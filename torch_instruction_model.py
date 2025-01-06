import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import gc

# need a permanent solution for this
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set the PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
 
# Clear the cache directory
cache_dir = "C:/Users/MahtabSarwar/.cache/huggingface"
 
# Set your Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HF_MODULES_CACHE"] = cache_dir
hf_token="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"

instruction_tuning_data_files = [
        "pm_data_all/PMData_fatigue_train_all.json",
        "pm_data_all/PMData_readiness_train_all.json",
        "pm_data_all/PMData_sleep_quality_train_all.json",
        "pm_data_all/PMData_stress_train_all.json"
    ]

def load_data(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            data.extend(file_data)
    return data


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, device):
        self.data = data
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item['instruction']
        input_text = item['input']
        output_text = item['output']
        prompt = f"{instruction}\n{input_text}\n{output_text}"
        encoding = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        encoding['labels'] = encoding['input_ids']
        return {key: val.squeeze().to(self.device) for key, val in encoding.items()}

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def main():
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(local_rank))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(local_rank) / 1024**3, 1), 'GB')
        print('Reserved:', round(torch.cuda.memory_reserved(local_rank) / 1024**3, 1), 'GB')

    model_name = "meta-llama/Llama-3.2-1B"
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Load the dataset
    data = load_data(instruction_tuning_data_files)
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_data, val_data = random_split(data, [train_size, val_size])

    train_sampler = DistributedSampler(train_data)
    val_sampler = DistributedSampler(val_data)

    train_dataset = InstructionDataset([data[i] for i in train_data.indices], tokenizer, device)
    val_dataset = InstructionDataset([data[i] for i in val_data.indices], tokenizer, device)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn, sampler=val_sampler)

    accumulation_steps = 4  # Adjust this based on your memory constraints
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    scaler = GradScaler()
    num_epochs = 3

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast('cuda'):
                outputs = model(**batch)
                loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), batch['labels'].view(-1))
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if local_rank == 0:
                    print(f"Epoch {epoch + 1}, Step {i + 1}, Loss: {loss.item()}")

            # Clear cache and delete variables
            del batch, outputs, loss
            torch.cuda.empty_cache()
            gc.collect()

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast('cuda'):
                    outputs = model(**batch)
                    loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), batch['labels'].view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        if local_rank == 0:
            print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")

    # Final step for remaining gradients
    if (i + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    if local_rank == 0:
        model.module.save_pretrained("./instruction-tuned-llama3")
        tokenizer.save_pretrained("./instruction-tuned-llama3")
    
if __name__ == '__main__':
    main()
