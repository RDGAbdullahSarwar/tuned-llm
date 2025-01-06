import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import gc

# need a permanent solution for this
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set the PYTORCH_CUDA_ALLOC_CONF environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
 
# Clear the cache directory
cache_dir = "C:/Users/MahtabSarwar/.cache/huggingface"
 
# Set your Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir
os.environ["HF_MODULES_CACHE"] = cache_dir
token="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"


class InstructionDataset(Dataset):
    def __init__(self, file_path, tokenizer, device):
        self.data = []
        self.tokenizer = tokenizer
        self.device = device
        with open(file_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

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


class BioASQDataset(Dataset):
    def __init__(self, file_path, tokenizer, device, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

        with open(file_path, 'r') as f:
            lines = f.readlines()
            # Skip the first line and the last line
            #for line in lines[1:-1]:
            for line in lines[1:501]:
                line = line.rstrip(',\n')
                try:
                    article = json.loads(line)
                    self.data.append(article)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {line}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        mesh_terms = " ".join(item['meshMajor'])
        input_text = f"{mesh_terms} {item['abstractText']}"
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        labels = self.tokenizer(
            item['title'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze().to(self.device),
            'attention_mask': inputs['attention_mask'].squeeze().to(self.device),
            'labels': labels['input_ids'].squeeze().to(self.device)
        }
        
def main():
    
    # check we're using GPU and cuda
    print("Checking cuda availability: {}".format(torch.cuda.is_available))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
        print('Reserved:', round(torch.cuda.memory_reserved(0) / 1024**3, 1), 'GB')

    model_name = "meta-llama/Llama-3.2-1B"
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    
    model.to(device)


    # Load the dataset
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = BioASQDataset('datasets/training/chunk_1.json', tokenizer, device)
    #val_dataset = BioASQDataset('datasets/validation/chunk_10.json', tokenizer, device)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    #val_loader = DataLoader(val_dataset, batch_size=4, num_workers=4)

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
                print(f"Epoch {epoch + 1}, Step {i + 1}, Loss: {loss.item()}")
                
            # Clear cache and delete variables
            del batch, outputs, loss
            torch.cuda.empty_cache()
            gc.collect()
            
    # Final step for remaining gradients
    if (i + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    model.save_pretrained("./fine-tuned-llama3")
    tokenizer.save_pretrained("./fine-tuned-llama3")

    model = AutoModelForCausalLM.from_pretrained("./fine-tuned-llama3")
    tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-llama3")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    #inputs = tokenizer("Your instruction here", return_tensors="pt")

    inputs = tokenizer("Question: What is the prevalence of depression, anxiety, and insomnia among privileged and underprivileged classes in Pakistan during the COVID-19 lockdown? Context: Adult Anxiety COVID-19 Communicable Disease Control Cross-Sectional Studies Depression Female Humans Male Mental Health Middle Aged Pakistan Pandemics Prevalence Risk Factors SARS-CoV-2 Sleep Initiation and Maintenance Disorders Surveys and Questionnaires We aimed to determine the frequency of depression, anxiety and insomnia; identify associated factors; and compare these outcomes amongst a privileged and underprivileged class of Pakistan. A cross-sectional online and face to face survey was conducted in Karachi from April 2020 to May 2020. Validated depression (World Health Organization self-reporting questionnaire), anxiety (general anxiety and depression scale) and insomnia (insomnia severity index) scales were used. Out of 447 participants, the majority were less than 30 years (63.8%) and females (57.7%); 20.8% study participants belonged to poor or very poor socioeconomic status; 17% respondents were from lower middle status and 38% belonged to the higher middle or rich class. Depression, anxiety and insomnia were identified in 30%, 30.63% and 8.5% of participants, respectively. The prevalence of depression, anxiety and insomnia among privileged people was 37.8%, 16.6% and 11.3% respectively whereas among underprivileged were 17.8%, 16.6% and 4.1% respectively. There were significant differences in frequencies of depression (p<0.001), anxiety (p<0.001) and insomnia (p=0.009) among the privileged and underprivileged classes. We found a high prevalence of depression, anxiety and insomnia among both the privileged and underprivileged Pakistani population and a policy needs to be devised to ensure the mental health of Pakistani population.", return_tensors="pt")
    outputs = model.generate(**inputs)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
if __name__ == '__main__':
    main()
