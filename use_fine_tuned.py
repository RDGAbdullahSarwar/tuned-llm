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

def check_cuda():
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    print("PyTorch version:", torch.__version__)
    return torch.cuda.is_available()

def test_model( model_type ):
    model = None
    tokenizer = None
    
    if model_type == "instruction_tuned":
        model = AutoModelForCausalLM.from_pretrained("./instruction-tuned-llama3")
        tokenizer = AutoTokenizer.from_pretrained("./instruction-tuned-llama3")
    elif model_type == "data_tuned":
        model = AutoModelForCausalLM.from_pretrained("./fine-tuned-llama3")
        tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-llama3")
    else:  # base model
        model_name = "meta-llama/Llama-3.2-1B"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    
    # Move model to GPU
    device = torch.device("cuda")
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Prepare the input
    #question = "What is the prevalence of depression, anxiety, and insomnia among privileged and underprivileged classes in Pakistan during the COVID-19 lockdown?"
    #context = "Adult Anxiety COVID-19 Communicable Disease Control Cross-Sectional Studies Depression Female Humans Male Mental Health Middle Aged Pakistan Pandemics Prevalence Risk Factors SARS-CoV-2 Sleep Initiation and Maintenance Disorders Surveys and Questionnaires We aimed to determine the frequency of depression, anxiety and insomnia; identify associated factors; and compare these outcomes amongst a privileged and underprivileged class of Pakistan. A cross-sectional online and face to face survey was conducted in Karachi from April 2020 to May 2020. Validated depression (World Health Organization self-reporting questionnaire), anxiety (general anxiety and depression scale) and insomnia (insomnia severity index) scales were used. Out of 447 participants, the majority were less than 30 years (63.8%) and females (57.7%); 20.8% study participants belonged to poor or very poor socioeconomic status; 17% respondents were from lower middle status and 38% belonged to the higher middle or rich class. Depression, anxiety and insomnia were identified in 30%, 30.63% and 8.5% of participants, respectively. The prevalence of depression, anxiety and insomnia among privileged people was 37.8%, 16.6% and 11.3% respectively whereas among underprivileged were 17.8%, 16.6% and 4.1% respectively. There were significant differences in frequencies of depression (p<0.001), anxiety (p<0.001) and insomnia (p=0.009) among the privileged and underprivileged classes. We found a high prevalence of depression, anxiety and insomnia among both the privileged and underprivileged Pakistani population and a policy needs to be devised to ensure the mental health of Pakistani population."
    #input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
     # Prepare a simple input
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Move inputs to GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate the output
    # Generate the output
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.9,
        num_beams=5,
        repetition_penalty=1.5,
        early_stopping=True
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
if __name__ == "__main__":
    if check_cuda() == True:
        print("Select an option:")
        print("1) Run base model")
        print("2) Run data tuned model")
        print("3) Run instruction tuned model")
        
        choice = input("Enter the number of your choice: ")
        
        if choice == '1':
            model_type = "base"
        elif choice == '2':
            model_type = "data_tuned"
        elif choice == '3':
            model_type = "instruction_tuned"
        else:
            print("Invalid choice. Defaulting to base model.")
            model_type = "base"
        
        test_model(model_type)
