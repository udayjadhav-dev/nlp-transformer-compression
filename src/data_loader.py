# src/data_loader.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def get_sst2_pipeline(model_name="distilbert-base-uncased", batch_size=32):
    print(f"[+] Loading tokenizer and dataset for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Using the corrected, official namespaced HuggingFace repository path
    dataset = load_dataset("nyu-mll/glue", "sst2")

    def tokenize_function(examples):
        # Targeting 'sentence' instead of 'text' based on GLUE schema specs
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

    # 1. Map tokenizer transformations over text items
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 2. Rename singular 'label' column to plural 'labels' for PyTorch requirements
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    
    # 3. Explicitly keep only tensor arrays to protect against list processing faults
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 4. Generate the PyTorch validation DataLoader split
    eval_dataloader = torch.utils.data.DataLoader(
        tokenized_datasets["validation"], batch_size=batch_size
    )
    return eval_dataloader