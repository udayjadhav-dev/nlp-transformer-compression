# src/evaluate.py
import time
import torch
import os
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, dataloader, device="cpu"):
    model.eval()
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch in dataloader:
            # Safely routes standard tensors over to the target runtime memory
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"]
            
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    end_time = time.time()
    
    total_latency = (end_time - start_time) * 1000 # Milliseconds
    avg_batch_latency = total_latency / len(dataloader)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    
    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "avg_batch_latency_ms": avg_batch_latency
    }

def get_model_size_mb(model_path):
    if not os.path.exists(model_path):
        return 0
    size_bytes = os.path.getsize(model_path)
    return size_bytes / (1024 * 1024)