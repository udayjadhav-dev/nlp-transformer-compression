# src/quantize.py
import argparse
import torch
import os
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from data_loader import get_sst2_pipeline
from evaluate import evaluate_model, get_model_size_mb

# Import the modern torchao toolkit for Apple Silicon compatibility
try:
    import torchao
    # Importing the modern stable configuration class name
    from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
except ImportError:
    raise ImportError("[!] Please run 'pip install torchao' to install the required modern backend.")

def run_quantization_pipeline(model_name, strategy, epochs, lr):
    print(f"\n==== Starting TinyML Model Compression Framework [Strategy: {strategy.upper()}] ====")
    
    # 1. Fetch Pipeline Components
    dataloader = get_sst2_pipeline(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Vanilla Baseline Model (FP32)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Ensure directory existence before writing checkpoint states
    os.makedirs("models", exist_ok=True)
    baseline_path = "models/baseline_fp32.pt"
    torch.save(model.state_dict(), baseline_path)
    fp32_size = get_model_size_mb(baseline_path)
    
    print("[+] Evaluating baseline FP32 model performance...")
    baseline_results = evaluate_model(model, dataloader)
    
    # 3. Apply Selected Quantization Strategy
    if strategy == "ptq":
        print("[+] Applying Modern torchao Dynamic 8-bit Quantization to Linear Layers...")
        # Deepcopy protects the baseline model from in-place modifications
        import copy
        compressed_model = copy.deepcopy(model)
        
        # Uses the modern backend which compiles flawlessly on macOS
        quantize_(compressed_model, Int8DynamicActivationInt8WeightConfig())
        
        output_path = "models/quantized_ptq_int8.pt"
        
    elif strategy == "qat":
        print(f"[+] Initializing Quantization-Aware Training (QAT) Graph Transformation...")
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack') # Use qnnpack for arm64 target if available
        prepared_model = torch.quantization.prepare_qat(model, inplace=False)
        prepared_model.to(device)
        
        optimizer = torch.optim.AdamW(prepared_model.parameters(), lr=lr, eps=1e-8)
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
        
        print(f"[+] Fine-tuning model parameters with quantization constraints for {epochs} epochs...")
        for epoch in range(epochs):
            prepared_model.train()
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)
                
                outputs = prepared_model(**inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(prepared_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            print(f"    -> Epoch {epoch+1}/{epochs} Complete. Avg Loss: {total_loss/len(dataloader):.4f}")
        
        print("[+] Freezing quantization parameters and converting graph to INT8...")
        prepared_model.to("cpu")
        prepared_model.eval()
        compressed_model = torch.quantization.convert(prepared_model, inplace=False)
        output_path = "models/quantized_qat_int8.pt"
        
    # Save the compressed operational artifacts
    torch.save(compressed_model.state_dict(), output_path)
    compressed_size = get_model_size_mb(output_path)
    
    print("[+] Evaluating compressed model performance...")
    compressed_results = evaluate_model(compressed_model, dataloader)
    
    # 4. Print Empirical Research Report
    print("\n================ EMPIRICAL RESEARCH REPORT ================")
    print(f"{'Metric':<25} | {'Baseline (FP32)':<18} | {'Compressed (INT8)':<18}")
    print("-" * 68)
    print(f"{'Model Size':<25} | {fp32_size:>12.2f} MB | {compressed_size:>12.2f} MB")
    print(f"{'Reduction Percentage':<25} | {'0.0%':>18} | {((fp32_size - compressed_size)/fp32_size)*100:>11.2f}%")
    print(f"{'SST-2 Task Accuracy':<25} | {baseline_results['accuracy']:>17.4f} | {compressed_results['accuracy']:>17.4f}")
    print(f"{'F1-Score Performance':<25} | {baseline_results['f1_score']:>17.4f} | {compressed_results['f1_score']:>17.4f}")
    print(f"{'Avg Batch Latency':<25} | {baseline_results['avg_batch_latency_ms']:>12.2f} ms | {compressed_results['avg_batch_latency_ms']:>12.2f} ms")
    print("===========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyML NLP Transformer Compression Engine")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased", help="HuggingFace model stub")
    parser.add_argument("--strategy", type=str, default="ptq", choices=["ptq", "qat"], help="Compression strategy")
    parser.add_argument("--epochs", type=int, default=3, help="Number of QAT fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for QAT optimization loop")
    
    args = parser.parse_args()
    run_quantization_pipeline(args.model, args.strategy, args.epochs, args.lr)