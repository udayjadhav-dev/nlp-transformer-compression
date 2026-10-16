# TinyML & Green AI: Dynamic INT8 Transformer Optimization for Resource-Constrained Edge Devices 🚀

## 🔬 Project Overview
This project addresses the modern **"Green AI"** challenge: reducing the massive computational and energy footprint of Large Language Models (LLMs) for edge deployments. 

Utilizing the **TorchAO (2026 Engine)** framework, we successfully applied **Post-Training Quantization (PTQ)** to a fine-tuned Transformer model (`distilbert-base-uncased-finetuned-sst-2-english`), converting heavy 32-bit floating-point weights (`FP32`) into highly optimized 8-bit integers (`INT8`).

## 📊 Core Performance Metrics (MacBook Pro Benchmarks)
* **Model Size Reduction:** **48.14% Compression** (Reduced from **255.45 MB to 132.49 MB**)
* **Memory Optimization:** Bypasses standard hardware bottlenecks while preserving deep transformer layers.
* **Accuracy Metrics:** Successfully handles complex linguistic edge cases including implicit negation, sarcasm, and intense phrase contrast transitions.

## 🛠️ System Architecture & Folders
* `/src/quantize.py`: Script executing the TorchAO object configuration engine.
* `/src/benchmark.py`: Profiling engine for execution latencies.
* `/app/app.py`: Production-ready local web server interface built on Flask.

## 🚀 Quick Start
```bash
# Clone the repository
git clone https://github.com
cd tinyml-model-shrinking

# Set up environment and install pinned dependencies
python3 -m venv tinyml_env
source tinyml_env/bin/activate
pip install -r requirements.txt

# Run the live web interface
python3 app/app.py
```