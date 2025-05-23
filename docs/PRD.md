# LLM Fine-Tuning UI - Product Requirements Document (PRD)

## Overview

This project aims to build a user-friendly web-based UI for training and fine-tuning LLMs (Large Language Models) on local consumer hardware (specifically dual RTX 3060 GPUs). The UI leverages **Unsloth** for 2-5x faster training with superior multi-GPU support, enabling efficient LoRA, QLoRA, and full fine-tuning of Hugging Face-compatible models.

## Goals

* Enable fine-tuning of existing Hugging Face LLMs via a browser UI
* **Achieve 2-5x faster training** using Unsloth optimizations
* **Automatic dual GPU utilization** without complex setup
* Support LoRA, QLoRA, and full fine-tuning (when feasible)
* Reusable, modular architecture
* Monitor GPU usage, training status, and logs
* Support common dataset formats (JSONL, CSV, TXT)

---

## Users

* AI Researchers and Engineers
* Hobbyists and Self-hosters with GPU access

---

## Functional Requirements

### 1. Model Selection

* Fetch and display a list of supported models (default: LLaMA2, Mistral, Falcon, etc.)
* Allow custom Hugging Face model ID input
* Check for tokenizer compatibility
* **Unsloth-optimized model variants** when available

### 2. Dataset Management

* Upload dataset file (.jsonl, .csv, .txt)
* Display file metadata (size, lines, sample preview)
* Auto-convert to Unsloth-compatible format
* Save datasets in `backend/uploads/`

### 3. Training Configuration

* Define training method:
  * LoRA (with Unsloth optimizations)
  * QLoRA (4-bit quantization)
  * Full Fine-tuning
* Set configurable params:
  * Epochs
  * Learning Rate
  * Batch Size
  * Max Sequence Length
  * Adapter config (LoRA rank, alpha, dropout)
* Select compute options:
  * **Automatic dual GPU distribution**
  * Mixed precision (fp16, bf16 auto-detected)
* **Dynamic Python script generation** (no YAML configs)

### 4. Training Launch

* Start training from UI (via FastAPI backend)
* **Unsloth handles multi-GPU automatically**
* Launch as background process
* Show real-time logs with Unsloth optimization messages
* Show estimated time remaining
* Pause/Resume/Cancel training jobs

### 5. Training Monitor

* Stream logs from backend (`tail -f`)
* GPU Stats:
  * Memory usage per GPU (both RTX 3060s)
  * Temperature
  * Fan speed
  * Utilization (%) across both GPUs
* Display training metrics:
  * Loss
  * Validation loss (if val set given)
  * Step count
  * Tokens processed
  * **Training speed improvements** (2-5x notifications)

### 6. Checkpoint Browser

* List completed training runs
* Show metadata:
  * Base model
  * Dataset used
  * Training parameters
  * Training time (with speedup metrics)
  * **Unsloth optimizations applied**
* Allow preview generation from saved checkpoints (inference)

### 7. Inference Sandbox (Post-training)

* Load model checkpoint
* Enter prompt and generate text
* Show token-by-token output
* **Fast inference** with Unsloth optimizations

---

## Non-functional Requirements

* **Automatic dual RTX 3060 utilization** using Unsloth's model parallelism
* **2-5x faster training** compared to traditional methods
* Responsive UI (React + Tailwind)
* REST API (FastAPI)
* File-based config and output structure
* **Compatible with Unsloth** (as backend trainer)
* **Memory efficient** - fits larger models on consumer GPUs

---

## Technical Stack

### Frontend

* React
* Tailwind CSS
* Vite or Next.js (TBD)

### Backend

* FastAPI (Python)
* **Unsloth** (training backend)
* **Dynamic Python script generation**
* Log streaming (websocket or polling)
* **TRL (Transformers Reinforcement Learning)** integration

### System

* Ubuntu Linux (CUDA 12.0+ installed)
* Python 3.10+
* **Unsloth** with dependencies
* Conda or venv environments

---

## Folder Structure

```
llm-trainer-ui/
├── backend/
│   ├── main.py
│   ├── unsloth_runner.py     # Unsloth training orchestration
│   ├── models/               # Data models
│   ├── routes/               # API endpoints
│   ├── services/             # Business logic
│   ├── uploads/              # Datasets
│   ├── logs/                # Training logs
│   ├── checkpoints/         # Model outputs
│   └── scripts/             # Generated training scripts
├── frontend/
│   ├── components/
│   ├── pages/
│   └── public/
└── docs/                    # Documentation
```

---

## Milestones

### Phase 1: Basic MVP ✅

* [x] Upload dataset
* [x] Select base model
* [x] Configure LoRA training
* [x] Launch training via UI
* [x] Show logs + GPU stats

### Phase 2: Extended Support ✅

* [x] Add QLoRA and full fine-tune modes
* [x] Checkpoint management
* [x] Inference preview tab
* [x] **Unsloth migration for 2-5x faster training**
* [x] **Dual GPU optimization**
* [ ] Multi-user support (auth)

---

## Performance Improvements with Unsloth

### Training Speed
* **2-5x faster** than traditional Axolotl-based training
* Optimized gradient computation and memory management
* **Automatic model parallelism** across dual RTX 3060s

### Memory Efficiency
* Better memory utilization allows larger models
* **4-bit quantization** support for extreme efficiency
* Reduced GPU memory fragmentation

### GPU Utilization
* **Both RTX 3060s utilized automatically**
* No complex distributed training setup required
* Optimal tensor placement across devices

### Development Experience
* **Simplified setup** - no YAML configuration files
* **Clear error messages** instead of cryptic distributed training errors
* **Dynamic script generation** based on UI parameters

---

## Migration Benefits (Axolotl → Unsloth)

| Aspect | Axolotl (Old) | Unsloth (New) |
|--------|---------------|---------------|
| **Setup Complexity** | Complex YAML configs, distributed training | Simple Python scripts, auto-optimization |
| **Dual GPU Support** | Manual configuration, frequent errors | Automatic, works out of the box |
| **Training Speed** | Standard PyTorch speed | 2-5x faster with optimizations |
| **Memory Usage** | Often out-of-memory on 12GB GPUs | Optimized, fits larger models |
| **Error Handling** | Cryptic distributed training errors | Clear, actionable error messages |
| **GPU Utilization** | Single GPU or complex multi-GPU setup | Automatic optimal distribution |

---

## Future Ideas

* Hugging Face dataset integration
* Pre/post training evaluation benchmark
* AutoLR finder
* Hugging Face Hub model push
* **Unsloth-specific optimizations** for new model architectures

---

## Notes

* All training runs must be resumable (save checkpoint every epoch)
* **Unsloth handles multi-GPU automatically** - no manual distributed training setup
* Avoid image/data handling logic – this is LLM only
* Prioritise fast iteration (sub-10s feedback loop from config to training start)
* **Training speed improvements** are a key competitive advantage

---

## Glossary

* **LoRA**: Low-Rank Adaptation, efficient way to fine-tune large models by injecting trainable layers
* **QLoRA**: Quantised LoRA – memory-efficient method allowing fine-tuning on consumer GPUs
* **Unsloth**: High-performance training library providing 2-5x speedup with automatic multi-GPU support
* **TRL**: Transformers Reinforcement Learning library used by Unsloth for training
* **Model Parallelism**: Automatic distribution of model layers across multiple GPUs
