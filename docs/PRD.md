# LLM Fine-Tuning UI - Product Requirements Document (PRD)

## Overview

This project aims to build a user-friendly web-based UI for training and fine-tuning LLMs (Large Language Models) on local consumer hardware (specifically dual RTX 3060 GPUs). The UI will support a generic, extensible setup allowing users to fine-tune Hugging Face-compatible LLMs using LoRA, QLoRA, and other efficient training techniques.

## Goals

* Enable fine-tuning of existing Hugging Face LLMs via a browser UI
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

### 2. Dataset Management

* Upload dataset file (.jsonl, .csv, .txt)
* Display file metadata (size, lines, sample preview)
* Auto-convert to Axolotl-compatible format
* Save datasets in `backend/uploads/`

### 3. Training Configuration

* Define training method:

  * LoRA
  * QLoRA
  * Full Fine-tuning
* Set configurable params:

  * Epochs
  * Learning Rate
  * Batch Size
  * Max Sequence Length
  * Adapter config (LoRA rank, alpha, dropout)
* Select compute options:

  * Single or dual GPU
  * Mixed precision (fp16, bf16)
* Save configuration YAML for reproducibility

### 4. Training Launch

* Start training from UI (via FastAPI backend)
* Launch as background process
* Show real-time logs
* Show estimated time remaining
* Pause/Resume/Cancel training jobs

### 5. Training Monitor

* Stream logs from backend (`tail -f`)
* GPU Stats:

  * Memory usage per GPU
  * Temperature
  * Fan speed
  * Utilisation (%)
* Display training metrics:

  * Loss
  * Validation loss (if val set given)
  * Step count
  * Tokens processed

### 6. Checkpoint Browser

* List completed training runs
* Show metadata:

  * Base model
  * Dataset used
  * Params
  * Training time
* Allow preview generation from saved checkpoints (inference)

### 7. Inference Sandbox (Post-training)

* Load model checkpoint
* Enter prompt and generate text
* Show token-by-token output

---

## Non-functional Requirements

* Support dual RTX 3060s using model parallelism or QLoRA
* Responsive UI (React + Tailwind)
* REST API (FastAPI)
* File-based config and output structure
* Compatible with Axolotl (as backend trainer)

---

## Technical Stack

### Frontend

* React
* Tailwind CSS
* Vite or Next.js (TBD)

### Backend

* FastAPI (Python)
* Axolotl (subprocess runner)
* YAML config generator
* Log streaming (websocket or polling)

### System

* Ubuntu Linux (CUDA installed)
* Python 3.10+
* Conda or venv environments

---

## Folder Structure

```
llm-trainer-ui/
├── backend/
│   ├── main.py
│   ├── train_runner.py
│   ├── config_builder.py
│   ├── uploads/
│   ├── logs/
│   └── checkpoints/
├── frontend/
│   ├── components/
│   ├── pages/
│   └── public/
├── configs/
└── models/
```

---

## Milestones

### Phase 1: Basic MVP

* [ ] Upload dataset
* [ ] Select base model
* [ ] Configure LoRA training
* [ ] Launch training via UI
* [ ] Show logs + GPU stats

### Phase 2: Extended Support

* [ ] Add QLoRA and full fine-tune modes
* [ ] Checkpoint management
* [ ] Inference preview tab
* [ ] Multi-user support (auth)

---

## Future Ideas

* Hugging Face dataset integration
* Pre/post training evaluation benchmark
* AutoLR finder
* Hugging Face Hub model push

---

## Notes

* All training runs must be resumable (save checkpoint every X steps)
* Use accelerate or deepspeed only if needed for full fine-tune
* Avoid image/data handling logic – this is LLM only
* Prioritise fast iteration (sub-10s feedback loop from config to training start)

---

## Glossary

* **LoRA**: Low-Rank Adaptation, efficient way to fine-tune large models by injecting trainable layers
* **QLoRA**: Quantised LoRA – memory-efficient method allowing fine-tuning on consumer GPUs
* **Axolotl**: Open-source library for training and fine-tuning LLMs, supports LoRA/QLoRA
