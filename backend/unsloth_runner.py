#!/usr/bin/env python3

import asyncio
import subprocess
import uuid
import logging
import os
import signal
import torch
from typing import Dict, Optional
from datetime import datetime

from models.training import TrainingJob, TrainingConfig, TrainingStatus, TrainingMethod
from services.database import db_service
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)

class CustomTrainer(Trainer):
    """Custom trainer that lets accelerate handle device placement for multi-GPU"""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # No manual .to() or device movement here
        # Let accelerate + device_map handle tensor dispatching
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch
        )

class DualGPUTrainingRunner:
    """Training runner for dual GPU setups using Transformers + PEFT"""
    
    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self._log_files: Dict[str, open] = {}
    
    async def create_job(self, config: TrainingConfig) -> TrainingJob:
        """Create a new training job"""
        job_id = str(uuid.uuid4())[:8]
        
        job = TrainingJob(
            id=job_id,
            config=config,
            status=TrainingStatus.PENDING,
            total_epochs=config.epochs
        )
        
        self.jobs[job_id] = job
        logger.info(f"Created dual GPU training job {job_id}")
        
        return job
    
    async def start_job(self, job_id: str) -> bool:
        """Start training job with dual GPU support"""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status not in [TrainingStatus.PENDING, TrainingStatus.PAUSED]:
            raise ValueError(f"Cannot start job in status: {job.status}")
        
        try:
            # Create training script for this job
            script_path = self._create_training_script(job)
            
            # Set environment variables
            env = os.environ.copy()
            
            # Set HF token if available
            hf_token = db_service.get_active_hf_token()
            if hf_token:
                env["HUGGINGFACE_HUB_TOKEN"] = hf_token
                env["HF_TOKEN"] = hf_token
                logger.info(f"Using HF authentication for job {job_id}")
            
            # Create directories
            os.makedirs("logs", exist_ok=True)
            os.makedirs("checkpoints", exist_ok=True)
            
            # Create log file
            log_file_path = f"logs/{job_id}.log"
            job.log_file = log_file_path
            
            # Command to run the training script 
            # Use direct python execution with device_map="auto" for dual GPU
            cmd = ["python", script_path]
            
            if job.config.use_dual_gpu and torch.cuda.device_count() >= 2:
                logger.info(f"Starting dual GPU training for job {job_id} with device_map='auto'")
            else:
                logger.info(f"Starting single GPU training for job {job_id}")
            
            # Start process
            log_file = open(log_file_path, 'w')
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid,
                bufsize=1,
                universal_newlines=True
            )
            
            self.processes[job_id] = process
            self._log_files[job_id] = log_file
            
            # Update job status
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now()
            job.checkpoint_dir = f"checkpoints/{job_id}"
            
            logger.info(f"Started dual GPU training job {job_id} with PID {process.pid}")
            
            # Start monitoring
            asyncio.create_task(self._monitor_job(job_id))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start dual GPU job {job_id}: {e}")
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            return False
    
    def _create_training_script(self, job: TrainingJob) -> str:
        """Create a Python training script for this job"""
        script_path = f"scripts/train_{job.id}.py"
        os.makedirs("scripts", exist_ok=True)
        
        # Generate training script based on config
        script_content = f'''#!/usr/bin/env python3
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

class CustomTrainer(Trainer):
    """Custom trainer that lets accelerate handle device placement for multi-GPU"""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # No manual .to() or device movement here
        # Let accelerate + device_map handle tensor dispatching
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch
        )

def formatting_prompts_func(examples):
    """Format dataset for training"""
    # Handle both single examples and batched examples
    if isinstance(examples["instruction"], str):
        # Single example
        instruction = examples["instruction"]
        input_text = examples.get("input", "")
        output = examples["output"]
        
        # Create prompt in Alpaca format
        if input_text and input_text.strip():
            text = "### Instruction:\\n" + instruction + "\\n\\n### Input:\\n" + input_text + "\\n\\n### Response:\\n" + output
        else:
            text = "### Instruction:\\n" + instruction + "\\n\\n### Response:\\n" + output
        
        return [text]
    else:
        # Batched examples
        texts = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples.get("input", [""] * len(examples["instruction"]))[i]
            output = examples["output"][i]
            
            # Create prompt in Alpaca format
            if input_text and input_text.strip():
                text = "### Instruction:\\n" + instruction + "\\n\\n### Input:\\n" + input_text + "\\n\\n### Response:\\n" + output
            else:
                text = "### Instruction:\\n" + instruction + "\\n\\n### Response:\\n" + output
            
            texts.append(text)
        
        return texts

def main():
    print("🚀 Starting standard LoRA training...")
    print(f"📊 GPUs available: {{torch.cuda.device_count()}}")
    print(f"💾 Memory per GPU: {{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}}GB")
    
    # Load model with standard transformers
    print("📥 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("{job.config.base_model}")
    
    # Define explicit device map for dual GPU distribution
    if torch.cuda.device_count() >= 2:
        print("🔄 Setting up dual GPU distribution...")
        device_map = {{
            "model.embed_tokens": 0,
            "model.layers.0": 0, "model.layers.1": 0, "model.layers.2": 0, "model.layers.3": 0,
            "model.layers.4": 0, "model.layers.5": 0, "model.layers.6": 0, "model.layers.7": 0,
            "model.layers.8": 0, "model.layers.9": 0, "model.layers.10": 0, "model.layers.11": 0,
            "model.layers.12": 1, "model.layers.13": 1, "model.layers.14": 1, "model.layers.15": 1,
            "model.layers.16": 1, "model.layers.17": 1, "model.layers.18": 1, "model.layers.19": 1,
            "model.layers.20": 1, "model.layers.21": 1, "model.layers.22": 1, "model.layers.23": 1,
            "model.layers.24": 1, "model.layers.25": 1, "model.layers.26": 1, "model.layers.27": 1,
            "model.layers.28": 1, "model.layers.29": 1, "model.layers.30": 1, "model.layers.31": 1,
            "model.norm": 0,
            "lm_head": 0,
        }}
    else:
        print("📍 Using single GPU...")
        device_map = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(
        "{job.config.base_model}",
        torch_dtype=torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16,
        device_map=device_map,  # Use explicit or auto mapping
        load_in_4bit={job.config.method == TrainingMethod.QLORA},
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure LoRA with PEFT
    print("🎯 Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r={job.config.lora_config.rank},
        lora_alpha={job.config.lora_config.alpha},
        lora_dropout={job.config.lora_config.dropout},
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Check GPU memory usage after model loading
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  GPU {{i}} allocated: {{allocated:.2f}}GB")
    
    # Load dataset
    print("📊 Loading dataset...")
    dataset = load_dataset("json", data_files="{job.config.dataset_path}", split="train")
    
    # Tokenize dataset using standard approach
    def tokenize_function(examples):
        formatted_texts = formatting_prompts_func(examples)
        tokenized = tokenizer(
            formatted_texts,
            truncation=True,
            padding=True,
            max_length={job.config.max_sequence_length},
            return_tensors=None  # Don't return tensors yet
        )
        # Add labels for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    print("🎯 Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling
    )
    
    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size={job.config.batch_size},
        gradient_accumulation_steps={job.config.gradient_accumulation_steps},
        warmup_steps=10,
        num_train_epochs={job.config.epochs},
        learning_rate={job.config.learning_rate},
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_torch",  # Use standard optimizer
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="./checkpoints/{job.id}",
        save_strategy="epoch",
        save_total_limit=3,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        report_to="none",  # Disable wandb logging
        remove_unused_columns=False,
    )
    
    # Create CustomTrainer with device mismatch handling
    print("🎯 Creating trainer...")
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        args=training_args,
    )
    
    print("🎯 Starting training...")
    trainer.train()
    
    print("💾 Saving model...")
    trainer.save_model()
    
    print("✅ Training completed successfully!")

if __name__ == "__main__":
    main()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        return script_path
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        if job_id in self.processes:
            process = self.processes[job_id]
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait()
                
                del self.processes[job_id]
                logger.info(f"Terminated dual GPU process for job {job_id}")
                
            except Exception as e:
                logger.error(f"Failed to terminate process for job {job_id}: {e}")
        
        job.status = TrainingStatus.CANCELLED
        job.completed_at = datetime.now()
        
        # Clean up script file
        script_path = f"scripts/train_{job_id}.py"
        if os.path.exists(script_path):
            os.remove(script_path)
        
        return True
    
    async def pause_job(self, job_id: str) -> bool:
        """Pause a training job (note: dual GPU doesn't support pause/resume)"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        # For now, pause is implemented as cancel since dual GPU doesn't support pause/resume
        logger.warning(f"Pause requested for job {job_id}, but dual GPU doesn't support pause/resume. Cancelling instead.")
        return await self.cancel_job(job_id)
    
    async def resume_job(self, job_id: str) -> bool:
        """Resume a paused training job (note: dual GPU doesn't support pause/resume)"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        # For now, resume is implemented as restart since dual GPU doesn't support pause/resume
        logger.warning(f"Resume requested for job {job_id}, but dual GPU doesn't support pause/resume. Restarting instead.")
        return await self.restart_job(job_id)
    
    async def restart_job(self, job_id: str) -> bool:
        """Restart a training job"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        # Cancel current job if running
        if job.status in [TrainingStatus.RUNNING, TrainingStatus.PAUSED]:
            await self.cancel_job(job_id)
        
        # Reset job status
        job.status = TrainingStatus.PENDING
        job.started_at = None
        job.completed_at = None
        job.current_epoch = 0
        job.current_step = 0
        job.loss = None
        job.error_message = None
        
        # Start the job again
        return await self.start_job(job_id)
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def list_jobs(self) -> list[TrainingJob]:
        """List all jobs"""
        return list(self.jobs.values())
    
    async def _monitor_job(self, job_id: str):
        """Monitor training job progress"""
        if job_id not in self.processes:
            return
        
        process = self.processes[job_id]
        job = self.jobs[job_id]
        
        try:
            # Parse logs in background
            asyncio.create_task(self._parse_training_logs(job_id))
            
            # Wait for completion
            return_code = await asyncio.create_task(
                self._wait_for_process(process)
            )
            
            if return_code == 0:
                job.status = TrainingStatus.COMPLETED
                job.completed_at = datetime.now()
                logger.info(f"Dual GPU job {job_id} completed successfully")
            else:
                job.status = TrainingStatus.FAILED
                job.completed_at = datetime.now()
                job.error_message = f"Process exited with code {return_code}"
                logger.error(f"Dual GPU job {job_id} failed with exit code {return_code}")
            
            # Cleanup
            if job_id in self.processes:
                del self.processes[job_id]
            if job_id in self._log_files:
                try:
                    self._log_files[job_id].close()
                    del self._log_files[job_id]
                except Exception:
                    pass
            
            # Clean up script file
            script_path = f"scripts/train_{job_id}.py"
            if os.path.exists(script_path):
                os.remove(script_path)
                
        except Exception as e:
            logger.error(f"Error monitoring dual GPU job {job_id}: {e}")
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
    
    async def _parse_training_logs(self, job_id: str):
        """Parse training logs to extract progress"""
        job = self.jobs[job_id]
        if not job.log_file:
            return
            
        import re
        last_position = 0
        
        while job_id in self.processes and job.status == TrainingStatus.RUNNING:
            try:
                if os.path.exists(job.log_file):
                    with open(job.log_file, 'r') as f:
                        f.seek(last_position)
                        new_lines = f.read()
                        last_position = f.tell()
                        
                        # Parse dual GPU training logs
                        # Look for: {'loss': 1.234, 'learning_rate': 0.0002, 'epoch': 1.0}
                        
                        # Extract step info
                        step_matches = re.findall(r"'step': (\d+)", new_lines)
                        if step_matches:
                            job.current_step = int(step_matches[-1])
                        
                        # Extract loss
                        loss_matches = re.findall(r"'train_loss': ([\d.]+)", new_lines)
                        if loss_matches:
                            job.loss = float(loss_matches[-1])
                        
                        # Extract epoch
                        epoch_matches = re.findall(r"'epoch': ([\d.]+)", new_lines)
                        if epoch_matches:
                            job.current_epoch = int(float(epoch_matches[-1]))
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error parsing dual GPU logs for job {job_id}: {e}")
                await asyncio.sleep(5)
    
    async def _wait_for_process(self, process: subprocess.Popen) -> int:
        """Async wrapper for process.wait()"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, process.wait)

# Global dual GPU training runner instance
dual_gpu_runner = DualGPUTrainingRunner() 