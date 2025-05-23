import yaml
import os
from typing import Dict, Any
from models.training import TrainingConfig, TrainingMethod

class AxolotlConfigBuilder:
    """Build Axolotl YAML configurations from training config"""
    
    @staticmethod
    def build_config(training_config: TrainingConfig, job_id: str) -> Dict[str, Any]:
        """Build Axolotl configuration from training config"""
        
        config = {
            "base_model": training_config.base_model,
            "tokenizer_type": "AutoTokenizer",
            
            # Don't force special tokens - let the model use its native tokenizer
            # This prevents vocabulary expansion and tokenizer corruption
            # The model's original special tokens will be used automatically
            
            # Dataset configuration
            "datasets": [
                {
                    "path": training_config.dataset_path,
                    "ds_type": "json",
                    "type": "alpaca"
                }
            ],
            
            # Training parameters - adjusted for small datasets
            "sequence_len": min(training_config.max_sequence_length, 512),  # Smaller for small datasets
            "sample_packing": False,  # Disable sample packing for small datasets
            "pad_to_sequence_len": True,
            
            # Training settings
            "micro_batch_size": 1,  # Use batch size 1 for small datasets
            "gradient_accumulation_steps": max(training_config.gradient_accumulation_steps, training_config.batch_size),
            "num_epochs": training_config.epochs,
            "learning_rate": training_config.learning_rate,
            
            # Optimizer settings
            "optimizer": "adamw_bnb_8bit" if training_config.method == TrainingMethod.QLORA else "adamw_torch",
            "lr_scheduler": "cosine",
            
            # Validation
            "val_set_size": 0.0,  # Disable validation for small datasets
            "save_strategy": "epoch",  # Save per epoch instead of steps for small datasets
            "logging_steps": 1,  # Log every step for small datasets
            
            # Output configuration
            "output_dir": f"./checkpoints/{job_id}",
            "hub_model_id": None,
            "push_dataset_to_hub": None,
            
            # Compute configuration
            "bf16": training_config.precision.value == "bf16",
            "fp16": training_config.precision.value == "fp16",
            "tf32": False,
            
            # Hardware optimization - conservative settings for small datasets
            "gradient_checkpointing": True,
            "dataloader_pin_memory": False,
            "group_by_length": False,  # Disable grouping for small datasets
            "dataloader_num_workers": 0,  # Single threaded for small datasets
            
            # Wandb (disabled by default)
            "wandb_project": None,
            "wandb_entity": None,
            "wandb_watch": None,
            "wandb_run_id": None,
            "wandb_log_model": None,
        }
        
        # Add adapter configuration for LoRA/QLoRA
        if training_config.method in [TrainingMethod.LORA, TrainingMethod.QLORA]:
            config.update({
                "adapter": "lora",
                "lora_r": training_config.lora_config.rank,
                "lora_alpha": training_config.lora_config.alpha,
                "lora_dropout": training_config.lora_config.dropout,
                "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "lora_fan_in_fan_out": False,
            })
        
        # QLoRA specific settings
        if training_config.method == TrainingMethod.QLORA:
            config.update({
                "load_in_8bit": False,
                "load_in_4bit": True,
                "strict": False,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": "bfloat16" if training_config.precision.value == "bf16" else "float16",
                "bnb_4bit_use_double_quant": True,
            })
        
        # Multi-GPU configuration (disabled for now to avoid issues)
        # if training_config.use_dual_gpu:
        #     config["ddp"] = True
        #     config["ddp_timeout"] = 1800
        #     config["ddp_bucket_cap_mb"] = 25
        
        # Force single GPU to avoid device mismatch errors
        config.update({
            "ddp": False,
            "fsdp": [],
            "deepspeed": None,
            # Force everything to cuda:0 to avoid device placement issues
            "local_rank": None,
            "device_map": None,  # Let model decide placement
            "auto_resume_from_checkpoints": False,
        })
        
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> None:
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def build_and_save(training_config: TrainingConfig, job_id: str) -> str:
        """Build and save Axolotl configuration, return config file path"""
        config = AxolotlConfigBuilder.build_config(training_config, job_id)
        config_path = f"./configs/{job_id}.yaml"
        AxolotlConfigBuilder.save_config(config, config_path)
        return config_path 