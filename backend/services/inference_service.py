import os
import torch
from typing import Dict, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import warnings


class InferenceService:
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        # Use specific GPU 0 to avoid multi-GPU device conflicts
        if torch.cuda.is_available():
            self.device = "cuda:0"
            # Set the default CUDA device to ensure consistency
            torch.cuda.set_device(0)
        else:
            self.device = "cpu"
        print(f"InferenceService initialized with device: {self.device}")
        
    def load_model(self, job_id: str, job_config: Any) -> Dict[str, Any]:
        """Load a trained model for inference"""
        if job_id in self.loaded_models:
            return self.loaded_models[job_id]
            
        checkpoint_dir = f"checkpoints/{job_id}"
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            
        print(f"Loading model from checkpoint: {checkpoint_dir}")
        print(f"Files in checkpoint: {os.listdir(checkpoint_dir)}")
        
        base_model_name = job_config.base_model
        print(f"Base model: {base_model_name}")
        
        # Load tokenizer with better error handling and recovery
        tokenizer = None
        tokenizer_source = None
        expected_vocab_size = None
        
        # First, check for added tokens to understand expected vocab size
        added_tokens_path = os.path.join(checkpoint_dir, "added_tokens.json")
        if os.path.exists(added_tokens_path):
            import json
            with open(added_tokens_path, 'r') as f:
                added_tokens = json.load(f)
                print(f"Found added tokens: {added_tokens}")
                if added_tokens:
                    # Find the maximum token ID to determine expected vocab size
                    max_token_id = max(added_tokens.values())
                    expected_vocab_size = max_token_id + 1
                    print(f"Expected vocab size based on added tokens: {expected_vocab_size}")
        
        # Try to load tokenizer from checkpoint first
        try:
            if os.path.exists(os.path.join(checkpoint_dir, "tokenizer_config.json")):
                print(f"Attempting to load tokenizer from checkpoint...")
                tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
                tokenizer_source = "checkpoint"
                print(f"Successfully loaded tokenizer from checkpoint: vocab_size={len(tokenizer)}")
            else:
                print(f"No tokenizer config found in checkpoint")
                raise FileNotFoundError("No tokenizer in checkpoint")
                
        except Exception as e:
            print(f"Error loading tokenizer from checkpoint: {e}")
            print(f"Falling back to base model tokenizer and reconstructing...")
            
            # Load base model tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer_source = "base_model_reconstructed"
            
            # If we have added tokens, we need to add them to the tokenizer
            if expected_vocab_size and expected_vocab_size > len(tokenizer):
                print(f"Reconstructing tokenizer with added tokens...")
                print(f"Original tokenizer vocab size: {len(tokenizer)}")
                
                # Add the special tokens that were added during training
                if os.path.exists(added_tokens_path):
                    with open(added_tokens_path, 'r') as f:
                        added_tokens = json.load(f)
                        for token_text, token_id in added_tokens.items():
                            print(f"Adding special token: '{token_text}' with ID {token_id}")
                            # Add the token if it's not already in the tokenizer
                            if token_text not in tokenizer.get_vocab():
                                tokenizer.add_tokens([token_text])
                                print(f"Added token '{token_text}', new vocab size: {len(tokenizer)}")
                
                print(f"Reconstructed tokenizer vocab size: {len(tokenizer)}")
            
        # Set pad token properly
        if tokenizer.pad_token is None:
            # Check for common pad tokens
            if "<|endoftext|>" in tokenizer.get_vocab():
                tokenizer.pad_token = "<|endoftext|>"
            elif hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # Use the first available special token
                special_tokens_map_path = os.path.join(checkpoint_dir, "special_tokens_map.json")
                if os.path.exists(special_tokens_map_path):
                    import json
                    with open(special_tokens_map_path, 'r') as f:
                        special_tokens_map = json.load(f)
                        if 'pad_token' in special_tokens_map:
                            pad_token_info = special_tokens_map['pad_token']
                            if isinstance(pad_token_info, dict) and 'content' in pad_token_info:
                                tokenizer.pad_token = pad_token_info['content']
                            elif isinstance(pad_token_info, str):
                                tokenizer.pad_token = pad_token_info
                        print(f"Set pad token from special_tokens_map: {tokenizer.pad_token}")
        
        print(f"Final tokenizer vocab size: {len(tokenizer)} (source: {tokenizer_source})")
        print(f"Pad token: {tokenizer.pad_token}")
        
        # Get special tokens info
        special_tokens = {
            'pad_token': tokenizer.pad_token,
            'eos_token': tokenizer.eos_token,
            'bos_token': getattr(tokenizer, 'bos_token', None),
            'unk_token': getattr(tokenizer, 'unk_token', None),
        }
        print(f"Special tokens: {special_tokens}")
        
        # Check adapter config
        adapter_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            import json
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
                print(f"Adapter config: {adapter_config}")
        
        # Determine torch dtype
        torch_dtype = torch.float16 if job_config.precision.value == "fp16" else torch.float32
        if job_config.precision.value == "bf16":
            torch_dtype = torch.bfloat16
        
        # Helper function to resize model embeddings if needed
        def resize_model_embeddings_if_needed(model, tokenizer_vocab_size, model_name="model"):
            original_vocab_size = model.config.vocab_size
            if tokenizer_vocab_size != original_vocab_size:
                print(f"Vocab size mismatch detected for {model_name}!")
                print(f"  {model_name} vocab size: {original_vocab_size}")
                print(f"  Tokenizer vocab size: {tokenizer_vocab_size}")
                print(f"  Resizing {model_name} embeddings...")
                
                model.resize_token_embeddings(tokenizer_vocab_size)
                print(f"  {model_name} vocab size after resize: {model.config.vocab_size}")
                
                # Verify the resize worked
                if model.config.vocab_size != tokenizer_vocab_size:
                    raise Exception(f"Failed to resize {model_name} embeddings: expected {tokenizer_vocab_size}, got {model.config.vocab_size}")
                print(f"  Successfully resized {model_name} embeddings")
            else:
                print(f"{model_name} vocab size matches tokenizer: {original_vocab_size}")
        
        # Load base model with error handling
        try:
            print(f"Loading base model: {base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch_dtype,
                device_map=None,  # Don't use automatic device mapping to avoid multi-GPU issues
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print(f"Base model loaded successfully. Original vocab size: {base_model.config.vocab_size}")
            
        except Exception as e:
            print(f"Error loading base model: {e}")
            raise e
        
        # CRITICAL: Resize embeddings BEFORE loading adapter
        tokenizer_vocab_size = len(tokenizer)
        resize_model_embeddings_if_needed(base_model, tokenizer_vocab_size, "base_model")
        
        # Move base model to device early
        base_model = base_model.to(self.device)
        print(f"Moved base model to device: {self.device}")
        
        # Load adapter or full model
        if job_config.method.value in ["lora", "qlora"]:
            print(f"Loading LoRA adapter...")
            model = None
            
            try:
                print("Attempt 1: Standard PEFT loading")
                model = PeftModel.from_pretrained(base_model, checkpoint_dir)
                print(f"Successfully loaded LoRA adapter from {checkpoint_dir}")
                
            except Exception as e1:
                print(f"Standard LoRA loading failed: {str(e1)}")
                
                try:
                    print("Attempt 2: Loading with explicit PEFT config")
                    from peft import PeftConfig
                    
                    peft_config = PeftConfig.from_pretrained(checkpoint_dir)
                    print(f"PEFT config loaded: {peft_config}")
                    
                    model = PeftModel.from_pretrained(base_model, checkpoint_dir, config=peft_config)
                    print("Loaded LoRA adapter with explicit config")
                    
                except Exception as e2:
                    print(f"Config-based loading failed: {str(e2)}")
                    
                    try:
                        print("Attempt 3: Try loading as complete model from checkpoint")
                        # This might be a merged model saved as a complete model
                        merged_model = AutoModelForCausalLM.from_pretrained(
                            checkpoint_dir,
                            torch_dtype=torch_dtype,
                            device_map=None,  # Don't use automatic device mapping
                            trust_remote_code=True,
                            low_cpu_mem_usage=True
                        )
                        print(f"Loaded as complete model. Vocab size: {merged_model.config.vocab_size}")
                        
                        # Resize this model too if needed
                        resize_model_embeddings_if_needed(merged_model, tokenizer_vocab_size, "merged_model")
                        
                        # Move to device
                        merged_model = merged_model.to(self.device)
                        
                        model = merged_model
                        base_model = None  # Don't need base model in this case
                        print("Successfully loaded as complete merged model")
                        
                    except Exception as e3:
                        print(f"Complete model loading failed: {str(e3)}")
                        
                        try:
                            print("Attempt 4: Try loading base model and merging adapter")
                            # Last resort: load and immediately merge
                            temp_model = PeftModel.from_pretrained(base_model, checkpoint_dir)
                            model = temp_model.merge_and_unload()
                            print("Successfully loaded and merged LoRA adapter")
                            
                        except Exception as e4:
                            raise Exception(f"All loading attempts failed. Errors:\n1: {e1}\n2: {e2}\n3: {e3}\n4: {e4}")
        else:
            # For full fine-tuning, load the complete model
            print("Loading complete fine-tuned model...")
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_dir,
                torch_dtype=torch_dtype,
                device_map=None,  # Don't use automatic device mapping
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Resize if needed
            resize_model_embeddings_if_needed(model, tokenizer_vocab_size, "fine_tuned_model")
            
            # Move to device
            model = model.to(self.device)
        
        model.eval()
        
        # Ensure all model components are on the same device
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'to'):
            model.base_model = model.base_model.to(self.device)
        
        # Move the entire model to device one more time to be sure
        model = model.to(self.device)
        
        print(f"Model successfully loaded and ready for inference on device: {self.device}")
        print(f"Model device: {next(model.parameters()).device if hasattr(model, 'parameters') else 'unknown'}")
        
        model_info = {
            "model": model,
            "tokenizer": tokenizer,
            "base_model": base_model if job_config.method.value in ["lora", "qlora"] and base_model is not None else None,
            "torch_dtype": torch_dtype,
            "job_config": job_config
        }
        
        # Cache the loaded model (be careful with memory)
        self.loaded_models[job_id] = model_info
        
        return model_info
    
    def generate_response(
        self, 
        job_id: str, 
        prompt: str, 
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate a response using the loaded model"""
        
        if job_id not in self.loaded_models:
            raise ValueError(f"Model {job_id} not loaded")
            
        model_info = self.loaded_models[job_id]
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        generation_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            use_cache=True
        )
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode response
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response_text.strip()
    
    def unload_model(self, job_id: str):
        """Unload a model to free memory"""
        if job_id in self.loaded_models:
            model_info = self.loaded_models[job_id]
            del model_info["model"]
            if model_info["base_model"]:
                del model_info["base_model"]
            del self.loaded_models[job_id]
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def unload_all_models(self):
        """Unload all models to free memory"""
        for job_id in list(self.loaded_models.keys()):
            self.unload_model(job_id)


# Global inference service instance
inference_service = InferenceService() 