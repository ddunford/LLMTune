from fastapi import APIRouter, HTTPException
from typing import List
import os
import logging

from models.training import (
    TrainingConfig, TrainingJob, TrainingJobResponse, 
    TrainingControlRequest, TrainingStatus
)
from unsloth_runner import dual_gpu_runner as training_runner
from services.inference_service import inference_service
from services.database import db_service

router = APIRouter()

@router.post("/jobs", response_model=TrainingJobResponse)
async def create_training_job(config: TrainingConfig):
    """Create a new training job"""
    try:
        job = await training_runner.create_job(config)
        return TrainingJobResponse(
            job=job,
            message=f"Training job {job.id} created successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs")
async def list_training_jobs():
    """List all training jobs"""
    jobs = training_runner.list_jobs()
    return {"jobs": jobs}

@router.get("/jobs/{job_id}", response_model=TrainingJob)
async def get_training_job(job_id: str):
    """Get training job by ID"""
    job = training_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job

@router.post("/jobs/{job_id}/control")
async def control_training_job(job_id: str, request: TrainingControlRequest):
    """Control training job (start, pause, resume, cancel)"""
    job = training_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    try:
        if request.action == "start":
            success = await training_runner.start_job(job_id)
            if not success:
                raise HTTPException(status_code=400, detail="Failed to start training job")
            return {"message": f"Training job {job_id} started successfully"}
        
        elif request.action == "pause":
            success = await training_runner.pause_job(job_id)
            if not success:
                raise HTTPException(status_code=400, detail="Failed to pause training job")
            return {"message": f"Training job {job_id} paused successfully"}
        
        elif request.action == "resume":
            success = await training_runner.resume_job(job_id)
            if not success:
                raise HTTPException(status_code=400, detail="Failed to resume training job")
            return {"message": f"Training job {job_id} resumed successfully"}
        
        elif request.action == "cancel":
            success = await training_runner.cancel_job(job_id)
            if not success:
                raise HTTPException(status_code=400, detail="Failed to cancel training job")
            return {"message": f"Training job {job_id} cancelled successfully"}
        
        elif request.action == "restart":
            success = await training_runner.restart_job(job_id)
            if not success:
                raise HTTPException(status_code=400, detail="Failed to restart training job")
            return {"message": f"Training job {job_id} restarted successfully"}
        
        else:
            raise HTTPException(status_code=400, detail="Invalid action")
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/jobs/{job_id}")
async def delete_training_job(job_id: str):
    """Delete a training job and its artifacts"""
    job = training_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    # Cancel if running
    if job.status in [TrainingStatus.RUNNING, TrainingStatus.PAUSED]:
        await training_runner.cancel_job(job_id)
    
    # Clean up associated files
    import shutil
    
    files_removed = []
    
    try:
        # Remove config file
        config_file = f"configs/{job_id}.yaml"
        if os.path.exists(config_file):
            os.remove(config_file)
            files_removed.append(config_file)
        
        # Remove log file
        log_file = f"logs/{job_id}.log"
        if os.path.exists(log_file):
            os.remove(log_file)
            files_removed.append(log_file)
        
        # Remove checkpoint directory
        checkpoint_dir = f"checkpoints/{job_id}"
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            files_removed.append(checkpoint_dir)
            
        # Remove any test checkpoint directories (like test_small_checkpoints)
        if hasattr(job, 'output_dir') and job.output_dir and os.path.exists(job.output_dir):
            if job.output_dir != f"checkpoints/{job_id}":  # Don't double-delete
                shutil.rmtree(job.output_dir)
                files_removed.append(job.output_dir)
    
    except Exception as e:
        # Log the error but continue with job deletion
        logger = logging.getLogger(__name__)
        logger.warning(f"Error cleaning up files for job {job_id}: {e}")
    
    # Remove from memory
    if job_id in training_runner.jobs:
        del training_runner.jobs[job_id]
    
    message = f"Training job {job_id} deleted successfully"
    if files_removed:
        message += f". Cleaned up: {', '.join(files_removed)}"
    
    return {"message": message, "files_removed": files_removed}

@router.get("/models")
async def get_supported_models():
    """Get list of supported base models"""
    # This would typically come from a configuration file or database
    supported_models = [
        # General Chat & Conversation
        {
            "id": "meta-llama/Llama-3.3-70B-Instruct",
            "name": "Llama 3.3 70B Instruct",
            "description": "Meta's latest model - excellent for conversation, reasoning, and general tasks",
            "size": "70B parameters",
            "category": "General & Chat",
            "recommended": True
        },
        {
            "id": "microsoft/DialoGPT-medium",
            "name": "DialoGPT Medium",
            "description": "Specialized conversational AI model for chatbots",
            "size": "117M parameters",
            "category": "General & Chat",
            "recommended": False
        },
        
        # Coding & Development
        {
            "id": "codellama/CodeLlama-7b-Instruct-hf",
            "name": "Code Llama 7B Instruct",
            "description": "Meta's specialized coding model based on Llama 2",
            "size": "7B parameters",
            "category": "Coding",
            "recommended": True
        },
        {
            "id": "WizardLM/WizardCoder-Python-7B-V1.0",
            "name": "WizardCoder Python 7B",
            "description": "Specialized Python coding model with strong performance",
            "size": "7B parameters",
            "category": "Coding",
            "recommended": True
        },
        {
            "id": "bigcode/starcoder2-7b",
            "name": "StarCoder2 7B",
            "description": "Advanced code generation model supporting 80+ languages",
            "size": "7B parameters",
            "category": "Coding",
            "recommended": False
        },
        
        # Reasoning & Math
        {
            "id": "microsoft/phi-2",
            "name": "Phi-2",
            "description": "Microsoft's efficient model optimized for reasoning tasks",
            "size": "2.7B parameters",
            "category": "Reasoning & Math",
            "recommended": True
        },
        {
            "id": "mistralai/Mistral-7B-v0.1",
            "name": "Mistral 7B v0.1",
            "description": "Open source Mistral 7B model - excellent for general tasks (non-gated)",
            "size": "7B parameters",
            "category": "General & Chat",
            "recommended": True
        },
        
        # Small & Efficient Models
        {
            "id": "microsoft/phi-1_5",
            "name": "Phi-1.5",
            "description": "Compact but powerful model with strong performance",
            "size": "1.3B parameters",
            "category": "Small & Efficient",
            "recommended": True
        },
        {
            "id": "google/gemma-2b",
            "name": "Gemma 2B",
            "description": "Google's efficient open model with strong performance",
            "size": "2B parameters",
            "category": "Small & Efficient",
            "recommended": True
        },
        
        # Multilingual
        {
            "id": "bigscience/bloom-7b1",
            "name": "BLOOM 7B",
            "description": "Multilingual model supporting dozens of languages",
            "size": "7B parameters",
            "category": "Multilingual",
            "recommended": True
        },
        {
            "id": "facebook/xglm-7.5B",
            "name": "XGLM 7.5B",
            "description": "Cross-lingual generative model supporting 30+ languages",
            "size": "7.5B parameters",
            "category": "Multilingual",
            "recommended": False
        },
        
        # Legacy models (keeping for backward compatibility)
        {
            "id": "meta-llama/Llama-2-7b-hf",
            "name": "Llama 2 7B",
            "description": "Meta's Llama 2 7B model (legacy)",
            "size": "7B parameters",
            "category": "Legacy",
            "recommended": False
        },
        {
            "id": "tiiuae/falcon-7b",
            "name": "Falcon 7B",
            "description": "TII's Falcon 7B model (legacy)",
            "size": "7B parameters",
            "category": "Legacy",
            "recommended": False
        },
        {
            "id": "EleutherAI/gpt-neo-2.7B",
            "name": "GPT-Neo 2.7B",
            "description": "EleutherAI's GPT-Neo model (legacy)",
            "size": "2.7B parameters",
            "category": "Legacy",
            "recommended": False
        }
    ]
    
    return {"models": supported_models}

@router.post("/validate-model")
async def validate_model(model_id: str):
    """Validate if a model ID is accessible and compatible"""
    try:
        # Get stored HF token
        hf_token = db_service.get_active_hf_token()
        
        # Set token in environment if available
        original_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        
        try:
            from transformers import AutoConfig, AutoTokenizer
            from huggingface_hub import model_info
            
            # Try to get model info first (this tests basic accessibility)
            try:
                info = model_info(model_id, token=hf_token)
                model_name = info.modelId
                
                # Try to load config to test deeper accessibility
                config = AutoConfig.from_pretrained(model_id, token=hf_token)
                
                # Try to load tokenizer to test compatibility
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
                
                return {
                    "valid": True,
                    "name": model_name,
                    "description": f"✅ Model validated successfully. Architecture: {config.architectures[0] if config.architectures else 'Unknown'}",
                    "message": "Model is accessible and compatible",
                    "tokenizer_compatible": True,
                    "requires_auth": bool(hf_token),
                    "architecture": config.architectures[0] if config.architectures else None,
                    "auth_status": "authenticated" if hf_token else "public_access"
                }
                
            except Exception as model_error:
                error_str = str(model_error).lower()
                
                # Check if this is an authentication error
                if "401" in str(model_error) or "gated" in error_str or "unauthorized" in error_str:
                    if hf_token:
                        return {
                            "valid": False,
                            "message": "❌ Your Hugging Face token doesn't have access to this gated model. You may need to request access on the model's page.",
                            "tokenizer_compatible": False,
                            "requires_auth": True,
                            "error_type": "permission_denied",
                            "auth_status": "token_insufficient",
                            "access_request_url": f"https://huggingface.co/{model_id}",
                            "help_text": "Click the link above to visit the model page and request access. Once approved, you'll be able to use this model for training."
                        }
                    else:
                        return {
                            "valid": False,
                            "message": "🔐 This model is gated and requires Hugging Face authentication. Please configure your token in Settings → Hugging Face.",
                            "tokenizer_compatible": False,
                            "requires_auth": True,
                            "error_type": "authentication_required",
                            "auth_status": "no_token",
                            "access_request_url": f"https://huggingface.co/{model_id}",
                            "help_text": "This model requires both authentication and permission. First configure your HF token, then request access at the link above."
                        }
                elif "not found" in error_str or "does not exist" in error_str:
                    return {
                        "valid": False,
                        "message": f"❌ Model '{model_id}' not found. Please check the model ID.",
                        "tokenizer_compatible": False,
                        "error_type": "not_found",
                        "auth_status": hf_token and "authenticated" or "public_access"
                    }
                else:
                    return {
                        "valid": False,
                        "message": f"❌ Model validation failed: {str(model_error)}",
                        "tokenizer_compatible": False,
                        "error_type": "validation_error",
                        "auth_status": hf_token and "authenticated" or "public_access"
                    }
        
        finally:
            # Restore original token
            if original_token:
                os.environ["HUGGINGFACE_HUB_TOKEN"] = original_token
            elif hf_token:
                os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)
                
    except Exception as e:
        return {
            "valid": False,
            "message": f"❌ System error validating model: {str(e)}",
            "tokenizer_compatible": False,
            "error_type": "system_error",
            "auth_status": "unknown"
        }

@router.get("/jobs/{job_id}/logs")
async def get_training_logs(job_id: str, limit: int = 100):
    """Get training logs for a specific job"""
    job = training_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    try:
        logs = []
        log_file_path = f"logs/{job_id}.log"
        
        # Check both the job's log_file attribute and the expected path
        if hasattr(job, 'log_file') and job.log_file and os.path.exists(job.log_file):
            log_file_path = job.log_file
        elif not os.path.exists(log_file_path):
            # Return helpful message if no logs exist yet
            if job.status == TrainingStatus.PENDING:
                return {"logs": ["Job is pending - no logs available yet"], "job_id": job_id, "total_lines": 1}
            elif job.status == TrainingStatus.RUNNING:
                return {"logs": ["Job is running but no logs available yet..."], "job_id": job_id, "total_lines": 1}
            else:
                return {"logs": ["No log file found for this job"], "job_id": job_id, "total_lines": 1}
        
        # Read logs from the log file
        with open(log_file_path, 'r') as f:
            all_lines = f.readlines()
            # Return last 'limit' lines, strip whitespace
            logs = [line.rstrip() for line in all_lines[-limit:]]
            
            # Filter out empty lines
            logs = [log for log in logs if log.strip()]
        
        return {"logs": logs, "job_id": job_id, "total_lines": len(logs)}
        
    except Exception as e:
        # Return the error in a user-friendly way
        error_msg = f"Error reading logs: {str(e)}"
        return {"logs": [error_msg], "job_id": job_id, "total_lines": 1}

@router.post("/cleanup")
async def cleanup_orphaned_files():
    """Clean up orphaned config and log files that don't have corresponding jobs"""
    import glob
    
    files_removed = []
    active_job_ids = set(training_runner.jobs.keys())
    
    try:
        # Clean up orphaned config files
        config_files = glob.glob("configs/*.yaml")
        for config_file in config_files:
            job_id = os.path.basename(config_file).replace('.yaml', '')
            if job_id not in active_job_ids:
                os.remove(config_file)
                files_removed.append(config_file)
        
        # Clean up orphaned log files
        log_files = glob.glob("logs/*.log")
        for log_file in log_files:
            job_id = os.path.basename(log_file).replace('.log', '')
            if job_id not in active_job_ids:
                os.remove(log_file)
                files_removed.append(log_file)
        
        # Clean up orphaned checkpoint directories
        if os.path.exists("checkpoints"):
            for item in os.listdir("checkpoints"):
                if item == ".gitkeep":
                    continue
                checkpoint_path = os.path.join("checkpoints", item)
                if os.path.isdir(checkpoint_path) and item not in active_job_ids:
                    import shutil
                    shutil.rmtree(checkpoint_path)
                    files_removed.append(checkpoint_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")
    
    message = f"Cleanup completed. Removed {len(files_removed)} orphaned files."
    return {"message": message, "files_removed": files_removed, "count": len(files_removed)}

@router.post("/restore")
async def restore_jobs_from_configs():
    """Restore jobs from existing config files"""
    try:
        # Call the restore method
        await training_runner.restore_jobs_from_configs()
        
        # Get the current job count
        jobs = training_runner.list_jobs()
        restored_jobs = [job for job in jobs if job.status in [TrainingStatus.FAILED, TrainingStatus.COMPLETED]]
        
        message = f"Job restoration completed. Found {len(restored_jobs)} jobs from config files."
        return {
            "message": message, 
            "restored_count": len(restored_jobs),
            "jobs": [{"id": job.id, "status": job.status, "base_model": job.config.base_model} for job in restored_jobs]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error restoring jobs: {str(e)}")

@router.get("/jobs/{job_id}/download")
async def download_model_files(job_id: str, file_type: str = "adapter"):
    """Download trained model files"""
    from fastapi.responses import FileResponse
    import zipfile
    import tempfile
    
    job = training_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if job.status != TrainingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job must be completed to download model files")
    
    checkpoint_dir = f"checkpoints/{job_id}"
    if not os.path.exists(checkpoint_dir):
        raise HTTPException(status_code=404, detail="Model checkpoint directory not found")
    
    try:
        if file_type == "adapter":
            # Create ZIP with adapter files
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
                # Add adapter files
                for root, dirs, files in os.walk(checkpoint_dir):
                    for file in files:
                        if file.endswith(('.safetensors', '.bin', '.json')):
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, checkpoint_dir)
                            zipf.write(file_path, arcname)
                
                # Add training config and logs
                config_file = f"configs/{job_id}.yaml"
                if os.path.exists(config_file):
                    zipf.write(config_file, "training_config.yaml")
                
                log_file = f"logs/{job_id}.log"
                if os.path.exists(log_file):
                    zipf.write(log_file, "training_log.txt")
            
            return FileResponse(
                temp_zip.name,
                media_type="application/zip",
                filename=f"{job_id}_adapter.zip"
            )
        
        elif file_type == "config":
            # Download just the training config
            config_file = f"configs/{job_id}.yaml"
            if os.path.exists(config_file):
                return FileResponse(
                    config_file,
                    media_type="application/x-yaml",
                    filename=f"{job_id}_config.yaml"
                )
            else:
                raise HTTPException(status_code=404, detail="Config file not found")
        
        elif file_type == "logs":
            # Download training logs
            log_file = f"logs/{job_id}.log"
            if os.path.exists(log_file):
                return FileResponse(
                    log_file,
                    media_type="text/plain",
                    filename=f"{job_id}_training.log"
                )
            else:
                raise HTTPException(status_code=404, detail="Log file not found")
        
        else:
            raise HTTPException(status_code=400, detail="Invalid file_type. Use 'adapter', 'config', or 'logs'")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating download: {str(e)}")

@router.post("/jobs/{job_id}/inference")
async def test_model_inference(job_id: str, request: dict):
    """Test inference with a completed model"""
    job = training_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if job.status != TrainingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job must be completed to run inference")
    
    try:
        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 100)
        temperature = request.get("temperature", 0.7)
        top_p = request.get("top_p", 0.9)
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        # Load model if not already loaded
        try:
            model_info = inference_service.load_model(job_id, job.config)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        
        # Generate response
        try:
            response_text = inference_service.generate_response(
                job_id=job_id,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during generation: {str(e)}")
        
        return {
            "prompt": prompt,
            "response": response_text,
            "model_id": job_id,
            "base_model": job.config.base_model,
            "method": job.config.method.value,
            "parameters": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            },
            "device": inference_service.device,
            "torch_dtype": str(model_info["torch_dtype"])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

@router.post("/jobs/{job_id}/unload")
async def unload_model(job_id: str):
    """Unload a model from memory to free GPU resources"""
    try:
        inference_service.unload_model(job_id)
        return {"message": f"Model {job_id} unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error unloading model: {str(e)}")

@router.post("/inference/unload-all")
async def unload_all_models():
    """Unload all models from memory"""
    try:
        inference_service.unload_all_models()
        return {"message": "All models unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error unloading models: {str(e)}") 