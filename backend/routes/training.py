from fastapi import APIRouter, HTTPException
from typing import List
import os

from models.training import (
    TrainingConfig, TrainingJob, TrainingJobResponse, 
    TrainingControlRequest, TrainingStatus
)
from train_runner import training_runner
from services.inference_service import inference_service

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
        print(f"Warning: Error cleaning up files for job {job_id}: {e}")
    
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
            "id": "mistralai/Mistral-7B-Instruct-v0.3",
            "name": "Mistral 7B Instruct v0.3",
            "description": "Efficient 7B model with great performance and function calling support",
            "size": "7B parameters",
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
            "id": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
            "name": "DeepSeek Coder 7B Instruct",
            "description": "Specialized for coding tasks, supports 80+ programming languages",
            "size": "7B parameters",
            "category": "Coding",
            "recommended": True
        },
        {
            "id": "codellama/CodeLlama-7b-Instruct-hf",
            "name": "Code Llama 7B Instruct",
            "description": "Meta's specialized coding model based on Llama 2",
            "size": "7B parameters",
            "category": "Coding",
            "recommended": True
        },
        {
            "id": "WizardLM/WizardCoder-15B-V1.0",
            "name": "WizardCoder 15B",
            "description": "Strong coding performance across multiple programming languages",
            "size": "15B parameters",
            "category": "Coding",
            "recommended": False
        },
        
        # Reasoning & Math
        {
            "id": "deepseek-ai/deepseek-math-7b-instruct",
            "name": "DeepSeek Math 7B Instruct",
            "description": "Specialized for mathematical reasoning and problem solving",
            "size": "7B parameters",
            "category": "Reasoning & Math",
            "recommended": True
        },
        {
            "id": "microsoft/phi-3-mini-4k-instruct",
            "name": "Phi-3 Mini 4K Instruct",
            "description": "Microsoft's efficient model optimized for reasoning tasks",
            "size": "3.8B parameters",
            "category": "Reasoning & Math",
            "recommended": True
        },
        
        # Small & Efficient Models
        {
            "id": "microsoft/phi-3-mini-128k-instruct",
            "name": "Phi-3 Mini 128K Instruct",
            "description": "Compact but powerful model with large context window",
            "size": "3.8B parameters",
            "category": "Small & Efficient",
            "recommended": True
        },
        {
            "id": "google/gemma-2-9b-it",
            "name": "Gemma 2 9B Instruct",
            "description": "Google's efficient open model with strong performance",
            "size": "9B parameters",
            "category": "Small & Efficient",
            "recommended": True
        },
        
        # Multilingual
        {
            "id": "Qwen/Qwen2.5-7B-Instruct",
            "name": "Qwen 2.5 7B Instruct",
            "description": "Alibaba's multilingual model with strong reasoning capabilities",
            "size": "7B parameters",
            "category": "Multilingual",
            "recommended": True
        },
        {
            "id": "bigscience/bloom-7b1",
            "name": "BLOOM 7B",
            "description": "Multilingual model supporting dozens of languages",
            "size": "7B parameters",
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
        # This would typically try to load the model/tokenizer
        # For now, just return success for valid-looking model IDs
        if "/" in model_id and len(model_id.split("/")) == 2:
            return {
                "valid": True,
                "message": "Model ID appears valid",
                "tokenizer_compatible": True
            }
        else:
            return {
                "valid": False,
                "message": "Invalid model ID format",
                "tokenizer_compatible": False
            }
    except Exception as e:
        return {
            "valid": False,
            "message": f"Error validating model: {str(e)}",
            "tokenizer_compatible": False
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

@router.get("/jobs/{job_id}/debug")
async def debug_job_status(job_id: str):
    """Debug endpoint to check job status and monitoring info"""
    job = training_runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    debug_info = {
        "job_id": job_id,
        "status": job.status,
        "started_at": job.started_at,
        "current_step": job.current_step,
        "total_steps": job.total_steps,
        "current_epoch": job.current_epoch,
        "total_epochs": job.total_epochs,
        "loss": job.loss,
        "log_file": getattr(job, 'log_file', None),
        "process_running": job_id in training_runner.processes,
        "log_file_exists": False,
        "log_file_size": 0,
        "last_log_lines": []
    }
    
    # Check log file status
    if debug_info["log_file"]:
        if os.path.exists(debug_info["log_file"]):
            debug_info["log_file_exists"] = True
            debug_info["log_file_size"] = os.path.getsize(debug_info["log_file"])
            
            # Get last few lines
            try:
                with open(debug_info["log_file"], 'r') as f:
                    lines = f.readlines()
                    debug_info["last_log_lines"] = [line.strip() for line in lines[-5:]]
            except Exception as e:
                debug_info["log_read_error"] = str(e)
    
    return debug_info

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