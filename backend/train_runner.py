import asyncio
import subprocess
import uuid
import logging
import os
import signal
from typing import Dict, Optional
from datetime import datetime

from models.training import TrainingJob, TrainingConfig, TrainingStatus
from config_builder import AxolotlConfigBuilder

logger = logging.getLogger(__name__)

class TrainingRunner:
    """Manages training jobs and Axolotl subprocess execution"""
    
    def __init__(self):
        self.jobs: Dict[str, TrainingJob] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self._log_files: Dict[str, open] = {}
        
        # Load existing jobs from config files on startup (schedule for later execution)
        self._restore_on_startup = True
    
    async def restore_jobs_from_configs(self):
        """Restore jobs from existing config files"""
        import glob
        import yaml
        
        try:
            config_files = glob.glob("configs/*.yaml")
            for config_file in config_files:
                try:
                    job_id = os.path.basename(config_file).replace('.yaml', '')
                    
                    # Skip if job already exists in memory
                    if job_id in self.jobs:
                        continue
                    
                    # Load the config file
                    with open(config_file, 'r') as f:
                        config_data = yaml.safe_load(f)
                    
                    # Reconstruct TrainingConfig from YAML
                    training_config = self._reconstruct_training_config(config_data)
                    
                    # Create job with restored config
                    job = TrainingJob(
                        id=job_id,
                        config=training_config,
                        status=TrainingStatus.FAILED,  # Assume failed since process is gone
                        log_file=f"logs/{job_id}.log",
                        checkpoint_dir=f"checkpoints/{job_id}"
                    )
                    
                    # Check if log file exists to determine status
                    if os.path.exists(f"logs/{job_id}.log"):
                        # Try to determine final status from log
                        job.status = self._determine_job_status_from_log(job_id)
                    
                    self.jobs[job_id] = job
                    logger.info(f"Restored job {job_id} from config file")
                    
                except Exception as e:
                    logger.error(f"Failed to restore job from {config_file}: {e}")
            
            logger.info(f"Restored {len(config_files)} jobs from config files")
            
        except Exception as e:
            logger.error(f"Error restoring jobs from configs: {e}")
    
    def _reconstruct_training_config(self, config_data: dict) -> TrainingConfig:
        """Reconstruct TrainingConfig from Axolotl YAML config"""
        from models.training import TrainingMethod, Precision, LoRAConfig
        
        # Extract original training parameters from the config
        method = TrainingMethod.LORA if 'adapter' in config_data else TrainingMethod.FULL
        if config_data.get('load_in_4bit') or config_data.get('bnb_4bit_quant_type'):
            method = TrainingMethod.QLORA
        
        # Extract dataset path
        dataset_path = config_data.get('datasets', [{}])[0].get('path', 'unknown')
        
        # Extract precision
        precision = Precision.FP16
        if config_data.get('bf16'):
            precision = Precision.BF16
        elif config_data.get('fp16') is False and config_data.get('bf16') is False:
            precision = Precision.FP32
        
        # Extract LoRA config if present
        lora_config = None
        if method in [TrainingMethod.LORA, TrainingMethod.QLORA]:
            lora_config = LoRAConfig(
                rank=config_data.get('lora_r', 16),
                alpha=config_data.get('lora_alpha', 32),
                dropout=config_data.get('lora_dropout', 0.1),
                target_modules=config_data.get('lora_target_modules', ["q_proj", "v_proj"])
            )
        
        return TrainingConfig(
            base_model=config_data.get('base_model', 'unknown'),
            method=method,
            dataset_path=dataset_path,
            epochs=config_data.get('num_epochs', 3),
            learning_rate=config_data.get('learning_rate', 2e-4),
            batch_size=config_data.get('micro_batch_size', 4),
            max_sequence_length=config_data.get('sequence_len', 2048),
            lora_config=lora_config,
            use_dual_gpu=False,  # Default to safe setting
            precision=precision,
            gradient_accumulation_steps=config_data.get('gradient_accumulation_steps', 1)
        )
    
    def _determine_job_status_from_log(self, job_id: str) -> TrainingStatus:
        """Determine job status by reading the log file"""
        log_file = f"logs/{job_id}.log"
        
        if not os.path.exists(log_file):
            return TrainingStatus.FAILED
        
        try:
            with open(log_file, 'r') as f:
                log_content = f.read().lower()
                
            # Check for completion indicators
            if 'training completed' in log_content or 'training finished' in log_content:
                return TrainingStatus.COMPLETED
            elif 'error' in log_content or 'traceback' in log_content:
                return TrainingStatus.FAILED
            else:
                # If no clear completion or error, assume failed (since process is gone)
                return TrainingStatus.FAILED
                
        except Exception:
            return TrainingStatus.FAILED
    
    async def create_job(self, config: TrainingConfig) -> TrainingJob:
        """Create a new training job"""
        job_id = str(uuid.uuid4())[:8]  # Short job ID
        
        job = TrainingJob(
            id=job_id,
            config=config,
            status=TrainingStatus.PENDING,
            total_epochs=config.epochs
        )
        
        self.jobs[job_id] = job
        logger.info(f"Created training job {job_id}")
        
        return job
    
    async def start_job(self, job_id: str) -> bool:
        """Start a training job"""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.jobs[job_id]
        
        if job.status != TrainingStatus.PENDING:
            raise ValueError(f"Job {job_id} is not in pending status")
        
        try:
            # Build Axolotl configuration
            config_path = AxolotlConfigBuilder.build_and_save(job.config, job_id)
            logger.info(f"Built config for job {job_id}: {config_path}")
            
            # Create log file
            log_file_path = f"logs/{job_id}.log"
            os.makedirs("logs", exist_ok=True)
            job.log_file = log_file_path
            
            # Start Axolotl training process
            cmd = [
                "python", "-m", "axolotl.cli.train",
                config_path
            ]
            
            # Set environment variables for CUDA
            env = os.environ.copy()
            if job.config.use_dual_gpu:
                env["CUDA_VISIBLE_DEVICES"] = "0,1"
            else:
                env["CUDA_VISIBLE_DEVICES"] = "0"
            
            # Start process
            log_file = open(log_file_path, 'w')
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid,  # Create new process group
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            self.processes[job_id] = process
            
            # Store the log file handle so we can close it later
            self._log_files[job_id] = log_file
            
            # Update job status
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now()
            
            logger.info(f"Started training job {job_id} with PID {process.pid}")
            
            # Start monitoring task
            asyncio.create_task(self._monitor_job(job_id))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start job {job_id}: {e}")
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            return False
    
    async def pause_job(self, job_id: str) -> bool:
        """Pause a running training job"""
        if job_id not in self.jobs or job_id not in self.processes:
            return False
        
        job = self.jobs[job_id]
        process = self.processes[job_id]
        
        if job.status != TrainingStatus.RUNNING:
            return False
        
        try:
            # Send SIGSTOP to pause the process
            os.killpg(os.getpgid(process.pid), signal.SIGSTOP)
            job.status = TrainingStatus.PAUSED
            logger.info(f"Paused job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to pause job {job_id}: {e}")
            return False
    
    async def resume_job(self, job_id: str) -> bool:
        """Resume a paused training job"""
        if job_id not in self.jobs or job_id not in self.processes:
            return False
        
        job = self.jobs[job_id]
        process = self.processes[job_id]
        
        if job.status != TrainingStatus.PAUSED:
            return False
        
        try:
            # Send SIGCONT to resume the process
            os.killpg(os.getpgid(process.pid), signal.SIGCONT)
            job.status = TrainingStatus.RUNNING
            logger.info(f"Resumed job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to resume job {job_id}: {e}")
            return False
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        if job_id in self.processes:
            process = self.processes[job_id]
            try:
                # Terminate the process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                
                # Wait for termination with timeout
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate gracefully
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait()
                
                del self.processes[job_id]
                logger.info(f"Terminated process for job {job_id}")
                
            except Exception as e:
                logger.error(f"Failed to terminate process for job {job_id}: {e}")
        
        job.status = TrainingStatus.CANCELLED
        job.completed_at = datetime.now()
        
        return True
    
    async def restart_job(self, job_id: str) -> bool:
        """Restart a failed training job"""
        if job_id not in self.jobs:
            return False
        
        job = self.jobs[job_id]
        
        # Only allow restarting failed jobs
        if job.status != TrainingStatus.FAILED:
            return False
        
        try:
            # Clean up any existing process (shouldn't be any for failed jobs)
            if job_id in self.processes:
                del self.processes[job_id]
            
            # Reset job state
            job.status = TrainingStatus.PENDING
            job.started_at = None
            job.completed_at = None
            job.current_step = 0
            job.current_epoch = 0
            job.loss = None
            job.validation_loss = None
            job.error_message = None
            
            # Clean up any partial checkpoint outputs from failed run
            if job.checkpoint_dir and os.path.exists(job.checkpoint_dir):
                import shutil
                try:
                    shutil.rmtree(job.checkpoint_dir)
                    logger.info(f"Cleaned up partial checkpoints for job {job_id}")
                except Exception as e:
                    logger.warning(f"Failed to clean up checkpoints for job {job_id}: {e}")
            
            logger.info(f"Restarted job {job_id} - status reset to pending")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart job {job_id}: {e}")
            return False
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def list_jobs(self) -> list[TrainingJob]:
        """List all jobs"""
        return list(self.jobs.values())
    
    async def _monitor_job(self, job_id: str):
        """Monitor a training job's progress"""
        if job_id not in self.processes:
            return
        
        process = self.processes[job_id]
        job = self.jobs[job_id]
        
        try:
            # Monitor job by parsing logs in the background
            asyncio.create_task(self._parse_training_logs(job_id))
            
            # Wait for process to complete
            return_code = await asyncio.create_task(
                self._wait_for_process(process)
            )
            
            # Update job status based on return code
            if return_code == 0:
                job.status = TrainingStatus.COMPLETED
                job.completed_at = datetime.now()
                logger.info(f"Job {job_id} completed successfully")
            else:
                job.status = TrainingStatus.FAILED
                job.completed_at = datetime.now()
                job.error_message = f"Process exited with code {return_code}"
                logger.error(f"Job {job_id} failed with exit code {return_code}")
            
            # Clean up process reference
            if job_id in self.processes:
                del self.processes[job_id]
                
            # Clean up log file handle
            if job_id in self._log_files:
                try:
                    self._log_files[job_id].close()
                    del self._log_files[job_id]
                except Exception as e:
                    logger.error(f"Error closing log file for job {job_id}: {e}")
                
        except Exception as e:
            logger.error(f"Error monitoring job {job_id}: {e}")
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            
            # Clean up on error
            if job_id in self.processes:
                del self.processes[job_id]
            if job_id in self._log_files:
                try:
                    self._log_files[job_id].close()
                    del self._log_files[job_id]
                except Exception:
                    pass
    
    async def _parse_training_logs(self, job_id: str):
        """Parse training logs to extract progress information"""
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
                        
                        # Parse for common training metrics with multiple patterns
                        # Axolotl patterns: {'loss': 1.234, 'learning_rate': 0.0002, 'epoch': 1.0}
                        # HuggingFace patterns: {'train_loss': 1.234, 'epoch': 1.0, 'step': 100}
                        
                        # Extract step/global_step
                        step_patterns = [
                            r"'step': (\d+)",
                            r"'global_step': (\d+)",
                            r"(\d+)/(\d+) \[",  # Progress bar format
                            r"Step (\d+)/"
                        ]
                        
                        for pattern in step_patterns:
                            matches = re.findall(pattern, new_lines)
                            if matches:
                                if len(matches[0]) == 2:  # Progress bar format
                                    job.current_step = int(matches[-1][0])
                                    if job.total_steps is None:
                                        job.total_steps = int(matches[-1][1])
                                else:
                                    job.current_step = int(matches[-1])
                                break
                        
                        # Extract loss
                        loss_patterns = [
                            r"'loss': ([\d.]+)",
                            r"'train_loss': ([\d.]+)",
                            r"loss=([\d.]+)",
                            r"Loss: ([\d.]+)"
                        ]
                        
                        for pattern in loss_patterns:
                            matches = re.findall(pattern, new_lines)
                            if matches:
                                job.loss = float(matches[-1])
                                break
                        
                        # Extract epoch
                        epoch_patterns = [
                            r"'epoch': ([\d.]+)",
                            r"Epoch (\d+)/",
                            r"epoch=(\d+)"
                        ]
                        
                        for pattern in epoch_patterns:
                            matches = re.findall(pattern, new_lines)
                            if matches:
                                job.current_epoch = int(float(matches[-1]))
                                break
                        
                        # Extract learning rate
                        lr_patterns = [
                            r"'learning_rate': ([\d.e-]+)",
                            r"lr=([\d.e-]+)"
                        ]
                        
                        for pattern in lr_patterns:
                            matches = re.findall(pattern, new_lines)
                            if matches:
                                # Store learning rate if needed
                                break
                        
                        # Estimate total steps if not found
                        if job.total_steps is None and job.current_step > 0:
                            # Rough estimate: steps_per_epoch * total_epochs
                            if job.current_epoch > 0:
                                estimated_steps_per_epoch = job.current_step / job.current_epoch
                                job.total_steps = int(estimated_steps_per_epoch * job.total_epochs)
                            
                await asyncio.sleep(2)  # Check logs every 2 seconds
                
            except Exception as e:
                logger.error(f"Error parsing logs for job {job_id}: {e}")
                await asyncio.sleep(5)
    
    async def _wait_for_process(self, process: subprocess.Popen) -> int:
        """Async wrapper for process.wait()"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, process.wait)

# Global training runner instance
training_runner = TrainingRunner() 