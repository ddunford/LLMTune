from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from enum import Enum
from datetime import datetime

class TrainingMethod(str, Enum):
    LORA = "lora"
    QLORA = "qlora"
    FULL = "full"

class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Precision(str, Enum):
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"

class LoRAConfig(BaseModel):
    rank: int = Field(default=16, ge=1, le=256, description="LoRA rank")
    alpha: int = Field(default=32, ge=1, le=512, description="LoRA alpha")
    dropout: float = Field(default=0.1, ge=0.0, le=0.5, description="LoRA dropout")
    target_modules: List[str] = Field(default=["q_proj", "v_proj"], description="Target modules for LoRA")

class TrainingConfig(BaseModel):
    # Model configuration
    base_model: str = Field(..., description="Hugging Face model ID")
    model_type: Optional[str] = Field(None, description="Model type (auto-detected)")
    
    # Training method
    method: TrainingMethod = Field(default=TrainingMethod.LORA, description="Training method")
    
    # Dataset
    dataset_path: str = Field(..., description="Path to dataset file")
    
    # Training parameters
    epochs: int = Field(default=3, ge=1, le=100, description="Number of epochs")
    learning_rate: float = Field(default=2e-4, ge=1e-6, le=1e-3, description="Learning rate")
    batch_size: int = Field(default=4, ge=1, le=64, description="Batch size")
    max_sequence_length: int = Field(default=2048, ge=128, le=4096, description="Max sequence length")
    
    # LoRA configuration (only for LoRA/QLoRA)
    lora_config: Optional[LoRAConfig] = Field(default_factory=LoRAConfig)
    
    # Compute configuration
    use_dual_gpu: bool = Field(default=True, description="Use dual GPU setup")
    precision: Precision = Field(default=Precision.FP16, description="Training precision")
    gradient_accumulation_steps: int = Field(default=1, ge=1, description="Gradient accumulation steps")
    
    # Checkpointing
    save_steps: int = Field(default=500, ge=100, description="Save checkpoint every N steps")
    
    # Validation
    validation_split: float = Field(default=0.1, ge=0.0, le=0.3, description="Validation split")

class TrainingJob(BaseModel):
    id: str = Field(..., description="Unique job ID")
    config: TrainingConfig
    status: TrainingStatus = Field(default=TrainingStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: Optional[int] = None
    loss: Optional[float] = None
    validation_loss: Optional[float] = None
    log_file: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    error_message: Optional[str] = None

class TrainingMetrics(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    epoch: int
    step: int
    loss: float
    validation_loss: Optional[float] = None
    learning_rate: float
    tokens_processed: int
    gpu_memory_used: List[float] = Field(description="Memory usage per GPU in GB")
    gpu_utilization: List[float] = Field(description="GPU utilization percentage")

class TrainingJobResponse(BaseModel):
    job: TrainingJob
    message: str

class TrainingControlRequest(BaseModel):
    action: Literal["start", "pause", "resume", "cancel", "restart"] 