from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class GPUStats(BaseModel):
    gpu_id: int = Field(..., description="GPU index")
    name: str = Field(..., description="GPU name")
    memory_used: float = Field(..., description="Memory used in GB")
    memory_total: float = Field(..., description="Total memory in GB")
    memory_percent: float = Field(..., description="Memory usage percentage")
    utilization: float = Field(..., description="GPU utilization percentage")
    temperature: float = Field(..., description="GPU temperature in Celsius")
    power_draw: Optional[float] = Field(None, description="Power draw in Watts")
    fan_speed: Optional[float] = Field(None, description="Fan speed percentage")

class SystemStats(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    cpu_percent: float = Field(..., description="CPU usage percentage")
    ram_used: float = Field(..., description="RAM used in GB")
    ram_total: float = Field(..., description="Total RAM in GB")
    ram_percent: float = Field(..., description="RAM usage percentage")
    disk_used: float = Field(..., description="Disk used in GB")
    disk_total: float = Field(..., description="Total disk in GB")
    disk_percent: float = Field(..., description="Disk usage percentage")
    gpus: List[GPUStats] = Field(..., description="GPU statistics")

class LogEntry(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    level: str = Field(..., description="Log level (INFO, WARNING, ERROR)")
    message: str = Field(..., description="Log message")
    source: str = Field(..., description="Log source (training, system, etc.)")

class MonitoringResponse(BaseModel):
    system_stats: SystemStats
    recent_logs: List[LogEntry] 