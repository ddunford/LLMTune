import psutil
import GPUtil
import pynvml
from typing import List
import logging
from models.monitoring import GPUStats, SystemStats

logger = logging.getLogger(__name__)

class GPUMonitor:
    def __init__(self):
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"Initialized GPU monitor with {self.device_count} GPUs")
        except Exception as e:
            logger.warning(f"Failed to initialize NVIDIA ML: {e}")
            self.device_count = 0
    
    def get_gpu_stats(self) -> List[GPUStats]:
        """Get current GPU statistics"""
        stats = []
        
        if self.device_count == 0:
            return stats
        
        try:
            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get basic info
                name_result = pynvml.nvmlDeviceGetName(handle)
                # Handle both string and bytes return types
                if isinstance(name_result, bytes):
                    name = name_result.decode('utf-8')
                else:
                    name = str(name_result)
                
                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used = mem_info.used / (1024**3)  # Convert to GB
                memory_total = mem_info.total / (1024**3)
                memory_percent = (mem_info.used / mem_info.total) * 100
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Power (if available)
                power_draw = None
                try:
                    power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts
                except:
                    pass
                
                # Fan speed (if available)
                fan_speed = None
                try:
                    fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                except:
                    pass
                
                stats.append(GPUStats(
                    gpu_id=i,
                    name=name,
                    memory_used=round(memory_used, 2),
                    memory_total=round(memory_total, 2),
                    memory_percent=round(memory_percent, 1),
                    utilization=gpu_util,
                    temperature=temp,
                    power_draw=power_draw,
                    fan_speed=fan_speed
                ))
                
        except Exception as e:
            logger.error(f"Error getting GPU stats: {e}")
        
        return stats
    
    def get_system_stats(self) -> SystemStats:
        """Get system-wide statistics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            ram_used = memory.used / (1024**3)  # Convert to GB
            ram_total = memory.total / (1024**3)
            ram_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_used = disk.used / (1024**3)  # Convert to GB
            disk_total = disk.total / (1024**3)
            disk_percent = (disk.used / disk.total) * 100
            
            # GPU stats
            gpu_stats = self.get_gpu_stats()
            
            return SystemStats(
                cpu_percent=round(cpu_percent, 1),
                ram_used=round(ram_used, 2),
                ram_total=round(ram_total, 2),
                ram_percent=round(ram_percent, 1),
                disk_used=round(disk_used, 2),
                disk_total=round(disk_total, 2),
                disk_percent=round(disk_percent, 1),
                gpus=gpu_stats
            )
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            raise

# Global monitor instance
gpu_monitor = GPUMonitor() 