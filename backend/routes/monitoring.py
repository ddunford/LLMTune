from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import asyncio
import json
import logging
from typing import List

from models.monitoring import SystemStats, LogEntry, MonitoringResponse
from services.gpu_monitor import gpu_monitor

router = APIRouter()
logger = logging.getLogger(__name__)

# Store active WebSocket connections
active_connections: List[WebSocket] = []

@router.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """Get current system and GPU statistics"""
    try:
        return gpu_monitor.get_system_stats()
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to get system stats", "detail": str(e)}
        )

@router.get("/logs")
async def get_recent_logs(limit: int = 100):
    """Get recent log entries"""
    # This would read from actual log files in production
    # For now, return mock data
    mock_logs = [
        LogEntry(
            level="INFO",
            message="Training started for job abc123",
            source="training"
        ),
        LogEntry(
            level="INFO", 
            message="Epoch 1/3 completed",
            source="training"
        ),
        LogEntry(
            level="WARNING",
            message="GPU temperature is high: 85°C",
            source="system"
        )
    ]
    
    return {"logs": mock_logs}

@router.websocket("/ws")
async def websocket_monitoring(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Send system stats every 2 seconds
            try:
                stats = gpu_monitor.get_system_stats()
                await websocket.send_text(json.dumps({
                    "type": "system_stats",
                    "data": stats.dict()
                }))
            except Exception as e:
                logger.error(f"Error sending stats via WebSocket: {e}")
                break
            
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)

@router.websocket("/ws/logs/{job_id}")
async def websocket_logs(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time log streaming"""
    await websocket.accept()
    
    try:
        # In production, this would tail the actual log file
        log_file_path = f"logs/{job_id}.log"
        
        while True:
            # Mock log streaming - in production, use file watching
            mock_log = LogEntry(
                level="INFO",
                message=f"Training step completed for job {job_id}",
                source="training"
            )
            
            await websocket.send_text(json.dumps({
                "type": "log_entry",
                "data": mock_log.dict()
            }))
            
            await asyncio.sleep(5)  # Send log every 5 seconds
            
    except WebSocketDisconnect:
        logger.info(f"Log WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"Log WebSocket error for job {job_id}: {e}")

async def broadcast_to_monitoring_clients(message: dict):
    """Broadcast message to all monitoring WebSocket clients"""
    if not active_connections:
        return
    
    message_text = json.dumps(message)
    disconnected = []
    
    for connection in active_connections:
        try:
            await connection.send_text(message_text)
        except:
            disconnected.append(connection)
    
    # Remove disconnected clients
    for connection in disconnected:
        active_connections.remove(connection) 