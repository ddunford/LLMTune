from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import logging

from routes import training, datasets, monitoring, settings
from services.database import db_service

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Fine-Tuning UI Backend",
    description="Backend API for fine-tuning LLMs with LoRA, QLoRA, and full fine-tuning",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Starting LLM Fine-Tuning UI...")
    
    # Database initialization
    try:
        db_service.init_database()
        logger.info("Database connected successfully")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
    
    logger.info("Application started successfully")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:5173", 
        "http://localhost:55155",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173", 
        "http://127.0.0.1:55155"
    ],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directories if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Mount static files for serving uploaded datasets and checkpoints
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/checkpoints", StaticFiles(directory="checkpoints"), name="checkpoints")

# Include API routes
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["monitoring"])
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])

@app.get("/")
async def root():
    return {"message": "LLM Fine-Tuning UI Backend", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Backend is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    ) 