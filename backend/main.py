from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from routes import training, datasets, monitoring

app = FastAPI(
    title="LLM Fine-Tuning UI Backend",
    description="Backend API for fine-tuning LLMs with LoRA, QLoRA, and full fine-tuning",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    from train_runner import training_runner
    
    # Restore jobs from existing config files
    await training_runner.restore_jobs_from_configs()

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