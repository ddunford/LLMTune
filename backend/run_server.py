#!/usr/bin/env python3

import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import uvicorn
    from main import app
    
    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8001)
        
except ImportError:
    print("uvicorn not available, trying basic server...")
    try:
        from main import app
        import threading
        from wsgiref.simple_server import make_server
        
        # Try to use a basic ASGI to WSGI adapter
        print("Starting basic development server on port 8001...")
        print("Note: This is for testing only. Install uvicorn for production.")
        
        # Simple test function
        def test_backend():
            from train_runner import training_runner
            from services.inference_service import inference_service
            
            print("Backend components loaded successfully:")
            print(f"- Training runner: {training_runner}")
            print(f"- Inference service: {inference_service}")
            print(f"- Jobs in memory: {len(training_runner.jobs)}")
            
            # Test model loading
            if training_runner.jobs:
                job_id = list(training_runner.jobs.keys())[0]
                job = training_runner.jobs[job_id]
                print(f"- Testing model loading for job: {job_id}")
                
                if job.status.value == "completed":
                    try:
                        model_info = inference_service.load_model(job_id, job.config)
                        print("✅ Model loaded successfully!")
                        print(f"- Model type: {type(model_info['model'])}")
                        print(f"- Tokenizer vocab size: {len(model_info['tokenizer'])}")
                    except Exception as e:
                        print(f"❌ Model loading failed: {e}")
                else:
                    print(f"- Job status: {job.status.value} (not ready for inference)")
        
        test_backend()
        
    except Exception as e:
        print(f"Error running backend test: {e}")
        import traceback
        traceback.print_exc() 