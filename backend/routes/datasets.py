from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import os
import json
import pandas as pd
import uuid
from typing import List
import aiofiles
import logging
import glob
import asyncio

from models.dataset import (
    DatasetMetadata, DatasetUploadResponse, DatasetListResponse, 
    DatasetFormat, DatasetStatus, DatasetProcessingRequest
)
from services.file_handler import FileHandler

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory storage for demo (in production, use a database)
datasets_db = {}

def reload_datasets_from_disk():
    """Reload datasets from uploads directory on startup"""
    uploads_pattern = "uploads/*_*"
    uploaded_files = glob.glob(uploads_pattern)
    
    for file_path in uploaded_files:
        if file_path.endswith('.gitkeep') or not os.path.isfile(file_path):
            continue
            
        try:
            # Extract ID and filename from path like "uploads/UUID_filename"
            filename_with_id = os.path.basename(file_path)
            if '_' not in filename_with_id:
                continue
                
            dataset_id = filename_with_id.split('_')[0]
            original_filename = '_'.join(filename_with_id.split('_')[1:])
            
            # Skip if already loaded
            if dataset_id in datasets_db:
                continue
            
            # Get file stats
            file_stats = os.stat(file_path)
            file_size = file_stats.st_size
            
            # Determine format
            filename_lower = original_filename.lower()
            if filename_lower.endswith('.jsonl') or filename_lower.endswith('.jsonl.txt'):
                file_format = 'jsonl'
            elif filename_lower.endswith('.csv'):
                file_format = 'csv'
            else:
                file_format = 'txt'
            
            # Create dataset metadata
            dataset = DatasetMetadata(
                id=dataset_id,
                filename=original_filename,
                format=DatasetFormat(file_format),
                size_bytes=file_size,
                status=DatasetStatus.READY,
                uploaded_at=file_stats.st_mtime
            )
            
            # Process metadata
            try:
                # Run the async function synchronously
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                dataset = loop.run_until_complete(process_dataset_metadata(dataset, file_path))
                loop.close()
            except Exception as e:
                logger.warning(f"Could not process metadata for {file_path}: {e}")
                
            datasets_db[dataset_id] = dataset
            logger.info(f"Reloaded dataset: {original_filename} ({dataset_id})")
            
        except Exception as e:
            logger.warning(f"Could not reload dataset from {file_path}: {e}")

# Reload datasets on module import
reload_datasets_from_disk()

@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file"""
    try:
        # Validate file format - check for both extensions and content
        filename_lower = file.filename.lower()
        
        # Determine format by checking file extensions
        if filename_lower.endswith('.jsonl') or filename_lower.endswith('.jsonl.txt'):
            file_format = 'jsonl'
        elif filename_lower.endswith('.csv'):
            file_format = 'csv'
        elif filename_lower.endswith('.txt'):
            file_format = 'txt'
        else:
            # Try to extract last known extension
            file_ext = file.filename.split('.')[-1].lower()
            if file_ext not in ['jsonl', 'csv', 'txt']:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file format. Please upload .jsonl, .csv, or .txt files."
                )
            file_format = file_ext
        
        # Generate unique dataset ID
        dataset_id = str(uuid.uuid4())
        
        # Create dataset metadata
        dataset = DatasetMetadata(
            id=dataset_id,
            filename=file.filename,
            format=DatasetFormat(file_format),
            size_bytes=0,  # Will be updated after saving
            status=DatasetStatus.UPLOADING
        )
        
        # Save file
        upload_path = f"uploads/{dataset_id}_{file.filename}"
        async with aiofiles.open(upload_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
            dataset.size_bytes = len(content)
        
        # Process the file to extract metadata
        try:
            dataset = await process_dataset_metadata(dataset, upload_path)
            dataset.status = DatasetStatus.READY
        except Exception as e:
            dataset.status = DatasetStatus.ERROR
            dataset.error_message = str(e)
            logger.error(f"Error processing dataset {dataset_id}: {e}")
        
        # Store in database
        datasets_db[dataset_id] = dataset
        
        return DatasetUploadResponse(
            dataset=dataset,
            message="Dataset uploaded successfully"
        )
        
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=DatasetListResponse)
async def list_datasets():
    """List all uploaded datasets"""
    datasets = list(datasets_db.values())
    return DatasetListResponse(
        datasets=datasets,
        total=len(datasets)
    )

@router.get("/{dataset_id}", response_model=DatasetMetadata)
async def get_dataset(dataset_id: str):
    """Get dataset metadata by ID"""
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return datasets_db[dataset_id]

@router.get("/{dataset_id}/preview")
async def preview_dataset(dataset_id: str):
    """Get dataset preview with formatted sample data"""
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = datasets_db[dataset_id]
    
    try:
        # Return preview data in a format suitable for the frontend
        preview_data = {
            "samples": dataset.sample_data or [],
            "total_samples": dataset.num_rows or 0,
            "format": dataset.format.value if dataset.format else "unknown",
            "columns": getattr(dataset, 'columns', None),
            "estimated_tokens": dataset.num_tokens or 0
        }
        
        return preview_data
        
    except Exception as e:
        logger.error(f"Error getting dataset preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = datasets_db[dataset_id]
    
    # Delete file
    upload_path = f"uploads/{dataset_id}_{dataset.filename}"
    try:
        os.remove(upload_path)
    except FileNotFoundError:
        pass
    
    # Remove from database
    del datasets_db[dataset_id]
    
    return {"message": "Dataset deleted successfully"}

@router.post("/{dataset_id}/process")
async def process_dataset(dataset_id: str, request: DatasetProcessingRequest):
    """Process dataset with custom formatting options"""
    if dataset_id not in datasets_db:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    dataset = datasets_db[dataset_id]
    upload_path = f"uploads/{dataset_id}_{dataset.filename}"
    
    try:
        # Process with custom options
        processed_dataset = await process_dataset_with_options(dataset, upload_path, request)
        datasets_db[dataset_id] = processed_dataset
        
        return {"message": "Dataset processed successfully", "dataset": processed_dataset}
        
    except Exception as e:
        logger.error(f"Error processing dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_dataset_metadata(dataset: DatasetMetadata, file_path: str) -> DatasetMetadata:
    """Process dataset to extract metadata and sample data"""
    try:
        if dataset.format == DatasetFormat.JSONL:
            # Process JSONL file
            rows = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 5:  # Only read first 5 rows for sample
                        break
                    try:
                        row = json.loads(line.strip())
                        rows.append(row)
                    except json.JSONDecodeError:
                        continue
            
            dataset.sample_data = rows
            dataset.num_rows = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
            
        elif dataset.format == DatasetFormat.CSV:
            # Process CSV file
            df = pd.read_csv(file_path)
            dataset.sample_data = df.head(5).to_dict('records')
            dataset.num_rows = len(df)
            dataset.columns = df.columns.tolist()
            
        elif dataset.format == DatasetFormat.TXT:
            # Process text file
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Sample first 5 lines
            sample_lines = [{"text": line.strip()} for line in lines[:5]]
            dataset.sample_data = sample_lines
            dataset.num_rows = len(lines)
        
        # Estimate token count (rough approximation)
        if dataset.sample_data:
            avg_text_length = sum(len(str(row)) for row in dataset.sample_data) / len(dataset.sample_data)
            estimated_tokens = int((avg_text_length * dataset.num_rows) / 4)  # Rough token estimation
            dataset.num_tokens = estimated_tokens
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error processing dataset metadata: {e}")
        raise

async def process_dataset_with_options(
    dataset: DatasetMetadata, 
    file_path: str, 
    options: DatasetProcessingRequest
) -> DatasetMetadata:
    """Process dataset with custom formatting options"""
    # This would implement custom dataset processing logic
    # For now, just return the original dataset
    return dataset 