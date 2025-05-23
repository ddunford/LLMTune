from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

class DatasetFormat(str, Enum):
    JSONL = "jsonl"
    CSV = "csv"
    TXT = "txt"

class DatasetStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"

class DatasetMetadata(BaseModel):
    id: str = Field(..., description="Unique dataset ID")
    filename: str = Field(..., description="Original filename")
    format: DatasetFormat = Field(..., description="Dataset format")
    size_bytes: int = Field(..., description="File size in bytes")
    num_rows: Optional[int] = Field(None, description="Number of rows/samples")
    num_tokens: Optional[int] = Field(None, description="Estimated number of tokens")
    uploaded_at: datetime = Field(default_factory=datetime.now)
    status: DatasetStatus = Field(default=DatasetStatus.UPLOADING)
    error_message: Optional[str] = None
    sample_data: Optional[List[Dict[str, Any]]] = Field(None, description="Sample rows for preview")
    columns: Optional[List[str]] = Field(None, description="Column names (for CSV)")
    
class DatasetUploadResponse(BaseModel):
    dataset: DatasetMetadata
    message: str

class DatasetListResponse(BaseModel):
    datasets: List[DatasetMetadata]
    total: int

class DatasetProcessingRequest(BaseModel):
    text_column: Optional[str] = Field(None, description="Column name containing text (for CSV)")
    instruction_column: Optional[str] = Field(None, description="Column name containing instructions")
    response_column: Optional[str] = Field(None, description="Column name containing responses")
    format_template: Optional[str] = Field(None, description="Custom format template") 