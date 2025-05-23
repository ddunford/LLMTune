from sqlalchemy import Column, String, Boolean, DateTime, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

Base = declarative_base()

class Settings(Base):
    """Database model for application settings"""
    __tablename__ = "settings"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, index=True, nullable=False)
    value = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    is_sensitive = Column(Boolean, default=False)  # For passwords, tokens, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class HuggingFaceAuth(Base):
    """Database model specifically for Hugging Face authentication"""
    __tablename__ = "huggingface_auth"
    
    id = Column(Integer, primary_key=True, index=True)
    token = Column(Text, nullable=False)
    username = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    last_validated = Column(DateTime(timezone=True), nullable=True)
    validation_error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

# Pydantic models for API
class SettingsCreate(BaseModel):
    key: str
    value: Optional[str] = None
    description: Optional[str] = None
    is_sensitive: bool = False

class SettingsUpdate(BaseModel):
    value: Optional[str] = None
    description: Optional[str] = None
    is_sensitive: Optional[bool] = None

class SettingsResponse(BaseModel):
    id: int
    key: str
    value: Optional[str] = None
    description: Optional[str] = None
    is_sensitive: bool = False
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class HuggingFaceAuthCreate(BaseModel):
    token: str
    username: Optional[str] = None

class HuggingFaceAuthUpdate(BaseModel):
    token: Optional[str] = None
    username: Optional[str] = None
    is_active: Optional[bool] = None

class HuggingFaceAuthResponse(BaseModel):
    id: int
    token_preview: str  # Only show first/last few characters
    username: Optional[str] = None
    is_active: bool = True
    last_validated: Optional[datetime] = None
    validation_error: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class HuggingFaceTokenValidation(BaseModel):
    token: str
    
class HuggingFaceTokenValidationResponse(BaseModel):
    valid: bool
    username: Optional[str] = None
    error: Optional[str] = None
    can_access_gated: bool = False
    organizations: list = [] 