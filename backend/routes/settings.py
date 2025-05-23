from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
import os

from models.settings import (
    Settings, HuggingFaceAuth,
    SettingsCreate, SettingsUpdate, SettingsResponse,
    HuggingFaceAuthCreate, HuggingFaceAuthUpdate, HuggingFaceAuthResponse,
    HuggingFaceTokenValidation, HuggingFaceTokenValidationResponse
)
from services.database import db_service

router = APIRouter()

@router.get("/settings", response_model=List[SettingsResponse])
async def get_all_settings():
    """Get all application settings"""
    with db_service.get_session() as db:
        settings = db.query(Settings).all()
        
        # Mask sensitive values
        result = []
        for setting in settings:
            setting_dict = {
                "id": setting.id,
                "key": setting.key,
                "value": "***" if setting.is_sensitive and setting.value else setting.value,
                "description": setting.description,
                "is_sensitive": setting.is_sensitive,
                "created_at": setting.created_at,
                "updated_at": setting.updated_at
            }
            result.append(SettingsResponse(**setting_dict))
        
        return result

@router.get("/settings/{key}", response_model=SettingsResponse)
async def get_setting(key: str):
    """Get a specific setting by key"""
    with db_service.get_session() as db:
        setting = db.query(Settings).filter(Settings.key == key).first()
        if not setting:
            raise HTTPException(status_code=404, detail="Setting not found")
        
        # Mask sensitive values
        setting_dict = {
            "id": setting.id,
            "key": setting.key,
            "value": "***" if setting.is_sensitive and setting.value else setting.value,
            "description": setting.description,
            "is_sensitive": setting.is_sensitive,
            "created_at": setting.created_at,
            "updated_at": setting.updated_at
        }
        
        return SettingsResponse(**setting_dict)

@router.post("/settings", response_model=SettingsResponse)
async def create_setting(setting: SettingsCreate):
    """Create a new setting"""
    with db_service.get_session() as db:
        # Check if setting already exists
        existing = db.query(Settings).filter(Settings.key == setting.key).first()
        if existing:
            raise HTTPException(status_code=400, detail="Setting already exists")
        
        db_setting = Settings(
            key=setting.key,
            value=setting.value,
            description=setting.description,
            is_sensitive=setting.is_sensitive
        )
        
        db.add(db_setting)
        db.commit()
        db.refresh(db_setting)
        
        # Mask sensitive values in response
        setting_dict = {
            "id": db_setting.id,
            "key": db_setting.key,
            "value": "***" if db_setting.is_sensitive and db_setting.value else db_setting.value,
            "description": db_setting.description,
            "is_sensitive": db_setting.is_sensitive,
            "created_at": db_setting.created_at,
            "updated_at": db_setting.updated_at
        }
        
        return SettingsResponse(**setting_dict)

@router.put("/settings/{key}", response_model=SettingsResponse)
async def update_setting(key: str, setting_update: SettingsUpdate):
    """Update an existing setting"""
    with db_service.get_session() as db:
        db_setting = db.query(Settings).filter(Settings.key == key).first()
        if not db_setting:
            raise HTTPException(status_code=404, detail="Setting not found")
        
        # Update only provided fields
        if setting_update.value is not None:
            db_setting.value = setting_update.value
        if setting_update.description is not None:
            db_setting.description = setting_update.description
        if setting_update.is_sensitive is not None:
            db_setting.is_sensitive = setting_update.is_sensitive
        
        db.commit()
        db.refresh(db_setting)
        
        # Mask sensitive values in response
        setting_dict = {
            "id": db_setting.id,
            "key": db_setting.key,
            "value": "***" if db_setting.is_sensitive and db_setting.value else db_setting.value,
            "description": db_setting.description,
            "is_sensitive": db_setting.is_sensitive,
            "created_at": db_setting.created_at,
            "updated_at": db_setting.updated_at
        }
        
        return SettingsResponse(**setting_dict)

@router.delete("/settings/{key}")
async def delete_setting(key: str):
    """Delete a setting"""
    with db_service.get_session() as db:
        db_setting = db.query(Settings).filter(Settings.key == key).first()
        if not db_setting:
            raise HTTPException(status_code=404, detail="Setting not found")
        
        db.delete(db_setting)
        db.commit()
        
        return {"message": f"Setting '{key}' deleted successfully"}

# Hugging Face Authentication Routes

@router.get("/huggingface/auth", response_model=List[HuggingFaceAuthResponse])
async def get_hf_auth_configs():
    """Get all Hugging Face authentication configurations"""
    with db_service.get_session() as db:
        auths = db.query(HuggingFaceAuth).all()
        
        result = []
        for auth in auths:
            # Create masked token preview
            token_preview = ""
            if auth.token:
                if len(auth.token) > 10:
                    token_preview = f"hf_***...{auth.token[-4:]}"
                else:
                    token_preview = "hf_***"
            
            auth_dict = {
                "id": auth.id,
                "token_preview": token_preview,
                "username": auth.username,
                "is_active": auth.is_active,
                "last_validated": auth.last_validated,
                "validation_error": auth.validation_error,
                "created_at": auth.created_at,
                "updated_at": auth.updated_at
            }
            result.append(HuggingFaceAuthResponse(**auth_dict))
        
        return result

@router.post("/huggingface/auth", response_model=HuggingFaceAuthResponse)
async def create_hf_auth(auth: HuggingFaceAuthCreate):
    """Create or update Hugging Face authentication"""
    with db_service.get_session() as db:
        # Deactivate existing auths if this one is being set as active
        db.query(HuggingFaceAuth).update({"is_active": False})
        
        # Validate token first
        validation_result = await validate_hf_token_internal(auth.token)
        
        db_auth = HuggingFaceAuth(
            token=auth.token,
            username=auth.username or validation_result.get("username"),
            is_active=True,
            last_validated=datetime.utcnow() if validation_result["valid"] else None,
            validation_error=validation_result.get("error") if not validation_result["valid"] else None
        )
        
        db.add(db_auth)
        db.commit()
        db.refresh(db_auth)
        
        # Create masked response
        token_preview = ""
        if db_auth.token:
            if len(db_auth.token) > 10:
                token_preview = f"hf_***...{db_auth.token[-4:]}"
            else:
                token_preview = "hf_***"
        
        auth_dict = {
            "id": db_auth.id,
            "token_preview": token_preview,
            "username": db_auth.username,
            "is_active": db_auth.is_active,
            "last_validated": db_auth.last_validated,
            "validation_error": db_auth.validation_error,
            "created_at": db_auth.created_at,
            "updated_at": db_auth.updated_at
        }
        
        return HuggingFaceAuthResponse(**auth_dict)

@router.put("/huggingface/auth/{auth_id}", response_model=HuggingFaceAuthResponse)
async def update_hf_auth(auth_id: int, auth_update: HuggingFaceAuthUpdate):
    """Update Hugging Face authentication"""
    with db_service.get_session() as db:
        db_auth = db.query(HuggingFaceAuth).filter(HuggingFaceAuth.id == auth_id).first()
        if not db_auth:
            raise HTTPException(status_code=404, detail="Authentication config not found")
        
        # Update fields
        if auth_update.token is not None:
            validation_result = await validate_hf_token_internal(auth_update.token)
            db_auth.token = auth_update.token
            db_auth.last_validated = datetime.utcnow() if validation_result["valid"] else None
            db_auth.validation_error = validation_result.get("error") if not validation_result["valid"] else None
            
        if auth_update.username is not None:
            db_auth.username = auth_update.username
            
        if auth_update.is_active is not None:
            if auth_update.is_active:
                # Deactivate other auths
                db.query(HuggingFaceAuth).filter(HuggingFaceAuth.id != auth_id).update({"is_active": False})
            db_auth.is_active = auth_update.is_active
        
        db.commit()
        db.refresh(db_auth)
        
        # Create masked response
        token_preview = ""
        if db_auth.token:
            if len(db_auth.token) > 10:
                token_preview = f"hf_***...{db_auth.token[-4:]}"
            else:
                token_preview = "hf_***"
        
        auth_dict = {
            "id": db_auth.id,
            "token_preview": token_preview,
            "username": db_auth.username,
            "is_active": db_auth.is_active,
            "last_validated": db_auth.last_validated,
            "validation_error": db_auth.validation_error,
            "created_at": db_auth.created_at,
            "updated_at": db_auth.updated_at
        }
        
        return HuggingFaceAuthResponse(**auth_dict)

@router.delete("/huggingface/auth/{auth_id}")
async def delete_hf_auth(auth_id: int):
    """Delete Hugging Face authentication"""
    with db_service.get_session() as db:
        db_auth = db.query(HuggingFaceAuth).filter(HuggingFaceAuth.id == auth_id).first()
        if not db_auth:
            raise HTTPException(status_code=404, detail="Authentication config not found")
        
        db.delete(db_auth)
        db.commit()
        
        return {"message": "Hugging Face authentication deleted successfully"}

@router.post("/huggingface/validate-token", response_model=HuggingFaceTokenValidationResponse)
async def validate_hf_token(validation: HuggingFaceTokenValidation):
    """Validate a Hugging Face token"""
    result = await validate_hf_token_internal(validation.token)
    return HuggingFaceTokenValidationResponse(**result)

async def validate_hf_token_internal(token: str) -> dict:
    """Internal function to validate HF token"""
    try:
        from huggingface_hub import whoami, HfApi
        
        # Set token in environment temporarily
        original_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        
        try:
            # Test the token
            user_info = whoami(token)
            
            # Test access to a gated model to check permissions
            api = HfApi(token=token)
            can_access_gated = False
            
            try:
                # Try to access model info for a known gated model
                api.model_info("meta-llama/Llama-2-7b-hf")
                can_access_gated = True
            except Exception:
                # Can't access gated models - that's okay
                pass
            
            return {
                "valid": True,
                "username": user_info.get("name", ""),
                "error": None,
                "can_access_gated": can_access_gated,
                "organizations": user_info.get("orgs", [])
            }
            
        finally:
            # Restore original token
            if original_token:
                os.environ["HUGGINGFACE_HUB_TOKEN"] = original_token
            else:
                os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)
                
    except Exception as e:
        return {
            "valid": False,
            "username": None,
            "error": str(e),
            "can_access_gated": False,
            "organizations": []
        }

@router.get("/huggingface/test-connection")
async def test_hf_connection():
    """Test current Hugging Face connection"""
    token = db_service.get_active_hf_token()
    
    if not token:
        return {
            "connected": False,
            "message": "No Hugging Face token configured",
            "username": None
        }
    
    validation_result = await validate_hf_token_internal(token)
    
    return {
        "connected": validation_result["valid"],
        "message": validation_result.get("error") or "Connected successfully",
        "username": validation_result.get("username"),
        "can_access_gated": validation_result.get("can_access_gated", False)
    } 