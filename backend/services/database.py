import os
import sqlite3
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from models.settings import Base, Settings, HuggingFaceAuth
from contextlib import contextmanager
from typing import Generator

class DatabaseService:
    def __init__(self, db_path: str = "data/settings.db"):
        self.db_path = db_path
        
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Create SQLAlchemy engine
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,  # Set to True for SQL debugging
            connect_args={"check_same_thread": False}
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Initialize database
        self.init_db()
    
    def init_db(self):
        """Initialize database with tables"""
        Base.metadata.create_all(bind=self.engine)
        
        # Create default settings if they don't exist
        with self.get_session() as db:
            self._create_default_settings(db)
    
    def _create_default_settings(self, db: Session):
        """Create default application settings"""
        default_settings = [
            {
                "key": "app_name",
                "value": "LLM Fine-Tuning UI",
                "description": "Application name displayed in the UI",
                "is_sensitive": False
            },
            {
                "key": "max_concurrent_training_jobs",
                "value": "2",
                "description": "Maximum number of concurrent training jobs",
                "is_sensitive": False
            },
            {
                "key": "default_training_method",
                "value": "lora",
                "description": "Default training method for new jobs",
                "is_sensitive": False
            },
            {
                "key": "gpu_memory_threshold",
                "value": "0.9",
                "description": "GPU memory usage threshold (0.0-1.0)",
                "is_sensitive": False
            }
        ]
        
        for setting_data in default_settings:
            existing = db.query(Settings).filter(Settings.key == setting_data["key"]).first()
            if not existing:
                setting = Settings(**setting_data)
                db.add(setting)
        
        db.commit()
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_setting(self, key: str) -> str | None:
        """Get a setting value by key"""
        with self.get_session() as db:
            setting = db.query(Settings).filter(Settings.key == key).first()
            return setting.value if setting else None
    
    def set_setting(self, key: str, value: str, description: str = None, is_sensitive: bool = False) -> bool:
        """Set a setting value"""
        with self.get_session() as db:
            setting = db.query(Settings).filter(Settings.key == key).first()
            if setting:
                setting.value = value
                if description:
                    setting.description = description
                setting.is_sensitive = is_sensitive
            else:
                setting = Settings(
                    key=key,
                    value=value,
                    description=description,
                    is_sensitive=is_sensitive
                )
                db.add(setting)
            
            db.commit()
            return True
    
    def delete_setting(self, key: str) -> bool:
        """Delete a setting"""
        with self.get_session() as db:
            setting = db.query(Settings).filter(Settings.key == key).first()
            if setting:
                db.delete(setting)
                db.commit()
                return True
            return False
    
    def get_active_hf_token(self) -> str | None:
        """Get the active Hugging Face token"""
        with self.get_session() as db:
            auth = db.query(HuggingFaceAuth).filter(
                HuggingFaceAuth.is_active == True
            ).first()
            return auth.token if auth else None
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_session() as db:
                db.execute(text("SELECT 1"))
                return True
        except Exception:
            return False

# Global database instance
db_service = DatabaseService() 