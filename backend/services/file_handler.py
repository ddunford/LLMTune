import os
import shutil
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class FileHandler:
    """Handle file operations for datasets and checkpoints"""
    
    @staticmethod
    def ensure_directory(path: str) -> None:
        """Ensure directory exists"""
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def delete_file(file_path: str) -> bool:
        """Delete a file safely"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False
    
    @staticmethod
    def delete_directory(dir_path: str) -> bool:
        """Delete a directory and all its contents"""
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting directory {dir_path}: {e}")
            return False
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """Get file size in bytes"""
        try:
            return os.path.getsize(file_path)
        except Exception as e:
            logger.error(f"Error getting file size for {file_path}: {e}")
            return 0
    
    @staticmethod
    def list_files(directory: str, extension: str = None) -> List[str]:
        """List files in directory, optionally filtered by extension"""
        try:
            files = []
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    if extension is None or file.endswith(extension):
                        files.append(file)
            return files
        except Exception as e:
            logger.error(f"Error listing files in {directory}: {e}")
            return [] 