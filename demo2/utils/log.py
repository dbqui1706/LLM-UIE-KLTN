# Utility functions
import logging
from pathlib import Path
from typing import Union, Dict, Any

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('docling_pipeline.log')
        ]
    )

def validate_file_path(file_path: Union[str, Path]) -> Path:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path

def get_file_info(file_path: Path) -> Dict[str, Any]:
    """Get basic file information"""
    stat = file_path.stat()
    return {
        'name': file_path.name,
        'size': stat.st_size,
        'modified': stat.st_mtime,
        'extension': file_path.suffix.lower()
    }