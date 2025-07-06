
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

class ProcessingMode(Enum):
    """Processing modes for different performance requirements"""
    FAST = "fast"           # Fastest processing, lower quality
    BALANCED = "balanced"   # Balance between speed and quality
    ACCURATE = "accurate"   # Best quality, slower processing
    BATCH = "batch"         # Optimized for batch processing

class SupportedFormat(Enum):
    """Supported document formats"""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    MD = "md"
    TXT = "txt"
    HTML = "html"
    PPTX = "pptx"
    XLSX = "xlsx"

@dataclass
class ProcessingStats:
    """Statistics for processing operations"""
    total_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    total_pages: int = 0
    total_processing_time: float = 0.0
    average_time_per_page: float = 0.0
    peak_memory_usage: float = 0.0
    failed_files: List[str] = field(default_factory=list)

@dataclass
class DocumentResult:
    """Result of document processing"""
    file_path: str
    success: bool
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    page_count: int = 0
    error_message: Optional[str] = None