from typing import Dict, Any
from .base import BaseHandler
from .text import TextExtractionHandler
from .document import DocumentProcessingHandler
from .chunk import ChunkExtractionHandler
from .preset_manager import PresetManager

class HandlerFactory:
    """Factory để tạo và quản lý handlers"""
    
    _handler_classes = {
        'text_extraction': TextExtractionHandler,
        'document_processing': DocumentProcessingHandler,
        'chunk_extraction': ChunkExtractionHandler,
        'preset_manager': PresetManager
    }
    
    @classmethod
    def create_handler(cls, handler_type: str, context) -> BaseHandler:
        if handler_type not in cls._handler_classes:
            raise ValueError(f"Unknown handler type: {handler_type}")
        
        return cls._handler_classes[handler_type](context)
    
    @classmethod
    def create_all_handlers(cls, context) -> Dict[str, BaseHandler]:
        return {
            name: handler_class(context) 
            for name, handler_class in cls._handler_classes.items()
        }

class EventHandlers:
    """Wrapper class để maintain backward compatibility"""
    
    def __init__(self, context):
        self.context = context
        self._handlers = HandlerFactory.create_all_handlers(context)
    
    def load_sample_text(self, sample_name: str) -> str:
        return self._handlers['text_extraction'].load_sample_text(sample_name)
    
    def process_text_extraction(self, *args, **kwargs):
        return self._handlers['text_extraction'].process_text_extraction(*args, **kwargs)
    
    def process_document_upload(self, *args, **kwargs):
        return self._handlers['document_processing'].process_document_upload(*args, **kwargs)
    
    def refresh_chunks_info(self) -> str:
        return self._handlers['chunk_extraction'].refresh_chunks_info()
    
    def process_chunk_extraction(self, *args, **kwargs):
        return self._handlers['chunk_extraction'].process_chunk_extraction(*args, **kwargs)
    
    def apply_generation_preset(self, *args, **kwargs):
        return self._handlers['preset_manager'].apply_generation_preset(*args, **kwargs)
    
    # Combined methods
    def process_document_and_update_chunks_info(self, *args, **kwargs):
        doc_summary, doc_json, chunks_preview = self.process_document_upload(*args, **kwargs)
        chunks_info = self.refresh_chunks_info()
        return doc_summary, doc_json, chunks_preview, chunks_info
    
    def process_chunk_extraction_combined(self, *args, **kwargs):
        return self.process_chunk_extraction(*args, **kwargs)