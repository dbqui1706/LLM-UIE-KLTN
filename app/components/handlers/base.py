from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import logging
import json
import time

logger = logging.getLogger(__name__)

class BaseHandler(ABC):
    """Base handler cho tất cả event handlers"""
    
    def __init__(self, context):
        self.context = context
        self.model = context.model if hasattr(context, 'model') else None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    

    def _log_operation(self, operation: str, **kwargs):
        """Log operation với details"""
        self.logger.info(f"🔄 {operation} - {kwargs}")
    
    def _create_error_response(self, error_msg: str, operation: str = "Unknown") -> Tuple[str, str]:
        """Tạo error response chuẩn"""
        full_msg = f"❌ {operation} failed: {error_msg}"
        self.logger.error(full_msg)
        return full_msg, ""
    
    def _create_success_response(self, summary: str, details: Dict[str, Any]) -> Tuple[str, str]:
        """Tạo success response chuẩn"""
        detailed_json = json.dumps(details, indent=2, ensure_ascii=False)
        return summary, detailed_json

class MockResultMixin:
    """Mixin cho tạo mock results khi model không available"""
    
    def _create_mock_extraction_result(self, text: str, task: str) -> Dict[str, Any]:
        """Tạo mock extraction result"""
        return {
            "text": text,
            "task": task,
            "entities": [
                {"entity_type": "PERSON", "entity_mention": "Sample Person", "confidence": 0.95},
                {"entity_type": "ORG", "entity_mention": "Sample Organization", "confidence": 0.90}
            ] if task in ["NER", "ALL"] else [],
            "relations": [
                {"relation_type": "WORKS_AT", "head_entity": "Sample Person", 
                 "tail_entity": "Sample Organization", "confidence": 0.88}
            ] if task in ["RE", "ALL"] else [],
            "events": [
                {"trigger": "meeting", "trigger_type": "MEETING", "arguments": []}
            ] if task in ["EE", "ALL"] else [],
            "generation_info": {
                "parameters_used": {"temperature": 0.1, "max_new_tokens": 512}
            }
        }