from typing import Tuple, Dict
from .base import BaseHandler, MockResultMixin
from ...utils.sample import get_sample_texts

class TextExtractionHandler(BaseHandler, MockResultMixin):
    """Handler chuyên biệt cho text extraction"""
    
    def __init__(self, context):
        super().__init__(context)
        self.sample_texts = get_sample_texts()
    
    def load_sample_text(self, sample_name: str) -> str:
        """Load sample text khi user chọn từ dropdown"""
        if sample_name and sample_name in self.sample_texts:
            self._log_operation("Load sample text", sample=sample_name)
            return self.sample_texts[sample_name]
        return ""
    
    def process_text_extraction(self, text: str, task: str, entity_types: str, 
                               relation_types: str, event_types: str, argument_types: str, 
                               mode: str, *generation_params) -> Tuple[str, str]:
        """Xử lý text extraction với generation parameters"""
        
        # Validation
        if not text or not text.strip():
            return self._create_error_response("Please enter text to analyze", "Text Extraction")
        
        self._log_operation("Text extraction", task=task, mode=mode, text_length=len(text))
        
        try:
            # Extract information
            if hasattr(self.context, 'extract_information') and self.context.model:
                result = self.context.extract_information(
                    text=text, task=task, entity_types=entity_types,
                    relation_types=relation_types, event_types=event_types,
                    argument_types=argument_types, mode=mode, *generation_params
                )
                
                if "error" in result:
                    return self._create_error_response(result['error'], "Text Extraction")
            else:
                # Mock result for demo
                self.logger.warning("Model not available, creating mock results")
                result = self._create_mock_extraction_result(text, task)
            
            # Create summary
            summary = self._create_extraction_summary(result)
            return self._create_success_response(summary, result)
            
        except Exception as e:
            return self._create_error_response(str(e), "Text Extraction")
    
    def _create_extraction_summary(self, result: Dict) -> str:
        """Tạo summary cho extraction results"""
        parts = []
        
        if "entities" in result:
            parts.append(f"🏷️ Entities: {len(result['entities'])}")
        if "relations" in result:
            parts.append(f"🔗 Relations: {len(result['relations'])}")
        if "events" in result:
            parts.append(f"📅 Events: {len(result['events'])}")
        
        if "generation_info" in result:
            gen_info = result["generation_info"]["parameters_used"]
            parts.append(f"🎛️ Temp: {gen_info.get('temperature', 'N/A')}")
            parts.append(f"🎯 Tokens: {gen_info.get('max_new_tokens', 'N/A')}")
        
        return " | ".join(parts) if parts else "No results found"