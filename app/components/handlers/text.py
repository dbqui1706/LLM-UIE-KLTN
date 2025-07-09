from typing import Tuple, Dict
from .base import BaseHandler, MockResultMixin
from ...utils.sample import get_sample_texts
from .utils import create_extraction_summary

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
        # Validation
        if not text or not text.strip():
            return self._create_error_response("Please enter text to analyze", "Text Extraction")
        
        self._log_operation("Text extraction", task=task, mode=mode, text_length=len(text))
        
        try:
            gen_param_names = [
                'max_new_tokens', 'temperature', 'top_p', 'top_k', 'do_sample',
                'repetition_penalty', 'no_repeat_ngram_size', 'num_beams', 'early_stopping'
            ]

            # Build kwargs dict from generation parameters
            gen_kwargs = {}
            for i, param_name in enumerate(gen_param_names):
                if i < len(generation_params):
                    gen_kwargs[param_name] = generation_params[i]
            result = None
            result = self.context.extract_information(
                text=text, task=task, entity_types=entity_types,
                relation_types=relation_types, event_types=event_types,
                argument_types=argument_types, mode=mode, **gen_kwargs
            )
                
            if "error" in result:
                return self._create_error_response(result['error'], "Text Extraction")
            
            # Create summary
            summary = create_extraction_summary(result)
            return self._create_success_response(summary, result)
            
        except Exception as e:
            return self._create_error_response(str(e), "Text Extraction")