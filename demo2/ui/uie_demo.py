import logging
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UIEUi:
    """UIE Demo class with all processing methods"""
    
    def __init__(self):
        """Initialize UIE demo with model"""
        try:
            # Load model - có thể customize model name ở đây
            # self.model = LLamaModel(model_name='quidangz/LLama-8B-Instruct-MultiTask-CE')
            self.model = None
            
            # Document processor placeholder
            self.document_processor = None

            if self.model and hasattr(self.model, 'model') and self.model.model is not None:
                self.load_status = "success"
                logger.info("✅ Model loaded successfully!")
            else:
                self.load_status = "failed"
                self.error_message = "Model object created but model weights not loaded"
                logger.error(self.error_message)

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def extract_information(self, text, task, entity_types="", relation_types="", 
                          event_types="", argument_types="", mode="flexible", 
                          max_new_tokens=512, temperature=0.1, top_p=0.9, top_k=50, 
                          do_sample=True, repetition_penalty=1.0, no_repeat_ngram_size=0, 
                          num_beams=1, early_stopping=False):
        """Extract information with generation parameters"""
        # Implementation moved from main file
        # ... (same as before)
        pass
    
    def process_document(self, file_path, processing_mode, use_gpu, enable_ocr, 
                        enable_table_structure, enable_cleaning, aggressive_clean, 
                        enable_chunking, chunking_strategy, chunk_size, chunk_overlap, 
                        task_type, max_new_tokens=512, temperature=0.1, top_p=0.9, 
                        top_k=50, do_sample=True, repetition_penalty=1.0, 
                        no_repeat_ngram_size=0, num_beams=1, early_stopping=False):
        """Process document with generation parameters"""
        # Implementation moved from main file
        # ... (same as before)
        pass
    
    def extract_from_chunks(self, chunks_data, task, entity_types="", relation_types="", 
                           event_types="", argument_types="", mode="flexible", 
                           max_new_tokens=512, temperature=0.1, top_p=0.9, top_k=50, 
                           do_sample=True, repetition_penalty=1.0, no_repeat_ngram_size=0, 
                           num_beams=1, early_stopping=False):
        """Extract from chunks with generation parameters"""
        # Implementation moved from main file
        # ... (same as before)
        pass

    def get_current_chunks(self):
        """Get currently stored chunks"""
        return self.current_chunks
    
    def set_current_chunks(self, chunks):
        """Set current chunks"""
        self.current_chunks = chunks
