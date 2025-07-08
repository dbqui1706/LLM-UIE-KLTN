from typing import Tuple, List, Dict, Any
from .base import BaseHandler
import time

class ChunkExtractionHandler(BaseHandler):
    """Handler chuyên biệt cho chunk extraction"""
    
    def refresh_chunks_info(self) -> str:
        """Refresh chunks information display"""
        chunks = getattr(self.context, 'current_chunks', [])
        
        if not chunks:
            return "❌ No chunks available. Process a document first."
        
        chunk_sizes = [len(chunk['content']) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        
        return f"""
## 📊 Available Chunks: {len(chunks)}

**Statistics:**
- Average size: {avg_size:.0f} characters  
- Total characters: {sum(chunk_sizes):,}
- Size range: {min(chunk_sizes)} - {max(chunk_sizes)} chars

**Ready for batch extraction**
"""
    
    def process_chunk_extraction(self, task: str, entity_types: str, relation_types: str,
                                event_types: str, argument_types: str, mode: str, 
                                batch_size: int, aggregate_results: bool,
                                filter_duplicates: bool, *generation_params) -> Tuple[str, str]:

        chunks = getattr(self.context, 'current_chunks', [])

        if not chunks:
            return self._create_error_response("No chunks available for extraction", "Chunk Extraction")

        # Validate task for chunks
        if task.upper() not in ['NER', 'RE', 'EE']:
            return self._create_error_response("Chunk extraction only supports NER, RE, or EE tasks (not ALL)",
                                               "Chunk Extraction")

        self._log_operation("Chunk extraction",
                            num_chunks=len(chunks), task=task, mode=mode, batch_size=batch_size)

        try:
            result = self.context.extract_from_chunks(
                task=task,
                entity_types=entity_types,
                relation_types=relation_types,
                event_types=event_types,
                argument_types=argument_types,
                mode=mode,
                batch_size=batch_size,
                aggregate_results=aggregate_results,
                filter_duplicates=filter_duplicates,
                *generation_params
            )

            if "error" in result:
                return self._create_error_response(result['error'], "Chunk Extraction")

            summary = self._create_chunk_extraction_summary(result)
            return self._create_success_response(summary, result)

        except Exception as e:
            return self._create_error_response(str(e), "Chunk Extraction")
    
