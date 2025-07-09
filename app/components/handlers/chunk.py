from typing import Tuple, List, Dict, Any
from .base import BaseHandler
import time

class ChunkExtractionHandler(BaseHandler):

    def refresh_chunks_info(self) -> str:
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
            gen_kwargs = {}
            if generation_params:
                param_names = [
                    'max_new_tokens', 'temperature', 'top_p', 'top_k', 'do_sample', 
                    'repetition_penalty', 'no_repeat_ngram_size', 'num_beams', 'early_stopping'
                ]
                
                for i, value in enumerate(generation_params):
                    if i < len(param_names):
                        gen_kwargs[param_names[i]] = value
            result = self.context.extract_from_chunks(
                task=task,
                entity_types=entity_types,
                relation_types=relation_types,
                event_types=event_types,
                argument_types=argument_types,
                mode=mode,
                aggregate_results=aggregate_results,
                filter_duplicates=filter_duplicates,
                **gen_kwargs
            )

            if "error" in result:
                return self._create_error_response(result['error'], "Chunk Extraction")

            summary = self._create_chunk_extraction_summary(result)
            return self._create_success_response(summary, result)

        except Exception as e:
            return self._create_error_response(str(e), "Chunk Extraction")

    def _create_chunk_extraction_summary(self, result: Dict) -> str:
      """Create summary for chunk extraction results"""
      
      chunks_processed = result.get('chunks_processed', 0)
      processing_time = result.get('processing_time', 0)
      
      # Get aggregated results
      aggregated = result.get('aggregated_results', {})
      
      summary_parts = [
          f"📦 Processed: {chunks_processed} chunks",
          f"⏱️ Time: {processing_time:.2f}s"
      ]
      
      # Add counts based on task
      if 'entities' in aggregated:
          count = len(aggregated['entities'])
          summary_parts.append(f"🏷️ Entities: {count}")
      
      if 'relations' in aggregated:
          count = len(aggregated['relations'])
          summary_parts.append(f"🔗 Relations: {count}")
      
      if 'events' in aggregated:
          count = len(aggregated['events'])
          summary_parts.append(f"📅 Events: {count}")
      
      return " | ".join(summary_parts)
