from typing import Tuple, List, Dict, Any
from .base import BaseHandler, MockResultMixin
import time

class ChunkExtractionHandler(BaseHandler, MockResultMixin):
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
        """Process extraction từ document chunks"""
        
        chunks = getattr(self.context, 'current_chunks', [])
        
        if not chunks:
            return self._create_error_response("No chunks available for extraction", "Chunk Extraction")
        
        self._log_operation("Chunk extraction", 
                           num_chunks=len(chunks), task=task, mode=mode, batch_size=batch_size)
        
        try:
            start_time = time.time()
            
            # Process chunks in batches
            all_results, aggregated_results = self._process_chunks_in_batches(
                chunks, task, entity_types, relation_types, event_types, 
                argument_types, mode, batch_size, aggregate_results, 
                filter_duplicates, generation_params
            )
            
            processing_time = time.time() - start_time
            
            # Build final results
            results = self._build_extraction_results(
                chunks, all_results, aggregated_results, task, mode,
                batch_size, aggregate_results, filter_duplicates,
                processing_time, generation_params
            )
            
            summary = self._create_extraction_summary(results)
            return self._create_success_response(summary, results)
            
        except Exception as e:
            return self._create_error_response(str(e), "Chunk Extraction")
    
    def _process_chunks_in_batches(self, chunks: List[Dict], task: str, entity_types: str,
                                  relation_types: str, event_types: str, argument_types: str,
                                  mode: str, batch_size: int, aggregate_results: bool,
                                  filter_duplicates: bool, generation_params: tuple):
        """Process chunks in batches"""
        
        all_results = []
        aggregated_entities = []
        aggregated_relations = []
        aggregated_events = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_num = i//batch_size + 1
            self.logger.info(f"📦 Processing batch {batch_num}: {len(batch)} chunks")
            
            for chunk in batch:
                # Extract from chunk
                if hasattr(self.context, 'extract_information') and self.context.model:
                    chunk_result = self.context.extract_information(
                        text=chunk['content'], task=task, entity_types=entity_types,
                        relation_types=relation_types, event_types=event_types,
                        argument_types=argument_types, mode=mode, *generation_params
                    )
                else:
                    chunk_result = self._create_mock_chunk_extraction_result(chunk, task)
                
                chunk_result['chunk_metadata'] = chunk['metadata']
                all_results.append(chunk_result)
                
                # Aggregate if enabled
                if aggregate_results:
                    if 'entities' in chunk_result:
                        aggregated_entities.extend(chunk_result['entities'])
                    if 'relations' in chunk_result:
                        aggregated_relations.extend(chunk_result['relations'])
                    if 'events' in chunk_result:
                        aggregated_events.extend(chunk_result['events'])
        
        # Filter duplicates
        if filter_duplicates and aggregate_results:
            aggregated_entities = self._filter_duplicates(aggregated_entities, 'entities')
            aggregated_relations = self._filter_duplicates(aggregated_relations, 'relations')
            aggregated_events = self._filter_duplicates(aggregated_events, 'events')
        
        aggregated_results = {
            'entities': aggregated_entities,
            'relations': aggregated_relations,
            'events': aggregated_events
        }
        
        return all_results, aggregated_results
    
    def _create_mock_chunk_extraction_result(self, chunk: Dict, task: str) -> Dict:
        """Create mock extraction result for a chunk"""
        chunk_id = chunk['metadata']['chunk_id']
        
        result = {'chunk_id': chunk_id, 'entities': [], 'relations': [], 'events': []}
        
        if task in ['NER', 'ALL']:
            result['entities'] = [
                {'entity_type': 'PERSON', 'entity_mention': f'Person_{chunk_id}', 'confidence': 0.95}
            ]
        
        if task in ['RE', 'ALL']:
            result['relations'] = [
                {'relation_type': 'WORKS_AT', 'head_entity': f'Person_{chunk_id}', 
                 'tail_entity': f'Org_{chunk_id}', 'confidence': 0.88}
            ]
        
        if task in ['EE', 'ALL']:
            result['events'] = [
                {'trigger': f'meeting_{chunk_id}', 'trigger_type': 'MEETING', 'confidence': 0.85}
            ]
        
        return result
    
    def _filter_duplicates(self, items: List[Dict], item_type: str) -> List[Dict]:
        """Filter duplicate items based on type"""
        seen = set()
        filtered = []
        
        for item in items:
            if item_type == 'entities':
                key = (item.get('entity_type', ''), item.get('entity_mention', ''))
            elif item_type == 'relations':
                key = (item.get('relation_type', ''), item.get('head_entity', ''), item.get('tail_entity', ''))
            elif item_type == 'events':
                key = (item.get('trigger_type', ''), item.get('trigger', ''))
            else:
                key = str(item)
            
            if key not in seen:
                seen.add(key)
                filtered.append(item)
        
        return filtered
    
    def _build_extraction_results(self, chunks: List, all_results: List, 
                                 aggregated_results: Dict, task: str, mode: str,
                                 batch_size: int, aggregate_results: bool,
                                 filter_duplicates: bool, processing_time: float,
                                 generation_params: tuple) -> Dict:
        """Build comprehensive extraction results"""
        
        return {
            'chunks_processed': len(chunks),
            'processing_time': processing_time,
            'task': task,
            'mode': mode,
            'batch_size': batch_size,
            'aggregate_results': aggregate_results,
            'filter_duplicates': filter_duplicates,
            'aggregated_results': aggregated_results,
            'per_chunk_results': all_results,
            'generation_parameters': {
                'max_new_tokens': generation_params[0] if generation_params else 512,
                'temperature': generation_params[1] if generation_params else 0.1,
                'top_p': generation_params[2] if generation_params else 0.9,
                'top_k': generation_params[3] if generation_params else 50
            },
            'performance_metrics': {
                'avg_time_per_chunk': processing_time / len(chunks),
                'chunks_per_second': len(chunks) / processing_time if processing_time > 0 else 0
            }
        }
    
    def _create_extraction_summary(self, result: Dict) -> str:
        """Create extraction summary"""
        aggregated = result['aggregated_results']
        perf = result['performance_metrics']
        
        return f"""
## 📊 Chunk Extraction Results

### 📈 Processing: {result['chunks_processed']} chunks in {result['processing_time']:.2f}s
### 🎯 Extracted: {len(aggregated['entities'])} entities, {len(aggregated['relations'])} relations, {len(aggregated['events'])} events  
### 🚀 Performance: {perf['chunks_per_second']:.1f} chunks/sec
### 🎛️ Config: {result['task']} mode, batch size {result['batch_size']}
        """.strip()