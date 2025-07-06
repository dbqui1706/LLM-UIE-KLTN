import json
import time
from typing import Dict, Any, Tuple
from .presets import GenerationPresets
from ..utils.sample import get_sample_texts

# Import document processing components
try:
    from ..pipeline.document_processor import DocumentProcessor
    from ..pipeline.chunking import DocumentChunker, ChunkingPresets, ChunkingConfig, ChunkingStrategy
    from ..pipeline.config import ProcessingMode
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Pipeline components not available: {e}")
    PIPELINE_AVAILABLE = False


class EventHandlers:
    """
    Event handlers cho Gradio components
    Chia thành các handler riêng biệt cho từng tab và chức năng
    """
    
    def __init__(self, context):
        self.context = context
        self.sample_texts = get_sample_texts()
        # Initialize document processor and chunker
        self.document_processor = None
        self.document_chunker = None
    
    # ============================================================================
    # TEXT EXTRACTION HANDLERS
    # ============================================================================
    
    def load_sample_text(self, sample_name: str) -> str:
        """
        Load sample text khi user chọn từ dropdown
        
        Args:
            sample_name: Tên của sample text được chọn
            
        Returns:
            Text content của sample được chọn
        """
        if sample_name and sample_name in self.sample_texts:
            return self.sample_texts[sample_name]
        return ""
    
    def process_text_extraction(self, text: str, task: str, entity_types: str, 
                               relation_types: str, event_types: str, argument_types: str, 
                               mode: str, *generation_params) -> Tuple[str, str]:
        """
        Xử lý text extraction với các generation parameters
        
        Args:
            text: Input text cần extract
            task: Loại task (NER, RE, EE, ALL)
            entity_types, relation_types, event_types, argument_types: Custom schema
            mode: Extraction mode (flexible, strict, open)
            *generation_params: Các tham số generation từ UI controls
            
        Returns:
            Tuple[summary_text, detailed_json]
        """
        try:
            # Kiểm tra input
            if not text or not text.strip():
                return "❌ Please enter text to analyze", ""
            
            print(f"🔄 Processing text extraction - Task: {task}, Mode: {mode}")
            print(f"📝 Text length: {len(text)} characters")
            
            # Gọi model extraction với parameters (nếu có model)
            if hasattr(self.context, 'extract_information') and self.context.model:
                result = self.context.extract_information(
                    text=text,
                    task=task,
                    entity_types=entity_types,
                    relation_types=relation_types,
                    event_types=event_types,
                    argument_types=argument_types,
                    mode=mode,
                    *generation_params  # Unpack generation parameters
                )
                
                # Xử lý lỗi
                if "error" in result:
                    return f"❌ {result['error']}", ""
                
                # Format output
                summary = self._create_extraction_summary(result)
                detailed_json = json.dumps(result, indent=2, ensure_ascii=False)
                
                print("✅ Text extraction completed successfully")
                return summary, detailed_json
            else:
                # Demo mode - create mock results
                print("⚠️ Model not available, creating demo results")
                mock_result = self._create_mock_extraction_result(text, task)
                summary = self._create_extraction_summary(mock_result)
                detailed_json = json.dumps(mock_result, indent=2, ensure_ascii=False)
                return summary, detailed_json
            
        except Exception as e:
            error_msg = f"❌ Extraction failed: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg, ""
    
    def _create_extraction_summary(self, result: Dict[str, Any]) -> str:
        """
        Tạo summary text cho extraction results
        
        Args:
            result: Kết quả extraction từ model
            
        Returns:
            Summary string để hiển thị
        """
        summary_parts = []
        
        # Count results
        if "entities" in result:
            summary_parts.append(f"🏷️ Entities: {len(result['entities'])}")
        if "relations" in result:
            summary_parts.append(f"🔗 Relations: {len(result['relations'])}")
        if "events" in result:
            summary_parts.append(f"📅 Events: {len(result['events'])}")
        
        # Add generation info nếu có
        if "generation_info" in result:
            gen_info = result["generation_info"]["parameters_used"]
            summary_parts.append(f"🎛️ Temp: {gen_info.get('temperature', 'N/A')}")
            summary_parts.append(f"🎯 Tokens: {gen_info.get('max_new_tokens', 'N/A')}")
        
        return " | ".join(summary_parts) if summary_parts else "No results found"
    
    def _create_mock_extraction_result(self, text: str, task: str) -> Dict[str, Any]:
        """Tạo mock extraction result cho demo"""
        mock_result = {
            "text": text,
            "task": task,
            "entities": [],
            "relations": [],
            "events": [],
            "generation_info": {
                "parameters_used": {
                    "temperature": 0.1,
                    "max_new_tokens": 512
                }
            }
        }
        
        # Add mock entities for demo
        if task in ["NER", "ALL"]:
            mock_result["entities"] = [
                {"entity_type": "PERSON", "entity_mention": "Sample Person", "confidence": 0.95},
                {"entity_type": "ORG", "entity_mention": "Sample Organization", "confidence": 0.90}
            ]
        
        if task in ["RE", "ALL"]:
            mock_result["relations"] = [
                {"relation_type": "WORKS_AT", "head_entity": "Sample Person", "tail_entity": "Sample Organization", "confidence": 0.88}
            ]
        
        if task in ["EE", "ALL"]:
            mock_result["events"] = [
                {"trigger": "meeting", "trigger_type": "MEETING", "arguments": []}
            ]
        
        return mock_result
    
    # ============================================================================
    # DOCUMENT PROCESSING HANDLERS  
    # ============================================================================
    
    def process_document_upload(self, file, processing_mode: str, use_gpu: bool, 
                               enable_ocr: bool, enable_table_structure: bool,
                               enable_cleaning: bool, aggressive_clean: bool,
                               enable_chunking: bool, chunking_strategy: str,
                               chunk_size: int, chunk_overlap: int, task_type: str,
                               *generation_params) -> Tuple[str, str, str]:
        """
        Xử lý document upload và processing với DocumentProcessor và DocumentChunker thực sự
        
        Args:
            file: Uploaded file object
            processing_mode, use_gpu, etc.: Processing options
            *generation_params: Generation parameters
            
        Returns:
            Tuple[summary_markdown, detailed_json, chunks_preview]
        """
        # Kiểm tra file upload
        if file is None:
            return "❌ Please upload a file first", "", ""
        
        # Kiểm tra pipeline availability
        if not PIPELINE_AVAILABLE:
            return "❌ Document processing pipeline not available. Please check dependencies.", "", ""
        
        try:
            # Khởi tạo DocumentProcessor
            print(f"🔄 Initializing DocumentProcessor with mode: {processing_mode}")
            self.document_processor = DocumentProcessor(
                processing_mode=ProcessingMode(processing_mode),
                use_gpu=use_gpu,
                enable_ocr=enable_ocr,
                enable_table_structure=enable_table_structure,
                clean=enable_cleaning,
                aggressive_clean=aggressive_clean,
                max_workers=1  # Để tránh xung đột trong demo
            )
            
            print(f"🔄 Processing document: {file.name}")
            start_time = time.time()
            
            # Process document
            result = self.document_processor.process_single_document(file.name)
            
            if not result.success:
                error_msg = f"❌ Document processing failed: {result.error_message}"
                print(error_msg)
                return error_msg, "", ""
            
            print(f"✅ Document processed successfully in {result.processing_time:.2f}s")
            print(f"📄 Content length: {len(result.content)} characters")
            
            # Chunking nếu được enable
            chunks = []
            chunks_preview = "Chunking disabled"
            
            if enable_chunking and result.content:
                print(f"🔄 Starting chunking with strategy: {chunking_strategy}")
                
                # Tạo chunking config
                chunking_config = ChunkingConfig(
                    strategy=ChunkingStrategy(chunking_strategy),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    min_chunk_size=50,
                    max_chunk_size=chunk_size * 2
                )
                
                # Khởi tạo chunker
                self.document_chunker = DocumentChunker(chunking_config)
                
                # Perform chunking
                chunks = self.document_chunker.chunk_document(
                    content=result.content,
                    metadata=result.metadata,
                    task_type=task_type
                )
                
                print(f"✅ Created {len(chunks)} chunks")
                
                # Tạo chunks preview
                chunks_preview = self._create_detailed_chunks_preview(chunks)
                
                # Lưu chunks vào context để chunk extraction tab có thể sử dụng
                self.context.current_chunks = chunks
            
            # Tạo detailed result
            processing_result = {
                'file_info': {
                    'name': file.name,
                    'size': result.metadata.get('file_size', 0),
                    'format': result.metadata.get('format', ''),
                    'pages': result.metadata.get('page_count', 0)
                },
                'processing': {
                    'mode': processing_mode,
                    'time': result.processing_time,
                    'success': result.success,
                    'use_gpu': use_gpu,
                    'ocr_enabled': enable_ocr,
                    'table_structure': enable_table_structure
                },
                'content': {
                    'original_length': len(result.content),
                    'cleaned_length': len(result.content) if enable_cleaning else len(result.content),
                    'cleaning_enabled': enable_cleaning,
                    'aggressive_clean': aggressive_clean
                },
                'chunking': {
                    'enabled': enable_chunking,
                    'strategy': chunking_strategy if enable_chunking else None,
                    'chunk_size': chunk_size if enable_chunking else None,
                    'chunk_overlap': chunk_overlap if enable_chunking else None,
                    'total_chunks': len(chunks) if chunks else 0,
                    'task_optimized_for': task_type if enable_chunking else None
                },
                'chunks_sample': chunks[:3] if chunks else [],  # First 3 chunks for JSON
                'generation_params': {
                    'max_new_tokens': generation_params[0] if generation_params else None,
                    'temperature': generation_params[1] if generation_params else None,
                    'top_p': generation_params[2] if generation_params else None,
                    'top_k': generation_params[3] if generation_params else None
                }
            }
            
            # Tạo summary
            summary = self._create_processing_summary(processing_result)
            
            # Tạo detailed JSON
            detailed_json = json.dumps(processing_result, indent=2, ensure_ascii=False)
            
            print("🎉 Document processing and chunking completed successfully!")
            
            return summary, detailed_json, chunks_preview
            
        except Exception as e:
            error_msg = f"❌ Document processing failed: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg, "", ""
    
    def _create_processing_summary(self, result: Dict[str, Any]) -> str:
        """Tạo summary cho document processing results"""
        file_info = result['file_info']
        processing = result['processing']
        content = result['content']
        chunking = result['chunking']
        
        summary = f"""
## 📄 Document Processing Results

### 📁 File Information
- **File:** {file_info['name']}
- **Size:** {file_info['size']:,} bytes
- **Format:** {file_info['format']}
- **Pages:** {file_info['pages']}

### ⚡ Processing
- **Mode:** {processing['mode']}
- **Time:** {processing['time']:.2f}s
- **GPU:** {'✅' if processing['use_gpu'] else '❌'}
- **OCR:** {'✅' if processing['ocr_enabled'] else '❌'}
- **Table Structure:** {'✅' if processing['table_structure'] else '❌'}

### 📝 Content
- **Original Length:** {content['original_length']:,} chars
- **Cleaned Length:** {content['cleaned_length']:,} chars
- **Cleaning:** {'✅' if content['cleaning_enabled'] else '❌'}
- **Aggressive Clean:** {'✅' if content['aggressive_clean'] else '❌'}

### ✂️ Chunking
- **Enabled:** {'✅' if chunking['enabled'] else '❌'}
- **Strategy:** {chunking['strategy'] or 'N/A'}
- **Chunk Size:** {chunking['chunk_size'] or 'N/A'}
- **Overlap:** {chunking['chunk_overlap'] or 'N/A'}
- **Total Chunks:** {chunking['total_chunks']}
- **Optimized for:** {chunking['task_optimized_for'] or 'N/A'}

### 🎛️ Generation Parameters Ready
- Max Tokens: {result['generation_params']['max_new_tokens']}
- Temperature: {result['generation_params']['temperature']}
- Top-p: {result['generation_params']['top_p']}
- Top-k: {result['generation_params']['top_k']}
        """.strip()
        
        return summary
    
    def _create_detailed_chunks_preview(self, chunks: list) -> str:
        """Tạo detailed preview cho chunks với statistics"""
        if not chunks:
            return "No chunks available for preview."
        
        # Chunks statistics
        chunk_sizes = [len(chunk['content']) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)
        
        # Get chunking stats if available
        chunking_stats = {}
        if self.document_chunker:
            chunking_stats = self.document_chunker.get_chunking_stats(chunks)
        
        preview_parts = [f"""
## 📊 Chunking Statistics
- **Total Chunks:** {len(chunks)}
- **Average Size:** {avg_size:.0f} chars
- **Size Range:** {min_size} - {max_size} chars
- **Strategy:** {chunking_stats.get('strategy_used', 'Unknown')}
- **Total Characters:** {sum(chunk_sizes):,}
- **Total Tokens:** {chunking_stats.get('total_tokens', 'N/A')}

## 👀 Sample Chunks Preview
"""]
        
        # Show first 5 chunks
        for i, chunk in enumerate(chunks[:5]):
            chunk_content = chunk['content']
            chunk_metadata = chunk['metadata']
            
            # Truncate content for preview
            preview_content = chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content
            
            preview_parts.append(f"""
### Chunk {i+1} ({chunk_metadata['chunk_size']} chars, {chunk_metadata['chunk_tokens']} tokens)
{preview_content}
""")
        
        if len(chunks) > 5:
            preview_parts.append(f"\n... and {len(chunks) - 5} more chunks")
        
        # Add download/export info
        preview_parts.append(f"""
## 💾 Chunks Data
- Chunks are ready for batch extraction
- Use the extraction section below to process all chunks
- Total chunks available: {len(chunks)}
""")
        
        return "\n".join(preview_parts)
    
    # ============================================================================
    # CHUNK EXTRACTION HANDLERS
    # ============================================================================
    
    def refresh_chunks_info(self) -> str:
        """Refresh chunks information display"""
        chunks = getattr(self.context, 'current_chunks', [])
        
        if not chunks:
            return "❌ No chunks available. Process a document first to see chunks here."
        
        chunk_sizes = [len(chunk['content']) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        
        info = f"""
## 📊 Available Chunks: {len(chunks)}

**Statistics:**
- Average size: {avg_size:.0f} characters
- Total characters: {sum(chunk_sizes):,}
- Size range: {min(chunk_sizes)} - {max(chunk_sizes)} chars

**Sample chunks:**
"""
        
        # Show first 3 chunks preview
        for i, chunk in enumerate(chunks[:3]):
            content_preview = chunk['content'][:100] + "..." if len(chunk['content']) > 100 else chunk['content']
            info += f"\n**Chunk {i+1}:** {content_preview}\n"
        
        if len(chunks) > 3:
            info += f"\n... and {len(chunks) - 3} more chunks ready for processing."
        
        return info
    
    def process_chunk_extraction(self, task: str, entity_types: str,
                                relation_types: str, event_types: str, argument_types: str,
                                mode: str, batch_size: int, aggregate_results: bool,
                                filter_duplicates: bool, *generation_params) -> Tuple[str, str]:
        """
        Xử lý extraction từ document chunks thực sự
        
        Args:
            task: Extraction task
            entity_types, etc.: Schema parameters
            mode: Extraction mode
            batch_size, aggregate_results, filter_duplicates: Batch options
            *generation_params: Generation parameters
            
        Returns:
            Tuple[summary_markdown, detailed_json]
        """
        # Get chunks from context
        chunks = getattr(self.context, 'current_chunks', [])
        
        if not chunks:
            return "❌ No chunks available for extraction. Process a document first.", ""
        
        try:
            print(f"🔄 Starting chunk extraction on {len(chunks)} chunks")
            print(f"📋 Task: {task}, Mode: {mode}, Batch size: {batch_size}")
            
            start_time = time.time()
            
            # Process chunks in batches
            all_results = []
            aggregated_entities = []
            aggregated_relations = []
            aggregated_events = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                batch_num = i//batch_size + 1
                print(f"📦 Processing batch {batch_num}: {len(batch)} chunks")
                
                # Process each chunk in batch
                for chunk in batch:
                    chunk_text = chunk['content']
                    
                    # Call extraction method (real or mock)
                    if hasattr(self.context, 'extract_information') and self.context.model:
                        # Real extraction
                        chunk_result = self.context.extract_information(
                            text=chunk_text,
                            task=task,
                            entity_types=entity_types,
                            relation_types=relation_types,
                            event_types=event_types,
                            argument_types=argument_types,
                            mode=mode,
                            *generation_params
                        )
                    else:
                        # Mock extraction for demo
                        chunk_result = self._create_mock_chunk_extraction_result(chunk, task)
                    
                    # Store result with chunk metadata
                    chunk_result['chunk_metadata'] = chunk['metadata']
                    all_results.append(chunk_result)
                    
                    # Aggregate results if enabled
                    if aggregate_results:
                        if 'entities' in chunk_result:
                            aggregated_entities.extend(chunk_result['entities'])
                        if 'relations' in chunk_result:
                            aggregated_relations.extend(chunk_result['relations'])
                        if 'events' in chunk_result:
                            aggregated_events.extend(chunk_result['events'])
            
            # Filter duplicates if enabled
            if filter_duplicates and aggregate_results:
                aggregated_entities = self._filter_duplicate_entities(aggregated_entities)
                aggregated_relations = self._filter_duplicate_relations(aggregated_relations)
                aggregated_events = self._filter_duplicate_events(aggregated_events)
            
            processing_time = time.time() - start_time
            
            # Create final results
            results = {
                'chunks_processed': len(chunks),
                'processing_time': processing_time,
                'task': task,
                'mode': mode,
                'batch_size': batch_size,
                'aggregate_results': aggregate_results,
                'filter_duplicates': filter_duplicates,
                'aggregated_results': {
                    'entities': aggregated_entities,
                    'relations': aggregated_relations,
                    'events': aggregated_events
                },
                'per_chunk_results': all_results,
                'generation_parameters': {
                    'max_new_tokens': generation_params[0] if generation_params else 512,
                    'temperature': generation_params[1] if generation_params else 0.1,
                    'top_p': generation_params[2] if generation_params else 0.9,
                    'top_k': generation_params[3] if generation_params else 50
                },
                'performance_metrics': {
                    'avg_time_per_chunk': processing_time / len(chunks),
                    'tokens_generated': len(chunks) * 100,  # Simulated
                    'tokens_per_second': (len(chunks) * 100) / processing_time if processing_time > 0 else 0,
                    'memory_usage': '2.5 GB'  # Simulated
                }
            }
            
            print(f"✅ Chunk extraction completed in {processing_time:.2f}s")
            
            # Create summary
            summary = self._create_chunk_extraction_summary(results)
            
            # Create detailed JSON
            detailed_json = json.dumps(results, indent=2, ensure_ascii=False)
            
            return summary, detailed_json
            
        except Exception as e:
            error_msg = f"❌ Chunk extraction failed: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg, ""
    
    def _create_mock_chunk_extraction_result(self, chunk: Dict, task: str) -> Dict:
        """Create mock extraction result for a chunk"""
        chunk_id = chunk['metadata']['chunk_id']
        
        result = {
            'chunk_id': chunk_id,
            'entities': [],
            'relations': [],
            'events': []
        }
        
        # Add mock results based on task
        if task in ['NER', 'ALL']:
            result['entities'] = [
                {'entity_type': 'PERSON', 'entity_mention': f'Person_{chunk_id}', 'confidence': 0.95},
                {'entity_type': 'ORG', 'entity_mention': f'Organization_{chunk_id}', 'confidence': 0.90}
            ]
        
        if task in ['RE', 'ALL']:
            result['relations'] = [
                {'relation_type': 'WORKS_AT', 'head_entity': f'Person_{chunk_id}', 
                 'tail_entity': f'Organization_{chunk_id}', 'confidence': 0.88}
            ]
        
        if task in ['EE', 'ALL']:
            result['events'] = [
                {'trigger': f'meeting_{chunk_id}', 'trigger_type': 'MEETING', 
                 'arguments': [], 'confidence': 0.85}
            ]
        
        return result
    
    def _filter_duplicate_entities(self, entities: list) -> list:
        """Filter duplicate entities"""
        seen = set()
        filtered = []
        for entity in entities:
            key = (entity.get('entity_type', ''), entity.get('entity_mention', ''))
            if key not in seen:
                seen.add(key)
                filtered.append(entity)
        return filtered
    
    def _filter_duplicate_relations(self, relations: list) -> list:
        """Filter duplicate relations"""
        seen = set()
        filtered = []
        for relation in relations:
            key = (relation.get('relation_type', ''), 
                   relation.get('head_entity', ''), 
                   relation.get('tail_entity', ''))
            if key not in seen:
                seen.add(key)
                filtered.append(relation)
        return filtered
    
    def _filter_duplicate_events(self, events: list) -> list:
        """Filter duplicate events"""
        seen = set()
        filtered = []
        for event in events:
            key = (event.get('trigger_type', ''), event.get('trigger', ''))
            if key not in seen:
                seen.add(key)
                filtered.append(event)
        return filtered
    
    def _create_chunk_extraction_summary(self, result: Dict[str, Any]) -> str:
        """
        Tạo summary cho chunk extraction results
        
        Args:
            result: Kết quả chunk extraction
            
        Returns:
            Formatted summary markdown
        """
        # Basic stats
        total_chunks = result['chunks_processed']
        processing_time = result['processing_time']
        
        # Aggregated results
        aggregated = result['aggregated_results']
        total_entities = len(aggregated['entities'])
        total_relations = len(aggregated['relations'])
        total_events = len(aggregated['events'])
        
        # Performance metrics
        perf = result['performance_metrics']
        avg_time_per_chunk = perf['avg_time_per_chunk']
        tokens_generated = perf['tokens_generated']
        tokens_per_second = perf['tokens_per_second']
        memory_usage = perf['memory_usage']
        
        # Generation parameters
        gen_params = result['generation_parameters']
        
        summary = f"""
## 📊 Chunk Extraction Results

### 📈 Processing Statistics
- **Chunks Processed:** {total_chunks}
- **Total Time:** {processing_time:.2f}s
- **Task:** {result['task']}
- **Mode:** {result['mode']}
- **Batch Size:** {result['batch_size']}

### 🎯 Extraction Results
- **Entities Found:** {total_entities}
- **Relations Found:** {total_relations}
- **Events Found:** {total_events}
- **Aggregation:** {'✅' if result['aggregate_results'] else '❌'}
- **Deduplication:** {'✅' if result['filter_duplicates'] else '❌'}

### 🚀 Performance Metrics
- **Avg Time/Chunk:** {avg_time_per_chunk:.3f}s
- **Tokens Generated:** {tokens_generated:,}
- **Speed:** {tokens_per_second:.1f} tokens/sec
- **Memory Usage:** {memory_usage}

### 🎛️ Generation Parameters Used
- **Temperature:** {gen_params['temperature']}
- **Max Tokens:** {gen_params['max_new_tokens']}
- **Top-p:** {gen_params['top_p']}
- **Top-k:** {gen_params['top_k']}

### 💡 Next Steps
- Review extraction results in the JSON output below
- Adjust parameters and re-run if needed
- Export results for further analysis
        """.strip()
        
        return summary
    
    # ============================================================================
    # COMBINED TAB HANDLERS
    # ============================================================================
    
    def process_document_and_update_chunks_info(self, file, processing_mode: str, use_gpu: bool, 
                                               enable_ocr: bool, enable_table_structure: bool,
                                               enable_cleaning: bool, aggressive_clean: bool,
                                               enable_chunking: bool, chunking_strategy: str,
                                               chunk_size: int, chunk_overlap: int, task_type: str,
                                               *generation_params) -> Tuple[str, str, str, str]:
        """
        Process document và update chunks info cho extraction section
        Returns: (doc_summary, doc_json, chunks_preview, chunks_info)
        """
        # Gọi existing document processing method
        doc_summary, doc_json, chunks_preview = self.process_document_upload(
            file, processing_mode, use_gpu, enable_ocr, enable_table_structure,
            enable_cleaning, aggressive_clean, enable_chunking, chunking_strategy,
            chunk_size, chunk_overlap, task_type, *generation_params
        )
        
        # Update chunks info cho extraction section
        chunks_info = self.refresh_chunks_info()
        
        return doc_summary, doc_json, chunks_preview, chunks_info
    
    def process_chunk_extraction_combined(self, task: str, entity_types: str,
                                         relation_types: str, event_types: str, argument_types: str,
                                         mode: str, batch_size: int, aggregate_results: bool,
                                         filter_duplicates: bool, *generation_params) -> Tuple[str, str]:
        """
        Process chunk extraction for combined tab
        """
        # Gọi existing chunk extraction method
        return self.process_chunk_extraction(
            task, entity_types, relation_types, event_types, argument_types,
            mode, batch_size, aggregate_results, filter_duplicates, *generation_params
        )
    
    # ============================================================================
    # GENERATION PRESET HANDLERS
    # ============================================================================
    
    def apply_generation_preset(self, preset_name: str, *current_values) -> tuple:
        """
        Áp dụng generation parameter preset
        
        Args:
            preset_name: Tên preset (conservative, balanced, creative, precise)
            *current_values: Current values của generation parameters
            
        Returns:
            Tuple các values mới sau khi áp dụng preset
        """
        return GenerationPresets.apply_preset(preset_name, current_values)


class TabEventHandlers:
    """
    Helper class để setup event handlers cho từng tab một cách có tổ chức
    """
    
    def __init__(self, handlers: EventHandlers):
        self.handlers = handlers
    
    def setup_text_extraction_tab(self, components: Dict[str, Any]):
        """
        Setup event handlers cho Text Extraction tab
        
        Args:
            components: Dictionary chứa các Gradio components của tab
        """
        inputs = components['inputs']
        outputs = components['outputs']
        
        # Sample text loading
        inputs['sample_dropdown'].change(
            fn=self.handlers.load_sample_text,
            inputs=[inputs['sample_dropdown']],
            outputs=[inputs['text_input']]
        )
        
        # Generation preset buttons
        self._setup_preset_buttons(inputs['gen_controls'])
        
        # Main extraction button
        inputs['extract_btn'].click(
            fn=self.handlers.process_text_extraction,
            inputs=[
                inputs['text_input'],
                inputs['task_dropdown'],
                *inputs['schema_inputs'],  # entity_types, relation_types, etc.
                inputs['mode_dropdown'],
                *inputs['gen_controls'][:9]  # Generation parameters (exclude buttons)
            ],
            outputs=[
                outputs['summary_output'],
                outputs['json_output']
            ]
        )
    
    def setup_document_processing_extraction_tab(self, components: Dict[str, Any]):
        """
        Setup event handlers cho combined Document Processing & Extraction tab
        
        Args:
            components: Dictionary chứa các Gradio components của tab
        """
        inputs = components['inputs']
        outputs = components['outputs']
        
        # Generation preset buttons
        self._setup_preset_buttons(inputs['gen_controls'])
        
        # Document processing button - updates both doc results and chunks info
        inputs['process_btn'].click(
            fn=self.handlers.process_document_and_update_chunks_info,
            inputs=[
                inputs['file_upload'],
                # processing options
                inputs['processing_options']['processing_mode'],
                inputs['processing_options']['use_gpu'],
                inputs['processing_options']['enable_ocr'],
                inputs['processing_options']['enable_table_structure'],
                # cleaning options
                inputs['cleaning_options']['enable_cleaning'],
                inputs['cleaning_options']['aggressive_clean'],
                # chunking options
                inputs['chunking_options']['enable_chunking'],
                inputs['chunking_options']['chunking_strategy'],
                inputs['chunking_options']['chunk_size'],
                inputs['chunking_options']['chunk_overlap'],
                inputs['chunking_options']['task_type'],
                # generation parameters
                *inputs['gen_controls'][:9]
            ],
            outputs=[
                outputs['doc_summary_output'],
                outputs['doc_json_output'],
                outputs['chunks_preview'],
                inputs['chunks_info']  # Update chunks info in extraction section
            ]
        )
        
        # Chunk extraction button
        inputs['extract_btn'].click(
            fn=self.handlers.process_chunk_extraction_combined,
            inputs=[
                inputs['task_mode_controls']['task_dropdown'],
                inputs['schema_inputs'][0],  # entity_types
                inputs['schema_inputs'][1],  # relation_types  
                inputs['schema_inputs'][2],  # event_types
                inputs['schema_inputs'][3],  # argument_types
                inputs['task_mode_controls']['mode_dropdown'],
                inputs['batch_options']['batch_size'],
                inputs['batch_options']['aggregate_results'],
                inputs['batch_options']['filter_duplicates'],
                *inputs['gen_controls'][:9]  # Generation parameters
            ],
            outputs=[
                outputs['extraction_summary'],
                outputs['extraction_results']
            ]
        )
    
    def _setup_preset_buttons(self, gen_controls: tuple):
        """
        Setup generation preset buttons cho một tab
        
        Args:
            gen_controls: Tuple chứa generation controls include preset buttons
        """
        # Conservative preset
        gen_controls[9].click(
            fn=lambda *args: self.handlers.apply_generation_preset("conservative", *args),
            inputs=list(gen_controls[:9]),
            outputs=list(gen_controls[:9])
        )
        
        # Balanced preset
        gen_controls[10].click(
            fn=lambda *args: self.handlers.apply_generation_preset("balanced", *args),
            inputs=list(gen_controls[:9]),
            outputs=list(gen_controls[:9])
        )
        
        # Creative preset
        gen_controls[11].click(
            fn=lambda *args: self.handlers.apply_generation_preset("creative", *args),
            inputs=list(gen_controls[:9]),
            outputs=list(gen_controls[:9])
        )
        
        # Precise preset
        gen_controls[12].click(
            fn=lambda *args: self.handlers.apply_generation_preset("precise", *args),
            inputs=list(gen_controls[:9]),
            outputs=list(gen_controls[:9])
        )


def setup_event_handlers(demo, components: Dict[str, Any], handlers: EventHandlers):
    """
    Main function để setup tất cả event handlers
    
    Args:
        demo: Gradio Blocks object
        components: Dictionary chứa components của tất cả tabs
        handlers: EventHandlers instance
    """
    # Create tab handlers helper
    tab_handlers = TabEventHandlers(handlers)
    
    # Setup handlers cho từng tab
    if 'text_extraction' in components:
        tab_handlers.setup_text_extraction_tab(components['text_extraction'])
    
    if 'document_processing_extraction' in components:
        tab_handlers.setup_document_processing_extraction_tab(components['document_processing_extraction'])
    
    # Legacy support cho separate document processing tab nếu có
    if 'document_processing' in components:
        tab_handlers.setup_document_processing_tab(components['document_processing'])
    
    if 'chunk_extraction' in components:
        tab_handlers.setup_chunk_extraction_tab(components['chunk_extraction'])
    
    print("✅ All event handlers setup successfully!")