import json
from typing import Dict, Any, Tuple
from .presets import GenerationPresets
from ..utils.sample import get_sample_texts

class EventHandlers:
    """
    Event handlers cho Gradio components
    Chia thành các handler riêng biệt cho từng tab và chức năng
    """
    
    def __init__(self, context):
        self.context = context
        self.sample_texts = get_sample_texts()
    
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
            # Gọi model extraction với parameters
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
                return result["error"], ""
            
            # Format output
            summary = self._create_extraction_summary(result)
            detailed_json = json.dumps(result, indent=2, ensure_ascii=False)
            
            return summary, detailed_json
            
        except Exception as e:
            error_msg = f"❌ Extraction failed: {str(e)}"
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
            summary_parts.append(f"🎛️ Temp: {gen_info['temperature']}")
            summary_parts.append(f"🎯 Tokens: {gen_info['max_new_tokens']}")
        
        return " | ".join(summary_parts) if summary_parts else "No results found"
    
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
        Xử lý document upload và processing
        
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
        
        try:
            # Gọi document processor
            result = self.context.process_document(
                file_path=file.name,
                processing_mode=processing_mode,
                use_gpu=use_gpu,
                enable_ocr=enable_ocr,
                enable_table_structure=enable_table_structure,
                enable_cleaning=enable_cleaning,
                aggressive_clean=aggressive_clean,
                enable_chunking=enable_chunking,
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                task_type=task_type,
                *generation_params
            )
            
            # Xử lý lỗi
            if "error" in result:
                return f"❌ {result['error']}", "", ""
            
            # Format outputs
            summary = self._create_document_summary(result)
            detailed_json = json.dumps(result, indent=2, ensure_ascii=False)
            chunks_preview = self._create_chunks_preview(result.get("sample_chunks", []))
            
            return summary, detailed_json, chunks_preview
            
        except Exception as e:
            error_msg = f"❌ Document processing failed: {str(e)}"
            return error_msg, "", ""
    
    def _create_document_summary(self, result: Dict[str, Any]) -> str:
        """
        Tạo summary cho document processing results
        
        Args:
            result: Kết quả document processing
            
        Returns:
            Formatted summary markdown
        """
        from pathlib import Path
        
        # Basic info
        file_name = Path(result['file_path']).name
        processing_time = result['processing_time']
        original_length = result['original_content_length']
        cleaned_length = result['cleaned_content_length']
        chunks_created = result['chunks_created']
        
        # Chunking stats
        chunking_stats = result['metadata']['chunking_stats']
        avg_chunk_size = chunking_stats['avg_chunk_size']
        strategy_used = chunking_stats['strategy_used']
        
        # Generation info
        gen_info = result.get('generation_info', {})
        gen_params = gen_info.get('parameters_used', {})
        total_inference_time = gen_info.get('total_inference_time', 0)
        
        summary = f"""
📄 **File:** {file_name}
⏱️ **Processing Time:** {processing_time:.2f}s
📊 **Content:** {original_length:,} → {cleaned_length:,} chars
📝 **Chunks Created:** {chunks_created}
📈 **Avg Chunk Size:** {avg_chunk_size} chars
🎯 **Strategy:** {strategy_used}

**🎛️ Generation Settings:**
- Temperature: {gen_params.get('temperature', 'N/A')}
- Max Tokens: {gen_params.get('max_new_tokens', 'N/A')}
- Top-p: {gen_params.get('top_p', 'N/A')}
- Inference Time: {total_inference_time:.2f}s
        """.strip()
        
        return summary
    
    def _create_chunks_preview(self, sample_chunks: list) -> str:
        """
        Tạo preview cho sample chunks
        
        Args:
            sample_chunks: List các sample chunks
            
        Returns:
            Formatted chunks preview
        """
        if not sample_chunks:
            return "No chunks available for preview."
        
        preview_parts = []
        for chunk in sample_chunks:
            chunk_id = chunk['chunk_id']
            content = chunk['content']
            size = chunk['size']
            tokens = chunk['tokens']
            
            preview_parts.append(
                f"**Chunk {chunk_id} ({size} chars, {tokens} tokens):**\n{content}"
            )
        
        return "\n\n".join(preview_parts)
    
    # ============================================================================
    # CHUNK EXTRACTION HANDLERS
    # ============================================================================
    
    def process_chunk_extraction(self, chunks_data: str, task: str, entity_types: str,
                                relation_types: str, event_types: str, argument_types: str,
                                mode: str, *generation_params) -> Tuple[str, str]:
        """
        Xử lý extraction từ document chunks
        
        Args:
            chunks_data: JSON string chứa chunks data
            task: Extraction task
            entity_types, etc.: Schema parameters
            mode: Extraction mode
            *generation_params: Generation parameters
            
        Returns:
            Tuple[summary_markdown, detailed_json]
        """
        # Kiểm tra chunks data
        if not chunks_data or not chunks_data.strip():
            return "❌ No chunks data available", ""
        
        try:
            # Gọi chunk extraction
            result = self.context.extract_from_chunks(
                chunks_data=chunks_data,
                task=task,
                entity_types=entity_types,
                relation_types=relation_types,
                event_types=event_types,
                argument_types=argument_types,
                mode=mode,
                *generation_params
            )
            
            # Xử lý lỗi
            if "error" in result:
                return f"❌ {result['error']}", ""
            
            # Format outputs
            summary = self._create_chunk_extraction_summary(result)
            detailed_json = json.dumps(result, indent=2, ensure_ascii=False)
            
            return summary, detailed_json
            
        except Exception as e:
            error_msg = f"❌ Chunk extraction failed: {str(e)}"
            return error_msg, ""
    
    def _create_chunk_extraction_summary(self, result: Dict[str, Any]) -> str:
        """
        Tạo summary cho chunk extraction results
        
        Args:
            result: Kết quả chunk extraction
            
        Returns:
            Formatted summary markdown
        """
        # Basic stats
        total_chunks = result['total_chunks_processed']
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
📊 **Chunks Processed:** {total_chunks}
⏱️ **Total Time:** {processing_time:.2f}s
🏷️ **Entities Found:** {total_entities}
🔗 **Relations Found:** {total_relations}
📅 **Events Found:** {total_events}

**🚀 Performance:**
- Avg Time/Chunk: {avg_time_per_chunk:.3f}s
- Tokens Generated: {tokens_generated:,}
- Speed: {tokens_per_second:.1f} tokens/sec
- Memory: {memory_usage}

**🎛️ Generation Used:**
- Temperature: {gen_params['temperature']}
- Max Tokens: {gen_params['max_new_tokens']}
- Top-p: {gen_params['top_p']}
- Beams: {gen_params['num_beams']}
        """.strip()
        
        return summary
    
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
    
    def setup_document_processing_tab(self, components: Dict[str, Any]):
        """
        Setup event handlers cho Document Processing tab
        
        Args:
            components: Dictionary chứa các Gradio components của tab
        """
        inputs = components['inputs']
        outputs = components['outputs']
        
        # Generation preset buttons
        self._setup_preset_buttons(inputs['gen_controls'])
        
        # Document processing button
        inputs['process_btn'].click(
            fn=self.handlers.process_document_upload,
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
                outputs['summary_output'],
                outputs['json_output'],
                outputs['chunks_preview']
            ]
        )
    
    def setup_chunk_extraction_tab(self, components: Dict[str, Any]):
        """
        Setup event handlers cho Chunk Extraction tab
        
        Args:
            components: Dictionary chứa các Gradio components của tab
        """
        inputs = components['inputs']
        outputs = components['outputs']
        
        # Generation preset buttons
        self._setup_preset_buttons(inputs['gen_controls'])
        
        # Chunk extraction button
        inputs['extract_btn'].click(
            fn=self.handlers.process_chunk_extraction,
            inputs=[
                inputs['chunks_data_storage'],
                inputs['task_mode_controls']['task_dropdown'],
                inputs['schema_inputs']['entity_types'],
                inputs['schema_inputs']['relation_types'],
                inputs['schema_inputs']['event_types'],
                inputs['schema_inputs']['argument_types'],
                inputs['task_mode_controls']['mode_dropdown'],
                *inputs['gen_controls'][:9]  # Generation parameters
            ],
            outputs=[
                outputs['summary_output'],
                outputs['results_output']
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
    tab_handlers.setup_text_extraction_tab(components['text_extraction'])
    tab_handlers.setup_document_processing_tab(components['document_processing'])
    tab_handlers.setup_chunk_extraction_tab(components['chunk_extraction'])
    
    print("✅ All event handlers setup successfully!")