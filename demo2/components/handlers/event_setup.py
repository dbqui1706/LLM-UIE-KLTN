from typing import Dict, Any
from .factory import HandlerFactory
import logging
import gradio as gr

logger = logging.getLogger(__name__)

class TabEventHandlers:
    """Helper class để setup event handlers cho từng tab"""
    
    def __init__(self, handlers_dict: Dict[str, Any]):
        self.handlers = handlers_dict
    
    def setup_text_extraction_tab(self, components: Dict[str, Any]):
        """Setup event handlers cho Text Extraction tab"""
        inputs = components['inputs']
        outputs = components['outputs']
        
        logger.info("🔧 Setting up Text Extraction tab events")
        
        # Sample text loading
        inputs['sample_dropdown'].change(
            fn=self.handlers['text_extraction'].load_sample_text,
            inputs=[inputs['sample_dropdown']],
            outputs=[inputs['text_input']]
        )
        
        # Generation preset buttons
        self._setup_preset_buttons(inputs['gen_controls'])
        
        # Main extraction button
        inputs['extract_btn'].click(
            fn=self.handlers['text_extraction'].process_text_extraction,
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
        
        logger.info("✅ Text Extraction tab events configured")
    
    def setup_document_processing_extraction_tab(self, components: Dict[str, Any]):
        """Setup event handlers cho combined Document Processing & Extraction tab"""
        inputs = components['inputs']
        outputs = components['outputs']
        
        logger.info("🔧 Setting up Document Processing & Extraction tab events")
        
        # Generation preset buttons
        self._setup_preset_buttons(inputs['gen_controls'])
        
        # Document processing button
        inputs['process_btn'].click(
            fn=self._create_document_processing_handler(inputs, outputs),
            inputs=self._get_document_processing_inputs(inputs),
            outputs=self._get_document_processing_outputs(outputs, inputs)
        )
        
        # Chunk extraction button
        inputs['extract_btn'].click(
            fn=self.handlers['chunk_extraction'].process_chunk_extraction,
            inputs=self._get_chunk_extraction_inputs(inputs),
            outputs=[
                outputs['extraction_summary'],
                outputs['extraction_results']
            ]
        )
        
        logger.info("✅ Document Processing & Extraction tab events configured")
    
    def _setup_preset_buttons(self, gen_controls: tuple):
        """Setup generation preset buttons"""
        preset_handler = self.handlers['preset_manager']
        
        # Conservative preset
        gen_controls[9].click(
            fn=lambda *args: preset_handler.apply_generation_preset("conservative", *args),
            inputs=list(gen_controls[:9]),
            outputs=list(gen_controls[:9])
        )
        
        # Balanced preset
        gen_controls[10].click(
            fn=lambda *args: preset_handler.apply_generation_preset("balanced", *args),
            inputs=list(gen_controls[:9]),
            outputs=list(gen_controls[:9])
        )
        
        # Creative preset
        gen_controls[11].click(
            fn=lambda *args: preset_handler.apply_generation_preset("creative", *args),
            inputs=list(gen_controls[:9]),
            outputs=list(gen_controls[:9])
        )
        
        # Precise preset
        gen_controls[12].click(
            fn=lambda *args: preset_handler.apply_generation_preset("precise", *args),
            inputs=list(gen_controls[:9]),
            outputs=list(gen_controls[:9])
        )
    
    def _create_document_processing_handler(self, inputs: Dict, outputs: Dict):
        """Create combined document processing handler"""
        doc_handler = self.handlers['document_processing']
        chunk_handler = self.handlers['chunk_extraction']
        
        def combined_handler(*args):
            # Process document
            doc_summary, doc_json, chunks_preview, download_file = doc_handler.process_document_upload(*args)
            
            # Update chunks info
            chunks_info = chunk_handler.refresh_chunks_info()
            
            # Update download button
            if download_file:
                download_btn_update = gr.update(visible=True, value=download_file)
            else:
                download_btn_update = gr.update(visible=False)
            
            return doc_summary, doc_json, chunks_preview, chunks_info, download_btn_update
        
        return combined_handler
    
    def _get_document_processing_inputs(self, inputs: Dict) -> list:
        """Get inputs for document processing"""
        return [
            inputs['file_upload'],
            # Processing options
            inputs['processing_options']['processing_mode'],
            inputs['processing_options']['use_gpu'],
            inputs['processing_options']['enable_ocr'],
            inputs['processing_options']['enable_table_structure'],
            # Cleaning options
            inputs['cleaning_options']['enable_cleaning'],
            inputs['cleaning_options']['aggressive_clean'],
            # Chunking options
            inputs['chunking_options']['enable_chunking'],
            inputs['chunking_options']['chunking_strategy'],
            inputs['chunking_options']['chunk_size'],
            inputs['chunking_options']['chunk_overlap'],
            inputs['chunking_options']['task_type'],
            # Generation parameters
            *inputs['gen_controls'][:9]
        ]
    
    def _get_document_processing_outputs(self, outputs: Dict, inputs: Dict) -> list:
        """Get outputs for document processing"""
        return [
            outputs['doc_summary_output'],
            outputs['doc_json_output'], 
            outputs['chunks_preview'],
            inputs['chunks_info'],
            outputs['download_chunks_btn']
        ]
    
    def _get_chunk_extraction_inputs(self, inputs: Dict) -> list:
        """Get inputs for chunk extraction"""
        return [
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
        ]


class EventSetupManager:
    """Manager class để coordinate tất cả event setup"""
    
    def __init__(self, context):
        self.context = context
        self.handlers = HandlerFactory.create_all_handlers(context)
        self.tab_handlers = TabEventHandlers(self.handlers)
    
    def setup_all_events(self, demo, components: Dict[str, Any]):
        logger.info("🚀 Starting event setup for all tabs")
        
        try:
            # Setup Text Extraction tab
            if 'text_extraction' in components:
                self.tab_handlers.setup_text_extraction_tab(components['text_extraction'])
            
            # Setup Document Processing & Extraction tab
            if 'document_processing_extraction' in components:
                self.tab_handlers.setup_document_processing_extraction_tab(
                    components['document_processing_extraction']
                )
            
            # Setup legacy tabs if present
            self._setup_tabs(components)
            
            logger.info("✅ All event handlers setup successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error setting up event handlers: {e}")
            raise
    
    def _setup_tabs(self, components: Dict[str, Any]):
        
        # document processing tab
        if 'document_processing' in components:
            logger.info("🔧 Setting up legacy Document Processing tab")
            self._setup_document_processing_tab(components['document_processing'])
        
        # chunk extraction tab
        if 'chunk_extraction' in components:
            logger.info("🔧 Setting up legacy Chunk Extraction tab")
            self._setup_chunk_extraction_tab(components['chunk_extraction'])
    
    def _setup_document_processing_tab(self, components: Dict[str, Any]):
        inputs = components['inputs']
        outputs = components['outputs']
        
        inputs['process_btn'].click(
            fn=self.handlers['document_processing'].process_document_upload,
            inputs=self.tab_handlers._get_document_processing_inputs(inputs),
            outputs=[
                outputs['doc_summary_output'],
                outputs['doc_json_output'],
                outputs['chunks_preview']
            ]
        )
    
    def _setup_chunk_extraction_tab(self, components: Dict[str, Any]):
        """Setup legacy chunk extraction tab"""
        inputs = components['inputs']
        outputs = components['outputs']
        
        inputs['extract_btn'].click(
            fn=self.handlers['chunk_extraction'].process_chunk_extraction,
            inputs=self.tab_handlers._get_chunk_extraction_inputs(inputs),
            outputs=[
                outputs['extraction_summary'],
                outputs['extraction_results']
            ]
        )


def setup_event_handlers(demo, components: Dict[str, Any], context):
    """Main function setup event handlers"""
    manager = EventSetupManager(context)
    manager.setup_all_events(demo, components)
    
    return manager


# Convenience functions để maintain backward compatibility
def setup_text_extraction_events(components: Dict[str, Any], handlers: Dict[str, Any]):
    """Setup events text extraction tab"""
    tab_handlers = TabEventHandlers(handlers)
    tab_handlers.setup_text_extraction_tab(components)


def setup_document_processing_events(components: Dict[str, Any], handlers: Dict[str, Any]):
    """Setup events cho document processing tab"""
    tab_handlers = TabEventHandlers(handlers)
    tab_handlers.setup_document_processing_extraction_tab(components)


# Export functions để sử dụng trong main
__all__ = [
    'setup_event_handlers',
    'EventSetupManager', 
    'TabEventHandlers',
    'setup_text_extraction_events',
    'setup_document_processing_events'
]