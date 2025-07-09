import gradio as gr
from .generation_controls import create_generation_controls, create_schema_inputs

def document_processing_extraction_tab():
    """Create combined document processing and extraction tab"""
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 📁 Document Upload & Processing")
            
            # File upload
            file_upload = gr.File(
                label="Upload Document",
                file_types=[".pdf", ".docx", ".doc", ".txt", ".md", ".html"],
                file_count="single"
            )
            
            # Processing Options
            processing_options = _create_processing_options()
            
            # Text Cleaning Options
            cleaning_options = _create_cleaning_options()
            
            # Chunking Options
            chunking_options = _create_chunking_options()

            # Process document button
            process_doc_btn = gr.Button("🔄 Process Document", variant="primary", size="lg")
            
            gr.Markdown("---")
            gr.Markdown("## 🔍 Information Extraction")
            
            # Available chunks info
            chunks_info = gr.Markdown(
                label="Available Chunks",
                value="📄 Upload and process a document to see chunks here."
            )
            
            # Task and mode selection for extraction
            task_mode_controls = _create_extraction_task_controls()
            
            # Generation controls
            gen_controls = create_generation_controls()

            # Schema inputs
            schema_inputs = create_schema_inputs()
            
            # Batch processing options
            batch_options = _create_batch_processing_options()
            
            # Extract button
            extract_chunks_btn = gr.Button("🚀 Extract Information from Chunks", variant="secondary", size="lg")

        with gr.Column(scale=3):
            gr.Markdown("## 📊 Results")
            
            # Processing results section
            with gr.Accordion("📄 Document Processing Results", open=True):
                doc_summary_output = gr.Markdown(
                    label="Processing Summary",
                    value="Upload and process a document to see results here."
                )
                
                with gr.Accordion("📋 Processing Details", open=False):
                    doc_json_output = gr.Code(
                        label="Processing Details (JSON)",
                        language="json",
                        lines=10
                    )
            
            # Chunks preview section
            with gr.Accordion("👀 Document Chunks", open=True):
                # ✅ Add download button
                with gr.Row():
                    download_chunks_btn = gr.DownloadButton(
                        label="📥 Download Chunks",
                        visible=False
                    )
                
                chunks_preview = gr.Markdown(
                    label="Chunks Preview",
                    value="Process a document to see chunks here."
                )
            
            # Extraction results section
            with gr.Accordion("🔍 Extraction Results", open=False):
                extraction_summary = gr.Markdown(
                    label="Extraction Summary",
                    value="Extract information from chunks to see results here."
                )
                
                extraction_results = gr.Code(
                    label="Detailed Extraction Results (JSON)",
                    language="json",
                    lines=25
                )
    
    return {
        'inputs': {
            'file_upload': file_upload,
            'processing_options': processing_options,
            'cleaning_options': cleaning_options,
            'chunking_options': chunking_options,
            'process_btn': process_doc_btn,
            'chunks_info': chunks_info,
            'task_mode_controls': task_mode_controls,
            'gen_controls': gen_controls,
            'schema_inputs': schema_inputs,
            'batch_options': batch_options,
            'extract_btn': extract_chunks_btn
        },
        'outputs': {
            'doc_summary_output': doc_summary_output,
            'doc_json_output': doc_json_output,
            'chunks_preview': chunks_preview,
            'extraction_summary': extraction_summary,
            'extraction_results': extraction_results,
            'download_chunks_btn': download_chunks_btn
        }
    }


def _create_processing_options():
    """Create processing options section"""
    with gr.Accordion("⚙️ Processing Options", open=False):
        processing_mode = gr.Dropdown(
            choices=["fast", "balanced", "accurate", "batch"],
            label="Processing Mode",
            value="fast",
            info="fast: Quick processing | balanced: Good speed/quality | accurate: Best quality"
        )
        
        use_gpu = gr.Checkbox(
            label="Use GPU Acceleration",
            value=True,
            info="Enable GPU for faster processing"
        )
        
        with gr.Row():
            enable_ocr = gr.Checkbox(
                label="Enable OCR",
                value=False,
                info="Extract text from scanned documents"
            )
            
            enable_table_structure = gr.Checkbox(
                label="Enable Table Structure",
                value=False,
                info="Preserve table structure"
            )
    
    return {
        'processing_mode': processing_mode,
        'use_gpu': use_gpu,
        'enable_ocr': enable_ocr,
        'enable_table_structure': enable_table_structure
    }


def _create_cleaning_options():
    """Create text cleaning options section"""
    with gr.Accordion("🧹 Text Cleaning Options", open=False):
        enable_cleaning = gr.Checkbox(
            label="Enable Text Cleaning",
            value=False,
            info="Clean and normalize text"
        )
        
        aggressive_clean = gr.Checkbox(
            label="Aggressive Cleaning",
            value=False,
            info="More thorough cleaning (may remove some content)"
        )
    
    return {
        'enable_cleaning': enable_cleaning,
        'aggressive_clean': aggressive_clean
    }


def _create_chunking_options():
    with gr.Accordion("✂️ Chunking Options", open=True):
        enable_chunking = gr.Checkbox(
            label="Enable Chunking",
            value=True,
            info="Split document into chunks for processing"
        )
        
        chunking_strategy = gr.Dropdown(
            choices=[
                "sentence",
                "recursive", 
                "markdown",
                "semantic",
                # "spacy",
                # "hybrid"
            ],
            label="Chunking Strategy",
            value="sentence",
            info="Method for splitting document"
        )
        
        with gr.Row():
            chunk_size = gr.Slider(
                minimum=100,
                maximum=1000,
                value=300,
                step=50,
                label="Chunk Size",
                info="Maximum characters per chunk"
            )
            
            chunk_overlap = gr.Slider(
                minimum=0,
                maximum=200,
                value=50,
                step=10,
                label="Chunk Overlap",
                info="Overlap between chunks"
            )

    return {
        'enable_chunking': enable_chunking,
        'chunking_strategy': chunking_strategy,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
    }


def _create_extraction_task_controls():
    with gr.Accordion("🎯 Extraction Configuration", open=True):
        # Task selection
        task_dropdown = gr.Dropdown(
            choices=["NER", "RE", "EE", "ALL"],
            label="Extraction Task",
            value="ALL",
            info="Select which information to extract"
        )
        
        # Mode selection
        mode_dropdown = gr.Dropdown(
            choices=["flexible", "strict", "open"],
            label="Extraction Mode",
            value="flexible",
            info="flexible: use schema + default | strict: only schema | open: no schema"
        )
    
    return {
        'task_dropdown': task_dropdown,
        'mode_dropdown': mode_dropdown
    }


def _create_batch_processing_options():
    """Create batch processing options"""
    with gr.Accordion("⚙️ Batch Processing Options", open=False):
        batch_size = gr.Slider(
            minimum=1,
            maximum=20,
            value=8,
            step=1,
            label="Batch Size",
            info="Number of chunks to process simultaneously"
        )
        
        aggregate_results = gr.Checkbox(
            label="Aggregate Results",
            value=True,
            info="Combine results from all chunks"
        )
        
        filter_duplicates = gr.Checkbox(
            label="Filter Duplicates",
            value=True,
            info="Remove duplicate entities/relations across chunks"
        )
    
    return {
        'batch_size': batch_size,
        'aggregate_results': aggregate_results,
        'filter_duplicates': filter_duplicates
    }