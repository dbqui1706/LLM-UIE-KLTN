import gradio as gr
from .generation_controls import create_generation_controls, create_schema_inputs

def document_processing_tab():
    """Create document processing tab"""
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 📁 Document Upload & Processing")
            
            # File upload
            file_upload = gr.File(
                label="Upload Document",
                file_types=[".pdf", ".docx", ".doc", ".txt", ".md", ".html"],
                file_count="single"
            )
            
            # Generation controls
            doc_gen_controls = create_generation_controls()

            # Processing Options
            processing_options = _create_processing_options()
            
            # Text Cleaning Options
            cleaning_options = _create_cleaning_options()
            
            # Chunking Options
            chunking_options = _create_chunking_options()

            # Schema inputs
            schema_inputs = create_schema_inputs()
            
            # Process button
            process_doc_btn = gr.Button("🔄 Process Document", variant="primary", size="lg")

        with gr.Column(scale=3):
            gr.Markdown("## 📊 Processing Results")
            
            # Processing summary
            doc_summary_output = gr.Markdown(
                label="Processing Summary",
                value="Upload and process a document to see results here."
            )
            
            # Detailed results
            with gr.Accordion("📋 Detailed Results", open=False):
                doc_json_output = gr.Code(
                    label="Processing Details (JSON)",
                    language="json",
                    lines=15
                )
            
            # Chunks preview
            with gr.Accordion("👀 Chunks Preview", open=True):
                chunks_preview = gr.Markdown(
                    label="Sample Chunks",
                    value="Process a document to see chunk samples here."
                )
    
    return {
        'inputs': {
            'file_upload': file_upload,
            'processing_options': processing_options,
            'cleaning_options': cleaning_options,
            'chunking_options': chunking_options,
            'gen_controls': doc_gen_controls,
            'process_btn': process_doc_btn
        },
        'outputs': {
            'summary_output': doc_summary_output,
            'json_output': doc_json_output,
            'chunks_preview': chunks_preview
        }
    }


def _create_processing_options():
    """Create processing options section"""
    with gr.Accordion("⚙️ Processing Options", open=False):
        processing_mode = gr.Dropdown(
            choices=["fast", "balanced", "accurate", "batch"],
            label="Processing Mode",
            value="balanced",
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
                value=True,
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
            value=True,
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
    """Create chunking options section"""
    with gr.Accordion("✂️ Chunking Options", open=False):
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
                "markdown_header",
                "semantic",
                "spacy",
                "hybrid"
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
        
        task_type = gr.Dropdown(
            choices=["balanced", "ner", "re", "ee"],
            label="Optimize for Task",
            value="balanced",
            info="Optimize chunking for specific UIE task"
        )
    
    return {
        'enable_chunking': enable_chunking,
        'chunking_strategy': chunking_strategy,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'task_type': task_type
    }