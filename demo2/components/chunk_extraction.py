import gradio as gr
from .generation_controls import create_generation_controls, create_schema_inputs

def chunk_extraction_tab():
    """Create batch chunk extraction tab"""
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 🔄 Extract from Document Chunks")
            
            gr.Markdown("""
            This tab allows you to extract information from document chunks created in the Document Processing tab.
            """)
            
            # Hidden field to store chunks data
            chunks_data_storage = gr.Textbox(
                label="Chunks Data (Hidden)",
                visible=False,
                value='{"chunks": "sample_data"}'
            )
            
            # Task and mode selection
            task_mode_controls = _create_task_mode_controls()
            
            # Generation controls
            chunk_gen_controls = create_generation_controls()
            
            # Schema inputs
            schema_inputs = _create_chunk_schema_inputs()
            
            # Batch processing options
            batch_options = _create_batch_processing_options()
            
            # Extract button
            extract_chunks_btn = gr.Button("🚀 Extract from Chunks", variant="primary", size="lg")
            
            # Status info
            gr.Markdown("""
            **Note:** This is a demo interface. In the full implementation:
            - Chunks data would be automatically passed from Document Processing
            - Real extraction would be performed on each chunk
            - Results would be aggregated and deduplicated
            """)

        with gr.Column(scale=3):
            gr.Markdown("## 📊 Chunk Extraction Results")
            
            # Extraction summary
            chunk_extract_summary = gr.Markdown(
                label="Extraction Summary",
                value="Run chunk extraction to see results here."
            )
            
            # Detailed extraction results
            chunk_extract_results = gr.Code(
                label="Detailed Extraction Results (JSON)",
                language="json",
                lines=20
            )
    
    return {
        'inputs': {
            'chunks_data_storage': chunks_data_storage,
            'task_mode_controls': task_mode_controls,
            'gen_controls': chunk_gen_controls,
            'schema_inputs': schema_inputs,
            'batch_options': batch_options,
            'extract_btn': extract_chunks_btn
        },
        'outputs': {
            'summary_output': chunk_extract_summary,
            'results_output': chunk_extract_results
        }
    }


def _create_task_mode_controls():
    """Create task and mode selection controls"""
    # Task selection for chunk extraction
    chunk_task_dropdown = gr.Dropdown(
        choices=["NER", "RE", "EE", "ALL"],
        label="Extraction Task",
        value="ALL"
    )
    
    # Mode selection
    chunk_mode_dropdown = gr.Dropdown(
        choices=["flexible", "strict", "open"],
        label="Extraction Mode",
        value="flexible"
    )
    
    return {
        'task_dropdown': chunk_task_dropdown,
        'mode_dropdown': chunk_mode_dropdown
    }


def _create_chunk_schema_inputs():
    """Create schema inputs for chunk extraction"""
    with gr.Accordion("🎯 Custom Schema for Chunks", open=False):
        chunk_entity_types = gr.Textbox(
            label="Entity Types",
            placeholder="algorithm, researcher, institution...",
            info="Comma-separated list"
        )
        
        chunk_relation_types = gr.Textbox(
            label="Relation Types", 
            placeholder="works_at, published_in, invented_by...",
            info="Comma-separated list"
        )
        
        chunk_event_types = gr.Textbox(
            label="Event Types",
            placeholder="Publication, Conference, Research...",
            info="Comma-separated list"
        )
        
        chunk_argument_types = gr.Textbox(
            label="Argument Types",
            placeholder="author, venue, date, topic...",
            info="Comma-separated list"
        )
    
    return {
        'entity_types': chunk_entity_types,
        'relation_types': chunk_relation_types,
        'event_types': chunk_event_types,
        'argument_types': chunk_argument_types
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