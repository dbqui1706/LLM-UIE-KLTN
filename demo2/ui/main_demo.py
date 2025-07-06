import gradio as gr
from uie_demo import UIEUi
import sys
from pathlib import Path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from demo2.components import (
    text_extraction_tab,
    document_processing_tab,
    chunk_extraction_tab,
    setup_event_handlers
)
from demo2.components.handlers import EventHandlers


def main():
    """Create main Gradio demo interface"""
    
    # Initialize demo và handlers
    context = UIEUi()
    handlers = EventHandlers(context)
    
    with gr.Blocks(title="Universal Information Extraction Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔍 Universal Information Extraction (UIE) Demo
        
        Extract entities, relations, and events from text or documents using fine-tuned LLaMA models.
        """)

        with gr.Tabs():
            # Tab 1: Text Extraction
            with gr.TabItem("📝 Text Extraction"):
                text_extraction_comp = text_extraction_tab()
            
            # Tab 2: Document Processing
            with gr.TabItem("📄 Document Processing"):
                document_processing_comp = document_processing_tab()
            
            # Tab 3: Batch Extraction from Chunks
            with gr.TabItem("🔄 Batch Chunk Extraction"):
                chunk_extraction_comp = chunk_extraction_tab()

        # Organize components
        components = {
            'text_extraction': text_extraction_comp,
            'document_processing': document_processing_comp,
            'chunk_extraction': chunk_extraction_comp
        }
        
        # Setup tất cả event handlers
        setup_event_handlers(demo, components, handlers)
    
    return demo

if __name__ == "__main__":
    demo = main()
    demo.launch(share=True, debug=True, show_error=True)