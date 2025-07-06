import gradio as gr
from .generation_controls import create_generation_controls, create_schema_inputs
from demo2.utils.sample import get_sample_texts

def text_extraction_tab():
    sample_texts = get_sample_texts()
    
    with gr.Row():
        with gr.Column(scale=2):
            # Input section
            gr.Markdown("## 📝 Input")

            # Sample text dropdown
            sample_dropdown = gr.Dropdown(
                choices=list(sample_texts.keys()),
                label="Sample Texts (Optional)",
                value=None
            )

            # Main text input
            text_input = gr.Textbox(
                label="Text to Analyze",
                placeholder="Enter your text here...",
                lines=5,
                value=""
            )

            # Task selection
            task_dropdown = gr.Dropdown(
                choices=["NER", "RE", "EE", "ALL"],
                label="Task",
                value="ALL"
            )

            # Mode selection
            mode_dropdown = gr.Dropdown(
                choices=["flexible", "strict", "open"],
                label="Extraction Mode",
                value="flexible",
                info="flexible: use schema but allow others | strict: only use schema | open: no schema"
            )

            # Generation controls
            text_gen_controls = create_generation_controls()

            # Schema inputs
            schema_inputs = create_schema_inputs()

            # Extract button
            extract_btn = gr.Button("🚀 Extract Information", variant="primary", size="lg")

        with gr.Column(scale=3):
            # Output section
            gr.Markdown("## 📊 Results")

            # Summary output
            summary_output = gr.Textbox(
                label="Summary",
                interactive=False,
                lines=2
            )

            # JSON output
            json_output = gr.Code(
                label="Detailed Results (JSON)",
                language="json",
                lines=20
            )
    
    return {
        'inputs': {
            'sample_dropdown': sample_dropdown,
            'text_input': text_input,
            'task_dropdown': task_dropdown,
            'mode_dropdown': mode_dropdown,
            'schema_inputs': schema_inputs,
            'gen_controls': text_gen_controls,
            'extract_btn': extract_btn
        },
        'outputs': {
            'summary_output': summary_output,
            'json_output': json_output
        }
    }