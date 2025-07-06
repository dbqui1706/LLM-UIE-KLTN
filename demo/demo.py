import gradio as gr
import json
from typing import Dict, List, Any
import logging

# Import các class từ code của bạn
from demo.core.base import LLamaModel
from demo.core.extract import UIEResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UIEDemo:
    def __init__(self):
        """Initialize UIE demo with model"""
        try:
            # Load model - có thể customize model name ở đây
            # self.model = LLamaModel(model_name='quidangz/LLama-8B-Instruct-MultiTask-CE')
            self.model = None

            # Verify model is actually loaded
            if self.model and hasattr(self.model, 'model') and self.model.model is not None:
                self.load_status = "success"
                logger.info("✅ Model loaded successfully!")
            else:
                self.load_status = "failed"
                self.error_message = "Model object created but model weights not loaded"
                logger.error(self.error_message)

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def extract_information(self,
                            text: str,
                            task: str,
                            entity_types: str = "",
                            relation_types: str = "",
                            event_types: str = "",
                            argument_types: str = "",
                            mode: str = "flexible") -> Dict[str, Any]:
        """
        Extract information based on selected task

        Args:
            text: Input text
            task: Selected task (NER, RE, EE, ALL)
            entity_types: Comma-separated entity types
            relation_types: Comma-separated relation types
            event_types: Comma-separated event types
            argument_types: Comma-separated argument types
            mode: Extraction mode

        Returns:
            Dictionary with extraction results
        """
        if not self.model:
            return {"error": "Model not loaded"}

        if not text.strip():
            return {"error": "Please enter some text"}

        try:
            # Prepare user schema
            user_schema = None
            if any([entity_types.strip(), relation_types.strip(), event_types.strip(), argument_types.strip()]):
                user_schema = {}

                if entity_types.strip():
                    user_schema['entity_types'] = [t.strip() for t in entity_types.split(',') if t.strip()]

                if relation_types.strip():
                    user_schema['relation_types'] = [t.strip() for t in relation_types.split(',') if t.strip()]

                if event_types.strip():
                    user_schema['event_types'] = [t.strip() for t in event_types.split(',') if t.strip()]

                if argument_types.strip():
                    user_schema['argument_types'] = [t.strip() for t in argument_types.split(',') if t.strip()]

            # Perform extraction
            result = self.model.extract(
                text=text,
                task=task,
                user_schema=user_schema if user_schema else None,
                mode=mode
            )

            # Format output based on task
            if task == "ALL":
                return self._format_all_results(result)
            else:
                return self._format_single_task_result(result, task)

        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return {"error": f"Extraction failed: {str(e)}"}

    def _format_all_results(self, result: UIEResult) -> Dict[str, Any]:
        """Format results for ALL task"""
        return {
            "entities": [
                {
                    "type": ent.entity_type,
                    "text": ent.entity_mention,
                } for ent in result.entities
            ],
            "relations": [
                {
                    "type": rel.relation_type,
                    "head": rel.head_entity,
                    "tail": rel.tail_entity,
                } for rel in result.relations
            ],
            "events": [
                {
                    "event_type": event.trigger_type,
                    "trigger": event.trigger,
                    "arguments": [
                        {"role": arg.role, "entity": arg.entity}
                        for arg in event.arguments
                    ],
                } for ee in result.events for event in ee.events
            ],
            "statistics": result.get_statistics()
        }

    def _format_single_task_result(self, result: List, task: str) -> Dict[str, Any]:
        """Format results for single task"""
        if task == "NER":
            return {
                "entities": [
                    {
                        "type": ent.entity_type,
                        "text": ent.entity_mention,
                    } for ent in result
                ]
            }
        elif task == "RE":
            return {
                "relations": [
                    {
                        "type": rel.relation_type,
                        "head": rel.head_entity,
                        "tail": rel.tail_entity,
                    } for rel in result
                ]
            }
        elif task == "EE":
            return {
                "events": [
                    {
                        "event_type": event.trigger_type,
                        "trigger": event.trigger,
                        "arguments": [
                            {"role": arg.role, "entity": arg.entity}
                            for arg in event.arguments
                        ],
                    } for ee in result for event in ee.events
                ]
            }

        return {"result": result}


def create_demo():
    """Create Gradio demo interface"""

    # Initialize demo
    demo_instance = UIEDemo()

    # Sample texts for examples
    sample_texts = {
        "Business News": "Apple Inc announced that Tim Cook will step down as CEO next year. The company, founded by Steve Jobs in Cupertino, California, has been a major player in the technology industry.",
        "Sports News": "Lionel Messi scored two goals in the FIFA World Cup final held in Qatar. The Argentine forward helped his team defeat France 4-2 in the penalty shootout.",
        "Academic Text": "Dr. Sarah Johnson from MIT published a research paper on artificial intelligence in the Journal of Machine Learning. The study was conducted in collaboration with researchers from Stanford University."
    }

    def process_extraction(text, task, entity_types, relation_types, event_types, argument_types, mode):
        """Process extraction and return formatted results"""
        result = demo_instance.extract_information(
            text, task, entity_types, relation_types, event_types, argument_types, mode
        )

        if "error" in result:
            return result["error"], ""

        # Pretty print JSON
        formatted_json = json.dumps(result, indent=2, ensure_ascii=False)

        # Create summary
        summary_lines = []
        if "entities" in result:
            summary_lines.append(f"🏷️ Entities: {len(result['entities'])}")
        if "relations" in result:
            summary_lines.append(f"🔗 Relations: {len(result['relations'])}")
        if "events" in result:
            summary_lines.append(f"📅 Events: {len(result['events'])}")

        summary = " | ".join(summary_lines) if summary_lines else "No results found"

        return summary, formatted_json

    # Create Gradio interface
    with gr.Blocks(title="Universal Information Extraction Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔍 Universal Information Extraction (UIE) Demo

        Extract entities, relations, and events from text using fine-tuned LLaMA models.
        """)

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

                # Schema inputs (collapsible)
                with gr.Accordion("🎯 Custom Schema (Optional)", open=False):
                    entity_types_input = gr.Textbox(
                        label="Entity Types",
                        placeholder="algorithm, conference, product, researcher, university...",
                        info="Comma-separated list"
                    )

                    relation_types_input = gr.Textbox(
                        label="Relation Types",
                        placeholder="works_at, located_in, published_in...",
                        info="Comma-separated list"
                    )

                    event_types_input = gr.Textbox(
                        label="Event Types",
                        placeholder="Conference, Publication, Research...",
                        info="Comma-separated list"
                    )

                    argument_types_input = gr.Textbox(
                        label="Argument Types",
                        placeholder="researcher, venue, topic, date...",
                        info="Comma-separated list"
                    )

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

        # Event handlers
        def load_sample_text(sample_name):
            if sample_name:
                return sample_texts[sample_name]
            return ""

        sample_dropdown.change(
            fn=load_sample_text,
            inputs=[sample_dropdown],
            outputs=[text_input]
        )

        extract_btn.click(
            fn=process_extraction,
            inputs=[
                text_input, task_dropdown,
                entity_types_input, relation_types_input,
                event_types_input, argument_types_input,
                mode_dropdown
            ],
            outputs=[summary_output, json_output]
        )

    return demo


if __name__ == "__main__":
    # Create and launch demo
    demo = create_demo()

    # Launch with custom settings
    demo.launch(
        share=True,  # Set to True for public sharing
        debug=True,  # Enable debug mode
        show_error=True  # Show detailed errors
    )