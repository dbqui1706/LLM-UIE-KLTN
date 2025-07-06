import gradio as gr
import json
from typing import Dict, List, Any
import logging
import tempfile
import os
from pathlib import Path

import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)  # Go up two levels to reach llm_multitask
sys.path.append(project_root)

# Now imports should work
from demo2.core.extract import UIEResult

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

            # Document processor placeholder
            self.document_processor = None

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
                            mode: str = "flexible",
                            # Generation parameters
                            max_new_tokens: int = 512,
                            temperature: float = 0.1,
                            top_p: float = 0.9,
                            top_k: int = 50,
                            do_sample: bool = True,
                            repetition_penalty: float = 1.0,
                            no_repeat_ngram_size: int = 0,
                            num_beams: int = 1,
                            early_stopping: bool = False) -> Dict[str, Any]:
        """
        Extract information based on selected task with custom generation parameters
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

            # Prepare generation parameters
            generation_kwargs = {
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'do_sample': do_sample,
                'repetition_penalty': repetition_penalty,
                'no_repeat_ngram_size': no_repeat_ngram_size,
                'num_beams': num_beams,
                'early_stopping': early_stopping
            }

            # Perform extraction with custom generation parameters
            result = self.model.extract(
                text=text,
                task=task,
                user_schema=user_schema if user_schema else None,
                mode=mode,
                **generation_kwargs  # Pass generation parameters
            )

            # Format output based on task
            if task == "ALL":
                formatted_result = self._format_all_results(result)
            else:
                formatted_result = self._format_single_task_result(result, task)

            # Add generation info to result
            formatted_result['generation_info'] = {
                'parameters_used': generation_kwargs,
                'text_length': len(text),
                'task': task,
                'mode': mode
            }

            return formatted_result

        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return {"error": f"Extraction failed: {str(e)}"}

    def process_document(self,
                         file_path: str,
                         processing_mode: str,
                         use_gpu: bool,
                         enable_ocr: bool,
                         enable_table_structure: bool,
                         enable_cleaning: bool,
                         aggressive_clean: bool,
                         enable_chunking: bool,
                         chunking_strategy: str,
                         chunk_size: int,
                         chunk_overlap: int,
                         task_type: str,
                         # Generation parameters
                         max_new_tokens: int = 512,
                         temperature: float = 0.1,
                         top_p: float = 0.9,
                         top_k: int = 50,
                         do_sample: bool = True,
                         repetition_penalty: float = 1.0,
                         no_repeat_ngram_size: int = 0,
                         num_beams: int = 1,
                         early_stopping: bool = False) -> Dict[str, Any]:
        """
        Process document với các options và generation parameters
        """
        try:
            # Prepare generation parameters
            generation_kwargs = {
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'do_sample': do_sample,
                'repetition_penalty': repetition_penalty,
                'no_repeat_ngram_size': no_repeat_ngram_size,
                'num_beams': num_beams,
                'early_stopping': early_stopping
            }

            # Mock processing result for UI demo (includes generation parameters)
            mock_result = {
                "success": True,
                "file_path": file_path,
                "processing_time": 2.5,
                "original_content_length": 15420,
                "cleaned_content_length": 14850,
                "chunks_created": 12,
                "metadata": {
                    "file_size": 245760,
                    "format": ".pdf",
                    "page_count": 8,
                    "processing_mode": processing_mode,
                    "generation_parameters": generation_kwargs,
                    "chunking_stats": {
                        "total_chunks": 12,
                        "avg_chunk_size": 320,
                        "strategy_used": chunking_strategy
                    }
                },
                "sample_chunks": [
                    {
                        "chunk_id": 0,
                        "content": "This is a sample chunk from the document. It contains important information about machine learning algorithms and their applications in natural language processing. The transformer architecture has revolutionized the field.",
                        "size": 145,
                        "tokens": 25
                    },
                    {
                        "chunk_id": 1,
                        "content": "Another chunk discussing deep learning architectures, particularly transformer models and their effectiveness in understanding context and relationships in text data. Attention mechanisms enable better performance.",
                        "size": 157,
                        "tokens": 28
                    }
                ],
                "generation_info": {
                    "parameters_used": generation_kwargs,
                    "total_inference_time": 8.7,
                    "avg_time_per_chunk": 0.725
                }
            }

            return mock_result

        except Exception as e:
            return {"error": f"Document processing failed: {str(e)}"}

    def extract_from_chunks(self,
                            chunks_data: str,
                            task: str,
                            entity_types: str = "",
                            relation_types: str = "",
                            event_types: str = "",
                            argument_types: str = "",
                            mode: str = "flexible",
                            # Generation parameters
                            max_new_tokens: int = 512,
                            temperature: float = 0.1,
                            top_p: float = 0.9,
                            top_k: int = 50,
                            do_sample: bool = True,
                            repetition_penalty: float = 1.0,
                            no_repeat_ngram_size: int = 0,
                            num_beams: int = 1,
                            early_stopping: bool = False) -> Dict[str, Any]:
        """
        Extract information from document chunks với generation parameters
        """
        try:
            # Prepare generation parameters
            generation_kwargs = {
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'do_sample': do_sample,
                'repetition_penalty': repetition_penalty,
                'no_repeat_ngram_size': no_repeat_ngram_size,
                'num_beams': num_beams,
                'early_stopping': early_stopping
            }

            # Mock extraction from chunks với generation info
            mock_results = {
                "total_chunks_processed": 12,
                "processing_time": 8.7,
                "generation_parameters": generation_kwargs,
                "aggregated_results": {
                    "entities": [
                        {"type": "ALGORITHM", "text": "transformer", "frequency": 5, "chunks": [0, 1, 3, 7, 9],
                         "confidence": 0.92},
                        {"type": "PERSON", "text": "Vaswani", "frequency": 2, "chunks": [1, 3], "confidence": 0.88},
                        {"type": "ORGANIZATION", "text": "Google", "frequency": 3, "chunks": [1, 4, 8],
                         "confidence": 0.95}
                    ],
                    "relations": [
                        {"type": "INVENTED_BY", "head": "transformer", "tail": "Vaswani", "chunks": [1, 3],
                         "confidence": 0.87},
                        {"type": "WORKS_AT", "head": "Vaswani", "tail": "Google", "chunks": [1], "confidence": 0.91}
                    ],
                    "events": [
                        {"type": "PUBLICATION", "trigger": "published", "chunks": [1, 5], "confidence": 0.89}
                    ]
                },
                "chunk_level_results": [
                    {
                        "chunk_id": 0,
                        "entities": 3,
                        "relations": 1,
                        "events": 0,
                        "inference_time": 0.65,
                        "avg_confidence": 0.91
                    },
                    {
                        "chunk_id": 1,
                        "entities": 5,
                        "relations": 2,
                        "events": 1,
                        "inference_time": 0.72,
                        "avg_confidence": 0.89
                    }
                ],
                "performance_metrics": {
                    "total_inference_time": 8.7,
                    "avg_time_per_chunk": 0.725,
                    "tokens_generated": 1240,
                    "tokens_per_second": 142.5,
                    "memory_usage": "2.1 GB"
                }
            }

            return mock_results

        except Exception as e:
            return {"error": f"Chunk extraction failed: {str(e)}"}

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


def create_generation_controls():
    """Create reusable generation parameter controls"""
    with gr.Accordion("🎛️ Generation Parameters", open=False):
        gr.Markdown("**Adjust model generation parameters for fine-tuning output quality and style**")

        with gr.Row():
            max_new_tokens = gr.Slider(
                minimum=50,
                maximum=2048,
                value=512,
                step=50,
                label="Max New Tokens",
                info="Maximum number of tokens to generate"
            )

            temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.1,
                step=0.1,
                label="Temperature",
                info="Controls randomness (0.0 = deterministic, higher = more creative)"
            )

        with gr.Row():
            top_p = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="Top-p (Nucleus)",
                info="Probability threshold for nucleus sampling"
            )

            top_k = gr.Slider(
                minimum=1,
                maximum=100,
                value=50,
                step=5,
                label="Top-k",
                info="Consider only top-k tokens for sampling"
            )

        with gr.Row():
            repetition_penalty = gr.Slider(
                minimum=1.0,
                maximum=2.0,
                value=1.0,
                step=0.05,
                label="Repetition Penalty",
                info="Penalty for repeating tokens (1.0 = no penalty)"
            )

            no_repeat_ngram_size = gr.Slider(
                minimum=0,
                maximum=5,
                value=0,
                step=1,
                label="No Repeat N-gram Size",
                info="Prevent repeating n-grams of this size"
            )

        with gr.Row():
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                label="Number of Beams",
                info="Beam search size (1 = greedy, higher = more thorough)"
            )

            do_sample = gr.Checkbox(
                label="Enable Sampling",
                value=True,
                info="Use sampling instead of greedy decoding"
            )

            early_stopping = gr.Checkbox(
                label="Early Stopping",
                value=False,
                info="Stop when EOS token is generated in beam search"
            )

        # Preset buttons
        with gr.Row():
            gr.Markdown("**Quick Presets:**")

        with gr.Row():
            conservative_btn = gr.Button("🎯 Conservative", size="sm")
            balanced_btn = gr.Button("⚖️ Balanced", size="sm")
            creative_btn = gr.Button("🎨 Creative", size="sm")
            precise_btn = gr.Button("🔬 Precise", size="sm")

    return (max_new_tokens, temperature, top_p, top_k, do_sample, repetition_penalty,
            no_repeat_ngram_size, num_beams, early_stopping, conservative_btn,
            balanced_btn, creative_btn, precise_btn)


def create_demo():
    """Create Gradio demo interface with document processing and generation controls"""

    # Initialize demo
    demo_instance = UIEDemo()

    # Sample texts for examples
    sample_texts = {
        "Business News": "Apple Inc announced that Tim Cook will step down as CEO next year. The company, founded by Steve Jobs in Cupertino, California, has been a major player in the technology industry.",
        "Sports News": "Lionel Messi scored two goals in the FIFA World Cup final held in Qatar. The Argentine forward helped his team defeat France 4-2 in the penalty shootout.",
        "Academic Text": "Dr. Sarah Johnson from MIT published a research paper on artificial intelligence in the Journal of Machine Learning. The study was conducted in collaboration with researchers from Stanford University."
    }

    def apply_generation_preset(preset_name, *current_values):
        """Apply generation parameter presets"""
        presets = {
            "conservative": {
                "temperature": 0.05,
                "top_p": 0.8,
                "top_k": 10,
                "do_sample": True,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 2,
                "num_beams": 1,
                "early_stopping": False
            },
            "balanced": {
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True,
                "repetition_penalty": 1.0,
                "no_repeat_ngram_size": 0,
                "num_beams": 1,
                "early_stopping": False
            },
            "creative": {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 80,
                "do_sample": True,
                "repetition_penalty": 1.05,
                "no_repeat_ngram_size": 0,
                "num_beams": 1,
                "early_stopping": False
            },
            "precise": {
                "temperature": 0.01,
                "top_p": 0.7,
                "top_k": 5,
                "do_sample": False,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
                "num_beams": 3,
                "early_stopping": True
            }
        }

        preset = presets[preset_name]
        # Return updated values keeping max_new_tokens unchanged
        return (
            current_values[0],  # max_new_tokens unchanged
            preset["temperature"],
            preset["top_p"],
            preset["top_k"],
            preset["do_sample"],
            preset["repetition_penalty"],
            preset["no_repeat_ngram_size"],
            preset["num_beams"],
            preset["early_stopping"]
        )

    def process_extraction(text, task, entity_types, relation_types, event_types, argument_types, mode,
                           max_new_tokens, temperature, top_p, top_k, do_sample, repetition_penalty,
                           no_repeat_ngram_size, num_beams, early_stopping):
        """Process extraction with generation parameters"""
        result = demo_instance.extract_information(
            text, task, entity_types, relation_types, event_types, argument_types, mode,
            max_new_tokens, temperature, top_p, top_k, do_sample, repetition_penalty,
            no_repeat_ngram_size, num_beams, early_stopping
        )

        if "error" in result:
            return result["error"], ""

        # Pretty print JSON
        formatted_json = json.dumps(result, indent=2, ensure_ascii=False)

        # Create enhanced summary with generation info
        summary_lines = []
        if "entities" in result:
            summary_lines.append(f"🏷️ Entities: {len(result['entities'])}")
        if "relations" in result:
            summary_lines.append(f"🔗 Relations: {len(result['relations'])}")
        if "events" in result:
            summary_lines.append(f"📅 Events: {len(result['events'])}")

        # Add generation info
        if "generation_info" in result:
            gen_info = result["generation_info"]["parameters_used"]
            summary_lines.append(f"🎛️ Temp: {gen_info['temperature']}")
            summary_lines.append(f"🎯 Tokens: {gen_info['max_new_tokens']}")

        summary = " | ".join(summary_lines) if summary_lines else "No results found"

        return summary, formatted_json

    def process_uploaded_document(file, processing_mode, use_gpu, enable_ocr, enable_table_structure,
                                  enable_cleaning, aggressive_clean, enable_chunking, chunking_strategy,
                                  chunk_size, chunk_overlap, task_type,
                                  max_new_tokens, temperature, top_p, top_k, do_sample, repetition_penalty,
                                  no_repeat_ngram_size, num_beams, early_stopping):
        """Process uploaded document with generation parameters"""
        if file is None:
            return "Please upload a file first", "", ""

        try:
            result = demo_instance.process_document(
                file.name, processing_mode, use_gpu, enable_ocr, enable_table_structure,
                enable_cleaning, aggressive_clean, enable_chunking, chunking_strategy,
                chunk_size, chunk_overlap, task_type,
                max_new_tokens, temperature, top_p, top_k, do_sample, repetition_penalty,
                no_repeat_ngram_size, num_beams, early_stopping
            )

            if "error" in result:
                return result["error"], "", ""

            # Format enhanced summary with generation info
            gen_info = result["generation_info"]
            summary = f"""
📄 **File:** {Path(result['file_path']).name}
⏱️ **Processing Time:** {result['processing_time']:.2f}s
📊 **Original Content:** {result['original_content_length']:,} chars
🧹 **Cleaned Content:** {result['cleaned_content_length']:,} chars
📝 **Chunks Created:** {result['chunks_created']}
📈 **Avg Chunk Size:** {result['metadata']['chunking_stats']['avg_chunk_size']} chars
🎯 **Strategy:** {result['metadata']['chunking_stats']['strategy_used']}

**🎛️ Generation Settings:**
- Temperature: {gen_info['parameters_used']['temperature']}
- Max Tokens: {gen_info['parameters_used']['max_new_tokens']}
- Top-p: {gen_info['parameters_used']['top_p']}
- Inference Time: {gen_info['total_inference_time']:.2f}s
            """.strip()

            # Format detailed results
            detailed_json = json.dumps(result, indent=2, ensure_ascii=False)

            # Format chunks preview
            chunks_preview = ""
            if "sample_chunks" in result:
                chunks_preview = "\n\n".join([
                    f"**Chunk {chunk['chunk_id']} ({chunk['size']} chars, {chunk['tokens']} tokens):**\n{chunk['content']}"
                    for chunk in result["sample_chunks"]
                ])

            return summary, detailed_json, chunks_preview

        except Exception as e:
            return f"Error: {str(e)}", "", ""

    def extract_from_document_chunks(chunks_data, task, entity_types, relation_types, event_types, argument_types, mode,
                                     max_new_tokens, temperature, top_p, top_k, do_sample, repetition_penalty,
                                     no_repeat_ngram_size, num_beams, early_stopping):
        """Extract information from document chunks with generation parameters"""
        if not chunks_data.strip():
            return "No chunks data available", ""

        try:
            result = demo_instance.extract_from_chunks(
                chunks_data, task, entity_types, relation_types, event_types, argument_types, mode,
                max_new_tokens, temperature, top_p, top_k, do_sample, repetition_penalty,
                no_repeat_ngram_size, num_beams, early_stopping
            )

            if "error" in result:
                return result["error"], ""

            # Format enhanced summary with generation and performance info
            perf = result["performance_metrics"]
            gen_params = result["generation_parameters"]

            summary = f"""
📊 **Total Chunks Processed:** {result['total_chunks_processed']}
⏱️ **Processing Time:** {result['processing_time']:.2f}s
🏷️ **Total Entities:** {len(result['aggregated_results']['entities'])}
🔗 **Total Relations:** {len(result['aggregated_results']['relations'])}
📅 **Total Events:** {len(result['aggregated_results']['events'])}

**🚀 Performance:**
- Avg Time/Chunk: {perf['avg_time_per_chunk']:.3f}s
- Tokens Generated: {perf['tokens_generated']:,}
- Tokens/Second: {perf['tokens_per_second']:.1f}
- Memory Usage: {perf['memory_usage']}

**🎛️ Generation Settings:**
- Temperature: {gen_params['temperature']}
- Max Tokens: {gen_params['max_new_tokens']}
- Top-p: {gen_params['top_p']}
- Beams: {gen_params['num_beams']}
            """.strip()

            # Format detailed results
            detailed_json = json.dumps(result, indent=2, ensure_ascii=False)

            return summary, detailed_json

        except Exception as e:
            return f"Error: {str(e)}", ""

    # Create Gradio interface với tabs và generation controls
    with gr.Blocks(title="Universal Information Extraction Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔍 Universal Information Extraction (UIE) Demo

        Extract entities, relations, and events from text or documents using fine-tuned LLaMA models with customizable generation parameters.
        """)

        with gr.Tabs():
            # Tab 1: Text Extraction
            with gr.TabItem("📝 Text Extraction"):
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

                        # Generation controls for text extraction
                        text_gen_controls = create_generation_controls()

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

            # Tab 2: Document Processing
            with gr.TabItem("📄 Document Processing"):
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
                        with gr.Accordion("⚙️ Processing Options", open=True):
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

                        # Text Cleaning Options
                        with gr.Accordion("🧹 Text Cleaning Options", open=True):
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

                        # Chunking Options
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

                        # Generation controls for document processing
                        doc_gen_controls = create_generation_controls()

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

            # Tab 3: Batch Extraction from Chunks
            with gr.TabItem("🔄 Batch Chunk Extraction"):
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

                        # Generation controls for chunk extraction
                        chunk_gen_controls = create_generation_controls()

                        # Schema inputs for chunk extraction
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

                        # Batch processing options
                        with gr.Accordion("⚙️ Batch Processing Options", open=True):
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

                        # Extract from chunks button
                        extract_chunks_btn = gr.Button("🚀 Extract from Chunks", variant="primary", size="lg")

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

        # Event handlers for Text Extraction tab
        def load_sample_text(sample_name):
            if sample_name:
                return sample_texts[sample_name]
            return ""

        sample_dropdown.change(
            fn=load_sample_text,
            inputs=[sample_dropdown],
            outputs=[text_input]
        )

        # Generation preset handlers for text tab
        text_gen_controls[9].click(  # conservative_btn
            fn=lambda *args: apply_generation_preset("conservative", *args),
            inputs=list(text_gen_controls[:9]),
            outputs=list(text_gen_controls[:9])
        )

        text_gen_controls[10].click(  # balanced_btn
            fn=lambda *args: apply_generation_preset("balanced", *args),
            inputs=list(text_gen_controls[:9]),
            outputs=list(text_gen_controls[:9])
        )

        text_gen_controls[11].click(  # creative_btn
            fn=lambda *args: apply_generation_preset("creative", *args),
            inputs=list(text_gen_controls[:9]),
            outputs=list(text_gen_controls[:9])
        )

        text_gen_controls[12].click(  # precise_btn
            fn=lambda *args: apply_generation_preset("precise", *args),
            inputs=list(text_gen_controls[:9]),
            outputs=list(text_gen_controls[:9])
        )

        extract_btn.click(
            fn=process_extraction,
            inputs=[
                       text_input, task_dropdown,
                       entity_types_input, relation_types_input,
                       event_types_input, argument_types_input,
                       mode_dropdown
                   ] + list(text_gen_controls[:9]),
            outputs=[summary_output, json_output]
        )

        # Generation preset handlers for document tab
        doc_gen_controls[9].click(  # conservative_btn
            fn=lambda *args: apply_generation_preset("conservative", *args),
            inputs=list(doc_gen_controls[:9]),
            outputs=list(doc_gen_controls[:9])
        )

        doc_gen_controls[10].click(  # balanced_btn
            fn=lambda *args: apply_generation_preset("balanced", *args),
            inputs=list(doc_gen_controls[:9]),
            outputs=list(doc_gen_controls[:9])
        )

        doc_gen_controls[11].click(  # creative_btn
            fn=lambda *args: apply_generation_preset("creative", *args),
            inputs=list(doc_gen_controls[:9]),
            outputs=list(doc_gen_controls[:9])
        )

        doc_gen_controls[12].click(  # precise_btn
            fn=lambda *args: apply_generation_preset("precise", *args),
            inputs=list(doc_gen_controls[:9]),
            outputs=list(doc_gen_controls[:9])
        )

        # Event handlers for Document Processing tab
        process_doc_btn.click(
            fn=process_uploaded_document,
            inputs=[
                       file_upload, processing_mode, use_gpu, enable_ocr, enable_table_structure,
                       enable_cleaning, aggressive_clean, enable_chunking, chunking_strategy,
                       chunk_size, chunk_overlap, task_type
                   ] + list(doc_gen_controls[:9]),
            outputs=[doc_summary_output, doc_json_output, chunks_preview]
        )

        # Generation preset handlers for chunk tab
        chunk_gen_controls[9].click(  # conservative_btn
            fn=lambda *args: apply_generation_preset("conservative", *args),
            inputs=list(chunk_gen_controls[:9]),
            outputs=list(chunk_gen_controls[:9])
        )

        chunk_gen_controls[10].click(  # balanced_btn
            fn=lambda *args: apply_generation_preset("balanced", *args),
            inputs=list(chunk_gen_controls[:9]),
            outputs=list(chunk_gen_controls[:9])
        )

        chunk_gen_controls[11].click(  # creative_btn
            fn=lambda *args: apply_generation_preset("creative", *args),
            inputs=list(chunk_gen_controls[:9]),
            outputs=list(chunk_gen_controls[:9])
        )

        chunk_gen_controls[12].click(  # precise_btn
            fn=lambda *args: apply_generation_preset("precise", *args),
            inputs=list(chunk_gen_controls[:9]),
            outputs=list(chunk_gen_controls[:9])
        )

        # Event handlers for Chunk Extraction tab
        extract_chunks_btn.click(
            fn=extract_from_document_chunks,
            inputs=[
                       chunks_data_storage, chunk_task_dropdown,
                       chunk_entity_types, chunk_relation_types,
                       chunk_event_types, chunk_argument_types,
                       chunk_mode_dropdown
                   ] + list(chunk_gen_controls[:9]),
            outputs=[chunk_extract_summary, chunk_extract_results]
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