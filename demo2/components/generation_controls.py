import gradio as gr
from typing import Tuple, List

def create_generation_controls() -> Tuple:
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


def create_schema_inputs() -> Tuple:
    """Create reusable schema input controls"""
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
    
    return entity_types_input, relation_types_input, event_types_input, argument_types_input