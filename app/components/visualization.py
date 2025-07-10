import gradio as gr
from .generation_controls import create_generation_controls, create_schema_inputs

def visualization_tab():
    """Create visualization tab for graph networks"""
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 🔗 Graph Visualization")
            
            # Data source selection
            data_source = gr.Radio(
                choices=["Current Text Results", "Current Document Results", "Upload JSON"],
                label="Data Source",
                value="Current Text Results",
                info="Choose the source of data to visualize"
            )
            
            # File upload for JSON data
            json_upload = gr.File(
                label="Upload Extraction Results (JSON)",
                file_types=[".json"],
                file_count="single",
                visible=False
            )
            
            # Graph settings
            with gr.Accordion("🎨 Graph Settings", open=False):
                graph_layout = gr.Dropdown(
                    choices=["force_atlas_2based", "hierarchical", "random", "circular"],
                    label="Layout Algorithm",
                    value="force_atlas_2based",
                    info="Choose graph layout algorithm"
                )
                
                show_entities = gr.Checkbox(
                    label="Show Entities",
                    value=True,
                    info="Display entity nodes"
                )
                
                show_relations = gr.Checkbox(
                    label="Show Relations", 
                    value=True,
                    info="Display relation edges"
                )
                
                show_events = gr.Checkbox(
                    label="Show Events",
                    value=True,
                    info="Display event nodes"
                )
                
                with gr.Row():
                    node_size = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=25,
                        step=5,
                        label="Node Size"
                    )
                    
                    edge_width = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Edge Width"
                    )
                
                with gr.Row():
                    physics_enabled = gr.Checkbox(
                        label="Enable Physics",
                        value=True,
                        info="Enable graph physics simulation"
                    )
                    
                    show_buttons = gr.Checkbox(
                        label="Show Control Buttons",
                        value=True,
                        info="Show interaction buttons on graph"
                    )
            
            # Color scheme
            with gr.Accordion("🌈 Color Scheme", open=False):
                entity_color = gr.ColorPicker(
                    label="Entity Color",
                    value="#3498db"
                )
                
                relation_color = gr.ColorPicker(
                    label="Relation Color", 
                    value="#e74c3c"
                )
                
                event_color = gr.ColorPicker(
                    label="Event Color",
                    value="#f39c12"
                )
                
                background_color = gr.ColorPicker(
                    label="Background Color",
                    value="#ffffff"
                )
            
            # Generate button
            generate_viz_btn = gr.Button(
                "🚀 Generate Visualization", 
                variant="primary", 
                size="lg"
            )
            
            # Export options
            with gr.Accordion("💾 Export Options", open=False):
                export_format = gr.Radio(
                    choices=["HTML", "JSON", "GraphML"],
                    label="Export Format",
                    value="HTML"
                )
                
                export_btn = gr.Button("📥 Export Graph", variant="secondary")
                
                download_file = gr.DownloadButton(
                    label="📁 Download",
                    visible=False
                )

        with gr.Column(scale=3):
            gr.Markdown("## 📊 Interactive Graph Visualization")
            
            # Graph display
            graph_html = gr.HTML(
                label="Graph Visualization",
                value="<div style='text-align: center; padding: 50px; color: #666;'>Generate a visualization to see the graph here</div>"
            )
            
            # Graph statistics
            with gr.Accordion("📈 Graph Statistics", open=False):
                graph_stats = gr.Markdown(
                    value="No graph generated yet."
                )
    
    # Show/hide JSON upload based on data source
    def toggle_json_upload(source):
        return gr.update(visible=(source == "Upload JSON"))
    
    data_source.change(
        fn=toggle_json_upload,
        inputs=[data_source],
        outputs=[json_upload]
    )
    
    return {
        'inputs': {
            'data_source': data_source,
            'json_upload': json_upload,
            'graph_layout': graph_layout,
            'show_entities': show_entities,
            'show_relations': show_relations,
            'show_events': show_events,
            'node_size': node_size,
            'edge_width': edge_width,
            'physics_enabled': physics_enabled,
            'show_buttons': show_buttons,
            'entity_color': entity_color,
            'relation_color': relation_color,
            'event_color': event_color,
            'background_color': background_color,
            'export_format': export_format,
            'generate_btn': generate_viz_btn,
            'export_btn': export_btn
        },
        'outputs': {
            'graph_html': graph_html,
            'graph_stats': graph_stats,
            'download_file': download_file
        }
    }