from typing import Tuple, Dict, Any, Optional
from .base import BaseHandler
from ...utils.graph_utils import GraphBuilder, GraphVisualizer, calculate_graph_statistics, export_graph
from ...utils.data_extractor import extract_uie_data
import json
import logging

logger = logging.getLogger(__name__)


class VisualizationHandler(BaseHandler):
    """Handler for graph visualization operations"""

    def __init__(self, context):
        super().__init__(context)
        self.graph_builder = GraphBuilder()
        self.graph_visualizer = GraphVisualizer()
        self.current_graph = None
        self.current_results = None

    def generate_visualization(self, data_source: str, json_file, graph_layout: str,
                               show_entities: bool, show_relations: bool, show_events: bool,
                               node_size: int, edge_width: int, physics_enabled: bool,
                               show_buttons: bool, entity_color: str, relation_color: str,
                               event_color: str, background_color: str) -> Tuple[str, str]:
        try:
            raw_data = self._get_extraction_data(data_source, json_file)
            if not raw_data:
                return self._create_error_response("No extraction data available", "Visualization")

            print(f"📥 Raw data received: {type(raw_data)} - {str(raw_data)[:200]}...")

            # Extract and normalize UIE data using the smart extractor
            normalized_data = extract_uie_data(raw_data)
            print(f"🔄 Normalized data: {normalized_data}")

            if not any(normalized_data.values()):
                return self._create_empty_visualization(), self._create_stats_markdown({})

            # Build graph from normalized data
            self.current_graph = self.graph_builder.create_graph_from_results(normalized_data)
            self.current_results = normalized_data

            print(f"✅ Created Graph with {len(self.current_graph.nodes)} nodes, {len(self.current_graph.edges)} edges!")

            if len(self.current_graph.nodes) == 0:
                return self._create_empty_visualization(), self._create_stats_markdown({})

            # Prepare visualization settings
            viz_settings = {
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
                'background_color': background_color
            }

            # Create visualization
            html_content = self.graph_visualizer.create_interactive_graph(
                self.current_graph, **viz_settings
            )

            print("✅ Set HTML content for graph!")

            # Calculate statistics
            stats = calculate_graph_statistics(self.current_graph)
            print("✅ Statisctics!")

            stats_markdown = self._create_stats_markdown(stats)
            print("✅ Mardown")

            self._log_operation("Visualization generated",
                                nodes=len(self.current_graph.nodes),
                                edges=len(self.current_graph.edges))

            return html_content, stats_markdown

        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return self._create_error_response(str(e), "Visualization")

    def export_current_graph(self, export_format: str) -> str:
        """Export current graph in specified format"""

        if not self.current_graph:
            logger.warning("No graph available for export")
            return None

        try:
            export_file = export_graph(self.current_graph, export_format, "uie_extraction")
            self._log_operation("Graph exported", format=export_format, file=export_file)
            return export_file

        except Exception as e:
            logger.error(f"Graph export failed: {e}")
            return None

    def _get_extraction_data(self, data_source: str, json_file) -> Optional[Dict]:
        if data_source == "Upload JSON" and json_file:
            try:
                with open(json_file.name, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load JSON file: {e}")
                return None

        elif data_source == "Current Text Results":
            return self.context.get_last_text_results()

        elif data_source == "Current Document Results":
            return self.context.get_last_document_results()

    def _create_empty_visualization(self) -> str:
        """Create empty visualization message"""
        return """
        <div style='text-align: center; padding: 50px; color: #666; border: 2px dashed #ddd; border-radius: 10px;'>
            <h3>📊 No Data to Visualize</h3>
            <p>No entities, relations, or events found in the extraction results.</p>
            <p>Try adjusting your extraction parameters or use different input data.</p>
        </div>
        """

    def _create_stats_markdown(self, stats: Dict[str, Any]) -> str:
        """Create formatted statistics markdown"""

        if not stats:
            return "No statistics available."

        markdown = f"""
## 📈 Graph Statistics

### 📊 Basic Metrics
- **Nodes:** {stats.get('nodes', 0)}
- **Edges:** {stats.get('edges', 0)}
- **Density:** {stats.get('density', 0):.3f}
- **Connected:** {'✅' if stats.get('is_connected', False) else '❌'}
- **Components:** {stats.get('components', 0)}

### 🏷️ Node Types
"""

        node_types = stats.get('node_types', {})
        for node_type, count in node_types.items():
            icon = {'entity': '🏷️', 'event': '📅', 'unknown': '❓'}.get(node_type, '🔘')
            markdown += f"- **{icon} {node_type.title()}:** {count}\n"

        markdown += "\n### 🔗 Edge Types\n"

        edge_types = stats.get('edge_types', {})
        for edge_type, count in edge_types.items():
            icon = {'relation': '🔗', 'argument': '🎯', 'unknown': '❓'}.get(edge_type, '🔘')
            markdown += f"- **{icon} {edge_type.title()}:** {count}\n"

        # Add centrality info if available
        if 'avg_centrality' in stats:
            markdown += f"\n### 🎯 Centrality\n"
            markdown += f"- **Average Centrality:** {stats['avg_centrality']:.3f}\n"
            if stats.get('most_central_node'):
                markdown += f"- **Most Central Node:** {stats['most_central_node']}\n"

        return markdown

    def _create_error_response(self, error_msg: str, operation: str = "Visualization") -> Tuple[str, str]:
        """Create error response for visualization"""
        error_html = f"""
        <div style='text-align: center; padding: 50px; color: #e74c3c; border: 2px solid #e74c3c; border-radius: 10px; background-color: #fdf2f2;'>
            <h3>❌ {operation} Error</h3>
            <p>{error_msg}</p>
        </div>
        """

        error_stats = f"**❌ Error:** {error_msg}"

        return error_html, error_stats