# app/utils/graph_utils.py
import networkx as nx
from pyvis.network import Network
import json
import tempfile
from datetime import datetime
import base64
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build NetworkX graphs from UIE extraction results"""

    def __init__(self):
        self.graph = nx.Graph()
        self.node_counter = 0
        self.entity_nodes = {}
        self.event_nodes = {}

    def create_graph_from_results(self, results: Dict[str, Any]) -> nx.Graph:
        """Create NetworkX graph from extraction results"""
        logger.info(f"🔍 Building graph from results: {results}")

        self.graph.clear()
        self.node_counter = 0
        self.entity_nodes.clear()
        self.event_nodes.clear()

        # Add entities as nodes
        if 'entities' in results and results['entities']:
            logger.info(f"📊 Adding {len(results['entities'])} entities")
            self._add_entity_nodes(results['entities'])
        else:
            logger.info("📊 No entities found")

        # Add relations as edges
        if 'relations' in results and results['relations']:
            logger.info(f"🔗 Adding {len(results['relations'])} relations")
            self._add_relation_edges(results['relations'])
        else:
            logger.info("🔗 No relations found")

        # Add events as nodes with arguments
        if 'events' in results and results['events']:
            logger.info(f"📅 Adding {len(results['events'])} events")
            self._add_event_nodes(results['events'])
        else:
            logger.info("📅 No events found")

        logger.info(f"✅ Created graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")

        # Debug: Print node and edge info
        for node_id, data in self.graph.nodes(data=True):
            logger.debug(f"Node: {node_id} -> {data}")
        for source, target, data in self.graph.edges(data=True):
            logger.debug(f"Edge: {source} -> {target} -> {data}")

        return self.graph

    def _add_entity_nodes(self, entities: List[Dict]):
        """Add entity nodes to graph"""
        for entity in entities:
            entity_mention = entity.get('entity_mention', '')
            entity_type = entity.get('entity_type', 'UNKNOWN')

            if entity_mention and entity_mention not in self.entity_nodes:
                node_id = f"entity_{self.node_counter}"
                self.node_counter += 1

                self.graph.add_node(
                    node_id,
                    label=entity_mention,
                    title=f"Type: {entity_type}\nMention: {entity_mention}",
                    type='entity',
                    entity_type=entity_type,
                    group=entity_type,
                    size=25
                )

                self.entity_nodes[entity_mention] = node_id

    def _add_relation_edges(self, relations: List[Dict]):
        """Add relation edges to graph"""
        for relation in relations:
            relation_type = relation.get('relation_type', 'UNKNOWN')
            head_entity = relation.get('head_entity', '')
            tail_entity = relation.get('tail_entity', '')

            # Get or create nodes for entities
            head_node = self._get_or_create_entity_node(head_entity)
            tail_node = self._get_or_create_entity_node(tail_entity)

            if head_node and tail_node:
                self.graph.add_edge(
                    head_node,
                    tail_node,
                    label=relation_type,
                    title=f"{relation_type}: {head_entity} → {tail_entity}",
                    type='relation',
                    relation_type=relation_type,
                    weight=3
                )

    def _add_event_nodes(self, events: List[Dict]):
        """Add event nodes and their arguments"""
        for event_data in events:
            # Handle both nested format và flat format
            events_to_process = []

            if 'events' in event_data and isinstance(event_data['events'], list):
                # Nested format: {'events': [event1, event2, ...]}
                events_to_process = event_data['events']
            else:
                # Flat format: direct event object
                events_to_process = [event_data]

            for event in events_to_process:
                trigger = event.get('trigger', '')
                trigger_type = event.get('trigger_type', 'UNKNOWN')

                if trigger:
                    # Create event node
                    event_node_id = f"event_{self.node_counter}"
                    self.node_counter += 1

                    self.graph.add_node(
                        event_node_id,
                        label=trigger,
                        title=f"Event: {trigger_type}\nTrigger: {trigger}",
                        type='event',
                        event_type=trigger_type,
                        group='EVENT',
                        size=30
                    )

                    # Add arguments as edges to event
                    arguments = event.get('arguments', [])
                    for arg in arguments:
                        role = arg.get('role', '')
                        entity = arg.get('entity', '')

                        if entity:
                            entity_node = self._get_or_create_entity_node(entity)
                            if entity_node:
                                self.graph.add_edge(
                                    event_node_id,
                                    entity_node,
                                    label=role,
                                    title=f"Role: {role}\nEntity: {entity}",
                                    type='argument',
                                    role=role,
                                    weight=2
                                )

    def _get_or_create_entity_node(self, entity_mention: str) -> Optional[str]:
        """Get existing entity node or create new one"""
        if not entity_mention:
            return None

        if entity_mention in self.entity_nodes:
            return self.entity_nodes[entity_mention]

        # Create new entity node
        node_id = f"entity_{self.node_counter}"
        self.node_counter += 1

        self.graph.add_node(
            node_id,
            label=entity_mention,
            title=f"Entity: {entity_mention}",
            type='entity',
            entity_type='INFERRED',
            group='INFERRED',
            size=20
        )

        self.entity_nodes[entity_mention] = node_id
        return node_id


class GraphVisualizer:
    """Create interactive visualizations using PyVis"""

    def __init__(self):
        self.color_schemes = {
            'entity': '#3498db',
            'relation': '#e74c3c',
            'event': '#f39c12',
            'background': '#ffffff'
        }

    def create_interactive_graph(self, nx_graph: nx.Graph, **kwargs) -> str:
        """Create interactive PyVis graph from NetworkX graph"""
        try:
            # Extract settings
            layout = kwargs.get('graph_layout', 'force_atlas_2based')
            show_entities = kwargs.get('show_entities', True)
            show_relations = kwargs.get('show_relations', True)
            show_events = kwargs.get('show_events', True)
            node_size_multiplier = kwargs.get('node_size', 25) / 25
            edge_width = kwargs.get('edge_width', 3)
            physics_enabled = kwargs.get('physics_enabled', True)
            show_buttons = kwargs.get('show_buttons', True)

            # Update color scheme
            self.color_schemes.update({
                'entity': kwargs.get('entity_color', '#3498db'),
                'relation': kwargs.get('relation_color', '#e74c3c'),
                'event': kwargs.get('event_color', '#f39c12'),
                'background': kwargs.get('background_color', '#ffffff')
            })

            # Create PyVis network
            net = Network(
                height="600px",
                width="100%",
                bgcolor=self.color_schemes['background'],
                font_color="black" if self._is_light_color(self.color_schemes['background']) else "white",
            )

            # Filter and add nodes
            filtered_graph = self._filter_graph(nx_graph, show_entities, show_relations, show_events)

            # Process nodes
            for node_id, node_data in filtered_graph.nodes(data=True):
                node_type = node_data.get('type', 'unknown')
                color = self._get_node_color(node_type, node_data)
                size = node_data.get('size', 25) * node_size_multiplier

                net.add_node(
                    node_id,
                    label=node_data.get('label', str(node_id)),
                    title=node_data.get('title', ''),
                    color=color,
                    size=size,
                    font={'size': max(12, int(size * 0.6))}
                )

            # Process edges
            for source, target, edge_data in filtered_graph.edges(data=True):
                edge_type = edge_data.get('type', 'unknown')
                color = self._get_edge_color(edge_type)
                width = edge_data.get('weight', 1) * edge_width

                net.add_edge(
                    source,
                    target,
                    label=edge_data.get('label', ''),
                    title=edge_data.get('title', ''),
                    color=color,
                    width=width
                )

            # Apply layout
            self._apply_layout(net, layout)

            # Configure physics
            if physics_enabled:
                net.toggle_physics(True)
            else:
                net.toggle_physics(False)

            # Show control buttons
            if show_buttons:
                net.show_buttons(filter_=['physics'])

            # Generate HTML
            html_content = net.generate_html()
            with open('test_basic.html', 'w', encoding='utf-8') as f:
                f.write(html_content)

            return self._create_data_url_iframe(html_content)
        except Exception as e:
            logger.error(f"❌ PyVis visualization failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def _create_data_url_iframe(self, html_content: str) -> str:
        """Create iframe with data URL from HTML content"""
        try:
            # Encode as data URL
            html_bytes = html_content.encode('utf-8')
            html_b64 = base64.b64encode(html_bytes).decode('utf-8')
            data_url = f"data:text/html;base64,{html_b64}"

            # Create iframe with data URL
            iframe_html = f"""
            <div style="width: 100%; height: 620px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; background: white;">
                <iframe src="{data_url}" 
                        width="100%" 
                        height="100%" 
                        frameborder="0"
                        style="border: none;">
                    Your browser does not support data URL iframes.
                </iframe>
            </div>
            <div style="margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; font-size: 12px; color: #666;">
                🎮 Interactive graph: Drag nodes, scroll to zoom, pan to explore
            </div>
            """

            return iframe_html

        except Exception as e:
            logger.error(f"❌ Data URL creation failed: {e}")
            return f"""
            <div style="padding: 20px; border: 2px solid #e74c3c; border-radius: 10px; background: #fdf2f2;">
                <h3>❌ Data URL Method Failed</h3>
                <p>Error: {e}</p>
                <p>Falling back to direct HTML content...</p>
            </div>
            """

    def _create_data_url_iframe(self, html_content: str) -> str:
        """Create iframe with data URL from HTML content"""
        try:
            # Encode as data URL
            html_bytes = html_content.encode('utf-8')
            html_b64 = base64.b64encode(html_bytes).decode('utf-8')
            data_url = f"data:text/html;base64,{html_b64}"

            # Create iframe with data URL
            iframe_html = f"""
            <div style="width: 100%; height: 620px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden; background: white;">
                <iframe src="{data_url}" 
                        width="100%" 
                        height="100%" 
                        frameborder="0"
                        style="border: none;">
                    Your browser does not support data URL iframes.
                </iframe>
            </div>
            <div style="margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; font-size: 12px; color: #666;">
                🎮 Interactive graph: Drag nodes, scroll to zoom, pan to explore
            </div>
            """

            return iframe_html

        except Exception as e:
            logger.error(f"❌ Data URL creation failed: {e}")
            return f"""
            <div style="padding: 20px; border: 2px solid #e74c3c; border-radius: 10px; background: #fdf2f2;">
                <h3>❌ Data URL Method Failed</h3>
                <p>Error: {e}</p>
                <p>Falling back to direct HTML content...</p>
            </div>
            """

    def _filter_graph(self, graph: nx.Graph, show_entities: bool,
                      show_relations: bool, show_events: bool) -> nx.Graph:
        """Filter graph based on visibility settings"""
        filtered = nx.Graph()

        # Add nodes based on filters
        for node_id, node_data in graph.nodes(data=True):
            node_type = node_data.get('type', 'unknown')

            should_include = (
                    (node_type == 'entity' and show_entities) or
                    (node_type == 'event' and show_events) or
                    node_type not in ['entity', 'event']
            )

            if should_include:
                filtered.add_node(node_id, **node_data)

        # Add edges based on filters and node inclusion
        for source, target, edge_data in graph.edges(data=True):
            edge_type = edge_data.get('type', 'unknown')

            should_include_edge = (
                    (edge_type == 'relation' and show_relations) or
                    (edge_type in ['argument', 'unknown'])
            )

            if (should_include_edge and
                    source in filtered.nodes and
                    target in filtered.nodes):
                filtered.add_edge(source, target, **edge_data)

        return filtered

    def _get_node_color(self, node_type: str, node_data: Dict) -> str:
        """Get color for node based on type"""
        if node_type == 'entity':
            return self.color_schemes['entity']
        elif node_type == 'event':
            return self.color_schemes['event']
        else:
            return '#95a5a6'  # Default gray

    def _get_edge_color(self, edge_type: str) -> str:
        """Get color for edge based on type"""
        if edge_type == 'relation':
            return self.color_schemes['relation']
        elif edge_type == 'argument':
            return '#9b59b6'  # Purple for event arguments
        else:
            return '#bdc3c7'  # Light gray for others

    def _apply_layout(self, net: Network, layout: str):
        """Apply layout algorithm to network"""
        if layout == 'force_atlas_2based':
            net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100)
        elif layout == 'hierarchical':
            net.hierarchical_layout(direction='UD', sort_method='directed')
        elif layout == 'random':
            net.set_options('{"layout": {"randomSeed": 42}}')
        # circular layout is default for small graphs

    def _is_light_color(self, hex_color: str) -> bool:
        """Check if color is light (for font color selection)"""
        try:
            hex_color = hex_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            return brightness > 128
        except:
            return True

    def _patch_for_gradio(self, html_content: str) -> str:
        """Patch HTML for Gradio compatibility"""
        try:
            # Replace CDN links with inline notice
            html_content = html_content.replace(
                'src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"',
                'src="https://unpkg.com/vis-network@9.1.2/standalone/umd/vis-network.min.js"'
            )

            html_content = html_content.replace(
                'href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css"',
                'href="https://unpkg.com/vis-network@9.1.2/dist/vis-network.css"'
            )

            # Remove bootstrap (not needed for graph)
            import re
            html_content = re.sub(r'<link[^>]*bootstrap[^>]*>', '', html_content)
            html_content = re.sub(r'<script[^>]*bootstrap[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)

            # Remove config panel (causes issues in Gradio)
            html_content = html_content.replace('<div id="config"></div>', '')
            html_content = html_content.replace('options.configure["container"] = document.getElementById("config");',
                                                '')

            # Simplify body structure
            html_content = re.sub(
                r'<body>.*?<div id="mynetwork"[^>]*></div>.*?</body>',
                '<body><div id="mynetwork" style="width: 100%; height: 500px; border: 1px solid #ccc;"></div></body>',
                html_content,
                flags=re.DOTALL
            )

            logger.info(f"✅ Patched HTML for Gradio ({len(html_content)} chars)")
            return html_content
        except Exception as e:
            logger.error(f"❌ Failed to patch HTML: {e}")
            return html_content  # Return original if patching fails

    def _create_empty_graph_html(self) -> str:
        """Create HTML for empty graph"""
        return """
        <div style='text-align: center; padding: 50px; border: 2px dashed #ddd; border-radius: 10px; background-color: #f9f9f9;'>
            <h3>📊 Empty Graph</h3>
            <p>No nodes to display after filtering.</p>
            <p>Try adjusting your filter settings.</p>
        </div>
        """

    def _create_fallback_html(self, graph: nx.Graph) -> str:
        """Create fallback HTML when PyVis fails"""
        nodes_info = []
        for node_id, data in list(graph.nodes(data=True))[:10]:  # Show first 10
            label = data.get('label', str(node_id))
            node_type = data.get('type', 'unknown')
            nodes_info.append(f"<li><strong>{label}</strong> ({node_type})</li>")

        edges_info = []
        for source, target, data in list(graph.edges(data=True))[:10]:  # Show first 10
            label = data.get('label', 'connection')
            edges_info.append(f"<li>{source} --{label}--> {target}</li>")

        return f"""
        <div style='padding: 20px; border: 2px solid #f39c12; border-radius: 10px; background-color: #fffbf0;'>
            <h3>⚠️ Visualization Fallback Mode</h3>
            <p>Interactive visualization failed. Here's your graph data:</p>

            <h4>📊 Nodes ({len(graph.nodes)}):</h4>
            <ul>{''.join(nodes_info)}</ul>
            {f'<p>... and {len(graph.nodes) - 10} more nodes</p>' if len(graph.nodes) > 10 else ''}

            <h4>🔗 Edges ({len(graph.edges)}):</h4>
            <ul>{''.join(edges_info)}</ul>
            {f'<p>... and {len(graph.edges) - 10} more edges</p>' if len(graph.edges) > 10 else ''}

            <p style="margin-top: 15px; color: #666;">
                💡 <strong>Tip:</strong> Use the Export button to get a working interactive HTML file.
            </p>
        </div>
        """


def calculate_graph_statistics(graph: nx.Graph) -> Dict[str, Any]:
    """Calculate various graph statistics"""
    stats = {
        'nodes': len(graph.nodes),
        'edges': len(graph.edges),
        'density': nx.density(graph),
        'is_connected': nx.is_connected(graph),
        'components': nx.number_connected_components(graph)
    }

    # Node type breakdown
    node_types = {}
    for _, node_data in graph.nodes(data=True):
        node_type = node_data.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1

    stats['node_types'] = node_types

    # Edge type breakdown
    edge_types = {}
    for _, _, edge_data in graph.edges(data=True):
        edge_type = edge_data.get('type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

    stats['edge_types'] = edge_types

    # Centrality measures (for small graphs)
    if len(graph.nodes) < 100:
        try:
            centrality = nx.degree_centrality(graph)
            stats['avg_centrality'] = sum(centrality.values()) / len(centrality) if centrality else 0
            stats['most_central_node'] = max(centrality, key=centrality.get) if centrality else None
        except:
            stats['avg_centrality'] = 0
            stats['most_central_node'] = None

    return stats


def export_graph(graph: nx.Graph, format_type: str, filename_prefix: str = "graph_export") -> str:
    """Export graph in different formats"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if format_type == "GraphML":
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix=f'_{filename_prefix}_{timestamp}.graphml',
            delete=False,
            encoding='utf-8'
        )
        nx.write_graphml(graph, temp_file.name)
        temp_file.close()
        return temp_file.name

    elif format_type == "JSON":
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix=f'_{filename_prefix}_{timestamp}.json',
            delete=False,
            encoding='utf-8'
        )

        # Convert to JSON-serializable format
        data = nx.node_link_data(graph)
        json.dump(data, temp_file, indent=2, ensure_ascii=False)
        temp_file.close()
        return temp_file.name

    else:  # HTML
        visualizer = GraphVisualizer()
        html_content = visualizer.create_interactive_graph(graph)

        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix=f'_{filename_prefix}_{timestamp}.html',
            delete=False,
            encoding='utf-8'
        )
        temp_file.write(html_content)
        temp_file.close()
        return temp_file.name
