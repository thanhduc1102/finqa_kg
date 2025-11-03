"""
Knowledge Graph Visualizer for FinQA
"""

import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for graph visualization"""
    node_size: int = 1000
    font_size: int = 8
    width: int = 1200
    height: int = 800
    edge_width: float = 1.0
    show_labels: bool = True
    show_edge_labels: bool = True
    node_color_map: Dict[str, str] = None
    edge_color_map: Dict[str, str] = None
    layout: str = "spring"  # spring, circular, random, shell

class GraphVisualizer:
    """Enhanced visualizer for FinQA Knowledge Graph"""
    
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph
        self.default_config = VisualizationConfig()
        self._setup_default_colors()
        
    def _setup_default_colors(self):
        """Setup default color schemes"""
        self.default_config.node_color_map = {
            'document': '#1f77b4',  # Blue
            'table': '#2ca02c',    # Green
            'text': '#ff7f0e',     # Orange
            'table_header': '#9467bd',  # Purple
            'table_cell': '#8c564b',    # Brown
            'number': '#e377c2',    # Pink
            'date': '#7f7f7f',     # Gray
            'entity': '#bcbd22',    # Olive
            'question': '#17becf'   # Cyan
        }
        
        self.default_config.edge_color_map = {
            'contains': '#1f77b4',
            'has_cell': '#2ca02c',
            'contains_number': '#ff7f0e',
            'contains_date': '#9467bd',
            'semantically_related': '#e377c2',
            'supports_answer': '#17becf'
        }

    def _get_node_positions(self, layout: str) -> Dict:
        """Get node positions based on layout algorithm"""
        if layout == "spring":
            return nx.spring_layout(self.graph)
        elif layout == "circular":
            return nx.circular_layout(self.graph)
        elif layout == "random":
            return nx.random_layout(self.graph)
        elif layout == "shell":
            return nx.shell_layout(self.graph)
        else:
            return nx.spring_layout(self.graph)

    def visualize_full_graph(
        self,
        config: Optional[VisualizationConfig] = None,
        output_path: Optional[str] = None
    ):
        """Visualize the entire knowledge graph"""
        if config is None:
            config = self.default_config
        # Use provided color maps, else fall back to defaults
        node_color_map = config.node_color_map or self.default_config.node_color_map
        edge_color_map = config.edge_color_map or self.default_config.edge_color_map

        plt.figure(figsize=(config.width//100, config.height//100))
        
        # Get positions
        pos = self._get_node_positions(config.layout)
        
        # Draw nodes
        for node_type in set(nx.get_node_attributes(self.graph, 'type').values()):
            node_list = [
                node for node, attr in self.graph.nodes(data=True)
                if attr.get('type') == node_type
            ]
            if node_list:
                nx.draw_networkx_nodes(
                    self.graph,
                    pos,
                    nodelist=node_list,
                    node_color=node_color_map.get(node_type, '#666666'),
                    node_size=config.node_size,
                    alpha=0.7,
                    label=node_type
                )
                
        # Draw edges
        for edge_type in set(nx.get_edge_attributes(self.graph, 'relation').values()):
            edge_list = [
                (u, v) for u, v, attr in self.graph.edges(data=True)
                if attr.get('relation') == edge_type
            ]
            if edge_list:
                nx.draw_networkx_edges(
                    self.graph,
                    pos,
                    edgelist=edge_list,
                    edge_color=edge_color_map.get(edge_type, '#666666'),
                    width=config.edge_width,
                    alpha=0.5,
                    label=edge_type
                )
                
        # Add labels if requested
        if config.show_labels:
            labels = {}
            for node, attr in self.graph.nodes(data=True):
                if 'content' in attr:
                    labels[node] = str(attr['content'])[:20] + '...'
                else:
                    labels[node] = str(attr.get('type', ''))
            nx.draw_networkx_labels(
                self.graph,
                pos,
                labels,
                font_size=config.font_size
            )
            
        plt.title("FinQA Knowledge Graph")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()

    def visualize_subgraph(
        self,
        center_node: str,
        depth: int = 2,
        config: Optional[VisualizationConfig] = None,
        output_path: Optional[str] = None
    ):
        """Visualize a subgraph centered on a specific node"""
        if config is None:
            config = self.default_config
        node_color_map = config.node_color_map or self.default_config.node_color_map
        edge_color_map = config.edge_color_map or self.default_config.edge_color_map
        # Get subgraph nodes
        nodes = set([center_node])
        current_nodes = {center_node}
        
        for _ in range(depth):
            next_nodes = set()
            for node in current_nodes:
                next_nodes.update(self.graph.predecessors(node))
                next_nodes.update(self.graph.successors(node))
            nodes.update(next_nodes)
            current_nodes = next_nodes
            
        subgraph = self.graph.subgraph(nodes)
        
        plt.figure(figsize=(config.width//100, config.height//100))
        
        # Get positions
        pos = self._get_node_positions(config.layout)
        
        # Draw nodes with different sizes based on distance from center
        distances = nx.shortest_path_length(subgraph, source=center_node)
        sizes = {
            node: config.node_size / (1 + dist)
            for node, dist in distances.items()
        }
        
        for node_type in set(nx.get_node_attributes(subgraph, 'type').values()):
            node_list = [
                node for node, attr in subgraph.nodes(data=True)
                if attr.get('type') == node_type
            ]
            if node_list:
                nx.draw_networkx_nodes(
                    subgraph,
                    pos,
                    nodelist=node_list,
                    node_color=node_color_map.get(node_type, '#666666'),
                    node_size=[sizes[node] for node in node_list],
                    alpha=0.7,
                    label=node_type
                )
                
        # Draw edges
        for edge_type in set(nx.get_edge_attributes(subgraph, 'relation').values()):
            edge_list = [
                (u, v) for u, v, attr in subgraph.edges(data=True)
                if attr.get('relation') == edge_type
            ]
            if edge_list:
                nx.draw_networkx_edges(
                    subgraph,
                    pos,
                    edgelist=edge_list,
                    edge_color=edge_color_map.get(edge_type, '#666666'),
                    width=config.edge_width,
                    alpha=0.5,
                    label=edge_type
                )
                
        # Add labels
        if config.show_labels:
            labels = {}
            for node, attr in subgraph.nodes(data=True):
                if node == center_node:
                    labels[node] = str(attr.get('content', ''))[:30]
                elif 'content' in attr:
                    labels[node] = str(attr['content'])[:20] + '...'
                else:
                    labels[node] = str(attr.get('type', ''))
            nx.draw_networkx_labels(
                subgraph,
                pos,
                labels,
                font_size=config.font_size
            )
            
        plt.title(f"Subgraph centered on {center_node}")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()

    def visualize_path(
        self,
        path: List[str],
        config: Optional[VisualizationConfig] = None,
        output_path: Optional[str] = None
    ):
        """Visualize a specific path in the graph"""
        if config is None:
            config = self.default_config
        node_color_map = config.node_color_map or self.default_config.node_color_map
        edge_color_map = config.edge_color_map or self.default_config.edge_color_map
        # Create subgraph from path
        path_edges = list(zip(path[:-1], path[1:]))
        subgraph = self.graph.subgraph(path)
        
        plt.figure(figsize=(config.width//100, config.height//100))
        
        # Get positions - use special layout for paths
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # Draw nodes
        for idx, node in enumerate(path):
            node_attr = subgraph.nodes[node]
            nx.draw_networkx_nodes(
                subgraph,
                pos,
                nodelist=[node],
                node_color=node_color_map.get(node_attr.get('type'), '#666666'),
                node_size=config.node_size,
                alpha=0.7
            )
            
        # Draw edges with arrows
        nx.draw_networkx_edges(
            subgraph,
            pos,
            edgelist=path_edges,
            edge_color='r',
            width=2,
            arrows=True,
            arrowsize=20
        )
        
        # Add labels
        if config.show_labels:
            labels = {}
            for node, attr in subgraph.nodes(data=True):
                if 'content' in attr:
                    labels[node] = f"{attr.get('type', '')}\n{str(attr['content'])[:20]}"
                else:
                    labels[node] = str(attr.get('type', ''))
            nx.draw_networkx_labels(
                subgraph,
                pos,
                labels,
                font_size=config.font_size
            )
            
        # Add edge labels if requested
        if config.show_edge_labels:
            edge_labels = {}
            for u, v in path_edges:
                edge_data = subgraph.get_edge_data(u, v)
                if edge_data:
                    # Get the first edge if there are multiple
                    first_edge = next(iter(edge_data.values()))
                    edge_labels[(u, v)] = first_edge.get('relation', '')
            nx.draw_networkx_edge_labels(
                subgraph,
                pos,
                edge_labels,
                font_size=config.font_size-2
            )
            
        plt.title("Path Visualization")
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()

    def create_interactive_visualization(
        self,
        output_path: str,
        config: Optional[VisualizationConfig] = None
    ):
        """Create interactive HTML visualization using Plotly"""
        if config is None:
            config = self.default_config
        # Fallback color maps
        node_color_map = config.node_color_map or self.default_config.node_color_map
        edge_color_map = config.edge_color_map or self.default_config.edge_color_map

        # Get positions
        pos = self._get_node_positions(config.layout)
        
        # Prepare node traces
        node_traces = {}
        for node_type in set(nx.get_node_attributes(self.graph, 'type').values()):
            node_list = [
                node for node, attr in self.graph.nodes(data=True)
                if attr.get('type') == node_type
            ]
            if node_list:
                x = [pos[node][0] for node in node_list]
                y = [pos[node][1] for node in node_list]
                
                node_traces[node_type] = go.Scatter(
                    x=x,
                    y=y,
                    mode='markers+text',
                    name=node_type,
                    marker=dict(
                        size=20,
                        color=node_color_map.get(node_type, '#666666')
                    ),
                    text=[
                        self.graph.nodes[node].get('content', '')[:20]
                        for node in node_list
                    ],
                    hoverinfo='text',
                    showlegend=True
                )
                
        # Prepare edge traces
        edge_traces = {}
        for edge_type in set(nx.get_edge_attributes(self.graph, 'relation').values()):
            edge_list = [
                (u, v) for u, v, attr in self.graph.edges(data=True)
                if attr.get('relation') == edge_type
            ]
            if edge_list:
                edge_x = []
                edge_y = []
                for (u, v) in edge_list:
                    edge_x.extend([pos[u][0], pos[v][0], None])
                    edge_y.extend([pos[u][1], pos[v][1], None])
                    
                edge_traces[edge_type] = go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode='lines',
                    name=edge_type,
                    line=dict(
                        width=1,
                        color=edge_color_map.get(edge_type, '#666666')
                    ),
                    hoverinfo='none',
                    showlegend=True
                )
                
        # Create figure
        fig = go.Figure(
            data=list(edge_traces.values()) + list(node_traces.values()),
            layout=go.Layout(
                title='Interactive FinQA Knowledge Graph',
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=config.width,
                height=config.height
            )
        )
        
        # Save to HTML
        fig.write_html(output_path)