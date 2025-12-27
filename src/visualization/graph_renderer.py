"""
Graph Renderer - Version 4.0

Multi-layer graph visualization using vis.js.
Renders pub-sub system graphs with interactive HTML output.

Features:
- Multi-layer visualization (Application, Topic, Broker, Infrastructure)
- Interactive HTML with vis.js
- Criticality-based coloring
- Multiple layout algorithms
- Layer filtering

Author: Software-as-a-Graph Research Project
Version: 4.0
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Set

from ..simulation import SimulationGraph, ComponentType


# =============================================================================
# Enums and Constants
# =============================================================================

class Layer(Enum):
    """System layers for multi-layer visualization"""
    APPLICATION = "application"
    TOPIC = "topic"
    BROKER = "broker"
    INFRASTRUCTURE = "infrastructure"
    ALL = "all"


class LayoutAlgorithm(Enum):
    """Available layout algorithms"""
    PHYSICS = "physics"        # Force-directed (default)
    HIERARCHICAL = "hierarchical"
    LAYERED = "layered"        # Horizontal layers


class ColorScheme(Enum):
    """Color schemes for nodes"""
    TYPE = "type"              # Color by component type
    CRITICALITY = "criticality"  # Color by criticality level
    LAYER = "layer"            # Color by system layer


# Color palettes
COLORS = {
    # Node types
    "Application": "#3498db",   # Blue
    "Topic": "#2ecc71",         # Green
    "Broker": "#e74c3c",        # Red
    "Node": "#9b59b6",          # Purple
    
    # Layers
    Layer.APPLICATION: "#3498db",
    Layer.TOPIC: "#2ecc71",
    Layer.BROKER: "#e74c3c",
    Layer.INFRASTRUCTURE: "#9b59b6",
    
    # Criticality levels
    "critical": "#e74c3c",
    "high": "#e67e22",
    "medium": "#f39c12",
    "low": "#27ae60",
    "minimal": "#95a5a6",
    
    # Edge types
    "PUBLISHES_TO": "#27ae60",
    "SUBSCRIBES_TO": "#3498db",
    "ROUTES": "#e74c3c",
    "RUNS_ON": "#9b59b6",
    "HOSTS": "#9b59b6",
    "DEPENDS_ON": "#e74c3c",
}

# Node shapes by type
SHAPES = {
    "Application": "dot",
    "Topic": "diamond",
    "Broker": "square",
    "Node": "triangle",
}

# Node sizes by type
SIZES = {
    "Application": 20,
    "Topic": 15,
    "Broker": 25,
    "Node": 22,
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RenderConfig:
    """Configuration for graph rendering"""
    title: str = "Pub-Sub System Graph"
    width: str = "100%"
    height: str = "100vh"
    background: str = "#1a1a2e"
    color_scheme: ColorScheme = ColorScheme.TYPE
    layout: LayoutAlgorithm = LayoutAlgorithm.PHYSICS
    physics_enabled: bool = True
    show_labels: bool = True
    show_legend: bool = True
    node_spacing: int = 100
    layer_spacing: int = 150


@dataclass
class NodeData:
    """Processed node data for rendering"""
    id: str
    label: str
    type: str
    layer: Layer
    color: str
    shape: str
    size: int
    title: str  # Tooltip
    level: Optional[int] = None  # For hierarchical layout
    x: Optional[float] = None
    y: Optional[float] = None
    
    def to_vis(self) -> Dict:
        """Convert to vis.js format"""
        data = {
            "id": self.id,
            "label": self.label,
            "color": {"background": self.color, "border": self.color},
            "shape": self.shape,
            "size": self.size,
            "title": self.title,
            "group": self.layer.value,
        }
        if self.level is not None:
            data["level"] = self.level
        if self.x is not None:
            data["x"] = self.x
            data["y"] = self.y
            data["fixed"] = {"x": True, "y": True}
        return data


@dataclass
class EdgeData:
    """Processed edge data for rendering"""
    id: str
    source: str
    target: str
    type: str
    color: str
    width: int = 2
    dashes: bool = False
    arrows: str = "to"
    title: str = ""
    
    def to_vis(self) -> Dict:
        """Convert to vis.js format"""
        return {
            "id": self.id,
            "from": self.source,
            "to": self.target,
            "color": {"color": self.color, "opacity": 0.7},
            "width": self.width,
            "dashes": self.dashes,
            "arrows": self.arrows,
            "title": self.title,
            "smooth": {"type": "continuous"},
        }


# =============================================================================
# Graph Renderer
# =============================================================================

class GraphRenderer:
    """
    Renders pub-sub system graphs as interactive HTML.
    
    Features:
    - Multi-layer visualization
    - Criticality-based coloring
    - Interactive navigation
    - Multiple layout algorithms
    """

    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()

    def render(
        self,
        graph: SimulationGraph,
        criticality: Optional[Dict[str, Dict]] = None,
        layer: Layer = Layer.ALL,
    ) -> str:
        """
        Render graph to HTML.
        
        Args:
            graph: SimulationGraph to render
            criticality: Optional criticality scores {id: {score, level}}
            layer: Layer to display (ALL for all layers)
        
        Returns:
            HTML string
        """
        nodes, edges = self._process_graph(graph, criticality, layer)
        return self._generate_html(nodes, edges)

    def render_multi_layer(
        self,
        graph: SimulationGraph,
        criticality: Optional[Dict[str, Dict]] = None,
    ) -> str:
        """
        Render graph with horizontal layer layout.
        
        Args:
            graph: SimulationGraph to render
            criticality: Optional criticality scores
        
        Returns:
            HTML string with layered visualization
        """
        nodes, edges = self._process_graph(graph, criticality, Layer.ALL)
        
        # Assign layer positions
        nodes = self._assign_layer_positions(nodes)
        
        return self._generate_multi_layer_html(nodes, edges)

    def _process_graph(
        self,
        graph: SimulationGraph,
        criticality: Optional[Dict[str, Dict]],
        layer: Layer,
    ) -> Tuple[List[NodeData], List[EdgeData]]:
        """Process graph into render data"""
        nodes = []
        edges = []
        
        # Process components
        for comp_id, comp in graph.components.items():
            node_layer = self._get_layer(comp.type)
            
            # Filter by layer
            if layer != Layer.ALL and node_layer != layer:
                continue
            
            # Determine color
            if self.config.color_scheme == ColorScheme.CRITICALITY and criticality:
                crit = criticality.get(comp_id, {})
                level = crit.get("level", "minimal")
                color = COLORS.get(level, COLORS["minimal"])
            elif self.config.color_scheme == ColorScheme.LAYER:
                color = COLORS.get(node_layer, "#95a5a6")
            else:
                color = COLORS.get(comp.type.value, "#95a5a6")
            
            # Build tooltip
            tooltip = f"<b>{comp_id}</b><br>Type: {comp.type.value}"
            if criticality and comp_id in criticality:
                crit = criticality[comp_id]
                tooltip += f"<br>Score: {crit.get('score', 0):.4f}"
                tooltip += f"<br>Level: {crit.get('level', 'unknown')}"
            
            nodes.append(NodeData(
                id=comp_id,
                label=comp_id if self.config.show_labels else "",
                type=comp.type.value,
                layer=node_layer,
                color=color,
                shape=SHAPES.get(comp.type.value, "dot"),
                size=SIZES.get(comp.type.value, 20),
                title=tooltip,
            ))
        
        # Track valid node IDs for edge filtering
        node_ids = {n.id for n in nodes}
        
        # Process connections
        for i, conn in enumerate(graph.connections):
            if conn.source not in node_ids or conn.target not in node_ids:
                continue
            
            edges.append(EdgeData(
                id=f"e{i}",
                source=conn.source,
                target=conn.target,
                type=conn.type.value,
                color=COLORS.get(conn.type.value, "#7f8c8d"),
                title=conn.type.value,
            ))
        
        return nodes, edges

    def _get_layer(self, comp_type: ComponentType) -> Layer:
        """Map component type to layer"""
        mapping = {
            ComponentType.APPLICATION: Layer.APPLICATION,
            ComponentType.TOPIC: Layer.TOPIC,
            ComponentType.BROKER: Layer.BROKER,
            ComponentType.NODE: Layer.INFRASTRUCTURE,
        }
        return mapping.get(comp_type, Layer.APPLICATION)

    def _assign_layer_positions(self, nodes: List[NodeData]) -> List[NodeData]:
        """Assign x/y positions for layered layout"""
        # Group nodes by layer
        layer_nodes: Dict[Layer, List[NodeData]] = {
            Layer.APPLICATION: [],
            Layer.TOPIC: [],
            Layer.BROKER: [],
            Layer.INFRASTRUCTURE: [],
        }
        
        for node in nodes:
            if node.layer in layer_nodes:
                layer_nodes[node.layer].append(node)
        
        # Layer Y positions (top to bottom)
        layer_y = {
            Layer.APPLICATION: 0,
            Layer.TOPIC: 200,
            Layer.BROKER: 400,
            Layer.INFRASTRUCTURE: 600,
        }
        
        # Assign positions
        for layer, layer_node_list in layer_nodes.items():
            n = len(layer_node_list)
            if n == 0:
                continue
            
            # Spread horizontally
            total_width = (n - 1) * self.config.node_spacing
            start_x = -total_width / 2
            
            for i, node in enumerate(layer_node_list):
                node.x = start_x + i * self.config.node_spacing
                node.y = layer_y[layer]
                node.level = list(layer_y.keys()).index(layer)
        
        return nodes

    def _generate_html(
        self,
        nodes: List[NodeData],
        edges: List[EdgeData],
    ) -> str:
        """Generate interactive HTML with vis.js"""
        nodes_json = json.dumps([n.to_vis() for n in nodes])
        edges_json = json.dumps([e.to_vis() for e in edges])
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Count by type
        type_counts = {}
        for n in nodes:
            type_counts[n.type] = type_counts.get(n.type, 0) + 1
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: {self.config.background};
            color: #ecf0f1;
        }}
        
        #header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        #header h1 {{
            font-size: 1.4em;
            font-weight: 600;
        }}
        
        .stats {{
            display: flex;
            gap: 20px;
            font-size: 0.9em;
        }}
        
        .stat {{
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 1.3em;
            font-weight: bold;
        }}
        
        #controls {{
            background: rgba(0,0,0,0.3);
            padding: 10px 20px;
            display: flex;
            gap: 15px;
            align-items: center;
        }}
        
        button {{
            background: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
        }}
        
        button:hover {{
            background: #2980b9;
        }}
        
        #network {{
            width: 100%;
            height: calc(100vh - 120px);
        }}
        
        #legend {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0,0,0,0.85);
            padding: 15px;
            border-radius: 8px;
            font-size: 0.85em;
            max-width: 200px;
        }}
        
        #legend h4 {{
            margin-bottom: 10px;
            color: #fff;
            border-bottom: 1px solid #555;
            padding-bottom: 5px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 6px 0;
        }}
        
        .legend-color {{
            width: 14px;
            height: 14px;
            border-radius: 3px;
            margin-right: 8px;
            flex-shrink: 0;
        }}
        
        #info {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(0,0,0,0.7);
            padding: 10px 15px;
            border-radius: 6px;
            font-size: 0.8em;
            color: #aaa;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>{self.config.title}</h1>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{len(nodes)}</div>
                <div>Nodes</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(edges)}</div>
                <div>Edges</div>
            </div>
        </div>
    </div>
    
    <div id="controls">
        <button onclick="togglePhysics()">Toggle Physics</button>
        <button onclick="toggleLabels()">Toggle Labels</button>
        <button onclick="fitNetwork()">Fit View</button>
        <button onclick="exportPNG()">Export PNG</button>
    </div>
    
    <div id="network"></div>
    
    <div id="legend">
        <h4>Node Types</h4>
        <div class="legend-item">
            <div class="legend-color" style="background: {COLORS['Application']}"></div>
            <span>Application ({type_counts.get('Application', 0)})</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: {COLORS['Topic']}"></div>
            <span>Topic ({type_counts.get('Topic', 0)})</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: {COLORS['Broker']}"></div>
            <span>Broker ({type_counts.get('Broker', 0)})</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: {COLORS['Node']}"></div>
            <span>Infrastructure ({type_counts.get('Node', 0)})</span>
        </div>
    </div>
    
    <div id="info">Generated: {timestamp}</div>
    
    <script>
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});
        
        var container = document.getElementById('network');
        var data = {{ nodes: nodes, edges: edges }};
        
        var options = {{
            nodes: {{
                font: {{ size: 12, color: '#ecf0f1' }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                shadow: true
            }},
            physics: {{
                enabled: {str(self.config.physics_enabled).lower()},
                barnesHut: {{
                    gravitationalConstant: -4000,
                    springLength: {self.config.node_spacing},
                    springConstant: 0.04,
                    damping: 0.09
                }},
                stabilization: {{ iterations: 150, fit: true }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                navigationButtons: true,
                keyboard: true
            }}
        }};
        
        var network = new vis.Network(container, data, options);
        var physicsEnabled = {str(self.config.physics_enabled).lower()};
        var labelsVisible = true;
        
        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
        }}
        
        function toggleLabels() {{
            labelsVisible = !labelsVisible;
            var newNodes = nodes.get().map(function(n) {{
                n.font = {{ size: labelsVisible ? 12 : 0, color: '#ecf0f1' }};
                return n;
            }});
            nodes.update(newNodes);
        }}
        
        function fitNetwork() {{
            network.fit();
        }}
        
        function exportPNG() {{
            var canvas = container.getElementsByTagName('canvas')[0];
            var link = document.createElement('a');
            link.download = 'graph.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
        }}
        
        network.once('stabilized', function() {{
            network.fit();
        }});
    </script>
</body>
</html>"""

    def _generate_multi_layer_html(
        self,
        nodes: List[NodeData],
        edges: List[EdgeData],
    ) -> str:
        """Generate HTML with horizontal layer layout"""
        nodes_json = json.dumps([n.to_vis() for n in nodes])
        edges_json = json.dumps([e.to_vis() for e in edges])
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Count by layer
        layer_counts = {}
        for n in nodes:
            layer_counts[n.layer.value] = layer_counts.get(n.layer.value, 0) + 1
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title} - Multi-Layer View</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: {self.config.background};
            color: #ecf0f1;
        }}
        
        #header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px 25px;
            text-align: center;
        }}
        
        #header h1 {{
            font-size: 1.4em;
            margin-bottom: 5px;
        }}
        
        #header p {{
            opacity: 0.9;
            font-size: 0.9em;
        }}
        
        #network {{
            width: 100%;
            height: calc(100vh - 80px);
        }}
        
        #layer-labels {{
            position: fixed;
            left: 20px;
            top: 50%;
            transform: translateY(-50%);
            display: flex;
            flex-direction: column;
            gap: 120px;
        }}
        
        .layer-label {{
            writing-mode: vertical-rl;
            text-orientation: mixed;
            transform: rotate(180deg);
            padding: 10px 5px;
            border-radius: 5px;
            font-weight: 600;
            font-size: 0.8em;
            text-align: center;
        }}
        
        .layer-app {{ background: {COLORS[Layer.APPLICATION]}; }}
        .layer-topic {{ background: {COLORS[Layer.TOPIC]}; }}
        .layer-broker {{ background: {COLORS[Layer.BROKER]}; }}
        .layer-infra {{ background: {COLORS[Layer.INFRASTRUCTURE]}; }}
        
        #legend {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: rgba(0,0,0,0.85);
            padding: 15px;
            border-radius: 8px;
            font-size: 0.85em;
        }}
        
        #legend h4 {{
            margin-bottom: 10px;
            border-bottom: 1px solid #555;
            padding-bottom: 5px;
        }}
        
        .legend-section {{
            margin-bottom: 12px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
            margin-right: 8px;
        }}
        
        #info {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(0,0,0,0.7);
            padding: 10px 15px;
            border-radius: 6px;
            font-size: 0.8em;
            color: #aaa;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>{self.config.title}</h1>
        <p>Multi-Layer Architecture View • {len(nodes)} nodes • {len(edges)} edges</p>
    </div>
    
    <div id="network"></div>
    
    <div id="layer-labels">
        <div class="layer-label layer-app">Application ({layer_counts.get('application', 0)})</div>
        <div class="layer-label layer-topic">Topic ({layer_counts.get('topic', 0)})</div>
        <div class="layer-label layer-broker">Broker ({layer_counts.get('broker', 0)})</div>
        <div class="layer-label layer-infra">Infrastructure ({layer_counts.get('infrastructure', 0)})</div>
    </div>
    
    <div id="legend">
        <div class="legend-section">
            <h4>Edge Types</h4>
            <div class="legend-item">
                <div class="legend-color" style="background: {COLORS['PUBLISHES_TO']}"></div>
                <span>Publishes To</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: {COLORS['SUBSCRIBES_TO']}"></div>
                <span>Subscribes To</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: {COLORS['ROUTES']}"></div>
                <span>Routes / Depends</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: {COLORS['RUNS_ON']}"></div>
                <span>Runs On / Hosts</span>
            </div>
        </div>
    </div>
    
    <div id="info">Generated: {timestamp}</div>
    
    <script>
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});
        
        var container = document.getElementById('network');
        var data = {{ nodes: nodes, edges: edges }};
        
        var options = {{
            nodes: {{
                font: {{ size: 11, color: '#ecf0f1' }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                smooth: {{ type: 'cubicBezier', roundness: 0.4 }},
                shadow: true
            }},
            physics: {{
                enabled: false
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                navigationButtons: true,
                keyboard: true,
                zoomView: true,
                dragView: true
            }}
        }};
        
        var network = new vis.Network(container, data, options);
        network.fit();
    </script>
</body>
</html>"""
