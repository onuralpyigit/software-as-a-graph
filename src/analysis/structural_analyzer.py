"""
Structural Analyzer

Computes raw graph topological metrics using NetworkX.
Supports multi-layer analysis: Application, Infrastructure, Complete System.

Metrics computed:
- Centrality: PageRank, Reverse PageRank, Betweenness, Degree
- Resilience: Clustering Coefficient, Articulation Points, Bridges
- Flow: In-Degree, Out-Degree

Author: Software-as-a-Graph Research Project
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, Any, Set, List, Optional
from enum import Enum

import networkx as nx


class AnalysisLayer(Enum):
    """Enumeration of analysis layers."""
    APPLICATION = "application"
    INFRASTRUCTURE = "infrastructure"
    COMPLETE = "complete"
    APP_BROKER = "app_broker"
    NODE_BROKER = "node_broker"


# Layer definitions mapping
LAYER_DEFINITIONS = {
    AnalysisLayer.APPLICATION: {
        "name": "Application Layer",
        "component_types": {"Application"},
        "dependency_types": {"app_to_app"},
        "description": "Service-level reliability analysis",
    },
    AnalysisLayer.INFRASTRUCTURE: {
        "name": "Infrastructure Layer",
        "component_types": {"Node"},
        "dependency_types": {"node_to_node"},
        "description": "Network topology resilience analysis",
    },
    AnalysisLayer.APP_BROKER: {
        "name": "Application-Broker Layer",
        "component_types": {"Application", "Broker"},
        "dependency_types": {"app_to_broker"},
        "description": "Middleware dependency analysis",
    },
    AnalysisLayer.NODE_BROKER: {
        "name": "Node-Broker Layer",
        "component_types": {"Node", "Broker"},
        "dependency_types": {"node_to_broker"},
        "description": "Infrastructure-middleware coupling analysis",
    },
    AnalysisLayer.COMPLETE: {
        "name": "Complete System",
        "component_types": {"Application", "Broker", "Node", "Topic"},
        "dependency_types": {"app_to_app", "app_to_broker", "node_to_node", "node_to_broker"},
        "description": "System-wide analysis across all layers",
    },
}


@dataclass
class StructuralMetrics:
    """Raw topological metrics for a single component."""
    
    id: str
    type: str
    
    # Centrality (Importance)
    pagerank: float = 0.0           # Global importance in dependency graph
    reverse_pagerank: float = 0.0   # Failure propagation influence
    betweenness: float = 0.0        # Bridge/flow control centrality
    closeness: float = 0.0          # Average distance to all other nodes
    eigenvector: float = 0.0        # Influence based on neighbor importance
    
    # Degree Metrics
    degree: float = 0.0             # Total degree centrality
    in_degree: float = 0.0          # Number of dependents (who depends on this)
    out_degree: float = 0.0         # Number of dependencies (this depends on)
    in_degree_raw: int = 0          # Raw in-degree count
    out_degree_raw: int = 0         # Raw out-degree count
    
    # Resilience Metrics
    clustering_coefficient: float = 0.0  # Local redundancy measure
    
    # Criticality Flags
    is_articulation_point: bool = False  # Removal disconnects graph
    is_isolated: bool = False            # No connections
    bridge_count: int = 0                # Number of bridges incident to node
    bridge_ratio: float = 0.0            # Fraction of edges that are bridges
    
    # Weights
    weight: float = 1.0             # Intrinsic component weight
    dependency_weight_in: float = 0.0   # Sum of incoming dependency weights
    dependency_weight_out: float = 0.0  # Sum of outgoing dependency weights
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EdgeMetrics:
    """Raw topological metrics for a single edge."""
    
    source: str
    target: str
    source_type: str
    target_type: str
    dependency_type: str
    
    # Metrics
    betweenness: float = 0.0        # Edge betweenness centrality
    is_bridge: bool = False         # Removal disconnects graph
    weight: float = 1.0             # Edge weight
    
    @property
    def key(self) -> Tuple[str, str]:
        return (self.source, self.target)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphSummary:
    """Summary statistics for the analyzed graph."""
    
    layer: str
    nodes: int = 0
    edges: int = 0
    density: float = 0.0
    avg_degree: float = 0.0
    avg_clustering: float = 0.0
    is_connected: bool = False
    num_components: int = 0
    num_articulation_points: int = 0
    num_bridges: int = 0
    diameter: Optional[int] = None
    avg_path_length: Optional[float] = None
    
    # Node type breakdown
    node_types: Dict[str, int] = field(default_factory=dict)
    edge_types: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StructuralAnalysisResult:
    """Container for all raw structural metrics."""
    
    layer: AnalysisLayer
    components: Dict[str, StructuralMetrics]
    edges: Dict[Tuple[str, str], EdgeMetrics]
    graph_summary: GraphSummary
    
    def get_components_by_type(self, comp_type: str) -> List[StructuralMetrics]:
        """Get all components of a specific type."""
        return [c for c in self.components.values() if c.type == comp_type]
    
    def get_articulation_points(self) -> List[StructuralMetrics]:
        """Get all articulation points."""
        return [c for c in self.components.values() if c.is_articulation_point]
    
    def get_bridges(self) -> List[EdgeMetrics]:
        """Get all bridge edges."""
        return [e for e in self.edges.values() if e.is_bridge]


class StructuralAnalyzer:
    """
    Analyzes graph structure to compute topological metrics.
    
    Supports multi-layer analysis by filtering components and dependencies
    based on the specified analysis layer.
    """
    
    def __init__(self, damping_factor: float = 0.85):
        """
        Initialize the structural analyzer.
        
        Args:
            damping_factor: PageRank damping factor (default: 0.85)
        """
        self.damping_factor = damping_factor
        self.logger = logging.getLogger(__name__)
    
    def analyze(
        self, 
        graph_data: Any,
        layer: AnalysisLayer = AnalysisLayer.COMPLETE
    ) -> StructuralAnalysisResult:
        """
        Compute structural metrics for the provided graph data.
        
        Args:
            graph_data: GraphData object from graph_exporter
            layer: Analysis layer to use for filtering
            
        Returns:
            StructuralAnalysisResult containing all computed metrics
        """
        # Build filtered NetworkX graph
        G = self._build_graph(graph_data, layer)
        
        if len(G) == 0:
            return StructuralAnalysisResult(
                layer=layer,
                components={},
                edges={},
                graph_summary=GraphSummary(layer=layer.value)
            )
        
        # Compute all metrics
        component_metrics = self._compute_component_metrics(G, graph_data, layer)
        edge_metrics = self._compute_edge_metrics(G, graph_data, layer)
        graph_summary = self._compute_graph_summary(G, layer, component_metrics, edge_metrics)
        
        return StructuralAnalysisResult(
            layer=layer,
            components=component_metrics,
            edges=edge_metrics,
            graph_summary=graph_summary
        )
    
    def _build_graph(self, graph_data: Any, layer: AnalysisLayer) -> nx.DiGraph:
        """
        Build a NetworkX DiGraph from graph data, filtered by layer.
        """
        G = nx.DiGraph()
        layer_def = LAYER_DEFINITIONS[layer]
        component_types = layer_def["component_types"]
        dependency_types = layer_def["dependency_types"]
        
        # Add nodes (filtered by component type)
        for comp in graph_data.components:
            if layer == AnalysisLayer.COMPLETE or comp.component_type in component_types:
                G.add_node(
                    comp.id,
                    type=comp.component_type,
                    weight=comp.weight,
                    **comp.properties
                )
        
        # Add edges (filtered by dependency type)
        for edge in graph_data.edges:
            if layer == AnalysisLayer.COMPLETE or edge.dependency_type in dependency_types:
                # Only add edge if both endpoints are in the graph
                if G.has_node(edge.source_id) and G.has_node(edge.target_id):
                    G.add_edge(
                        edge.source_id,
                        edge.target_id,
                        dependency_type=edge.dependency_type,
                        weight=edge.weight,
                        source_type=edge.source_type,
                        target_type=edge.target_type,
                        **edge.properties
                    )
        
        return G
    
    def _compute_component_metrics(
        self, 
        G: nx.DiGraph, 
        graph_data: Any,
        layer: AnalysisLayer
    ) -> Dict[str, StructuralMetrics]:
        """Compute all metrics for components (nodes)."""
        
        if len(G) == 0:
            return {}
        
        # 1. PageRank metrics
        try:
            pagerank = nx.pagerank(G, alpha=self.damping_factor, weight="weight")
        except:
            pagerank = {n: 1.0 / len(G) for n in G}
        
        try:
            reverse_pagerank = nx.pagerank(G.reverse(), alpha=self.damping_factor, weight="weight")
        except:
            reverse_pagerank = {n: 1.0 / len(G) for n in G}
        
        # 2. Centrality metrics
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
        degree_cent = nx.degree_centrality(G)
        in_degree_cent = nx.in_degree_centrality(G)
        out_degree_cent = nx.out_degree_centrality(G)
        
        try:
            closeness = nx.closeness_centrality(G, wf_improved=True)
        except:
            closeness = {n: 0.0 for n in G}
        
        try:
            eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
        except:
            eigenvector = {n: 0.0 for n in G}
        
        # 3. Clustering (use undirected version)
        G_undirected = G.to_undirected()
        clustering = nx.clustering(G_undirected)
        
        # 4. Articulation points and bridges
        articulation_points = set(nx.articulation_points(G_undirected)) if nx.is_connected(G_undirected) else set()
        bridges = set(nx.bridges(G_undirected)) if nx.is_connected(G_undirected) else set()
        
        # Count bridges per node
        bridge_count = {n: 0 for n in G}
        for u, v in bridges:
            bridge_count[u] = bridge_count.get(u, 0) + 1
            bridge_count[v] = bridge_count.get(v, 0) + 1
        
        # Build lookup for weights
        weight_lookup = {comp.id: comp.weight for comp in graph_data.components}
        
        # 5. Assemble metrics for each node
        component_metrics = {}
        
        for node in G.nodes():
            node_data = G.nodes[node]
            node_type = node_data.get("type", "Unknown")
            
            # Compute degree and bridge ratio
            total_degree = G.degree(node)
            br = bridge_count.get(node, 0) / total_degree if total_degree > 0 else 0.0
            
            # Dependency weights
            dep_weight_in = sum(G.edges[pred, node].get("weight", 1.0) for pred in G.predecessors(node))
            dep_weight_out = sum(G.edges[node, succ].get("weight", 1.0) for succ in G.successors(node))
            
            metrics = StructuralMetrics(
                id=node,
                type=node_type,
                pagerank=pagerank.get(node, 0.0),
                reverse_pagerank=reverse_pagerank.get(node, 0.0),
                betweenness=betweenness.get(node, 0.0),
                closeness=closeness.get(node, 0.0),
                eigenvector=eigenvector.get(node, 0.0),
                degree=degree_cent.get(node, 0.0),
                in_degree=in_degree_cent.get(node, 0.0),
                out_degree=out_degree_cent.get(node, 0.0),
                in_degree_raw=G.in_degree(node),
                out_degree_raw=G.out_degree(node),
                clustering_coefficient=clustering.get(node, 0.0),
                is_articulation_point=node in articulation_points,
                is_isolated=total_degree == 0,
                bridge_count=bridge_count.get(node, 0),
                bridge_ratio=br,
                weight=weight_lookup.get(node, node_data.get("weight", 1.0)),
                dependency_weight_in=dep_weight_in,
                dependency_weight_out=dep_weight_out,
            )
            component_metrics[node] = metrics
        
        return component_metrics
    
    def _compute_edge_metrics(
        self, 
        G: nx.DiGraph, 
        graph_data: Any,
        layer: AnalysisLayer
    ) -> Dict[Tuple[str, str], EdgeMetrics]:
        """Compute metrics for edges."""
        
        if len(G.edges()) == 0:
            return {}
        
        # Edge betweenness
        edge_betweenness = nx.edge_betweenness_centrality(G, weight="weight", normalized=True)
        
        # Bridges (use undirected)
        G_undirected = G.to_undirected()
        bridges = set(nx.bridges(G_undirected)) if nx.is_connected(G_undirected) else set()
        
        edge_metrics = {}
        
        for u, v, data in G.edges(data=True):
            # Check if this edge is a bridge (either direction)
            is_bridge = (u, v) in bridges or (v, u) in bridges
            
            metrics = EdgeMetrics(
                source=u,
                target=v,
                source_type=data.get("source_type", G.nodes[u].get("type", "Unknown")),
                target_type=data.get("target_type", G.nodes[v].get("type", "Unknown")),
                dependency_type=data.get("dependency_type", "unknown"),
                betweenness=edge_betweenness.get((u, v), 0.0),
                is_bridge=is_bridge,
                weight=data.get("weight", 1.0),
            )
            edge_metrics[(u, v)] = metrics
        
        return edge_metrics
    
    def _compute_graph_summary(
        self, 
        G: nx.DiGraph, 
        layer: AnalysisLayer,
        component_metrics: Dict[str, StructuralMetrics],
        edge_metrics: Dict[Tuple[str, str], EdgeMetrics]
    ) -> GraphSummary:
        """Compute summary statistics for the graph."""
        
        n_nodes = len(G)
        n_edges = len(G.edges())
        
        # Density
        density = nx.density(G) if n_nodes > 0 else 0.0
        
        # Average degree
        avg_degree = sum(G.degree(n) for n in G) / n_nodes if n_nodes > 0 else 0.0
        
        # Average clustering
        G_undirected = G.to_undirected()
        avg_clustering = nx.average_clustering(G_undirected) if n_nodes > 0 else 0.0
        
        # Connectivity
        is_connected = nx.is_weakly_connected(G) if n_nodes > 0 else False
        num_components = nx.number_weakly_connected_components(G) if n_nodes > 0 else 0
        
        # Articulation points and bridges
        num_ap = sum(1 for m in component_metrics.values() if m.is_articulation_point)
        num_bridges = sum(1 for e in edge_metrics.values() if e.is_bridge)
        
        # Path metrics (only for connected graphs)
        diameter = None
        avg_path_length = None
        if is_connected and n_nodes > 1:
            try:
                diameter = nx.diameter(G_undirected)
                avg_path_length = nx.average_shortest_path_length(G_undirected)
            except:
                pass
        
        # Node type breakdown
        node_types: Dict[str, int] = {}
        for m in component_metrics.values():
            node_types[m.type] = node_types.get(m.type, 0) + 1
        
        # Edge type breakdown
        edge_types: Dict[str, int] = {}
        for e in edge_metrics.values():
            edge_types[e.dependency_type] = edge_types.get(e.dependency_type, 0) + 1
        
        return GraphSummary(
            layer=layer.value,
            nodes=n_nodes,
            edges=n_edges,
            density=density,
            avg_degree=avg_degree,
            avg_clustering=avg_clustering,
            is_connected=is_connected,
            num_components=num_components,
            num_articulation_points=num_ap,
            num_bridges=num_bridges,
            diameter=diameter,
            avg_path_length=avg_path_length,
            node_types=node_types,
            edge_types=edge_types,
        )


def extract_layer_subgraph(
    graph_data: Any, 
    layer: AnalysisLayer
) -> Tuple[List[Any], List[Any]]:
    """
    Extract components and edges for a specific layer.
    
    Useful for creating layer-specific datasets without full analysis.
    """
    layer_def = LAYER_DEFINITIONS[layer]
    component_types = layer_def["component_types"]
    dependency_types = layer_def["dependency_types"]
    
    filtered_components = [
        c for c in graph_data.components 
        if c.component_type in component_types
    ]
    
    component_ids = {c.id for c in filtered_components}
    
    filtered_edges = [
        e for e in graph_data.edges
        if e.dependency_type in dependency_types
        and e.source_id in component_ids
        and e.target_id in component_ids
    ]
    
    return filtered_components, filtered_edges