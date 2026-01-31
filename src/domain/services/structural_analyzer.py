"""
Structural Analyzer

Computes raw topological metrics from graph data using NetworkX.
Supports multi-layer analysis by filtering based on DEPENDS_ON relationship types.

Metrics computed:
    - Centrality: PageRank, Reverse PageRank, Betweenness, Closeness, Eigenvector
    - Degree: In-degree, Out-degree, Total degree
    - Resilience: Clustering coefficient, Articulation points, Bridges

Usage:
    analyzer = StructuralAnalyzer()
    result = analyzer.analyze(graph_data, layer=AnalysisLayer.APP)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import networkx as nx

from src.models.analysis.layers import AnalysisLayer, LAYER_DEFINITIONS, get_layer_definition
from src.domain.models.metrics import StructuralMetrics, EdgeMetrics, GraphSummary


@dataclass
class StructuralAnalysisResult:
    """Container for all raw structural analysis results."""
    layer: AnalysisLayer
    components: Dict[str, StructuralMetrics]
    edges: Dict[Tuple[str, str], EdgeMetrics]
    graph_summary: GraphSummary
    
    def get_components_by_type(self, comp_type: str) -> List[StructuralMetrics]:
        """Get all components of a specific type."""
        return [c for c in self.components.values() if c.type == comp_type]
    
    def get_articulation_points(self) -> List[StructuralMetrics]:
        """Get all articulation points (single points of failure)."""
        return [c for c in self.components.values() if c.is_articulation_point]
    
    def get_bridges(self) -> List[EdgeMetrics]:
        """Get all bridge edges."""
        return [e for e in self.edges.values() if e.is_bridge]
    
    def get_top_by_metric(
        self, 
        metric: str, 
        n: int = 10,
        reverse: bool = True
    ) -> List[StructuralMetrics]:
        """Get top N components by a specific metric."""
        return sorted(
            self.components.values(),
            key=lambda c: getattr(c, metric, 0),
            reverse=reverse
        )[:n]


class StructuralAnalyzer:
    """
    Analyzes graph structure to compute topological metrics.
    
    Supports multi-layer analysis by filtering components and edges
    based on the specified analysis layer's DEPENDS_ON relationship types.
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
        layer: AnalysisLayer = AnalysisLayer.SYSTEM
    ) -> StructuralAnalysisResult:
        """
        Compute structural metrics for the provided graph data.
        
        Args:
            graph_data: GraphData object containing components and edges
            layer: Analysis layer for filtering (determines which DEPENDS_ON types)
            
        Returns:
            StructuralAnalysisResult with all computed metrics
        """
        layer_def = get_layer_definition(layer)
        
        # Build filtered NetworkX graph (includes all component_types for edges)
        G = self._build_graph(graph_data, layer)
        
        if len(G) == 0:
            self.logger.warning(f"Empty graph for layer {layer.value}")
            return StructuralAnalysisResult(
                layer=layer,
                components={},
                edges={},
                graph_summary=GraphSummary(layer=layer.value)
            )
        
        self.logger.info(f"Analyzing {layer.value} layer: {len(G)} nodes, {len(G.edges())} edges")
        
        # Compute all metrics on full graph
        all_component_metrics = self._compute_component_metrics(G, graph_data, layer)
        edge_metrics = self._compute_edge_metrics(G, graph_data, layer)
        
        # Filter to only include components matching analyze_types
        types_to_analyze = layer_def.types_to_analyze
        component_metrics = {
            k: v for k, v in all_component_metrics.items()
            if v.type in types_to_analyze
        }
        
        self.logger.info(
            f"Filtered to {len(component_metrics)} components of types: {types_to_analyze}"
        )
        
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
        
        Only includes components and edges matching the layer's
        component_types and dependency_types.
        """
        G = nx.DiGraph()
        layer_def = get_layer_definition(layer)
        
        # Filter components by type
        component_map: Dict[str, Any] = {}
        for comp in graph_data.components:
            if comp.component_type in layer_def.component_types:
                component_map[comp.id] = comp
                G.add_node(
                    comp.id,
                    type=comp.component_type,
                    weight=getattr(comp, 'weight', 1.0),
                    **comp.properties
                )
        
        # Filter edges by dependency type
        for edge in graph_data.edges:
            if edge.dependency_type not in layer_def.dependency_types:
                continue
            if edge.source_id not in component_map:
                continue
            if edge.target_id not in component_map:
                continue
            
            G.add_edge(
                edge.source_id,
                edge.target_id,
                dependency_type=edge.dependency_type,
                weight=getattr(edge, 'weight', 1.0),
            )
        
        return G
    
    def _compute_component_metrics(
        self,
        G: nx.DiGraph,
        graph_data: Any,
        layer: AnalysisLayer
    ) -> Dict[str, StructuralMetrics]:
        """Compute all metrics for each node in the graph."""
        
        n_nodes = len(G)
        if n_nodes == 0:
            return {}
        
        # === Centrality Metrics ===
        
        # Forward PageRank: Measures importance as a dependency target.
        # High PageRank = many components depend on this (directly or indirectly).
        try:
            pagerank = nx.pagerank(G, alpha=self.damping_factor, weight='weight')
        except nx.NetworkXError:
            pagerank = {n: 1.0 / n_nodes for n in G}
        
        # Reverse PageRank: Measures importance as a dependency source.
        # High Reverse PageRank = this component depends on many others.
        # Useful for identifying components vulnerable to upstream failures.
        G_rev = G.reverse()
        try:
            reverse_pagerank = nx.pagerank(G_rev, alpha=self.damping_factor, weight='weight')
        except nx.NetworkXError:
            reverse_pagerank = {n: 1.0 / n_nodes for n in G}
        
        # Betweenness centrality (bottleneck/bridge importance)
        try:
            betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
        except nx.NetworkXError:
            betweenness = {n: 0.0 for n in G}
        
        # Closeness centrality (average distance to others)
        # Using edge weights as distance (higher weight = shorter path = more important)
        try:
            closeness = nx.closeness_centrality(G, distance='weight')
        except nx.NetworkXError:
            closeness = {n: 0.0 for n in G}
        
        # Eigenvector centrality (influence via neighbors)
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
        except (nx.NetworkXError, nx.PowerIterationFailedConvergence):
            eigenvector = {n: 0.0 for n in G}
        
        # === Degree Metrics ===
        
        in_degree_raw = dict(G.in_degree())
        out_degree_raw = dict(G.out_degree())
        
        max_degree = max(max(in_degree_raw.values(), default=1), 
                        max(out_degree_raw.values(), default=1))
        
        # Normalized degree centrality
        try:
            degree_centrality = nx.degree_centrality(G)
            in_degree_centrality = nx.in_degree_centrality(G)
            out_degree_centrality = nx.out_degree_centrality(G)
        except nx.NetworkXError:
            degree_centrality = {n: 0.0 for n in G}
            in_degree_centrality = {n: 0.0 for n in G}
            out_degree_centrality = {n: 0.0 for n in G}
        
        # === Resilience Metrics ===
        
        # Convert to undirected for some metrics
        G_undirected = G.to_undirected()
        
        # Clustering coefficient (local redundancy)
        try:
            clustering = nx.clustering(G_undirected)
        except nx.NetworkXError:
            clustering = {n: 0.0 for n in G}
        
        # Articulation points (single points of failure)
        try:
            articulation_points = set(nx.articulation_points(G_undirected))
        except nx.NetworkXError:
            articulation_points = set()
        
        # Bridges
        try:
            bridges = set(nx.bridges(G_undirected))
        except nx.NetworkXError:
            bridges = set()
        
        # Count bridges per node
        bridge_counts: Dict[str, int] = {n: 0 for n in G}
        for u, v in bridges:
            bridge_counts[u] = bridge_counts.get(u, 0) + 1
            bridge_counts[v] = bridge_counts.get(v, 0) + 1
        
        # === Build Component Metrics ===
        
        component_map = {c.id: c for c in graph_data.components}
        metrics: Dict[str, StructuralMetrics] = {}
        
        for node in G.nodes():
            comp = component_map.get(node)
            comp_name = comp.properties.get('name', node) if comp else node
            comp_type = G.nodes[node].get('type', 'Unknown')
            comp_weight = G.nodes[node].get('weight', 1.0)
            
            # Calculate dependency weights
            dep_weight_in = sum(G.edges[pred, node].get('weight', 1.0) 
                               for pred in G.predecessors(node))
            dep_weight_out = sum(G.edges[node, succ].get('weight', 1.0) 
                                for succ in G.successors(node))
            
            # Bridge ratio
            node_degree = in_degree_raw[node] + out_degree_raw[node]
            bridge_ratio = bridge_counts[node] / node_degree if node_degree > 0 else 0.0
            
            metrics[node] = StructuralMetrics(
                id=node,
                name=comp_name,
                type=comp_type,
                pagerank=pagerank.get(node, 0.0),
                reverse_pagerank=reverse_pagerank.get(node, 0.0),
                betweenness=betweenness.get(node, 0.0),
                closeness=closeness.get(node, 0.0),
                eigenvector=eigenvector.get(node, 0.0),
                degree=degree_centrality.get(node, 0.0),
                in_degree=in_degree_centrality.get(node, 0.0),
                out_degree=out_degree_centrality.get(node, 0.0),
                in_degree_raw=in_degree_raw.get(node, 0),
                out_degree_raw=out_degree_raw.get(node, 0),
                clustering_coefficient=clustering.get(node, 0.0),
                is_articulation_point=node in articulation_points,
                is_isolated=node_degree == 0,
                bridge_count=bridge_counts.get(node, 0),
                bridge_ratio=bridge_ratio,
                weight=comp_weight,
                dependency_weight_in=dep_weight_in,
                dependency_weight_out=dep_weight_out,
            )
        
        return metrics
    
    def _compute_edge_metrics(
        self,
        G: nx.DiGraph,
        graph_data: Any,
        layer: AnalysisLayer
    ) -> Dict[Tuple[str, str], EdgeMetrics]:
        """Compute metrics for each edge in the graph."""
        
        if len(G.edges()) == 0:
            return {}
        
        # Edge betweenness centrality
        try:
            edge_betweenness = nx.edge_betweenness_centrality(G, normalized=True)
        except:
            edge_betweenness = {e: 0.0 for e in G.edges()}
        
        # Bridges
        G_undirected = G.to_undirected()
        try:
            bridges = set(nx.bridges(G_undirected))
        except:
            bridges = set()
        
        # Build edge metrics
        edge_map = {(e.source_id, e.target_id): e for e in graph_data.edges}
        metrics: Dict[Tuple[str, str], EdgeMetrics] = {}
        
        for u, v, data in G.edges(data=True):
            edge_data = edge_map.get((u, v))
            is_bridge = (u, v) in bridges or (v, u) in bridges
            
            metrics[(u, v)] = EdgeMetrics(
                source=u,
                target=v,
                source_type=G.nodes[u].get('type', 'Unknown'),
                target_type=G.nodes[v].get('type', 'Unknown'),
                dependency_type=data.get('dependency_type', 'unknown'),
                betweenness=edge_betweenness.get((u, v), 0.0),
                is_bridge=is_bridge,
                weight=data.get('weight', 1.0),
            )
        
        return metrics
    
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
    
    Args:
        graph_data: Full graph data
        layer: Target layer
        
    Returns:
        Tuple of (filtered_components, filtered_edges)
    """
    layer_def = get_layer_definition(layer)
    
    filtered_components = [
        c for c in graph_data.components
        if c.component_type in layer_def.component_types
    ]
    
    component_ids = {c.id for c in filtered_components}
    
    filtered_edges = [
        e for e in graph_data.edges
        if e.dependency_type in layer_def.dependency_types
        and e.source_id in component_ids
        and e.target_id in component_ids
    ]
    
    return filtered_components, filtered_edges