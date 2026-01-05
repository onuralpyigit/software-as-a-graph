"""
Structural Analyzer

Computes raw graph topological metrics using NetworkX.
Supports analysis of both Components (Nodes) and Dependencies (Edges).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Set

import networkx as nx

from src.core.graph_exporter import GraphData

@dataclass
class StructuralMetrics:
    """Raw topological metrics for a single component."""
    id: str
    type: str
    
    # Centrality (Importance)
    pagerank: float = 0.0          # Global importance
    betweenness: float = 0.0       # Bridge/Flow control
    degree: float = 0.0            # Connectivity
    in_degree: float = 0.0         # Dependents
    out_degree: float = 0.0        # Dependencies
    
    # Resilience
    clustering_coefficient: float = 0.0  # Local redundancy
    
    # Criticality Flags
    is_articulation_point: bool = False  # If removed, graph disconnects
    is_isolated: bool = False
    bridge_ratio: float = 0.0
    
    weight: float = 1.0  # Intrinsic weight from Import

@dataclass
class EdgeMetrics:
    """Raw topological metrics for a single edge."""
    source: str
    target: str
    dependency_type: str
    
    betweenness: float = 0.0
    is_bridge: bool = False
    weight: float = 1.0

@dataclass
class StructuralAnalysisResult:
    """Container for all raw structural metrics."""
    components: Dict[str, StructuralMetrics]
    edges: Dict[Tuple[str, str], EdgeMetrics]
    graph_summary: Dict[str, Any]

class StructuralAnalyzer:
    def __init__(self, damping_factor: float = 0.85):
        self.damping_factor = damping_factor
        self.logger = logging.getLogger(__name__)

    def analyze(self, graph_data: GraphData) -> StructuralAnalysisResult:
        """Compute structural metrics for the provided GraphData."""
        G = self._build_graph(graph_data)
        
        if len(G) == 0:
            return StructuralAnalysisResult({}, {}, {"nodes": 0, "edges": 0})

        # 1. Component Metrics (Directed)
        try:
            pagerank = nx.pagerank(G, alpha=self.damping_factor, weight="weight")
        except:
            pagerank = {n: 1.0/len(G) for n in G}

        betweenness = nx.betweenness_centrality(G, weight="weight")
        degree_cent = nx.degree_centrality(G)
        in_degree_cent = nx.in_degree_centrality(G)
        out_degree_cent = nx.out_degree_centrality(G)

        # 2. Resilience Metrics (Undirected View)
        G_undir = G.to_undirected()
        clustering = nx.clustering(G_undir, weight="weight")
        
        # Articulation Points & Bridges (Connectivity Criticality)
        # Note: Only valid for connected components, so we iterate over them
        articulation_points: Set[str] = set()
        bridges = []
        
        for component in nx.connected_components(G_undir):
            subg = G_undir.subgraph(component)
            if len(subg) > 2:
                articulation_points.update(nx.articulation_points(subg))
                bridges.extend(nx.bridges(subg))

        # Bridge Ratio calculation
        bridge_counts = {n: 0 for n in G}
        bridge_set = set(bridges)
        for u, v in bridges:
            bridge_counts[u] += 1
            bridge_counts[v] += 1

        # 3. Edge Metrics
        edge_betweenness = nx.edge_betweenness_centrality(G, weight="weight")

        # 4. Assembly - Components
        comp_metrics = {}
        # Quick lookup for component raw data (type, intrinsic weight)
        comp_lookup = {c.id: c for c in graph_data.components}
        
        for n in G.nodes:
            deg = G_undir.degree(n)
            c_data = comp_lookup.get(n)
            
            comp_metrics[n] = StructuralMetrics(
                id=n,
                type=c_data.component_type if c_data else "Unknown",
                pagerank=pagerank.get(n, 0),
                betweenness=betweenness.get(n, 0),
                degree=degree_cent.get(n, 0),
                in_degree=in_degree_cent.get(n, 0),
                out_degree=out_degree_cent.get(n, 0),
                clustering_coefficient=clustering.get(n, 0),
                is_articulation_point=(n in articulation_points),
                is_isolated=(deg == 0),
                bridge_ratio=(bridge_counts[n] / deg) if deg > 0 else 0.0,
                weight=c_data.weight if c_data else 1.0
            )

        # 5. Assembly - Edges
        edge_metrics_dict = {}
        edge_lookup = {(e.source_id, e.target_id): e for e in graph_data.edges}
        
        for u, v in G.edges:
            e_data = edge_lookup.get((u, v))
            is_bridge_edge = (u, v) in bridge_set or (v, u) in bridge_set
            
            edge_metrics_dict[(u, v)] = EdgeMetrics(
                source=u,
                target=v,
                dependency_type=e_data.dependency_type if e_data else "structural",
                betweenness=edge_betweenness.get((u, v), 0),
                is_bridge=is_bridge_edge,
                weight=e_data.weight if e_data else 1.0
            )

        return StructuralAnalysisResult(
            components=comp_metrics,
            edges=edge_metrics_dict,
            graph_summary={
                "nodes": len(G.nodes), 
                "edges": len(G.edges),
                "density": nx.density(G)
            }
        )

    def _build_graph(self, data: GraphData) -> nx.DiGraph:
        G = nx.DiGraph()
        for c in data.components:
            G.add_node(c.id, weight=c.weight)
        for e in data.edges:
            G.add_edge(e.source_id, e.target_id, weight=e.weight)
        return G