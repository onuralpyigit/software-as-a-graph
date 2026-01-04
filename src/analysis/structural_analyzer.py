"""
Structural Analyzer

Computes raw graph topological metrics using NetworkX.
Supports analysis of both Components (Nodes) and Dependencies (Edges).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Any

import networkx as nx

from src.core.graph_exporter import GraphData

@dataclass
class StructuralMetrics:
    """Raw topological metrics for a single component."""
    id: str
    type: str
    
    # Centrality
    pagerank: float = 0.0
    betweenness: float = 0.0
    degree: float = 0.0
    in_degree: float = 0.0
    
    # Advanced
    failure_propagation: float = 0.0  # Reverse PageRank
    clustering_coefficient: float = 0.0
    
    # Criticality Flags
    is_articulation_point: bool = False
    bridge_ratio: float = 0.0
    
    weight: float = 1.0

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

        # 1. Node Metrics
        try:
            pagerank = nx.pagerank(G, alpha=self.damping_factor, weight="weight")
            fail_prop = nx.pagerank(G.reverse(), alpha=self.damping_factor, weight="weight")
        except:
            # Fallback for empty/disconnected graphs causing convergence issues
            pagerank = {n: 1.0/len(G) for n in G}
            fail_prop = {n: 1.0/len(G) for n in G}

        betweenness = nx.betweenness_centrality(G, weight="weight")
        degree_cent = nx.degree_centrality(G)
        in_degree_cent = nx.in_degree_centrality(G)
        
        G_undir = G.to_undirected()
        clustering = nx.clustering(G_undir, weight="weight")
        
        try:
            articulation_points = set(nx.articulation_points(G_undir))
            bridges = list(nx.bridges(G_undir))
        except:
            articulation_points = set()
            bridges = []

        # Bridge Ratio (fraction of incident edges that are bridges)
        bridge_counts = {n: 0 for n in G}
        bridge_set = set(bridges)
        for u, v in bridges:
            bridge_counts[u] += 1
            bridge_counts[v] += 1

        # 2. Edge Metrics
        edge_betweenness = nx.edge_betweenness_centrality(G, weight="weight")

        # 3. Assembly - Components
        comp_metrics = {}
        comp_lookup = {c.id: c for c in graph_data.components}
        
        for n in G.nodes:
            deg = G_undir.degree(n)
            comp_data = comp_lookup.get(n)
            
            comp_metrics[n] = StructuralMetrics(
                id=n,
                type=comp_data.component_type if comp_data else "Unknown",
                pagerank=pagerank.get(n, 0),
                in_degree=in_degree_cent.get(n, 0),
                failure_propagation=fail_prop.get(n, 0),
                betweenness=betweenness.get(n, 0),
                degree=degree_cent.get(n, 0),
                clustering_coefficient=clustering.get(n, 0),
                is_articulation_point=(n in articulation_points),
                bridge_ratio=(bridge_counts[n] / deg) if deg > 0 else 0.0,
                weight=comp_data.weight if comp_data else 1.0
            )

        # 4. Assembly - Edges
        edge_metrics_dict = {}
        edge_lookup = {(e.source_id, e.target_id): e for e in graph_data.edges}
        
        for u, v in G.edges:
            # Normalize edge key for lookup (directed)
            e_data = edge_lookup.get((u, v))
            is_bridge = (u, v) in bridge_set or (v, u) in bridge_set
            
            edge_metrics_dict[(u, v)] = EdgeMetrics(
                source=u,
                target=v,
                dependency_type=e_data.dependency_type if e_data else "unknown",
                betweenness=edge_betweenness.get((u, v), 0),
                is_bridge=is_bridge,
                weight=e_data.weight if e_data else 1.0
            )

        return StructuralAnalysisResult(
            components=comp_metrics,
            edges=edge_metrics_dict,
            graph_summary={"nodes": len(G.nodes), "edges": len(G.edges)}
        )

    def _build_graph(self, data: GraphData) -> nx.DiGraph:
        G = nx.DiGraph()
        for c in data.components:
            G.add_node(c.id)
        for e in data.edges:
            G.add_edge(e.source_id, e.target_id, weight=e.weight)
        return G