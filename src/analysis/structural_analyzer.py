"""
Structural Analyzer

Computes raw graph topological metrics using NetworkX.
This module is responsible ONLY for metric calculation, not scoring or interpretation.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Any

import networkx as nx

from src.core.graph_exporter import GraphData

@dataclass
class StructuralMetrics:
    """Raw topological metrics for a single component."""
    component_id: str
    component_type: str
    
    # Reliability inputs
    pagerank: float = 0.0
    in_degree: float = 0.0
    failure_propagation: float = 0.0  # Reverse PageRank
    
    # Maintainability inputs
    betweenness: float = 0.0
    degree: float = 0.0
    clustering_coefficient: float = 0.0
    
    # Availability inputs
    is_articulation_point: bool = False
    bridge_ratio: float = 0.0
    
    # Meta
    weight: float = 1.0

@dataclass
class StructuralAnalysisResult:
    """Container for all raw structural metrics."""
    metrics: Dict[str, StructuralMetrics]
    bridges: List[Tuple[str, str]]
    articulation_points: Set[str]

class StructuralAnalyzer:
    def __init__(self, damping_factor: float = 0.85):
        self.damping_factor = damping_factor
        self.logger = logging.getLogger(__name__)

    def analyze(self, graph_data: GraphData, weighted: bool = True) -> StructuralAnalysisResult:
        """Compute all raw structural metrics."""
        G = self._build_graph(graph_data, weighted)
        if len(G) == 0:
            return StructuralAnalysisResult({}, [], set())

        weight_attr = "weight" if weighted else None

        # 1. Centrality Metrics
        try:
            pagerank = nx.pagerank(G, alpha=self.damping_factor, weight=weight_attr)
        except:
            pagerank = {n: 1.0/len(G) for n in G}

        betweenness = nx.betweenness_centrality(G, weight=weight_attr)
        degree_cent = nx.degree_centrality(G)
        in_degree_cent = nx.in_degree_centrality(G)
        
        # 2. Advanced Metrics
        # Failure Propagation: Modeled as influence in the reverse graph
        try:
            fail_prop = nx.pagerank(G.reverse(), alpha=self.damping_factor, weight=weight_attr)
        except:
            fail_prop = nx.out_degree_centrality(G)

        # Clustering (Undirected view for modularity)
        G_undir = G.to_undirected()
        clustering = nx.clustering(G_undir, weight=weight_attr)
        
        # 3. Structural Weakness
        try:
            articulation_points = set(nx.articulation_points(G_undir))
            bridges = list(nx.bridges(G_undir))
        except:
            articulation_points = set()
            bridges = []

        # Bridge Ratio calculation
        bridge_counts = {n: 0 for n in G}
        for u, v in bridges:
            bridge_counts[u] += 1
            bridge_counts[v] += 1
            
        # 4. Assembly
        metrics = {}
        comp_lookup = {c.id: c for c in graph_data.components}
        
        for n in G.nodes:
            deg = G_undir.degree(n)
            comp = comp_lookup.get(n)
            
            metrics[n] = StructuralMetrics(
                component_id=n,
                component_type=comp.component_type if comp else "Unknown",
                pagerank=pagerank.get(n, 0),
                in_degree=in_degree_cent.get(n, 0),
                failure_propagation=fail_prop.get(n, 0),
                betweenness=betweenness.get(n, 0),
                degree=degree_cent.get(n, 0),
                clustering_coefficient=clustering.get(n, 0),
                is_articulation_point=(n in articulation_points),
                bridge_ratio=(bridge_counts[n] / deg) if deg > 0 else 0.0,
                weight=comp.weight if comp else 1.0
            )

        return StructuralAnalysisResult(metrics, bridges, articulation_points)

    def _build_graph(self, data: GraphData, weighted: bool) -> nx.DiGraph:
        G = nx.DiGraph()
        for c in data.components:
            G.add_node(c.id)
        for e in data.edges:
            w = e.weight if weighted else 1.0
            G.add_edge(e.source_id, e.target_id, weight=w)
        return G