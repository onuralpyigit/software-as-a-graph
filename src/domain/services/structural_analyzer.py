"""
Structural Analyzer

Computes raw topological metrics from graph data using NetworkX.
Supports multi-layer analysis by filtering based on DEPENDS_ON relationship types.

Metrics computed per component:
    Centrality : PageRank, Reverse PageRank, Betweenness, Closeness, Eigenvector
    Degree     : In-degree, Out-degree, Total degree
    Resilience : Clustering coefficient, Articulation point flag

Metrics computed per edge:
    Edge betweenness, Bridge flag

Metrics computed per graph:
    Node/edge count, Density, Connected components, Diameter, Avg path length

Layer filtering:
    Each AnalysisLayer defines which dependency_types and component_types to
    include, so the same analyzer can serve app, infra, mw, or system views.

Usage:
    analyzer = StructuralAnalyzer()
    result = analyzer.analyze(graph_data, layer=AnalysisLayer.APP)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Set, Tuple

import networkx as nx

from src.domain.config.layers import (
    AnalysisLayer,
    LAYER_DEFINITIONS,
    get_layer_definition,
)
from src.domain.models.metrics import StructuralMetrics, EdgeMetrics, GraphSummary


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class StructuralAnalysisResult:
    """Container for all raw structural analysis results for one layer."""

    layer: AnalysisLayer
    components: Dict[str, StructuralMetrics]
    edges: Dict[Tuple[str, str], EdgeMetrics]
    graph_summary: GraphSummary

    # -- convenience queries --------------------------------------------------

    def get_components_by_type(self, comp_type: str) -> List[StructuralMetrics]:
        return [c for c in self.components.values() if c.type == comp_type]

    def get_articulation_points(self) -> List[StructuralMetrics]:
        return [c for c in self.components.values() if c.is_articulation_point]

    def get_bridges(self) -> List[EdgeMetrics]:
        return [e for e in self.edges.values() if e.is_bridge]

    def get_top_by_metric(
        self, metric: str, n: int = 10, reverse: bool = True,
    ) -> List[StructuralMetrics]:
        return sorted(
            self.components.values(),
            key=lambda c: getattr(c, metric, 0),
            reverse=reverse,
        )[:n]


# ---------------------------------------------------------------------------
# Layer subgraph extraction (pure function)
# ---------------------------------------------------------------------------

def extract_layer_subgraph(
    graph_data: Any,
    layer: AnalysisLayer,
) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph containing only the components and DEPENDS_ON
    edges relevant to *layer*.

    The graph_data object is expected to provide:
      - .components  : iterable of component data objects with .id, .component_type
      - .edges       : iterable of edge data objects with .source, .target,
                       .dependency_type, .weight (DEPENDS_ON edges)

    Returns an empty DiGraph when no matching data is found.
    """
    defn = get_layer_definition(layer)
    G = nx.DiGraph()

    # Collect allowed component IDs
    allowed_ids: Set[str] = set()
    for comp in graph_data.components:
        if comp.component_type in defn.component_types:
            allowed_ids.add(comp.id)
            props = getattr(comp, "properties", {}).copy()
            props.pop("name", None)  # Avoid duplicate 'name' arg
            G.add_node(
                comp.id,
                component_type=comp.component_type,
                name=getattr(comp, "name", comp.id),
                **props,
            )

    # Add DEPENDS_ON edges whose dependency_type matches the layer
    for edge in graph_data.edges:
        if edge.dependency_type not in defn.dependency_types:
            continue
        if edge.source_id not in allowed_ids or edge.target_id not in allowed_ids:
            continue
        G.add_edge(
            edge.source_id,
            edge.target_id,
            dependency_type=edge.dependency_type,
            weight=getattr(edge, "weight", 1.0),
        )

    return G


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class StructuralAnalyzer:
    """
    Analyses graph structure to compute topological metrics.

    Supports multi-layer analysis by filtering components and edges
    based on the specified AnalysisLayer's definition.
    """

    def __init__(self, damping_factor: float = 0.85) -> None:
        self.damping_factor = damping_factor
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        graph_data: Any,
        layer: AnalysisLayer = AnalysisLayer.SYSTEM,
    ) -> StructuralAnalysisResult:
        """
        Compute structural metrics for *graph_data* filtered by *layer*.

        Steps:
            1. Build layer-specific subgraph
            2. Compute centrality metrics
            3. Compute degree metrics
            4. Detect articulation points / bridges
            5. Compute edge metrics
            6. Build graph-level summary
        """
        defn = get_layer_definition(layer)
        G = extract_layer_subgraph(graph_data, layer)

        self._logger.info(
            "Structural analysis [%s]: %d nodes, %d edges",
            layer.value, G.number_of_nodes(), G.number_of_edges(),
        )

        if G.number_of_nodes() == 0:
            return self._empty_result(layer)

        # Filter components to analyse (e.g. MW only reports Brokers)
        analyze_types = defn.types_to_analyze

        # --- Centrality metrics on the directed graph ---
        pagerank = nx.pagerank(G, alpha=self.damping_factor, weight="weight")
        reverse_pagerank = nx.pagerank(G.reverse(), alpha=self.damping_factor, weight="weight")
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
        closeness = nx.closeness_centrality(G, wf_improved=True)
        eigenvector = self._safe_eigenvector(G)

        # --- Degree ---
        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())

        # --- Resilience (on undirected view) ---
        U = G.to_undirected()
        clustering = nx.clustering(U)
        art_points = set(nx.articulation_points(U)) if nx.is_connected(U) else self._art_points_disconnected(U)
        bridges = set(nx.bridges(U)) if nx.is_connected(U) else self._bridges_disconnected(U)

        # --- Edge betweenness ---
        edge_betweenness = nx.edge_betweenness_centrality(G, weight="weight", normalized=True)

        # --- Assemble component metrics ---
        components: Dict[str, StructuralMetrics] = {}
        for nid in G.nodes:
            ntype = G.nodes[nid].get("component_type", "Unknown")
            if ntype not in analyze_types:
                continue
            components[nid] = StructuralMetrics(
                id=nid,
                name=G.nodes[nid].get("name", nid),
                type=ntype,
                pagerank=pagerank.get(nid, 0.0),
                reverse_pagerank=reverse_pagerank.get(nid, 0.0),
                betweenness=betweenness.get(nid, 0.0),
                closeness=closeness.get(nid, 0.0), # No change, just context
                eigenvector=eigenvector.get(nid, 0.0),
                in_degree_raw=in_deg.get(nid, 0),
                out_degree_raw=out_deg.get(nid, 0),
                clustering_coefficient=clustering.get(nid, 0.0),
                is_articulation_point=nid in art_points,
                is_isolated=(in_deg.get(nid, 0) + out_deg.get(nid, 0)) == 0,
            )

        # --- Assemble edge metrics ---
        edge_metrics: Dict[Tuple[str, str], EdgeMetrics] = {}
        for (u, v), data in G.edges.items():
            # Only report edges where at least one endpoint is in analyze_types
            u_type = G.nodes[u].get("component_type")
            v_type = G.nodes[v].get("component_type")
            if u_type not in analyze_types and v_type not in analyze_types:
                continue
            key = (u, v)
            edge_metrics[key] = EdgeMetrics(
                source=u,
                target=v,
                source_type=u_type,
                target_type=v_type,
                dependency_type=data.get("dependency_type", "unknown"),
                weight=data.get("weight", 1.0),
                betweenness=edge_betweenness.get(key, 0.0),
                is_bridge=(u, v) in bridges or (v, u) in bridges,
            )

        # --- Graph-level summary ---
        summary = self._build_summary(G, layer)

        return StructuralAnalysisResult(
            layer=layer,
            components=components,
            edges=edge_metrics,
            graph_summary=summary,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _safe_eigenvector(self, G: nx.DiGraph) -> Dict[str, float]:
        """Compute eigenvector centrality with fallback for non-convergence."""
        try:
            return nx.eigenvector_centrality(G, max_iter=500, weight="weight")
        except (nx.PowerIterationFailedConvergence, nx.NetworkXException):
            self._logger.warning("Eigenvector centrality did not converge; using zeros")
            return {n: 0.0 for n in G.nodes}

    @staticmethod
    def _art_points_disconnected(U: nx.Graph) -> Set[str]:
        """Articulation points across all connected components."""
        pts: Set[str] = set()
        for comp in nx.connected_components(U):
            sub = U.subgraph(comp)
            if len(sub) >= 3:
                pts.update(nx.articulation_points(sub))
        return pts

    @staticmethod
    def _bridges_disconnected(U: nx.Graph) -> Set[Tuple[str, str]]:
        """Bridges across all connected components."""
        br: Set[Tuple[str, str]] = set()
        for comp in nx.connected_components(U):
            sub = U.subgraph(comp)
            if len(sub) >= 2:
                br.update(nx.bridges(sub))
        return br

    def _build_summary(self, G: nx.DiGraph, layer: AnalysisLayer) -> GraphSummary:
        """Compute graph-level statistics."""
        U = G.to_undirected()
        cc = list(nx.connected_components(U))
        largest_cc = max(cc, key=len) if cc else set()

        diameter = 0
        avg_path = 0.0
        if len(largest_cc) >= 2:
            sub = G.subgraph(largest_cc)
            try:
                diameter = nx.diameter(sub.to_undirected())
                avg_path = nx.average_shortest_path_length(sub.to_undirected())
            except nx.NetworkXError:
                pass

        return GraphSummary(
            layer=layer.value,
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
            density=nx.density(G),
            num_components=len(cc),
            diameter=diameter,
            avg_path_length=avg_path,
        )

    @staticmethod
    def _empty_result(layer: AnalysisLayer) -> StructuralAnalysisResult:
        return StructuralAnalysisResult(
            layer=layer,
            components={},
            edges={},
            graph_summary=GraphSummary(
                layer=layer.value,
                nodes=0, edges=0, density=0.0,
                num_components=0, diameter=0, avg_path_length=0.0,
            ),
        )