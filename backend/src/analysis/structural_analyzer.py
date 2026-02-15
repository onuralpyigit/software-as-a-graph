"""
Structural Analyzer

Computes raw topological metrics from graph data using NetworkX.
Supports multi-layer analysis by filtering based on DEPENDS_ON relationship types.

Metrics computed per component:
    Centrality : PageRank, Reverse PageRank, Betweenness, Harmonic Closeness,
                 Eigenvector (with Katz fallback)
    Degree     : In-degree, Out-degree, Total degree (raw + normalized)
    Resilience : Clustering coefficient, Articulation point flag,
                 Bridge count, Bridge ratio
    Weights    : Component weight, Dependency weight in/out

Metrics computed per edge:
    Edge betweenness (with inverted weights), Bridge flag

Metrics computed per graph:
    Node/edge count, Density, Connected components, Diameter, Avg path length,
    Assortativity, Avg clustering, AP count, Bridge count

Layer filtering:
    Each AnalysisLayer defines which dependency_types and component_types to
    include, so the same analyzer can serve app, infra, mw, or system views.

Weight semantics:
    DEPENDS_ON edge weights represent dependency strength (importance).
    - PageRank, Eigenvector/Katz: weights used as-is (importance)
    - Betweenness, Edge Betweenness: weights inverted (1/w) for distance semantics
    - Harmonic Closeness: no weights (topological proximity)

Usage:
    analyzer = StructuralAnalyzer()
    result = analyzer.analyze(graph_data, layer=AnalysisLayer.APP)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Set, Tuple

import networkx as nx

from src.core.layers import (
    AnalysisLayer,
    LAYER_DEFINITIONS,
    get_layer_definition,
)
from src.core.metrics import StructuralMetrics, EdgeMetrics, GraphSummary


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

from .models import StructuralAnalysisResult


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
      - .edges       : iterable of edge data objects with .source_id, .target_id,
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
            # Extract name from properties (fallback to id if not present)
            name = props.pop("name", comp.id)
            G.add_node(
                comp.id,
                component_type=comp.component_type,
                name=name,
                weight=getattr(comp, "weight", 1.0),
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
            2. Compute centrality metrics (directed, with correct weight semantics)
            3. Compute degree metrics (directed, raw + normalized)
            4. Detect articulation points / bridges (undirected)
            5. Compute edge metrics (directed, inverted weights)
            6. Assemble component metrics with all fields populated
            7. Build graph-level summary
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
        n_nodes = G.number_of_nodes()

        # --- Build inverted-weight graph for distance-based metrics ---
        G_dist = self._build_distance_graph(G)

        # --- Centrality metrics on the directed graph ---
        # PageRank: weights as importance (raw)
        pagerank = nx.pagerank(G, alpha=self.damping_factor, weight="weight")
        reverse_pagerank = nx.pagerank(G.reverse(), alpha=self.damping_factor, weight="weight")

        # Betweenness: weights as distance (inverted)
        betweenness = nx.betweenness_centrality(G_dist, weight="weight", normalized=True)

        # Harmonic closeness: no weights (topological proximity)
        closeness = nx.harmonic_centrality(G)
        # Normalize harmonic centrality to [0, 1] by dividing by (n-1)
        if n_nodes > 1:
            closeness = {v: c / (n_nodes - 1) for v, c in closeness.items()}

        # Eigenvector with Katz fallback
        eigenvector = self._safe_eigenvector(G)

        # --- Degree ---
        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())

        # --- Resilience (on undirected view) ---
        U = G.to_undirected()
        clustering = nx.clustering(U)
        art_points = set(nx.articulation_points(U)) if nx.is_connected(U) else self._art_points_disconnected(U)
        bridges = set(nx.bridges(U)) if nx.is_connected(U) else self._bridges_disconnected(U)

        # --- Bridge count per node ---
        node_bridge_count: Dict[str, int] = {}
        for (u, v) in bridges:
            node_bridge_count[u] = node_bridge_count.get(u, 0) + 1
            node_bridge_count[v] = node_bridge_count.get(v, 0) + 1

        # --- Edge betweenness (inverted weights) ---
        edge_betweenness = nx.edge_betweenness_centrality(G_dist, weight="weight", normalized=True)

        # --- Dependency weights per node ---
        dep_weight_in: Dict[str, float] = {nid: 0.0 for nid in G.nodes}
        dep_weight_out: Dict[str, float] = {nid: 0.0 for nid in G.nodes}
        for u, v, data in G.edges(data=True):
            w = data.get("weight", 1.0)
            dep_weight_out[u] += w
            dep_weight_in[v] += w

        # --- Assemble component metrics ---
        components: Dict[str, StructuralMetrics] = {}
        for nid in G.nodes:
            ntype = G.nodes[nid].get("component_type", "Unknown")
            if ntype not in analyze_types:
                continue

            raw_in = in_deg.get(nid, 0)
            raw_out = out_deg.get(nid, 0)
            total_raw = raw_in + raw_out
            bc = node_bridge_count.get(nid, 0)

            components[nid] = StructuralMetrics(
                id=nid,
                name=G.nodes[nid].get("name", nid),
                type=ntype,
                # Centrality
                pagerank=pagerank.get(nid, 0.0),
                reverse_pagerank=reverse_pagerank.get(nid, 0.0),
                betweenness=betweenness.get(nid, 0.0),
                closeness=closeness.get(nid, 0.0),
                eigenvector=eigenvector.get(nid, 0.0),
                # Degree (normalized)
                degree=total_raw / (2 * (n_nodes - 1)) if n_nodes > 1 else 0.0,
                in_degree=raw_in / (n_nodes - 1) if n_nodes > 1 else 0.0,
                out_degree=raw_out / (n_nodes - 1) if n_nodes > 1 else 0.0,
                # Degree (raw)
                in_degree_raw=raw_in,
                out_degree_raw=raw_out,
                # Resilience
                clustering_coefficient=clustering.get(nid, 0.0),
                is_articulation_point=nid in art_points,
                is_isolated=(raw_in + raw_out) == 0,
                bridge_count=bc,
                bridge_ratio=bc / total_raw if total_raw > 0 else 0.0,
                # Weights
                weight=G.nodes[nid].get("weight", 1.0),
                dependency_weight_in=dep_weight_in.get(nid, 0.0),
                dependency_weight_out=dep_weight_out.get(nid, 0.0),
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
        summary = self._build_summary(G, layer, art_points, bridges)

        return StructuralAnalysisResult(
            layer=layer,
            components=components,
            edges=edge_metrics,
            graph_summary=summary,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_distance_graph(G: nx.DiGraph) -> nx.DiGraph:
        """
        Create a copy of G with inverted edge weights for distance-based metrics.

        DEPENDS_ON weights represent dependency strength (importance).
        Betweenness centrality interprets weights as distances.
        Inversion ensures strongly-weighted dependencies are treated as
        "close" (preferred for shortest paths).

        w_distance = 1.0 / w_importance
        """
        G_dist = G.copy()
        for u, v, data in G_dist.edges(data=True):
            w = data.get("weight", 1.0)
            data["weight"] = 1.0 / w if w > 0 else 1.0
        return G_dist

    def _safe_eigenvector(self, G: nx.DiGraph) -> Dict[str, float]:
        """
        Compute eigenvector centrality with Katz centrality fallback.

        Eigenvector centrality may fail to converge on directed acyclic
        graphs (DAGs) or nearly-acyclic graphs because the dominant
        eigenvalue doesn't exist. Katz centrality with attenuation
        factor α handles these cases gracefully.
        """
        try:
            return nx.eigenvector_centrality(G, max_iter=500, weight="weight")
        except (nx.PowerIterationFailedConvergence, nx.NetworkXException):
            self._logger.warning(
                "Eigenvector centrality did not converge; "
                "falling back to Katz centrality"
            )
            try:
                return nx.katz_centrality(G, alpha=0.01, beta=1.0, weight="weight")
            except (nx.NetworkXException, nx.PowerIterationFailedConvergence):
                self._logger.warning(
                    "Katz centrality also failed; returning zeros"
                )
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

    def _build_summary(
        self,
        G: nx.DiGraph,
        layer: AnalysisLayer,
        art_points: Set[str],
        bridges: Set[Tuple[str, str]],
    ) -> GraphSummary:
        """Compute graph-level statistics including assortativity."""
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

        # Assortativity (Pearson correlation of degree-degree at edge endpoints)
        assortativity = 0.0
        try:
            assortativity = nx.degree_assortativity_coefficient(G)
        except (nx.NetworkXError, ValueError):
            pass

        # Average clustering
        avg_clustering = nx.average_clustering(U) if U.number_of_nodes() > 0 else 0.0

        # Node and edge type breakdowns
        node_types: Dict[str, int] = {}
        for _, data in G.nodes(data=True):
            t = data.get("component_type", "Unknown")
            node_types[t] = node_types.get(t, 0) + 1

        edge_types: Dict[str, int] = {}
        for _, _, data in G.edges(data=True):
            t = data.get("dependency_type", "unknown")
            edge_types[t] = edge_types.get(t, 0) + 1

        return GraphSummary(
            layer=layer.value,
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
            density=nx.density(G),
            avg_degree=sum(d for _, d in G.degree()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0.0,
            avg_clustering=avg_clustering,
            is_connected=nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else False,
            num_components=len(cc),
            num_articulation_points=len(art_points),
            num_bridges=len(bridges),
            diameter=diameter,
            avg_path_length=avg_path,
            assortativity=assortativity,
            node_types=node_types,
            edge_types=edge_types,
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