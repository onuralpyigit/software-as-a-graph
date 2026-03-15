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
    Pub-Sub    : pubsub_degree, pubsub_betweenness, broker_exposure
                 (derived from raw PUBLISHES_TO/SUBSCRIBES_TO/ROUTES edges)

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
from networkx.utils import reverse_cuthill_mckee_ordering

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
            # Extract code-quality fields from props (written there by get_graph_data)
            code_qual = {
                "loc": props.pop("loc", 0),
                "cyclomatic_complexity": props.pop("cyclomatic_complexity", 0.0),
                "coupling_afferent": props.pop("coupling_afferent", 0),
                "coupling_efferent": props.pop("coupling_efferent", 0),
                "lcom": props.pop("lcom", 0.0),
            }
            # Extract subscriber_count for Topic nodes
            subscriber_count = 0
            if comp.component_type == "Topic":
                subscriber_count = props.pop("subscriber_count", 0)

            G.add_node(
                comp.id,
                component_type=comp.component_type,
                name=name,
                weight=getattr(comp, "weight", 1.0),
                subscriber_count=subscriber_count,
                **code_qual,
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
            path_count=getattr(edge, "path_count", 1),
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
        G_rev = G.reverse()
        reverse_pagerank = nx.pagerank(G_rev, alpha=self.damping_factor, weight="weight")

        # Betweenness: weights as distance (inverted)
        betweenness = nx.betweenness_centrality(G_dist, weight="weight", normalized=True)

        # Harmonic closeness: no weights (topological proximity)
        closeness = nx.harmonic_centrality(G)
        reverse_closeness = nx.harmonic_centrality(G_rev)
        # Normalize harmonic centrality to [0, 1] by dividing by (n-1)
        if n_nodes > 1:
            closeness = {v: c / (n_nodes - 1) for v, c in closeness.items()}
            reverse_closeness = {v: c / (n_nodes - 1) for v, c in reverse_closeness.items()}

        # Eigenvector with Katz fallback
        eigenvector = self._safe_eigenvector(G)
        reverse_eigenvector = self._safe_eigenvector(G_rev)

        # --- Degree ---
        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())

        # --- MPCI & FOC (Tier 1 New) ---
        mpci: Dict[str, float] = {}
        for nid in G.nodes:
            # MPCI(v) = Σ_{e ∈ InEdges(v)} max(path_count(e) − 1, 0) / (|V| − 1)
            raw_mpci = 0.0
            for u, v, data in G.in_edges(nid, data=True):
                raw_mpci += max(data.get("path_count", 1) - 1, 0)
            mpci[nid] = raw_mpci / (n_nodes - 1) if n_nodes > 1 else 0.0

        foc: Dict[str, float] = {}
        topic_nodes = [n for n, d in G.nodes(data=True) if d.get("component_type") == "Topic"]
        if topic_nodes:
            max_sub = max((G.nodes[n].get("subscriber_count", 0) for n in topic_nodes), default=0)
            for n in topic_nodes:
                foc[n] = G.nodes[n].get("subscriber_count", 0) / max_sub if max_sub > 0 else 0.0
        # For non-topic nodes, FOC stays 0.0 by default in StructuralMetrics

        # --- AP_c_directed & CDI (Tier 1 New - Migrated from QualityAnalyzer) ---
        ap_scores = self._compute_continuous_ap_scores(G)

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

        # --- Pub-sub topology metrics (raw PUBLISHES_TO / SUBSCRIBES_TO) ---
        # allowed_ids = all component IDs in the layer subgraph
        allowed_ids: Set[str] = set(G.nodes)
        pubsub_metrics = self._compute_pubsub_metrics(graph_data, allowed_ids)

        # --- QoS profile from topic nodes ---
        qos_profile = self._collect_qos_profile(graph_data)

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
            ps = pubsub_metrics.get(nid, {})

            # Raw code-quality values (0.0 for non-qualifying node types)
            # loc_norm / complexity_norm / lcom_norm store the RAW values here;
            # the post-loop normalization pass will overwrite them with [0,1] values.
            # Both Application and Library nodes carry code-quality attributes;
            # all other types (Broker, Node, Topic) get zeroed fields.
            is_cq_node = ntype in ("Application", "Library")
            raw_loc  = G.nodes[nid].get("loc", 0) if is_cq_node else 0
            raw_cc   = G.nodes[nid].get("cyclomatic_complexity", 0.0) if is_cq_node else 0.0
            raw_lcom = G.nodes[nid].get("lcom", 0.0) if is_cq_node else 0.0
            raw_ca   = G.nodes[nid].get("coupling_afferent", 0)
            raw_ce   = G.nodes[nid].get("coupling_efferent", 0)
            instability_val = raw_ce / max(raw_ca + raw_ce, 1) if is_cq_node and (raw_ca + raw_ce) > 0 else 0.0

            components[nid] = StructuralMetrics(
                id=nid,
                name=G.nodes[nid].get("name", nid),
                type=ntype,
                # Centrality
                pagerank=pagerank.get(nid, 0.0),
                reverse_pagerank=reverse_pagerank.get(nid, 0.0),
                betweenness=betweenness.get(nid, 0.0),
                closeness=closeness.get(nid, 0.0),
                reverse_closeness=reverse_closeness.get(nid, 0.0),
                eigenvector=eigenvector.get(nid, 0.0),
                reverse_eigenvector=reverse_eigenvector.get(nid, 0.0),
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
                ap_c_directed=ap_scores.get(nid, {}).get("ap_c_dir", 0.0),
                cdi=ap_scores.get(nid, {}).get("cdi", 0.0),
                is_isolated=(raw_in + raw_out) == 0,
                bridge_count=bc,
                bridge_ratio=bc / total_raw if total_raw > 0 else 0.0,
                # Pub-sub topology
                pubsub_degree=ps.get("pubsub_degree", 0.0),
                pubsub_betweenness=ps.get("pubsub_betweenness", 0.0),
                broker_exposure=ps.get("broker_exposure", 0.0),
                fan_out_criticality=foc.get(nid, 0.0),
                mpci=mpci.get(nid, 0.0),
                # Code quality — raw stored temporarily; post-loop normalisation fills them
                # loc_norm / complexity_norm / lcom_norm hold RAW values until the pass
                loc_norm=float(raw_loc),
                complexity_norm=raw_cc,
                lcom_norm=raw_lcom,
                instability_code=instability_val,
                code_quality_penalty=0.0,  # filled by _compute_code_quality_metrics
                # Weights
                weight=G.nodes[nid].get("weight", 1.0),
                dependency_weight_in=dep_weight_in.get(nid, 0.0),
                dependency_weight_out=dep_weight_out.get(nid, 0.0),
            )

        # --- Code-quality normalisation (Application and Library nodes, separately) ---
        # Min-max scales loc_norm, complexity_norm, lcom_norm across each type's population,
        # then computes code_quality_penalty = 0.40·CC + 0.35·instability + 0.25·LCOM
        self._compute_code_quality_metrics(components)

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

        # --- Bandwidth minimization (RCM Ordering) ---
        # Used for topological reordering of the dependency matrix to surface
        # structural clusters.
        try:
            rcm_order = list(reverse_cuthill_mckee_ordering(U))
        except Exception as e:
            self._logger.warning(f"RCM ordering failed: {e}")
            rcm_order = list(G.nodes)

        # --- Graph-level summary ---
        summary = self._build_summary(G, layer, art_points, bridges)

        return StructuralAnalysisResult(
            layer=layer,
            components=components,
            edges=edge_metrics,
            graph_summary=summary,
            graph=G,
            qos_profile=qos_profile,
            rcm_order=rcm_order,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_code_quality_metrics(
        components: Dict[str, "StructuralMetrics"],
    ) -> None:
        """
        Normalise code-quality fields in-place for Application and Library nodes.

        Application and Library nodes are normalised **independently** (separate
        population min-max passes) because their typical LOC / CC scales differ
        significantly.  Both use the same CQP formula:

            CQP = 0.40 * complexity_norm + 0.35 * instability_code + 0.25 * lcom_norm

        instability_code is already in [0,1] (Ce/(Ca+Ce)) — not re-normalised.

        All other component types (Broker, Node, Topic) are skipped; their
        code_quality_penalty stays 0.0.
        """
        W_CC   = 0.40
        W_INS  = 0.35
        W_LCOM = 0.25

        def _mm(values: list, lo: float, hi: float) -> list:
            span = hi - lo
            if span == 0:
                return [0.0] * len(values)
            return [(v - lo) / span for v in values]

        # Process each qualifying node type as an independent population
        for node_type in ("Application", "Library"):
            ids   = []
            locs  = []
            ccs   = []
            lcoms = []

            for nid, m in components.items():
                if m.type != node_type:
                    continue
                ids.append(nid)
                locs.append(m.loc_norm)        # holds raw LOC float at this point
                ccs.append(m.complexity_norm)  # holds raw CC float
                lcoms.append(m.lcom_norm)      # holds raw LCOM float

            if not ids:
                continue  # no nodes of this type in the layer

            norm_locs  = _mm(locs,  min(locs),  max(locs))
            norm_ccs   = _mm(ccs,   min(ccs),   max(ccs))
            norm_lcoms = _mm(lcoms, min(lcoms), max(lcoms))

            for idx, nid in enumerate(ids):
                m = components[nid]
                m.loc_norm        = norm_locs[idx]
                m.complexity_norm = norm_ccs[idx]
                m.lcom_norm       = norm_lcoms[idx]
                m.code_quality_penalty = (
                    W_CC   * m.complexity_norm
                    + W_INS  * m.instability_code
                    + W_LCOM * m.lcom_norm
                )

    @staticmethod
    def _compute_pubsub_metrics(
        graph_data: Any,
        allowed_ids: Set[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute pub-sub topology metrics for Application nodes by building a
        bipartite app↔topic graph from raw PUBLISHES_TO and SUBSCRIBES_TO edges.

        Also resolves broker exposure via ROUTES edges.

        Returns:
            Dict[app_id, {pubsub_degree, pubsub_betweenness, broker_exposure}]
        """
        import networkx as nx

        PUBSUB_EDGE_TYPES = {"PUBLISHES_TO", "SUBSCRIBES_TO", "publishes_to", "subscribes_to"}
        ROUTES_EDGE_TYPES = {"ROUTES", "routes"}

        # Build bipartite app-topic undirected graph
        G_ps = nx.Graph()
        topic_to_brokers: Dict[str, Set[str]] = {}

        for edge in graph_data.edges:
            dep = getattr(edge, "dependency_type", "") or ""
            src, tgt = edge.source_id, edge.target_id

            if dep in PUBSUB_EDGE_TYPES:
                # Only include edges from allowed (layer-filtered) app nodes
                if src in allowed_ids:
                    G_ps.add_node(src, bipartite=0)
                    G_ps.add_node(tgt, bipartite=1)
                    G_ps.add_edge(src, tgt)

            elif dep in ROUTES_EDGE_TYPES:
                # Broker → Topic routing (for broker_exposure)
                topic_to_brokers.setdefault(tgt, set()).add(src)

        result: Dict[str, Dict[str, float]] = {}

        if G_ps.number_of_nodes() == 0:
            return result

        # Betweenness centrality on the bipartite graph
        try:
            bt = nx.betweenness_centrality(G_ps, normalized=True)
        except Exception:
            bt = {n: 0.0 for n in G_ps.nodes}

        # Max values for normalization
        max_deg = max((G_ps.degree(n) for n in G_ps.nodes if G_ps.nodes[n].get("bipartite") == 0), default=1)
        max_exp = max(
            (len(topic_to_brokers.get(t, set()))
             for n in G_ps.nodes if G_ps.nodes[n].get("bipartite") == 0
             for t in G_ps.neighbors(n)),
            default=1,
        ) or 1

        for node in G_ps.nodes:
            if G_ps.nodes[node].get("bipartite") != 0:
                continue  # skip topic nodes
            if node not in allowed_ids:
                continue

            deg = G_ps.degree(node)
            # broker_exposure: average number of distinct brokers routing
            # topics this app touches (normalised to [0,1])
            touched_topics = list(G_ps.neighbors(node))
            if touched_topics:
                broker_counts = [len(topic_to_brokers.get(t, set())) for t in touched_topics]
                avg_broker_exp = sum(broker_counts) / len(broker_counts)
            else:
                avg_broker_exp = 0.0

            result[node] = {
                "pubsub_degree": deg / max_deg if max_deg > 0 else 0.0,
                "pubsub_betweenness": bt.get(node, 0.0),
                "broker_exposure": avg_broker_exp / max_exp if max_exp > 0 else 0.0,
            }

        return result

    @staticmethod
    def _collect_qos_profile(graph_data: Any) -> Dict[str, Any]:
        """
        Aggregate topic QoS distributions from graph_data for use by
        QoS-aware RMAV weight adjustment in QualityAnalyzer.

        Returns a dict:
            {
              "durability": {"volatile": N, "transient_local": N, ...},
              "reliability": {"best_effort": N, "reliable": N},
              "priority":    {"low": N, "medium": N, "high": N, "critical": N},
              "total_topics": N,
            }
        """
        durability: Dict[str, int] = {}
        reliability: Dict[str, int] = {}
        priority: Dict[str, int] = {}
        total = 0

        for comp in graph_data.components:
            if getattr(comp, "component_type", "") != "Topic":
                continue
            props = getattr(comp, "properties", {}) or {}
            qos = props.get("qos", {})

            dur = qos.get("durability", "").lower().replace(" ", "_") if qos else ""
            rel = qos.get("reliability", "").lower().replace(" ", "_") if qos else ""
            pri = qos.get("transport_priority", "").lower().replace(" ", "_") if qos else ""

            if dur:
                durability[dur] = durability.get(dur, 0) + 1
            if rel:
                reliability[rel] = reliability.get(rel, 0) + 1
            if pri:
                priority[pri] = priority.get(pri, 0) + 1
            total += 1

        return {
            "durability": durability,
            "reliability": reliability,
            "priority": priority,
            "total_topics": total,
        }

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

    def _compute_continuous_ap_scores(
        self, G_dir: nx.DiGraph
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute continuous AP scores, directed AP scores, and CDI per component.
        Migrated from QualityAnalyzer for better performance (one-pass pipeline).
        """
        import networkx as nx
        import random
        ap_scores: Dict[str, Dict[str, float]] = {}
        
        n = G_dir.number_of_nodes()
        if n <= 1:
            return {nid: {"ap_c_dir": 0.0, "cdi": 0.0} for nid in G_dir.nodes}

        # Undirected version for standard AP_c
        G_undir = G_dir.to_undirected()
        # Transposed version for in-SPOF
        G_T_undir = G_dir.reverse(copy=True).to_undirected()

        # Optimization: Sample BFS for large graphs
        use_sampling = n > 300
        sample_size = 50 if use_sampling else n
        random_nodes = random.sample(list(G_dir.nodes), sample_size) if use_sampling else list(G_dir.nodes)
        
        # Pre-compute baseline average shortest path length (undirected)
        # Using largest CC to avoid infinity issues
        cc = list(nx.connected_components(G_undir))
        largest_cc = max(cc, key=len) if cc else set()
        baseline_avg_path = 0.0
        if len(largest_cc) > 1:
            G_sub = G_undir.subgraph(largest_cc)
            try:
                baseline_avg_path = nx.average_shortest_path_length(G_sub)
            except (nx.NetworkXError, ZeroDivisionError):
                baseline_avg_path = 0.0

        cdi_raw: Dict[str, float] = {}

        for nid in G_dir.nodes:
            # --- AP_c_out (undirected removal) ---
            G_out = G_undir.copy()
            G_out.remove_node(nid)
            out_cc = list(nx.connected_components(G_out))
            largest_out = max(len(c) for c in out_cc) if out_cc else 0
            ap_c_out = 1.0 - (largest_out / (n - 1)) if n > 1 else 0.0

            # --- AP_c_in (transposed removal) ---
            G_in = G_T_undir.copy()
            G_in.remove_node(nid)
            in_cc = list(nx.connected_components(G_in))
            largest_in = max(len(c) for c in in_cc) if in_cc else 0
            ap_c_in = 1.0 - (largest_in / (n - 1)) if n > 1 else 0.0

            ap_c_dir = max(ap_c_out, ap_c_in)

            # --- CDI (Connectivity Degradation Index) ---
            cdi_val = 0.0
            if baseline_avg_path > 0 and len(out_cc) > 0:
                # Removal may have fragmented the graph
                # If disconnected, CDI = 1.0 (worst case)
                if len(out_cc) > len(cc):
                    cdi_val = 1.0
                else:
                    try:
                        G_out_main = G_out.subgraph(max(out_cc, key=len)).copy()
                        if G_out_main.number_of_nodes() > 1:
                            # Use sampling if requested
                            if use_sampling:
                                # Simple approximation for average path length increase
                                # BFS from sample of nodes
                                lengths = []
                                for s in random_nodes:
                                    if s == nid or s not in G_out_main: continue
                                    d = nx.single_source_shortest_path_length(G_out_main, s)
                                    lengths.extend(d.values())
                                after_avg = sum(lengths) / len(lengths) if lengths else baseline_avg_path
                            else:
                                after_avg = nx.average_shortest_path_length(G_out_main)
                            
                            cdi_val = min(max(0.0, (after_avg - baseline_avg_path) / baseline_avg_path), 1.0)
                    except (nx.NetworkXError, ZeroDivisionError):
                        cdi_val = 0.0
            
            ap_scores[nid] = {"ap_c_dir": ap_c_dir, "cdi": cdi_val}
            cdi_raw[nid] = cdi_val

        return ap_scores

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