"""
saag/prediction/structural_predictor.py — Standalone Structural Centrality Predictors
====================================================================================

Promotes training-free topological baselines (Topo and Topo-QoS) into first-class,
reusable predictors in the Software-as-a-Graph framework.

Models
------
TopoPredictor:
    Classical structural centrality (0.6 * betweenness + 0.4 * articulation)
    evaluated on the Application–Library flow projection (G_flow). Requires no training.

TopoQoSPredictor:
    QoS-weighted structural centrality (0.6 * QoS-weighted betweenness + 0.4 * articulation)
    evaluated on the flow projection. NetworkX weights are interpreted as distance,
    so distance = 1 / (qos_weight + eps).

DualEnginePredictor:
    Executes learned relational forecasting (HGT-QoS) and closed-form structural
    centrality (Topo-QoS) concurrently. Identifies:
      1. Consensus Critical Set: Top-K intersection of both models (high confidence).
      2. Divergence Escalation Set: Components with large rank disagreements,
         triggering human architectural triage (operationalizing JSS Section 8.1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx

logger = logging.getLogger(__name__)


def derive_depends_on_edges(topology: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Derive DEPENDS_ON edges from pub-sub topology relationships.

    Implements Rules 1 and 5 from CLAUDE.md (app_to_app, app_to_lib) without
    Neo4j. These edges are required so that StructuralAnalyzer can compute
    meaningful betweenness, bridge_ratio, and reverse_pagerank for Application
    and Library nodes.

    Direction: dependent -> dependency (subscriber depends on publisher, app depends on lib).
    """
    rels = topology.get("relationships", {})

    # Collect topic -> {publishers}, topic -> {subscribers}
    topic_publishers: Dict[str, Set[str]] = {}
    for r in rels.get("publishes_to", []):
        src = r.get("source") or r.get("application_id") or r.get("from")
        dst = r.get("target") or r.get("topic_id") or r.get("to")
        if src and dst:
            topic_publishers.setdefault(str(dst), set()).add(str(src))

    topic_subscribers: Dict[str, Set[str]] = {}
    for r in rels.get("subscribes_to", []):
        src = r.get("source") or r.get("application_id") or r.get("from")
        dst = r.get("target") or r.get("topic_id") or r.get("to")
        if src and dst:
            topic_subscribers.setdefault(str(dst), set()).add(str(src))

    from saag.core.models import topic_weight_from_node_attrs, compute_effective_edge_weight
    import statistics as _stats

    topic_attrs: Dict[str, Dict] = {
        str(t["id"]): t for t in topology.get("topics", []) if t.get("id") is not None
    }
    topic_weight: Dict[str, float] = {}
    for tid, attrs in topic_attrs.items():
        existing = attrs.get("weight")
        if isinstance(existing, (int, float)) and existing > 0:
            topic_weight[tid] = float(existing)
        else:
            topic_weight[tid] = float(topic_weight_from_node_attrs(attrs))

    neutral_weight = _stats.median(topic_weight.values()) if topic_weight else 1.0

    # Per-relationship qos_profile when a topology does supply one; otherwise the
    # Topic's own qos block is the authority.
    pub_qos: Dict[Tuple[str, str], Dict] = {}
    for r in rels.get("publishes_to", []):
        src = r.get("source") or r.get("application_id") or r.get("from")
        dst = r.get("target") or r.get("topic_id") or r.get("to")
        if src and dst and "qos_profile" in r:
            pub_qos[(str(src), str(dst))] = r["qos_profile"]

    sub_qos: Dict[Tuple[str, str], Dict] = {}
    for r in rels.get("subscribes_to", []):
        src = r.get("source") or r.get("application_id") or r.get("from")
        dst = r.get("target") or r.get("topic_id") or r.get("to")
        if src and dst and "qos_profile" in r:
            sub_qos[(str(src), str(dst))] = r["qos_profile"]

    edges: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()

    # Rule 1 -- app_to_app: subscriber depends on publisher (via shared topic)
    pair_topics: Dict[Tuple[str, str], List[str]] = {}
    for topic_id, publishers in topic_publishers.items():
        for subscriber in topic_subscribers.get(topic_id, set()):
            for publisher in publishers:
                if subscriber == publisher:
                    continue
                pair_topics.setdefault((subscriber, publisher), []).append(topic_id)

    for (subscriber, publisher), topic_ids in pair_topics.items():
        seen.add((subscriber, publisher))
        weights = [topic_weight.get(tid, 1.0) for tid in topic_ids]
        via_topic = max(topic_ids, key=lambda tid: topic_weight.get(tid, 1.0))
        qp = (pub_qos.get((publisher, via_topic))
              or sub_qos.get((subscriber, via_topic))
              or (topic_attrs.get(via_topic, {}).get("qos") or {}))
        edges.append({
            "source": subscriber,
            "target": publisher,
            "type": "app_to_app",
            "weight": 1.0,
            "qos_weight": compute_effective_edge_weight(weights),
            "qos_profile": qp,
            "via_topic": via_topic,
            "path_count": len(topic_ids),
        })

    # Rule 5 -- app_to_lib: application depends on library (USES edge)
    for r in rels.get("uses", []):
        src = r.get("source") or r.get("application_id") or r.get("from")
        dst = r.get("target") or r.get("library_id") or r.get("to")
        if src and dst:
            key = (str(src), str(dst))
            if key not in seen:
                seen.add(key)
                qp = r.get("qos_profile") or {}
                edges.append({
                    "source": str(src),
                    "target": str(dst),
                    "type": "app_to_lib",
                    "weight": 1.0,
                    "qos_weight": neutral_weight,
                    "qos_profile": qp,
                    "path_count": 1,
                })

    return edges


def derive_flow_projection(
    graph_or_topology: Union[nx.DiGraph, nx.MultiDiGraph, Dict[str, Any]],
) -> nx.DiGraph:
    """Derive the Application--Library DEPENDS_ON projection graph (G_flow).

    In raw publish--subscribe multigraphs, Application nodes never route messages
    directly, causing their betweenness centrality on raw topologies to be near-zero.
    This projection derives direct logical dependencies:
      - Rule 1 (app_to_app): subscriber -> publisher mediated by shared topic.
      - Rule 5 (app_to_lib): application -> library via USES relationship.

    Accepts either a scenario topology dictionary or a NetworkX multigraph.
    """
    if isinstance(graph_or_topology, dict):
        return _derive_from_topology_dict(graph_or_topology)
    elif isinstance(graph_or_topology, (nx.DiGraph, nx.MultiDiGraph)):
        return _derive_from_nx_graph(graph_or_topology)
    else:
        raise TypeError(f"Expected dict or nx.Graph, got {type(graph_or_topology)}")


def _derive_from_topology_dict(topology: Dict[str, Any]) -> nx.DiGraph:
    """Build G_flow from a scenario JSON/YAML topology structure."""
    deps = derive_depends_on_edges(topology)
    app_ids = {str(a["id"]) for a in topology.get("applications", []) if a.get("id") is not None}
    lib_ids = {str(lb["id"]) for lb in topology.get("libraries", []) if lb.get("id") is not None}
    allowed = app_ids | lib_ids

    dep_graph = nx.DiGraph()
    for nid in app_ids:
        dep_graph.add_node(nid, type="Application")
    for nid in lib_ids:
        dep_graph.add_node(nid, type="Library")

    for e in deps:
        src, dst = str(e["source"]), str(e["target"])
        if src in allowed and dst in allowed:
            dep_graph.add_edge(
                src,
                dst,
                weight=float(e.get("weight", 1.0)),
                qos_weight=float(e.get("qos_weight", 1.0)),
                type="DEPENDS_ON",
                dependency_type=e.get("type", "app_to_app"),
                path_count=e.get("path_count", 1),
            )

    return dep_graph


def _derive_from_nx_graph(graph: Union[nx.DiGraph, nx.MultiDiGraph]) -> nx.DiGraph:
    """Extract G_flow from an existing NetworkX graph containing pub/sub/uses edges."""
    from saag.core.models import compute_effective_edge_weight

    dep_graph = nx.DiGraph()
    for nid, d in graph.nodes(data=True):
        nt = d.get("type", "Application")
        if nt in ("Application", "Library"):
            dep_graph.add_node(nid, **d)

    # Check if DEPENDS_ON edges already exist
    existing_depends = [
        (u, v, d)
        for u, v, d in graph.edges(data=True)
        if d.get("type") in ("DEPENDS_ON", "app_to_app", "app_to_lib")
    ]
    if existing_depends:
        for u, v, d in existing_depends:
            if u in dep_graph and v in dep_graph:
                dep_graph.add_edge(u, v, **d)
        return dep_graph

    has_pubsub = any(
        d.get("type") in ("PUBLISHES_TO", "SUBSCRIBES_TO")
        for _, _, d in graph.edges(data=True)
    )
    if not has_pubsub and graph.number_of_edges() > 0:
        # Pre-derived dependency or custom graph
        for u, v, d in graph.edges(data=True):
            if u in dep_graph and v in dep_graph:
                dep_graph.add_edge(u, v, **d)
        return dep_graph

    # Otherwise derive from PUBLISHES_TO / SUBSCRIBES_TO / USES
    topic_publishers: Dict[str, Set[str]] = {}
    topic_subscribers: Dict[str, Set[str]] = {}
    topic_weights: Dict[str, float] = {}

    for u, v, d in graph.edges(data=True):
        etype = d.get("type")
        if etype == "PUBLISHES_TO":
            topic_publishers.setdefault(str(v), set()).add(str(u))
            topic_weights[str(v)] = float(d.get("qos_weight", d.get("weight", 1.0)))
        elif etype == "SUBSCRIBES_TO":
            topic_subscribers.setdefault(str(v), set()).add(str(u))
        elif etype == "USES":
            if u in dep_graph and v in dep_graph:
                dep_graph.add_edge(
                    str(u),
                    str(v),
                    weight=1.0,
                    qos_weight=float(d.get("qos_weight", d.get("weight", 1.0))),
                    type="DEPENDS_ON",
                    dependency_type="app_to_lib",
                )

    for tid, publishers in topic_publishers.items():
        w = topic_weights.get(tid, 1.0)
        for sub in topic_subscribers.get(tid, set()):
            for pub in publishers:
                if sub == pub or sub not in dep_graph or pub not in dep_graph:
                    continue
                if dep_graph.has_edge(sub, pub):
                    prev_w = dep_graph[sub][pub].get("qos_weight", 1.0)
                    dep_graph[sub][pub]["qos_weight"] = float(
                        compute_effective_edge_weight([prev_w, w])
                    )
                else:
                    dep_graph.add_edge(
                        sub,
                        pub,
                        weight=1.0,
                        qos_weight=w,
                        type="DEPENDS_ON",
                        dependency_type="app_to_app",
                    )

    return dep_graph


def qos_weighted_betweenness(flow_graph: nx.DiGraph, eps: float = 1e-6) -> Dict[str, float]:
    """Compute betweenness on a graph where edge distance = 1 / (qos_weight + eps)."""
    dist_g = nx.DiGraph()
    dist_g.add_nodes_from(flow_graph.nodes(data=True))
    n_qos_edges = 0
    for u, v, data in flow_graph.edges(data=True):
        w = float(data.get("qos_weight", data.get("weight", 1.0)))
        if abs(w - 1.0) > 1e-9:
            n_qos_edges += 1
        dist_g.add_edge(u, v, distance=1.0 / (w + eps))
    if n_qos_edges == 0:
        return {}
    bc = nx.betweenness_centrality(dist_g, weight="distance")
    return {str(k): float(v) for k, v in bc.items()}


class TopoPredictor:
    """Unweighted topological baseline: 0.6 * Betweenness + 0.4 * Articulation Score."""

    def __init__(self, bt_weight: float = 0.6, ap_weight: float = 0.4):
        self.bt_weight = bt_weight
        self.ap_weight = ap_weight

    def predict(
        self,
        graph_or_flow: Union[nx.DiGraph, nx.MultiDiGraph, Dict[str, Any]],
        structural_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """Compute unweighted topological criticality scores."""
        flow_graph = derive_flow_projection(graph_or_flow)
        if flow_graph.number_of_nodes() == 0:
            return {}

        ap = {
            str(nid): float(m.get("articulation_point", m.get("ap_c_score", m.get("ap_c_directed", 0.0))))
            for nid, m in (structural_metrics or {}).items()
        }

        if structural_metrics and any("betweenness" in m or "betweenness_centrality" in m for m in structural_metrics.values()):
            bt = {
                str(nid): float(m.get("betweenness", m.get("betweenness_centrality", 0.0)))
                for nid, m in structural_metrics.items()
            }
        else:
            bt = {str(n): float(v) for n, v in nx.betweenness_centrality(flow_graph).items()}

        nodes = set(bt) | set(ap)
        return {
            nid: self.bt_weight * bt.get(nid, 0.0) + self.ap_weight * ap.get(nid, 0.0)
            for nid in nodes
        }


class TopoQoSPredictor:
    """QoS-weighted topological baseline with distance inversion."""

    def __init__(
        self,
        bt_weight: float = 0.6,
        ap_weight: float = 0.4,
        eps: float = 1e-6,
    ):
        self.bt_weight = bt_weight
        self.ap_weight = ap_weight
        self.eps = eps

    def predict(
        self,
        graph_or_flow: Union[nx.DiGraph, nx.MultiDiGraph, Dict[str, Any]],
        structural_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """Compute QoS-weighted topological criticality scores."""
        flow_graph = derive_flow_projection(graph_or_flow)
        if flow_graph.number_of_nodes() == 0:
            return {}

        ap = {
            str(nid): float(m.get("articulation_point", m.get("ap_c_score", m.get("ap_c_directed", 0.0))))
            for nid, m in (structural_metrics or {}).items()
        }

        bt = qos_weighted_betweenness(flow_graph, eps=self.eps)
        if not bt:
            logger.warning(
                "Topo-QoS: no QoS weights on graph; falling back to "
                "topology betweenness (Topo-QoS equivalent to Topo for this cell)."
            )
            bt = {str(n): float(v) for n, v in nx.betweenness_centrality(flow_graph).items()}

        nodes = set(bt) | set(ap)
        return {
            nid: self.bt_weight * bt.get(nid, 0.0) + self.ap_weight * ap.get(nid, 0.0)
            for nid in nodes
        }


@dataclass
class DualEngineResult:
    """Result of concurrent dual-engine prediction (HGT-QoS + Topo-QoS)."""

    gnn_scores: Dict[str, float]
    topo_scores: Dict[str, float]
    combined_scores: Dict[str, float]
    gnn_ranks: Dict[str, int]
    topo_ranks: Dict[str, int]
    rank_divergences: Dict[str, int]
    consensus_top_k: List[str]
    divergence_escalations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class DualEnginePredictor:
    """Operationalizes JSS Section 8.1: runs HGT-QoS and Topo-QoS concurrently.

    Extracts:
      - Consensus Critical Set: components appearing in the top-K of BOTH models.
      - Divergence Escalation Set: components where |rank_gnn - rank_topo| > divergence_threshold.
    """

    def __init__(
        self,
        topo_predictor: Optional[TopoQoSPredictor] = None,
        divergence_threshold: Optional[int] = None,
        divergence_fraction: float = 0.25,
    ):
        self.topo_predictor = topo_predictor or TopoQoSPredictor()
        self.divergence_threshold = divergence_threshold
        self.divergence_fraction = divergence_fraction

    def evaluate_dual(
        self,
        gnn_scores: Dict[str, float],
        graph_or_flow: Union[nx.DiGraph, nx.MultiDiGraph, Dict[str, Any]],
        structural_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
        k: Optional[int] = None,
        top_k_fraction: float = 0.20,
    ) -> DualEngineResult:
        """Combine GNN predictions with Topo-QoS scores into consensus and divergence triage.
        
        Evaluates rank divergence strictly over components scored by both models to
        avoid fabricating spurious divergences on nodes missing from either input.
        """
        import math

        topo_scores = self.topo_predictor.predict(graph_or_flow, structural_metrics)

        # Restrict rank divergence to the intersection of components both engines scored
        common_nodes = sorted(set(gnn_scores) & set(topo_scores))
        if not common_nodes:
            return DualEngineResult({}, {}, {}, {}, {}, {}, [], [])

        # Determine effective K (critical set size) and divergence threshold
        eff_k = k if k is not None else max(1, int(math.ceil(top_k_fraction * len(common_nodes))))
        eff_k = min(eff_k, len(common_nodes))

        if self.divergence_threshold is not None:
            eff_div = self.divergence_threshold
        else:
            eff_div = max(1, int(math.ceil(self.divergence_fraction * len(common_nodes))))

        # Normalization over common nodes
        max_g = max((gnn_scores[n] for n in common_nodes), default=1.0)
        max_t = max((topo_scores[n] for n in common_nodes), default=1.0)
        norm_g = {n: (gnn_scores[n] / max_g) if max_g > 0 else 0.0 for n in common_nodes}
        norm_t = {n: (topo_scores[n] / max_t) if max_t > 0 else 0.0 for n in common_nodes}

        # Rankings (1-based, 1 is most critical)
        sorted_g = sorted(common_nodes, key=lambda n: norm_g[n], reverse=True)
        sorted_t = sorted(common_nodes, key=lambda n: norm_t[n], reverse=True)
        gnn_ranks = {n: i + 1 for i, n in enumerate(sorted_g)}
        topo_ranks = {n: i + 1 for i, n in enumerate(sorted_t)}

        # Rank divergences over common nodes
        rank_divs = {n: abs(gnn_ranks[n] - topo_ranks[n]) for n in common_nodes}

        # Consensus Top-K (intersection of top-k sets)
        top_k_g = set(sorted_g[:eff_k])
        top_k_t = set(sorted_t[:eff_k])
        consensus = [n for n in sorted_g[:eff_k] if n in top_k_t]

        # Divergence escalations
        escalations = [
            n for n in common_nodes
            if rank_divs[n] >= eff_div and (n in top_k_g or n in top_k_t)
        ]
        escalations.sort(key=lambda n: rank_divs[n], reverse=True)

        combined = {n: 0.5 * norm_g[n] + 0.5 * norm_t[n] for n in common_nodes}

        # Track any nodes present in only one engine's output
        unscored_by_topo = sorted(set(gnn_scores) - set(topo_scores))
        unscored_by_gnn = sorted(set(topo_scores) - set(gnn_scores))

        return DualEngineResult(
            gnn_scores=gnn_scores,
            topo_scores=topo_scores,
            combined_scores=combined,
            gnn_ranks=gnn_ranks,
            topo_ranks=topo_ranks,
            rank_divergences=rank_divs,
            consensus_top_k=consensus,
            divergence_escalations=escalations,
            metadata={
                "k": eff_k,
                "divergence_threshold": eff_div,
                "n_common": len(common_nodes),
                "unscored_by_topo": unscored_by_topo,
                "unscored_by_gnn": unscored_by_gnn,
            },
        )

