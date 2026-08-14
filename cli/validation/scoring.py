"""Q(v) scoring for the validation CLI: the RM baseline and the GNN predictor.

Both predictors need the same structural primitives from the graph, so those are
extracted once by `extract_primitives` and then shaped differently for each
consumer.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class NodeScores:
    """Per-node Q(v) prediction and I(v) simulation ground-truth."""
    node_id: str
    node_type: str                # Application | Broker | Topic | InfraNode | Library

    # RM dimensions (topology-only baseline when qos=False). FT and A are
    # Reliability's sub-characteristics, reported alongside R.
    R: float = 0.0                # Reliability exposure (hierarchical: FT + A)
    M: float = 0.0                # Maintainability proxy (betweenness + closeness)
    FT: float = 0.0               # Fault tolerance       (PageRank + in-degree)
    A: float = 0.0                # Availability risk     (articulation × QoS_SPOF)
    Q: float = 0.0                # Composite Q(v)

    # Simulation ground truth
    I: float = 0.0                # Normalised cascade impact score I(v) ∈ [0,1]
    cascade_depth: int = 0        # Max depth reached in cascade
    nodes_affected: int = 0       # Distinct nodes rendered unreachable

    # Structural flags
    is_articulation_point: bool = False
    degree_centrality: float = 0.0   # Baseline comparator


@dataclass
class StructuralPrimitives:
    """Graph-derived quantities shared by the RM and GNN scorers."""
    pagerank: Dict[str, float]
    reverse_pagerank: Dict[str, float]
    betweenness: Dict[str, float]
    articulation_points: set
    in_degree: Dict[str, int]
    out_degree: Dict[str, int]
    degree_centrality_physical: Dict[str, float]
    publisher_spof: Dict[str, float]
    max_betweenness: float

    def mpci(self, node_id: str) -> float:
        """Betweenness normalised by the graph maximum."""
        return self.betweenness.get(node_id, 0.0) / (self.max_betweenness + 1e-12)


def extract_primitives(G: nx.DiGraph, qos: bool = True) -> StructuralPrimitives:
    """Compute every structural quantity the scorers need, once."""
    pagerank = nx.pagerank(G, alpha=0.85)
    reverse_pagerank = nx.pagerank(G.reverse(), alpha=0.85)
    betweenness = nx.betweenness_centrality(G, normalized=True)

    # Articulation points are taken over the Application-only DEPENDS_ON graph:
    # a cut vertex there is a genuine single point of failure for the app mesh.
    try:
        g_depends = G.copy()
        g_depends.remove_edges_from(
            [(u, v) for u, v, d in g_depends.edges(data=True) if d.get("etype") != "DEPENDS_ON"]
        )
        g_depends.remove_nodes_from(
            [n for n, d in g_depends.nodes(data=True) if d.get("ntype") != "Application"]
        )
        articulation_points = set(nx.articulation_points(g_depends.to_undirected()))
    except (nx.NetworkXError, nx.NetworkXNotImplemented) as e:
        logger.warning("Articulation-point detection failed: %s", e)
        articulation_points = set()

    # Degree centrality on the physical graph only (DEPENDS_ON edges are derived,
    # so including them would double-count the same relationships).
    try:
        g_physical = G.copy()
        g_physical.remove_edges_from(
            [(u, v) for u, v, d in g_physical.edges(data=True) if d.get("etype") == "DEPENDS_ON"]
        )
        degree_centrality_physical = nx.degree_centrality(g_physical)
    except (nx.NetworkXError, nx.NetworkXNotImplemented) as e:
        logger.warning("Degree-centrality computation failed: %s", e)
        degree_centrality_physical = {}

    pub_count: Dict[str, int] = defaultdict(int)
    sub_count: Dict[str, int] = defaultdict(int)
    for _, v, d in G.edges(data=True):
        etype = d.get("etype", "")
        if etype == "PUBLISHES_TO":
            pub_count[v] += 1
        elif etype == "SUBSCRIBES_TO":
            sub_count[v] += 1

    # A sole publisher to a topic that has subscribers is a publisher SPOF.
    publisher_spof: Dict[str, float] = defaultdict(float)
    if qos:
        for u, topic, d in G.edges(data=True):
            if d.get("etype") != "PUBLISHES_TO":
                continue
            if pub_count[topic] == 1 and sub_count[topic] >= 1:
                publisher_spof[u] = max(
                    publisher_spof[u], 0.5 * min(sub_count[topic] / 5.0, 1.0)
                )

    return StructuralPrimitives(
        pagerank=pagerank,
        reverse_pagerank=reverse_pagerank,
        betweenness=betweenness,
        articulation_points=articulation_points,
        in_degree=dict(G.in_degree()),
        out_degree=dict(G.out_degree()),
        degree_centrality_physical=degree_centrality_physical,
        publisher_spof=publisher_spof,
        max_betweenness=max(betweenness.values()) if betweenness else 1.0,
    )


def compute_rm(G: nx.DiGraph, qos: bool = True, normalization: str = "robust") -> Dict[str, NodeScores]:
    """Compute RM for ALL nodes using the central QualityAnalyzer."""
    from saag.analysis.analyzer import QualityAnalyzer
    from saag.core.metrics import StructuralMetrics

    p = extract_primitives(G, qos=qos)

    all_metrics = [
        StructuralMetrics(
            id=nid,
            name=ndata.get("label", nid),
            type=ndata.get("ntype", "Application"),
            pagerank=p.pagerank.get(nid, 0.0),
            reverse_pagerank=p.reverse_pagerank.get(nid, 0.0),
            betweenness=p.betweenness.get(nid, 0.0),
            in_degree_raw=p.in_degree.get(nid, 0),
            out_degree_raw=p.out_degree.get(nid, 0),
            is_articulation_point=(nid in p.articulation_points),
            weight=0.5,
            publisher_spof=p.publisher_spof.get(nid, 0.0),
            mpci=p.mpci(nid),
            ap_c_directed=1.0 if nid in p.articulation_points else 0.0,
        )
        for nid, ndata in G.nodes(data=True)
    ]

    analyzer = QualityAnalyzer(normalization_method=normalization)
    norm_factors = analyzer._normalize(all_metrics)
    qualities = analyzer._score_and_classify_components(all_metrics, norm_factors)

    return {
        q.id: NodeScores(
            node_id=q.id,
            node_type=q.type,
            Q=q.scores.overall,
            R=q.scores.reliability,
            M=q.scores.maintainability,
            FT=q.scores.fault_tolerance,
            A=q.scores.availability,
            I=0.0,
            degree_centrality=p.degree_centrality_physical.get(q.id, 0.0),
            is_articulation_point=(q.id in p.articulation_points),
        )
        for q in qualities
    }


def compute_gnn_scores(G: nx.DiGraph, gnn_model: str, qos: bool = True) -> Dict[str, NodeScores]:
    """Score every node with a trained GNN checkpoint instead of the RM formula."""
    from saag.prediction.gnn_service import GNNService

    p = extract_primitives(G, qos=qos)

    # The GNN consumes the RM scores as input features.
    rm_scores = {
        nid: {
            "overall": ns.Q,
            "reliability": ns.R,
            "maintainability": ns.M,
            "availability": ns.A,
        }
        for nid, ns in compute_rm(G, qos=qos).items()
    }

    structural_metrics = {
        nid: {
            "pagerank": p.pagerank.get(nid, 0.0),
            "reverse_pagerank": p.reverse_pagerank.get(nid, 0.0),
            "betweenness_centrality": p.betweenness.get(nid, 0.0),
            "closeness_centrality": 0.0,
            "eigenvector_centrality": 0.0,
            "in_degree_centrality": p.in_degree.get(nid, 0),
            "out_degree_centrality": p.out_degree.get(nid, 0),
            "clustering_coefficient": 0.0,
            "ap_c_score": 1.0 if nid in p.articulation_points else 0.0,
            "ap_c_directed": 1.0 if nid in p.articulation_points else 0.0,
            "cdi": 0.0,
            "mpci": p.mpci(nid),
            "path_complexity": 0.0,
            "fan_out_criticality": 0.0,
            "bridge_ratio": 0.0,
            "qos_weight": 0.5,
            "qos_weight_in": 0.0,
            "qos_weight_out": p.publisher_spof.get(nid, 0.0),
        }
        for nid in G.nodes()
    }

    service = GNNService.from_checkpoint(gnn_model, graph=G)
    gnn_result = service.predict(
        graph=G,
        structural_metrics=structural_metrics,
        rm_scores=rm_scores,
        mode="gnn",
        qos_enabled=qos,
    )

    return {
        nid: NodeScores(
            node_id=nid,
            node_type=G.nodes[nid].get("ntype", "Application") if nid in G.nodes else "Application",
            Q=score.composite_score,
            R=score.reliability_score,
            M=score.maintainability_score,
            A=score.availability_score,
            I=0.0,
            degree_centrality=p.degree_centrality_physical.get(nid, 0.0),
            is_articulation_point=(nid in p.articulation_points),
        )
        for nid, score in gnn_result.node_scores.items()
    }
