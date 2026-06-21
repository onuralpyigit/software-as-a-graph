#!/usr/bin/env python3
"""
validate_graph.py — SaG Statistical Validation CLI
====================================================
Statistically proves that topology-based Q(v) predictions agree with
simulation-derived cascade impact I(v) as proxy ground truth.

Pipeline
--------
1. Load graph (JSON / Neo4j) and compute Q(v) scores
2. Derive simulation ground truth I(v) via cascade failure simulation
3. Run full statistical battery:
   • Spearman ρ  (primary gate ≥ 0.80)
   • Kendall τ   (robustness cross-check)
   • Wilcoxon signed-rank vs. degree-centrality baseline
   • Bootstrap 95% CI on ρ  (B = 2000 resamples)
4. Compute specialist metrics:
   • ICR@K  — In-Cluster Recall at K
   • RCR    — Rank Consistency Rate across seeds
   • BCE    — Binary Classification Error on top-K
   • SPOF-F1 — Articulation-point detection F1
   • FTR    — False Top Rate (critical false positives)
   • PG     — Predictive Gain over degree-centrality baseline
5. Node-type stratified reporting (Application, Broker, Topic, Infra, Library)
6. Topology-class gate evaluation (sparse / medium / dense / hub-spoke)
7. Multi-seed stability sweep  (seeds: 42, 123, 456, 789, 2024)

Usage
-----
# Single run — topology-only baseline
python cli/validate_graph.py single --input data/system.json

# Single run — QoS-enriched
python cli/validate_graph.py single --input data/system.json --qos

# Multi-seed stability sweep
python cli/validate_graph.py sweep --input data/system.json --qos

# Full report (sweep + topology-class gates + node-type strata)
python cli/validate_graph.py report --input data/system.json \\
    --output output/validation_report.json --qos

# Run against ATM dataset with custom top-k
python cli/validate_graph.py report --input datasets/atm_system.json \
    --top-k 10 --qos --output output/atm_validation.json

# Methodological-guard harness on pre-computed JSON artifacts
python cli/validate_graph.py harness \
    --predictions output/predictions.json \
    --ground-truth cascade=output/impact_scores.json \
    --ground-truth latency=output/latency_delta.json \
    --out output/harness_report.json

Options
-------
--input     PATH    Graph JSON (system.json format) or Neo4j bolt URI
--qos               Enable QoS-weighted RMAV scoring (default: off = topology-only)
--top-k     INT     K for ICR@K, BCE, FTR classification metrics (default: 20% of nodes)
--seeds     INTS    Comma-separated seed list (default: 42,123,456,789,2024)
--cascade   INT     Cascade simulation depth limit (default: 5)
--bootstrap INT     Bootstrap resamples for CI (default: 2000)
--alpha     FLOAT   Significance level for Wilcoxon test (default: 0.05)
--output    PATH    Write JSON report to PATH
--csv               Also write CSV per-node table
--verbose           Print per-node scores during run
--no-color          Disable ANSI colours in console output
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

# Add project root to sys.path to support direct execution (python cli/validate_graph.py)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

from cli.common.arguments import setup_logging

import networkx as nx
import numpy as np
from scipy import stats
from scipy.stats import kendalltau, spearmanr

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (int, float, bool)): return obj
        import numpy as np
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        return super(NpEncoder, self).default(obj)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NodeScores:
    """Per-node Q(v) prediction and I(v) simulation ground-truth."""
    node_id: str
    node_type: str                # Application | Broker | Topic | InfraNode | Library

    # RMAS dimensions (topology-only baseline when qos=False)
    R: float = 0.0                # Reliability exposure  (PageRank + in-degree)
    M: float = 0.0                # Maintainability proxy (betweenness + closeness)
    A: float = 0.0                # Availability risk     (articulation × QoS_SPOF)
    S: float = 0.0                # Security              (out-degree + dep-density)
    Q: float = 0.0                # Composite Q(v)

    # Simulation ground truth
    I: float = 0.0                # Normalised cascade impact score I(v) ∈ [0,1]
    cascade_depth: int = 0        # Max depth reached in cascade
    nodes_affected: int = 0       # Distinct nodes rendered unreachable

    # Structural flags
    is_articulation_point: bool = False
    degree_centrality: float = 0.0   # Baseline comparator


@dataclass
class ValidationResult:
    """Full statistical report for one (graph, seed, qos_mode) triple."""
    seed: int
    qos_enabled: bool
    n_nodes: int
    n_app_nodes: int

    # ── rank correlation ───────────────────────────────────────────────────────
    spearman_rho: float = 0.0
    spearman_p: float = 1.0
    kendall_tau: float = 0.0
    kendall_p: float = 1.0
    bootstrap_ci_lo: float = 0.0
    bootstrap_ci_hi: float = 0.0

    # ── classification ─────────────────────────────────────────────────────────
    top_k: int = 0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    f1_at_k: float = 0.0
    spof_f1: float = 0.0
    ftr: float = 0.0              # False Top Rate

    # ── specialist metrics ─────────────────────────────────────────────────────
    icr_at_k: float = 0.0        # In-Cluster Recall @K
    bce: float = 0.0              # Binary Classification Error
    pg: float = 0.0               # Predictive Gain over degree-centrality

    # ── Wilcoxon vs degree baseline ────────────────────────────────────────────
    wilcoxon_stat: float = 0.0
    wilcoxon_p: float = 1.0
    wilcoxon_significant: bool = False

    # ── node-type strata ───────────────────────────────────────────────────────
    strata: Dict[str, Dict] = field(default_factory=dict)

    # ── gate evaluation ────────────────────────────────────────────────────────
    gates_passed: Dict[str, bool] = field(default_factory=dict)
    overall_pass: bool = False


@dataclass
class SweepReport:
    """Aggregate over the multi-seed sweep."""
    qos_enabled: bool
    seeds: List[int]
    rho_mean: float
    rho_std: float
    rho_min: float
    rho_max: float
    f1_mean: float
    pg_mean: float
    rcr: float                    # Rank Consistency Rate = 1 − (mean Kendall distance)
    all_gates_pass_rate: float    # Fraction of seeds that pass all gates
    per_seed: List[ValidationResult] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_graph(path: str) -> Tuple[nx.DiGraph, dict]:
    """
    Load a SaG system.json / dataset.json and build a typed DiGraph.

    Returns (G, raw_data).
    G nodes carry attribute 'ntype' ∈ {Application, Broker, Topic,
    InfraNode, Library} and 'label'.
    """
    raw = json.loads(Path(path).read_text())
    G = nx.DiGraph()

    # ── nodes ──────────────────────────────────────────────────────────────────
    def _add(collection, ntype):
        for item in raw.get(collection, []):
            nid = item.get("id") or item.get("name")
            model_type = "Node" if ntype == "InfraNode" else ntype
            G.add_node(nid, ntype=ntype, type=model_type, label=item.get("name", nid), raw=item)

    _add("applications", "Application")
    _add("brokers",      "Broker")
    _add("topics",       "Topic")
    _add("nodes",        "InfraNode")
    _add("libraries",    "Library")

    # ── edges ──────────────────────────────────────────────────────────────────
    rels = raw.get("relationships", {})

    def _edges(key, src_field, tgt_field, etype):
        # Support both root-level and relationships-level keys
        # Support both singular and plural (publishes vs publishes_to)
        items = raw.get(key, []) + rels.get(key, [])
        if not items and "_" not in key:
            items += raw.get(f"{key}_to", []) + rels.get(f"{key}_to", [])
        
        for e in items:
            s = e.get(src_field) or e.get("from") or e.get("app") or e.get("src")
            t = e.get(tgt_field) or e.get("to") or e.get("topic") or e.get("tgt")
            if s and t and G.has_node(s) and G.has_node(t):
                G.add_edge(s, t, etype=etype, type=etype)

    _edges("publishes",    "app", "topic", "PUBLISHES_TO")
    _edges("subscribes",   "app", "topic", "SUBSCRIBES_TO")
    _edges("routes",       "from", "to",   "ROUTES")
    _edges("runs_on",      "from", "to",   "RUNS_ON")
    # Legacy support
    _edges("publish_edges",   "app", "topic", "PUBLISHES_TO")
    _edges("subscribe_edges", "app", "topic", "SUBSCRIBES_TO")
    _edges("broker_connections", "broker", "topic", "ROUTES")

    # ── derive DEPENDS_ON (app_to_app via shared topics) ──────────────────────
    pub_map: Dict[str, List[str]] = defaultdict(list)   # topic → publishers
    sub_map: Dict[str, List[str]] = defaultdict(list)   # topic → subscribers

    for u, v, d in G.edges(data=True):
        if d["etype"] == "PUBLISHES_TO":
            pub_map[v].append(u)
        elif d["etype"] == "SUBSCRIBES_TO":
            sub_map[v].append(u)

    for topic, pubs in pub_map.items():
        for sub in sub_map.get(topic, []):
            for pub in pubs:
                if pub != sub and not G.has_edge(sub, pub):
                    G.add_edge(sub, pub, etype="DEPENDS_ON", type="DEPENDS_ON", via=topic)

    return G, raw


# ═══════════════════════════════════════════════════════════════════════════════
# RMAV SCORING  (topology-only baseline; QoS enrichment is additive)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rmav(G: nx.DiGraph, qos: bool = True, normalization: str = "robust") -> Dict[str, NodeScores]:
    """
    Compute RMAV for ALL nodes using central QualityAnalyzer.
    """
    from saag.prediction.analyzer import QualityAnalyzer
    from saag.core.metrics import StructuralMetrics
    
    # 1. Primitives
    pagerank = nx.pagerank(G, alpha=0.85)
    reverse_pagerank = nx.pagerank(G.reverse(), alpha=0.85)
    betweenness = nx.betweenness_centrality(G, normalized=True)
    try:
        G_depends = G.copy()
        other_edges = [(u, v) for u, v, d in G_depends.edges(data=True) if d.get("etype") != "DEPENDS_ON"]
        G_depends.remove_edges_from(other_edges)
        non_apps = [n for n, d in G_depends.nodes(data=True) if d.get("ntype") != "Application"]
        G_depends.remove_nodes_from(non_apps)
        art_points = set(nx.articulation_points(G_depends.to_undirected()))
    except:
        art_points = set()
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    
    # Calculate degree centrality on physical G (without DEPENDS_ON edges)
    try:
        G_phys = G.copy()
        dep_edges = [(u, v) for u, v, d in G_phys.edges(data=True) if d.get("etype") == "DEPENDS_ON"]
        G_phys.remove_edges_from(dep_edges)
        deg_cent_phys = nx.degree_centrality(G_phys)
    except Exception:
        deg_cent_phys = {}
    
    pub_count = defaultdict(int)
    sub_count = defaultdict(int)
    for u, v, d in G.edges(data=True):
        et = d.get("etype", "")
        if et == "PUBLISHES_TO":
            pub_count[v] += 1
        elif et == "SUBSCRIBES_TO":
            sub_count[v] += 1
            
    pspof_vals = defaultdict(float)
    for u, topic, d in G.edges(data=True):
        if d.get("etype") == "PUBLISHES_TO":
            if pub_count[topic] == 1 and sub_count[topic] >= 1:
                pspof_vals[u] = max(pspof_vals[u], 0.5 * min(sub_count[topic] / 5.0, 1.0))

    # 2. Map
    all_metrics = []
    for nid, ndata in G.nodes(data=True):
        max_bc = max(betweenness.values()) if betweenness.values() else 1.0
        node_mpci = betweenness.get(nid, 0.0) / (max_bc + 1e-12)
        all_metrics.append(StructuralMetrics(
            id=nid,
            name=ndata.get("label", nid),
            type=ndata.get("ntype", "Application"),
            pagerank=pagerank.get(nid, 0.0),
            reverse_pagerank=reverse_pagerank.get(nid, 0.0),
            betweenness=betweenness.get(nid, 0.0),
            in_degree_raw=in_deg.get(nid, 0),
            out_degree_raw=out_deg.get(nid, 0),
            is_articulation_point=(nid in art_points),
            weight=0.5,
            publisher_spof=pspof_vals.get(nid, 0.0) if qos else 0.0,
            mpci=node_mpci,
            ap_c_directed=1.0 if nid in art_points else 0.0 
        ))
        
    # 3. Analyze
    analyzer = QualityAnalyzer(normalization_method=normalization)
    norm_factors = analyzer._normalize(all_metrics)
    qualities = analyzer._score_and_classify_components(all_metrics, norm_factors)
    
    # 4. Return
    results = {}
    for q in qualities:
        results[q.id] = NodeScores(
            node_id=q.id,
            node_type=q.type,
            Q=q.scores.overall,
            R=q.scores.reliability,
            M=q.scores.maintainability,
            A=q.scores.availability,
            S=q.scores.security,
            I=0.0,
            degree_centrality=deg_cent_phys.get(q.id, 0.0),
            is_articulation_point=(q.id in art_points)
        )
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CASCADE SIMULATION  — produces proxy ground truth I(v)
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_cascade(G: nx.DiGraph, origin: str, depth_limit: int = 5, seed: int = 42) -> Tuple[float, int, int]:
    """
    LEGACY WRAPPER: Now uses central FaultInjector for consistency.
    """
    from saag.simulation.fault_injector import FaultInjector
    injector = FaultInjector(graph=G, seeds=[seed], cascade_depth_limit=depth_limit)
    # _inject_node is a private helper that runs a single node injection
    rec = injector._inject_node(origin)
    return rec.impact_score, rec.cascade_depth, rec.total_impacted_subscribers


def derive_ground_truth(
    G: nx.DiGraph,
    scores: Dict[str, NodeScores],
    depth_limit: int = 5,
    seed: int = 42,
    n_repeats: int = 5,
) -> Dict[str, NodeScores]:
    """
    Run cascade simulation for every node and record I(v).

    Uses `n_repeats` stochastic runs per node; I(v) = mean impact.
    """
    rng_seeds = [seed + i * 37 for i in range(n_repeats)]
    n = G.number_of_nodes()

    for v, ns in scores.items():
        impacts = []
        for s in rng_seeds:
            imp, depth, affected = simulate_cascade(G, v, depth_limit, seed=s)
            impacts.append(imp)
        ns.I = float(np.mean(impacts))
        ns.cascade_depth = depth
        ns.nodes_affected = affected
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def _bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray, B: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    """Non-parametric bootstrap CI for Spearman ρ."""
    n = len(x)
    rhos = []
    rng = np.random.default_rng(42)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        xi, yi = x[idx], y[idx]
        if np.std(xi) == 0 or np.std(yi) == 0:
            continue
        r, _ = stats.spearmanr(xi, yi)
        rhos.append(r)
    if not rhos:
        return 0.0, 0.0
    rhos = np.array(rhos)
    lo = float(np.percentile(rhos, 100 * alpha / 2))
    hi = float(np.percentile(rhos, 100 * (1 - alpha / 2)))
    return lo, hi


def run_statistical_tests(
    node_scores: Dict[str, NodeScores],
    top_k: int,
    B: int = 2000,
    alpha: float = 0.05,
    primary_type: str = "Application",
) -> dict:
    """
    Core statistical battery.

    Primary rank-correlation metrics (Spearman ρ, Kendall τ, PG, Wilcoxon)
    are computed on ``primary_type`` nodes only (default: Application).
    Classification metrics (F1, SPOF-F1, FTR, ICR, BCE) use all nodes but
    restrict the candidate pool to primary_type as well.

    This matches the RMAV thesis claim: topology predicts *application-layer*
    cascade criticality — not generic structural centrality of topics/brokers.
    """
    # Primary-type subset for rank correlation
    primary = [ns for ns in node_scores.values() if ns.node_type == primary_type]
    if len(primary) < 4:
        # Fall back to all nodes if not enough primary-type nodes
        primary = list(node_scores.values())

    items = primary
    Q_arr = np.array([ns.Q for ns in items])
    I_arr = np.array([ns.I for ns in items])
    DC_arr= np.array([ns.degree_centrality for ns in items])

    # ── rank correlation ───────────────────────────────────────────────────────
    rho, rho_p = stats.spearmanr(Q_arr, I_arr)
    tau, tau_p = stats.kendalltau(Q_arr, I_arr)
    ci_lo, ci_hi = _bootstrap_spearman_ci(Q_arr, I_arr, B=B, alpha=alpha)

    # Guard NaN (can occur on constant arrays in tiny graphs)
    if math.isnan(rho):
        rho, rho_p = 0.0, 1.0
    if math.isnan(tau):
        tau, tau_p = 0.0, 1.0

    # ── classification: top-K by I (ground truth) vs top-K by Q ──────────────
    n = len(items)
    actual_k = min(top_k, n)
    gt_top_k  = set(sorted((ns.node_id for ns in items), key=lambda v: node_scores[v].I,  reverse=True)[:actual_k])
    pred_top_k= set(sorted((ns.node_id for ns in items), key=lambda v: node_scores[v].Q, reverse=True)[:actual_k])

    tp = len(gt_top_k & pred_top_k)
    fp = len(pred_top_k - gt_top_k)
    fn = len(gt_top_k  - pred_top_k)

    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    ftr   = fp / actual_k if actual_k > 0 else 0.0

    # ── SPOF-F1 ───────────────────────────────────────────────────────────────
    spof_actual = {ns.node_id for ns in items if ns.is_articulation_point and ns.I > 0.3}
    spof_pred   = {ns.node_id for ns in items if ns.is_articulation_point}
    if len(spof_actual) == 0 and len(spof_pred) == 0:
        spof_f1 = 1.0
    else:
        sp_tp = len(spof_actual & spof_pred)
        sp_fp = len(spof_pred - spof_actual)
        sp_fn = len(spof_actual - spof_pred)
        sp_p  = sp_tp / (sp_tp + sp_fp) if (sp_tp + sp_fp) > 0 else 0.0
        sp_r  = sp_tp / (sp_tp + sp_fn) if (sp_tp + sp_fn) > 0 else 0.0
        spof_f1 = 2 * sp_p * sp_r / (sp_p + sp_r) if (sp_p + sp_r) > 0 else 0.0

    # ── ICR@K (In-Cluster Recall): fraction of true-top-K that are "clustered"
    #    with a correct prediction in Q-rank neighbourhood ±K/2 ─────────────
    rank_by_Q  = {v: i for i, v in enumerate(sorted(node_scores, key=lambda v: node_scores[v].Q, reverse=True))}
    rank_by_I  = {v: i for i, v in enumerate(sorted(node_scores, key=lambda v: node_scores[v].I, reverse=True))}
    window = max(1, actual_k // 2)
    icr_hits = sum(1 for v in gt_top_k if abs(rank_by_Q[v] - rank_by_I[v]) <= window)
    icr = icr_hits / actual_k if actual_k > 0 else 0.0

    # ── Binary Classification Error ───────────────────────────────────────────
    # Binary labels: 1 if in gt_top_k, else 0
    y_true = np.array([1 if v in gt_top_k else 0 for v in node_scores])
    y_pred = np.array([1 if v in pred_top_k else 0 for v in node_scores])
    bce = float(np.mean(y_true != y_pred))

    # ── Wilcoxon: Q(v) ranks better than degree centrality ranks ──────────────
    # Difference signal: |ρ(Q,I)| vs |ρ(DC,I)| per bootstrap resample
    rho_dc, _ = stats.spearmanr(DC_arr, I_arr)
    diff_scores = np.abs(Q_arr - I_arr) - np.abs(DC_arr - I_arr)

    if len(diff_scores) >= 10:
        w_stat, w_p = stats.wilcoxon(diff_scores, alternative='less')
    else:
        w_stat, w_p = 0.0, 1.0

    pg = float(abs(rho) - abs(rho_dc))

    return dict(
        spearman_rho=float(rho),
        spearman_p=float(rho_p),
        kendall_tau=float(tau),
        kendall_p=float(tau_p),
        bootstrap_ci_lo=ci_lo,
        bootstrap_ci_hi=ci_hi,
        top_k=actual_k,
        precision_at_k=prec,
        recall_at_k=rec,
        f1_at_k=f1,
        spof_f1=spof_f1,
        ftr=ftr,
        icr_at_k=icr,
        bce=bce,
        pg=pg,
        wilcoxon_stat=float(w_stat),
        wilcoxon_p=float(w_p),
        wilcoxon_significant=(w_p < alpha),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# NODE-TYPE STRATIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def stratified_metrics(node_scores: Dict[str, NodeScores], top_k: int) -> Dict[str, Dict]:
    """
    Compute Spearman ρ and F1@K for each node type independently.
    """
    from collections import defaultdict
    by_type: Dict[str, List[NodeScores]] = defaultdict(list)
    for ns in node_scores.values():
        by_type[ns.node_type].append(ns)

    strata = {}
    for ntype, items in by_type.items():
        if len(items) < 4:
            strata[ntype] = {"n": len(items), "note": "too few nodes for ρ"}
            continue
        Q = np.array([ns.Q for ns in items])
        I = np.array([ns.I for ns in items])
        # Skip strata where I is constant (degenerate — e.g. Topics with no cascade)
        if np.std(I) < 1e-9 or np.std(Q) < 1e-9:
            strata[ntype] = {"n": len(items), "note": "constant signal (not a primary failure type)"}
            continue
        rho, p = stats.spearmanr(Q, I)
        if math.isnan(rho):
            rho, p = 0.0, 1.0
        k = max(1, min(top_k, len(items) // 5))
        gt_k  = set(sorted((ns.node_id for ns in items), key=lambda v: node_scores[v].I,  reverse=True)[:k])
        pr_k  = set(sorted((ns.node_id for ns in items), key=lambda v: node_scores[v].Q, reverse=True)[:k])
        tp = len(gt_k & pr_k)
        prec = tp / k
        rec  = tp / k
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        strata[ntype] = {
            "n": len(items),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": round(float(p), 4),
            "f1_at_k": round(f1, 4),
            "k_used": k,
        }
    return strata


# ═══════════════════════════════════════════════════════════════════════════════
# TOPOLOGY-CLASS GATE EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

GATE_THRESHOLDS = {
    # class           rho_min  f1_min  spof_f1_min  ftr_max  pg_min
    "sparse":        (0.75,   0.65,   0.60,        0.30,    0.02),
    "medium":        (0.80,   0.70,   0.65,        0.25,    0.03),
    "dense":         (0.82,   0.72,   0.65,        0.25,    0.03),
    "hub_spoke":     (0.85,   0.75,   0.70,        0.20,    0.03),
}

def classify_topology(G: nx.DiGraph) -> str:
    """Heuristic topology class from degree distribution."""
    G_phys = G.copy()
    dep_edges = [(u, v) for u, v, d in G_phys.edges(data=True) if d.get("etype") == "DEPENDS_ON"]
    G_phys.remove_edges_from(dep_edges)
    
    n = G_phys.number_of_nodes()
    m = G_phys.number_of_edges()
    if n == 0:
        return "sparse"
    density = m / (n * (n - 1)) if n > 1 else 0
    degrees = [d for _, d in G_phys.degree()]
    max_d = max(degrees) if degrees else 0
    avg_d = np.mean(degrees) if degrees else 0
    hub_ratio = max_d / (avg_d + 1e-9)
    if hub_ratio > 10 and density < 0.10:
        return "hub_spoke"
    if density < 0.05:
        return "sparse"
    if density > 0.20:
        return "dense"
    return "medium"


def evaluate_gates(res: ValidationResult, topo_class: str) -> Dict[str, bool]:
    """Return pass/fail for each gate threshold given topology class."""
    thresholds = GATE_THRESHOLDS.get(topo_class, GATE_THRESHOLDS["medium"])
    rho_min, f1_min, spof_min, ftr_max, pg_min = thresholds
    return {
        f"rho >= {rho_min}":      res.spearman_rho >= rho_min,
        f"f1 >= {f1_min}":        res.f1_at_k >= f1_min,
        f"spof_f1 >= {spof_min}": res.spof_f1 >= spof_min,
        f"ftr <= {ftr_max}":      res.ftr <= ftr_max,
        f"pg >= {pg_min}":        res.pg >= pg_min,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RANK CONSISTENCY RATE (across seeds)
# ═══════════════════════════════════════════════════════════════════════════════

def rank_consistency_rate(per_seed_scores: List[Dict[str, NodeScores]]) -> float:
    """
    RCR = 1 − mean_normalised_Kendall_distance between all seed pairs.

    Normalised Kendall distance ∈ [0, 1]; RCR = 1 means identical rankings.
    """
    rankings = []
    for sc in per_seed_scores:
        order = sorted(sc, key=lambda v: sc[v].Q, reverse=True)
        rankings.append({v: i for i, v in enumerate(order)})

    if len(rankings) < 2:
        return 1.0

    nodes_common = set(rankings[0].keys())
    for r in rankings[1:]:
        nodes_common &= set(r.keys())
    nodes_common = sorted(nodes_common)
    n = len(nodes_common)
    if n < 2:
        return 1.0

    distances = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            ri = [rankings[i][v] for v in nodes_common]
            rj = [rankings[j][v] for v in nodes_common]
            tau, _ = stats.kendalltau(ri, rj)
            # Normalised Kendall distance = (1 − τ) / 2
            distances.append((1 - tau) / 2)

    return 1.0 - float(np.mean(distances))


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_gnn_scores(G: nx.DiGraph, gnn_model: str, qos: bool = True) -> Dict[str, NodeScores]:
    from saag.prediction.gnn_service import GNNService
    from collections import defaultdict
    
    # 1. First compute the baseline metrics and RMAV (just like compute_rmav)
    rmav_node_scores = compute_rmav(G, qos=qos)
    
    # Format them as the dicts GNNService.predict expects:
    rmav_dict = {}
    for nid, ns in rmav_node_scores.items():
        rmav_dict[nid] = {
            "overall": ns.Q,
            "reliability": ns.R,
            "maintainability": ns.M,
            "availability": ns.A,
            "security": ns.S
        }
        
    # Extract structural metrics dict.
    pagerank = nx.pagerank(G, alpha=0.85)
    reverse_pagerank = nx.pagerank(G.reverse(), alpha=0.85)
    betweenness = nx.betweenness_centrality(G, normalized=True)
    try:
        G_depends = G.copy()
        other_edges = [(u, v) for u, v, d in G_depends.edges(data=True) if d.get("etype") != "DEPENDS_ON"]
        G_depends.remove_edges_from(other_edges)
        non_apps = [n for n, d in G_depends.nodes(data=True) if d.get("ntype") != "Application"]
        G_depends.remove_nodes_from(non_apps)
        art_points = set(nx.articulation_points(G_depends.to_undirected()))
    except:
        art_points = set()
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    
    # Calculate degree centrality on physical G (without DEPENDS_ON edges)
    try:
        G_phys = G.copy()
        dep_edges = [(u, v) for u, v, d in G_phys.edges(data=True) if d.get("etype") == "DEPENDS_ON"]
        G_phys.remove_edges_from(dep_edges)
        deg_cent_phys = nx.degree_centrality(G_phys)
    except Exception:
        deg_cent_phys = {}
    
    pub_count = defaultdict(int)
    sub_count = defaultdict(int)
    for u, v, d in G.edges(data=True):
        et = d.get("etype", "")
        if et == "PUBLISHES_TO":
            pub_count[v] += 1
        elif et == "SUBSCRIBES_TO":
            sub_count[v] += 1
            
    pspof_vals = defaultdict(float)
    for u, topic, d in G.edges(data=True):
        if d.get("etype") == "PUBLISHES_TO":
            if pub_count[topic] == 1 and sub_count[topic] >= 1:
                pspof_vals[u] = max(pspof_vals[u], 0.5 * min(sub_count[topic] / 5.0, 1.0))
                
    structural_metrics = {}
    for nid, ndata in G.nodes(data=True):
        max_bc = max(betweenness.values()) if betweenness.values() else 1.0
        node_mpci = betweenness.get(nid, 0.0) / (max_bc + 1e-12)
        structural_metrics[nid] = {
            "pagerank": pagerank.get(nid, 0.0),
            "reverse_pagerank": reverse_pagerank.get(nid, 0.0),
            "betweenness_centrality": betweenness.get(nid, 0.0),
            "closeness_centrality": 0.0,
            "eigenvector_centrality": 0.0,
            "in_degree_centrality": in_deg.get(nid, 0),
            "out_degree_centrality": out_deg.get(nid, 0),
            "clustering_coefficient": 0.0,
            "ap_c_score": 1.0 if nid in art_points else 0.0,
            "ap_c_directed": 1.0 if nid in art_points else 0.0,
            "cdi": 0.0,
            "mpci": node_mpci,
            "path_complexity": 0.0,
            "fan_out_criticality": 0.0,
            "bridge_ratio": 0.0,
            "qos_weight": 0.5,
            "qos_weight_in": 0.0,
            "qos_weight_out": pspof_vals.get(nid, 0.0) if qos else 0.0,
        }
        
    # Load GNN Service
    service = GNNService.from_checkpoint(gnn_model, graph=G)
    
    # Run GNN predict
    gnn_result = service.predict(
        graph=G,
        structural_metrics=structural_metrics,
        rmav_scores=rmav_dict,
        mode="gnn",
        qos_enabled=qos
    )
    
    # Use GNN node_scores directly
    scores_map = gnn_result.node_scores
    
    # Convert back to Dict[str, NodeScores]
    results = {}
    for nid, score in scores_map.items():
        node_type = "Application"
        if nid in G.nodes:
            node_type = G.nodes[nid].get("ntype", "Application")
            
        results[nid] = NodeScores(
            node_id=nid,
            node_type=node_type,
            Q=score.composite_score,
            R=score.reliability_score,
            M=score.maintainability_score,
            A=score.availability_score,
            S=score.security_score,
            I=0.0,
            degree_centrality=deg_cent_phys.get(nid, 0.0),
            is_articulation_point=(nid in art_points)
        )
    return results

def run_single(
    G: nx.DiGraph,
    raw: dict,
    seed: int,
    qos: bool,
    top_k_frac: float,
    depth_limit: int,
    B: int,
    alpha: float,
    gnn_model: Optional[str] = None,
) -> Tuple[ValidationResult, Dict[str, NodeScores]]:
    """Run one full validation pass and return (ValidationResult, node_scores)."""
    random.seed(seed)
    np.random.seed(seed)

    if gnn_model:
        scores = compute_gnn_scores(G, gnn_model, qos=qos)
    else:
        scores = compute_rmav(G, qos=qos)
    scores = derive_ground_truth(G, scores, depth_limit=depth_limit, seed=seed)

    n = len(scores)
    app_nodes = [v for v, ns in scores.items() if ns.node_type == "Application"]
    top_k = max(3, int(n * top_k_frac))

    stat = run_statistical_tests(scores, top_k=top_k, B=B, alpha=alpha, primary_type="Application")
    strata = stratified_metrics(scores, top_k=top_k)

    topo_class = classify_topology(G)
    vr = ValidationResult(
        seed=seed,
        qos_enabled=qos,
        n_nodes=n,
        n_app_nodes=len(app_nodes),
        strata=strata,
        **stat,
    )
    vr.gates_passed = evaluate_gates(vr, topo_class)
    vr.overall_pass = all(vr.gates_passed.values())
    return vr, scores


def run_sweep(
    G: nx.DiGraph,
    raw: dict,
    seeds: List[int],
    qos: bool,
    top_k_frac: float,
    depth_limit: int,
    B: int,
    alpha: float,
    gnn_model: Optional[str] = None,
) -> SweepReport:
    """Run multi-seed sweep and compute aggregate stability metrics."""
    results = []
    all_scores = []

    for s in seeds:
        vr, sc = run_single(G, raw, seed=s, qos=qos, top_k_frac=top_k_frac,
                            depth_limit=depth_limit, B=B, alpha=alpha, gnn_model=gnn_model)
        results.append(vr)
        all_scores.append(sc)

    rhos = [r.spearman_rho for r in results]
    f1s  = [r.f1_at_k for r in results]
    pgs  = [r.pg for r in results]

    rcr = rank_consistency_rate(all_scores)

    return SweepReport(
        qos_enabled=qos,
        seeds=seeds,
        rho_mean=float(np.mean(rhos)),
        rho_std=float(np.std(rhos)),
        rho_min=float(np.min(rhos)),
        rho_max=float(np.max(rhos)),
        f1_mean=float(np.mean(f1s)),
        pg_mean=float(np.mean(pgs)),
        rcr=rcr,
        all_gates_pass_rate=float(np.mean([r.overall_pass for r in results])),
        per_seed=results,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CONSOLE REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

_COLOR = {
    "green":  "\033[92m",
    "red":    "\033[91m",
    "yellow": "\033[93m",
    "cyan":   "\033[96m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

def _c(text: str, color: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{_COLOR[color]}{text}{_COLOR['reset']}"


def _tick(ok: bool, use_color: bool) -> str:
    return _c("✓", "green", use_color) if ok else _c("✗", "red", use_color)


def print_single_report(vr: ValidationResult, topo_class: str, use_color: bool = True):
    bold = _COLOR["bold"] if use_color else ""
    reset = _COLOR["reset"] if use_color else ""

    print(f"\n{bold}{'═'*64}{reset}")
    print(f"{bold}  VALIDATION REPORT  seed={vr.seed}  QoS={'ON' if vr.qos_enabled else 'OFF'}{reset}")
    print(f"{bold}{'═'*64}{reset}")
    print(f"  Nodes: {vr.n_nodes}  (Applications: {vr.n_app_nodes})  "
          f"Topology class: {_c(topo_class, 'cyan', use_color)}")

    print(f"\n{bold}  Rank Correlation{reset}")
    print(f"    Spearman ρ  = {_c(f'{vr.spearman_rho:.4f}', 'green' if vr.spearman_rho>=0.80 else 'red', use_color)}"
          f"  (p={vr.spearman_p:.4f})"
          f"  95% CI [{vr.bootstrap_ci_lo:.4f}, {vr.bootstrap_ci_hi:.4f}]")
    print(f"    Kendall τ   = {vr.kendall_tau:.4f}  (p={vr.kendall_p:.4f})")

    print(f"\n{bold}  Classification @ K={vr.top_k}{reset}")
    print(f"    Precision   = {vr.precision_at_k:.4f}")
    print(f"    Recall      = {vr.recall_at_k:.4f}")
    print(f"    F1          = {_c(f'{vr.f1_at_k:.4f}', 'green' if vr.f1_at_k>=0.70 else 'red', use_color)}")
    print(f"    SPOF-F1     = {vr.spof_f1:.4f}")
    print(f"    FTR         = {vr.ftr:.4f}")

    print(f"\n{bold}  Specialist Metrics{reset}")
    print(f"    ICR@K       = {vr.icr_at_k:.4f}")
    print(f"    BCE         = {vr.bce:.4f}")
    print(f"    PG (vs DC)  = {_c(f'{vr.pg:.4f}', 'green' if vr.pg>=0.03 else 'yellow', use_color)}")

    print(f"\n{bold}  Wilcoxon (Q > DC){reset}")
    print(f"    stat={vr.wilcoxon_stat:.2f}  p={vr.wilcoxon_p:.4f}  "
          f"{'significant' if vr.wilcoxon_significant else 'not significant'}")

    print(f"\n{bold}  Gate Evaluation  ({topo_class}){reset}")
    for gate, passed in vr.gates_passed.items():
        print(f"    {_tick(passed, use_color)} {gate}")

    overall = _c("PASS", "green", use_color) if vr.overall_pass else _c("FAIL", "red", use_color)
    print(f"\n  Overall: {bold}{overall}{reset}\n")

    if vr.strata:
        print(f"{bold}  Node-type Strata{reset}")
        for ntype, s in vr.strata.items():
            n_str = s.get("n", 0)
            if "note" in s:
                print(f"    {ntype:16s} n={n_str}  {s['note']}")
            else:
                rho_str = _c(f'{s["spearman_rho"]:.4f}', 'green' if s["spearman_rho"] >= 0.70 else 'yellow', use_color)
                print(f"    {ntype:16s} n={n_str:4d}  ρ={rho_str}  F1={s['f1_at_k']:.4f}")


def print_sweep_report(sr: SweepReport, use_color: bool = True):
    bold = _COLOR["bold"] if use_color else ""
    reset = _COLOR["reset"] if use_color else ""

    print(f"\n{bold}{'═'*64}{reset}")
    print(f"{bold}  SWEEP REPORT  QoS={'ON' if sr.qos_enabled else 'OFF'}  "
          f"seeds={sr.seeds}{reset}")
    print(f"{bold}{'═'*64}{reset}")
    print(f"  ρ  mean={_c(f'{sr.rho_mean:.4f}','green' if sr.rho_mean>=0.80 else 'red',use_color)}"
          f"  std={sr.rho_std:.4f}  "
          f"[{sr.rho_min:.4f}, {sr.rho_max:.4f}]")
    print(f"  F1 mean={sr.f1_mean:.4f}")
    print(f"  PG mean={sr.pg_mean:.4f}")
    print(f"  RCR     = {_c(f'{sr.rcr:.4f}','green' if sr.rcr>=0.90 else 'yellow',use_color)}")
    print(f"  All-gates pass rate = {sr.all_gates_pass_rate:.2%}\n")

    print(f"{bold}  Per-seed ρ{reset}")
    for r in sr.per_seed:
        ok = _tick(r.overall_pass, use_color)
        print(f"    seed={r.seed}  ρ={r.spearman_rho:.4f}  F1={r.f1_at_k:.4f}  PG={r.pg:.4f}  {ok}")


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION COMPARISON (topology-only vs QoS-enriched)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AblationReport:
    """Side-by-side comparison: topology-only baseline vs QoS-enriched."""
    topology_class: str
    n_nodes: int
    n_app_nodes: int
    seeds: List[int]

    # baseline (qos=False)
    base_rho_mean: float
    base_rho_std: float
    base_f1_mean: float
    base_pg_mean: float
    base_rcr: float

    # enriched (qos=True)
    enr_rho_mean: float
    enr_rho_std: float
    enr_f1_mean: float
    enr_pg_mean: float
    enr_rcr: float

    # deltas
    delta_rho: float          # Δρ = enr − base  (primary Middleware 2026 claim)
    delta_f1: float
    delta_pg: float
    rho_lift_significant: bool  # bootstrap overlap test: CI(enr) does not overlap CI(base)

    # per-seed raw series (for plotting)
    base_rhos: List[float] = field(default_factory=list)
    enr_rhos: List[float] = field(default_factory=list)


def run_ablation(
    G: nx.DiGraph,
    raw: dict,
    seeds: List[int],
    top_k_frac: float,
    depth_limit: int,
    B: int,
    alpha: float,
) -> AblationReport:
    """
    Run sweep for both QoS=False and QoS=True and compute ablation deltas.

    The Δρ = ρ(Q_QoS, I) − ρ(Q_topo, I) is the primary Middleware 2026
    evidence that QoS contract topology carries predictive signal beyond
    purely structural topology.
    """
    sr_base = run_sweep(G, raw, seeds=seeds, qos=False,
                        top_k_frac=top_k_frac, depth_limit=depth_limit,
                        B=B, alpha=alpha)
    sr_enr  = run_sweep(G, raw, seeds=seeds, qos=True,
                        top_k_frac=top_k_frac, depth_limit=depth_limit,
                        B=B, alpha=alpha)

    topo_class = classify_topology(G)
    n_nodes    = sr_base.per_seed[0].n_nodes if sr_base.per_seed else 0
    n_apps     = sr_base.per_seed[0].n_app_nodes if sr_base.per_seed else 0

    base_rhos = [r.spearman_rho for r in sr_base.per_seed]
    enr_rhos  = [r.spearman_rho for r in sr_enr.per_seed]

    # Non-overlap bootstrap test: does the enriched 95% CI sit above base CI?
    # Approximated here as t-test on seed-level rho series.
    if len(base_rhos) >= 3 and len(enr_rhos) >= 3:
        _, p_lift = stats.ttest_rel(enr_rhos, base_rhos, alternative="greater")
        significant = p_lift < alpha
    else:
        significant = (sr_enr.rho_mean - sr_base.rho_mean) > 0.01

    return AblationReport(
        topology_class=topo_class,
        n_nodes=n_nodes,
        n_app_nodes=n_apps,
        seeds=seeds,
        base_rho_mean=sr_base.rho_mean,
        base_rho_std=sr_base.rho_std,
        base_f1_mean=sr_base.f1_mean,
        base_pg_mean=sr_base.pg_mean,
        base_rcr=sr_base.rcr,
        enr_rho_mean=sr_enr.rho_mean,
        enr_rho_std=sr_enr.rho_std,
        enr_f1_mean=sr_enr.f1_mean,
        enr_pg_mean=sr_enr.pg_mean,
        enr_rcr=sr_enr.rcr,
        delta_rho=sr_enr.rho_mean - sr_base.rho_mean,
        delta_f1=sr_enr.f1_mean  - sr_base.f1_mean,
        delta_pg=sr_enr.pg_mean  - sr_base.pg_mean,
        rho_lift_significant=significant,
        base_rhos=base_rhos,
        enr_rhos=enr_rhos,
    )


def print_ablation_report(ar: AblationReport, use_color: bool = True):
    bold = _COLOR["bold"] if use_color else ""
    reset = _COLOR["reset"] if use_color else ""

    def _delta(v: float) -> str:
        sign = "+" if v >= 0 else ""
        col = "green" if v > 0.005 else ("yellow" if v > -0.005 else "red")
        return _c(f"{sign}{v:.4f}", col, use_color)

    print(f"\n{bold}{'═'*64}{reset}")
    print(f"{bold}  ABLATION REPORT  (topology-only vs QoS-enriched){reset}")
    print(f"{bold}{'═'*64}{reset}")
    print(f"  Topology class: {ar.topology_class}   "
          f"Nodes: {ar.n_nodes}  Apps: {ar.n_app_nodes}   "
          f"Seeds: {ar.seeds}")

    header = f"\n  {'Metric':<20} {'Topo-only':>12} {'QoS-enr':>12} {'Δ':>10}"
    sep    = f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}"
    print(header)
    print(sep)

    def row(label, b, e):
        d = e - b
        print(f"  {label:<20} {b:>12.4f} {e:>12.4f} {_delta(d):>10}")

    row("ρ  (mean)",   ar.base_rho_mean,  ar.enr_rho_mean)
    row("ρ  (std)",    ar.base_rho_std,   ar.enr_rho_std)
    row("F1 (mean)",   ar.base_f1_mean,   ar.enr_f1_mean)
    row("PG (mean)",   ar.base_pg_mean,   ar.enr_pg_mean)
    row("RCR",         ar.base_rcr,       ar.enr_rcr)

    sig_str = _c("significant (p<α)", "green", use_color) if ar.rho_lift_significant \
              else _c("not significant", "yellow", use_color)
    print(f"\n  QoS-enriched ρ lift: {sig_str}")
    print(f"  Δρ = {ar.delta_rho:+.4f}  ΔF1 = {ar.delta_f1:+.4f}  ΔPG = {ar.delta_pg:+.4f}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# LaTeX TABLE EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def write_latex_table(ar: AblationReport, path: str):
    """
    Write a ready-to-paste IEEE two-column LaTeX table (booktabs style)
    suitable for Middleware 2026 / VISSOFT 2026 / UYMS 2026.
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation Study: Topology-Only vs.\ QoS-Enriched Prediction}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{@{}lSSS@{}}",
        r"\toprule",
        r"Metric & {Topo-Only} & {QoS-Enr.} & {$\Delta$} \\",
        r"\midrule",
        rf"Spearman $\rho$ (mean) & {ar.base_rho_mean:.4f} & {ar.enr_rho_mean:.4f} & {ar.delta_rho:+.4f} \\",
        rf"Spearman $\rho$ (std)  & {ar.base_rho_std:.4f}  & {ar.enr_rho_std:.4f}  & {ar.enr_rho_std - ar.base_rho_std:+.4f} \\",
        rf"F1 @ $K$               & {ar.base_f1_mean:.4f} & {ar.enr_f1_mean:.4f} & {ar.delta_f1:+.4f} \\",
        rf"Predictive Gain (PG)   & {ar.base_pg_mean:.4f} & {ar.enr_pg_mean:.4f} & {ar.delta_pg:+.4f} \\",
        rf"RCR                    & {ar.base_rcr:.4f}      & {ar.enr_rcr:.4f}     & {ar.enr_rcr - ar.base_rcr:+.4f} \\",
        r"\midrule",
    ]
    sig_note = r"$p < \alpha$, significant" if ar.rho_lift_significant \
               else r"not significant"
    lines += [
        rf"\multicolumn{{4}}{{l}}{{\small QoS $\rho$-lift: {sig_note}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    Path(path).write_text("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════════
# GUARD: minimum application-node count for reliable statistics
# ═══════════════════════════════════════════════════════════════════════════════

_MIN_APPS_FOR_RELIABLE_RHO = 10

def _check_min_apps(G: nx.DiGraph, use_color: bool = True):
    """Warn if fewer than MIN_APPS Application nodes are present."""
    n_apps = sum(1 for _, d in G.nodes(data=True) if d.get("ntype") == "Application")
    if n_apps < _MIN_APPS_FOR_RELIABLE_RHO:
        msg = (f"WARNING: only {n_apps} Application nodes found "
               f"(minimum recommended: {_MIN_APPS_FOR_RELIABLE_RHO}). "
               f"Spearman ρ will have high variance on small n.")
        print(_c(msg, "yellow", use_color))
    return n_apps




def write_csv(node_scores: Dict[str, NodeScores], path: str):
    import csv
    rows = sorted(node_scores.values(), key=lambda ns: ns.Q, reverse=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "node_id", "node_type", "Q", "R", "M", "A", "S",
                    "I", "cascade_depth", "nodes_affected",
                    "is_articulation_point", "degree_centrality"])
        for rank, ns in enumerate(rows, 1):
            w.writerow([
                rank, ns.node_id, ns.node_type,
                round(ns.Q, 5), round(ns.R, 5), round(ns.M, 5),
                round(ns.A, 5), round(ns.S, 5),
                round(ns.I, 5), ns.cascade_depth, ns.nodes_affected,
                ns.is_articulation_point, round(ns.degree_centrality, 5),
            ])


# ═══════════════════════════════════════════════════════════════════════════════
# HARNESS — methodological-guard validation on pre-computed Q(v) / I(v) files
# ═══════════════════════════════════════════════════════════════════════════════

MIN_N = 3  # minimum aligned points required before reporting a correlation


@dataclass
class Prediction:
    """Predicted criticality for one node (pre-computed Q(v))."""
    node_id: str
    node_type: str
    q: float


@dataclass
class GroundTruthSource:
    """One independent ground-truth impact signal I(v) or I_dyn(v).

    scores    : {node_id: impact} for a single seed, or the per-seed mean.
    per_seed  : optional list of {node_id: impact} per seed; `scores` is
                derived automatically if omitted.
    qos_coupled : True if this signal shares QoS weights with Q(v). Triggers
                  the ablation caveat in the report.
    independence: free-text note on how I(v) was kept independent of Q(v).
    """
    name: str
    scores: Dict[str, float] = field(default_factory=dict)
    per_seed: Optional[List[Dict[str, float]]] = None
    qos_coupled: bool = False
    independence: str = ""

    def __post_init__(self) -> None:
        if self.per_seed and not self.scores:
            keys = set().union(*[set(d) for d in self.per_seed])
            self.scores = {
                k: float(np.mean([d[k] for d in self.per_seed if k in d]))
                for k in keys
            }


@dataclass
class Corr:
    rho: float
    p: float
    tau: float
    n: int

    @property
    def reportable(self) -> bool:
        return self.n >= MIN_N and not np.isnan(self.rho)


def _safe_corr(x: Sequence[float], y: Sequence[float]) -> Corr:
    n = len(x)
    if n < MIN_N or len(set(x)) < 2 or len(set(y)) < 2:
        return Corr(float("nan"), float("nan"), float("nan"), n)
    rho, p = spearmanr(x, y)
    tau, _ = kendalltau(x, y)
    return Corr(float(rho), float(p), float(tau), n)


def align(
    preds: Dict[str, Prediction], gt: Dict[str, float]
) -> List[Tuple[str, str, float, float]]:
    """Inner-join prediction and ground truth on node_id.

    Returns [(node_id, node_type, q, i), ...] for nodes present in both.
    """
    common = sorted(set(preds) & set(gt))
    return [(nid, preds[nid].node_type, preds[nid].q, gt[nid]) for nid in common]


def stratified(
    rows: List[Tuple[str, str, float, float]]
) -> Dict[str, Corr]:
    """Per-node-type Spearman/Kendall correlation (Simpson's-paradox guard)."""
    by_type: Dict[str, List[Tuple[float, float]]] = {}
    for _, ntype, q, i in rows:
        by_type.setdefault(ntype, []).append((q, i))
    out: Dict[str, Corr] = {}
    for ntype, pairs in by_type.items():
        qs = [p[0] for p in pairs]
        is_ = [p[1] for p in pairs]
        out[ntype] = _safe_corr(qs, is_)
    return out


def precision_at_k(
    rows: List[Tuple[str, str, float, float]], k: int
) -> Optional[float]:
    """|top-k by Q ∩ top-k by I| / k."""
    if k > len(rows):
        return None
    top_q = {r[0] for r in sorted(rows, key=lambda r: r[2], reverse=True)[:k]}
    top_i = {r[0] for r in sorted(rows, key=lambda r: r[3], reverse=True)[:k]}
    return len(top_q & top_i) / k


def rank_displacement(
    rows: List[Tuple[str, str, float, float]], top: int = 5
) -> List[Tuple[str, str, int, float, float]]:
    """Nodes most under-predicted by Q relative to I (structural blind spots)."""
    q_rank = {
        r[0]: idx + 1
        for idx, r in enumerate(sorted(rows, key=lambda r: r[2], reverse=True))
    }
    i_rank = {
        r[0]: idx + 1
        for idx, r in enumerate(sorted(rows, key=lambda r: r[3], reverse=True))
    }
    disp = [
        (r[0], r[1], i_rank[r[0]] - q_rank[r[0]], r[2], r[3]) for r in rows
    ]
    disp.sort(key=lambda d: d[2])
    return disp[:top]


def per_seed_spread(
    preds: Dict[str, Prediction], src: GroundTruthSource
) -> Optional[Tuple[float, float, int]]:
    """Mean ± std of pooled Spearman ρ across seeds. None if not multi-seed."""
    if not src.per_seed:
        return None
    rhos: List[float] = []
    for seed_scores in src.per_seed:
        rows = align(preds, seed_scores)
        c = _safe_corr([r[2] for r in rows], [r[3] for r in rows])
        if c.reportable:
            rhos.append(c.rho)
    if not rhos:
        return None
    return float(np.mean(rhos)), float(np.std(rhos)), len(rhos)


def load_predictions(path: Path) -> Dict[str, Prediction]:
    """Load Q(v) JSON.

    Accepts either:
      A) {"<node_id>": {"type": "...", "Q": <float>, ...}}
      B) [{"node_id": "...", "type": "...", "Q": <float>}, ...]
    """
    raw = json.loads(path.read_text())
    out: Dict[str, Prediction] = {}
    rows = raw.values() if isinstance(raw, dict) else raw
    keys = raw.keys() if isinstance(raw, dict) else [None] * len(raw)
    for key, row in zip(keys, rows):
        nid = row.get("node_id", key)
        if nid is None:
            raise ValueError("prediction row missing node_id")
        q = row.get("Q", row.get("q"))
        ntype = row.get("type", row.get("node_type", "Unknown"))
        if q is None:
            continue
        out[str(nid)] = Prediction(str(nid), str(ntype), float(q))
    if not out:
        raise ValueError(f"no predictions parsed from {path}")
    return out


def load_ground_truth(
    path: Path,
    name: str,
    *,
    field_name: str = "impact_score",
    qos_coupled: bool = False,
    independence: str = "",
) -> GroundTruthSource:
    """Load I(v) JSON.

    Accepts either:
      A) {"<node_id>": <float>}
      B) FaultInjector impact_scores.json: {"records": {"<nid>": {"impact_score": ...}}}
      C) {"<node_id>": {"<field_name>": <float>}}
    """
    raw = json.loads(path.read_text())
    records = raw.get("records", raw) if isinstance(raw, dict) else raw
    scores: Dict[str, float] = {}
    for nid, val in records.items():
        if isinstance(val, dict):
            v = val.get(field_name)
            if v is None:
                continue
            scores[str(nid)] = float(v)
        else:
            scores[str(nid)] = float(val)
    if not scores:
        raise ValueError(f"no ground-truth scores parsed from {path}")
    return GroundTruthSource(
        name=name, scores=scores, qos_coupled=qos_coupled, independence=independence
    )


def _fmt(c: Corr) -> str:
    if not c.reportable:
        return f"n/a (n={c.n}, insufficient)"
    star = "*" if c.p < 0.05 else " "
    return f"ρ={c.rho:+.3f}{star}  τ={c.tau:+.3f}  p={c.p:.3f}  n={c.n}"


def build_report(
    preds: Dict[str, Prediction], sources: List[GroundTruthSource]
) -> Tuple[str, dict]:
    lines: List[str] = []
    blob: dict = {"sources": {}, "convergent_validity": {}}
    w = lines.append

    w("=" * 74)
    w("SaG PREDICTION VALIDATION REPORT")
    w("=" * 74)
    w(f"Predicted nodes: {len(preds)}   "
      f"types: {sorted({p.node_type for p in preds.values()})}")
    w("")

    for src in sources:
        rows = align(preds, src.scores)
        pooled = _safe_corr([r[2] for r in rows], [r[3] for r in rows])
        strat = stratified(rows)
        sblob: dict = {"pooled": pooled.__dict__, "per_type": {}}

        w("─" * 74)
        w(f"GROUND TRUTH: {src.name}   (aligned nodes: {len(rows)})")
        if src.independence:
            w(f"  independence: {src.independence}")
        w("─" * 74)

        w(f"  POOLED  (across all types — Simpson's-paradox risk): {_fmt(pooled)}")
        w("          ▲ pooled ρ mixes node types with different (Q,I) regimes;")
        w("            interpret the per-type rows below as the real signal.")
        w("")
        w("  STRATIFIED by node type:")
        for ntype in sorted(strat):
            w(f"    {ntype:<22} {_fmt(strat[ntype])}")
            sblob["per_type"][ntype] = strat[ntype].__dict__
        w("")

        w("  PRECISION@k (predicted top-k vs ground-truth top-k):")
        pk = {}
        for k in (3, 5, 10):
            val = precision_at_k(rows, k)
            if val is not None:
                w(f"    P@{k:<3} = {val:.2f}")
                pk[k] = val
        sblob["precision_at_k"] = pk
        w("")

        disp = rank_displacement(rows, top=5)
        w("  RANK-DISPLACEMENT outliers (I ranks node more critical than Q):")
        w(f"    {'node':<22}{'type':<14}{'Δrank':>6}  {'Q':>6}  {'I':>6}")
        for nid, ntype, d, q, i in disp:
            flag = "  ← blind spot" if d <= -2 else ""
            w(f"    {nid:<22}{ntype:<14}{d:>+6}  {q:>6.3f}  {i:>6.3f}{flag}")
        sblob["rank_displacement"] = [
            {"node_id": n, "type": t, "delta_rank": d, "Q": q, "I": i}
            for n, t, d, q, i in disp
        ]
        w("")

        spread = per_seed_spread(preds, src)
        if spread is not None:
            mean, std, ns = spread
            w(f"  MULTI-SEED pooled ρ: {mean:+.3f} ± {std:.3f}  (seeds={ns})")
            if std > 0:
                w("            ▲ std > 0 → I(v) is cascade-order sensitive for "
                  "some nodes; report as fragility.")
            sblob["multi_seed"] = {"mean_rho": mean, "std_rho": std, "seeds": ns}
            w("")

        if src.qos_coupled:
            w("  ⚠  INDEPENDENCE: this source is declared QoS-coupled. Its")
            w("    correlation with Q(v) may be inflated by shared QoS weights.")
            w("    A QoS-ablation arm is REQUIRED before citing this number.")
            w("")

        blob["sources"][src.name] = sblob

    if len(sources) >= 2:
        w("─" * 74)
        w("CONVERGENT VALIDITY (independent ground-truth sources vs each other)")
        w("─" * 74)
        for a in range(len(sources)):
            for b in range(a + 1, len(sources)):
                sa, sb = sources[a], sources[b]
                common = sorted(set(sa.scores) & set(sb.scores))
                c = _safe_corr(
                    [sa.scores[n] for n in common],
                    [sb.scores[n] for n in common],
                )
                w(f"  {sa.name}  vs  {sb.name}:  {_fmt(c)}")
                blob["convergent_validity"][f"{sa.name}__{sb.name}"] = c.__dict__
        w("    ▲ strong agreement between independent oracles is the convergent-")
        w("      validity argument; weak agreement bounds what either can claim.")
        w("")

    w("=" * 74)
    return "\n".join(lines), blob


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        prog="validate_graph.py",
        description="SaG topology prediction vs. simulation-based ground-truth validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=False)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input",   required=True, help="Path to system.json")
    common.add_argument("--qos",     action="store_true", help="Enable QoS-enriched scoring")
    common.add_argument("--gnn-model", default=None, help="Path to GNN checkpoint directory")
    common.add_argument("--top-k",   type=int, default=None,
                        help="K for classification metrics (default: 20%% of nodes)")
    common.add_argument("--seeds",   default="42,123,456,789,2024",
                        help="Comma-separated seed list")
    common.add_argument("--cascade", type=int, default=5, help="Cascade depth limit")
    common.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples")
    common.add_argument("--alpha",   type=float, default=0.05, help="Significance level")
    common.add_argument("--output",  default=None, help="Write JSON report to path")
    common.add_argument("--csv",     action="store_true", help="Write per-node CSV")
    common.add_argument("--latex",   action="store_true", help="Write LaTeX ablation table")
    common.add_argument("--verbose", action="store_true")
    common.add_argument("--no-color", action="store_true")

    # subcommands
    sub.add_parser("single",  parents=[common], help="One-seed run (first seed only)")
    sub.add_parser("sweep",   parents=[common], help="Multi-seed stability sweep")
    sub.add_parser("report",  parents=[common], help="Full sweep + strata + gates JSON report")
    sub.add_parser("compare", parents=[common],
                   help="Ablation: topology-only vs QoS-enriched side-by-side")

    harness_p = sub.add_parser(
        "harness",
        help="Methodological-guard harness: validate pre-computed Q(v) vs I(v) JSON files",
    )
    harness_p.add_argument("--predictions", required=True, type=Path,
                            help="Q(v) JSON ({node_id:{type,Q}} or list).")
    harness_p.add_argument("--ground-truth", action="append", default=[],
                            metavar="NAME=PATH",
                            help="Ground-truth source (e.g. cascade=output/impact_scores.json). "
                                 "Repeatable. Append ':qos' to mark QoS-coupled.")
    harness_p.add_argument("--out", type=Path, default=None,
                            help="Optional JSON report path.")
    harness_p.add_argument("--no-color", action="store_true")

    return p.parse_args()


def main():
    # Check if a subcommand is specified; if not, default to "single"
    subcommands = {"single", "sweep", "report", "compare", "harness"}
    has_subcommand = any(arg in subcommands for arg in sys.argv[1:] if not arg.startswith("-"))
    if not has_subcommand:
        sys.argv.insert(1, "single")

    args = _parse_args()
    setup_logging(args)
    use_color = not args.no_color

    G, raw = load_graph(args.input)
    seeds = [int(s) for s in args.seeds.split(",")]
    _check_min_apps(G, use_color)
    n_total = G.number_of_nodes()
    top_k_frac = 0.20 if args.top_k is None else max(0.01, args.top_k / max(n_total, 1))
    topo_class = classify_topology(G)

    gnn_model = getattr(args, "gnn_model", None)

    if args.command == "single":
        vr, node_scores = run_single(
            G, raw, seed=seeds[0], qos=args.qos,
            top_k_frac=top_k_frac, depth_limit=args.cascade,
            B=args.bootstrap, alpha=args.alpha,
            gnn_model=gnn_model,
        )
        print_single_report(vr, topo_class, use_color)

        if args.verbose:
            print("\nTop-20 nodes by Q(v):")
            ranked = sorted(node_scores.values(), key=lambda ns: ns.Q, reverse=True)[:20]
            for rank, ns in enumerate(ranked, 1):
                print(f"  {rank:3d}. {ns.node_id:30s} Q={ns.Q:.4f}  I={ns.I:.4f}  "
                      f"{'AP ' if ns.is_articulation_point else '   '}{ns.node_type}")

        if args.csv:
            csv_path = (args.output or "validation").replace(".json", "") + "_nodes.csv"
            write_csv(node_scores, csv_path)
            print(f"\nPer-node CSV written to: {csv_path}")

        if args.output:
            report = {"validation": asdict(vr), "topology_class": topo_class}
            Path(args.output).write_text(json.dumps(report, indent=2, cls=NpEncoder))
            print(f"JSON report written to: {args.output}")

        sys.exit(0 if vr.overall_pass else 1)

    elif args.command == "sweep":
        sr = run_sweep(
            G, raw, seeds=seeds, qos=args.qos,
            top_k_frac=top_k_frac, depth_limit=args.cascade,
            B=args.bootstrap, alpha=args.alpha,
            gnn_model=gnn_model,
        )
        print_sweep_report(sr, use_color)

        if args.output:
            Path(args.output).write_text(json.dumps(asdict(sr), indent=2, cls=NpEncoder))
            print(f"JSON report written to: {args.output}")

        sys.exit(0 if sr.all_gates_pass_rate == 1.0 else 1)

    elif args.command == "report":
        sr = run_sweep(
            G, raw, seeds=seeds, qos=args.qos,
            top_k_frac=top_k_frac, depth_limit=args.cascade,
            B=args.bootstrap, alpha=args.alpha,
            gnn_model=gnn_model,
        )
        print_sweep_report(sr, use_color)
        if sr.per_seed:
            print_single_report(sr.per_seed[0], topo_class, use_color)

        if args.output:
            out = {
                "sweep": asdict(sr),
                "topology_class": topo_class,
                "gate_thresholds": GATE_THRESHOLDS.get(topo_class),
            }
            Path(args.output).write_text(json.dumps(out, indent=2, cls=NpEncoder))
            print(f"\nFull report written to: {args.output}")

        sys.exit(0 if sr.all_gates_pass_rate == 1.0 else 1)

    elif args.command == "compare":
        ar = run_ablation(
            G, raw, seeds=seeds,
            top_k_frac=top_k_frac, depth_limit=args.cascade,
            B=args.bootstrap, alpha=args.alpha,
        )
        print_ablation_report(ar, use_color)

        if args.output:
            Path(args.output).write_text(json.dumps(asdict(ar), indent=2, cls=NpEncoder))
            print(f"Ablation JSON written to: {args.output}")

        if args.latex:
            latex_path = (args.output or "ablation").replace(".json", "") + "_table.tex"
            write_latex_table(ar, latex_path)
            print(f"LaTeX table written to:   {latex_path}")

        # Exit 0 only if Δρ > 0 and significant (validates the QoS claim)
        sys.exit(0 if (ar.delta_rho > 0 and ar.rho_lift_significant) else 1)

    elif args.command == "harness":
        preds = load_predictions(args.predictions)
        sources: List[GroundTruthSource] = []
        for spec in args.ground_truth:
            name, _, rest = spec.partition("=")
            path_str, _, tag = rest.partition(":")
            sources.append(load_ground_truth(
                Path(path_str), name,
                qos_coupled=(tag == "qos"),
                independence="declared independent" if tag != "qos" else "QoS-coupled",
            ))
        if not sources:
            print("error: harness subcommand requires at least one --ground-truth source",
                  file=sys.stderr)
            sys.exit(2)
        text, blob = build_report(preds, sources)
        print(text)
        if args.out:
            args.out.write_text(json.dumps(blob, indent=2, cls=NpEncoder))
            print(f"\nJSON report → {args.out}")
        sys.exit(0)


if __name__ == "__main__":
    main()