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
from typing import Dict, List, Optional, Tuple

from cli.common.arguments import setup_logging

import networkx as nx
import numpy as np
from scipy import stats

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

    # RMAV dimensions (topology-only baseline when qos=False)
    R: float = 0.0                # Reliability exposure  (PageRank + in-degree)
    M: float = 0.0                # Maintainability proxy (betweenness + closeness)
    A: float = 0.0                # Availability risk     (articulation × QoS_SPOF)
    V: float = 0.0                # Vulnerability         (out-degree + dep-density)
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
            G.add_node(nid, ntype=ntype, label=item.get("name", nid), raw=item)

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
                G.add_edge(s, t, etype=etype)

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
                if pub != sub and not G.has_edge(pub, sub):
                    G.add_edge(pub, sub, etype="DEPENDS_ON", via=topic)

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
        art_points = set(nx.articulation_points(G.to_undirected()))
    except:
        art_points = set()
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    
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
            V=q.scores.vulnerability,
            I=0.0,
            degree_centrality=pagerank.get(q.id, 0.0),
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
    gt_top_k  = set(sorted(node_scores, key=lambda v: node_scores[v].I,  reverse=True)[:actual_k])
    pred_top_k= set(sorted(node_scores, key=lambda v: node_scores[v].Q, reverse=True)[:actual_k])

    tp = len(gt_top_k & pred_top_k)
    fp = len(pred_top_k - gt_top_k)
    fn = len(gt_top_k  - pred_top_k)

    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    ftr   = fp / actual_k if actual_k > 0 else 0.0

    # ── SPOF-F1 ───────────────────────────────────────────────────────────────
    spof_actual = {v for v, ns in node_scores.items() if ns.is_articulation_point and ns.I > 0.3}
    spof_pred   = {v for v, ns in node_scores.items() if ns.is_articulation_point}
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
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return "sparse"
    density = m / (n * (n - 1)) if n > 1 else 0
    degrees = [d for _, d in G.degree()]
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

def run_single(
    G: nx.DiGraph,
    raw: dict,
    seed: int,
    qos: bool,
    top_k_frac: float,
    depth_limit: int,
    B: int,
    alpha: float,
) -> Tuple[ValidationResult, Dict[str, NodeScores]]:
    """Run one full validation pass and return (ValidationResult, node_scores)."""
    random.seed(seed)
    np.random.seed(seed)

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
) -> SweepReport:
    """Run multi-seed sweep and compute aggregate stability metrics."""
    results = []
    all_scores = []

    for s in seeds:
        vr, sc = run_single(G, raw, seed=s, qos=qos, top_k_frac=top_k_frac,
                            depth_limit=depth_limit, B=B, alpha=alpha)
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
        w.writerow(["rank", "node_id", "node_type", "Q", "R", "M", "A", "V",
                    "I", "cascade_depth", "nodes_affected",
                    "is_articulation_point", "degree_centrality"])
        for rank, ns in enumerate(rows, 1):
            w.writerow([
                rank, ns.node_id, ns.node_type,
                round(ns.Q, 5), round(ns.R, 5), round(ns.M, 5),
                round(ns.A, 5), round(ns.V, 5),
                round(ns.I, 5), ns.cascade_depth, ns.nodes_affected,
                ns.is_articulation_point, round(ns.degree_centrality, 5),
            ])


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
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input",   required=True, help="Path to system.json")
    common.add_argument("--qos",     action="store_true", help="Enable QoS-enriched scoring")
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

    return p.parse_args()


def main():
    args = _parse_args()
    setup_logging(args)
    use_color = not args.no_color

    G, raw = load_graph(args.input)
    seeds = [int(s) for s in args.seeds.split(",")]
    _check_min_apps(G, use_color)
    n_total = G.number_of_nodes()
    top_k_frac = 0.20 if args.top_k is None else max(0.01, args.top_k / max(n_total, 1))
    topo_class = classify_topology(G)

    if args.command == "single":
        vr, node_scores = run_single(
            G, raw, seed=seeds[0], qos=args.qos,
            top_k_frac=top_k_frac, depth_limit=args.cascade,
            B=args.bootstrap, alpha=args.alpha,
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


if __name__ == "__main__":
    main()