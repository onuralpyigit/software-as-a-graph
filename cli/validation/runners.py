"""End-to-end run orchestration: single seed, multi-seed sweep, QoS ablation."""
from __future__ import annotations

import random

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import stats

from .ground_truth import derive_ground_truth
from .scoring import NodeScores, compute_gnn_scores, compute_rm
from .statistics import (
    GATE_THRESHOLDS, SweepReport, ValidationResult, classify_topology,
    evaluate_gates, rank_consistency_rate, run_statistical_tests, stratified_metrics,
)


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
        scores = compute_rm(G, qos=qos)
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
