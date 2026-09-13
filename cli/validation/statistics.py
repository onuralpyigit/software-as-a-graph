"""Statistical battery, topology classification and gate evaluation."""
from __future__ import annotations

import math

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy import stats
from scipy.stats import kendalltau, spearmanr

from .scoring import NodeScores

#: Minimum number of strictly-positive labels for the zero-excluded rho to be
#: reported at all. Matches ``saag/evaluation/metrics.py::_MIN_STRATUM``.
_MIN_POSITIVE_FOR_RHO = 3


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

    # ── zero-exclusion sensitivity ─────────────────────────────────────────────
    # rho over the strictly-positive-impact subset, with the count it was taken
    # over. Reported as a bound on the headline rho, never in place of it.
    n_zero_impact: int = 0
    n_positive_impact: int = 0
    spearman_rho_positive: Optional[float] = None

    # ── classification ─────────────────────────────────────────────────────────
    top_k: int = 0
    #: |Top-K(Q) ∩ Top-K(I)| / K. This one number *is* precision@K, recall@K and
    #: F1@K -- both sets are "the top K", so they are the same size. It used to be
    #: reported under all three names plus its own complement (ftr), which made one
    #: statistic look like four independent gate conditions.
    overlap_at_k: float = 0.0
    spof_f1: float = 0.0

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
    overlap_mean: float
    pg_mean: float
    rcr: float                    # Rank Consistency Rate = 1 − (mean Kendall distance)
    all_gates_pass_rate: float    # Fraction of seeds that pass all gates
    per_seed: List[ValidationResult] = field(default_factory=list)

    # ── zero-exclusion sensitivity (see ValidationResult) ──────────────────────
    # ``rho_positive_mean`` is None when too few seeds yielded a defined value.
    rho_positive_mean: Optional[float] = None
    n_zero_impact_mean: float = 0.0
    n_positive_impact_mean: float = 0.0


def top_k_sets(items, node_scores: Dict[str, "NodeScores"], k: int) -> Tuple[set, set]:
    """The top-K node sets by ground truth I(v) and by prediction Q(v)."""
    ids = [ns.node_id for ns in items]
    gt_top = set(sorted(ids, key=lambda v: (-node_scores[v].I, v))[:k])
    pred_top = set(sorted(ids, key=lambda v: (-node_scores[v].Q, v))[:k])
    return gt_top, pred_top


def top_k_agreement(items, node_scores: Dict[str, "NodeScores"], k: int) -> float:
    """Fraction of the top-K set that Q(v) and I(v) agree on.

    This single number *is* precision@K, recall@K and F1@K: the predicted and
    ground-truth positive sets are both "the top K", so they are the same size,
    TP+FP = TP+FN = K, and the three coincide by construction. It is now reported
    once, under one name.
    """
    if k <= 0 or not items:
        return 0.0
    gt_top, pred_top = top_k_sets(items, node_scores, k)
    return len(gt_top & pred_top) / k


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

    This matches the RM thesis claim: topology predicts *application-layer*
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
    gt_top_k, pred_top_k = top_k_sets(items, node_scores, actual_k)

    # Both sets have exactly `actual_k` members, so this is precision, recall and
    # F1 at once. See `top_k_agreement`.
    overlap = top_k_agreement(items, node_scores, actual_k)

    # ── SPOF-F1 ───────────────────────────────────────────────────────────────
    # The predicted set is the top-K by Q(v). It used to be "every articulation
    # point", which made spof_actual a subset of spof_pred: recall was then
    # identically 1.0 and the whole statistic never read Q at all, so it returned
    # the same value for RM, RM-QoS and every GNN variant and could not
    # discriminate between the models it was gating.
    spof_actual = {ns.node_id for ns in items if ns.is_articulation_point and ns.I > 0.3}
    spof_pred   = pred_top_k
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

    # Signed, not |rho| - |rho_dc|: taking absolute values scored a perfectly
    # anti-correlated predictor (rho = -0.9) as a large gain over degree.
    pg = float(rho - rho_dc)

    # ── zero-exclusion sensitivity ────────────────────────────────────────────
    # On several transcribed architectures 40-66% of Applications carry I(v)=0,
    # and Spearman over a heavily tied population is dominated by midrank tie
    # handling rather than by the ordering under test. rho over the strictly
    # positive subset bounds how much of the headline figure is the separation
    # of inert components from active ones. It is a sensitivity check, not a
    # replacement: a zero label is a real measurement ("its failure reaches
    # nobody"), so the reported rho keeps them.
    positive = I_arr > 0.0
    n_positive = int(positive.sum())
    rho_pos: Optional[float] = None
    if n_positive >= _MIN_POSITIVE_FOR_RHO:
        Q_pos, I_pos = Q_arr[positive], I_arr[positive]
        if np.std(Q_pos) > 0 and np.std(I_pos) > 0:
            r_pos, _ = stats.spearmanr(Q_pos, I_pos)
            if not math.isnan(r_pos):
                rho_pos = float(r_pos)

    return dict(
        spearman_rho=float(rho),
        spearman_p=float(rho_p),
        n_zero_impact=int(n - n_positive),
        n_positive_impact=n_positive,
        spearman_rho_positive=rho_pos,
        kendall_tau=float(tau),
        kendall_p=float(tau_p),
        bootstrap_ci_lo=ci_lo,
        bootstrap_ci_hi=ci_hi,
        top_k=actual_k,
        overlap_at_k=overlap,
        spof_f1=spof_f1,
        icr_at_k=icr,
        bce=bce,
        pg=pg,
        wilcoxon_stat=float(w_stat),
        wilcoxon_p=float(w_p),
        wilcoxon_significant=(w_p < alpha),
    )


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
        f1 = top_k_agreement(items, node_scores, k)
        strata[ntype] = {
            "n": len(items),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": round(float(p), 4),
            "overlap_at_k": round(f1, 4),
            "k_used": k,
        }
    return strata


#: Four conditions, not five. The retired one was FTR, which was defined as
#: 1 - overlap_at_k -- the exact complement of the overlap condition, with a
#: strictly tighter implied threshold in every class (sparse 0.70 > 0.65,
#: medium 0.75 > 0.70, dense 0.75 > 0.72, hub_spoke 0.80 > 0.75), so the overlap
#: condition could never bind and the gate was never really five conditions.
GATE_THRESHOLDS = {
    # class           rho_min  overlap_min  spof_f1_min  pg_min
    "sparse":        (0.75,   0.70,        0.60,        0.02),
    "medium":        (0.80,   0.75,        0.65,        0.03),
    "dense":         (0.82,   0.75,        0.65,        0.03),
    "hub_spoke":     (0.85,   0.80,        0.70,        0.03),
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
    rho_min, overlap_min, spof_min, pg_min = thresholds
    return {
        f"rho >= {rho_min}":          res.spearman_rho >= rho_min,
        f"overlap >= {overlap_min}":  res.overlap_at_k >= overlap_min,
        f"spof_f1 >= {spof_min}":     res.spof_f1 >= spof_min,
        f"pg >= {pg_min}":            res.pg >= pg_min,
    }


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
