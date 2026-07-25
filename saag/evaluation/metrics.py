"""Canonical metric contract shared by every predictor variant.

Moved here verbatim from ``cli/loso_evaluate.py`` (which now re-exports it) so
that the in-distribution table, the LOSO table and the k-fold table all score
variants with the same code on the same declared node population.

Two properties matter for the published tables:

**One population per table.** Structural baselines need no training and were
historically scored on every node, while learned variants were scored on the 20%
test split of an ``{Application, Library}`` subset. Mixing those conventions
inside one table compares different estimators on different samples.
:func:`resolve_eval_keys` makes the population an explicit, recorded choice.

**Absent is not zero.** A stratum whose labels are constant (Topic and Node
carry no ground truth at all — see ``docs/failure-simulation.md`` L6/L10) has an
*undefined* rank correlation, not a correlation of 0.0. Emitting 0.0 there
silently converts a coverage gap into evidence of a bad model.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

#: Sentinel written to JSON when a statistic cannot be computed. Distinct from
#: 0.0 (a real, poor correlation) and from a missing key (nothing was attempted).
UNDEFINED = "undefined"

#: Node populations a table may be scored on. The value is the set of node
#: ``type`` attributes retained; ``None`` means "every type present".
EVAL_POPULATIONS: Dict[str, Optional[frozenset]] = {
    # Default. Matches the thesis claim — topology predicts *application-layer*
    # cascade criticality — and is the only population every variant can score.
    "application": frozenset({"Application"}),
    # Applications plus Libraries: the full DEPENDS_ON projection substrate.
    "app_lib": frozenset({"Application", "Library"}),
    # Every node that carries a non-degenerate label, whatever its type.
    "labeled": None,
}

#: Minimum stratum size for a Spearman correlation to be reported at all.
_MIN_STRATUM = 3


def resolve_eval_keys(
    pred_scores: Mapping[str, float],
    true_impact: Mapping[str, float],
    graph,
    population: str = "application",
    unlabeled_ids: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return the sorted node ids every variant in a table must be scored on.

    The key set is a function of the *graph and labels only* — never of the
    predictions — so two variants evaluated with the same ``population`` on the
    same scenario are guaranteed to see an identical sample.

    A label of exactly ``0.0`` is **kept**. For a node the simulator actually
    injected, zero impact is a real measurement ("its failure reaches nobody"),
    and discarding it would be a results-favourable filter. Only nodes the
    simulator never injected are excluded, and those are identified explicitly
    via ``unlabeled_ids`` (the artifact's ``unlabeled_node_ids``) rather than
    inferred from the value, because the two are indistinguishable once written.
    """
    if population not in EVAL_POPULATIONS:
        raise ValueError(
            f"unknown eval population {population!r}; "
            f"expected one of {sorted(EVAL_POPULATIONS)}"
        )
    allowed = EVAL_POPULATIONS[population]
    excluded = {str(n) for n in (unlabeled_ids or ())}

    keys = []
    for nid in set(pred_scores) & set(true_impact):
        if nid in excluded:
            continue
        if allowed is not None:
            node_type = graph.nodes[nid].get("type") if nid in graph else None
            if node_type not in allowed:
                continue
        keys.append(nid)
    return sorted(keys)


def _per_type_rho(
    keys: Sequence[str],
    pred_scores: Mapping[str, float],
    true_impact: Mapping[str, float],
    graph,
) -> Dict[str, Dict[str, Any]]:
    """Stratified Spearman ρ, with undefined strata reported as undefined.

    A stratum is undefined when it has fewer than three members, or when either
    vector is constant (no ranks to correlate). Both cases previously collapsed
    to ``0.0``, which is what filled the per-node-type table with zeros for
    Topic/Node/Library and made "consistent strength across component types"
    unsupportable.
    """
    by_type: Dict[str, List[Tuple[float, float]]] = {}
    for nid in keys:
        node_type = graph.nodes[nid].get("type", "Unknown") if nid in graph else "Unknown"
        by_type.setdefault(node_type, []).append(
            (float(pred_scores[nid]), float(true_impact[nid]))
        )

    out: Dict[str, Dict[str, Any]] = {}
    for node_type, pairs in sorted(by_type.items()):
        n = len(pairs)
        if n < _MIN_STRATUM:
            out[node_type] = {"rho": UNDEFINED, "n": n, "reason": "too_few_nodes"}
            continue
        preds, trues = (np.asarray(v, dtype=np.float64) for v in zip(*pairs))
        if np.ptp(preds) == 0.0 or np.ptp(trues) == 0.0:
            out[node_type] = {"rho": UNDEFINED, "n": n, "reason": "constant_signal"}
            continue
        rho, _ = spearmanr(preds, trues)
        if np.isnan(rho):
            out[node_type] = {"rho": UNDEFINED, "n": n, "reason": "undefined_rho"}
            continue
        out[node_type] = {"rho": float(rho), "n": n}
    return out


def aggregate_per_type(
    entries: Sequence[Mapping[str, Any]],
    value_key: str = "rho",
    count_key: str = "n_seeds",
) -> Dict[str, Dict[str, Any]]:
    """Average per-node-type statistics while preserving ``undefined``.

    Every consumer of :func:`compute_inductive_metrics`' ``per_type_rho`` needs
    the same rule: a stratum that was undefined on some runs contributes nothing
    to the mean, and a stratum undefined on *all* runs stays ``undefined``
    instead of collapsing to 0.0. Centralised so a table, a figure and a CSV
    cannot disagree about what an unmeasurable stratum looks like.

    Parameters
    ----------
    entries:
        One ``{node_type: {value_key: float|"undefined", ...}}`` mapping per run.
    """
    values: Dict[str, List[float]] = {}
    undefined: Dict[str, int] = {}
    n_nodes: Dict[str, int] = {}

    for entry in entries:
        for node_type, info in (entry or {}).items():
            raw = info.get(value_key) if isinstance(info, Mapping) else info
            if isinstance(info, Mapping) and "n" in info:
                n_nodes[node_type] = info["n"]
            if isinstance(raw, (int, float)) and not isinstance(raw, bool) and not np.isnan(raw):
                values.setdefault(node_type, []).append(float(raw))
            else:
                undefined[node_type] = undefined.get(node_type, 0) + 1

    out: Dict[str, Dict[str, Any]] = {}
    for node_type in sorted(set(values) | set(undefined)):
        vs = values.get(node_type, [])
        out[node_type] = {
            "mean": float(np.mean(vs)) if vs else UNDEFINED,
            "std": float(np.std(vs)) if vs else UNDEFINED,
            "n_nodes": n_nodes.get(node_type, 0),
            count_key: len(vs),
            f"{count_key}_undefined": undefined.get(node_type, 0),
        }
    return out


def _minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


#: Boundaries of the QoS tiers used for stratified reporting, as w(t) cut points.
#: Chosen to mirror CRITICALITY_THRESHOLDS in saag/core/models.py so a "high" tier
#: here means the same thing as a "high" Topic.criticality label.
_QOS_TIER_BOUNDS: Sequence[Tuple[str, float]] = (
    ("minimal", 0.19),
    ("low", 0.43),
    ("medium", 0.64),
    ("high", 1.01),
)


def _qos_tier(weight: float) -> str:
    """Bucket a QoS weight w(t) into a reporting tier."""
    for name, upper in _QOS_TIER_BOUNDS:
        if weight < upper:
            return name
    return "critical"


def _topic_qos_weights(graph) -> Dict[str, float]:
    """w(t) for every Topic node, resolved from whichever QoS shape is present."""
    from saag.core.models import topic_weight_from_node_attrs

    weights: Dict[str, float] = {}
    for nid, attrs in graph.nodes(data=True):
        if attrs.get("type") != "Topic":
            continue
        existing = attrs.get("weight")
        if isinstance(existing, (int, float)) and existing > 0:
            weights[nid] = float(existing)
        else:
            weights[nid] = topic_weight_from_node_attrs(attrs)
    return weights


def component_qos_exposure(graph) -> Dict[str, float]:
    """QoS mass each component carries: Σ w(t) over topics it publishes or routes.

    Publish and route edges are what a component is *responsible for delivering*;
    subscriptions are excluded because losing a subscriber does not silence the
    channel for anyone else.
    """
    topic_weights = _topic_qos_weights(graph)
    exposure: Dict[str, float] = {}
    for src, dst, attrs in graph.edges(data=True):
        etype = (attrs.get("type") or attrs.get("etype") or "").upper()
        if etype not in ("PUBLISHES_TO", "ROUTES"):
            continue
        weight = topic_weights.get(dst)
        if weight is not None:
            exposure[src] = exposure.get(src, 0.0) + weight
    return exposure


def critical_topic_coverage_at_k(
    keys: Sequence[str],
    pred_scores: Mapping[str, float],
    graph,
    k: int,
) -> Dict[str, Any]:
    """Share of the system's QoS mass covered by the top-K predicted components.

    Answers the question a hardening budget actually poses — "if I harden K
    components, how much of what matters have I protected?" — which a global rank
    correlation cannot express: a model can rank well overall while missing the
    handful of components carrying the critical channels.

    ``lift`` compares the achieved coverage against what K components chosen at
    random would cover, so a value of 1.0 means the ranking added nothing.
    """
    exposure = component_qos_exposure(graph)
    total = sum(exposure.get(nid, 0.0) for nid in keys)
    if total <= 0 or not keys:
        return {"coverage": UNDEFINED, "reason": "no_qos_exposure", "k": k}

    ranked = sorted(keys, key=lambda nid: -float(pred_scores[nid]))[:k]
    covered = sum(exposure.get(nid, 0.0) for nid in ranked)
    coverage = covered / total

    random_share = len(ranked) / len(keys)
    return {
        "coverage": float(coverage),
        "random_baseline": float(random_share),
        "lift": float(coverage / random_share) if random_share > 0 else UNDEFINED,
        "k": k,
        "total_qos_mass": float(total),
    }


def per_qos_tier_rho(
    keys: Sequence[str],
    pred_scores: Mapping[str, float],
    true_impact: Mapping[str, float],
    graph,
) -> Dict[str, Dict[str, Any]]:
    """Spearman ρ stratified by the QoS tier of the mass a component carries.

    A pooled ρ can be carried by the many low-QoS components while the model
    ranks the critical ones badly — the same Simpson's-paradox risk the existing
    node-type stratification guards against, on the axis the QoS claim is about.
    Follows the same conventions as :func:`_per_type_rho`: fewer than
    ``_MIN_STRATUM`` members or a constant vector is ``undefined``, never 0.0.
    """
    exposure = component_qos_exposure(graph)

    by_tier: Dict[str, List[Tuple[float, float]]] = {}
    for nid in keys:
        mass = exposure.get(nid)
        if mass is None:
            continue  # carries no topic; the QoS axis is undefined for it
        by_tier.setdefault(_qos_tier(mass), []).append(
            (float(pred_scores[nid]), float(true_impact[nid]))
        )

    out: Dict[str, Dict[str, Any]] = {}
    for tier, pairs in sorted(by_tier.items()):
        n = len(pairs)
        if n < _MIN_STRATUM:
            out[tier] = {"rho": UNDEFINED, "n": n, "reason": "too_few_nodes"}
            continue
        preds, trues = (np.asarray(v, dtype=np.float64) for v in zip(*pairs))
        if np.ptp(preds) == 0.0 or np.ptp(trues) == 0.0:
            out[tier] = {"rho": UNDEFINED, "n": n, "reason": "constant_signal"}
            continue
        rho, _ = spearmanr(preds, trues)
        out[tier] = (
            {"rho": UNDEFINED, "n": n, "reason": "undefined_rho"}
            if np.isnan(rho) else {"rho": float(rho), "n": n}
        )
    return out


def compute_inductive_metrics(
    pred_scores: Dict[str, float],
    true_impact: Dict[str, float],
    graph,
    top_k_frac: float = 0.20,
    tau_frac: float = 0.50,
    population: str = "labeled",
    eval_keys: Optional[Iterable[str]] = None,
    label_stability: Optional[Mapping[str, Any]] = None,
    unlabeled_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Compute the full metric family for one (scenario, variant, seed) cell.

    Three families are reported because they answer different questions:

    * ``*_at_k`` — top-K overlap. ``precision_at_k``, ``recall_at_k`` and
      ``f1_at_k`` are **identically equal** here: both sets contain exactly K
      elements, so tp/K == tp/K. They are retained for backward compatibility
      with existing CSVs; ``overlap_at_k`` is the honest name.
    * ``*_at_tau`` — precision/recall against ``tau = tau_frac * max(y_true)``.
      The true critical set is sized by the data rather than fixed at K, so the
      two genuinely diverge. Relative to the max because label magnitude is not
      comparable across scenarios (max I*(v) spans ~4x over the cohort).
    * ``pr_auc`` — average precision over the full ranking against that same
      truth set. No K and no prediction-side threshold; the best single summary
      for cross-scenario comparison.

    ``rmse``/``mae`` compare sigmoid-scale predictions against raw labels and are
    dominated by label scale; ``rmse_scaled``/``mae_scaled`` min-max both vectors
    first, and ``label_scale_max`` keeps the raw figures interpretable.

    Parameters
    ----------
    population:
        Node population to score on. Ignored when ``eval_keys`` is given.
    eval_keys:
        Explicit key set, so a caller can force several variants onto one sample
        computed once via :func:`resolve_eval_keys`.
    label_stability:
        The labeler's self-agreement block (``test_retest_spearman``,
        ``topk_jaccard``). Carried through unchanged so every reported ρ can be
        read next to the ceiling no method can exceed.
    """
    n_predicted = len(pred_scores)
    n_labeled = len(true_impact)

    if eval_keys is None:
        common = resolve_eval_keys(
            pred_scores, true_impact, graph, population, unlabeled_ids
        )
    else:
        common = sorted(eval_keys)

    base = {
        "eval_population": population,
        "n_predicted": n_predicted,
        "n_labeled": n_labeled,
        "n_evaluated": len(common),
        "label_stability": dict(label_stability) if label_stability else {},
    }

    if len(common) < _MIN_STRATUM:
        return {
            **base,
            "spearman_rho": float("nan"),
            "n": len(common),
            "per_type_rho": {},
            "note": "insufficient_overlap",
        }

    y_pred = np.array([pred_scores[v] for v in common], dtype=np.float64)
    y_true = np.array([true_impact[v] for v in common], dtype=np.float64)

    if np.ptp(y_pred) == 0.0 or np.ptp(y_true) == 0.0:
        rho, p_val = float("nan"), 1.0
    else:
        rho, p_val = spearmanr(y_pred, y_true)
        rho = float(rho) if not np.isnan(rho) else float("nan")

    # F1 @ top-K. Both sets have exactly k elements, so precision == recall == f1.
    k = max(1, int(round(len(common) * top_k_frac)))
    pred_top = set(np.argsort(-y_pred)[:k].tolist())
    true_top = set(np.argsort(-y_true)[:k].tolist())
    tp = len(pred_top & true_top)
    precision = tp / k if k > 0 else 0.0
    recall = tp / len(true_top) if true_top else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    label_scale_max = float(y_true.max())
    tau = tau_frac * label_scale_max
    true_critical = y_true >= tau if label_scale_max > 0 else np.zeros_like(y_true, dtype=bool)
    n_true_critical = int(true_critical.sum())

    if 0 < n_true_critical < len(common):
        tp_tau = int(true_critical[np.argsort(-y_pred)[:k]].sum())
        precision_tau = tp_tau / k
        recall_tau = tp_tau / n_true_critical
        f1_tau = (
            2 * precision_tau * recall_tau / (precision_tau + recall_tau)
            if (precision_tau + recall_tau) > 0 else 0.0
        )
        pr_auc = float(average_precision_score(true_critical.astype(int), y_pred))
    else:
        # Degenerate truth set: every node critical or none. Precision/recall
        # carry no information, so report them as undefined rather than 0.0.
        precision_tau = recall_tau = f1_tau = float("nan")
        pr_auc = float("nan")

    k_ndcg = min(10, len(common))
    ideal_order = np.argsort(-y_true)[:k_ndcg]
    pred_order = np.argsort(-y_pred)[:k_ndcg]
    discount = np.log2(np.arange(2, k_ndcg + 2))
    dcg = float(np.sum(y_true[pred_order] / discount))
    idcg = float(np.sum(y_true[ideal_order] / discount))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    pred_s, true_s = _minmax(y_pred), _minmax(y_true)
    rmse_scaled = float(np.sqrt(np.mean((pred_s - true_s) ** 2)))
    mae_scaled = float(np.mean(np.abs(pred_s - true_s)))

    return {
        **base,
        "spearman_rho": rho,
        "spearman_p": float(p_val) if not np.isnan(p_val) else 1.0,
        "f1_at_k": f1,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "ndcg_10": ndcg,
        "rmse": rmse,
        "mae": mae,
        "n": len(common),
        "k": k,
        "per_type_rho": _per_type_rho(common, pred_scores, true_impact, graph),
        # QoS axis: does the ranking hold up on the components carrying the
        # critical channels, and how much of that mass does a top-K budget cover?
        "per_qos_tier_rho": per_qos_tier_rho(common, pred_scores, true_impact, graph),
        "critical_topic_coverage_at_k": critical_topic_coverage_at_k(
            common, pred_scores, graph, k
        ),
        # Honest name for f1_at_k/precision_at_k/recall_at_k, which are equal.
        "overlap_at_k": f1,
        # Absolute-threshold critical set: precision and recall diverge here.
        "precision_at_tau": precision_tau,
        "recall_at_tau": recall_tau,
        "f1_at_tau": f1_tau,
        "n_true_critical": n_true_critical,
        "tau": tau,
        "pr_auc": pr_auc,
        # Scale diagnostics for the otherwise-uninterpretable rmse/mae above.
        "rmse_scaled": rmse_scaled,
        "mae_scaled": mae_scaled,
        "label_scale_max": label_scale_max,
    }
