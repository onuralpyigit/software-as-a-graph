"""
Parameter uncertainty for pre-deployment runs.

Before deployment every runtime parameter -- the operating point, the service
distribution, how long an input stays fresh, how bursty traffic is -- is a
guess. Rather than report a label computed at one guessed point, a
pre-deployment study draws the parameters from declared ranges, recomputes the
label under each draw, and reports how stable the resulting ranking is.

`sample_settings` turns `PARAMETER_RANGES` (or a caller's own) into
`MessageFlowSimulator` keyword arguments; `rank_stability` summarises the
labels obtained across draws.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

#: Supported parameter-design sampling strategies for pre-deployment sweeps.
SAMPLING_METHODS: Tuple[str, ...] = ("random", "lhs", "sobol")

#: Default ranges. Numeric ranges are (low, high) and drawn uniformly; tuples of
#: strings are categorical choices. Every key is a `MessageFlowSimulator`
#: keyword argument.
PARAMETER_RANGES: Dict[str, Any] = {
    "target_utilization": (0.45, 0.85),
    "service_distribution": ("exponential", "uniform"),
    "staleness_periods": (2.0, 6.0),
    "burst_ratio": (2.0, 6.0),
}


#: Drawn keys that the simulator reads only under one setting, as (key, value).
#: Under any other value a draw of the key changes nothing in the run.
ONLY_UNDER: Dict[str, tuple] = {
    "burst_ratio": ("workload_model", "mmpp"),
    "burst_fraction": ("workload_model", "mmpp"),
    "mean_burst_s": ("workload_model", "mmpp"),
    "staleness_periods": ("behavior_model", "reactive"),
}

#: The simulator's own defaults for the keys in `ONLY_UNDER`.
_SIMULATOR_DEFAULTS = {"workload_model": "declared", "behavior_model": "open_loop"}


def inert_parameters(settings: Mapping[str, Any], ranges: Mapping[str, Any] = PARAMETER_RANGES) -> List[str]:
    """Keys of *ranges* that a run under *settings* never reads.

    `burst_ratio` is drawn by default, but it shapes traffic only under
    ``workload_model="mmpp"``; with declared workload every draw gives the same
    run, so its spread in `rank_stability` is no evidence about burstiness.
    """
    out = []
    for key in sorted(ranges):
        if key in ONLY_UNDER:
            setting, needed = ONLY_UNDER[key]
            if settings.get(setting, _SIMULATOR_DEFAULTS[setting]) != needed:
                out.append(key)
    return out


def sample_settings(rng: random.Random, ranges: Mapping[str, Any] = PARAMETER_RANGES) -> Dict[str, Any]:
    """One draw from *ranges*, in sorted key order so a seed fixes the draw."""
    out: Dict[str, Any] = {}
    for key in sorted(ranges):
        spec = ranges[key]
        if spec and all(isinstance(v, str) for v in spec):
            out[key] = spec[rng.randrange(len(spec))]
        else:
            lo, hi = spec
            out[key] = lo + (hi - lo) * rng.random()
    return out


def _is_categorical(spec: Any) -> bool:
    return bool(spec) and all(isinstance(v, str) for v in spec)


def sample_design(
    n: int,
    seed: int = 42,
    ranges: Mapping[str, Any] = PARAMETER_RANGES,
    method: str = "lhs",
) -> List[Dict[str, Any]]:
    """Draw *n* parameter settings from *ranges* using a space-filling design.

    Methods:
    - ``"random"``: iid uniform draws (`sample_settings`).
    - ``"lhs"``: stratified Latin Hypercube Sampling across numeric dimensions
      and balanced shuffled cycles across categorical dimensions.
    - ``"sobol"``: scrambled Sobol quasi-Monte Carlo sequence (`scipy.stats.qmc`).
    """
    if method not in SAMPLING_METHODS:
        raise ValueError(f"method must be one of {SAMPLING_METHODS}, got {method!r}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n!r}")
    if n == 0:
        return []
    keys = sorted(ranges)
    if not keys:
        return [{} for _ in range(n)]

    rng = random.Random(seed)
    if method == "random":
        return [sample_settings(rng, ranges) for _ in range(n)]

    if method == "lhs":
        cols: Dict[str, List[Any]] = {}
        for key in keys:
            spec = ranges[key]
            if _is_categorical(spec):
                col = [spec[i % len(spec)] for i in range(n)]
                rng.shuffle(col)
                cols[key] = col
            else:
                lo, hi = float(spec[0]), float(spec[1])
                strata = [(i + rng.random()) / n for i in range(n)]
                rng.shuffle(strata)
                cols[key] = [lo + (hi - lo) * u for u in strata]
        return [{k: cols[k][i] for k in keys} for i in range(n)]

    # method == "sobol"
    from scipy.stats.qmc import Sobol

    m = max(1, math.ceil(math.log2(n))) if n > 1 else 0
    pts = Sobol(d=len(keys), scramble=True, seed=seed).random_base2(m)[:n]
    out: List[Dict[str, Any]] = []
    for row in pts:
        draw: Dict[str, Any] = {}
        for key, u in zip(keys, row):
            spec = ranges[key]
            if _is_categorical(spec):
                idx = min(len(spec) - 1, int(float(u) * len(spec)))
                draw[key] = spec[idx]
            else:
                lo, hi = float(spec[0]), float(spec[1])
                draw[key] = lo + (hi - lo) * float(u)
        out.append(draw)
    return out


def _quantile(sorted_vals: Sequence[float], alpha: float) -> float:
    m = len(sorted_vals)
    if m == 1:
        return float(sorted_vals[0])
    pos = alpha * (m - 1)
    lo = int(math.floor(pos))
    hi = min(m - 1, lo + 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def risk_summary(
    labels: Sequence[Mapping[str, float]],
    alpha: float = 0.9,
) -> Dict[str, Dict[str, float]]:
    """Per-node distribution and upper-tail risk (VaR / CVaR) across parameter draws."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    nodes = sorted(set().union(*labels)) if labels else []
    out: Dict[str, Dict[str, float]] = {}
    for n in nodes:
        vals = sorted(float(draw[n]) for draw in labels if n in draw)
        if not vals:
            continue
        m = len(vals)
        tail_k = max(1, math.ceil((1.0 - alpha) * m))
        out[n] = {
            "mean": statistics.fmean(vals),
            "std": statistics.stdev(vals) if m > 1 else 0.0,
            "min": vals[0],
            "max": vals[-1],
            "quantile": _quantile(vals, alpha),
            "cvar": statistics.fmean(vals[-tail_k:]),
        }
    return out


def cvar_ranking(
    labels: Sequence[Mapping[str, float]],
    alpha: float = 0.9,
) -> List[str]:
    """Component ids ranked descending by upper-tail CVaR (ties broken by mean, id)."""
    summary = risk_summary(labels, alpha=alpha)
    return sorted(summary, key=lambda n: (-summary[n]["cvar"], -summary[n]["mean"], n))


def parameter_sensitivity(
    settings: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, float]],
    inert: Iterable[str] = (),
) -> Dict[str, Any]:
    """Attribute impact variation across draws to each uncertain parameter.

    - Numeric parameters: Spearman correlation with system-mean impact, mean
      absolute node-level Spearman correlation (unified ``score`` in [0, 1]),
      and Spearman correlation with rank-shift from the ensemble mean ranking.
    - Categorical parameters: variance explained ``eta_sq`` averaged over
      non-constant nodes (unified ``score = sqrt(eta_sq)`` in [0, 1]) and
      category-level system mean impacts.
    """
    from scipy.stats import spearmanr

    inert_set = set(inert)
    if not settings or not labels or len(settings) != len(labels):
        return {"parameters": {}, "top_drivers": []}
    nodes = sorted(set().union(*labels))
    if not nodes or len(settings) < 2:
        return {
            "parameters": {
                k: {"kind": "inert" if k in inert_set else "insufficient_draws", "score": 0.0}
                for k in sorted(settings[0])
            },
            "top_drivers": [],
        }

    # Draw-by-node matrix
    table = [[float(draw.get(n, 0.0)) for n in nodes] for draw in labels]
    sys_means = [statistics.fmean(row) for row in table]
    node_cols = list(zip(*table))
    mean_vec = [statistics.fmean(col) for col in node_cols]

    # Per-draw rank shift relative to the ensemble mean ranking
    rank_shifts: List[float] = []
    mean_const = len(set(mean_vec)) <= 1
    for row in table:
        if mean_const or len(set(row)) <= 1:
            rank_shifts.append(0.0)
        else:
            rho = spearmanr(row, mean_vec).correlation
            rank_shifts.append(0.0 if math.isnan(rho) else 1.0 - float(rho))

    def _safe_rho(xs: Sequence[float], ys: Sequence[float]) -> float:
        if len(xs) < 2 or len(set(xs)) <= 1 or len(set(ys)) <= 1:
            return 0.0
        r = spearmanr(xs, ys).correlation
        return 0.0 if math.isnan(r) else float(r)

    params: Dict[str, Dict[str, Any]] = {}
    for key in sorted(settings[0]):
        if key in inert_set:
            params[key] = {"kind": "inert", "score": 0.0}
            continue
        vals = [s[key] for s in settings]
        if all(isinstance(v, str) for v in vals):
            cats = sorted(set(vals))
            cat_means = {
                c: statistics.fmean(sys_means[i] for i, v in enumerate(vals) if v == c)
                for c in cats
            }
            etas: List[float] = []
            for col in node_cols:
                grand = statistics.fmean(col)
                ss_tot = sum((x - grand) ** 2 for x in col)
                if ss_tot <= 1e-18:
                    continue
                ss_between = 0.0
                for c in cats:
                    grp = [col[i] for i, v in enumerate(vals) if v == c]
                    if grp:
                        ss_between += len(grp) * (statistics.fmean(grp) - grand) ** 2
                etas.append(min(1.0, max(0.0, ss_between / ss_tot)))
            eta_sq = statistics.fmean(etas) if etas else 0.0
            params[key] = {
                "kind": "categorical",
                "eta_sq": round(eta_sq, 6),
                "score": round(math.sqrt(eta_sq), 6),
                "category_means": {c: round(m, 6) for c, m in cat_means.items()},
            }
        else:
            fvals = [float(v) for v in vals]
            sys_rho = _safe_rho(fvals, sys_means)
            node_rhos = [abs(_safe_rho(fvals, col)) for col in node_cols if len(set(col)) > 1]
            mean_abs = statistics.fmean(node_rhos) if node_rhos else 0.0
            shift_rho = _safe_rho(fvals, rank_shifts)
            params[key] = {
                "kind": "numeric",
                "system_mean_rho": round(sys_rho, 6),
                "mean_abs_node_rho": round(mean_abs, 6),
                "rank_shift_rho": round(shift_rho, 6),
                "score": round(mean_abs, 6),
            }

    top_drivers = sorted(
        [k for k, v in params.items() if v["kind"] != "inert"],
        key=lambda k: (-params[k]["score"], k),
    )
    return {"parameters": params, "top_drivers": top_drivers}


def rank_stability(labels: Sequence[Mapping[str, float]], k_frac: float = 0.2) -> Dict[str, Any]:
    """Spread of per-node labels and agreement of rankings across draws.

    Returns per-node mean and std, the worst pairwise Spearman rho between
    draws, and the worst pairwise top-K Jaccard. Worst rather than mean, for the
    same reason as `FaultInjector._compute_label_stability`: the ceiling is set
    by the weakest agreement.
    """
    from scipy.stats import spearmanr

    nodes = sorted(set().union(*labels)) if labels else []
    out: Dict[str, Any] = {"n_draws": len(labels), "n_nodes": len(nodes)}
    if not nodes:
        return out
    table = [[draw.get(n, 0.0) for n in nodes] for draw in labels]
    out["mean"] = {n: statistics.fmean(col) for n, col in zip(nodes, zip(*table))}
    out["std"] = {n: (statistics.stdev(col) if len(col) > 1 else 0.0)
                  for n, col in zip(nodes, zip(*table))}
    if len(labels) < 2:
        out["worst_spearman"] = None
        out["worst_topk_jaccard"] = None
        return out
    k = max(1, int(round(len(nodes) * k_frac)))

    def top(row: List[float]) -> set:
        return set(sorted(range(len(row)), key=lambda i: (-row[i], i))[:k])

    rhos, jacc = [], []
    for i in range(len(table)):
        for j in range(i + 1, len(table)):
            rho = spearmanr(table[i], table[j]).correlation
            if not math.isnan(rho):
                rhos.append(float(rho))
            a, b = top(table[i]), top(table[j])
            jacc.append(len(a & b) / len(a | b))
    out["worst_spearman"] = min(rhos) if rhos else None
    out["worst_topk_jaccard"] = min(jacc)
    return out
