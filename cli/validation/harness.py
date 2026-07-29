"""Methodological-guard harness over pre-computed Q(v) and I(v) JSON artifacts.

Independent of the graph pipeline: it reads prediction and ground-truth files
and reports the guards (stratification, rank displacement, seed spread) that
catch a pooled correlation hiding a per-stratum failure.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr


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
