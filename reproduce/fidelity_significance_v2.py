#!/usr/bin/env python3
"""
Statistical analysis for Study 2 (docs/research/fidelity/PREREGISTRATION_V2.md).

Evaluates:
  - H2.1 (Primary): Steady-state p50 latency error reduction, S0 - S2 (n=13, kappa=1)
  - H2.2a (Secondary, Holm): Infrastructure (broker+host) rank agreement rho(S2) - 0.85
  - H2.2b (Secondary, Holm): Infrastructure stall-set Jaccard J(S2) - 0.85
  - H2.3 (Secondary, Holm): Perturbed-counterpart delivery error reduction, S2 - S3
  - H2.4 (Secondary, Holm): Gray-failure (omission) impact error reduction vs crash ref

Usage:
    python3 reproduce/fidelity_significance_v2.py --input results/fidelity_study_v2/summary.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from scipy.stats import wilcoxon


def hodges_lehmann_ci(diffs: Sequence[float], alpha: float = 0.05) -> Tuple[float, Tuple[float, float]]:
    """Hodges-Lehmann median of Walsh averages and exact/bootstrap-free distribution-free CI."""
    xs = sorted(float(x) for x in diffs)
    n = len(xs)
    if n == 0:
        return float("nan"), (float("nan"), float("nan"))
    walsh = sorted(0.5 * (xs[i] + xs[j]) for i in range(n) for j in range(i, n))
    hl = statistics.median(walsh)
    if n < 5:
        return float(hl), (float(walsh[0]), float(walsh[-1]))
    # Exact Wilcoxon critical value via normal/exact tail on signed-rank sum
    # Find largest k such that P(W <= k) <= alpha / 2 for n
    total = 2 ** n
    if n <= 16:
        # Exact distribution of W+ over all 2^n sign assignments (no ties)
        max_w = n * (n + 1) // 2
        counts = [0] * (max_w + 1)
        counts[0] = 1
        for r in range(1, n + 1):
            for w in range(max_w, r - 1, -1):
                counts[w] += counts[w - r]
        cum = 0
        k = 0
        for w, c in enumerate(counts):
            cum += c
            if cum / total <= alpha / 2.0:
                k = w + 1
            else:
                break
        lo = walsh[max(0, min(k - 1, len(walsh) - 1))] if k > 0 else walsh[0]
        hi = walsh[max(0, min(len(walsh) - k, len(walsh) - 1))] if k > 0 else walsh[-1]
        return float(hl), (float(lo), float(hi))
    # Normal approximation for n > 16
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    k = max(1, int(math.floor(mu - 1.96 * sd)))
    return float(hl), (float(walsh[k - 1]), float(walsh[-k]))


def exact_wilcoxon(diffs: Sequence[float], alternative: str = "greater") -> float:
    """Exact Wilcoxon signed-rank p-value (handling all-zero diffs safely)."""
    nz = [float(x) for x in diffs if abs(float(x)) > 1e-12]
    if not nz:
        return 1.0
    res = wilcoxon(nz, alternative=alternative, method="exact")
    return float(res.pvalue)


def tost_wilcoxon(diffs: Sequence[float], delta: float = 0.05) -> float:
    """Two one-sided exact Wilcoxon tests for equivalence within [-delta, +delta]."""
    p_lo = exact_wilcoxon([x + delta for x in diffs], alternative="greater")
    p_hi = exact_wilcoxon([x - delta for x in diffs], alternative="less")
    return max(p_lo, p_hi)


def rank_biserial(diffs: Sequence[float]) -> float:
    nz = [float(x) for x in diffs if abs(float(x)) > 1e-12]
    if not nz:
        return 0.0
    # Simple rank-biserial r = (W+ - W-) / (W+ + W-)
    indexed = sorted(enumerate(nz), key=lambda kv: abs(kv[1]))
    w_pos = w_neg = 0.0
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and abs(abs(indexed[j + 1][1]) - abs(indexed[i][1])) <= 1e-12:
            j += 1
        avg_rank = 0.5 * (i + 1 + j + 1)
        for k in range(i, j + 1):
            if indexed[k][1] > 0:
                w_pos += avg_rank
            else:
                w_neg += avg_rank
        i = j + 1
    return float((w_pos - w_neg) / (w_pos + w_neg)) if (w_pos + w_neg) > 0 else 0.0


def holm_adjust(pvals: Mapping[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
    """Step-down Holm-Bonferroni correction."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, Dict[str, Any]] = {}
    rejected_so_far = True
    prev_adj = 0.0
    for idx, (key, p) in enumerate(items):
        thresh = alpha / (m - idx)
        adj = min(1.0, max(prev_adj, p * (m - idx)))
        prev_adj = adj
        sig = rejected_so_far and (p <= thresh)
        if not sig:
            rejected_so_far = False
        out[key] = {"raw_p": float(p), "holm_p": float(adj), "threshold": float(thresh), "significant": sig}
    return out


def analyze_study2(summary: Mapping[str, Any], delta_lat: float = 0.05, alpha: float = 0.05) -> Dict[str, Any]:
    """Run all Study 2 confirmatory tests from a scenario-level summary dict."""
    scenarios = summary.get("scenarios", {})

    # H2.1: Primary — steady-state p50 latency error reduction (S0 - S2), kappa == 1 only
    d_lat = [
        float(s["latency_p50_err_S0"]) - float(s["latency_p50_err_S2"])
        for s in scenarios.values()
        if abs(float(s.get("kappa", 1.0)) - 1.0) < 1e-6
        and s.get("latency_p50_err_S0") is not None
        and s.get("latency_p50_err_S2") is not None
    ]
    hl_lat, ci_lat = hodges_lehmann_ci(d_lat, alpha=alpha)
    p_sup = exact_wilcoxon(d_lat, alternative="greater")
    p_tost = tost_wilcoxon(d_lat, delta=delta_lat)
    p_two = exact_wilcoxon(d_lat, alternative="two-sided")
    if p_sup < alpha:
        h21_outcome = "Better"
    elif p_tost < alpha:
        h21_outcome = "Equivalent"
    elif hl_lat < 0 and ci_lat[1] < 0 and p_two < alpha:
        h21_outcome = "Worse"
    else:
        h21_outcome = "Inconclusive"

    # Secondary family (two-sided exact Wilcoxon, Holm-corrected)
    sec_diffs: Dict[str, List[float]] = {
        "H2.2a_infra_rho": [
            float(s["infra_rho_S2"]) - 0.85
            for s in scenarios.values()
            if s.get("infra_rho_S2") is not None
        ],
        "H2.2b_infra_stall_J": [
            float(s["infra_stall_J_S2"]) - 0.85
            for s in scenarios.values()
            if s.get("infra_stall_J_S2") is not None
        ],
        "H2.3_perturbed_cal_delivery": [
            float(s["perturbed_delivery_err_S2"]) - float(s["perturbed_delivery_err_S3"])
            for s in scenarios.values()
            if s.get("perturbed_delivery_err_S2") is not None
            and s.get("perturbed_delivery_err_S3") is not None
        ],
        "H2.4_gray_omission_err": [
            float(s["omission_err_crash_ref"]) - float(s["omission_err_S2"])
            for s in scenarios.values()
            if s.get("omission_err_crash_ref") is not None
            and s.get("omission_err_S2") is not None
        ],
    }

    raw_p = {k: exact_wilcoxon(v, alternative="two-sided") for k, v in sec_diffs.items() if v}
    holm = holm_adjust(raw_p, alpha=alpha)
    secondary_out: Dict[str, Any] = {}
    for k, vals in sec_diffs.items():
        if not vals:
            secondary_out[k] = {"n": 0, "skipped": True}
            continue
        hl, ci = hodges_lehmann_ci(vals, alpha=alpha)
        secondary_out[k] = {
            "n": len(vals),
            "hl": hl,
            "hl_ci": list(ci),
            "rank_biserial": rank_biserial(vals),
            **holm[k],
        }

    return {
        "H2.1_latency_p50": {
            "n": len(d_lat),
            "delta": delta_lat,
            "hl": hl_lat,
            "hl_ci": list(ci_lat),
            "rank_biserial": rank_biserial(d_lat),
            "p_superiority": p_sup,
            "p_tost": p_tost,
            "p_two_sided": p_two,
            "outcome": h21_outcome,
        },
        "secondary": secondary_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--delta", type=float, default=0.05)
    args = ap.parse_args()

    summary = json.loads(args.input.read_text())
    result = analyze_study2(summary, delta_lat=args.delta)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
