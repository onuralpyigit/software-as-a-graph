#!/usr/bin/env python3
"""
reproduce/loso_significance.py — the pre-registered tests, with their own floor
==============================================================================

Reads a `results/loso_all_variants.json`-shaped artifact and reports the
comparisons `docs/research/jss/PREREGISTRATION.md` commits to:

  * primary   — hgl_qos (HGT-QoS) vs topo_qos (Topo-QoS)
  * secondary — hgl     (HGT)     vs topo_qos

two-sided Wilcoxon signed-rank over folds, Holm-corrected across the two, each
printed next to the **attainable-p floor** for the realised fold count.

That floor is the part these tables have historically left out. The signed-rank
statistic is discrete: at n=8 the smallest two-sided p any result can produce is
0.0078, so a p of 0.11 does not mean "nearly significant", it means the design
tolerated a loss it cannot afford. Printing the floor next to the p-value makes
an underpowered comparison legible as underpowered rather than as a near miss.

Every other variant pair is printed too, marked exploratory.

Usage:
    PYTHONPATH=. python reproduce/loso_significance.py
    PYTHONPATH=. python reproduce/loso_significance.py \\
        --input results/loso_all_variants_v2.json --output results/loso_significance.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import wilcoxon

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saag.evaluation import variant_registry as _registry

BASELINE = "topo_qos"
PRIMARY = ("hgl_qos", BASELINE)
SECONDARY = ("hgl", BASELINE)

#: RQ2 confound controls, each pairing HGT-QoS against an arm that differs from
#: it in exactly one respect (PREREGISTRATION.md, Amendment 2). Post-hoc and so
#: Holm-corrected inside their own family -- pooling them with the two
#: pre-registered contrasts would penalise those for questions asked later.
#: Each is skipped silently when the artifact lacks the arm, so a sweep that
#: ran only the manuscript variants produces exactly its previous output.
CONTROL_CONTRASTS = (
    ("hgl_qos", "gl_full_qos_cap", "capacity"),
    ("hgl_qos", "gl_full_qos16_cap", "edge_channel"),
    ("hgl_qos", "hgl_qos_uni", "directionality"),
    ("hgl", "gl_full_cap", "capacity_qos_off"),
)

#: The architecture contrasts the manuscript actually headlines: RQ2 is a
#: typed-vs-untyped comparison and RQ3's ablation is QoS-on vs QoS-off, and
#: neither is a comparison against BASELINE. Both were reported with p-values
#: that appeared in no significance artifact under any version, because the
#: exploratory block below only ever pairs a variant against topo_qos. They are
#: not pre-registered, so they are Holm-corrected inside their own family for
#: the same reason the controls are: pooling questions asked later with the two
#: pre-registered contrasts would penalise those for the later questions.
ARCHITECTURE_CONTRASTS = (
    ("hgl_qos", "gl_qos", "typing_qos"),
    ("hgl", "gl", "typing_unweighted"),
    ("hgl_qos", "hgl", "qos_edge_typed"),
    ("gl_qos", "gl", "qos_edge_untyped"),
)


def attainable_floor(n: int) -> float:
    """Smallest two-sided p a signed-rank test on ``n`` pairs can produce."""
    if n < 1:
        return float("nan")
    return float(wilcoxon(np.arange(1, n + 1, dtype=float),
                          np.zeros(n)).pvalue)


def loss_budget(n: int, alpha: float = 0.05) -> int:
    """How many folds may be lost and still reach ``alpha`` — best case.

    Best case means the lost folds are the *smallest* |Δ| in the set. A large
    loss costs far more rank mass, so this is an upper bound on what the design
    tolerates, not a promise.
    """
    budget = 0
    for k in range(0, n + 1):
        d = np.arange(1, n + 1, dtype=float)
        d[:k] *= -1
        if wilcoxon(d).pvalue < alpha:
            budget = k
        else:
            break
    return budget


def _per_fold(table: Dict[str, Any], variant: str) -> Dict[str, float]:
    entry = table.get(variant)
    if not entry:
        return {}
    return {f["holdout"]: f["mean_rho"] for f in entry.get("per_fold", [])
            if f.get("mean_rho") is not None}


def _bootstrap_delta_ci(
    diff: np.ndarray, b: int = 2000, alpha: float = 0.05, seed: int = 42
) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI for the mean per-fold delta.

    The LOSO table has never carried one: ``reproduce/loso_all_variants.py``
    computes no interval, and EXPERIMENTS.md 2.D's claim of bootstrap CIs is
    true only of the in-distribution table. Resampling is over folds, matching
    the unit of analysis the signed-rank test uses.

    The folds are not independent -- any two LOSO models share 10 of their 11
    training graphs -- so this interval, like the p-value beside it, understates
    the true dispersion. It is reported as the conventional summary, not as an
    unbiased one.
    """
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n < 2:
        return (float("nan"), float("nan"))
    means = diff[rng.integers(0, n, size=(b, n))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def compare(table: Dict[str, Any], a: str, b: str) -> Optional[Dict[str, Any]]:
    """Paired comparison of variant ``a`` against variant ``b`` over folds."""
    pa, pb = _per_fold(table, a), _per_fold(table, b)
    folds = sorted(set(pa) & set(pb))
    if len(folds) < 2:
        return None

    xa = np.array([pa[f] for f in folds])
    xb = np.array([pb[f] for f in folds])
    diff = xa - xb
    n = len(folds)
    try:
        stat, p = wilcoxon(xa, xb)
    except ValueError:      # all-zero differences
        stat, p = float("nan"), 1.0

    return {
        "variant": a,
        "baseline": b,
        "label": _registry.label(a, harness="loso"),
        "baseline_label": _registry.label(b, harness="loso"),
        "n_folds": n,
        "mean_delta": float(diff.mean()),
        "wins": int((diff > 0).sum()),
        "losses": int((diff < 0).sum()),
        "W": float(stat),
        "p": float(p),
        "attainable_floor_p": attainable_floor(n),
        "loss_budget_at_005": loss_budget(n),
        "delta_ci95": _bootstrap_delta_ci(diff),
        "per_fold_delta": {f: float(d) for f, d in zip(folds, diff)},
    }


def stratified_qos_ablation(
    table: Dict[str, Any], diagnostic_path: Path
) -> Optional[Dict[str, Any]]:
    """QoS ablation (HGT-QoS - HGT) split by whether the holdout's QoS varies.

    EXPLORATORY. This split was not pre-registered: it was motivated by
    ``reproduce/qos_corpus_diagnostic.py``, which found that 7 of the 12
    scenarios declare an identical QoS triple on every topic, so the 7 QoS
    edge-feature dimensions are a constant offset on those graphs and cannot
    discriminate between their components. Pooling them with the graphs where
    QoS does vary averages a treatment over folds where it is inert.

    No significance test is reported per stratum, and deliberately so: at n = 4
    the attainable two-sided signed-rank floor is already above 0.05, so no
    arrangement of four folds can reach significance. The strata are descriptive.
    """
    if not diagnostic_path.exists():
        return None
    diag = json.loads(diagnostic_path.read_text())
    degenerate = set(diag.get("qos_degenerate_scenarios", []))
    if not degenerate:
        return None

    pa, pb = _per_fold(table, "hgl_qos"), _per_fold(table, "hgl")
    folds = sorted(set(pa) & set(pb))
    if not folds:
        return None

    strata: Dict[str, Dict[str, Any]] = {}
    for name, members in (
        ("qos_varying", [f for f in folds if f not in degenerate]),
        ("qos_degenerate", [f for f in folds if f in degenerate]),
        ("all", folds),
    ):
        if not members:
            continue
        diff = np.array([pa[f] - pb[f] for f in members])
        strata[name] = {
            "n_folds": len(members),
            "folds": members,
            "mean_delta": float(diff.mean()),
            "wins": int((diff > 0).sum()),
            "losses": int((diff < 0).sum()),
            "attainable_floor_p": attainable_floor(len(members)),
            "per_fold_delta": {f: float(d) for f, d in zip(members, diff)},
        }

    return {
        "contrast": "hgl_qos - hgl",
        "label": "HGT-QoS - HGT (QoS edge-encoding ablation)",
        "role": "exploratory",
        "not_preregistered": True,
        "source": str(diagnostic_path),
        "dimensions_dead_in_every_scenario":
            diag.get("dimensions_dead_in_every_scenario", []),
        "strata": strata,
    }


def holm(results: List[Dict[str, Any]]) -> None:
    """Holm-Bonferroni over the pre-registered family, in place."""
    ordered = sorted(range(len(results)), key=lambda i: results[i]["p"])
    m = len(results)
    running = 0.0
    for rank, idx in enumerate(ordered):
        adj = min(1.0, (m - rank) * results[idx]["p"])
        running = max(running, adj)      # monotone by construction
        results[idx]["p_holm"] = running


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-registered LOSO significance tests.")
    ap.add_argument("--input", type=Path,
                    default=Path("results/loso_all_variants.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("results/loso_significance.json"))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--stratify", type=Path,
                    default=Path("results/qos_corpus_diagnostic.json"),
                    help="qos_corpus_diagnostic.json used to split the QoS "
                         "ablation into QoS-varying and QoS-degenerate strata. "
                         "Reported as exploratory; pass a non-existent path to "
                         "suppress the block.")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found.", file=sys.stderr)
        return 2

    table = json.loads(args.input.read_text())["comparison_table"]

    family = [compare(table, *PRIMARY), compare(table, *SECONDARY)]
    family = [r for r in family if r]
    if not family:
        print("Error: neither pre-registered comparison is present in the artifact.",
              file=sys.stderr)
        return 2
    holm(family)
    for r, role in zip(family, ("primary", "secondary")):
        r["role"] = role

    exploratory = [
        r for v in table
        if v not in {PRIMARY[0], SECONDARY[0], BASELINE}
        and (r := compare(table, v, BASELINE))
    ]
    for r in exploratory:
        r["role"] = "exploratory"

    def _family(contrasts, role):
        out = []
        for a, b, what in contrasts:
            if a not in table or b not in table:
                continue
            r = compare(table, a, b)
            if not r:
                continue
            r["role"] = role
            r["controls_for"] = what
            r["not_preregistered"] = True
            out.append(r)
        # Each family is corrected within itself, per Amendment 2.
        if out:
            holm(out)
        return out

    architecture = _family(ARCHITECTURE_CONTRASTS, "architecture")
    controls = _family(CONTROL_CONTRASTS, "control")

    n = family[0]["n_folds"]
    print(f"\n  Pre-registered LOSO comparisons vs "
          f"{family[0]['baseline_label']}   (n = {n} folds)")
    print(f"  Attainable two-sided p floor at n={n}: {attainable_floor(n):.4f}")
    print(f"  Folds that may be lost and still reach a={args.alpha} "
          f"(best case, smallest |d|): {loss_budget(n, args.alpha)}")
    print("  " + "─" * 78)
    print(f"  {'variant':<12}{'role':<13}{'d rho':>9}{'wins':>7}{'W':>7}"
          f"{'p':>9}{'p_holm':>9}")
    for r in family + exploratory + architecture + controls:
        holm_s = f"{r['p_holm']:.4f}" if "p_holm" in r else "—"
        print(f"  {r['label']:<12}{r['role']:<13}{r['mean_delta']:>+9.4f}"
              f"{r['wins']:>4}/{r['n_folds']:<2}{r['W']:>7.1f}"
              f"{r['p']:>9.4f}{holm_s:>9}")

    print("\n  Per-fold deltas (primary):")
    for fold, d in sorted(family[0]["per_fold_delta"].items(),
                          key=lambda kv: kv[1]):
        print(f"    {fold:<32}{d:>+8.4f}")

    stratified = stratified_qos_ablation(table, args.stratify)
    if stratified:
        print("\n  QoS edge-encoding ablation, stratified   [EXPLORATORY, "
              "not pre-registered]")
        print(f"  Split source: {stratified['source']}")
        dead = stratified["dimensions_dead_in_every_scenario"]
        if dead:
            print(f"  Constant in ALL scenarios: {', '.join(dead)}")
        print("  " + "─" * 78)
        print(f"  {'stratum':<18}{'n':>4}{'d rho':>10}{'wins':>8}"
              f"{'floor p':>10}   reachable at a=0.05?")
        for name in ("qos_varying", "qos_degenerate", "all"):
            st = stratified["strata"].get(name)
            if not st:
                continue
            reach = "yes" if st["attainable_floor_p"] < args.alpha else "NO"
            print(f"  {name:<18}{st['n_folds']:>4}{st['mean_delta']:>+10.4f}"
                  f"{st['wins']:>5}/{st['n_folds']:<2}"
                  f"{st['attainable_floor_p']:>10.4f}   {reach}")
        print("  No per-stratum significance test is reported: see the "
              "docstring of stratified_qos_ablation().")

    payload = {
        "input": str(args.input),
        "alpha": args.alpha,
        "n_folds": n,
        "attainable_floor_p": attainable_floor(n),
        "loss_budget_at_alpha": loss_budget(n, args.alpha),
        "preregistered": family,
        "exploratory": exploratory,
        # Post-hoc architecture contrasts (RQ2 typing, RQ3 QoS edge ablation);
        # Holm-corrected within this list only.
        "architecture": architecture,
        # Post-hoc RQ2 confound controls; Holm-corrected within this list only.
        "rq2_controls": controls,
        "qos_stratified_ablation": stratified,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\n  Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
