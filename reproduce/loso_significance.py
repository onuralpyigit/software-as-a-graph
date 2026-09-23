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

from reproduce._provenance import stamp
from saag.evaluation import variant_registry as _registry

BASELINE = _registry.PREREGISTERED_BASELINE
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

#: SaG-Hybrid (PREREGISTRATION.md Amendment 5): primary against the
#: closed-form engine it corrects, secondary against the learned engine it
#: extends. Holm-corrected across these two only. Skipped when the artifact
#: lacks the arm, so existing sweeps reproduce their previous output.
HYBRID_CONTRASTS = (
    ("hgl_qos_prior", "topo_qos", "hybrid_vs_closed_form"),
    ("hgl_qos_prior", "hgl_qos", "hybrid_vs_learned"),
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


#: The 2x2 design underlying the four architecture contrasts. Factors are
#: T (relation typing) and Q (QoS edge channel); the four cells are the four
#: reported learned arms.
FACTORIAL_CELLS = {
    ("T0", "Q0"): "gl",          # GAT-N      -- untyped, no edge channel
    ("T1", "Q0"): "hgl",         # HGT        -- typed, no edge channel
    ("T0", "Q1"): "gl_qos",      # GAT-N-QoS  -- untyped, edge channel
    ("T1", "Q1"): "hgl_qos",     # HGT-QoS    -- typed, edge channel
}


def factorial(
    table: Dict[str, Any],
    *,
    fisher_z: bool = False,
    cells: Optional[Dict[Tuple[str, str], str]] = None,
) -> List[Dict[str, Any]]:
    """The three orthogonal quantities of the 2x2, tested per fold.

    With ``fisher_z=True`` every cell's per-fold rho is transformed by
    ``arctanh`` before the contrasts are formed. This matters for exactly one
    of the three quantities. Spearman rho is bounded on [-1, 1] and its scale
    compresses as it approaches either end, so a difference of differences
    computed on raw rho is not scale-free: two mechanisms that each move a
    predictor toward the attainable ceiling will appear to interact
    sub-additively even when they contribute independently on any monotone
    rescaling. The interaction row is the paper's substitution claim, so it has
    to survive the transformation that removes that artefact. Main effects are
    reported on both scales for completeness; they are differences, not
    differences of differences, and are far less exposed to it.

    The transform is applied per fold per cell, never to an already-averaged
    rho. No cell in the shipped artifact reaches |rho| = 1, so arctanh is
    finite throughout; a cell that did would return inf and is guarded below.

    The four pairwise contrasts in ARCHITECTURE_CONTRASTS are *simple effects*,
    and they are algebraically linked: given the four cell means, any three
    determine the fourth, because

        (TQ - T0Q) - (TQ0 - T00)  ==  (TQ - TQ0) - (T0Q - T00)

    -- both sides are the interaction. Reporting the four as if they were four
    independent questions, and reading "one significant, one not" as evidence
    that the two differ, is the classic error of inferring an interaction from a
    difference in significance. It does not follow: a contrast at p = 0.0005 and
    one at p = 0.13 have not thereby been shown to differ from each other.

    So the correction belongs on the three quantities that *are* distinct: the
    two main effects and the interaction. The simple effects stay in the
    manuscript because they carry the narrative ("typing helps when the QoS
    channel is absent"), but they are reported descriptively, uncorrected, with
    the claim about their difference resting on the interaction row here.

    Main effects are averaged over the other factor's levels, which is what a
    main effect means in a balanced 2x2 -- not the simple effect at one level.

    ``cells`` maps each (T, Q) level to a variant id and defaults to
    FACTORIAL_CELLS. Passing the capacity- and channel-matched arms instead
    (gl_full_cap, hgl, gl_full_qos16_cap, hgl_qos) forms the matched 2x2 of
    PREREGISTRATION.md Amendment 2.
    """
    names = cells or FACTORIAL_CELLS
    cells = {k: _per_fold(table, v) for k, v in names.items()}
    if not all(cells.values()):
        return []
    folds = sorted(set.intersection(*(set(c) for c in cells.values())))
    if len(folds) < 2:
        return []

    t00 = np.array([cells[("T0", "Q0")][f] for f in folds])
    t10 = np.array([cells[("T1", "Q0")][f] for f in folds])
    t01 = np.array([cells[("T0", "Q1")][f] for f in folds])
    t11 = np.array([cells[("T1", "Q1")][f] for f in folds])

    if fisher_z:
        for name, arr in ((names[("T0", "Q0")], t00), (names[("T1", "Q0")], t10),
                          (names[("T0", "Q1")], t01), (names[("T1", "Q1")], t11)):
            if np.any(np.abs(arr) >= 1.0):
                raise ValueError(
                    f"{name} carries |rho| = 1 on some fold; arctanh is undefined "
                    "there and the z-scale contrast cannot be formed."
                )
        t00, t10, t01, t11 = (np.arctanh(a) for a in (t00, t10, t01, t11))

    quantities = (
        ("main_typing", "Typing (main effect)", ((t10 - t00) + (t11 - t01)) / 2.0),
        ("main_qos", "QoS edge channel (main effect)", ((t01 - t00) + (t11 - t10)) / 2.0),
        # Negative => typing buys less when the QoS channel is already present,
        # i.e. the two mechanisms substitute rather than compose.
        ("interaction", "Typing x QoS interaction", (t11 - t01) - (t10 - t00)),
    )

    out: List[Dict[str, Any]] = []
    for key, label, diff in quantities:
        n = len(folds)
        try:
            stat, p = wilcoxon(diff)
        except ValueError:                     # all-zero differences
            stat, p = float("nan"), 1.0
        out.append({
            "quantity": f"{key}_z" if fisher_z else key,
            "label": f"{label}, Fisher z" if fisher_z else label,
            "role": "factorial_fisher_z" if fisher_z else "factorial",
            "scale": "fisher_z" if fisher_z else "spearman_rho",
            "not_preregistered": True,
            "n_folds": n,
            "mean_delta": float(diff.mean()),
            "wins": int((diff > 0).sum()),
            "losses": int((diff < 0).sum()),
            "W": float(stat),
            "p": float(p),
            "attainable_floor_p": attainable_floor(n),
            "delta_ci95": _bootstrap_delta_ci(diff),
            "per_fold_delta": {f: float(d) for f, d in zip(folds, diff)},
        })
    holm(out)
    return out


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
    ap.add_argument("--factorial-cells", default=None,
                    help="Comma-separated variant ids for the 2x2 cells in the "
                         "order T0Q0,T1Q0,T0Q1,T1Q1 (default: gl,hgl,gl_qos,hgl_qos).")
    args = ap.parse_args()

    cells = None
    if args.factorial_cells:
        ids = [v.strip() for v in args.factorial_cells.split(",")]
        if len(ids) != 4:
            print("Error: --factorial-cells needs exactly four variant ids.", file=sys.stderr)
            return 2
        cells = dict(zip((("T0", "Q0"), ("T1", "Q0"), ("T0", "Q1"), ("T1", "Q1")), ids))

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

    def _family(contrasts, role, correct=True):
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
        if out and correct:
            holm(out)
        return out

    factors = factorial(table, cells=cells)
    # The same three quantities on the variance-stabilised scale. Reported
    # beside the rho-scale block rather than instead of it: the manuscript
    # states the interaction in rho units, and the z-scale row is what says
    # whether that interaction is a property of the mechanisms or of the metric.
    factors_z = factorial(table, fisher_z=True, cells=cells)
    # Simple effects are reported descriptively: they are algebraically linked
    # to each other and to the interaction, so correcting across them would
    # treat one structural fact as four questions. The correction lives on the
    # three orthogonal quantities in `factors`.
    architecture = _family(ARCHITECTURE_CONTRASTS, "architecture", correct=False)
    controls = _family(CONTROL_CONTRASTS, "control")
    hybrid = _family(HYBRID_CONTRASTS, "hybrid")

    n = family[0]["n_folds"]
    print(f"\n  Pre-registered LOSO comparisons vs "
          f"{family[0]['baseline_label']}   (n = {n} folds)")
    print(f"  Attainable two-sided p floor at n={n}: {attainable_floor(n):.4f}")
    print(f"  Folds that may be lost and still reach a={args.alpha} "
          f"(best case, smallest |d|): {loss_budget(n, args.alpha)}")
    print("  " + "─" * 78)
    print(f"  {'variant':<12}{'role':<13}{'d rho':>9}{'wins':>7}{'W':>7}"
          f"{'p':>9}{'p_holm':>9}")
    for r in family + exploratory + architecture + controls + hybrid:  # noqa: E501
        holm_s = f"{r['p_holm']:.4f}" if "p_holm" in r else "—"
        print(f"  {r['label']:<12}{r['role']:<13}{r['mean_delta']:>+9.4f}"
              f"{r['wins']:>4}/{r['n_folds']:<2}{r['W']:>7.1f}"
              f"{r['p']:>9.4f}{holm_s:>9}")

    if factors:
        print(f"\n  2x2 factorial   [POST-HOC]   Holm-corrected across these three")
        print("  " + "\u2500" * 78)
        print(f"  {'quantity':<34}{'d rho':>9}{'wins':>7}{'W':>7}{'p':>9}{'p_holm':>9}")
        for r in factors:
            print(f"  {r['label']:<34}{r['mean_delta']:>+9.4f}"
                  f"{r['wins']:>4}/{r['n_folds']:<2}{r['W']:>7.1f}"
                  f"{r['p']:>9.4f}{r['p_holm']:>9.4f}")
        ci = next(r for r in factors if r["quantity"] == "interaction")["delta_ci95"]
        print(f"  interaction bootstrap 95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}]")

    if factors_z:
        print(f"\n  2x2 factorial on Fisher z   [POST-HOC]   "
              f"rho is bounded; z is not")
        print("  " + "─" * 78)
        print(f"  {'quantity':<34}{'d z':>9}{'wins':>7}{'W':>7}{'p':>9}{'p_holm':>9}")
        for r in factors_z:
            print(f"  {r['label']:<34}{r['mean_delta']:>+9.4f}"
                  f"{r['wins']:>4}/{r['n_folds']:<2}{r['W']:>7.1f}"
                  f"{r['p']:>9.4f}{r['p_holm']:>9.4f}")
        ciz = next(r for r in factors_z
                   if r["quantity"] == "interaction_z")["delta_ci95"]
        print(f"  interaction bootstrap 95% CI (z): [{ciz[0]:+.4f}, {ciz[1]:+.4f}]")

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
        # The 2x2's three orthogonal quantities, Holm-corrected across them.
        # This is where the substitution claim is tested.
        "factorial": factors,
        # The same three quantities with each fold's rho passed through
        # arctanh first. Spearman rho is bounded, so a difference of
        # differences on it can read as sub-additive purely because the cells
        # sit at different points on a compressing scale. This block is what
        # distinguishes a mechanism-level interaction from a metric artefact.
        "factorial_fisher_z": factors_z,
        # Post-hoc simple effects (RQ2 typing, RQ3 QoS edge ablation), reported
        # descriptively: algebraically linked, so not separately corrected.
        "architecture": architecture,
        # Post-hoc RQ2 confound controls; Holm-corrected within this list only.
        "rq2_controls": controls,
        "qos_stratified_ablation": stratified,
    }
    if hybrid:
        # Amendment 5's registered pair, Holm-corrected within itself.
        payload["hybrid"] = hybrid
    if cells:
        payload["factorial_cells"] = {f"{t}{q}": v for (t, q), v in cells.items()}
    # Say which commit and which corpus produced these contrasts. Without it
    # reconcile_manuscript.py falls back to comparing timestamps, and a table's
    # licensing statistics are the last place that should be guesswork.
    payload["provenance"] = stamp(input=str(args.input), alpha=args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\n  Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
