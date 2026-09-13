#!/usr/bin/env python3
"""
reproduce/render_results_figure.py — the manuscript's results-at-a-glance figure
===============================================================================

Writes ``docs/research/jss/latex/figures/Figure_3.{png,pdf}``: three panels that
carry the study's headline numbers and its principal limitation side by side.

    A. Out-of-distribution generalisation — LOSO Spearman rho per variant.
    B. Critical-set detection — F1@K per variant.
    C. Inter-oracle rank agreement, with the chance baseline drawn in.

Everything is read from committed artifacts:

    results/loso_all_variants_v4.json -> panels A and B
    results/convergent_validity.json -> panel C

**Nothing here is hardcoded, and that is deliberate.** An earlier version of this
figure circulated with the pooled-population numbers transcribed by hand; when the
evaluation population was corrected, the figure kept printing the superseded values
(including a negative RM bar that no longer exists). A figure that reads its own
artifact cannot disagree with the table built from the same artifact.

Panel A is sorted on the values actually read, not on a fixed variant order, so the
ordering shown is whatever the data says -- currently the untrained Topo-QoS
baseline places second, above the base typed model.

Usage
-----
    PYTHONPATH=. python reproduce/render_results_figure.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saag.evaluation import variant_registry as _registry

RESULTS_DIR = Path("results")
_DEFAULT_OUTPUT = Path("docs/research/jss/latex/figures/Figure_3")

#: Printed labels, and the colour family each variant belongs to. Labels come
#: from saag/evaluation/variant_registry.py under the "loso" harness — this
#: figure reads results/loso_all_variants_v4.json, where gl_qos is run on the
#: native graph and is therefore reported as GAT-N-QoS.
_VARIANTS: List[Tuple[str, str, str]] = [
    ("hgl_qos",       f"{_registry.label('hgl_qos', 'loso')} (Typed Heterogeneous)", "learned_best"),
    ("gl_qos",        f"{_registry.label('gl_qos', 'loso')} (Homogeneous GNN)",      "homogeneous"),
    ("topo_qos",      f"{_registry.label('topo_qos', 'loso')} (Weighted)",           "untrained"),
    ("topo_baseline", f"{_registry.label('topo_baseline', 'loso')} (Centrality)",    "untrained"),
    ("topology_rm",   f"{_registry.label('topology_rm', 'loso')} / $Q(v)$ (diagnostic reference)", "diagnostic"),
]

_COLOURS = {
    "learned_best": "#D68910",
    "learned":      "#1F6FB4",
    "homogeneous":  "#C77DBA",
    "untrained":    "#8A8A8A",
    # RM/Q(v) is the diagnostic pathway (Section 1.2), not a competing ranking
    # predictor -- same neutral colour as the untrained baselines, but hatched
    # so a reader can tell at a glance it is not part of that comparison.
    "diagnostic":   "#8A8A8A",
}

#: Families rendered with a hatch pattern instead of a solid fill, to mark
#: bars that are not part of the baseline/predictor ranking comparison.
_HATCH = {"diagnostic": "///"}

_ORACLE_LABELS = {
    "i_dyn__i_star":  "MessageFlow vs. FaultInjector\n($I_{dyn}$ vs. $I^*$)",
    "i_comp__i_dyn":  "FailureSim vs. MessageFlow\n($I_{comp}$ vs. $I_{dyn}$)",
    "i_comp__i_star": "FailureSim vs. FaultInjector\n($I_{comp}$ vs. $I^*$)",
}
_ORACLE_COLOURS = {
    "i_dyn__i_star":  "#C0392B",
    "i_comp__i_dyn":  "#C89060",
    "i_comp__i_star": "#159C7B",
}


def _load_variants(path: Path) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    table = json.loads(path.read_text()).get("comparison_table", {})
    if not table:
        raise SystemExit(f"no comparison_table in {path}")

    population = next(
        (table[k].get("eval_population") for k, _, _ in _VARIANTS
         if k in table and table[k].get("eval_population")), None)

    rows = []
    for key, label, family in _VARIANTS:
        r = table.get(key)
        if not r or r.get("mean_rho") is None:
            continue
        rows.append({
            "label": label,
            "family": family,
            "rho": float(r["mean_rho"]),
            "std": float(r.get("std_rho") or 0.0),
            "f1": float(r.get("mean_f1") or 0.0),
        })
    if not rows:
        raise SystemExit(f"no usable variant rows in {path}")
    return rows, population


def _load_oracles(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[str]]:
    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    per_scenario = data.get("per_scenario", [])

    rows = []
    for pair in ("i_dyn__i_star", "i_comp__i_dyn", "i_comp__i_star"):
        blk = summary.get(pair)
        if not blk or blk.get("mean_spearman_rho") is None:
            continue
        vals = [
            r["pairs"][pair]["spearman_rho"] for r in per_scenario
            if isinstance(r.get("pairs", {}).get(pair, {}).get("spearman_rho"), float)
        ]
        mean = float(blk["mean_spearman_rho"])
        rows.append({
            "key": pair,
            "label": _ORACLE_LABELS.get(pair, pair),
            "rho": mean,
            # Asymmetric whiskers: the observed range across scenarios is the
            # honest spread here, not a symmetric sigma around the mean.
            "lo": mean - min(vals) if vals else 0.0,
            "hi": max(vals) - mean if vals else 0.0,
        })

    jac = [
        blk.get("mean_topk_jaccard_tie_robust") or blk.get("mean_topk_jaccard")
        for blk in summary.values()
        if isinstance(blk, dict) and (
            blk.get("mean_topk_jaccard_tie_robust") or blk.get("mean_topk_jaccard"))
    ]
    chance = next(
        (blk.get("mean_topk_jaccard_random_baseline") for blk in summary.values()
         if isinstance(blk, dict) and blk.get("mean_topk_jaccard_random_baseline")), None)
    # Read the labeler's own floor from results/label_stability.json, which is
    # where it is measured, rather than from the copy convergent_validity.json
    # denormalises into itself at write time. That copy is only as fresh as the
    # convergent-validity run: refreshing label_stability alone left the figure
    # annotating a 12-fold result with a superseded floor, and nothing said so.
    ceiling = None
    stability = Path("results/label_stability.json")
    if stability.exists():
        try:
            ceiling = (json.loads(stability.read_text()).get("summary") or {}).get(
                "topk_jaccard_range")
        except (OSError, ValueError):
            ceiling = None
    if ceiling is None:
        ceiling = (data.get("self_agreement_ceiling") or {}).get("topk_jaccard_range")

    return rows, {
        "jaccard_min": min(jac) if jac else None,
        "jaccard_max": max(jac) if jac else None,
        "chance": chance,
        "ceiling": ceiling,
    }, data.get("eval_population")


def _footnote(j: Dict[str, Any]) -> str:
    if j["jaccard_min"] is None:
        return ""
    line = (f"Mean top-$K$ Jaccard overlap: {100*j['jaccard_min']:.1f}%"
            f"–{100*j['jaccard_max']:.1f}%")
    if j["chance"]:
        line += f"  (chance = {100*j['chance']:.1f}%)"
    if j["ceiling"]:
        lo, hi = j["ceiling"]
        line += (f"\nRead against the labeler's own seed-to-seed floor: "
                 f"{100*lo:.0f}%–{100*hi:.0f}%")
    return line


def _load_interaction(sig_path: Path):
    """The 2x2 interaction, read from the significance artifact.

    Returns None when the artifact is absent or predates the factorial block, so
    an older replication still renders the three original panels rather than
    failing. Panel D is drawn only from measured values -- nothing is recomputed
    here, so the figure and Table 8 cannot drift apart.
    """
    if not sig_path.exists():
        return None
    try:
        d = json.loads(sig_path.read_text())
    except Exception:                                   # noqa: BLE001
        return None
    row = next((r for r in (d.get("factorial") or [])
                if r.get("quantity") == "interaction"), None)
    if not row:
        return None
    per = row.get("per_fold_delta") or {}
    if not per:
        return None
    return {
        "per_fold": per,
        "mean": row["mean_delta"],
        "ci": row.get("delta_ci95") or (row["mean_delta"], row["mean_delta"]),
        "p": row["p"],
        "neg": sum(1 for v in per.values() if v < 0),
    }


def render(variants, oracles, jac, population, output: Path, dpi: int = 300, inter=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ((axa, axb), (axc, axd)) = plt.subplots(2, 2, figsize=(13.0, 9.6))

    pop = f" — {population} population" if population else ""
    fig.suptitle(
        f"Software-as-a-Graph (SaG) Pre-Deployment Reliability Evaluation{pop}",
        fontsize=15, fontweight="bold", y=0.99,
    )

    # ── A: LOSO rho, sorted on the data ──────────────────────────────────────
    a = sorted(variants, key=lambda r: r["rho"])
    ypos = range(len(a))
    bars_a = axa.barh(list(ypos), [r["rho"] for r in a],
             xerr=[r["std"] for r in a], capsize=3,
             color=[_COLOURS[r["family"]] for r in a], zorder=3)
    for bar, r in zip(bars_a, a):
        if r["family"] in _HATCH:
            bar.set_hatch(_HATCH[r["family"]])
            bar.set_edgecolor("#444")
    axa.set_yticks(list(ypos))
    axa.set_yticklabels([r["label"] for r in a], fontsize=10)
    axa.axvline(0.0, color="#555", linestyle="--", linewidth=1.0, zorder=2)
    axa.set_xlabel("Spearman Rank Correlation (Mean $\\rho$)", fontsize=11, fontweight="bold")
    axa.set_title("A. Out-of-Distribution Generalization\n(Leave-One-Scenario-Out $\\rho$)",
                  fontsize=12, fontweight="bold")
    # Labels sit clear of the whisker end, not inside the bar: an error bar drawn
    # over its own value label is unreadable at print size.
    a_right = max(r["rho"] + r["std"] for r in a)
    axa.set_xlim(min(0.0, min(r["rho"] - r["std"] for r in a)) - 0.04, a_right + 0.13)
    for i, r in enumerate(a):
        axa.text(r["rho"] + r["std"] + 0.02, i, f"{r['rho']:.3f}",
                 va="center", ha="left", fontsize=9.5, fontweight="bold", color="#111")

    # ── B: F1@K, same variant order as A so the eye can track a row ──────────
    bars_b = axb.barh(list(ypos), [r["f1"] for r in a],
             color=[_COLOURS[r["family"]] for r in a], zorder=3)
    for bar, r in zip(bars_b, a):
        if r["family"] in _HATCH:
            bar.set_hatch(_HATCH[r["family"]])
            bar.set_edgecolor("#444")
    axb.set_yticks(list(ypos))
    axb.set_yticklabels([])
    axb.set_xlabel("Critical-Set Detection ($F_1@K$)", fontsize=11, fontweight="bold")
    axb.set_title("B. Top-K Critical Component Search\n($F_1@K$, K = top 20%)",
                  fontsize=12, fontweight="bold")
    for i, r in enumerate(a):
        inside = r["f1"] > 0.09
        axb.text(r["f1"] - 0.01 if inside else r["f1"] + 0.01, i, f"{r['f1']:.3f}",
                 va="center", ha="right" if inside else "left",
                 fontsize=9.5, fontweight="bold",
                 color="white" if inside else "#111")

    # ── C: oracle agreement, with the chance line drawn ──────────────────────
    c = list(reversed(oracles))
    cpos = range(len(c))
    axc.barh(list(cpos), [r["rho"] for r in c],
             xerr=[[r["lo"] for r in c], [r["hi"] for r in c]], capsize=3,
             color=[_ORACLE_COLOURS.get(r["key"], "#888") for r in c], zorder=3)
    axc.set_yticks(list(cpos))
    axc.set_yticklabels([r["label"] for r in c], fontsize=9.5)
    axc.set_xlim(0, 1.18)
    axc.set_xlabel("Spearman Rank Correlation (Mean $\\rho$)", fontsize=11, fontweight="bold")
    axc.set_title("C. Simulation Oracle Agreement\n(Rank Convergence & Range Across Scenarios)",
                  fontsize=12, fontweight="bold")
    for i, r in enumerate(c):
        axc.text(r["rho"] + r["hi"] + 0.02, i, f"{r['rho']:.3f}", va="center", ha="left",
                 fontsize=9.5, fontweight="bold", color="#111")

    note = _footnote(jac)
    if note:
        axc.text(0.5, -0.30, note, transform=axc.transAxes, ha="center", va="top",
                 fontsize=9, style="italic",
                 bbox=dict(boxstyle="round,pad=0.45", facecolor="#FDF6E3",
                           edgecolor="#D9C89A", linewidth=0.8))

    # ── D: the typing x QoS interaction, per fold ────────────────────────────
    # The substitution claim is an interaction, so it is shown as one: how much
    # typing buys with the QoS channel present, minus how much it buys without.
    # Every fold below zero is the claim; a mix of signs would refute it.
    if inter:
        deltas = sorted(inter["per_fold"].values())
        ypos = range(len(deltas))
        axd.barh(list(ypos), deltas, color="#C0392B", alpha=0.85,
                 edgecolor="#7B241C", linewidth=0.6, zorder=3)
        axd.axvline(0, color="#111", linewidth=1.1, zorder=4)
        axd.axvline(inter["mean"], color="#1F6FB4", linewidth=1.6,
                    linestyle="--", zorder=5,
                    label=f"mean {inter['mean']:+.3f}")
        axd.axvspan(inter["ci"][0], inter["ci"][1], color="#1F6FB4", alpha=0.12,
                    zorder=1, label="95% CI")
        axd.set_yticks(list(ypos))
        axd.set_yticklabels(
            [k.replace("_system", "").replace("_", " ")
             for k in sorted(inter["per_fold"], key=inter["per_fold"].get)],
            fontsize=8.5)
        axd.set_xlabel(r"$(\rho_{HGT\text{-}QoS}-\rho_{GAT\text{-}N\text{-}QoS})-"
                       r"(\rho_{HGT}-\rho_{GAT\text{-}N})$", fontsize=10)
        axd.set_title("D. Typing $\\times$ QoS Interaction\n"
                      f"({inter['neg']}/{len(deltas)} folds negative, "
                      f"$p$ = {inter['p']:.4f})",
                      fontsize=13, fontweight="bold")
        axd.legend(fontsize=8.5, loc="lower left", framealpha=0.9)
    else:
        axd.axis("off")

    for ax in (axa, axb, axc, axd):
        ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = output.with_suffix(f".{ext}")
        plt.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close()


def _find_default_loso_path() -> Path:
    candidates = [
        RESULTS_DIR / "loso_all_variants_v5.json",
        RESULTS_DIR / "loso_all_variants_v4.json",
        RESULTS_DIR / "loso_all_variants_v3.json",
        RESULTS_DIR / "loso_all_variants_v2.json",
        RESULTS_DIR / "loso_all_variants.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    found = sorted(RESULTS_DIR.glob("loso_all_variants*.json"), reverse=True)
    if found:
        return found[0]
    return RESULTS_DIR / "loso_all_variants_v5.json"


def parse_args():
    p = argparse.ArgumentParser(description="Three-panel results figure (Figure 5)")
    p.add_argument(
        "--loso", type=Path, default=None,
        help="Path to LOSO results JSON (default: latest found in results/loso_all_variants*.json)",
    )
    p.add_argument("--oracles", type=Path, default=RESULTS_DIR / "convergent_validity.json")
    p.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    loso_path = args.loso if args.loso is not None else _find_default_loso_path()

    if not loso_path.exists():
        print(f"Error: LOSO results file not found: {loso_path}", file=sys.stderr)
        print("Run reproduce/loso_all_variants.py first to generate the artifact.", file=sys.stderr)
        return 1

    if not args.oracles.exists():
        print(f"Error: Convergent validity results file not found: {args.oracles}", file=sys.stderr)
        print("Run reproduce/convergent_validity.py first to generate the artifact.", file=sys.stderr)
        return 1

    variants, pop_a = _load_variants(loso_path)
    oracles, jac, pop_c = _load_oracles(args.oracles)

    # Both panels must describe the same population or the figure is comparing
    # measurements that are not comparable -- say so rather than drawing it.
    if pop_a and pop_c and pop_a != pop_c:
        print(f"  WARNING: panels A/B are '{pop_a}' but panel C is '{pop_c}'; "
              f"these are different populations.")

    print(f"Figure 5 from {loso_path} and {args.oracles} (population: {pop_a})")
    for r in sorted(variants, key=lambda r: -r["rho"]):
        print(f"  {r['label']:28s} rho={r['rho']:.4f}  F1@K={r['f1']:.4f}")
    for r in oracles:
        print(f"  {r['key']:16s} rho={r['rho']:.4f}")

    inter = _load_interaction(RESULTS_DIR / "loso_significance_v5.json")
    if inter:
        print(f"  interaction      mean={inter['mean']:+.4f} "
              f"({inter['neg']}/{len(inter['per_fold'])} folds negative, p={inter['p']:.4f})")
    render(variants, oracles, jac, pop_a, args.output, dpi=args.dpi, inter=inter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
