#!/usr/bin/env python3
"""
reproduce/render_graphical_abstract.py — JSS graphical abstract
================================================================

A 13 x 5 cm graphical abstract (the Guide's 1328 x 531 px, exceeded at 300 dpi):
(1) a pub-sub topology and the dependency edges derived from it, (2) the predictors on
the reachability oracle against reference rankings that restate that oracle's first
wave (Amendment 13: dependency counts are references, not predictors), (3) what the
confirmatory tests, the other two oracles and the cost measurements add. Every number
is read from a committed artifact.

Usage:
    PYTHONPATH=. python reproduce/render_graphical_abstract.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "benchmarks"
OUT = ROOT / "docs" / "research" / "jss" / "latex" / "figures" / "graphical_abstract"

INK, INK2, SURFACE = "#1F2937", "#475569", "#FCFCFB"
#: Highlight pattern: one accent for the best predictor, a neutral for the others
#: (validated: CVD and normal-vision separation and contrast pass; the neutral's
#: chroma is intentionally below the categorical floor).
ACCENT, NEUTRAL = "#0072B2", "#8C939E"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def _numbers() -> dict:
    tf = json.loads((BENCH / "tf_baselines.json").read_text())["summary"]
    dg = json.loads((ROOT / "results" / "dependency_graph_contrasts.json").read_text())["means"]
    raw = json.loads((BENCH / "referee_round7_raw_baselines.json").read_text())["summary"]
    rec = json.loads((BENCH / "referee_round7_recall.json").read_text())["curves"]
    ioe = json.loads((ROOT / "results" / "independent_oracle_evaluation.json").read_text())["summary"]
    hyb = json.loads((ROOT / "results" / "loso_hybrid_gat_cpu.json").read_text())["comparison_table"]
    lrn = json.loads((BENCH / "referee_round7_learned_oracles.json").read_text())
    return {
        "Topo-QoS": tf["Topo-QoS"]["loso_mean_rho"],
        "Hybrid-GAT": hyb["gl_qos16_prior"]["mean_rho"],
        "GAT-P-QoS": dg["gl_proj_qos16_cap"]["loso_mean_rho"],
        "InDeg": tf["InDeg"]["loso_mean_rho"],
        "Analytic-I*": ioe["i_star"]["Analytic-I*"]["mean_rho"],
        "comp_topo": raw["i_comp"]["Topo-QoS"]["loso"]["mean"],
        "comp_learned": max(v["summary"]["i_comp"]["rho"]["mean"] for k, v in lrn.items()
                            if isinstance(v, dict) and "summary" in v and k != "GAT-P+InDeg"),
        "margin": rec["i_star"]["GAT-P-QoS"]["safety_margin"]["0.80"],
    }


def _node(ax, xy, text, colour=INK2, fill="white", r=0.055):
    ax.add_patch(plt.Circle(xy, r, facecolor=fill, edgecolor=colour, lw=0.9, zorder=3))
    ax.text(*xy, text, ha="center", va="center", fontsize=6, color=INK, zorder=4)


def _arrow(ax, a, b, colour, ls="-", lw=0.9):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, color=colour, lw=lw,
                                 ls=ls, shrinkA=6, shrinkB=6, zorder=2))


def panel_graph(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("1. Derive dependencies", loc="left", fontsize=7.4, fontweight="bold", color=INK)
    a1, t, a2, a3 = (0.15, 0.55), (0.5, 0.55), (0.85, 0.8), (0.85, 0.3)
    _node(ax, a1, "a1")
    _node(ax, a2, "a2")
    _node(ax, a3, "a3")
    ax.add_patch(plt.Rectangle((t[0] - 0.05, t[1] - 0.045), 0.1, 0.09, facecolor="white",
                               edgecolor=INK2, lw=0.9, zorder=3))
    ax.text(*t, "t", ha="center", va="center", fontsize=6, color=INK, zorder=4)
    _arrow(ax, a1, t, INK2)
    _arrow(ax, a2, t, INK2)
    _arrow(ax, a3, t, INK2)
    for s in (a2, a3):  # derived DEPENDS_ON: subscriber -> publisher
        ax.add_patch(FancyArrowPatch(s, a1, arrowstyle="-|>", mutation_scale=6, color=ACCENT, lw=1.2,
                                     connectionstyle="arc3,rad=0.35" if s == a2 else "arc3,rad=-0.35",
                                     shrinkA=6, shrinkB=6, zorder=2))
    ax.text(0.5, 0.06, "publish / subscribe (grey)\nderived DEPENDS_ON (blue)\nInDeg(a1) = 2: the oracle's first wave",
            ha="center", va="bottom", fontsize=5.6, color=INK2, linespacing=1.25)


def panel_bars(ax, n: dict) -> None:
    rows = [("Training-free baseline", n["Topo-QoS"], NEUTRAL),
            ("Hybrid GNN", n["Hybrid-GAT"], NEUTRAL),
            ("GNN, dependency graph", n["GAT-P-QoS"], ACCENT)]
    ys = range(len(rows))
    ax.barh(list(ys), [v for _, v, _ in rows], color=[c for *_, c in rows], height=0.42)
    for y, (name, v, _) in zip(ys, rows):
        ax.text(0.0, y + 0.3, name, va="bottom", ha="left", fontsize=5.8, color=INK)
        ax.text(v - 0.015, y, f"{v:.2f}", va="center", ha="right", fontsize=6, color="white")
    # References restate I*'s rule (Amendment 13): drawn as lines, not as predictors.
    for x, ls in ((n["InDeg"], (0, (3, 1.5))), (n["Analytic-I*"], (0, (1, 1.2)))):
        ax.axvline(x, color=INK2, lw=0.8, ls=ls, ymin=0.02, ymax=0.97)
    ax.set_facecolor(SURFACE)
    ax.set_ylim(-0.4, len(rows) - 0.2)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel(f"Spearman ρ, 12 synthetic architectures\nlines: references InDeg {n['InDeg']:.2f}, "
                  f"first-order I* {n['Analytic-I*']:.2f}", fontsize=5.3, color=INK2, linespacing=1.15)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("2. Predictors vs. references", loc="left", fontsize=7.4,
                 fontweight="bold", color=INK)


def panel_text(ax, n: dict) -> None:
    ax.axis("off")
    ax.set_title("3. Beyond one simulator", loc="left", fontsize=7.4, fontweight="bold", color=INK)
    lines = [
        "Dependency counts restate the\nsimulator: references, not predictors.",
        "Registered primary contrast null;\nhybrids beat baseline, not base learners.",
        "Queue-flow surrogate pays:\nGBM 0.80 vs 0.71; GNN fails (0.60).",
        f"80% of the critical set needs\nthe top {100 * n['margin']:.0f}% by the best GNN.",
        "One simulation: seconds; GNN\nfeature extraction: ~5.6x that.",
    ]
    for i, line in enumerate(lines):
        ax.text(0.0, 0.97 - i * 0.2, line, ha="left", va="top", fontsize=5.5, color=INK, linespacing=1.15)


def main() -> None:
    n = _numbers()
    cm = 1 / 2.54
    fig = plt.figure(figsize=(13 * cm, 5 * cm), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.05], wspace=0.22,
                          left=0.01, right=0.98, bottom=0.2, top=0.86)
    panel_graph(fig.add_subplot(gs[0]))
    panel_bars(fig.add_subplot(gs[1]), n)
    panel_text(fig.add_subplot(gs[2]), n)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), facecolor=SURFACE)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, facecolor=SURFACE)
    print(f"wrote {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
