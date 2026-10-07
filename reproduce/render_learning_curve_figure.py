#!/usr/bin/env python3
"""
reproduce/render_learning_curve_figure.py — manuscript Figure 6 (Amendment 19, LC)
==================================================================================

LOSO Spearman rho against I* as a function of the number K of training scenarios per
fold, one line per learner with its 95% bootstrap interval over folds, and the
direct-dependent count (afferent coupling, InDeg) as a flat reference. Reads
data/benchmarks/referee_round14_lc.json; trains nothing.

Usage:
    PYTHONPATH=. python reproduce/render_learning_curve_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "benchmarks" / "referee_round14_lc.json"
OUT = ROOT / "docs" / "research" / "jss" / "latex" / "figures" / "Figure_6"

INK, INK2, GRID = "#1F2937", "#475569", "#E5E7EB"
#: Okabe-Ito hues, validated as a categorical set (light surface); GAT-P-QoS keeps
#: its colour from Figures 4 and 5. Markers differ too, so identity is not colour alone.
SERIES = [("GAT-P-QoS", "#7B3F8C", "o"), ("GIN-P-QoS", "#009E73", "s"), ("GAT-QoS", "#D55E00", "^")]
KS = ("1", "2", "4", "8", "11")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})


def main() -> None:
    d = json.loads(SRC.read_text())
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = [int(k) for k in KS]
    indeg = d["indeg_mean"]
    ax.axhline(indeg, color=INK2, lw=1.2, ls=(0, (4, 2)), zorder=1)
    ax.text(1.0, indeg + 0.006, "afferent coupling (InDeg, reference)", va="bottom", ha="left",
            color=INK2, fontsize=6.5)
    for name, colour, marker in SERIES:
        m = d["means"][name]
        y = [m[k]["mean"] for k in KS]
        lo = [m[k]["ci95"][0] for k in KS]
        hi = [m[k]["ci95"][1] for k in KS]
        ax.fill_between(x, lo, hi, color=colour, alpha=0.08, lw=0, zorder=2)
        ax.plot(x, y, color=colour, lw=2, marker=marker, ms=4.5, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8, label=name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels(KS)
    ax.set_xlim(0.85, 12.5)
    ax.set_xlabel("Training scenarios per fold, $K$")
    ax.set_ylabel("LOSO Spearman $\\rho$ against $I^*$")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}.pdf/.png")


if __name__ == "__main__":
    main()
