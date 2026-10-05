#!/usr/bin/env python3
"""
reproduce/render_recall_figure.py — manuscript Figure 6 (Amendment 12, R4)
===========================================================================

Recall of the true top-20% set by each ranker's top-k%, mean over the twelve LOSO
folds, against I* and the exhaustive I_dyn labels. Reads
data/benchmarks/referee_round7_recall.json (I*) and referee_round10_recall_idyn_full.json (I_dyn); trains and simulates nothing.

Usage:
    PYTHONPATH=. python reproduce/render_recall_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "benchmarks" / "referee_round7_recall.json"
#: Panel B reads the exhaustive I_dyn labels (``referee_round7.py recall --idyn-full``).
SRC_IDYN = ROOT / "data" / "benchmarks" / "referee_round10_recall_idyn_full.json"
OUT = ROOT / "docs" / "research" / "jss" / "latex" / "figures" / "Figure_5"

INK, INK2, GRID = "#1F2937", "#475569", "#E5E7EB"
#: Predictors in their Figure 5 colours (Okabe-Ito), solid with markers. Reference
#: rankings restate I*'s propagation rule (Amendment 13) and are drawn as grey
#: unmarked lines with their own dash pattern.
SERIES = [("GAT-P-QoS", "GAT-P-QoS", "#7B3F8C", "-"), ("Topo-QoS", "Topo-QoS", "#0072B2", "-"),
          ("Analytic-I*", "First-order $I^*$ (ref.)", "#9CA3AF", (0, (1, 1.5))),
          ("InDeg", "InDeg (ref.)", "#475569", (0, (4, 2))),
          ("Reach", "Reach (ref.)", "#94A3B8", (0, (5, 1.5, 1, 1.5)))]
REFERENCES = {"Analytic-I*", "InDeg", "Reach", "Rate-weighted"}
#: Panel B only: Eq. 7, the reference for I_dyn (Amendment 15).
IDYN_EXTRA = [("Rate-weighted", "Rate-weighted Eq. 7 (ref.)", "#1F2937", (0, (3, 1, 1, 1, 1, 1)))]
TITLES = {"i_star": "A. Against $I^*$ (all Applications)",
          "i_dyn": "B. Against $I_{\\mathrm{dyn}}$ (all Applications)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})


def main() -> None:
    curves = json.loads(SRC.read_text())["curves"]
    curves["i_dyn"] = json.loads(SRC_IDYN.read_text())["curves"]["i_dyn"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharey=True)
    for ax, oracle in zip(axes, ("i_star", "i_dyn")):
        for key, name, colour, ls in SERIES + (IDYN_EXTRA if oracle == "i_dyn" else []):
            c = curves[oracle][key]["curve"]
            ks = sorted(c, key=float)
            x = [100 * float(k) for k in ks]
            if key == "InDeg":  # Reach's zero-reach ties span most of the axis; see caption
                ax.fill_between(x, [c[k]["pessimistic"] for k in ks],
                                [c[k]["optimistic"] for k in ks], color=colour, alpha=0.15, lw=0)
            ref = key in REFERENCES
            ax.plot(x, [c[k]["expected"] for k in ks], color=colour, lw=1.1 if ref else 1.6, ls=ls,
                    marker=None if ref else "o", ms=2.8, label=name, zorder=2 if ref else 3)
        ax.axhline(0.8, color=INK2, lw=0.7, ls=(0, (1, 2)))
        ax.set_xlabel("Share of Applications flagged (top $k$%)")
        ax.set_title(TITLES[oracle], loc="left", fontsize=7.6, fontweight="bold", color=INK)
        ax.grid(color=GRID, lw=0.6)
        ax.set_ylim(0, 1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Recall of the true top-20% set")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=6.2,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
