#!/usr/bin/env python3
"""
reproduce/render_recall_figure.py — manuscript Figure 6 (Amendment 12, R4)
===========================================================================

Recall of the true top-20% set by each ranker's top-k%, mean over the twelve LOSO
folds, against I* and the n = 30 I_dyn sample. Reads
data/benchmarks/referee_round7_recall.json; trains and simulates nothing.

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
OUT = ROOT / "docs" / "research" / "jss" / "latex" / "figures" / "Figure_6"

INK, INK2, GRID = "#1F2937", "#475569", "#E5E7EB"
#: Same engine colours as Figure 5 (Okabe-Ito); the first-order expansion is grey.
SERIES = [("InDeg", "InDeg", "#000000"), ("GAT-P-QoS", "GAT-P-QoS", "#7B3F8C"),
          ("Analytic-I*", "First-order $I^*$", "#999999"), ("Reach", "Reach", "#56B4E9"),
          ("Topo-QoS", "Topo-QoS", "#0072B2")]
TITLES = {"i_star": "A. Against $I^*$ (all Applications)",
          "i_dyn": "B. Against $I_{\\mathrm{dyn}}$ ($n = 30$ per fold)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})


def main() -> None:
    curves = json.loads(SRC.read_text())["curves"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharey=True)
    for ax, oracle in zip(axes, ("i_star", "i_dyn")):
        for key, name, colour in SERIES:
            c = curves[oracle][key]["curve"]
            ks = sorted(c, key=float)
            x = [100 * float(k) for k in ks]
            if key == "InDeg":  # Reach's zero-reach ties span most of the axis; see caption
                ax.fill_between(x, [c[k]["pessimistic"] for k in ks],
                                [c[k]["optimistic"] for k in ks], color=colour, alpha=0.15, lw=0)
            ax.plot(x, [c[k]["expected"] for k in ks], color=colour, lw=1.5, marker="o", ms=2.8,
                    label=name)
        ax.axhline(0.8, color=INK2, lw=0.7, ls=(0, (1, 2)))
        ax.set_xlabel("Share of Applications flagged (top $k$%)")
        ax.set_title(TITLES[oracle], loc="left", fontsize=7.6, fontweight="bold", color=INK)
        ax.grid(color=GRID, lw=0.6)
        ax.set_ylim(0, 1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Recall of the true top-20% set")
    axes[1].legend(loc="lower right", frameon=False, fontsize=6.2)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
