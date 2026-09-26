#!/usr/bin/env python3
"""
reproduce/render_amendment7_figure.py — the JSS manuscript's results figure (Figure 5)
=====================================================================================

Writes ``docs/research/jss/latex/figures/Figure_5.{pdf,png}``, three panels:

    A. Mean LOSO Spearman rho with 95% bootstrap intervals for every engine:
       dependency counts, betweenness engines, learned and hybrid engines.
    B. Per held-out fold: InDeg, the best hybrid (Hybrid-GAT) and Topo-QoS.
    C. Where the closed-form gain comes from: the registered Topo, unweighted and
       QoS-weighted betweenness on the projection, and the Amendment 7 controls.

Training-free values and intervals are read from the Amendment 7 artifacts
(``results/tf_baselines.json``, ``results/qos_attribution_controls.json``,
``results/qos_indep_corpus.json``). Learned and hybrid values are the published
CPU-sweep figures: per-fold values from Supplementary Table S23 (the constants in
``reproduce/training_free_suite.py``) and intervals from Table 7 of the manuscript,
whose backing artifacts are in the Zenodo bundle.

Usage:
    PYTHONPATH=. python reproduce/render_amendment7_figure.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from reproduce.training_free_suite import (  # noqa: E402
    FOLDS,
    PUBLISHED_HYBRID_GAT_CPU,
    PUBLISHED_TOPO,
    mean_ci,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "research" / "jss" / "latex" / "figures" / "Figure_5"

INK, INK2, GRID = "#1F2937", "#475569", "#E5E7EB"
# Family colours (validated with the dataviz palette checker, all pairs, light
# surface); grey is the neutral reference for the betweenness engines, not a slot.
COUNT, LEARN, HYBRID, REF = "#0072B2", "#E69F00", "#AA3377", "#8C8C8C"

# Published CPU-sweep means and fold-bootstrap intervals (manuscript Table 7;
# GBM-Feat from the attribution controls of Section 7.2, Amendment 8).
PUBLISHED = {
    "Topo": (0.349, 0.254, 0.452),
    "HGT-QoS": (0.622, 0.547, 0.690),
    "GAT-QoS": (0.635, 0.567, 0.696),
    "Hybrid-HGT": (0.657, 0.572, 0.733),
    "Hybrid-GAT": (0.683, 0.603, 0.753),
    "GBM-Feat": (0.642, 0.547, 0.725),
}
SHORT = {
    "ATM": "ATM", "AV System": "AV", "Enterprise": "Enterprise",
    "Financial Trading": "Financial", "Healthcare": "Healthcare",
    "Enterprise Integration (ESB)": "Ent. Integr.", "Industrial SCADA": "SCADA",
    "IoT Smart City": "IoT City", "Logistics Fleet": "Logistics",
    "Microservices": "Microserv.", "Real-Time Gaming": "Gaming", "Telecom RAN": "Telecom",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def panel_a(ax, tf: dict) -> None:
    s = tf["summary"]
    rows = [  # (label, mean, lo, hi, colour, marker)
        ("InDeg", s["InDeg"]["loso_mean_rho"], *s["InDeg"]["loso_rho_ci95"], COUNT, "o"),
        ("Reach", s["Reach"]["loso_mean_rho"], *s["Reach"]["loso_rho_ci95"], COUNT, "o"),
        ("Reach-QoS", s["Reach-QoS"]["loso_mean_rho"], *s["Reach-QoS"]["loso_rho_ci95"], COUNT, "o"),
        ("Hybrid-GAT", *PUBLISHED["Hybrid-GAT"], HYBRID, "D"),
        ("Hybrid-HGT", *PUBLISHED["Hybrid-HGT"], HYBRID, "D"),
        ("GBM-Feat", *PUBLISHED["GBM-Feat"], LEARN, "s"),
        ("GAT-QoS", *PUBLISHED["GAT-QoS"], LEARN, "s"),
        ("HGT-QoS", *PUBLISHED["HGT-QoS"], LEARN, "s"),
        ("Betweenness (proj.)", s["Topo (projection)"]["loso_mean_rho"],
         *s["Topo (projection)"]["loso_rho_ci95"], REF, "o"),
        ("Topo-QoS", s["Topo-QoS"]["loso_mean_rho"], *s["Topo-QoS"]["loso_rho_ci95"], REF, "o"),
        ("Topo", *PUBLISHED["Topo"], REF, "o"),
    ]
    ys = np.arange(len(rows))[::-1]
    for y, (lab, m, lo, hi, c, mk) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=c, lw=1.4, solid_capstyle="round", zorder=2)
        ax.plot(m, y, marker=mk, ms=5.2, mfc=c, mec="white", mew=0.8, zorder=3, ls="none")
        ax.text(hi + 0.012, y, f"{m:.3f}", color=INK2, fontsize=6.0, va="center")
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlim(0.2, 0.95)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(r"mean Spearman $\rho$, twelve held-out folds (95% CI)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(handles=[
        Line2D([], [], marker="o", color=COUNT, ls="none", ms=5, label="dependency counts"),
        Line2D([], [], marker="D", color=HYBRID, ls="none", ms=4.6, label="hybrid"),
        Line2D([], [], marker="s", color=LEARN, ls="none", ms=4.6, label="learned"),
        Line2D([], [], marker="o", color=REF, ls="none", ms=5, label="betweenness"),
    ], loc="upper left", frameon=False, fontsize=6.0, handletextpad=0.2, borderaxespad=0.1)
    _style(ax)
    ax.set_title("A. Counting dependents ranks best", loc="left", fontsize=7.6,
                 fontweight="bold", color=INK)


def panel_b(ax, tf: dict) -> None:
    per = tf["per_fold"]
    order = sorted(per, key=lambda k: per[k]["Topo-QoS"]["rho"], reverse=True)
    xs = np.arange(len(order))
    series = [
        ("InDeg", [per[k]["InDeg"]["rho"] for k in order], COUNT, "o"),
        ("Hybrid-GAT", [PUBLISHED_HYBRID_GAT_CPU[k] for k in order], HYBRID, "D"),
        ("Topo-QoS", [per[k]["Topo-QoS"]["rho"] for k in order], REF, "o"),
    ]
    for lab, ys, c, mk in series:
        ax.plot(xs, ys, color=c, lw=1.0, alpha=0.55, zorder=2)
        ax.plot(xs, ys, marker=mk, ms=4.8, mfc=c, mec="white", mew=0.8, ls="none", zorder=3)
        ax.text(xs[-1] + 0.25, ys[-1], lab, color=INK2, fontsize=6.0, va="center")
    ax.set_xticks(xs, [SHORT[k] for k in order], fontsize=6.0, rotation=35, ha="right")
    ax.set_xlim(-0.5, len(order) + 1.3)
    ax.set_ylim(0.2, 0.95)
    ax.set_ylabel(r"Spearman $\rho$")
    ax.grid(axis="y", color=GRID, lw=0.6)
    _style(ax)
    ax.set_title("B. Per held-out fold (ordered by Topo-QoS)", loc="left", fontsize=7.6,
                 fontweight="bold", color=INK)


def panel_c(ax, ctl: dict, ind: dict, tf: dict) -> None:
    names = list(FOLDS.values())
    pf = ctl["per_fold"]
    rows = [
        ("Topo (registered)", [PUBLISHED_TOPO[n] for n in names]),
        ("Betweenness (proj.)", [tf["per_fold"][n]["Topo (projection)"]["rho"] for n in names]),
        ("Topo-QoS (declared QoS)", [pf[n]["Topo-QoS"] for n in names]),
        ("constant topic weight", [pf[n]["Topo-Mult"] for n in names]),
        ("permuted QoS", [pf[n]["Topo-QoS-Perm_mean"] for n in names]),
        ("QoS-indep. corpus", [ind["per_fold"][n]["Topo-QoS"] for n in names]),
    ]
    ys = np.arange(len(rows))[::-1]
    for y, (lab, vals) in zip(ys, rows):
        m = float(np.mean(vals))
        lo, hi = mean_ci(vals)
        c = INK2 if lab.startswith("Topo (") else REF
        ax.plot([lo, hi], [y, y], color=c, lw=1.4, solid_capstyle="round", zorder=2)
        ax.plot(m, y, "o", ms=5.2, mfc=c, mec="white", mew=0.8, zorder=3)
        ax.text(hi + 0.012, y, f"{m:.3f}", color=INK2, fontsize=6.0, va="center")
    ax.axhline(ys[1] + 0.5, color=GRID, lw=0.8)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(r"mean Spearman $\rho$ (95% CI)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    _style(ax)
    ax.set_title("C. The gain is the projection,\nnot the QoS content", loc="left",
                 fontsize=7.6, fontweight="bold", color=INK)


def main() -> int:
    tf = _load("tf_baselines.json")
    ctl = _load("qos_attribution_controls.json")
    ind = _load("qos_indep_corpus.json")
    fig = plt.figure(figsize=(6.6, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 0.85],
                          wspace=0.75, hspace=0.62, left=0.17, right=0.97, bottom=0.12, top=0.95)
    ax_a, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, :])
    panel_a(ax_a, tf)
    panel_b(ax_b, tf)
    panel_c(ax_c, ctl, ind, tf)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {OUT}.pdf / .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
