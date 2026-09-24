#!/usr/bin/env python3
"""
reproduce/render_headline_figure.py — the JSS manuscript's results figure
=========================================================================

Writes ``docs/research/jss/latex/figures/Figure_5.{pdf,png}``, three panels that
carry the paper's three findings at a glance:

    A. Accuracy on unseen synthetic architectures (LOSO) against zero-shot
       transfer to the five system models, per engine, with 95% CIs.
       -> hybrids lead in distribution, pure learned engines transfer best.
    B. Per-fold Delta-rho against Topo-QoS for HGT-QoS and Hybrid-HGT, folds
       ordered by the closed-form engine's own score.
       -> the engines are complementary; the prior removes the learned
          engine's losses on the folds where closed-form structure is strongest.
    C. Cell means of the capacity- and channel-matched 2x2.
       -> the QoS channel moves accuracy; typing does not.

Everything is read from the same artifacts that back Tables 7-10 and is
reconciled against them by ``reconcile_manuscript.py``:

    results/loso_hybrid_cpu.json, results/loso_hybrid_gat_cpu.json   (A, B)
    results/realworld_zeroshot_<variant>_cpu.json                     (A)
    results/loso_rq2_matched.json                                     (C)

LOSO intervals use the same fold bootstrap as the tables
(``loso_significance._bootstrap_delta_ci`` applied to fold means, B=2000,
seed 42); zero-shot intervals are the artifacts' own bootstrap over systems.

Usage
-----
    PYTHONPATH=. python reproduce/render_headline_figure.py
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

from reproduce.loso_significance import _bootstrap_delta_ci  # noqa: E402
from saag.evaluation.variant_registry import label  # noqa: E402

RESULTS = Path("results")
OUT = Path("docs/research/jss/latex/figures/Figure_5")

INK, INK2, GRID = "#1F2937", "#475569", "#E5E7EB"
# Engine identity, fixed across the paper's figures (Okabe-Ito; validated with
# the dataviz palette checker: CVD and normal-vision separation pass).
ENGINES = [  # (variant, printed label from the registry, colour)
    (v, label(v, "loso"), c) for v, c in (
        ("topo_baseline", "#999999"),
        ("topo_qos", "#0072B2"),
        ("hgl_qos", "#E69F00"),
        ("gl_full_qos16_cap", "#CC79A7"),
        ("hgl_qos_prior", "#D55E00"),
        ("gl_qos16_prior", "#009E73"),
    )
]
COLOUR = {v: c for v, _, c in ENGINES}
FOLD = {  # two-line tick labels for panel B
    "atm_system": "ATM\n", "av_system": "AV\nSystem", "enterprise_system": "Enter-\nprise",
    "financial_trading_system": "Financial\nTrading", "healthcare_system": "Health-\ncare",
    "hub_and_spoke_system": "Ent. Inte-\ngration", "industrial_scada_system": "Industrial\nSCADA",
    "iot_smart_city_system": "IoT Smart\nCity", "logistics_fleet_system": "Logistics\nFleet",
    "microservices_system": "Micro-\nservices", "realtime_gaming_system": "Real-Time\nGaming",
    "telecom_ran_system": "Telecom\nRAN",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def _folds(table: dict, variant: str) -> dict:
    return {f["holdout"]: f["mean_rho"] for f in table[variant]["per_fold"]}


def load():
    loso = {**_load("loso_hybrid_gat_cpu.json")["comparison_table"],
            **_load("loso_hybrid_cpu.json")["comparison_table"]}
    loso_ci = {}
    for v, _, _ in ENGINES:
        f = np.array(list(_folds(loso, v).values()))
        loso_ci[v] = (float(f.mean()), *_bootstrap_delta_ci(f))
    rw = {}
    for v in ("hgl_qos", "gl_full_qos16_cap", "hgl_qos_prior", "gl_qos16_prior"):
        b = _load(f"realworld_zeroshot_{v}_cpu.json")["bootstrap_ci"]
        rw[v] = tuple(b["learned"]["rho"][k] for k in ("mean", "lo", "hi"))
        rw.setdefault("topo_baseline", tuple(b["Topo"]["rho"][k] for k in ("mean", "lo", "hi")))
        rw.setdefault("topo_qos", tuple(b["Topo-QoS"]["rho"][k] for k in ("mean", "lo", "hi")))
    matched = _load("loso_rq2_matched.json")["comparison_table"]
    return loso, loso_ci, rw, matched


def panel_a(ax, loso_ci, rw):
    ys = np.arange(len(ENGINES))[::-1]
    for y, (v, label, c) in zip(ys, ENGINES):
        for (m, lo, hi), dy, marker, face in ((loso_ci[v], 0.15, "o", c), (rw[v], -0.15, "D", "white")):
            ax.plot([lo, hi], [y + dy, y + dy], color=c, lw=1.2, solid_capstyle="round", zorder=2)
            ax.plot(m, y + dy, marker=marker, ms=5.0 if marker == "o" else 4.2, mfc=face, mec=c,
                    mew=1.3, zorder=3, ls="none")
    ax.set_yticks(ys, [lab for _, lab, _ in ENGINES])
    ax.set_xlim(0.2, 1.0)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.6, len(ENGINES) + 0.3)  # headroom row for the legend
    ax.set_xlabel(r"Spearman $\rho$ with simulated impact (95% CI)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Line2D([], [], marker="o", color=INK2, mfc=INK2, ls="none", ms=5, label="LOSO (12 folds)"),
        Line2D([], [], marker="D", color=INK2, mfc="white", mew=1.3, ls="none", ms=4.2,
               label="zero-shot (5 systems)"),
    ], loc="upper right", ncol=2, frameon=False, fontsize=6.2, handletextpad=0.2,
       columnspacing=1.0, borderaxespad=0.0)
    ax.set_title("A. In-distribution accuracy vs. transfer", loc="left", fontsize=7.6,
                 fontweight="bold", color=INK)


def panel_b(ax, loso):
    base = _folds(loso, "topo_qos")
    learned, hybrid = _folds(loso, "hgl_qos"), _folds(loso, "hgl_qos_prior")
    order = sorted(base, key=base.get, reverse=True)
    xs = np.arange(len(order))
    for x, k in zip(xs, order):
        dl, dh = learned[k] - base[k], hybrid[k] - base[k]
        ax.annotate("", xy=(x, dh), xytext=(x, dl),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.2",
                                    color="#9CA3AF", lw=0.9, shrinkA=3, shrinkB=3))
        ax.plot(x, dl, "o", ms=5, mfc=COLOUR["hgl_qos"], mec="white", mew=0.8, zorder=3)
        ax.plot(x, dh, "o", ms=5, mfc=COLOUR["hgl_qos_prior"], mec="white", mew=0.8, zorder=3)
    ax.axhline(0, color=COLOUR["topo_qos"], lw=1.0)
    ax.text(len(order) - 0.45, 0.012, "Topo-QoS", color=INK2, fontsize=6.0, ha="right", va="bottom")
    ax.set_xticks(xs, [f"{FOLD[k]}\n({base[k]:.2f})" for k in order], fontsize=5.9)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(-0.42, 0.42)
    ax.set_ylabel(r"$\Delta\rho$ vs. Topo-QoS")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("held-out fold, ordered by Topo-QoS ρ (in brackets): closed-form engine strongest → weakest",
                  fontsize=6.4, color=INK2)
    ax.legend(handles=[
        Line2D([], [], marker="o", color=COLOUR["hgl_qos"], ls="none", ms=5, label=label("hgl_qos", "loso")),
        Line2D([], [], marker="o", color=COLOUR["hgl_qos_prior"], ls="none", ms=5,
               label=f"{label('hgl_qos_prior', 'loso')} ({label('hgl_qos', 'loso')} + closed-form prior)"),
    ], loc="upper left", frameon=False, fontsize=6.2, handletextpad=0.2, borderaxespad=0.1, ncol=2)
    ax.set_title("B. Per fold: the prior repairs HGT-QoS where the closed form is strong",
                 loc="left", fontsize=7.6, fontweight="bold", color=INK)


def panel_c(ax, matched):
    x = [0, 1]
    for (a, b), name, c, marker, dy in ((("gl_full_cap", "gl_full_qos16_cap"), "untyped (GAT)",
                                          COLOUR["gl_full_qos16_cap"], "s", 0.0035),
                                         (("hgl", "hgl_qos"), "typed (HGT)", COLOUR["hgl_qos"], "o", -0.0035)):
        y = [matched[a]["mean_rho"], matched[b]["mean_rho"]]
        ax.plot(x, y, color=c, lw=1.6, marker=marker, ms=5, mec="white", mew=0.8)
        ax.text(1.08, y[1] + dy, name, color=INK2, fontsize=6.0, va="center")
    ref = matched["topo_qos"]["mean_rho"]
    ax.axhline(ref, color=COLOUR["topo_qos"], lw=0.9, ls=(0, (4, 2)))
    ax.text(1.08, ref + 0.002, "Topo-QoS", color=INK2, fontsize=6.0, va="bottom")
    ax.set_xticks(x, ["no QoS\nchannel", "16-D QoS\nchannel"])
    ax.set_xlim(-0.25, 2.05)
    ax.set_ylim(0.53, 0.65)
    ax.set_ylabel(r"mean LOSO $\rho$")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("C. Matched capacity: QoS\nmatters, typing does not", loc="left", fontsize=7.6,
                 fontweight="bold", color=INK)


def main() -> None:
    loso, loso_ci, rw, matched = load()
    fig = plt.figure(figsize=(6.5, 4.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 0.95],
                          wspace=0.42, hspace=0.62, left=0.15, right=0.99, bottom=0.14, top=0.95)
    ax_a, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, :])
    for ax in (ax_a, ax_b, ax_c):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    panel_a(ax_a, loso_ci, rw)
    panel_b(ax_b, loso)
    panel_c(ax_c, matched)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"✓ {OUT}.pdf / .png")


if __name__ == "__main__":
    main()
