#!/usr/bin/env python3
"""
reproduce/render_jss_diagrams.py — the JSS manuscript's explanatory diagrams
===========================================================================

Writes three conceptual figures into ``docs/research/jss/latex/figures/``:

    Figure_2  running example: structural graph and its DEPENDS_ON projection (§3.2)
    Figure_3  the three ranking engines and the evaluation design (§4)
    Figure_4  the explanation layer: metrics -> FT / A / M -> Q(v) -> remediation (§5)

They carry no measured numbers -- only the declared model constants of §3 and
§5 -- so nothing here needs an artifact. Figure_5 (results) is data-driven and
lives in ``render_headline_figure.py``.

Every canvas is drawn at the manuscript's text width (6.5 in = 468 pt) and
included at ``width=\\linewidth``, so the fonts set here are the fonts that print.
Do not widen the canvas: the Graphviz version of Figure_2 was 1014 pt wide and
printed its labels at ~5.5 pt.

Usage
-----
    python reproduce/render_jss_diagrams.py            # all three
    python reproduce/render_jss_diagrams.py --only 3   # one figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path("docs/research/jss/latex/figures")
WIDTH_IN = 6.5

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
})

# Ink and surfaces
INK, INK2, MUTED, RULE = "#1F2937", "#475569", "#94A3B8", "#CBD5E1"
# Entity types (same hues as Table 1 in the text; Host is neutral so that it does
# not read as the vermillion "simultaneous blast" edge).
ENTITY = {"app": "#4C72B0", "topic": "#DD8452", "broker": "#55A868",
          "host": "#7F7F7F", "lib": "#8172B3"}
# Derived-edge semantics
CASCADE, BLAST = "#0072B2", "#D55E00"
# Engine identity, fixed across Figure_3 and Figure_5 (Okabe-Ito, validated).
ENGINE = {"closed": "#0072B2", "learned": "#E69F00", "hybrid": "#D55E00"}
ORACLE = "#B45309"


# ── primitives ────────────────────────────────────────────────────────────────

def canvas(height_in: float, xmax: float = 100.0):
    fig = plt.figure(figsize=(WIDTH_IN, height_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ymax = xmax * height_in / WIDTH_IN
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax, ymax


def box(ax, x, y, w, h, title=None, body=None, fc="#F8FAFC", ec=INK2, lw=0.9,
        title_size=8.2, body_size=7.2, dashed=False, align="center", pad=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                fc=fc, ec=ec, lw=lw, ls=(0, (4, 2)) if dashed else "-"))
    cx = x + w / 2 if align == "center" else x + pad
    ha = "center" if align == "center" else "left"
    if title and body:
        ax.text(cx, y + h - pad - 0.6, title, ha=ha, va="top", fontsize=title_size,
                fontweight="bold", color=INK)
        ax.text(cx, y + h - pad - 3.6, body, ha=ha, va="top", fontsize=body_size,
                color=INK2, linespacing=1.35)
    elif title:
        ax.text(cx, y + h / 2, title, ha=ha, va="center", fontsize=title_size,
                fontweight="bold", color=INK)


def arrow(ax, p, q, color=INK2, lw=1.0, ls="-", rad=0.0, head=6, shrink=0.0, z=2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=f"-|>,head_length={head/10},head_width={head/16}",
                                 connectionstyle=f"arc3,rad={rad}", color=color, lw=lw,
                                 ls=ls, shrinkA=shrink, shrinkB=shrink, zorder=z,
                                 mutation_scale=10))


def node(ax, x, y, label, kind, r=2.6):
    ax.add_patch(Circle((x, y), r, fc=ENTITY[kind], ec="white", lw=1.2, zorder=4))
    ax.text(x, y, label, ha="center", va="center", fontsize=7.5, color="white",
            fontweight="bold", zorder=5)


def edge(ax, a, b, r=2.6, **kw):
    """Arrow between two circle centres, clipped to the circle rims."""
    import math
    (x1, y1), (x2, y2) = a, b
    d = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / d, (y2 - y1) / d
    arrow(ax, (x1 + ux * r, y1 + uy * r), (x2 - ux * (r + 0.3), y2 - uy * (r + 0.3)), **kw)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"✓ {OUT / name}.pdf / .png")


# ── Figure 2: running example ─────────────────────────────────────────────────

def figure2():
    fig, ax, ymax = canvas(2.75)
    top = ymax - 1
    for x0, title in ((1, r"(a) Structural graph $G_{\mathrm{structural}}$"),
                      (51, r"(b) Derived $\mathtt{DEPENDS\_ON}$ edges in $G_{\mathrm{analysis}}$")):
        ax.add_patch(FancyBboxPatch((x0, 8.5), 48, top - 8.5, boxstyle="round,pad=0,rounding_size=1.5",
                                    fc="white", ec=RULE, lw=0.8))
        ax.text(x0 + 24, top - 2.6, title, ha="center", va="center", fontsize=8, color=INK)

    # (a) library above, the hosted processes in the middle, the topic below.
    lib, t = (21, 32.5), (21, 13.5)
    A = {"a1": (9, 23), "a2": (21, 23), "a3": (33, 23)}
    b = (44, 23)
    ax.add_patch(FancyBboxPatch((4.2, 19.3), 44.2, 7.4, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="#F3F4F6", ec=ENTITY["host"], lw=0.8, ls=(0, (3, 2)), zorder=1))
    ax.text(48.0, 27.2, "host n (runs_on)", fontsize=6.4, color=INK2, ha="right", va="bottom")
    for k, pnt in A.items():
        node(ax, *pnt, k, "app")
        edge(ax, pnt, lib, color=MUTED, lw=0.8, ls=(0, (1, 1.2)), head=4)
    node(ax, *b, "b", "broker"); node(ax, *lib, "ℓ", "lib"); node(ax, *t, "t", "topic")
    ax.text(10.5, 29.0, "uses", fontsize=6.5, color=INK2)
    edge(ax, A["a1"], t, color=INK, lw=1.0, head=4)
    edge(ax, A["a2"], t, color=INK, lw=1.0, head=4)
    edge(ax, A["a3"], t, color=INK, lw=1.0, head=4)
    edge(ax, b, t, color=INK, lw=1.0, head=4, rad=-0.2)
    ax.text(11.0, 16.2, "pub", fontsize=6.5, color=INK2)
    ax.text(26.2, 16.4, "sub", fontsize=6.5, color=INK2)
    ax.text(33.5, 13.8, "routes", fontsize=6.5, color=INK2)

    # (b) projection
    B = {"a1": (60, 21), "a2": (74, 29.5), "a3": (74, 12.5)}
    lib2, b2 = (91, 29.5), (91, 12.5)
    for k, pnt in B.items():
        node(ax, *pnt, k, "app")
    node(ax, *lib2, "ℓ", "lib"); node(ax, *b2, "b", "broker")
    for k in ("a1", "a2", "a3"):
        edge(ax, B[k], b2, color=MUTED, lw=0.8, ls=(0, (3, 2)), head=4, rad=0.08)
    edge(ax, B["a2"], B["a1"], color=CASCADE, lw=1.8, head=4.5)
    edge(ax, B["a3"], B["a1"], color=CASCADE, lw=1.8, head=4.5)
    edge(ax, B["a1"], lib2, color=BLAST, lw=1.8, head=4.5, rad=-0.08)
    edge(ax, B["a2"], lib2, color=BLAST, lw=1.8, head=4.5)
    edge(ax, B["a3"], lib2, color=BLAST, lw=1.8, head=4.5, rad=0.1)
    ax.text(62.5, 28.5, "a2, a3 subscribe to\nt, which a1 publishes", fontsize=6.2, color=INK2,
            ha="center", va="center")
    ax.text(62.5, 12.2, "all three use ℓ;\nall publish or\nsubscribe via b", fontsize=6.2,
            color=INK2, ha="center", va="center")

    # legend: entity types, then edge semantics
    renderer = fig.canvas.get_renderer()
    units_per_px = ax.get_xlim()[1] / ax.get_window_extent(renderer).width

    def legend_row(y, items):
        x = 2.0
        for draw, name in items:
            draw(x, y)
            txt = ax.text(x + 3.6, y, name, fontsize=6.6, va="center", color=INK2)
            x += 3.6 + txt.get_window_extent(renderer).width * units_per_px + 2.5

    def dot(kind):
        return lambda x, y: ax.add_patch(Circle((x + 1.2, y), 1.1, fc=ENTITY[kind], ec="white", lw=0.8))

    def line(color, lw, ls):
        return lambda x, y: ax.plot([x, x + 3], [y, y], color=color, lw=lw, ls=ls)

    def host_swatch(x, y):
        ax.add_patch(FancyBboxPatch((x, y - 1.0), 2.6, 2.0, boxstyle="round,pad=0,rounding_size=0.5",
                                    fc="#F3F4F6", ec=ENTITY["host"], lw=0.8, ls=(0, (2, 1.5))))

    legend_row(5.6, [(dot("app"), "Application"), (dot("topic"), "Topic"), (dot("broker"), "Broker"),
                     (dot("lib"), "Library"), (host_swatch, "Host (contains what runs on it)")])
    legend_row(2.0, [(line(CASCADE, 1.8, "-"), "sequential cascade (Rule 1)"),
                     (line(BLAST, 1.8, "-"), "simultaneous blast (Rule 5)"),
                     (line(MUTED, 0.9, (0, (3, 2))), "app → broker (Rule 2)")])
    save(fig, "Figure_2")


# ── Figure 3: engines and evaluation design ───────────────────────────────────

def figure3():
    fig, ax, ymax = canvas(3.0)
    ax.text(1, ymax - 2.2, "(a) Three ranking engines, one analysis graph",
            fontsize=8.2, fontweight="bold", color=INK, va="center")
    ax.text(62, ymax - 2.2, "(b) Ground truth and evaluation",
            fontsize=8.2, fontweight="bold", color=INK, va="center")

    # (a) engines
    box(ax, 1, 14, 13.5, 17, r"$G_{\mathrm{analysis}}$",
        "QoS-weighted\nDEPENDS_ON\nedges, typed\nnode features", fc="#EEF2FF", ec="#3730A3",
        body_size=6.8)
    box(ax, 19, 29, 22, 9.5, "Closed-form engine",
        "QoS-weighted betweenness\n" r"(Topo-QoS) $\rightarrow p(v)$", fc="#E6F1F8",
        ec=ENGINE["closed"], body_size=6.8)
    box(ax, 19, 5, 22, 13.5, "Learned engine",
        "GNN over the typed multi-\ngraph with 16-D QoS edge\n" r"vectors $\rightarrow$ logit $z(v)$",
        fc="#FDF4E3", ec=ENGINE["learned"], body_size=6.8)
    arrow(ax, (14.5, 27), (19, 32), color=INK2, head=4)
    arrow(ax, (14.5, 18), (19, 13), color=INK2, head=4)
    box(ax, 45.5, 14.5, 14, 10, "Hybrid engine",
        r"$\sigma\!\left(z + \alpha\,\mathrm{logit}\,p\right)$", fc="#FBEAE1", ec=ENGINE["hybrid"],
        title_size=7.6, body_size=7.4)
    arrow(ax, (41, 32), (50, 24.5), color=ENGINE["hybrid"], lw=1.1, rad=-0.25, head=4)
    arrow(ax, (41, 12), (46, 17), color=ENGINE["hybrid"], lw=1.1, head=4)
    arrow(ax, (25, 29), (25, 18.5), color=ENGINE["hybrid"], lw=1.1, ls=(0, (3, 2)), head=4)
    ax.text(26, 23.8, "prior $p(v)$, rank-\nnormalized, as an\nextra input feature",
            fontsize=6.1, color=INK2, va="center")
    # outputs
    arrow(ax, (41, 36.5), (45, 36.5), color=ENGINE["closed"], head=4)
    ax.text(45.8, 36.5, "ranking", fontsize=6.6, color=INK2, va="center")
    arrow(ax, (30, 5), (30, 2.4), color=ENGINE["learned"], head=4)
    ax.text(31, 2.8, r"ranking $\sigma(z)$", fontsize=6.6, color=INK2, va="center")
    arrow(ax, (52.5, 14.5), (52.5, 10.5), color=ENGINE["hybrid"], head=4)
    ax.text(52.5, 9.2, "ranking", fontsize=6.6, color=INK2, ha="center", va="top")
    ax.text(1, 7.5, "Hybrids add one scalar $\\alpha$;\notherwise identical to\ntheir learned engine.",
            fontsize=6.0, color=INK2, va="center")

    ax.plot([60.5, 60.5], [1, ymax - 1], color=RULE, lw=0.8)

    # (b) ground truth + protocols
    box(ax, 62.5, 30.5, 14, 6, r"$G_{\mathrm{structural}}$", fc="#DBEAFE", ec="#1E40AF", title_size=7.6)
    box(ax, 81.5, 30.5, 17.5, 6, r"oracles $\rightarrow I^*(v)$", fc="#FFEDD5", ec=ORACLE, title_size=7.6)
    arrow(ax, (76.5, 33.5), (81.5, 33.5), color=ORACLE, head=4)
    ax.text(80.7, 28.8, "labels only; no predictor reads $G_{\\mathrm{structural}}$",
            fontsize=6.1, color=INK2, ha="center", va="top")
    gx, gy, c = 64, 4.5, 1.45
    for i in range(12):
        for j in range(12):
            ax.add_patch(Rectangle((gx + j * c, gy + (11 - i) * c), c - 0.25, c - 0.25,
                                   fc=INK2 if i == j else "#CBD5E1", ec="none"))
    ax.text(gx + 6 * c, gy + 12 * c + 1.0, "LOSO: 12 folds", fontsize=7.0, ha="center",
            color=INK, fontweight="bold")
    ax.text(gx - 0.6, gy + 6 * c, "fold", rotation=90, fontsize=6.1, ha="right", va="center", color=INK2)
    ax.text(gx + 6 * c, gy - 1.0, "scenario", fontsize=6.1, ha="center", va="top", color=INK2)
    lx, ly = gx + 12 * c + 1.2, gy + 10.5 * c
    ax.add_patch(Rectangle((lx, ly), 1.2, 1.2, fc="#CBD5E1", ec="none"))
    ax.text(lx + 1.8, ly + 0.6, "train", fontsize=6.1, va="center", color=INK2)
    ax.add_patch(Rectangle((lx, ly - 2.4), 1.2, 1.2, fc=INK2, ec="none"))
    ax.text(lx + 1.8, ly - 1.8, "test", fontsize=6.1, va="center", color=INK2)
    box(ax, 89, 4.5, 10, 13, "Zero-shot",
        "train on all\n12; score 5\nopen-source\nsystem\nmodels", fc="#F1F5F9", ec=INK2,
        title_size=7.0, body_size=6.1, pad=0.9)
    save(fig, "Figure_3")


# ── Figure 4: explanation layer ───────────────────────────────────────────────

def figure4():
    fig, ax, ymax = canvas(2.55)
    GREEN, GREEN_BG = "#15803D", "#ECFDF5"
    for x, t in ((1, "Graph metrics"), (29, "Sub-characteristics"), (57, "Composite"),
                 (77.5, "Diagnosis → remedy")):
        ax.text(x, ymax - 2.0, t, fontsize=7.6, fontweight="bold", color=INK)

    rows = [
        ("Reverse PageRank,\nin-degree, cascade\ndepth on $G^{\\top}$", "Fault Tolerance FT",
         "0.45 RPR + 0.30 Deg$_{\\mathrm{in}}$\n+ 0.25 CDPot", 26.5),
        ("Directed articulation,\nQoS-SPOF, bridges,\nCDI, QoS weight", "Availability A",
         "0.25 AP + 0.20 QSPOF + 0.20 BR\n+ 0.25 CDI + 0.10 w", 15.5),
        ("Betweenness, QoS fan-\nout, code quality,\ncoupling, clustering", "Maintainability M",
         "0.35 BT + 0.30 w$_{\\mathrm{out}}$ + 0.15 CQP\n+ 0.12 CR + 0.08 (1−CC)", 4.5),
    ]
    for metrics, dim, formula, y in rows:
        box(ax, 1, y, 23.5, 9.2, None, None, fc="#F8FAFC", ec=RULE)
        ax.text(2.2, y + 4.6, metrics, fontsize=6.4, color=INK2, va="center", linespacing=1.25)
        box(ax, 29, y, 24.5, 9.2, None, None, fc=GREEN_BG, ec=GREEN)
        ax.text(41.25, y + 6.8, dim, fontsize=7.2, fontweight="bold", color=INK, ha="center", va="center")
        ax.text(41.25, y + 3.0, formula, fontsize=5.9, color=INK2, ha="center", va="center",
                linespacing=1.25)
        arrow(ax, (24.5, y + 4.6), (29, y + 4.6), head=4)
    box(ax, 57, 19.5, 15.5, 9.5, "Reliability R", r"$0.36\,FT + 0.64\,A$", fc="#DCFCE7", ec=GREEN,
        title_size=7.2, body_size=6.9)
    box(ax, 57, 4.5, 15.5, 11, "Quality Q(v)", r"$0.80\,R + 0.20\,M$" "\nTukey fence" r"$\,\rightarrow$" "\nCRITICAL tier",
        fc="#DCFCE7", ec=GREEN, title_size=7.2, body_size=6.5)
    arrow(ax, (53.5, 31), (57, 26.5), head=4)
    arrow(ax, (53.5, 20), (57, 22.5), head=4)
    arrow(ax, (64.75, 19.5), (64.75, 15.5), head=4)
    arrow(ax, (53.5, 9.1), (57, 9.1), head=4)
    diag = [("High A, low FT", "single point of failure", "→ replication", 26.5),
            ("High FT", "error-cascade hub", "→ circuit breaker", 15.5),
            ("High M", "coupling bottleneck", "→ decoupling, refactoring", 4.5)]
    for cond, what, fix, y in diag:
        box(ax, 77.5, y, 21.5, 9.2, None, None, fc="white", ec=GREEN, dashed=True)
        ax.text(78.7, y + 6.6, cond, fontsize=6.9, fontweight="bold", color=INK, va="center")
        ax.text(78.7, y + 2.9, f"{what}\n{fix}", fontsize=6.2, color=INK2, va="center", linespacing=1.3)
        arrow(ax, (72.5, 10), (77.5, y + 4.6), color=GREEN, head=4)
    ax.text(88.25, 2.2, "each flagged component is read\nby its FT / A / M profile",
            fontsize=5.9, color=INK2, ha="center", va="center")
    save(fig, "Figure_4")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--only", type=int, choices=[2, 3, 4])
    args = ap.parse_args()
    for n, fn in ((2, figure2), (3, figure3), (4, figure4)):
        if args.only in (None, n):
            fn()


if __name__ == "__main__":
    main()
