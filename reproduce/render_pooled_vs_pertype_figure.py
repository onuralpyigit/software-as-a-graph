#!/usr/bin/env python3
"""
reproduce/render_pooled_vs_pertype_figure.py — RETIRED from the JSS manuscript
================================================================================

RETIRED: this figure is no longer in docs/research/jss/draft.md. When the
manuscript was condensed to fit JSS's page guidance, §5.5 (the stratified
correlation check) was compressed to a single paragraph and its figure dropped;
the full treatment now lives in
docs/research/thesis/material/oracles_and_labels.md. The script is kept because
the analysis is still current and a thesis chapter may want the figure -- but it
is NOT part of `make -f reproduce/Makefile jss-figures` any more, and writing to
Figure_4 would now overwrite the AHP shrinkage figure. Point --output somewhere
else (or at the thesis material) before running it.

Produces the pooled-versus-per-node-type Spearman rho figure between Q(v)
and I_comp(v), with per-type sample sizes. See draft.md §5.5 (the Simpson's-
paradox consistency check) -- the six values below (one pooled + five
per-type rho, each with its n) are the published numbers stated directly in
that section's prose; there is no separate JSON artifact behind them (the
underlying per-node (Q, I_comp) pairs are not retained -- see the
Declarations/data-availability note), so they are transcribed here rather
than recomputed. If the artifact is later retained, point this script at it
instead of the hardcoded _RESULTS table.

NOTE on the file number: this is draft.md's own "Figure 3" (the caption at
§5.5 says so, and \label{fig:3} in the LaTeX cites it that way), but it is
NOT the 3rd figure by physical position in the manuscript -- the attention-
subgraph figure (draft.md's own "Figure 6") appears earlier, at §5.2. LaTeX
numbers floats by reading order regardless of the source's own label, so
the file that lands as the *printed* "Fig. 3" is Figure_3.pdf (the attention
subgraph, produced by extract_attention.py + render_attention_subgraph.py);
this script's output is the printed "Fig. 4". See the numbering table in
docs/research/jss/latex/README.md before renaming anything here.

Output:
  docs/research/thesis/material/figures/pooled_vs_pertype.png  (300 dpi)
  docs/research/thesis/material/figures/pooled_vs_pertype.pdf  (vector)

Usage
-----
  python reproduce/render_pooled_vs_pertype_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Was hardcoded to docs/research/jss/latex/figures/Figure_4, silently
# overwriting the AHP shrinkage figure (reproduce/render_shrinkage_figure.py)
# on every run -- exactly the landmine the docstring above warns about.
# Retargeted at the thesis material this script's analysis actually lives in.
_OUTPUT = Path("docs/research/thesis/material/figures/pooled_vs_pertype")

# (label, rho, n) -- draft.md §5.5.
_POOLED = ("Pooled\n(all types)", 0.374, 1545)
_PER_TYPE = [
    ("Broker", 0.429, 36),
    ("InfraNode", 0.409, 119),
    ("Library", 0.351, 165),
    ("Application", 0.346, 850),
    ("Topic", 0.322, 375),
]

_POOLED_COLOR = "#C44E52"
_PER_TYPE_COLOR = "#4C72B0"


def _make_figure(output_path: Path, dpi: int = 300):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Skipping plot.", file=sys.stderr)
        return

    labels = [_POOLED[0]] + [r[0] for r in _PER_TYPE]
    rhos = [_POOLED[1]] + [r[1] for r in _PER_TYPE]
    ns = [_POOLED[2]] + [r[2] for r in _PER_TYPE]
    colors = [_POOLED_COLOR] + [_PER_TYPE_COLOR] * len(_PER_TYPE)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.8))

    bars = ax.bar(x, rhos, 0.6, color=colors, alpha=0.85, zorder=3)
    for xi, rho, n in zip(x, rhos, ns):
        ax.annotate(f"$\\rho={rho:.3f}$\n$n={n}$", xy=(xi, rho), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=9)

    # Reference band spanning the per-type range, to show the pooled figure
    # sits inside it rather than diverging (no Simpson's-paradox effect).
    per_type_rhos = [r[1] for r in _PER_TYPE]
    ax.axhspan(min(per_type_rhos), max(per_type_rhos), color=_PER_TYPE_COLOR,
               alpha=0.08, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"Spearman $\rho$ between $Q(v)$ and $I_{\mathrm{comp}}(v)$", fontsize=11)
    ax.set_title(
        "Pooled versus Per-Node-Type Correlation\n"
        "(pooled sits inside the per-type range -- no Simpson's-paradox effect)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(0.0, 0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        p = output_path.with_suffix(f".{ext}")
        plt.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close()


def main():
    print("\n  Pooled versus Per-Node-Type Spearman rho (draft.md §5.5, printed as Fig. 4)")
    print(f"    Pooled: rho={_POOLED[1]:.3f} n={_POOLED[2]}")
    for label, rho, n in _PER_TYPE:
        print(f"    {label}: rho={rho:.3f} n={n}")
    _make_figure(_OUTPUT)
    print("  Done.")


if __name__ == "__main__":
    main()
