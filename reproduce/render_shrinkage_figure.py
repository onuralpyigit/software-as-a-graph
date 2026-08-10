#!/usr/bin/env python3
"""
reproduce/render_shrinkage_figure.py — JSS Figure 5 generator
===============================================================

Reads results/ahp_shrinkage_sweep.json and produces draft.md's own "Figure 4"
(§8.3: mean Spearman ρ against I*(v) as the AHP shrinkage parameter λ blends
the stated composite weighting toward a uniform prior; λ=0 equal weights,
λ=1 raw AHP judgement). See draft.md Table 21 / §8.3.

NOTE on the file number: LaTeX numbers floats by reading order, not by the
source's own figure label -- because the attention-subgraph figure (draft.md's
"Figure 6") physically appears earlier in the text (§5.2), this one prints as
"Fig. 5". See the numbering table in docs/research/jss/latex/README.md.

Output:
  docs/research/jss/latex/figures/Figure_5.png  (300 dpi)
  docs/research/jss/latex/figures/Figure_5.pdf  (vector)

Usage
-----
  python reproduce/render_shrinkage_figure.py
  python reproduce/render_shrinkage_figure.py --input results/ahp_shrinkage_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DEFAULT_INPUT = Path("results/ahp_shrinkage_sweep.json")
_DEFAULT_OUTPUT = Path("docs/research/jss/latex/figures/Figure_5")
_DEFAULT_LAMBDA = 0.70  # the stated/canonical operating point (draft.md Table 21)

# House palette, matching reproduce/render_stratified_figure.py's _VARIANT_COLORS.
_LINE_COLOR = "#4C72B0"
_MARK_COLOR = "#C44E52"


def _make_figure(rows: list[dict], output_path: Path, dpi: int = 300):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.", file=sys.stderr)
        return

    lambdas = [r["lambda"] for r in rows]
    means = [r["mean_rho"] for r in rows]
    stds = [r.get("std_rho", 0.0) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.errorbar(lambdas, means, yerr=stds, fmt="o-", color=_LINE_COLOR,
                linewidth=2.0, markersize=6, capsize=3, zorder=3,
                label=r"mean $\rho$ (± std across 7 scenarios)")

    # Mark the canonical/default lambda.
    default_row = min(rows, key=lambda r: abs(r["lambda"] - _DEFAULT_LAMBDA))
    ax.scatter([default_row["lambda"]], [default_row["mean_rho"]],
               s=110, facecolors="none", edgecolors=_MARK_COLOR, linewidths=2.0,
               zorder=4, label=rf"stated default ($\lambda={_DEFAULT_LAMBDA:.2f}$)")

    ax.set_xlabel(r"$\lambda$ (0 = equal weights, 1 = raw AHP judgement)", fontsize=11)
    ax.set_ylabel(r"Spearman $\rho$ against $I^*(v)$ (mean over 7 scenarios)", fontsize=11)
    ax.set_title(
        "AHP Shrinkage Sensitivity\n"
        r"(monotone decline in $\rho$; no plateau anywhere in $[0,1]$)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(-0.03, 1.03)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
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


def parse_args():
    p = argparse.ArgumentParser(description="AHP shrinkage sensitivity (draft.md Figure 4, printed as Fig. 5)")
    p.add_argument("--input", type=Path, default=_DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    print("\n  AHP Shrinkage Sensitivity (draft.md Figure 4, printed as Fig. 5)")
    if not args.input.exists():
        print(f"\n  {args.input} not found. Run reproduce/ahp_sensitivity.py first.")
        sys.exit(0)

    data = json.loads(args.input.read_text())
    rows = sorted(data["rows"], key=lambda r: r["lambda"])
    for r in rows:
        marker = " *" if abs(r["lambda"] - _DEFAULT_LAMBDA) < 1e-9 else ""
        print(f"    lambda={r['lambda']:.2f}  mean_rho={r['mean_rho']:.3f}{marker}")

    _make_figure(rows, args.output, dpi=args.dpi)
    print("  Done.")


if __name__ == "__main__":
    main()
