#!/usr/bin/env python3
"""
reproduce/render_threshold_figure.py — RETIRED from the JSS manuscript
========================================================================

RETIRED: this figure is no longer in docs/research/jss/draft.md. When the
manuscript was condensed to fit JSS's page guidance it was dropped as redundant
-- Table 12 (§8.3) carries exactly the same numbers -- so the sweep is still
reported, just not plotted. The script is kept because the analysis is current
and a thesis chapter may want the plot, but it is NOT part of
`make -f reproduce/Makefile jss-figures` any more.

Reads results/threshold_sensitivity.json and produces the propagation_threshold
sweep plot (§8.3: mean Spearman ρ against I*(v) across the sweep, with the
canonical 0.20 default marked). See draft.md Table 12 / §8.3.

Output:
  docs/research/jss/latex/figures/Figure_6.png  (300 dpi)
  docs/research/jss/latex/figures/Figure_6.pdf  (vector)

Usage
-----
  python reproduce/render_threshold_figure.py
  python reproduce/render_threshold_figure.py --input results/threshold_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DEFAULT_INPUT = Path("results/threshold_sensitivity.json")
_DEFAULT_OUTPUT = Path("docs/research/jss/latex/figures/Figure_6")
_CANONICAL_THRESHOLD = 0.20  # draft.md's canonical propagation_threshold default

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

    thresholds = [r["propagation_threshold"] for r in rows]
    means = [r["mean_rho"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(thresholds, means, "o-", color=_LINE_COLOR, linewidth=2.0,
            markersize=6, zorder=3, label=r"mean $\rho$ (over 7 scenarios)")

    default_row = min(rows, key=lambda r: abs(r["propagation_threshold"] - _CANONICAL_THRESHOLD))
    ax.scatter([default_row["propagation_threshold"]], [default_row["mean_rho"]],
               s=110, facecolors="none", edgecolors=_MARK_COLOR, linewidths=2.0,
               zorder=4, label=rf"canonical default ($t={_CANONICAL_THRESHOLD:.2f}$)")

    ax.set_xlabel("propagation_threshold", fontsize=11, family="monospace")
    ax.set_ylabel(r"Spearman $\rho$ against $I^*(v)$ (mean over 7 scenarios)", fontsize=11)
    ax.set_title(
        "Propagation-Threshold Sensitivity\n"
        r"($\rho$ spans the sweep; vanishes entirely at $t=0$)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(-0.03, 1.03)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
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
    p = argparse.ArgumentParser(description="Propagation-threshold sensitivity (draft.md Figure 5, printed as Fig. 6)")
    p.add_argument("--input", type=Path, default=_DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    print("\n  Propagation-Threshold Sensitivity (draft.md Figure 5, printed as Fig. 6)")
    if not args.input.exists():
        print(f"\n  {args.input} not found. Run reproduce/threshold_sensitivity.py first.")
        sys.exit(0)

    data = json.loads(args.input.read_text())
    rows = sorted(data["threshold_sweep"], key=lambda r: r["propagation_threshold"])
    for r in rows:
        marker = " *" if abs(r["propagation_threshold"] - _CANONICAL_THRESHOLD) < 1e-9 else ""
        print(f"    t={r['propagation_threshold']:.2f}  mean_rho={r['mean_rho']:.3f}{marker}")

    _make_figure(rows, args.output, dpi=args.dpi)
    print("  Done.")


if __name__ == "__main__":
    main()
