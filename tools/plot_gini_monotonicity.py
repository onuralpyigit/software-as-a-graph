#!/usr/bin/env python3
"""
tools/plot_gini_monotonicity.py — Block D: Figure 3 generator
=============================================================

Reads results/gini_sweep.json and produces:
  - Figure 3 (PNG + PDF): Spearman ρ vs QoS Gini for all 4 variants
  - Console monotonicity verdict table for paper §6.3

Usage
-----
  python tools/plot_gini_monotonicity.py
  python tools/plot_gini_monotonicity.py --input results/gini_sweep.json \\
      --output results/figure3_gini_monotonicity.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Variant display config ─────────────────────────────────────────────────────

_VARIANT_CONFIG = {
    "hetero_qos":     {"label": "Q-HGL (ours)",          "color": "#4C72B0", "ls": "-",  "marker": "o", "lw": 2.5, "zorder": 10},
    "homo_scalar":    {"label": "Homo-Scalar QoS",        "color": "#DD8452", "ls": "--", "marker": "s", "lw": 1.8, "zorder": 7},
    "homo_unweighted":{"label": "Homo-Unweighted",        "color": "#55A868", "ls": ":",  "marker": "^", "lw": 1.8, "zorder": 6},
    "topology_rmav":  {"label": "RMAV (topology only)",   "color": "#C44E52", "ls": "-.", "marker": "D", "lw": 1.8, "zorder": 5},
}

_VARIANT_ORDER = ["hetero_qos", "homo_scalar", "homo_unweighted", "topology_rmav"]


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_sweep_data(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Sweep results not found: {path}\n"
            "Run:  python tools/qos_gini_sweep.py"
        )
    return json.loads(path.read_text())


def _extract_scenario_data(data: Dict, scenario: str) -> Dict[str, List[Tuple[float, float, float]]]:
    """Extract {variant: [(gini, mean_rho, std_rho)]} for a given scenario."""
    agg = data.get("aggregate", {})
    result = {}
    for var in _VARIANT_ORDER:
        key = f"{scenario}|{var}"
        if key not in agg:
            continue
        entries = agg[key].get("gini_rho", [])
        result[var] = [(e["gini"], e["mean_rho"], e["std_rho"]) for e in entries]
    return result


def _get_scenarios(data: Dict) -> List[str]:
    agg = data.get("aggregate", {})
    scenarios = []
    for key in agg:
        sc = key.split("|")[0]
        if sc not in scenarios:
            scenarios.append(sc)
    return scenarios


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_single_scenario(ax, var_data: Dict, scenario: str, show_legend: bool = True):
    """Plot ρ vs Gini for one scenario onto ax."""
    for var in _VARIANT_ORDER:
        if var not in var_data:
            continue
        entries = var_data[var]
        if not entries:
            continue
        cfg = _VARIANT_CONFIG[var]
        ginis = [e[0] for e in entries]
        rhos  = [e[1] for e in entries]
        stds  = [e[2] for e in entries]

        ax.plot(ginis, rhos,
                label=cfg["label"], color=cfg["color"],
                linestyle=cfg["ls"], marker=cfg["marker"],
                linewidth=cfg["lw"], markersize=7, zorder=cfg["zorder"])
        ax.fill_between(
            ginis,
            [r - s for r, s in zip(rhos, stds)],
            [r + s for r, s in zip(rhos, stds)],
            color=cfg["color"], alpha=0.12, zorder=cfg["zorder"] - 1
        )

    ax.set_xlabel("QoS Gini Coefficient", fontsize=11)
    ax.set_ylabel("Spearman ρ (composite)", fontsize=11)
    ax.set_title(f"{scenario.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.05, 0.85)
    ax.set_ylim(bottom=0.0)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.grid(True, linestyle="--", alpha=0.4, color="grey")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(fontsize=9, loc="lower right", framealpha=0.9)


def make_figure(data: Dict, output_path: Path, dpi: int = 180):
    """Create the full Figure 3 panel (one sub-plot per scenario)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    scenarios = _get_scenarios(data)
    if not scenarios:
        print("No aggregate data found. Run qos_gini_sweep.py first.")
        return

    ncols = min(3, len(scenarios))
    nrows = (len(scenarios) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             squeeze=False)
    fig.suptitle(
        "Figure 3: Spearman ρ vs QoS Gini Coefficient\n"
        "(Q-HGL gains more from heterogeneous QoS; baselines are flat)",
        fontsize=13, fontweight="bold", y=0.98
    )

    for i, sc in enumerate(scenarios):
        row, col = divmod(i, ncols)
        var_data = _extract_scenario_data(data, sc)
        plot_single_scenario(axes[row][col], var_data, sc, show_legend=(i == 0))

    # Hide unused sub-plots
    for j in range(len(scenarios), nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for ext in ["png", "pdf"]:
        p = output_path.with_suffix(f".{ext}")
        plt.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {p}")

    plt.close()


# ── Console table ─────────────────────────────────────────────────────────────

def _print_monotonicity_table(data: Dict):
    agg = data.get("aggregate", {})
    scenarios = _get_scenarios(data)

    print("\n  Monotonicity Table — paper §6.3")
    print(f"  {'Scenario':<30} {'Variant':<22} {'ρ@G=0':<7} {'ρ@G=0.4':<8} "
          f"{'ρ@G=0.8':<8} {'ρ(Gini)':<8} {'Monotone?'}")
    print("  " + "-" * 90)

    for sc in scenarios:
        for var in _VARIANT_ORDER:
            key = f"{sc}|{var}"
            if key not in agg:
                continue
            entry = agg[key]
            mono = entry.get("monotonicity", {})
            gr = {e["gini"]: e["mean_rho"] for e in entry.get("gini_rho", [])}

            rho_g0  = f"{gr.get(0.0, float('nan')):.3f}"
            rho_g4  = f"{gr.get(0.4, float('nan')):.3f}"
            rho_g8  = f"{gr.get(0.8, float('nan')):.3f}"
            rho_gini = f"{mono.get('spearman_with_gini', float('nan')):.3f}"
            is_mono = "✓" if mono.get("is_monotone") else "✗"
            cfg_label = _VARIANT_CONFIG.get(var, {}).get("label", var)

            print(f"  {sc:<30} {cfg_label:<22} {rho_g0:<7} {rho_g4:<8} "
                  f"{rho_g8:<8} {rho_gini:<8} {is_mono}")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate Figure 3: ρ vs Gini (Block D).")
    p.add_argument("--input",  type=Path, default=Path("results/gini_sweep.json"))
    p.add_argument("--output", type=Path, default=Path("results/figure3_gini_monotonicity"))
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--table-only", action="store_true",
                   help="Print console table only, skip PNG/PDF generation")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n  Figure 3 Generator — Gini Monotonicity")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}.[png|pdf]")

    data = _load_sweep_data(args.input)
    _print_monotonicity_table(data)

    if not args.table_only:
        make_figure(data, args.output, dpi=args.dpi)
    print("\n  Done.")


if __name__ == "__main__":
    main()
