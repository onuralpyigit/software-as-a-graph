#!/usr/bin/env python3
"""
reproduce/render_stratified_figure.py — Block F: Figure 4 generator
================================================================

Reads EvalMetrics per_node_type data from training run JSON outputs and
produces Figure 4: per-node-type Spearman ρ stratified by variant.

Sources (tries each in order):
  1. results/main_table.json  (from reproduce/middleware26_main_table.py)
  2. output/loso/<variant>/results.json  (per-variant LOSO)

Output:
  results/figure4_stratified_rho.png  / .pdf
  results/figure4_stratified_rho.md   (console table)

Usage
-----
  python reproduce/render_stratified_figure.py
  python reproduce/render_stratified_figure.py --source main_table
  python reproduce/render_stratified_figure.py --source loso
  python reproduce/render_stratified_figure.py --table-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Config ────────────────────────────────────────────────────────────────────

_NODE_TYPE_ORDER  = ["Application", "Broker", "Topic", "Node", "Library"]
_NODE_TYPE_LABELS = {
    "Application": "App",
    "Broker":      "Broker",
    "Topic":       "Topic",
    "Node":        "Node",
    "Library":     "Lib",
}
_VARIANT_ORDER = ["topology_rmav", "gl", "gl_qos", "hgl", "hgl_qos"]
_VARIANT_LABELS = {
    "hgl_qos":         "HGL-QoS",
    "hgl":             "HGL",
    "gl_qos":          "GL-QoS",
    "gl":              "GL",
    "topology_rmav":   "RMAV baseline",
}
_VARIANT_COLORS = {
    "hgl_qos":         "#4C72B0",
    "hgl":             "#55A868",
    "gl_qos":          "#DD8452",
    "gl":              "#8172B3",
    "topology_rmav":   "#C44E52",
}

_RESULTS_DIR = Path("results")
_LOSO_DIR    = Path("output/loso")


# ── Data extraction ───────────────────────────────────────────────────────────

def _extract_from_main_table(path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Extract per_node_type ρ from main_table.json cells.

    Returns: {variant: {node_type: {"mean": float, "std": float}}}
    """
    import numpy as np

    data = json.loads(path.read_text())
    cells = data.get("cells", [])

    # Aggregate per (variant, node_type) across scenarios and seeds
    buf: Dict[str, Dict[str, List[float]]] = {}

    for cell in cells:
        if "error" in cell:
            continue
        variant = cell.get("variant")
        per_type = cell.get("per_node_type_rho") or cell.get("per_node_type", {})
        if not variant or not per_type:
            continue
        buf.setdefault(variant, {})
        for nt, rho in per_type.items():
            if rho is not None and not (isinstance(rho, float) and (rho != rho)):  # skip NaN
                buf[variant].setdefault(nt, []).append(float(rho))

    result = {}
    for var, nt_dict in buf.items():
        result[var] = {}
        for nt, vals in nt_dict.items():
            result[var][nt] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)) if len(vals) > 1 else 0.0,
                "n":    len(vals),
            }
    return result


def _extract_from_loso(loso_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Extract per_node_type ρ from loso/<variant>/results.json files."""
    import numpy as np

    result = {}
    for var in _VARIANT_ORDER:
        rp = loso_dir / var / "results.json"
        if not rp.exists():
            continue
        data = json.loads(rp.read_text())
        folds = data.get("folds", [])
        nt_buf: Dict[str, List[float]] = {}
        for fold in folds:
            per_type = fold.get("per_type_rho", {})
            for nt, info in per_type.items():
                nt_buf.setdefault(nt, []).append(float(info.get("mean", 0.0)))
        if nt_buf:
            result[var] = {}
            for nt, vals in nt_buf.items():
                result[var][nt] = {
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals)) if len(vals) > 1 else 0.0,
                    "n":    len(vals),
                }
    return result


def _load_stratified_data(
    source: str,
    main_table_path: Path,
    loso_dir: Path,
) -> Optional[Dict[str, Dict[str, Dict[str, float]]]]:
    """Load per-node-type ρ from the specified source."""
    if source in ("main_table", "auto"):
        if main_table_path.exists():
            data = _extract_from_main_table(main_table_path)
            if data:
                print(f"  Source: {main_table_path}")
                return data
    if source in ("loso", "auto"):
        data = _extract_from_loso(loso_dir)
        if data:
            print(f"  Source: {loso_dir}/<variant>/results.json")
            return data
    return None


# ── Console table ─────────────────────────────────────────────────────────────

def _print_stratified_table(stratified: Dict[str, Dict[str, Dict[str, float]]]):
    node_types = [nt for nt in _NODE_TYPE_ORDER
                  if any(nt in stratified.get(v, {}) for v in _VARIANT_ORDER)]
    if not node_types:
        node_types = sorted({nt for v in stratified.values() for nt in v})

    print("\n  Figure 4 — Per-Node-Type Spearman ρ")
    col_w = 14
    header = f"  {'Variant':<22}" + "".join(
        f"{_NODE_TYPE_LABELS.get(nt, nt):<{col_w}}" for nt in node_types
    )
    print(header)
    print("  " + "─" * (22 + col_w * len(node_types)))

    for var in _VARIANT_ORDER:
        if var not in stratified:
            continue
        label = _VARIANT_LABELS.get(var, var)
        row = f"  {label:<22}"
        for nt in node_types:
            info = stratified[var].get(nt, {})
            mean = info.get("mean")
            std  = info.get("std")
            if mean is None:
                row += f"{'—':<{col_w}}"
            else:
                cell = f"{mean:.3f}±{std:.3f}"
                row += f"{cell:<{col_w}}"
        print(row)
    print()


# ── Plot ──────────────────────────────────────────────────────────────────────

def _make_figure(
    stratified: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
    dpi: int = 180,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Skipping plot.", file=sys.stderr)
        return

    node_types = [nt for nt in _NODE_TYPE_ORDER
                  if any(nt in stratified.get(v, {}) for v in _VARIANT_ORDER)]
    if not node_types:
        node_types = sorted({nt for v in stratified.values() for nt in v})

    variants_present = [v for v in _VARIANT_ORDER if v in stratified]
    n_vars = len(variants_present)
    n_types = len(node_types)
    bar_w = 0.8 / n_vars
    x = np.arange(n_types)

    fig, ax = plt.subplots(figsize=(max(8, n_types * 1.8), 5))

    for i, var in enumerate(variants_present):
        offsets = x + (i - n_vars / 2 + 0.5) * bar_w
        means = [stratified[var].get(nt, {}).get("mean", 0.0) or 0.0 for nt in node_types]
        stds  = [stratified[var].get(nt, {}).get("std",  0.0) or 0.0 for nt in node_types]

        bars = ax.bar(offsets, means, bar_w * 0.9,
                      label=_VARIANT_LABELS.get(var, var),
                      color=_VARIANT_COLORS.get(var, f"C{i}"),
                      alpha=0.85, zorder=3)
        ax.errorbar(offsets, means, yerr=stds,
                    fmt="none", color="black", capsize=3,
                    linewidth=1.0, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([_NODE_TYPE_LABELS.get(nt, nt) for nt in node_types], fontsize=11)
    ax.set_ylabel("Spearman ρ (mean ± std)", fontsize=11)
    ax.set_title(
        "Figure 4: Per-Node-Type Spearman ρ by Variant\n"
        "(HGL-QoS gains most on QoS-bearing node types: App, Topic)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(bottom=0.0)
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


# ── Markdown output ───────────────────────────────────────────────────────────

def _save_md(stratified: Dict, output: Path):
    node_types = [nt for nt in _NODE_TYPE_ORDER
                  if any(nt in stratified.get(v, {}) for v in _VARIANT_ORDER)]
    headers = ["Variant"] + [_NODE_TYPE_LABELS.get(nt, nt) for nt in node_types]
    rows = [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for var in _VARIANT_ORDER:
        if var not in stratified:
            continue
        label = _VARIANT_LABELS.get(var, var)
        cells = [label]
        for nt in node_types:
            info = stratified[var].get(nt, {})
            mean = info.get("mean")
            std  = info.get("std")
            cells.append(f"{mean:.3f}±{std:.3f}" if mean is not None else "—")
        rows.append("| " + " | ".join(cells) + " |")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n")
    print(f"  Saved MD: {output}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Block F: Figure 4 — per-node-type ρ stratification.")
    p.add_argument("--source", choices=["auto", "main_table", "loso"], default="auto",
                   help="Data source (default: auto-detect)")
    p.add_argument("--main-table", type=Path, default=_RESULTS_DIR / "main_table.json")
    p.add_argument("--loso-dir", type=Path, default=_LOSO_DIR)
    p.add_argument("--output", type=Path, default=_RESULTS_DIR / "figure4_stratified_rho")
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--table-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print("\n  Figure 4 — Stratified ρ by Node Type (Block F)")

    data = _load_stratified_data(args.source, args.main_table, args.loso_dir)
    if data is None:
        print("\n  No stratified data found. Run one of:")
        print("    python reproduce/middleware26_main_table.py   (for main_table.json)")
        print("    python reproduce/loso_all_variants.py         (for LOSO results)")
        sys.exit(0)

    _print_stratified_table(data)

    # Save markdown table
    _save_md(data, args.output.with_suffix(".md"))

    if not args.table_only:
        _make_figure(data, args.output, dpi=args.dpi)
    print("  Done.")


if __name__ == "__main__":
    main()
