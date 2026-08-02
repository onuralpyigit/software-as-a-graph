#!/usr/bin/env python3
"""
reproduce/feature_shift_diagnostic.py
======================================

Quantifies cross-scenario feature scale drift in the node feature matrices the
GNN actually trains on (each scenario's ``bundle.hetero_data[node_type].x``,
built by the same :func:`networkx_to_hetero_data` path used in training/LOSO).

Motivation: several of the scale-dependent columns in ``BASE_METRIC_KEYS``
(pagerank, eigenvector/closeness/betweenness centrality, mpci, cdi,
qos_weight, ...) are computed on a per-graph basis and are not
scale-invariant across graphs of very different size — e.g. PageRank sums to
1 over all nodes, so its mean tracks 1/N. Under Leave-One-Scenario-Out, a
column whose mean varies by an order of magnitude across scenarios pushes
held-out scenarios off the distribution the model was trained on. This script
reports, per node type and per column, the per-scenario mean/std/max, so the
magnitude of that drift can be read off directly rather than inferred.

Read-only: loads cached scenario bundles, computes summary statistics, writes
a Markdown table to ``results/feature_shift_diagnostic.md``. Does not modify
any model, cache, or training code.

Usage:
    PYTHONPATH=. python reproduce/feature_shift_diagnostic.py \\
        --cache-dir output/loso_cache --output results/feature_shift_diagnostic.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.loso_evaluate import discover_scenarios  # noqa: E402
from saag.prediction.data_preparation import KEYS_BY_TYPE  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def _shift_ratio(values: List[float]) -> float:
    """max(|mean|) / min(|mean|) across scenarios for one column, guarding zeros."""
    abs_means = [abs(v) for v in values if abs(v) > 1e-12]
    if len(abs_means) < 2:
        return float("nan")
    return max(abs_means) / min(abs_means)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=Path("output/loso_cache"))
    p.add_argument("--output", type=Path, default=Path("results/feature_shift_diagnostic.md"))
    p.add_argument("--skip", default="", help="Comma-separated scenario ids to skip")
    args = p.parse_args()

    skip = [s.strip() for s in args.skip.split(",") if s.strip()]
    bundles = discover_scenarios(args.cache_dir, skip)
    if not bundles:
        print(f"No scenario bundles found under {args.cache_dir}", file=sys.stderr)
        sys.exit(1)

    lines: List[str] = []
    lines.append("# Feature-Shift Diagnostic\n")
    lines.append(
        "Per-scenario mean/std/max of each node-feature column, as it actually "
        "reaches the model (`bundle.hetero_data[node_type].x`). "
        "`shift` = max(|mean|) / min(|mean|) across scenarios — "
        "large values indicate a column whose scale is not comparable across "
        "graphs of different size, which is a liability under LOSO.\n"
    )

    node_types = sorted({nt for b in bundles for nt in b.hetero_data.node_types})
    scenario_ids = [b.scenario_id for b in bundles]

    for node_type in node_types:
        keys = KEYS_BY_TYPE.get(node_type)
        if keys is None:
            continue
        lines.append(f"\n## {node_type}\n")
        header = "| feature | " + " | ".join(f"{sid} (mean)" for sid in scenario_ids) + " | shift |"
        sep = "|---" * (len(scenario_ids) + 2) + "|"
        lines.append(header)
        lines.append(sep)

        # Gather per-scenario column stats.
        per_scenario_matrix: Dict[str, np.ndarray] = {}
        for b in bundles:
            if node_type not in b.hetero_data.node_types:
                continue
            store = b.hetero_data[node_type]
            if not hasattr(store, "x") or store.x is None or store.x.numel() == 0:
                continue
            per_scenario_matrix[b.scenario_id] = store.x.detach().cpu().numpy()

        for col_idx, key in enumerate(keys):
            row_means = []
            for sid in scenario_ids:
                mat = per_scenario_matrix.get(sid)
                if mat is None or col_idx >= mat.shape[1] or mat.shape[0] == 0:
                    row_means.append(float("nan"))
                    continue
                row_means.append(float(mat[:, col_idx].mean()))
            shift = _shift_ratio(row_means)
            cells = " | ".join(
                "—" if np.isnan(m) else f"{m:.4g}" for m in row_means
            )
            shift_str = "—" if np.isnan(shift) else f"{shift:.1f}x"
            lines.append(f"| {key} | {cells} | {shift_str} |")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output}")

    # Console summary: top-10 worst-shift columns across all node types.
    print("\nTop feature-scale shifts (max(|mean|)/min(|mean|) across scenarios):")
    worst: List[tuple] = []
    for node_type in node_types:
        keys = KEYS_BY_TYPE.get(node_type)
        if keys is None:
            continue
        per_scenario_matrix = {
            b.scenario_id: b.hetero_data[node_type].x.detach().cpu().numpy()
            for b in bundles
            if node_type in b.hetero_data.node_types
            and hasattr(b.hetero_data[node_type], "x")
            and b.hetero_data[node_type].x is not None
            and b.hetero_data[node_type].x.numel() > 0
        }
        for col_idx, key in enumerate(keys):
            row_means = [
                float(mat[:, col_idx].mean())
                for mat in per_scenario_matrix.values()
                if col_idx < mat.shape[1] and mat.shape[0] > 0
            ]
            shift = _shift_ratio(row_means)
            if not np.isnan(shift):
                worst.append((shift, node_type, key))
    worst.sort(reverse=True)
    for shift, node_type, key in worst[:10]:
        print(f"  {shift:8.1f}x  {node_type:12s} {key}")


if __name__ == "__main__":
    main()
