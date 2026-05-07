#!/usr/bin/env python3
"""
tools/middleware26_main_table.py — Block C: Main Results Table Harness
======================================================================

Orchestrates the 8×4×5 training matrix for Table 3 (paper §6.2):
  8 scenarios × 4 variants × 5 seeds = 160 training runs.

For each cell, emits:
  - Spearman ρ (composite), per-node-type ρ, F1, RMSE, NDCG@10
  - Paired Wilcoxon signed-rank p-value (hetero_qos vs each baseline)
  - Bootstrap 95% CI (B=2000)

Output: results/main_table.json + results/main_table.tex (via render_table.py)

Usage
-----
  # Full matrix (≈ 160 runs, 30-60 min on CPU)
  python tools/middleware26_main_table.py

  # Quick smoke test: 1 scenario, 2 seeds
  python tools/middleware26_main_table.py --scenarios atm_system --seeds 42 123

  # Resume from partial results (skips completed cells)
  python tools/middleware26_main_table.py --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path for direct execution
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

ALL_SCENARIOS = [
    "atm_system",
    "av_system",
    "iot_smart_city_system",
    "financial_trading_system",
    "healthcare_system",
    "hub_and_spoke_system",
    "microservices_system",
    "enterprise_system",
]

ALL_VARIANTS = ["topology_rmav", "homo_unweighted", "homo_scalar", "hetero_qos"]

DEFAULT_SEEDS = [42, 123, 456, 789, 2024]

RESULTS_DIR = Path("results")
SCENARIOS_DIR = Path("data/scenarios")
LOSO_CACHE_DIR = Path("output/loso_cache")


# ── Per-cell training ─────────────────────────────────────────────────────────

def _load_scenario_data(scenario: str) -> Tuple[Any, Dict, Dict, Dict]:
    """Load graph + structural/simulation/RMAV data for a scenario.

    Tries LOSO cache first, then falls back to generating from scenario JSON.
    """
    import networkx as nx
    from saag.prediction.data_preparation import (
        extract_structural_metrics_dict,
        extract_simulation_dict,
        extract_rmav_scores_dict,
    )

    # Try loso cache
    cache_dir = LOSO_CACHE_DIR / scenario
    json_path = SCENARIOS_DIR / f"{scenario}.json"

    if not json_path.exists():
        # Try without _system suffix
        for ext in ["", ".yaml"]:
            alt = SCENARIOS_DIR / f"{scenario}{ext}"
            if alt.exists():
                json_path = alt
                break

    if not json_path.exists():
        raise FileNotFoundError(f"Scenario file not found for '{scenario}'")

    # Build graph from JSON
    from cli.loso_evaluate import _build_graph_from_json
    topology = json.loads(json_path.read_text()) if json_path.suffix == ".json" else None
    if topology is None:
        raise ValueError(f"Non-JSON scenario not yet supported: {json_path}")

    nx_graph = _build_graph_from_json(topology)

    # Load pre-computed metrics if available
    structural_dict: Dict = {}
    simulation_dict: Dict = {}
    rmav_dict: Dict = {}

    if cache_dir.exists():
        sm_path = cache_dir / "structural_metrics.json"
        fi_path = cache_dir / "failure_impact.json"
        qi_path = cache_dir / "quality_scores.json"
        if sm_path.exists():
            structural_dict = json.loads(sm_path.read_text())
        if fi_path.exists():
            simulation_dict = json.loads(fi_path.read_text())
        if qi_path.exists():
            rmav_dict = json.loads(qi_path.read_text())
    else:
        logger.warning("No LOSO cache for '%s'. Structural/simulation data will be empty.", scenario)

    return nx_graph, structural_dict, simulation_dict, rmav_dict


def _train_cell(
    scenario: str,
    variant: str,
    seed: int,
    hidden: int = 64,
    num_heads: int = 4,
    num_layers: int = 3,
    dropout: float = 0.2,
    num_epochs: int = 200,
    patience: int = 30,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Dict[str, Any]:
    """Train one cell of the 8×4×5 matrix and return metrics dict."""
    import torch
    import numpy as np
    from saag.prediction.data_preparation import networkx_to_hetero_data, create_node_splits
    from saag.prediction.trainer import GNNTrainer, evaluate

    torch.manual_seed(seed)
    np.random.seed(seed)

    nx_graph, structural_dict, simulation_dict, rmav_dict = _load_scenario_data(scenario)

    if nx_graph.number_of_nodes() == 0:
        return {"error": "empty_graph"}

    start = time.time()

    if variant == "topology_rmav":
        # RMAV baseline: use simulation ground truth directly as prediction
        if not rmav_dict or not simulation_dict:
            return {"error": "no_rmav_or_simulation_data"}
        from scipy.stats import spearmanr
        # Align keys
        keys = sorted(set(rmav_dict) & set(simulation_dict))
        if len(keys) < 3:
            return {"error": "insufficient_overlap"}
        pred_r = [rmav_dict[k].get("composite", rmav_dict[k].get("overall", 0)) for k in keys]
        true_r = [simulation_dict[k].get("composite", 0) for k in keys]
        rho, _ = spearmanr(pred_r, true_r)
        return {
            "scenario": scenario, "variant": variant, "seed": seed,
            "spearman_rho": float(rho), "f1_score": 0.0, "rmse": 0.0,
            "mae": 0.0, "ndcg_10": 0.0, "per_node_type": {},
            "runtime_s": round(time.time() - start, 2),
        }

    elif variant in ("homo_unweighted", "homo_scalar"):
        from saag.prediction.models.baselines import build_baseline
        conv = networkx_to_hetero_data(nx_graph, structural_dict, simulation_dict, rmav_dict)
        data = conv.hetero_data
        create_node_splits(data, train_ratio, val_ratio, seed=seed)
        model = build_baseline(variant, hidden_channels=hidden, num_heads=num_heads,
                               num_layers=num_layers, dropout=dropout)
        ckpt_dir = f"output/gnn_checkpoints/{scenario}_{variant}_s{seed}"
        trainer = GNNTrainer(model=model, checkpoint_dir=ckpt_dir, lr=3e-4,
                             num_epochs=num_epochs, patience=patience)
        trainer.train(data)
        device = torch.device("cpu")
        metrics = evaluate(model, data, "test_mask", device)

    else:  # hetero_qos
        from saag.prediction.gnn_service import GNNService
        svc = GNNService(hidden_channels=hidden, num_heads=num_heads, num_layers=num_layers,
                         dropout=dropout, predict_edges=False,
                         checkpoint_dir=f"output/gnn_checkpoints/{scenario}_hetero_qos_s{seed}")
        result = svc.train(
            graph=nx_graph, structural_metrics=structural_dict,
            simulation_results=simulation_dict, rmav_scores=rmav_dict,
            train_ratio=train_ratio, val_ratio=val_ratio,
            num_epochs=num_epochs, lr=3e-4, patience=patience,
            seeds=[seed], mode="gnn",
        )
        metrics = result.gnn_metrics

    if metrics is None:
        return {"error": "no_metrics"}

    d = metrics.to_dict()
    d.update({"scenario": scenario, "variant": variant, "seed": seed,
              "runtime_s": round(time.time() - start, 2)})
    return d


# ── Statistics helpers ─────────────────────────────────────────────────────────

def _bootstrap_ci(values: List[float], stat_fn=None, B: int = 2000, alpha: float = 0.05):
    """Bootstrap confidence interval for stat_fn(values). Default: mean."""
    import numpy as np
    if not values:
        return float("nan"), float("nan")
    stat_fn = stat_fn or np.mean
    arr = np.array(values)
    boot = np.array([stat_fn(np.random.choice(arr, len(arr), replace=True)) for _ in range(B)])
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def _wilcoxon_p(a: List[float], b: List[float]) -> float:
    """Two-sided Wilcoxon signed-rank test p-value."""
    from scipy.stats import wilcoxon
    import numpy as np
    if len(a) < 2 or len(a) != len(b):
        return float("nan")
    try:
        _, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        return float(p)
    except Exception:
        return float("nan")


def _aggregate_cells(cells: List[Dict]) -> Dict:
    """Aggregate seed-level cells into mean ± CI and Wilcoxon p-values."""
    import numpy as np

    by_sv: Dict[Tuple[str, str], List[Dict]] = {}
    for c in cells:
        if "error" in c:
            continue
        k = (c["scenario"], c["variant"])
        by_sv.setdefault(k, []).append(c)

    aggregate = {}
    for (sc, var), cs in by_sv.items():
        rhos = [c["spearman_rho"] for c in cs]
        mean_r = float(np.mean(rhos))
        lo, hi = _bootstrap_ci(rhos)
        aggregate[(sc, var)] = {
            "mean_rho": round(mean_r, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "n_seeds": len(cs),
            "rhos": rhos,
        }

    # Per-scenario Wilcoxon p-values (hetero_qos vs each baseline)
    scenarios = sorted({sc for sc, _ in aggregate})
    for sc in scenarios:
        ref_cells = [(sc, "hetero_qos")]
        ref_rhos = aggregate.get(ref_cells[0], {}).get("rhos", [])
        for var in ALL_VARIANTS:
            if var == "hetero_qos":
                continue
            base_rhos = aggregate.get((sc, var), {}).get("rhos", [])
            p = _wilcoxon_p(ref_rhos, base_rhos)
            if (sc, var) in aggregate:
                aggregate[(sc, var)]["wilcoxon_p_vs_hetero"] = round(p, 4)

    return aggregate


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Middleware 2026 main table harness (Block C).")
    p.add_argument("--scenarios", nargs="+", default=None,
                   help=f"Scenarios to run (default: all {len(ALL_SCENARIOS)})")
    p.add_argument("--variants", nargs="+", default=None, choices=ALL_VARIANTS,
                   help="Variants to include (default: all 4)")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help=f"Seeds to use (default: {DEFAULT_SEEDS})")
    p.add_argument("--epochs", type=int, default=200, help="Training epochs per run (default: 200)")
    p.add_argument("--patience", type=int, default=30, help="Early stopping patience (default: 30)")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "main_table.json")
    p.add_argument("--resume", action="store_true",
                   help="Skip cells already present in the output file")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned matrix without training")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    scenarios = args.scenarios or ALL_SCENARIOS
    variants = args.variants or ALL_VARIANTS
    seeds = args.seeds or DEFAULT_SEEDS

    total = len(scenarios) * len(variants) * len(seeds)
    print(f"\n  Middleware 2026 Main Table Harness")
    print(f"  Scenarios : {scenarios}")
    print(f"  Variants  : {variants}")
    print(f"  Seeds     : {seeds}")
    print(f"  Total runs: {total}")
    print()

    if args.dry_run:
        for sc in scenarios:
            for v in variants:
                for s in seeds:
                    print(f"  [DRY-RUN] {sc} × {v} × seed={s}")
        return

    # Load existing results for --resume
    existing_cells: List[Dict] = []
    if args.resume and args.output.exists():
        data = json.loads(args.output.read_text())
        existing_cells = data.get("cells", [])
        done_keys = {(c["scenario"], c["variant"], c["seed"]) for c in existing_cells
                     if "error" not in c}
        print(f"  Resuming: {len(done_keys)} cells already completed.")
    else:
        done_keys = set()

    cells: List[Dict] = list(existing_cells)
    n_done = len(done_keys)
    n_fail = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)

    for sc in scenarios:
        for v in variants:
            for s in seeds:
                key = (sc, v, s)
                if key in done_keys:
                    continue

                label = f"[{n_done+1}/{total}] {sc} × {v} × seed={s}"
                print(f"  {label} ...", end="", flush=True)

                try:
                    cell = _train_cell(
                        scenario=sc, variant=v, seed=s,
                        hidden=args.hidden, num_heads=args.heads,
                        num_layers=args.layers, num_epochs=args.epochs,
                        patience=args.patience,
                    )
                except Exception as exc:
                    cell = {"scenario": sc, "variant": v, "seed": s, "error": str(exc)}
                    n_fail += 1
                    print(f" ERROR: {exc}")
                else:
                    rho = cell.get("spearman_rho", "?")
                    rt = cell.get("runtime_s", "?")
                    print(f" rho={rho:.4f}  ({rt}s)")

                cells.append(cell)
                n_done += 1

                # Save incrementally
                args.output.write_text(json.dumps({"cells": cells}, indent=2))

    # Aggregate
    aggregate = _aggregate_cells(cells)
    agg_serializable = {
        f"{sc}|{var}": v for (sc, var), v in aggregate.items()
    }

    output = {
        "cells": cells,
        "aggregate": agg_serializable,
        "config": {
            "scenarios": scenarios, "variants": variants, "seeds": seeds,
            "epochs": args.epochs, "hidden": args.hidden,
        },
    }
    args.output.write_text(json.dumps(output, indent=2))

    print(f"\n  Completed: {n_done} runs, {n_fail} failures.")
    print(f"  Results saved to: {args.output}")

    if n_fail == 0:
        print("\n  ✓ All cells complete. Run tools/render_table.py to generate LaTeX.")
    else:
        print(f"\n  ⚠  {n_fail} cells failed. Check logs for details.")


if __name__ == "__main__":
    main()
