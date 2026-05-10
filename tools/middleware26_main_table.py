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

    # Try LOSO cache — exact match first, then fuzzy (dir may be named scenario_10_atm_system)
    cache_dir = LOSO_CACHE_DIR / scenario
    if not cache_dir.exists() and LOSO_CACHE_DIR.exists():
        for d in LOSO_CACHE_DIR.iterdir():
            if d.is_dir() and scenario in d.name:
                cache_dir = d
                logger.debug("Fuzzy cache match: %s → %s", scenario, d)
                break

    json_path = SCENARIOS_DIR / f"{scenario}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Scenario file not found for '{scenario}'")

    from cli.loso_evaluate import _build_graph_from_json
    topology = json.loads(json_path.read_text())
    nx_graph = _build_graph_from_json(topology)

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
            simulation_dict = _parse_failure_impact(json.loads(fi_path.read_text()))
        if qi_path.exists():
            rmav_dict = _parse_quality_scores(json.loads(qi_path.read_text()))
        # Re-map node IDs from cache format (A0, A1) to graph format (A01, A02)
        graph_nodes = set(str(n) for n in nx_graph.nodes())
        simulation_dict = _remap_node_ids(simulation_dict, graph_nodes)
        rmav_dict = _remap_node_ids(rmav_dict, graph_nodes)
    else:
        logger.warning("No LOSO cache for '%s'. Structural/simulation data will be empty.", scenario)

    return nx_graph, structural_dict, simulation_dict, rmav_dict


def _remap_node_ids(d: Dict, graph_nodes: set) -> Dict:
    """Re-map dictionary keys to match the exact format used by the graph.

    E.g. converts cache keys 'A0', 'A1' → graph keys 'A01', 'A02' when the
    graph uses zero-padded IDs. Uses prefix+number decomposition.
    Always attempts remap and accepts whichever form gives better match rate.
    """
    import re
    if not d or not graph_nodes:
        return d

    # Build lookup: (prefix_upper, int_num) → graph_node_id
    graph_index: Dict[Tuple[str, int], str] = {}
    for nid in graph_nodes:
        m = re.match(r'^([A-Za-z]+)(\d+)$', nid)
        if m:
            graph_index[(m.group(1).upper(), int(m.group(2)))] = nid

    if not graph_index:
        return d  # Graph nodes don't follow prefix+num pattern

    remapped: Dict[str, Any] = {}
    for k, v in d.items():
        m = re.match(r'^([A-Za-z]+)(\d+)$', str(k))
        if m:
            key = (m.group(1).upper(), int(m.group(2)))
            new_id = graph_index.get(key, k)
            remapped[new_id] = v
        else:
            remapped[k] = v

    # Accept remap only if it improves coverage
    before = sum(1 for k in d if k in graph_nodes)
    after  = sum(1 for k in remapped if k in graph_nodes)
    logger.debug("_remap_node_ids: coverage %d→%d / %d graph nodes", before, after, len(graph_nodes))
    return remapped if after >= before else d







def _parse_failure_impact(raw: Dict) -> Dict:
    """Normalise failure_impact.json → {node_id: {composite: float}}.

    Supports:
      1. {records: {node_id: {impact_score, ...}}}  (dict-keyed format)
      2. {records: [{node_id, impact_score, ...}]}  (list format)
      3. Flat {node_id: {composite: ...}}            (already normalised)
    """
    if "records" in raw:
        records = raw["records"]
        result = {}
        # Dict keyed by node_id
        if isinstance(records, dict):
            for nid, rec in records.items():
                if not isinstance(rec, dict):
                    continue
                result[str(nid)] = {
                    "composite":     float(rec.get("impact_score", 0.0)),
                    "reliability":   float(rec.get("total_impacted_subscribers", 0.0)),
                    "cascade_depth": float(rec.get("cascade_depth", 0.0)),
                }
        # List of dicts
        elif isinstance(records, list):
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                nid = rec.get("node_id") or rec.get("id")
                if nid is None:
                    continue
                result[str(nid)] = {
                    "composite":     float(rec.get("impact_score", 0.0)),
                    "reliability":   float(rec.get("total_impacted_subscribers", 0.0)),
                    "cascade_depth": float(rec.get("cascade_depth", 0.0)),
                }
        return result
    # Filter metadata-only keys and return flat dicts
    skip = {"schema_version", "graph_id", "total_nodes_injected",
            "total_application_nodes", "total_broker_nodes",
            "total_subscribers", "seeds_used", "top_k_by_impact", "records"}
    return {k: v for k, v in raw.items() if k not in skip and isinstance(v, dict)}



def _parse_quality_scores(raw: Dict) -> Dict:
    """Normalise quality_scores.json → {node_id: {overall, reliability, ...}}.

    Supports:
      1. {layers: {<layer>: {rmav: {node_id: {...}}}}}   (nested format)
      2. Flat {node_id: {...}}
    """
    if "layers" in raw:
        for layer_val in raw["layers"].values():
            if not isinstance(layer_val, dict):
                continue
            rmav = layer_val.get("rmav")
            if isinstance(rmav, dict):
                return {str(k): v for k, v in rmav.items() if isinstance(v, dict)}
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


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
        # RMAV baseline: use structural graph metrics as proxy when cache unavailable
        if not simulation_dict:
            # Auto-generate lightweight simulation proxy from graph degree
            import networkx as nx
            deg = dict(nx.degree(nx_graph))
            max_deg = max(deg.values()) if deg else 1
            simulation_dict = {
                str(n): {"composite": d / max_deg}
                for n, d in deg.items()
            }
        n_nodes = nx_graph.number_of_nodes()
        effective_layers = 1 if n_nodes <= 200 else (2 if n_nodes <= 500 else num_layers)

        if not rmav_dict:
            # Use in-degree centrality as RMAV proxy
            import networkx as nx
            ic = nx.in_degree_centrality(nx_graph)
            rmav_dict = {
                str(n): {"composite": float(v)}
                for n, v in ic.items()
            }
        from scipy.stats import spearmanr
        start = time.time()
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
        start = time.time()
        conv = networkx_to_hetero_data(nx_graph, structural_dict, simulation_dict, rmav_dict)
        data = conv.hetero_data
        create_node_splits(data, train_ratio, val_ratio, seed=seed)
        effective_lr = 1e-3  # higher LR for faster convergence on small labelled sets
        effective_patience = max(patience, 60)
        model = build_baseline(variant, hidden_channels=hidden, num_heads=num_heads,
                               num_layers=effective_layers, dropout=dropout)
        ckpt_dir = f"output/gnn_checkpoints/{scenario}_{variant}_s{seed}"
        trainer = GNNTrainer(model=model, checkpoint_dir=ckpt_dir, lr=effective_lr,
                             num_epochs=num_epochs, patience=effective_patience)
        trainer.train(data)
        device = torch.device("cpu")
        metrics = evaluate(model, data, "test_mask", device)

    else:  # hetero_qos
        from saag.prediction.gnn_service import GNNService
        start = time.time()
        svc = GNNService(hidden_channels=hidden, num_heads=num_heads, num_layers=effective_layers,
                         dropout=dropout, predict_edges=False,
                         checkpoint_dir=f"output/gnn_checkpoints/{scenario}_hetero_qos_s{seed}")
        result = svc.train(
            graph=nx_graph, structural_metrics=structural_dict,
            simulation_results=simulation_dict, rmav_scores=rmav_dict,
            train_ratio=train_ratio, val_ratio=val_ratio,
            num_epochs=num_epochs, lr=1e-3, patience=patience,
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
                    if "error" in cell:
                        print(f" SKIP ({cell['error']})")
                        n_fail += 1
                    else:
                        rho = cell.get("spearman_rho", float("nan"))
                        rt  = cell.get("runtime_s", 0)
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
