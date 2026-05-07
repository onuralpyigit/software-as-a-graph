#!/usr/bin/env python3
"""
tools/qos_gini_sweep.py — Block D: QoS Gini Monotonicity Sweep
===============================================================

For each Gini level (0.0, 0.2, 0.4, 0.6, 0.8) × each model variant
× each seed, trains a GNN and records Spearman ρ.

Expected output: results/gini_sweep.json (source for Figure 3: ρ vs Gini).

The core hypothesis: hetero_qos ρ should be monotonically increasing
with Gini (more heterogeneous QoS → more signal for the heterogeneous
model), while homogeneous baselines should be flat or degraded.

Usage
-----
  # Full sweep over all scenarios × 5 Gini levels × 4 variants × 5 seeds
  python tools/qos_gini_sweep.py

  # Quick test: 1 scenario, 1 Gini level, 2 seeds
  python tools/qos_gini_sweep.py --scenario atm_system --gini 0.4 --seeds 42 123

  # Resume from partial
  python tools/qos_gini_sweep.py --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

ALL_SCENARIOS = [
    "atm_system",
    "av_system",
    "iot_smart_city_system",
]

GINI_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8]
ALL_VARIANTS = ["topology_rmav", "homo_unweighted", "homo_scalar", "hetero_qos"]
DEFAULT_SEEDS = [42, 123, 456, 789, 2024]

SCENARIOS_DIR    = Path("data/scenarios")
GINI_VARIANTS_DIR = Path("data/scenarios/gini_variants")
RESULTS_DIR      = Path("results")
LOSO_CACHE_DIR   = Path("output/loso_cache")


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_or_generate_gini_scenario(
    base_scenario: str, target_gini: float, seed: int = 42
) -> Path:
    """Return path to the Gini-variant scenario JSON, generating it if needed."""
    gini_tag = f"gini_{str(target_gini).replace('.', '')}"
    out_path = GINI_VARIANTS_DIR / f"{base_scenario}_{gini_tag}.json"

    if not out_path.exists():
        base_path = SCENARIOS_DIR / f"{base_scenario}.json"
        if not base_path.exists():
            raise FileNotFoundError(f"Base scenario not found: {base_path}")

        from tools.qos_gini_generator import _generate_one
        generated = _generate_one(base_path, target_gini, seed=seed, output_path=out_path)
        if generated is None:
            raise RuntimeError(f"Failed to generate Gini={target_gini} for {base_scenario}")

    return out_path


def _load_graph_and_metrics(
    scenario_json: Path, base_scenario: str
) -> Tuple[Any, Dict, Dict, Dict]:
    """Load NetworkX graph + structural/simulation/RMAV dicts."""
    from cli.loso_evaluate import _build_graph_from_json

    topology = json.loads(scenario_json.read_text())
    nx_graph = _build_graph_from_json(topology)

    structural_dict: Dict = {}
    simulation_dict: Dict = {}
    rmav_dict: Dict = {}

    cache_dir = LOSO_CACHE_DIR / base_scenario
    if cache_dir.exists():
        sm_path = cache_dir / "structural_metrics.json"
        fi_path = cache_dir / "failure_impact.json"
        qi_path = cache_dir / "quality_scores.json"
        if sm_path.exists(): structural_dict = json.loads(sm_path.read_text())
        if fi_path.exists(): simulation_dict = json.loads(fi_path.read_text())
        if qi_path.exists(): rmav_dict      = json.loads(qi_path.read_text())

    return nx_graph, structural_dict, simulation_dict, rmav_dict


# ── Per-cell training ─────────────────────────────────────────────────────────

def _train_gini_cell(
    scenario: str,
    gini: float,
    variant: str,
    seed: int,
    hidden: int = 64,
    num_heads: int = 4,
    num_layers: int = 3,
    dropout: float = 0.2,
    num_epochs: int = 150,
    patience: int = 25,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Dict[str, Any]:
    """Train one cell (scenario × gini × variant × seed) → metrics dict."""
    import numpy as np
    import torch
    from saag.prediction.data_preparation import networkx_to_hetero_data, create_node_splits
    from saag.prediction.trainer import GNNTrainer, evaluate

    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    try:
        scenario_json = _load_or_generate_gini_scenario(scenario, gini, seed=42)
        nx_graph, struct, sim, rmav = _load_graph_and_metrics(scenario_json, scenario)
    except Exception as e:
        return {"scenario": scenario, "gini": gini, "variant": variant, "seed": seed,
                "error": f"data_load: {e}"}

    if nx_graph.number_of_nodes() == 0:
        return {"scenario": scenario, "gini": gini, "variant": variant, "seed": seed,
                "error": "empty_graph"}

    try:
        conv = networkx_to_hetero_data(nx_graph, struct, sim, rmav)
        data = conv.hetero_data
        create_node_splits(data, train_ratio, val_ratio, seed=seed)
    except Exception as e:
        return {"scenario": scenario, "gini": gini, "variant": variant, "seed": seed,
                "error": f"hetero_data: {e}"}

    try:
        if variant == "topology_rmav":
            if not rmav or not sim:
                return {"scenario": scenario, "gini": gini, "variant": variant, "seed": seed,
                        "error": "no_rmav_data"}
            from scipy.stats import spearmanr
            keys = sorted(set(rmav) & set(sim))
            if len(keys) < 3:
                return {"scenario": scenario, "gini": gini, "variant": variant, "seed": seed,
                        "error": "insufficient_overlap"}
            pred_r = [rmav[k].get("composite", 0) for k in keys]
            true_r = [sim[k].get("composite", 0) for k in keys]
            rho, _ = spearmanr(pred_r, true_r)
            metrics_d = {"spearman_rho": float(rho), "f1_score": 0.0, "rmse": 0.0, "mae": 0.0}

        elif variant in ("homo_unweighted", "homo_scalar"):
            from saag.prediction.models.baselines import build_baseline
            model = build_baseline(variant, hidden_channels=hidden, num_heads=num_heads,
                                   num_layers=num_layers, dropout=dropout)
            ckpt_dir = f"output/gnn_checkpoints/gini_{scenario}_{variant}_s{seed}"
            trainer = GNNTrainer(model=model, checkpoint_dir=ckpt_dir, lr=3e-4,
                                 num_epochs=num_epochs, patience=patience)
            trainer.train(data)
            metrics = evaluate(model, data, "test_mask", torch.device("cpu"))
            metrics_d = metrics.to_dict()

        else:  # hetero_qos
            from saag.prediction.gnn_service import GNNService
            svc = GNNService(hidden_channels=hidden, num_heads=num_heads, num_layers=num_layers,
                             dropout=dropout, predict_edges=False,
                             checkpoint_dir=f"output/gnn_checkpoints/gini_{scenario}_hetero_s{seed}")
            result = svc.train(
                graph=nx_graph, structural_metrics=struct, simulation_results=sim,
                rmav_scores=rmav, train_ratio=train_ratio, val_ratio=val_ratio,
                num_epochs=num_epochs, lr=3e-4, patience=patience, seeds=[seed], mode="gnn",
            )
            if result.gnn_metrics:
                metrics_d = result.gnn_metrics.to_dict()
            else:
                return {"scenario": scenario, "gini": gini, "variant": variant, "seed": seed,
                        "error": "no_metrics"}

    except Exception as e:
        return {"scenario": scenario, "gini": gini, "variant": variant, "seed": seed,
                "error": str(e)}

    metrics_d.update({
        "scenario": scenario, "gini": gini, "variant": variant, "seed": seed,
        "runtime_s": round(time.time() - t0, 2),
    })
    return metrics_d


# ── Aggregation ───────────────────────────────────────────────────────────────

def _check_monotonicity(
    gini_levels: List[float], rhos: List[float]
) -> Dict[str, Any]:
    """Check if rho is monotonically non-decreasing with Gini.

    Returns dict with 'is_monotone', 'spearman_with_gini', 'violations'.
    """
    from scipy.stats import spearmanr
    if len(rhos) < 2:
        return {"is_monotone": None, "spearman_with_gini": None, "violations": []}

    rho_g, _ = spearmanr(gini_levels, rhos)
    violations = []
    for i in range(1, len(rhos)):
        if rhos[i] < rhos[i-1] - 0.02:  # tolerance 0.02
            violations.append({"from_gini": gini_levels[i-1], "to_gini": gini_levels[i],
                                "rho_drop": round(rhos[i-1] - rhos[i], 4)})
    return {
        "is_monotone": len(violations) == 0,
        "spearman_with_gini": round(float(rho_g), 4),
        "violations": violations,
    }


def _aggregate(cells: List[Dict]) -> Dict:
    """Aggregate cells into (scenario, gini, variant) → mean ρ + monotonicity."""
    import numpy as np

    by_sgv: Dict[Tuple, List[float]] = {}
    for c in cells:
        if "error" in c:
            continue
        key = (c["scenario"], c["gini"], c["variant"])
        by_sgv.setdefault(key, []).append(c["spearman_rho"])

    # Per (scenario, variant): check monotonicity across Gini levels
    results = {}
    scenarios = sorted({k[0] for k in by_sgv})
    variants = sorted({k[2] for k in by_sgv})

    for sc in scenarios:
        for var in variants:
            gini_mean_rho = []
            for g in sorted({k[1] for k in by_sgv if k[0] == sc and k[2] == var}):
                rs = by_sgv.get((sc, g, var), [])
                if rs:
                    gini_mean_rho.append((g, float(np.mean(rs)),
                                          float(np.std(rs)) if len(rs) > 1 else 0.0))
            if not gini_mean_rho:
                continue

            gini_vals = [x[0] for x in gini_mean_rho]
            rho_vals  = [x[1] for x in gini_mean_rho]
            mono = _check_monotonicity(gini_vals, rho_vals)

            results[f"{sc}|{var}"] = {
                "scenario": sc, "variant": var,
                "gini_rho": [{"gini": g, "mean_rho": r, "std_rho": s}
                             for g, r, s in gini_mean_rho],
                "monotonicity": mono,
            }

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Block D: QoS Gini Monotonicity Sweep.")
    p.add_argument("--scenarios", nargs="+", default=None,
                   help="Scenarios to sweep (default: all 3 reference scenarios)")
    p.add_argument("--gini", type=float, nargs="+", default=None,
                   help="Gini levels (default: 0.0 0.2 0.4 0.6 0.8)")
    p.add_argument("--variants", nargs="+", default=None, choices=ALL_VARIANTS,
                   help="Model variants (default: all 4)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Seeds (default: 5 seeds)")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "gini_sweep.json")
    p.add_argument("--resume", action="store_true",
                   help="Skip already-completed cells")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    scenarios = args.scenarios or ALL_SCENARIOS
    gini_levels = sorted(args.gini or GINI_LEVELS)
    variants = args.variants or ALL_VARIANTS
    seeds = args.seeds or DEFAULT_SEEDS

    total = len(scenarios) * len(gini_levels) * len(variants) * len(seeds)
    print(f"\n  QoS Gini Monotonicity Sweep (Block D)")
    print(f"  Scenarios : {scenarios}")
    print(f"  Gini lvls : {gini_levels}")
    print(f"  Variants  : {variants}")
    print(f"  Seeds     : {seeds}")
    print(f"  Total runs: {total}")
    print()

    if args.dry_run:
        for sc in scenarios:
            for g in gini_levels:
                for v in variants:
                    for s in seeds:
                        print(f"  [DRY-RUN] {sc} × gini={g} × {v} × seed={s}")
        return

    # Load existing for --resume
    existing: List[Dict] = []
    done_keys = set()
    if args.resume and args.output.exists():
        saved = json.loads(args.output.read_text())
        existing = saved.get("cells", [])
        done_keys = {(c["scenario"], c["gini"], c["variant"], c["seed"])
                     for c in existing if "error" not in c}
        print(f"  Resuming: {len(done_keys)} cells already completed.")

    cells: List[Dict] = list(existing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    n_fail = 0

    for sc in scenarios:
        for g in gini_levels:
            for v in variants:
                for s in seeds:
                    key = (sc, g, v, s)
                    if key in done_keys:
                        continue

                    n_done += 1
                    label = f"[{n_done}/{total}] {sc} gini={g} {v} s={s}"
                    print(f"  {label} ...", end="", flush=True)

                    try:
                        cell = _train_gini_cell(
                            scenario=sc, gini=g, variant=v, seed=s,
                            hidden=args.hidden, num_heads=args.heads,
                            num_layers=args.layers, num_epochs=args.epochs,
                            patience=args.patience,
                        )
                    except Exception as exc:
                        cell = {"scenario": sc, "gini": g, "variant": v, "seed": s,
                                "error": str(exc)}
                        n_fail += 1
                        print(f" ERROR: {exc}")
                    else:
                        if "error" in cell:
                            print(f" SKIP ({cell['error']})")
                            n_fail += 1
                        else:
                            rho = cell.get("spearman_rho", "?")
                            rt  = cell.get("runtime_s", "?")
                            print(f" rho={rho:.4f}  ({rt}s)")

                    cells.append(cell)
                    args.output.write_text(json.dumps({"cells": cells}, indent=2))

    aggregate = _aggregate(cells)
    output = {"cells": cells, "aggregate": aggregate,
               "config": {"scenarios": scenarios, "gini_levels": gini_levels,
                           "variants": variants, "seeds": seeds}}
    args.output.write_text(json.dumps(output, indent=2))

    # Print monotonicity summary
    print(f"\n  ──────────────────────────────────────────────────")
    print(f"  Monotonicity Summary (hetero_qos)")
    for sc in scenarios:
        key = f"{sc}|hetero_qos"
        if key in aggregate:
            mono = aggregate[key]["monotonicity"]
            status = "✓ MONOTONE" if mono.get("is_monotone") else f"✗ {len(mono.get('violations', []))} violations"
            rho_g = mono.get("spearman_with_gini", "?")
            print(f"    {sc:<35} {status}  ρ(Gini)={rho_g}")

    print(f"\n  {n_done} runs, {n_fail} failures. Saved: {args.output}")
    print(f"  Run: python tools/plot_gini_monotonicity.py to generate Figure 3.")


if __name__ == "__main__":
    main()
