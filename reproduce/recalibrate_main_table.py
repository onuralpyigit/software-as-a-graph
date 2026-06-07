#!/usr/bin/env python3
"""
reproduce/recalibrate_main_table.py — Post-hoc F1 recalibration for MW26
=====================================================================

Walks an existing results/main_table.json and replaces F1/Precision/Recall
with rank-matched binarization, salvaging cells produced before the F1
calibration patch landed in _train_cell.

Strategy
--------
  • Topo-BL / Q-Topo-BL cells:
      Recompute deterministically from the scenario's structural metrics
      and graph.  Cheap, no checkpoint needed.

  • Homo / Hetero GAT cells:
      Load the model checkpoint from
        output/gnn_checkpoints/{scenario}_{variant}_s{seed}
      and re-run inference on the same test mask.  When the checkpoint is
      missing or the architecture has drifted, the cell is left untouched
      and flagged for re-training.

  • Cells with a recent `calibration: rank_matched` field:
      Skipped (already calibrated by the patched harness).

The script writes a new JSON rather than mutating the input, so the original
run remains available for audit.

Usage
-----
  # Audit: print what would change, no writes
  python reproduce/recalibrate_main_table.py \
      --input results/main_table.json --audit

  # Recalibrate everything possible
  python reproduce/recalibrate_main_table.py \
      --input  results/main_table.json \
      --output results/main_table_recalibrated.json

  # Only the topo variants (safest, always works)
  python reproduce/recalibrate_main_table.py \
      --input  results/main_table.json \
      --output results/main_table_recalibrated.json \
      --variants topo_baseline q_topo_baseline

  # Constrain to a subset of scenarios
  python reproduce/recalibrate_main_table.py \
      --input  results/main_table.json \
      --output results/main_table_recalibrated.json \
      --scenarios atm_system financial_trading_system
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the harness's data-prep + helpers so recalibration uses the exact
# same DEPENDS_ON graph, RMAV substitution policy, and topo scoring formula.
from reproduce.middleware26_main_table import (
    _load_scenario_data,
    _compute_topo_baseline_scores,
    _mask_qos_in_graph,
    _mask_qos_in_structural,
    _NATIVE_QOS_FLAG_AVAILABLE,
    ALL_VARIANTS,
)

logger = logging.getLogger("recalibrate_main_table")

CKPT_ROOT = Path("output/gnn_checkpoints")

# Heuristic: a cell is "suspicious" (broken F1) when ranking is strong but
# F1 is near zero.  This is what the fixed-0.5 threshold artifact looks like.
_BROKEN_F1_HEURISTIC = lambda c: (
    c.get("spearman_rho") is not None
    and c.get("spearman_rho", 0.0) > 0.30
    and c.get("f1_score") is not None
    and c.get("f1_score", 1.0) < 0.05
)


# ── Rank-matched F1 (mirrors the harness helper) ─────────────────────────────

def _rank_matched_f1(pred_scores, true_scores, gt_threshold: float = 0.5):
    import numpy as np
    from sklearn.metrics import f1_score, precision_score, recall_score

    p = np.asarray(pred_scores, dtype=float)
    t = np.asarray(true_scores, dtype=float)
    n = len(p)
    if n != len(t) or n == 0:
        return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0,
                "n_critical_in_truth": 0, "calibration": "rank_matched"}

    # Adaptive threshold: mirror evaluate_scores behaviour.
    if np.max(t) < 0.5 and np.max(t) > 1e-6:
        gt_threshold = float(np.percentile(t, 90))

    true_bin = t >= gt_threshold
    k = int(true_bin.sum())
    if k == 0 or k == n:
        return {"f1_score": float("nan"), "precision": float("nan"),
                "recall": float("nan"), "n_critical_in_truth": k,
                "calibration": "rank_matched_degenerate"}

    order = np.argsort(-p)
    pred_bin = np.zeros(n, dtype=bool)
    pred_bin[order[:k]] = True
    return {
        "f1_score":            float(f1_score(true_bin, pred_bin, zero_division=0)),
        "precision":           float(precision_score(true_bin, pred_bin, zero_division=0)),
        "recall":              float(recall_score(true_bin, pred_bin, zero_division=0)),
        "n_critical_in_truth": k,
        "calibration":         "rank_matched",
    }


# ── Topo recompute path ──────────────────────────────────────────────────────

def _recompute_topo(
    scenario: str, variant: str, seed: int,
    cached_data: Optional[Tuple] = None,
) -> Tuple[Optional[Dict], Tuple]:
    """Recompute F1 for a topo cell.  Returns (calib_dict, scenario_data_cache)."""
    if cached_data is None:
        cached_data = _load_scenario_data(scenario)

    nx_graph, structural_dict, simulation_dict, _, _ = cached_data
    use_qos = (variant == "topo_qos")

    struct_pred = _compute_topo_baseline_scores(
        nx_graph, structural_dict, use_qos=use_qos
    )
    if struct_pred is None:
        return None, cached_data

    keys = sorted(set(struct_pred) & set(simulation_dict))
    if len(keys) < 3:
        logger.debug("topo recompute %s|%s: n_keys=%d (too few)", scenario, variant, len(keys))
        return None, cached_data

    logger.debug("topo recompute %s|%s: n_keys=%d", scenario, variant, len(keys))
    pred_list = [struct_pred[k] for k in keys]
    true_list = [simulation_dict[k].get("composite", 0.0) for k in keys]
    return _rank_matched_f1(pred_list, true_list), cached_data


# ── GAT recompute path (best-effort) ─────────────────────────────────────────

def _try_load_homo_model(variant: str, ckpt_dir: Path,
                          hidden: int, num_heads: int, num_layers: int,
                          dropout: float):
    """Best-effort loader for a homogeneous baseline checkpoint."""
    import torch
    from saag.prediction.models.baselines import build_baseline

    model = build_baseline(
        variant, hidden_channels=hidden, num_heads=num_heads,
        num_layers=num_layers, dropout=dropout,
    )
    for fname in ("best_model.pt", "model.pt", "checkpoint.pt"):
        p = ckpt_dir / fname
        if not p.exists():
            continue
        state = torch.load(p, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        return model
    raise FileNotFoundError(f"No homo checkpoint in {ckpt_dir}")


def _extract_test_scores_from_heterodata(data, out):
    """Return (pred, true) numpy arrays for the test mask, unioned over types."""
    import numpy as np
    preds, trues = [], []
    try:
        for nt in data.node_types:
            mask = data[nt].get("test_mask", None)
            y    = data[nt].get("y", None)
            if mask is None or y is None:
                continue
            type_pred = out[nt] if isinstance(out, dict) else None
            if type_pred is None:
                continue
            preds.append(type_pred[mask, 0].cpu().numpy())
            trues.append(y[mask, 0].cpu().numpy())
    except Exception:
        pass
    if preds:
        return np.concatenate(preds), np.concatenate(trues)
    # Single-tensor fallback
    try:
        import torch
        mask = getattr(data, "test_mask", None)
        y    = getattr(data, "y", None)
        if mask is None or y is None or out is None:
            return None, None
        return out[mask, 0].cpu().numpy(), y[mask, 0].cpu().numpy()
    except Exception:
        return None, None


def _try_recompute_homo(
    scenario: str, variant: str, seed: int,
    hidden: int, num_heads: int, num_layers: int, dropout: float,
    train_ratio: float, val_ratio: float,
    cached_data: Optional[Tuple] = None,
) -> Tuple[Optional[Dict], Tuple]:
    import torch
    from saag.prediction.data_preparation import (
        networkx_to_hetero_data, create_node_splits,
    )

    if cached_data is None:
        cached_data = _load_scenario_data(scenario)
    nx_graph, structural_dict, simulation_dict, rmav_dict, _ = cached_data

    ckpt_dir = CKPT_ROOT / f"{scenario}_{variant}_s{seed}"
    if not ckpt_dir.exists():
        logger.debug("[%s|%s|s%d] checkpoint dir missing: %s", scenario, variant, seed, ckpt_dir)
        return None, cached_data

    # Mirror the ablation gate in _train_cell: GL must use the masked graph
    # so that inference runs on the same input distribution as training.
    use_qos = (variant == "gl_qos")
    if use_qos:
        train_graph = nx_graph
        train_sm    = structural_dict
    else:
        train_graph = _mask_qos_in_graph(nx_graph)
        train_sm    = _mask_qos_in_structural(structural_dict)

    conv = networkx_to_hetero_data(
        train_graph, train_sm, simulation_dict, rmav_dict
    )
    data = conv.hetero_data
    create_node_splits(data, train_ratio, val_ratio, seed=seed)

    try:
        model = _try_load_homo_model(variant, ckpt_dir, hidden, num_heads, num_layers, dropout)
    except Exception as e:
        logger.warning("[%s|%s|s%d] homo checkpoint load failed: %s", scenario, variant, seed, e)
        return None, cached_data

    model.eval()
    with torch.no_grad():
        try:
            out = model(data.x, data.edge_index)
        except (TypeError, AttributeError):
            try:
                out = model(data.x_dict, data.edge_index_dict)
                out = torch.cat([v for v in out.values()], dim=0)
            except Exception as e:
                logger.warning("[%s|%s|s%d] homo forward failed: %s", scenario, variant, seed, e)
                return None, cached_data

    pred_t, true_t = _extract_test_scores_from_heterodata(data, out)
    if pred_t is None:
        return None, cached_data

    return _rank_matched_f1(pred_t.tolist(), true_t.tolist()), cached_data


def _coerce_gnnservice_output(result, simulation_dict):
    """Map GNNService.predict() output onto (pred_list, true_list)."""
    if result is None:
        return None, None
    pred_map = None
    if isinstance(result, dict):
        pred_map = result
    elif hasattr(result, "predictions"):
        pred_map = result.predictions
    if pred_map is None:
        return None, None

    keys = sorted(set(pred_map) & set(simulation_dict))
    if len(keys) < 3:
        return None, None
    pred_list, true_list = [], []
    for k in keys:
        v = pred_map[k]
        pred_list.append(float(v.get("composite", 0.0)) if isinstance(v, dict) else float(v))
        true_list.append(float(simulation_dict[k].get("composite", 0.0)))
    return pred_list, true_list


def _try_recompute_hetero(
    scenario: str, variant: str, seed: int,
    hidden: int, num_heads: int, num_layers: int, dropout: float,
    train_ratio: float, val_ratio: float,
    cached_data: Optional[Tuple] = None,
) -> Tuple[Optional[Dict], Tuple]:
    """Best-effort recompute for HGL and Q-HGL via GNNService."""
    if cached_data is None:
        cached_data = _load_scenario_data(scenario)
    nx_graph, structural_dict, simulation_dict, rmav_dict, _ = cached_data

    ckpt_dir = CKPT_ROOT / f"{scenario}_{variant}_s{seed}"
    if not ckpt_dir.exists():
        logger.debug("[%s|%s|s%d] checkpoint dir missing: %s", scenario, variant, seed, ckpt_dir)
        return None, cached_data

    use_qos = (variant == "hgl_qos")
    train_graph = nx_graph if use_qos else _mask_qos_in_graph(nx_graph)
    train_sm    = structural_dict if use_qos else _mask_qos_in_structural(structural_dict)

    try:
        from saag.prediction.gnn_service import GNNService
        svc = GNNService(
            hidden_channels=hidden, num_heads=num_heads, num_layers=num_layers,
            dropout=dropout, predict_edges=False,
            checkpoint_dir=str(ckpt_dir),
        )
        if hasattr(svc, "load"):
            svc.load()
        if not hasattr(svc, "predict"):
            logger.warning(
                "[%s|%s|s%d] GNNService.predict() not available; re-train this cell.",
                scenario, variant, seed)
            return None, cached_data

        kwargs: Dict[str, Any] = dict(
            graph=train_graph, structural_metrics=train_sm,
            simulation_results=simulation_dict, rmav_scores=rmav_dict,
        )
        if _NATIVE_QOS_FLAG_AVAILABLE:
            kwargs["qos_enabled"] = use_qos

        result = svc.predict(**kwargs)
        preds, trues = _coerce_gnnservice_output(result, simulation_dict)
        if preds is None or trues is None:
            return None, cached_data
        return _rank_matched_f1(preds, trues), cached_data

    except Exception as e:
        logger.warning("[%s|%s|s%d] hetero recompute failed: %s", scenario, variant, seed, e)
        return None, cached_data


# ── NaN-safe round ────────────────────────────────────────────────────────────

def _safe_round(x, ndigits=4):
    """Round x, returning None when x is NaN (so JSON serializes as null)."""
    import math
    if x is None:
        return None
    try:
        return None if math.isnan(x) else round(x, ndigits)
    except (TypeError, ValueError):
        return None


# ── Driver ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Post-hoc F1 recalibration.")
    p.add_argument("--input", type=Path, required=True,
                   help="Path to existing main_table.json")
    p.add_argument("--output", type=Path, default=None,
                   help="Where to write the recalibrated JSON (required unless --audit)")
    p.add_argument("--audit", action="store_true",
                   help="Print what would change without writing.")
    p.add_argument("--variants", nargs="+", default=None, choices=ALL_VARIANTS)
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument("--hidden",  type=int,   default=64)
    p.add_argument("--heads",   type=int,   default=4)
    p.add_argument("--layers",  type=int,   default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--val-ratio",   type=float, default=0.2)
    p.add_argument("--force", action="store_true",
                   help="Recalibrate even cells already marked rank_matched.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="  %(levelname)-7s %(message)s",
    )

    if not args.audit and args.output is None:
        sys.exit("Either --audit or --output is required.")

    raw = json.loads(args.input.read_text())
    cells: List[Dict] = raw.get("cells", [])

    def keep(c: Dict) -> bool:
        if "error" in c:
            return False
        if args.variants and c.get("variant") not in args.variants:
            return False
        if args.scenarios and c.get("scenario") not in args.scenarios:
            return False
        if not args.force and c.get("calibration") == "rank_matched":
            return False
        return True

    candidates = [c for c in cells if keep(c)]
    suspicious = [c for c in candidates if _BROKEN_F1_HEURISTIC(c)]

    print(f"\n  Total cells in input    : {len(cells)}")
    print(f"  Candidates for recal    : {len(candidates)}")
    print(f"  Suspicious (heuristic)  : {len(suspicious)}")

    if args.audit:
        print("\n  Audit mode — would attempt the following:\n")
        for c in candidates:
            tag = " ★" if _BROKEN_F1_HEURISTIC(c) else "  "
            print(f"   {tag} {c['scenario']:<28} {c['variant']:<18} "
                  f"s={c['seed']:<6}  ρ={c.get('spearman_rho', 0):.3f}  "
                  f"F1={c.get('f1_score') or 0:.3f}")
        print("\n  (★ = heuristic flag for broken F1)\n")
        return

    # Group by scenario to avoid reloading data per cell.
    by_scenario: Dict[str, List[Dict]] = {}
    for c in candidates:
        by_scenario.setdefault(c["scenario"], []).append(c)

    n_recal_topo = 0
    n_recal_gnn  = 0
    n_skipped    = 0
    failures: List[Tuple[str, str, int, str]] = []

    new_cells = deepcopy(cells)
    cell_index: Dict[Tuple[str, str, int], int] = {
        (c["scenario"], c["variant"], c["seed"]): i
        for i, c in enumerate(new_cells) if "error" not in c
    }

    for sc, sc_cells in by_scenario.items():
        try:
            scenario_data = _load_scenario_data(sc)
        except Exception as e:
            logger.warning("[%s] data load failed: %s — skipping %d cells",
                           sc, e, len(sc_cells))
            for c in sc_cells:
                failures.append((sc, c["variant"], c["seed"], f"data_load: {e}"))
                n_skipped += 1
            continue

        for c in sc_cells:
            variant = c["variant"]; seed = c["seed"]
            t0 = time.time()
            calib = None
            try:
                if variant in ("topo_baseline", "topo_qos"):
                    calib, scenario_data = _recompute_topo(
                        sc, variant, seed, cached_data=scenario_data
                    )
                    if calib is not None:
                        n_recal_topo += 1
                elif variant in ("gl", "gl_qos"):
                    calib, scenario_data = _try_recompute_homo(
                        sc, variant, seed,
                        args.hidden, args.heads, args.layers, args.dropout,
                        args.train_ratio, args.val_ratio,
                        cached_data=scenario_data,
                    )
                    if calib is not None:
                        n_recal_gnn += 1
                elif variant in ("hgl", "hgl_qos"):
                    calib, scenario_data = _try_recompute_hetero(
                        sc, variant, seed,
                        args.hidden, args.heads, args.layers, args.dropout,
                        args.train_ratio, args.val_ratio,
                        cached_data=scenario_data,
                    )
                    if calib is not None:
                        n_recal_gnn += 1
            except Exception as e:
                calib = None
                logger.warning("[%s|%s|s%d] exception during recalibration: %s",
                               sc, variant, seed, e)
                failures.append((sc, variant, seed, str(e)))

            dt = time.time() - t0

            if calib is None:
                n_skipped += 1
                if not any(f[:3] == (sc, variant, seed) for f in failures):
                    failures.append((sc, variant, seed, "no_predictions_recoverable"))
                idx = cell_index.get((sc, variant, seed))
                if idx is not None:
                    new_cells[idx]["needs_recalibration"] = True
                print(f"   ✗ {sc:<28} {variant:<18} s={seed:<6}  (skipped, {dt:.1f}s)")
                continue

            idx = cell_index[(sc, variant, seed)]
            old_f1 = c.get("f1_score") or 0.0
            new_cells[idx].update({
                "f1_score":            _safe_round(calib["f1_score"]),
                "precision":           _safe_round(calib["precision"]),
                "recall":              _safe_round(calib["recall"]),
                "calibration":         calib["calibration"],
                "n_critical_in_truth": calib["n_critical_in_truth"],
                "needs_recalibration": False,
            })
            new_f1_disp = new_cells[idx]["f1_score"]
            print(f"   ✓ {sc:<28} {variant:<18} s={seed:<6}  "
                  f"F1: {old_f1:.3f} → {new_f1_disp if new_f1_disp is not None else 'NaN'}"
                  f"  ({dt:.1f}s)")

    # ── Re-aggregate so downstream renderers see the new F1 numbers ──────────
    from reproduce.middleware26_main_table import _aggregate_cells
    aggregate = _aggregate_cells(new_cells)

    agg_serializable: Dict[str, Any] = {}
    for k, v in aggregate.items():
        if isinstance(k, tuple):
            agg_serializable[f"{k[0]}|{k[1]}"] = v
        else:
            agg_serializable[k] = v  # meta keys like _factorial_contrasts

    output_data = {
        "cells": new_cells,
        "aggregate": agg_serializable,
        "config": raw.get("config", {}),
        "recalibration": {
            "tool": "reproduce/recalibrate_main_table.py",
            "input": str(args.input),
            "n_recal_topo": n_recal_topo,
            "n_recal_gnn":  n_recal_gnn,
            "n_skipped":    n_skipped,
            "failures":     [
                {"scenario": s, "variant": v, "seed": sd, "reason": r}
                for s, v, sd, r in failures
            ],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_data, indent=2))

    print()
    print(f"  ─── Recalibration summary ──────────────────────────────────")
    print(f"  Topo cells recalibrated   : {n_recal_topo}")
    print(f"  GAT cells recalibrated    : {n_recal_gnn}")
    print(f"  Skipped (needs re-train)  : {n_skipped}")
    print(f"  Output                    : {args.output}")
    print()

    if n_skipped > 0:
        skipped_variants  = sorted({v for _, v, _, _ in failures})
        skipped_scenarios = sorted({s for s, _, _, _ in failures})
        print(f"  To re-train the skipped cells:")
        print(f"    bash scripts/run_main_table.sh --resume \\")
        print(f"        --output {args.output} \\")
        print(f"        --variants  \"{ ' '.join(skipped_variants) }\" \\")
        print(f"        --scenarios \"{ ' '.join(skipped_scenarios)}\"")
        print()


if __name__ == "__main__":
    main()
