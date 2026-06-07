#!/usr/bin/env python3
"""
cli/loso_evaluate.py — Leave-One-Scenario-Out Inductive Evaluation
==================================================================

Closes G4 (transductive leakage) for the GNN Predict stage by establishing
a strict inductive evaluation protocol: for every scenario k in the suite,
train the HeteroGAT on the N-1 remaining scenarios and evaluate on k. The
held-out scenario is never observed during training — its node features
never participate in any forward pass, its labels never enter any loss.

This is the evidence required for inductive generalisation claims in:
    - Middleware 2026 (cross-system QoS-ablation)
    - SoSE 2026 (systems-of-systems generality)
    - Thesis Chapter 6 (validity threats)

────────────────────────────────────────────────────────────────────────────
Protocol
────────────────────────────────────────────────────────────────────────────
For each scenario k ∈ {1..N}:
    train_set := scenarios \\ {k}             (N-1 scenarios)
    primary  := argmax_{j ∈ train_set} |V_j|  (most signal for early-stopping)
    inductive := train_set \\ {primary}       (passed via inductive_graphs)

For each seed s ∈ {42, 123, 456, 789, 2024}:
    GNNService.train(primary, inductive_graphs=inductive, seeds=[s])
    GNNService.predict(holdout_graph)         # holdout never seen
    Compute ρ, F1@K, NDCG@10, RMSE, MAE — overall and per-node-type

Reports: per-fold mean ± std across seeds, then cross-fold mean ± std.

────────────────────────────────────────────────────────────────────────────
Cache layout (one directory per scenario)
────────────────────────────────────────────────────────────────────────────
    output/loso_cache/<scenario_id>/
        topology.json              (input — same JSON as cli/import_graph.py)
        structural_metrics.json    (output of cli/analyze_graph.py)
        quality_scores.json        (output of cli/predict_graph.py --mode rmav)
        failure_impact.json        (output of cli/simulate_graph.py fault-inject)

To populate the cache from existing pipeline outputs:

    for cfg in data/scenario_*.yaml; do
        sid=$(basename "$cfg" .yaml)
        out="output/loso_cache/$sid"
        mkdir -p "$out"
        PYTHONPATH=. python cli/generate_graph.py --config "$cfg" --output "$out/topology.json"
        PYTHONPATH=. python cli/import_graph.py    --input "$out/topology.json" --clear
        PYTHONPATH=. python cli/analyze_graph.py   --layer app --output "$out/structural_metrics.json"
        PYTHONPATH=. python cli/predict_graph.py   --layer app --mode rmav --output "$out/quality_scores.json"
        PYTHONPATH=. python cli/simulate_graph.py  fault-inject --input "$out/topology.json" \
                                                   --output "$out/" --export-json --seeds 42
        # rename impact_scores.json -> failure_impact.json if needed
    done

────────────────────────────────────────────────────────────────────────────
Usage
────────────────────────────────────────────────────────────────────────────
    PYTHONPATH=. python cli/loso_evaluate.py \
        --cache-dir output/loso_cache \
        --output-dir output/loso \
        --layer app

    # Skip xlarge scenarios for fast iteration
    PYTHONPATH=. python cli/loso_evaluate.py --skip scenario_07,scenario_09

    # Use ensemble mode for inductive eval (compares Q_GNN vs Q_ens generalisation)
    PYTHONPATH=. python cli/loso_evaluate.py --mode ensemble
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from scipy.stats import spearmanr
from torch_geometric.data import HeteroData

# ── SaG SDK imports ──────────────────────────────────────────────────────────
from saag.prediction.gnn_service import GNNService
from saag.prediction.data_preparation import (
    networkx_to_hetero_data,
    extract_simulation_dict,
    extract_structural_metrics_dict,
    extract_rmav_scores_dict,
)

logger = logging.getLogger("loso_evaluate")


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioBundle:
    """All artefacts needed to use a scenario as a training or evaluation graph."""
    scenario_id: str
    graph: nx.DiGraph
    structural: Dict[str, Any]
    rmav: Dict[str, Any]
    simulation: Dict[str, Any]
    hetero_data: HeteroData
    n_nodes: int
    n_edges: int
    n_labelled: int


@dataclass
class FoldResult:
    """Result of a single LOSO fold (one held-out scenario, multi-seed)."""
    holdout_id: str
    train_ids: List[str]
    primary_id: str
    seed_metrics: List[Dict[str, Any]] = field(default_factory=list)
    mean_metrics: Dict[str, float] = field(default_factory=dict)
    std_metrics: Dict[str, float] = field(default_factory=dict)
    per_type_rho: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Mean predictions across seeds: {node_id: {score_type: mean_val}}
    node_predictions: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class LOSOReport:
    """Aggregate LOSO results across all folds."""
    fold_results: List[FoldResult] = field(default_factory=list)
    overall_mean_rho: float = 0.0
    overall_std_rho: float = 0.0
    overall_mean_f1: float = 0.0
    overall_mean_ndcg: float = 0.0
    n_folds: int = 0
    n_seeds_per_fold: int = 0
    per_type_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # All scenarios: {scenario_id: {node_id: {score_type: mean_val}}}
    scenario_predictions: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Cache loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _build_graph_from_json(topology: Dict[str, Any]) -> nx.DiGraph:
    """Lightweight builder, peer of cli/simulate_graph.py:_load_graph fallback path."""
    g = nx.DiGraph()

    type_buckets = [
        ("applications", "Application"),
        ("brokers", "Broker"),
        ("topics", "Topic"),
        ("nodes", "Node"),
        ("libraries", "Library"),
    ]
    for key, type_label in type_buckets:
        for entity in topology.get(key, []):
            g.add_node(
                entity["id"],
                type=type_label,
                name=entity.get("name", entity["id"]),
                **{k: v for k, v in entity.items() if k not in ("id", "name")},
            )

    rels = topology.get("relationships", {}) or {}

    edge_buckets = [
        (rels.get("publishes_to", []) + topology.get("publishes", []), "PUBLISHES_TO"),
        (rels.get("subscribes_to", []) + topology.get("subscribes", []), "SUBSCRIBES_TO"),
        (rels.get("routes", []) + topology.get("routes", []), "ROUTES"),
        (rels.get("runs_on", []) + topology.get("runs_on", []), "RUNS_ON"),
        (rels.get("connects_to", []) + topology.get("connects_to", []), "CONNECTS_TO"),
        (rels.get("uses", []) + topology.get("uses", []), "USES"),
        (rels.get("depends_on", []), "DEPENDS_ON"),
    ]
    for items, type_label in edge_buckets:
        for r in items:
            src = (
                r.get("source") or r.get("from")
                or r.get("application_id") or r.get("topic_id")
                or r.get("node_id") or r.get("broker_id")
            )
            dst = (
                r.get("target") or r.get("to")
                or r.get("topic_id") or r.get("broker_id")
                or r.get("application_id") or r.get("node_id")
            )
            if src and dst and src != dst:
                g.add_edge(
                    src, dst,
                    type=r.get("type", type_label),
                    weight=float(r.get("weight", 1.0)),
                    qos_profile=r.get("qos_profile", {}),
                )
    return g


def load_scenario_bundle(scenario_dir: Path) -> Optional[ScenarioBundle]:
    """Load one scenario's full artefact bundle from cache. Returns None on incomplete cache."""
    scenario_id = scenario_dir.name
    topology = _load_json(scenario_dir / "topology.json")
    structural_raw = _load_json(scenario_dir / "structural_metrics.json")
    rmav_raw = _load_json(scenario_dir / "quality_scores.json")
    sim_raw = _load_json(scenario_dir / "failure_impact.json")

    missing = [
        name for name, val in [
            ("topology.json", topology),
            ("structural_metrics.json", structural_raw),
            ("failure_impact.json", sim_raw),
        ] if val is None
    ]
    if missing:
        logger.warning("  [%s] missing %s — skipping.", scenario_id, ", ".join(missing))
        return None

    graph = _build_graph_from_json(topology)
    structural = extract_structural_metrics_dict(structural_raw)

    try:
        from reproduce.middleware26_main_table import _parse_failure_impact, _parse_quality_scores, _remap_node_ids
        sim_parsed = _parse_failure_impact(sim_raw)
        rmav_parsed = _parse_quality_scores(rmav_raw) if rmav_raw else {}
        graph_nodes = set(str(n) for n in graph.nodes())
        simulation = _remap_node_ids(sim_parsed, graph_nodes)
        rmav = _remap_node_ids(rmav_parsed, graph_nodes)
    except ImportError:
        rmav = extract_rmav_scores_dict(rmav_raw) if rmav_raw else {}
        simulation = extract_simulation_dict(sim_raw)

    conv = networkx_to_hetero_data(graph, structural, simulation, rmav)

    bundle = ScenarioBundle(
        scenario_id=scenario_id,
        graph=graph,
        structural=structural,
        rmav=rmav,
        simulation=simulation,
        hetero_data=conv.hetero_data,
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        n_labelled=conv.num_labelled_nodes,
    )
    logger.info(
        "  [%s] %d nodes, %d edges, %d labelled%s",
        scenario_id, bundle.n_nodes, bundle.n_edges, bundle.n_labelled,
        "" if rmav else "  (rmav missing — ensemble mode unavailable)",
    )
    return bundle


def discover_scenarios(cache_dir: Path, skip: List[str]) -> List[ScenarioBundle]:
    """Walk cache_dir and load all valid scenario bundles."""
    bundles: List[ScenarioBundle] = []
    for sub in sorted(p for p in cache_dir.iterdir() if p.is_dir()):
        if any(s in sub.name for s in skip):
            logger.info("  Skipping %s (matches --skip filter)", sub.name)
            continue
        b = load_scenario_bundle(sub)
        if b is not None and b.n_labelled >= 3:
            bundles.append(b)

    if len(bundles) < 2:
        raise ValueError(
            f"Need ≥ 2 scenarios for LOSO; found {len(bundles)} usable in {cache_dir}."
        )
    return bundles


# ──────────────────────────────────────────────────────────────────────────────
# Inductive metric computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_inductive_metrics(
    pred_scores: Dict[str, float],
    true_impact: Dict[str, float],
    graph: nx.DiGraph,
    top_k_frac: float = 0.20,
) -> Dict[str, Any]:
    """
    Compute Spearman ρ, F1@K, NDCG@10, RMSE, MAE between predicted and ground-truth
    composite I*(v). Also returns per-node-type stratified ρ.
    """
    common = sorted(set(pred_scores.keys()) & set(true_impact.keys()))
    if len(common) < 3:
        return {"spearman_rho": float("nan"), "n": len(common), "per_type_rho": {}}

    y_pred = np.array([pred_scores[v] for v in common], dtype=np.float64)
    y_true = np.array([true_impact[v] for v in common], dtype=np.float64)

    rho, p_val = spearmanr(y_pred, y_true)
    rho = float(rho) if not np.isnan(rho) else 0.0

    # F1 @ top-K
    k = max(1, int(round(len(common) * top_k_frac)))
    pred_top = set(np.argsort(-y_pred)[:k].tolist())
    true_top = set(np.argsort(-y_true)[:k].tolist())
    tp = len(pred_top & true_top)
    precision = tp / k if k > 0 else 0.0
    recall = tp / len(true_top) if true_top else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # NDCG @ 10
    k_ndcg = min(10, len(common))
    ideal_order = np.argsort(-y_true)[:k_ndcg]
    pred_order = np.argsort(-y_pred)[:k_ndcg]
    dcg = float(np.sum(y_true[pred_order] / np.log2(np.arange(2, k_ndcg + 2))))
    idcg = float(np.sum(y_true[ideal_order] / np.log2(np.arange(2, k_ndcg + 2))))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))

    # Per-node-type stratified ρ
    by_type: Dict[str, List[Tuple[float, float]]] = {}
    for v in common:
        nt = graph.nodes[v].get("type", "Unknown")
        by_type.setdefault(nt, []).append((pred_scores[v], true_impact[v]))

    per_type_rho: Dict[str, Dict[str, float]] = {}
    for nt, pairs in by_type.items():
        if len(pairs) < 3:
            continue
        ps, ts = zip(*pairs)
        type_rho, _ = spearmanr(ps, ts)
        per_type_rho[nt] = {
            "rho": float(type_rho) if not np.isnan(type_rho) else 0.0,
            "n": len(pairs),
        }

    return {
        "spearman_rho": rho,
        "spearman_p": float(p_val) if not np.isnan(p_val) else 1.0,
        "f1_at_k": f1,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "ndcg_10": ndcg,
        "rmse": rmse,
        "mae": mae,
        "n": len(common),
        "k": k,
        "per_type_rho": per_type_rho,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Single-fold execution
# ──────────────────────────────────────────────────────────────────────────────

def run_one_fold(
    bundles: List[ScenarioBundle],
    holdout_idx: int,
    seeds: List[int],
    layer: str,
    epochs: int,
    lr: float,
    hidden: int,
    heads: int,
    layers: int,
    dropout: float,
    workdir: Path,
    mode: str,
    global_metadata: Optional[Tuple] = None,
    variant: str = "hetero_qos",
) -> FoldResult:
    """
    One LOSO fold: train on N-1 scenarios with multi-seed, predict on held-out.

    Defensive invariants:
      - holdout never appears in train_ids
      - holdout's structural/rmav are passed at predict() time (needed for features)
      - holdout's simulation is passed only for evaluation, never for training
    """
    holdout = bundles[holdout_idx]
    train_set = [b for i, b in enumerate(bundles) if i != holdout_idx]
    train_ids = [b.scenario_id for b in train_set]

    assert holdout.scenario_id not in train_ids, (
        f"G4 leakage violation: holdout {holdout.scenario_id} found in train ids"
    )

    # Pick the largest non-holdout as the primary (longest val masks → stable early stop)
    primary = max(train_set, key=lambda b: b.n_nodes)
    inductives = [b for b in train_set if b.scenario_id != primary.scenario_id]

    logger.info(
        "Fold[holdout=%s]  primary=%s (|V|=%d)  inductive=%d scenarios",
        holdout.scenario_id, primary.scenario_id, primary.n_nodes, len(inductives),
    )

    fold_dir = workdir / f"fold_{holdout.scenario_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    seed_metrics: List[Dict[str, Any]] = []

    for seed in seeds:
        logger.info("  ── seed %d ──", seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        ckpt_dir = fold_dir / f"seed_{seed}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        try:
            if variant in ("gl", "gl_qos"):
                # Baseline variants use GNNTrainer directly
                from saag.prediction.models.baselines import build_baseline
                from saag.prediction.data_preparation import create_node_splits
                from saag.prediction.trainer import GNNTrainer, evaluate

                use_qos = (variant == "gl_qos")
                if use_qos:
                    train_graph = primary.graph
                    train_sm    = primary.structural
                    holdout_graph = holdout.graph
                    holdout_sm    = holdout.structural
                else:
                    from reproduce.middleware26_main_table import _mask_qos_in_graph, _mask_qos_in_structural
                    train_graph = _mask_qos_in_graph(primary.graph)
                    train_sm    = _mask_qos_in_structural(primary.structural)
                    holdout_graph = _mask_qos_in_graph(holdout.graph)
                    holdout_sm    = _mask_qos_in_structural(holdout.structural)

                conv = networkx_to_hetero_data(
                    train_graph, train_sm, primary.simulation, primary.rmav, qos_enabled=use_qos
                )
                data = conv.hetero_data
                create_node_splits(data, seed=seed)
                baseline_name = "homo_unweighted" if variant == "gl" else "homo_scalar"
                model = build_baseline(baseline_name, hidden_channels=hidden, num_heads=heads,
                                       num_layers=layers, dropout=dropout)
                best_path = ckpt_dir / "best_model.pt"
                if best_path.exists():
                    logger.info("  Found baseline checkpoint %s. Skipping training.", best_path)
                    model.load_state_dict(torch.load(best_path, map_location="cpu"))
                else:
                    trainer = GNNTrainer(model=model, checkpoint_dir=str(ckpt_dir),
                                         lr=lr, num_epochs=epochs, patience=min(60, epochs))
                    trainer.train(data)

                # Evaluate on holdout
                conv_h = networkx_to_hetero_data(
                    holdout_graph, holdout_sm, holdout.simulation, holdout.rmav, qos_enabled=use_qos
                )
                data_h = conv_h.hetero_data
                create_node_splits(data_h, seed=seed)
                device = torch.device("cpu")
                metrics = evaluate(model, data_h, "test_mask", device)

                # Build pred_scores from model output for inductive metrics
                model.eval()
                with torch.no_grad():
                    x_h = {nt: data_h[nt].x for nt in data_h.node_types if hasattr(data_h[nt], "x")}
                    ei_h = {r: data_h[r].edge_index for r in data_h.edge_types}
                    ea_h = {r: data_h[r].edge_attr for r in data_h.edge_types if hasattr(data_h[r], "edge_attr")}
                    out_h = model(x_h, ei_h, ea_h)

                pred_scores: Dict[str, float] = {}
                full_node_scores: Dict[str, Dict[str, float]] = {}
                # node_id_map is Dict[str, List[str]]: node_type → ordered list of node IDs
                for nt, preds in out_h.items():
                    node_list = conv_h.node_id_map.get(nt, [])
                    for local_idx, nid in enumerate(node_list):
                        if local_idx < preds.shape[0]:
                            pred_scores[nid] = float(preds[local_idx, 0])
                            full_node_scores[nid] = {
                                "overall":         float(preds[local_idx, 0]),
                                "reliability":     float(preds[local_idx, 1]),
                                "maintainability": float(preds[local_idx, 2]),
                                "availability":    float(preds[local_idx, 3]),
                                "vulnerability":   float(preds[local_idx, 4]),
                            }

            else:
                # hgl_qos (default) or hgl or topology_rmav → GNNService path
                effective_mode = "rmav" if variant == "topology_rmav" else mode
                effective_layers = 1 if primary.n_nodes <= 200 else (2 if primary.n_nodes <= 500 else layers)
                use_qos = (variant == "hgl_qos")

                if use_qos:
                    train_graph = primary.graph
                    train_sm    = primary.structural
                    holdout_graph = holdout.graph
                    holdout_sm    = holdout.structural
                else:
                    from reproduce.middleware26_main_table import _mask_qos_in_graph, _mask_qos_in_structural
                    train_graph = _mask_qos_in_graph(primary.graph)
                    train_sm    = _mask_qos_in_structural(primary.structural)
                    holdout_graph = _mask_qos_in_graph(holdout.graph)
                    holdout_sm    = _mask_qos_in_structural(holdout.structural)
                
                best_path = ckpt_dir / "best_model.pt"
                if best_path.exists():
                    logger.info("  Found GNN checkpoint %s. Skipping training.", best_path)
                    service = GNNService.from_checkpoint(
                        str(ckpt_dir),
                        graph=train_graph,
                        layer=layer,
                    )
                else:
                    service = GNNService(
                        checkpoint_dir=str(ckpt_dir),
                        hidden_channels=hidden,
                        num_heads=heads,
                        num_layers=effective_layers,
                        dropout=dropout,
                        predict_edges=False,
                    )
                    service.train(
                        graph=train_graph,
                        structural_metrics=train_sm,
                        simulation_results=primary.simulation,
                        rmav_scores=primary.rmav,
                        inductive_graphs=[
                            networkx_to_hetero_data(
                                b.graph if use_qos else _mask_qos_in_graph(b.graph),
                                b.structural if use_qos else _mask_qos_in_structural(b.structural),
                                b.simulation,
                                b.rmav,
                                qos_enabled=use_qos
                            ).hetero_data
                            for b in inductives
                        ],
                        seeds=[seed],
                        num_epochs=1 if variant == "topology_rmav" else epochs,
                        lr=lr,
                        patience=min(60, epochs),
                        layer=layer,
                        qos_enabled=use_qos,
                    )
                result = service.predict(
                    graph=holdout_graph,
                    structural_metrics=holdout_sm,
                    rmav_scores=holdout.rmav,
                    simulation_results=holdout.simulation,
                    mode=effective_mode,
                    qos_enabled=use_qos,
                )
                pred_scores = {nid: float(ns.composite_score)
                               for nid, ns in result.node_scores.items()}
                full_node_scores = {
                    nid: {
                        "overall":         float(ns.composite_score),
                        "reliability":     float(ns.reliability_score),
                        "maintainability": float(ns.maintainability_score),
                        "availability":    float(ns.availability_score),
                        "vulnerability":   float(ns.vulnerability_score),
                    }
                    for nid, ns in result.node_scores.items()
                }


        except Exception as e:
            logger.error("  Fold seed %d failed: %s", seed, e, exc_info=True)
            continue

        true_impact = {nid: float(d.get("composite", 0.0)) for nid, d in holdout.simulation.items()}

        m = compute_inductive_metrics(pred_scores, true_impact, holdout.graph)
        m["seed"] = seed
        m["prediction_mode"] = mode
        m["variant"] = variant
        m["_full_scores"] = full_node_scores  # temporary storage for aggregation
        seed_metrics.append(m)


        logger.info(
            "    ρ=%.4f  F1=%.4f  NDCG=%.4f  RMSE=%.4f  (n=%d, mode=%s)",
            m["spearman_rho"], m["f1_at_k"], m["ndcg_10"], m["rmse"], m["n"],
            m["prediction_mode"],
        )

    # Aggregate node scores across seeds
    all_nodes = set()
    for m in seed_metrics:
        all_nodes.update(m["_full_scores"].keys())
    
    score_keys = ["overall", "reliability", "maintainability", "availability", "vulnerability"]
    node_agg: Dict[str, Dict[str, float]] = {}
    
    for nid in all_nodes:
        node_agg[nid] = {}
        for k in score_keys:
            vals = [m["_full_scores"][nid][k] for m in seed_metrics if nid in m["_full_scores"]]
            node_agg[nid][k] = float(np.mean(vals)) if vals else 0.0
    
    # Cleanup temporary storage
    for m in seed_metrics:
        if "_full_scores" in m:
            del m["_full_scores"]

    # Aggregate across seeds
    rho_vals = [m["spearman_rho"] for m in seed_metrics]
    f1_vals = [m["f1_at_k"] for m in seed_metrics]
    ndcg_vals = [m["ndcg_10"] for m in seed_metrics]
    rmse_vals = [m["rmse"] for m in seed_metrics]

    per_type_agg: Dict[str, List[float]] = {}
    for m in seed_metrics:
        for nt, info in m.get("per_type_rho", {}).items():
            per_type_agg.setdefault(nt, []).append(info["rho"])
    per_type_summary = {
        nt: {
            "mean": float(np.mean(vs)),
            "std": float(np.std(vs)),
            "n_seeds": len(vs),
        }
        for nt, vs in per_type_agg.items()
    }

    return FoldResult(
        holdout_id=holdout.scenario_id,
        train_ids=train_ids,
        primary_id=primary.scenario_id,
        seed_metrics=seed_metrics,
        mean_metrics={
            "spearman_rho": float(np.mean(rho_vals)),
            "f1_at_k": float(np.mean(f1_vals)),
            "ndcg_10": float(np.mean(ndcg_vals)),
            "rmse": float(np.mean(rmse_vals)),
        },
        std_metrics={
            "spearman_rho": float(np.std(rho_vals)),
            "f1_at_k": float(np.std(f1_vals)),
            "ndcg_10": float(np.std(ndcg_vals)),
            "rmse": float(np.std(rmse_vals)),
        },
        per_type_rho=per_type_summary,
        node_predictions=node_agg,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Full LOSO orchestration
# ──────────────────────────────────────────────────────────────────────────────

def run_loso(
    bundles: List[ScenarioBundle],
    seeds: List[int],
    output_dir: Path,
    layer: str,
    epochs: int,
    lr: float,
    hidden: int,
    heads: int,
    layers: int,
    dropout: float,
    mode: str,
    variant: str = "hetero_qos",
) -> LOSOReport:
    """Run leave-one-scenario-out across all loaded bundles."""
    output_dir.mkdir(parents=True, exist_ok=True)
    workdir = output_dir / "workspace"
    workdir.mkdir(exist_ok=True)

    # Compute global metadata across all bundles to ensure GNN has matrices for all possible types
    all_node_types = set()
    all_edge_types = set()
    for b in bundles:
        m = b.hetero_data.metadata()
        all_node_types.update(m[0])
        all_edge_types.update(m[1])
    global_metadata = (list(all_node_types), list(all_edge_types))
    logger.info("Global metadata: %d node types, %d edge types", len(all_node_types), len(all_edge_types))

    fold_results: List[FoldResult] = []
    for k in range(len(bundles)):
        logger.info("════════════════════════════════════════════════════════════")
        logger.info("LOSO fold %d / %d   holdout = %s",
                    k + 1, len(bundles), bundles[k].scenario_id)
        logger.info("════════════════════════════════════════════════════════════")

        try:
            fold = run_one_fold(
                bundles=bundles, holdout_idx=k, seeds=seeds,
                layer=layer, epochs=epochs, lr=lr,
                hidden=hidden, heads=heads, layers=layers, dropout=dropout,
                workdir=workdir, mode=mode,
                global_metadata=global_metadata,
                variant=variant,
            )
            fold_results.append(fold)
        except Exception as exc:
            logger.exception("  Fold failed (holdout=%s): %s", bundles[k].scenario_id, exc)
            continue

    # Cross-fold aggregation
    if not fold_results:
        raise RuntimeError("All LOSO folds failed — nothing to report.")

    all_rhos = [f.mean_metrics["spearman_rho"] for f in fold_results]
    all_f1s = [f.mean_metrics["f1_at_k"] for f in fold_results]
    all_ndcgs = [f.mean_metrics["ndcg_10"] for f in fold_results]

    type_to_rhos: Dict[str, List[float]] = {}
    for f in fold_results:
        for nt, info in f.per_type_rho.items():
            type_to_rhos.setdefault(nt, []).append(info["mean"])

    per_type_summary = {
        nt: {
            "mean": float(np.mean(vs)),
            "std": float(np.std(vs)),
            "n_folds": len(vs),
        }
        for nt, vs in type_to_rhos.items()
    }

    return LOSOReport(
        fold_results=fold_results,
        overall_mean_rho=float(np.mean(all_rhos)),
        overall_std_rho=float(np.std(all_rhos)),
        overall_mean_f1=float(np.mean(all_f1s)),
        overall_mean_ndcg=float(np.mean(all_ndcgs)),
        n_folds=len(fold_results),
        n_seeds_per_fold=len(seeds),
        per_type_summary=per_type_summary,
        scenario_predictions={f.holdout_id: f.node_predictions for f in fold_results},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Output writers
# ──────────────────────────────────────────────────────────────────────────────

def write_results_json(report: LOSOReport, path: Path) -> None:
    payload = {
        "summary": {
            "n_folds": report.n_folds,
            "n_seeds_per_fold": report.n_seeds_per_fold,
            "overall_mean_spearman_rho": report.overall_mean_rho,
            "overall_std_spearman_rho": report.overall_std_rho,
            "overall_mean_f1_at_k": report.overall_mean_f1,
            "overall_mean_ndcg_10": report.overall_mean_ndcg,
        },
        "per_type_summary": report.per_type_summary,
        "folds": [
            {
                "holdout_id": f.holdout_id,
                "primary_id": f.primary_id,
                "train_ids": f.train_ids,
                "mean_metrics": f.mean_metrics,
                "std_metrics": f.std_metrics,
                "per_type_rho": f.per_type_rho,
                "seed_metrics": f.seed_metrics,
            }
            for f in report.fold_results
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", path)


def write_per_fold_csv(report: LOSOReport, path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "holdout_id", "primary_id", "seed",
            "spearman_rho", "f1_at_k", "ndcg_10", "rmse", "mae",
            "n", "prediction_mode",
        ])
        for fold in report.fold_results:
            for m in fold.seed_metrics:
                w.writerow([
                    fold.holdout_id, fold.primary_id, m["seed"],
                    f"{m['spearman_rho']:.4f}",
                    f"{m['f1_at_k']:.4f}",
                    f"{m['ndcg_10']:.4f}",
                    f"{m['rmse']:.4f}",
                    f"{m['mae']:.4f}",
                    m["n"], m.get("prediction_mode", ""),
                ])
    logger.info("Wrote %s", path)


def write_summary_md(report: LOSOReport, path: Path) -> None:
    L: List[str] = []
    L.append("# LOSO Evaluation Summary (G4 closure)")
    L.append("")
    L.append(f"**Folds:** {report.n_folds}  ·  **Seeds per fold:** {report.n_seeds_per_fold}")
    L.append("")
    L.append("## Cross-fold")
    L.append("")
    L.append(f"- Spearman ρ : **{report.overall_mean_rho:.4f} ± {report.overall_std_rho:.4f}**")
    L.append(f"- F1 @ K     : {report.overall_mean_f1:.4f}")
    L.append(f"- NDCG @ 10  : {report.overall_mean_ndcg:.4f}")
    L.append("")
    L.append("## Per node type (cross-fold)")
    L.append("")
    L.append("| Node type | mean ρ | std | folds |")
    L.append("|-----------|--------|-----|-------|")
    for nt, info in sorted(report.per_type_summary.items()):
        L.append(f"| {nt} | {info['mean']:.4f} | {info['std']:.4f} | {info['n_folds']} |")
    L.append("")
    L.append("## Per-fold details")
    L.append("")
    L.append("| Holdout | Primary | mean ρ | std ρ | mean F1 | mean NDCG@10 |")
    L.append("|---------|---------|--------|-------|---------|--------------|")
    for f in report.fold_results:
        L.append(
            f"| {f.holdout_id} | {f.primary_id} "
            f"| {f.mean_metrics['spearman_rho']:.4f} "
            f"| {f.std_metrics['spearman_rho']:.4f} "
            f"| {f.mean_metrics['f1_at_k']:.4f} "
            f"| {f.mean_metrics['ndcg_10']:.4f} |"
        )
    path.write_text("\n".join(L) + "\n")
    logger.info("Wrote %s", path)


def write_predictions_json(report: LOSOReport, path: Path) -> None:
    """Save inductive predictions for use in Step 4/Step 6."""
    # Format: {scenario_id: {node_id: {overall, reliability, ...}}}
    path.write_text(json.dumps(report.scenario_predictions, indent=2))
    logger.info("Wrote %s (prediction-step format)", path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Leave-One-Scenario-Out inductive evaluation (G4 closure).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cache-dir", type=Path, default=Path("output/loso_cache"),
                   help="Per-scenario cache root")
    p.add_argument("--output-dir", type=Path, default=Path("output/loso"),
                   help="Output directory for LOSO results")
    p.add_argument("--layer", default="app", choices=["app", "infra", "mw", "system"])
    p.add_argument("--seeds", default="42,123,456,789,2024",
                   help="Comma-separated training seeds")
    p.add_argument("--skip", default="",
                   help="Comma-separated scenario id substrings to skip")
    p.add_argument("--mode", default="gnn", choices=["gnn", "rmav", "ensemble"],
                   help="Prediction mode for evaluation (default: gnn)")
    p.add_argument(
        "--variant",
        choices=["hgl_qos", "hgl", "gl_qos", "gl", "topology_rmav"],
        default="hgl_qos",
        help=(
            "Model architecture variant (default: hgl_qos). "
            "hgl_qos = QoS-embedded HeteroGAT on native graph; "
            "hgl     = QoS-masked HeteroGAT on native graph; "
            "gl_qos  = QoS-weighted homogeneous GAT on projection; "
            "gl      = unweighted homogeneous GAT on projection; "
            "topology_rmav = RMAV scores only (no GNN)."
        ),
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    skip = [s.strip() for s in args.skip.split(",") if s.strip()]

    logger.info("LOSO Evaluation — G4 closure")
    logger.info("  Cache:     %s", args.cache_dir)
    logger.info("  Output:    %s", args.output_dir)
    logger.info("  Layer:     %s", args.layer)
    logger.info("  Seeds:     %s", seeds)
    logger.info("  Mode:      %s", args.mode)
    logger.info("  Variant:   %s", getattr(args, 'variant', 'hetero_qos'))
    logger.info("  Skip:      %s", skip if skip else "(none)")

    if not args.cache_dir.exists():
        logger.error("Cache dir not found: %s", args.cache_dir)
        logger.error("See module docstring for the cache-population shell loop.")
        return 2

    bundles = discover_scenarios(args.cache_dir, skip=skip)
    logger.info("Loaded %d scenarios for LOSO.", len(bundles))

    t0 = time.time()
    report = run_loso(
        bundles=bundles, seeds=seeds, output_dir=args.output_dir,
        layer=args.layer, epochs=args.epochs, lr=args.lr,
        hidden=args.hidden, heads=args.heads, layers=args.layers,
        dropout=args.dropout, mode=args.mode,
        variant=getattr(args, 'variant', 'hetero_qos'),
    )
    elapsed = time.time() - t0
    logger.info("LOSO complete in %.1f s.", elapsed)

    write_results_json(report, args.output_dir / "results.json")
    write_predictions_json(report, args.output_dir / "inductive_predictions.json")
    write_per_fold_csv(report, args.output_dir / "per_fold_metrics.csv")
    write_summary_md(report, args.output_dir / "summary.md")

    print()
    print("=" * 64)
    print(f"  LOSO ρ = {report.overall_mean_rho:.4f} ± {report.overall_std_rho:.4f}"
          f"  (n_folds = {report.n_folds})")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
