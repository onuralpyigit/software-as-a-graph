#!/usr/bin/env python3
"""
cli/kfold_evaluate.py — Per-Domain Repeated Stratified K-Fold Evaluation
==========================================================================

Replaces cross-scenario Leave-One-Scenario-Out (see cli/loso_evaluate.py) as
the primary validation protocol. Each of the 11 domain scenarios
(data/scenarios/*.yaml — autonomous vehicle, IoT, HFT, healthcare,
hub-and-spoke SPOF, microservices, enterprise, tiny/xlarge stress, ATM,
broker redundancy) represents a genuinely distinct domain with its own
topology and QoS dynamics. Rather than testing zero-shot transfer across
unrelated domains (LOSO's claim), this protocol validates how well the model
fits *each domain's own* dynamics: for every scenario, independently run
repeated stratified k-fold over that scenario's own labelled nodes. No
scenario ever contributes training data to another scenario's evaluation.

────────────────────────────────────────────────────────────────────────────
Protocol
────────────────────────────────────────────────────────────────────────────
For each scenario s (independently, no cross-scenario data):
    For each seed r ∈ {42, 123, 456, 789, 2024}    (repeats)
        For each fold f ∈ {0 .. k-1}:
            create_kfold_masks(data_s, k, fold_idx=f, seed=1000+f)  # fixed
                                                                     # per fold
            train on data_s.train_mask / val_mask (transductive, single graph)
            evaluate on data_s.test_mask nodes only
            Compute ρ, F1@K, NDCG@10, RMSE, MAE — overall and per-node-type

Reports: per-(scenario, fold) mean ± std across repeats, then per-scenario
mean ± std across folds, then cross-scenario summary.

────────────────────────────────────────────────────────────────────────────
Cache layout — identical to cli/loso_evaluate.py, reused as-is
────────────────────────────────────────────────────────────────────────────
    output/loso_cache/<scenario_id>/
        topology.json              (input — same JSON as cli/import_graph.py)
        structural_metrics.json    (output of cli/analyze_graph.py)
        quality_scores.json        (output of cli/predict_graph.py --mode rmav)
        failure_impact.json        (output of cli/simulate_graph.py fault-inject)

See cli/loso_evaluate.py's module docstring for the cache-population loop
(scripts/populate_loso_cache.sh automates it).

────────────────────────────────────────────────────────────────────────────
Usage
────────────────────────────────────────────────────────────────────────────
    PYTHONPATH=. python cli/kfold_evaluate.py \
        --cache-dir output/loso_cache \
        --output-dir output/kfold \
        --layer app --k 5

    # Skip xlarge scenarios for fast iteration
    PYTHONPATH=. python cli/kfold_evaluate.py --skip scenario_07,scenario_09
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
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import torch

# ── SaG SDK imports ──────────────────────────────────────────────────────────
from saag.prediction.gnn_service import GNNService
from saag.prediction.data_preparation import (
    networkx_to_hetero_data,
    create_kfold_masks,
)

# Reuse LOSO's cache-loading, bundle dataclass and metric computation unchanged.
from cli.loso_evaluate import (
    ScenarioBundle,
    discover_scenarios,
    compute_inductive_metrics,
)

logger = logging.getLogger("kfold_evaluate")


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    """Result of one (scenario, fold) pair, aggregated across repeat seeds."""
    scenario_id: str
    fold_idx: int
    n_test: int
    seed_metrics: List[Dict[str, Any]] = field(default_factory=list)
    mean_metrics: Dict[str, float] = field(default_factory=dict)
    std_metrics: Dict[str, float] = field(default_factory=dict)
    per_type_rho: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Aggregate k-fold results for a single scenario (domain)."""
    scenario_id: str
    fold_results: List[FoldResult] = field(default_factory=list)
    mean_metrics: Dict[str, float] = field(default_factory=dict)
    std_metrics: Dict[str, float] = field(default_factory=dict)
    per_type_rho: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class KFoldReport:
    """Aggregate k-fold results across all scenarios."""
    scenario_results: List[ScenarioResult] = field(default_factory=list)
    overall_mean_rho: float = 0.0
    overall_std_rho: float = 0.0
    overall_mean_f1: float = 0.0
    overall_mean_ndcg: float = 0.0
    n_scenarios: int = 0
    k_folds: int = 0
    n_seeds_per_fold: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# Single-scenario k-fold execution
# ──────────────────────────────────────────────────────────────────────────────

def _test_node_ids(data, node_id_map: Dict[str, List[str]]) -> set:
    """Node ids whose test_mask is True across all node stores."""
    ids = set()
    for nt in data.node_types:
        store = data[nt]
        if not hasattr(store, "test_mask"):
            continue
        mask = store.test_mask.numpy()
        for local_idx, nid in enumerate(node_id_map.get(nt, [])):
            if local_idx < len(mask) and mask[local_idx]:
                ids.add(nid)
    return ids


def run_one_scenario(
    bundle: ScenarioBundle,
    k: int,
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
    variant: str,
    weight_decay: float,
    warmup_T0: Optional[int],
    multitask_weight: float,
    rmav_consistency_weight: float,
    ranking_weight: float,
    pairwise_ranking_weight: float,
) -> ScenarioResult:
    """Repeated stratified k-fold within a single scenario's own graph."""
    scenario_dir = workdir / bundle.scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    use_qos = variant in ("hgl_qos", "gl_qos")
    if variant in ("hgl", "gl"):
        from reproduce.middleware26_main_table import _mask_qos_in_graph, _mask_qos_in_structural
        train_graph = _mask_qos_in_graph(bundle.graph)
        train_sm = _mask_qos_in_structural(bundle.structural)
    else:
        train_graph = bundle.graph
        train_sm = bundle.structural

    true_impact = {nid: float(d.get("composite", 0.0)) for nid, d in bundle.simulation.items()}

    fold_results: List[FoldResult] = []

    for fold_idx in range(k):
        fold_dir = scenario_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        split_seed = 1000 + fold_idx  # fixed per fold: same test nodes across repeat seeds

        seed_metrics: List[Dict[str, Any]] = []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            ckpt_dir = fold_dir / f"seed_{seed}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            try:
                if variant in ("gl", "gl_qos"):
                    from saag.prediction.models.baselines import build_baseline
                    from saag.prediction.trainer import GNNTrainer

                    conv = networkx_to_hetero_data(
                        train_graph, train_sm, bundle.simulation, bundle.rmav, qos_enabled=use_qos
                    )
                    data = conv.hetero_data
                    create_kfold_masks(data, k=k, fold_idx=fold_idx, seed=split_seed)

                    baseline_name = "homo_scalar" if variant == "gl_qos" else "homo_unweighted"
                    model = build_baseline(
                        baseline_name, hidden_channels=hidden, num_heads=heads,
                        num_layers=layers, dropout=dropout,
                    )
                    trainer = GNNTrainer(
                        model=model, checkpoint_dir=str(ckpt_dir),
                        lr=lr, num_epochs=epochs, patience=min(60, epochs),
                        weight_decay=weight_decay, warmup_T0=warmup_T0,
                        multitask_weight=multitask_weight,
                        rmav_consistency_weight=rmav_consistency_weight,
                        ranking_weight=ranking_weight,
                        pairwise_ranking_weight=pairwise_ranking_weight,
                    )
                    trainer.train(data)

                    test_ids = _test_node_ids(data, conv.node_id_map)
                    model.eval()
                    with torch.no_grad():
                        x = {nt: data[nt].x for nt in data.node_types if hasattr(data[nt], "x")}
                        ei = {r: data[r].edge_index for r in data.edge_types}
                        ea = {r: data[r].edge_attr for r in data.edge_types if hasattr(data[r], "edge_attr")}
                        out = model(x, ei, ea)
                    pred_scores = {}
                    for nt, preds in out.items():
                        node_list = conv.node_id_map.get(nt, [])
                        for local_idx, nid in enumerate(node_list):
                            if nid in test_ids and local_idx < preds.shape[0]:
                                pred_scores[nid] = float(preds[local_idx, 0])

                else:
                    # hgl_qos (default), hgl, or topology_rmav → GNNService path.
                    # GNNService.train() always calls (module-level) create_node_splits
                    # internally; monkeypatch it to our fixed k-fold split for the
                    # duration of this call so the rest of GNNService's training/
                    # inference machinery (model construction, IQR label norm,
                    # multi-seed best-state tracking, RMAV blending) is reused unchanged.
                    import saag.prediction.gnn_service as gnn_service_mod

                    effective_mode = "rmav" if variant == "topology_rmav" else mode
                    effective_epochs = 1 if variant == "topology_rmav" else epochs

                    def _fold_split(hd, train_ratio=0.6, val_ratio=0.2, seed=None):
                        create_kfold_masks(hd, k=k, fold_idx=fold_idx, val_ratio=val_ratio, seed=split_seed)

                    original_splitter = gnn_service_mod.create_node_splits
                    gnn_service_mod.create_node_splits = _fold_split
                    try:
                        service = GNNService(
                            checkpoint_dir=str(ckpt_dir),
                            hidden_channels=hidden,
                            num_heads=heads,
                            num_layers=layers,
                            dropout=dropout,
                            predict_edges=False,
                        )
                        service.train(
                            graph=train_graph,
                            structural_metrics=train_sm,
                            simulation_results=bundle.simulation,
                            rmav_scores=bundle.rmav,
                            num_epochs=effective_epochs,
                            lr=lr,
                            patience=min(60, epochs),
                            layer=layer,
                            qos_enabled=use_qos,
                            weight_decay=weight_decay,
                            warmup_T0=warmup_T0,
                            multitask_weight=multitask_weight,
                            rmav_consistency_weight=rmav_consistency_weight,
                            ranking_weight=ranking_weight,
                            pairwise_ranking_weight=pairwise_ranking_weight,
                            seeds=[seed],
                            mode=effective_mode,
                        )
                        conv = service._conversion_result
                        test_ids = _test_node_ids(conv.hetero_data, conv.node_id_map)
                        result = service.predict_from_data(conv.hetero_data, bundle.simulation, mode=effective_mode)
                        pred_scores = {
                            nid: float(ns.composite_score)
                            for nid, ns in result.node_scores.items()
                            if nid in test_ids
                        }
                    finally:
                        gnn_service_mod.create_node_splits = original_splitter

            except Exception as e:
                logger.error(
                    "  [%s] fold %d seed %d failed: %s",
                    bundle.scenario_id, fold_idx, seed, e, exc_info=True,
                )
                continue

            m = compute_inductive_metrics(pred_scores, true_impact, bundle.graph)
            m["seed"] = seed
            m["prediction_mode"] = mode
            m["variant"] = variant
            seed_metrics.append(m)

            logger.info(
                "  [%s] fold %d seed %d: ρ=%.4f F1=%.4f NDCG=%.4f RMSE=%.4f (n_test=%d)",
                bundle.scenario_id, fold_idx, seed,
                m["spearman_rho"], m["f1_at_k"], m["ndcg_10"], m["rmse"], m["n"],
            )

        if not seed_metrics:
            logger.warning("  [%s] fold %d: all seeds failed, skipping fold.", bundle.scenario_id, fold_idx)
            continue

        rho_vals = [m["spearman_rho"] for m in seed_metrics]
        f1_vals = [m["f1_at_k"] for m in seed_metrics]
        ndcg_vals = [m["ndcg_10"] for m in seed_metrics]
        rmse_vals = [m["rmse"] for m in seed_metrics]

        per_type_agg: Dict[str, List[float]] = {}
        for m in seed_metrics:
            for nt, info in m.get("per_type_rho", {}).items():
                per_type_agg.setdefault(nt, []).append(info["rho"])
        per_type_summary = {
            nt: {"mean": float(np.mean(vs)), "std": float(np.std(vs)), "n_seeds": len(vs)}
            for nt, vs in per_type_agg.items()
        }

        fold_results.append(FoldResult(
            scenario_id=bundle.scenario_id,
            fold_idx=fold_idx,
            n_test=seed_metrics[0]["n"],
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
        ))

    if not fold_results:
        raise RuntimeError(f"All folds failed for scenario {bundle.scenario_id}.")

    fold_rhos = [f.mean_metrics["spearman_rho"] for f in fold_results]
    fold_f1s = [f.mean_metrics["f1_at_k"] for f in fold_results]
    fold_ndcgs = [f.mean_metrics["ndcg_10"] for f in fold_results]
    fold_rmses = [f.mean_metrics["rmse"] for f in fold_results]

    type_to_rhos: Dict[str, List[float]] = {}
    for f in fold_results:
        for nt, info in f.per_type_rho.items():
            type_to_rhos.setdefault(nt, []).append(info["mean"])
    per_type_summary = {
        nt: {"mean": float(np.mean(vs)), "std": float(np.std(vs)), "n_folds": len(vs)}
        for nt, vs in type_to_rhos.items()
    }

    return ScenarioResult(
        scenario_id=bundle.scenario_id,
        fold_results=fold_results,
        mean_metrics={
            "spearman_rho": float(np.mean(fold_rhos)),
            "f1_at_k": float(np.mean(fold_f1s)),
            "ndcg_10": float(np.mean(fold_ndcgs)),
            "rmse": float(np.mean(fold_rmses)),
        },
        std_metrics={
            "spearman_rho": float(np.std(fold_rhos)),
            "f1_at_k": float(np.std(fold_f1s)),
            "ndcg_10": float(np.std(fold_ndcgs)),
            "rmse": float(np.std(fold_rmses)),
        },
        per_type_rho=per_type_summary,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Full k-fold orchestration
# ──────────────────────────────────────────────────────────────────────────────

def run_kfold(
    bundles: List[ScenarioBundle],
    k: int,
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
    variant: str,
    weight_decay: float,
    warmup_T0: Optional[int],
    multitask_weight: float,
    rmav_consistency_weight: float,
    ranking_weight: float,
    pairwise_ranking_weight: float,
) -> KFoldReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    workdir = output_dir / "workspace"
    workdir.mkdir(exist_ok=True)

    scenario_results: List[ScenarioResult] = []
    for i, bundle in enumerate(bundles):
        logger.info("════════════════════════════════════════════════════════════")
        logger.info("Scenario %d / %d   %s (%d nodes, %d labelled)",
                     i + 1, len(bundles), bundle.scenario_id, bundle.n_nodes, bundle.n_labelled)
        logger.info("════════════════════════════════════════════════════════════")

        this_k = min(k, bundle.n_labelled) if bundle.n_labelled >= 2 else k
        if this_k < k:
            logger.warning(
                "  [%s] only %d labelled nodes — reducing k %d -> %d for this scenario.",
                bundle.scenario_id, bundle.n_labelled, k, this_k,
            )

        try:
            result = run_one_scenario(
                bundle=bundle, k=this_k, seeds=seeds,
                layer=layer, epochs=epochs, lr=lr,
                hidden=hidden, heads=heads, layers=layers, dropout=dropout,
                workdir=workdir, mode=mode, variant=variant,
                weight_decay=weight_decay, warmup_T0=warmup_T0,
                multitask_weight=multitask_weight,
                rmav_consistency_weight=rmav_consistency_weight,
                ranking_weight=ranking_weight,
                pairwise_ranking_weight=pairwise_ranking_weight,
            )
            scenario_results.append(result)
        except Exception as exc:
            logger.exception("  Scenario failed (%s): %s", bundle.scenario_id, exc)
            continue

    if not scenario_results:
        raise RuntimeError("All scenarios failed — nothing to report.")

    all_rhos = [r.mean_metrics["spearman_rho"] for r in scenario_results]
    all_f1s = [r.mean_metrics["f1_at_k"] for r in scenario_results]
    all_ndcgs = [r.mean_metrics["ndcg_10"] for r in scenario_results]

    return KFoldReport(
        scenario_results=scenario_results,
        overall_mean_rho=float(np.mean(all_rhos)),
        overall_std_rho=float(np.std(all_rhos)),
        overall_mean_f1=float(np.mean(all_f1s)),
        overall_mean_ndcg=float(np.mean(all_ndcgs)),
        n_scenarios=len(scenario_results),
        k_folds=k,
        n_seeds_per_fold=len(seeds),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Output writers
# ──────────────────────────────────────────────────────────────────────────────

def write_results_json(report: KFoldReport, path: Path) -> None:
    payload = {
        "summary": {
            "n_scenarios": report.n_scenarios,
            "k_folds": report.k_folds,
            "n_seeds_per_fold": report.n_seeds_per_fold,
            "overall_mean_spearman_rho": report.overall_mean_rho,
            "overall_std_spearman_rho": report.overall_std_rho,
            "overall_mean_f1_at_k": report.overall_mean_f1,
            "overall_mean_ndcg_10": report.overall_mean_ndcg,
        },
        "scenarios": [
            {
                "scenario_id": r.scenario_id,
                "n_folds": len(r.fold_results),
                "mean_metrics": r.mean_metrics,
                "std_metrics": r.std_metrics,
                "per_type_rho": r.per_type_rho,
                "folds": [
                    {
                        "fold_idx": f.fold_idx,
                        "n_test": f.n_test,
                        "mean_metrics": f.mean_metrics,
                        "std_metrics": f.std_metrics,
                        "per_type_rho": f.per_type_rho,
                        "seed_metrics": f.seed_metrics,
                    }
                    for f in r.fold_results
                ],
            }
            for r in report.scenario_results
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", path)


def write_per_fold_csv(report: KFoldReport, path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "scenario_id", "fold_idx", "seed",
            "spearman_rho", "f1_at_k", "ndcg_10", "rmse", "mae",
            "n", "prediction_mode",
        ])
        for r in report.scenario_results:
            for fold in r.fold_results:
                for m in fold.seed_metrics:
                    w.writerow([
                        r.scenario_id, fold.fold_idx, m["seed"],
                        f"{m['spearman_rho']:.4f}",
                        f"{m['f1_at_k']:.4f}",
                        f"{m['ndcg_10']:.4f}",
                        f"{m['rmse']:.4f}",
                        f"{m['mae']:.4f}",
                        m["n"], m.get("prediction_mode", ""),
                    ])
    logger.info("Wrote %s", path)


def write_summary_md(report: KFoldReport, path: Path) -> None:
    L: List[str] = []
    L.append("# Per-Domain K-Fold Evaluation Summary")
    L.append("")
    L.append(f"**Scenarios:** {report.n_scenarios}  ·  **k:** {report.k_folds}  ·  "
             f"**Seeds per fold:** {report.n_seeds_per_fold}")
    L.append("")
    L.append("## Cross-scenario")
    L.append("")
    L.append(f"- Spearman ρ : **{report.overall_mean_rho:.4f} ± {report.overall_std_rho:.4f}**")
    L.append(f"- F1 @ K     : {report.overall_mean_f1:.4f}")
    L.append(f"- NDCG @ 10  : {report.overall_mean_ndcg:.4f}")
    L.append("")
    L.append("## Per-scenario (in-domain fit)")
    L.append("")
    L.append("| Scenario | mean ρ | std ρ | mean F1 | mean NDCG@10 | folds |")
    L.append("|----------|--------|-------|---------|--------------|-------|")
    for r in report.scenario_results:
        L.append(
            f"| {r.scenario_id} "
            f"| {r.mean_metrics['spearman_rho']:.4f} "
            f"| {r.std_metrics['spearman_rho']:.4f} "
            f"| {r.mean_metrics['f1_at_k']:.4f} "
            f"| {r.mean_metrics['ndcg_10']:.4f} "
            f"| {len(r.fold_results)} |"
        )
    path.write_text("\n".join(L) + "\n")
    logger.info("Wrote %s", path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-domain repeated stratified k-fold evaluation (replaces cross-scenario LOSO).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cache-dir", type=Path, default=Path("output/loso_cache"),
                   help="Per-scenario cache root (same layout as loso_evaluate.py)")
    p.add_argument("--output-dir", type=Path, default=Path("output/kfold"),
                   help="Output directory for k-fold results")
    p.add_argument("--layer", default="app", choices=["app", "infra", "mw", "system"])
    p.add_argument("--k", type=int, default=5, help="Number of folds per scenario")
    p.add_argument("--seeds", default="42,123,456,789,2024",
                   help="Comma-separated repeat seeds (model init randomness within each fold)")
    p.add_argument("--skip", default="",
                   help="Comma-separated scenario id substrings to skip")
    p.add_argument("--mode", default="gnn", choices=["gnn", "rmav"],
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
    p.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    p.add_argument("--warmup-t0", type=int, default=None,
                    help="T_0 for CosineAnnealingWarmRestarts (default: max(50, epochs//4))")
    p.add_argument("--multitask-weight", type=float, default=0.5,
                    help="CriticalityLoss weight for per-dimension R/M/A/V MSE term")
    p.add_argument("--ranking-weight", type=float, default=0.3,
                    help="CriticalityLoss weight for the ListMLE ranking term")
    p.add_argument("--pairwise-ranking-weight", type=float, default=0.1,
                    help="CriticalityLoss weight for the pairwise margin-ranking term")
    p.add_argument("--rmav-consistency-weight", type=float, default=0.1,
                    help="CriticalityLoss weight for RMAV consistency regularization on unlabeled nodes")
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

    logger.info("Per-Domain K-Fold Evaluation")
    logger.info("  Cache:     %s", args.cache_dir)
    logger.info("  Output:    %s", args.output_dir)
    logger.info("  Layer:     %s", args.layer)
    logger.info("  k:         %s", args.k)
    logger.info("  Seeds:     %s", seeds)
    logger.info("  Mode:      %s", args.mode)
    logger.info("  Variant:   %s", args.variant)
    logger.info("  Skip:      %s", skip if skip else "(none)")

    if not args.cache_dir.exists():
        logger.error("Cache dir not found: %s", args.cache_dir)
        logger.error("See cli/loso_evaluate.py's module docstring for the cache-population loop.")
        return 2

    bundles = discover_scenarios(args.cache_dir, skip=skip)
    logger.info("Loaded %d scenarios for k-fold evaluation.", len(bundles))

    t0 = time.time()
    report = run_kfold(
        bundles=bundles, k=args.k, seeds=seeds, output_dir=args.output_dir,
        layer=args.layer, epochs=args.epochs, lr=args.lr,
        hidden=args.hidden, heads=args.heads, layers=args.layers,
        dropout=args.dropout, mode=args.mode, variant=args.variant,
        weight_decay=args.weight_decay, warmup_T0=args.warmup_t0,
        multitask_weight=args.multitask_weight,
        rmav_consistency_weight=args.rmav_consistency_weight,
        ranking_weight=args.ranking_weight,
        pairwise_ranking_weight=args.pairwise_ranking_weight,
    )
    elapsed = time.time() - t0
    logger.info("K-fold evaluation complete in %.1f s.", elapsed)

    write_results_json(report, args.output_dir / "results.json")
    write_per_fold_csv(report, args.output_dir / "per_fold_metrics.csv")
    write_summary_md(report, args.output_dir / "summary.md")

    print()
    print("=" * 64)
    print(f"  K-Fold ρ = {report.overall_mean_rho:.4f} ± {report.overall_std_rho:.4f}"
          f"  (n_scenarios = {report.n_scenarios}, k = {report.k_folds})")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
