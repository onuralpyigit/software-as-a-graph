#!/usr/bin/env python3
"""
cli/loso_evaluate.py — Leave-One-Scenario-Out Inductive Evaluation
==================================================================

Closes G4 (transductive leakage) for the GNN Predict stage by establishing
a strict inductive evaluation protocol: for every scenario k in the suite,
train the HGT on the N-1 remaining scenarios and evaluate on k. The
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
    primary  := argmax_{j ∈ train_set} |V_j|  (the graph splits are drawn on)
    val      := median-sized inductive        (--inner-val-scenario auto only)
    inductive := train_set \\ {primary, val}  (passed via inductive_graphs)

For each seed s ∈ {42, 123, 456, 789, 2024}:
    GNNService.train(primary, inductive_graphs=inductive, val_graph=val, seeds=[s])
    GNNService.predict(holdout_graph)         # holdout never seen
    Compute ρ, F1@K, NDCG@10, RMSE, MAE — overall and per-node-type

Reports: per-fold mean ± std across seeds, then cross-fold mean ± std.

Two options change what is being measured rather than how well:

  --inner-val-scenario auto
      Early stopping and checkpoint selection move off a within-`primary` split
      onto a training scenario held out of the loss entirely. Selecting on a
      split of a training scenario selects for in-distribution fit, under a
      protocol whose entire point is distribution shift.

  --no-auto-layers
      Fixes the depth at --layers for every fold. With the default auto-downgrade
      the depth is derived from `primary`'s size, and `primary` changes with the
      holdout — on the 8-scenario corpus, seven folds train 3 layers and the fold
      that holds out enterprise_system trains 2. A capacity difference that
      tracks the fold is a confound in whatever that table reports.
      reproduce/loso_all_variants.py passes this by default.

────────────────────────────────────────────────────────────────────────────
Cache layout (one directory per scenario)
────────────────────────────────────────────────────────────────────────────
    output/loso_cache/<scenario_id>/
        topology.json              (input — same JSON as cli/import_graph.py)
        structural_metrics.json    (output of cli/analyze_graph.py)
        quality_scores.json        (output of cli/predict_graph.py --mode rm)
        failure_impact.json        (output of cli/simulate_graph.py fault-inject)

To populate the cache from existing pipeline outputs:

    for cfg in data/scenario_*.yaml; do
        sid=$(basename "$cfg" .yaml)
        out="output/loso_cache/$sid"
        mkdir -p "$out"
        PYTHONPATH=. python cli/generate_graph.py --config "$cfg" --output "$out/topology.json"
        PYTHONPATH=. python cli/import_graph.py    --input "$out/topology.json" --clear
        PYTHONPATH=. python cli/analyze_graph.py   --layer app --output "$out/structural_metrics.json"
        PYTHONPATH=. python cli/predict_graph.py   --layer app --mode rm --output "$out/quality_scores.json"
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

    # Use gnn mode for inductive eval
    PYTHONPATH=. python cli/loso_evaluate.py --mode gnn
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from torch_geometric.data import HeteroData

# ── SaG SDK imports ──────────────────────────────────────────────────────────
from saag.evaluation.fingerprint import fit_fingerprint
from saag.evaluation.metrics import (
    aggregate_per_type,
    compute_inductive_metrics as _shared_inductive_metrics,
    resolve_eval_keys,
)
from saag.prediction.gnn_service import GNNService
from saag.prediction.data_preparation import (
    networkx_to_hetero_data,
    normalize_labels_robust,
    extract_simulation_dict,
    extract_structural_metrics_dict,
    extract_rm_scores_dict,
    extract_edge_simulation_dict,
)
from saag.core.models import QoSPolicy, topic_weight_from_node_attrs
from saag.evaluation import variant_registry as _registry

logger = logging.getLogger("loso_evaluate")


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

#: Which dispatch branch each variant id takes. Declared here rather than
#: inline so the branches and the ``--variant`` choices list can be checked
#: against each other by tests/test_variant_dispatch.py. Before these existed
#: the final branch was a bare ``else``, so a typo'd or newly-added id ran
#: silently as an HGT and reported the result under its own name.
#:
#: ``topology_rm`` belongs with the HGT ids: it reaches the GNNService branch
#: only to be forced to ``mode="rm"`` and one epoch, and moving it would change
#: a published baseline.
_STRUCTURAL_VARIANTS = ("topo_baseline", "topo_qos")
_HOMOGENEOUS_VARIANTS = (
    "gl", "gl_qos", "gl_full_cap", "gl_full_qos_cap", "gl_full_qos16_cap",
)
_HGT_VARIANTS = ("hgl", "hgl_qos", "hgl_qos_uni", "topology_rm")
KNOWN_VARIANTS = _STRUCTURAL_VARIANTS + _HOMOGENEOUS_VARIANTS + _HGT_VARIANTS


@dataclass
class ScenarioBundle:
    """All artefacts needed to use a scenario as a training or evaluation graph."""
    scenario_id: str
    graph: nx.DiGraph
    structural: Dict[str, Any]
    rm: Dict[str, Any]
    simulation: Dict[str, Any]
    hetero_data: HeteroData
    n_nodes: int
    n_edges: int
    n_labelled: int
    #: Provenance carried through from failure_impact.json: which engine wrote
    #: the labels, over how many seeds, and how well they agree with themselves.
    #: The test-retest rho here is the ceiling on any rho reported against them.
    label_stability: Dict[str, Any] = field(default_factory=dict)
    labeler: str = ""
    #: ``{(source, target): combined_impact}`` from the edge-removal sweep, when
    #: the cache carries an ``edge_criticality.json``. Empty leaves the edge head
    #: unsupervised rather than substituting a structural proxy for a
    #: measurement — see ``networkx_to_hetero_data``.
    edge_simulation: Dict[Any, float] = field(default_factory=dict)


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
    #: Depth this fold actually trained at. Reported because ``--auto-layers``
    #: derives it from the *primary* scenario's size, which changes with the
    #: holdout: enterprise_system is primary in 7 of 8 folds (3 layers) and the
    #: fold that holds it out silently drops to 2. A capacity difference that
    #: tracks the fold is a confound, so it is recorded rather than inferred.
    effective_layers: Optional[int] = None
    #: Scenario held out of training to drive early stopping, when
    #: ``--inner-val-scenario auto`` is in effect. None = legacy behaviour
    #: (selection on a within-primary split).
    val_scenario_id: Optional[str] = None


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
    #: {scenario_id: label_stability} carried from each cache artifact, so the
    #: report can state the ceiling alongside the achieved rho.
    label_stability: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: Node population every fold was scored on (see ``--eval-population``).
    eval_population: str = "application"


# ──────────────────────────────────────────────────────────────────────────────
# Cache loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


# Graph construction moved to saag/core/graph_io.py: fifteen modules under
# reproduce/ imported these by their private names, which made a core
# primitive depend on a CLI script. Re-exported here so existing imports
# keep working.
from saag.core.graph_io import (  # noqa: E402
    build_graph_from_json as _build_graph_from_json,
    _project_topic_qos_onto_edges,
    _TOPIC_MEDIATED_EDGES,
)


def load_scenario_bundle(scenario_dir: Path) -> Optional[ScenarioBundle]:
    """Load one scenario's full artefact bundle from cache. Returns None on incomplete cache."""
    scenario_id = scenario_dir.name
    topology = _load_json(scenario_dir / "topology.json")
    structural_raw = _load_json(scenario_dir / "structural_metrics.json")
    rm_raw = _load_json(scenario_dir / "quality_scores.json")
    sim_raw = _load_json(scenario_dir / "failure_impact.json")
    # Optional: absent in caches built before the edge-removal sweep existed.
    edge_raw = _load_json(scenario_dir / "edge_criticality.json")

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

    # A cache whose topology has drifted from its committed dataset describes a graph
    # that no longer exists in the repository; evaluating against it silently publishes
    # unreproducible numbers. Same guard reproduce/main_table.py applies.
    from reproduce.main_table import _assert_cache_matches_dataset
    _assert_cache_matches_dataset(
        scenario_id, topology, Path("data/scenarios") / f"{scenario_id}.json"
    )

    graph = _build_graph_from_json(topology)
    structural = extract_structural_metrics_dict(structural_raw)

    try:
        from reproduce.main_table import _parse_failure_impact, _parse_quality_scores, _remap_node_ids
        sim_parsed = _parse_failure_impact(sim_raw)
        rm_parsed = _parse_quality_scores(rm_raw) if rm_raw else {}
        graph_nodes = set(str(n) for n in graph.nodes())
        simulation = _remap_node_ids(sim_parsed, graph_nodes)
        rm = _remap_node_ids(rm_parsed, graph_nodes)
    except ImportError:
        rm = extract_rm_scores_dict(rm_raw) if rm_raw else {}
        simulation = extract_simulation_dict(sim_raw)

    edge_simulation = extract_edge_simulation_dict(edge_raw) if edge_raw else {}
    conv = networkx_to_hetero_data(
        graph, structural, simulation, rm,
        edge_simulation_results=edge_simulation or None,
    )

    bundle = ScenarioBundle(
        scenario_id=scenario_id,
        graph=graph,
        structural=structural,
        rm=rm,
        simulation=simulation,
        hetero_data=conv.hetero_data,
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        n_labelled=conv.num_labelled_nodes,
        label_stability=sim_raw.get("label_stability", {}) if isinstance(sim_raw, dict) else {},
        labeler=sim_raw.get("labeler", "") if isinstance(sim_raw, dict) else "",
        edge_simulation=edge_simulation,
    )
    logger.info(
        "  [%s] %d nodes, %d edges, %d labelled%s",
        scenario_id, bundle.n_nodes, bundle.n_edges, bundle.n_labelled,
        "" if rm else "  (rm missing)",
    )
    return bundle


def discover_scenarios(
    cache_dir: Path, skip: List[str], min_scenarios: int = 2
) -> List[ScenarioBundle]:
    """Walk cache_dir and load all valid scenario bundles.

    ``min_scenarios`` defaults to 2 because LOSO (this module's own use) is
    undefined on fewer — there is no "other scenario" to train on. K-fold
    evaluation (``cli/kfold_evaluate.py``) trains and tests within one
    scenario's own graph and has no such requirement; it passes
    ``min_scenarios=1``.
    """
    bundles: List[ScenarioBundle] = []
    for sub in sorted(p for p in cache_dir.iterdir() if p.is_dir()):
        if any(s in sub.name for s in skip):
            logger.info("  Skipping %s (matches --skip filter)", sub.name)
            continue
        b = load_scenario_bundle(sub)
        if b is not None and b.n_labelled >= 3:
            bundles.append(b)

    if len(bundles) < min_scenarios:
        raise ValueError(
            f"Need >= {min_scenarios} scenario(s); found {len(bundles)} usable in {cache_dir}."
        )
    return bundles


# ──────────────────────────────────────────────────────────────────────────────
# Inductive metric computation
# ──────────────────────────────────────────────────────────────────────────────

#: Canonical implementation now lives in ``saag.evaluation.metrics`` so the
#: in-distribution table (``reproduce/main_table.py``), this LOSO harness and
#: ``cli/kfold_evaluate.py`` cannot drift apart. Re-exported here unchanged so
#: existing callers and the emitted CSV columns keep working.
#:
#: The node population is selected per run via ``--eval-population`` and defaults
#: to ``"application"``, matching ``reproduce/main_table.py`` so the LOSO table and
#: the in-distribution table are scored on the same node type. ``"labeled"`` (every
#: node the cache carries a label for) is the historical behaviour of this file and
#: remains available, but it pools node types with different scales and base rates,
#: which inverts the sign of weakly-predictive variants (see the Simpson's-paradox
#: note in the manuscript's Conclusion Validity discussion).
compute_inductive_metrics = _shared_inductive_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Single-fold execution
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_bundle_graph(
    bundle: ScenarioBundle, use_qos: bool
) -> Tuple[Any, Dict[str, Any]]:
    """Return ``(graph, structural_metrics)`` with QoS masked out when disabled.

    The unweighted variants must never see QoS through *any* door — the node
    features, the edge features, or the structural metrics — so the masking is
    applied once here rather than repeated at each of the four call sites that
    used to inline it.
    """
    if use_qos:
        return bundle.graph, bundle.structural
    from reproduce.main_table import _mask_qos_in_graph, _mask_qos_in_structural

    return _mask_qos_in_graph(bundle.graph), _mask_qos_in_structural(bundle.structural)


def _build_training_hetero(
    bundle: ScenarioBundle, use_qos: bool, rank_normalize_features: bool
) -> HeteroData:
    """HeteroData for one training scenario, splits left to the caller."""
    graph, sm = _prepare_bundle_graph(bundle, use_qos)
    return networkx_to_hetero_data(
        graph, sm, bundle.simulation, bundle.rm,
        qos_enabled=use_qos,
        rank_normalize_features=rank_normalize_features,
        edge_simulation_results=bundle.edge_simulation or None,
    ).hetero_data


def _build_validation_hetero(
    bundle: ScenarioBundle, use_qos: bool, rank_normalize_features: bool
) -> HeteroData:
    """HeteroData for the inner validation scenario.

    Every labelled node goes in ``val_mask`` and nothing goes in ``train_mask``
    or ``test_mask``: this graph is scored, never fitted. Using the whole
    labelled population (rather than a 20% slice of it) makes the selection
    signal as stable as the scenario allows, which is the point of moving
    selection off the training distribution in the first place.
    """
    from saag.prediction.data_preparation import _labelled_index_mask

    data = _build_training_hetero(bundle, use_qos, rank_normalize_features)
    for store in data.node_stores:
        n = store.num_nodes
        if hasattr(store, "y") and store.y.numel() > 0:
            labelled = torch.from_numpy(_labelled_index_mask(store))
        else:
            labelled = torch.zeros(n, dtype=torch.bool)
        store.train_mask = torch.zeros(n, dtype=torch.bool)
        store.val_mask = labelled
        store.test_mask = torch.zeros(n, dtype=torch.bool)
    return data


def _select_val_bundle(
    inductives: List[ScenarioBundle], inner_val: str
) -> Optional[ScenarioBundle]:
    """Pick the inner validation scenario, or None for the legacy behaviour.

    Deterministic and independent of the holdout's identity: the median-sized
    inductive scenario, ties broken by scenario id. Picking the largest would
    hand validation to whichever scenario also dominates the training signal;
    picking the smallest would validate on the noisiest ρ available.
    """
    if inner_val != "auto" or not inductives:
        return None
    ordered = sorted(inductives, key=lambda b: (b.n_nodes, b.scenario_id))
    return ordered[len(ordered) // 2]


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
    variant: str = "hgl_qos",
    eval_population: str = "application",
    auto_layers: bool = True,
    weight_decay: float = 1e-4,
    warmup_T0: Optional[int] = None,
    multitask_weight: float = 0.5,
    rm_consistency_weight: float = 0.0,
    ranking_weight: float = 0.3,
    pairwise_ranking_weight: float = 0.1,
    inner_val: str = "none",
    rank_normalize_features: bool = False,
    rank_normalize_labels: bool = False,
    device: Optional[str] = "auto",
    resume: bool = False,
    cache_dir: Optional[Path] = None,
) -> FoldResult:
    """
    One LOSO fold: train on N-1 scenarios with multi-seed, predict on held-out.

    Defensive invariants:
      - holdout never appears in train_ids
      - holdout never appears in the inner validation set
      - holdout's structural/rm are passed at predict() time (needed for features)
      - holdout's simulation is passed only for evaluation, never for training
    """
    # Pre-flight, before any work: the dispatch below sits inside a per-seed
    # try/except that logs and continues, so an unknown id would otherwise show
    # up as twelve folds of nan rather than as an error.
    if variant not in KNOWN_VARIANTS:
        raise ValueError(
            f"unrecognised variant {variant!r}; expected one of "
            f"{sorted(KNOWN_VARIANTS)}"
        )

    if device == "cuda" or (device in ("auto", None) and torch.cuda.is_available()):
        target_device = torch.device("cuda")
    else:
        target_device = torch.device("cpu")
    target_device = _resolve_device(device)
    plan = _plan_fold(bundles, holdout_idx, layers, auto_layers, inner_val, workdir)
    cfg = _seed_cfg(
        layer=layer, epochs=epochs, lr=lr, hidden=hidden, heads=heads,
        layers=layers, dropout=dropout, mode=mode, variant=variant,
        eval_population=eval_population, weight_decay=weight_decay,
        warmup_T0=warmup_T0, multitask_weight=multitask_weight,
        rm_consistency_weight=rm_consistency_weight, ranking_weight=ranking_weight,
        pairwise_ranking_weight=pairwise_ranking_weight,
        rank_normalize_features=rank_normalize_features,
        rank_normalize_labels=rank_normalize_labels,
    )

    seed_metrics: List[Dict[str, Any]] = []
    for seed in seeds:
        m = _seed_metrics(
            plan, seed, cfg, target_device,
            resume=resume, cache_dir=cache_dir, done=seed_metrics,
        )
        if m is not None:
            seed_metrics.append(m)

    return _aggregate_fold(plan, seed_metrics)


@dataclass
class _FoldPlan:
    """Everything about a fold that does not depend on the seed.

    Split out of ``run_one_fold`` so a (fold, seed) fit is addressable on its
    own: it is the unit ``--jobs`` dispatches, the unit ``--resume`` reuses, and
    the unit whose failure aborts a sweep early.
    """

    holdout: ScenarioBundle
    train_set: List[ScenarioBundle]
    train_ids: List[str]
    primary: ScenarioBundle
    inductives: List[ScenarioBundle]
    val_bundle: Optional[ScenarioBundle]
    effective_layers: int
    fold_dir: Path


class SeedFailed(RuntimeError):
    """One (fold, seed) fit raised. Carries the message for fail-fast triage."""


def _resolve_device(device: Optional[str]) -> torch.device:
    if device == "cuda" or (device in ("auto", None) and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def _plan_fold(
    bundles: List[ScenarioBundle],
    holdout_idx: int,
    layers: int,
    auto_layers: bool,
    inner_val: str,
    workdir: Path,
) -> _FoldPlan:
    """Resolve holdout/primary/inductive/val membership and depth for one fold."""
    holdout = bundles[holdout_idx]
    train_set = [b for i, b in enumerate(bundles) if i != holdout_idx]
    train_ids = [b.scenario_id for b in train_set]

    assert holdout.scenario_id not in train_ids, (
        f"G4 leakage violation: holdout {holdout.scenario_id} found in train ids"
    )

    # Pick the largest non-holdout as the primary (longest val masks → stable early stop)
    primary = max(train_set, key=lambda b: b.n_nodes)
    inductives = [b for b in train_set if b.scenario_id != primary.scenario_id]

    # Inner validation scenario, when enabled, is held out of the training
    # loader as well — otherwise early stopping would be selecting on a graph
    # the loss is already fitting.
    val_bundle = _select_val_bundle(inductives, inner_val)
    if val_bundle is not None:
        inductives = [b for b in inductives if b.scenario_id != val_bundle.scenario_id]
        assert val_bundle.scenario_id != holdout.scenario_id, (
            f"G4 leakage violation: holdout {holdout.scenario_id} selected as "
            "inner validation scenario"
        )

    logger.info(
        "Fold[holdout=%s]  primary=%s (|V|=%d)  inductive=%d scenarios  inner_val=%s",
        holdout.scenario_id, primary.scenario_id, primary.n_nodes, len(inductives),
        val_bundle.scenario_id if val_bundle else "(none)",
    )

    # Depth is a property of the fold, not of the seed. Computed once here so it
    # can be reported: with --auto-layers on, it is derived from the primary
    # scenario's size, which changes with the holdout.
    if auto_layers:
        effective_layers = 1 if primary.n_nodes <= 200 else (2 if primary.n_nodes <= 500 else layers)
        if effective_layers != layers:
            logger.info(
                "  [auto-layers] primary.n_nodes=%d -> downgrading layers %d -> %d "
                "(disable with --no-auto-layers)",
                primary.n_nodes, layers, effective_layers,
            )
    else:
        effective_layers = layers

    fold_dir = workdir / f"fold_{holdout.scenario_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    return _FoldPlan(
        holdout=holdout, train_set=train_set, train_ids=train_ids,
        primary=primary, inductives=inductives, val_bundle=val_bundle,
        effective_layers=effective_layers, fold_dir=fold_dir,
    )



#: Keys of one fit's configuration. Collected in one place so the fingerprint and
#: the dispatch cannot drift: anything that changes what a fit produces has to
#: appear here, or a stale shard is silently reusable.
_SEED_CFG_KEYS = (
    "layer", "epochs", "lr", "hidden", "heads", "layers", "dropout", "mode",
    "variant", "eval_population", "weight_decay", "warmup_T0",
    "multitask_weight", "rm_consistency_weight", "ranking_weight",
    "pairwise_ranking_weight", "rank_normalize_features", "rank_normalize_labels",
)


def _seed_cfg(**kwargs) -> Dict[str, Any]:
    missing = set(_SEED_CFG_KEYS) - set(kwargs)
    if missing:
        raise TypeError(f"_seed_cfg missing {sorted(missing)}")
    return {k: kwargs[k] for k in _SEED_CFG_KEYS}


def _seed_fingerprint(
    plan: _FoldPlan, seed: int, cfg: Dict[str, Any],
    target_device: torch.device, cache_dir: Optional[Path],
) -> str:
    """Content hash of everything this fit depends on.

    Fold membership is included explicitly rather than implied by the holdout id:
    dropping a scenario with ``--skip`` changes the training set of every
    remaining fold without changing any holdout's name.
    """
    return fit_fingerprint(
        {
            **cfg,
            "seed": seed,
            "holdout": plan.holdout.scenario_id,
            "primary": plan.primary.scenario_id,
            "inductives": [b.scenario_id for b in plan.inductives],
            "val": plan.val_bundle.scenario_id if plan.val_bundle else None,
            "effective_layers": plan.effective_layers,
            # A CPU fit and a CUDA fit of the same configuration are not
            # interchangeable rows; reproduce/loso_all_variants.py already
            # records the device for exactly this reason.
            "device": target_device.type,
        },
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )


def _replicate_structural(done: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Copy a training-free variant's result onto another seed.

    ``topo_baseline``/``topo_qos`` score a fixed graph with a deterministic
    centrality: every seed recomputes the identical numbers. Replicating rather
    than recomputing keeps the reported ``n_seeds_per_fold`` and the (zero)
    across-seed std exactly as they were, while doing the work once.
    """
    m = {k: (dict(v) if isinstance(v, dict) else v) for k, v in done.items()}
    m["_full_scores"] = {k: dict(v) for k, v in done["_full_scores"].items()}
    m["seed"] = seed
    return m


def _seed_metrics(
    plan: _FoldPlan,
    seed: int,
    cfg: Dict[str, Any],
    target_device: torch.device,
    resume: bool = False,
    cache_dir: Optional[Path] = None,
    done: Optional[List[Dict[str, Any]]] = None,
    errors: Optional[List[str]] = None,
    info: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """One fit, reusing a stored one when its fingerprint still matches.

    Returns ``None`` when the fit failed, preserving ``run_one_fold``'s
    log-and-continue behaviour; the caller decides whether a run of failures is
    worth aborting for.
    """
    if cfg["variant"] in _STRUCTURAL_VARIANTS and done:
        return _replicate_structural(done[0], seed)

    seed_dir = plan.fold_dir / f"seed_{seed}"
    shard = seed_dir / "seed_result.json"
    fingerprint = _seed_fingerprint(plan, seed, cfg, target_device, cache_dir)

    if resume and shard.exists():
        try:
            payload = json.loads(shard.read_text())
        except json.JSONDecodeError:
            payload = {}
        if payload.get("fingerprint") == fingerprint:
            logger.info("  ── seed %d ── reusing stored fit (fingerprint match)", seed)
            if info is not None:
                info["reused"] = True
            return payload["metrics"]
        logger.info(
            "  ── seed %d ── stored fit does not match this configuration or the "
            "current model code; re-running", seed,
        )

    # Anything left in the seed directory is from a fit this one is replacing.
    # It must go before training starts: both learned branches restore from a
    # `best_model.pt` found here and skip training entirely, which is how a
    # dirty workspace once reported hgl LOSO rho = -0.576 in 3.2 s.
    if seed_dir.exists():
        shutil.rmtree(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    try:
        m = _run_seed(plan, seed, cfg, target_device)
    except SeedFailed as exc:
        if errors is not None:
            errors.append(str(exc))
        return None

    shard.write_text(json.dumps({"fingerprint": fingerprint, "metrics": m}))
    return m


def _run_seed(
    plan: _FoldPlan,
    seed: int,
    cfg: Dict[str, Any],
    target_device: torch.device,
) -> Dict[str, Any]:
    """Train and score one (fold, seed) fit. Raises :class:`SeedFailed`.

    The body is the former inner block of ``run_one_fold``'s seed loop, moved
    verbatim so that the parallel and serial paths cannot diverge.
    """
    holdout = plan.holdout
    primary = plan.primary
    inductives = plan.inductives
    val_bundle = plan.val_bundle
    effective_layers = plan.effective_layers
    fold_dir = plan.fold_dir

    variant = cfg["variant"]
    layer = cfg["layer"]
    epochs = cfg["epochs"]
    lr = cfg["lr"]
    hidden = cfg["hidden"]
    heads = cfg["heads"]
    layers = cfg["layers"]
    dropout = cfg["dropout"]
    mode = cfg["mode"]
    eval_population = cfg["eval_population"]
    weight_decay = cfg["weight_decay"]
    warmup_T0 = cfg["warmup_T0"]
    multitask_weight = cfg["multitask_weight"]
    rm_consistency_weight = cfg["rm_consistency_weight"]
    ranking_weight = cfg["ranking_weight"]
    pairwise_ranking_weight = cfg["pairwise_ranking_weight"]
    rank_normalize_features = cfg["rank_normalize_features"]
    rank_normalize_labels = cfg["rank_normalize_labels"]

    logger.info("  ── seed %d ──", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_dir = fold_dir / f"seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    try:
        if variant in _STRUCTURAL_VARIANTS:
            # Training-free structural centrality. It has no notion of a
            # train set, so its held-out score is simply its score — which
            # is exactly why it belongs in the LOSO table: an out-of-domain
            # comparison a model must beat to justify being trained at all.
            # Omitting it (as the published Table 4 did) leaves the strongest
            # non-learning competitor unmeasured under the harder protocol.
            from reproduce.main_table import (
                _compute_topo_baseline_scores, _load_scenario_data,
            )

            # Score on the DEPENDS_ON projection, not the native graph.
            # Application nodes never route messages, so their betweenness
            # on the raw pub-sub graph is identically 0 — the baseline would
            # emit a constant for the entire Application stratum and its
            # pooled rho would be carried purely by between-type offsets.
            # This is the same substrate the in-distribution table gives it.
            try:
                proj_graph, proj_struct, _sim, _rm, _gt = _load_scenario_data(
                    holdout.scenario_id, substrate="projection"
                )
            except Exception as exc:      # noqa: BLE001 - fall back to native
                logger.warning("  %s: projection unavailable (%s); using native graph", variant, exc)
                proj_graph, proj_struct = holdout.graph, holdout.structural

            struct_pred = _compute_topo_baseline_scores(
                proj_graph, proj_struct,
                use_qos=(variant == "topo_qos"),
            )
            if not struct_pred:
                logger.warning("  %s: no structural signal on holdout; skipping seed", variant)
                raise SeedFailed("no structural signal on holdout")
            pred_scores = {str(k): float(v) for k, v in struct_pred.items()}
            full_node_scores = {
                k: {"overall": v, "reliability": v, "maintainability": v}
                for k, v in pred_scores.items()
            }

        elif variant in _HOMOGENEOUS_VARIANTS:
            # Baseline variants use GNNTrainer directly
            from saag.prediction.models.baselines import build_baseline
            from saag.prediction.data_preparation import create_node_splits
            from saag.prediction.trainer import GNNTrainer, evaluate

            # Any arm with an edge channel needs QoS on the graph; the
            # registry owns which those are.
            use_qos = _registry.edge_dim(variant, "loso") is not None
            train_graph, train_sm = _prepare_bundle_graph(primary, use_qos)
            holdout_graph, holdout_sm = _prepare_bundle_graph(holdout, use_qos)

            conv = networkx_to_hetero_data(
                train_graph, train_sm, primary.simulation, primary.rm,
                qos_enabled=use_qos,
                rank_normalize_features=rank_normalize_features,
            )
            data = conv.hetero_data
            create_node_splits(data, seed=seed)

            # Training-set parity with the HGT branch below. This branch
            # used to train on `primary` alone while the HGT branch trained
            # on primary + every inductive scenario, so the published
            # typed-vs-untyped LOSO margin compared a model with N-1
            # training graphs against one with a single graph. Same folds,
            # same substrate, same graph count.
            inductive_data = [
                _build_training_hetero(b, use_qos, rank_normalize_features)
                for b in inductives
            ]
            for ig in inductive_data:
                create_node_splits(ig, seed=seed)

            # Multi-graph training also requires the label normalization the
            # GNNService path has always applied and this branch never did:
            # without it each scenario's labels reach the same scale-sensitive
            # loss terms on its own raw scale.
            normalize_labels_robust(data, rank_normalize=rank_normalize_labels)
            for ig in inductive_data:
                normalize_labels_robust(ig, rank_normalize=rank_normalize_labels)

            val_data = None
            if val_bundle is not None:
                val_data = _build_validation_hetero(
                    val_bundle, use_qos, rank_normalize_features
                )
                normalize_labels_robust(val_data, rank_normalize=rank_normalize_labels)

            if inductive_data:
                from torch_geometric.loader import DataLoader as _PyGDataLoader
                training_input = _PyGDataLoader(
                    [data] + inductive_data, batch_size=1, shuffle=True
                )
            else:
                training_input = data

            # Width and edge-channel width come from the registry: they are
            # identity for every reported variant and differ only for the
            # RQ2 capacity / edge-channel controls.
            edge_dim = _registry.edge_dim(variant, "loso")
            baseline_name = "homo_unweighted" if edge_dim is None else "homo_scalar"
            model = build_baseline(baseline_name,
                                   hidden_channels=_registry.hidden_for(variant, hidden, "loso"),
                                   num_heads=heads,
                                   num_layers=layers, dropout=dropout,
                                   edge_dim=edge_dim)
            model.to(target_device)
            best_path = ckpt_dir / "best_model.pt"
            if best_path.exists():
                logger.info("  Found baseline checkpoint %s. Skipping training.", best_path)
                model.load_state_dict(torch.load(best_path, map_location=target_device))
            else:
                trainer = GNNTrainer(model=model, checkpoint_dir=str(ckpt_dir),
                                     lr=lr, num_epochs=epochs, patience=min(60, epochs),
                                     weight_decay=weight_decay, warmup_T0=warmup_T0,
                                     multitask_weight=multitask_weight,
                                     rm_consistency_weight=rm_consistency_weight,
                                     ranking_weight=ranking_weight,
                                     pairwise_ranking_weight=pairwise_ranking_weight,
                                     # Also parity: without the labeler's
                                     # dimension mask the unmeasured
                                     # maintainability head is regressed
                                     # toward a fabricated zero, which the
                                     # HGT branch has never done.
                                     dimension_mask=conv.dimension_mask)
                trainer.train(
                    training_input,
                    primary_data=data if inductive_data else None,
                    val_data=val_data,
                )

            # Evaluate on holdout
            conv_h = networkx_to_hetero_data(
                holdout_graph, holdout_sm, holdout.simulation, holdout.rm,
                qos_enabled=use_qos,
                rank_normalize_features=rank_normalize_features,
            )
            data_h = conv_h.hetero_data
            create_node_splits(data_h, seed=seed)
            metrics = evaluate(model, data_h, "test_mask", target_device)

            # Build pred_scores from model output for inductive metrics
            model.eval()
            data_h_dev = data_h.to(target_device)
            with torch.no_grad():
                x_h = {nt: data_h_dev[nt].x for nt in data_h_dev.node_types if hasattr(data_h_dev[nt], "x")}
                ei_h = {r: data_h_dev[r].edge_index for r in data_h_dev.edge_types}
                ea_h = {r: data_h_dev[r].edge_attr for r in data_h_dev.edge_types if hasattr(data_h_dev[r], "edge_attr")}
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
                        }

        elif variant in _HGT_VARIANTS:
            # hgl_qos (default), hgl, hgl_qos_uni or topology_rm → GNNService
            effective_mode = "rm" if variant == "topology_rm" else mode
            use_qos = variant in ("hgl_qos", "hgl_qos_uni")
            train_graph, train_sm = _prepare_bundle_graph(primary, use_qos)
            holdout_graph, holdout_sm = _prepare_bundle_graph(holdout, use_qos)

            best_path = ckpt_dir / "best_model.pt"
            if best_path.exists():
                logger.info("  Found GNN checkpoint %s. Skipping training.", best_path)
                service = GNNService.from_checkpoint(
                    str(ckpt_dir),
                    graph=train_graph,
                    layer=layer,
                    device=target_device,
                )
            else:
                service = GNNService(
                    checkpoint_dir=str(ckpt_dir),
                    hidden_channels=hidden,
                    num_heads=heads,
                    num_layers=effective_layers,
                    dropout=dropout,
                    predict_edges=False,
                    device=target_device,
                    # Variant-derived, unlike --qos-injection, which is a
                    # cross-cutting CLI ablation. The directionality control
                    # is the only arm that turns this off.
                    use_bidirectional=_registry.bidirectional_for(variant),
                )
                service.train(
                    graph=train_graph,
                    structural_metrics=train_sm,
                    simulation_results=primary.simulation,
                    edge_simulation_results=primary.edge_simulation or None,
                    rm_scores=primary.rm,
                    inductive_graphs=[
                        _build_training_hetero(b, use_qos, rank_normalize_features)
                        for b in inductives
                    ],
                    val_graph=(
                        _build_validation_hetero(
                            val_bundle, use_qos, rank_normalize_features
                        )
                        if val_bundle is not None else None
                    ),
                    seeds=[seed],
                    num_epochs=1 if variant == "topology_rm" else epochs,
                    lr=lr,
                    patience=min(60, epochs),
                    layer=layer,
                    qos_enabled=use_qos,
                    weight_decay=weight_decay,
                    warmup_T0=warmup_T0,
                    multitask_weight=multitask_weight,
                    rm_consistency_weight=rm_consistency_weight,
                    ranking_weight=ranking_weight,
                    pairwise_ranking_weight=pairwise_ranking_weight,
                    rank_normalize_features=rank_normalize_features,
                    rank_normalize_labels=rank_normalize_labels,
                )
            result = service.predict(
                graph=holdout_graph,
                structural_metrics=holdout_sm,
                rm_scores=holdout.rm,
                # GNNService.train() names this simulation_results, predict() names
                # it eval_labels. Passing the train() spelling here raised TypeError
                # inside the per-seed try/except, so every HGT/HGT-QoS seed was
                # skipped and the fold aggregated to nan.
                eval_labels=holdout.simulation,
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
                }
                for nid, ns in result.node_scores.items()
            }

        else:
            # Unreachable via the CLI (argparse `choices` is checked against
            # KNOWN_VARIANTS by tests/test_variant_dispatch.py) and via the
            # pre-flight check at the top of run_one_fold. Kept so a direct
            # programmatic call cannot silently get the HGT branch.
            raise ValueError(
                f"unrecognised variant {variant!r}; expected one of "
                f"{sorted(KNOWN_VARIANTS)}"
            )

    except SeedFailed:
        # Already reported by the branch that raised it (a structural variant
        # with no signal on this holdout); not an unexpected fault.
        raise
    except Exception as e:
        logger.error("  Fold seed %d failed: %s", seed, e, exc_info=True)
        raise SeedFailed(str(e)) from e

    true_impact = {nid: float(d.get("composite", 0.0)) for nid, d in holdout.simulation.items()}

    m = compute_inductive_metrics(
        pred_scores, true_impact, holdout.graph, population=eval_population,
    )
    m["seed"] = seed
    m["prediction_mode"] = mode
    m["variant"] = variant
    m["_full_scores"] = full_node_scores  # temporary storage for aggregation


    logger.info(
        "    ρ=%.4f  F1=%.4f  NDCG=%.4f  RMSE=%.4f  (n=%d, mode=%s)",
        m["spearman_rho"], m["f1_at_k"], m["ndcg_10"], m["rmse"], m["n"],
        m["prediction_mode"],
    )

    return m


def _aggregate_fold(plan: _FoldPlan, seed_metrics: List[Dict[str, Any]]) -> FoldResult:
    """Collapse a fold's per-seed metrics into its FoldResult."""
    holdout = plan.holdout
    train_ids = plan.train_ids
    primary = plan.primary
    effective_layers = plan.effective_layers
    val_bundle = plan.val_bundle

    # Aggregate node scores across seeds
    all_nodes = set()
    for m in seed_metrics:
        all_nodes.update(m["_full_scores"].keys())
    
    score_keys = ["overall", "reliability", "maintainability"]
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

    # Metrics added alongside the originals. Aggregated the same way, and
    # tolerant of seed dicts written before these keys existed.
    def _agg(key: str) -> List[float]:
        # ``None`` is skipped as well as NaN: the active-stratum rho is None when
        # a seed's holdout has fewer than three positive-impact components, which
        # is an undefined statistic rather than a zero one (see metrics.py's
        # "absent is not zero" doctrine). np.isnan raises on None, so the None
        # test has to come first.
        out: List[float] = []
        for m in seed_metrics:
            v = m.get(key)
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            out.append(v)
        return out

    added_keys = [
        "precision_at_tau", "recall_at_tau", "f1_at_tau", "pr_auc",
        "rmse_scaled", "mae_scaled", "n_true_critical",
        "n_predicted", "n_labeled", "n_evaluated",
        # Active-stratum ranking: rho restricted to components the oracle scores
        # strictly positive, plus that stratum's size. Reported alongside the
        # full-population rho because I*(v) is heavily zero-inflated.
        "spearman_rho_positive", "n_positive",
    ]
    added_mean = {k: float(np.mean(v)) for k in added_keys if (v := _agg(k))}
    added_std = {k: float(np.std(v)) for k in added_keys if (v := _agg(k))}

    # Undefined strata (Topic and Node carry no ground truth at all) must stay
    # undefined through aggregation rather than averaging in as 0.0.
    per_type_summary = aggregate_per_type(
        [m.get("per_type_rho", {}) for m in seed_metrics], value_key="rho"
    )

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
            **added_mean,
        },
        std_metrics={
            "spearman_rho": float(np.std(rho_vals)),
            "f1_at_k": float(np.std(f1_vals)),
            "ndcg_10": float(np.std(ndcg_vals)),
            "rmse": float(np.std(rmse_vals)),
            **added_std,
        },
        per_type_rho=per_type_summary,
        node_predictions=node_agg,
        effective_layers=effective_layers,
        val_scenario_id=val_bundle.scenario_id if val_bundle else None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Full LOSO orchestration
# ──────────────────────────────────────────────────────────────────────────────

#: How many consecutive failed fits before a sweep is abandoned. A systematic
#: fault — a device mismatch, a missing artefact, a bad flag — fails every fit
#: identically, and the per-seed handler's log-and-continue turns that into
#: hours of wasted training before anything is raised. One fold's worth of
#: identical failures is already conclusive.
_FAIL_FAST_STREAK = 5


class SweepAborted(RuntimeError):
    """Raised when a run of fits fails the same way, instead of training on."""


class _FailFast:
    """Counts consecutive failures and trips once they look systematic."""

    def __init__(self, streak: int = _FAIL_FAST_STREAK):
        self.streak = streak
        self.count = 0
        self.last = ""

    def record(self, error: Optional[str]) -> None:
        if error is None:
            self.count = 0
            return
        self.count += 1
        self.last = error
        if self.count >= self.streak:
            raise SweepAborted(
                f"{self.count} consecutive fits failed with the same fault; "
                f"abandoning the sweep rather than training on. Last error: {self.last}"
            )


def _preflight(plan: _FoldPlan, cfg: Dict[str, Any], target_device: torch.device,
               workdir: Path) -> None:
    """Run one throwaway one-epoch fit before committing to the sweep.

    A sweep is hours of work whose first fault may only surface at the end of a
    fit. This exercises the whole path — conversion, model construction,
    training step, inference, metric computation — for a few seconds, on this
    device, so an environment fault is reported before the sweep rather than
    after it.
    """
    probe_dir = workdir / ".preflight"
    shutil.rmtree(probe_dir, ignore_errors=True)
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_plan = _FoldPlan(**{**plan.__dict__, "fold_dir": probe_dir})
    probe_cfg = {**cfg, "epochs": 1}
    t0 = time.time()
    try:
        _run_seed(probe_plan, 42, probe_cfg, target_device)
    except SeedFailed as exc:
        raise SweepAborted(
            f"pre-flight fit failed on device '{target_device.type}' before the "
            f"sweep started: {exc}. Nothing was trained. Re-run with "
            f"--no-preflight to skip this check."
        ) from exc
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)
    logger.info("Pre-flight fit OK (%s, 1 epoch, %.1fs).", target_device.type, time.time() - t0)


def _run_folds_serial(
    plans: List[_FoldPlan], seeds: List[int], cfg: Dict[str, Any],
    target_device: torch.device, resume: bool, cache_dir: Optional[Path],
) -> Dict[str, List[Dict[str, Any]]]:
    guard = _FailFast()
    out: Dict[str, List[Dict[str, Any]]] = {}
    for i, plan in enumerate(plans):
        logger.info("════════════════════════════════════════════════════════════")
        logger.info("LOSO fold %d / %d   holdout = %s",
                    i + 1, len(plans), plan.holdout.scenario_id)
        logger.info("════════════════════════════════════════════════════════════")
        collected: List[Dict[str, Any]] = []
        for seed in seeds:
            errors: List[str] = []
            m = _seed_metrics(plan, seed, cfg, target_device, resume=resume,
                              cache_dir=cache_dir, done=collected, errors=errors)
            guard.record(None if m is not None else (errors[-1] if errors else "unknown"))
            if m is not None:
                collected.append(m)
        out[plan.holdout.scenario_id] = collected
    return out


#: Per-worker state. Populated once per process by _worker_init so the corpus is
#: parsed once per worker rather than pickled once per fit.
_WORKER_STATE: Dict[str, Any] = {}


def _worker_init(cache_dir: str, skip: List[str], expected_ids: List[str],
                 torch_threads: int) -> None:
    torch.set_num_threads(max(1, torch_threads))
    logging.getLogger().setLevel(logging.WARNING)
    bundles = discover_scenarios(Path(cache_dir), skip=skip)
    got = [b.scenario_id for b in bundles]
    if got != list(expected_ids):
        # Fold indices are positional; a worker that enumerated the corpus
        # differently would train on one fold and report it as another.
        raise RuntimeError(
            f"worker corpus differs from the parent's: {got} != {list(expected_ids)}"
        )
    _WORKER_STATE["bundles"] = bundles


def _worker_seed(job: Tuple) -> Tuple[int, int, Optional[Dict[str, Any]], Optional[str], bool]:
    (k, seed, cfg, workdir, layers, auto_layers, inner_val,
     device_type, resume, cache_dir) = job
    plan = _plan_fold(_WORKER_STATE["bundles"], k, layers, auto_layers,
                      inner_val, Path(workdir))
    errors: List[str] = []
    info: Dict[str, Any] = {}
    m = _seed_metrics(
        plan, seed, cfg, torch.device(device_type), resume=resume,
        cache_dir=Path(cache_dir) if cache_dir else None, errors=errors, info=info,
    )
    return k, seed, m, (errors[-1] if errors else None), bool(info.get("reused"))


def _run_folds_parallel(
    plans: List[_FoldPlan], seeds: List[int], cfg: Dict[str, Any],
    target_device: torch.device, workdir: Path, jobs: int, resume: bool,
    cache_dir: Optional[Path], skip: List[str], torch_threads: int,
    expected_ids: List[str], auto_layers: bool, inner_val: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Dispatch every (fold, seed) fit to a process pool.

    The unit is the fit, not the fold: folds differ in cost by roughly an order
    of magnitude (the primary graph changes with the holdout), so a fold-level
    pool would spend its tail waiting on a single worker.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if cache_dir is None:
        raise ValueError("--jobs > 1 needs --cache-dir so each worker can load the corpus")

    queue = [
        (k, seed, cfg, str(workdir), cfg["layers"], auto_layers, inner_val,
         target_device.type, resume, str(cache_dir))
        for k in range(len(plans)) for seed in seeds
    ]
    logger.info("Dispatching %d fits (%d folds x %d seeds) across %d workers.",
                len(queue), len(plans), len(seeds), jobs)

    out: Dict[str, List[Dict[str, Any]]] = {p.holdout.scenario_id: [] for p in plans}
    by_fold: Dict[int, Dict[int, Dict[str, Any]]] = {k: {} for k in range(len(plans))}
    guard = _FailFast()
    done_count = 0

    with ProcessPoolExecutor(
        max_workers=jobs,
        initializer=_worker_init,
        initargs=(str(cache_dir), list(skip), list(expected_ids), torch_threads),
    ) as pool:
        futures = {pool.submit(_worker_seed, job): job for job in queue}
        try:
            for fut in as_completed(futures):
                k, seed, m, err, reused = fut.result()
                done_count += 1
                if m is not None:
                    by_fold[k][seed] = m
                if m is None:
                    outcome = f"FAILED ({err})"
                else:
                    outcome = f"rho={m['spearman_rho']:.4f}"
                    if reused:
                        outcome += "  (reused)"
                logger.info("  [%d/%d] fold=%s seed=%d %s", done_count, len(queue),
                            plans[k].holdout.scenario_id, seed, outcome)
                guard.record(None if m is not None else (err or "unknown"))
        except SweepAborted:
            for fut in futures:
                fut.cancel()
            raise

    # Seed order is the caller's, not completion order: the per-seed lists are
    # reported verbatim and an aggregate over a shuffled list would differ from
    # the serial path's in its `seed_metrics` ordering.
    for k, plan in enumerate(plans):
        out[plan.holdout.scenario_id] = [by_fold[k][s] for s in seeds if s in by_fold[k]]
    return out


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
    variant: str = "hgl_qos",
    auto_layers: bool = True,
    weight_decay: float = 1e-4,
    warmup_T0: Optional[int] = None,
    multitask_weight: float = 0.5,
    rm_consistency_weight: float = 0.0,
    ranking_weight: float = 0.3,
    pairwise_ranking_weight: float = 0.1,
    eval_population: str = "application",
    inner_val: str = "none",
    rank_normalize_features: bool = False,
    rank_normalize_labels: bool = False,
    device: Optional[str] = "auto",
    jobs: int = 1,
    resume: bool = False,
    cache_dir: Optional[Path] = None,
    skip: Optional[List[str]] = None,
    torch_threads: int = 1,
    preflight: bool = True,
) -> LOSOReport:
    """Run leave-one-scenario-out across all loaded bundles.

    ``jobs`` > 1 dispatches (fold, seed) fits to a process pool. Each fit seeds
    torch and numpy from its own seed and writes to its own directory, so the
    result does not depend on how many workers ran it — only the wall-clock does.
    """
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

    cfg = _seed_cfg(
        layer=layer, epochs=epochs, lr=lr, hidden=hidden, heads=heads,
        layers=layers, dropout=dropout, mode=mode, variant=variant,
        eval_population=eval_population, weight_decay=weight_decay,
        warmup_T0=warmup_T0, multitask_weight=multitask_weight,
        rm_consistency_weight=rm_consistency_weight, ranking_weight=ranking_weight,
        pairwise_ranking_weight=pairwise_ranking_weight,
        rank_normalize_features=rank_normalize_features,
        rank_normalize_labels=rank_normalize_labels,
    )
    target_device = _resolve_device(device)
    plans = [
        _plan_fold(bundles, k, layers, auto_layers, inner_val, workdir)
        for k in range(len(bundles))
    ]

    if preflight:
        _preflight(plans[0], cfg, target_device, workdir)

    # Training-free variants recompute the same deterministic numbers for every
    # seed, so the pool would buy nothing and the seed-replication shortcut in
    # _seed_metrics needs the seeds of a fold to be in one process.
    if jobs > 1 and variant in _STRUCTURAL_VARIANTS:
        logger.info("Variant %s is training-free; running it serially.", variant)
        jobs = 1

    if jobs > 1:
        per_fold = _run_folds_parallel(
            plans, seeds, cfg, target_device, workdir=workdir, jobs=jobs,
            resume=resume, cache_dir=cache_dir, skip=skip or [],
            torch_threads=torch_threads, expected_ids=[b.scenario_id for b in bundles],
            auto_layers=auto_layers, inner_val=inner_val,
        )
    else:
        per_fold = _run_folds_serial(
            plans, seeds, cfg, target_device, resume=resume, cache_dir=cache_dir,
        )

    fold_results: List[FoldResult] = []
    for plan in plans:
        metrics = per_fold.get(plan.holdout.scenario_id, [])
        if not metrics:
            logger.error("  Fold produced no usable seed (holdout=%s).",
                         plan.holdout.scenario_id)
            continue
        fold_results.append(_aggregate_fold(plan, metrics))

    # Cross-fold aggregation
    if not fold_results:
        raise RuntimeError("All LOSO folds failed — nothing to report.")

    all_rhos = [f.mean_metrics["spearman_rho"] for f in fold_results]
    all_f1s = [f.mean_metrics["f1_at_k"] for f in fold_results]
    all_ndcgs = [f.mean_metrics["ndcg_10"] for f in fold_results]

    per_type_summary = aggregate_per_type(
        [f.per_type_rho for f in fold_results], value_key="mean", count_key="n_folds"
    )

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
        label_stability={
            b.scenario_id: b.label_stability for b in bundles if b.label_stability
        },
        eval_population=eval_population,
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
            # The node population every fold was scored on. A rho computed on a
            # different population is a different measurement, not a noisier one,
            # so it is recorded next to the number rather than left implicit.
            "eval_population": report.eval_population,
        },
        "per_type_summary": report.per_type_summary,
        "folds": [
            {
                "holdout_id": f.holdout_id,
                "primary_id": f.primary_id,
                "train_ids": f.train_ids,
                "effective_layers": f.effective_layers,
                "val_scenario_id": f.val_scenario_id,
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
        # New columns are appended after the original ten so that any reader
        # indexing by position keeps working.
        w.writerow([
            "holdout_id", "primary_id", "seed",
            "spearman_rho", "f1_at_k", "ndcg_10", "rmse", "mae",
            "n", "prediction_mode",
            "pr_auc", "precision_at_tau", "recall_at_tau", "f1_at_tau",
            "n_true_critical", "rmse_scaled", "mae_scaled", "label_scale_max",
            "n_predicted", "n_labeled", "n_evaluated",
        ])

        def _f(m: Dict[str, Any], key: str) -> str:
            v = m.get(key)
            return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"

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
                    _f(m, "pr_auc"),
                    _f(m, "precision_at_tau"),
                    _f(m, "recall_at_tau"),
                    _f(m, "f1_at_tau"),
                    m.get("n_true_critical", ""),
                    _f(m, "rmse_scaled"),
                    _f(m, "mae_scaled"),
                    _f(m, "label_scale_max"),
                    m.get("n_predicted", ""),
                    m.get("n_labeled", ""),
                    m.get("n_evaluated", ""),
                ])
    logger.info("Wrote %s", path)


def _metric_caveats(report: LOSOReport) -> str:
    """Interpretation notes that must travel with the numbers.

    Each line documents a way these metrics can be over-read. They are emitted
    into the report rather than kept in a docstring because the report is what
    gets copied into papers and issues.
    """
    lines = ["### Reading these numbers", ""]
    lines.append(
        "- **Overlap@K** (reported as `f1_at_k`, `precision_at_k`, `recall_at_k`) — "
        "the predicted and true top-K sets both contain exactly K elements, so all "
        "three are numerically identical. Treat them as one quantity: set overlap."
    )
    lines.append(
        "- **P@τ / R@τ** use an absolute critical set (`I*(v) >= 0.5 * max`), so they "
        "size the truth set from the data and genuinely diverge. `crit` is how many "
        "nodes cleared that bar — when it is 2 or 3, a single ranking error moves "
        "recall by 30-50 points."
    )
    lines.append(
        "- **PR-AUC** is the K-free summary; prefer it when comparing across scenarios."
    )
    lines.append(
        "- **rmse/mae** compare sigmoid-scale predictions against raw labels whose "
        "maximum varies ~4x across scenarios; they largely reflect label scale, not "
        "error. Use `rmse_scaled`/`mae_scaled`."
    )

    ceilings = [
        (sid, st.get("test_retest_spearman"), st.get("topk_jaccard"))
        for sid, st in sorted(report.label_stability.items())
    ]
    measured = [(s, r, j) for s, r, j in ceilings if r is not None]
    if measured:
        worst_rho = min(r for _, r, _ in measured)
        worst_sid = next(s for s, r, _ in measured if r == worst_rho)
        lines.append("")
        lines.append(
            f"**Label noise ceiling.** The ground truth agrees with itself at "
            f"test-retest ρ = **{worst_rho:.4f}** (worst: `{worst_sid}`). A model ρ at or "
            f"near this value has saturated the labels, not underperformed — no method "
            f"can exceed the reproducibility of what it is scored against."
        )
        churn = [(s, j) for s, _, j in measured if j is not None and j < 0.9]
        if churn:
            lines.append("")
            lines.append(
                "Top-K critical sets are themselves unstable across seeds in: "
                + ", ".join(f"`{s}` (Jaccard {j:.2f})" for s, j in sorted(churn))
                + ". Overlap@K and P@τ on those scenarios inherit that churn."
            )
    elif report.label_stability:
        lines.append("")
        lines.append(
            "**Label noise ceiling: not measured.** The cache was built from a single "
            "seed, so the labels' own reproducibility is unknown and ρ has no stated "
            "ceiling. Regenerate with the five recommended seeds to establish one."
        )

    coverages = [
        (f.holdout_id, f.mean_metrics.get("n_evaluated"), f.mean_metrics.get("n_predicted"))
        for f in report.fold_results
    ]
    gaps = [
        f"{h} ({int(ev)}/{int(pr)})"
        for h, ev, pr in coverages
        if ev is not None and pr is not None and pr > 0 and ev < pr
    ]
    if gaps:
        lines.append(
            "- **Coverage gap** — scored on fewer nodes than were predicted: "
            + ", ".join(gaps)
            + ". Unlabelled nodes are dropped from scoring; they are not evidence "
            "either way."
        )
    return "\n".join(lines)


def _fmt(value: Any, ndigits: int = 4) -> str:
    """Render a statistic, passing ``undefined`` through as text, not as 0.0."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and not np.isnan(value):
        return f"{value:.{ndigits}f}"
    return "undefined"


def write_summary_md(report: LOSOReport, path: Path) -> None:
    L: List[str] = []
    L.append("# LOSO Evaluation Summary (G4 closure)")
    L.append("")
    L.append(f"**Folds:** {report.n_folds}  ·  **Seeds per fold:** {report.n_seeds_per_fold}")
    L.append("")
    L.append("## Cross-fold")
    L.append("")
    L.append(f"- Spearman ρ : **{report.overall_mean_rho:.4f} ± {report.overall_std_rho:.4f}**")
    L.append(f"- Overlap @ K : {report.overall_mean_f1:.4f}")
    L.append(f"- NDCG @ 10  : {report.overall_mean_ndcg:.4f}")
    L.append("")
    L.append(_metric_caveats(report))
    L.append("")
    L.append("## Per node type (cross-fold)")
    L.append("")
    L.append("| Node type | mean ρ | std | nodes | folds | folds undefined |")
    L.append("|-----------|--------|-----|-------|-------|-----------------|")
    for nt, info in sorted(report.per_type_summary.items()):
        # "undefined" is a real outcome here — Topic and Node carry no ground
        # truth (failure-simulation.md L6), so their ρ is not a number and must
        # not be printed as one.
        L.append(
            f"| {nt} | {_fmt(info.get('mean'))} | {_fmt(info.get('std'))} | "
            f"{info.get('n_nodes', '—')} | {info.get('n_folds', 0)} | "
            f"{info.get('n_folds_undefined', 0)} |"
        )
    L.append("")
    L.append("## Per-fold details")
    L.append("")
    L.append("| Holdout | Primary | mean ρ | std ρ | Overlap@K | NDCG@10 | PR-AUC | P@τ | R@τ | crit | labelled |")
    L.append("|---------|---------|--------|-------|-----------|---------|--------|-----|-----|------|----------|")
    for f in report.fold_results:
        m = f.mean_metrics

        def _c(key: str, fmt: str = ".4f") -> str:
            v = m.get(key)
            return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, fmt)

        L.append(
            f"| {f.holdout_id} | {f.primary_id} "
            f"| {m['spearman_rho']:.4f} "
            f"| {f.std_metrics['spearman_rho']:.4f} "
            f"| {m['f1_at_k']:.4f} "
            f"| {m['ndcg_10']:.4f} "
            f"| {_c('pr_auc')} "
            f"| {_c('precision_at_tau')} "
            f"| {_c('recall_at_tau')} "
            f"| {_c('n_true_critical', '.0f')} "
            f"| {_c('n_evaluated', '.0f')}/{_c('n_predicted', '.0f')} |"
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
    p.add_argument("--mode", default="gnn", choices=["gnn", "rm"],
                   help="Prediction mode for evaluation (default: gnn)")
    p.add_argument(
        "--variant",
        choices=list(KNOWN_VARIANTS),
        default="hgl_qos",
        help=(
            "Model architecture variant (default: hgl_qos). Display names come "
            "from saag/evaluation/variant_registry.py. "
            "hgl_qos [HGT-QoS] = QoS-embedded HGT on native graph; "
            "hgl     [HGT]     = QoS-masked HGT on native graph; "
            "gl_qos  [GAT-N-QoS] = homogeneous GAT with scalar w(e), native graph; "
            "gl      [GAT-N]     = unweighted homogeneous GAT, native graph; "
            "topology_rm [RM]  = RM composite scores only (no GNN); "
            "topo_baseline [Topo] / topo_qos [Topo-QoS] = training-free centrality "
            "on the DEPENDS_ON projection. "
            "NOTE: this harness runs gl/gl_qos on the NATIVE graph (unlike "
            "reproduce/main_table.py, which runs them on the projection), which is "
            "why they are reported as GAT-N/GAT-N-QoS."
        ),
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument(
        "--auto-layers", dest="auto_layers", action="store_true", default=True,
        help="Auto-downgrade --layers for small/medium scenarios (n_nodes<=200 -> 1 layer, "
             "<=500 -> 2 layers) to reduce overfitting risk. Default: on.",
    )
    p.add_argument(
        "--no-auto-layers", dest="auto_layers", action="store_false",
        help="Disable the layer-count auto-downgrade; always use --layers as given.",
    )
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
    p.add_argument("--rm-consistency-weight", type=float, default=0.0,
                    help="CriticalityLoss weight for RM consistency regularization on unlabeled nodes. "
                    "Default 0.0: the GNN and RM diagnostic pathways are trained independently. "
                    "Pass 0.1 (the pre-decoupling default) to reproduce the ablation arm.")
    p.add_argument(
        "--eval-population", default="application",
        choices=["application", "app_lib", "labeled"],
        help="Node population every variant is scored on. 'application' (default) "
             "matches reproduce/main_table.py, so the LOSO table and the "
             "in-distribution table compare like with like. 'labeled' pools every "
             "node type the cache carries a label for — the historical behaviour, "
             "retained for reproducing older runs, but it mixes populations with "
             "different scales and base rates (Simpson's paradox).",
    )
    p.add_argument(
        "--inner-val-scenario", default="none", choices=["none", "auto"],
        help="Where early stopping and checkpoint selection get their metric. "
             "'none' (default) uses a within-scenario val_mask split of the "
             "primary training graph — i.e. selection on the training "
             "distribution, under a protocol whose point is distribution "
             "shift. 'auto' holds one training scenario (the median-sized "
             "inductive, ties by id) out of the loss entirely and selects on "
             "its whole labelled population. The outer holdout is never "
             "eligible either way.",
    )
    p.add_argument(
        "--rank-normalize-features", action="store_true",
        help="Within-graph rank-normalize the base structural feature columns. "
             "results/feature_shift_diagnostic.md measures up to 115.7x "
             "cross-scenario drift in these columns (mpci), which puts every "
             "LOSO holdout off the training distribution. Off by default; the "
             "un-normalized path remains the ablation arm.",
    )
    p.add_argument(
        "--rank-normalize-labels", action="store_true",
        help="Rank-normalize label targets instead of the IQR+sigmoid default. "
             "The reported metric is Spearman, so ranks are the matched target "
             "scale. Off by default.",
    )
    p.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="Device for model training/inference (default: auto -> cuda if available else cpu)",
    )
    p.add_argument(
        "--jobs", type=int, default=1,
        help="Number of (fold, seed) fits to run concurrently (default: 1). "
             "Every fit seeds torch and numpy from its own seed and writes to its "
             "own directory, so results do not depend on this — only wall-clock "
             "does. The corpus here is small enough that a fit is latency-bound "
             "rather than compute-bound, so several concurrent workers share one "
             "GPU (or one CPU) without contending for it.",
    )
    p.add_argument(
        "--torch-threads", type=int, default=1,
        help="Intra-op threads per process (default: 1). On graphs this small, "
             "every tensor op is smaller than its own threading overhead: "
             "measured on the enterprise scenario, one forward+backward takes "
             "1540 ms at 14 threads and 54 ms at 1. Results are unchanged (the "
             "fold reproduces to 10 decimal places either way).",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Reuse any completed (fold, seed) fit whose stored fingerprint "
             "still matches this configuration, the cache contents and the "
             "current model code. Anything that does not match is re-run from "
             "scratch, its stale checkpoint deleted first.",
    )
    p.add_argument(
        "--no-preflight", dest="preflight", action="store_false", default=True,
        help="Skip the one-epoch probe fit that runs before the sweep. The probe "
             "exercises the whole path on the target device in a few seconds; "
             "without it, an environment fault surfaces only after every fit has "
             "failed (a CUDA-only fault in the k-fold harness cost 3.8 hours "
             "before it raised).",
    )
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

    # Set before the first tensor op, and inherited by pool workers.
    torch.set_num_threads(max(1, args.torch_threads))

    dev_desc = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    if dev_desc == "cuda":
        dev_desc += f" ({torch.cuda.get_device_name(0)})"

    logger.info("LOSO Evaluation — G4 closure")
    logger.info("  Cache:     %s", args.cache_dir)
    logger.info("  Output:    %s", args.output_dir)
    logger.info("  Device:    %s", dev_desc)
    logger.info("  Layer:     %s", args.layer)
    logger.info("  Seeds:     %s", seeds)
    logger.info("  Mode:      %s", args.mode)
    logger.info("  Variant:   %s", getattr(args, 'variant', 'hgl_qos'))
    logger.info("  Skip:      %s", skip if skip else "(none)")
    logger.info("  Jobs:      %d (%d intra-op thread(s) each)", args.jobs, args.torch_threads)
    logger.info("  Resume:    %s", "on (fingerprinted)" if args.resume else "off")

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
        variant=getattr(args, 'variant', 'hgl_qos'),
        auto_layers=args.auto_layers,
        weight_decay=args.weight_decay,
        warmup_T0=args.warmup_t0,
        multitask_weight=args.multitask_weight,
        rm_consistency_weight=args.rm_consistency_weight,
        ranking_weight=args.ranking_weight,
        pairwise_ranking_weight=args.pairwise_ranking_weight,
        eval_population=args.eval_population,
        inner_val=args.inner_val_scenario,
        rank_normalize_features=args.rank_normalize_features,
        rank_normalize_labels=args.rank_normalize_labels,
        device=args.device,
        jobs=args.jobs,
        resume=args.resume,
        cache_dir=args.cache_dir,
        skip=skip,
        torch_threads=args.torch_threads,
        preflight=args.preflight,
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
