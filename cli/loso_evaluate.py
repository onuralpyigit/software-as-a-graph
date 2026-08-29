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
from saag.evaluation.metrics import (
    aggregate_per_type,
    compute_inductive_metrics as _shared_inductive_metrics,
    resolve_eval_keys,
)
from saag.prediction.gnn_service import GNNService
from saag.prediction.data_preparation import (
    networkx_to_hetero_data,
    extract_simulation_dict,
    extract_structural_metrics_dict,
    extract_rm_scores_dict,
)
from saag.core.models import QoSPolicy, topic_weight_from_node_attrs

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
            # ``type`` is excluded from the splat, not just ``id``/``name``: the
            # three real-world topologies carry a per-entity ``type`` field, and
            # letting it through collided with the keyword above
            # (``TypeError: got multiple values for keyword argument 'type'``),
            # so this builder raised on every one of them. The canonical bucket
            # label wins because the node-type contract downstream
            # (``resolve_eval_keys``, ``networkx_to_hetero_data``) is keyed on it;
            # where both are present they agree anyway.
            g.add_node(
                entity["id"],
                type=type_label,
                name=entity.get("name", entity["id"]),
                **{k: v for k, v in entity.items() if k not in ("id", "name", "type")},
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
                attrs: Dict[str, Any] = {
                    "type": r.get("type", type_label),
                    "qos_profile": r.get("qos_profile", {}),
                }
                # Only pin a weight the topology actually stated, so the
                # projection below can tell "unset" from "deliberately 1.0".
                if r.get("weight") is not None:
                    attrs["weight"] = float(r["weight"])
                g.add_edge(src, dst, **attrs)

    _project_topic_qos_onto_edges(g)
    for _, _, data in g.edges(data=True):
        data.setdefault("weight", 1.0)
    return g


#: Edge types that inherit their weight and QoS profile from the Topic endpoint.
_TOPIC_MEDIATED_EDGES = ("PUBLISHES_TO", "SUBSCRIBES_TO", "ROUTES")


def _project_topic_qos_onto_edges(g: nx.DiGraph) -> None:
    """Inherit each Topic's w(t) and QoS profile onto its incident pub/sub edges.

    Mirrors the ``SET r.weight = t.weight`` inheritance the repositories perform
    on import (see ``Neo4jRepository._calculate_intrinsic_weights``). Topology
    JSON carries QoS on Topic *nodes* only and states no edge attributes at all,
    so without this pass every pub/sub edge reaching a consumer has
    ``weight=1.0`` and ``qos_profile={}`` — constant across the whole graph.
    That is what made the GNN's QoS edge dimensions carry no signal.

    Existing non-default values are left alone, so a topology that does state
    edge-level QoS keeps it.
    """
    for u, v, data in g.edges(data=True):
        etype = (data.get("type") or data.get("etype") or "").upper()
        if etype not in _TOPIC_MEDIATED_EDGES:
            continue

        # Topic is the target on PUBLISHES_TO/SUBSCRIBES_TO/ROUTES, but tolerate
        # either orientation rather than silently skipping a reversed edge.
        topic = v if g.nodes.get(v, {}).get("type") == "Topic" else u
        attrs = g.nodes.get(topic, {})
        if attrs.get("type") != "Topic":
            continue

        if not data.get("qos_profile"):
            data["qos_profile"] = QoSPolicy.from_node_attrs(attrs).to_dict()
        if "weight" not in data:
            data["weight"] = topic_weight_from_node_attrs(attrs)


def load_scenario_bundle(scenario_dir: Path) -> Optional[ScenarioBundle]:
    """Load one scenario's full artefact bundle from cache. Returns None on incomplete cache."""
    scenario_id = scenario_dir.name
    topology = _load_json(scenario_dir / "topology.json")
    structural_raw = _load_json(scenario_dir / "structural_metrics.json")
    rm_raw = _load_json(scenario_dir / "quality_scores.json")
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

    conv = networkx_to_hetero_data(graph, structural, simulation, rm)

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
) -> FoldResult:
    """
    One LOSO fold: train on N-1 scenarios with multi-seed, predict on held-out.

    Defensive invariants:
      - holdout never appears in train_ids
      - holdout's structural/rm are passed at predict() time (needed for features)
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
            if variant in ("topo_baseline", "topo_qos"):
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
                    continue
                pred_scores = {str(k): float(v) for k, v in struct_pred.items()}
                full_node_scores = {
                    k: {"overall": v, "reliability": v, "maintainability": v}
                    for k, v in pred_scores.items()
                }

            elif variant in ("gl", "gl_qos"):
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
                    from reproduce.main_table import _mask_qos_in_graph, _mask_qos_in_structural
                    train_graph = _mask_qos_in_graph(primary.graph)
                    train_sm    = _mask_qos_in_structural(primary.structural)
                    holdout_graph = _mask_qos_in_graph(holdout.graph)
                    holdout_sm    = _mask_qos_in_structural(holdout.structural)

                conv = networkx_to_hetero_data(
                    train_graph, train_sm, primary.simulation, primary.rm, qos_enabled=use_qos
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
                                         lr=lr, num_epochs=epochs, patience=min(60, epochs),
                                         weight_decay=weight_decay, warmup_T0=warmup_T0,
                                         multitask_weight=multitask_weight,
                                         rm_consistency_weight=rm_consistency_weight,
                                         ranking_weight=ranking_weight,
                                         pairwise_ranking_weight=pairwise_ranking_weight)
                    trainer.train(data)

                # Evaluate on holdout
                conv_h = networkx_to_hetero_data(
                    holdout_graph, holdout_sm, holdout.simulation, holdout.rm, qos_enabled=use_qos
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
                            }

            else:
                # hgl_qos (default) or hgl or topology_rm → GNNService path
                effective_mode = "rm" if variant == "topology_rm" else mode
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
                use_qos = (variant == "hgl_qos")

                if use_qos:
                    train_graph = primary.graph
                    train_sm    = primary.structural
                    holdout_graph = holdout.graph
                    holdout_sm    = holdout.structural
                else:
                    from reproduce.main_table import _mask_qos_in_graph, _mask_qos_in_structural
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
                        rm_scores=primary.rm,
                        inductive_graphs=[
                            networkx_to_hetero_data(
                                b.graph if use_qos else _mask_qos_in_graph(b.graph),
                                b.structural if use_qos else _mask_qos_in_structural(b.structural),
                                b.simulation,
                                b.rm,
                                qos_enabled=use_qos
                            ).hetero_data
                            for b in inductives
                        ],
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
                    )
                result = service.predict(
                    graph=holdout_graph,
                    structural_metrics=holdout_sm,
                    rm_scores=holdout.rm,
                    # GNNService.train() names this simulation_results, predict() names
                    # it eval_labels. Passing the train() spelling here raised TypeError
                    # inside the per-seed try/except, so every HGL/HGL-QoS seed was
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


        except Exception as e:
            logger.error("  Fold seed %d failed: %s", seed, e, exc_info=True)
            continue

        true_impact = {nid: float(d.get("composite", 0.0)) for nid, d in holdout.simulation.items()}

        m = compute_inductive_metrics(
            pred_scores, true_impact, holdout.graph, population=eval_population,
        )
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
        return [m[key] for m in seed_metrics if key in m and not np.isnan(m[key])]

    added_keys = [
        "precision_at_tau", "recall_at_tau", "f1_at_tau", "pr_auc",
        "rmse_scaled", "mae_scaled", "n_true_critical",
        "n_predicted", "n_labeled", "n_evaluated",
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
    variant: str = "hgl_qos",
    auto_layers: bool = True,
    weight_decay: float = 1e-4,
    warmup_T0: Optional[int] = None,
    multitask_weight: float = 0.5,
    rm_consistency_weight: float = 0.0,
    ranking_weight: float = 0.3,
    pairwise_ranking_weight: float = 0.1,
    eval_population: str = "application",
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
                eval_population=eval_population,
                auto_layers=auto_layers,
                weight_decay=weight_decay,
                warmup_T0=warmup_T0,
                multitask_weight=multitask_weight,
                rm_consistency_weight=rm_consistency_weight,
                ranking_weight=ranking_weight,
                pairwise_ranking_weight=pairwise_ranking_weight,
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
        choices=["hgl_qos", "hgl", "gl_qos", "gl", "topology_rm", "topo_baseline", "topo_qos"],
        default="hgl_qos",
        help=(
            "Model architecture variant (default: hgl_qos). "
            "hgl_qos = QoS-embedded HGT on native graph; "
            "hgl     = QoS-masked HGT on native graph; "
            "gl_qos  = QoS-weighted homogeneous GAT on projection; "
            "gl      = unweighted homogeneous GAT on projection; "
            "topology_rm = RM scores only (no GNN)."
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
    logger.info("  Variant:   %s", getattr(args, 'variant', 'hgl_qos'))
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
        variant=getattr(args, 'variant', 'hgl_qos'),
        auto_layers=args.auto_layers,
        weight_decay=args.weight_decay,
        warmup_T0=args.warmup_t0,
        multitask_weight=args.multitask_weight,
        rm_consistency_weight=args.rm_consistency_weight,
        ranking_weight=args.ranking_weight,
        pairwise_ranking_weight=args.pairwise_ranking_weight,
        eval_population=args.eval_population,
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
