#!/usr/bin/env python3
"""
reproduce/realworld_zeroshot.py — the learned model on systems we did not generate
=================================================================================

Trains HGT-QoS once on the full synthetic corpus and scores it zero-shot on the
five open-source reference systems. Closes the gap Section 7.4 states plainly:
the real-world table reports only the deterministic explanation layer Q(v) and
the training-free topological baselines, so the typing claim of Section 7.2 has
never been tested outside our own generator.

Oracle
------
This is the part that cannot be shortcut. Table 9's ground truth is
``cli/validate_graph.py``'s composite cascade impact I_comp(v), produced by
``FailureSimulator`` — the Validate-stage oracle. The GNN is trained against
I*(v), produced by ``FaultInjector`` — the Predict-stage labeler. CLAUDE.md's
ground-truth contract forbids substituting one for the other within a stage
(``tests/test_groundtruth_contract.py``), so this script scores the learned
model against I*(v) and reports it as its **own table**, not as a column
appended to Table 9. Mixing the two oracles in one table would make the
learned column incomparable to the ones beside it while looking comparable.

Populate the real-world cache first, into a directory that is NOT
``output/loso_cache`` — ``discover_scenarios`` treats every directory it finds
there as a LOSO fold, so caching the real systems alongside the synthetic ones
would silently change the twelve-fold corpus behind every published number::

    CACHE_DIR=output/realworld_cache bash scripts/populate_loso_cache.sh \\
        realworld_autoware_ros2 realworld_cloud_microservices \\
        realworld_trainticket realworld_homeassistant realworld_edgex

Usage
-----
    PYTHONPATH=. python reproduce/realworld_zeroshot.py
    PYTHONPATH=. python reproduce/realworld_zeroshot.py --seeds 42,123
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from reproduce._provenance import stamp

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.loso_evaluate import (  # noqa: E402
    ScenarioBundle,
    _build_training_hetero,
    _build_validation_hetero,
    _dependency_bundle,
    _graft_qos_edge_attr,
    _prepare_bundle_graph,
    _select_val_bundle,
    _with_prior,
    compute_inductive_metrics,
    discover_scenarios,
)
from saag.evaluation import variant_registry as _registry  # noqa: E402
from saag.evaluation.metrics import resolve_eval_keys  # noqa: E402
from saag.prediction.gnn_service import GNNService  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def train_once(
    bundles: List[ScenarioBundle],
    seed: int,
    ckpt_dir: Path,
    *,
    use_qos: bool,
    epochs: int,
    layers: int,
    rank_normalize_features: bool,
    rank_normalize_labels: bool,
    device: Optional[str] = "auto",
    topo_prior: bool = False,
    use_bidirectional: bool = True,
    hidden_channels: int = 64,
) -> GNNService:
    """Train one HGT on the whole synthetic corpus.

    Mirrors the ``hgl_qos`` branch of ``cli.loso_evaluate.run_one_fold``: the
    largest scenario is the primary graph the splits are drawn on, one
    median-sized scenario is held out of the loss to drive early stopping, and
    the rest arrive through the inductive-graph channel. The difference is that
    nothing is held out for *testing* — the test set is the real-world corpus,
    which lives in a different cache entirely.
    """
    import torch
    target_device = torch.device(
        "cuda" if (device == "cuda" or (device in ("auto", None) and torch.cuda.is_available())) else "cpu"
    )
    primary = max(bundles, key=lambda b: b.n_nodes)
    inductives = [b for b in bundles if b.scenario_id != primary.scenario_id]
    val_bundle = _select_val_bundle(inductives, "auto")
    if val_bundle is not None:
        inductives = [b for b in inductives if b.scenario_id != val_bundle.scenario_id]

    logger.info(
        "  seed %d: primary=%s (|V|=%d), inductive=%d, val=%s",
        seed, primary.scenario_id, primary.n_nodes, len(inductives),
        val_bundle.scenario_id if val_bundle else "(none)",
    )

    train_graph, train_sm = _prepare_bundle_graph(primary, use_qos)
    if topo_prior:
        train_sm = _with_prior(primary, train_sm)
    service = GNNService(
        checkpoint_dir=str(ckpt_dir),
        hidden_channels=hidden_channels,
        num_heads=4,
        num_layers=layers,
        dropout=0.2,
        predict_edges=False,
        device=target_device,
        topo_prior=topo_prior,
        use_bidirectional=use_bidirectional,
    )
    service.train(
        graph=train_graph,
        structural_metrics=train_sm,
        simulation_results=primary.simulation,
        rm_scores=primary.rm,
        inductive_graphs=[
            _build_training_hetero(b, use_qos, rank_normalize_features, topo_prior)
            for b in inductives
        ],
        val_graph=(
            _build_validation_hetero(val_bundle, use_qos, rank_normalize_features, topo_prior)
            if val_bundle is not None else None
        ),
        seeds=[seed],
        num_epochs=epochs,
        lr=3e-4,
        patience=min(60, epochs),
        layer="app",
        qos_enabled=use_qos,
        rank_normalize_features=rank_normalize_features,
        rank_normalize_labels=rank_normalize_labels,
    )
    return service


class HomogeneousScorer:
    """A trained untyped GAT arm, exposing just what ``score`` needs.

    Built by :func:`train_once_homogeneous`.
    """

    def __init__(self, model, device, rank_normalize_features: bool, topo_prior: bool = False,
                 graft_edges: bool = False):
        self.model = model
        self.device = device
        self.rank_normalize_features = rank_normalize_features
        #: Prior column: falsy, True / "topo_qos" (Hybrid-GAT, Amendment 6) or
        #: "indeg" (Hybrid-GAT-P, Amendment 9).
        self.topo_prior = topo_prior
        #: QoS node features masked, QoS edge channel kept (GAT-QoS-nf).
        self.graft_edges = graft_edges

    def predict_scores(self, bundle: ScenarioBundle, use_qos: bool) -> Dict[str, float]:
        import torch
        from saag.prediction.data_preparation import networkx_to_hetero_data

        graph, sm = _prepare_bundle_graph(bundle, use_qos)
        if self.topo_prior:
            sm = _with_prior(bundle, sm, self.topo_prior)
        conv = networkx_to_hetero_data(
            graph, sm, bundle.simulation, bundle.rm,
            qos_enabled=use_qos, rank_normalize_features=self.rank_normalize_features,
            append_prior=bool(self.topo_prior),
        )
        if self.graft_edges:
            _graft_qos_edge_attr(conv.hetero_data, bundle, self.rank_normalize_features)
        data = conv.hetero_data.to(self.device)
        self.model.eval()
        with torch.no_grad():
            x = {nt: data[nt].x for nt in data.node_types if hasattr(data[nt], "x")}
            ei = {r: data[r].edge_index for r in data.edge_types}
            ea = {r: data[r].edge_attr for r in data.edge_types if hasattr(data[r], "edge_attr")}
            out = self.model(x, ei, ea)
        pred: Dict[str, float] = {}
        for nt, preds in out.items():
            for i, nid in enumerate(conv.node_id_map.get(nt, [])):
                if i < preds.shape[0]:
                    pred[nid] = float(preds[i, 0])
        return pred


def train_once_homogeneous(
    bundles: List[ScenarioBundle],
    seed: int,
    ckpt_dir: Path,
    *,
    variant: str,
    epochs: int,
    layers: int,
    rank_normalize_features: bool,
    rank_normalize_labels: bool,
    device: Optional[str] = "auto",
) -> HomogeneousScorer:
    """Train one untyped GAT arm on the whole synthetic corpus.

    Mirrors the homogeneous branch of ``cli.loso_evaluate.run_one_fold`` (same
    model builder, width and edge channel from the registry, same trainer,
    label normalisation and dimension mask) with ``train_once``'s split of the
    corpus: largest scenario as primary, median-sized one for early stopping,
    the rest as inductive graphs.
    """
    import torch
    from torch_geometric.loader import DataLoader as _PyGDataLoader
    from saag.prediction.data_preparation import (
        create_node_splits, networkx_to_hetero_data, normalize_labels_robust,
    )
    from saag.prediction.models.baselines import build_baseline
    from saag.prediction.trainer import GNNTrainer

    torch.manual_seed(seed)
    np.random.seed(seed)
    target_device = torch.device(
        "cuda" if (device == "cuda" or (device in ("auto", None) and torch.cuda.is_available())) else "cpu"
    )
    primary = max(bundles, key=lambda b: b.n_nodes)
    inductives = [b for b in bundles if b.scenario_id != primary.scenario_id]
    val_bundle = _select_val_bundle(inductives, "auto")
    if val_bundle is not None:
        inductives = [b for b in inductives if b.scenario_id != val_bundle.scenario_id]

    edge_dim = _registry.edge_dim(variant, "loso")
    use_qos = _registry.node_qos_for(variant, "loso")
    graft_edges = edge_dim is not None and not use_qos
    topo_prior = _registry.prior_for(variant)
    train_graph, train_sm = _prepare_bundle_graph(primary, use_qos)
    if topo_prior:
        train_sm = _with_prior(primary, train_sm, topo_prior)
    conv = networkx_to_hetero_data(
        train_graph, train_sm, primary.simulation, primary.rm,
        qos_enabled=use_qos, rank_normalize_features=rank_normalize_features,
        append_prior=bool(topo_prior),
    )
    data = conv.hetero_data
    if graft_edges:
        _graft_qos_edge_attr(data, primary, rank_normalize_features)
    create_node_splits(data, seed=seed)
    inductive_data = [
        _build_training_hetero(b, use_qos, rank_normalize_features, topo_prior)
        for b in inductives
    ]
    if graft_edges:
        for b, ig in zip(inductives, inductive_data):
            _graft_qos_edge_attr(ig, b, rank_normalize_features)
    for ig in inductive_data:
        create_node_splits(ig, seed=seed)
    normalize_labels_robust(data, rank_normalize=rank_normalize_labels)
    for ig in inductive_data:
        normalize_labels_robust(ig, rank_normalize=rank_normalize_labels)
    val_data = None
    if val_bundle is not None:
        val_data = _build_validation_hetero(
            val_bundle, use_qos, rank_normalize_features, topo_prior
        )
        if graft_edges:
            _graft_qos_edge_attr(val_data, val_bundle, rank_normalize_features)
        normalize_labels_robust(val_data, rank_normalize=rank_normalize_labels)

    model = build_baseline(
        "homo_unweighted" if edge_dim is None else "homo_scalar",
        hidden_channels=_registry.hidden_for(variant, 64, "loso"),
        num_heads=4, num_layers=layers, dropout=0.2, edge_dim=edge_dim,
        topo_prior=bool(topo_prior),
    )
    model.to(target_device)
    trainer = GNNTrainer(
        model=model, checkpoint_dir=str(ckpt_dir), lr=3e-4, num_epochs=epochs,
        patience=min(60, epochs), dimension_mask=conv.dimension_mask,
    )
    trainer.train(
        _PyGDataLoader([data] + inductive_data, batch_size=1, shuffle=True),
        primary_data=data, val_data=val_data,
    )
    return HomogeneousScorer(model, target_device, rank_normalize_features, topo_prior,
                             graft_edges=graft_edges)


class TabularScorer:
    """A fitted GBM-Feat arm, exposing just what ``score`` needs.

    Built by :func:`train_once_tabular`.
    """

    def __init__(self, train_data, seed: int, rank_normalize_features: bool):
        self.train_data = train_data
        self.seed = seed
        self.rank_normalize_features = rank_normalize_features

    def predict_scores(self, bundle: ScenarioBundle, use_qos: bool) -> Dict[str, float]:
        from saag.prediction.data_preparation import networkx_to_hetero_data
        from saag.prediction.models.tabular import fit_predict_tabular

        graph, sm = _prepare_bundle_graph(bundle, use_qos)
        conv = networkx_to_hetero_data(
            graph, sm, bundle.simulation, bundle.rm,
            qos_enabled=use_qos, rank_normalize_features=self.rank_normalize_features,
        )
        return fit_predict_tabular(
            self.train_data, conv.hetero_data, conv.node_id_map, seed=self.seed,
        )


def train_once_tabular(
    bundles: List[ScenarioBundle],
    seed: int,
    *,
    variant: str,
    rank_normalize_features: bool,
    rank_normalize_labels: bool,
) -> TabularScorer:
    """Assemble GBM-Feat's training set: the same graphs the GAT arms fit on.

    Same corpus split as :func:`train_once_homogeneous` (the early-stopping
    scenario is left out, since the GAT arms never fit it) and the same label
    normalisation. Fitting is deferred to scoring, as in the LOSO branch.
    """
    from saag.prediction.data_preparation import normalize_labels_robust

    use_qos = _registry.node_qos_for(variant, "loso")
    primary = max(bundles, key=lambda b: b.n_nodes)
    inductives = [b for b in bundles if b.scenario_id != primary.scenario_id]
    val_bundle = _select_val_bundle(inductives, "auto")
    if val_bundle is not None:
        inductives = [b for b in inductives if b.scenario_id != val_bundle.scenario_id]
    train_data = [
        _build_training_hetero(b, use_qos, rank_normalize_features)
        for b in [primary, *inductives]
    ]
    for td in train_data:
        normalize_labels_robust(td, rank_normalize=rank_normalize_labels)
    return TabularScorer(train_data, seed, rank_normalize_features)


def score(service: GNNService, bundle: ScenarioBundle, *, use_qos: bool,
          population: str, rank_normalize_features: bool = True) -> Dict[str, Any]:
    """Zero-shot predict on one real system and score against its I*(v) labels."""
    if isinstance(service, (HomogeneousScorer, TabularScorer)):
        pred = service.predict_scores(bundle, use_qos)
    else:
        graph, sm = _prepare_bundle_graph(bundle, use_qos)
        if service.topo_prior:
            sm = _with_prior(bundle, sm)
        result = service.predict(
            graph=graph,
            structural_metrics=sm,
            rm_scores=bundle.rm,
            eval_labels=bundle.simulation,
            mode="gnn",
            qos_enabled=use_qos,
            rank_normalize_features=rank_normalize_features,
        )
        pred = {nid: float(ns.composite_score) for nid, ns in result.node_scores.items()}
    true_impact = {
        nid: float(d.get("composite", 0.0)) for nid, d in bundle.simulation.items()
    }
    # A dependency-graph bundle (Amendment 9) is scored on its native graph, so
    # every arm is scored on the same population.
    graph = bundle.graph.graph.get("infra_source", bundle.graph)
    m = compute_inductive_metrics(
        pred, true_impact, graph, population=population
    )

    # Topology prior for hybrid evaluation
    topo_pred = {
        nid: 0.6 * float(m_dict.get("betweenness_centrality", 0.0))
           + 0.4 * float(m_dict.get("ap_c_score", 0.0))
        for nid, m_dict in (bundle.structural or {}).items()
    }
    max_t = max(topo_pred.values()) if topo_pred and max(topo_pred.values()) > 0 else 1.0
    norm_t = {k: v / max_t for k, v in topo_pred.items()}
    max_g = max(pred.values()) if pred and max(pred.values()) > 0 else 1.0
    norm_g = {k: v / max_g for k, v in pred.items()}
    pred_hybrid = {k: 0.5 * norm_t.get(k, 0.0) + 0.5 * norm_g.get(k, 0.0) for k in pred}
    m_hybrid = compute_inductive_metrics(
        pred_hybrid, true_impact, graph, population=population
    )
    m["hybrid_spearman_rho"] = float(m_hybrid.get("spearman_rho", 0.0))
    m["hybrid_f1_at_k"] = float(m_hybrid.get("f1_at_k", 0.0))

    # Active strata (positive ground truth only, per tests/test_zero_exclusion.py)
    pos_impact = {nid: val for nid, val in true_impact.items() if val > 0}
    if len(pos_impact) >= 3:
        m_pos = compute_inductive_metrics(
            pred, pos_impact, graph, population=population
        )
        m["spearman_rho_positive"] = (
            float(m_pos.get("spearman_rho")) if m_pos.get("spearman_rho") is not None else None
        )
        # Count what the correlation was actually computed on. `pos_impact`
        # spans every labelled node type, so reporting its length alongside an
        # Application-population rho printed n_positive > n_evaluated.
        m["n_positive"] = m_pos.get("n_evaluated", len(pos_impact))
    else:
        m["spearman_rho_positive"] = None
        m["n_positive"] = len(pos_impact)

    m["eval_points"] = _eval_points(pred, true_impact, graph, population)
    return m


def _eval_points(pred, true_impact, graph, population) -> List[Dict[str, Any]]:
    """The (prediction, label) pairs behind one system's figures.

    Kept so a stratification nobody ran at the time — by node type, by whether
    the oracle scores the component above zero, by QoS tier, or pooled across
    the five systems instead of averaged over them — can be checked against a
    published number without re-training. Every pooled figure in this artifact
    is a candidate for Simpson's paradox and these rows are what make that
    testable.
    """
    keys = resolve_eval_keys(pred, true_impact, graph, population)
    return [
        {
            "id": nid,
            "type": graph.nodes[nid].get("type") if nid in graph else None,
            "pred": round(float(pred[nid]), 6),
            "true": round(float(true_impact[nid]), 6),
        }
        for nid in keys
    ]


def score_references(bundle: ScenarioBundle, *, population: str) -> Dict[str, Any]:
    """Score the training-free references against the SAME oracle and population.

    Without these the learned number is uninterpretable. The Q(v) figures in
    Section 7.4 are scored against I_comp(v) (FailureSimulator), so they cannot
    be set beside a model trained and scored on I*(v) — the comparison has to be
    rebuilt on one oracle, which is what this does.

    ``Topo`` is 0.6*betweenness + 0.4*articulation, the same combination
    ``reproduce.main_table._compute_topo_baseline_scores`` uses, read off the
    cached app-layer metrics (already the DEPENDS_ON projection, since the cache
    is built with ``analyze_graph.py --layer app``). ``Topo-QoS`` is the same
    combination with QoS-weighted betweenness, computed by the same function.

    Topo-QoS was previously omitted here on the grounds that the cache carried no
    QoS edge weights. It did not, but the cause was a defect rather than a
    property of the data: the architecture adapters emit ``weight: 1.0`` on every
    edge while the generator omits the key, and ``_project_topic_qos_onto_edges``
    guarded on key *absence*, so the QoS-derived weight was applied to generated
    topologies and skipped on transcribed ones. With that guard corrected the
    weights are present (48-66% of edges non-unit across the five systems) and
    the baseline is computable on the same footing as everywhere else.
    """
    true_impact = {
        nid: float(d.get("composite", 0.0)) for nid, d in bundle.simulation.items()
    }
    out: Dict[str, Any] = {}

    rm_pred = {nid: float(v.get("overall", 0.0)) for nid, v in (bundle.rm or {}).items()}
    if rm_pred:
        out["RM"] = compute_inductive_metrics(
            rm_pred, true_impact, bundle.graph, population=population
        )

    topo_pred = {
        nid: 0.6 * float(m.get("betweenness_centrality", 0.0))
           + 0.4 * float(m.get("ap_c_score", 0.0))
        for nid, m in (bundle.structural or {}).items()
    }
    if topo_pred and any(v > 0 for v in topo_pred.values()):
        out["Topo"] = compute_inductive_metrics(
            topo_pred, true_impact, bundle.graph, population=population
        )

    # Topo-QoS via the canonical implementation, so this column and the synthetic
    # tables cannot drift apart. It returns None when the graph yields no signal,
    # and _qos_weighted_betweenness falls back to unweighted betweenness when no
    # QoS weights are present -- in which case Topo-QoS would equal Topo, so we
    # record whether the weights were actually there rather than leaving a
    # silently duplicated column.
    from reproduce.main_table import (
        _compute_topo_baseline_scores,
        _qos_weighted_betweenness,
    )
    n_weighted = sum(
        1 for _, _, d in bundle.graph.edges(data=True)
        if abs(float(d.get("qos_weight", d.get("weight", 1.0))) - 1.0) > 1e-9
    )
    topo_qos_pred = _compute_topo_baseline_scores(
        bundle.graph, bundle.structural, use_qos=True
    )
    app_ids = [n for n, d in bundle.graph.nodes(data=True)
               if d.get("type") == "Application"]
    scored = {v for nid, v in (topo_qos_pred or {}).items() if nid in set(app_ids)}
    if topo_qos_pred and len(scored) > 1:
        m = compute_inductive_metrics(
            topo_qos_pred, true_impact, bundle.graph, population=population
        )
        m["n_qos_weighted_edges"] = n_weighted
        out["Topo-QoS"] = m
    else:
        # Degenerate, and the reason is structural rather than incidental, so it
        # is recorded instead of emitted as a NaN column. ``Topo`` reads
        # betweenness off the cached *app-layer projection*; the QoS-weighted
        # variant recomputes it on the raw multigraph, where Applications never
        # route messages and betweenness is identically zero for all of them
        # (the degeneracy Section 6.2.2 gives as the reason topological
        # baselines run on the projection at all). Scoring Topo-QoS here needs
        # the derived DEPENDS_ON projection as a graph object; this cache stores
        # that layer only as precomputed scalars.
        out["Topo-QoS"] = {
            "unavailable": "qos_weighted_betweenness_degenerate_on_raw_multigraph",
            "n_qos_weighted_edges": n_weighted,
            "n_distinct_app_scores": len(scored),
        }
    return out


#: Identification metrics carried from the metric contract into this artifact.
#: ``f1_at_k`` is deliberately not in the list: it is overlap@K, where the
#: predicted and true sets both have exactly K members, so precision, recall and
#: F1 are identically equal and none of them can be read as an F1 score.
_ID_KEYS = (
    "precision_at_tau", "recall_at_tau", "f1_at_tau",
    "precision_at_threshold", "recall_at_threshold", "f1_at_threshold",
    "f1_max", "f1_all_positive", "pr_auc",
    "n_true_critical", "n_pred_critical",
)


def _finite(value):
    """Numbers through unchanged; NaN and absent values become ``None``.

    A degenerate truth set (every node critical, or none) leaves the metric
    contract returning NaN. JSON has no NaN, and averaging it in as 0.0 would
    turn "not measurable here" into "the model scored zero".
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return None if np.isnan(value) else float(value)
    return value


def bootstrap_over_systems(
    per_system: Dict[str, Any], references: Dict[str, Any],
    b: int = 2000, alpha: float = 0.05, seed: int = 42,
) -> Dict[str, Any]:
    """Percentile bootstrap over the five systems, for every predictor.

    Five systems is a small sample and resampling cannot make it larger. What
    the interval does is stop a five-point mean from being read as if it were
    an estimate with negligible error: the headline transfer figures are
    averages over five numbers that range across half the scale, and an
    interval that spans most of that range says so where a point estimate does
    not. Resampling is over systems because the system is the unit the mean is
    taken over, matching the unit of analysis used for folds under LOSO.

    The intervals are descriptive. At n = 5 the percentile bootstrap has no
    coverage guarantee worth quoting, and we do not attach a significance
    claim to it.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    def _ci(values: List[float]) -> Optional[Dict[str, float]]:
        vals = np.array([v for v in values if v is not None], dtype=float)
        if vals.size < 2:
            return None
        draws = vals[rng.integers(0, vals.size, size=(b, vals.size))].mean(axis=1)
        lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return {"mean": float(vals.mean()), "lo": float(lo), "hi": float(hi),
                "n_systems": int(vals.size)}

    out: Dict[str, Any] = {
        "method": f"percentile bootstrap over systems, B={b}, seed={seed}",
        "caveat": ("n = 5 systems. Descriptive only: no coverage guarantee and "
                   "no significance claim is attached to these intervals."),
        "learned": {
            "rho": _ci([s.get("mean_rho") for s in per_system.values()]),
            "rho_positive": _ci([s.get("mean_rho_positive") for s in per_system.values()]),
            "f1_at_k": _ci([s.get("mean_f1_at_k") for s in per_system.values()]),
            "pr_auc": _ci([s.get("mean_pr_auc") for s in per_system.values()]),
        },
    }
    for name, block in references.items():
        out[name] = {
            "rho": _ci([s.get("rho") for s in block.values()]),
            "rho_positive": _ci([s.get("rho_positive") for s in block.values()]),
            "f1_at_k": _ci([s.get("f1_at_k") for s in block.values()]),
            "pr_auc": _ci([s.get("pr_auc") for s in block.values()]),
        }
    return out


def _identification_summary(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean/std of the identification family over the seeds where it is defined."""
    out: Dict[str, Any] = {}
    for key in _ID_KEYS:
        vals = [v for v in (_finite(r.get(key)) for r in runs) if v is not None]
        if key.startswith("n_"):
            out[key] = int(vals[0]) if vals else None
            continue
        out[f"mean_{key}"] = float(np.mean(vals)) if vals else None
        out[f"std_{key}"] = float(np.std(vals)) if vals else None
        out[f"n_seeds_{key}_defined"] = len(vals)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic-cache", type=Path, default=Path("output/loso_cache"))
    p.add_argument("--realworld-cache", type=Path, default=Path("output/realworld_cache"))
    p.add_argument("--variant", default="hgl_qos", choices=["hgl_qos", "hgl", "hgl_qos_prior",
                            "gl_full_cap", "gl_full_qos16_cap", "gl_qos16_prior",
                            "gl_full_qos16_nfmask", "tab_gbm", "tab_gbm_qos", "hgl_qos_uni",
                            "gl_full_qos_cap", "gl_proj_cap", "gl_proj_qos16_cap",
                            "gl_proj_qos16_indeg_prior", "hgl_proj_qos"])
    p.add_argument("--seeds", default="42,123,456,789,2024")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--eval-population", default="application",
                   choices=["application", "app_lib", "labeled"])
    p.add_argument("--rank-normalize-features", action="store_true", default=True)
    p.add_argument("--no-rank-normalize-features", dest="rank_normalize_features", action="store_false")
    p.add_argument("--rank-normalize-labels", action="store_true", default=True)
    p.add_argument("--no-rank-normalize-labels", dest="rank_normalize_labels", action="store_false")
    p.add_argument("--workdir", type=Path, default=Path("output/realworld_zeroshot"))
    p.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="Device for training/inference (default: auto -> cuda if available else cpu)",
    )
    p.add_argument("--output", type=Path,
                   default=RESULTS_DIR / "realworld_zeroshot.json")
    args = p.parse_args()

    if not args.realworld_cache.exists():
        print(
            f"Error: {args.realworld_cache} not found. Populate it first:\n"
            f"  CACHE_DIR={args.realworld_cache} bash scripts/populate_loso_cache.sh "
            f"realworld_autoware_ros2 realworld_cloud_microservices "
            f"realworld_trainticket realworld_homeassistant realworld_edgex",
            file=sys.stderr,
        )
        return 2

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    homogeneous = args.variant in ("gl_full_cap", "gl_full_qos16_cap", "gl_qos16_prior",
                                   "gl_full_qos16_nfmask", "gl_full_qos_cap", "gl_proj_cap",
                                   "gl_proj_qos16_cap", "gl_proj_qos16_indeg_prior")
    tabular = args.variant in ("tab_gbm", "tab_gbm_qos")
    use_qos = (
        _registry.node_qos_for(args.variant, "loso") if homogeneous or tabular
        else args.variant in ("hgl_qos", "hgl_qos_prior", "hgl_qos_uni", "hgl_proj_qos")
    )
    topo_prior = args.variant == "hgl_qos_prior"
    # Amendment 9: learn and predict on the DEPENDS_ON projection of every graph.
    on_projection = _registry.learns_on_projection(args.variant, "loso")

    synthetic = discover_scenarios(args.synthetic_cache, [])
    real = discover_scenarios(args.realworld_cache, [], min_scenarios=1)
    if on_projection:
        synthetic = [_dependency_bundle(b) for b in synthetic]
    logger.info("Synthetic training corpus: %d scenarios", len(synthetic))
    logger.info("Real-world evaluation corpus: %d systems", len(real))

    # A real system that leaked into the training cache would make this a
    # transductive evaluation while still being reported as zero-shot.
    overlap = {b.scenario_id for b in synthetic} & {b.scenario_id for b in real}
    if overlap:
        print(f"Error: {sorted(overlap)} appear in BOTH caches; "
              "this would not be a zero-shot evaluation.", file=sys.stderr)
        return 2

    per_system: Dict[str, List[Dict[str, Any]]] = {b.scenario_id: [] for b in real}
    t0 = time.time()
    for seed in seeds:
        ckpt = args.workdir / args.variant / f"seed_{seed}"
        ckpt.mkdir(parents=True, exist_ok=True)
        if tabular:
            service = train_once_tabular(
                synthetic, seed, variant=args.variant,
                rank_normalize_features=args.rank_normalize_features,
                rank_normalize_labels=args.rank_normalize_labels,
            )
        elif homogeneous:
            service = train_once_homogeneous(
                synthetic, seed, ckpt, variant=args.variant,
                epochs=args.epochs, layers=args.layers,
                rank_normalize_features=args.rank_normalize_features,
                rank_normalize_labels=args.rank_normalize_labels,
                device=args.device,
            )
        else:
            service = train_once(
                synthetic, seed, ckpt,
                use_qos=use_qos, epochs=args.epochs, layers=args.layers,
                rank_normalize_features=args.rank_normalize_features,
                rank_normalize_labels=args.rank_normalize_labels,
                device=args.device,
                topo_prior=topo_prior,
                use_bidirectional=_registry.bidirectional_for(args.variant),
                hidden_channels=_registry.hidden_for(args.variant, 64, "loso"),
            )
        for b in real:
            try:
                m = score(service, _dependency_bundle(b) if on_projection else b,
                          use_qos=use_qos,
                          population=args.eval_population,
                          rank_normalize_features=args.rank_normalize_features)
            except Exception as exc:                      # noqa: BLE001
                logger.error("  %s seed %d failed: %s", b.scenario_id, seed, exc,
                             exc_info=True)
                continue
            m["seed"] = seed
            per_system[b.scenario_id].append(m)
            logger.info("    %-32s rho=%.4f  hybrid_rho=%.4f  F1@K=%.4f  n=%d",
                        b.scenario_id, m["spearman_rho"], m.get("hybrid_spearman_rho", 0.0),
                        m["f1_at_k"], m["n"])

    summary: Dict[str, Any] = {}
    for sid, runs in per_system.items():
        if not runs:
            summary[sid] = {"n_seeds": 0}
            continue
        rho = [r["spearman_rho"] for r in runs]
        hybrid_rho = [r.get("hybrid_spearman_rho", 0.0) for r in runs]
        f1 = [r["f1_at_k"] for r in runs]
        hybrid_f1 = [r.get("hybrid_f1_at_k", 0.0) for r in runs]
        pos_rho = [r["spearman_rho_positive"] for r in runs if r.get("spearman_rho_positive") is not None]

        bundle = next(b for b in real if b.scenario_id == sid)
        summary[sid] = {
            "n_seeds": len(runs),
            "n_nodes": bundle.n_nodes,
            "n_evaluated": runs[0].get("n"),
            "mean_rho": float(np.mean(rho)),
            "std_rho": float(np.std(rho)),
            "mean_hybrid_rho": float(np.mean(hybrid_rho)),
            "std_hybrid_rho": float(np.std(hybrid_rho)),
            "mean_f1_at_k": float(np.mean(f1)),
            "std_f1_at_k": float(np.std(f1)),
            "mean_hybrid_f1_at_k": float(np.mean(hybrid_f1)),
            "std_hybrid_f1_at_k": float(np.std(hybrid_f1)),
            "mean_rho_positive": float(np.mean(pos_rho)) if pos_rho else None,
            "std_rho_positive": float(np.std(pos_rho)) if pos_rho else None,
            "n_positive": runs[0].get("n_positive"),
            # mean_f1_at_k above is overlap@K: both sets have exactly K members,
            # so precision == recall == F1 there by construction and it measures
            # rank agreement, not identification. The block below is the
            # identification family proper, on the same seeds.
            **_identification_summary(runs),
            # Seed 0's rows. The per-seed spread is already carried as std on
            # every figure above; what these add is the ability to re-stratify.
            "eval_points": runs[0].get("eval_points", []),
            "label_stability": bundle.label_stability,
            "labeler": bundle.labeler,
        }

    # Training-free references on the same oracle, same population, same labels.
    references: Dict[str, Dict[str, Any]] = {}
    for b in real:
        for name, m in score_references(b, population=args.eval_population).items():
            # A reference that could not be computed records why; it is carried
            # through as-is rather than coerced to a number.
            if "spearman_rho" not in m:
                references.setdefault(name, {})[b.scenario_id] = dict(m)
                continue
            references.setdefault(name, {})[b.scenario_id] = {
                "rho": float(m["spearman_rho"]),
                "f1_at_k": float(m["f1_at_k"]),
                "rho_positive": _finite(m.get("spearman_rho_positive")),
                "n_positive": m.get("n_positive"),
                **{k: _finite(m.get(k)) for k in _ID_KEYS},
            }
    ref_means = {}
    for name, per in references.items():
        rhos = [v["rho"] for v in per.values() if "rho" in v]
        ref_means[name] = float(np.mean(rhos)) if len(rhos) == len(per) else None

    scored = [s for s in summary.values() if s.get("n_seeds")]
    mean_rho_all = float(np.mean([s["mean_rho"] for s in scored])) if scored else None
    mean_hybrid_rho_all = float(np.mean([s["mean_hybrid_rho"] for s in scored])) if scored else None

    payload = {
        "variant": args.variant,
        "label": _registry.label(args.variant, harness="loso"),
        "oracle": "I*(v) / FaultInjector",
        "oracle_note": (
            "Scored against I*(v), the oracle the model was trained on. Table 9's "
            "Q(v) and Topo columns are scored against I_comp(v) (FailureSimulator) "
            "and are NOT directly comparable to these numbers."
        ),
        "training_corpus": sorted(b.scenario_id for b in synthetic),
        "eval_population": args.eval_population,
        "seeds": seeds,
        "epochs": args.epochs,
        "layers": args.layers,
        "rank_normalize_features": args.rank_normalize_features,
        "rank_normalize_labels": args.rank_normalize_labels,
        "elapsed_s": time.time() - t0,
        "per_system": summary,
        "references": references,
        "reference_mean_rho": ref_means,
        "reference_note": (
            "Training-free references scored against the same I*(v) labels, "
            "population and node set as the learned model. Topo-QoS is present: "
            "it was previously omitted on the grounds that the cache carried no "
            "QoS edge weights, which was a defect in the projection guard rather "
            "than a property of the data. With that corrected, 48-66% of edges "
            "carry non-unit weights across the five systems."
        ),
        "mean_rho_across_systems": mean_rho_all,
        "mean_hybrid_rho_across_systems": mean_hybrid_rho_all,
        # Five systems averaged into one number, with the spread that average
        # conceals. Reported for the training-free references too, so the
        # comparison down the column carries the same uncertainty statement.
        "bootstrap_ci": bootstrap_over_systems(summary, references),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Stamp the corpus this artifact describes, by content. Without it
    # reconcile_manuscript.py can only compare timestamps, which the
    # byte-identical corpus regeneration routinely invalidates in both
    # directions.
    payload["provenance"] = stamp()
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\n  {payload['label']} zero-shot on real systems "
          f"(oracle: {payload['oracle']})")
    print("  " + "─" * 84)
    print(f"  {'system':<32}{'|V|':>5}{'GNN ρ':>9}{'sd':>7}{'Hybrid ρ':>10}{'sd':>7}{'F1@K':>8}")
    for sid in sorted(summary):
        s = summary[sid]
        if not s.get("n_seeds"):
            print(f"  {sid:<32}{'—':>5}{'failed':>9}")
            continue
        print(f"  {sid:<32}{s['n_nodes']:>5}{s['mean_rho']:>9.4f}"
              f"{s['std_rho']:>7.4f}{s['mean_hybrid_rho']:>10.4f}"
              f"{s['std_hybrid_rho']:>7.4f}{s['mean_f1_at_k']:>8.4f}")
    if mean_rho_all is not None:
        print(f"  {'mean across systems':<32}{'':>5}"
              f"{mean_rho_all:>9.4f}{'':>7}{mean_hybrid_rho_all:>10.4f}")

    if references:
        print(f"\n  Comparison Across Models (Application Stratum, Oracle: I*(v))")
        print("  " + "─" * 84)
        hdr = f"  {'system':<32}"
        for name in sorted(references):
            hdr += f"{name:>12}"
        hdr += f"{'HGT-QoS':>12}{'SaG-Hybrid':>12}"
        print(hdr)
        for sid in sorted(summary):
            row = f"  {sid:<32}"
            for name in sorted(references):
                v = references[name].get(sid) or {}
                row += f"{v['rho']:>12.4f}" if "rho" in v else f"{'—':>12}"
            s_ = summary[sid]
            if s_.get("n_seeds"):
                row += f"{s_['mean_rho']:>12.4f}{s_['mean_hybrid_rho']:>12.4f}"
            else:
                row += f"{'—':>12}{'—':>12}"
            print(row)
        row = f"  {'mean':<32}"
        for name in sorted(references):
            m = ref_means[name]
            row += f"{m:>12.4f}" if m is not None else f"{'—':>12}"
        row += f"{mean_rho_all:>12.4f}{mean_hybrid_rho_all:>12.4f}"
        print(row)
    print(f"\n  Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
