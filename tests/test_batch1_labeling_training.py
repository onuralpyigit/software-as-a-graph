"""
Regression tests for Batch 1 labeling & training fixes:
- Signed/zero label ordering under normalize_labels_robust (IQR and rank)
- Preventing double normalization and in-place mutation in evaluate_scores
- _collect_samples and _compute_per_type_rho respecting label_mask
- GNNTrainer / evaluate / GNNService multi-node-type training (Broker, Node)
- labels_from_run (i_flow) for non-crash sink faults
- ref_flow_policy k_of_n:K degradation and telemetry_profile policy_overrides
- select_label_stds extraction
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pytest
import torch

from saag.prediction.data_preparation import (
    create_node_splits,
    networkx_to_hetero_data,
    normalize_labels_robust,
)
from saag.prediction.label_artifacts import select_label_stds
from saag.prediction.models.core import NodeCriticalityGNN
from saag.prediction.trainer import (
    GNNTrainer,
    _collect_samples,
    _compute_per_type_rho,
    _metrics_from_preds,
    evaluate,
    evaluate_scores,
)

logging.disable(logging.WARNING)


def _hetero_graph_with_brokers_and_nodes() -> nx.DiGraph:
    g = nx.DiGraph()
    for i in range(4):
        g.add_node(f"A{i}", type="Application")
    for i in range(4):
        g.add_node(f"B{i}", type="Broker")
    for i in range(4):
        g.add_node(f"N{i}", type="Node")
    g.add_node("T0", type="Topic", frequency=10.0)
    for i in range(4):
        g.add_edge(f"A{i}", "T0", type="PUBLISHES_TO")
        g.add_edge(f"A{i}", "T0", type="SUBSCRIBES_TO")
        g.add_edge(f"B{i}", "T0", type="ROUTES")
        g.add_edge(f"A{i}", f"N{i}", type="RUNS_ON")
    return g


# ── 1. normalize_labels_robust signed & zero ordering ────────────────────────

@pytest.mark.parametrize("rank_normalize", [False, True])
def test_normalize_labels_robust_preserves_signed_and_zero_order(rank_normalize):
    g = _hetero_graph_with_brokers_and_nodes()
    # A0=-0.10 (contention relief), A1=0.0 (genuine zero), A2=0.25, A3 unlabelled
    sim = {
        "A0": {"composite": -0.10},
        "A1": {"composite": 0.0},
        "A2": {"composite": 0.25},
    }
    data = networkx_to_hetero_data(g, structural_metrics={}, simulation_results=sim).hetero_data
    normalize_labels_robust(data, rank_normalize=rank_normalize)

    assert getattr(data, "labels_normalized", False) is True
    y = data["Application"].y[:, 0]
    # Strict ordering: negative < genuine zero < positive, all inside (0, 1)
    assert 0.0 < y[0].item() < y[1].item() < y[2].item() < 1.0
    # Unlabelled node A3 stays exactly 0.0
    assert y[3].item() == 0.0


def test_normalize_labels_robust_explicit_use_label_mask():
    g = _hetero_graph_with_brokers_and_nodes()
    sim = {
        "A0": {"composite": 0.0},
        "A1": {"composite": 0.2},
        "A2": {"composite": 0.8},
    }
    data = networkx_to_hetero_data(g, structural_metrics={}, simulation_results=sim).hetero_data
    normalize_labels_robust(data, rank_normalize=True, use_label_mask=True)

    y = data["Application"].y[:, 0]
    assert 0.0 < y[0].item() < y[1].item() < y[2].item() < 1.0
    assert y[3].item() == 0.0


# ── 2. evaluate_scores: no double normalization & no in-place mutation ───────

def test_evaluate_scores_no_mutation_and_no_double_normalization():
    y_true = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0], [0.9, 0.0, 0.0]])
    y_true_orig = y_true.copy()
    y_pred = y_true.copy()

    m_no_norm = evaluate_scores(y_pred, y_true, normalize_targets=False)
    np.testing.assert_array_equal(y_true, y_true_orig)
    assert m_no_norm.rmse == pytest.approx(0.0, abs=1e-9)
    assert m_no_norm.mae == pytest.approx(0.0, abs=1e-9)
    assert m_no_norm.n_critical_in_truth == 1

    m_with_norm = evaluate_scores(y_pred, y_true, normalize_targets=True)
    np.testing.assert_array_equal(y_true, y_true_orig)
    # With normalization, median-centered sigmoid pushes >= 2 items to >= 0.5 and RMSE > 0
    assert m_with_norm.rmse > 0.05


def test_metrics_from_preds_skips_double_norm_when_stamped():
    g = _hetero_graph_with_brokers_and_nodes()
    sim = {f"A{i}": {"composite": 0.1 * (i + 1)} for i in range(4)}
    data = networkx_to_hetero_data(g, structural_metrics={}, simulation_results=sim).hetero_data
    normalize_labels_robust(data)
    data["Application"].test_mask = torch.ones(4, dtype=torch.bool)

    # Perfect predictions matching the normalized labels should yield RMSE == 0
    node_preds = {"Application": data["Application"].y.clone()}
    metrics = _metrics_from_preds(node_preds, data, "test_mask")
    assert metrics.rmse == pytest.approx(0.0, abs=1e-6)


# ── 3. _collect_samples & _compute_per_type_rho respect label_mask ───────────

def test_collect_samples_and_per_type_rho_respect_label_mask():
    g = _hetero_graph_with_brokers_and_nodes()
    # A0 is genuinely 0.0 (labelled), A1=0.2, A2=0.8; A3 is unlabelled
    sim = {
        "A0": {"composite": 0.0},
        "A1": {"composite": 0.2},
        "A2": {"composite": 0.8},
    }
    data = networkx_to_hetero_data(g, structural_metrics={}, simulation_results=sim).hetero_data
    data["Application"].test_mask = torch.ones(4, dtype=torch.bool)
    preds = {"Application": torch.tensor([[0.1, 0.0, 0.0], [0.4, 0.0, 0.0], [0.9, 0.0, 0.0], [0.99, 0.0, 0.0]])}

    p, t = _collect_samples(preds, data, "test_mask")
    # Should collect A0, A1, A2 (3 labelled nodes), excluding unlabelled A3
    assert p.shape[0] == 3
    assert t[0, 0] == pytest.approx(0.0)

    per_type = _compute_per_type_rho(preds, data, "test_mask")
    # A3 (unlabelled, pred=0.99, y=0.0) would destroy rho if included; with label_mask rho == 1.0
    assert per_type["Application"] == pytest.approx(1.0)


# ── 4. Multi-type GNN training (Broker & Node receive gradients) ─────────────

def test_trainer_supports_broker_and_node_targets(tmp_path):
    g = _hetero_graph_with_brokers_and_nodes()
    sim = {
        **{f"A{i}": {"composite": 0.1 * (i + 1)} for i in range(4)},
        **{f"B{i}": {"composite": 0.2 * (i + 1)} for i in range(4)},
        **{f"N{i}": {"composite": 0.15 * (i + 1)} for i in range(4)},
    }
    conv = networkx_to_hetero_data(g, structural_metrics={}, simulation_results=sim)
    data = conv.hetero_data
    normalize_labels_robust(data)
    create_node_splits(data, train_ratio=0.5, val_ratio=0.25, seed=42)

    model = NodeCriticalityGNN(
        metadata=data.metadata(),
        hidden_channels=16,
        num_heads=2,
        num_layers=1,
    )
    trainer = GNNTrainer(
        model=model,
        checkpoint_dir=str(tmp_path),
        num_epochs=2,
        patience=2,
        target_node_types={"Broker", "Node"},
    )
    # Run one loss computation on Broker/Node only
    x_dict = {nt: data[nt].x for nt in data.node_types}
    ei_dict = {rel: data[rel].edge_index for rel in data.edge_types}
    ea_dict = {rel: data[rel].edge_attr for rel in data.edge_types}
    preds = model(x_dict, ei_dict, ea_dict)
    loss = trainer._node_loss(preds, data)
    assert loss.item() > 0.0

    # Evaluate on Broker & Node
    m = evaluate(model, data, "test_mask", torch.device("cpu"), target_node_types={"Broker", "Node"})
    assert m.rmse >= 0.0


# ── 5. Runtime labeling: i_flow non-crash sink & ref_flow_policy ─────────────

def test_i_flow_non_crash_sink_fault_is_partial():
    pytest.importorskip("simpy")
    from saag.simulation.runtime.labels import runtime_labels

    g = nx.DiGraph()
    g.add_node("A", type="Application", app_type="sensor")
    g.add_node("S", type="Application", app_type="actuator", criticality="HIGH")
    g.add_node("t1", type="Topic", frequency=10.0)
    g.add_edge("A", "t1", type="PUBLISHES_TO")
    g.add_edge("S", "t1", type="SUBSCRIBES_TO")

    out = runtime_labels(
        g,
        candidates=["S"],
        seeds=[42],
        settings={"duration": 20.0, "deployment_model": "none"},
        fault={"mode": "transient_crash", "duration_s": 2.0},
    )
    # S only crashed for 2s out of the 10s post-window, so i_flow must be strictly < 0.8 (not 1.0)
    i_flow = out["records"]["S"]["i_flow"]["mean"]
    assert 0.05 < i_flow < 0.8


def test_ref_flow_policy_k_of_n_and_policy_overrides():
    from saag.simulation.runtime.labels import _Topology, ref_flow_policy

    g = nx.DiGraph()
    g.add_node("S1", type="Application", app_type="sensor")
    g.add_node("S2", type="Application", app_type="sensor")
    g.add_node("S3", type="Application", app_type="sensor")
    g.add_node("M", type="Application", app_type="processor", trigger="k_of_n:2")
    g.add_node("K", type="Application", app_type="actuator")
    for t in ("t1", "t2", "t3", "tout"):
        g.add_node(t, type="Topic", frequency=10.0)
    g.add_edge("S1", "t1", type="PUBLISHES_TO")
    g.add_edge("S2", "t2", type="PUBLISHES_TO")
    g.add_edge("S3", "t3", type="PUBLISHES_TO")
    for t in ("t1", "t2", "t3"):
        g.add_edge("M", t, type="SUBSCRIBES_TO")
    g.add_edge("M", "tout", type="PUBLISHES_TO")
    g.add_edge("K", "tout", type="SUBSCRIBES_TO")

    topo = _Topology(g)
    # 3 inputs, k=2: losing S1 leaves 2 fresh >= 2 -> 0.0 loss
    assert ref_flow_policy(topo, g, "S1") == pytest.approx(0.0)

    # With k_of_n:3 via policy_overrides: losing S1 leaves 2/3 -> 1/3 loss at sink K
    assert ref_flow_policy(topo, g, "S1", policy_overrides={"M": "k_of_n:3"}) == pytest.approx(1 / 3)

    # With all_of via policy_overrides: losing S1 stalls M completely -> 1.0 loss at sink K
    assert ref_flow_policy(topo, g, "S1", policy_overrides={"M": "all_of"}) == pytest.approx(1.0)


# ── 6. select_label_stds ─────────────────────────────────────────────────────

def test_select_label_stds():
    fi_raw = {
        "labeler": "FaultInjector",
        "records": {
            "A0": {"impact_score": 0.5, "impact_score_std": 0.02},
            "A1": {"impact_score": 0.1},
        },
    }
    assert select_label_stds(fi_raw) == {"A0": 0.02}

    rt_raw = {
        "labeler": "MessageFlowSimulator/runtime",
        "records": {
            "A0": {"i_flow": {"mean": 0.8, "std": 0.05}},
            "A1": {"i_flow": None},
        },
    }
    assert select_label_stds(rt_raw, label="i_flow") == {"A0": 0.05}
