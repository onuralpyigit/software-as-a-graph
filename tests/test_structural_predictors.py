"""
tests/test_structural_predictors.py
===================================
Tests for TopoPredictor, TopoQoSPredictor, DualEnginePredictor,
and temperature-scaled CriticalityLoss.
"""

import pytest
import networkx as nx
import torch

from saag.prediction.structural_predictor import (
    TopoPredictor,
    TopoQoSPredictor,
    DualEnginePredictor,
    derive_flow_projection,
)
from saag.prediction.models.core import CriticalityLoss
from saag.prediction.service import PredictionService
from saag.analysis.models import StructuralAnalysisResult


@pytest.fixture
def sample_pubsub_graph():
    """Create a multi-application pub-sub graph with varied QoS."""
    g = nx.DiGraph()
    # Applications
    g.add_node("AppA", type="Application")
    g.add_node("AppB", type="Application")
    g.add_node("AppC", type="Application")
    g.add_node("AppD", type="Application")
    # Topics
    g.add_node("Topic1", type="Topic")
    g.add_node("Topic2", type="Topic")
    # Library
    g.add_node("Lib1", type="Library")

    # PubSub edges
    g.add_edge("AppA", "Topic1", type="PUBLISHES_TO", weight=1.0, qos_weight=1.5)
    g.add_edge("AppB", "Topic1", type="SUBSCRIBES_TO", weight=1.0, qos_weight=1.5)
    g.add_edge("AppB", "Topic2", type="PUBLISHES_TO", weight=1.0, qos_weight=0.5)
    g.add_edge("AppC", "Topic2", type="SUBSCRIBES_TO", weight=1.0, qos_weight=0.5)
    g.add_edge("AppD", "Topic2", type="SUBSCRIBES_TO", weight=1.0, qos_weight=0.5)
    g.add_edge("AppA", "Lib1", type="USES", weight=1.0, qos_weight=1.0)
    return g


def test_derive_flow_projection(sample_pubsub_graph):
    flow_g = derive_flow_projection(sample_pubsub_graph)
    assert flow_g.number_of_nodes() > 0
    # AppB subscribes to Topic1 published by AppA -> AppB depends on AppA
    assert flow_g.has_edge("AppB", "AppA")
    # AppC subscribes to Topic2 published by AppB -> AppC depends on AppB
    assert flow_g.has_edge("AppC", "AppB")
    # AppD subscribes to Topic2 published by AppB -> AppD depends on AppB
    assert flow_g.has_edge("AppD", "AppB")
    # AppA uses Lib1 -> AppA depends on Lib1
    assert flow_g.has_edge("AppA", "Lib1")


def test_topo_predictor(sample_pubsub_graph):
    predictor = TopoPredictor()
    scores = predictor.predict(sample_pubsub_graph)
    assert len(scores) == 5  # 4 apps + 1 lib
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    # AppA or AppB should be higher than leaf nodes AppC, AppD
    assert scores["AppB"] > scores["AppC"] or scores["AppA"] > scores["AppC"]


def test_topo_qos_predictor(sample_pubsub_graph):
    predictor = TopoQoSPredictor()
    scores = predictor.predict(sample_pubsub_graph)
    assert len(scores) == 5
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_dual_engine_consensus_and_divergence(sample_pubsub_graph):
    gnn_scores = {
        "AppA": 0.9,
        "AppB": 0.8,
        "AppC": 0.2,
        "AppD": 0.1,
        "Lib1": 0.05,
    }
    dual_pred = DualEnginePredictor(divergence_threshold=2)
    res = dual_pred.evaluate_dual(gnn_scores, sample_pubsub_graph, k=2)

    assert len(res.combined_scores) == 5
    assert len(res.gnn_ranks) == 5
    assert len(res.topo_ranks) == 5
    # Consensus top-K should be a subset of top-K
    assert isinstance(res.consensus_top_k, list)
    assert isinstance(res.divergence_escalations, list)


def test_criticality_loss_temperature_scaling():
    pred = torch.tensor([[0.8, 0.5, 0.3], [0.4, 0.2, 0.1]], requires_grad=True)
    target = torch.tensor([[0.9, 0.5, 0.3], [0.1, 0.2, 0.1]])
    mask = torch.tensor([True, True])

    # Standard loss (tau = 1.0)
    loss_fn_1 = CriticalityLoss(temperature=1.0)
    total_1, comps_1 = loss_fn_1(pred, target, mask)

    # Temperature scaled loss (tau = 0.1)
    loss_fn_sharp = CriticalityLoss(temperature=0.1)
    total_sharp, comps_sharp = loss_fn_sharp(pred, target, mask)

    assert total_1.item() > 0
    assert total_sharp.item() > 0
    # Ranking loss component should differ when temperature changes
    assert comps_1["ranking"] != comps_sharp["ranking"]

    # Test backward pass
    total_sharp.backward()
    assert pred.grad is not None


def test_prediction_service_structural_and_dual(sample_pubsub_graph):
    from saag.core.metrics import StructuralMetrics, GraphSummary
    from saag.core.layers import AnalysisLayer

    service = PredictionService(prefer_gnn=False)
    comps = {
        "AppA": StructuralMetrics(id="AppA", name="AppA", type="Application", betweenness=0.5, is_articulation_point=False),
        "AppB": StructuralMetrics(id="AppB", name="AppB", type="Application", betweenness=0.8, is_articulation_point=True),
        "AppC": StructuralMetrics(id="AppC", name="AppC", type="Application", betweenness=0.1, is_articulation_point=False),
        "AppD": StructuralMetrics(id="AppD", name="AppD", type="Application", betweenness=0.1, is_articulation_point=False),
        "Lib1": StructuralMetrics(id="Lib1", name="Lib1", type="Library", betweenness=0.0, is_articulation_point=False),
    }
    struct_res = StructuralAnalysisResult(
        layer=AnalysisLayer.SYSTEM,
        components=comps,
        edges={},
        graph_summary=GraphSummary(layer="system", nodes=5),
        graph=sample_pubsub_graph,
    )

    # Test Topo-QoS mode
    topo_res = service.predict_structural(struct_res, graph=sample_pubsub_graph, use_qos=True, diagnose=False)
    assert topo_res.prediction_mode == "topo_qos"
    assert len(topo_res.node_scores) > 0

    # Test Dual mode
    dual_res = service.predict_dual(struct_res, graph=sample_pubsub_graph, diagnose=False, k=2)
    assert dual_res.prediction_mode == "dual"
    assert hasattr(dual_res, "dual_result")
    assert len(dual_res.dual_result.combined_scores) > 0
