"""Tests for Predict-stage alignment: projection rankers, multi-arch GNNService, and bug fixes."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from saag.analysis.models import StructuralAnalysisResult
from saag.core.layers import AnalysisLayer
from saag.core.metrics import GraphSummary, StructuralMetrics
from saag.infrastructure.memory_repo import MemoryRepository
from saag.prediction.service import PredictionService
from saag.prediction.structural_predictor import (
    InDegPredictor,
    ReachPredictor,
    ReachQoSPredictor,
    STRUCTURAL_PREDICTOR_MODES,
    derive_flow_projection,
    get_structural_predictor,
)
from saag.usecases.predict_graph import PredictGraphUseCase
from saag.usecases.predictive import PredictiveUseCase

TOPOLOGY = Path(__file__).resolve().parent.parent / "data" / "scenarios" / "atm_system_tiny.json"


@pytest.fixture
def pubsub_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for nid, ntype in (
        ("AppA", "Application"),
        ("AppB", "Application"),
        ("AppC", "Application"),
        ("AppD", "Application"),
        ("Lib1", "Library"),
        ("Topic1", "Topic"),
        ("Topic2", "Topic"),
    ):
        g.add_node(nid, type=ntype)
    # AppB subscribes to Topic1 (published by AppA) -> AppB depends on AppA
    g.add_edge("AppA", "Topic1", type="PUBLISHES_TO", weight=1.0, qos_weight=0.9)
    g.add_edge("AppB", "Topic1", type="SUBSCRIBES_TO", weight=1.0, qos_weight=0.9)
    # AppC and AppD subscribe to Topic2 (published by AppB) -> AppC, AppD depend on AppB
    g.add_edge("AppB", "Topic2", type="PUBLISHES_TO", weight=1.0, qos_weight=0.5)
    g.add_edge("AppC", "Topic2", type="SUBSCRIBES_TO", weight=1.0, qos_weight=0.5)
    g.add_edge("AppD", "Topic2", type="SUBSCRIBES_TO", weight=1.0, qos_weight=0.5)
    # AppA uses Lib1 -> AppA depends on Lib1
    g.add_edge("AppA", "Lib1", type="USES", weight=1.0, qos_weight=0.8)
    return g


@pytest.fixture
def structural_result(pubsub_graph: nx.DiGraph) -> StructuralAnalysisResult:
    comps = {
        nid: StructuralMetrics(
            id=nid,
            name=nid,
            type=data["type"],
            betweenness=0.2,
            is_articulation_point=(nid == "AppB"),
        )
        for nid, data in pubsub_graph.nodes(data=True)
        if data["type"] in ("Application", "Library")
    }
    return StructuralAnalysisResult(
        layer=AnalysisLayer.SYSTEM,
        components=comps,
        edges={},
        graph_summary=GraphSummary(layer="system", nodes=len(comps)),
        graph=pubsub_graph,
    )


def test_projection_rankers_match_training_free_suite(pubsub_graph):
    from reproduce.training_free_suite import indeg, reach, reach_qos

    flow = derive_flow_projection(pubsub_graph)

    # InDeg (unnormalized matches exact integer count; normalized preserves rank & [0, 1])
    raw_indeg = InDegPredictor(normalize=False).predict(pubsub_graph)
    assert raw_indeg == indeg(flow)
    norm_indeg = InDegPredictor(normalize=True).predict(pubsub_graph)
    assert max(norm_indeg.values()) == pytest.approx(1.0)
    assert norm_indeg["AppB"] > norm_indeg["AppC"]

    # Reach matches exact fraction
    assert ReachPredictor().predict(pubsub_graph) == pytest.approx(reach(flow))

    # Reach-QoS (unnormalized matches exact value; normalized stays in [0, 1])
    raw_rqos = ReachQoSPredictor(normalize=False).predict(pubsub_graph)
    assert raw_rqos == pytest.approx(reach_qos(flow))
    norm_rqos = ReachQoSPredictor(normalize=True).predict(pubsub_graph)
    assert max(norm_rqos.values()) == pytest.approx(1.0)
    assert norm_rqos["Lib1"] > norm_rqos["AppC"]


def test_get_structural_predictor_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown structural predictor mode"):
        get_structural_predictor("nonexistent")


@pytest.mark.parametrize("mode", STRUCTURAL_PREDICTOR_MODES)
def test_prediction_service_structural_modes_do_not_duplicate_rm_subscores(
    pubsub_graph, structural_result, mode
):
    service = PredictionService(prefer_gnn=False)
    res = service.predict_quality_with_gnn(
        structural_result, graph=pubsub_graph, predictor_mode=mode, diagnose=False
    )
    assert res.prediction_mode == mode
    assert res.node_scores
    for score in res.node_scores.values():
        assert score.reliability_score == 0.0
        assert score.maintainability_score == 0.0


def test_predictive_use_case_forwards_k_and_divergence_threshold(pubsub_graph, structural_result):
    uc = PredictiveUseCase(prefer_gnn=False)
    res = uc.execute(
        layer="system",
        structural_result=structural_result,
        graph=pubsub_graph,
        k=3,
        divergence_threshold=1,
        predictor_mode="dual",
        diagnose=False,
    )
    assert res.prediction_mode == "dual"
    assert res.dual_result.metadata["k"] == 3
    assert res.dual_result.metadata["divergence_threshold"] == 1


def test_predict_graph_use_case_emits_deprecation_warning():
    service = PredictionService(prefer_gnn=False)
    with pytest.warns(DeprecationWarning, match="PredictGraphUseCase is deprecated"):
        PredictGraphUseCase(service)


def test_client_predict_supports_projection_modes():
    from saag import Client

    topology = json.loads(TOPOLOGY.read_text())
    repo = MemoryRepository()
    repo.save_graph(topology, clear=True)
    repo.derive_dependencies()
    client = Client(repo=repo)
    analysis = client.analyze(layer="app")

    for mode in ("indeg", "reach", "reach_qos"):
        pred = client.predict(analysis, mode=mode, diagnose=False)
        assert pred.raw.prediction_mode == mode
        assert pred.raw.node_scores


def test_gnn_service_defaults_and_validation():
    pytest.importorskip("torch_geometric", reason="torch_geometric not installed")
    from saag.prediction.gnn_service import GNNService

    svc = GNNService()
    assert svc.predict_edges is False
    assert svc.model_arch == "hgt"
    assert svc.substrate == "native"

    with pytest.raises(ValueError, match="predict_edges=True is only supported"):
        GNNService(model_arch="homo_scalar", predict_edges=True)

    with pytest.raises(ValueError, match="Unknown model_arch"):
        GNNService(model_arch="invalid_arch")

    with pytest.raises(ValueError, match="Unknown substrate"):
        GNNService(substrate="invalid_substrate")


@pytest.mark.parametrize("model_arch", ["homo_scalar", "homo_gin"])
def test_gnn_service_projection_arch_roundtrip_and_layer_guard(tmp_path, model_arch):
    pytest.importorskip("torch_geometric", reason="torch_geometric not installed")
    from saag import Client
    from saag.core.graph_io import build_graph_from_json
    from saag.prediction.data_preparation import extract_structural_metrics_dict
    from saag.prediction.gnn_service import GNNService

    topology = json.loads(TOPOLOGY.read_text())
    repo = MemoryRepository()
    repo.save_graph(topology, clear=True)
    repo.derive_dependencies()
    client = Client(repo=repo)
    analysis = client.analyze(layer="app")

    graph = build_graph_from_json(topology)
    structural = extract_structural_metrics_dict(analysis.raw.structural)
    labels = {
        node: {"composite": (i % 5) / 4, "reliability": (i % 5) / 4, "maintainability": 0.5}
        for i, node in enumerate(sorted(graph.nodes()))
    }
    ckpt = tmp_path / f"ckpt_{model_arch}"
    svc = GNNService(
        checkpoint_dir=str(ckpt),
        hidden_channels=16,
        num_heads=2,
        num_layers=2,
        model_arch=model_arch,
        edge_dim=16,
        substrate="projection",
    )
    svc.train(
        graph=graph,
        structural_metrics=structural,
        simulation_results=labels,
        seeds=[42],
        num_epochs=3,
        layer="app",
    )

    # Reload and verify config persistence + inference on raw graph (auto-projected)
    reloaded = GNNService.from_checkpoint(str(ckpt), layer="app")
    assert reloaded.model_arch == model_arch
    assert reloaded.substrate == "projection"
    assert reloaded.edge_dim == 16
    pred = reloaded.predict(graph=graph, structural_metrics=structural, mode="gnn")
    assert pred.node_scores

    # PredictionService must enforce checkpoint layer guard (fail loud on mismatch)
    pred_service = PredictionService(gnn_checkpoint_dir=str(ckpt), prefer_gnn=True)
    with pytest.raises(ValueError, match="GNN Layer Mismatch"):
        pred_service.predict_quality_with_gnn(
            analysis.raw.structural, graph=graph, layer="infra", diagnose=False
        )
