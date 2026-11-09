"""Client.predict(mode="gnn") must run the GNN, not fall back to RM.

The GNN is trained on the raw structural graph (typed nodes, PUBLISHES_TO /
SUBSCRIBES_TO / ROUTES / ...). Client.predict used to hand it the layer's
derived DEPENDS_ON analysis graph instead, which carries none of those edge
types: the forward pass raised, PredictionService caught it and returned RM
scores, and every caller of the SDK got RM numbers under a GNN label.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch_geometric", reason="torch_geometric not installed")

from saag import Client
from saag.core.graph_io import build_graph_from_json
from saag.infrastructure.memory_repo import MemoryRepository
from saag.prediction.data_preparation import extract_structural_metrics_dict
from saag.prediction.gnn_service import GNNService

TOPOLOGY = Path(__file__).resolve().parent.parent / "data" / "scenarios" / "atm_system_tiny.json"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A client on the tiny ATM system and a GNN checkpoint trained on its raw graph."""
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
    checkpoint = tmp_path_factory.mktemp("gnn_ckpt")
    GNNService(
        checkpoint_dir=str(checkpoint), hidden_channels=16, num_heads=2, num_layers=2,
        predict_edges=False,
    ).train(
        graph=graph, structural_metrics=structural, simulation_results=labels,
        seeds=[42], num_epochs=3, layer="app",
    )
    return client, analysis, graph, structural, checkpoint


def test_client_predict_gnn_runs_the_gnn(trained):
    client, analysis, _graph, _structural, checkpoint = trained
    result = client.predict(analysis, mode="gnn", gnn_checkpoint=str(checkpoint), diagnose=False)
    assert result.raw.prediction_mode == "gnn_only"


def test_client_predict_gnn_matches_direct_inference(trained):
    """The client must feed the model the same graph a direct GNNService call does."""
    client, analysis, graph, structural, checkpoint = trained
    via_client = client.predict(analysis, mode="gnn", gnn_checkpoint=str(checkpoint), diagnose=False)
    direct = GNNService.from_checkpoint(str(checkpoint)).predict(
        graph=graph, structural_metrics=structural, mode="gnn"
    )
    assert via_client.raw.node_scores
    for node, score in via_client.raw.node_scores.items():
        assert score.composite_score == pytest.approx(
            direct.node_scores[node].composite_score, abs=1e-6
        )


def test_client_predict_gnn_reports_the_analysed_layer(trained):
    """The GNN scores Topics, Brokers and Nodes too; an app-layer call must not return them."""
    client, analysis, _graph, _structural, checkpoint = trained
    result = client.predict(analysis, mode="gnn", gnn_checkpoint=str(checkpoint), diagnose=False)
    layer = analysis.raw.structural.components
    assert set(result.raw.node_scores) == set(layer)
    assert {c.id: c.type for c in result.all_components} == {i: m.type for i, m in layer.items()}


def test_client_predict_without_a_checkpoint_argument_returns_rm(trained, monkeypatch):
    """No default directory is searched: a checkpoint left in output/ must not change the result."""
    client, analysis, _graph, _structural, checkpoint = trained
    default = checkpoint.parent / "output" / "gnn_checkpoints"
    default.parent.mkdir()
    default.symlink_to(checkpoint, target_is_directory=True)
    monkeypatch.chdir(checkpoint.parent)
    result = client.predict(analysis, diagnose=False)
    assert result.raw.prediction_mode == "rm"
