"""
Tests for cli/train_graph.py's DB-sourced RM computation path.

Regression coverage for two related defects: (1) --structural + --simulated
supplied as files with no --rm never connected to Neo4j to compute RM
scores, because the outer guard only checked structural_dict/simulation_dict;
(2) even when the inner branch did run, it read layer_result.quality, which
AnalysisService.analyze_layer() never populates (only .structural), so RM
scores were always {}.
"""
import json
import sys
from unittest.mock import MagicMock, patch

import cli.train_graph as train_graph
from saag.analysis.models import QualityAnalysisResult
from saag.core.metrics import ComponentQuality, QualityLevels, QualityScores, StructuralMetrics


def _quality_result() -> QualityAnalysisResult:
    comp = ComponentQuality(
        id="App1", type="Application",
        scores=QualityScores(overall=0.7, reliability=0.6, maintainability=0.5,
                              fault_tolerance=0.45, availability=0.4),
        levels=QualityLevels(),
        structural=StructuralMetrics(id="App1", name="App1", type="Application"),
    )
    return QualityAnalysisResult(
        timestamp="t", layer="app", context="test",
        components=[comp], edges=[],
        classification_summary=MagicMock(),
    )


def test_rm_scores_computed_when_structural_and_simulated_given_without_rm(tmp_path):
    structural_path = tmp_path / "structural.json"
    structural_path.write_text(json.dumps({"App1": {"pagerank": 0.1}}))
    simulated_path = tmp_path / "simulated.json"
    simulated_path.write_text(json.dumps({"App1": {"composite": 0.5}}))

    layer_result = MagicMock()
    mock_analysis_service = MagicMock()
    mock_analysis_service.analyze_layer.return_value = layer_result

    mock_client = MagicMock()
    mock_client.repo = MagicMock()

    mock_prediction_service = MagicMock()
    mock_prediction_service.predict_quality.return_value = _quality_result()

    mock_train_result = MagicMock(
        gnn_metrics=None,
        summary=MagicMock(return_value={}),
        top_critical_nodes=MagicMock(return_value=[]),
    )
    mock_gnn_instance = MagicMock()
    mock_gnn_instance.train.return_value = mock_train_result

    argv = [
        "train_graph.py", "--layer", "app",
        "--structural", str(structural_path),
        "--simulated", str(simulated_path),
    ]

    with patch.object(sys, "argv", argv), \
         patch("cli.train_graph.ConsoleDisplay"), \
         patch("saag.Client", return_value=mock_client), \
         patch("saag.analysis.AnalysisService", return_value=mock_analysis_service), \
         patch("saag.simulation.SimulationService"), \
         patch("saag.prediction.service.PredictionService", return_value=mock_prediction_service), \
         patch("saag.prediction.GNNService", return_value=mock_gnn_instance):
        # saag.prediction.GNNService is lazily resolved via module __getattr__,
        # which caches the *real* class into saag.prediction's own namespace on
        # first access anywhere in the process (see saag/prediction/__init__.py).
        # Once cached — which happens the moment any other GNN test in the
        # suite runs first — patching the submodule's saag.prediction.
        # gnn_service.GNNService no longer has any effect, since attribute
        # lookup is satisfied by the cached value before __getattr__ ever runs.
        # Patching the package-level name directly works either way.
        train_graph.main()

    # C11: PredictionService.predict_quality must actually run — not read
    # layer_result.quality (always None) as extract_rm_scores_dict's input.
    mock_prediction_service.predict_quality.assert_called_once_with(layer_result.structural)

    # C13: the DB branch must trigger even though structural_dict and
    # simulation_dict were both already file-supplied, because rm_dict
    # alone was still missing.
    mock_analysis_service.analyze_layer.assert_called_once_with("app")

    # The RM scores computed above must actually reach GNNService.train(),
    # non-empty and keyed by the real component id.
    train_kwargs = mock_gnn_instance.train.call_args.kwargs
    assert train_kwargs["rm_scores"] == {
        "App1": {
            "overall": 0.7, "reliability": 0.6, "maintainability": 0.5,
            "fault_tolerance": 0.45, "availability": 0.4,
        }
    }


def _write_file_only_inputs(tmp_path):
    """--structural/--rm/--simulated as files for every node of atm_system."""
    from pathlib import Path
    from saag.core.graph_io import load_graph
    nodes = list(load_graph(Path("data/scenarios/atm_system.json")).nodes)
    paths = {}
    for name in ("structural", "rm", "simulated"):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps({n: {} for n in nodes}))
    return paths


def _run_main_capturing_graph(argv):
    """Run main() with GNNService mocked; return the graph passed to train()."""
    from saag.prediction.data_preparation import networkx_to_hetero_data
    mock_gnn_instance = MagicMock()
    mock_gnn_instance.train.return_value = MagicMock(
        gnn_metrics=None,
        summary=MagicMock(return_value={}),
        top_critical_nodes=MagicMock(return_value=[]),
    )
    with patch.object(sys, "argv", argv), \
         patch("cli.train_graph.ConsoleDisplay"), \
         patch("saag.Client") as mock_client, \
         patch("saag.prediction.GNNService", return_value=mock_gnn_instance):
        train_graph.main()
    mock_client.assert_not_called()  # file-only: Neo4j is never contacted
    kwargs = mock_gnn_instance.train.call_args.kwargs
    return networkx_to_hetero_data(
        kwargs["graph"], kwargs["structural_metrics"],
        kwargs["simulation_results"], kwargs["rm_scores"],
    ).hetero_data


def test_file_only_path_loads_topology_with_relation_types(tmp_path):
    # Regression: with every input file-supplied, the graph used to be
    # rebuilt as structural-metrics keys typed "Application" with no edges —
    # "1 node types, 0 relation types" — and hetero_qos crashed in
    # torch.cat() on the empty relation list.
    paths = _write_file_only_inputs(tmp_path)
    data = _run_main_capturing_graph([
        "train_graph.py", "--layer", "system",
        "--input", "data/scenarios/atm_system.json",
        "--structural", str(paths["structural"]),
        "--rm", str(paths["rm"]),
        "--simulated", str(paths["simulated"]),
    ])
    assert len(data.edge_types) > 0
    assert len(data.node_types) > 1


def test_file_only_path_without_input_exits(tmp_path):
    paths = _write_file_only_inputs(tmp_path)
    argv = [
        "train_graph.py", "--layer", "system",
        "--structural", str(paths["structural"]),
        "--rm", str(paths["rm"]),
        "--simulated", str(paths["simulated"]),
    ]
    with patch.object(sys, "argv", argv), \
         patch("cli.train_graph.ConsoleDisplay"), \
         patch("saag.Client") as mock_client:
        try:
            train_graph.main()
        except SystemExit as exc:
            assert exc.code == 1
        else:
            raise AssertionError("expected SystemExit when --input is missing")
    mock_client.assert_not_called()


def test_multi_scenario_loads_topology_and_skips_dirs_without_it(tmp_path, monkeypatch):
    import shutil
    paths = _write_file_only_inputs(tmp_path)
    work = tmp_path / "work"
    for name, has_topology in (("with_results", True), ("without_results", False)):
        d = work / "output" / name
        d.mkdir(parents=True)
        shutil.copy(paths["structural"], d / "structural_metrics.json")
        shutil.copy(paths["simulated"], d / "failure_impact.json")
        if has_topology:
            shutil.copy("data/scenarios/atm_system.json", d / "topology.json")
    topology = str((work / "output" / "with_results" / "topology.json").resolve())
    monkeypatch.chdir(work)

    mock_gnn_instance = MagicMock()
    mock_gnn_instance.train.return_value = MagicMock(
        gnn_metrics=None,
        summary=MagicMock(return_value={}),
        top_critical_nodes=MagicMock(return_value=[]),
    )
    argv = [
        "train_graph.py", "--layer", "system", "--multi-scenario",
        "--input", topology,
        "--structural", str(paths["structural"]),
        "--rm", str(paths["rm"]),
        "--simulated", str(paths["simulated"]),
    ]
    with patch.object(sys, "argv", argv), \
         patch("cli.train_graph.ConsoleDisplay"), \
         patch("saag.Client"), \
         patch("saag.prediction.GNNService", return_value=mock_gnn_instance):
        train_graph.main()

    inductive = mock_gnn_instance.train.call_args.kwargs["inductive_graphs"]
    assert len(inductive) == 1  # the dir without topology.json is skipped
    assert len(inductive[0].edge_types) > 0
