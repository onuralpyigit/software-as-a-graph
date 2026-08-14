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
