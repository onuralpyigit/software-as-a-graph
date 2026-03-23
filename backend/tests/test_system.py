import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools.generation import generate_graph
from src.adapters import create_repository
from src.analysis import AnalysisService
from src.simulation import SimulationService
from src.validation import ValidationService
from src.prediction import GNNService

@pytest.mark.slow
def test_system_end_to_end(tmp_path):
    """
    A proper system test that replaces the legacy subprocess-based CLI scripts.
    This exercises the actual Python API backend flow end-to-end.
    If Neo4j is unavailable, we gracefully skip the test.
    """
    # 1. Generate local graph
    graph_data = generate_graph(scale="tiny", seed=42)
    assert "nodes" in graph_data
    assert len(graph_data["nodes"]) > 0

    # 2. Try to connect to a local Neo4j DB (assumed for slow tests)
    # If connection fails, pytest.skip()
    try:
        repo = create_repository(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )
        repo.save_graph(graph_data, clear=True)
    except Exception as e:
        pytest.skip(f"Skipping system test: could not connect to Neo4j database or save graph ({e})")
        return

    try:
        # 3. Analyze
        analysis_service = AnalysisService(repo, use_ahp=False)
        analysis_result = analysis_service.analyze_layer("app")
        assert analysis_result is not None
        assert len(analysis_result.quality.components) > 0

        # 4. Simulate
        simulation_service = SimulationService(repo)
        sim_results = simulation_service.run_failure_simulation_exhaustive(layer="app")
        assert len(sim_results) > 0

        # 5. Validate
        validation_service = ValidationService(analysis_service, simulation_service, ndcg_k=5)
        val_result = validation_service.validate_single_layer("app")
        assert val_result is not None

        # 6. Predict (GNN)
        from src.prediction import extract_structural_metrics_dict, extract_rmav_scores_dict, extract_simulation_dict
        structural_dict = extract_structural_metrics_dict(analysis_result.structural)
        rmav_dict       = extract_rmav_scores_dict(analysis_result.quality)
        simulation_dict = extract_simulation_dict(sim_results)

        gnn_service = GNNService(
            hidden_channels=16,
            num_heads=1,
            num_layers=1,
            dropout=0.1,
            predict_edges=False,
            checkpoint_dir=str(tmp_path / "gnn_checkpoint"),
        )
        
        # We only train for 2 epochs in a test to ensure it runs fast
        train_result = gnn_service.train(
            graph=analysis_result.graph,
            structural_metrics=structural_dict,
            simulation_results=simulation_dict,
            rmav_scores=rmav_dict,
            num_epochs=2,
            lr=1e-3,
        )
        assert train_result is not None
        top_critical = train_result.top_critical_nodes(3)
        assert isinstance(top_critical, list)

    finally:
        repo.close()
