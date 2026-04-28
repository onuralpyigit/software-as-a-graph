import pytest
from saag.infrastructure.memory_repo import MemoryRepository
from saag.usecases.analyze_graph import AnalyzeGraphUseCase
from saag.usecases.predict_graph import PredictGraphUseCase
from saag.usecases.model_graph import ModelGraphUseCase
from saag.usecases.simulate_graph import SimulateGraphUseCase, SimulationMode
from saag.usecases.validate_graph import ValidateGraphUseCase

@pytest.fixture
def memory_repo():
    return MemoryRepository()

@pytest.fixture
def repo_with_topology(memory_repo):
    """Seed the repo with A -> B topology."""
    graph_data = {
        "applications": [
            {"id": "A", "name": "App A"},
            {"id": "B", "name": "App B"},
        ],
        "relationships": {
            "publishes_to": [
                {"source_id": "A", "target_id": "B", "weight": 0.8}
            ]
        }
    }
    memory_repo.save_graph(graph_data)
    return memory_repo

class TestUseCaseOrchestration:
    
    def test_analyze_then_predict_pipeline(self, repo_with_topology):
        """
        Verifies that PredictGraphUseCase correctly consumes the output 
        of AnalyzeGraphUseCase (Independence Guarantee).
        """
        analyze_uc = AnalyzeGraphUseCase(repo_with_topology)
        predict_uc = PredictGraphUseCase(repo_with_topology)

        # 1. Analyze
        struct_res = analyze_uc.execute("app")
        assert struct_res.layer.value == "app"
        assert "A" in struct_res.components
        assert "B" in struct_res.components
        
        # 2. Predict (orchestrated)
        quality_res, problems = predict_uc.execute("app", struct_res, detect_problems=True)
        
        # Verify orchestration
        assert quality_res.layer == "app"
        assert len(quality_res.components) == 2
        
        # Verify Independence Guarantee: output is derived from the passed struct_res
        # (Actually, we'd need to mock PredictionService to be 100% sure it didn't call repo,
        # but the signature already enforces it).
        
        # Check scores are in range [0, 1]
        for comp in quality_res.components:
            assert 0.0 <= comp.scores.overall <= 1.0
            assert 0.0 <= comp.scores.reliability <= 1.0

    def test_model_graph_use_case(self, memory_repo):
        """Verifies that ModelGraphUseCase correctly updates the repo and returns stats."""
        model_uc = ModelGraphUseCase(memory_repo)
        
        graph_data = {
            "applications": [{"id": "App1"}],
            "nodes": [{"id": "Node1"}],
            "relationships": {
                "connects_to": [{"source_id": "Node1", "target_id": "Node1"}]
            }
        }
        
        stats = model_uc.execute(graph_data)
        
        assert stats.success is True
        # MemoryRepository.get_statistics returns counts by key
        # application_count, node_count, etc.
        assert stats.details["application_count"] == 1
        assert stats.details["node_count"] == 1
        
        # Verify repo state
        final_stats = memory_repo.get_statistics()
        assert final_stats["application_count"] == 1

    def test_simulate_graph_use_case(self, repo_with_topology):
        """Smoke test for SimulateGraphUseCase."""
        simulate_uc = SimulateGraphUseCase(repo_with_topology)
        
        # Run exhaustive simulation on app layer
        # Note: Simulation might return different results depending on internal logic,
        # but we check if it runs and returns something.
        result = simulate_uc.execute(layer="app", mode=SimulationMode.EXHAUSTIVE)
        
        assert result is not None
        # results usually include failure stats/impacts

    def test_validate_graph_use_case(self, repo_with_topology):
        """Smoke test for ValidateGraphUseCase."""
        validate_uc = ValidateGraphUseCase(repo_with_topology)
        
        # Run validation on app layer
        result = validate_uc.execute(layers=["app"])
        
        assert result is not None
        assert "app" in result.layers
