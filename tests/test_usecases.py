import pytest
from saag.infrastructure.memory_repo import MemoryRepository
from saag.usecases.analyze_graph import AnalyzeGraphUseCase
from saag.usecases.predict_graph import PredictGraphUseCase
from saag.usecases.model_graph import ModelGraphUseCase
from saag.usecases.simulate_graph import SimulateGraphUseCase, SimulationMode
from saag.usecases.validate_graph import ValidateGraphUseCase

from saag.analysis.service import AnalysisService
from saag.prediction.service import PredictionService
from saag.simulation.service import SimulationService
from saag.validation.service import ValidationService


@pytest.fixture
def memory_repo():
    return MemoryRepository()


@pytest.fixture
def repo_with_topology(memory_repo):
    """Seed the repo with a valid App A -> Topic T -> App B topology hosted on NodeMain."""
    graph_data = {
        "applications": [
            {"id": "A", "name": "App A", "role": "pub"},
            {"id": "B", "name": "App B", "role": "sub"},
        ],
        "brokers": [
            {"id": "BrokerMain", "name": "Broker Main"}
        ],
        "nodes": [
            {"id": "NodeMain", "name": "Node Main"}
        ],
        "topics": [
            {"id": "T", "name": "Topic T", "size": 1024}
        ],
        "relationships": {
            "publishes_to": [
                {"source": "A", "target": "T", "weight": 0.8}
            ],
            "subscribes_to": [
                {"source": "B", "target": "T", "weight": 0.8}
            ],
            "routes": [
                {"source": "BrokerMain", "target": "T", "weight": 1.0}
            ],
            "runs_on": [
                {"source": "A", "target": "NodeMain", "weight": 1.0},
                {"source": "BrokerMain", "target": "NodeMain", "weight": 1.0}
            ]
        }
    }
    memory_repo.save_graph(graph_data)
    return memory_repo


@pytest.fixture
def analysis_service(repo_with_topology):
    return AnalysisService(repo_with_topology)


@pytest.fixture
def prediction_service():
    return PredictionService()


@pytest.fixture
def simulation_service(repo_with_topology):
    return SimulationService(repo_with_topology)


@pytest.fixture
def validation_service(analysis_service, prediction_service, simulation_service):
    return ValidationService(
        analysis_service=analysis_service,
        prediction_service=prediction_service,
        simulation_service=simulation_service,
    )


class TestUseCaseOrchestration:

    def test_analyze_then_predict_pipeline(self, analysis_service, prediction_service):
        """
        Verifies that PredictGraphUseCase correctly consumes the output
        of AnalyzeGraphUseCase (Independence Guarantee).
        """
        analyze_uc = AnalyzeGraphUseCase(analysis_service)
        predict_uc = PredictGraphUseCase(prediction_service)

        # 1. Analyze
        layer_res = analyze_uc.execute("app")
        struct_res = layer_res.structural

        # 2. Predict (orchestrated)
        quality_res, problems = predict_uc.execute("app", struct_res, detect_problems=True)

        assert quality_res is not None
        assert len(quality_res.components) >= 1

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
                "connects_to": [{"source": "Node1", "target": "Node1"}]
            }
        }

        stats = model_uc.execute(graph_data)

        assert stats.success is True

        final_stats = memory_repo.get_statistics()
        assert final_stats.get("application_count", 0) >= 1
        assert final_stats.get("node_count", 0) >= 1

    def test_simulate_graph_use_case(self, simulation_service):
        """Smoke test for SimulateGraphUseCase."""
        simulate_uc = SimulateGraphUseCase(simulation_service)
        result = simulate_uc.execute(layer="app", mode=SimulationMode.EXHAUSTIVE)
        assert result is not None
        assert len(result) > 0

    def test_validate_graph_use_case(self, validation_service):
        """Smoke test for ValidateGraphUseCase."""
        validate_uc = ValidateGraphUseCase(validation_service)

        result = validate_uc.execute(layers=["app"])

        assert result is not None
        assert "app" in result.layers
