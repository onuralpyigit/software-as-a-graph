
"""
Unit Tests for ValidationService
"""
import pytest
from unittest.mock import MagicMock, patch
from src.application.services.validation_service import ValidationService
from src.domain.config.layers import SimulationLayer
from src.domain.models.validation.results import LayerValidationResult, ValidationResult
from src.domain.models.analysis.results import LayerAnalysisResult
from src.domain.services import QualityAnalysisResult
from src.domain.services.failure_simulator import FailureResult, ImpactMetrics

@pytest.fixture
def mock_analysis_service():
    service = MagicMock()
    # Mock analyze_layer return value
    mock_result = MagicMock(spec=LayerAnalysisResult)
    mock_result.quality = MagicMock(spec=QualityAnalysisResult)
    mock_result.quality.components = []
    service.analyze_layer.return_value = mock_result
    return service

@pytest.fixture
def mock_simulation_service():
    service = MagicMock()
    # Mock run_failure_simulation_exhaustive return value
    service.run_failure_simulation_exhaustive.return_value = []
    return service

@pytest.fixture
def validation_service(mock_analysis_service, mock_simulation_service):
    return ValidationService(
        analysis_service=mock_analysis_service,
        simulation_service=mock_simulation_service
    )

class TestValidationService:
    
    def test_validate_layers_success(self, validation_service, mock_analysis_service, mock_simulation_service):
        """Test validating multiple valid layers."""
        # Setup mocks to return some data to avoid "insufficient data" warnings if possible,
        # but for this test we mainly care about flow control.
        
        result = validation_service.validate_layers(layers=["app", "infra"])
        
        assert result.total_components == 0 # internal validation returns 0 if empty
        assert "app" in result.layers
        assert "infra" in result.layers
        
        # Verify calls
        assert mock_analysis_service.analyze_layer.call_count == 2
        mock_analysis_service.analyze_layer.assert_any_call("app")
        mock_analysis_service.analyze_layer.assert_any_call("infra")
        
        assert mock_simulation_service.run_failure_simulation_exhaustive.call_count == 2
        mock_simulation_service.run_failure_simulation_exhaustive.assert_any_call(layer="app")
        mock_simulation_service.run_failure_simulation_exhaustive.assert_any_call(layer="infra")

    def test_validate_layers_skip_invalid(self, validation_service):
        """Test that invalid layers are skipped."""
        result = validation_service.validate_layers(layers=["app", "invalid_layer"])
        
        assert "app" in result.layers
        assert "invalid_layer" not in result.layers
        assert len(result.layers) == 1

    def test_validate_single_layer_flow(self, validation_service, mock_analysis_service, mock_simulation_service):
        """Test the flow of efficient sinlge layer validation."""
        # Setup Analysis Result
        comps = []
        sim_results = []
        for i, char in enumerate(['A', 'B', 'C']):
            # Analysis
            comp = MagicMock()
            comp.id = char
            comp.type = "Application"
            comp.scores.overall = 0.8  # Perfect match
            comp.structural.name = f"App {char}"
            comps.append(comp)
            
            # Simulation
            fail_res = MagicMock(spec=FailureResult)
            fail_res.target_id = char
            fail_res.impact = MagicMock(spec=ImpactMetrics)
            fail_res.impact.composite_impact = 0.8 # Perfect match
            sim_results.append(fail_res)
        
        mock_analysis_res = MagicMock(spec=LayerAnalysisResult)
        mock_analysis_res.quality = MagicMock(spec=QualityAnalysisResult)
        mock_analysis_res.quality.components = comps
        mock_analysis_service.analyze_layer.return_value = mock_analysis_res
        
        mock_simulation_service.run_failure_simulation_exhaustive.return_value = sim_results
        
        # Execute
        result = validation_service.validate_layers(layers=["app"])
        
        layer_res = result.layers["app"]
        assert layer_res.predicted_components == 3
        assert layer_res.simulated_components == 3
        assert layer_res.matched_components == 3
        assert layer_res.passed is True
        
        # Verify data passed to validator (implicitly via result checks)
        assert layer_res.component_names["A"] == "App A"
