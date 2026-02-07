"""
Tests for Visualization Service and Data Collector
"""
import pytest
from unittest.mock import MagicMock, patch
from src.application.services.visualization_service import VisualizationService
from src.application.services.visualization.data_collector import LayerDataCollector
from src.application.services.analysis_service import AnalysisService
from src.application.services.simulation_service import SimulationService
from src.application.services.validation_service import ValidationService
from src.domain.models.visualization.layer_data import LayerData

@pytest.fixture
def mock_analysis_service():
    service = MagicMock(spec=AnalysisService)
    # Mock analyze_layer return
    analysis_result = MagicMock()
    analysis_result.structural.graph_summary.nodes = 10
    analysis_result.structural.graph_summary.edges = 20
    analysis_result.structural.graph_summary.density = 0.5
    analysis_result.structural.graph_summary.num_components = 1
    analysis_result.structural.graph_summary.node_types = {"Application": 5, "Broker": 5}
    analysis_result.structural.graph_summary.num_articulation_points = 2
    
    # Mock quality components
    comp1 = MagicMock()
    comp1.id = "app1"
    comp1.type = "Application"
    comp1.scores.overall = 0.9
    comp1.levels.overall.name = "HIGH"
    comp1.structural.name = "App 1"
    
    analysis_result.quality.components = [comp1]
    analysis_result.problems = []
    
    # Mock structural edges
    analysis_result.structural.edges = {
        ("app1", "broker1"): MagicMock(weight=1.0, dependency_type="app_to_broker")
    }
    
    service.analyze_layer.return_value = analysis_result
    service._repository = MagicMock()
    return service

@pytest.fixture
def mock_simulation_service():
    service = MagicMock(spec=SimulationService)
    metrics = MagicMock()
    metrics.event_throughput = 100
    metrics.event_delivery_rate = 99.9
    metrics.avg_reachability_loss = 0.1
    metrics.max_impact = 0.5
    service.analyze_layer.return_value = metrics
    return service

@pytest.fixture
def mock_validation_service():
    service = MagicMock(spec=ValidationService)
    val_result = MagicMock()
    val_result.spearman = 0.8
    val_result.f1_score = 0.9
    val_result.passed = True
    
    layers_result = MagicMock()
    layers_result.layers = {"mw": val_result}
    service.validate_layers.return_value = layers_result
    return service

def test_collector_mw_layer(mock_analysis_service, mock_simulation_service, mock_validation_service):
    """Test data collection for the MW layer."""
    mock_repository = MagicMock()
    collector = LayerDataCollector(
        mock_analysis_service,
        mock_simulation_service,
        mock_validation_service,
        mock_repository
    )
    
    data = collector.collect_layer_data("mw", include_validation=True)
    
    assert data.layer == "mw"
    assert data.nodes == 10
    assert data.critical_count == 0 
    assert data.high_count == 1
    assert data.event_throughput == 100
    assert data.spearman == 0.8
    assert len(data.network_nodes) == 1
    assert len(data.network_edges) == 1
    assert data.network_edges[0]["relation_type"] == "DEPENDS_ON"

def test_visualization_service_integration(mock_analysis_service, mock_simulation_service, mock_validation_service):
    """Test full visualization service integration with mocked collector."""
    
    # Mock LayerDataCollector
    with patch("src.application.services.visualization.data_collector.LayerDataCollector") as MockCollector:
        # Setup the mock collector instance
        collector_instance = MockCollector.return_value
        
        # Create a real LayerData object for the return value
        mock_data = LayerData(layer="mw", name="Middleware Layer")
        mock_data.nodes = 10
        mock_data.spearman = 0.8
        mock_data.f1_score = 0.9
        mock_data.precision = 0.85
        mock_data.recall = 0.85
        mock_data.validation_passed = True
        
        collector_instance.collect_layer_data.return_value = mock_data
        
        # Initialize service with all required parameters
        mock_repository = MagicMock()
        service = VisualizationService(
            mock_analysis_service,
            mock_simulation_service,
            mock_validation_service,
            mock_repository
        )
        
        # Mock DashboardGenerator to avoid writing files
        with patch("src.application.services.visualization_service.DashboardGenerator") as MockDash:
            dash_instance = MockDash.return_value
            dash_instance.generate.return_value = "<html>Test Dashboard</html>"
            
            output = service.generate_dashboard(
                output_file="test_dashboard.html",
                layers=["mw"],
                include_network=False
            )
            
            assert "test_dashboard.html" in output
            MockDash.assert_called_once()
            
            # Verify collector was called correctly
            collector_instance.collect_layer_data.assert_called_with("mw", True)
