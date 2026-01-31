
import sys
import pytest
import importlib
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

# Ensure project root is in path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "bin"))

def test_simulate_graph_cli():
    """Test simulate_graph.py main function with mocks."""
    # Mock Container
    mock_container = MagicMock()
    mock_display = MagicMock()
    mock_sim_service = MagicMock()
    
    mock_container.display_service.return_value = mock_display
    mock_container.simulation_service.return_value = mock_sim_service
    
    # Mock simulation result
    mock_event_result = MagicMock()
    mock_event_result.to_dict.return_value = {}
    mock_sim_service.run_event_simulation.return_value = mock_event_result
    
    with patch.object(sys, 'argv', ['simulate_graph.py', '--event', 'App1']), \
         patch('src.config.Container', return_value=mock_container) as MockContainer:
        
        import simulate_graph
        importlib.reload(simulate_graph)
        
        ret = simulate_graph.main()
        
        assert ret == 0
        MockContainer.assert_called_once()
        mock_container.display_service.assert_called_once()
        mock_container.simulation_service.assert_called_once()
        mock_sim_service.run_event_simulation.assert_called_once()
        mock_display.display_event_result.assert_called_once()
        mock_container.close.assert_called_once()


def test_analyze_graph_cli():
    """Test analyze_graph.py main function with mocks."""
    mock_container = MagicMock()
    mock_display = MagicMock()
    mock_analysis_service = MagicMock()
    
    mock_container.display_service.return_value = mock_display
    mock_container.analysis_service.return_value = mock_analysis_service
    
    # Mock analysis result
    mock_results = MagicMock()
    mock_results.to_dict.return_value = {}
    mock_analysis_service.analyze_layer.return_value = MagicMock()
    
    with patch.object(sys, 'argv', ['analyze_graph.py', '--layer', 'app']), \
         patch('src.config.Container', return_value=mock_container) as MockContainer, \
         patch('analyze_graph.MultiLayerAnalysisResult', create=True) as MockMLAR:
        
        import analyze_graph
        importlib.reload(analyze_graph)
        ret = analyze_graph.main()
        
        assert ret == 0
        MockContainer.assert_called_once()
        mock_container.display_service.assert_called_once()
        mock_container.analysis_service.assert_called_once()
        mock_display.display_multi_layer_analysis_result.assert_called_once()
        mock_container.close.assert_called_once()


def test_validate_graph_cli():
    """Test validate_graph.py main function with mocks."""
    mock_container = MagicMock()
    mock_display = MagicMock()
    mock_val_service = MagicMock()
    
    mock_container.display_service.return_value = mock_display
    mock_container.validation_service.return_value = mock_val_service
    
    mock_result = MagicMock()
    mock_result.all_passed = True
    mock_result.to_dict.return_value = {}
    mock_val_service.validate_layers.return_value = mock_result
    
    with patch.object(sys, 'argv', ['validate_graph.py', '--layer', 'app']), \
         patch('src.config.Container', return_value=mock_container) as MockContainer:
        
        import validate_graph
        importlib.reload(validate_graph)
        ret = validate_graph.main()
        
        assert ret == 0
        MockContainer.assert_called_once()
        mock_container.display_service.assert_called_once()
        mock_container.validation_service.assert_called_once()
        mock_display.display_pipeline_validation_result.assert_called_once()
        mock_container.close.assert_called_once()


def test_visualize_graph_cli():
    """Test visualize_graph.py main function with mocks."""
    mock_container = MagicMock()
    mock_display = MagicMock()
    mock_viz_service = MagicMock()
    
    mock_container.display_service.return_value = mock_display
    mock_container.visualization_service.return_value = mock_viz_service
    
    mock_viz_service.generate_dashboard.return_value = "dashboard.html"
    
    with patch.object(sys, 'argv', ['visualize_graph.py', '--layer', 'app', '--output', 'test.html']), \
         patch('src.config.Container', return_value=mock_container) as MockContainer:
        
        import visualize_graph
        importlib.reload(visualize_graph)
        ret = visualize_graph.main()
        
        assert ret == 0
        MockContainer.assert_called_once()
        mock_container.display_service.assert_called_once()
        mock_container.visualization_service.assert_called_once()
        mock_viz_service.generate_dashboard.assert_called_once()
        mock_container.close.assert_called_once()
