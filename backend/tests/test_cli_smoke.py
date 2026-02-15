
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
    mock_repo = MagicMock()
    mock_display = MagicMock()
    mock_sim_service = MagicMock()
    
    # Mock simulation result
    mock_event_result = MagicMock()
    mock_event_result.to_dict.return_value = {}
    mock_sim_service.run_event_simulation.return_value = mock_event_result
    
    with patch.object(sys, 'argv', ['simulate_graph.py', 'event', '--source', 'App1']):
        with patch('src.core.create_repository', return_value=mock_repo) as MockCreateRepo, \
             patch('src.simulation.SimulationService', return_value=mock_sim_service) as MockSimService, \
             patch('src.cli.console.ConsoleDisplay', return_value=mock_display):
            
            import simulate_graph
            importlib.reload(simulate_graph)
            
            ret = simulate_graph.main()
            
            assert ret == 0
            MockCreateRepo.assert_called_once()
            MockSimService.assert_called_once_with(mock_repo)
            mock_sim_service.run_event_simulation.assert_called_once()
            mock_display.display_event_result.assert_called_once()
            mock_repo.close.assert_called_once()


def test_analyze_graph_cli():
    """Test analyze_graph.py main function with mocks."""
    mock_repo = MagicMock()
    mock_display = MagicMock()
    mock_analysis_service = MagicMock()
    
    # Mock analysis result
    mock_results = MagicMock()
    mock_results.to_dict.return_value = {}
    mock_analysis_service.analyze_layer.return_value = MagicMock()
    
    with patch.object(sys, 'argv', ['analyze_graph.py', '--layer', 'app']), \
         patch('src.core.create_repository', return_value=mock_repo) as MockCreateRepo, \
         patch('src.analysis.AnalysisService', return_value=mock_analysis_service), \
         patch('src.cli.console.ConsoleDisplay', return_value=mock_display):
        
        import analyze_graph
        importlib.reload(analyze_graph)
        ret = analyze_graph.main()
        
        assert ret == 0
        MockCreateRepo.assert_called_once()
        mock_display.display_multi_layer_analysis_result.assert_called()
        mock_repo.close.assert_called()


def test_validate_graph_cli():
    """Test validate_graph.py main function with mocks."""
    mock_repo = MagicMock()
    mock_display = MagicMock()
    mock_val_service = MagicMock()
    
    mock_result = MagicMock()
    mock_result.all_passed = True
    mock_result.to_dict.return_value = {}
    mock_val_service.validate_layers.return_value = mock_result
    
    with patch.object(sys, 'argv', ['validate_graph.py', '--layer', 'app']), \
         patch('src.core.create_repository', return_value=mock_repo) as MockCreateRepo, \
         patch('src.validation.ValidationService', return_value=mock_val_service) as MockValService, \
         patch('src.cli.console.ConsoleDisplay', return_value=mock_display):
        
        import validate_graph
        importlib.reload(validate_graph)
        ret = validate_graph.main()
        
        assert ret == 0
        MockCreateRepo.assert_called_once()
        # ValidationService now takes analysis and simulation services optionally, 
        # but in validate_graph.py it's created with (analysis_service, simulation_service, targets)
        # We just check it was called.
        assert MockValService.call_count == 1
        mock_display.display_pipeline_validation_result.assert_called_once()
        mock_repo.close.assert_called_once()


def test_visualize_graph_cli():
    """Test visualize_graph.py main function with mocks."""
    mock_repo = MagicMock()
    mock_display = MagicMock()
    mock_viz_service = MagicMock()
    
    mock_viz_service.generate_dashboard.return_value = "dashboard.html"
    
    import src.visualization
    import src.analysis
    import src.simulation
    import src.validation

    with patch.object(sys, 'argv', ['visualize_graph.py', '--layer', 'app', '--output', 'test.html']), \
         patch('src.core.create_repository', return_value=mock_repo) as MockCreateRepo, \
         patch('src.analysis.AnalysisService', return_value=MagicMock()), \
         patch('src.simulation.SimulationService', return_value=MagicMock()), \
         patch('src.validation.ValidationService', return_value=MagicMock()), \
         patch('src.visualization.VisualizationService', return_value=mock_viz_service), \
         patch('src.cli.console.ConsoleDisplay', return_value=mock_display), \
         patch('os.path.getsize', return_value=1024):
        
        import visualize_graph
        importlib.reload(visualize_graph)
        ret = visualize_graph.main()
        
        assert ret == 0
        MockCreateRepo.assert_called_once()
        mock_viz_service.generate_dashboard.assert_called_once()
        mock_repo.close.assert_called_once()
