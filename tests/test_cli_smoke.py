
import sys
import pytest
import importlib
from unittest.mock import MagicMock, patch

# Ensure project root is in path
sys.path.append('/home/onuralpyigit/Workspace/SoftwareAsAGraph')

def test_simulate_graph_cli():
    """Test simulate_graph.py main function with mocks."""
    import simulate_graph
    
    # Mock Container
    mock_container = MagicMock()
    mock_repo = MagicMock()
    mock_container.graph_repository.return_value = mock_repo
    
    # Mock Simulator context manager
    mock_simulator = MagicMock()
    mock_simulator.__enter__.return_value = mock_simulator
    mock_simulator.__exit__.return_value = None
    
    # Use patch for sys.argv
    with patch.object(sys, 'argv', ['simulate_graph.py', '--event', 'App1']), \
         patch('src.simulation.Simulator', return_value=mock_simulator) as MockSimulator, \
         patch('src.infrastructure.Container', return_value=mock_container) as MockContainer:
        
        importlib.reload(simulate_graph)
        from simulate_graph import main
        
        ret = main()
        
        assert ret == 0
        MockContainer.assert_called_once()
        mock_container.graph_repository.assert_called_once()
        MockSimulator.assert_called_once()
        # Verify repository was passed to Simulator
        _, kwargs = MockSimulator.call_args
        assert kwargs['repository'] == mock_repo
        
        # Verify simulation method called
        mock_simulator.run_event_simulation.assert_called_once()
        # container.close called by context manager or finally
        mock_container.close.assert_called_once()

def test_validate_graph_cli():
    """Test validate_graph.py main function with mocks."""
    import validate_graph
    
    # Mock Container
    mock_container = MagicMock()
    mock_repo = MagicMock()
    mock_container.graph_repository.return_value = mock_repo
    
    # Mock ValidationPipeline
    mock_pipeline = MagicMock()
    mock_result = MagicMock()
    mock_result.all_passed = True
    mock_pipeline.run.return_value = mock_result
    
    with patch.object(sys, 'argv', ['validate_graph.py', '--layer', 'app']), \
         patch('src.validation.ValidationPipeline', return_value=mock_pipeline) as MockPipeline, \
         patch('src.infrastructure.Container', return_value=mock_container) as MockContainer:
        
        importlib.reload(validate_graph)
        from validate_graph import main
        
        ret = main()
        
        assert ret == 0
        MockContainer.assert_called_once()
        mock_container.graph_repository.assert_called_once()
        MockPipeline.assert_called_once()
        _, kwargs = MockPipeline.call_args
        assert kwargs['repository'] == mock_repo
        
        mock_pipeline.run.assert_called_once()
        mock_container.close.assert_called_once()

def test_visualize_graph_cli():
    """Test visualize_graph.py main function with mocks."""
    import visualize_graph
    
    # Mock Container
    mock_container = MagicMock()
    mock_repo = MagicMock()
    mock_container.graph_repository.return_value = mock_repo
    
    # Mock Visualizer context manager
    mock_viz = MagicMock()
    mock_viz.__enter__.return_value = mock_viz
    mock_viz.__exit__.return_value = None
    mock_viz.analyzer = MagicMock()
    mock_viz.simulator = MagicMock()
    mock_viz.validator = MagicMock()
    mock_viz.generate_dashboard.return_value = "test.html"
    
    with patch.object(sys, 'argv', ['visualize_graph.py', '--layer', 'system', '--output', 'test.html', '--no-network']), \
         patch('src.visualization.GraphVisualizer', return_value=mock_viz) as MockViz, \
         patch('src.infrastructure.Container', return_value=mock_container) as MockContainer:
        
        importlib.reload(visualize_graph)
        from visualize_graph import main
        
        # main() calls generate_dashboard which we didn't patch, but we patched GraphVisualizer
        ret = main() # visualize_graph.py main returns output path string
        
        assert ret == "test.html"
        MockContainer.assert_called_once()
        mock_container.graph_repository.assert_called_once()
        MockViz.assert_called_once()
        _, kwargs = MockViz.call_args
        assert kwargs['repository'] == mock_repo
        
        mock_viz.generate_dashboard.assert_called_once()
        # container.close NOT called in visualize_graph.py anymore (handled by Visualizer)
        # mock_container.close.assert_called_once()

def test_analyze_graph_cli():
    """Test analyze_graph.py main function with mocks."""
    import analyze_graph
    
    # Mock Container
    mock_container = MagicMock()
    mock_repo = MagicMock()
    mock_container.graph_repository.return_value = mock_repo
    
    # Mock GraphAnalyzer
    mock_analyzer = MagicMock()
    mock_analyzer.__enter__.return_value = mock_analyzer
    mock_analyzer.__exit__.return_value = None
    
    # Mock analysis result
    mock_result = MagicMock()
    mock_result.layer = "app"
    mock_result.to_dict.return_value = {}
    mock_result.layers = {} 
    mock_analyzer.analyze_layer.return_value = mock_result
    
    with patch.object(sys, 'argv', ['analyze_graph.py', '--layer', 'app']), \
         patch('src.analysis.GraphAnalyzer', return_value=mock_analyzer) as MockAnalyzer, \
         patch('src.infrastructure.Container', return_value=mock_container) as MockContainer:
        
        importlib.reload(analyze_graph)
        from analyze_graph import main
        
        ret = main()
        
        assert ret == 0
        MockContainer.assert_called_once()
        mock_container.graph_repository.assert_called_once()
        MockAnalyzer.assert_called_once()
        _, kwargs = MockAnalyzer.call_args
        assert kwargs['repository'] == mock_repo
        
        mock_analyzer.analyze_layer.assert_called_once()
        mock_container.close.assert_called_once()
