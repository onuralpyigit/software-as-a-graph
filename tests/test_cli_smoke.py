
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
    
    # Mock run_event_simulation return value
    mock_sim_result = MagicMock()
    mock_sim_result.scenario = "test_scenario"
    mock_sim_result.duration = 1.0 # float
    mock_sim_result.trace = []
    mock_sim_result.affected_components = []
    mock_sim_result.system_impact.initial_components = 0
    mock_sim_result.system_impact.total_cascade = 0
    mock_sim_result.system_impact.failed_providers = 0
    mock_sim_result.services_lost = []
    
    # Mock metrics
    mock_metrics = MagicMock()
    mock_metrics.messages_published = 100
    mock_metrics.messages_delivered = 90
    mock_metrics.messages_dropped = 10
    mock_metrics.delivery_rate = 90.0
    mock_metrics.drop_rate = 10.0
    mock_metrics.avg_latency = 0.05
    mock_metrics.min_latency = 0.01
    mock_metrics.max_latency = 1.0
    mock_metrics.p50_latency = 0.04
    mock_metrics.p99_latency = 0.5
    mock_metrics.throughput = 500.0
    mock_sim_result.metrics = mock_metrics

    mock_simulator.run_event_simulation.return_value = mock_sim_result
    
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
    # Mock numeric properties accessed by display.py
    mock_result.timestamp = "2023-01-01"
    mock_result.total_components = 100
    mock_result.layers_passed = 1
    mock_result.layers_passed = 1
    mock_result.layer_results = {} 
    mock_result.summary = {"passed": True}
    
    # Mock targets
    mock_targets = MagicMock()
    mock_targets.spearman = 0.5
    mock_targets.f1_score = 0.5
    mock_targets.precision = 0.5
    mock_targets.recall = 0.5
    mock_targets.top_5_overlap = 0.5
    mock_targets.top_10_overlap = 0.5
    mock_targets.rmse_max = 1.0
    mock_targets.pearson = 0.5
    mock_targets.kendall = 0.5
    mock_result.targets = mock_targets

    # Mock layers
    mock_layer_res = MagicMock()
    mock_layer_res.matched_components = 10
    mock_layer_res.spearman = 0.6
    mock_layer_res.f1_score = 0.6
    mock_layer_res.precision = 0.6
    mock_layer_res.recall = 0.6
    mock_layer_res.top_5_overlap = 0.6
    mock_layer_res.passed = True
    mock_result.layers = {"app": mock_layer_res}
    
    # Mock validation_result deep structure
    mock_val_res = MagicMock()
    mock_val_res.overall.correlation.spearman = 0.6
    mock_val_res.overall.correlation.pearson = 0.6
    mock_val_res.overall.correlation.kendall = 0.6
    mock_val_res.overall.classification.f1_score = 0.6
    mock_val_res.overall.classification.precision = 0.6
    mock_val_res.overall.classification.recall = 0.6
    mock_val_res.overall.classification.accuracy = 0.8
    mock_val_res.overall.classification.confusion_matrix = {'tp': 1, 'fp': 0, 'fn': 0, 'tn': 1}
    mock_val_res.overall.ranking.top_5_overlap = 0.6
    mock_val_res.overall.ranking.top_10_overlap = 0.6
    mock_val_res.overall.ranking.top_5_predicted = []
    mock_val_res.overall.ranking.top_5_actual = []
    mock_val_res.overall.ranking.top_5_common = []
    mock_val_res.overall.error.rmse = 0.1
    mock_val_res.overall.error.mae = 0.1
    mock_val_res.overall.error.max_error = 0.2
    mock_val_res.by_type = {}
    
    # Ensure layer result has this structure too if used by display_layer_validation_result
    mock_layer_res.validation_result = mock_val_res

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
    
    # Mock return values for mocked modules inside visualizer to avoid format errors if they are called
    # But generate_dashboard is mocked, so maybe we don't need to mock deep structure if we mock os.path.getsize
    
    with patch.object(sys, 'argv', ['visualize_graph.py', '--layer', 'system', '--output', 'test.html', '--no-network']), \
         patch('src.visualization.GraphVisualizer', return_value=mock_viz) as MockViz, \
         patch('src.infrastructure.Container', return_value=mock_container) as MockContainer, \
         patch('os.path.getsize', return_value=1024), \
         patch('os.path.exists', return_value=True):
        
        importlib.reload(visualize_graph)
        from visualize_graph import main
        
        # main() calls generate_dashboard which we didn't patch, but we patched GraphVisualizer
        ret = main() # visualize_graph.py main returns 0 on success
        
        assert ret == 0
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
    mock_result.layer_name = "Application"
    mock_result.description = "App Description"
    # Structure for display_graph_summary
    mock_result.structural.graph_summary.nodes = 10
    mock_result.structural.graph_summary.edges = 20
    mock_result.structural.graph_summary.density = 0.1
    mock_result.structural.graph_summary.avg_degree = 2.0
    mock_result.structural.graph_summary.avg_clustering = 0.5
    mock_result.structural.graph_summary.diameter = 5
    mock_result.structural.graph_summary.avg_path_length = 3.0
    mock_result.structural.graph_summary.components = 1
    mock_result.structural.graph_summary.num_components = 1 # Added
    mock_result.structural.graph_summary.num_articulation_points = 0 # Added
    mock_result.structural.graph_summary.num_bridges = 0 # Added
    mock_result.structural.graph_summary.node_types = {"Application": 10}
    mock_result.structural.graph_summary.edge_types = {"app_to_app": 20}
    
    # Structure for display_quality_summary
    mock_result.quality.classification_summary.total_components = 10
    mock_result.quality.classification_summary.critical_components = 0
    mock_result.quality.classification_summary.high_components = 0
    mock_result.quality.classification_summary.total_edges = 20
    mock_result.quality.classification_summary.critical_edges = 0
    mock_result.quality.classification_summary.high_edges = 0
    
    # Structure for display_problems
    mock_result.problems = []
    # Structure for display_layer_result -> display_problem_summary
    mock_result.problem_summary.total_problems = 0
    mock_result.problem_summary.requires_attention = 0
    mock_result.problem_summary.by_severity = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

    mock_result.to_dict.return_value = {}
    mock_result.layers = {} 
    mock_analyzer.analyze_layer.return_value = mock_result
    
    with patch.object(sys, 'argv', ['analyze_graph.py', '--layer', 'app']), \
         patch('src.application.services.AnalysisService', return_value=mock_analyzer) as MockAnalyzer, \
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
