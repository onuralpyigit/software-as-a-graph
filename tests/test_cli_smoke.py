
import sys
import pytest
import importlib
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure project root is in path
sys.path.append('/home/onuralpyigit/Workspace/SoftwareAsAGraph')

def test_simulate_graph_cli():
    """Test simulate_graph.py main function with mocks."""
    import simulate_graph
    
    # Mock Container
    mock_container = MagicMock()
    mock_repo = MagicMock()
    mock_container.graph_repository.return_value = mock_repo
    
    # Mock SimulationService context manager
    mock_service = MagicMock()
    mock_service.__enter__.return_value = mock_service
    mock_service.__exit__.return_value = None
    
    # Mock run_event_simulation return value
    mock_sim_result = MagicMock()
    mock_sim_result.scenario = "test_scenario"
    mock_sim_result.duration = 1.0 # float
    # ... (rest of mock setup same as before mostly)
    mock_sim_result.trace = []
    mock_sim_result.affected_components = []
    # mock_sim_result.system_impact... (legacy fields?)
    # New EventResult has "metrics", "affected_topics", etc.
    # Check EventResult structure in src/domain/services/simulation/event_simulator.py
    
    # Mock metrics
    mock_metrics = MagicMock()
    # Explicitly set attributes to valid numbers
    type(mock_metrics).messages_published = PropertyMock(return_value=100)
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
    mock_metrics.to_dict.return_value = {}
    
    mock_sim_result.metrics = mock_metrics
    mock_sim_result.to_dict.return_value = {}

    mock_service.run_event_simulation.return_value = mock_sim_result
    
    # Use patch for sys.argv
    with patch.object(sys, 'argv', ['simulate_graph.py', '--event', 'App1']), \
         patch('src.application.services.SimulationService', return_value=mock_service) as MockService, \
         patch('src.infrastructure.Container', return_value=mock_container) as MockContainer:
        
        importlib.reload(simulate_graph)
        from simulate_graph import main
        
        ret = main()
        
        assert ret == 0
        MockContainer.assert_called_once()
        mock_container.graph_repository.assert_called_once()
        MockService.assert_called_once()
        # Verify repository was passed to Service
        _, kwargs = MockService.call_args
        assert kwargs['repository'] == mock_repo
        
        # Verify simulation method called
        mock_service.run_event_simulation.assert_called_once()
        # container.close called by finally
        mock_container.close.assert_called_once()

@patch('src.infrastructure.container.Container.validation_service')
def test_validate_graph_cli(mock_validation_service):
    """Test the graph validation CLI."""
    # Setup mock
    mock_service_instance = MagicMock()
    
    # Mock result object pattern from ValidationService
    mock_pipeline_result = MagicMock()
    mock_pipeline_result.all_passed = True
    mock_pipeline_result.layers_passed = 1
    mock_pipeline_result.total_components = 10
    mock_pipeline_result.timestamp = "2023-01-01"
    
    mock_layer_result = MagicMock()
    mock_layer_result.passed = True
    mock_layer_result.layer = "app"
    mock_layer_result.layer_name = "Application"
    mock_layer_result.matched_components = 10
    mock_layer_result.predicted_components = 10
    mock_layer_result.simulated_components = 10
    mock_layer_result.spearman = 0.85
    mock_layer_result.f1_score = 0.9
    mock_layer_result.precision = 0.9
    mock_layer_result.recall = 0.9
    mock_layer_result.top_5_overlap = 0.8
    mock_layer_result.rmse = 0.1
    mock_layer_result.warnings = []
    
    # Mock nested validation_result for display_layer_validation_result
    mock_val_res = MagicMock()
    mock_val_res.overall.correlation.spearman = 0.85
    mock_val_res.overall.correlation.pearson = 0.8
    mock_val_res.overall.correlation.kendall = 0.7
    mock_val_res.overall.classification.f1_score = 0.9
    mock_val_res.overall.classification.precision = 0.9
    mock_val_res.overall.classification.recall = 0.9
    mock_val_res.overall.classification.accuracy = 0.95
    mock_val_res.overall.classification.confusion_matrix = {"tp": 5, "fp": 0, "tn": 5, "fn": 0}
    mock_val_res.overall.ranking.top_5_overlap = 0.8
    mock_val_res.overall.ranking.top_10_overlap = 0.8
    mock_val_res.overall.ranking.top_5_predicted = ["A", "B"]
    mock_val_res.overall.ranking.top_5_actual = ["A", "B"]
    mock_val_res.overall.ranking.top_5_common = ["A", "B"]
    mock_val_res.overall.error.rmse = 0.1
    mock_val_res.overall.error.mae = 0.1
    mock_val_res.overall.error.max_error = 0.2
    # Ensure components list has items to avoid empty list in display logic
    mock_comp = MagicMock()
    mock_comp.id = "A"
    mock_comp.type = "Application"
    mock_comp.predicted = 0.8
    mock_comp.actual = 0.85
    mock_comp.error = 0.05
    mock_comp.classification = "TP"
    mock_val_res.overall.components = [mock_comp]
    mock_val_res.by_type = {}
    mock_val_res.predicted_count = 10
    mock_val_res.actual_count = 10
    mock_val_res.matched_count = 10
    mock_val_res.warnings = []
    
    mock_layer_result.validation_result = mock_val_res
    
    mock_pipeline_result.layers = {"app": mock_layer_result}
    mock_pipeline_result.targets = MagicMock(
        spearman=0.7, pearson=0.65, kendall=0.5, 
        f1_score=0.8, precision=0.8, recall=0.8, 
        top_5_overlap=0.6, top_10_overlap=0.5,
        rmse_max=0.25
    )
    mock_pipeline_result.cross_layer_insights = []
    
    mock_service_instance.validate_layers.return_value = mock_pipeline_result
    mock_validation_service.return_value = mock_service_instance

    with patch.object(sys, 'argv', ['validate_graph.py', '--layer', 'app']), \
         patch('src.infrastructure.container.Container.graph_repository'):
        
        # Import and run
        import validate_graph
        importlib.reload(validate_graph)
        
        # Should return 0 (success)
        return_code = validate_graph.main()
        
        assert return_code == 0
        
        # Verify validation service was called
        mock_service_instance.validate_layers.assert_called_once()
        args = mock_service_instance.validate_layers.call_args[1]
        assert 'app' in args['layers']

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
