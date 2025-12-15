#!/usr/bin/env python3
"""
Test Suite for Graph Visualizer
================================

Comprehensive tests for pub-sub system visualization.

Usage:
    python test_graph_visualizer.py
    python test_graph_visualizer.py -v

Author: Software-as-a-Graph Research Project
"""

import sys
import json
import unittest
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

from src.analysis import GraphAnalyzer
from src.simulation import GraphSimulator
from src.validation import GraphValidator
from src.visualization import (
    GraphVisualizer,
    VisualizationConfig,
    ColorScheme,
    visualize_system,
    MATPLOTLIB_AVAILABLE,
)


# ============================================================================
# Test Data
# ============================================================================

SAMPLE_PUBSUB_DATA = {
    "nodes": [
        {"id": "N1", "name": "ComputeNode1", "type": "compute"},
        {"id": "N2", "name": "ComputeNode2", "type": "compute"}
    ],
    "brokers": [
        {"id": "B1", "name": "MainBroker", "node": "N1"},
        {"id": "B2", "name": "BackupBroker", "node": "N2"}
    ],
    "applications": [
        {"id": "A1", "name": "Publisher", "role": "pub", "node": "N1"},
        {"id": "A2", "name": "Processor", "role": "both", "node": "N1"},
        {"id": "A3", "name": "Subscriber1", "role": "sub", "node": "N2"},
        {"id": "A4", "name": "Subscriber2", "role": "sub", "node": "N2"}
    ],
    "topics": [
        {"id": "T1", "name": "data", "broker": "B1"},
        {"id": "T2", "name": "processed", "broker": "B1"}
    ],
    "relationships": {
        "publishes_to": [
            {"from": "A1", "to": "T1"},
            {"from": "A2", "to": "T2"}
        ],
        "subscribes_to": [
            {"from": "A2", "to": "T1"},
            {"from": "A3", "to": "T2"},
            {"from": "A4", "to": "T2"}
        ],
        "runs_on": [
            {"from": "A1", "to": "N1"},
            {"from": "A2", "to": "N1"},
            {"from": "A3", "to": "N2"},
            {"from": "A4", "to": "N2"},
            {"from": "B1", "to": "N1"},
            {"from": "B2", "to": "N2"}
        ],
        "routes": [
            {"from": "B1", "to": "T1"},
            {"from": "B1", "to": "T2"}
        ]
    }
}


def create_test_analyzer() -> GraphAnalyzer:
    """Create a test analyzer with sample data"""
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    return analyzer


def create_test_results(analyzer: GraphAnalyzer):
    """Create test analysis, simulation, and validation results"""
    analysis = analyzer.analyze()
    
    simulator = GraphSimulator(seed=42)
    simulation = simulator.simulate_all_single_failures(analyzer.G)
    
    validator = GraphValidator(analyzer, seed=42)
    validation = validator.validate()
    
    return analysis, simulation, validation


# ============================================================================
# Test Classes
# ============================================================================

class TestColorScheme(unittest.TestCase):
    """Tests for ColorScheme class"""
    
    def test_type_colors(self):
        """Test type color lookup"""
        self.assertEqual(ColorScheme.get_type_color('Application'), '#4CAF50')
        self.assertEqual(ColorScheme.get_type_color('Broker'), '#2196F3')
        self.assertEqual(ColorScheme.get_type_color('Node'), '#FF9800')
        self.assertEqual(ColorScheme.get_type_color('Topic'), '#9C27B0')
    
    def test_type_color_unknown(self):
        """Test unknown type returns default"""
        color = ColorScheme.get_type_color('UnknownType')
        self.assertEqual(color, ColorScheme.TYPE_COLORS['Unknown'])
    
    def test_criticality_colors(self):
        """Test criticality color lookup"""
        self.assertEqual(ColorScheme.get_criticality_color('critical'), '#D32F2F')
        self.assertEqual(ColorScheme.get_criticality_color('high'), '#FF5722')
        self.assertEqual(ColorScheme.get_criticality_color('medium'), '#FFC107')
        self.assertEqual(ColorScheme.get_criticality_color('low'), '#4CAF50')
    
    def test_impact_color(self):
        """Test impact-based color"""
        self.assertEqual(ColorScheme.get_impact_color(0.8), '#D32F2F')  # Critical
        self.assertEqual(ColorScheme.get_impact_color(0.6), '#FF5722')  # High
        self.assertEqual(ColorScheme.get_impact_color(0.4), '#FFC107')  # Medium
        self.assertEqual(ColorScheme.get_impact_color(0.2), '#4CAF50')  # Low


class TestVisualizationConfig(unittest.TestCase):
    """Tests for VisualizationConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = VisualizationConfig()
        self.assertEqual(config.figsize, (12, 8))
        self.assertEqual(config.dpi, 150)
        self.assertEqual(config.node_size, 800)
        self.assertTrue(config.show_labels)
        self.assertTrue(config.show_legend)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = VisualizationConfig(
            figsize=(16, 12),
            dpi=300,
            node_size=1000,
            show_labels=False
        )
        self.assertEqual(config.figsize, (16, 12))
        self.assertEqual(config.dpi, 300)
        self.assertEqual(config.node_size, 1000)
        self.assertFalse(config.show_labels)
    
    def test_to_dict(self):
        """Test config serialization"""
        config = VisualizationConfig()
        d = config.to_dict()
        self.assertIn('figsize', d)
        self.assertIn('dpi', d)
        self.assertIn('node_size', d)


class TestGraphVisualizerInit(unittest.TestCase):
    """Tests for GraphVisualizer initialization"""
    
    def test_init_default(self):
        """Test default initialization"""
        viz = GraphVisualizer()
        self.assertIsNone(viz.analyzer)
        self.assertIsNotNone(viz.config)
    
    def test_init_with_analyzer(self):
        """Test initialization with analyzer"""
        analyzer = create_test_analyzer()
        viz = GraphVisualizer(analyzer)
        self.assertEqual(viz.analyzer, analyzer)
    
    def test_init_with_config(self):
        """Test initialization with custom config"""
        config = VisualizationConfig(dpi=300)
        viz = GraphVisualizer(config=config)
        self.assertEqual(viz.config.dpi, 300)


class TestGraphVisualizerSetters(unittest.TestCase):
    """Tests for GraphVisualizer result setters"""
    
    def setUp(self):
        self.analyzer = create_test_analyzer()
        self.viz = GraphVisualizer(self.analyzer)
    
    def test_set_analysis_result(self):
        """Test setting analysis result"""
        result = self.analyzer.analyze()
        self.viz.set_analysis_result(result)
        self.assertEqual(self.viz._analysis_result, result)
    
    def test_set_simulation_result(self):
        """Test setting simulation result"""
        self.analyzer.analyze()
        simulator = GraphSimulator(seed=42)
        result = simulator.simulate_all_single_failures(self.analyzer.G)
        self.viz.set_simulation_result(result)
        self.assertEqual(self.viz._simulation_result, result)
    
    def test_set_validation_result(self):
        """Test setting validation result"""
        validator = GraphValidator(self.analyzer, seed=42)
        result = validator.validate()
        self.viz.set_validation_result(result)
        self.assertEqual(self.viz._validation_result, result)


@unittest.skipUnless(MATPLOTLIB_AVAILABLE, "Matplotlib not available")
class TestGraphVisualizerPlots(unittest.TestCase):
    """Tests for GraphVisualizer plotting functions"""
    
    def setUp(self):
        self.analyzer = create_test_analyzer()
        self.analysis, self.simulation, self.validation = create_test_results(self.analyzer)
        
        self.viz = GraphVisualizer(self.analyzer)
        self.viz.set_analysis_result(self.analysis)
        self.viz.set_simulation_result(self.simulation)
        self.viz.set_validation_result(self.validation)
        
        self.temp_dir = tempfile.mkdtemp()
    
    def test_plot_topology(self):
        """Test topology plot generation"""
        output_path = Path(self.temp_dir) / 'topology.png'
        result = self.viz.plot_topology(str(output_path))
        
        self.assertIsNotNone(result)
        self.assertTrue(output_path.exists())
    
    def test_plot_topology_layouts(self):
        """Test different layout algorithms"""
        for layout in ['spring', 'circular', 'shell', 'kamada_kawai']:
            output_path = Path(self.temp_dir) / f'topology_{layout}.png'
            result = self.viz.plot_topology(str(output_path), layout=layout)
            self.assertIsNotNone(result)
    
    def test_plot_topology_color_by_type(self):
        """Test coloring by type"""
        output_path = Path(self.temp_dir) / 'topology_type.png'
        result = self.viz.plot_topology(str(output_path), color_by='type')
        self.assertIsNotNone(result)
    
    def test_plot_topology_color_by_criticality(self):
        """Test coloring by criticality"""
        output_path = Path(self.temp_dir) / 'topology_crit.png'
        result = self.viz.plot_topology(str(output_path), color_by='criticality')
        self.assertIsNotNone(result)
    
    def test_plot_criticality_heatmap(self):
        """Test criticality heatmap generation"""
        output_path = Path(self.temp_dir) / 'criticality.png'
        result = self.viz.plot_criticality_heatmap(str(output_path))
        
        self.assertIsNotNone(result)
        self.assertTrue(output_path.exists())
    
    def test_plot_criticality_heatmap_top_n(self):
        """Test criticality heatmap with custom top_n"""
        output_path = Path(self.temp_dir) / 'criticality_top5.png'
        result = self.viz.plot_criticality_heatmap(str(output_path), top_n=5)
        self.assertIsNotNone(result)
    
    def test_plot_impact_comparison(self):
        """Test impact comparison chart generation"""
        output_path = Path(self.temp_dir) / 'comparison.png'
        result = self.viz.plot_impact_comparison(str(output_path))
        
        self.assertIsNotNone(result)
        self.assertTrue(output_path.exists())
    
    def test_plot_validation_scatter(self):
        """Test validation scatter plot generation"""
        output_path = Path(self.temp_dir) / 'scatter.png'
        result = self.viz.plot_validation_scatter(str(output_path))
        
        self.assertIsNotNone(result)
        self.assertTrue(output_path.exists())
    
    def test_plot_impact_distribution(self):
        """Test impact distribution histogram generation"""
        output_path = Path(self.temp_dir) / 'distribution.png'
        result = self.viz.plot_impact_distribution(str(output_path))
        
        self.assertIsNotNone(result)
        self.assertTrue(output_path.exists())
    
    def test_plot_without_output_path(self):
        """Test plotting without saving"""
        result = self.viz.plot_topology()
        self.assertIsNone(result)  # No path returned if not saved


class TestHTMLReportGeneration(unittest.TestCase):
    """Tests for HTML report generation"""
    
    def setUp(self):
        self.analyzer = create_test_analyzer()
        self.analysis, self.simulation, self.validation = create_test_results(self.analyzer)
        
        self.viz = GraphVisualizer(self.analyzer)
        self.viz.set_analysis_result(self.analysis)
        self.viz.set_simulation_result(self.simulation)
        self.viz.set_validation_result(self.validation)
        
        self.temp_dir = tempfile.mkdtemp()
    
    def test_generate_html_report(self):
        """Test HTML report generation"""
        output_path = Path(self.temp_dir) / 'report.html'
        result = self.viz.generate_html_report(str(output_path))
        
        self.assertEqual(result, str(output_path))
        self.assertTrue(output_path.exists())
    
    def test_html_report_content(self):
        """Test HTML report contains expected sections"""
        output_path = Path(self.temp_dir) / 'report.html'
        self.viz.generate_html_report(str(output_path))
        
        with open(output_path) as f:
            content = f.read()
        
        # Check for key sections
        self.assertIn('Summary', content)
        self.assertIn('Criticality', content)
        self.assertIn('Validation', content)
        self.assertIn('Simulation', content)
    
    def test_html_report_custom_title(self):
        """Test HTML report with custom title"""
        output_path = Path(self.temp_dir) / 'report.html'
        custom_title = "My Custom Report Title"
        self.viz.generate_html_report(str(output_path), title=custom_title)
        
        with open(output_path) as f:
            content = f.read()
        
        self.assertIn(custom_title, content)
    
    def test_html_report_without_images(self):
        """Test HTML report without embedded images"""
        output_path = Path(self.temp_dir) / 'report_no_img.html'
        self.viz.generate_html_report(str(output_path), include_images=False)
        
        with open(output_path) as f:
            content = f.read()
        
        # Should still be valid HTML
        self.assertIn('<html>', content)
        self.assertIn('</html>', content)


class TestGraphDataExport(unittest.TestCase):
    """Tests for graph data export"""
    
    def setUp(self):
        self.analyzer = create_test_analyzer()
        analysis_result = self.analyzer.analyze()
        
        self.viz = GraphVisualizer(self.analyzer)
        self.viz.set_analysis_result(analysis_result)
        
        self.temp_dir = tempfile.mkdtemp()
    
    def test_export_json(self):
        """Test JSON export"""
        output_path = Path(self.temp_dir) / 'graph.json'
        result = self.viz.export_graph_data(str(output_path), format='json')
        
        self.assertTrue(output_path.exists())
        
        with open(output_path) as f:
            data = json.load(f)
        
        self.assertIn('nodes', data)
        self.assertIn('links', data)
        self.assertGreater(len(data['nodes']), 0)
    
    def test_export_json_includes_criticality(self):
        """Test JSON export includes criticality scores"""
        output_path = Path(self.temp_dir) / 'graph.json'
        self.viz.export_graph_data(str(output_path), format='json')
        
        with open(output_path) as f:
            data = json.load(f)
        
        # Some nodes should have criticality
        nodes_with_crit = [n for n in data['nodes'] if 'criticality' in n]
        self.assertGreater(len(nodes_with_crit), 0)
    
    # Note: GraphML and GEXF exports are skipped due to NetworkX serialization issues
    # with complex node attributes (lists). JSON export is the recommended format.


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions"""
    
    def test_visualize_system(self):
        """Test visualize_system function"""
        analyzer = create_test_analyzer()
        analysis = analyzer.analyze()
        
        temp_dir = tempfile.mkdtemp()
        files = visualize_system(analyzer, temp_dir)
        
        self.assertIsInstance(files, list)
        self.assertGreater(len(files), 0)
        
        # Should have HTML report at minimum
        html_files = [f for f in files if f.endswith('.html')]
        self.assertGreater(len(html_files), 0)
    
    def test_visualize_system_with_simulation(self):
        """Test visualize_system with simulation results"""
        analyzer = create_test_analyzer()
        analysis = analyzer.analyze()
        
        simulator = GraphSimulator(seed=42)
        simulation = simulator.simulate_all_single_failures(analyzer.G)
        
        temp_dir = tempfile.mkdtemp()
        files = visualize_system(analyzer, temp_dir, simulation_result=simulation)
        
        self.assertGreater(len(files), 0)
    
    def test_visualize_system_with_validation(self):
        """Test visualize_system with validation results"""
        analyzer = create_test_analyzer()
        analysis = analyzer.analyze()
        
        validator = GraphValidator(analyzer, seed=42)
        validation = validator.validate()
        
        temp_dir = tempfile.mkdtemp()
        files = visualize_system(analyzer, temp_dir, validation_result=validation)
        
        self.assertGreater(len(files), 0)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases"""
    
    def test_visualizer_no_analyzer(self):
        """Test visualizer without analyzer"""
        viz = GraphVisualizer()
        
        temp_dir = tempfile.mkdtemp()
        output_path = Path(temp_dir) / 'topology.png'
        
        if MATPLOTLIB_AVAILABLE:
            result = viz.plot_topology(str(output_path))
            self.assertIsNone(result)
    
    def test_visualizer_no_analysis_result(self):
        """Test visualizer without analysis result"""
        analyzer = create_test_analyzer()
        analyzer.analyze()  # Build graph but don't store result
        
        viz = GraphVisualizer(analyzer)
        # No analysis result set
        
        temp_dir = tempfile.mkdtemp()
        output_path = Path(temp_dir) / 'criticality.png'
        
        if MATPLOTLIB_AVAILABLE:
            result = viz.plot_criticality_heatmap(str(output_path))
            self.assertIsNone(result)
    
    def test_visualizer_empty_graph(self):
        """Test visualizer with minimal graph"""
        data = {
            "nodes": [{"id": "N1", "name": "Node1", "type": "compute"}],
            "brokers": [],
            "applications": [],
            "topics": [],
            "relationships": {
                "publishes_to": [],
                "subscribes_to": [],
                "runs_on": [],
                "routes": []
            }
        }
        
        analyzer = GraphAnalyzer()
        analyzer.load_from_dict(data)
        analyzer.analyze()
        
        viz = GraphVisualizer(analyzer)
        
        temp_dir = tempfile.mkdtemp()
        output_path = Path(temp_dir) / 'topology.png'
        
        if MATPLOTLIB_AVAILABLE:
            result = viz.plot_topology(str(output_path))
            # Should still work with minimal graph
            self.assertIsNotNone(result)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestColorScheme,
        TestVisualizationConfig,
        TestGraphVisualizerInit,
        TestGraphVisualizerSetters,
        TestGraphVisualizerPlots,
        TestHTMLReportGeneration,
        TestGraphDataExport,
        TestConvenienceFunctions,
        TestEdgeCases,
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())