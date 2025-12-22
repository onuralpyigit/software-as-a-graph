#!/usr/bin/env python3
"""
Test Suite for Graph Visualization Module
===========================================

Comprehensive tests for multi-layer graph visualization.

Run with:
    python -m pytest tests/test_visualization.py -v
"""

import sys
import unittest
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

from src.visualization import (
    GraphVisualizer,
    DashboardGenerator,
    VisualizationConfig,
    DashboardConfig,
    Layer,
    LayoutAlgorithm,
    ColorScheme,
    Colors,
    MATPLOTLIB_AVAILABLE
)


class TestGraphBuilder:
    """Helper to build test graphs"""
    
    @staticmethod
    def create_simple_graph():
        """Create simple test graph"""
        G = nx.DiGraph()
        G.add_node('app1', type='Application', name='App 1')
        G.add_node('app2', type='Application', name='App 2')
        G.add_node('topic1', type='Topic', name='Topic 1')
        G.add_node('broker1', type='Broker', name='Broker 1')
        
        G.add_edge('app1', 'topic1', type='PUBLISHES_TO')
        G.add_edge('topic1', 'app2', type='SUBSCRIBES_TO')
        G.add_edge('broker1', 'topic1', type='RUNS_ON')
        
        return G
    
    @staticmethod
    def create_multi_layer_graph():
        """Create multi-layer test graph"""
        G = nx.DiGraph()
        
        # Infrastructure layer
        G.add_node('node1', type='Node', name='Server 1')
        G.add_node('node2', type='Node', name='Server 2')
        
        # Broker layer
        G.add_node('broker1', type='Broker', name='MQTT Broker')
        G.add_node('broker2', type='Broker', name='Kafka Broker')
        
        # Topic layer
        G.add_node('topic1', type='Topic', name='Temperature')
        G.add_node('topic2', type='Topic', name='Humidity')
        G.add_node('topic3', type='Topic', name='Alerts')
        
        # Application layer
        G.add_node('sensor1', type='Application', name='Sensor 1')
        G.add_node('sensor2', type='Application', name='Sensor 2')
        G.add_node('aggregator', type='Application', name='Aggregator')
        G.add_node('dashboard', type='Application', name='Dashboard')
        
        # Cross-layer edges
        G.add_edge('broker1', 'node1', type='RUNS_ON')
        G.add_edge('broker2', 'node2', type='RUNS_ON')
        G.add_edge('topic1', 'broker1', type='DEPENDS_ON')
        G.add_edge('topic2', 'broker1', type='DEPENDS_ON')
        G.add_edge('topic3', 'broker2', type='DEPENDS_ON')
        G.add_edge('sensor1', 'topic1', type='PUBLISHES_TO')
        G.add_edge('sensor2', 'topic2', type='PUBLISHES_TO')
        G.add_edge('topic1', 'aggregator', type='SUBSCRIBES_TO')
        G.add_edge('topic2', 'aggregator', type='SUBSCRIBES_TO')
        G.add_edge('aggregator', 'topic3', type='PUBLISHES_TO')
        G.add_edge('topic3', 'dashboard', type='SUBSCRIBES_TO')
        
        return G


class TestGraphVisualizer(unittest.TestCase):
    """Test GraphVisualizer class"""
    
    def setUp(self):
        self.graph = TestGraphBuilder.create_simple_graph()
        self.multi_layer_graph = TestGraphBuilder.create_multi_layer_graph()
        self.visualizer = GraphVisualizer()
    
    def test_classify_layers(self):
        """Test layer classification"""
        layers = self.visualizer.classify_layers(self.multi_layer_graph)
        
        self.assertIn(Layer.APPLICATION, layers)
        self.assertIn(Layer.TOPIC, layers)
        self.assertIn(Layer.BROKER, layers)
        self.assertIn(Layer.INFRASTRUCTURE, layers)
        
        self.assertEqual(len(layers[Layer.APPLICATION]), 4)
        self.assertEqual(len(layers[Layer.TOPIC]), 3)
        self.assertEqual(len(layers[Layer.BROKER]), 2)
        self.assertEqual(len(layers[Layer.INFRASTRUCTURE]), 2)
    
    def test_get_node_style(self):
        """Test node style generation"""
        node_data = {'type': 'Application'}
        style = self.visualizer.get_node_style('test', node_data)
        
        self.assertEqual(style.color, Colors.NODE_TYPES['Application'])
        self.assertIsInstance(style.size, int)
    
    def test_get_node_style_with_criticality(self):
        """Test node style with criticality"""
        node_data = {'type': 'Application'}
        crit = {'score': 0.8, 'level': 'critical', 'is_articulation_point': True}
        
        self.visualizer.config.color_scheme = ColorScheme.CRITICALITY
        style = self.visualizer.get_node_style('test', node_data, crit)
        
        self.assertEqual(style.color, Colors.CRITICALITY['critical'])
        self.assertGreater(style.size, 25)
        self.assertEqual(style.border_width, 4)
    
    def test_get_edge_style(self):
        """Test edge style generation"""
        edge_data = {'type': 'PUBLISHES_TO'}
        style = self.visualizer.get_edge_style('app1', 'topic1', edge_data, self.graph)
        
        self.assertEqual(style.color, Colors.EDGES['PUBLISHES_TO'])
    
    def test_calculate_layout_spring(self):
        """Test spring layout calculation"""
        self.visualizer.classify_layers(self.graph)
        positions = self.visualizer.calculate_layout(
            self.graph, LayoutAlgorithm.SPRING
        )
        
        self.assertEqual(len(positions), self.graph.number_of_nodes())
        for node, (x, y) in positions.items():
            self.assertIsInstance(x, float)
            self.assertIsInstance(y, float)
    
    def test_calculate_layout_hierarchical(self):
        """Test hierarchical layout calculation"""
        self.visualizer.classify_layers(self.multi_layer_graph)
        positions = self.visualizer.calculate_layout(
            self.multi_layer_graph, LayoutAlgorithm.HIERARCHICAL
        )
        
        self.assertEqual(len(positions), self.multi_layer_graph.number_of_nodes())
    
    def test_render_html(self):
        """Test HTML rendering"""
        html = self.visualizer.render_html(self.graph)
        
        self.assertIn('<!DOCTYPE html>', html)
        self.assertIn('vis.Network', html)
        self.assertIn('vis.DataSet', html)
        
        # Check nodes are present
        for node in self.graph.nodes():
            self.assertIn(node, html)
    
    def test_render_html_with_criticality(self):
        """Test HTML rendering with criticality"""
        criticality = {
            'app1': {'score': 0.8, 'level': 'critical'},
            'app2': {'score': 0.3, 'level': 'low'},
            'topic1': {'score': 0.6, 'level': 'high'},
            'broker1': {'score': 0.5, 'level': 'medium'}
        }
        
        html = self.visualizer.render_html(self.graph, criticality)
        
        self.assertIn('critical', html)
        self.assertIn('Criticality', html)
    
    def test_render_html_layer_filter(self):
        """Test HTML rendering with layer filter"""
        html = self.visualizer.render_html(
            self.multi_layer_graph,
            layer=Layer.APPLICATION
        )
        
        self.assertIn('sensor1', html)
        self.assertIn('dashboard', html)
        # Broker should not be in application layer view
        self.assertNotIn('"id": "broker1"', html)
    
    def test_render_multi_layer_html(self):
        """Test multi-layer HTML rendering"""
        html = self.visualizer.render_multi_layer_html(self.multi_layer_graph)
        
        self.assertIn('Multi-Layer', html)
        self.assertIn('APPLICATION', html)
        self.assertIn('TOPIC', html)
        self.assertIn('BROKER', html)
        self.assertIn('INFRASTRUCTURE', html)
    
    def test_get_layer_statistics(self):
        """Test layer statistics"""
        self.visualizer.classify_layers(self.multi_layer_graph)
        stats = self.visualizer.get_layer_statistics(self.multi_layer_graph)
        
        self.assertIn('total_nodes', stats)
        self.assertIn('total_edges', stats)
        self.assertIn('layers', stats)
        
        for layer_name, layer_stats in stats['layers'].items():
            self.assertIn('node_count', layer_stats)
            self.assertIn('internal_edges', layer_stats)
            self.assertIn('cross_layer_edges', layer_stats)
            self.assertIn('density', layer_stats)
    
    def test_export_for_d3(self):
        """Test D3.js export"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            self.visualizer.classify_layers(self.graph)
            self.visualizer.export_for_d3(self.graph, output_path)
            
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            self.assertIn('nodes', data)
            self.assertIn('links', data)
            self.assertEqual(len(data['nodes']), self.graph.number_of_nodes())
            self.assertEqual(len(data['links']), self.graph.number_of_edges())
        finally:
            Path(output_path).unlink(missing_ok=True)


class TestVisualizationConfig(unittest.TestCase):
    """Test VisualizationConfig"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = VisualizationConfig()
        
        self.assertEqual(config.layout, LayoutAlgorithm.SPRING)
        self.assertEqual(config.color_scheme, ColorScheme.TYPE)
        self.assertTrue(config.physics_enabled)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = VisualizationConfig(
            title="Custom Title",
            layout=LayoutAlgorithm.HIERARCHICAL,
            color_scheme=ColorScheme.CRITICALITY,
            physics_enabled=False
        )
        
        self.assertEqual(config.title, "Custom Title")
        self.assertEqual(config.layout, LayoutAlgorithm.HIERARCHICAL)
        self.assertFalse(config.physics_enabled)


class TestDashboardGenerator(unittest.TestCase):
    """Test DashboardGenerator class"""
    
    def setUp(self):
        self.graph = TestGraphBuilder.create_multi_layer_graph()
        self.generator = DashboardGenerator()
        
        self.criticality = {
            node: {
                'score': 0.5,
                'level': 'medium',
                'is_articulation_point': False
            }
            for node in self.graph.nodes()
        }
    
    def test_generate_dashboard(self):
        """Test dashboard generation"""
        html = self.generator.generate(
            graph=self.graph,
            criticality=self.criticality
        )
        
        self.assertIn('<!DOCTYPE html>', html)
        self.assertIn('Dashboard', html)
        self.assertIn('System Overview', html)
    
    def test_generate_with_validation(self):
        """Test dashboard with validation results"""
        validation = {
            'status': 'passed',
            'correlation': {
                'spearman': {'coefficient': 0.85}
            },
            'classification': {
                'overall': {
                    'f1_score': 0.92,
                    'precision': 0.88,
                    'recall': 0.95
                }
            }
        }
        
        html = self.generator.generate(
            graph=self.graph,
            criticality=self.criticality,
            validation=validation
        )
        
        self.assertIn('Validation', html)
        self.assertIn('PASSED', html)
        self.assertIn('Spearman', html)
    
    def test_generate_with_simulation(self):
        """Test dashboard with simulation results"""
        simulation = {
            'total_simulations': 10,
            'results': [
                {'primary_failures': ['broker1'], 'impact_score': 0.8, 'affected_nodes': 5},
                {'primary_failures': ['topic1'], 'impact_score': 0.6, 'affected_nodes': 3}
            ]
        }
        
        html = self.generator.generate(
            graph=self.graph,
            criticality=self.criticality,
            simulation=simulation
        )
        
        self.assertIn('Simulation', html)
        self.assertIn('broker1', html)
    
    def test_dashboard_config(self):
        """Test dashboard configuration"""
        config = DashboardConfig(
            title="Custom Dashboard",
            theme="light",
            include_network=False
        )
        generator = DashboardGenerator(config)
        
        html = generator.generate(
            graph=self.graph,
            criticality=self.criticality
        )
        
        self.assertIn('Custom Dashboard', html)
        # Network section should not be present
        self.assertNotIn('network-graph', html)


class TestColorSchemes(unittest.TestCase):
    """Test color scheme functionality"""
    
    def setUp(self):
        self.graph = TestGraphBuilder.create_simple_graph()
    
    def test_type_color_scheme(self):
        """Test type-based coloring"""
        config = VisualizationConfig(color_scheme=ColorScheme.TYPE)
        visualizer = GraphVisualizer(config)
        
        app_style = visualizer.get_node_style('app1', {'type': 'Application'})
        topic_style = visualizer.get_node_style('topic1', {'type': 'Topic'})
        
        self.assertEqual(app_style.color, Colors.NODE_TYPES['Application'])
        self.assertEqual(topic_style.color, Colors.NODE_TYPES['Topic'])
    
    def test_criticality_color_scheme(self):
        """Test criticality-based coloring"""
        config = VisualizationConfig(color_scheme=ColorScheme.CRITICALITY)
        visualizer = GraphVisualizer(config)
        
        crit_style = visualizer.get_node_style(
            'app1', {'type': 'Application'},
            {'level': 'critical', 'score': 0.9}
        )
        low_style = visualizer.get_node_style(
            'app2', {'type': 'Application'},
            {'level': 'low', 'score': 0.2}
        )
        
        self.assertEqual(crit_style.color, Colors.CRITICALITY['critical'])
        self.assertEqual(low_style.color, Colors.CRITICALITY['low'])


class TestLayoutAlgorithms(unittest.TestCase):
    """Test layout algorithms"""
    
    def setUp(self):
        self.graph = TestGraphBuilder.create_multi_layer_graph()
        self.visualizer = GraphVisualizer()
        self.visualizer.classify_layers(self.graph)
    
    def test_all_layouts(self):
        """Test all layout algorithms work"""
        layouts = [
            LayoutAlgorithm.SPRING,
            LayoutAlgorithm.CIRCULAR,
            LayoutAlgorithm.SHELL,
            LayoutAlgorithm.HIERARCHICAL,
            LayoutAlgorithm.LAYERED
        ]
        
        for layout in layouts:
            with self.subTest(layout=layout):
                positions = self.visualizer.calculate_layout(self.graph, layout)
                self.assertEqual(len(positions), self.graph.number_of_nodes())


class TestEdgeCases(unittest.TestCase):
    """Test edge cases"""
    
    def test_empty_graph(self):
        """Test with empty graph"""
        G = nx.DiGraph()
        visualizer = GraphVisualizer()
        
        html = visualizer.render_html(G)
        self.assertIn('<!DOCTYPE html>', html)
    
    def test_single_node(self):
        """Test with single node"""
        G = nx.DiGraph()
        G.add_node('single', type='Application')
        
        visualizer = GraphVisualizer()
        html = visualizer.render_html(G)
        
        self.assertIn('single', html)
    
    def test_disconnected_graph(self):
        """Test with disconnected components"""
        G = nx.DiGraph()
        G.add_node('a', type='Application')
        G.add_node('b', type='Application')
        G.add_node('c', type='Topic')
        G.add_edge('a', 'c')
        # b is disconnected
        
        visualizer = GraphVisualizer()
        html = visualizer.render_html(G)
        
        self.assertIn('a', html)
        self.assertIn('b', html)
        self.assertIn('c', html)
    
    def test_unknown_node_type(self):
        """Test with unknown node type"""
        G = nx.DiGraph()
        G.add_node('unknown', type='CustomType')
        
        visualizer = GraphVisualizer()
        style = visualizer.get_node_style('unknown', {'type': 'CustomType'})
        
        self.assertEqual(style.color, Colors.NODE_TYPES['Unknown'])


@unittest.skipUnless(MATPLOTLIB_AVAILABLE, "Matplotlib not available")
class TestImageExport(unittest.TestCase):
    """Test image export (requires matplotlib)"""
    
    def setUp(self):
        self.graph = TestGraphBuilder.create_simple_graph()
        self.visualizer = GraphVisualizer()
    
    def test_png_export(self):
        """Test PNG export"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name
        
        try:
            result = self.visualizer.render_image(
                self.graph, output_path, format='png'
            )
            self.assertEqual(result, output_path)
            self.assertTrue(Path(output_path).exists())
        finally:
            Path(output_path).unlink(missing_ok=True)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestGraphVisualizer))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualizationConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestDashboardGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestColorSchemes))
    suite.addTests(loader.loadTestsFromTestCase(TestLayoutAlgorithms))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    if MATPLOTLIB_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestImageExport))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())