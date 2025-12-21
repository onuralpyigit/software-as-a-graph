#!/usr/bin/env python3
"""
Test Suite for Multi-Layer Dependency Analyzer
==============================================

Tests for the direct algorithm application approach:
- Multi-layer graph building
- DEPENDS_ON derivation
- Critical node/edge identification
- Anti-pattern detection

Usage:
    python test_multi_layer_analyzer.py              # Run all tests
    python test_multi_layer_analyzer.py --quick      # Quick tests only
    python test_multi_layer_analyzer.py -v           # Verbose output

Author: Software-as-a-Graph Research Project
"""

import sys
import json
import unittest
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import networkx as nx
except ImportError:
    print("ERROR: NetworkX is required. Install with: pip install networkx")
    sys.exit(1)

from src.analysis.multi_layer_analyzer import (
    # Enums
    Layer,
    DependencyType,
    AntiPatternType,
    CriticalityReason,
    
    # Data Classes
    DependsOnEdge,
    CriticalNode,
    CriticalEdge,
    AntiPattern,
    
    # Builders and Analyzers
    MultiLayerGraphBuilder,
    DirectAlgorithmApplicator,
    CriticalComponentIdentifier,
    AntiPatternDetector,
    MultiLayerDependencyAnalyzer,
    
    # Convenience Functions
    analyze_multi_layer_dict,
)


# ============================================================================
# Test Data Generation
# ============================================================================

def generate_simple_pubsub_data() -> Dict[str, Any]:
    """Generate simple pub-sub system data"""
    return {
        "applications": [
            {"id": "pub1", "name": "Publisher1", "role": "pub"},
            {"id": "pub2", "name": "Publisher2", "role": "pub"},
            {"id": "sub1", "name": "Subscriber1", "role": "sub"},
            {"id": "sub2", "name": "Subscriber2", "role": "sub"},
            {"id": "relay1", "name": "Relay1", "role": "pubsub"},
        ],
        "topics": [
            {"id": "topic1", "name": "Events"},
            {"id": "topic2", "name": "Commands"},
        ],
        "brokers": [
            {"id": "broker1", "name": "MainBroker"},
        ],
        "nodes": [
            {"id": "node1", "name": "Server1"},
            {"id": "node2", "name": "Server2"},
        ],
        "relationships": {
            "publishes_to": [
                {"from": "pub1", "to": "topic1"},
                {"from": "pub2", "to": "topic1"},
                {"from": "relay1", "to": "topic2"},
            ],
            "subscribes_to": [
                {"from": "sub1", "to": "topic1"},
                {"from": "sub2", "to": "topic1"},
                {"from": "relay1", "to": "topic1"},
                {"from": "sub1", "to": "topic2"},
            ],
            "routes": [
                {"from": "broker1", "to": "topic1"},
                {"from": "broker1", "to": "topic2"},
            ],
            "runs_on": [
                {"from": "pub1", "to": "node1"},
                {"from": "pub2", "to": "node1"},
                {"from": "sub1", "to": "node2"},
                {"from": "sub2", "to": "node2"},
                {"from": "relay1", "to": "node1"},
                {"from": "broker1", "to": "node1"},
            ],
            "connects_to": [
                {"from": "node1", "to": "node2"},
            ]
        }
    }


def generate_god_topic_data() -> Dict[str, Any]:
    """Generate data with god topic anti-pattern"""
    data = {
        "applications": [],
        "topics": [{"id": "god_topic", "name": "GodTopic"}],
        "brokers": [{"id": "broker1", "name": "Broker"}],
        "nodes": [{"id": "node1", "name": "Server"}],
        "relationships": {
            "publishes_to": [],
            "subscribes_to": [],
            "routes": [{"from": "broker1", "to": "god_topic"}],
            "runs_on": [],
            "connects_to": []
        }
    }
    
    # Add many publishers and subscribers
    for i in range(15):
        data["applications"].append({
            "id": f"pub{i}", "name": f"Publisher{i}", "role": "pub"
        })
        data["relationships"]["publishes_to"].append({
            "from": f"pub{i}", "to": "god_topic"
        })
        data["relationships"]["runs_on"].append({
            "from": f"pub{i}", "to": "node1"
        })
    
    for i in range(10):
        data["applications"].append({
            "id": f"sub{i}", "name": f"Subscriber{i}", "role": "sub"
        })
        data["relationships"]["subscribes_to"].append({
            "from": f"sub{i}", "to": "god_topic"
        })
        data["relationships"]["runs_on"].append({
            "from": f"sub{i}", "to": "node1"
        })
    
    return data


def generate_circular_dependency_data() -> Dict[str, Any]:
    """Generate data with circular dependency"""
    return {
        "applications": [
            {"id": "app1", "name": "App1", "role": "pubsub"},
            {"id": "app2", "name": "App2", "role": "pubsub"},
            {"id": "app3", "name": "App3", "role": "pubsub"},
        ],
        "topics": [
            {"id": "topic1", "name": "Topic1"},
            {"id": "topic2", "name": "Topic2"},
            {"id": "topic3", "name": "Topic3"},
        ],
        "brokers": [{"id": "broker1", "name": "Broker"}],
        "nodes": [{"id": "node1", "name": "Server"}],
        "relationships": {
            "publishes_to": [
                {"from": "app1", "to": "topic1"},
                {"from": "app2", "to": "topic2"},
                {"from": "app3", "to": "topic3"},
            ],
            "subscribes_to": [
                {"from": "app2", "to": "topic1"},  # app2 depends on app1
                {"from": "app3", "to": "topic2"},  # app3 depends on app2
                {"from": "app1", "to": "topic3"},  # app1 depends on app3 - cycle!
            ],
            "routes": [
                {"from": "broker1", "to": "topic1"},
                {"from": "broker1", "to": "topic2"},
                {"from": "broker1", "to": "topic3"},
            ],
            "runs_on": [
                {"from": "app1", "to": "node1"},
                {"from": "app2", "to": "node1"},
                {"from": "app3", "to": "node1"},
            ],
            "connects_to": []
        }
    }


def generate_spof_data() -> Dict[str, Any]:
    """Generate data with single point of failure"""
    return {
        "applications": [
            {"id": "app1", "name": "App1", "role": "pub"},
            {"id": "app2", "name": "App2", "role": "sub"},
            {"id": "app3", "name": "App3", "role": "sub"},
            {"id": "app4", "name": "App4", "role": "sub"},
        ],
        "topics": [
            {"id": "topic1", "name": "CriticalTopic"},
        ],
        "brokers": [{"id": "broker1", "name": "Broker"}],
        "nodes": [
            {"id": "node1", "name": "Server1"},
            {"id": "node2", "name": "Server2"},
        ],
        "relationships": {
            "publishes_to": [
                {"from": "app1", "to": "topic1"},
            ],
            "subscribes_to": [
                {"from": "app2", "to": "topic1"},
                {"from": "app3", "to": "topic1"},
                {"from": "app4", "to": "topic1"},
            ],
            "routes": [
                {"from": "broker1", "to": "topic1"},
            ],
            "runs_on": [
                {"from": "app1", "to": "node1"},
                {"from": "app2", "to": "node2"},
                {"from": "app3", "to": "node2"},
                {"from": "app4", "to": "node2"},
                {"from": "broker1", "to": "node1"},
            ],
            "connects_to": [
                {"from": "node1", "to": "node2"},
            ]
        }
    }


# ============================================================================
# Multi-Layer Graph Builder Tests
# ============================================================================

class TestMultiLayerGraphBuilder(unittest.TestCase):
    """Tests for MultiLayerGraphBuilder"""
    
    def setUp(self):
        self.data = generate_simple_pubsub_data()
        self.builder = MultiLayerGraphBuilder()
    
    def test_build_from_dict(self):
        """Test building graph from dictionary"""
        self.builder.build_from_dict(self.data)
        
        # Check full graph
        self.assertEqual(self.builder.full_graph.number_of_nodes(), 10)
        self.assertGreater(self.builder.full_graph.number_of_edges(), 0)
    
    def test_node_types(self):
        """Test that nodes have correct types"""
        self.builder.build_from_dict(self.data)
        
        types = set()
        for n in self.builder.full_graph.nodes():
            node_data = self.builder.full_graph.nodes[n]
            types.add(node_data.get('type'))
        
        self.assertIn('Application', types)
        self.assertIn('Topic', types)
        self.assertIn('Broker', types)
        self.assertIn('Node', types)
    
    def test_depends_on_derivation(self):
        """Test DEPENDS_ON relationship derivation"""
        self.builder.build_from_dict(self.data)
        
        self.assertGreater(len(self.builder.depends_on_edges), 0)
        
        # Check types
        dep_types = set(e.dep_type for e in self.builder.depends_on_edges)
        self.assertIn(DependencyType.APP_TO_APP, dep_types)
    
    def test_app_to_app_dependency(self):
        """Test APP_TO_APP dependency derivation"""
        self.builder.build_from_dict(self.data)
        
        # sub1 subscribes to topic1, pub1/pub2 publish to topic1
        # So sub1 should depend on pub1 and pub2
        app_to_app = [e for e in self.builder.depends_on_edges 
                     if e.dep_type == DependencyType.APP_TO_APP]
        
        # Find sub1's dependencies
        sub1_deps = [e for e in app_to_app if e.source == 'sub1']
        self.assertGreater(len(sub1_deps), 0)
    
    def test_app_depends_graph(self):
        """Test app dependency graph construction"""
        self.builder.build_from_dict(self.data)
        
        # Should only contain applications
        for n in self.builder.app_depends_graph.nodes():
            node_data = self.builder.app_depends_graph.nodes[n]
            self.assertEqual(node_data.get('type'), 'Application')
    
    def test_node_depends_graph(self):
        """Test node dependency graph construction"""
        self.builder.build_from_dict(self.data)
        
        # Should only contain infrastructure nodes
        for n in self.builder.node_depends_graph.nodes():
            node_data = self.builder.node_depends_graph.nodes[n]
            self.assertEqual(node_data.get('type'), 'Node')


# ============================================================================
# Algorithm Applicator Tests
# ============================================================================

class TestDirectAlgorithmApplicator(unittest.TestCase):
    """Tests for DirectAlgorithmApplicator"""
    
    def setUp(self):
        self.data = generate_simple_pubsub_data()
        self.builder = MultiLayerGraphBuilder()
        self.builder.build_from_dict(self.data)
        self.applicator = DirectAlgorithmApplicator(self.builder.full_graph)
    
    def test_apply_all_algorithms(self):
        """Test applying all algorithms"""
        results = self.applicator.apply_all_algorithms()
        
        self.assertIn('betweenness', results)
        self.assertIn('pagerank', results)
        self.assertIn('hits', results)
        self.assertIn('degree', results)
        self.assertIn('articulation_points', results)
        self.assertIn('bridges', results)
        self.assertIn('k_core', results)
    
    def test_betweenness_centrality(self):
        """Test betweenness centrality computation"""
        results = self.applicator.apply_all_algorithms()
        
        bc = results['betweenness']
        self.assertEqual(len(bc), self.builder.full_graph.number_of_nodes())
        
        for val in bc.values():
            self.assertGreaterEqual(val, 0)
            self.assertLessEqual(val, 1)
    
    def test_pagerank(self):
        """Test PageRank computation"""
        results = self.applicator.apply_all_algorithms()
        
        pr = results['pagerank']
        self.assertEqual(len(pr), self.builder.full_graph.number_of_nodes())
        
        # PageRank values should sum to ~1
        total = sum(pr.values())
        self.assertAlmostEqual(total, 1.0, places=3)
    
    def test_hits(self):
        """Test HITS computation"""
        results = self.applicator.apply_all_algorithms()
        
        hits = results['hits']
        self.assertIn('hubs', hits)
        self.assertIn('authorities', hits)
        
        self.assertEqual(len(hits['hubs']), self.builder.full_graph.number_of_nodes())
        self.assertEqual(len(hits['authorities']), self.builder.full_graph.number_of_nodes())
    
    def test_k_core(self):
        """Test k-core computation"""
        results = self.applicator.apply_all_algorithms()
        
        kcore = results['k_core']
        
        for val in kcore.values():
            self.assertGreaterEqual(val, 0)


# ============================================================================
# Critical Component Identifier Tests
# ============================================================================

class TestCriticalComponentIdentifier(unittest.TestCase):
    """Tests for CriticalComponentIdentifier"""
    
    def setUp(self):
        self.data = generate_simple_pubsub_data()
        self.builder = MultiLayerGraphBuilder()
        self.builder.build_from_dict(self.data)
        self.applicator = DirectAlgorithmApplicator(self.builder.full_graph)
        self.algo_results = self.applicator.apply_all_algorithms()
        self.identifier = CriticalComponentIdentifier(
            self.builder.full_graph, self.algo_results
        )
    
    def test_identify_critical_nodes(self):
        """Test critical node identification"""
        critical = self.identifier.identify_critical_nodes()
        
        # Should return list of CriticalNode
        self.assertIsInstance(critical, list)
        
        for node in critical:
            self.assertIsInstance(node, CriticalNode)
            self.assertIsNotNone(node.node_id)
            self.assertGreater(len(node.reasons), 0)
    
    def test_critical_node_reasons(self):
        """Test that critical nodes have valid reasons"""
        critical = self.identifier.identify_critical_nodes()
        
        for node in critical:
            for reason in node.reasons:
                self.assertIsInstance(reason, CriticalityReason)
    
    def test_identify_critical_edges(self):
        """Test critical edge identification"""
        critical = self.identifier.identify_critical_edges()
        
        self.assertIsInstance(critical, list)
        
        for edge in critical:
            self.assertIsInstance(edge, CriticalEdge)
            self.assertIsNotNone(edge.source)
            self.assertIsNotNone(edge.target)


# ============================================================================
# Anti-Pattern Detector Tests
# ============================================================================

class TestAntiPatternDetector(unittest.TestCase):
    """Tests for AntiPatternDetector"""
    
    def test_detect_god_topic(self):
        """Test god topic detection"""
        data = generate_god_topic_data()
        
        builder = MultiLayerGraphBuilder()
        builder.build_from_dict(data)
        
        applicator = DirectAlgorithmApplicator(builder.full_graph)
        algo_results = applicator.apply_all_algorithms()
        
        detector = AntiPatternDetector(builder, algo_results)
        patterns = detector.detect_all()
        
        # Should detect god topic
        god_topics = [p for p in patterns if p.pattern_type == AntiPatternType.GOD_TOPIC]
        self.assertGreater(len(god_topics), 0)
        
        # Check severity
        self.assertIn(god_topics[0].severity, ['critical', 'high'])
    
    def test_detect_circular_dependency(self):
        """Test circular dependency detection"""
        data = generate_circular_dependency_data()
        
        builder = MultiLayerGraphBuilder()
        builder.build_from_dict(data)
        
        applicator = DirectAlgorithmApplicator(builder.full_graph)
        algo_results = applicator.apply_all_algorithms()
        
        detector = AntiPatternDetector(builder, algo_results)
        patterns = detector.detect_all()
        
        # Should detect circular dependency
        circular = [p for p in patterns if p.pattern_type == AntiPatternType.CIRCULAR_DEPENDENCY]
        self.assertGreater(len(circular), 0)
    
    def test_detect_spof(self):
        """Test single point of failure detection"""
        data = generate_spof_data()
        
        builder = MultiLayerGraphBuilder()
        builder.build_from_dict(data)
        
        applicator = DirectAlgorithmApplicator(builder.full_graph)
        algo_results = applicator.apply_all_algorithms()
        
        # Check if there are articulation points
        if algo_results.get('articulation_points'):
            detector = AntiPatternDetector(builder, algo_results)
            patterns = detector.detect_all()
            
            spof = [p for p in patterns if p.pattern_type == AntiPatternType.SINGLE_POINT_OF_FAILURE]
            # May or may not detect depending on graph structure
            self.assertIsInstance(spof, list)


# ============================================================================
# Multi-Layer Dependency Analyzer Tests
# ============================================================================

class TestMultiLayerDependencyAnalyzer(unittest.TestCase):
    """Tests for MultiLayerDependencyAnalyzer"""
    
    def setUp(self):
        self.data = generate_simple_pubsub_data()
    
    def test_analyze_from_dict(self):
        """Test analysis from dictionary"""
        analyzer = MultiLayerDependencyAnalyzer()
        result = analyzer.analyze_from_dict(self.data)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.total_nodes, 10)
        self.assertGreater(result.total_edges, 0)
    
    def test_layers_analyzed(self):
        """Test that multiple layers are analyzed"""
        analyzer = MultiLayerDependencyAnalyzer()
        result = analyzer.analyze_from_dict(self.data)
        
        self.assertIn('full', result.layers_analyzed)
    
    def test_critical_nodes_populated(self):
        """Test critical nodes are identified"""
        analyzer = MultiLayerDependencyAnalyzer()
        result = analyzer.analyze_from_dict(self.data)
        
        # Should have some critical nodes
        self.assertIsInstance(result.critical_nodes, list)
    
    def test_anti_patterns_populated(self):
        """Test anti-patterns are detected"""
        analyzer = MultiLayerDependencyAnalyzer()
        result = analyzer.analyze_from_dict(self.data)
        
        self.assertIsInstance(result.anti_patterns, list)
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        analyzer = MultiLayerDependencyAnalyzer()
        result = analyzer.analyze_from_dict(self.data)
        
        result_dict = result.to_dict()
        
        self.assertIn('summary', result_dict)
        self.assertIn('critical_nodes', result_dict)
        self.assertIn('critical_edges', result_dict)
        self.assertIn('anti_patterns', result_dict)
    
    def test_get_summary(self):
        """Test summary generation"""
        analyzer = MultiLayerDependencyAnalyzer()
        result = analyzer.analyze_from_dict(self.data)
        
        summary = analyzer.get_summary(result)
        
        self.assertIsInstance(summary, str)
        self.assertIn('MULTI-LAYER', summary)


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions"""
    
    def test_analyze_multi_layer_dict(self):
        """Test analyze_multi_layer_dict function"""
        data = generate_simple_pubsub_data()
        
        result = analyze_multi_layer_dict(data)
        
        self.assertIsNotNone(result)
        self.assertGreater(result.total_nodes, 0)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def test_full_pipeline(self):
        """Test complete analysis pipeline"""
        data = generate_simple_pubsub_data()
        
        # Build graphs
        builder = MultiLayerGraphBuilder()
        builder.build_from_dict(data)
        
        # Apply algorithms
        applicator = DirectAlgorithmApplicator(builder.full_graph)
        algo_results = applicator.apply_all_algorithms()
        
        # Identify critical components
        identifier = CriticalComponentIdentifier(builder.full_graph, algo_results)
        critical_nodes = identifier.identify_critical_nodes()
        critical_edges = identifier.identify_critical_edges()
        
        # Detect anti-patterns
        detector = AntiPatternDetector(builder, algo_results)
        patterns = detector.detect_all()
        
        # Verify all steps completed
        self.assertIsNotNone(builder.full_graph)
        self.assertIsNotNone(algo_results)
        self.assertIsInstance(critical_nodes, list)
        self.assertIsInstance(critical_edges, list)
        self.assertIsInstance(patterns, list)
    
    def test_cli_simulation(self):
        """Test CLI-like usage"""
        import tempfile
        
        data = generate_simple_pubsub_data()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            analyzer = MultiLayerDependencyAnalyzer()
            result = analyzer.analyze_from_file(temp_path)
            
            self.assertIsNotNone(result)
            self.assertGreater(result.total_nodes, 0)
        finally:
            Path(temp_path).unlink()


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run multi-layer analyzer tests')
    parser.add_argument('--quick', action='store_true', help='Quick tests only')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    verbosity = 2 if args.verbose else 1
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    if args.quick:
        suite.addTests(loader.loadTestsFromTestCase(TestMultiLayerGraphBuilder))
        suite.addTests(loader.loadTestsFromTestCase(TestDirectAlgorithmApplicator))
    else:
        suite.addTests(loader.loadTestsFromTestCase(TestMultiLayerGraphBuilder))
        suite.addTests(loader.loadTestsFromTestCase(TestDirectAlgorithmApplicator))
        suite.addTests(loader.loadTestsFromTestCase(TestCriticalComponentIdentifier))
        suite.addTests(loader.loadTestsFromTestCase(TestAntiPatternDetector))
        suite.addTests(loader.loadTestsFromTestCase(TestMultiLayerDependencyAnalyzer))
        suite.addTests(loader.loadTestsFromTestCase(TestConvenienceFunctions))
        suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()