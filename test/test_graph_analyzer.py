#!/usr/bin/env python3
"""
Test Suite for Graph Analyzer
=============================

Comprehensive tests for the DEPENDS_ON relationship derivation
and graph analysis functionality.

Usage:
    # Run all tests
    python test_graph_analyzer.py
    
    # Run specific test class
    python test_graph_analyzer.py TestDependencyDerivation
    
    # Verbose output
    python test_graph_analyzer.py -v

Author: Software-as-a-Graph Research Project
"""

import sys
import json
import unittest
import tempfile
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import (
    GraphAnalyzer,
    DependsOnEdge,
    CriticalityScore,
    AnalysisResult,
    DependencyType,
    CriticalityLevel,
    analyze_pubsub_system,
    derive_dependencies,
)


# ============================================================================
# Test Data Generators
# ============================================================================

def create_minimal_system() -> Dict[str, Any]:
    """Create a minimal pub-sub system for basic tests"""
    return {
        "nodes": [
            {"id": "N1", "name": "Node1"},
            {"id": "N2", "name": "Node2"}
        ],
        "brokers": [
            {"id": "B1", "name": "Broker1", "node": "N1"}
        ],
        "applications": [
            {"id": "A1", "name": "Publisher", "role": "pub", "node": "N1"},
            {"id": "A2", "name": "Subscriber", "role": "sub", "node": "N2"}
        ],
        "topics": [
            {"id": "T1", "name": "Topic1", "broker": "B1"}
        ],
        "relationships": {
            "publishes_to": [
                {"from": "A1", "to": "T1"}
            ],
            "subscribes_to": [
                {"from": "A2", "to": "T1"}
            ]
        }
    }


def create_complex_system() -> Dict[str, Any]:
    """Create a more complex pub-sub system for comprehensive tests"""
    return {
        "nodes": [
            {"id": "N1", "name": "Node1"},
            {"id": "N2", "name": "Node2"},
            {"id": "N3", "name": "Node3"}
        ],
        "brokers": [
            {"id": "B1", "name": "Broker1", "node": "N1"},
            {"id": "B2", "name": "Broker2", "node": "N2"}
        ],
        "applications": [
            {"id": "A1", "name": "Publisher1", "role": "pub", "node": "N1"},
            {"id": "A2", "name": "Publisher2", "role": "pub", "node": "N1"},
            {"id": "A3", "name": "Processor", "role": "pubsub", "node": "N2"},
            {"id": "A4", "name": "Subscriber1", "role": "sub", "node": "N3"},
            {"id": "A5", "name": "Subscriber2", "role": "sub", "node": "N3"}
        ],
        "topics": [
            {"id": "T1", "name": "InputTopic", "broker": "B1"},
            {"id": "T2", "name": "ProcessedTopic", "broker": "B2"},
            {"id": "T3", "name": "SharedTopic", "broker": "B1"}
        ],
        "relationships": {
            "publishes_to": [
                {"from": "A1", "to": "T1"},
                {"from": "A2", "to": "T3"},
                {"from": "A3", "to": "T2"}
            ],
            "subscribes_to": [
                {"from": "A3", "to": "T1"},
                {"from": "A3", "to": "T3"},
                {"from": "A4", "to": "T2"},
                {"from": "A5", "to": "T2"},
                {"from": "A5", "to": "T3"}
            ]
        }
    }


def create_spof_system() -> Dict[str, Any]:
    """Create a system with a clear single point of failure"""
    return {
        "nodes": [
            {"id": "N1", "name": "Node1"},
            {"id": "N2", "name": "Node2"},
            {"id": "N3", "name": "Node3"}
        ],
        "brokers": [
            {"id": "B1", "name": "CentralBroker", "node": "N2"}
        ],
        "applications": [
            {"id": "A1", "name": "App1", "role": "pub", "node": "N1"},
            {"id": "A2", "name": "App2", "role": "pub", "node": "N1"},
            {"id": "A3", "name": "App3", "role": "sub", "node": "N3"},
            {"id": "A4", "name": "App4", "role": "sub", "node": "N3"}
        ],
        "topics": [
            {"id": "T1", "name": "Topic1", "broker": "B1"},
            {"id": "T2", "name": "Topic2", "broker": "B1"}
        ],
        "relationships": {
            "publishes_to": [
                {"from": "A1", "to": "T1"},
                {"from": "A2", "to": "T2"}
            ],
            "subscribes_to": [
                {"from": "A3", "to": "T1"},
                {"from": "A4", "to": "T2"}
            ]
        }
    }


def create_circular_dependency_system() -> Dict[str, Any]:
    """Create a system with circular dependencies"""
    return {
        "nodes": [
            {"id": "N1", "name": "Node1"}
        ],
        "brokers": [
            {"id": "B1", "name": "Broker1", "node": "N1"}
        ],
        "applications": [
            {"id": "A1", "name": "App1", "role": "pubsub", "node": "N1"},
            {"id": "A2", "name": "App2", "role": "pubsub", "node": "N1"},
            {"id": "A3", "name": "App3", "role": "pubsub", "node": "N1"}
        ],
        "topics": [
            {"id": "T1", "name": "Topic1", "broker": "B1"},
            {"id": "T2", "name": "Topic2", "broker": "B1"},
            {"id": "T3", "name": "Topic3", "broker": "B1"}
        ],
        "relationships": {
            "publishes_to": [
                {"from": "A1", "to": "T1"},  # A1 -> T1
                {"from": "A2", "to": "T2"},  # A2 -> T2
                {"from": "A3", "to": "T3"}   # A3 -> T3
            ],
            "subscribes_to": [
                {"from": "A2", "to": "T1"},  # A2 subscribes to A1's output
                {"from": "A3", "to": "T2"},  # A3 subscribes to A2's output
                {"from": "A1", "to": "T3"}   # A1 subscribes to A3's output -> CYCLE
            ]
        }
    }


# ============================================================================
# Test Classes
# ============================================================================

class TestGraphAnalyzerBasic(unittest.TestCase):
    """Basic tests for GraphAnalyzer initialization and loading"""
    
    def test_initialization(self):
        """Test analyzer initialization with default weights"""
        analyzer = GraphAnalyzer()
        self.assertEqual(analyzer.alpha, 0.4)
        self.assertEqual(analyzer.beta, 0.3)
        self.assertEqual(analyzer.gamma, 0.3)
    
    def test_custom_weights(self):
        """Test analyzer initialization with custom weights"""
        analyzer = GraphAnalyzer(alpha=0.5, beta=0.25, gamma=0.25)
        self.assertEqual(analyzer.alpha, 0.5)
        self.assertEqual(analyzer.beta, 0.25)
        self.assertEqual(analyzer.gamma, 0.25)
    
    def test_load_from_dict(self):
        """Test loading data from dictionary"""
        analyzer = GraphAnalyzer()
        data = create_minimal_system()
        analyzer.load_from_dict(data)
        
        self.assertEqual(analyzer.raw_data, data)
        self.assertIn('A1', analyzer._app_to_node)
        self.assertIn('B1', analyzer._broker_to_node)
    
    def test_load_from_file(self):
        """Test loading data from JSON file"""
        data = create_minimal_system()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            analyzer = GraphAnalyzer()
            analyzer.load_from_file(temp_path)
            self.assertEqual(analyzer.raw_data, data)
        finally:
            Path(temp_path).unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file raises error"""
        analyzer = GraphAnalyzer()
        with self.assertRaises(FileNotFoundError):
            analyzer.load_from_file('/nonexistent/file.json')


class TestDependencyDerivation(unittest.TestCase):
    """Tests for DEPENDS_ON relationship derivation"""
    
    def setUp(self):
        """Set up analyzer with minimal system"""
        self.analyzer = GraphAnalyzer()
        self.analyzer.load_from_dict(create_minimal_system())
    
    def test_derive_app_to_app(self):
        """Test APP_TO_APP dependency derivation"""
        edges = self.analyzer.derive_depends_on()
        
        app_to_app = [e for e in edges if e.dep_type == DependencyType.APP_TO_APP]
        
        # A2 (subscriber) should depend on A1 (publisher)
        self.assertEqual(len(app_to_app), 1)
        self.assertEqual(app_to_app[0].source, 'A2')
        self.assertEqual(app_to_app[0].target, 'A1')
        self.assertIn('T1', app_to_app[0].via_topics)
    
    def test_derive_app_to_broker(self):
        """Test APP_TO_BROKER dependency derivation"""
        edges = self.analyzer.derive_depends_on()
        
        app_to_broker = [e for e in edges if e.dep_type == DependencyType.APP_TO_BROKER]
        
        # Both A1 and A2 should depend on B1
        sources = {e.source for e in app_to_broker}
        self.assertIn('A1', sources)
        self.assertIn('A2', sources)
        
        for edge in app_to_broker:
            self.assertEqual(edge.target, 'B1')
    
    def test_derive_node_to_node(self):
        """Test NODE_TO_NODE dependency derivation"""
        edges = self.analyzer.derive_depends_on()
        
        node_to_node = [e for e in edges if e.dep_type == DependencyType.NODE_TO_NODE]
        
        # N2 should depend on N1 (A2 on N2 depends on A1 on N1)
        if node_to_node:  # May not exist if apps on same node
            self.assertEqual(node_to_node[0].source, 'N2')
            self.assertEqual(node_to_node[0].target, 'N1')
    
    def test_derive_node_to_broker(self):
        """Test NODE_TO_BROKER dependency derivation"""
        edges = self.analyzer.derive_depends_on()
        
        node_to_broker = [e for e in edges if e.dep_type == DependencyType.NODE_TO_BROKER]
        
        # N2 should depend on B1 (B1 is on N1)
        if node_to_broker:
            n2_to_b1 = [e for e in node_to_broker if e.source == 'N2' and e.target == 'B1']
            self.assertTrue(len(n2_to_b1) > 0)
    
    def test_dependency_weight_increases_with_shared_topics(self):
        """Test that weight increases with multiple shared topics"""
        # Create system with multiple shared topics
        data = {
            "nodes": [{"id": "N1", "name": "Node1"}],
            "brokers": [{"id": "B1", "name": "Broker1", "node": "N1"}],
            "applications": [
                {"id": "A1", "name": "Pub", "role": "pub", "node": "N1"},
                {"id": "A2", "name": "Sub", "role": "sub", "node": "N1"}
            ],
            "topics": [
                {"id": "T1", "name": "Topic1", "broker": "B1"},
                {"id": "T2", "name": "Topic2", "broker": "B1"},
                {"id": "T3", "name": "Topic3", "broker": "B1"}
            ],
            "relationships": {
                "publishes_to": [
                    {"from": "A1", "to": "T1"},
                    {"from": "A1", "to": "T2"},
                    {"from": "A1", "to": "T3"}
                ],
                "subscribes_to": [
                    {"from": "A2", "to": "T1"},
                    {"from": "A2", "to": "T2"},
                    {"from": "A2", "to": "T3"}
                ]
            }
        }
        
        analyzer = GraphAnalyzer()
        analyzer.load_from_dict(data)
        edges = analyzer.derive_depends_on()
        
        app_to_app = [e for e in edges if e.dep_type == DependencyType.APP_TO_APP]
        self.assertEqual(len(app_to_app), 1)
        self.assertEqual(len(app_to_app[0].via_topics), 3)
        self.assertGreater(app_to_app[0].weight, 1.0)


class TestComplexSystem(unittest.TestCase):
    """Tests using the complex system"""
    
    def setUp(self):
        """Set up analyzer with complex system"""
        self.analyzer = GraphAnalyzer()
        self.analyzer.load_from_dict(create_complex_system())
    
    def test_multiple_app_to_app_dependencies(self):
        """Test multiple APP_TO_APP dependencies are derived correctly"""
        edges = self.analyzer.derive_depends_on()
        
        app_to_app = [e for e in edges if e.dep_type == DependencyType.APP_TO_APP]
        
        # A3 depends on A1 (via T1) and A2 (via T3)
        a3_deps = [e for e in app_to_app if e.source == 'A3']
        self.assertGreaterEqual(len(a3_deps), 2)
        
        targets = {e.target for e in a3_deps}
        self.assertIn('A1', targets)
        self.assertIn('A2', targets)
    
    def test_chain_dependencies(self):
        """Test chain dependencies: A4/A5 -> A3 -> A1/A2"""
        edges = self.analyzer.derive_depends_on()
        
        app_to_app = [e for e in edges if e.dep_type == DependencyType.APP_TO_APP]
        
        # A4 and A5 depend on A3
        a4_a5_deps = [e for e in app_to_app if e.source in ('A4', 'A5') and e.target == 'A3']
        self.assertGreaterEqual(len(a4_a5_deps), 1)
    
    def test_multiple_brokers(self):
        """Test APP_TO_BROKER with multiple brokers"""
        edges = self.analyzer.derive_depends_on()
        
        app_to_broker = [e for e in edges if e.dep_type == DependencyType.APP_TO_BROKER]
        
        brokers_used = {e.target for e in app_to_broker}
        self.assertIn('B1', brokers_used)
        self.assertIn('B2', brokers_used)


class TestGraphBuilding(unittest.TestCase):
    """Tests for building NetworkX graph"""
    
    def test_build_dependency_graph(self):
        """Test building NetworkX graph from dependencies"""
        analyzer = GraphAnalyzer()
        analyzer.load_from_dict(create_minimal_system())
        
        G = analyzer.build_dependency_graph()
        
        # Check nodes
        self.assertIn('N1', G.nodes())
        self.assertIn('B1', G.nodes())
        self.assertIn('A1', G.nodes())
        self.assertIn('T1', G.nodes())
        
        # Check node types
        self.assertEqual(G.nodes['N1']['type'], 'Node')
        self.assertEqual(G.nodes['B1']['type'], 'Broker')
        self.assertEqual(G.nodes['A1']['type'], 'Application')
        
        # Check edges exist
        self.assertGreater(G.number_of_edges(), 0)
    
    def test_edge_attributes(self):
        """Test that edges have correct attributes"""
        analyzer = GraphAnalyzer()
        analyzer.load_from_dict(create_minimal_system())
        
        G = analyzer.build_dependency_graph()
        
        # Check edge attributes
        for u, v, data in G.edges(data=True):
            self.assertEqual(data['type'], 'DEPENDS_ON')
            self.assertIn('dependency_type', data)
            self.assertIn('weight', data)


class TestCriticalityScoring(unittest.TestCase):
    """Tests for criticality scoring"""
    
    def test_criticality_scores_computed(self):
        """Test that criticality scores are computed for all nodes"""
        analyzer = GraphAnalyzer()
        analyzer.load_from_dict(create_complex_system())
        
        result = analyzer.analyze()
        
        # All nodes should have scores
        scored_nodes = {s.node_id for s in result.criticality_scores}
        all_nodes = set()
        for n in analyzer.raw_data.get('nodes', []):
            all_nodes.add(n['id'])
        for b in analyzer.raw_data.get('brokers', []):
            all_nodes.add(b['id'])
        for a in analyzer.raw_data.get('applications', []):
            all_nodes.add(a['id'])
        for t in analyzer.raw_data.get('topics', []):
            all_nodes.add(t['id'])
        
        self.assertEqual(scored_nodes, all_nodes)
    
    def test_scores_sorted_descending(self):
        """Test that scores are sorted in descending order"""
        analyzer = GraphAnalyzer()
        analyzer.load_from_dict(create_complex_system())
        
        result = analyzer.analyze()
        
        scores = [s.composite_score for s in result.criticality_scores]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_articulation_points_marked_critical(self):
        """Test that articulation points are marked appropriately"""
        analyzer = GraphAnalyzer()
        analyzer.load_from_dict(create_spof_system())
        
        result = analyzer.analyze()
        
        # Check articulation points
        ap_scores = [s for s in result.criticality_scores if s.is_articulation_point]
        
        for score in ap_scores:
            self.assertEqual(score.level, CriticalityLevel.CRITICAL)
    
    def test_weight_formula(self):
        """Test the C_score = α·BC + β·AP + γ·I formula"""
        alpha, beta, gamma = 0.4, 0.3, 0.3
        analyzer = GraphAnalyzer(alpha=alpha, beta=beta, gamma=gamma)
        analyzer.load_from_dict(create_minimal_system())
        
        result = analyzer.analyze()
        
        for score in result.criticality_scores:
            expected = alpha * score.betweenness + beta * (1.0 if score.is_articulation_point else 0.0) + gamma * score.impact_score
            self.assertAlmostEqual(score.composite_score, expected, places=4)


class TestStructuralAnalysis(unittest.TestCase):
    """Tests for structural analysis"""
    
    def test_articulation_points_detected(self):
        """Test articulation points are detected"""
        analyzer = GraphAnalyzer()
        analyzer.load_from_dict(create_spof_system())
        
        result = analyzer.analyze()
        
        # B1 should be an articulation point (central broker)
        self.assertGreater(result.structural_analysis['articulation_point_count'], 0)
    
    def test_circular_dependencies_detected(self):
        """Test circular dependencies are detected"""
        analyzer = GraphAnalyzer()
        analyzer.load_from_dict(create_circular_dependency_system())
        
        result = analyzer.analyze()
        
        # Should detect the A1 -> A2 -> A3 -> A1 cycle
        self.assertTrue(result.structural_analysis['has_cycles'])
        self.assertGreater(len(result.structural_analysis['cycles']), 0)
    
    def test_no_false_cycles(self):
        """Test that simple linear systems don't report cycles"""
        analyzer = GraphAnalyzer()
        analyzer.load_from_dict(create_minimal_system())
        
        result = analyzer.analyze()
        
        # Simple pub-sub should not have cycles
        self.assertFalse(result.structural_analysis['has_cycles'])


class TestAnalysisResult(unittest.TestCase):
    """Tests for AnalysisResult"""
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        analyzer = GraphAnalyzer()
        analyzer.load_from_dict(create_minimal_system())
        
        result = analyzer.analyze()
        result_dict = result.to_dict()
        
        self.assertIn('graph_summary', result_dict)
        self.assertIn('depends_on', result_dict)
        self.assertIn('criticality', result_dict)
        self.assertIn('structural', result_dict)
        self.assertIn('recommendations', result_dict)
    
    def test_count_by_type(self):
        """Test counting edges by type"""
        analyzer = GraphAnalyzer()
        analyzer.load_from_dict(create_complex_system())
        
        result = analyzer.analyze()
        result_dict = result.to_dict()
        
        by_type = result_dict['depends_on']['by_type']
        self.assertIn('app_to_app', by_type)
        self.assertIn('app_to_broker', by_type)
    
    def test_count_by_level(self):
        """Test counting scores by level"""
        analyzer = GraphAnalyzer()
        analyzer.load_from_dict(create_complex_system())
        
        result = analyzer.analyze()
        result_dict = result.to_dict()
        
        by_level = result_dict['criticality']['by_level']
        total_by_level = sum(by_level.values())
        total_scores = len(result_dict['criticality']['scores'])
        
        self.assertEqual(total_by_level, total_scores)


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions"""
    
    def test_analyze_pubsub_system(self):
        """Test analyze_pubsub_system function"""
        data = create_minimal_system()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            result = analyze_pubsub_system(temp_path)
            
            self.assertIsInstance(result, AnalysisResult)
            self.assertGreater(len(result.depends_on_edges), 0)
        finally:
            Path(temp_path).unlink()
    
    def test_derive_dependencies(self):
        """Test derive_dependencies function"""
        data = create_minimal_system()
        
        deps = derive_dependencies(data)
        
        self.assertIsInstance(deps, list)
        self.assertGreater(len(deps), 0)
        
        for dep in deps:
            self.assertIn('source', dep)
            self.assertIn('target', dep)
            self.assertIn('type', dep)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases"""
    
    def test_empty_system(self):
        """Test handling of empty system"""
        analyzer = GraphAnalyzer()
        analyzer.load_from_dict({
            "nodes": [],
            "brokers": [],
            "applications": [],
            "topics": [],
            "relationships": {"publishes_to": [], "subscribes_to": []}
        })
        
        result = analyzer.analyze()
        
        self.assertEqual(result.graph_summary['total_nodes'], 0)
        self.assertEqual(len(result.depends_on_edges), 0)
    
    def test_no_relationships(self):
        """Test system with components but no relationships"""
        analyzer = GraphAnalyzer()
        analyzer.load_from_dict({
            "nodes": [{"id": "N1", "name": "Node1"}],
            "brokers": [{"id": "B1", "name": "Broker1", "node": "N1"}],
            "applications": [{"id": "A1", "name": "App1", "role": "pub", "node": "N1"}],
            "topics": [{"id": "T1", "name": "Topic1", "broker": "B1"}],
            "relationships": {"publishes_to": [], "subscribes_to": []}
        })
        
        result = analyzer.analyze()
        
        # Should have nodes but no DEPENDS_ON edges
        self.assertGreater(result.graph_summary['total_nodes'], 0)
        self.assertEqual(len(result.depends_on_edges), 0)
    
    def test_self_loop_prevention(self):
        """Test that self-loops are not created"""
        analyzer = GraphAnalyzer()
        analyzer.load_from_dict({
            "nodes": [{"id": "N1", "name": "Node1"}],
            "brokers": [{"id": "B1", "name": "Broker1", "node": "N1"}],
            "applications": [{"id": "A1", "name": "App1", "role": "pubsub", "node": "N1"}],
            "topics": [{"id": "T1", "name": "Topic1", "broker": "B1"}],
            "relationships": {
                "publishes_to": [{"from": "A1", "to": "T1"}],
                "subscribes_to": [{"from": "A1", "to": "T1"}]
            }
        })
        
        edges = analyzer.derive_depends_on()
        
        # No self-loops in APP_TO_APP
        app_to_app = [e for e in edges if e.dep_type == DependencyType.APP_TO_APP]
        for edge in app_to_app:
            self.assertNotEqual(edge.source, edge.target)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGraphAnalyzerBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestDependencyDerivation))
    suite.addTests(loader.loadTestsFromTestCase(TestComplexSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphBuilding))
    suite.addTests(loader.loadTestsFromTestCase(TestCriticalityScoring))
    suite.addTests(loader.loadTestsFromTestCase(TestStructuralAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalysisResult))
    suite.addTests(loader.loadTestsFromTestCase(TestConvenienceFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())