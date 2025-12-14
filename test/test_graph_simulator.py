#!/usr/bin/env python3
"""
Test Suite for Graph Simulator
===============================

Comprehensive tests for failure simulation in pub-sub systems.

Usage:
    python test_graph_simulator.py
    python test_graph_simulator.py -v

Author: Software-as-a-Graph Research Project
"""

import sys
import json
import unittest
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

from src.simulation import (
    GraphSimulator,
    SimulationResult,
    BatchSimulationResult,
    FailureEvent,
    FailureMode,
    SimulationMode,
    simulate_single_failure,
    simulate_and_rank,
)


# ============================================================================
# Test Data Generators
# ============================================================================

def create_simple_graph() -> nx.DiGraph:
    """Create a simple test graph with DEPENDS_ON edges"""
    G = nx.DiGraph()
    
    # Add nodes
    G.add_node('A1', type='Application', name='Publisher')
    G.add_node('A2', type='Application', name='Subscriber')
    G.add_node('B1', type='Broker', name='MainBroker')
    G.add_node('T1', type='Topic', name='Topic1')
    G.add_node('N1', type='Node', name='Node1')
    G.add_node('N2', type='Node', name='Node2')
    
    # Add DEPENDS_ON edges (subscriber depends on publisher)
    G.add_edge('A2', 'A1', type='DEPENDS_ON', dependency_type='app_to_app', weight=1.0)
    G.add_edge('A1', 'B1', type='DEPENDS_ON', dependency_type='app_to_broker', weight=1.0)
    G.add_edge('A2', 'B1', type='DEPENDS_ON', dependency_type='app_to_broker', weight=1.0)
    G.add_edge('N2', 'N1', type='DEPENDS_ON', dependency_type='node_to_node', weight=1.0)
    
    return G


def create_chain_graph() -> nx.DiGraph:
    """Create a chain: A1 <- A2 <- A3 <- A4"""
    G = nx.DiGraph()
    
    for i in range(1, 5):
        G.add_node(f'A{i}', type='Application', name=f'App{i}')
    
    # Chain dependencies
    G.add_edge('A2', 'A1', type='DEPENDS_ON', dependency_type='app_to_app', weight=1.0)
    G.add_edge('A3', 'A2', type='DEPENDS_ON', dependency_type='app_to_app', weight=1.0)
    G.add_edge('A4', 'A3', type='DEPENDS_ON', dependency_type='app_to_app', weight=1.0)
    
    return G


def create_hub_graph() -> nx.DiGraph:
    """Create a hub-spoke graph with B1 as central hub"""
    G = nx.DiGraph()
    
    # Central hub
    G.add_node('B1', type='Broker', name='CentralBroker')
    
    # Spokes (all depend on hub)
    for i in range(1, 6):
        G.add_node(f'A{i}', type='Application', name=f'App{i}')
        G.add_edge(f'A{i}', 'B1', type='DEPENDS_ON', dependency_type='app_to_broker', weight=1.0)
    
    return G


def create_redundant_graph() -> nx.DiGraph:
    """Create a graph with redundant paths"""
    G = nx.DiGraph()
    
    # Nodes
    G.add_node('A1', type='Application', name='Publisher1')
    G.add_node('A2', type='Application', name='Publisher2')
    G.add_node('A3', type='Application', name='Subscriber')
    G.add_node('B1', type='Broker', name='Broker1')
    G.add_node('B2', type='Broker', name='Broker2')
    
    # A3 depends on both A1 and A2 (redundant publishers)
    G.add_edge('A3', 'A1', type='DEPENDS_ON', dependency_type='app_to_app', weight=1.0)
    G.add_edge('A3', 'A2', type='DEPENDS_ON', dependency_type='app_to_app', weight=1.0)
    
    # A1 uses B1, A2 uses B2
    G.add_edge('A1', 'B1', type='DEPENDS_ON', dependency_type='app_to_broker', weight=1.0)
    G.add_edge('A2', 'B2', type='DEPENDS_ON', dependency_type='app_to_broker', weight=1.0)
    
    return G


def create_complex_graph() -> nx.DiGraph:
    """Create a more complex test graph"""
    G = nx.DiGraph()
    
    # Infrastructure
    G.add_node('N1', type='Node', name='ComputeNode1')
    G.add_node('N2', type='Node', name='ComputeNode2')
    G.add_node('N3', type='Node', name='EdgeNode')
    
    # Brokers
    G.add_node('B1', type='Broker', name='MainBroker')
    G.add_node('B2', type='Broker', name='BackupBroker')
    
    # Applications
    for i in range(1, 8):
        G.add_node(f'A{i}', type='Application', name=f'App{i}')
    
    # Topics
    for i in range(1, 4):
        G.add_node(f'T{i}', type='Topic', name=f'Topic{i}')
    
    # App-to-app dependencies
    G.add_edge('A2', 'A1', type='DEPENDS_ON', dependency_type='app_to_app', weight=1.0)
    G.add_edge('A3', 'A1', type='DEPENDS_ON', dependency_type='app_to_app', weight=1.0)
    G.add_edge('A4', 'A2', type='DEPENDS_ON', dependency_type='app_to_app', weight=1.0)
    G.add_edge('A5', 'A3', type='DEPENDS_ON', dependency_type='app_to_app', weight=1.0)
    G.add_edge('A6', 'A4', type='DEPENDS_ON', dependency_type='app_to_app', weight=1.0)
    G.add_edge('A7', 'A5', type='DEPENDS_ON', dependency_type='app_to_app', weight=1.0)
    
    # App-to-broker dependencies
    G.add_edge('A1', 'B1', type='DEPENDS_ON', dependency_type='app_to_broker', weight=1.0)
    G.add_edge('A2', 'B1', type='DEPENDS_ON', dependency_type='app_to_broker', weight=1.0)
    G.add_edge('A3', 'B2', type='DEPENDS_ON', dependency_type='app_to_broker', weight=1.0)
    
    # Node-to-node dependencies
    G.add_edge('N2', 'N1', type='DEPENDS_ON', dependency_type='node_to_node', weight=1.0)
    G.add_edge('N3', 'N2', type='DEPENDS_ON', dependency_type='node_to_node', weight=1.0)
    
    return G


# ============================================================================
# Test Classes
# ============================================================================

class TestSimulatorBasic(unittest.TestCase):
    """Basic functionality tests"""
    
    def test_init_default(self):
        """Test default initialization"""
        sim = GraphSimulator()
        self.assertEqual(sim.cascade_threshold, 0.7)
        self.assertEqual(sim.cascade_probability, 0.5)
        self.assertEqual(sim.max_cascade_depth, 5)
    
    def test_init_custom(self):
        """Test custom initialization"""
        sim = GraphSimulator(
            cascade_threshold=0.5,
            cascade_probability=0.3,
            max_cascade_depth=10,
            seed=42
        )
        self.assertEqual(sim.cascade_threshold, 0.5)
        self.assertEqual(sim.cascade_probability, 0.3)
        self.assertEqual(sim.max_cascade_depth, 10)
    
    def test_component_not_found(self):
        """Test error when component not in graph"""
        G = create_simple_graph()
        sim = GraphSimulator()
        
        with self.assertRaises(ValueError):
            sim.simulate_failure(G, 'nonexistent')


class TestSingleFailure(unittest.TestCase):
    """Tests for single component failure simulation"""
    
    def setUp(self):
        self.simulator = GraphSimulator(seed=42)
    
    def test_single_failure_result_structure(self):
        """Test that result has correct structure"""
        G = create_simple_graph()
        result = self.simulator.simulate_failure(G, 'A1')
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.failed_components, ['A1'])
        self.assertEqual(result.simulation_mode, SimulationMode.SINGLE)
        self.assertIsInstance(result.impact_score, float)
        self.assertIsInstance(result.resilience_score, float)
    
    def test_single_failure_metrics(self):
        """Test impact metrics are calculated correctly"""
        G = create_simple_graph()
        result = self.simulator.simulate_failure(G, 'A1')
        
        # Impact + Resilience should equal 1
        self.assertAlmostEqual(result.impact_score + result.resilience_score, 1.0, places=4)
        
        # Original reachability should be > 0
        self.assertGreater(result.original_reachability, 0)
        
        # Reachability loss should be >= 0
        self.assertGreaterEqual(result.reachability_loss, 0)
    
    def test_hub_failure_high_impact(self):
        """Test that hub failure causes high impact"""
        G = create_hub_graph()
        result = self.simulator.simulate_failure(G, 'B1')
        
        # Hub failure should have significant impact
        self.assertGreater(result.impact_score, 0.3)
    
    def test_leaf_failure_low_impact(self):
        """Test that leaf node failure causes lower impact"""
        G = create_hub_graph()
        result = self.simulator.simulate_failure(G, 'A1')
        
        # Leaf failure should have lower impact than hub
        hub_result = self.simulator.simulate_failure(G, 'B1')
        self.assertLess(result.impact_score, hub_result.impact_score)
    
    def test_chain_head_failure(self):
        """Test failure of head in chain"""
        G = create_chain_graph()
        result = self.simulator.simulate_failure(G, 'A1')
        
        # Failing A1 should affect A2, A3, A4
        self.assertGreater(len(result.affected_components), 0)


class TestMultipleFailures(unittest.TestCase):
    """Tests for multiple simultaneous failures"""
    
    def setUp(self):
        self.simulator = GraphSimulator(seed=42)
    
    def test_multiple_failures_result(self):
        """Test multiple failures result structure"""
        G = create_simple_graph()
        result = self.simulator.simulate_multiple_failures(G, ['A1', 'A2'])
        
        self.assertEqual(set(result.failed_components), {'A1', 'A2'})
        self.assertEqual(result.simulation_mode, SimulationMode.MULTIPLE)
    
    def test_multiple_failures_higher_impact(self):
        """Test that multiple failures have higher impact than single"""
        G = create_hub_graph()
        
        single_result = self.simulator.simulate_failure(G, 'A1')
        multi_result = self.simulator.simulate_multiple_failures(G, ['A1', 'A2'])
        
        self.assertGreater(multi_result.impact_score, single_result.impact_score)
    
    def test_redundant_failures(self):
        """Test failures in redundant system"""
        G = create_redundant_graph()
        
        # Single publisher failure should be survivable
        single_result = self.simulator.simulate_failure(G, 'A1')
        
        # Both publishers failing should be worse
        both_result = self.simulator.simulate_multiple_failures(G, ['A1', 'A2'])
        
        self.assertGreater(both_result.impact_score, single_result.impact_score)


class TestCascadeFailures(unittest.TestCase):
    """Tests for cascading failure simulation"""
    
    def setUp(self):
        # Use low threshold to make cascades more likely
        self.simulator = GraphSimulator(
            cascade_threshold=0.3,
            cascade_probability=0.8,
            seed=42
        )
    
    def test_cascade_mode(self):
        """Test cascade simulation mode"""
        G = create_chain_graph()
        result = self.simulator.simulate_failure(G, 'A1', enable_cascade=True)
        
        self.assertEqual(result.simulation_mode, SimulationMode.CASCADE)
    
    def test_cascade_events(self):
        """Test cascade failure events are recorded"""
        G = create_chain_graph()
        result = self.simulator.simulate_failure(G, 'A1', enable_cascade=True)
        
        # Check events are recorded
        self.assertGreater(len(result.failure_events), 0)
        
        # First event should be primary failure
        self.assertEqual(result.failure_events[0].component, 'A1')
        self.assertFalse(result.failure_events[0].is_cascade)
    
    def test_cascade_depth(self):
        """Test cascade depth tracking"""
        G = create_chain_graph()
        result = self.simulator.simulate_failure(G, 'A1', enable_cascade=True)
        
        # Check cascade events have increasing depth
        depths = [e.cascade_depth for e in result.failure_events if e.is_cascade]
        if depths:
            self.assertEqual(depths, sorted(depths))


class TestBatchSimulation(unittest.TestCase):
    """Tests for exhaustive batch simulation"""
    
    def setUp(self):
        self.simulator = GraphSimulator(seed=42)
    
    def test_exhaustive_simulation(self):
        """Test exhaustive simulation runs all components"""
        G = create_simple_graph()
        result = self.simulator.simulate_all_single_failures(G)
        
        self.assertIsInstance(result, BatchSimulationResult)
        self.assertEqual(result.total_simulations, G.number_of_nodes())
    
    def test_exhaustive_with_type_filter(self):
        """Test exhaustive simulation with type filter"""
        G = create_complex_graph()
        
        result = self.simulator.simulate_all_single_failures(
            G, component_types=['Application']
        )
        
        # Should only test Application nodes
        app_count = sum(1 for _, d in G.nodes(data=True) if d.get('type') == 'Application')
        self.assertEqual(result.total_simulations, app_count)
    
    def test_batch_summary(self):
        """Test batch summary statistics"""
        G = create_hub_graph()
        result = self.simulator.simulate_all_single_failures(G)
        
        summary = result.summary
        
        self.assertIn('impact_score', summary)
        self.assertIn('reachability_loss', summary)
        self.assertIn('most_impactful', summary)
    
    def test_impact_ranking(self):
        """Test impact ranking is sorted correctly"""
        G = create_hub_graph()
        result = self.simulator.simulate_all_single_failures(G)
        
        ranking = result.get_impact_ranking()
        
        # Should be sorted descending
        impacts = [score for _, score in ranking]
        self.assertEqual(impacts, sorted(impacts, reverse=True))
        
        # Hub should be most impactful
        self.assertEqual(ranking[0][0], 'B1')


class TestReachability(unittest.TestCase):
    """Tests for reachability calculations"""
    
    def setUp(self):
        self.simulator = GraphSimulator(seed=42)
    
    def test_reachability_chain(self):
        """Test reachability in chain graph"""
        G = create_chain_graph()
        
        # Original reachability: (A1->none), (A2->A1), (A3->A1,A2), (A4->A1,A2,A3)
        # = 0 + 1 + 2 + 3 = 6 pairs
        
        result = self.simulator.simulate_failure(G, 'A2')
        
        # Failing A2 should reduce reachability
        self.assertGreater(result.reachability_loss, 0)
    
    def test_reachability_loss_percentage(self):
        """Test reachability loss percentage calculation"""
        G = create_chain_graph()
        result = self.simulator.simulate_failure(G, 'A1')
        
        if result.original_reachability > 0:
            expected_pct = (result.reachability_loss / result.original_reachability) * 100
            self.assertAlmostEqual(result.reachability_loss_pct, expected_pct, places=2)


class TestConnectivity(unittest.TestCase):
    """Tests for connectivity analysis"""
    
    def setUp(self):
        self.simulator = GraphSimulator(seed=42)
    
    def test_fragmentation(self):
        """Test fragmentation detection"""
        G = create_simple_graph()
        result = self.simulator.simulate_failure(G, 'B1')
        
        # Fragmentation should be >= 0
        self.assertGreaterEqual(result.fragmentation, 0)
    
    def test_isolated_components(self):
        """Test isolated component detection"""
        G = nx.DiGraph()
        G.add_node('A1', type='Application')
        G.add_node('A2', type='Application')
        G.add_node('A3', type='Application')
        G.add_edge('A2', 'A1', type='DEPENDS_ON')
        # A3 is already isolated
        
        result = self.simulator.simulate_failure(G, 'A1')
        
        # A3 should be isolated
        self.assertIn('A3', result.isolated_components)


class TestReporting(unittest.TestCase):
    """Tests for report generation"""
    
    def setUp(self):
        self.simulator = GraphSimulator(seed=42)
    
    def test_generate_report(self):
        """Test report generation"""
        G = create_simple_graph()
        result = self.simulator.simulate_failure(G, 'A1')
        
        report = self.simulator.generate_report(result)
        
        self.assertIn('summary', report)
        self.assertIn('failures', report)
        self.assertIn('impact', report)
        self.assertIn('recommendations', report)
    
    def test_severity_classification(self):
        """Test severity classification"""
        sim = GraphSimulator()
        
        self.assertEqual(sim._classify_severity(0.8), "CRITICAL")
        self.assertEqual(sim._classify_severity(0.5), "HIGH")
        self.assertEqual(sim._classify_severity(0.3), "MEDIUM")
        self.assertEqual(sim._classify_severity(0.1), "LOW")


class TestSerialization(unittest.TestCase):
    """Tests for result serialization"""
    
    def setUp(self):
        self.simulator = GraphSimulator(seed=42)
    
    def test_result_to_dict(self):
        """Test SimulationResult.to_dict()"""
        G = create_simple_graph()
        result = self.simulator.simulate_failure(G, 'A1')
        
        d = result.to_dict()
        
        self.assertIn('simulation_id', d)
        self.assertIn('impact_metrics', d)
        self.assertIn('reachability', d)
        self.assertIn('connectivity', d)
    
    def test_result_json_serializable(self):
        """Test result can be serialized to JSON"""
        G = create_simple_graph()
        result = self.simulator.simulate_failure(G, 'A1')
        
        # Should not raise
        json_str = json.dumps(result.to_dict(), default=str)
        self.assertIsInstance(json_str, str)
    
    def test_batch_result_to_dict(self):
        """Test BatchSimulationResult.to_dict()"""
        G = create_simple_graph()
        result = self.simulator.simulate_all_single_failures(G)
        
        d = result.to_dict()
        
        self.assertIn('total_simulations', d)
        self.assertIn('summary', d)
        self.assertIn('results', d)
    
    def test_failure_event_to_dict(self):
        """Test FailureEvent.to_dict()"""
        event = FailureEvent(
            component='A1',
            component_type='Application',
            failure_mode=FailureMode.CRASH,
            timestamp=datetime.now(),
            is_cascade=False,
            cascade_depth=0,
            cause='primary'
        )
        
        d = event.to_dict()
        
        self.assertEqual(d['component'], 'A1')
        self.assertEqual(d['failure_mode'], 'crash')


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions"""
    
    def test_simulate_single_failure(self):
        """Test simulate_single_failure function"""
        G = create_simple_graph()
        result = simulate_single_failure(G, 'A1')
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.failed_components, ['A1'])
    
    def test_simulate_and_rank(self):
        """Test simulate_and_rank function"""
        G = create_hub_graph()
        ranking = simulate_and_rank(G)
        
        self.assertIsInstance(ranking, list)
        self.assertGreater(len(ranking), 0)
        
        # Should be sorted by impact descending
        impacts = [score for _, score in ranking]
        self.assertEqual(impacts, sorted(impacts, reverse=True))


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases"""
    
    def setUp(self):
        self.simulator = GraphSimulator(seed=42)
    
    def test_single_node_graph(self):
        """Test graph with single node"""
        G = nx.DiGraph()
        G.add_node('A1', type='Application')
        
        result = self.simulator.simulate_failure(G, 'A1')
        
        self.assertEqual(result.failed_components, ['A1'])
        self.assertEqual(result.impact_score, 1.0)
    
    def test_disconnected_graph(self):
        """Test disconnected graph"""
        G = nx.DiGraph()
        G.add_node('A1', type='Application')
        G.add_node('A2', type='Application')
        # No edges
        
        result = self.simulator.simulate_failure(G, 'A1')
        
        # A2 should be isolated
        self.assertIn('A2', result.isolated_components)
    
    def test_empty_cascade(self):
        """Test no cascades when disabled"""
        G = create_chain_graph()
        result = self.simulator.simulate_failure(G, 'A1', enable_cascade=False)
        
        # Should have no cascade failures
        self.assertEqual(result.cascade_failures, [])


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegrationWithAnalyzer(unittest.TestCase):
    """Integration tests with GraphAnalyzer"""
    
    def test_analyzer_to_simulator(self):
        """Test using analyzer output with simulator"""
        from src.analysis import GraphAnalyzer
        
        # Create test data
        data = {
            'nodes': [
                {'id': 'N1', 'name': 'Node1', 'type': 'compute'},
                {'id': 'N2', 'name': 'Node2', 'type': 'edge'}
            ],
            'brokers': [
                {'id': 'B1', 'name': 'Broker1', 'node': 'N1'}
            ],
            'applications': [
                {'id': 'A1', 'name': 'Publisher', 'role': 'pub', 'node': 'N1'},
                {'id': 'A2', 'name': 'Subscriber', 'role': 'sub', 'node': 'N2'}
            ],
            'topics': [
                {'id': 'T1', 'name': 'Topic1', 'broker': 'B1'}
            ],
            'relationships': {
                'publishes_to': [{'from': 'A1', 'to': 'T1'}],
                'subscribes_to': [{'from': 'A2', 'to': 'T1'}],
                'runs_on': [
                    {'from': 'A1', 'to': 'N1'},
                    {'from': 'A2', 'to': 'N2'},
                    {'from': 'B1', 'to': 'N1'}
                ],
                'routes': [{'from': 'B1', 'to': 'T1'}]
            }
        }
        
        # Analyze
        analyzer = GraphAnalyzer()
        analyzer.load_from_dict(data)
        analyzer.derive_depends_on()
        graph = analyzer.build_dependency_graph()
        
        # Simulate
        simulator = GraphSimulator(seed=42)
        result = simulator.simulate_failure(graph, 'A1')
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.failed_components, ['A1'])


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestSimulatorBasic,
        TestSingleFailure,
        TestMultipleFailures,
        TestCascadeFailures,
        TestBatchSimulation,
        TestReachability,
        TestConnectivity,
        TestReporting,
        TestSerialization,
        TestConvenienceFunctions,
        TestEdgeCases,
        TestIntegrationWithAnalyzer,
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())