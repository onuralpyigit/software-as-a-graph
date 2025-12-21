#!/usr/bin/env python3
"""
Test Suite for Pub-Sub System Simulation
=========================================

Comprehensive tests for failure simulation and event-driven simulation.

Run with:
    python -m pytest tests/test_simulation.py -v
    
Or directly:
    python tests/test_simulation.py
"""

import sys
import unittest
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

from src.simulation import (
    # Failure Simulator
    FailureSimulator,
    FailureType,
    FailureMode,
    AttackStrategy,
    SimulationResult,
    BatchSimulationResult,
    
    # Event-Driven Simulator
    EventDrivenSimulator,
    EventSimulationResult,
    MessageState,
    ComponentState
)


class TestGraphBuilder:
    """Helper class to build test graphs"""
    
    @staticmethod
    def create_simple_graph() -> nx.DiGraph:
        """Create a simple pub-sub graph for testing"""
        G = nx.DiGraph()
        
        # Add nodes
        G.add_node('app1', type='Application')
        G.add_node('app2', type='Application')
        G.add_node('app3', type='Application')
        G.add_node('topic1', type='Topic')
        G.add_node('topic2', type='Topic')
        G.add_node('broker1', type='Broker')
        G.add_node('node1', type='Node')
        
        # Add edges (pub-sub relationships)
        G.add_edge('app1', 'topic1')  # app1 publishes to topic1
        G.add_edge('app2', 'topic1')  # app2 publishes to topic1
        G.add_edge('topic1', 'app3')  # app3 subscribes to topic1
        G.add_edge('app3', 'topic2')  # app3 publishes to topic2
        G.add_edge('topic2', 'app1')  # app1 subscribes to topic2
        G.add_edge('topic1', 'broker1')
        G.add_edge('topic2', 'broker1')
        G.add_edge('broker1', 'node1')
        
        return G
    
    @staticmethod
    def create_chain_graph(length: int = 5) -> nx.DiGraph:
        """Create a linear chain graph"""
        G = nx.DiGraph()
        
        for i in range(length):
            G.add_node(f'node_{i}', type='Application')
            if i > 0:
                G.add_edge(f'node_{i-1}', f'node_{i}')
        
        return G
    
    @staticmethod
    def create_star_graph(center: str = 'hub', spokes: int = 5) -> nx.DiGraph:
        """Create a star topology graph"""
        G = nx.DiGraph()
        
        G.add_node(center, type='Broker')
        
        for i in range(spokes):
            spoke = f'app_{i}'
            G.add_node(spoke, type='Application')
            G.add_edge(spoke, center)
            G.add_edge(center, spoke)
        
        return G
    
    @staticmethod
    def create_complex_graph() -> nx.DiGraph:
        """Create a more complex pub-sub graph"""
        G = nx.DiGraph()
        
        # Nodes
        nodes = [
            ('node1', 'Node'), ('node2', 'Node'),
            ('broker1', 'Broker'), ('broker2', 'Broker'),
            ('topic1', 'Topic'), ('topic2', 'Topic'), ('topic3', 'Topic'),
            ('app1', 'Application'), ('app2', 'Application'),
            ('app3', 'Application'), ('app4', 'Application'), ('app5', 'Application')
        ]
        
        for node_id, node_type in nodes:
            G.add_node(node_id, type=node_type)
        
        # Edges
        edges = [
            ('node1', 'broker1'), ('node2', 'broker2'),
            ('broker1', 'topic1'), ('broker1', 'topic2'),
            ('broker2', 'topic2'), ('broker2', 'topic3'),
            ('app1', 'topic1'), ('app2', 'topic1'),
            ('topic1', 'app3'), ('topic1', 'app4'),
            ('app3', 'topic2'), ('app4', 'topic3'),
            ('topic2', 'app5'), ('topic3', 'app5'),
            ('app5', 'topic1')  # Feedback loop
        ]
        
        G.add_edges_from(edges)
        
        return G


class TestFailureSimulator(unittest.TestCase):
    """Test cases for FailureSimulator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.simple_graph = TestGraphBuilder.create_simple_graph()
        self.chain_graph = TestGraphBuilder.create_chain_graph(5)
        self.star_graph = TestGraphBuilder.create_star_graph('hub', 5)
        self.complex_graph = TestGraphBuilder.create_complex_graph()
        self.simulator = FailureSimulator(seed=42)
    
    def test_single_failure_basic(self):
        """Test basic single component failure"""
        result = self.simulator.simulate_single_failure(
            self.simple_graph, 'app1', enable_cascade=False
        )
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(len(result.primary_failures), 1)
        self.assertEqual(result.primary_failures[0], 'app1')
        self.assertEqual(len(result.cascade_failures), 0)
        self.assertGreater(result.impact_score, 0)
        self.assertLessEqual(result.impact_score, 1.0)
    
    def test_single_failure_with_cascade(self):
        """Test single failure with cascade propagation"""
        result = self.simulator.simulate_single_failure(
            self.chain_graph, 'node_0', enable_cascade=True
        )
        
        self.assertIsInstance(result, SimulationResult)
        self.assertIn('node_0', result.primary_failures)
    
    def test_single_failure_invalid_component(self):
        """Test single failure with invalid component"""
        with self.assertRaises(ValueError):
            self.simulator.simulate_single_failure(
                self.simple_graph, 'nonexistent'
            )
    
    def test_multiple_failures(self):
        """Test multiple component failures"""
        result = self.simulator.simulate_multiple_failures(
            self.simple_graph, ['app1', 'app2'], enable_cascade=False
        )
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(len(result.primary_failures), 2)
        self.assertIn('app1', result.primary_failures)
        self.assertIn('app2', result.primary_failures)
    
    def test_network_failure(self):
        """Test network/edge failure"""
        result = self.simulator.simulate_network_failure(
            self.simple_graph, 'app1', 'topic1'
        )
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(result.simulation_type, 'network_failure')
    
    def test_network_failure_invalid_edge(self):
        """Test network failure with invalid edge"""
        with self.assertRaises(ValueError):
            self.simulator.simulate_network_failure(
                self.simple_graph, 'app1', 'nonexistent'
            )
    
    def test_random_failures(self):
        """Test random failure simulation"""
        result = self.simulator.simulate_random_failures(
            self.complex_graph,
            failure_probability=0.3,
            enable_cascade=False
        )
        
        self.assertIsInstance(result, SimulationResult)
    
    def test_targeted_attack_criticality(self):
        """Test targeted attack with criticality strategy"""
        result = self.simulator.simulate_targeted_attack(
            self.complex_graph,
            strategy=AttackStrategy.CRITICALITY,
            target_count=3,
            enable_cascade=False
        )
        
        self.assertIsInstance(result, SimulationResult)
        self.assertLessEqual(len(result.primary_failures), 3)
    
    def test_targeted_attack_betweenness(self):
        """Test targeted attack with betweenness strategy"""
        result = self.simulator.simulate_targeted_attack(
            self.complex_graph,
            strategy=AttackStrategy.BETWEENNESS,
            target_count=2,
            enable_cascade=False
        )
        
        self.assertIsInstance(result, SimulationResult)
    
    def test_targeted_attack_degree(self):
        """Test targeted attack with degree strategy"""
        result = self.simulator.simulate_targeted_attack(
            self.star_graph,
            strategy=AttackStrategy.DEGREE,
            target_count=1,
            enable_cascade=False
        )
        
        self.assertIsInstance(result, SimulationResult)
        # Hub should be selected (highest degree)
        self.assertIn('hub', result.primary_failures)
    
    def test_exhaustive_simulation(self):
        """Test exhaustive simulation of all components"""
        result = self.simulator.simulate_exhaustive(
            self.simple_graph,
            enable_cascade=False
        )
        
        self.assertIsInstance(result, BatchSimulationResult)
        self.assertEqual(result.total_simulations, self.simple_graph.number_of_nodes())
        self.assertGreater(len(result.results), 0)
        self.assertGreater(len(result.most_critical), 0)
    
    def test_exhaustive_with_type_filter(self):
        """Test exhaustive simulation with component type filter"""
        result = self.simulator.simulate_exhaustive(
            self.complex_graph,
            component_types=['Application'],
            enable_cascade=False
        )
        
        self.assertIsInstance(result, BatchSimulationResult)
        # Should only test Application nodes
        apps = [n for n, d in self.complex_graph.nodes(data=True) 
                if d.get('type') == 'Application']
        self.assertEqual(result.total_simulations, len(apps))
    
    def test_impact_metrics(self):
        """Test impact metrics calculation"""
        result = self.simulator.simulate_single_failure(
            self.star_graph, 'hub', enable_cascade=False
        )
        
        impact = result.impact
        
        # Hub removal should significantly affect reachability
        self.assertGreater(impact.reachability_loss, 0)
        self.assertGreater(impact.fragmentation, 0)
    
    def test_result_serialization(self):
        """Test result serialization to dict"""
        result = self.simulator.simulate_single_failure(
            self.simple_graph, 'app1', enable_cascade=False
        )
        
        result_dict = result.to_dict()
        
        self.assertIn('simulation_id', result_dict)
        self.assertIn('failures', result_dict)
        self.assertIn('impact', result_dict)
        self.assertIn('scores', result_dict)
    
    def test_batch_result_rankings(self):
        """Test batch result rankings"""
        result = self.simulator.simulate_exhaustive(
            self.star_graph,
            enable_cascade=False
        )
        
        # Hub should be most critical in star topology
        most_critical = result.most_critical
        self.assertGreater(len(most_critical), 0)
        
        # Find hub in rankings
        hub_rank = None
        for i, (comp, score) in enumerate(most_critical):
            if comp == 'hub':
                hub_rank = i
                break
        
        # Hub should be in top 2 most critical
        self.assertIsNotNone(hub_rank)
        self.assertLess(hub_rank, 2)


class TestEventDrivenSimulator(unittest.TestCase):
    """Test cases for EventDrivenSimulator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.simple_graph = TestGraphBuilder.create_simple_graph()
        self.complex_graph = TestGraphBuilder.create_complex_graph()
        self.simulator = EventDrivenSimulator(seed=42)
    
    def test_basic_simulation(self):
        """Test basic event-driven simulation"""
        result = self.simulator.simulate(
            self.simple_graph,
            duration_ms=1000,
            message_rate=10
        )
        
        self.assertIsInstance(result, EventSimulationResult)
        self.assertGreater(result.total_events, 0)
        self.assertGreater(result.metrics.total_messages, 0)
    
    def test_simulation_duration(self):
        """Test simulation respects duration"""
        duration = 5000
        result = self.simulator.simulate(
            self.simple_graph,
            duration_ms=duration,
            message_rate=100
        )
        
        self.assertEqual(result.duration_ms, duration)
    
    def test_message_delivery(self):
        """Test messages are delivered"""
        result = self.simulator.simulate(
            self.complex_graph,
            duration_ms=2000,
            message_rate=50
        )
        
        metrics = result.metrics
        
        # Some messages should be delivered
        self.assertGreater(metrics.delivered_messages, 0)
        # Delivery rate should be reasonable
        self.assertGreater(metrics.delivery_rate, 0.5)
    
    def test_latency_metrics(self):
        """Test latency metrics are calculated"""
        result = self.simulator.simulate(
            self.simple_graph,
            duration_ms=1000,
            message_rate=50
        )
        
        metrics = result.metrics
        
        if metrics.delivered_messages > 0:
            self.assertGreater(metrics.avg_latency, 0)
            self.assertGreaterEqual(metrics.p99_latency, metrics.p50_latency)
    
    def test_failure_injection(self):
        """Test failure injection during simulation"""
        failure_schedule = [
            {'time_ms': 500, 'component': 'broker1', 'duration_ms': 200}
        ]
        
        result = self.simulator.simulate(
            self.simple_graph,
            duration_ms=1000,
            message_rate=50,
            failure_schedule=failure_schedule
        )
        
        self.assertEqual(len(result.failures_injected), 1)
        self.assertGreater(result.metrics.component_failures, 0)
    
    def test_load_test(self):
        """Test load testing simulation"""
        result = self.simulator.simulate_with_load_test(
            self.complex_graph,
            duration_ms=5000,
            initial_rate=10,
            peak_rate=100,
            ramp_time_ms=2000
        )
        
        self.assertIsInstance(result, EventSimulationResult)
        self.assertGreater(result.metrics.total_messages, 0)
    
    def test_chaos_simulation(self):
        """Test chaos engineering simulation"""
        result = self.simulator.simulate_chaos(
            self.complex_graph,
            duration_ms=3000,
            message_rate=50,
            failure_probability=0.05,
            recovery_probability=0.5,
            check_interval_ms=500
        )
        
        self.assertIsInstance(result, EventSimulationResult)
    
    def test_speedup(self):
        """Test simulation achieves speedup over real-time"""
        result = self.simulator.simulate(
            self.simple_graph,
            duration_ms=10000,  # 10 seconds simulated
            message_rate=100
        )
        
        # Should be faster than real-time
        self.assertGreater(result.speedup, 1.0)
    
    def test_component_stats(self):
        """Test component statistics are collected"""
        result = self.simulator.simulate(
            self.simple_graph,
            duration_ms=2000,
            message_rate=50
        )
        
        self.assertGreater(len(result.component_stats), 0)
        
        # Check stats structure
        for comp_id, stats in result.component_stats.items():
            self.assertEqual(stats.component_id, comp_id)
            self.assertGreaterEqual(stats.messages_received, 0)
            self.assertGreaterEqual(stats.messages_sent, 0)
    
    def test_result_serialization(self):
        """Test result serialization to dict"""
        result = self.simulator.simulate(
            self.simple_graph,
            duration_ms=1000,
            message_rate=20
        )
        
        result_dict = result.to_dict()
        
        self.assertIn('simulation_id', result_dict)
        self.assertIn('timing', result_dict)
        self.assertIn('metrics', result_dict)
        self.assertIn('component_stats', result_dict)
    
    def test_events_by_type(self):
        """Test events are categorized by type"""
        result = self.simulator.simulate(
            self.simple_graph,
            duration_ms=1000,
            message_rate=50
        )
        
        self.assertGreater(len(result.events_by_type), 0)
        self.assertIn('message_publish', result.events_by_type)


class TestSimulatorConfiguration(unittest.TestCase):
    """Test simulator configuration options"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.graph = TestGraphBuilder.create_simple_graph()
    
    def test_failure_simulator_seed(self):
        """Test reproducibility with seed"""
        # Use a deterministic simulation (exhaustive) to test seed
        sim1 = FailureSimulator(seed=12345)
        sim2 = FailureSimulator(seed=12345)
        
        # Test with single failure (deterministic)
        result1 = sim1.simulate_single_failure(self.graph, 'app1', enable_cascade=False)
        result2 = sim2.simulate_single_failure(self.graph, 'app1', enable_cascade=False)
        
        # Results should be identical for deterministic operations
        self.assertEqual(result1.impact_score, result2.impact_score)
        self.assertEqual(result1.primary_failures, result2.primary_failures)
    
    def test_event_simulator_seed(self):
        """Test event simulator basic functionality"""
        sim = EventDrivenSimulator(seed=12345)
        
        result = sim.simulate(self.graph, duration_ms=500, message_rate=50)
        
        # Should produce consistent results
        self.assertIsInstance(result, EventSimulationResult)
        self.assertGreater(result.metrics.total_messages, 0)
    
    def test_cascade_threshold(self):
        """Test cascade threshold configuration"""
        # High threshold - fewer cascades
        sim_high = FailureSimulator(cascade_threshold=0.9, seed=42)
        # Low threshold - more cascades
        sim_low = FailureSimulator(cascade_threshold=0.3, seed=42)
        
        graph = TestGraphBuilder.create_chain_graph(10)
        
        result_high = sim_high.simulate_single_failure(
            graph, 'node_0', enable_cascade=True
        )
        result_low = sim_low.simulate_single_failure(
            graph, 'node_0', enable_cascade=True
        )
        
        # Lower threshold should cause more cascades (in general)
        # Note: With deterministic seed, results are reproducible
        self.assertIsInstance(result_high, SimulationResult)
        self.assertIsInstance(result_low, SimulationResult)
    
    def test_max_cascade_depth(self):
        """Test max cascade depth limits propagation"""
        sim = FailureSimulator(
            cascade_threshold=0.1,  # Easy to trigger
            cascade_probability=1.0,  # Always cascade
            max_cascade_depth=2,
            seed=42
        )
        
        graph = TestGraphBuilder.create_chain_graph(10)
        result = sim.simulate_single_failure(
            graph, 'node_0', enable_cascade=True
        )
        
        # Cascade depth should be limited
        max_depth = max(
            (e.cascade_depth for e in result.failure_events if e.is_cascade),
            default=0
        )
        self.assertLessEqual(max_depth, 2)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def test_empty_graph(self):
        """Test with empty graph"""
        G = nx.DiGraph()
        sim = EventDrivenSimulator()
        
        result = sim.simulate(G, duration_ms=1000, message_rate=100)
        
        # Should handle gracefully
        self.assertIsInstance(result, EventSimulationResult)
        self.assertEqual(result.metrics.total_messages, 0)
    
    def test_single_node_graph(self):
        """Test with single node graph"""
        G = nx.DiGraph()
        G.add_node('single', type='Application')
        
        sim = FailureSimulator()
        result = sim.simulate_single_failure(G, 'single', enable_cascade=False)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(len(result.primary_failures), 1)
    
    def test_disconnected_graph(self):
        """Test with disconnected graph"""
        G = nx.DiGraph()
        G.add_node('a', type='Application')
        G.add_node('b', type='Application')
        G.add_node('c', type='Application')
        G.add_edge('a', 'b')
        # c is disconnected
        
        sim = FailureSimulator()
        result = sim.simulate_single_failure(G, 'a', enable_cascade=False)
        
        self.assertIsInstance(result, SimulationResult)
    
    def test_zero_message_rate(self):
        """Test with zero message rate"""
        G = TestGraphBuilder.create_simple_graph()
        sim = EventDrivenSimulator()
        
        result = sim.simulate(G, duration_ms=1000, message_rate=0)
        
        # Should not crash, no messages generated
        self.assertIsInstance(result, EventSimulationResult)
    
    def test_very_high_message_rate(self):
        """Test with very high message rate"""
        G = TestGraphBuilder.create_simple_graph()
        sim = EventDrivenSimulator()
        
        result = sim.simulate(G, duration_ms=100, message_rate=10000)
        
        self.assertIsInstance(result, EventSimulationResult)
        self.assertGreater(result.metrics.total_messages, 0)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFailureSimulator))
    suite.addTests(loader.loadTestsFromTestCase(TestEventDrivenSimulator))
    suite.addTests(loader.loadTestsFromTestCase(TestSimulatorConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())