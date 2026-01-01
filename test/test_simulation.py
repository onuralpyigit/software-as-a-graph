#!/usr/bin/env python3
"""
Test Suite for Simulation Module - Version 5.0

Comprehensive tests for:
- Simulation graph model
- Failure simulation
- Event-driven simulation
- Component-type specific simulation

Run with: python -m pytest tests/test_simulation.py -v
Or:       python tests/test_simulation.py

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import unittest
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation import (
    # Graph Model
    SimulationGraph,
    Component,
    Edge,
    ComponentType,
    EdgeType,
    ComponentStatus,
    QoSPolicy,
    create_simulation_graph,
    # Failure Simulator
    FailureSimulator,
    FailureMode,
    FailureResult,
    CampaignResult,
    simulate_single_failure,
    simulate_all_components,
    # Event Simulator
    EventSimulator,
    EventType,
    MessageStatus,
    Message,
    SimulationResult,
    run_event_simulation,
)


# =============================================================================
# Test Data Factory
# =============================================================================

def create_test_graph() -> SimulationGraph:
    """Create a small test graph for unit tests"""
    graph = SimulationGraph()
    
    # Add components
    # 2 Applications
    graph.add_component(Component(
        id="app1", type=ComponentType.APPLICATION, name="Publisher App"
    ))
    graph.add_component(Component(
        id="app2", type=ComponentType.APPLICATION, name="Subscriber App"
    ))
    
    # 1 Broker
    graph.add_component(Component(
        id="broker1", type=ComponentType.BROKER, name="Main Broker"
    ))
    
    # 2 Topics
    graph.add_component(Component(
        id="topic1", type=ComponentType.TOPIC, name="Orders Topic"
    ))
    graph.add_component(Component(
        id="topic2", type=ComponentType.TOPIC, name="Events Topic"
    ))
    
    # 1 Node
    graph.add_component(Component(
        id="node1", type=ComponentType.NODE, name="Server 1"
    ))
    
    # Add edges
    # app1 publishes to topic1
    graph.add_edge(Edge(
        source="app1", target="topic1", edge_type=EdgeType.PUBLISHES_TO
    ))
    # app2 subscribes to topic1
    graph.add_edge(Edge(
        source="app2", target="topic1", edge_type=EdgeType.SUBSCRIBES_TO
    ))
    # broker1 routes both topics
    graph.add_edge(Edge(
        source="broker1", target="topic1", edge_type=EdgeType.ROUTES
    ))
    graph.add_edge(Edge(
        source="broker1", target="topic2", edge_type=EdgeType.ROUTES
    ))
    # Components run on node1
    graph.add_edge(Edge(
        source="app1", target="node1", edge_type=EdgeType.RUNS_ON
    ))
    graph.add_edge(Edge(
        source="app2", target="node1", edge_type=EdgeType.RUNS_ON
    ))
    graph.add_edge(Edge(
        source="broker1", target="node1", edge_type=EdgeType.RUNS_ON
    ))
    
    return graph


def create_larger_test_graph() -> SimulationGraph:
    """Create a larger test graph for campaign tests"""
    graph = SimulationGraph()
    
    # 5 Applications
    for i in range(1, 6):
        graph.add_component(Component(
            id=f"app{i}", type=ComponentType.APPLICATION
        ))
    
    # 2 Brokers
    for i in range(1, 3):
        graph.add_component(Component(
            id=f"broker{i}", type=ComponentType.BROKER
        ))
    
    # 5 Topics
    for i in range(1, 6):
        graph.add_component(Component(
            id=f"topic{i}", type=ComponentType.TOPIC
        ))
    
    # 2 Nodes
    for i in range(1, 3):
        graph.add_component(Component(
            id=f"node{i}", type=ComponentType.NODE
        ))
    
    # Pub/Sub relationships
    graph.add_edge(Edge(source="app1", target="topic1", edge_type=EdgeType.PUBLISHES_TO))
    graph.add_edge(Edge(source="app1", target="topic2", edge_type=EdgeType.PUBLISHES_TO))
    graph.add_edge(Edge(source="app2", target="topic1", edge_type=EdgeType.SUBSCRIBES_TO))
    graph.add_edge(Edge(source="app2", target="topic2", edge_type=EdgeType.SUBSCRIBES_TO))
    graph.add_edge(Edge(source="app3", target="topic3", edge_type=EdgeType.PUBLISHES_TO))
    graph.add_edge(Edge(source="app4", target="topic3", edge_type=EdgeType.SUBSCRIBES_TO))
    graph.add_edge(Edge(source="app5", target="topic4", edge_type=EdgeType.SUBSCRIBES_TO))
    
    # Broker routes
    graph.add_edge(Edge(source="broker1", target="topic1", edge_type=EdgeType.ROUTES))
    graph.add_edge(Edge(source="broker1", target="topic2", edge_type=EdgeType.ROUTES))
    graph.add_edge(Edge(source="broker1", target="topic3", edge_type=EdgeType.ROUTES))
    graph.add_edge(Edge(source="broker2", target="topic4", edge_type=EdgeType.ROUTES))
    graph.add_edge(Edge(source="broker2", target="topic5", edge_type=EdgeType.ROUTES))
    
    # RUNS_ON
    graph.add_edge(Edge(source="app1", target="node1", edge_type=EdgeType.RUNS_ON))
    graph.add_edge(Edge(source="app2", target="node1", edge_type=EdgeType.RUNS_ON))
    graph.add_edge(Edge(source="app3", target="node2", edge_type=EdgeType.RUNS_ON))
    graph.add_edge(Edge(source="broker1", target="node1", edge_type=EdgeType.RUNS_ON))
    graph.add_edge(Edge(source="broker2", target="node2", edge_type=EdgeType.RUNS_ON))
    
    return graph


# =============================================================================
# Test: Enums
# =============================================================================

class TestEnums(unittest.TestCase):
    """Test enum definitions"""
    
    def test_component_type_values(self):
        """Test ComponentType enum values"""
        self.assertEqual(ComponentType.APPLICATION.value, "Application")
        self.assertEqual(ComponentType.BROKER.value, "Broker")
        self.assertEqual(ComponentType.TOPIC.value, "Topic")
        self.assertEqual(ComponentType.NODE.value, "Node")
    
    def test_component_type_from_string(self):
        """Test ComponentType.from_string()"""
        self.assertEqual(ComponentType.from_string("application"), ComponentType.APPLICATION)
        self.assertEqual(ComponentType.from_string("BROKER"), ComponentType.BROKER)
        self.assertEqual(ComponentType.from_string("Topic"), ComponentType.TOPIC)
    
    def test_edge_type_values(self):
        """Test EdgeType enum values"""
        self.assertEqual(EdgeType.PUBLISHES_TO.value, "PUBLISHES_TO")
        self.assertEqual(EdgeType.SUBSCRIBES_TO.value, "SUBSCRIBES_TO")
        self.assertEqual(EdgeType.ROUTES.value, "ROUTES")
        self.assertEqual(EdgeType.RUNS_ON.value, "RUNS_ON")
        self.assertEqual(EdgeType.CONNECTS_TO.value, "CONNECTS_TO")
    
    def test_failure_mode_impact_factor(self):
        """Test FailureMode impact factors"""
        self.assertEqual(FailureMode.CRASH.impact_factor, 1.0)
        self.assertEqual(FailureMode.DEGRADED.impact_factor, 0.5)
        self.assertEqual(FailureMode.NETWORK_PARTITION.impact_factor, 0.8)
        self.assertEqual(FailureMode.OVERLOAD.impact_factor, 0.3)
    
    def test_event_type_values(self):
        """Test EventType enum"""
        self.assertIn(EventType.PUBLISH, EventType)
        self.assertIn(EventType.ROUTE, EventType)
        self.assertIn(EventType.DELIVER, EventType)
        self.assertIn(EventType.FAILURE, EventType)
    
    def test_message_status_values(self):
        """Test MessageStatus enum"""
        self.assertEqual(MessageStatus.PENDING.value, "pending")
        self.assertEqual(MessageStatus.DELIVERED.value, "delivered")
        self.assertEqual(MessageStatus.FAILED.value, "failed")


# =============================================================================
# Test: Data Classes
# =============================================================================

class TestDataClasses(unittest.TestCase):
    """Test data class serialization"""
    
    def test_qos_policy_to_dict(self):
        """Test QoSPolicy serialization"""
        qos = QoSPolicy(
            reliability="reliable",
            durability="persistent",
            priority=5,
            bandwidth=2.0,
            latency_ms=20.0
        )
        d = qos.to_dict()
        
        self.assertEqual(d["reliability"], "reliable")
        self.assertEqual(d["durability"], "persistent")
        self.assertEqual(d["priority"], 5)
        self.assertEqual(d["bandwidth"], 2.0)
    
    def test_qos_policy_from_dict(self):
        """Test QoSPolicy deserialization"""
        d = {"reliability": "reliable", "priority": 3}
        qos = QoSPolicy.from_dict(d)
        
        self.assertEqual(qos.reliability, "reliable")
        self.assertEqual(qos.priority, 3)
        # Defaults
        self.assertEqual(qos.durability, "volatile")
    
    def test_component_to_dict(self):
        """Test Component serialization"""
        comp = Component(
            id="test_app",
            type=ComponentType.APPLICATION,
            name="Test App",
            status=ComponentStatus.HEALTHY,
            properties={"version": "1.0"}
        )
        d = comp.to_dict()
        
        self.assertEqual(d["id"], "test_app")
        self.assertEqual(d["type"], "Application")
        self.assertEqual(d["status"], "healthy")
        self.assertEqual(d["properties"]["version"], "1.0")
    
    def test_edge_to_dict(self):
        """Test Edge serialization"""
        edge = Edge(
            source="app1",
            target="topic1",
            edge_type=EdgeType.PUBLISHES_TO,
            weight=2.0
        )
        d = edge.to_dict()
        
        self.assertEqual(d["source"], "app1")
        self.assertEqual(d["target"], "topic1")
        self.assertEqual(d["type"], "PUBLISHES_TO")
        self.assertEqual(d["weight"], 2.0)
    
    def test_edge_key(self):
        """Test Edge key property"""
        edge = Edge(source="a", target="b", edge_type=EdgeType.ROUTES)
        self.assertEqual(edge.key, ("a", "b", "ROUTES"))


# =============================================================================
# Test: Simulation Graph
# =============================================================================

class TestSimulationGraph(unittest.TestCase):
    """Test SimulationGraph class"""
    
    def setUp(self):
        self.graph = create_test_graph()
    
    def test_component_count(self):
        """Test component counts"""
        self.assertEqual(len(self.graph.components), 6)
    
    def test_edge_count(self):
        """Test edge counts"""
        self.assertEqual(len(self.graph.edges), 7)
    
    def test_get_component(self):
        """Test get_component()"""
        comp = self.graph.get_component("app1")
        self.assertIsNotNone(comp)
        self.assertEqual(comp.id, "app1")
        self.assertEqual(comp.type, ComponentType.APPLICATION)
    
    def test_get_components_by_type(self):
        """Test get_components_by_type()"""
        apps = self.graph.get_components_by_type(ComponentType.APPLICATION)
        self.assertEqual(len(apps), 2)
        
        brokers = self.graph.get_components_by_type(ComponentType.BROKER)
        self.assertEqual(len(brokers), 1)
    
    def test_get_publishers(self):
        """Test get_publishers()"""
        publishers = self.graph.get_publishers("topic1")
        self.assertEqual(publishers, {"app1"})
    
    def test_get_subscribers(self):
        """Test get_subscribers()"""
        subscribers = self.graph.get_subscribers("topic1")
        self.assertEqual(subscribers, {"app2"})
    
    def test_get_topics_published_by(self):
        """Test get_topics_published_by()"""
        topics = self.graph.get_topics_published_by("app1")
        self.assertEqual(topics, {"topic1"})
    
    def test_get_broker_for_topic(self):
        """Test get_broker_for_topic()"""
        broker = self.graph.get_broker_for_topic("topic1")
        self.assertEqual(broker, "broker1")
    
    def test_get_node_for_component(self):
        """Test get_node_for_component()"""
        node = self.graph.get_node_for_component("app1")
        self.assertEqual(node, "node1")
    
    def test_get_components_on_node(self):
        """Test get_components_on_node()"""
        components = self.graph.get_components_on_node("node1")
        self.assertIn("app1", components)
        self.assertIn("app2", components)
        self.assertIn("broker1", components)
    
    def test_get_downstream_components(self):
        """Test get_downstream_components()"""
        downstream = self.graph.get_downstream_components("app1")
        self.assertIn("topic1", downstream)
        self.assertIn("app2", downstream)
    
    def test_statistics(self):
        """Test get_statistics()"""
        stats = self.graph.get_statistics()
        
        self.assertEqual(stats["total_components"], 6)
        self.assertEqual(stats["total_edges"], 7)
        self.assertEqual(stats["by_component_type"]["Application"], 2)
        self.assertEqual(stats["by_edge_type"]["PUBLISHES_TO"], 1)
    
    def test_to_dict(self):
        """Test to_dict() serialization"""
        d = self.graph.to_dict()
        
        self.assertIn("components", d)
        self.assertIn("edges", d)
        self.assertIn("statistics", d)
        self.assertEqual(len(d["components"]), 6)
    
    def test_from_dict(self):
        """Test from_dict() deserialization"""
        d = self.graph.to_dict()
        restored = SimulationGraph.from_dict(d)
        
        self.assertEqual(len(restored.components), 6)
        self.assertEqual(len(restored.edges), 7)
    
    def test_subgraph_by_type(self):
        """Test get_subgraph_by_type()"""
        subgraph = self.graph.get_subgraph_by_type(ComponentType.APPLICATION)
        
        self.assertEqual(len(subgraph.components), 2)
        self.assertEqual(len(subgraph.edges), 0)  # No edges between apps
    
    def test_remove_component(self):
        """Test remove_component()"""
        removed = self.graph.remove_component("app1")
        
        self.assertIsNotNone(removed)
        self.assertEqual(removed.id, "app1")
        self.assertIsNone(self.graph.get_component("app1"))


# =============================================================================
# Test: Failure Simulator
# =============================================================================

class TestFailureSimulator(unittest.TestCase):
    """Test FailureSimulator class"""
    
    def setUp(self):
        self.graph = create_test_graph()
        self.simulator = FailureSimulator(seed=42, cascade=True)
    
    def test_single_failure(self):
        """Test single component failure"""
        result = self.simulator.simulate_failure(self.graph, "broker1")
        
        self.assertEqual(result.failed_component, "broker1")
        self.assertEqual(result.component_type, ComponentType.BROKER)
        self.assertGreater(result.impact_score, 0)
        self.assertLessEqual(result.impact_score, 1.0)
    
    def test_broker_failure_affects_topics(self):
        """Test that broker failure affects routed topics"""
        result = self.simulator.simulate_failure(self.graph, "broker1")
        
        # Broker routes topic1 and topic2
        self.assertIn("topic1", result.directly_affected)
        self.assertIn("topic2", result.directly_affected)
    
    def test_node_failure_affects_components(self):
        """Test that node failure affects components on it"""
        result = self.simulator.simulate_failure(self.graph, "node1")
        
        # app1, app2, broker1 run on node1
        self.assertIn("app1", result.directly_affected)
        self.assertIn("app2", result.directly_affected)
        self.assertIn("broker1", result.directly_affected)
    
    def test_topic_failure_affects_subscribers(self):
        """Test that topic failure affects subscribers"""
        result = self.simulator.simulate_failure(self.graph, "topic1")
        
        self.assertIn("app2", result.directly_affected)  # subscriber
        self.assertIn("app1", result.directly_affected)  # publisher
    
    def test_failure_mode_impact(self):
        """Test different failure modes have different impacts"""
        crash_result = self.simulator.simulate_failure(
            self.graph, "broker1", FailureMode.CRASH
        )
        degraded_result = self.simulator.simulate_failure(
            self.graph, "broker1", FailureMode.DEGRADED
        )
        
        # CRASH has higher impact factor than DEGRADED
        self.assertGreater(crash_result.impact_score, degraded_result.impact_score)
    
    def test_no_cascade(self):
        """Test simulation without cascade"""
        no_cascade_sim = FailureSimulator(seed=42, cascade=False)
        result = no_cascade_sim.simulate_failure(self.graph, "broker1")
        
        self.assertEqual(len(result.cascade_affected), 0)
    
    def test_result_to_dict(self):
        """Test FailureResult serialization"""
        result = self.simulator.simulate_failure(self.graph, "broker1")
        d = result.to_dict()
        
        self.assertIn("failed_component", d)
        self.assertIn("impact_score", d)
        self.assertIn("directly_affected", d)
        self.assertIn("affected_by_type", d)


class TestFailureCampaign(unittest.TestCase):
    """Test failure campaign simulation"""
    
    def setUp(self):
        self.graph = create_larger_test_graph()
        self.simulator = FailureSimulator(seed=42)
    
    def test_full_campaign(self):
        """Test simulating all component failures"""
        campaign = self.simulator.simulate_all_failures(self.graph)
        
        self.assertEqual(len(campaign.results), len(self.graph.components))
        self.assertIsInstance(campaign.component_impacts, dict)
    
    def test_campaign_by_type(self):
        """Test campaign for specific component type"""
        campaign = self.simulator.simulate_by_component_type(
            self.graph, ComponentType.APPLICATION
        )
        
        # Should only test applications
        app_count = len(self.graph.get_components_by_type(ComponentType.APPLICATION))
        self.assertEqual(len(campaign.results), app_count)
    
    def test_ranked_by_impact(self):
        """Test impact ranking"""
        campaign = self.simulator.simulate_all_failures(self.graph)
        ranked = campaign.ranked_by_impact
        
        # Should be sorted descending by impact
        for i in range(len(ranked) - 1):
            self.assertGreaterEqual(ranked[i][1], ranked[i + 1][1])
    
    def test_critical_components(self):
        """Test critical component identification"""
        campaign = self.simulator.simulate_all_failures(self.graph)
        
        # All critical components should have impact >= threshold
        for comp_id in campaign.critical_components:
            self.assertGreaterEqual(
                campaign.component_impacts[comp_id],
                campaign.critical_threshold
            )
    
    def test_campaign_to_dict(self):
        """Test CampaignResult serialization"""
        campaign = self.simulator.simulate_all_failures(self.graph)
        d = campaign.to_dict()
        
        self.assertIn("results", d)
        self.assertIn("component_impacts", d)
        self.assertIn("critical_count", d)


class TestSimultaneousFailures(unittest.TestCase):
    """Test simultaneous failure simulation"""
    
    def setUp(self):
        self.graph = create_larger_test_graph()
        self.simulator = FailureSimulator(seed=42)
    
    def test_simultaneous_failures(self):
        """Test multiple simultaneous failures"""
        result = self.simulator.simulate_simultaneous_failures(
            self.graph, ["broker1", "broker2"]
        )
        
        # Impact should be higher than single failure
        single_result = self.simulator.simulate_failure(self.graph, "broker1")
        self.assertGreaterEqual(result.impact_score, single_result.impact_score)


# =============================================================================
# Test: Event Simulator
# =============================================================================

class TestEventSimulator(unittest.TestCase):
    """Test EventSimulator class"""
    
    def setUp(self):
        self.graph = create_test_graph()
        self.simulator = EventSimulator(self.graph, seed=42)
    
    def test_basic_simulation(self):
        """Test basic event simulation"""
        result = self.simulator.run(duration=100, message_rate=1.0)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertGreater(result.statistics.total_messages, 0)
    
    def test_message_delivery(self):
        """Test messages are delivered"""
        result = self.simulator.run(duration=500, message_rate=5.0)
        
        # Some messages should be delivered
        self.assertGreater(result.statistics.delivered_messages, 0)
        self.assertGreater(result.statistics.delivery_rate, 0)
    
    def test_statistics_calculation(self):
        """Test statistics are calculated correctly"""
        result = self.simulator.run(duration=200, message_rate=10.0)
        stats = result.statistics
        
        # Sanity checks
        total = stats.delivered_messages + stats.failed_messages + \
                stats.dropped_messages + stats.timeout_messages
        self.assertLessEqual(total, stats.total_messages)
        
        # Delivery rate should be between 0 and 1
        self.assertGreaterEqual(stats.delivery_rate, 0)
        self.assertLessEqual(stats.delivery_rate, 1)
    
    def test_component_load_tracking(self):
        """Test component load statistics"""
        result = self.simulator.run(duration=500, message_rate=10.0)
        
        # Should have load data for components
        self.assertGreater(len(result.component_loads), 0)
        
        # At least some components should have processed messages
        total_processed = sum(
            load.messages_processed 
            for load in result.component_loads.values()
        )
        self.assertGreater(total_processed, 0)
    
    def test_inject_failure(self):
        """Test failure injection during simulation"""
        self.simulator.inject_failure("broker1", at_time=50.0)
        result = self.simulator.run(duration=200, message_rate=10.0)
        
        # Broker should be in failed state
        self.assertIn("broker1", result.failed_components)
    
    def test_inject_temporary_failure(self):
        """Test temporary failure injection"""
        self.simulator.inject_failure("broker1", at_time=50.0, duration=50.0)
        result = self.simulator.run(duration=200, message_rate=10.0)
        
        # Broker should have recovered
        self.assertNotIn("broker1", result.failed_components)
    
    def test_reset(self):
        """Test simulator reset"""
        self.simulator.run(duration=100, message_rate=5.0)
        self.simulator.reset()
        
        self.assertEqual(self.simulator.current_time, 0.0)
        self.assertEqual(len(self.simulator.event_queue), 0)
        self.assertEqual(len(self.simulator.messages), 0)
    
    def test_result_to_dict(self):
        """Test SimulationResult serialization"""
        result = self.simulator.run(duration=100, message_rate=5.0)
        d = result.to_dict()
        
        self.assertIn("statistics", d)
        self.assertIn("component_loads", d)
        self.assertIn("message_count", d)


class TestEventSimulatorPerformance(unittest.TestCase):
    """Test event simulator with larger workloads"""
    
    def setUp(self):
        self.graph = create_larger_test_graph()
    
    def test_high_message_rate(self):
        """Test simulation with high message rate"""
        simulator = EventSimulator(self.graph, seed=42)
        result = simulator.run(duration=1000, message_rate=50.0)
        
        # Should handle many messages
        self.assertGreater(result.statistics.total_messages, 100)
    
    def test_long_duration(self):
        """Test longer simulation duration"""
        simulator = EventSimulator(self.graph, seed=42)
        result = simulator.run(duration=5000, message_rate=10.0)
        
        self.assertEqual(result.statistics.simulation_time, 5000)


# =============================================================================
# Test: Factory Functions
# =============================================================================

class TestFactoryFunctions(unittest.TestCase):
    """Test module factory functions"""
    
    def setUp(self):
        self.graph = create_test_graph()
    
    def test_simulate_single_failure(self):
        """Test simulate_single_failure()"""
        result = simulate_single_failure(self.graph, "broker1")
        
        self.assertIsInstance(result, FailureResult)
        self.assertEqual(result.failed_component, "broker1")
    
    def test_simulate_all_components(self):
        """Test simulate_all_components()"""
        campaign = simulate_all_components(self.graph)
        
        self.assertIsInstance(campaign, CampaignResult)
        self.assertEqual(len(campaign.results), len(self.graph.components))
    
    def test_run_event_simulation(self):
        """Test run_event_simulation()"""
        result = run_event_simulation(self.graph, duration=100, message_rate=5.0)
        
        self.assertIsInstance(result, SimulationResult)


# =============================================================================
# Test: Integration
# =============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_full_failure_analysis_workflow(self):
        """Test complete failure analysis workflow"""
        # Create graph
        graph = create_larger_test_graph()
        
        # Run campaign
        simulator = FailureSimulator(seed=42)
        campaign = simulator.simulate_all_failures(graph)
        
        # Get critical components
        critical = campaign.critical_components
        
        # Verify results are consistent
        for comp_id in critical:
            self.assertIn(comp_id, campaign.component_impacts)
            self.assertGreaterEqual(
                campaign.component_impacts[comp_id],
                campaign.critical_threshold
            )
    
    def test_full_event_simulation_workflow(self):
        """Test complete event simulation workflow"""
        # Create graph
        graph = create_larger_test_graph()
        
        # Run simulation
        simulator = EventSimulator(graph, seed=42)
        
        # Inject a failure
        simulator.inject_failure("broker1", at_time=250, duration=100)
        
        # Run
        result = simulator.run(duration=1000, message_rate=20.0)
        
        # Verify results
        self.assertGreater(result.statistics.total_messages, 0)
        self.assertIsNotNone(result.statistics.delivery_rate)
        
        # Get bottlenecks
        bottlenecks = result.get_bottlenecks(threshold=10)
        self.assertIsInstance(bottlenecks, list)
    
    def test_graph_serialization_roundtrip(self):
        """Test graph can be serialized and restored"""
        import json
        
        original = create_larger_test_graph()
        
        # Serialize
        json_str = json.dumps(original.to_dict(), default=str)
        
        # Deserialize
        data = json.loads(json_str)
        restored = SimulationGraph.from_dict(data)
        
        # Verify
        self.assertEqual(len(restored.components), len(original.components))
        self.assertEqual(len(restored.edges), len(original.edges))


# =============================================================================
# Main
# =============================================================================

def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEnums))
    suite.addTests(loader.loadTestsFromTestCase(TestDataClasses))
    suite.addTests(loader.loadTestsFromTestCase(TestSimulationGraph))
    suite.addTests(loader.loadTestsFromTestCase(TestFailureSimulator))
    suite.addTests(loader.loadTestsFromTestCase(TestFailureCampaign))
    suite.addTests(loader.loadTestsFromTestCase(TestSimultaneousFailures))
    suite.addTests(loader.loadTestsFromTestCase(TestEventSimulator))
    suite.addTests(loader.loadTestsFromTestCase(TestEventSimulatorPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestFactoryFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
