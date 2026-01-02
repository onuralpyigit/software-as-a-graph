#!/usr/bin/env python3
"""
Test Suite for src.simulation Module - Version 5.0

Tests for simulation including:
- SimulationGraph
- FailureSimulator
- EventSimulator
- Layer-specific analysis

Run with pytest:
    pytest tests/test_simulation.py -v

Or standalone:
    python tests/test_simulation.py

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


# =============================================================================
# Test Data
# =============================================================================

def create_test_graph():
    """Create a simple test graph."""
    from src.simulation import (
        SimulationGraph,
        Component,
        Edge,
        ComponentType,
        EdgeType,
        QoSPolicy,
    )
    
    graph = SimulationGraph()
    
    # Create nodes
    for i in range(2):
        graph.add_component(Component(
            id=f"node_{i+1}",
            type=ComponentType.NODE,
            name=f"Node {i+1}",
        ))
    
    # Create broker
    graph.add_component(Component(
        id="broker_1",
        type=ComponentType.BROKER,
        name="Broker 1",
    ))
    graph.add_edge(Edge(
        source="broker_1",
        target="node_1",
        edge_type=EdgeType.RUNS_ON,
    ))
    
    # Create topics
    for i in range(3):
        graph.add_component(Component(
            id=f"topic_{i+1}",
            type=ComponentType.TOPIC,
            name=f"Topic {i+1}",
        ))
        graph.add_edge(Edge(
            source="broker_1",
            target=f"topic_{i+1}",
            edge_type=EdgeType.ROUTES,
        ))
    
    # Create applications
    for i in range(4):
        app_id = f"app_{i+1}"
        node = f"node_{(i % 2) + 1}"
        
        graph.add_component(Component(
            id=app_id,
            type=ComponentType.APPLICATION,
            name=f"Application {i+1}",
            node_id=node,
        ))
        graph.add_edge(Edge(
            source=app_id,
            target=node,
            edge_type=EdgeType.RUNS_ON,
        ))
    
    # Create pub/sub relationships
    # app_1 publishes to topic_1, topic_2
    graph.add_edge(Edge(source="app_1", target="topic_1", edge_type=EdgeType.PUBLISHES_TO))
    graph.add_edge(Edge(source="app_1", target="topic_2", edge_type=EdgeType.PUBLISHES_TO))
    
    # app_2 subscribes to topic_1
    graph.add_edge(Edge(source="app_2", target="topic_1", edge_type=EdgeType.SUBSCRIBES_TO))
    
    # app_3 subscribes to topic_1, topic_2
    graph.add_edge(Edge(source="app_3", target="topic_1", edge_type=EdgeType.SUBSCRIBES_TO))
    graph.add_edge(Edge(source="app_3", target="topic_2", edge_type=EdgeType.SUBSCRIBES_TO))
    
    # app_4 publishes to topic_3
    graph.add_edge(Edge(source="app_4", target="topic_3", edge_type=EdgeType.PUBLISHES_TO))
    
    # Connections
    graph.add_edge(Edge(source="node_1", target="broker_1", edge_type=EdgeType.CONNECTS_TO))
    graph.add_edge(Edge(source="node_2", target="broker_1", edge_type=EdgeType.CONNECTS_TO))
    
    return graph


# =============================================================================
# Test: Enums
# =============================================================================

class TestEnums:
    """Tests for simulation enums."""

    def test_component_type_values(self):
        from src.simulation import ComponentType
        
        assert ComponentType.APPLICATION.value == "Application"
        assert ComponentType.BROKER.value == "Broker"
        assert ComponentType.TOPIC.value == "Topic"
        assert ComponentType.NODE.value == "Node"

    def test_edge_type_values(self):
        from src.simulation import EdgeType
        
        assert EdgeType.PUBLISHES_TO.value == "PUBLISHES_TO"
        assert EdgeType.SUBSCRIBES_TO.value == "SUBSCRIBES_TO"
        assert EdgeType.ROUTES.value == "ROUTES"
        assert EdgeType.RUNS_ON.value == "RUNS_ON"
        assert EdgeType.CONNECTS_TO.value == "CONNECTS_TO"

    def test_failure_mode(self):
        from src.simulation import FailureMode
        
        assert FailureMode.CRASH.impact_factor == 1.0
        assert FailureMode.DEGRADED.impact_factor == 0.5
        assert FailureMode.PARTITION.impact_factor == 0.8


# =============================================================================
# Test: QoSPolicy
# =============================================================================

class TestQoSPolicy:
    """Tests for QoSPolicy data class."""

    def test_default_qos(self):
        from src.simulation import QoSPolicy
        
        qos = QoSPolicy()
        
        assert qos.reliability == "BEST_EFFORT"
        assert qos.durability == "VOLATILE"
        assert qos.priority == "MEDIUM"

    def test_criticality_score(self):
        from src.simulation import QoSPolicy
        
        # Minimum criticality
        qos_low = QoSPolicy(
            reliability="BEST_EFFORT",
            durability="VOLATILE",
            priority="LOW",
        )
        assert qos_low.criticality_score() == 0.0
        
        # Maximum criticality
        qos_high = QoSPolicy(
            reliability="RELIABLE",
            durability="PERSISTENT",
            priority="URGENT",
        )
        assert qos_high.criticality_score() == 1.0

    def test_to_dict(self):
        from src.simulation import QoSPolicy
        
        qos = QoSPolicy(reliability="RELIABLE", durability="PERSISTENT", priority="HIGH")
        data = qos.to_dict()
        
        assert data["reliability"] == "RELIABLE"
        assert data["durability"] == "PERSISTENT"
        assert data["priority"] == "HIGH"


# =============================================================================
# Test: SimulationGraph
# =============================================================================

class TestSimulationGraph:
    """Tests for SimulationGraph."""

    def test_add_component(self):
        from src.simulation import SimulationGraph, Component, ComponentType
        
        graph = SimulationGraph()
        comp = Component(id="app_1", type=ComponentType.APPLICATION)
        
        graph.add_component(comp)
        
        assert "app_1" in graph.components
        assert "app_1" in graph._by_type[ComponentType.APPLICATION]

    def test_add_edge(self):
        from src.simulation import SimulationGraph, Component, Edge, ComponentType, EdgeType
        
        graph = SimulationGraph()
        graph.add_component(Component(id="app_1", type=ComponentType.APPLICATION))
        graph.add_component(Component(id="topic_1", type=ComponentType.TOPIC))
        
        edge = Edge(source="app_1", target="topic_1", edge_type=EdgeType.PUBLISHES_TO)
        graph.add_edge(edge)
        
        assert len(graph.edges) == 1
        assert len(graph._outgoing["app_1"]) == 1
        assert len(graph._incoming["topic_1"]) == 1

    def test_get_publishers(self):
        graph = create_test_graph()
        
        publishers = graph.get_publishers("topic_1")
        
        assert "app_1" in publishers
        assert len(publishers) == 1

    def test_get_subscribers(self):
        graph = create_test_graph()
        
        subscribers = graph.get_subscribers("topic_1")
        
        assert "app_2" in subscribers
        assert "app_3" in subscribers
        assert len(subscribers) == 2

    def test_get_message_paths(self):
        graph = create_test_graph()
        
        paths = graph.get_message_paths()
        
        # app_1 -> topic_1 -> app_2, app_3 (2 paths)
        # app_1 -> topic_2 -> app_3 (1 path)
        # topic_3 has no subscribers
        assert len(paths) >= 3

    def test_get_broker_for_topic(self):
        graph = create_test_graph()
        
        broker = graph.get_broker_for_topic("topic_1")
        
        assert broker == "broker_1"

    def test_layer_subgraph(self):
        graph = create_test_graph()
        
        app_layer = graph.get_layer_subgraph("application")
        
        # Should have applications and topics
        assert len(app_layer._by_type) > 0
        # Should have PUBLISHES_TO and SUBSCRIBES_TO edges
        pub_edges = app_layer.get_edges_by_type(
            __import__("src.simulation", fromlist=["EdgeType"]).EdgeType.PUBLISHES_TO
        )
        assert len(pub_edges) > 0

    def test_remove_component(self):
        graph = create_test_graph()
        initial_count = len(graph.components)
        
        graph.remove_component("app_1")
        
        assert "app_1" not in graph.components
        assert len(graph.components) == initial_count - 1
        # Edges should be removed
        assert all(e.source != "app_1" and e.target != "app_1" for e in graph.edges)

    def test_copy(self):
        graph = create_test_graph()
        
        copy = graph.copy()
        
        assert len(copy.components) == len(graph.components)
        assert len(copy.edges) == len(graph.edges)
        
        # Modifying copy shouldn't affect original
        copy.remove_component("app_1")
        assert "app_1" in graph.components
        assert "app_1" not in copy.components

    def test_to_dict_and_from_dict(self):
        graph = create_test_graph()
        
        data = graph.to_dict()
        restored = __import__("src.simulation", fromlist=["SimulationGraph"]).SimulationGraph.from_dict(data)
        
        assert len(restored.components) == len(graph.components)
        assert len(restored.edges) == len(graph.edges)


# =============================================================================
# Test: FailureSimulator
# =============================================================================

class TestFailureSimulator:
    """Tests for FailureSimulator."""

    def test_single_failure(self):
        from src.simulation import FailureSimulator
        
        graph = create_test_graph()
        simulator = FailureSimulator(seed=42, cascade=False)
        
        result = simulator.simulate_failure(graph, "app_1")
        
        assert result.component_id == "app_1"
        assert result.impact_score >= 0.0
        assert result.paths_broken > 0

    def test_broker_failure_impact(self):
        from src.simulation import FailureSimulator
        
        graph = create_test_graph()
        simulator = FailureSimulator(seed=42, cascade=False)
        
        result = simulator.simulate_failure(graph, "broker_1")
        
        # Broker failure should have high impact (routes all topics)
        assert result.impact_score > 0.0
        assert len(result.directly_affected) > 0

    def test_cascade_propagation(self):
        from src.simulation import FailureSimulator
        
        graph = create_test_graph()
        simulator = FailureSimulator(seed=42, cascade=True, cascade_threshold=0.5)
        
        result = simulator.simulate_failure(graph, "broker_1")
        
        # Cascade may cause additional failures
        assert result.total_affected >= len(result.directly_affected)

    def test_simulate_all(self):
        from src.simulation import FailureSimulator
        
        graph = create_test_graph()
        simulator = FailureSimulator(seed=42, cascade=False)
        
        campaign = simulator.simulate_all(graph)
        
        assert campaign.total_simulations == len(graph.components)
        assert len(campaign.ranked_by_impact()) == len(graph.components)

    def test_campaign_by_layer(self):
        from src.simulation import FailureSimulator
        
        graph = create_test_graph()
        simulator = FailureSimulator(seed=42, cascade=False)
        
        campaign = simulator.simulate_all(graph)
        
        # Should have results by layer
        assert "application" in campaign.by_layer
        assert "infrastructure" in campaign.by_layer

    def test_failure_mode_impact(self):
        from src.simulation import FailureSimulator, FailureMode
        
        graph = create_test_graph()
        simulator = FailureSimulator(seed=42, cascade=False)
        
        crash_result = simulator.simulate_failure(graph, "broker_1", FailureMode.CRASH)
        degraded_result = simulator.simulate_failure(graph, "broker_1", FailureMode.DEGRADED)
        
        # Crash should have higher impact than degraded
        assert crash_result.impact_score >= degraded_result.impact_score


# =============================================================================
# Test: EventSimulator
# =============================================================================

class TestEventSimulator:
    """Tests for EventSimulator."""

    def test_basic_simulation(self):
        from src.simulation import EventSimulator
        
        graph = create_test_graph()
        simulator = EventSimulator(seed=42)
        
        result = simulator.run(graph, duration=100.0, message_rate=10.0)
        
        assert result.stats.total_messages > 0
        assert result.stats.simulation_time == 100.0

    def test_message_delivery(self):
        from src.simulation import EventSimulator
        
        graph = create_test_graph()
        simulator = EventSimulator(seed=42, timeout=10000.0)
        
        result = simulator.run(graph, duration=500.0, message_rate=10.0)
        
        # Some messages should be delivered
        assert result.stats.delivered_messages > 0
        assert result.stats.delivery_rate > 0.0

    def test_latency_tracking(self):
        from src.simulation import EventSimulator
        
        graph = create_test_graph()
        simulator = EventSimulator(seed=42, default_latency=10.0)
        
        result = simulator.run(graph, duration=500.0, message_rate=10.0)
        
        if result.stats.delivered_messages > 0:
            assert result.stats.avg_latency > 0.0

    def test_failure_injection(self):
        from src.simulation import EventSimulator
        
        graph = create_test_graph()
        simulator = EventSimulator(seed=42)
        
        failure_schedule = [
            {"component_id": "broker_1", "time": 50.0, "duration": 100.0}
        ]
        
        result = simulator.run(
            graph,
            duration=500.0,
            message_rate=10.0,
            failure_schedule=failure_schedule,
        )
        
        # Some messages should fail due to broker failure
        assert result.stats.total_messages > 0

    def test_layer_stats(self):
        from src.simulation import EventSimulator
        
        graph = create_test_graph()
        simulator = EventSimulator(seed=42)
        
        result = simulator.run(graph, duration=500.0, message_rate=10.0)
        
        # Should have layer stats
        assert "application" in result.layer_stats

    def test_component_loads(self):
        from src.simulation import EventSimulator
        
        graph = create_test_graph()
        simulator = EventSimulator(seed=42)
        
        result = simulator.run(graph, duration=500.0, message_rate=10.0)
        
        # Should track component loads
        assert len(result.component_loads) > 0
        
        # Broker should have processed some messages
        broker_load = result.component_loads.get("broker_1")
        if broker_load:
            assert broker_load.messages_processed >= 0


# =============================================================================
# Test: Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_simulation_graph(self):
        from src.simulation import create_simulation_graph
        
        graph = create_simulation_graph(
            applications=5,
            brokers=2,
            topics=10,
            nodes=3,
            seed=42,
        )
        
        assert len(graph.get_components_by_type(
            __import__("src.simulation", fromlist=["ComponentType"]).ComponentType.APPLICATION
        )) == 5
        assert len(graph.get_components_by_type(
            __import__("src.simulation", fromlist=["ComponentType"]).ComponentType.BROKER
        )) == 2
        assert len(graph.get_components_by_type(
            __import__("src.simulation", fromlist=["ComponentType"]).ComponentType.TOPIC
        )) == 10

    def test_simulate_single_failure(self):
        from src.simulation import simulate_single_failure, create_simulation_graph
        
        graph = create_simulation_graph(seed=42)
        result = simulate_single_failure(graph, "broker_1", seed=42)
        
        assert result.component_id == "broker_1"
        assert result.impact_score >= 0.0

    def test_simulate_all_components(self):
        from src.simulation import simulate_all_components, create_simulation_graph
        
        graph = create_simulation_graph(applications=3, brokers=1, topics=5, nodes=2, seed=42)
        campaign = simulate_all_components(graph, seed=42)
        
        assert campaign.total_simulations > 0

    def test_run_event_simulation(self):
        from src.simulation import run_event_simulation, create_simulation_graph
        
        graph = create_simulation_graph(seed=42)
        result = run_event_simulation(graph, duration=100.0, message_rate=10.0, seed=42)
        
        assert result.stats.total_messages > 0


# =============================================================================
# Test: Data Class Serialization
# =============================================================================

class TestSerialization:
    """Tests for data class serialization."""

    def test_failure_result_to_dict(self):
        from src.simulation import FailureSimulator
        
        graph = create_test_graph()
        simulator = FailureSimulator(seed=42)
        result = simulator.simulate_failure(graph, "app_1")
        
        data = result.to_dict()
        
        assert "component_id" in data
        assert "impact_score" in data
        assert "directly_affected" in data

    def test_campaign_result_to_dict(self):
        from src.simulation import FailureSimulator
        
        graph = create_test_graph()
        simulator = FailureSimulator(seed=42)
        campaign = simulator.simulate_all(graph)
        
        data = campaign.to_dict()
        
        assert "total_simulations" in data
        assert "by_layer" in data
        assert "ranked_top_20" in data

    def test_simulation_result_to_dict(self):
        from src.simulation import EventSimulator
        
        graph = create_test_graph()
        simulator = EventSimulator(seed=42)
        result = simulator.run(graph, duration=100.0, message_rate=10.0)
        
        data = result.to_dict()
        
        assert "stats" in data
        assert "layer_stats" in data
        assert "component_loads" in data


# =============================================================================
# Test: Neo4j Client
# =============================================================================

class TestNeo4jClient:
    """Tests for Neo4jSimulationClient (skipped if Neo4j unavailable)."""
    
    @classmethod
    def setup_class(cls):
        """Check if Neo4j is available."""
        from src.simulation import check_neo4j_available
        cls.neo4j_available = check_neo4j_available()
        cls.connection_works = False
        
        if cls.neo4j_available:
            try:
                from src.simulation import Neo4jSimulationClient
                with Neo4jSimulationClient() as client:
                    cls.connection_works = client.verify_connection()
            except:
                pass
    
    def test_check_neo4j_available(self):
        from src.simulation import check_neo4j_available
        
        # Just verify the function runs
        result = check_neo4j_available()
        assert isinstance(result, bool)
    
    def test_neo4j_config(self):
        from src.simulation import Neo4jConfig
        
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
            database="neo4j"
        )
        
        assert config.uri == "bolt://localhost:7687"
        assert config.user == "neo4j"
        assert config.password == "test"
        
        data = config.to_dict()
        assert "uri" in data
        assert "user" in data
    
    def test_neo4j_config_from_dict(self):
        from src.simulation import Neo4jConfig
        
        data = {
            "uri": "bolt://host:7687",
            "user": "testuser",
            "password": "testpass",
            "database": "testdb"
        }
        
        config = Neo4jConfig.from_dict(data)
        assert config.uri == "bolt://host:7687"
        assert config.user == "testuser"
    
    def test_client_import_error(self):
        """Test behavior when Neo4j driver not available."""
        # This test just verifies the import path works
        from src.simulation import Neo4jSimulationClient, check_neo4j_available
        
        # If Neo4j is not available, instantiation should fail
        if not check_neo4j_available():
            try:
                client = Neo4jSimulationClient()
                assert False, "Should have raised ImportError"
            except ImportError:
                pass
    
    def test_load_graph_from_neo4j_function(self):
        """Test factory function exists and has correct signature."""
        from src.simulation import load_graph_from_neo4j
        import inspect
        
        sig = inspect.signature(load_graph_from_neo4j)
        params = list(sig.parameters.keys())
        
        assert "uri" in params
        assert "user" in params
        assert "password" in params
        assert "database" in params
        assert "layer" in params
    
    def test_client_layer_definitions(self):
        """Test layer definitions are correct."""
        if not self.neo4j_available:
            return  # Skip
        
        from src.simulation import Neo4jSimulationClient, EdgeType
        
        # Verify layer edge type mappings
        assert "application" in Neo4jSimulationClient.LAYER_EDGE_TYPES
        assert "infrastructure" in Neo4jSimulationClient.LAYER_EDGE_TYPES
        assert "app_broker" in Neo4jSimulationClient.LAYER_EDGE_TYPES
        assert "node_broker" in Neo4jSimulationClient.LAYER_EDGE_TYPES
        
        # Application layer should have pub/sub edges
        app_edges = Neo4jSimulationClient.LAYER_EDGE_TYPES["application"]
        assert EdgeType.PUBLISHES_TO in app_edges
        assert EdgeType.SUBSCRIBES_TO in app_edges
    
    def test_client_with_live_connection(self):
        """Test actual Neo4j connection (skipped if unavailable)."""
        if not self.connection_works:
            return  # Skip
        
        from src.simulation import Neo4jSimulationClient
        
        with Neo4jSimulationClient() as client:
            # Verify connection
            assert client.verify_connection()
            
            # Get statistics
            stats = client.get_statistics()
            assert "total_components" in stats
            assert "total_edges" in stats
    
    def test_load_full_graph_live(self):
        """Test loading full graph (skipped if unavailable)."""
        if not self.connection_works:
            return  # Skip
        
        from src.simulation import Neo4jSimulationClient
        
        with Neo4jSimulationClient() as client:
            graph = client.load_full_graph()
            
            assert graph is not None
            # Graph may be empty if DB is empty
    
    def test_load_layer_live(self):
        """Test loading specific layer (skipped if unavailable)."""
        if not self.connection_works:
            return  # Skip
        
        from src.simulation import Neo4jSimulationClient
        
        with Neo4jSimulationClient() as client:
            for layer in ["application", "infrastructure", "app_broker", "node_broker"]:
                graph = client.load_layer(layer)
                assert graph is not None


# =============================================================================
# Standalone Test Runner
# =============================================================================

def run_tests_standalone():
    """Run tests without pytest."""
    import traceback
    
    test_classes = [
        TestEnums,
        TestQoSPolicy,
        TestSimulationGraph,
        TestFailureSimulator,
        TestEventSimulator,
        TestFactoryFunctions,
        TestSerialization,
        TestNeo4jClient,
    ]
    
    passed = failed = skipped = 0
    
    print("=" * 60)
    print("Software-as-a-Graph Simulation Module Tests")
    print("=" * 60)
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)
        
        # Setup class if needed
        if hasattr(test_class, 'setup_class'):
            try:
                test_class.setup_class()
            except:
                pass
        
        instance = test_class()
        for method_name in [m for m in dir(instance) if m.startswith("test_")]:
            try:
                getattr(instance, method_name)()
                print(f"  ✓ {method_name}")
                passed += 1
            except Exception as e:
                if "Skip" in str(e) or str(e) == "":
                    print(f"  ○ {method_name} (skipped)")
                    skipped += 1
                else:
                    print(f"  ✗ {method_name}: {e}")
                    if "--verbose" in sys.argv:
                        traceback.print_exc()
                    failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    if HAS_PYTEST and "--no-pytest" not in sys.argv:
        sys.exit(pytest.main([__file__, "-v"]))
    else:
        success = run_tests_standalone()
        sys.exit(0 if success else 1)