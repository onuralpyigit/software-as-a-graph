#!/usr/bin/env python3
"""
Test Suite for src.core Module - Version 5.0

Comprehensive tests for the core module including:
- Graph generation
- Graph model and data structures
- QoS policy and weight calculation
- Component weight calculation
- DEPENDS_ON relationship derivation
- Serialization/deserialization

Run with pytest:
    pytest tests/test_core.py -v

Or standalone:
    python tests/test_core.py

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import pytest, fall back to unittest
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


# =============================================================================
# Test Fixtures / Setup
# =============================================================================

def create_minimal_graph_data() -> Dict[str, Any]:
    """Create minimal graph data for testing."""
    return {
        "applications": [
            {"id": "app_pub", "name": "Publisher", "role": "pub"},
            {"id": "app_sub", "name": "Subscriber", "role": "sub"},
        ],
        "brokers": [
            {"id": "broker_1", "name": "Main Broker"},
        ],
        "topics": [
            {
                "id": "topic_1",
                "name": "/sensor/data",
                "size": 1024,
                "qos": {
                    "durability": "PERSISTENT",
                    "reliability": "RELIABLE",
                    "transport_priority": "HIGH",
                },
            },
        ],
        "nodes": [
            {"id": "node_1", "name": "Server 1"},
            {"id": "node_2", "name": "Server 2"},
        ],
        "relationships": {
            "publishes_to": [{"from": "app_pub", "to": "topic_1"}],
            "subscribes_to": [{"from": "app_sub", "to": "topic_1"}],
            "routes": [{"from": "broker_1", "to": "topic_1"}],
            "runs_on": [
                {"from": "app_pub", "to": "node_1"},
                {"from": "app_sub", "to": "node_2"},
                {"from": "broker_1", "to": "node_1"},
            ],
            "connects_to": [{"from": "node_1", "to": "node_2"}],
        },
    }


# =============================================================================
# Test Classes
# =============================================================================

class TestEnums:
    """Tests for enumeration types."""

    def test_vertex_types(self):
        """Test VertexType enum values."""
        from src.core import VertexType

        assert VertexType.APPLICATION.value == "Application"
        assert VertexType.BROKER.value == "Broker"
        assert VertexType.TOPIC.value == "Topic"
        assert VertexType.NODE.value == "Node"

    def test_edge_types(self):
        """Test EdgeType enum values."""
        from src.core import EdgeType

        assert EdgeType.PUBLISHES_TO.value == "PUBLISHES_TO"
        assert EdgeType.SUBSCRIBES_TO.value == "SUBSCRIBES_TO"
        assert EdgeType.ROUTES.value == "ROUTES"
        assert EdgeType.RUNS_ON.value == "RUNS_ON"
        assert EdgeType.CONNECTS_TO.value == "CONNECTS_TO"
        assert EdgeType.DEPENDS_ON.value == "DEPENDS_ON"

    def test_dependency_types(self):
        """Test DependencyType enum values."""
        from src.core import DependencyType

        assert DependencyType.APP_TO_APP.value == "app_to_app"
        assert DependencyType.NODE_TO_NODE.value == "node_to_node"
        assert DependencyType.APP_TO_BROKER.value == "app_to_broker"
        assert DependencyType.NODE_TO_BROKER.value == "node_to_broker"


class TestQoSPolicy:
    """Tests for QoS policy and weight calculation."""

    def test_default_qos(self):
        """Test default QoS values."""
        from src.core import QoSPolicy

        qos = QoSPolicy()
        assert qos.durability == "VOLATILE"
        assert qos.reliability == "BEST_EFFORT"
        assert qos.transport_priority == "MEDIUM"

    def test_qos_criticality_score_minimum(self):
        """Test minimum criticality score."""
        from src.core import QoSPolicy

        qos = QoSPolicy(
            durability="VOLATILE",
            reliability="BEST_EFFORT",
            transport_priority="LOW",
        )
        assert qos.criticality_score() == 0.0

    def test_qos_criticality_score_maximum(self):
        """Test maximum criticality score."""
        from src.core import QoSPolicy

        qos = QoSPolicy(
            durability="PERSISTENT",
            reliability="RELIABLE",
            transport_priority="URGENT",
        )
        # PERSISTENT (0.40) + RELIABLE (0.30) + URGENT (0.30) = 1.0
        assert qos.criticality_score() == 1.0

    def test_qos_criticality_score_partial(self):
        """Test partial criticality scores."""
        from src.core import QoSPolicy

        # TRANSIENT (0.25) + RELIABLE (0.30) + MEDIUM (0.10) = 0.65
        qos = QoSPolicy(
            durability="TRANSIENT",
            reliability="RELIABLE",
            transport_priority="MEDIUM",
        )
        assert abs(qos.criticality_score() - 0.65) < 0.001

    def test_qos_serialization_roundtrip(self):
        """Test QoS serialization and deserialization."""
        from src.core import QoSPolicy

        original = QoSPolicy(
            durability="PERSISTENT",
            reliability="RELIABLE",
            transport_priority="HIGH",
        )
        data = original.to_dict()
        loaded = QoSPolicy.from_dict(data)
        
        assert loaded.durability == original.durability
        assert loaded.reliability == original.reliability
        assert loaded.criticality_score() == original.criticality_score()

    def test_qos_invalid_values(self):
        """Test QoS with invalid values returns 0 weight for those values."""
        from src.core import QoSPolicy

        qos = QoSPolicy(
            durability="INVALID",
            reliability="INVALID",
            transport_priority="INVALID",
        )
        # Invalid values should contribute 0 to score
        assert qos.criticality_score() == 0.0


class TestVertexClasses:
    """Tests for vertex data classes."""

    def test_application_creation(self):
        """Test Application creation and properties."""
        from src.core import Application, VertexType

        app = Application(id="app_1", name="Test App", role="pubsub")
        
        assert app.id == "app_1"
        assert app.name == "Test App"
        assert app.role == "pubsub"
        assert app.weight == 0.0
        assert app.vertex_type == VertexType.APPLICATION

    def test_application_with_weight(self):
        """Test Application with weight property."""
        from src.core import Application

        app = Application(id="app_1", name="Critical App", weight=5.5)
        assert app.weight == 5.5

    def test_topic_with_qos(self):
        """Test Topic creation with QoS."""
        from src.core import Topic, QoSPolicy

        qos = QoSPolicy(durability="PERSISTENT", reliability="RELIABLE")
        topic = Topic(id="topic_1", name="/data/stream", size=512, qos=qos)
        
        assert topic.size == 512
        assert topic.qos.durability == "PERSISTENT"

    def test_vertex_serialization(self):
        """Test vertex to_dict and from_dict."""
        from src.core import Application, Broker, Topic, Node

        # Test each vertex type
        for cls, kwargs in [
            (Application, {"id": "a1", "name": "App", "role": "pub"}),
            (Broker, {"id": "b1", "name": "Broker"}),
            (Topic, {"id": "t1", "name": "Topic", "size": 1024}),
            (Node, {"id": "n1", "name": "Node"}),
        ]:
            original = cls(**kwargs)
            data = original.to_dict()
            loaded = cls.from_dict(data)
            assert original.id == loaded.id
            assert original.name == loaded.name


class TestEdgeClasses:
    """Tests for edge data classes."""

    def test_edge_creation(self):
        """Test basic Edge creation."""
        from src.core import Edge

        edge = Edge(source="app_1", target="topic_1", edge_type="PUBLISHES_TO")
        
        assert edge.source == "app_1"
        assert edge.target == "topic_1"
        assert edge.edge_type == "PUBLISHES_TO"

    def test_depends_on_edge(self):
        """Test DependsOnEdge creation."""
        from src.core import DependsOnEdge, DependencyType

        edge = DependsOnEdge(
            source="app_sub",
            target="app_pub",
            dependency_type=DependencyType.APP_TO_APP,
            weight=3.5,
            via_topics=["topic_1", "topic_2"],
        )
        
        assert edge.weight == 3.5
        assert len(edge.via_topics) == 2

    def test_depends_on_serialization(self):
        """Test DependsOnEdge serialization."""
        from src.core import DependsOnEdge, DependencyType

        edge = DependsOnEdge(
            source="node_1",
            target="broker_1",
            dependency_type=DependencyType.NODE_TO_BROKER,
            weight=2.0,
        )
        
        data = edge.to_dict()
        assert data["dependency_type"] == "node_to_broker"
        assert data["weight"] == 2.0


class TestGraphModel:
    """Tests for GraphModel class."""

    def test_empty_graph_model(self):
        """Test creating an empty graph model."""
        from src.core import GraphModel

        model = GraphModel()
        assert len(model.applications) == 0
        assert len(model.brokers) == 0

    def test_add_and_get_vertices(self):
        """Test adding and retrieving vertices."""
        from src.core import GraphModel, Application

        model = GraphModel()
        model.add_application(Application(id="app_1", name="Test App"))
        
        app = model.get_application("app_1")
        assert app is not None
        assert app.name == "Test App"
        
        assert model.get_application("nonexistent") is None

    def test_from_dict(self):
        """Test creating graph model from dictionary."""
        from src.core import GraphModel

        data = create_minimal_graph_data()
        model = GraphModel.from_dict(data)
        
        assert len(model.applications) == 2
        assert len(model.brokers) == 1
        assert len(model.topics) == 1
        assert len(model.nodes) == 2

    def test_to_dict_roundtrip(self):
        """Test full serialization roundtrip."""
        from src.core import GraphModel

        data = create_minimal_graph_data()
        model = GraphModel.from_dict(data)
        exported = model.to_dict()
        
        assert len(exported["applications"]) == 2
        assert len(exported["brokers"]) == 1

    def test_query_methods(self):
        """Test query methods for publishers and subscribers."""
        from src.core import GraphModel

        data = create_minimal_graph_data()
        model = GraphModel.from_dict(data)
        
        publishers = model.get_publishers("topic_1")
        assert len(publishers) == 1
        assert publishers[0].id == "app_pub"
        
        subscribers = model.get_subscribers("topic_1")
        assert len(subscribers) == 1
        assert subscribers[0].id == "app_sub"

    def test_summary_and_statistics(self):
        """Test summary and statistics methods."""
        from src.core import GraphModel

        data = create_minimal_graph_data()
        model = GraphModel.from_dict(data)
        
        summary = model.summary()
        assert "Applications: 2" in summary
        
        stats = model.statistics()
        assert stats["vertices"]["total"] == 6


class TestGraphGeneration:
    """Tests for graph generation."""

    def test_generate_scales(self):
        """Test generating graphs at different scales."""
        from src.core import generate_graph

        scales = {"tiny": 5, "small": 10, "medium": 30}
        
        for scale, expected_apps in scales.items():
            graph = generate_graph(scale=scale, seed=42)
            assert len(graph["applications"]) == expected_apps

    def test_generate_scenarios(self):
        """Test all scenario types."""
        from src.core import generate_graph

        scenarios = ["generic", "iot", "financial", "healthcare", "autonomous_vehicle"]
        
        for scenario in scenarios:
            graph = generate_graph(scale="tiny", scenario=scenario, seed=42)
            assert len(graph["applications"]) > 0

    def test_deterministic_generation(self):
        """Test that same seed produces same graph."""
        from src.core import generate_graph

        g1 = generate_graph(scale="small", seed=123)
        g2 = generate_graph(scale="small", seed=123)
        
        assert len(g1["applications"]) == len(g2["applications"])
        assert {a["id"] for a in g1["applications"]} == {a["id"] for a in g2["applications"]}

    def test_qos_in_topics(self):
        """Test that generated topics have QoS."""
        from src.core import generate_graph

        graph = generate_graph(scale="small", scenario="healthcare", seed=42)
        
        for topic in graph["topics"]:
            assert "qos" in topic
            assert "durability" in topic["qos"]


class TestJsonSerialization:
    """Tests for JSON serialization."""

    def test_full_roundtrip(self):
        """Test complete JSON serialization roundtrip."""
        from src.core import generate_graph, GraphModel

        original = generate_graph(scale="small", seed=42)
        json_str = json.dumps(original)
        parsed = json.loads(json_str)
        
        model = GraphModel.from_dict(parsed)
        exported = model.to_dict()
        
        assert len(exported["applications"]) == len(original["applications"])

    def test_file_roundtrip(self):
        """Test saving and loading from file."""
        from src.core import generate_graph

        graph = generate_graph(scale="tiny", seed=42)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(graph, f)
            filepath = f.name
        
        try:
            with open(filepath) as f:
                loaded = json.load(f)
            assert len(loaded["applications"]) == len(graph["applications"])
        finally:
            Path(filepath).unlink()


# =============================================================================
# Standalone Test Runner
# =============================================================================

def run_tests_standalone():
    """Run tests without pytest."""
    import traceback
    
    test_classes = [
        TestEnums,
        TestQoSPolicy,
        TestVertexClasses,
        TestEdgeClasses,
        TestGraphModel,
        TestGraphGeneration,
        TestJsonSerialization,
    ]
    
    passed = failed = 0
    
    print("=" * 60)
    print("Software-as-a-Graph Core Module Tests")
    print("=" * 60)
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)
        
        instance = test_class()
        for method_name in [m for m in dir(instance) if m.startswith("test_")]:
            try:
                getattr(instance, method_name)()
                print(f"  ✓ {method_name}")
                passed += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
                if "--verbose" in sys.argv:
                    traceback.print_exc()
                failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    if HAS_PYTEST and "--no-pytest" not in sys.argv:
        sys.exit(pytest.main([__file__, "-v"]))
    else:
        success = run_tests_standalone()
        sys.exit(0 if success else 1)
