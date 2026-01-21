"""
Unit Tests for src/core module

Tests for:
    - graph_model.py: QoSPolicy, Topic, Application, Node, Broker, Library weight calculations
    - graph_generator.py: GraphGenerator scale configurations
"""

import pytest
import math
from src.core.graph_model import (
    QoSPolicy,
    Topic,
    Application,
    Broker,
    Node,
    Library,
)


# =============================================================================
# QoSPolicy Tests
# =============================================================================

class TestQoSPolicy:
    """Tests for QoSPolicy weight calculation."""
    
    def test_default_qos_weight(self):
        """Default QoS (VOLATILE, BEST_EFFORT, MEDIUM) should have low weight."""
        policy = QoSPolicy()
        weight = policy.calculate_weight()
        # VOLATILE=0, BEST_EFFORT=0, MEDIUM=0.1
        assert weight == pytest.approx(0.1, abs=0.01)
    
    def test_reliable_qos_adds_weight(self):
        """RELIABLE reliability adds 0.30 to weight."""
        policy = QoSPolicy(reliability="RELIABLE")
        weight = policy.calculate_weight()
        # VOLATILE=0, RELIABLE=0.3, MEDIUM=0.1
        assert weight == pytest.approx(0.4, abs=0.01)
    
    def test_persistent_qos_adds_weight(self):
        """PERSISTENT durability adds 0.40 to weight."""
        policy = QoSPolicy(durability="PERSISTENT")
        weight = policy.calculate_weight()
        # PERSISTENT=0.4, BEST_EFFORT=0, MEDIUM=0.1
        assert weight == pytest.approx(0.5, abs=0.01)
    
    def test_urgent_priority_adds_weight(self):
        """URGENT priority adds 0.30 to weight."""
        policy = QoSPolicy(transport_priority="URGENT")
        weight = policy.calculate_weight()
        # VOLATILE=0, BEST_EFFORT=0, URGENT=0.3
        assert weight == pytest.approx(0.3, abs=0.01)
    
    def test_highest_qos_weight(self):
        """Maximum QoS settings should give weight of 1.0."""
        policy = QoSPolicy(
            reliability="RELIABLE",      # +0.3
            durability="PERSISTENT",     # +0.4
            transport_priority="URGENT"  # +0.3
        )
        weight = policy.calculate_weight()
        assert weight == pytest.approx(1.0, abs=0.01)
    
    def test_qos_to_dict(self):
        """to_dict should return correct dictionary."""
        policy = QoSPolicy(durability="PERSISTENT", reliability="RELIABLE")
        d = policy.to_dict()
        assert d["durability"] == "PERSISTENT"
        assert d["reliability"] == "RELIABLE"
    
    def test_qos_from_dict(self):
        """from_dict should create correct QoSPolicy."""
        data = {"durability": "TRANSIENT", "reliability": "RELIABLE", "transport_priority": "HIGH"}
        policy = QoSPolicy.from_dict(data)
        assert policy.durability == "TRANSIENT"
        assert policy.reliability == "RELIABLE"
        assert policy.transport_priority == "HIGH"


# =============================================================================
# Topic Tests
# =============================================================================

class TestTopic:
    """Tests for Topic weight calculation including size score."""
    
    def test_small_topic_weight(self):
        """Small message size (1KB) should have low size score."""
        qos = QoSPolicy()  # Default QoS = 0.1
        topic = Topic(id="t1", name="small_topic", size=1024, qos=qos)
        weight = topic.calculate_weight()
        # QoS=0.1, Size: log2(1+1)/10 = 0.1
        assert weight == pytest.approx(0.2, abs=0.05)
    
    def test_medium_topic_weight(self):
        """Medium message size (64KB) should have moderate size score."""
        qos = QoSPolicy()
        topic = Topic(id="t2", name="medium_topic", size=65536, qos=qos)
        weight = topic.calculate_weight()
        # QoS=0.1, Size: log2(1+64)/10 ≈ 0.6
        assert 0.6 < weight < 0.8
    
    def test_large_topic_weight_capped(self):
        """Large message size should be capped at 1.0 for size score."""
        qos = QoSPolicy()
        topic = Topic(id="t3", name="large_topic", size=1048576, qos=qos)  # 1MB
        weight = topic.calculate_weight()
        # Size score should be capped at 1.0
        # QoS=0.1, Size=1.0 → Total=1.1
        assert weight == pytest.approx(1.1, abs=0.05)
    
    def test_full_topic_weight(self):
        """Highest QoS + large size should give maximum weight."""
        qos = QoSPolicy(reliability="RELIABLE", durability="PERSISTENT", transport_priority="URGENT")
        topic = Topic(id="t4", name="critical_topic", size=1048576, qos=qos)
        weight = topic.calculate_weight()
        # QoS=1.0, Size=1.0 → Total=2.0
        assert weight == pytest.approx(2.0, abs=0.05)
    
    def test_topic_to_dict(self):
        """to_dict should include all topic properties."""
        qos = QoSPolicy(durability="PERSISTENT")
        topic = Topic(id="t1", name="test", size=2048, qos=qos)
        d = topic.to_dict()
        assert d["id"] == "t1"
        assert d["name"] == "test"
        assert d["size"] == 2048
        assert d["qos"]["durability"] == "PERSISTENT"


# =============================================================================
# Entity Tests
# =============================================================================

class TestEntities:
    """Tests for Application, Broker, Node, Library entities."""
    
    def test_application_to_dict(self):
        """Application to_dict should include all fields."""
        app = Application(
            id="app1",
            name="Test App",
            role="publisher",
            app_type="service",
            criticality="high",
            version="1.0"
        )
        d = app.to_dict()
        assert d["id"] == "app1"
        assert d["role"] == "publisher"
        assert d["criticality"] == "high"
    
    def test_broker_to_dict(self):
        """Broker to_dict should include id and name."""
        broker = Broker(id="b1", name="Main Broker")
        d = broker.to_dict()
        assert d["id"] == "b1"
        assert d["name"] == "Main Broker"
    
    def test_node_to_dict(self):
        """Node to_dict should include id and name."""
        node = Node(id="n1", name="Server1")
        d = node.to_dict()
        assert d["id"] == "n1"
        assert d["name"] == "Server1"

    
    def test_library_to_dict(self):
        """Library to_dict should include version if set."""
        lib = Library(id="lib1", name="NavLib", version="2.1.0")
        d = lib.to_dict()
        assert d["id"] == "lib1"
        assert d["version"] == "2.1.0"
    
    def test_library_without_version(self):
        """Library to_dict should omit version if not set."""
        lib = Library(id="lib2", name="DataLib")
        d = lib.to_dict()
        assert "version" not in d


# =============================================================================
# Weight Formula Verification
# =============================================================================

class TestWeightFormulas:
    """Verify weight formulas match documentation."""
    
    def test_size_score_formula(self):
        """Verify S_size = min(log₂(1 + size/1024) / 10, 1.0)."""
        test_cases = [
            (1024, 0.1),      # 1KB
            (8192, 0.32),     # 8KB  
            (65536, 0.60),    # 64KB
            (1048576, 1.0),   # 1MB (capped)
        ]
        
        for size_bytes, expected_score in test_cases:
            calculated = min(math.log2(1 + size_bytes / 1024) / 10, 1.0)
            assert calculated == pytest.approx(expected_score, abs=0.05), \
                f"Size {size_bytes} expected {expected_score}, got {calculated}"
    
    def test_qos_formula_components(self):
        """Verify individual QoS scores match documentation."""
        # From docs: RELIABLE=+0.30, PERSISTENT=+0.40, URGENT=+0.30
        assert QoSPolicy(reliability="RELIABLE").calculate_weight() == pytest.approx(0.4, abs=0.01)
        assert QoSPolicy(durability="PERSISTENT").calculate_weight() == pytest.approx(0.5, abs=0.01)
        assert QoSPolicy(transport_priority="URGENT").calculate_weight() == pytest.approx(0.3, abs=0.01)
