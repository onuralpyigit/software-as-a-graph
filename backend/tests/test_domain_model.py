"""
Tests for Step 1: Graph Model Construction

Validates the formal definitions from docs/graph-model.md:
    - Definition 1: Graph model G = (V, E, τ_V, τ_E, L, w, QoS)
    - Definition 2: Dependency derivation rules (Rules 1-4)
    - Definition 3: Layer projection π_l
    - Definition 4: Graph constraints (C1-C5)
    - §1.5: Weight calculation with minimum floor
"""

import math
import pytest

from src.core import (
    QoSPolicy, MIN_TOPIC_WEIGHT,
    Application, Topic, Broker, Node, Library,
    AnalysisLayer, LAYER_DEFINITIONS, SIMULATION_LAYERS,
    DEPENDENCY_TO_LAYER, resolve_layer,
)


# =========================================================================
# §1.5 QoS Weight Calculation
# =========================================================================

class TestQoSPolicy:
    """Tests for QoSPolicy weight calculation matching §1.5 scoring table."""

    def test_default_qos_weight(self):
        """Default QoS (BEST_EFFORT, VOLATILE, MEDIUM) → ~0.1 (0.3*0 + 0.4*0 + 0.3*0.33)."""
        policy = QoSPolicy()
        assert policy.calculate_weight() == pytest.approx(0.1, abs=0.01)

    def test_maximum_qos_weight(self):
        """Maximum QoS (RELIABLE + PERSISTENT + URGENT) → 1.0 (0.3*1 + 0.4*1 + 0.3*1)."""
        policy = QoSPolicy(
            reliability="RELIABLE",
            durability="PERSISTENT",
            transport_priority="URGENT",
        )
        assert policy.calculate_weight() == pytest.approx(1.0, abs=0.01)

    def test_lowest_qos_weight(self):
        """Lowest QoS (BEST_EFFORT, VOLATILE, LOW) → 0.0."""
        policy = QoSPolicy(
            reliability="BEST_EFFORT",
            durability="VOLATILE",
            transport_priority="LOW",
        )
        assert policy.calculate_weight() == pytest.approx(0.0, abs=0.01)

    # --- Individual QoS attribute tests (justified weights) ---

    def test_reliable_adds_0_3(self):
        """Reliability weight = 0.30."""
        policy = QoSPolicy(reliability="RELIABLE", durability="VOLATILE", transport_priority="LOW")
        assert policy.calculate_weight() == pytest.approx(0.3, abs=0.01)

    def test_persistent_adds_0_4(self):
        """Durability weight = 0.40."""
        policy = QoSPolicy(reliability="BEST_EFFORT", durability="PERSISTENT", transport_priority="LOW")
        assert policy.calculate_weight() == pytest.approx(0.4, abs=0.01)

    def test_transient_local_adds_0_2(self):
        """Transient local (0.5) * Durability weight (0.40) = 0.20."""
        policy = QoSPolicy(reliability="BEST_EFFORT", durability="TRANSIENT_LOCAL", transport_priority="LOW")
        assert policy.calculate_weight() == pytest.approx(0.2, abs=0.01)

    def test_transient_adds_0_24(self):
        """Transient (0.6) * Durability weight (0.40) = 0.24."""
        policy = QoSPolicy(reliability="BEST_EFFORT", durability="TRANSIENT", transport_priority="LOW")
        assert policy.calculate_weight() == pytest.approx(0.24, abs=0.01)

    def test_urgent_priority_adds_0_3(self):
        """Urgent (1.0) * Priority weight (0.30) = 0.30."""
        policy = QoSPolicy(reliability="BEST_EFFORT", durability="VOLATILE", transport_priority="URGENT")
        assert policy.calculate_weight() == pytest.approx(0.3, abs=0.01)

    def test_high_priority_adds_0_2(self):
        """High (0.66) * Priority weight (0.30) ≈ 0.20."""
        policy = QoSPolicy(reliability="BEST_EFFORT", durability="VOLATILE", transport_priority="HIGH")
        assert policy.calculate_weight() == pytest.approx(0.2, abs=0.01)

    def test_medium_priority_adds_0_1(self):
        """Medium (0.33) * Priority weight (0.30) ≈ 0.10."""
        policy = QoSPolicy(reliability="BEST_EFFORT", durability="VOLATILE", transport_priority="MEDIUM")
        assert policy.calculate_weight() == pytest.approx(0.1, abs=0.01)

    def test_low_priority_adds_0(self):
        policy = QoSPolicy(reliability="BEST_EFFORT", durability="VOLATILE", transport_priority="LOW")
        assert policy.calculate_weight() == pytest.approx(0.0, abs=0.01)

    # --- Serialization ---

    def test_to_dict(self):
        policy = QoSPolicy(reliability="RELIABLE", durability="PERSISTENT", transport_priority="URGENT")
        d = policy.to_dict()
        assert d == {
            "reliability": "RELIABLE",
            "durability": "PERSISTENT",
            "transport_priority": "URGENT",
        }

    def test_from_dict(self):
        policy = QoSPolicy.from_dict({"reliability": "RELIABLE", "durability": "TRANSIENT"})
        assert policy.reliability == "RELIABLE"
        assert policy.durability == "TRANSIENT"
        assert policy.transport_priority == "MEDIUM"  # default


# =========================================================================
# §1.5 Topic Weight with Minimum Floor
# =========================================================================

class TestTopicWeight:
    """Tests for Topic.calculate_weight() with minimum weight floor."""

    def test_minimum_weight_floor(self):
        """Lowest possible topic (LOW QoS, tiny size) still gets ε = MIN_TOPIC_WEIGHT."""
        topic = Topic(
            id="t0", name="minimal",
            size=1,  # 1 byte → size_kb ≈ 0.001 → score ≈ log2(1.001)/50 ≈ 0.00002
            qos=QoSPolicy(reliability="BEST_EFFORT", durability="VOLATILE", transport_priority="LOW"),
        )
        weight = topic.calculate_weight()
        assert weight == pytest.approx(MIN_TOPIC_WEIGHT, abs=0.001)
        assert weight > 0.0, "Topic weight must never be zero"

    def test_small_topic_weight(self):
        """1 KB topic with default QoS → ~0.12 (0.1 QoS + 0.02 size)."""
        topic = Topic(id="t1", name="small", size=1024)
        weight = topic.calculate_weight()
        # QoS default ≈ 0.1, size score = log2(1+1)/50 = 0.02
        assert 0.11 < weight < 0.13

    def test_medium_topic_weight(self):
        """64 KB topic with RELIABLE QoS → ~0.52."""
        topic = Topic(
            id="t2", name="medium",
            size=65536,
            qos=QoSPolicy(reliability="RELIABLE"),
        )
        weight = topic.calculate_weight()
        # Rel(1.0)*0.3 + Pri(0.33)*0.3 ≈ 0.4. Size score = log2(1+64)/50 ≈ 0.12.
        assert 0.50 < weight < 0.54

    def test_max_topic_weight(self):
        """Maximum QoS + large size → capped at 1.2."""
        topic = Topic(
            id="t3", name="max",
            size=1_048_576,  # 1 MB
            qos=QoSPolicy(reliability="RELIABLE", durability="PERSISTENT", transport_priority="URGENT"),
        )
        weight = topic.calculate_weight()
        # QoS = 1.0. Size score = min(log2(1+1024)/50, 0.20) = 0.20.
        assert weight == pytest.approx(1.2, abs=0.01)

    def test_size_score_formula(self):
        """Verify S_size = min(log₂(1 + size/1024) / 50, 0.20) exactly."""
        test_cases = [
            (64, min(math.log2(1 + 64 / 1024) / 50, 0.20)),
            (1024, min(math.log2(1 + 1024 / 1024) / 50, 0.20)),
            (65536, min(math.log2(1 + 65536 / 1024) / 50, 0.20)),
            (10_000_000, 0.20),  # capped
        ]
        for size, expected_size_score in test_cases:
            topic = Topic(
                id="t", name="test", size=size,
                qos=QoSPolicy(reliability="BEST_EFFORT", durability="VOLATILE", transport_priority="LOW"),
            )
            weight = topic.calculate_weight()
            expected = max(MIN_TOPIC_WEIGHT, 0.0 + expected_size_score)
            assert weight == pytest.approx(expected, abs=0.001), f"Failed for size={size}"

    def test_weight_range(self):
        """All weights must be in [MIN_TOPIC_WEIGHT, 1.2]."""
        configs = [
            (1, QoSPolicy(reliability="BEST_EFFORT", durability="VOLATILE", transport_priority="LOW")),
            (256, QoSPolicy()),
            (65536, QoSPolicy(reliability="RELIABLE", durability="TRANSIENT", transport_priority="HIGH")),
            (1_048_576, QoSPolicy(reliability="RELIABLE", durability="PERSISTENT", transport_priority="URGENT")),
        ]
        for size, qos in configs:
            topic = Topic(id="t", name="test", size=size, qos=qos)
            weight = topic.calculate_weight()
            assert MIN_TOPIC_WEIGHT <= weight <= 1.2, f"Weight {weight} out of range for size={size}"


# =========================================================================
# Entity Tests
# =========================================================================

class TestEntities:
    """Tests for graph entity domain models."""

    def test_application_to_dict(self):
        app = Application(id="A1", name="Sensor", role="pub", app_type="driver", criticality=True, version="1.0")
        d = app.to_dict()
        assert d["id"] == "A1"
        assert d["role"] == "pub"
        assert d["criticality"] is True
        assert d["version"] == "1.0"

    def test_application_defaults(self):
        app = Application(id="A1", name="Service")
        assert app.role == "pubsub"
        assert app.app_type == "service"
        assert app.criticality is False

    def test_broker_to_dict(self):
        broker = Broker(id="B1", name="DDS-0")
        d = broker.to_dict()
        assert d["id"] == "B1"
        assert d["name"] == "DDS-0"

    def test_node_to_dict(self):
        node = Node(id="N1", name="Host-0")
        d = node.to_dict()
        assert d["id"] == "N1"

    def test_library_with_version(self):
        lib = Library(id="L1", name="NavLib", version="2.1")
        d = lib.to_dict()
        assert d["version"] == "2.1"

    def test_library_without_version(self):
        lib = Library(id="L1", name="NavLib")
        d = lib.to_dict()
        assert "version" not in d

    def test_topic_to_dict(self):
        topic = Topic(id="T1", name="/sensor/lidar", size=8192)
        d = topic.to_dict()
        assert d["size"] == 8192
        assert "qos" in d


# =========================================================================
# Definition 3: Layer Projection
# =========================================================================

class TestLayerDefinitions:
    """Tests for layer projection definitions (Definition 3)."""

    def test_all_layers_defined(self):
        """All four canonical layers must be defined."""
        for layer in AnalysisLayer:
            assert layer in LAYER_DEFINITIONS
            assert layer in SIMULATION_LAYERS

    def test_app_layer_projection(self):
        """π_app: only Application components, only app_to_app dependencies."""
        defn = LAYER_DEFINITIONS[AnalysisLayer.APP]
        assert defn.component_types == frozenset({"Application"})
        assert defn.dependency_types == frozenset({"app_to_app"})
        assert defn.types_to_analyze == frozenset({"Application"})
        assert defn.quality_focus == "reliability"

    def test_infra_layer_projection(self):
        """π_infra: only Node components, only node_to_node dependencies."""
        defn = LAYER_DEFINITIONS[AnalysisLayer.INFRA]
        assert defn.component_types == frozenset({"Node"})
        assert defn.dependency_types == frozenset({"node_to_node"})
        assert defn.quality_focus == "availability"

    def test_mw_layer_projection(self):
        """π_mw: includes App+Broker+Node for edges, but only Broker in results."""
        defn = LAYER_DEFINITIONS[AnalysisLayer.MW]
        assert "Application" in defn.component_types
        assert "Broker" in defn.component_types
        assert "Node" in defn.component_types
        assert defn.types_to_analyze == frozenset({"Broker"})
        assert defn.dependency_types == frozenset({"app_to_broker", "node_to_broker"})
        assert defn.quality_focus == "maintainability"

    def test_system_layer_projection(self):
        """π_system: all components, all dependency types."""
        defn = LAYER_DEFINITIONS[AnalysisLayer.SYSTEM]
        assert len(defn.component_types) == 5
        assert len(defn.dependency_types) == 4

    def test_canonical_layer_resolution(self):
        """Canonical names resolve correctly."""
        assert resolve_layer("app") == AnalysisLayer.APP
        assert resolve_layer("infra") == AnalysisLayer.INFRA
        assert resolve_layer("mw") == AnalysisLayer.MW
        assert resolve_layer("system") == AnalysisLayer.SYSTEM

    def test_legacy_alias_resolution(self):
        """Legacy aliases resolve to canonical layers."""
        assert resolve_layer("application") == AnalysisLayer.APP
        assert resolve_layer("infrastructure") == AnalysisLayer.INFRA
        assert resolve_layer("app_broker") == AnalysisLayer.MW
        assert resolve_layer("middleware") == AnalysisLayer.MW
        assert resolve_layer("complete") == AnalysisLayer.SYSTEM
        assert resolve_layer("all") == AnalysisLayer.SYSTEM

    def test_invalid_layer_raises(self):
        """Unknown layer names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown layer"):
            resolve_layer("nonexistent")

    def test_dependency_to_layer_mapping(self):
        """Every DEPENDS_ON subtype maps to the correct layer."""
        assert DEPENDENCY_TO_LAYER["app_to_app"] == AnalysisLayer.APP
        assert DEPENDENCY_TO_LAYER["node_to_node"] == AnalysisLayer.INFRA
        assert DEPENDENCY_TO_LAYER["app_to_broker"] == AnalysisLayer.MW
        assert DEPENDENCY_TO_LAYER["node_to_broker"] == AnalysisLayer.MW


# =========================================================================
# Simulation Layer Tests
# =========================================================================

class TestSimulationLayers:
    """Tests for simulation layer definitions (G_structural)."""

    def test_app_simulation_uses_raw_relationships(self):
        """App simulation uses PUBLISHES_TO/SUBSCRIBES_TO, not DEPENDS_ON."""
        defn = SIMULATION_LAYERS[AnalysisLayer.APP]
        assert "PUBLISHES_TO" in defn.relationships
        assert "SUBSCRIBES_TO" in defn.relationships
        assert defn.analyze_types == frozenset({"Application"})

    def test_infra_simulation_uses_physical_relationships(self):
        """Infra simulation uses RUNS_ON/CONNECTS_TO."""
        defn = SIMULATION_LAYERS[AnalysisLayer.INFRA]
        assert "RUNS_ON" in defn.relationships
        assert "CONNECTS_TO" in defn.relationships
        assert defn.analyze_types == frozenset({"Node"})

    def test_mw_simulation_uses_routing_relationships(self):
        """MW simulation uses ROUTES + pub/sub."""
        defn = SIMULATION_LAYERS[AnalysisLayer.MW]
        assert "ROUTES" in defn.relationships
        assert defn.analyze_types == frozenset({"Broker"})

    def test_system_simulation_uses_all_relationships(self):
        """System simulation includes all 6 structural relationship types."""
        defn = SIMULATION_LAYERS[AnalysisLayer.SYSTEM]
        assert len(defn.relationships) == 6


# =========================================================================
# MIN_TOPIC_WEIGHT Constant
# =========================================================================

class TestConstants:
    """Tests for module-level constants."""

    def test_min_topic_weight_is_positive(self):
        assert MIN_TOPIC_WEIGHT > 0.0

    def test_min_topic_weight_is_small(self):
        """ε should be small enough not to distort relative rankings."""
        assert MIN_TOPIC_WEIGHT < 0.1
