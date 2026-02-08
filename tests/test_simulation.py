"""
Tests for Failure Simulation — Step 4

Covers:
    - Physical cascade (Node → hosted components)
    - Logical cascade: Broker → Topics (C1 fix)
    - Logical cascade: Publisher starvation (C1 fix)
    - Network cascade: Node isolation (C1 fix)
    - Broker-aware reachability (C6 fix)
    - True fragmentation via connected components (C2 fix)
    - QoS-weighted throughput loss (C3 fix)
    - Baseline caching in exhaustive mode (C5 fix)
    - Monte Carlo stochastic simulation
    - Cascade tree export
"""

import pytest
from unittest.mock import MagicMock

from src.domain.services.failure_simulator import (
    FailureSimulator,
    FailureScenario,
    ImpactMetrics,
    MonteCarloResult,
    CascadeEvent,
)
from src.domain.models.simulation.graph import SimulationGraph
from src.domain.models.simulation.types import (
    ComponentState,
    FailureMode,
    CascadeRule,
)

# Assuming GraphData/ComponentData/EdgeData are importable test fixtures
# from src.domain.models.simulation.components import GraphData, ComponentData, EdgeData


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def raw_graph_data():
    """
    Test topology:
    
        Node1 hosts: App1, App2, Broker1
        Node2 hosts: App3
        Node1 <-> Node2 (CONNECTS_TO)
        
        App1 -[PUBLISHES_TO]-> Topic1, Topic2
        App2 -[SUBSCRIBES_TO]-> Topic1
        App3 -[SUBSCRIBES_TO]-> Topic2
        
        Broker1 -[ROUTES]-> Topic1, Topic2
        
        App1 -[USES]-> Lib1
    """
    return GraphData(
        components=[
            ComponentData("App1", "Application"),
            ComponentData("App2", "Application"),
            ComponentData("App3", "Application"),
            ComponentData("Topic1", "Topic"),
            ComponentData("Topic2", "Topic"),
            ComponentData("Broker1", "Broker"),
            ComponentData("Node1", "Node"),
            ComponentData("Node2", "Node"),
            ComponentData("Lib1", "Library"),
        ],
        edges=[
            EdgeData("App1", "Topic1", "Application", "Topic", "PUBLISHES_TO", "PUBLISHES_TO"),
            EdgeData("App2", "Topic1", "Application", "Topic", "SUBSCRIBES_TO", "SUBSCRIBES_TO"),
            EdgeData("App1", "Topic2", "Application", "Topic", "PUBLISHES_TO", "PUBLISHES_TO"),
            EdgeData("App3", "Topic2", "Application", "Topic", "SUBSCRIBES_TO", "SUBSCRIBES_TO"),
            EdgeData("Broker1", "Topic1", "Broker", "Topic", "ROUTES", "ROUTES"),
            EdgeData("Broker1", "Topic2", "Broker", "Topic", "ROUTES", "ROUTES"),
            EdgeData("App1", "Node1", "Application", "Node", "RUNS_ON", "RUNS_ON"),
            EdgeData("App2", "Node1", "Application", "Node", "RUNS_ON", "RUNS_ON"),
            EdgeData("App3", "Node2", "Application", "Node", "RUNS_ON", "RUNS_ON"),
            EdgeData("Broker1", "Node1", "Broker", "Node", "RUNS_ON", "RUNS_ON"),
            EdgeData("Node1", "Node2", "Node", "Node", "CONNECTS_TO", "CONNECTS_TO"),
            EdgeData("App1", "Lib1", "Application", "Library", "USES", "USES"),
        ],
    )


@pytest.fixture
def sim_graph(raw_graph_data):
    """SimulationGraph instance from the fixture data."""
    return SimulationGraph(graph_data=raw_graph_data)


# =============================================================================
# Physical Cascade Tests
# =============================================================================

class TestPhysicalCascade:
    """Tests for Node → hosted components cascade."""

    def test_node_failure_cascades_to_hosted_apps(self, sim_graph):
        """Failing Node1 should cascade to App1, App2, Broker1 (all hosted)."""
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("Node1", "test"))

        assert "App1" in result.cascaded_failures
        assert "App2" in result.cascaded_failures
        assert "Broker1" in result.cascaded_failures
        assert result.impact.cascade_count >= 3

    def test_cascade_probability_zero_no_cascade(self, sim_graph):
        """With cascade_probability=0, no cascade should propagate."""
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario(
            "Node1", "test", cascade_probability=0.0,
        ))
        assert len(result.cascaded_failures) == 0

    def test_broker_failure_does_not_cascade_to_nodes(self, sim_graph):
        """Failing a broker should not cascade to infrastructure nodes."""
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("Broker1", "test"))

        assert "Node1" not in result.cascaded_failures
        assert "Node2" not in result.cascaded_failures


# =============================================================================
# Logical Cascade Tests (C1 Fix)
# =============================================================================

class TestLogicalCascade:
    """Tests for Broker → Topic and Publisher → Subscriber starvation cascades."""

    def test_broker_failure_cascades_to_topics(self, sim_graph):
        """
        C1 fix: When Broker1 fails, Topic1 and Topic2 (routed exclusively
        through Broker1) should appear in the cascade sequence.
        """
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("Broker1", "test"))

        cascaded_ids = {e.component_id for e in result.cascade_sequence}
        assert "Topic1" in cascaded_ids or "Topic1" in result.cascaded_failures
        assert "Topic2" in cascaded_ids or "Topic2" in result.cascaded_failures

    def test_broker_cascade_cause_tracking(self, sim_graph):
        """Cascade events from broker failure should have 'no_active_brokers' cause."""
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("Broker1", "test"))

        topic_events = [
            e for e in result.cascade_sequence
            if e.component_type == "Topic"
        ]
        assert len(topic_events) > 0
        for event in topic_events:
            assert "no_active_brokers" in event.cause

    def test_publisher_starvation_cascade(self, sim_graph):
        """
        C1 fix: When App1 (sole publisher for Topic1 and Topic2) fails,
        both topics should be marked as starved in the cascade.
        """
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("App1", "test"))

        cascaded_ids = set(result.cascaded_failures)
        # Topic1 and Topic2 lose their only publisher
        assert "Topic1" in cascaded_ids
        assert "Topic2" in cascaded_ids

    def test_publisher_starvation_cause_tracking(self, sim_graph):
        """Starvation events should have 'publisher_starvation' cause."""
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("App1", "test"))

        starvation_events = [
            e for e in result.cascade_sequence
            if "publisher_starvation" in e.cause
        ]
        assert len(starvation_events) > 0

    def test_node_failure_triggers_logical_cascade_chain(self, sim_graph):
        """
        Node1 hosts Broker1 and App1. Failing Node1 should:
        1. Physical: Cascade to App1, App2, Broker1
        2. Logical: Broker1 failure → Topic1, Topic2 unreachable
        3. Logical: App1 failure → publisher starvation (already covered by broker)
        """
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("Node1", "test"))

        cascaded_ids = set(result.cascaded_failures)
        # Physical cascade
        assert "App1" in cascaded_ids
        assert "App2" in cascaded_ids
        assert "Broker1" in cascaded_ids
        # Logical cascade from Broker1
        assert "Topic1" in cascaded_ids
        assert "Topic2" in cascaded_ids


# =============================================================================
# Network Cascade Tests (C1 Fix)
# =============================================================================

class TestNetworkCascade:
    """Tests for Node → isolated neighbor cascade."""

    def test_node_isolation_triggers_partition(self, sim_graph):
        """
        C1 fix: Node2 connects only to Node1. When Node1 fails,
        Node2 becomes isolated and should be marked as partitioned.
        """
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("Node1", "test"))

        cascaded_ids = set(result.cascaded_failures)
        assert "Node2" in cascaded_ids

    def test_partition_cause_tracking(self, sim_graph):
        """Network partition events should have 'network_partition' cause."""
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("Node1", "test"))

        partition_events = [
            e for e in result.cascade_sequence
            if "network_partition" in e.cause
        ]
        assert len(partition_events) > 0


# =============================================================================
# Broker-Aware Reachability Tests (C6 Fix)
# =============================================================================

class TestBrokerAwareReachability:
    """Tests for broker-aware path counting in reachability loss."""

    def test_broker_failure_increases_reachability_loss(self, sim_graph):
        """
        C6 fix: When Broker1 fails, reachability loss should be high
        because no topics have active routing brokers, even though
        publishers and subscribers are still alive.
        """
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("Broker1", "test"))

        # All paths go through Broker1 — reachability should be 1.0
        assert result.impact.reachability_loss == pytest.approx(1.0, abs=0.01)

    def test_pub_sub_paths_exclude_unroutable_topics(self, sim_graph):
        """
        C6 fix: get_pub_sub_paths should return no paths for topics
        whose brokers are all failed, even if pub/sub apps are active.
        """
        sim_graph.fail_component("Broker1")
        paths = sim_graph.get_pub_sub_paths(active_only=True)
        assert len(paths) == 0


# =============================================================================
# Fragmentation Tests (C2 Fix)
# =============================================================================

class TestFragmentation:
    """Tests for true graph fragmentation via connected components."""

    def test_fragmentation_measures_connectivity_not_loss(self, sim_graph):
        """
        C2 fix: Fragmentation should measure increase in connected
        components, not just component loss ratio.
        """
        sim = FailureSimulator(sim_graph)

        # App1 failure should NOT significantly fragment the graph
        # (App2, Broker1, Node1, Node2, App3 remain connected)
        result_app = sim.simulate(FailureScenario("App1", "test"))

        # Node1 failure fragments: Node2+App3 become a separate island
        result_node = sim.simulate(FailureScenario("Node1", "test"))

        # Node failure should cause higher fragmentation
        assert result_node.impact.fragmentation >= result_app.impact.fragmentation

    def test_connected_components_tracked(self, sim_graph):
        """Impact metrics should include initial and final connected component counts."""
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("Node1", "test"))

        assert result.impact.initial_connected_components >= 1
        assert result.impact.final_connected_components >= result.impact.initial_connected_components


# =============================================================================
# QoS-Weighted Throughput Tests (C3 Fix)
# =============================================================================

class TestQoSWeightedThroughput:
    """Tests for QoS-weighted throughput loss calculation."""

    def test_throughput_loss_uses_topic_weights(self, sim_graph):
        """
        C3 fix: Topics with higher QoS weight should contribute more
        to throughput loss than low-weight topics.
        """
        # Manually set different weights on topics
        if "Topic1" in sim_graph.topics:
            sim_graph.topics["Topic1"].weight = 10.0  # High priority
        if "Topic2" in sim_graph.topics:
            sim_graph.topics["Topic2"].weight = 1.0   # Low priority

        sim = FailureSimulator(sim_graph)

        # When Broker1 fails, both topics are affected
        result = sim.simulate(FailureScenario("Broker1", "test"))

        # Total weight = 11.0, lost = 11.0 → throughput_loss = 1.0
        assert result.impact.throughput_loss == pytest.approx(1.0, abs=0.01)

    def test_throughput_reports_weight_values(self, sim_graph):
        """Impact metrics should include initial and remaining throughput."""
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("Broker1", "test"))

        assert result.impact.initial_throughput > 0
        assert result.impact.remaining_throughput >= 0


# =============================================================================
# Configurable Impact Weights Tests
# =============================================================================

class TestImpactWeights:
    """Tests for configurable composite impact weights."""

    def test_default_weights(self):
        """Default weights: reachability=0.4, fragmentation=0.3, throughput=0.3."""
        metrics = ImpactMetrics(
            reachability_loss=1.0,
            fragmentation=0.0,
            throughput_loss=0.0,
        )
        assert metrics.composite_impact == pytest.approx(0.4, abs=0.01)

    def test_custom_weights(self):
        """Custom weights should change composite impact."""
        metrics = ImpactMetrics(
            reachability_loss=1.0,
            fragmentation=0.0,
            throughput_loss=0.0,
            impact_weights={"reachability": 1.0, "fragmentation": 0.0, "throughput": 0.0},
        )
        assert metrics.composite_impact == pytest.approx(1.0, abs=0.01)

    def test_all_dimensions_contribute(self):
        """Each dimension should contribute to composite when all equal."""
        metrics = ImpactMetrics(
            reachability_loss=0.5,
            fragmentation=0.5,
            throughput_loss=0.5,
        )
        expected = 0.4 * 0.5 + 0.3 * 0.5 + 0.3 * 0.5
        assert metrics.composite_impact == pytest.approx(expected, abs=0.01)


# =============================================================================
# Exhaustive Simulation Tests
# =============================================================================

class TestExhaustiveSimulation:
    """Tests for exhaustive failure analysis."""

    def test_exhaustive_covers_all_layer_components(self, sim_graph):
        """Exhaustive simulation should cover all analyzable components in layer."""
        sim = FailureSimulator(sim_graph)
        results = sim.simulate_exhaustive(layer="app")

        analyzed_ids = {r.target_id for r in results}
        expected_apps = {"App1", "App2", "App3"}
        assert expected_apps == analyzed_ids

    def test_exhaustive_sorted_by_impact(self, sim_graph):
        """Exhaustive results should be sorted by impact (descending)."""
        sim = FailureSimulator(sim_graph)
        results = sim.simulate_exhaustive(layer="system")

        impacts = [r.impact.composite_impact for r in results]
        assert impacts == sorted(impacts, reverse=True)

    def test_exhaustive_baseline_caching(self, sim_graph):
        """
        C5 fix: Exhaustive mode should compute baseline once,
        not per simulation. Verify by checking the flag.
        """
        sim = FailureSimulator(sim_graph)
        # After exhaustive, flag should be cleared
        sim.simulate_exhaustive(layer="system")
        assert sim._baseline_computed is False

    def test_layer_impacts_present(self, sim_graph):
        """Failure result should include per-layer impact breakdown."""
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("Node1", "test"))

        assert "app" in result.layer_impacts
        assert "infra" in result.layer_impacts
        assert result.layer_impacts["infra"] > 0


# =============================================================================
# Monte Carlo Tests
# =============================================================================

class TestMonteCarlo:
    """Tests for Monte Carlo stochastic simulation."""

    def test_monte_carlo_returns_distribution(self, sim_graph):
        """Monte Carlo should return mean, std, and confidence interval."""
        sim = FailureSimulator(sim_graph)
        mc = sim.simulate_monte_carlo(
            FailureScenario("Node1", "test", cascade_probability=0.7),
            n_trials=50,
        )

        assert isinstance(mc, MonteCarloResult)
        assert mc.n_trials == 50
        assert 0.0 <= mc.mean_impact <= 1.0
        assert mc.std_impact >= 0.0
        assert mc.ci_95[0] <= mc.ci_95[1]

    def test_monte_carlo_deterministic_at_prob_1(self, sim_graph):
        """With cascade_probability=1.0, all trials should be identical."""
        sim = FailureSimulator(sim_graph)
        mc = sim.simulate_monte_carlo(
            FailureScenario("Node1", "test", cascade_probability=1.0),
            n_trials=10,
        )

        # All trial impacts should be identical (deterministic cascade)
        assert mc.std_impact == pytest.approx(0.0, abs=0.01)


# =============================================================================
# Cascade Visualization Tests
# =============================================================================

class TestCascadeVisualization:
    """Tests for cascade tree export."""

    def test_cascade_to_graph_structure(self, sim_graph):
        """cascade_to_graph should return nodes and edges."""
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("Node1", "test"))

        tree = result.cascade_to_graph()
        assert "nodes" in tree
        assert "edges" in tree
        assert len(tree["nodes"]) > 0

    def test_cascade_to_graph_root_is_target(self, sim_graph):
        """First node in cascade tree should be the failure target."""
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("Node1", "test"))

        tree = result.cascade_to_graph()
        assert tree["nodes"][0]["id"] == "Node1"
        assert tree["nodes"][0]["depth"] == 0


# =============================================================================
# Normalization & Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge cases and normalization tests."""

    def test_nonexistent_target_returns_empty(self, sim_graph):
        """Targeting a nonexistent component returns empty result."""
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("NoSuchComp", "test"))

        assert result.impact.composite_impact == 0.0

    def test_impact_metrics_normalized(self, sim_graph):
        """Impact metrics should be in [0, 1] range."""
        sim = FailureSimulator(sim_graph)
        results = sim.simulate_exhaustive(layer="system")

        for result in results:
            assert 0.0 <= result.impact.reachability_loss <= 1.0
            assert 0.0 <= result.impact.fragmentation <= 1.0
            assert 0.0 <= result.impact.throughput_loss <= 1.0
            assert 0.0 <= result.impact.composite_impact <= 1.0

    def test_to_dict_roundtrip(self, sim_graph):
        """FailureResult.to_dict should produce valid serializable output."""
        sim = FailureSimulator(sim_graph)
        result = sim.simulate(FailureScenario("Broker1", "test"))

        d = result.to_dict()
        assert "target_id" in d
        assert "impact" in d
        assert "cascade_sequence" in d
        assert d["target_id"] == "Broker1"
        assert isinstance(d["impact"]["composite_impact"], float)