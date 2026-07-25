"""
tests/test_qos_weighted_impact.py — QoS severity inside composite_impact
========================================================================

``ImpactMetrics.composite_impact`` weighted every broken path, island and flow
equally, so losing a safety-critical RELIABLE/PERSISTENT channel scored the same
as losing a best-effort telemetry one. Three of its four terms now weight by
w(t)·rate; ``qos_weighting=False`` restores the count-based form and is the
topology-only arm of the label ablation.

Also pinned here: throughput loss is continuous and broker-aware. It used to be
binary on publishers/subscribers only, so a topic whose sole routing broker died
was scored as fully delivering.
"""

from __future__ import annotations

import pytest

from saag.core.models import GraphData, ComponentData, EdgeData
from saag.simulation.graph import SimulationGraph
from saag.simulation.failure_simulator import FailureSimulator
from saag.simulation.models import FailureScenario

CRITICAL_QOS = {
    "qos_reliability": "RELIABLE",
    "qos_durability": "PERSISTENT",
    "qos_transport_priority": "CRITICAL",
}
TRIVIAL_QOS = {
    "qos_reliability": "BEST_EFFORT",
    "qos_durability": "VOLATILE",
    "qos_transport_priority": "LOW",
}


def _two_topic_system(critical_weight: float, trivial_weight: float) -> GraphData:
    """One publisher per topic, one shared broker, one subscriber each.

    ``TopicCritical`` and ``TopicTrivial`` are structurally identical, so any
    difference in impact between killing PubCritical and PubTrivial can only
    come from QoS.
    """
    return GraphData(
        components=[
            ComponentData(id="PubCritical", component_type="Application", properties={"layer": "app"}),
            ComponentData(id="PubTrivial", component_type="Application", properties={"layer": "app"}),
            ComponentData(id="SubCritical", component_type="Application", properties={"layer": "app"}),
            ComponentData(id="SubTrivial", component_type="Application", properties={"layer": "app"}),
            ComponentData(id="TopicCritical", component_type="Topic", weight=critical_weight,
                          properties={"layer": "mw", **CRITICAL_QOS}),
            ComponentData(id="TopicTrivial", component_type="Topic", weight=trivial_weight,
                          properties={"layer": "mw", **TRIVIAL_QOS}),
            ComponentData(id="Broker1", component_type="Broker", properties={"layer": "mw"}),
            ComponentData(id="Node1", component_type="Node", properties={"layer": "infra"}),
        ],
        edges=[
            # Host everything on Node1 so the healthy graph is fully connected;
            # an isolated node would otherwise be a pre-existing island.
            EdgeData(source_id="PubCritical", target_id="Node1", source_type="Application",
                     target_type="Node", relation_type="RUNS_ON", dependency_type="deployment"),
            EdgeData(source_id="PubTrivial", target_id="Node1", source_type="Application",
                     target_type="Node", relation_type="RUNS_ON", dependency_type="deployment"),
            EdgeData(source_id="SubCritical", target_id="Node1", source_type="Application",
                     target_type="Node", relation_type="RUNS_ON", dependency_type="deployment"),
            EdgeData(source_id="SubTrivial", target_id="Node1", source_type="Application",
                     target_type="Node", relation_type="RUNS_ON", dependency_type="deployment"),
            EdgeData(source_id="Broker1", target_id="Node1", source_type="Broker",
                     target_type="Node", relation_type="RUNS_ON", dependency_type="deployment"),
            EdgeData(source_id="PubCritical", target_id="TopicCritical",
                     source_type="Application", target_type="Topic", relation_type="PUBLISHES_TO", dependency_type="pubsub"),
            EdgeData(source_id="SubCritical", target_id="TopicCritical",
                     source_type="Application", target_type="Topic", relation_type="SUBSCRIBES_TO", dependency_type="pubsub"),
            EdgeData(source_id="PubTrivial", target_id="TopicTrivial",
                     source_type="Application", target_type="Topic", relation_type="PUBLISHES_TO", dependency_type="pubsub"),
            EdgeData(source_id="SubTrivial", target_id="TopicTrivial",
                     source_type="Application", target_type="Topic", relation_type="SUBSCRIBES_TO", dependency_type="pubsub"),
            EdgeData(source_id="Broker1", target_id="TopicCritical",
                     source_type="Broker", target_type="Topic", relation_type="ROUTES", dependency_type="routing"),
            EdgeData(source_id="Broker1", target_id="TopicTrivial",
                     source_type="Broker", target_type="Topic", relation_type="ROUTES", dependency_type="routing"),
        ],
    )


def _impact(graph_data: GraphData, target: str, qos_weighting: bool):
    sim = FailureSimulator(SimulationGraph(graph_data=graph_data), qos_weighting=qos_weighting)
    return sim.simulate(FailureScenario(target_ids=[target])).impact


# ── Severity weighting ────────────────────────────────────────────────────────

def test_losing_a_critical_topic_outweighs_losing_a_trivial_one():
    data = _two_topic_system(critical_weight=0.95, trivial_weight=0.05)
    critical = _impact(data, "PubCritical", qos_weighting=True)
    trivial = _impact(data, "PubTrivial", qos_weighting=True)

    assert critical.throughput_loss > trivial.throughput_loss
    assert critical.composite_impact > trivial.composite_impact


def test_without_qos_weighting_the_two_are_indistinguishable():
    """The structural symmetry is real: only QoS can separate these two."""
    data = _two_topic_system(critical_weight=0.95, trivial_weight=0.05)
    critical = _impact(data, "PubCritical", qos_weighting=False)
    trivial = _impact(data, "PubTrivial", qos_weighting=False)

    assert critical.throughput_loss == pytest.approx(trivial.throughput_loss)
    assert critical.composite_impact == pytest.approx(trivial.composite_impact)


def test_severity_helper_is_neutral_when_weighting_disabled():
    data = _two_topic_system(critical_weight=0.95, trivial_weight=0.05)
    sim = FailureSimulator(SimulationGraph(graph_data=data), qos_weighting=False)
    assert sim.topic_severity("TopicCritical") == 1.0
    assert sim.topic_severity("TopicTrivial") == 1.0


def test_severity_tracks_topic_weight_when_enabled():
    data = _two_topic_system(critical_weight=0.95, trivial_weight=0.05)
    sim = FailureSimulator(SimulationGraph(graph_data=data), qos_weighting=True)
    assert sim.topic_severity("TopicCritical") == pytest.approx(0.95)
    assert sim.topic_severity("TopicTrivial") == pytest.approx(0.05)


# ── Continuous, broker-aware throughput ───────────────────────────────────────

def test_broker_loss_registers_as_throughput_loss():
    """A topic whose only router died delivers nothing.

    The previous test — ``if not publishers or not subscribers`` — never looked
    at brokers, so this scored as zero throughput loss.
    """
    data = _two_topic_system(critical_weight=0.5, trivial_weight=0.5)
    impact = _impact(data, "Broker1", qos_weighting=True)
    assert impact.throughput_loss > 0.0


def test_partial_publisher_loss_is_partial_throughput_loss():
    """Two publishers on one topic: losing one costs half, not all or nothing."""
    data = _two_topic_system(critical_weight=0.5, trivial_weight=0.5)
    data.components.append(
        ComponentData(id="PubCritical2", component_type="Application", properties={"layer": "app"})
    )
    data.edges.append(
        EdgeData(source_id="PubCritical2", target_id="TopicCritical",
                 source_type="Application", target_type="Topic", relation_type="PUBLISHES_TO", dependency_type="pubsub")
    )

    impact = _impact(data, "PubCritical", qos_weighting=True)
    # TopicCritical keeps one of two publishers → 0.5 of its severity is lost,
    # out of a total severity of 1.0 across both topics.
    assert impact.throughput_loss == pytest.approx(0.25, abs=1e-6)


def test_throughput_loss_stays_bounded():
    data = _two_topic_system(critical_weight=0.9, trivial_weight=0.9)
    for target in ("Broker1", "PubCritical", "Node1"):
        impact = _impact(data, target, qos_weighting=True)
        assert 0.0 <= impact.throughput_loss <= 1.0
        assert 0.0 <= impact.composite_impact <= 1.0


# ── Fragmentation ─────────────────────────────────────────────────────────────

def test_fragmentation_is_zero_while_survivors_stay_connected():
    """Stranded-mass weighting must not manufacture fragmentation from failures alone."""
    data = _two_topic_system(critical_weight=0.9, trivial_weight=0.1)
    impact = _impact(data, "SubTrivial", qos_weighting=True)
    assert impact.fragmentation == pytest.approx(0.0)


def test_weighted_fragmentation_is_not_double_blended():
    """IA(v)'s weighted_fragmentation used to re-apply the 0.70/0.30 split."""
    from saag.simulation.models import ImpactMetrics

    im = ImpactMetrics(fragmentation=0.4)
    im.weighted_fragmentation = im.fragmentation
    assert im.weighted_fragmentation == pytest.approx(0.4)


# ── AHP wiring ────────────────────────────────────────────────────────────────

def test_impact_weights_come_from_the_ahp_processor():
    """The weights were hardcoded literals, so perturbing the AHP matrix did nothing."""
    from saag.analysis.weight_calculator import AHPProcessor
    from saag.simulation.models import ImpactMetrics

    expected = AHPProcessor().compute_weights()
    weights = ImpactMetrics().impact_weights

    assert weights["reachability"] == pytest.approx(expected.i_reachability)
    assert weights["fragmentation"] == pytest.approx(expected.i_fragmentation)
    assert weights["throughput"] == pytest.approx(expected.i_throughput)
    assert weights["flow_disruption"] == pytest.approx(expected.i_flow_disruption)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_flow_disruption_weight_is_not_dropped():
    """AHPProcessor computed a fourth impact weight and then discarded it."""
    from saag.analysis.weight_calculator import AHPProcessor

    assert AHPProcessor().compute_weights().i_flow_disruption > 0.0
