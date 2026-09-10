"""
test_telemetry_impact_calculator.py
───────────────────────────────────
Unit tests for TelemetryImpactCalculator and its integration with GNN labeling and validation.
"""

import pytest
from saag.prediction.data_preparation import extract_simulation_dict
from saag.simulation.telemetry.models import (
    ComponentTelemetry,
    NodeTelemetry,
    SystemTelemetry,
    TopicTelemetry,
)
from saag.simulation.telemetry.impact_calculator import TelemetryImpactCalculator


def _build_test_telemetries():
    """Build pre-fault baseline and post-fault degraded telemetries."""
    base = SystemTelemetry(
        graph_id="test_graph",
        simulation_duration=10.0,
        total_messages_generated=100,
        total_messages_delivered=100,
        system_delivery_rate=1.0,
        system_latency_p95_ms=10.0,
        component_metrics={
            "Pub1": ComponentTelemetry("Pub1", "Application", messages_sent=100),
            "Sub1": ComponentTelemetry("Sub1", "Application", messages_received=100),
            "Sub2": ComponentTelemetry("Sub2", "Application", messages_received=100),
            "Broker1": ComponentTelemetry("Broker1", "Broker", messages_routed=100),
        },
        topic_metrics={
            "T1": TopicTelemetry("T1", "Topic1", published_count=100, delivered_count=100),
        },
        node_metrics={
            "N1": NodeTelemetry("N1", "Node1"),
        },
    )

    # Degraded: Pub1 died at t=5.0s, so Sub1 and Sub2 received 0 msgs in post window
    degraded = SystemTelemetry(
        graph_id="test_graph",
        simulation_duration=10.0,
        total_messages_generated=100,
        total_messages_delivered=0,
        system_delivery_rate=0.0,
        system_latency_p95_ms=10.0,
        component_metrics={
            "Pub1": ComponentTelemetry("Pub1", "Application", messages_sent=0, is_failed=True),
            "Sub1": ComponentTelemetry("Sub1", "Application", messages_received=0),
            "Sub2": ComponentTelemetry("Sub2", "Application", messages_received=0),
            "Broker1": ComponentTelemetry("Broker1", "Broker", messages_routed=0),
        },
        topic_metrics={
            "T1": TopicTelemetry("T1", "Topic1", published_count=100, delivered_count=0, dropped_no_route=100),
        },
        node_metrics={
            "N1": NodeTelemetry("N1", "Node1"),
        },
        faulted_nodes={"Pub1"},
    )
    return base, degraded


def test_impact_calculation_detects_publisher_loss():
    base, degraded = _build_test_telemetries()
    calc = TelemetryImpactCalculator(propagation_threshold=0.2)

    impact = calc.compute_node_impact(base, degraded, target_id="Pub1", target_type="Application")
    assert impact["target_id"] == "Pub1"
    assert impact["delivery_loss"] == pytest.approx(1.0)
    assert impact["starvation_reach"] == pytest.approx(1.0)
    assert impact["impact_score"] > 0.6
    assert "Sub1" in impact["impacted_subscribers"]
    assert "Sub2" in impact["impacted_subscribers"]


def test_impact_calculation_normalizes_leaf_consumer():
    """Failing a leaf consumer (Sub1) should not report high delivery loss on Sub2."""
    base, _ = _build_test_telemetries()
    # Degraded where only Sub1 is down; Sub2 still receives all messages
    degraded_sub1 = SystemTelemetry(
        graph_id="test_graph",
        simulation_duration=10.0,
        total_messages_generated=100,
        total_messages_delivered=100,
        system_delivery_rate=0.5,
        system_latency_p95_ms=10.0,
        component_metrics={
            "Pub1": ComponentTelemetry("Pub1", "Application", messages_sent=100),
            "Sub1": ComponentTelemetry("Sub1", "Application", messages_received=0, is_failed=True),
            "Sub2": ComponentTelemetry("Sub2", "Application", messages_received=100),
            "Broker1": ComponentTelemetry("Broker1", "Broker", messages_routed=100),
        },
        topic_metrics={
            "T1": TopicTelemetry("T1", "Topic1", published_count=100, delivered_count=100),
        },
        node_metrics={
            "N1": NodeTelemetry("N1", "Node1"),
        },
        faulted_nodes={"Sub1"},
    )

    calc = TelemetryImpactCalculator(propagation_threshold=0.2)
    impact = calc.compute_node_impact(base, degraded_sub1, target_id="Sub1", target_type="Application")

    # Because Sub1's own consumption is excluded from surviving subscriber evaluation,
    # Sub2's feed rate did not drop at all -> delivery loss on surviving population is 0.0!
    assert impact["delivery_loss"] == pytest.approx(0.0)
    assert impact["starvation_reach"] == pytest.approx(0.0)
    assert impact["impact_score"] == pytest.approx(0.0)


def test_to_fault_injection_result_schema_compatibility():
    """Verify generated FaultInjectionResult feeds GNN data preparation directly."""
    base, degraded = _build_test_telemetries()
    calc = TelemetryImpactCalculator()
    impact = calc.compute_node_impact(base, degraded, target_id="Pub1", target_type="Application")

    rec = calc.to_fault_injection_record(impact, seed_scores={42: impact["impact_score"]})
    res = calc.to_fault_injection_result(
        graph_id="test_graph",
        records={"Pub1": rec},
        seeds=[42],
        unlabeled_node_ids=["Unlabeled1"],
    )

    # Validate schema
    d = res.to_dict()
    assert "records" in d
    assert "Pub1" in d["records"]
    assert "impact_score" in d["records"]["Pub1"]
    assert d["labeler"] == "RuntimeTelemetrySimulator"
    assert d["unlabeled_node_ids"] == ["Unlabeled1"]

    # Verify extract_simulation_dict from GNN data preparation parses it cleanly
    sim_dict = extract_simulation_dict(d)
    assert "Pub1" in sim_dict
    assert "composite" in sim_dict["Pub1"]
    assert sim_dict["Pub1"]["composite"] == pytest.approx(impact["impact_score"])


def test_to_impact_metrics_compatibility():
    """Verify conversion to ImpactMetrics for validation gates."""
    base, degraded = _build_test_telemetries()
    calc = TelemetryImpactCalculator()
    impact = calc.compute_node_impact(base, degraded, target_id="Pub1", target_type="Application")

    metrics = calc.to_impact_metrics(impact, base, degraded)
    assert metrics.reachability_loss == pytest.approx(1.0)
    assert metrics.composite_impact == pytest.approx(impact["impact_score"])
    assert metrics.throughput_loss >= 0.0
