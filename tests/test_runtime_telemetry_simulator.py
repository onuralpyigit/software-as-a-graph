"""
test_runtime_telemetry_simulator.py
───────────────────────────────────
Unit tests for the Unified RuntimeTelemetrySimulator.
"""

import pytest
from saag.core.models import GraphData, ComponentData, EdgeData
from saag.simulation.graph import SimulationGraph
from saag.simulation.runtime_telemetry_simulator import RuntimeTelemetrySimulator
from saag.simulation.telemetry.models import TelemetryScenario


def _sample_pubsub_graph():
    """App1 -> Topic1 -> Broker1 -> Sub1, hosted on Node1 and Node2, App1 uses Lib1."""
    return GraphData(
        components=[
            ComponentData(id="App1", component_type="Application", properties={"layer": "app"}),
            ComponentData(id="Sub1", component_type="Application", properties={"layer": "app"}),
            ComponentData(id="Topic1", component_type="Topic", properties={"layer": "mw", "qos_reliability": "RELIABLE"}),
            ComponentData(id="Broker1", component_type="Broker", properties={"layer": "infra"}),
            ComponentData(id="Node1", component_type="Node", properties={"layer": "infra"}),
            ComponentData(id="Node2", component_type="Node", properties={"layer": "infra"}),
            ComponentData(id="Lib1", component_type="Library", properties={"layer": "app"}),
        ],
        edges=[
            EdgeData(source_id="App1", target_id="Topic1", source_type="Application", target_type="Topic",
                     dependency_type="pubsub", relation_type="PUBLISHES_TO", weight=1.0),
            EdgeData(source_id="Sub1", target_id="Topic1", source_type="Application", target_type="Topic",
                     dependency_type="pubsub", relation_type="SUBSCRIBES_TO", weight=1.0),
            EdgeData(source_id="Topic1", target_id="Broker1", source_type="Topic", target_type="Broker",
                     dependency_type="routing", relation_type="ROUTES", weight=1.0),
            EdgeData(source_id="App1", target_id="Node1", source_type="Application", target_type="Node",
                     dependency_type="deployment", relation_type="RUNS_ON"),
            EdgeData(source_id="Broker1", target_id="Node1", source_type="Broker", target_type="Node",
                     dependency_type="deployment", relation_type="RUNS_ON"),
            EdgeData(source_id="Sub1", target_id="Node2", source_type="Application", target_type="Node",
                     dependency_type="deployment", relation_type="RUNS_ON"),
            EdgeData(source_id="App1", target_id="Lib1", source_type="Application", target_type="Library",
                     dependency_type="library", relation_type="USES"),
        ],
    )


def test_baseline_simulation_delivers_traffic():
    graph = SimulationGraph(graph_data=_sample_pubsub_graph())
    sim = RuntimeTelemetrySimulator(graph)

    telemetry = sim.run_baseline(duration=2.0)
    assert telemetry.total_messages_generated > 0
    assert telemetry.total_messages_delivered > 0
    assert telemetry.system_delivery_rate == pytest.approx(1.0, abs=0.1)
    assert "App1" in telemetry.component_metrics
    assert telemetry.component_metrics["App1"].messages_sent > 0
    assert telemetry.component_metrics["Sub1"].messages_received > 0
    assert telemetry.topic_metrics["Topic1"].delivered_count > 0


def test_broker_fault_injection_drops_delivery():
    graph = SimulationGraph(graph_data=_sample_pubsub_graph())
    sim = RuntimeTelemetrySimulator(graph)

    telemetry = sim.run_fault(target_id="Broker1", duration=4.0, fault_time=2.0)
    assert "Broker1" in telemetry.faulted_nodes
    assert telemetry.pre_fault_delivery_rate is not None
    assert telemetry.post_fault_delivery_rate is not None
    assert telemetry.pre_fault_delivery_rate > 0.8
    # After Broker1 fails, topic delivery drops completely
    assert telemetry.post_fault_delivery_rate < 0.2


def test_node_failure_cascades_to_hosted_components():
    graph = SimulationGraph(graph_data=_sample_pubsub_graph())
    sim = RuntimeTelemetrySimulator(graph)

    # Node1 hosts App1 and Broker1. Failing Node1 must fail both.
    telemetry = sim.run_fault(target_id="Node1", duration=4.0, fault_time=2.0)
    assert "Node1" in telemetry.faulted_nodes
    assert "App1" in telemetry.faulted_nodes
    assert "Broker1" in telemetry.faulted_nodes
    assert telemetry.component_metrics["App1"].is_failed
    assert telemetry.component_metrics["Broker1"].is_failed
    assert telemetry.node_metrics["Node1"].is_failed


def test_library_failure_cascades_to_uses_consumers():
    graph = SimulationGraph(graph_data=_sample_pubsub_graph())
    sim = RuntimeTelemetrySimulator(graph)

    # Lib1 is used by App1. Failing Lib1 must silence App1.
    telemetry = sim.run_fault(target_id="Lib1", duration=4.0, fault_time=2.0)
    assert "Lib1" in telemetry.faulted_nodes
    assert "App1" in telemetry.faulted_nodes
    assert telemetry.component_metrics["App1"].is_failed


def test_qos_deadline_violation_tagged():
    data = _sample_pubsub_graph()
    # Add a strict deadline (0.01 ms) that is impossible to satisfy given 1 ms processing latency
    for c in data.components:
        if c.id == "Topic1":
            c.properties["qos_deadline_ms"] = 0.01

    graph = SimulationGraph(graph_data=data)
    sc = TelemetryScenario(duration=2.0, default_processing_time_s=0.005)
    sim = RuntimeTelemetrySimulator(graph, scenario=sc)

    telemetry = sim.simulate()
    assert len(telemetry.qos_violations) > 0
    assert any(v.violation_type == "deadline" for v in telemetry.qos_violations)
    assert telemetry.topic_metrics["Topic1"].dropped_deadline > 0


def test_sweep_all_components_produces_fault_injection_result():
    graph = SimulationGraph(graph_data=_sample_pubsub_graph())
    sim = RuntimeTelemetrySimulator(graph)

    result = sim.sweep_all_components(
        node_types=["Application", "Broker", "Node", "Library"],
        duration=2.0,
        seeds=[42],
    )
    assert isinstance(result.records, dict)
    assert "Broker1" in result.records
    assert "Node1" in result.records
    assert "App1" in result.records
    assert "Sub1" in result.records

    # Broker1 failure is critical because it breaks Topic1
    assert result.records["Broker1"].impact_score > 0.0
    # Sub1 failure is a leaf consumer; downstream starvation is 0
    assert result.records["Sub1"].impact_score < result.records["Broker1"].impact_score
