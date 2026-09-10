"""
test_runtime_telemetry_models.py
────────────────────────────────
Unit tests for runtime telemetry data structures and serialization.
"""

import json
from pathlib import Path
import pytest

from saag.simulation.telemetry.models import (
    ComponentTelemetry,
    NodeTelemetry,
    QoSViolationEvent,
    StarvationEvent,
    SystemTelemetry,
    TelemetryScenario,
    TopicTelemetry,
)


def test_component_telemetry_to_dict():
    c = ComponentTelemetry(
        component_id="App1",
        component_type="Application",
        component_name="FlightProcessor",
        messages_sent=150,
        messages_received=200,
        messages_dropped=5,
        feed_starvation_ratio=0.15,
        processing_time_total=0.45,
        active_duration=60.0,
        is_failed=False,
    )
    d = c.to_dict()
    assert d["component_id"] == "App1"
    assert d["messages_sent"] == 150
    assert d["messages_received"] == 200
    assert d["feed_starvation_ratio"] == 0.15
    assert not d["is_failed"]


def test_topic_telemetry_to_dict():
    t = TopicTelemetry(
        topic_id="T_radar",
        topic_name="RadarTracks",
        published_count=100,
        delivered_count=98,
        dropped_queue_full=1,
        dropped_deadline=1,
        latency_p50_ms=4.5,
        latency_p95_ms=12.2,
        delivery_rate=0.98,
        qos_reliability="RELIABLE",
        deadline_ms=20.0,
    )
    d = t.to_dict()
    assert d["topic_id"] == "T_radar"
    assert d["delivered_count"] == 98
    assert d["dropped_deadline"] == 1
    assert d["latency_p95_ms"] == 12.2
    assert d["qos_reliability"] == "RELIABLE"


def test_node_telemetry_to_dict():
    n = NodeTelemetry(
        node_id="Node1",
        node_name="HostAlpha",
        messages_in=500,
        messages_out=600,
        bandwidth_bps_in=512000.0,
        bandwidth_bps_out=614400.0,
        estimated_cpu_load=0.45,
        hosted_components=["App1", "Broker1"],
    )
    d = n.to_dict()
    assert d["node_id"] == "Node1"
    assert d["hosted_components"] == ["App1", "Broker1"]
    assert d["bandwidth_bps_in"] == 512000.0


def test_system_telemetry_serialization(tmp_path: Path):
    sys_tel = SystemTelemetry(
        graph_id="atm_test",
        simulation_duration=60.0,
        seed=42,
        total_messages_generated=100,
        total_messages_delivered=95,
        total_messages_dropped=5,
        system_delivery_rate=0.95,
        system_drop_rate=0.05,
        system_latency_p50_ms=3.2,
        system_latency_p95_ms=8.5,
        component_metrics={
            "App1": ComponentTelemetry("App1", "Application", messages_sent=100),
            "App2": ComponentTelemetry("App2", "Application", messages_received=95),
        },
        topic_metrics={
            "T1": TopicTelemetry("T1", "Topic1", published_count=100, delivered_count=95),
        },
        node_metrics={
            "N1": NodeTelemetry("N1", "Node1", hosted_components=["App1"]),
        },
        qos_violations=[
            QoSViolationEvent(time=12.5, message_id="m_1", topic_id="T1", subscriber_id="App2", violation_type="deadline", latency_ms=25.0, threshold_ms=20.0),
        ],
        starvation_events=[
            StarvationEvent(time=30.0, subscriber_id="App2", topic_id="T1", feed_loss=0.5),
        ],
    )

    out_file = tmp_path / "telemetry.json"
    sys_tel.save(out_file)

    loaded = json.loads(out_file.read_text())
    assert loaded["graph_id"] == "atm_test"
    assert loaded["system_delivery_rate"] == 0.95
    assert loaded["qos_violations_count"] == 1
    assert loaded["starvation_events_count"] == 1
    assert "App1" in loaded["components"]
    assert "T1" in loaded["topics"]
    assert "N1" in loaded["nodes"]
