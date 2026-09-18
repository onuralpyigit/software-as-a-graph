"""
test_message_flow_golden_signals.py
───────────────────────────────────
Automated tests for Google SRE Four Golden Signals (Latency, Traffic, Errors, Saturation)
in MessageFlowSimulator and simulation_results.
"""
import json
import pytest

pytest.importorskip("simpy")

import networkx as nx

from saag.simulation.message_flow_simulator import MessageFlowSimulator
from saag.simulation.simulation_results import (
    GoldenSignalsReport,
    LatencySignal,
    TrafficSignal,
    ErrorSignal,
    SaturationSignal,
)


def _build_test_graph(
    n_publishers: int = 1,
    n_subscribers: int = 1,
    pub_frequency: float = 20.0,
    queue_size: int = 20,
    deadline_ms: float = 200.0,
) -> nx.DiGraph:
    """Helper to build a clean pub/sub test graph with declared QoS."""
    g = nx.DiGraph()
    topic_qos = {
        "reliability": "RELIABLE",
        "durability": "VOLATILE",
        "queue_size": queue_size,
        "deadline_ms": deadline_ms,
    }
    g.add_node("/telemetry", type="Topic", frequency=pub_frequency, qos=topic_qos)

    for i in range(n_publishers):
        pub_id = f"Pub{i}"
        g.add_node(pub_id, type="Application")
        g.add_edge(pub_id, "/telemetry", type="PUBLISHES_TO")

    for j in range(n_subscribers):
        sub_id = f"Sub{j}"
        g.add_node(sub_id, type="Application")
        g.add_edge(sub_id, "/telemetry", type="SUBSCRIBES_TO")

    return g


def test_golden_signals_presence_and_types():
    """Verify that MessageFlowResult includes a valid GoldenSignalsReport."""
    g = _build_test_graph(n_publishers=1, n_subscribers=2, pub_frequency=25.0)
    sim = MessageFlowSimulator(
        graph=g,
        duration=5.0,
        seed=42,
        target_utilization=0.65,
    )
    result = sim.run()

    gs = result.golden_signals
    assert isinstance(gs, GoldenSignalsReport)
    assert isinstance(gs.latency, LatencySignal)
    assert isinstance(gs.traffic, TrafficSignal)
    assert isinstance(gs.errors, ErrorSignal)
    assert isinstance(gs.saturation, SaturationSignal)


def test_latency_decomposition_and_percentiles():
    """Verify that mean_e2e_ms decomposes into mean_queue_wait_ms and mean_service_time_ms."""
    g = _build_test_graph(n_publishers=1, n_subscribers=1, pub_frequency=30.0)
    sim = MessageFlowSimulator(
        graph=g,
        duration=6.0,
        seed=42,
        target_utilization=0.60,
    )
    result = sim.run()
    lat = result.golden_signals.latency

    assert lat.p50_ms is not None
    assert lat.p95_ms is not None
    assert lat.p99_ms is not None
    assert lat.p50_ms <= lat.p95_ms <= lat.p99_ms

    assert lat.mean_e2e_ms is not None
    assert lat.mean_queue_wait_ms is not None
    assert lat.mean_service_time_ms is not None

    # End-to-end latency should match queue wait + compute service time
    expected_sum = lat.mean_queue_wait_ms + lat.mean_service_time_ms
    assert lat.mean_e2e_ms == pytest.approx(expected_sum, rel=1e-2)


def test_traffic_signal_throughput():
    """Verify published/delivered rates and byte throughput."""
    frequency = 20.0
    duration = 5.0
    g = _build_test_graph(n_publishers=1, n_subscribers=2, pub_frequency=frequency)
    sim = MessageFlowSimulator(
        graph=g,
        duration=duration,
        seed=42,
        target_utilization=0.50,
    )
    result = sim.run()
    tr = result.golden_signals.traffic

    # Publication rate should closely track the declared 20 Hz
    assert tr.published_rate_hz == pytest.approx(frequency, rel=0.05)
    assert tr.total_messages_published == pytest.approx(frequency * duration, abs=2)
    assert tr.total_messages_delivered > 0
    assert tr.delivered_rate_hz > 0.0

    # Byte throughput checks (64 bytes payload default)
    assert tr.total_bytes_delivered == tr.total_messages_delivered * 64
    assert tr.throughput_bytes_per_s == pytest.approx(tr.total_bytes_delivered / duration, rel=1e-3)
    assert tr.throughput_kbps == pytest.approx((tr.throughput_bytes_per_s * 8.0) / 1000.0, rel=1e-3)


def test_error_accounting_under_congestion():
    """Verify that error signal correctly aggregates queue drops and deadline misses."""
    # Saturated setup: fast publisher, very small queue, and tight deadline
    g = _build_test_graph(
        n_publishers=1,
        n_subscribers=1,
        pub_frequency=100.0,
        queue_size=2,
        deadline_ms=5.0,
    )
    sim = MessageFlowSimulator(
        graph=g,
        duration=5.0,
        seed=42,
        default_processing_time_s=0.02, # 20 ms processing > 10 ms inter-arrival
        target_utilization=None,
    )
    result = sim.run()
    err = result.golden_signals.errors

    assert err.total_errors > 0
    assert 0.0 <= err.error_rate <= 1.0
    assert err.total_errors == (
        err.deadline_violations + err.queue_overflows + err.best_effort_drops + err.unserved_demand
    )


def test_saturation_signal_metrics():
    """Verify CPU utilization and queue buffer occupancy bounds."""
    g = _build_test_graph(n_publishers=2, n_subscribers=2, pub_frequency=25.0)
    sim = MessageFlowSimulator(
        graph=g,
        duration=6.0,
        seed=42,
        target_utilization=0.65,
    )
    result = sim.run()
    sat = result.golden_signals.saturation

    # Calibrated compute utilization should track ~0.65
    assert 0.0 <= sat.cpu_utilization <= 1.0
    assert sat.cpu_utilization == pytest.approx(0.65, abs=0.15)
    assert 0.0 <= sat.peak_cpu_utilization <= 1.0

    # Queue buffer occupancy should be within [0, 1]
    assert 0.0 <= sat.mean_queue_depth <= 20.0
    assert 0.0 <= sat.mean_queue_occupancy <= 1.0
    assert 0.0 <= sat.peak_queue_occupancy <= 1.0


def test_fault_event_golden_signals_shift():
    """Verify that fault injection updates signals_before and signals_after in FaultEventRecord."""
    g = _build_test_graph(n_publishers=1, n_subscribers=2, pub_frequency=20.0)
    sim = MessageFlowSimulator(
        graph=g,
        duration=10.0,
        fault_node="Pub0",
        fault_time=5.0,
        seed=42,
        target_utilization=0.65,
    )
    result = sim.run()

    fe = result.fault_event
    assert fe is not None
    assert isinstance(fe.signals_before, GoldenSignalsReport)
    assert isinstance(fe.signals_after, GoldenSignalsReport)

    # Pre-fault should have steady traffic and near-zero errors
    assert fe.signals_before.traffic.delivered_rate_hz > 0
    assert fe.signals_before.errors.error_rate == pytest.approx(0.0, abs=0.05)

    # Post-fault: publisher crashed, so delivered rate drops and errors/unserved spike
    assert fe.signals_after.traffic.delivered_rate_hz < fe.signals_before.traffic.delivered_rate_hz
    assert fe.signals_after.errors.unserved_demand > 0


def test_json_serialization_roundtrip(tmp_path):
    """Verify to_dict() and save() cleanly serialize all Golden Signals."""
    g = _build_test_graph(n_publishers=1, n_subscribers=1, pub_frequency=20.0)
    sim = MessageFlowSimulator(
        graph=g,
        duration=4.0,
        fault_node="Pub0",
        fault_time=2.0,
        seed=42,
        target_utilization=0.65,
    )
    result = sim.run()

    d = result.to_dict()
    assert "golden_signals" in d
    gs_dict = d["golden_signals"]
    assert "latency" in gs_dict
    assert "traffic" in gs_dict
    assert "errors" in gs_dict
    assert "saturation" in gs_dict

    # Check fault event serialization
    fe_dict = d["fault_event"]
    assert "signals_before" in fe_dict
    assert "signals_after" in fe_dict

    # Check disk saving
    out_file = tmp_path / "test_results.json"
    result.save(out_file)
    with open(out_file) as f:
        loaded = json.load(f)
    assert loaded["golden_signals"]["traffic"]["total_messages_published"] > 0
