"""
Tests for Priority 1 post-deployment calibration:
- Broker service-time telemetry (saag.forward), fitting, and replay in RuntimeController
- Testbed BrokerRelay saag.forward emission and calibration
- Per-topic MMPP burstiness fitting (fit_workload_burstiness) and replay in MessageFlowSimulator
- Zero false positives on periodic and Poisson workloads (including with outages)
"""

from __future__ import annotations

import logging

import networkx as nx
import pytest

pytest.importorskip("simpy")

from saag.simulation.calibration import (  # noqa: E402
    TelemetryCalibrator,
    TelemetryTrace,
    fit_workload_burstiness,
)
from saag.simulation.message_flow_simulator import MessageFlowSimulator  # noqa: E402
from saag.simulation.runtime.faults import FaultSpec  # noqa: E402

logging.disable(logging.WARNING)


def _brokered_graph(workload_type: str = "", capacity_msgs_s: float = 20.0) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node("A", type="Application", app_type="sensor")
    g.add_node("B", type="Application", app_type="actuator")
    t_attrs = {"type": "Topic", "frequency": 10.0}
    if workload_type:
        t_attrs["workload_type"] = workload_type
    g.add_node("t1", **t_attrs)
    g.add_node("BR1", type="Broker", capacity_msgs_s=capacity_msgs_s)
    g.add_node("H1", type="Node")
    g.add_node("H2", type="Node")
    g.add_edge("A", "t1", type="PUBLISHES_TO")
    g.add_edge("B", "t1", type="SUBSCRIBES_TO")
    g.add_edge("BR1", "t1", type="ROUTES")
    g.add_edge("A", "H1", type="RUNS_ON")
    g.add_edge("B", "H2", type="RUNS_ON")
    g.add_edge("BR1", "H1", type="RUNS_ON")
    return g


# ── 1. Broker service-time round-trip ────────────────────────────────────────

@pytest.mark.parametrize("dist,expected_family", [
    ("exponential", "exponential"),
    ("uniform", "lognormal"),
])
def test_broker_service_time_round_trip(dist, expected_family):
    # capacity_msgs_s = 20 -> true mean service time = 0.05 s (50 ms)
    g = _brokered_graph(capacity_msgs_s=20.0)
    res = MessageFlowSimulator(
        g,
        duration=40.0,
        seed=7,
        deployment_model="stations",
        service_distribution=dist,
        rng_streams="per_process",
        record_events=True,
    ).run()

    forwards = [e for e in res.events if e["name"] == "saag.forward"]
    assert len(forwards) >= 300

    trace = TelemetryTrace.from_records(res.events, start_s=0.0, end_s=40.0)
    prof = TelemetryCalibrator(infer_policies=False).calibrate(trace)

    assert "BR1" in prof.broker_service_time_dist
    fit = prof.broker_service_time_dist["BR1"]
    assert fit["dist"] == expected_family
    assert fit["mean_s"] == pytest.approx(0.05, rel=0.15)

    # Replay on a graph with a DIFFERENT declared broker capacity (100 msgs/s = 10 ms):
    # the calibrated profile must override it back to ~50 ms!
    g_wrong_cap = _brokered_graph(capacity_msgs_s=100.0)
    res_cal = MessageFlowSimulator(
        g_wrong_cap,
        duration=40.0,
        seed=11,
        deployment_model="stations",
        telemetry_profile=prof,
        rng_streams="per_process",
        record_events=True,
    ).run()
    cal_trace = TelemetryTrace.from_records(res_cal.events, start_s=0.0, end_s=40.0)
    cal_prof = TelemetryCalibrator(infer_policies=False).calibrate(cal_trace)
    assert cal_prof.broker_service_time_dist["BR1"]["mean_s"] == pytest.approx(0.05, rel=0.20)


# ── 2. Testbed BrokerRelay emits saag.forward ────────────────────────────────

def test_testbed_broker_relay_emits_forward_telemetry(tmp_path):
    from pathlib import Path
    from saag.testbed.agent import run_host
    from saag.testbed.deployment import build_deployment
    from saag.testbed.runtime import LocalTransport, TelemetrySink

    scenario = Path(__file__).resolve().parents[1] / "data" / "scenarios" / "tiny_system.json"
    plan = build_deployment(scenario)
    out_file = tmp_path / "all.jsonl"
    run_host(plan, "all", LocalTransport(), TelemetrySink(out_file), duration_s=1.5, seed=3)

    trace = TelemetryTrace.load([out_file])
    forwards = [e for e in trace.events if e["kind"] == "forward"]
    assert len(forwards) >= 5
    prof = TelemetryCalibrator(infer_policies=False).calibrate(trace)
    assert len(prof.broker_service_time_dist) >= 1
    assert all(fit["mean_s"] > 0.0 for fit in prof.broker_service_time_dist.values())


# ── 3. MMPP burstiness calibration & replay ──────────────────────────────────

def test_mmpp_burstiness_round_trip():
    g = _brokered_graph()
    res = MessageFlowSimulator(
        g,
        duration=120.0,
        seed=42,
        workload_model="mmpp",
        burst_ratio=4.0,
        burst_fraction=0.1,
        mean_burst_s=1.0,
        rng_streams="per_process",
        record_events=True,
    ).run()

    trace = TelemetryTrace.from_records(res.events, start_s=0.0, end_s=120.0)
    prof = TelemetryCalibrator(infer_policies=False).calibrate(trace)

    wp = prof.workload_params["t1"]
    assert wp["model"] == "mmpp"
    assert wp["cv"] > 1.15
    assert wp["fano"] > 1.5
    assert 1.5 <= wp["burst_ratio"] <= 6.0
    assert 0.02 <= wp["burst_fraction"] <= 0.35
    assert wp["mean_burst_s"] >= 0.25

    # Replay with workload_model="declared" (default) + telemetry_profile=prof:
    # the simulator must replay t1 as bursty MMPP while keeping the ~10 Hz mean rate.
    res_replay = MessageFlowSimulator(
        g,
        duration=240.0,
        seed=99,
        telemetry_profile=prof,
        rng_streams="per_process",
        record_events=True,
    ).run()
    replay_trace = TelemetryTrace.from_records(res_replay.events, start_s=0.0, end_s=240.0)
    replay_wp = fit_workload_burstiness(replay_trace.times("publish", topic="t1"), 0.0, 240.0)
    assert replay_wp is not None and replay_wp["model"] == "mmpp"
    # Replay mean rate matches the calibrated rate within MMPP finite-horizon sampling variance
    replay_rate = len(replay_trace.times("publish", topic="t1")) / 240.0
    assert replay_rate == pytest.approx(prof.msg_rate_per_sec["t1"], rel=0.20)


# ── 4. No false positives on periodic, Poisson, or crashed periodic traces ───

def test_periodic_and_poisson_are_not_misclassified_as_mmpp():
    # Periodic
    g_per = _brokered_graph()
    r_per = MessageFlowSimulator(
        g_per, duration=40.0, seed=5, rng_streams="per_process", record_events=True
    ).run()
    t_per = TelemetryTrace.from_records(r_per.events, start_s=0.0, end_s=40.0)
    p_per = TelemetryCalibrator(infer_policies=False).calibrate(t_per)
    assert p_per.workload_params["t1"]["model"] == "periodic"

    # Homogeneous Poisson
    g_poi = _brokered_graph(workload_type="poisson")
    r_poi = MessageFlowSimulator(
        g_poi, duration=60.0, seed=5, rng_streams="per_process", record_events=True
    ).run()
    t_poi = TelemetryTrace.from_records(r_poi.events, start_s=0.0, end_s=60.0)
    p_poi = TelemetryCalibrator(infer_policies=False).calibrate(t_poi)
    assert p_poi.workload_params["t1"]["model"] == "poisson"

    # Periodic with a 15-second mid-run crash outage must NOT be mistaken for MMPP
    r_crash = MessageFlowSimulator(
        g_per,
        duration=50.0,
        seed=5,
        faults=[FaultSpec("A", "transient_crash", 15.0, 15.0)],
        rng_streams="per_process",
        record_events=True,
    ).run()
    t_crash = TelemetryTrace.from_records(r_crash.events, start_s=0.0, end_s=50.0)
    p_crash = TelemetryCalibrator(infer_policies=False).calibrate(t_crash)
    assert p_crash.workload_params["t1"]["model"] == "periodic"
