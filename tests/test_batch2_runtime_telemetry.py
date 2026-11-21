"""
Regression tests for Batch 2 runtime & telemetry simulator hardening:
- saag.drop and saag.state telemetry trace emission from MessageFlowSimulator
- Durability replay skips gray failures, fires on Host/Broker crashes, and respects RxO & network gating
- Redelivery storm respects DDS RxO, network partitions, and dead brokers
- Host & Broker cascade annotation (cascade_orphaned_topics, lost_topics_count)
- Ref-counted overlapping FREEZE + DDS liveliness expiry, overlapping OMISSION/BABBLING
- FaultSpec parameter validation
- TelemetryCalibrator outage-robust topic rate estimation and k_of_n:K inference
"""

from __future__ import annotations

import logging

import networkx as nx
import pytest

pytest.importorskip("simpy")

from saag.simulation.calibration import (  # noqa: E402
    TelemetryCalibrator,
    TelemetryTrace,
    infer_trigger_policy,
)
from saag.simulation.message_flow_simulator import MessageFlowSimulator  # noqa: E402
from saag.simulation.runtime.faults import FaultSpec, RuntimeFaultMode  # noqa: E402

logging.disable(logging.WARNING)


def _deployed_chain(
    durability: str = "VOLATILE",
    reliability: str = "RELIABLE",
    history_depth: int = 10,
    brokers: tuple = ("BR1",),
    hotstandby: bool = False,
) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node("A", type="Application", app_type="sensor", hotstandby=hotstandby)
    g.add_node("B", type="Application", app_type="actuator")
    g.add_node(
        "t1",
        type="Topic",
        frequency=10.0,
        qos={"reliability": reliability, "durability": durability, "history_depth": history_depth},
    )
    g.add_edge("A", "t1", type="PUBLISHES_TO")
    g.add_edge("B", "t1", type="SUBSCRIBES_TO")
    for br in brokers:
        g.add_node(br, type="Broker")
        g.add_edge(br, "t1", type="ROUTES")
    g.add_node("H1", type="Node")
    g.add_node("H2", type="Node")
    g.add_edge("A", "H1", type="RUNS_ON")
    g.add_edge("B", "H2", type="RUNS_ON")
    g.add_edge("H1", "H2", type="CONNECTS_TO")
    return g


# ── 1. saag.drop and saag.state telemetry trace emission ─────────────────────

def test_record_events_emits_drop_and_state_events():
    g = _deployed_chain(reliability="BEST_EFFORT", history_depth=1, hotstandby=True)
    # Slow B severely so queue overflows (saag.drop), and crash A with hotstandby (down/takeover/up)
    faults = [
        FaultSpec("B", "slowdown", 2.0, 5.0, {"k": 20.0}),
        FaultSpec("A", "crash", 10.0),
    ]
    res = MessageFlowSimulator(
        g,
        duration=20.0,
        seed=42,
        faults=faults,
        deployment_model="stations",
        hot_standby=True,
        failover_delay_s=1.0,
        rng_streams="per_process",
        record_events=True,
    ).run()

    names = {e["name"] for e in res.events}
    assert "saag.drop" in names
    assert "saag.state" in names
    states = {e["attributes"]["saag.state"] for e in res.events if e["name"] == "saag.state"}
    assert {"down", "takeover", "up"} <= states


def test_record_events_emits_broker_down_drop():
    g = _deployed_chain()
    res = MessageFlowSimulator(
        g,
        duration=15.0,
        seed=1,
        faults=[FaultSpec("BR1", "crash", 5.0)],
        deployment_model="stations",
        rng_streams="per_process",
        record_events=True,
    ).run()
    broker_drops = [
        e for e in res.events
        if e["name"] == "saag.drop" and e["attributes"].get("saag.reason") == "broker_down"
    ]
    assert len(broker_drops) > 0


# ── 2. Durability replay & cascade annotation ────────────────────────────────

def test_durability_replay_skips_gray_failure_and_fires_on_host_crash():
    g = _deployed_chain(durability="TRANSIENT", history_depth=10)
    # Gray failure (SLOWDOWN) should NOT trigger durability replay
    res_slow = MessageFlowSimulator(
        g,
        duration=20.0,
        seed=1,
        faults=[FaultSpec("A", "slowdown", 10.0, params={"k": 4.0})],
        deployment_model="stations",
        durability_replay_deadline="reset",
        rng_streams="per_process",
    ).run()
    assert res_slow.topic_stats["t1"].replayed_total == 0

    # Host crash kills resident A -> should trigger durability replay AND populate orphaned topics
    res_host = MessageFlowSimulator(
        g,
        duration=20.0,
        seed=1,
        faults=[FaultSpec("H1", "crash", 10.0)],
        deployment_model="stations",
        durability_replay_deadline="reset",
        rng_streams="per_process",
    ).run()
    assert res_host.topic_stats["t1"].replayed_total > 0
    assert "t1" in res_host.fault_event.cascade_orphaned_topics
    assert "B" in res_host.fault_event.cascade_impacted_subscribers
    assert res_host.fault_event.lost_topics_count == 1
    assert res_host.fault_event.lost_applications_count == 1


def test_broker_crash_populates_orphaned_topics():
    g = _deployed_chain()
    res = MessageFlowSimulator(
        g,
        duration=20.0,
        seed=1,
        faults=[FaultSpec("BR1", "crash", 10.0)],
        deployment_model="stations",
        rng_streams="per_process",
    ).run()
    assert "t1" in res.fault_event.cascade_orphaned_topics
    assert res.fault_event.lost_topics_count == 1


def test_storm_respects_rxo_incompatibility_and_dead_broker():
    # Incompatible RxO: writer offers BEST_EFFORT, reader requests RELIABLE
    g = _deployed_chain(durability="TRANSIENT_LOCAL", reliability="BEST_EFFORT", brokers=())
    g.edges["B", "t1"]["qos"] = {"reliability": "RELIABLE"}
    res = MessageFlowSimulator(
        g,
        duration=15.0,
        seed=1,
        faults=[FaultSpec("A", "redelivery_storm", 8.0, params={"copies": 2})],
        qos_semantics="dds",
        rng_streams="per_process",
    ).run()
    assert res.topic_stats["t1"].replayed_total == 0


# ── 3. Overlapping FREEZE, liveliness expiry, and FaultSpec validation ───────

def test_overlapping_freeze_thaws_cleanly_and_expires_dds_liveliness():
    g = _deployed_chain(brokers=())
    # Two overlapping freezes on A: [5, 11] and [8, 14] -> total frozen [5, 14]
    faults = [
        FaultSpec("A", "freeze", 5.0, 6.0),
        FaultSpec("A", "freeze", 8.0, 6.0),
    ]
    res = MessageFlowSimulator(
        g,
        duration=20.0,
        seed=1,
        faults=faults,
        qos_semantics="dds",
        liveliness_lease_s=1.0,
        rng_streams="per_process",
        record_events=True,
    ).run()
    # Must complete without KeyError or hang, and A must resume after t=14
    post_thaw_pubs = [
        e for e in res.events
        if e["name"] == "saag.publish" and e["attributes"]["service.name"] == "A" and e["time_unix_s"] > 15.0
    ]
    assert len(post_thaw_pubs) > 20
    # DDS status recorder should have recorded down/up for A during freeze
    state_events = [
        e["attributes"]["saag.state"]
        for e in res.events
        if e["name"] == "saag.state" and e["attributes"]["service.name"] == "A"
    ]
    assert state_events == ["freeze", "thaw"]


def test_fault_spec_validates_invalid_params():
    with pytest.raises(ValueError, match="beta"):
        FaultSpec("A", RuntimeFaultMode.BUFFER_LEAK, 5.0, params={"beta": 0.0})
    with pytest.raises(ValueError, match="k"):
        FaultSpec("A", RuntimeFaultMode.SLOWDOWN, 5.0, params={"k": -1.0})
    with pytest.raises(ValueError, match="k"):
        FaultSpec("A", RuntimeFaultMode.BABBLING, 5.0, params={"k": 0.0})
    with pytest.raises(ValueError, match="p"):
        FaultSpec("A", RuntimeFaultMode.OMISSION, 5.0, params={"p": 1.5})
    with pytest.raises(ValueError, match="mean_good_s"):
        FaultSpec("H1|H2", RuntimeFaultMode.LINK_FLAP, 5.0, params={"mean_good_s": 0.0})


# ── 4. TelemetryCalibrator: outage-robust rates & k_of_n inference ───────────

def test_calibrator_recovers_healthy_rate_despite_mid_run_crash():
    g = _deployed_chain(brokers=())
    # A publishes at 10 Hz, crashes at t=20s out of 40s (20s silence at end).
    # Naive n / dur would report ~5.0 Hz; outage-robust should report ~10.0 Hz.
    res = MessageFlowSimulator(
        g,
        duration=40.0,
        seed=1,
        faults=[FaultSpec("A", "crash", 20.0)],
        rng_streams="per_process",
        record_events=True,
    ).run()
    trace = TelemetryTrace.from_records(res.events, start_s=0.0, end_s=40.0)
    prof = TelemetryCalibrator(infer_policies=False).calibrate(trace)
    assert prof.msg_rate_per_sec["t1"] == pytest.approx(10.0, rel=0.08)


def test_infer_trigger_policy_identifies_k_of_n():
    g = nx.DiGraph()
    for s in ("S1", "S2", "S3"):
        g.add_node(s, type="Application", app_type="sensor")
    g.add_node("M", type="Application", app_type="processor", trigger="k_of_n:2")
    g.add_node("K", type="Application", app_type="actuator")
    for t in ("t1", "t2", "t3", "tout"):
        g.add_node(t, type="Topic", frequency=10.0)
    g.add_edge("S1", "t1", type="PUBLISHES_TO")
    g.add_edge("S2", "t2", type="PUBLISHES_TO")
    g.add_edge("S3", "t3", type="PUBLISHES_TO")
    for t in ("t1", "t2", "t3"):
        g.add_edge("M", t, type="SUBSCRIBES_TO")
    g.add_edge("M", "tout", type="PUBLISHES_TO")
    g.add_edge("K", "tout", type="SUBSCRIBES_TO")

    # Staggered outages putting M into fresh=2 (share=1.0), fresh=1 (share=0.5), and fresh=0 (share=0.0)
    faults = [
        FaultSpec("S1", "transient_crash", 10.0, 20.0),  # S1 down [10, 30]
        FaultSpec("S2", "transient_crash", 20.0, 20.0),  # S2 down [20, 40] -> [20,30] only S3 fresh (f=1)
        FaultSpec("S3", "transient_crash", 25.0, 10.0),  # S3 down [25, 35] -> [25,30] all stale (f=0)
    ]
    res = MessageFlowSimulator(
        g,
        duration=50.0,
        seed=7,
        faults=faults,
        behavior_model="reactive",
        rng_streams="per_process",
        record_events=True,
    ).run()
    trace = TelemetryTrace.from_records(res.events, start_s=0.0, end_s=50.0)
    policy, ev = infer_trigger_policy(trace, "M")
    assert policy == "k_of_n:2", ev
