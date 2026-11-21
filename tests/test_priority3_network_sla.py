"""
Tests for Priority 3 runtime physics upgrades:
- Queued network links (network_model="queued"): serialization delay, propagation latency,
  bandwidth congestion from a babbling neighbor, and zero-delay co-located bypass
- DDS latency-SLA reactive degradation (reactive_latency_sla=True)
- MMPP calm-state staleness floor adjustment in ReactiveBehavior
"""

from __future__ import annotations

import logging

import networkx as nx
import pytest

pytest.importorskip("simpy")

from saag.simulation._stats import percentile  # noqa: E402
from saag.simulation.message_flow_simulator import MessageFlowSimulator  # noqa: E402
from saag.simulation.runtime.faults import FaultSpec  # noqa: E402

logging.disable(logging.WARNING)


def _two_host_network(
    capacity_mbps: float = 10.0,
    latency_ms: float = 0.0,
    payload_bytes: int = 1000,
    colocate: bool = False,
) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node("A", type="Application", app_type="sensor")
    g.add_node("B", type="Application", app_type="actuator")
    g.add_node("N1", type="Application", app_type="sensor")
    g.add_node("N2", type="Application", app_type="actuator")
    g.add_node("t_victim", type="Topic", frequency=10.0, size=payload_bytes)
    g.add_node("t_noisy", type="Topic", frequency=10.0, size=payload_bytes)
    g.add_edge("A", "t_victim", type="PUBLISHES_TO")
    g.add_edge("B", "t_victim", type="SUBSCRIBES_TO")
    g.add_edge("N1", "t_noisy", type="PUBLISHES_TO")
    g.add_edge("N2", "t_noisy", type="SUBSCRIBES_TO")

    g.add_node("H1", type="Node")
    g.add_node("H2", type="Node")
    g.add_edge("A", "H1", type="RUNS_ON")
    g.add_edge("B", "H1" if colocate else "H2", type="RUNS_ON")
    g.add_edge("N1", "H1", type="RUNS_ON")
    g.add_edge("N2", "H2", type="RUNS_ON")
    link_attrs = {"type": "CONNECTS_TO", "capacity_mbps": capacity_mbps}
    if latency_ms > 0:
        link_attrs["latency_ms"] = latency_ms
    g.add_edge("H1", "H2", **link_attrs)
    return g


# ── 1. Queued network links: propagation delay, serialization & congestion ───

def test_queued_network_adds_propagation_and_serialization_delay():
    # 12500 bytes = 100,000 bits on a 10 Mbps link -> 10 ms serialization + 20 ms prop = 30 ms
    g_cross = _two_host_network(capacity_mbps=10.0, latency_ms=20.0, payload_bytes=12500, colocate=False)
    g_local = _two_host_network(capacity_mbps=10.0, latency_ms=20.0, payload_bytes=12500, colocate=True)

    kw = dict(
        duration=15.0,
        seed=42,
        deployment_model="stations",
        network_model="queued",
        target_utilization=0.1,
        service_distribution="deterministic",
        rng_streams="per_process",
    )
    r_cross = MessageFlowSimulator(g_cross, **kw).run()
    r_local = MessageFlowSimulator(g_local, **kw).run()

    p50_cross = percentile(r_cross.topic_stats["t_victim"].latency_samples, 50)
    p50_local = percentile(r_local.topic_stats["t_victim"].latency_samples, 50)
    # Cross-host sees ~30 ms extra network delay compared to co-located
    assert p50_cross - p50_local >= 25.0
    assert r_cross.runtime_config["network_model"] == "queued"


def test_babbling_neighbor_congests_shared_network_link():
    # 12500 bytes @ 10 Hz = 1 Mbps per flow; on a 5 Mbps link baseline is 40% utilized.
    # Babbling N1 at 6x pushes t_noisy to 6 Mbps -> link saturates (> 100%) and delays t_victim!
    g = _two_host_network(capacity_mbps=5.0, latency_ms=1.0, payload_bytes=12500, colocate=False)
    kw = dict(
        duration=20.0,
        seed=7,
        deployment_model="stations",
        network_model="queued",
        target_utilization=0.1,
        rng_streams="per_process",
    )
    base = MessageFlowSimulator(g, **kw).run()
    babb = MessageFlowSimulator(
        g, faults=[FaultSpec("N1", "babbling", 5.0, params={"k": 6.0})], **kw
    ).run()

    p95_base = percentile(base.topic_stats["t_victim"].latency_samples, 95)
    p95_babb = percentile(babb.topic_stats["t_victim"].latency_samples, 95)
    assert p95_babb > 5.0 * p95_base


def test_network_model_validation():
    g = _two_host_network()
    with pytest.raises(ValueError, match="network_model"):
        MessageFlowSimulator(g, network_model="invalid")
    with pytest.raises(ValueError, match="deployment_model='stations'"):
        MessageFlowSimulator(g, deployment_model="none", network_model="queued")
    with pytest.raises(ValueError, match="default_link_latency_ms"):
        MessageFlowSimulator(g, default_link_latency_ms=-1.0)


# ── 2. DDS latency-SLA reactive degradation ──────────────────────────────────

def test_dds_reactive_latency_sla_propagates_gray_slowdown():
    g = nx.DiGraph()
    g.add_node("A", type="Application", app_type="sensor")
    g.add_node("B", type="Application", app_type="processor")  # all_of
    g.add_node("C", type="Application", app_type="actuator")
    g.add_node("t1", type="Topic", frequency=10.0, qos={"deadline_ms": 150.0})
    g.add_node("t2", type="Topic", frequency=10.0)
    g.add_edge("A", "t1", type="PUBLISHES_TO")
    g.add_edge("B", "t1", type="SUBSCRIBES_TO")
    g.add_edge("B", "t2", type="PUBLISHES_TO")
    g.add_edge("C", "t2", type="SUBSCRIBES_TO")

    kw = dict(
        duration=30.0,
        seed=42,
        faults=[FaultSpec("B", "slowdown", 10.0, params={"k": 5.0})],
        behavior_model="reactive",
        qos_semantics="dds",
        target_utilization=0.65,
        rng_streams="per_process",
    )
    r_off = MessageFlowSimulator(g, reactive_latency_sla=False, **kw).run()
    r_on = MessageFlowSimulator(g, reactive_latency_sla=True, **kw).run()

    # Without reactive_latency_sla, DDS delivers late samples and B keeps feeding C
    assert r_off.subscriber_stats["C"].received_post_fault > 100
    # With reactive_latency_sla, B's late deliveries stop refreshing t1 -> B stalls -> C loses feed
    assert r_on.subscriber_stats["C"].received_post_fault < r_off.subscriber_stats["C"].received_post_fault * 0.25
    assert r_on.fault_event.delivery_rate_after < r_off.fault_event.delivery_rate_after
    assert r_on.runtime_config["reactive_latency_sla"] is True


# ── 3. MMPP calm-state staleness widening ────────────────────────────────────

def test_mmpp_reactive_baseline_does_not_falsely_stall():
    g = nx.DiGraph()
    g.add_node("A", type="Application", app_type="sensor")
    g.add_node("B", type="Application", app_type="processor")
    g.add_node("C", type="Application", app_type="actuator")
    g.add_node("t1", type="Topic", frequency=10.0)
    g.add_node("t2", type="Topic", frequency=10.0)
    g.add_edge("A", "t1", type="PUBLISHES_TO")
    g.add_edge("B", "t1", type="SUBSCRIBES_TO")
    g.add_edge("B", "t2", type="PUBLISHES_TO")
    g.add_edge("C", "t2", type="SUBSCRIBES_TO")

    res = MessageFlowSimulator(
        g,
        duration=40.0,
        seed=42,
        behavior_model="reactive",
        workload_model="mmpp",
        burst_ratio=4.0,
        burst_fraction=0.1,
        mean_burst_s=1.0,
        rng_streams="per_process",
    ).run()
    # With calm_ratio-scaled staleness, healthy B should deliver the vast majority of t2 samples to C
    assert res.subscriber_stats["C"].received_per_topic["t2"] > 300
