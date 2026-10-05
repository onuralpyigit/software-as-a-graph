"""
Contention and operating-point contracts for MessageFlowSimulator.

Stage 3 introduces the engine's only contended resource: one `ServiceStation`
per subscriber, shared across every topic it reads, sized to hit a declared
target utilization. Before it, `run()` spawned one server per SUBSCRIBES_TO edge
and nothing ever queued behind anything else — which is why no QoS contract on
the corpus was ever binding, and why I_dyn reduced to a re-measurement of the
topological oracle.

The operating point is the load-bearing part: if `target_utilization` is not
actually realised, every downstream number is derived from a requested value
rather than a measured one.
"""
import json
import statistics

import pytest
pytest.importorskip("simpy")

import networkx as nx

from saag.simulation.message_flow_simulator import MessageFlowSimulator


def _fan_in(n_topics: int, frequency: float = 20.0, **topic_qos):
    """One subscriber reading `n_topics` topics, one publisher each."""
    g = nx.DiGraph()
    g.add_node("Sub", type="Application")
    for i in range(n_topics):
        g.add_node(f"/t{i}", type="Topic", frequency=frequency, qos=dict(topic_qos))
        g.add_node(f"Pub{i}", type="Application")
        g.add_edge(f"Pub{i}", f"/t{i}", type="PUBLISHES_TO")
        g.add_edge("Sub", f"/t{i}", type="SUBSCRIBES_TO")
    return g


def _run(graph, **kw):
    kw.setdefault("duration", 60.0)
    kw.setdefault("seed", 42)
    kw.setdefault("warmup_s", 10.0)
    kw.setdefault("qos_mode", "full")
    return MessageFlowSimulator(graph=graph, **kw).run()


# ── Parameter validation ─────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.2])
def test_target_utilization_must_be_a_proper_fraction(bad):
    """rho >= 1 has no steady state; rho <= 0 is not a load."""
    with pytest.raises(ValueError, match="target_utilization"):
        MessageFlowSimulator(graph=_fan_in(1), target_utilization=bad)


def test_unknown_utilization_mode_is_rejected():
    with pytest.raises(ValueError, match="utilization_mode"):
        MessageFlowSimulator(graph=_fan_in(1), utilization_mode="vibes")


def test_unknown_service_distribution_is_rejected():
    with pytest.raises(ValueError, match="service_distribution"):
        MessageFlowSimulator(graph=_fan_in(1), service_distribution="gamma")


# ── The operating point is realised, not just requested ──────────────────────

@pytest.mark.parametrize("rho", [0.3, 0.5, 0.65, 0.8])
def test_measured_utilization_tracks_the_target(rho):
    result = _run(_fan_in(4), target_utilization=rho)
    measured = list(result.measured_utilization.values())

    assert measured, "a calibrated subscriber must report its realised utilization"
    assert measured[0] == pytest.approx(rho, abs=0.05)


@pytest.mark.parametrize("frequency", [1.0, 20.0, 500.0])
def test_operating_point_is_scenario_invariant(frequency):
    """rho must mean the same thing at 1 Hz and at 500 Hz.

    This is the entire case for per-subscriber calibration over a global service
    constant. Corpus subscribers span 3-4 orders of magnitude of offered load
    (financial_trading medians 700 Hz, iot_smart_city 2 Hz), so any single
    service-time constant necessarily saturates one end and leaves the other
    idle — which is exactly what a global 200 ms service time did, stressing
    av_system while iot_smart_city stayed at 0.9989 delivery.
    """
    result = _run(_fan_in(3, frequency=frequency), target_utilization=0.8)
    assert list(result.measured_utilization.values())[0] == pytest.approx(0.8, abs=0.05)


def test_calibration_is_off_without_a_target():
    result = _run(_fan_in(3), target_utilization=None)
    assert result.measured_utilization == {}
    assert result.target_utilization is None


def test_operating_point_survives_serialization(tmp_path):
    """The realised-vs-requested check has to be performable on a saved run.

    `to_dict()` previously dropped every operating-point field, so a saved
    result stated neither which QoS arm produced it nor whether the requested
    utilization was ever reached — leaving the only recorded number the
    requested one, which is what the field's own comment warns against.
    """
    result = _run(_fan_in(4), target_utilization=0.65)
    path = tmp_path / "message_flow_results.json"
    result.save(path)
    raw = json.loads(path.read_text())

    for key in ("qos_mode", "target_utilization", "utilization_mode",
                "service_distribution", "measured_utilization", "service_time_s"):
        assert key in raw, f"{key} never reached the artifact"

    assert raw["target_utilization"] == pytest.approx(0.65)
    assert raw["qos_mode"] == "full"
    assert raw["measured_utilization"], "a calibrated run must record what it realised"
    assert statistics.mean(raw["measured_utilization"].values()) == pytest.approx(
        raw["target_utilization"], abs=0.05
    )


def test_legacy_mode_ignores_the_target():
    """`legacy` is defined as the uncalibrated arm; asking for load must not
    silently turn it into a calibrated one."""
    result = _run(_fan_in(3), qos_mode="legacy", target_utilization=0.9)
    assert result.measured_utilization == {}


def test_declared_processing_time_overrides_calibration():
    """A scenario that deliberately models a slow component keeps doing so."""
    g = _fan_in(2)
    g.nodes["Sub"]["processing_time"] = 0.02
    result = _run(g, target_utilization=0.5)

    assert result.measured_utilization == {}, "an overridden station is not calibrated"
    assert result.service_time_s["Sub"] == pytest.approx(0.02)


def test_subscriber_with_no_inbound_traffic_does_not_divide_by_zero():
    g = _fan_in(1, frequency=0.0)
    result = _run(g, target_utilization=0.8)
    assert result.measured_utilization == {}


# ── Contention actually exists now ───────────────────────────────────────────

def test_fan_in_loads_one_shared_server():
    """A subscriber's topics contend for the same compute, so delivery degrades
    as fan-in grows: N topics at 20 Hz costing 10 ms each offer N * 0.2 of one
    server, and past N = 5 the server cannot keep up.

    Under the old one-server-per-edge model this was flat — a subscriber reading
    eight topics was exactly as idle as one reading a single topic, which is why
    no amount of fan-in ever made a QoS contract bind.

    Calibration is deliberately off here: per-subscriber calibration *equalises*
    utilization across fan-in widths by construction, so it would mask the very
    effect under test. The service time is pinned instead.
    """
    def served_fraction(n_topics):
        graph = _fan_in(n_topics)
        graph.nodes["Sub"]["processing_time"] = 0.01
        result = _run(graph, target_utilization=None, qos_mode="contracts")
        delivered = sum(s.total_delivered for s in result.topic_stats.values())
        offered = sum(s.total_expected for s in result.topic_stats.values())
        return delivered / offered if offered else 0.0

    light, moderate, saturated = (served_fraction(n) for n in (1, 3, 8))

    assert light == pytest.approx(1.0, abs=0.02)
    assert moderate == pytest.approx(1.0, abs=0.02)
    assert saturated < 0.75, (
        f"eight topics offer 1.6x one server's capacity but {saturated:.3f} "
        "of demand still landed — the topics are not sharing a server"
    )


def test_deadline_violations_increase_with_utilization():
    """The binding condition is monotone in the operating point.

    Absolute rates are NOT asserted against the M/M/1 closed form. That formula
    assumes Poisson arrivals and the corpus is periodic; D/M/1 waits are much
    shorter at the same rho, so M/M/1 is a conservative *upper bound* here
    (measured: observed/predicted 0.33-0.41 under periodic arrivals, rising to
    0.42-0.88 when the same topics are switched to Poisson). Monotonicity is the
    property that actually has to hold.
    """
    graph = _fan_in(4, deadline_ms=40.0)
    violations = [
        _run(graph, target_utilization=rho).total_deadline_violations
        for rho in (0.3, 0.5, 0.8, 0.9)
    ]
    assert violations == sorted(violations), violations
    assert violations[-1] > 10 * max(1, violations[0]), (
        f"load must dominate the noise floor across the swept range: {violations}"
    )
    # Deliberately not asserting zero violations at rho = 0.3. Service is
    # exponential, so its tail produces occasional long services at any load —
    # a handful of misses under light load is the distribution behaving, not a
    # calibration fault.


# ── transport_priority orders service, and only in `full` ────────────────────

def _priority_graph():
    """One subscriber, two equal-rate topics differing only in priority."""
    g = nx.DiGraph()
    g.add_node("Sub", type="Application")
    for name, priority in (("/hi", "CRITICAL"), ("/lo", "LOW")):
        g.add_node(name, type="Topic", frequency=60.0,
                   qos={"transport_priority": priority, "deadline_ms": 30.0})
        g.add_node(f"Pub{name}", type="Application")
        g.add_edge(f"Pub{name}", name, type="PUBLISHES_TO")
        g.add_edge("Sub", name, type="SUBSCRIBES_TO")
    return g


def test_priority_favours_the_critical_topic_under_contention():
    result = _run(_priority_graph(), target_utilization=0.9, qos_mode="full")
    hi = result.topic_stats["/hi"]
    lo = result.topic_stats["/lo"]

    assert hi.total_dropped_deadline < lo.total_dropped_deadline, (
        f"CRITICAL missed {hi.total_dropped_deadline}, LOW missed "
        f"{lo.total_dropped_deadline} — priority is not ordering service"
    )


def test_priority_is_confined_to_full_mode():
    """`contracts` enforces deadlines but must stay FIFO, so the two topics --
    identical but for a field that mode does not read -- fare alike."""
    result = _run(_priority_graph(), target_utilization=0.9, qos_mode="contracts")
    hi = result.topic_stats["/hi"].total_dropped_deadline
    lo = result.topic_stats["/lo"].total_dropped_deadline

    assert hi == pytest.approx(lo, rel=0.25), (hi, lo)


# ── payload_model: declared payload size costs service, opt-in ───────────────

def _sized_fan_in(sizes, frequency: float = 20.0, **topic_qos):
    """One subscriber reading one topic per entry of `sizes` (bytes)."""
    g = _fan_in(len(sizes), frequency=frequency, **topic_qos)
    for i, size in enumerate(sizes):
        g.nodes[f"/t{i}"]["size"] = size
    return g


def test_unknown_payload_model_is_rejected():
    with pytest.raises(ValueError, match="payload_model"):
        MessageFlowSimulator(graph=_fan_in(1), payload_model="bytes")
    with pytest.raises(ValueError, match="payload_overhead_bytes"):
        MessageFlowSimulator(graph=_fan_in(1), payload_model="size",
                             payload_overhead_bytes=0.0)


def test_fixed_payload_model_is_the_default_bit_for_bit():
    """The published I_dyn oracle must not move: `fixed` is the historical engine."""
    g = _sized_fan_in([64, 8192, 32768])
    kw = dict(fault_node="Pub1", target_utilization=0.65)
    default = _run(g, **kw)
    fixed = _run(g, payload_model="fixed", **kw)

    assert fixed.fault_event.delivery_rate_before == default.fault_event.delivery_rate_before
    assert fixed.fault_event.delivery_rate_after == default.fault_event.delivery_rate_after
    assert fixed.service_time_s == default.service_time_s
    assert fixed.payload_model == "fixed"


def test_size_payload_model_keeps_the_operating_point():
    """Payload redistributes a subscriber's compute across its topics; it must
    not change how busy the subscriber is."""
    result = _run(_sized_fan_in([64, 8192, 32768]), target_utilization=0.65,
                  payload_model="size")
    assert list(result.measured_utilization.values())[0] == pytest.approx(0.65, abs=0.05)


def test_size_payload_model_scales_service_by_declared_size():
    import simpy

    g = _sized_fan_in([0, 4096], frequency=30.0)
    sim = MessageFlowSimulator(graph=g, payload_model="size", payload_overhead_bytes=1024.0)
    sub_edges = sim._edges_by_type()["SUBSCRIBES_TO"]
    station = sim._build_service_stations(
        simpy.Environment(), sub_edges, sim._node_processing_times())["Sub"]

    light, heavy = station.scale_for("/t0"), station.scale_for("/t1")
    assert heavy / light == pytest.approx(5.0)       # (1 + 4096/1024) / (1 + 0)
    assert (light + heavy) / 2 == pytest.approx(1.0)  # equal rates: mean scale is 1


def test_messages_carry_their_topic_size_only_under_size():
    g = _sized_fan_in([512, 4096])
    for model, expected in (("fixed", {"/t0": 64, "/t1": 64}),
                            ("size", {"/t0": 512, "/t1": 4096})):
        result = _run(g, target_utilization=0.5, payload_model=model)
        for tid, size in expected.items():
            stats = result.topic_stats[tid]
            assert stats.total_bytes_delivered == size * stats.total_delivered, (model, tid)


def test_heavy_payload_misses_more_deadlines_under_size():
    """Same rate, same deadline, FIFO service: under `size` the heavy topic's
    longer service shows up as deadline misses; under `fixed` the two topics are
    indistinguishable to the engine. Poisson arrivals, so the two topics do not
    arrive in lock-step and one always queue behind the other."""
    g = _sized_fan_in([0, 8192], frequency=60.0, deadline_ms=20.0)
    for tid in ("/t0", "/t1"):
        g.nodes[tid]["workload_type"] = "poisson"
    sized = _run(g, target_utilization=0.5, qos_mode="contracts", payload_model="size")
    light = sized.topic_stats["/t0"].total_dropped_deadline
    heavy = sized.topic_stats["/t1"].total_dropped_deadline
    assert heavy > 2 * light, (light, heavy)

    fixed = _run(g, target_utilization=0.5, qos_mode="contracts", payload_model="fixed")
    assert fixed.topic_stats["/t0"].total_dropped_deadline == pytest.approx(
        fixed.topic_stats["/t1"].total_dropped_deadline, rel=0.25)
