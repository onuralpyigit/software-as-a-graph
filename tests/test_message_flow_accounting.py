"""
Accounting contracts for MessageFlowSimulator.

Every defect pinned here was invisible on the shipped corpus, because nothing in
it ever filled a queue or missed a deadline: fault-free delivery was exactly
1.0000 with zero overflows on every scenario. They become first-order
measurement errors as soon as the engine is run under contention, so they are
fixed — and pinned — before any load is introduced.
"""
import pytest
pytest.importorskip("simpy")

import networkx as nx

from saag.simulation.message_flow_simulator import MessageFlowSimulator


def _one_topic(n_subscribers: int = 1, *, frequency: float = 20.0, **qos):
    """One publisher, `n_subscribers` subscribers, one topic."""
    g = nx.DiGraph()
    g.add_node("Pub", type="Application")
    g.add_node("/t", type="Topic", frequency=frequency, qos=qos or {})
    g.add_edge("Pub", "/t", type="PUBLISHES_TO")
    for i in range(n_subscribers):
        g.add_node(f"Sub{i}", type="Application")
        g.add_edge(f"Sub{i}", "/t", type="SUBSCRIBES_TO")
    return g


# ── BUG-MFS-7: the publisher's own delay inflated the publication period ──────

@pytest.mark.parametrize("frequency", [10.0, 50.0, 200.0])
@pytest.mark.parametrize("processing_time", [0.0, 0.001, 0.01])
def test_publish_rate_is_not_eroded_by_processing_time(frequency, processing_time):
    """Effective rate is min(f, 1/proc) — the publisher's compute never adds to
    the period, it only caps the rate when it exceeds one whole period.

    The loop used to be timeout(1/f) -> ... -> timeout(proc), so the effective
    rate was 1/(1/f + proc) — degraded even when proc was a small fraction of
    the period. That is a 4.8% shortfall at 50 Hz and 16.7% at 200 Hz with the
    1 ms default, so every high-rate corpus topic under-published, and the only
    test covering it allowed +/-5% at exactly 50 Hz.

    The 200 Hz / 10 ms cell is the saturated case: 10 ms of per-message compute
    physically caps a publisher at 100 Hz whatever it declares. That is a real
    limit, not the defect, and it is asserted here rather than excluded so the
    boundary between the two stays pinned.
    """
    ceiling = 1.0 / processing_time if processing_time > 0 else float("inf")
    expected = min(frequency, ceiling) * 20.0

    result = MessageFlowSimulator(
        graph=_one_topic(frequency=frequency),
        duration=20.0,
        seed=42,
        default_processing_time_s=processing_time,
        # Calibration forces publisher compute to zero -- all service belongs at
        # the subscriber -- which is exactly the quantity under test here.
        target_utilization=None,
    ).run()

    actual = result.topic_stats["/t"].total_published
    assert abs(actual - expected) <= 0.01 * expected, (
        f"f={frequency} proc={processing_time}: expected ~{expected}, got {actual}"
    )


# ── Fan-out inflation: per-topic delivery_rate could exceed 1.0 ───────────────

@pytest.mark.parametrize("n_subscribers", [1, 2, 5])
def test_topic_delivery_rate_is_a_fraction(n_subscribers):
    """delivery_rate stays in [0, 1] however wide the fan-out.

    `total_delivered` counts (subscriber, message) pairs and `total_published`
    counts messages, so dividing one by the other returned N on a healthy topic
    with N subscribers — and a matching negative drop_rate.
    """
    result = MessageFlowSimulator(
        graph=_one_topic(n_subscribers), duration=10.0, seed=42,
    ).run()
    stats = result.topic_stats["/t"]

    assert stats.n_subscribers == n_subscribers
    assert stats.total_expected == stats.total_published * n_subscribers
    assert 0.0 <= stats.delivery_rate <= 1.0
    assert 0.0 <= stats.drop_rate <= 1.0
    # Nothing is contended here, so every copy should land.
    assert stats.delivery_rate == pytest.approx(1.0, abs=0.02)


# ── Invisible drops: overflow was counted per topic but never per subscriber ──

@pytest.mark.parametrize("reliability", ["RELIABLE", "BEST_EFFORT"])
def test_overflow_drops_are_attributed_to_the_subscriber(reliability):
    """A dropped sample is charged to the subscriber that lost it.

    Overflow used to touch TopicFlowStats only. Under RELIABLE the head-drop is
    the *only* loss mode, so a RELIABLE topic's loss was unmeasurable from the
    subscriber side: the message appeared in neither received_per_topic nor
    missed_per_topic.
    """
    graph = _one_topic(frequency=100.0, reliability=reliability, durability="VOLATILE")
    graph.nodes["Sub0"]["processing_time"] = 0.05   # slow enough to back the queue up

    result = MessageFlowSimulator(
        graph=graph, duration=20.0, seed=42, default_queue_size=5,
    ).run()
    topic = result.topic_stats["/t"]
    sub = result.subscriber_stats["Sub0"]

    assert topic.total_dropped_queue_full > 0, "test graph failed to create overflow"
    assert sub.missed_per_topic["/t"] == topic.total_dropped_queue_full

    # Every published message is either received, missed, or still resident in
    # the queue at run end (capacity + at most one in service).
    accounted = sub.received_per_topic["/t"] + sub.missed_per_topic["/t"]
    assert 0 <= topic.total_published - accounted <= 6


# ── BUG-MFS-8: co-publishers of one topic all fired on the same instants ─────

@pytest.mark.parametrize("n_publishers", [2, 4, 9])
def test_co_publishers_are_phase_staggered(n_publishers):
    """A topic's publishers must spread across the period, not burst together.

    Each publisher emits at frequency/N and they all used to start at t=0, so a
    topic's aggregate stream was a burst of N every N/f seconds instead of an
    even stream at f. The aggregate *rate* was right, which is all the existing
    frequency test checked, so nothing caught it — until queue capacity came
    from history_depth, at which point a depth-1 reader kept exactly one message
    per burst and delivery collapsed to precisely 1/N. That is a phase artifact
    and would have been read as a QoS effect.
    """
    g = nx.DiGraph()
    g.add_node("Sub", type="Application")
    g.add_node("/t", type="Topic", frequency=20.0, qos={"history_depth": 1})
    g.add_edge("Sub", "/t", type="SUBSCRIBES_TO")
    for i in range(n_publishers):
        g.add_node(f"Pub{i}", type="Application")
        g.add_edge(f"Pub{i}", "/t", type="PUBLISHES_TO")

    stats = MessageFlowSimulator(
        graph=g, duration=20.0, seed=42, qos_mode="contracts",
        # Uncalibrated: this pins emission *phase*, and calibrated exponential
        # service would add queueing loss unrelated to it.
        target_utilization=None,
    ).run().topic_stats["/t"]

    assert stats.total_dropped_queue_full == 0
    assert stats.delivery_rate == pytest.approx(1.0, abs=0.02), (
        f"delivery {stats.delivery_rate:.4f} vs 1/N = {1.0 / n_publishers:.4f} "
        "— publishers are still firing in phase"
    )


def test_no_spurious_drops_when_nothing_is_contended():
    """The drop counters stay at zero on an uncontended topic."""
    result = MessageFlowSimulator(
        graph=_one_topic(3), duration=10.0, seed=42,
    ).run()
    topic = result.topic_stats["/t"]

    assert topic.total_dropped_queue_full == 0
    assert topic.total_dropped_best_effort == 0
    for sub_id in ("Sub0", "Sub1", "Sub2"):
        assert result.subscriber_stats[sub_id].missed_per_topic["/t"] == 0
