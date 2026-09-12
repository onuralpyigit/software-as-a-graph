"""
Durability-replay contracts for MessageFlowSimulator.

Durability carries the largest AHP sub-weight in the topic QoS model
(`QoSPolicy.W_DURABILITY = 0.62`) and until now no oracle in the project read it:
`FaultInjector`'s ladder says so in its own docstring, and this engine copied the
value into a stats field and did nothing with it.

The fault is permanent — `failed_nodes` is never cleared — so replay cannot be
triggered by a publisher recovering. It fires on *takeover*: surviving
co-publishers, or the durability service for the levels backed by one, serve the
gap the dead writer left.
"""
import pytest
pytest.importorskip("simpy")

import networkx as nx

from saag.simulation.message_flow_simulator import MessageFlowSimulator

DURABILITIES = ["VOLATILE", "TRANSIENT_LOCAL", "TRANSIENT", "PERSISTENT"]


def _topic(durability, n_publishers=2, deadline_ms=None, history_depth=20,
           frequency=20.0):
    g = nx.DiGraph()
    qos = {"durability": durability, "history_depth": history_depth}
    if deadline_ms is not None:
        qos["deadline_ms"] = deadline_ms
    g.add_node("/t", type="Topic", frequency=frequency, qos=qos)
    g.add_node("Sub", type="Application")
    g.add_edge("Sub", "/t", type="SUBSCRIBES_TO")
    for i in range(n_publishers):
        g.add_node(f"P{i}", type="Application")
        g.add_edge(f"P{i}", "/t", type="PUBLISHES_TO")
    return g


def _run(graph, **kw):
    kw.setdefault("duration", 60.0)
    kw.setdefault("fault_time", 30.0)
    kw.setdefault("seed", 42)
    kw.setdefault("qos_mode", "full")
    kw.setdefault("target_utilization", 0.65)
    kw.setdefault("fault_node", "P0")
    return MessageFlowSimulator(graph=graph, **kw).run()


# ── The eligibility ladder ───────────────────────────────────────────────────

@pytest.mark.parametrize("durability,replays", [
    ("VOLATILE", False),          # retains nothing
    ("TRANSIENT_LOCAL", True),    # a surviving co-publisher still holds history
    ("TRANSIENT", True),
    ("PERSISTENT", True),
])
def test_replay_when_a_co_publisher_survives(durability, replays):
    stats = _run(_topic(durability)).topic_stats["/t"]
    assert (stats.replayed_total > 0) is replays


@pytest.mark.parametrize("durability,replays", [
    ("VOLATILE", False),
    ("TRANSIENT_LOCAL", False),   # history lived in the writer and died with it
    ("TRANSIENT", True),          # backed by a durability service
    ("PERSISTENT", True),
])
def test_replay_when_the_topic_is_orphaned(durability, replays):
    stats = _run(_topic(durability, n_publishers=1)).topic_stats["/t"]
    assert (stats.replayed_total > 0) is replays


def test_replay_effect_is_monotone_in_durability():
    """Losing the same publisher must hurt less as durability strengthens.

    Monotone in `QoSPolicy.DURABILITY_SCORES`, which is what makes the mechanism
    a defensible reading of the AHP weight rather than an arbitrary rule.
    """
    losses = [_run(_topic(d)).fault_event.i_dyn for d in DURABILITIES]
    assert losses == sorted(losses, reverse=True), dict(zip(DURABILITIES, losses))
    assert losses[0] > losses[-1], "durability made no difference at all"


# ── Replay is confined to its modes ──────────────────────────────────────────

@pytest.mark.parametrize("mode,replays", [
    ("none", False), ("contracts", False),
    ("recovery", True), ("full", True), ("legacy", False),
])
def test_replay_follows_the_mode(mode, replays):
    """`recovery` exists so durability's contribution reads separately from the
    deadline/capacity contribution of `contracts`."""
    stats = _run(_topic("PERSISTENT"), qos_mode=mode).topic_stats["/t"]
    assert (stats.replayed_total > 0) is replays


# ── Windowing: a recovery is a post-fault event ──────────────────────────────

def test_replay_does_not_inflate_the_pre_fault_window():
    """A replayed sample carries a pre-fault `created_at` but fills a post-fault
    gap, so it must be credited to the post window.

    Bucketing it by `created_at` counted the recovery in the pre window on top of
    the original delivery — double-counting there, which drove
    `delivery_rate_before` above 1.0 and reported I_dyn > 1.0 for an orphaned
    PERSISTENT topic.
    """
    event = _run(_topic("PERSISTENT", n_publishers=1)).fault_event

    assert event.delivery_rate_before == pytest.approx(1.0, abs=0.02)
    assert event.delivery_rate_after > 0.0, "replay must show up after the fault"
    assert event.i_dyn <= 1.0


def test_replay_is_capped_by_what_was_actually_lost():
    """A durability service returns what the reader missed, never more.

    Without the cap a 1 Hz topic with history_depth=100 injects 100 samples into
    a post window that only ever expected ~30 — which the removed [0,1] clamp
    would no longer hide.
    """
    stats = _run(
        _topic("PERSISTENT", n_publishers=1, history_depth=100, frequency=1.0),
    ).topic_stats["/t"]

    assert stats.replayed_total <= stats.published_post
    assert _run(
        _topic("PERSISTENT", n_publishers=1, history_depth=100, frequency=1.0),
    ).fault_event.delivery_rate_after <= 1.0


def test_replay_does_not_push_delivery_above_one():
    """Recovery must not make a topic look better than fully served.

    Replay recovers demand the dead writer never emitted, so crediting it to the
    lifetime `total_delivered` — whose denominator counts messages publishers
    actually emitted — reported delivery rates above 1.0 (observed: 1.0500 on a
    faulted healthcare topic). The windowed counters are safe to credit, because
    their denominator is demand and replay is capped by it.
    """
    result = _run(_topic("PERSISTENT", n_publishers=1, history_depth=50))
    stats = result.topic_stats["/t"]

    assert stats.replayed_delivered > 0, "the test graph produced no replay"
    assert 0.0 <= stats.delivery_rate <= 1.0
    for windowed in (stats.delivery_rate_pre, stats.delivery_rate_post):
        if windowed is not None:
            assert 0.0 <= windowed <= 1.0
    assert result.fault_event.delivery_rate_after <= 1.0


# ── Durability recovers state, not timeliness ────────────────────────────────

def test_stale_replay_is_dropped_by_a_declared_deadline():
    """Under `original`, replayed samples keep their age and a deadline drops them.

    This is the substantive modelling claim of the stage: durability restores
    *state*, not *timeliness*. Its consequence is that durability is inert on the
    60-80% of corpus topics that declare a deadline, which is a real finding
    about the limits of durability under a real-time contract.
    """
    stats = _run(_topic("PERSISTENT", deadline_ms=500.0),
                 durability_replay_deadline="original").topic_stats["/t"]

    assert stats.replayed_enqueued > 0, "samples must reach the queue"
    assert stats.replayed_delivered < stats.replayed_enqueued, (
        "a declared deadline must reject some stale replay"
    )


def test_reset_policy_makes_replay_timely_again():
    """The sensitivity arm: treating a replayed sample as a fresh state snapshot.

    The gap between the two policies is how far the conclusion depends on this
    choice, which is the number a reviewer will ask for.
    """
    kw = dict(graph=_topic("PERSISTENT", deadline_ms=500.0))
    original = _run(**kw, durability_replay_deadline="original").topic_stats["/t"]
    reset = _run(**kw, durability_replay_deadline="reset").topic_stats["/t"]

    assert reset.replayed_delivered > original.replayed_delivered
    assert reset.replayed_delivered == reset.replayed_enqueued


def test_unknown_replay_deadline_policy_is_rejected():
    with pytest.raises(ValueError, match="durability_replay_deadline"):
        MessageFlowSimulator(
            graph=_topic("PERSISTENT"), durability_replay_deadline="whenever",
        )
