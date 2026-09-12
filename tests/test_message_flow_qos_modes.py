"""
QoS-mode contracts for MessageFlowSimulator.

`qos_mode` is the engine's ablation switch, shaped after
`FaultInjector.QOS_FACTOR_MODES` so the two oracles' QoS arms are declared the
same way. These tests pin what each mode enforces, and — more importantly — the
two places where a profile can be silently replaced by a default:

  * `history_depth` becoming the reader queue capacity must fire only where the
    depth was actually *declared*, and
  * merging an edge's profile onto its topic's must override only the fields the
    edge actually declared.

Both hinge on the `*_declared` flags, because `QoSPolicy.from_node_attrs`
substitutes DEFAULT_HISTORY_DEPTH and MEDIUM for absent keys and a resolved
profile cannot otherwise tell "wrote 10" from "wrote nothing".
"""
import pytest
pytest.importorskip("simpy")

import networkx as nx

from saag.simulation.message_flow_simulator import (
    QOS_MODES,
    MessageFlowSimulator,
    QoSProfile,
    _apply_qos_mode,
    _extract_qos,
    _merge_qos,
)


def _graph(topic_qos=None, edge_qos=None, frequency: float = 20.0):
    g = nx.DiGraph()
    g.add_node("Pub", type="Application")
    g.add_node("Sub", type="Application")
    g.add_node("/t", type="Topic", frequency=frequency, qos=topic_qos or {})
    g.add_edge("Pub", "/t", type="PUBLISHES_TO")
    g.add_edge("Sub", "/t", type="SUBSCRIBES_TO", **(edge_qos or {}))
    return g


# ── Mode validation ──────────────────────────────────────────────────────────

def test_unknown_qos_mode_is_rejected():
    """Mirrors tests/test_qos_resolution.py::test_unknown_qos_factor_mode_is_rejected."""
    with pytest.raises(ValueError, match="qos_mode must be one of"):
        MessageFlowSimulator(graph=_graph(), qos_mode="qos-please")


@pytest.mark.parametrize("mode", QOS_MODES)
def test_every_declared_mode_runs(mode):
    result = MessageFlowSimulator(graph=_graph(), duration=5.0, seed=42, qos_mode=mode).run()
    assert result.qos_mode == mode, "the result must state its own ablation arm"


def test_declared_qos_is_enforced_by_default():
    """Declared QoS binds unless a caller opts out.

    The switch shipped defaulting to `legacy` while the operating point was
    still being chosen. It is `full` now: a topic that declares a deadline, a
    history depth or a priority gets them honoured without the caller having to
    know the flag exists.
    """
    sim = MessageFlowSimulator(graph=_graph())
    assert sim.qos_mode == "full"
    assert sim.target_utilization == 0.65


def test_legacy_remains_reachable():
    """The uncalibrated historical policy stays available for reproducing
    pre-QoS artifacts."""
    sim = MessageFlowSimulator(graph=_graph(), qos_mode="legacy")
    assert sim.qos_mode == "legacy"
    assert MessageFlowSimulator(
        graph=_graph(), qos_mode="legacy",
    ).run().measured_utilization == {}


# ── history_depth -> queue capacity ──────────────────────────────────────────

def test_declared_history_depth_becomes_queue_capacity():
    """What docs/failure-simulation.md 4.3 has always claimed, now true in code."""
    qos = _extract_qos({"qos": {"history_depth": 4}})
    assert qos.history_depth_declared

    effective = _apply_qos_mode(qos, "contracts", default_queue=100)
    assert effective.queue_size == 4


def test_undeclared_history_depth_does_not_impose_keep_last_10():
    """An absent depth stays an unconstrained cache, not DEFAULT_HISTORY_DEPTH.

    `from_node_attrs` fills in 10, so treating the resolved value as declared
    would silently apply the corpus's second-tightest contract to every topic
    that omits the field — including most real-world fixture topics.
    """
    qos = _extract_qos({"qos": {"reliability": "RELIABLE"}})
    assert qos.history_depth == 10 and not qos.history_depth_declared

    effective = _apply_qos_mode(qos, "contracts", default_queue=100)
    assert effective.queue_size == 100


def test_explicit_queue_size_outranks_history_depth():
    qos = _extract_qos({"queue_size": 7, "qos": {"history_depth": 4}})
    assert _apply_qos_mode(qos, "full", default_queue=100).queue_size == 7


@pytest.mark.parametrize("mode", ["none", "recovery", "legacy"])
def test_capacity_rule_is_confined_to_its_modes(mode):
    qos = _extract_qos({"qos": {"history_depth": 4}})
    assert _apply_qos_mode(qos, mode, default_queue=100).queue_size == 100


# ── Deadline gating ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode,enforced", [
    ("none", False), ("contracts", True), ("recovery", False),
    ("full", True), ("legacy", True),
])
def test_deadline_enforcement_follows_the_mode(mode, enforced):
    qos = _extract_qos({"qos": {"deadline_ms": 25.0}})
    assert qos.deadline_declared
    effective = _apply_qos_mode(qos, mode, default_queue=100)
    assert (effective.deadline_ms is not None) is enforced


def test_none_mode_neutralises_policy_but_not_load():
    """The QoS-off arm must differ in policy only.

    If `none` also dropped the load, the ablation would confound "QoS policies
    do nothing" with "nothing was ever contended" — the exact confound the whole
    exercise exists to remove.
    """
    qos = _extract_qos({
        "qos": {"reliability": "BEST_EFFORT", "durability": "PERSISTENT",
                "transport_priority": "CRITICAL", "deadline_ms": 5.0,
                "history_depth": 2},
    })
    effective = _apply_qos_mode(qos, "none", default_queue=100)

    assert effective.reliability == "RELIABLE"
    assert effective.durability == "VOLATILE"
    assert effective.transport_priority == "MEDIUM"
    assert effective.deadline_ms is None
    assert effective.queue_size == 100


# ── The (topic, subscriber) merge — the 0e regression guard ──────────────────

def test_undeclared_edge_does_not_overwrite_the_topic():
    """An edge that declares nothing must leave the topic's policy intact.

    `_extract_qos` on a bare edge returns the *engine's* fallbacks (RELIABLE /
    VOLATILE / default_queue_size), so merging on values alone would replace
    every topic's declared policy with defaults on the majority of edges.
    """
    topic = _extract_qos({
        "qos": {"reliability": "BEST_EFFORT", "durability": "PERSISTENT",
                "transport_priority": "HIGH", "history_depth": 3,
                "deadline_ms": 40.0},
    })
    edge = _extract_qos({})
    merged = _merge_qos(topic, edge)

    assert merged.reliability == "BEST_EFFORT"
    assert merged.durability == "PERSISTENT"
    assert merged.transport_priority == "HIGH"
    assert merged.history_depth == 3
    assert merged.deadline_ms == 40.0


def test_declared_edge_overrides_the_topic():
    topic = _extract_qos({"qos": {"reliability": "BEST_EFFORT", "durability": "VOLATILE"}})
    edge = _extract_qos({"qos": {"reliability": "RELIABLE"}})
    merged = _merge_qos(topic, edge)

    assert merged.reliability == "RELIABLE"     # edge declared it
    assert merged.durability == "VOLATILE"      # edge did not


@pytest.mark.parametrize("topic_ms,edge_ms,expected", [
    (40.0, 10.0, 10.0),   # reader asks tighter -> tighter wins
    (10.0, 40.0, 10.0),   # writer offers tighter -> tighter still wins
    (None, 15.0, 15.0),   # only the reader declared one
    (15.0, None, 15.0),   # only the writer declared one
])
def test_timing_terms_take_the_stricter_of_the_pair(topic_ms, edge_ms, expected):
    topic = _extract_qos({"qos": {"deadline_ms": topic_ms} if topic_ms else {}})
    edge = _extract_qos({"qos": {"deadline_ms": edge_ms} if edge_ms else {}})
    assert _merge_qos(topic, edge).deadline_ms == expected


def test_merge_is_identity_on_the_research_path():
    """`_project_topic_qos_onto_edges` copies the topic profile onto its edges,
    so the merge must not perturb anything there."""
    topic = _extract_qos({
        "qos": {"reliability": "RELIABLE", "durability": "TRANSIENT_LOCAL",
                "transport_priority": "CRITICAL", "history_depth": 5,
                "deadline_ms": 30.0},
    })
    merged = _merge_qos(topic, QoSProfile(**{**topic.__dict__}))
    assert merged == topic


def _overflows(topic_qos, edge_qos, qos_mode):
    """Run one slow-consumer graph and report how many samples overflowed."""
    graph = _graph(topic_qos=topic_qos, edge_qos=edge_qos, frequency=100.0)
    graph.nodes["Sub"]["processing_time"] = 0.05
    result = MessageFlowSimulator(
        graph=graph, duration=10.0, seed=42, qos_mode=qos_mode,
    ).run()
    return result.topic_stats["/t"].total_dropped_queue_full


def test_edge_declared_depth_reaches_the_real_queue():
    """End-to-end: a reader's shallower history has to bind the actual queue.

    The queue used to be built from the topic's profile while the edge's parsed
    profile fed nothing but the deadline check, so a reader-declared depth was
    read and discarded. A depth of 2 overflows far sooner than one of 500, which
    makes the difference observable without reaching into the engine.
    """
    shallow = _overflows({"history_depth": 500}, {"qos": {"history_depth": 2}}, "contracts")
    deep = _overflows({"history_depth": 500}, {"qos": {"history_depth": 500}}, "contracts")

    assert shallow > 0, "a 2-deep reader queue must overflow under a slow consumer"
    assert shallow > deep


def test_capacity_rule_is_inert_in_legacy_mode():
    """The same pair of graphs must be indistinguishable under `legacy`."""
    shallow = _overflows({"history_depth": 500}, {"qos": {"history_depth": 2}}, "legacy")
    deep = _overflows({"history_depth": 500}, {"qos": {"history_depth": 500}}, "legacy")
    assert shallow == deep
