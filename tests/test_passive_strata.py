"""
tests/test_passive_strata.py — Topic and host-Node ground truth
===============================================================

``FaultInjector`` could not express the failure of a Topic or a physical Node:
``compute_topic_loss`` derived a topic's feed loss from its publishers and
routers and never asked whether the topic itself was down, and RUNS_ON was not
indexed at all. Both types scored a constant 0.0, which is what left 30-47% of
each system without ground truth (JSS Section 8.3, Limitation L1).

These pin the two added branches and, more importantly, the property that makes
them safe to add mid-revision: they are **additive**. No Application, Broker or
Library label moves, so every table already computed against those strata stands.

They also pin the determinism fix the new strata exposed. Float addition is not
associative, so summing an unordered set gives a result whose last bits depend on
iteration order; those bits decide ``sub_loss >= propagation_threshold`` for a
subscriber sitting exactly on the boundary (a k/n feed-loss fraction lands on 0.2
exactly), and a flipped comparison shifts every subsequent RNG draw in the wave.
Topic injection makes such exact fractions common, which is how this surfaced.
"""

from __future__ import annotations

import networkx as nx
import pytest

from saag.simulation.fault_injector import FaultInjector

SEEDS = [42, 123, 456]


def _graph(edge_order_reversed: bool = False) -> nx.DiGraph:
    """Three publishers on two hosts, five subscribers, deliberately asymmetric.

        N0 hosts P0 (publishes T0, 1 subscriber)
        N1 hosts P1, P2 (publish T1 and T2, 2 subscribers each)

    So losing N1 costs four of five subscribers their feed and losing N0 costs
    one. A symmetric fixture makes every host cost the same mean feed loss and
    the ordering assertions below pass vacuously.

    ``edge_order_reversed`` builds the identical graph with its edges inserted
    the other way round. Every index in ``_PubSubIndex`` is a set, so this is the
    in-process stand-in for a different PYTHONHASHSEED: any label that depends on
    set iteration order will differ between the two builds.
    """
    g = nx.DiGraph()
    for i in range(3):
        g.add_node(f"P{i}", type="Application", name=f"pub{i}")
    for i in range(5):
        g.add_node(f"S{i}", type="Application", name=f"sub{i}")
    for i in range(3):
        g.add_node(f"T{i}", type="Topic", name=f"topic{i}",
                   qos={"reliability": "RELIABLE", "durability": "VOLATILE",
                        "transport_priority": "HIGH"})
    g.add_node("B0", type="Broker", name="broker")
    g.add_node("N0", type="Node", name="light-host")
    g.add_node("N1", type="Node", name="heavy-host")

    edges = [
        ("P0", "T0", "PUBLISHES_TO"),
        ("P1", "T1", "PUBLISHES_TO"),
        ("P2", "T2", "PUBLISHES_TO"),
        ("S0", "T0", "SUBSCRIBES_TO"),
        ("S1", "T1", "SUBSCRIBES_TO"), ("S2", "T1", "SUBSCRIBES_TO"),
        ("S3", "T2", "SUBSCRIBES_TO"), ("S4", "T2", "SUBSCRIBES_TO"),
        ("B0", "T0", "ROUTES"), ("B0", "T1", "ROUTES"), ("B0", "T2", "ROUTES"),
        ("P0", "N0", "RUNS_ON"), ("S0", "N0", "RUNS_ON"),
        ("P1", "N1", "RUNS_ON"), ("P2", "N1", "RUNS_ON"),
        ("S1", "N1", "RUNS_ON"), ("S2", "N1", "RUNS_ON"),
        ("S3", "N1", "RUNS_ON"), ("S4", "N1", "RUNS_ON"),
    ]
    for src, tgt, etype in (reversed(edges) if edge_order_reversed else edges):
        g.add_edge(src, tgt, type=etype)
    return g


def _labels(node_types, graph=None):
    result = FaultInjector(graph=graph or _graph(), seeds=SEEDS).run(node_types=node_types)
    return {nid: rec.impact_score for nid, rec in result.records.items()}


# ── The two added branches produce measurements ───────────────────────────────

def test_topic_failure_is_expressible():
    """A severed Topic delivers nothing, however many publishers survive."""
    labels = _labels(["Topic"])
    assert set(labels) == {"T0", "T1", "T2"}
    assert all(v > 0 for v in labels.values()), labels
    # T1 and T2 each starve two of five subscribers; T0 starves one.
    assert labels["T1"] == labels["T2"] > labels["T0"]


def test_host_failure_takes_its_residents_with_it():
    """N1 carries two publishers feeding four subscribers; N0 carries one."""
    labels = _labels(["Node"])
    assert labels["N1"] > labels["N0"] > 0


def test_host_failure_is_at_least_as_bad_as_any_single_resident():
    """A simultaneous blast cannot hurt less than one of its own components."""
    hosts = _labels(["Node"])
    apps = _labels(["Application"])
    assert hosts["N0"] >= max(apps["P0"], apps["S0"]) - 1e-9


# ── The property that keeps the published tables valid ────────────────────────

def test_active_strata_are_unaffected_by_labelling_the_passive_ones():
    """Application/Broker/Library labels must be bit-identical either way.

    A Topic never enters ``failed_nodes`` during an Application injection and a
    host has no residents to blast, so the new branches are unreachable from the
    existing strata. This is the claim that every table scored on the Application
    population still holds after the extension.
    """
    active = ["Application", "Broker", "Library"]
    alone = _labels(active)
    together = _labels(active + ["Topic", "Node"])

    assert alone, "no active-stratum nodes were labelled"
    for nid, score in alone.items():
        assert together[nid] == score, f"{nid} moved: {score} -> {together[nid]}"


# ── Determinism: labels must not depend on set iteration order ────────────────

@pytest.mark.parametrize("node_types", [
    ["Application", "Broker", "Library"],
    ["Topic", "Node"],
])
def test_labels_do_not_depend_on_edge_insertion_order(node_types):
    """The in-process stand-in for PYTHONHASHSEED.

    Before the sorted-summation fix, the largest corpus scenario produced three
    distinct Topic/Node label sets across eight processes, and its Application
    labels were unstable too.
    """
    forward = _labels(node_types, graph=_graph())
    reverse = _labels(node_types, graph=_graph(edge_order_reversed=True))
    assert forward == reverse


# ── Degenerate strata are recorded, and optionally fatal ──────────────────────

def test_degenerate_types_are_recorded_on_the_result():
    g = nx.DiGraph()
    g.add_node("L0", type="Library", name="lib")
    g.add_node("A0", type="Application", name="app")
    g.add_edge("A0", "L0", type="USES")

    result = FaultInjector(graph=g, seeds=SEEDS).run(node_types=["Library"])
    assert "Library" in result.degenerate_node_types
    assert "degenerate_node_types" in result.to_dict()


def test_strict_labels_rejects_a_degenerate_stratum():
    g = nx.DiGraph()
    g.add_node("L0", type="Library", name="lib")
    g.add_node("A0", type="Application", name="app")
    g.add_edge("A0", "L0", type="USES")

    with pytest.raises(ValueError, match="degenerate labels"):
        FaultInjector(graph=g, seeds=SEEDS, strict_labels=True).run(node_types=["Library"])
