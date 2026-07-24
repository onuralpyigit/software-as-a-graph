"""
test_label_stability.py
───────────────────────
The labels carry their own noise floor.

A model scoring rho=0.93 against ground truth that agrees with itself at 0.91
has saturated the labels, not underperformed. Without that ceiling travelling
alongside the labels, every reported correlation is unbounded above and easy to
over-read. These tests pin the diagnostic and the reproducibility it measures.
"""

import networkx as nx
import pytest

from saag.simulation.fault_injector import FaultInjector, RECOMMENDED_SEEDS


def _pubsub_graph(n_apps: int = 12, n_topics: int = 5) -> nx.DiGraph:
    """A pub/sub graph with sole publishers, so failures actually orphan topics.

    Both PUBLISHES_TO and SUBSCRIBES_TO point app -> topic; the index reads the
    edge target as the topic in each case. A topic is orphaned only when its
    SOLE publisher fails, so the first `n_topics` apps are given exclusive
    publish rights to create real cascade signal.

    An app must not publish and subscribe to the same topic: this is a DiGraph,
    so the second edge would overwrite the first and silently erase one relation.

    One topic is left publisher-less (an externally-fed channel). Broker failure
    only registers on such topics — the cascade computes topic feed loss from
    publisher rates whenever any publisher exists, and falls back to routing
    brokers only when there are none.
    """
    g = nx.DiGraph()
    g.add_node("B0", type="Broker")
    g.add_node("L0", type="Library")
    for t in range(n_topics):
        g.add_node(f"T{t}", type="Topic", qos_reliability="RELIABLE", qos_priority="HIGH")
        g.add_edge("B0", f"T{t}", type="ROUTES")

    for a in range(n_apps):
        g.add_node(f"A{a}", type="Application")
        # Only the first n_topics apps publish, each to exactly one topic, so
        # each is the sole publisher and its failure orphans that topic.
        # T0 is deliberately left without a publisher; see the docstring.
        if 0 < a < n_topics:
            g.add_edge(f"A{a}", f"T{a}", type="PUBLISHES_TO", rate_hz=10.0)
        # Subscribe to a different topic than the one published, so the two
        # edges never collide on the same (app, topic) pair.
        g.add_edge(f"A{a}", f"T{(a + 1) % n_topics}", type="SUBSCRIBES_TO")
        if a % 3 == 0:
            g.add_edge(f"A{a}", "L0", type="USES")
    return g


def _run(seeds, node_types=("Application", "Broker", "Library")):
    return FaultInjector(graph=_pubsub_graph(), seeds=list(seeds)).run(
        node_types=list(node_types)
    )


def test_labels_are_reproducible_across_runs():
    """Identical seeds must yield identical labels, run to run."""
    a = {n: r.impact_score for n, r in _run([42, 123]).records.items()}
    b = {n: r.impact_score for n, r in _run([42, 123]).records.items()}
    assert a == b


def test_stability_block_is_populated():
    stability = _run(RECOMMENDED_SEEDS).to_dict()["label_stability"]

    assert stability["n_seeds"] == 5
    assert stability["n_nodes"] > 0
    assert 0.0 <= stability["test_retest_spearman"] <= 1.0
    assert 0.0 <= stability["topk_jaccard"] <= 1.0
    assert stability["mean_std"] >= 0.0


def test_single_seed_reports_unmeasured_rather_than_perfect():
    """One seed cannot establish a ceiling; it must not claim rho=1.0."""
    stability = _run([42]).to_dict()["label_stability"]

    assert stability["n_seeds"] == 1
    assert stability["test_retest_spearman"] is None, (
        "a single seed has no spread to measure; reporting 1.0 would overstate "
        "label quality and give rho a fictitious ceiling"
    )
    assert "note" in stability


def test_artifact_round_trips_stability(tmp_path):
    """The ceiling must survive serialisation — that is how it reaches reports."""
    import json

    out = tmp_path / "impact_scores.json"
    _run(RECOMMENDED_SEEDS).save(out)
    raw = json.loads(out.read_text())

    assert raw["schema_version"] == "2.1"
    assert raw["label_stability"]["n_seeds"] == 5
    assert raw["label_stability"]["test_retest_spearman"] is not None


def test_degenerate_node_type_is_flagged(caplog):
    """A type the cascade cannot express must warn, not pass as measurement.

    Topic and Node score I(v)=0 for every instance because the cascade derives
    DEPENDS_ON only from PUBLISHES_TO/SUBSCRIBES_TO/USES. Training on such a
    block teaches the model a constant.
    """
    import logging

    with caplog.at_level(logging.WARNING):
        _run([42], node_types=("Application", "Topic"))

    assert any("DEGENERATE LABELS" in r.message and "Topic" in r.message
               for r in caplog.records), (
        f"expected a degenerate-label warning for Topic; got {[r.message for r in caplog.records]}"
    )


def test_healthy_node_types_do_not_warn(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        _run(RECOMMENDED_SEEDS, node_types=("Application", "Broker", "Library"))

    assert not any("DEGENERATE LABELS" in r.message for r in caplog.records)


@pytest.mark.parametrize("node_type", ["Application", "Broker", "Library"])
def test_default_labeled_types_produce_signal(node_type):
    """Each default type must produce a non-degenerate label distribution."""
    result = _run(RECOMMENDED_SEEDS)
    scores = [r.impact_score for r in result.records.values() if r.node_type == node_type]

    assert scores, f"no {node_type} records"
    assert max(scores) > 0.0, f"all {node_type} labels are zero — not measurements"
