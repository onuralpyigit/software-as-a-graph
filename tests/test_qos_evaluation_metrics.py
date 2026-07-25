"""
tests/test_qos_evaluation_metrics.py — the QoS axis of the validation report
============================================================================

``saag/validation`` and ``saag/evaluation`` contained no reference to QoS at all,
so "does the model rank the components carrying critical channels correctly?"
could not be answered from any emitted artifact. A pooled Spearman ρ cannot
express it: a model can rank well overall while missing the few components that
carry the critical topics.
"""

from __future__ import annotations

import networkx as nx
import pytest

from saag.evaluation.metrics import (
    UNDEFINED,
    component_qos_exposure,
    critical_topic_coverage_at_k,
    compute_inductive_metrics,
    per_qos_tier_rho,
)

CRITICAL = {"reliability": "RELIABLE", "durability": "PERSISTENT", "transport_priority": "CRITICAL"}
TRIVIAL = {"reliability": "BEST_EFFORT", "durability": "VOLATILE", "transport_priority": "LOW"}


def _graph(n_trivial: int = 6) -> nx.DiGraph:
    """One app publishing a critical topic, several publishing trivial ones."""
    g = nx.DiGraph()
    g.add_node("TopicCritical", type="Topic", qos=dict(CRITICAL), size=256)
    g.add_node("AppCritical", type="Application")
    g.add_edge("AppCritical", "TopicCritical", type="PUBLISHES_TO")

    for i in range(n_trivial):
        g.add_node(f"TopicTrivial{i}", type="Topic", qos=dict(TRIVIAL), size=256)
        g.add_node(f"AppTrivial{i}", type="Application")
        g.add_edge(f"AppTrivial{i}", f"TopicTrivial{i}", type="PUBLISHES_TO")
    return g


def test_exposure_counts_published_mass_not_subscriptions():
    """Losing a subscriber does not silence a channel for anyone else."""
    g = _graph(n_trivial=1)
    g.add_node("AppSub", type="Application")
    g.add_edge("AppSub", "TopicCritical", type="SUBSCRIBES_TO")

    exposure = component_qos_exposure(g)
    assert exposure["AppCritical"] > 0.0
    assert "AppSub" not in exposure


def test_exposure_ranks_critical_above_trivial():
    exposure = component_qos_exposure(_graph())
    assert exposure["AppCritical"] > exposure["AppTrivial0"]


def test_coverage_rewards_ranking_the_critical_carrier_first():
    g = _graph()
    keys = ["AppCritical"] + [f"AppTrivial{i}" for i in range(6)]

    good = {"AppCritical": 1.0, **{f"AppTrivial{i}": 0.1 for i in range(6)}}
    bad = {"AppCritical": 0.0, **{f"AppTrivial{i}": 1.0 for i in range(6)}}

    good_cov = critical_topic_coverage_at_k(keys, good, g, k=1)
    bad_cov = critical_topic_coverage_at_k(keys, bad, g, k=1)

    assert good_cov["coverage"] > bad_cov["coverage"]
    assert good_cov["lift"] > 1.0, "ranking the critical carrier first must beat random"
    assert bad_cov["lift"] < 1.0


def test_coverage_is_undefined_without_qos_exposure():
    """No topics at all is a coverage gap, not a coverage of zero."""
    g = nx.DiGraph()
    g.add_node("A", type="Application")
    g.add_node("B", type="Application")
    g.add_edge("A", "B", type="DEPENDS_ON")

    result = critical_topic_coverage_at_k(["A", "B"], {"A": 1.0, "B": 0.0}, g, k=1)
    assert result["coverage"] == UNDEFINED
    assert result["reason"] == "no_qos_exposure"


def test_coverage_of_everything_is_total():
    g = _graph()
    keys = ["AppCritical"] + [f"AppTrivial{i}" for i in range(6)]
    scores = {k: 1.0 for k in keys}
    result = critical_topic_coverage_at_k(keys, scores, g, k=len(keys))
    assert result["coverage"] == pytest.approx(1.0)


# ── QoS-tier stratification ───────────────────────────────────────────────────

def test_tier_strata_separate_critical_from_trivial_carriers():
    g = _graph()
    keys = ["AppCritical"] + [f"AppTrivial{i}" for i in range(6)]
    pred = {k: float(i) for i, k in enumerate(keys)}
    truth = {k: float(i) for i, k in enumerate(keys)}

    tiers = per_qos_tier_rho(keys, pred, truth, g)
    assert len(tiers) >= 2, f"expected several QoS tiers, got {sorted(tiers)}"
    assert sum(t["n"] for t in tiers.values()) == len(keys)


def test_small_tier_is_undefined_not_zero():
    """Same contract as the node-type strata: absent is not zero."""
    g = _graph()
    keys = ["AppCritical"] + [f"AppTrivial{i}" for i in range(6)]
    pred = {k: float(i) for i, k in enumerate(keys)}
    truth = {k: float(i) for i, k in enumerate(keys)}

    tiers = per_qos_tier_rho(keys, pred, truth, g)
    critical_tier = [t for t in tiers.values() if t["n"] < 3]
    for tier in critical_tier:
        assert tier["rho"] == UNDEFINED
        assert tier["reason"] == "too_few_nodes"


def test_components_carrying_no_topic_are_omitted_not_bucketed():
    g = _graph(n_trivial=3)
    g.add_node("Orphan", type="Application")
    keys = ["AppCritical", "Orphan"] + [f"AppTrivial{i}" for i in range(3)]
    pred = {k: 1.0 for k in keys}
    truth = {k: 1.0 for k in keys}

    tiers = per_qos_tier_rho(keys, pred, truth, g)
    assert sum(t["n"] for t in tiers.values()) == len(keys) - 1


# ── Wiring into the reported metric family ────────────────────────────────────

def test_inductive_metrics_report_the_qos_axis():
    g = _graph()
    keys = ["AppCritical"] + [f"AppTrivial{i}" for i in range(6)]
    pred = {k: float(i) for i, k in enumerate(keys)}
    truth = {k: float(i) for i, k in enumerate(keys)}

    metrics = compute_inductive_metrics(pred, truth, g, population="labeled")
    assert "per_qos_tier_rho" in metrics
    assert "critical_topic_coverage_at_k" in metrics
    assert metrics["critical_topic_coverage_at_k"]["k"] == metrics["k"]
