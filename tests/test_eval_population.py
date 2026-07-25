"""Guards for the one-population evaluation contract.

The published Table 3 compared variants that had been scored on different node
sets: the training-free structural baselines on every node of the DEPENDS_ON
projection, the learned variants on the 20% test split of an
``{Application, Library}`` pool. Two estimators measured on two samples is not a
comparison, and the gap was large enough to invert the paper's RQ1 conclusion.

These tests pin the properties that make a row comparable:
  1. every variant is scored on an identical key set;
  2. the key set depends only on the graph and labels, never on predictions;
  3. an unmeasurable stratum reports ``undefined``, never ``0.0``.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from saag.evaluation.metrics import (
    UNDEFINED,
    aggregate_per_type,
    compute_inductive_metrics,
    resolve_eval_keys,
)


def _graph():
    """Six Applications, three Libraries, four unlabelled Topics."""
    g = nx.DiGraph()
    for i in range(6):
        g.add_node(f"A{i}", type="Application")
    for i in range(3):
        g.add_node(f"L{i}", type="Library")
    for i in range(4):
        g.add_node(f"T{i}", type="Topic")
    return g


def _labels():
    labels = {f"A{i}": 0.1 * (i + 1) for i in range(6)}
    labels.update({f"L{i}": 0.2 * (i + 1) for i in range(3)})
    # Topic labels exist but are all zero — the simulator cannot express a topic
    # failure, so this stratum is constant by construction.
    labels.update({f"T{i}": 0.0 for i in range(4)})
    return labels


def test_key_set_is_independent_of_predictions():
    """Two variants with different outputs must be scored on the same nodes."""
    g, labels = _graph(), _labels()
    good = {n: float(i) for i, n in enumerate(labels)}
    bad = {n: 1.0 for n in labels}          # constant predictor

    keys_good = resolve_eval_keys(good, labels, g, population="application")
    keys_bad = resolve_eval_keys(bad, labels, g, population="application")

    assert keys_good == keys_bad
    assert keys_good == sorted(f"A{i}" for i in range(6))


def test_population_selects_declared_types_only():
    g, labels = _graph(), _labels()
    preds = dict.fromkeys(labels, 1.0)

    assert len(resolve_eval_keys(preds, labels, g, "application")) == 6
    assert len(resolve_eval_keys(preds, labels, g, "app_lib")) == 9
    assert len(resolve_eval_keys(preds, labels, g, "labeled")) == 13


def test_zero_labels_are_kept_but_unlabelled_ids_are_dropped():
    """A simulated zero is a measurement; a node never simulated is not.

    Dropping genuine zeros would be a results-favourable filter, so membership is
    decided by an explicit unlabelled list rather than by the label value.
    """
    g, labels = _graph(), _labels()
    preds = dict.fromkeys(labels, 1.0)

    kept = resolve_eval_keys(preds, labels, g, "labeled")
    assert "T0" in kept                      # zero-valued but simulated

    dropped = resolve_eval_keys(
        preds, labels, g, "labeled", unlabeled_ids=[f"T{i}" for i in range(4)]
    )
    assert not any(k.startswith("T") for k in dropped)
    assert len(dropped) == 9


def test_constant_stratum_reports_undefined_not_zero():
    """The bug that filled Table 5 with 0.000 for Topic/Node/Library."""
    g, labels = _graph(), _labels()
    preds = {n: float(i) for i, n in enumerate(labels)}

    m = compute_inductive_metrics(preds, labels, g, population="labeled")
    per_type = m["per_type_rho"]

    assert per_type["Topic"]["rho"] == UNDEFINED
    assert per_type["Topic"]["reason"] == "constant_signal"
    assert isinstance(per_type["Application"]["rho"], float)


def test_undefined_survives_aggregation():
    """An always-undefined stratum must not average to 0.0 across runs."""
    runs = [
        {"Application": {"rho": 0.8, "n": 6}, "Topic": {"rho": UNDEFINED, "n": 4}},
        {"Application": {"rho": 0.6, "n": 6}, "Topic": {"rho": UNDEFINED, "n": 4}},
    ]
    agg = aggregate_per_type(runs, value_key="rho")

    assert agg["Topic"]["mean"] == UNDEFINED
    assert agg["Topic"]["n_seeds_undefined"] == 2
    assert agg["Application"]["mean"] == pytest.approx(0.7)
    assert agg["Application"]["n_seeds"] == 2


def test_partially_undefined_stratum_averages_only_defined_runs():
    runs = [
        {"Library": {"rho": 0.4, "n": 3}},
        {"Library": {"rho": UNDEFINED, "n": 3}},
        {"Library": {"rho": 0.6, "n": 3}},
    ]
    agg = aggregate_per_type(runs, value_key="rho")

    assert agg["Library"]["mean"] == pytest.approx(0.5)
    assert agg["Library"]["n_seeds"] == 2
    assert agg["Library"]["n_seeds_undefined"] == 1


def test_coverage_is_always_reported():
    """A labelling gap must be visible, not absorbed by the set intersection."""
    g, labels = _graph(), _labels()
    preds = {f"A{i}": float(i) for i in range(6)}   # predicts Applications only

    m = compute_inductive_metrics(preds, labels, g, population="labeled")

    assert m["n_predicted"] == 6
    assert m["n_labeled"] == 13
    assert m["n_evaluated"] == 6
    assert m["eval_population"] == "labeled"


def test_pinned_keys_override_population():
    g, labels = _graph(), _labels()
    preds = {n: float(i) for i, n in enumerate(labels)}
    pinned = ["A0", "A1", "A2", "L0"]

    m = compute_inductive_metrics(preds, labels, g, eval_keys=pinned)

    assert m["n_evaluated"] == 4


def test_unknown_population_is_rejected():
    g, labels = _graph(), _labels()
    with pytest.raises(ValueError, match="unknown eval population"):
        resolve_eval_keys(labels, labels, g, population="everything")
