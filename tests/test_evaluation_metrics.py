"""
test_evaluation_metrics.py
──────────────────────────
Pins that the reported metrics carry independent information.

`precision_at_k`, `recall_at_k` and `f1_at_k` are identically equal by
construction — both the predicted and the true top-K set contain exactly K
elements, so tp/K == tp/K. Three report columns carried one number. They are
kept for backward compatibility; these tests document the degeneracy and check
that the metrics added alongside them actually diverge.
"""

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from cli.loso_evaluate import compute_inductive_metrics  # noqa: E402


def _graph(node_ids, node_type="Application") -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(node_ids, type=node_type)
    return g


def _skewed_truth(n=50):
    """A few high-impact nodes over a long near-zero tail — the real shape.

    Measured label distributions have p50 ~= 0.001-0.02 with only 2-9 nodes
    above half the maximum.
    """
    true = {f"n{i}": 0.001 * i for i in range(n)}
    true["n0"] = 1.0
    true["n1"] = 0.8
    true["n2"] = 0.6
    return true


def test_topk_precision_recall_f1_are_identical_by_construction():
    """Documents the degeneracy rather than pretending it is three metrics."""
    true = _skewed_truth()
    rng = np.random.default_rng(0)
    pred = {k: v + rng.normal(0, 0.1) for k, v in true.items()}

    m = compute_inductive_metrics(pred, true, _graph(true))

    assert m["precision_at_k"] == pytest.approx(m["recall_at_k"])
    assert m["f1_at_k"] == pytest.approx(m["precision_at_k"])
    assert m["overlap_at_k"] == pytest.approx(m["f1_at_k"]), "alias must match"


def test_tau_precision_and_recall_diverge():
    """The absolute-threshold set is sized by the data, so P and R differ."""
    true = _skewed_truth()
    # Rank-preserving prediction: perfect ordering, so the top-K window is right
    # but is far wider than the 3-node critical set.
    pred = dict(true)

    m = compute_inductive_metrics(pred, true, _graph(true))

    assert m["n_true_critical"] == 3, "only n0,n1,n2 clear 0.5 * max"
    assert m["k"] == 10, "top-20% of 50 nodes"
    assert m["recall_at_tau"] == pytest.approx(1.0), "all 3 critical nodes are in the top 10"
    assert m["precision_at_tau"] == pytest.approx(0.3), "only 3 of those 10 are critical"
    assert m["precision_at_tau"] != pytest.approx(m["recall_at_tau"])


def test_pr_auc_separates_a_good_ranker_from_a_random_one():
    true = _skewed_truth()
    rng = np.random.default_rng(1)

    good = compute_inductive_metrics(dict(true), true, _graph(true))
    noise = {k: float(rng.random()) for k in true}
    bad = compute_inductive_metrics(noise, true, _graph(true))

    assert good["pr_auc"] == pytest.approx(1.0)
    assert bad["pr_auc"] < good["pr_auc"]


def test_scaled_error_is_insensitive_to_label_magnitude():
    """Raw rmse tracks label scale; the scaled variant should not.

    Label maxima range from 0.053 to 0.731 across the cohort, so an unscaled
    error is dominated by which scenario it came from.
    """
    true_big = _skewed_truth()
    true_small = {k: v * 0.05 for k, v in true_big.items()}

    pred_big = {k: v * 0.9 for k, v in true_big.items()}
    pred_small = {k: v * 0.9 for k, v in true_small.items()}

    m_big = compute_inductive_metrics(pred_big, true_big, _graph(true_big))
    m_small = compute_inductive_metrics(pred_small, true_small, _graph(true_small))

    assert m_big["rmse"] != pytest.approx(m_small["rmse"], rel=0.1), (
        "sanity: raw rmse should move with label scale"
    )
    assert m_big["rmse_scaled"] == pytest.approx(m_small["rmse_scaled"], abs=1e-9), (
        "scaled error must be invariant to label magnitude"
    )
    assert m_big["label_scale_max"] == pytest.approx(1.0)


def test_coverage_counts_expose_a_labeling_gap():
    """Predicted-but-unlabelled nodes must be counted, not silently dropped."""
    true = {f"n{i}": 0.1 * i for i in range(10)}
    pred = {f"n{i}": 0.1 * i for i in range(25)}   # 15 nodes have no ground truth

    m = compute_inductive_metrics(pred, true, _graph(pred))

    assert m["n_predicted"] == 25
    assert m["n_labeled"] == 10
    assert m["n_evaluated"] == 10, "scoring happens only on the intersection"


def test_degenerate_truth_does_not_crash():
    """An all-zero label block must not raise; pr_auc is undefined there."""
    true = {f"n{i}": 0.0 for i in range(10)}
    pred = {f"n{i}": float(i) for i in range(10)}

    m = compute_inductive_metrics(pred, true, _graph(true))

    assert m["n_true_critical"] == 0
    assert np.isnan(m["pr_auc"])
    assert m["precision_at_tau"] == 0.0


def test_too_few_common_nodes_still_reports_coverage():
    """The early-return path must not hide the coverage it failed on."""
    m = compute_inductive_metrics({"a": 1.0, "b": 2.0}, {"c": 1.0}, _graph(["a", "b", "c"]))

    assert m["n_predicted"] == 2
    assert m["n_labeled"] == 1
    assert m["n_evaluated"] == 0
