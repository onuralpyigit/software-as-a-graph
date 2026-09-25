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


def test_threshold_f1_is_not_capped_by_the_top_k_window():
    """A perfect ranking must be able to score F1 = 1.0.

    ``f1_at_tau`` cannot: it cuts the prediction at top-K (K = 20% of the
    population) and the labels at tau, so on a skewed truth set the two set
    sizes disagree and precision is capped at n_true_critical/K no matter how
    good the ranking is. Cutting both vectors by the same relative rule removes
    the cap, which is what makes ``f1_at_threshold`` readable as an F1 score.
    """
    true = _skewed_truth()
    pred = dict(true)

    m = compute_inductive_metrics(pred, true, _graph(true))

    assert m["f1_at_threshold"] == pytest.approx(1.0)
    assert m["n_pred_critical"] == m["n_true_critical"] == 3
    assert m["f1_at_tau"] < 0.5, "the same perfect ranking, capped by K != 3"


def test_threshold_precision_and_recall_diverge():
    """The predicted set floats, so the two are free to differ."""
    true = _skewed_truth()
    # Over-predicts: n3 is pushed above the prediction-side threshold although
    # its label is nowhere near tau.
    pred = dict(true)
    pred["n3"] = 0.9

    m = compute_inductive_metrics(pred, true, _graph(true))

    assert m["n_pred_critical"] == 4
    assert m["recall_at_threshold"] == pytest.approx(1.0)
    assert m["precision_at_threshold"] == pytest.approx(0.75)
    assert m["precision_at_threshold"] != pytest.approx(m["recall_at_threshold"])


def test_f1_max_bounds_the_operating_point_and_the_trivial_floor():
    """F1max is a ceiling over cuts; F1 of "everything is critical" is the floor.

    Reported together because a bare F1max is unreadable: on a high-prevalence
    truth set the all-positive cut already scores well, so an F1max near the
    floor is evidence of nothing.
    """
    true = _skewed_truth()
    rng = np.random.default_rng(7)
    pred = {k: v + rng.normal(0, 0.05) for k, v in true.items()}

    m = compute_inductive_metrics(pred, true, _graph(true))

    assert m["f1_max"] >= m["f1_at_threshold"]
    assert m["f1_max"] >= m["f1_all_positive"]
    # 3 criticals in 50 nodes: 2 * 0.06 / 1.06.
    assert m["f1_all_positive"] == pytest.approx(2 * (3 / 50) / (1 + 3 / 50))

    reversed_pred = {k: -v for k, v in true.items()}
    worst = compute_inductive_metrics(reversed_pred, true, _graph(true))
    assert worst["f1_max"] == pytest.approx(worst["f1_all_positive"]), (
        "an exactly inverted ranking can do no better than calling everything critical"
    )


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
    """An all-zero label block must not raise, and must not report a score.

    When no node clears the tau threshold there is no critical set to be precise
    *about*, so precision/recall are undefined rather than zero. They previously
    returned 0.0, which is indistinguishable from "the model ranked every
    critical node last" — a measurable failure — and let a labelling gap read as
    a model result.
    """
    true = {f"n{i}": 0.0 for i in range(10)}
    pred = {f"n{i}": float(i) for i in range(10)}

    m = compute_inductive_metrics(pred, true, _graph(true))

    assert m["n_true_critical"] == 0
    assert np.isnan(m["pr_auc"])
    assert np.isnan(m["precision_at_tau"])
    assert np.isnan(m["recall_at_tau"])
    assert np.isnan(m["f1_at_threshold"])
    assert np.isnan(m["f1_max"])
    assert np.isnan(m["f1_all_positive"])


def test_too_few_common_nodes_still_reports_coverage():
    """The early-return path must not hide the coverage it failed on."""
    m = compute_inductive_metrics({"a": 1.0, "b": 2.0}, {"c": 1.0}, _graph(["a", "b", "c"]))

    assert m["n_predicted"] == 2
    assert m["n_labeled"] == 1
    assert m["n_evaluated"] == 0


def _low_scale_truth(n=20):
    """Max I* of 0.3, so half the max (0.15) and an absolute 0.2 disagree."""
    true = {f"n{i}": 0.001 * i for i in range(n)}
    true.update({"n0": 0.3, "n1": 0.25, "n2": 0.21, "n3": 0.15})
    return true


def test_absolute_tau_cuts_at_the_value_not_the_scenario_max():
    true = _low_scale_truth()
    pred = dict(true)

    relative = compute_inductive_metrics(pred, true, _graph(true))
    absolute = compute_inductive_metrics(pred, true, _graph(true), tau_abs=0.2)

    assert relative["tau_mode"] == "relative"
    assert relative["tau"] == pytest.approx(0.15), "tau_frac * max(I*)"
    assert relative["n_true_critical"] == 4
    assert absolute["tau_mode"] == "absolute"
    assert absolute["tau"] == pytest.approx(0.2)
    assert absolute["n_true_critical"] == 3, "only n0,n1,n2 lose >= 20% of feeds"
    assert absolute["pr_auc"] == pytest.approx(1.0), "perfect ranking of that set"


def test_absolute_tau_does_not_move_with_label_scale():
    """The same components clear 0.2 however high the scenario's max is."""
    true = _low_scale_truth()
    scaled = dict(true, n0=1.0)

    m = compute_inductive_metrics(scaled, scaled, _graph(scaled), tau_abs=0.2)
    m_rel = compute_inductive_metrics(scaled, scaled, _graph(scaled))

    assert m["n_true_critical"] == 3
    assert m_rel["n_true_critical"] == 1, "the relative cut rises to 0.5 with the max"


def test_absolute_tau_above_every_label_is_undefined_not_zero():
    true = _low_scale_truth()

    m = compute_inductive_metrics(dict(true), true, _graph(true), tau_abs=0.5)

    assert m["n_true_critical"] == 0
    assert np.isnan(m["pr_auc"])
    assert np.isnan(m["precision_at_tau"])
