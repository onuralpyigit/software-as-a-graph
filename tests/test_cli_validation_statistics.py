"""
The CLI validation gate suite (`cli/validation/statistics.py`).

This module had no test coverage while it was the only gate suite the JSS
manuscript cites (supplementary.tex, the release-gate bullet). Each test below
pins one of the properties that were wrong when it went unmeasured.
"""
import pytest

from cli.validation.scoring import NodeScores
from cli.validation.statistics import (
    GATE_THRESHOLDS, ValidationResult, evaluate_gates, run_statistical_tests,
    top_k_agreement, top_k_sets,
)


def _scores(spec):
    """spec: {node_id: (Q, I, node_type, is_articulation_point)}"""
    out = {}
    for nid, (q, i, ntype, ap) in spec.items():
        out[nid] = NodeScores(
            node_id=nid, node_type=ntype, Q=q, I=i,
            is_articulation_point=ap, degree_centrality=0.0,
        )
    return out


def _apps(n, *, agreeing=True):
    spec = {}
    for i in range(n):
        q = (n - i) / n if agreeing else (i + 1) / n
        spec[f"A{i}"] = (q, (n - i) / n, "Application", False)
    return _scores(spec)


# ---------------------------------------------------------------------------
# One statistic, one name
# ---------------------------------------------------------------------------

def test_overlap_is_reported_once_not_under_four_names():
    """precision@K, recall@K, F1@K and FTR were one number wearing four hats.

    Both the predicted and the ground-truth positive set are "the top K", so they
    are the same size and precision == recall == F1 by construction; FTR was
    defined as 1 - F1, its exact complement.
    """
    fields = set(ValidationResult.__dataclass_fields__)
    assert "overlap_at_k" in fields
    for retired in ("precision_at_k", "recall_at_k", "f1_at_k", "ftr"):
        assert retired not in fields, f"{retired} should have been retired"


def test_gate_table_has_four_conditions():
    """FTR's implied threshold was strictly tighter than the overlap condition's
    in every topology class, so the overlap condition could never bind."""
    for topo_class, thresholds in GATE_THRESHOLDS.items():
        assert len(thresholds) == 4, f"{topo_class} should carry four thresholds"

    result = ValidationResult(seed=1, qos_enabled=False, n_nodes=10, n_app_nodes=10)
    gates = evaluate_gates(result, "medium")
    assert len(gates) == 4
    assert not any("ftr" in key for key in gates)


# ---------------------------------------------------------------------------
# SPOF-F1 must depend on the predictor
# ---------------------------------------------------------------------------

def test_spof_f1_distinguishes_two_predictors():
    """SPOF-F1 used to compare articulation points against articulation points.

    `spof_actual` was a subset of `spof_pred`, so recall was identically 1.0 and
    the statistic never read Q(v) at all -- it returned the same value for RM,
    RM-QoS and every GNN variant, while gating all of them.
    """
    # A0..A3 are articulation points; A0 and A1 are the impactful ones.
    good = _scores({
        "A0": (0.99, 0.9, "Application", True),
        "A1": (0.98, 0.8, "Application", True),
        "A2": (0.10, 0.1, "Application", True),
        "A3": (0.05, 0.0, "Application", True),
        "A4": (0.04, 0.0, "Application", False),
        "A5": (0.03, 0.0, "Application", False),
    })
    bad = _scores({
        "A0": (0.05, 0.9, "Application", True),
        "A1": (0.04, 0.8, "Application", True),
        "A2": (0.99, 0.1, "Application", True),
        "A3": (0.98, 0.0, "Application", True),
        "A4": (0.97, 0.0, "Application", False),
        "A5": (0.96, 0.0, "Application", False),
    })

    good_f1 = run_statistical_tests(good, top_k=2, B=50)["spof_f1"]
    bad_f1 = run_statistical_tests(bad, top_k=2, B=50)["spof_f1"]
    assert good_f1 > bad_f1, (
        "a predictor that ranks the impactful articulation points first must score "
        f"higher than one that ranks them last (got {good_f1} vs {bad_f1})"
    )


# ---------------------------------------------------------------------------
# Predictive gain must be signed
# ---------------------------------------------------------------------------

def test_predictive_gain_penalises_an_anti_correlated_predictor():
    """pg was |rho| - |rho_deg|, which scored rho = -0.9 as a large gain."""
    spec = {}
    for i in range(10):
        # Q is perfectly inverted against I; degree carries no signal.
        spec[f"A{i}"] = ((i + 1) / 10, (10 - i) / 10, "Application", False)
    scores = _scores(spec)
    for idx, ns in enumerate(scores.values()):
        ns.degree_centrality = (idx % 3) / 3.0

    stat = run_statistical_tests(scores, top_k=2, B=50)
    assert stat["spearman_rho"] < 0
    assert stat["pg"] < 0, "an inverted ranking must not report a positive gain"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_top_k_sets_break_ties_by_node_id():
    tied = _scores({f"A{i}": (0.5, 0.5, "Application", False) for i in range(6)})
    items = list(tied.values())
    gt_a, pred_a = top_k_sets(items, tied, 3)
    gt_b, pred_b = top_k_sets(list(reversed(items)), tied, 3)
    assert gt_a == gt_b == {"A0", "A1", "A2"}
    assert pred_a == pred_b == {"A0", "A1", "A2"}


def test_top_k_agreement_is_bounded():
    apps = _apps(10)
    assert top_k_agreement(list(apps.values()), apps, 2) == pytest.approx(1.0)
    assert top_k_agreement(list(apps.values()), apps, 0) == 0.0
