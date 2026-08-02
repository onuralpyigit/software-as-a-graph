"""
test_kfold_comparison_stats.py
────────────────────────────────
Pins the Stage 6 reporting-statistics helpers in reproduce/kfold_all_variants.py:
a proper confidence interval (not bare std) and paired per-scenario deltas
(not a difference of two independently-computed means) for comparing a
learned variant against a baseline. Pure-function tests — no training run
required, matching "build + verify the machinery here, apply at full corpus."
"""

import pytest

from reproduce.kfold_all_variants import (
    _build_comparison_table,
    _confidence_interval,
    _paired_deltas,
)


def test_confidence_interval_empty():
    assert _confidence_interval([]) == (None, None)


def test_confidence_interval_single_value():
    lo, hi = _confidence_interval([0.5])
    assert lo == hi == pytest.approx(0.5)


def test_confidence_interval_zero_variance():
    lo, hi = _confidence_interval([0.6, 0.6, 0.6])
    assert lo == hi == pytest.approx(0.6)


def test_confidence_interval_contains_mean_and_has_width():
    values = [0.4, 0.5, 0.6, 0.7, 0.8]
    lo, hi = _confidence_interval(values)
    mean = sum(values) / len(values)
    assert lo < mean < hi, "a dispersed sample must yield a CI with nonzero width around the mean"


def test_paired_deltas_matches_by_scenario_id():
    a = {"atm_system": 0.6, "av_system": 0.7, "enterprise_system": 0.5}
    b = {"atm_system": 0.5, "av_system": 0.5, "enterprise_system": 0.5}
    result = _paired_deltas(a, b)
    assert result["n_paired_scenarios"] == 3
    assert result["per_scenario_delta"] == {
        "atm_system": pytest.approx(0.1),
        "av_system": pytest.approx(0.2),
        "enterprise_system": pytest.approx(0.0),
    }
    assert result["mean_delta"] == pytest.approx(0.1)


def test_paired_deltas_only_uses_common_scenarios():
    a = {"atm_system": 0.6, "av_system": 0.7, "only_in_a": 0.9}
    b = {"atm_system": 0.5, "av_system": 0.5, "only_in_b": 0.1}
    result = _paired_deltas(a, b)
    assert result["n_paired_scenarios"] == 2
    assert "only_in_a" not in result["per_scenario_delta"]
    assert "only_in_b" not in result["per_scenario_delta"]


def test_paired_deltas_no_overlap_returns_none():
    assert _paired_deltas({"a": 0.5}, {"b": 0.5}) is None


def test_paired_deltas_single_scenario_degenerate_but_defined():
    """The ATM-only case: n=1 paired scenario. Must not raise, must not
    fabricate a meaningful CI out of a single point."""
    result = _paired_deltas({"atm_system": 0.6}, {"atm_system": 0.5})
    assert result["n_paired_scenarios"] == 1
    assert result["mean_delta"] == pytest.approx(0.1)
    assert result["ci_95"] == [pytest.approx(0.1), pytest.approx(0.1)]


def _fake_variant_result(scenario_rhos: dict) -> dict:
    return {
        "scenarios": [
            {
                "scenario_id": sid,
                "mean_metrics": {"spearman_rho": rho, "f1_at_k": 0.5},
                "std_metrics": {"spearman_rho": 0.05},
            }
            for sid, rho in scenario_rhos.items()
        ]
    }


def test_build_comparison_table_paired_delta_replaces_pooled_mean_diff():
    """The scenario this guards against: pooled-mean deltas can mask a
    consistent per-scenario direction, or manufacture one that isn't real.
    Here hgl_qos beats topo_qos on every single scenario by exactly 0.1 —
    the paired mean_delta must reflect that exactly, matching the pooled
    difference in this case since every scenario is offset identically."""
    results_by_variant = {
        "hgl_qos": _fake_variant_result({"atm_system": 0.6, "av_system": 0.7}),
        "topo_qos": _fake_variant_result({"atm_system": 0.5, "av_system": 0.6}),
    }
    table = _build_comparison_table(results_by_variant)

    assert table["hgl_qos"]["ci95_rho"][0] is not None
    paired = table["hgl_qos"]["paired_delta_vs_baseline"]["topo_qos"]
    assert paired["mean_delta"] == pytest.approx(0.1)
    assert paired["n_paired_scenarios"] == 2
    assert table["hgl_qos"]["delta_vs_best_baseline"] == pytest.approx(0.1)


def test_build_comparison_table_single_scenario_atm_only():
    """The actual ATM-scoped run shape: one scenario per variant."""
    results_by_variant = {
        "hgl_qos": _fake_variant_result({"atm_system": 0.567}),
        "topo_qos": _fake_variant_result({"atm_system": 0.609}),
    }
    table = _build_comparison_table(results_by_variant)

    paired = table["hgl_qos"]["paired_delta_vs_baseline"]["topo_qos"]
    assert paired["n_paired_scenarios"] == 1
    assert paired["mean_delta"] == pytest.approx(0.567 - 0.609, abs=1e-4)
    # Single-point CI must degenerate to (value, value), not error or lie
    # about certainty it doesn't have.
    assert paired["ci_95"][0] == pytest.approx(paired["ci_95"][1])
