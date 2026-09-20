"""
test_loso_delta_comparator.py
─────────────────────────────
Pins the comparator behind Table 4's Δρ column.

The column read "Δρ vs best baseline" and was computed as `max` over every other
row — including the learned ones. On the shipped 12-fold artifact that picked
GAT-N-QoS and printed HGT-QoS's margin as +0.0346, with no interval, under a
heading that says "baseline". The pre-registered contrast is against Topo-QoS,
where the same gap is +0.0851 with a 95% CI of [-0.029, +0.194] — an interval
that includes zero. The table and the test that licenses it must not use
different reference points, so both ends are pinned here:

  * the artifact's own Δ is paired by fold against ``topo_qos``;
  * the rendered cell carries the interval from the significance artifact.
"""

import pytest

from reproduce.loso_all_variants import PREREGISTERED_BASELINE, _build_comparison_table
from reproduce.loso_significance import BASELINE as SIGNIFICANCE_BASELINE
from reproduce.render_table import _delta_cell, _loso_contrasts


def _fold(holdout: str, rho: float) -> dict:
    return {"holdout_id": holdout, "mean_metrics": {"spearman_rho": rho, "f1_at_k": 0.4},
            "std_metrics": {"spearman_rho": 0.01}}


@pytest.fixture
def results():
    """Three variants where "best other row" and "the baseline" disagree.

    ``hgl_qos`` (0.60) trails ``gl_qos`` (0.70) and leads ``topo_qos`` (0.50),
    so a max-over-rows comparator reports -0.10 where the pre-registered one
    reports +0.10. That sign flip is the defect.
    """
    def variant(a, b):
        return {"folds": [_fold("atm_system", a), _fold("av_system", b)],
                "summary": {"eval_population": "application"}, "per_type_summary": {}}
    return {"topo_qos": variant(0.4, 0.6),
            "gl_qos":   variant(0.6, 0.8),
            "hgl_qos":  variant(0.5, 0.7)}


def test_delta_is_measured_against_the_preregistered_baseline(results):
    table = _build_comparison_table(results, eval_population="application")
    assert table["hgl_qos"]["delta_baseline"] == PREREGISTERED_BASELINE
    assert table["hgl_qos"]["delta_vs_baseline"] == pytest.approx(0.10, abs=1e-9)
    # Not -0.10, which is what comparing against the best-scoring other row gives.
    assert table["gl_qos"]["delta_vs_baseline"] == pytest.approx(0.20, abs=1e-9)


def test_the_baseline_row_has_no_delta_against_itself(results):
    table = _build_comparison_table(results, eval_population="application")
    assert "delta_vs_baseline" not in table[PREREGISTERED_BASELINE]


def test_delta_is_paired_by_fold_not_a_difference_of_means(results):
    """A variant missing a fold differences the folds it shares, not unequal sets."""
    results["hgl_qos"]["folds"] = [_fold("av_system", 0.7)]  # atm dropped
    table = _build_comparison_table(results, eval_population="application")
    assert table["hgl_qos"]["delta_n_folds"] == 1
    # Paired on av_system alone: 0.7 - 0.6. Differencing the means would give
    # 0.70 - 0.50 = +0.20, an artifact of the missing fold.
    assert table["hgl_qos"]["delta_vs_baseline"] == pytest.approx(0.10, abs=1e-9)


def test_table_and_significance_test_share_one_comparator():
    """One name, two scripts. Drifting them apart is how the defect got in."""
    assert PREREGISTERED_BASELINE == SIGNIFICANCE_BASELINE


def test_rendered_cell_carries_the_interval():
    sig = {"preregistered": [{"variant": "hgl_qos", "baseline": "topo_qos",
                              "mean_delta": 0.0851, "delta_ci95": [-0.029, 0.194],
                              "role": "primary"}]}
    contrasts = _loso_contrasts(sig)
    cell = _delta_cell("hgl_qos", {}, contrasts, latex=False)
    assert cell == "+0.0851 [-0.029, +0.194]"
    assert _delta_cell("hgl_qos", {}, contrasts, latex=True) == \
        "+0.0851 $[-0.029, +0.194]$"


def test_rendered_cell_falls_back_to_the_artifact_delta():
    """Without a significance artifact the column shows the point estimate, not nothing."""
    cell = _delta_cell("hgl_qos", {"delta_vs_baseline": 0.0851}, {}, latex=False)
    assert cell == "+0.0851"


def test_rendered_baseline_row_is_blank():
    contrasts = _loso_contrasts({"preregistered": [
        {"variant": "topo_qos", "baseline": "topo_qos", "mean_delta": 0.0}]})
    assert _delta_cell("topo_qos", {"delta_vs_baseline": 0.0}, contrasts, latex=False) == "—"


def test_contrasts_ignore_rows_measured_against_something_else():
    """The significance artifact also carries factorial contrasts with no baseline."""
    sig = {"preregistered": [{"variant": "hgl_qos", "baseline": "gl_qos",
                              "mean_delta": 0.5, "delta_ci95": [0.4, 0.6]}],
           "factorial": [{"quantity": "interaction", "mean_delta": -0.199}]}
    assert _loso_contrasts(sig) == {}
