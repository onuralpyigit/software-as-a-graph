"""tests/test_referee_round8.py — statistics helpers of reproduce/referee_round8.py (Amendment 14)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import t as t_dist, ttest_1samp

from reproduce.referee_round8 import nb_corrected_t, tost_t, tost_wilcoxon, weighted_signflip


def test_tost_t_matches_two_one_sided_t_tests():
    d = np.array([0.01, -0.02, 0.03, 0.0, 0.015, -0.01, 0.02, 0.005])
    m = 0.05
    out = tost_t(d, m)
    p_lo = ttest_1samp(d, -m, alternative="greater").pvalue
    p_hi = ttest_1samp(d, m, alternative="less").pvalue
    assert out["p"] == pytest.approx(max(p_lo, p_hi), abs=1e-12)
    half = t_dist.ppf(0.95, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    assert out["ci90"] == pytest.approx([d.mean() - half, d.mean() + half])
    assert out["equivalence_bound"] == pytest.approx(max(abs(d.mean() - half), abs(d.mean() + half)))


def test_tost_rejects_a_large_difference():
    d = np.full(12, 0.2) + np.linspace(-0.01, 0.01, 12)
    assert tost_t(d, 0.05)["p"] > 0.5
    assert tost_wilcoxon(d, 0.05)["p"] > 0.5


def test_nadeau_bengio_reduces_to_paired_t_at_zero_ratio():
    rng = np.random.default_rng(1)
    d = rng.normal(0.05, 0.1, 12)
    assert nb_corrected_t(d, 0.0)["p"] == pytest.approx(ttest_1samp(d, 0.0).pvalue, abs=1e-12)
    assert nb_corrected_t(d, 1 / 11)["p"] > nb_corrected_t(d, 0.0)["p"]


def test_weighted_signflip_is_exact_on_a_small_case():
    d, w = [1.0, 1.0, 1.0, -1.0], [1.0, 1.0, 1.0, 1.0]
    out = weighted_signflip(d, w)
    assert out["delta_weighted"] == pytest.approx(0.5)
    # |sum of signed ones| >= 2 in 10 of 16 sign patterns.
    assert out["p_signflip"] == pytest.approx(10 / 16)
