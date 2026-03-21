"""
Unit tests for the Q*(v) composite metric improvements.

Covers:
- R*(v) v5: DG_in replaces w_in; RPR weight 0.45; r_w_in deprecated to 0.0
- CriticalityProfile: pattern lookup, to_dict(), defaults
- I*(v) composite ground truth: weighted sum of dimension ground truths
- System health metrics: SRI boundary conditions, RCI Gini formula
- ValidationTargets: composite fields present with correct defaults
"""
import math
import pytest

from src.prediction.weight_calculator import QualityWeights, AHPMatrices, AHPProcessor
from src.prediction.analyzer import CriticalityProfile
from src.validation.models import ValidationTargets, LayerValidationResult


# ===========================================================================
# R*(v) v5: Formula and Weight Verification
# ===========================================================================

class TestRStarV5:

    def test_default_r_reverse_pagerank_increased(self):
        """RPR weight now 0.45 (up from 0.40 in v4)."""
        w = QualityWeights()
        assert w.r_reverse_pagerank == pytest.approx(0.45, abs=0.01)

    def test_r_in_degree_active_at_030(self):
        """DG_in (r_in_degree) reinstated at 0.30."""
        w = QualityWeights()
        assert w.r_in_degree == pytest.approx(0.30, abs=0.01)

    def test_r_w_in_deprecated_to_zero(self):
        """w_in exclusively assigned to V*(v) as QADS; r_w_in=0.0 in R*(v)."""
        w = QualityWeights()
        assert w.r_w_in == 0.0, "r_w_in must be deprecated (0.0) in R*(v) v5"

    def test_r_dim_weights_sum_to_one(self):
        """Active R*(v) sub-weights must sum to 1.0."""
        w = QualityWeights()
        total = w.r_reverse_pagerank + w.r_in_degree + w.r_cdpot
        assert total == pytest.approx(1.0, abs=0.02), (
            f"R*(v) active weights sum={total:.4f}, expected ≈1.0"
        )

    def test_ahp_reliability_matrix_is_3x3(self):
        """AHP reliability matrix is still 3×3 (RPR, DG_in, CDPot)."""
        m = AHPMatrices()
        assert len(m.criteria_reliability) == 3
        assert all(len(row) == 3 for row in m.criteria_reliability)

    def test_ahp_computed_rpr_is_highest(self):
        """AHP-derived RPR weight should exceed the other two terms."""
        proc = AHPProcessor()
        w = proc.compute_weights()
        assert w.r_reverse_pagerank > w.r_in_degree, "RPR should outweigh DG_in"
        assert w.r_reverse_pagerank > w.r_cdpot,    "RPR should outweigh CDPot"
        assert w.r_w_in == 0.0, "r_w_in must remain 0.0 after AHP compute"

    def test_ahp_overall_matrix_not_balanced(self):
        """AHP overall matrix is no longer all-1.0; A should receive the highest weight."""
        proc = AHPProcessor()
        w = proc.compute_weights()
        # A > R > M > V according to the theoretically motivated matrix
        assert w.q_availability > w.q_reliability, "Availability should outweigh Reliability"
        assert w.q_reliability  > w.q_maintainability, "Reliability should outweigh Maintainability"

    def test_ahp_overall_weights_sum_to_one(self):
        """Q*(v) overall weights must sum to ~1.0."""
        proc = AHPProcessor()
        w = proc.compute_weights()
        total = w.q_reliability + w.q_maintainability + w.q_availability + w.q_vulnerability
        assert total == pytest.approx(1.0, abs=0.02)


# ===========================================================================
# CriticalityProfile
# ===========================================================================

class TestCriticalityProfile:

    def test_default_all_false(self):
        """Default CriticalityProfile has all flags False."""
        p = CriticalityProfile()
        assert not p.r_crit
        assert not p.m_crit
        assert not p.a_crit
        assert not p.v_crit
        assert not p.q_crit

    def test_default_pattern_is_composite_risk(self):
        """Default profile (all False) is 'Composite Risk'."""
        assert CriticalityProfile().pattern == "Composite Risk"

    def test_total_hub_pattern(self):
        """All four dimension flags → Total Hub."""
        p = CriticalityProfile(r_crit=True, m_crit=True, a_crit=True, v_crit=True)
        assert p.pattern == "Total Hub"

    def test_reliability_hub_pattern(self):
        """Only R flag → Reliability Hub."""
        p = CriticalityProfile(r_crit=True)
        assert p.pattern == "Reliability Hub"

    def test_bottleneck_pattern(self):
        """Only M flag → Bottleneck."""
        p = CriticalityProfile(m_crit=True)
        assert p.pattern == "Bottleneck"

    def test_spof_pattern(self):
        """Only A flag → SPOF."""
        p = CriticalityProfile(a_crit=True)
        assert p.pattern == "SPOF"

    def test_attack_target_pattern(self):
        """Only V flag → Attack Target."""
        p = CriticalityProfile(v_crit=True)
        assert p.pattern == "Attack Target"

    def test_fragile_hub_pattern(self):
        """R+A flags → Fragile Hub."""
        p = CriticalityProfile(r_crit=True, a_crit=True)
        assert p.pattern == "Fragile Hub"

    def test_exposed_bottleneck_pattern(self):
        """M+V flags → Exposed Bottleneck."""
        p = CriticalityProfile(m_crit=True, v_crit=True)
        assert p.pattern == "Exposed Bottleneck"

    def test_q_crit_independent(self):
        """q_crit is independent — composite outlier with no single dominant dimension."""
        p = CriticalityProfile(q_crit=True)
        assert p.q_crit
        assert p.pattern == "Composite Risk"  # no RMAV flags set

    def test_to_dict_keys(self):
        """to_dict must include all five flags and pattern."""
        d = CriticalityProfile(r_crit=True, q_crit=True).to_dict()
        assert set(d.keys()) == {"r_crit", "m_crit", "a_crit", "v_crit", "q_crit", "pattern"}
        assert d["r_crit"] is True
        assert d["m_crit"] is False
        assert d["pattern"] == "Reliability Hub"


# ===========================================================================
# I*(v) Composite Ground Truth
# ===========================================================================

class TestIStarComposite:
    """White-box tests for the I*(v) = 0.25×IR + 0.25×IM + 0.25×IA + 0.25×IV formula."""

    @staticmethod
    def _compute_i_star(ir, im, ia, iv, weights=None):
        w = weights or dict(r=0.25, m=0.25, a=0.25, v=0.25)
        return w["r"] * ir + w["m"] * im + w["a"] * ia + w["v"] * iv

    def test_equal_weights_mean(self):
        """Equal weights: I*(v) is the arithmetic mean of the four sub-scores."""
        val = self._compute_i_star(0.4, 0.6, 0.8, 0.2)
        assert val == pytest.approx((0.4 + 0.6 + 0.8 + 0.2) / 4, abs=1e-9)

    def test_all_zero_gives_zero(self):
        assert self._compute_i_star(0, 0, 0, 0) == pytest.approx(0.0)

    def test_all_one_gives_one(self):
        assert self._compute_i_star(1, 1, 1, 1) == pytest.approx(1.0)

    def test_single_dominant_dimension(self):
        """When only one dimension is 1.0, I*(v) = 0.25 with equal weights."""
        assert self._compute_i_star(1.0, 0, 0, 0) == pytest.approx(0.25, abs=1e-9)

    def test_custom_weights_sum_to_one(self):
        """Custom weights that sum to 1.0 must produce a value in [0,1]."""
        w = dict(r=0.45, m=0.30, a=0.15, v=0.10)
        val = self._compute_i_star(0.3, 0.5, 0.7, 0.2, weights=w)
        assert 0.0 <= val <= 1.0

    def test_custom_weights_correctly_weight_dominant_dim(self):
        """Higher weight on IR should increase I*(v) when IR is high."""
        equal = self._compute_i_star(1.0, 0, 0, 0, weights=dict(r=0.25, m=0.25, a=0.25, v=0.25))
        skewed = self._compute_i_star(1.0, 0, 0, 0, weights=dict(r=0.50, m=0.20, a=0.20, v=0.10))
        assert skewed > equal


# ===========================================================================
# System Health: SRI and RCI
# ===========================================================================

class TestSystemHealth:

    @staticmethod
    def _sri(h_r, h_m, h_a, h_v, w=None):
        """SRI = Σ w_d × (1 − H_d)."""
        w = w or dict(r=0.25, m=0.25, a=0.25, v=0.25)
        return w["r"] * (1 - h_r) + w["m"] * (1 - h_m) + w["a"] * (1 - h_a) + w["v"] * (1 - h_v)

    @staticmethod
    def _gini(scores):
        """Gini coefficient of a list of scores."""
        n = len(scores)
        if n == 0:
            return 0.0
        q_sorted = sorted(scores)
        gini_sum = sum((2 * (i + 1) - n - 1) * q_sorted[i] for i in range(n))
        return abs(gini_sum) / (n * sum(q_sorted)) if sum(q_sorted) > 0 else 0.0

    def test_sri_perfect_health_is_zero(self):
        """H_d = 1.0 for all dims → SRI = 0."""
        assert self._sri(1, 1, 1, 1) == pytest.approx(0.0)

    def test_sri_maximum_risk_is_one(self):
        """H_d = 0.0 for all dims → SRI = 1."""
        assert self._sri(0, 0, 0, 0) == pytest.approx(1.0)

    def test_sri_partial_risk(self):
        """Partial health: SRI = weighted risk contribution."""
        sri = self._sri(h_r=0.5, h_m=1.0, h_a=0.5, h_v=1.0)
        assert sri == pytest.approx(0.25, abs=1e-9)

    def test_sri_in_unit_interval(self):
        """SRI must always be in [0, 1]."""
        import random
        rng = random.Random(99)
        for _ in range(50):
            vals = [rng.random() for _ in range(4)]
            sri = self._sri(*vals)
            assert 0.0 <= sri <= 1.0 + 1e-9

    def test_rci_uniform_is_zero(self):
        """Equal scores → Gini = 0 → RCI = 0."""
        assert self._gini([0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0, abs=1e-9)

    def test_rci_single_dominant_approaches_one(self):
        """All risk in one node → Gini approaches 1."""
        scores = [0.0, 0.0, 0.0, 1.0]
        gini = self._gini(scores)
        assert gini > 0.5, f"Expected high Gini for concentrated risk, got {gini:.4f}"

    def test_rci_in_unit_interval(self):
        """Gini must be in [0, 1]."""
        import random
        rng = random.Random(42)
        for _ in range(50):
            scores = [rng.random() for _ in range(10)]
            gini = self._gini(scores)
            assert 0.0 <= gini <= 1.0 + 1e-9


# ===========================================================================
# ValidationTargets: Composite Fields
# ===========================================================================

class TestCompositeValidationTargets:

    def test_composite_spearman_present_and_correct(self):
        """composite_spearman target must be 0.85."""
        t = ValidationTargets()
        assert hasattr(t, "composite_spearman")
        assert t.composite_spearman == pytest.approx(0.85, abs=0.01)

    def test_composite_f1_present_and_correct(self):
        """composite_f1 target must be 0.90."""
        t = ValidationTargets()
        assert hasattr(t, "composite_f1")
        assert t.composite_f1 == pytest.approx(0.90, abs=0.01)

    def test_composite_top5_overlap_raised(self):
        """composite_top5_overlap target must be 0.80 (raised from 0.60)."""
        t = ValidationTargets()
        assert hasattr(t, "composite_top5_overlap")
        assert t.composite_top5_overlap >= 0.80

    def test_predictive_gain_present(self):
        """predictive_gain target must be 0.03."""
        t = ValidationTargets()
        assert hasattr(t, "predictive_gain")
        assert t.predictive_gain == pytest.approx(0.03, abs=0.005)

    def test_max_interdim_correlation_present(self):
        """max_interdim_correlation must be 0.70."""
        t = ValidationTargets()
        assert hasattr(t, "max_interdim_correlation")
        assert t.max_interdim_correlation == pytest.approx(0.40, abs=0.01)

    def test_layer_result_has_composite_spearman(self):
        """LayerValidationResult must expose composite_spearman and predictive_gain."""
        lr = LayerValidationResult(layer="app", layer_name="Application")
        assert hasattr(lr, "composite_spearman")
        assert hasattr(lr, "predictive_gain")
        assert hasattr(lr, "system_health")
        assert isinstance(lr.system_health, dict)

    def test_layer_result_to_dict_includes_composite(self):
        """to_dict() summary must contain composite_spearman, predictive_gain, system_health."""
        lr = LayerValidationResult(
            layer="app", layer_name="Application",
            composite_spearman=0.87, predictive_gain=0.05,
            system_health={"SRI": 0.42, "RCI": 0.31},
        )
        d = lr.to_dict()
        summary = d["summary"]
        assert "composite_spearman" in summary
        assert "predictive_gain"    in summary
        assert "system_health"      in summary
        assert summary["composite_spearman"] == pytest.approx(0.87, abs=0.0001)
