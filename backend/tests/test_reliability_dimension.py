"""
Unit tests for the Reliability dimension improvements (v4).

Covers:
- CDPot(v) formula correctness
- R(v) v4 formula uses r_w_in / r_cdpot and not r_pagerank / r_in_degree
- IR(v): reliability_impact property arithmetic
- ImpactMetrics: cascade_reach, weighted_cascade_impact, normalized_cascade_depth
- CCR@K metric function
- CME metric function
"""
import pytest
from src.analysis.weight_calculator import QualityWeights, AHPMatrices, AHPProcessor
from src.simulation.models import ImpactMetrics
from src.validation.metric_calculator import calculate_ccr_at_k, calculate_cme


# ===========================================================================
# QualityWeights: new fields present, deprecated fields zero
# ===========================================================================

class TestQualityWeightsV4:

    def test_default_weights_have_r_w_in_and_r_cdpot(self):
        """QualityWeights must expose r_w_in and r_cdpot with sensible defaults."""
        w = QualityWeights()
        assert hasattr(w, 'r_w_in'), "r_w_in field missing from QualityWeights"
        assert hasattr(w, 'r_cdpot'), "r_cdpot field missing from QualityWeights"
        assert w.r_w_in > 0.0, "r_w_in should be positive"
        assert w.r_cdpot > 0.0, "r_cdpot should be positive"

    def test_deprecated_r_pagerank_and_r_in_degree_are_zero(self):
        """r_pagerank and r_in_degree must be 0.0 (deprecated in v4)."""
        w = QualityWeights()
        assert w.r_pagerank == 0.0, "r_pagerank should be deprecated (0.0) in v4"
        assert w.r_in_degree == 0.0, "r_in_degree should be deprecated (0.0) in v4"

    def test_ahp_computed_weights_are_positive_and_sum_near_active_terms(self):
        """AHP-computed reliability weights should be positive and roughly sum to 1."""
        proc = AHPProcessor()
        w = proc.compute_weights()
        active = w.r_reverse_pagerank + w.r_w_in + w.r_cdpot
        assert active == pytest.approx(1.0, abs=0.05), (
            f"Active R(v) weights should sum ~1.0, got {active:.4f}"
        )
        assert w.r_reverse_pagerank > 0.0
        assert w.r_w_in > 0.0
        assert w.r_cdpot > 0.0

    def test_ahp_reliability_matrix_is_3x3(self):
        """Reliability AHP matrix should be 3×3 (RPR, w_in, CDPot)."""
        m = AHPMatrices()
        assert len(m.criteria_reliability) == 3
        assert all(len(row) == 3 for row in m.criteria_reliability)


# ===========================================================================
# CDPot formula (computed inline in quality_analyzer._compute_rmav)
# Tested via the formula itself rather than a public function.
# ===========================================================================

class TestCDPotFormula:
    """
    CDPot(v) = ((RPR + DG_in) / 2) * (1 - min(DG_out / DG_in, 1))

    Properties:
      - Fan-out node (DG_out >> DG_in): CDPot → 0  (wide but shallow cascade)
      - Absorber node (DG_in >> DG_out): CDPot high  (narrow but deep cascade)
      - Equal in/out: CDPot = 0 (depth penalty = 1, cancelled)
    """

    @staticmethod
    def _cdpot(rpr: float, id_n: float, od_n: float) -> float:
        _denom = max(id_n, 1e-9)
        reach = (rpr + id_n) / 2.0
        depth = 1.0 - min(od_n / _denom, 1.0)
        return reach * depth

    def test_pure_absorber_node(self):
        """High in-degree, zero out-degree → high CDPot."""
        cdpot = self._cdpot(rpr=0.8, id_n=0.9, od_n=0.0)
        assert cdpot > 0.5, f"Absorber should have high CDPot, got {cdpot:.4f}"

    def test_pure_fanout_node(self):
        """Low in-degree, very high out-degree → CDPot ≈ 0."""
        cdpot = self._cdpot(rpr=0.6, id_n=0.1, od_n=1.0)
        assert cdpot == pytest.approx(0.0, abs=0.05), (
            f"Fan-out should have CDPot≈0, got {cdpot:.4f}"
        )

    def test_equal_in_out(self):
        """Equal normalised in/out-degree → depth penalty = 0 → CDPot = 0."""
        cdpot = self._cdpot(rpr=0.5, id_n=0.5, od_n=0.5)
        assert cdpot == pytest.approx(0.0, abs=1e-9)

    def test_zero_in_degree_safe(self):
        """Zero in-degree should not raise ZeroDivisionError."""
        cdpot = self._cdpot(rpr=0.0, id_n=0.0, od_n=0.5)
        assert cdpot == pytest.approx(0.0, abs=0.05)

    def test_output_in_range(self):
        """CDPot should always be in [0, 1]."""
        from itertools import product
        for rpr, id_n, od_n in product([0.0, 0.3, 0.7, 1.0], repeat=3):
            c = self._cdpot(rpr, id_n, od_n)
            assert 0.0 <= c <= 1.0 + 1e-9, f"CDPot out of range: {c} for ({rpr},{id_n},{od_n})"


# ===========================================================================
# ImpactMetrics: IR(v) fields
# ===========================================================================

class TestImpactMetricsIRv:

    def test_default_ir_fields_are_zero(self):
        """New IR(v) fields should default to 0.0 (backward compat)."""
        im = ImpactMetrics()
        assert im.cascade_reach == 0.0
        assert im.weighted_cascade_impact == 0.0
        assert im.normalized_cascade_depth == 0.0

    def test_reliability_impact_zero_when_defaults(self):
        """reliability_impact property should be 0.0 with all default sub-fields."""
        assert ImpactMetrics().reliability_impact == pytest.approx(0.0)

    def test_reliability_impact_formula(self):
        """IR(v) = 0.45*CR + 0.35*WCI + 0.20*ND."""
        im = ImpactMetrics(
            cascade_reach=1.0,
            weighted_cascade_impact=1.0,
            normalized_cascade_depth=1.0,
        )
        expected = 0.45 * 1.0 + 0.35 * 1.0 + 0.20 * 1.0
        assert im.reliability_impact == pytest.approx(expected, abs=1e-6)

    def test_reliability_impact_partial(self):
        """Partial field set gives correct weighted sum."""
        im = ImpactMetrics(
            cascade_reach=0.5,
            weighted_cascade_impact=0.3,
            normalized_cascade_depth=0.8,
        )
        expected = 0.45 * 0.5 + 0.35 * 0.3 + 0.20 * 0.8
        assert im.reliability_impact == pytest.approx(expected, abs=1e-6)

    def test_to_dict_includes_reliability_key(self):
        """to_dict() should include a 'reliability' section with IR(v) fields."""
        im = ImpactMetrics(cascade_reach=0.4, weighted_cascade_impact=0.2)
        d = im.to_dict()
        assert "reliability" in d
        assert "cascade_reach" in d["reliability"]
        assert "reliability_impact" in d["reliability"]

    def test_composite_impact_unchanged(self):
        """Legacy composite_impact should still work correctly."""
        im = ImpactMetrics(
            reachability_loss=0.5,
            fragmentation=0.0,
            throughput_loss=0.0,
            flow_disruption=0.0,
        )
        assert im.composite_impact == pytest.approx(0.35 * 0.5, abs=1e-6)


# ===========================================================================
# CCR@K
# ===========================================================================

class TestCCRAtK:

    def test_perfect_overlap(self):
        """Same ordering → CCR@5 = 1.0."""
        scores = {f"c{i}": 10 - i for i in range(10)}
        assert calculate_ccr_at_k(scores, scores, k=5) == pytest.approx(1.0)

    def test_zero_overlap(self):
        """Completely reversed top-K → CCR@5 = 0.0 (for K <= n/2)."""
        pred  = {"a": 1.0, "b": 0.9, "c": 0.8, "d": 0.2, "e": 0.1}
        actual = {"d": 1.0, "e": 0.9, "a": 0.3, "b": 0.2, "c": 0.1}
        # pred top-2: {a,b}; actual top-2: {d,e} — no overlap
        ccr = calculate_ccr_at_k(pred, actual, k=2)
        assert ccr == pytest.approx(0.0, abs=1e-9)

    def test_partial_overlap(self):
        """Partial top-K overlap → CCR@K = |intersection| / K."""
        pred  = {"a": 1.0, "b": 0.9, "c": 0.8, "d": 0.7, "e": 0.1}
        actual = {"a": 1.0, "c": 0.9, "f": 0.8, "g": 0.7, "b": 0.1}
        # Top-4 pred: {a,b,c,d}; Top-4 actual: {a,c,f,g}; common: {a,c} → 2/4
        ccr = calculate_ccr_at_k(pred, actual, k=4)
        assert ccr == pytest.approx(2 / 4, abs=1e-9)

    def test_empty_inputs(self):
        """Empty inputs should return 0.0 safely."""
        assert calculate_ccr_at_k({}, {}) == 0.0
        assert calculate_ccr_at_k({"a": 1.0}, {}) == 0.0

    def test_k_larger_than_n(self):
        """k > |components| should still return a valid value."""
        scores = {"a": 0.9, "b": 0.5}
        result = calculate_ccr_at_k(scores, scores, k=10)
        assert 0.0 <= result <= 1.0


# ===========================================================================
# CME
# ===========================================================================

class TestCME:

    def test_perfect_ranking_gives_zero_cme(self):
        """Identical predicted and actual → CME = 0.0."""
        scores = {f"c{i}": 10 - i for i in range(10)}
        assert calculate_cme(scores, scores) == pytest.approx(0.0, abs=1e-9)

    def test_fully_reversed_gives_max_cme(self):
        """Reversed ranking → maximum CME."""
        pred   = {"a": 1.0, "b": 0.67, "c": 0.33}
        actual = {"a": 0.0, "b": 0.33, "c": 1.0}
        cme = calculate_cme(pred, actual)
        # rank_R: a=1, b=2, c=3; rank_IR: c=1, b=2, a=3
        # |1-3| + |2-2| + |3-1| = 2+0+2 = 4; mean = 4/3; normalised by n=3: (4/3)/3 ≈ 0.444
        assert cme > 0.3, f"Fully reversed ranking should give high CME, got {cme:.4f}"

    def test_cme_range(self):
        """CME should always be in [0, 1]."""
        import random
        rng = random.Random(42)
        keys = [f"c{i}" for i in range(20)]
        pred   = {k: rng.random() for k in keys}
        actual = {k: rng.random() for k in keys}
        cme = calculate_cme(pred, actual)
        assert 0.0 <= cme <= 1.0, f"CME out of range: {cme}"

    def test_insufficient_data(self):
        """n < 2 should return 0.0 safely."""
        assert calculate_cme({}, {}) == 0.0
        assert calculate_cme({"a": 0.5}, {"a": 0.5}) == 0.0
