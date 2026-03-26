"""
Unit tests for the Availability dimension improvements (A(v) v3).

Covers:
  - QualityWeights: new a_qspof, a_ap_c_directed, a_cdi, a_qos_weight fields present
  - AHPMatrices: criteria_availability is 5×5
  - AHPProcessor: positive weights with correct ordering (AP primary)
  - QSPOF(v) formula correctness
  - AP_c_directed = max(AP_c_out, AP_c_in)
  - CDI normalisation
  - ImpactMetrics: IA(v) default fields, availability_impact property arithmetic, to_dict()
  - calculate_spof_f1: perfect, zero, partial coverage
  - calculate_hsrr: pairs recovered vs not recovered
  - calculate_rri: true-negative rate arithmetic
"""
import pytest
from src.prediction.weight_calculator import QualityWeights, AHPMatrices, AHPProcessor
from src.prediction.analyzer import QualityAnalyzer
from src.analysis.structural_analyzer import StructuralAnalyzer
from src.simulation.models import ImpactMetrics
from src.validation.metric_calculator import (
    calculate_spof_f1,
    calculate_hsrr,
    calculate_rri,
)


# ===========================================================================
# QualityWeights: new v2 A fields present, deprecated fields zero
# ===========================================================================

class TestQualityWeightsV3A:

    def test_new_a_fields_are_present_with_defaults(self):
        """A(v) v3 fields must exist and be positive by default."""
        w = QualityWeights()
        assert hasattr(w, 'a_qspof'),        "a_qspof missing"
        assert hasattr(w, 'a_ap_c_directed'), "a_ap_c_directed missing"
        assert hasattr(w, 'a_cdi'),           "a_cdi missing"
        assert hasattr(w, 'a_qos_weight'),    "a_qos_weight missing"
        assert w.a_qspof > 0.0,        "a_qspof should be positive"
        assert w.a_ap_c_directed > 0.0, "a_ap_c_directed should be positive"
        assert w.a_cdi > 0.0,           "a_cdi should be positive"
        assert w.a_qos_weight > 0.0,    "a_qos_weight should be positive"

    def test_legacy_a_fields_are_removed(self):
        """a_articulation and a_importance should be removed."""
        w = QualityWeights()
        assert not hasattr(w, 'a_articulation'), "a_articulation should be removed"
        assert not hasattr(w, 'a_importance'),   "a_importance should be removed"

    def test_default_a_weights_sum_to_one(self):
        """Active A(v) v3 weights must sum to 1.0."""
        w = QualityWeights()
        total = (
            w.a_qspof + 
            w.a_bridge_ratio + 
            w.a_ap_c_directed + 
            w.a_cdi + 
            w.a_qos_weight
        )
        assert total == pytest.approx(1.0, abs=0.1), (
            f"Active A(v) v3 weights should sum ~1.0, got {total:.4f}"
        )


# ===========================================================================
# AHPMatrices: criteria_availability is now 4×4
# ===========================================================================

class TestAHPAvailabilityMatrix:

    def test_availability_matrix_is_5x5(self):
        """criteria_availability must be 5×5 (AP_c_d, QSPOF, BR, CDI, w)."""
        m = AHPMatrices()
        assert len(m.criteria_availability) == 5, (
            f"Expected 5×5, got {len(m.criteria_availability)} rows"
        )
        assert all(len(row) == 5 for row in m.criteria_availability), (
            "All rows in criteria_availability must have 5 columns"
        )

    def test_ahp_computed_a_weights_sum_near_one(self):
        """AHP-computed A(v) v3 weights should be positive and sum ~1.0."""
        proc = AHPProcessor()
        w = proc.compute_weights()
        active = (
            w.a_qspof + 
            w.a_bridge_ratio + 
            w.a_ap_c_directed + 
            w.a_cdi + 
            w.a_qos_weight
        )
        assert active == pytest.approx(1.0, abs=0.05), (
            f"Computed A(v) weights should sum ~1.0, got {active:.4f}"
        )
        assert w.a_qspof > 0.0
        assert w.a_bridge_ratio > 0.0
        assert w.a_ap_c_directed > 0.0
        assert w.a_cdi > 0.0
        assert w.a_qos_weight > 0.0

    def test_ahp_weight_ordering_ap_c_directed_largest(self):
        """AP_c_directed should have the largest weight (baseline)."""
        proc = AHPProcessor()
        w = proc.compute_weights()
        assert w.a_ap_c_directed >= w.a_qspof, (
            f"AP_c_directed ({w.a_ap_c_directed:.4f}) should be >= QSPOF ({w.a_qspof:.4f})"
        )
        assert w.a_qos_weight <= w.a_cdi + 0.05, (
            f"w ({w.a_qos_weight:.4f}) should be the smallest"
        )


# ===========================================================================
# QSPOF formula: QSPOF(v) = AP_c_directed(v) × w(v)
# ===========================================================================

class TestQSPOFFormula:

    @staticmethod
    def _qspof(ap_c_dir: float, qw: float) -> float:
        return ap_c_dir * qw

    def test_zero_ap_gives_zero_qspof(self):
        """Non-articulation point → QSPOF = 0 regardless of QoS weight."""
        assert self._qspof(0.0, 1.0) == pytest.approx(0.0)

    def test_zero_weight_gives_zero_qspof(self):
        """Zero QoS weight → QSPOF = 0 regardless of AP score."""
        assert self._qspof(0.8, 0.0) == pytest.approx(0.0)

    def test_full_ap_full_weight_gives_one(self):
        """Full AP and full QoS weight → QSPOF = 1.0."""
        assert self._qspof(1.0, 1.0) == pytest.approx(1.0)

    def test_partial_values(self):
        """Partial AP and QoS weight multiply correctly."""
        assert self._qspof(0.5, 0.6) == pytest.approx(0.30, abs=1e-9)

    def test_qspof_in_range(self):
        """QSPOF must always be in [0, 1] for valid inputs."""
        for ap in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for qw in [0.0, 0.25, 0.5, 0.75, 1.0]:
                q = self._qspof(ap, qw)
                assert 0.0 <= q <= 1.0 + 1e-9, f"QSPOF out of range: {q}"


# ===========================================================================
# AP_c_directed = max(AP_c_out, AP_c_in)
# ===========================================================================

class TestAPcDirectedFormula:

    @staticmethod
    def _ap_c_directed(out: float, inp: float) -> float:
        return max(out, inp)

    def test_both_zero(self):
        assert self._ap_c_directed(0.0, 0.0) == 0.0

    def test_out_dominates(self):
        assert self._ap_c_directed(0.8, 0.3) == pytest.approx(0.8)

    def test_in_dominates(self):
        assert self._ap_c_directed(0.2, 0.9) == pytest.approx(0.9)

    def test_equal_values(self):
        assert self._ap_c_directed(0.5, 0.5) == pytest.approx(0.5)

    def test_result_in_range(self):
        for a in [0.0, 0.1, 0.5, 1.0]:
            for b in [0.0, 0.1, 0.5, 1.0]:
                result = self._ap_c_directed(a, b)
                assert 0.0 <= result <= 1.0


# ===========================================================================
# Property Tests: Availability via Analyzer Pipeline
# ===========================================================================

class TestAvailabilityProperty:
    """
    Property tests for Availability: A(v) = f(QSPOF, BR, AP_c, CDI).
    Verifies that the QualityAnalyzer correctly applies the formula and that 
    the ap_graph fixture yields expected directional results.
    """

    def test_articulation_point_availability_risk(self, ap_graph):
        """
        In an ap_graph (A-B-C, B-D), B is an articulation point.
        B should have a higher Availability score than others because 
        it is a Single Point of Failure (QSPOF > 0).
        """
        # 1. Structural Analysis
        struct_analyzer = StructuralAnalyzer()
        struct_res = struct_analyzer.analyze(ap_graph)

        # 2. Quality Analysis
        quality_analyzer = QualityAnalyzer(normalization_method="max")
        quality_res = quality_analyzer.analyze(struct_res)

        # 3. Property Asserts
        comp_map = {c.id: c for c in quality_res.components}
        b_quality = comp_map["B"]
        a_b = b_quality.scores.availability

        leaf_scores = [
            comp.scores.availability 
            for cid, comp in comp_map.items() 
            if cid != "B"
        ]

        # B should have higher availability risk (A score) than any leaf.
        assert all(a_b > a_leaf for a_leaf in leaf_scores), (
            f"Articulation Point B availability ({a_b:.4f}) should be higher than leaf nodes {leaf_scores}"
        )
        # B should have non-zero QSPOF contributing to it
        assert b_quality.structural.ap_c_directed > 0.0
        assert a_b > 0.1  # Sanity check it's non-negligible


# ===========================================================================
# ImpactMetrics: IA(v) fields and availability_impact property
# ===========================================================================

class TestImpactMetricsIAv:

    def test_default_ia_fields_are_zero(self):
        """New IA(v) fields should default to 0.0 (backward compat)."""
        im = ImpactMetrics()
        assert im.weighted_reachability_loss == 0.0
        assert im.weighted_fragmentation == 0.0
        assert im.path_breaking_throughput_loss == 0.0

    def test_availability_impact_zero_when_defaults(self):
        """availability_impact should be 0.0 with all-zero sub-fields."""
        assert ImpactMetrics().availability_impact == pytest.approx(0.0)

    def test_availability_impact_formula(self):
        """IA(v) = 0.50*WRL + 0.35*WFrag + 0.15*PBTL."""
        im = ImpactMetrics(
            weighted_reachability_loss=1.0,
            weighted_fragmentation=1.0,
            path_breaking_throughput_loss=1.0,
        )
        expected = 0.50 * 1.0 + 0.35 * 1.0 + 0.15 * 1.0
        assert im.availability_impact == pytest.approx(expected, abs=1e-6)

    def test_availability_impact_partial_values(self):
        """Partial sub-fields give correct weighted sum."""
        im = ImpactMetrics(
            weighted_reachability_loss=0.6,
            weighted_fragmentation=0.4,
            path_breaking_throughput_loss=0.2,
        )
        expected = 0.50 * 0.6 + 0.35 * 0.4 + 0.15 * 0.2
        assert im.availability_impact == pytest.approx(expected, abs=1e-6)

    def test_to_dict_includes_availability_section(self):
        """to_dict() should include an 'availability' section with IA(v) fields."""
        im = ImpactMetrics(
            weighted_reachability_loss=0.5,
            weighted_fragmentation=0.3,
            path_breaking_throughput_loss=0.1,
        )
        d = im.to_dict()
        assert "availability" in d, "'availability' key missing from to_dict()"
        av = d["availability"]
        assert "weighted_reachability_loss"    in av
        assert "weighted_fragmentation"        in av
        assert "path_breaking_throughput_loss" in av
        assert "availability_impact"           in av

    def test_reliability_impact_unchanged(self):
        """Existing IR(v) reliability_impact should still be correct."""
        im = ImpactMetrics(
            cascade_reach=0.5,
            weighted_cascade_impact=0.3,
            normalized_cascade_depth=0.8,
        )
        expected = 0.45 * 0.5 + 0.35 * 0.3 + 0.20 * 0.8
        assert im.reliability_impact == pytest.approx(expected, abs=1e-6)

    def test_maintainability_impact_unchanged(self):
        """Existing IM(v) maintainability_impact should still be correct."""
        im = ImpactMetrics(
            change_reach=0.5,
            weighted_change_impact=0.3,
            normalized_change_depth=0.8,
        )
        expected = 0.45 * 0.5 + 0.35 * 0.3 + 0.20 * 0.8
        assert im.maintainability_impact == pytest.approx(expected, abs=1e-6)


# ===========================================================================
# calculate_spof_f1
# ===========================================================================

class TestCalculateSPOFF1:

    def test_perfect_coverage(self):
        """All predicted SPOFs are actual SPOFs → F1 = 1.0."""
        ap  = {"a": 0.8, "b": 0.7, "c": 0.0}  # a,b predicted SPOF; c not
        ia  = {"a": 0.9, "b": 0.8, "c": 0.1}  # a,b actual SPOF; c not
        res = calculate_spof_f1(ap, ia, ap_threshold=0.0, ia_threshold=0.5)
        assert res["f1"] == pytest.approx(1.0)
        assert res["precision"] == pytest.approx(1.0)
        assert res["recall"] == pytest.approx(1.0)

    def test_zero_coverage(self):
        """No predicted SPOFs match actual SPOFs → F1 = 0.0."""
        ap  = {"a": 0.0, "b": 0.0, "c": 0.8}  # c predicted SPOF
        ia  = {"a": 0.9, "b": 0.8, "c": 0.1}  # a,b actual SPOF; c not
        res = calculate_spof_f1(ap, ia, ap_threshold=0.0, ia_threshold=0.5)
        assert res["f1"] == pytest.approx(0.0, abs=0.01)

    def test_partial_coverage(self):
        """Partial overlap → F1 between 0 and 1."""
        # a: TP; b: FP; c: FN
        ap  = {"a": 0.8, "b": 0.7, "c": 0.0}
        ia  = {"a": 0.9, "b": 0.1, "c": 0.8}
        res = calculate_spof_f1(ap, ia, ia_threshold=0.5)
        assert 0.0 < res["f1"] < 1.0
        # precision = 1/(1+1) = 0.5; recall = 1/(1+1) = 0.5; F1 = 0.5
        assert res["f1"] == pytest.approx(0.5, abs=0.01)

    def test_empty_inputs_safe(self):
        """Empty inputs → all metrics 0.0, no error."""
        res = calculate_spof_f1({}, {})
        assert res["f1"] == 0.0
        assert res["precision"] == 0.0
        assert res["recall"] == 0.0

    def test_result_in_range(self):
        """All metric values should be in [0, 1]."""
        import random
        rng = random.Random(42)
        keys = [f"c{i}" for i in range(20)]
        ap = {k: rng.random() for k in keys}
        ia = {k: rng.random() for k in keys}
        res = calculate_spof_f1(ap, ia)
        for metric_name, val in res.items():
            assert 0.0 <= val <= 1.0, f"{metric_name} out of range: {val}"


# ===========================================================================
# calculate_hsrr
# ===========================================================================

class TestCalculateHSRR:

    def test_all_recovered(self):
        """All hidden SPOFs (AP_c=0, IA>0.5) have QSPOF > 0 → HSRR = 1.0."""
        qspof = {"a": 0.7, "b": 0.8}
        ia    = {"a": 0.9, "b": 0.9}
        ap_c  = {"a": 0.0, "b": 0.0}
        assert calculate_hsrr(qspof, ia, ap_c) == pytest.approx(1.0)

    def test_none_recovered(self):
        """No hidden SPOFs have QSPOF > 0 → HSRR = 0.0."""
        qspof = {"a": 0.0, "b": 0.0}
        ia    = {"a": 0.9, "b": 0.9}
        ap_c  = {"a": 0.0, "b": 0.0}
        assert calculate_hsrr(qspof, ia, ap_c) == pytest.approx(0.0)

    def test_partial_recovered(self):
        """Half hidden SPOFs recovered → HSRR = 0.5."""
        qspof = {"a": 0.7, "b": 0.0}
        ia    = {"a": 0.9, "b": 0.9}
        ap_c  = {"a": 0.0, "b": 0.0}
        assert calculate_hsrr(qspof, ia, ap_c) == pytest.approx(0.5)

    def test_empty_inputs_returns_zero(self):
        assert calculate_hsrr({}, {}, {}) == 0.0

    def test_missing_components_safe(self):
        qspof = {"a": 0.9}
        ia    = {"x": 0.9} # x is hidden SPOF but not in qspof dict
        ap_c  = {"x": 0.0}
        assert calculate_hsrr(qspof, ia, ap_c) == pytest.approx(0.0)


# ===========================================================================
# calculate_rri
# ===========================================================================

class TestCalculateRRI:

    def test_perfect_rri(self):
        """All non-bridge components (BR=0) have IA < 0.30 → RRI = 1.0."""
        ia     = {"r1": 0.1, "r2": 0.2}
        br     = {"r1": 0.0, "r2": 0.0}
        assert calculate_rri(ia, br, ia_threshold=0.30) == pytest.approx(1.0)

    def test_zero_rri(self):
        """All non-bridge components have high IA → RRI = 0.0."""
        ia     = {"r1": 0.9, "r2": 0.8}
        br     = {"r1": 0.0, "r2": 0.0}
        assert calculate_rri(ia, br, ia_threshold=0.30) == pytest.approx(0.0)

    def test_partial_rri(self):
        """Half non-bridge components have low IA → RRI = 0.5."""
        ia     = {"r1": 0.1, "r2": 0.9}
        br     = {"r1": 0.0, "r2": 0.0}
        assert calculate_rri(ia, br, ia_threshold=0.30) == pytest.approx(0.5)

    def test_no_redundant_components(self):
        """No components have BR=0 → RRI = 0.0 (degenerate)."""
        ia     = {"a": 0.1}
        br     = {"a": 0.5}
        assert calculate_rri(ia, br, ia_threshold=0.30) == pytest.approx(0.0)

    def test_empty_inputs_safe(self):
        """Empty inputs → RRI = 0.0, no error."""
        assert calculate_rri({}, {}) == 0.0

    def test_rri_in_range(self):
        """RRI should always be in [0, 1]."""
        import random
        rng = random.Random(42)
        keys = [f"c{i}" for i in range(20)]
        av   = {k: rng.random() for k in keys}
        ia   = {k: rng.random() for k in keys}
        br   = {k: 0.0 if rng.random() < 0.5 else rng.random() for k in keys}
        rri  = calculate_rri(ia, br)
        assert 0.0 <= rri <= 1.0 + 1e-9, f"RRI out of range: {rri}"
