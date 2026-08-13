"""
Tests for saag/core/quality_model.py — the layered SQuaRE quality model.

Covers:
  - QIU_PROJECTION is row-stochastic (the property that makes Layer 3 a
    reweighting rather than an independent prediction stage).
  - derive_rmav_weights sums to 1.0 for every declared domain, and falls back
    to the static default for an unknown/absent domain.
  - The equivalence this whole design rests on: scoring an RMAV vector with
    Mᵀω as composite weights is identical to projecting onto quality-in-use
    harm and scalarising with ω.
  - QualityAnalyzer(domain_weights=None) — the default — is byte-identical to
    QualityAnalyzer() on a real component set.
  - QualityAnalyzer(domain_weights=...) is opt-in, traceable via
    QualityAnalysisResult.domain_weighting, and composes with QoS adaptation
    in a fixed, documented order.
"""
import pytest

from saag.core import GraphData, ComponentData, EdgeData
from saag.analysis.structural_analyzer import StructuralAnalyzer
from saag.analysis.analyzer import QualityAnalyzer
from saag.analysis.weight_calculator import QualityWeights
from saag.core.quality_model import (
    QIU_PROJECTION,
    QIU_ORDER,
    RMAV_ORDER,
    DOMAIN_PRIORITIES,
    Provenance,
    Layer,
    LAYER_SPEC,
    qiu_harm,
    effective_rmav_weights,
    derive_rmav_weights,
)


# ===========================================================================
# The algebra: QIU_PROJECTION is row-stochastic
# ===========================================================================

class TestProjectionIsRowStochastic:

    def test_every_row_sums_to_one(self):
        """Each ISO/IEC 25019 characteristic's row must sum to 1.0 — this is
        exactly the property that collapses Layer 3 into a reweighting of
        RMAV rather than an independent score."""
        for row in QIU_PROJECTION:
            assert sum(row) == pytest.approx(1.0)

    def test_shape_matches_rmav_and_qiu_order(self):
        assert len(QIU_PROJECTION) == len(QIU_ORDER) == 3
        for row in QIU_PROJECTION:
            assert len(row) == len(RMAV_ORDER) == 4


class TestDomainPriorities:

    @pytest.mark.parametrize("domain", list(DOMAIN_PRIORITIES.keys()))
    def test_every_declared_domain_sums_to_one(self, domain):
        omega = DOMAIN_PRIORITIES[domain]
        assert len(omega) == 3
        assert sum(omega) == pytest.approx(1.0)

    def test_ros2_autonomous_vehicle_prioritises_risk(self):
        """docs/criticality.md §3.3: omega_Risk >> omega_Ben for ROS 2 / AV."""
        omega = DOMAIN_PRIORITIES["autoware_ros2"]
        ben, risk, acc = omega
        assert risk > ben
        assert risk > acc

    def test_atm_prioritises_beneficialness_over_risk(self):
        """docs/criticality.md §3.3: omega_Ben > omega_Risk for ATM/aviation."""
        ben, risk, acc = DOMAIN_PRIORITIES["air_traffic_management"]
        assert ben > risk

    def test_hub_and_spoke_is_deliberately_absent(self):
        """Names a topology class, not a deployment domain — must take the
        documented fallback rather than an invented priority."""
        assert "hub-and-spoke" not in DOMAIN_PRIORITIES


class TestEffectiveRmavWeights:

    def test_sums_to_one_for_any_valid_omega(self):
        for omega in DOMAIN_PRIORITIES.values():
            weights = effective_rmav_weights(omega)
            assert sum(weights) == pytest.approx(1.0)

    def test_matches_hand_computed_atm_example(self):
        """Pin the documented worked example: ATM omega=(0.50,0.30,0.20) ->
        R=0.265 M=0.125 A=0.390 V=0.220."""
        r, m, a, v = effective_rmav_weights((0.50, 0.30, 0.20))
        assert r == pytest.approx(0.265)
        assert m == pytest.approx(0.125)
        assert a == pytest.approx(0.390)
        assert v == pytest.approx(0.220)


class TestDeriveRmavWeights:

    def test_known_domain_is_derived(self):
        weights, derived = derive_rmav_weights("healthcare")
        assert derived is True
        assert weights.q_reliability + weights.q_maintainability + weights.q_availability + weights.q_security == pytest.approx(1.0)

    def test_unknown_domain_falls_back_to_base_unchanged(self):
        base = QualityWeights()
        weights, derived = derive_rmav_weights("not_a_real_domain", base=base)
        assert derived is False
        assert weights is base

    def test_none_domain_falls_back(self):
        base = QualityWeights()
        weights, derived = derive_rmav_weights(None, base=base)
        assert derived is False
        assert weights is base

    def test_domain_lookup_is_case_and_whitespace_insensitive(self):
        _, derived_a = derive_rmav_weights("Healthcare")
        _, derived_b = derive_rmav_weights("  healthcare  ")
        assert derived_a is True
        assert derived_b is True

    def test_derivation_does_not_mutate_shared_base(self):
        """Same discipline as equal_weights: never mutate a caller-owned
        QualityWeights instance in place."""
        shared = QualityWeights()
        original = shared.q_availability
        derive_rmav_weights("autoware_ros2", base=shared)
        assert shared.q_availability == pytest.approx(original)


# ===========================================================================
# The equivalence the whole design rests on
# ===========================================================================

class TestQiuCollapseEquivalence:

    def test_scalarised_qiu_harm_equals_direct_reweighting(self):
        """Projecting an RMAV vector onto quality-in-use harm and scalarising
        with omega must equal scoring the same RMAV vector with M^T omega as
        composite weights, for an arbitrary score vector and every declared
        domain. This is the finding the whole Layer 3 design rests on —
        pinned so it cannot silently stop being true."""
        scores = (0.7, 0.2, 0.9, 0.4)  # (R, M, A, V)
        for omega in DOMAIN_PRIORITIES.values():
            harm = qiu_harm(scores)
            via_projection = sum(
                omega[i] * harm[name] for i, name in enumerate(QIU_ORDER)
            )

            weights = effective_rmav_weights(omega)
            via_reweighting = sum(w * s for w, s in zip(weights, scores))

            assert via_projection == pytest.approx(via_reweighting)


class TestLayerSpec:

    def test_only_external_layer_has_an_oracle(self):
        for layer, spec in LAYER_SPEC.items():
            if layer is Layer.EXTERNAL:
                assert spec["oracle"] is not None
            else:
                assert spec["oracle"] is None

    def test_provenance_escalates_to_declared_at_qiu_weighting(self):
        assert LAYER_SPEC[Layer.QME]["provenance"] is Provenance.MEASURED
        assert LAYER_SPEC[Layer.INTERNAL]["provenance"] is Provenance.DERIVED
        assert LAYER_SPEC[Layer.EXTERNAL]["provenance"] is Provenance.DERIVED
        assert LAYER_SPEC[Layer.QIU_WEIGHTING]["provenance"] is Provenance.DECLARED


# ===========================================================================
# QualityAnalyzer wiring: opt-in, traceable, order-preserving
# ===========================================================================

class TestQualityAnalyzerDomainWeighting:

    @pytest.fixture
    def graph(self):
        """A small graph with a structural SPOF, enough for RMAV scores to
        differ meaningfully across weightings."""
        return GraphData(
            components=[
                ComponentData(id="X", component_type="Application"),
                ComponentData(id="Y", component_type="Application"),
                ComponentData(id="Z", component_type="Application"),
                ComponentData(id="W", component_type="Application"),
            ],
            edges=[
                EdgeData("Y", "X", "Application", "Application", "app_to_app", "DEPENDS_ON", 1.0),
                EdgeData("W", "X", "Application", "Application", "app_to_app", "DEPENDS_ON", 1.0),
                EdgeData("X", "Z", "Application", "Application", "app_to_app", "DEPENDS_ON", 1.0),
            ],
        )

    def test_default_is_byte_identical_to_no_flag(self, graph):
        """QualityAnalyzer(domain_weights=None) — the default — must produce
        exactly today's scores. No published number may move unless the flag
        is passed explicitly."""
        struct = StructuralAnalyzer().analyze(graph)

        baseline = QualityAnalyzer().analyze(struct)
        explicit_none = QualityAnalyzer(domain_weights=None).analyze(struct)

        base_scores = {c.id: c.scores.to_dict() for c in baseline.components}
        none_scores = {c.id: c.scores.to_dict() for c in explicit_none.components}
        assert base_scores == none_scores
        assert baseline.domain_weighting is None
        assert explicit_none.domain_weighting is None

    def test_known_domain_changes_weights_and_records_provenance(self, graph):
        struct = StructuralAnalyzer().analyze(graph)
        result = QualityAnalyzer(domain_weights="autoware_ros2", adapt_qos_weights=False).analyze(struct)

        assert result.domain_weighting == {
            "domain": "autoware_ros2",
            "derived": True,
            "provenance": Provenance.DECLARED.value,
        }
        assert result.weights.q_reliability + result.weights.q_maintainability + result.weights.q_availability + result.weights.q_security == pytest.approx(1.0)

    def test_unknown_domain_records_fallback_not_derived(self, graph):
        struct = StructuralAnalyzer().analyze(graph)
        result = QualityAnalyzer(domain_weights="atlantis", adapt_qos_weights=False).analyze(struct)

        assert result.domain_weighting == {
            "domain": "atlantis",
            "derived": False,
            "provenance": Provenance.DECLARED.value,
        }
        # Falls back to the static defaults, not to an invented vector.
        assert result.weights.q_availability == pytest.approx(QualityWeights().q_availability)

    def test_domain_weighting_leaves_original_weights_object_untouched(self, graph):
        """The instance's self.weights must be restored after analyze(), same
        discipline as the existing QoS-adaptation code."""
        struct = StructuralAnalyzer().analyze(graph)
        analyzer = QualityAnalyzer(domain_weights="healthcare")
        original = analyzer.weights

        analyzer.analyze(struct)

        assert analyzer.weights is original
