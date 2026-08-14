"""
Tests for saag/core/metric_registry.py — the StructuralMetrics role declaration.

Covers:
  - Every StructuralMetrics data field has a registry entry (no silent drift
    between the dataclass and the declared role map).
  - The registry declares no field that doesn't exist on StructuralMetrics.
  - SCORING_METRICS matches the terms actually read by
    QualityAnalyzer._compute_rm — 19 fields (see analyzer.py and
    structural_analyzer.py's _compute_code_quality_metrics).
"""
import dataclasses

from saag.core.metrics import StructuralMetrics
from saag.core.metric_registry import (
    METRIC_ROLES,
    MetricRole,
    SCORING_METRICS,
    role_of,
    metrics_with_role,
)


def _structural_metric_fields():
    """Data fields on StructuralMetrics, excluding identifiers (id/name/type)."""
    return {
        f.name for f in dataclasses.fields(StructuralMetrics)
    } - {"id", "name", "type"}


class TestRegistryCompleteness:
    def test_every_structural_metric_has_a_role(self):
        missing = _structural_metric_fields() - set(METRIC_ROLES)
        assert not missing, f"StructuralMetrics fields with no registry entry: {missing}"

    def test_no_registry_entry_names_a_nonexistent_field(self):
        extra = set(METRIC_ROLES) - _structural_metric_fields()
        assert not extra, f"Registry entries with no matching StructuralMetrics field: {extra}"

    def test_every_entry_declares_at_least_one_role(self):
        empty = [m for m, roles in METRIC_ROLES.items() if not roles]
        assert not empty, f"Metrics with an empty role set: {empty}"


class TestScoringMetrics:
    def test_scoring_metrics_count(self):
        # The number a reader should be able to trust as "how many raw metrics
        # feed Q(v)" — see structural-analysis.md §10, which derives its table
        # from this constant.
        assert len(SCORING_METRICS) == 19

    def test_scoring_metrics_match_compute_rm_terms(self):
        expected = {
            "reverse_pagerank", "betweenness", "in_degree_raw", "out_degree_raw",
            "clustering_coefficient", "weight", "ap_c_directed", "cdi", "mpci",
            "fan_out_criticality", "dependency_weight_in", "dependency_weight_out",
            "path_complexity", "bridge_ratio",
            # CQP inputs (code_quality_penalty is itself a fifth M(v) term)
            "loc_norm", "complexity_norm", "instability_code", "lcom_norm",
            "code_quality_penalty",
        }
        assert SCORING_METRICS == frozenset(expected)


class TestRoleLookups:
    def test_role_of_unknown_metric_is_empty(self):
        assert role_of("not_a_real_metric") == frozenset()

    def test_metrics_with_role_scoring_matches_constant(self):
        assert metrics_with_role(MetricRole.SCORING) == SCORING_METRICS

    def test_reverse_closeness_and_reverse_eigenvector_are_gone(self):
        # Regression guard: these were retired (dead V(v)/REV/RCL residue) —
        # they must not reappear in either StructuralMetrics or the registry.
        assert "reverse_closeness" not in METRIC_ROLES
        assert "reverse_eigenvector" not in METRIC_ROLES
        assert "reverse_closeness" not in _structural_metric_fields()
        assert "reverse_eigenvector" not in _structural_metric_fields()
