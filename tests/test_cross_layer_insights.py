"""
Tests for cross-layer insight derivation.

compute_cross_layer_insights was extracted from AnalysisService._compute_cross_layer_insights;
these tests pin the behaviour that extraction had to preserve.
"""

import pytest

from saag.analysis.cross_layer import CONCENTRATION_THRESHOLD, compute_cross_layer_insights
from saag.analysis.models import LayerAnalysisResult
from saag.core.criticality import CriticalityLevel


class _Struct:
    def __init__(self, name, is_ap=False):
        self.name = name
        self.is_articulation_point = is_ap


class _Levels:
    def __init__(self, overall):
        self.overall = overall


class _Comp:
    def __init__(self, cid, overall=CriticalityLevel.LOW, is_ap=False, name=None):
        self.id = cid
        self.levels = _Levels(overall)
        self.structural = _Struct(name or f"{cid}-name", is_ap)


def _layer(name, components):
    """Build a LayerAnalysisResult carrying only what the insight code reads."""
    quality = type("Q", (), {"components": components})()
    return LayerAnalysisResult(
        layer=name, layer_name=name, description="", structural=None, quality=quality,
    )


def test_no_insights_for_single_layer_components():
    results = {
        "app": _layer("app", [_Comp("a", CriticalityLevel.CRITICAL)]),
        "infra": _layer("infra", [_Comp("b", CriticalityLevel.CRITICAL)]),
    }
    # Each id appears in exactly one layer — no cross-layer signal.
    assert [i for i in compute_cross_layer_insights(results)
            if i.insight_type != "layer_concentration"] == []


def test_compound_critical_requires_two_layers():
    results = {
        "app": _layer("app", [_Comp("a", CriticalityLevel.CRITICAL, name="Router")]),
        "system": _layer("system", [_Comp("a", CriticalityLevel.HIGH, name="Router")]),
    }
    compound = [i for i in compute_cross_layer_insights(results)
                if i.insight_type == "compound_critical"]

    assert len(compound) == 1
    assert compound[0].component_id == "a"
    assert compound[0].csc_name == "Router"
    assert compound[0].layers_affected == ["app", "system"]
    # CRITICAL in any contributing layer escalates the insight severity
    assert compound[0].severity == "CRITICAL"


def test_compound_critical_is_high_when_no_layer_is_critical():
    results = {
        "app": _layer("app", [_Comp("a", CriticalityLevel.HIGH)]),
        "system": _layer("system", [_Comp("a", CriticalityLevel.HIGH)]),
    }
    compound = [i for i in compute_cross_layer_insights(results)
                if i.insight_type == "compound_critical"]

    assert len(compound) == 1
    assert compound[0].severity == "HIGH"


def test_medium_in_both_layers_produces_nothing():
    results = {
        "app": _layer("app", [_Comp("a", CriticalityLevel.MEDIUM)]),
        "system": _layer("system", [_Comp("a", CriticalityLevel.MEDIUM)]),
    }
    assert compute_cross_layer_insights(results) == []


def test_systemic_spof_requires_two_layers():
    results = {
        "infra": _layer("infra", [_Comp("n1", is_ap=True, name="NodeA")]),
        "mw": _layer("mw", [_Comp("n1", is_ap=True, name="NodeA")]),
        "system": _layer("system", [_Comp("n1", is_ap=False, name="NodeA")]),
    }
    spof = [i for i in compute_cross_layer_insights(results)
            if i.insight_type == "systemic_spof"]

    assert len(spof) == 1
    assert spof[0].severity == "CRITICAL"
    # Only the layers where it is actually an articulation point are listed
    assert spof[0].layers_affected == ["infra", "mw"]


def test_layer_concentration_threshold():
    """Fires strictly above the threshold, not at it."""
    # 2/4 = 0.50 > 0.30 → fires
    over = _layer("mw", [
        _Comp(f"c{i}", CriticalityLevel.CRITICAL if i < 2 else CriticalityLevel.LOW)
        for i in range(4)
    ])
    # 1/4 = 0.25 <= 0.30 → does not fire
    under = _layer("app", [
        _Comp(f"d{i}", CriticalityLevel.CRITICAL if i < 1 else CriticalityLevel.LOW)
        for i in range(4)
    ])

    insights = compute_cross_layer_insights({"mw": over, "app": under})
    concentration = [i for i in insights if i.insight_type == "layer_concentration"]

    assert [i.layers_affected for i in concentration] == [["mw"]]
    assert concentration[0].severity == "HIGH"
    assert concentration[0].component_id == ""
    assert "50%" in concentration[0].description
    assert CONCENTRATION_THRESHOLD == 0.30


def test_empty_layer_is_skipped():
    assert compute_cross_layer_insights({"app": _layer("app", [])}) == []


def test_sorted_critical_first_then_by_layer_count():
    """CRITICAL before HIGH; within a severity, more affected layers first."""
    results = {
        "app": _layer("app", [
            _Comp("spof", is_ap=True),
            _Comp("hi", CriticalityLevel.HIGH),
        ]),
        "mw": _layer("mw", [
            _Comp("spof", is_ap=True),
            _Comp("hi", CriticalityLevel.HIGH),
        ]),
    }
    insights = compute_cross_layer_insights(results)
    severities = [i.severity for i in insights]

    assert severities == sorted(severities, key={"CRITICAL": 0, "HIGH": 1}.get)
    assert insights[0].insight_type == "systemic_spof"
