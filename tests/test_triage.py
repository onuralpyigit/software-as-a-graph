"""
tests/test_triage.py

Tests for the Triage bridge (saag/analysis/triage.py): Pathway B's Top-K
ranking joined to Pathway A's RM root-cause diagnosis for the same
component ids.
"""
import dataclasses
import inspect
import logging
from types import SimpleNamespace

import pytest

from saag.core.metrics import (
    ComponentQuality, QualityScores, QualityLevels, StructuralMetrics
)
from saag.core.criticality import CriticalityLevel
from saag.analysis.analyzer import CriticalityProfile
from saag.analysis.models import QualityAnalysisResult
from saag.analysis import triage as triage_mod
from saag.analysis.triage import triage, select_top_k
from saag.explanation.engine import ComponentExplanation, resolve_roles


# ── Fixtures: a small RM population mirroring test_explanation_engine.py ───

@pytest.fixture
def total_hub():
    """CRITICAL, 'Total Hub' pattern -- mirrors test_explanation_engine.py."""
    scores = QualityScores(
        reliability=0.9, maintainability=0.85, fault_tolerance=0.91, availability=0.88, overall=0.9
    )
    levels = QualityLevels(
        reliability=CriticalityLevel.CRITICAL, maintainability=CriticalityLevel.CRITICAL,
        fault_tolerance=CriticalityLevel.CRITICAL, availability=CriticalityLevel.CRITICAL,
        overall=CriticalityLevel.CRITICAL,
    )
    structural = StructuralMetrics(
        id="App_Controller", name="Controller", type="Application",
        in_degree_raw=14, out_degree_raw=5, reverse_pagerank=0.87, betweenness=0.6,
        ap_c_directed=1.0, bridge_ratio=0.75,
    )
    profile = CriticalityProfile(ft_crit=True, a_crit=True, m_crit=True, r_crit=True, q_crit=True)
    return ComponentQuality(id="App_Controller", type="Application", scores=scores, levels=levels,
                             structural=structural, profile=profile)


@pytest.fixture
def spof():
    """HIGH, 'SPOF' pattern -- mirrors test_explanation_engine.py's sample_quality_high."""
    scores = QualityScores(
        reliability=0.75, maintainability=0.7, fault_tolerance=0.68, availability=0.72, overall=0.75
    )
    levels = QualityLevels(
        reliability=CriticalityLevel.HIGH, maintainability=CriticalityLevel.HIGH,
        fault_tolerance=CriticalityLevel.HIGH, availability=CriticalityLevel.HIGH,
        overall=CriticalityLevel.HIGH,
    )
    structural = StructuralMetrics(
        id="Aux_Service", name="Aux", type="Infrastructure",
        in_degree_raw=3, out_degree_raw=2, reverse_pagerank=0.3, betweenness=0.1,
        ap_c_directed=0.2, bridge_ratio=0.1,
    )
    profile = CriticalityProfile(ft_crit=False, a_crit=True, m_crit=False, r_crit=False, q_crit=False)
    return ComponentQuality(id="Aux_Service", type="Infrastructure", scores=scores, levels=levels,
                             structural=structural, profile=profile)


@pytest.fixture
def unremarkable():
    """LOW, no elevated dimension, no profile -- 'Composite Risk' default."""
    scores = QualityScores(reliability=0.2, maintainability=0.2, overall=0.2)
    levels = QualityLevels(overall=CriticalityLevel.LOW)
    structural = StructuralMetrics(id="Leaf_Service", name="Leaf", type="Application")
    return ComponentQuality(id="Leaf_Service", type="Application", scores=scores, levels=levels,
                             structural=structural, profile=None)


@pytest.fixture
def rm_result(total_hub, spof, unremarkable):
    return QualityAnalysisResult(
        timestamp="2026-03-24T00:00:00Z",
        layer="system",
        context="test",
        components=[total_hub, spof, unremarkable],
        edges=[],
        classification_summary=None,
        problems=[],
        prediction_mode="rm",
    )


class _FakeGNNResult:
    """Duck-typed stand-in for GNNAnalysisResult -- only the surface
    ``triage()`` reads: ``.components``, ``.prediction_mode``, ``.rm_result``."""
    def __init__(self, components, rm_result):
        self.components = components
        self.prediction_mode = "gnn_only"
        self.rm_result = rm_result


# ── select_top_k ────────────────────────────────────────────────────────────

def test_select_top_k_orders_by_overall_desc_with_id_tiebreak():
    tied_a = SimpleNamespace(id="A", type="Application", scores=SimpleNamespace(overall=0.5))
    tied_b = SimpleNamespace(id="B", type="Application", scores=SimpleNamespace(overall=0.5))
    top = SimpleNamespace(id="C", type="Application", scores=SimpleNamespace(overall=0.9))
    fake = SimpleNamespace(components=[tied_b, top, tied_a])

    result = select_top_k(fake, k=3)

    assert result == [("C", 0.9), ("A", 0.5), ("B", 0.5)]


def test_select_top_k_filters_by_node_types():
    app = SimpleNamespace(id="App", type="Application", scores=SimpleNamespace(overall=0.5))
    infra = SimpleNamespace(id="Infra", type="Infrastructure", scores=SimpleNamespace(overall=0.9))
    fake = SimpleNamespace(components=[app, infra])

    result = select_top_k(fake, k=5, node_types=["Infrastructure"])

    assert result == [("Infra", 0.9)]


@pytest.mark.parametrize("k", [0, -1])
def test_select_top_k_rejects_non_positive_k(k):
    fake = SimpleNamespace(components=[SimpleNamespace(id="A", type="Application",
                                                         scores=SimpleNamespace(overall=0.5))])
    with pytest.raises(ValueError):
        select_top_k(fake, k=k)


# ── triage(): cold start (RM-only, no GNN checkpoint) ───────────────────────

def test_triage_cold_start_ranking_source_is_rm(rm_result):
    result = triage(rm_result, k=2)
    assert result.ranking_source == "rm"
    assert result.population == 3


def test_triage_cold_start_entries_carry_rm_diagnosis(rm_result):
    result = triage(rm_result, k=2)

    ids = [e.component_id for e in result.entries]
    assert ids == ["App_Controller", "Aux_Service"]  # overall desc: 0.9, 0.75

    top = result.entries[0]
    assert top.rank == 1
    assert top.pattern == "Total Hub"
    assert top.level == "CRITICAL"
    assert set(top.roles) == {"SRE", "DevOps", "Architect"}

    second = result.entries[1]
    assert second.pattern == "SPOF"
    assert second.roles == ["DevOps"]


def test_triage_k_larger_than_population_returns_all(rm_result):
    result = triage(rm_result, k=100)
    assert len(result.entries) == 3
    assert result.population == 3


def test_triage_rejects_non_positive_k(rm_result):
    with pytest.raises(ValueError):
        triage(rm_result, k=0)


# ── triage(): GNN mode joins B's ranking to A's (RM) diagnosis ──────────────

def test_triage_gnn_ranking_overrides_rm_order_but_reuses_rm_diagnosis(rm_result):
    """The GNN ranking is deliberately the reverse of RM's, so a shortlist
    that matches RM's order would mean the join silently fell back to RM
    scores instead of the GNN's own ranking."""
    rm_components = rm_result.components
    gnn_components = [
        dataclasses.replace(rm_components[0], scores=dataclasses.replace(rm_components[0].scores, overall=0.1)),
        dataclasses.replace(rm_components[1], scores=dataclasses.replace(rm_components[1].scores, overall=0.4)),
        dataclasses.replace(rm_components[2], scores=dataclasses.replace(rm_components[2].scores, overall=0.99)),
    ]
    gnn_result = _FakeGNNResult(gnn_components, rm_result)

    result = triage(gnn_result, k=2)

    assert result.ranking_source == "gnn"
    ids = [e.component_id for e in result.entries]
    assert ids == ["Leaf_Service", "Aux_Service"]  # GNN's own order: 0.99, 0.4 (App_Controller 0.1 excluded)

    # Diagnosis still comes from the RM substrate's CriticalityProfile, not
    # the GNN shim (which would carry profile=None on a real GNNAnalysisResult).
    leaf_entry, aux_entry = result.entries
    assert leaf_entry.pattern == "Composite Risk"
    assert aux_entry.pattern == "SPOF"
    assert aux_entry.roles == ["DevOps"]


def test_triage_skips_component_absent_from_rm_substrate(rm_result, caplog):
    ghost = ComponentQuality(
        id="Ghost", type="Application",
        scores=QualityScores(overall=0.99),
        levels=QualityLevels(overall=CriticalityLevel.CRITICAL),
        structural=StructuralMetrics(id="Ghost", name="Ghost", type="Application"),
        profile=None,
    )
    gnn_result = _FakeGNNResult([ghost], rm_result)

    with caplog.at_level(logging.WARNING):
        result = triage(gnn_result, k=1)

    assert result.entries == []
    assert "Ghost" in caplog.text


# ── resolve_roles: branches not exercised by the fixtures above ────────────

def _explanation(pattern, dimensions=()):
    return ComponentExplanation(
        component_id="X", pattern=pattern, level="HIGH",
        one_line="", top_risk="", dimensions=list(dimensions),
        priority_action="", anti_patterns=[],
    )


def test_resolve_roles_fragile_hub_and_exposed_bottleneck():
    assert set(resolve_roles(_explanation("Fragile Hub"))) == {"SRE", "DevOps"}
    assert set(resolve_roles(_explanation("Exposed Bottleneck"))) == {"Architect", "Security"}


def test_resolve_roles_defaults_to_architect_when_nothing_matches():
    assert resolve_roles(_explanation("Composite Risk")) == ["Architect"]


# ── Independence: triage never imports the simulation oracle ───────────────

def test_triage_module_imports_no_simulation_symbol():
    src = inspect.getsource(triage_mod)
    assert "import saag.simulation" not in src
    assert "from saag.simulation" not in src
