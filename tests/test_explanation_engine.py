"""
tests/test_explanation_engine.py

Tests for the ExplanationEngine module.
"""
import pytest
from saag.core.metrics import (
    ComponentQuality, QualityScores, QualityLevels, StructuralMetrics
)
from saag.core.criticality import CriticalityLevel
from saag.prediction.analyzer import CriticalityProfile
from saag.prediction.models import QualityAnalysisResult, DetectedProblem
from saag.analysis.smells import AntiPatternReport
from saag.explanation.engine import ExplanationEngine


@pytest.fixture
def sample_quality():
    """Create a sample ComponentQuality object with 'Total Hub' profile."""
    scores = QualityScores(
        reliability=0.9, maintainability=0.85, availability=0.88, vulnerability=0.92, overall=0.9
    )
    levels = QualityLevels(
        reliability=CriticalityLevel.CRITICAL,
        maintainability=CriticalityLevel.CRITICAL,
        availability=CriticalityLevel.CRITICAL,
        vulnerability=CriticalityLevel.CRITICAL,
        overall=CriticalityLevel.CRITICAL
    )
    structural = StructuralMetrics(
        id="App_Controller",
        name="Controller",
        type="Application",
        in_degree_raw=14,
        out_degree_raw=5,
        reverse_pagerank=0.87,
        betweenness=0.6,
        ap_c_directed=1.0,
        reverse_eigenvector=0.95,
        bridge_ratio=0.75
    )
    profile = CriticalityProfile(r_crit=True, m_crit=True, a_crit=True, v_crit=True, q_crit=True)
    
    return ComponentQuality(
        id="App_Controller",
        type="Application",
        scores=scores,
        levels=levels,
        structural=structural,
        profile=profile
    )


@pytest.fixture
def sample_quality_high():
    """Create a sample ComponentQuality object with HIGH level."""
    scores = QualityScores(
        reliability=0.75, maintainability=0.7, availability=0.72, vulnerability=0.78, overall=0.75
    )
    levels = QualityLevels(
        reliability=CriticalityLevel.HIGH,
        maintainability=CriticalityLevel.HIGH,
        availability=CriticalityLevel.HIGH,
        vulnerability=CriticalityLevel.HIGH,
        overall=CriticalityLevel.HIGH
    )
    structural = StructuralMetrics(
        id="Aux_Service",
        name="Aux",
        type="Infrastructure",
        in_degree_raw=3,
        out_degree_raw=2,
        reverse_pagerank=0.3,
        betweenness=0.1,
        ap_c_directed=0.2,
        reverse_eigenvector=0.4,
        bridge_ratio=0.1
    )
    profile = CriticalityProfile(r_crit=False, m_crit=False, a_crit=True, v_crit=False, q_crit=False)
    
    return ComponentQuality(
        id="Aux_Service",
        type="Infrastructure",
        scores=scores,
        levels=levels,
        structural=structural,
        profile=profile
    )


@pytest.fixture
def sample_smells():
    return [
        DetectedProblem(
            entity_id="App_Controller",
            entity_type="Component",
            category="Coupling",
            severity="CRITICAL",
            name="CyclicDependency",
            description="Cycle detected",
            recommendation="Break cycle"
        )
    ]


@pytest.fixture
def sample_analysis_result(sample_quality, sample_quality_high):
    return QualityAnalysisResult(
        timestamp="2026-03-24T00:00:00Z",
        layer="system",
        context="test_context",
        components=[sample_quality, sample_quality_high],
        edges=[],
        classification_summary=None
    )


@pytest.fixture
def sample_smell_report(sample_smells):
    return AntiPatternReport(
        problems=sample_smells,
        summary={"total": 1}
    )


def test_explain_component(sample_quality, sample_smells):
    engine = ExplanationEngine()
    explanation = engine.explain_component(sample_quality, sample_smells)

    assert explanation.component_id == "App_Controller"
    assert explanation.pattern == "Total Hub"
    assert explanation.level == "CRITICAL"
    assert "App_Controller is a critical hub" in explanation.one_line
    assert "three independent failure modes" in explanation.top_risk
    assert "CyclicDependency" in explanation.anti_patterns
    
    # Check dimensions
    assert len(explanation.dimensions) == 4
    
    # Check interpolation in plain meaning (using default templates now)
    rel_dim = next((d for d in explanation.dimensions if d.dimension == "Reliability"), None)
    assert rel_dim is not None
    assert "14" in rel_dim.plain_meaning
    assert "Primary driver" in rel_dim.plain_meaning
    
    maint_dim = next((d for d in explanation.dimensions if d.dimension == "Maintainability"), None)
    assert maint_dim is not None
    assert "5" in maint_dim.plain_meaning
    assert "Primary driver" in maint_dim.plain_meaning


def test_explain_system(sample_analysis_result, sample_smell_report):
    engine = ExplanationEngine()
    system_report = engine.explain_system(sample_analysis_result, sample_smell_report)
    
    assert system_report.total_components == 2
    assert system_report.critical_count == 1
    assert system_report.high_count == 1
    assert system_report.deployment_blocked is True
    assert "App_Controller" in system_report.top_risk_summary
    
    assert len(system_report.component_explanations) == 2
    
    # Check remediation steps
    assert len(system_report.remediation_plan) > 0
    assert "App_Controller" in system_report.remediation_plan[0].components

    assert len(system_report.by_stakeholder["DevOps"]) == 2
    assert "URGENT" in system_report.by_stakeholder["DevOps"][0]
    assert "STRATEGIC" in system_report.by_stakeholder["DevOps"][1]
