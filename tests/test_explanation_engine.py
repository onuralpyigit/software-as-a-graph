"""
tests/test_explanation_engine.py

Tests for the ExplanationEngine module.
"""
import pytest
from saag.core.metrics import (
    ComponentQuality, QualityScores, QualityLevels, StructuralMetrics
)
from saag.core.criticality import CriticalityLevel
from saag.analysis.analyzer import CriticalityProfile
from saag.analysis.models import QualityAnalysisResult, DetectedProblem
from saag.analysis.smells import AntiPatternReport
from saag.explanation.engine import ExplanationEngine


@pytest.fixture
def sample_quality():
    """Create a sample ComponentQuality object with 'Total Hub' profile."""
    scores = QualityScores(
        reliability=0.9, maintainability=0.85, fault_tolerance=0.91, availability=0.88, overall=0.9
    )
    levels = QualityLevels(
        reliability=CriticalityLevel.CRITICAL,
        maintainability=CriticalityLevel.CRITICAL,
        fault_tolerance=CriticalityLevel.CRITICAL,
        availability=CriticalityLevel.CRITICAL,
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
        bridge_ratio=0.75
    )
    profile = CriticalityProfile(ft_crit=True, a_crit=True, m_crit=True, r_crit=True, q_crit=True)
    
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
        reliability=0.75, maintainability=0.7, fault_tolerance=0.68, availability=0.72, overall=0.75
    )
    levels = QualityLevels(
        reliability=CriticalityLevel.HIGH,
        maintainability=CriticalityLevel.HIGH,
        fault_tolerance=CriticalityLevel.HIGH,
        availability=CriticalityLevel.HIGH,
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
        bridge_ratio=0.1
    )
    profile = CriticalityProfile(ft_crit=False, a_crit=True, m_crit=False, r_crit=False, q_crit=False)
    
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
    
    # Check dimensions (reliability, fault_tolerance, maintainability, availability)
    assert len(explanation.dimensions) == 4
    assert {d.dimension for d in explanation.dimensions} == {
        "Reliability", "Fault Tolerance", "Maintainability", "Availability"
    }
    assert set(explanation.to_dict()["stakeholders"]) == {"SRE", "DevOps", "Architect"}
    
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


def test_identify_driver_uses_weighted_contributions_not_raw_scale():
    """identify_driver must not be hijacked by large raw dependency_weight_out
    when weighted dimension_contributions show CQP is the dominant driver."""
    from saag.explanation.engine import identify_driver

    cq = ComponentQuality(
        id="App_Complex",
        type="Application",
        scores=QualityScores(maintainability=0.85, overall=0.7),
        levels=QualityLevels(maintainability=CriticalityLevel.CRITICAL, overall=CriticalityLevel.HIGH),
        structural=StructuralMetrics(
            id="App_Complex", name="Complex", type="Application",
            betweenness=0.1, dependency_weight_out=42.0, code_quality_penalty=0.95,
        ),
        dimension_contributions={
            "maintainability": {
                "betweenness": 0.02,
                "w_out": 0.05,
                "code_quality_penalty": 0.45,
                "coupling_risk_enh": 0.10,
                "clustering_deficit": 0.03,
            }
        },
    )
    metric_name, val, _ = identify_driver(cq, "Maintainability")
    assert metric_name == "Code Quality Penalty (CQP)"
    assert val == pytest.approx(0.45)


def test_template_interpolation_uses_accurate_structural_metrics():
    """SPOF uses ap_c_directed, Fault-Tolerance Hub uses blast_radius,
    and Bottleneck uses in_degree_raw."""
    engine = ExplanationEngine()

    spof_cq = ComponentQuality(
        id="Broker_1", type="Broker",
        scores=QualityScores(availability=0.9, overall=0.8),
        levels=QualityLevels(availability=CriticalityLevel.CRITICAL, overall=CriticalityLevel.CRITICAL),
        structural=StructuralMetrics(
            id="Broker_1", name="B1", type="Broker",
            ap_c_directed=0.60, bridge_ratio=0.10,
        ),
        profile=CriticalityProfile(a_crit=True),
    )
    spof_exp = engine.explain_component(spof_cq, [])
    assert "60%" in spof_exp.top_risk

    ft_cq = ComponentQuality(
        id="App_Root", type="Application",
        scores=QualityScores(fault_tolerance=0.9, reliability=0.85, overall=0.8),
        levels=QualityLevels(fault_tolerance=CriticalityLevel.CRITICAL, reliability=CriticalityLevel.CRITICAL, overall=CriticalityLevel.CRITICAL),
        structural=StructuralMetrics(
            id="App_Root", name="Root", type="Application",
            in_degree_raw=3, blast_radius=19,
        ),
        profile=CriticalityProfile(ft_crit=True),
    )
    ft_exp = engine.explain_component(ft_cq, [])
    assert "19 downstream components" in ft_exp.top_risk

    bn_cq = ComponentQuality(
        id="App_Mid", type="Application",
        scores=QualityScores(maintainability=0.9, overall=0.8),
        levels=QualityLevels(maintainability=CriticalityLevel.CRITICAL, overall=CriticalityLevel.CRITICAL),
        structural=StructuralMetrics(
            id="App_Mid", name="Mid", type="Application",
            in_degree_raw=4, out_degree_raw=15,
        ),
        profile=CriticalityProfile(m_crit=True),
    )
    bn_exp = engine.explain_component(bn_cq, [])
    assert "4 components must be updated" in bn_exp.top_risk


def test_explain_system_blocks_on_high_and_allows_medium(sample_analysis_result):
    """SystemReport.deployment_blocked aligns with cli/diagnose_graph.py and delta_gate.py:
    HIGH or CRITICAL blocks; MEDIUM alone does not."""
    engine = ExplanationEngine()

    high_smell = AntiPatternReport(
        problems=[DetectedProblem(
            entity_id="A", entity_type="Component", category="Availability",
            severity="HIGH", name="Broker Saturation", description="", recommendation="",
        )],
        summary={"total": 1},
    )
    assert engine.explain_system(sample_analysis_result, high_smell).deployment_blocked is True

    med_smell = AntiPatternReport(
        problems=[DetectedProblem(
            entity_id="A", entity_type="Component", category="Architecture",
            severity="MEDIUM", name="Isolated Component", description="", recommendation="",
        )],
        summary={"total": 1},
    )
    assert engine.explain_system(sample_analysis_result, med_smell).deployment_blocked is False
