"""
Tests for CLIFormatter
"""
import pytest
from unittest.mock import MagicMock, patch

from src.explanation.cli import CLIFormatter
from src.explanation.engine import ComponentExplanation, DimensionExplanation
from src.prediction.models import QualityAnalysisResult, ComponentQuality
from src.core.metrics import QualityLevels
from src.core.criticality import CriticalityLevel

def test_format_component_card():
    # Mock ComponentExplanation
    dim1 = DimensionExplanation(
        dimension="Reliability",
        score=0.891,
        level="CRITICAL",
        driving_metric="Reverse PageRank (RPR)",
        driving_value=0.87,
        plain_meaning="14 downstream components depend on it transitively.",
        risk_sentence="A failure here cascades widely."
    )
    
    exp = ComponentExplanation(
        component_id="App_Controller",
        pattern="Total Hub",
        level="CRITICAL",
        one_line="App_Controller is a critical hub.",
        top_risk="A single failure here activates three independent failure modes.",
        dimensions=[dim1],
        priority_action="Introduce standard redundancy.",
        anti_patterns=[]
    )
    
    card = CLIFormatter.format_component_card(exp)
    
    assert "App_Controller" in card
    assert "Total Hub" in card
    assert "CRITICAL" in card
    assert "Reverse PageRank (RPR)" in card
    assert "downstream" in card
    assert "components" in card
    assert "Introduce" in card
    assert "standard" in card
    assert "redundancy." in card

def test_print_critical_report():
    # Mock QualityAnalysisResult
    comp = MagicMock()
    comp.id = "TestComp"
    comp.levels.overall.value = "CRITICAL"
    comp.scores.overall = 0.9
    
    res = MagicMock(spec=QualityAnalysisResult)
    res.components = [comp]
    
    # Mock ExplanationEngine in the module
    with patch("src.explanation.cli.ExplanationEngine") as mock_engine_class:
        mock_engine = mock_engine_class.return_value
        mock_engine.explain_component.return_value = ComponentExplanation(
            component_id="TestComp",
            pattern="SPOF",
            level="CRITICAL",
            one_line="one liner",
            top_risk="top risk",
            dimensions=[],
            priority_action="fix it",
            anti_patterns=[]
        )
        
        with patch("builtins.print") as mock_print:
            CLIFormatter.print_critical_report(res)
            assert mock_print.called
            mock_engine.explain_component.assert_called_once()
def test_format_system_report():
    from src.explanation.engine import SystemReport, RemediationStep
    
    report = SystemReport(
        total_components=32,
        critical_count=3,
        high_count=5,
        deployment_blocked=True,
        reason="2 CRITICAL anti-patterns detected (SPOF, CYCLIC_DEPENDENCY).",
        top_risk_summary="The system's primary architectural risk is concentration.",
        by_stakeholder={
            "Reliability Engineer": ["Add circuit breakers."],
            "Software Architect": ["Extract stable interface."],
            "DevOps / SRE": ["Deploy failover replica."],
            "Security Engineer": ["Apply input validation."]
        },
        component_explanations=[],
        remediation_plan=[
            RemediationStep(action="Redundancy for App_Controller", components=["App_Controller"], priority=1)
        ]
    )
    
    output = CLIFormatter.format_system_report(report, "app")
    
    assert "System Report — app layer" in output
    assert "BLOCKED" in output
    assert "SPOF" in output
    assert "Reliability Engineer" in output
    assert "backlog" in output
    assert "P1 [CRITICAL" in output
