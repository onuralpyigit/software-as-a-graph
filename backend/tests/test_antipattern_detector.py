"""
Tests for the consolidated AntiPatternDetector.
"""
import pytest
from types import SimpleNamespace
from src.analysis.antipattern_detector import AntiPatternDetector
from src.prediction.models import QualityAnalysisResult, DetectedProblem
from src.core.metrics import ComponentQuality, QualityScores, QualityLevels, StructuralMetrics
from src.core.criticality import CriticalityLevel

@pytest.fixture
def mock_quality_result():
    """Create a mock QualityAnalysisResult with a SPOF and a God Component."""
    # Component A: SPOF (articulation point)
    comp_a = ComponentQuality(
        id="A",
        type="Application",
        scores=QualityScores(overall=0.8, reliability=0.8, maintainability=0.8, availability=0.8, vulnerability=0.2),
        levels=QualityLevels(overall=CriticalityLevel.HIGH, reliability=CriticalityLevel.HIGH, maintainability=CriticalityLevel.HIGH, availability=CriticalityLevel.HIGH, vulnerability=CriticalityLevel.LOW),
        structural=StructuralMetrics(id="A", name="A", type="Application", is_articulation_point=True, betweenness=0.4, pagerank=0.2, in_degree_raw=5, out_degree_raw=5)
    )
    
    # Component B: Normal
    comp_b = ComponentQuality(
        id="B",
        type="Application",
        scores=QualityScores(overall=0.4, reliability=0.4, maintainability=0.4, availability=0.4, vulnerability=0.2),
        levels=QualityLevels(overall=CriticalityLevel.MEDIUM, reliability=CriticalityLevel.MEDIUM, maintainability=CriticalityLevel.MEDIUM, availability=CriticalityLevel.MEDIUM, vulnerability=CriticalityLevel.LOW),
        structural=StructuralMetrics(id="B", name="B", type="Application", is_articulation_point=False, betweenness=0.1, pagerank=0.1, in_degree_raw=1, out_degree_raw=1)
    )
    
    return QualityAnalysisResult(
        timestamp="2026-03-22T00:00:00",
        layer="system",
        context="test",
        components=[comp_a, comp_b],
        edges=[],
        classification_summary=SimpleNamespace(total_components=2, component_distribution={"high": 1, "medium": 1})
    )

def test_detector_spof(mock_quality_result):
    """Verify SPOF detection."""
    detector = AntiPatternDetector()
    shim = SimpleNamespace(quality=mock_quality_result)
    problems = detector.detect(shim, "system")
    
    spofs = [p for p in problems if "Single Point of Failure" in p.name]
    assert len(spofs) == 1
    assert spofs[0].entity_id == "A"
    assert spofs[0].severity in ["CRITICAL", "HIGH"]

def test_detector_god_component(mock_quality_result):
    """Verify God Component detection."""
    detector = AntiPatternDetector()
    # Modify comp_a to be CRITICAL and high betweenness
    mock_quality_result.components[0].levels.maintainability = CriticalityLevel.CRITICAL
    mock_quality_result.components[0].structural.betweenness = 0.5
    
    shim = SimpleNamespace(quality=mock_quality_result)
    problems = detector.detect(shim, "system")
    
    gods = [p for p in problems if "God Component" in p.name]
    assert len(gods) == 1
    assert gods[0].entity_id == "A"

def test_problem_detector_wrapper(mock_quality_result):
    """Verify the backward-compatible ProblemDetector wrapper."""
    from src.prediction.problem_detector import ProblemDetector
    
    wrapper = ProblemDetector()
    problems = wrapper.detect(mock_quality_result)
    
    assert len(problems) > 0
    assert isinstance(problems[0], DetectedProblem)
    
    summary = wrapper.summarize(problems)
    assert summary.total_problems == len(problems)
    assert summary.affected_components >= 1
