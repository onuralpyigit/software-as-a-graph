import pytest
from saag.prediction.weight_calculator import QualityWeights
from saag.prediction.analyzer import QualityAnalyzer

def test_quality_weights_defaults_are_ahp_derived():
    """Verify that QualityWeights defaults match the AHP-derived values for Q."""
    w = QualityWeights()
    # A=0.43, R=0.24, M=0.17, V=0.16
    assert w.q_availability == pytest.approx(0.43)
    assert w.q_reliability == pytest.approx(0.24)
    assert w.q_maintainability == pytest.approx(0.17)
    assert w.q_vulnerability == pytest.approx(0.16)
    
    # Check sum
    assert (w.q_availability + w.q_reliability + w.q_maintainability + w.q_vulnerability) == pytest.approx(1.0)

def test_analyzer_equal_weights_override():
    """Verify that QualityAnalyzer(equal_weights=True) overrides defaults to 0.25."""
    analyzer = QualityAnalyzer(equal_weights=True)
    w = analyzer.weights
    assert w.q_availability == 0.25
    assert w.q_reliability == 0.25
    assert w.q_maintainability == 0.25
    assert w.q_vulnerability == 0.25

def test_analyzer_default_uses_ahp_derived():
    """Verify that QualityAnalyzer uses the new defaults by default."""
    analyzer = QualityAnalyzer()
    w = analyzer.weights
    assert w.q_availability == pytest.approx(0.43)
    assert w.q_reliability == pytest.approx(0.24)
    assert w.q_maintainability == pytest.approx(0.17)
    assert w.q_vulnerability == pytest.approx(0.16)
