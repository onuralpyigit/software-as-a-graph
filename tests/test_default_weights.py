import pytest
from saag.analysis.weight_calculator import QualityWeights
from saag.analysis.analyzer import QualityAnalyzer

def test_quality_weights_defaults_are_derived():
    """Verify that QualityWeights defaults match the values derived from the
    retired 4-D AHP composite (A=0.43, R=0.24, M=0.17, S=0.16) by dropping
    Vulnerability/Security and renormalising: q_reliability=(0.24+0.43)/0.84,
    q_maintainability=0.17/0.84. Not AHP-derived at this level — see class
    docstring for why a 2x2 composite matrix would add nothing."""
    w = QualityWeights()
    assert w.q_reliability == pytest.approx(0.80)
    assert w.q_maintainability == pytest.approx(0.20)
    assert w.r_alpha == pytest.approx(0.36)

    # Check sum
    assert (w.q_reliability + w.q_maintainability) == pytest.approx(1.0)

def test_analyzer_equal_weights_override():
    """Verify that QualityAnalyzer(equal_weights=True) overrides defaults to 0.5,
    including r_alpha — "equal weights" must mean equal at every level, not just
    at the composite, or it would silently keep the declared 0.36/0.64 FT-vs-A
    split baked into R."""
    analyzer = QualityAnalyzer(equal_weights=True)
    w = analyzer.weights
    assert w.q_reliability == 0.5
    assert w.q_maintainability == 0.5
    assert w.r_alpha == 0.5

def test_equal_weights_does_not_mutate_shared_weights_instance():
    """equal_weights=True must not mutate a caller-supplied QualityWeights in
    place — two analyzers sharing one weights object would otherwise corrupt
    each other's configuration."""
    shared = QualityWeights()
    original_reliability = shared.q_reliability

    QualityAnalyzer(weights=shared, equal_weights=True)

    assert shared.q_reliability == pytest.approx(original_reliability)
    assert shared.q_reliability != 0.5


def test_analyzer_default_uses_derived_weights():
    """Verify that QualityAnalyzer uses the new defaults by default."""
    analyzer = QualityAnalyzer()
    w = analyzer.weights
    assert w.q_reliability == pytest.approx(0.80)
    assert w.q_maintainability == pytest.approx(0.20)
