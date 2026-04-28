"""
Availability Orthogonality Test.

Verifies that A(v) is driven by structural SPOF risk (AP_c_directed, QSPOF),
not by QoS weight alone, and that Reliability and Availability remain orthogonal.

Updated for A(v) v2: uses a_qspof instead of deprecated a_articulation / a_qos_weight.
"""
import pytest
from saag.prediction.analyzer import QualityAnalyzer
from saag.core.metrics import StructuralMetrics
from saag.analysis.models import StructuralAnalysisResult
from saag.core.layers import AnalysisLayer
from saag.prediction.weight_calculator import QualityWeights


def test_availability_orthogonality():
    """Components with high AP_c score should dominate Availability.
    
    A has higher AP impact (articulation-point-like, weight=0.5).
    B has high PageRank but no structural SPOF role.
    A(A) should dominate; R(B) should dominate.
    """
    comp_a = StructuralMetrics(id="A", name="AppA", type="app", pagerank=0.1, weight=0.5,
                               is_articulation_point=True)
    comp_b = StructuralMetrics(id="B", name="AppB", type="app", pagerank=1.0, weight=0.1,
                               is_articulation_point=False)

    result = StructuralAnalysisResult(
        layer=AnalysisLayer.APP,
        components={"A": comp_a, "B": comp_b},
        edges={},
        graph_summary=None,
    )

    # v2 weights: QSPOF-dominant, no legacy fields
    weights = QualityWeights(
        a_qspof=1.0,           # Only test QSPOF contribution
        a_bridge_ratio=0.0,
        a_ap_c_directed=0.0,
        a_cdi=0.0,
        a_articulation=0.0,    # Deprecated
        a_qos_weight=0.0,      # Deprecated
        r_reverse_pagerank=1.0,
        r_pagerank=0.0,
        r_in_degree=0.0,
    )

    analyzer = QualityAnalyzer(weights=weights, use_ahp=False, normalization_method="max")
    quality_result = analyzer.analyze(result)

    qa = next(c for c in quality_result.components if c.id == "A")
    qb = next(c for c in quality_result.components if c.id == "B")

    # With is_articulation_point=True, A gets AP_c > 0 → QSPOF > 0 → A(A) > A(B)
    assert qa.scores.availability >= qb.scores.availability, (
        f"A(A)={qa.scores.availability:.4f} should be >= A(B)={qb.scores.availability:.4f}: "
        "A is an articulation point, B is not"
    )

    print(
        f"\nApp A: PR={qa.structural.pagerank:.2f}, AP={qa.structural.is_articulation_point}, "
        f"R={qa.scores.reliability:.2f}, A={qa.scores.availability:.2f}"
    )
    print(
        f"App B: PR={qb.structural.pagerank:.2f}, AP={qb.structural.is_articulation_point}, "
        f"R={qb.scores.reliability:.2f}, A={qb.scores.availability:.2f}"
    )
    print("\nAvailability Orthogonality test passed!")


if __name__ == "__main__":
    test_availability_orthogonality()
