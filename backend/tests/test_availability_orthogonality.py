
import sys
import os
from pathlib import Path

# Add backend to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.quality_analyzer import QualityAnalyzer
from src.core.metrics import StructuralMetrics, StructuralAnalysisResult
from src.core.layers import AnalysisLayer
from src.analysis.weight_calculator import QualityWeights

def test_availability_orthogonality():
    # 1. Create a simple graph: A -> B
    # B is a hub (high PR). A is not.
    # If we set w(A) = 1.0 and w(B) = 0.1, A should have higher QoS weight component in Availability.
    
    comp_a = StructuralMetrics(id="A", name="AppA", type="app", pagerank=0.1, weight=1.0)
    comp_b = StructuralMetrics(id="B", name="AppB", type="app", pagerank=1.0, weight=0.1)
    
    # Mock result
    result = StructuralAnalysisResult(
        layer=AnalysisLayer.APPLICATION,
        components={"A": comp_a, "B": comp_b},
        edges={},
        graph_summary=None
    )
    
    # Initialize analyzer with manual weights (no shrinkage to make it clear)
    weights = QualityWeights(
        a_articulation=0.0,
        a_bridge_ratio=0.0,
        a_qos_weight=1.0,  # Only test the weight contribution
        r_pagerank=1.0,
        r_reverse_pagerank=0.0,
        r_in_degree=0.0
    )
    
    analyzer = QualityAnalyzer(weights=weights, use_ahp=False, normalization_method="max")
    quality_result = analyzer.analyze(result)
    
    qa = next(c for c in quality_result.components if c.id == "A")
    qb = next(c for c in quality_result.components if c.id == "B")
    
    print(f"App A: PR={qa.structural.pagerank:.2f}, Weight={qa.structural.weight:.2f}, R={qa.scores.reliability:.2f}, A={qa.scores.availability:.2f}")
    print(f"App B: PR={qb.structural.pagerank:.2f}, Weight={qb.structural.weight:.2f}, R={qb.scores.reliability:.2f}, A={qb.scores.availability:.2f}")
    
    # With a_qos_weight=1.0 and weight(A)=1.0, A(A) should be 1.0 (normalized)
    # With a_qos_weight=1.0 and weight(B)=0.1, A(B) should be 0.1 (normalized)
    # But for Reliability, R(B) should be 1.0 (normalized PR) and R(A) should be 0.1
    
    assert qa.scores.availability > qb.scores.availability, "A should have higher availability score due to higher QoS weight"
    assert qb.scores.reliability > qa.scores.reliability, "B should have higher reliability score due to higher PageRank"
    
    print("\nAvailability Orthogonality test passed!")

if __name__ == "__main__":
    test_availability_orthogonality()
