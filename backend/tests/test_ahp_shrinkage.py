
import sys
import os
from pathlib import Path

# Add backend to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.weight_calculator import AHPProcessor, AHPMatrices

def test_shrinkage():
    # 1. Pure AHP (lambda=1.0)
    processor_pure = AHPProcessor(shrinkage_factor=1.0)
    weights_pure = processor_pure.compute_weights()
    print(f"Pure R weights: {weights_pure.r_pagerank:.3f}, {weights_pure.r_reverse_pagerank:.3f}, {weights_pure.r_in_degree:.3f}")
    assert abs(weights_pure.r_pagerank - 0.50) < 0.01

    # 2. Uniform (lambda=0.0)
    processor_uniform = AHPProcessor(shrinkage_factor=0.0)
    weights_uniform = processor_uniform.compute_weights()
    print(f"Uniform R weights: {weights_uniform.r_pagerank:.3f}, {weights_uniform.r_reverse_pagerank:.3f}, {weights_uniform.r_in_degree:.3f}")
    assert abs(weights_uniform.r_pagerank - 0.333) < 0.01

    # 3. Default Blend (lambda=0.7)
    processor_blend = AHPProcessor(shrinkage_factor=0.7)
    weights_blend = processor_blend.compute_weights()
    print(f"Blend (0.7) R weights: {weights_blend.r_pagerank:.3f}, {weights_blend.r_reverse_pagerank:.3f}, {weights_blend.r_in_degree:.3f}")
    # w = 0.7*0.5 + 0.3*0.333 = 0.35 + 0.10 = 0.45
    assert abs(weights_blend.r_pagerank - 0.45) < 0.01
    
    # 4. Availability Blend
    print(f"Blend (0.7) A weights: {weights_blend.a_articulation:.3f}, {weights_blend.a_bridge_ratio:.3f}, {weights_blend.a_importance:.3f}")
    # w = 0.7*0.65 + 0.3*0.333 = 0.455 + 0.10 = 0.555
    assert abs(weights_blend.a_articulation - 0.555) < 0.01

    print("\nAll AHP shrinkage tests passed!")

if __name__ == "__main__":
    test_shrinkage()
