
import sys
import os
from pathlib import Path

# Add backend to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.weight_calculator import AHPProcessor, AHPMatrices

def test_shrinkage():
    # v4: r_pagerank is deprecated (0.0); the primary active Reliability weight
    # is r_reverse_pagerank (RPR). Test uses the new AHP matrix (RPR, w_in, CDPot).

    # 1. Pure AHP (lambda=1.0)
    processor_pure = AHPProcessor(shrinkage_factor=1.0)
    weights_pure = processor_pure.compute_weights()
    # Deprecated fields must be 0.0
    assert weights_pure.r_pagerank == 0.0, "r_pagerank must be 0.0 in v4"
    assert weights_pure.r_in_degree == 0.0, "r_in_degree must be 0.0 in v4"
    # Primary weight (RPR) must be positive and dominant
    print(f"Pure R weights: rpr={weights_pure.r_reverse_pagerank:.3f}, w_in={weights_pure.r_w_in:.3f}, cdpot={weights_pure.r_cdpot:.3f}")
    assert weights_pure.r_reverse_pagerank > 0.0
    assert weights_pure.r_w_in > 0.0
    assert weights_pure.r_cdpot > 0.0
    # Active weights must sum to ~ 1.0
    active = weights_pure.r_reverse_pagerank + weights_pure.r_w_in + weights_pure.r_cdpot
    assert abs(active - 1.0) < 0.05

    # 2. Uniform (lambda=0.0): all three active weights should be equal to ~0.333
    processor_uniform = AHPProcessor(shrinkage_factor=0.0)
    weights_uniform = processor_uniform.compute_weights()
    print(f"Uniform R weights: rpr={weights_uniform.r_reverse_pagerank:.3f}, w_in={weights_uniform.r_w_in:.3f}, cdpot={weights_uniform.r_cdpot:.3f}")
    assert abs(weights_uniform.r_reverse_pagerank - 0.333) < 0.02
    assert abs(weights_uniform.r_w_in - 0.333) < 0.02
    assert abs(weights_uniform.r_cdpot - 0.333) < 0.02

    # 3. Default Blend (lambda=0.7): intermediate values between pure and uniform
    processor_blend = AHPProcessor(shrinkage_factor=0.7)
    weights_blend = processor_blend.compute_weights()
    print(f"Blend (0.7) R weights: rpr={weights_blend.r_reverse_pagerank:.3f}, w_in={weights_blend.r_w_in:.3f}, cdpot={weights_blend.r_cdpot:.3f}")
    # RPR row is [1.0, 0.67, 2.0] → AHP weight < w_in (1.5 row)
    # After blending, w_in should be the largest active weight
    assert weights_blend.r_w_in >= weights_blend.r_cdpot, "w_in should dominate CDPot"

    # 4. Availability Blend v2: a_qspof is now the primary SPOF weight; deprecated a_articulation=0
    print(f"Blend (0.7) A weights v2: qspof={weights_blend.a_qspof:.3f}, br={weights_blend.a_bridge_ratio:.3f}, ap_dir={weights_blend.a_ap_c_directed:.3f}, cdi={weights_blend.a_cdi:.3f}")
    # Deprecated fields must be 0.0 in v2
    assert weights_blend.a_articulation == 0.0, "a_articulation must be 0.0 (deprecated in v2)"
    assert weights_blend.a_qos_weight == 0.0,  "a_qos_weight must be 0.0 (deprecated in v2)"
    # Active v2 weights must be positive
    assert weights_blend.a_qspof > 0.0
    assert weights_blend.a_bridge_ratio > 0.0
    assert weights_blend.a_ap_c_directed > 0.0
    assert weights_blend.a_cdi > 0.0
    # Active weights must sum roughly to 1.0
    active_a = weights_blend.a_qspof + weights_blend.a_bridge_ratio + weights_blend.a_ap_c_directed + weights_blend.a_cdi
    assert abs(active_a - 1.0) < 0.05, f"A(v) v2 active weights should sum ~1.0, got {active_a:.4f}"

    print("\nAll AHP shrinkage tests passed!")

if __name__ == "__main__":
    test_shrinkage()
