
import pytest

from saag.analysis.weight_calculator import AHPProcessor, AHPMatrices
from saag.core.models import AHP_SHRINKAGE_LAMBDA, QoSPolicy

def test_shrinkage():
    # v5: ft_pagerank is deprecated (0.0); the three active Reliability weights
    # are ft_reverse_pagerank (RPR), ft_in_degree (DG_in), ft_cdpot (CDPot).

    # 1. Pure AHP (lambda=1.0)
    processor_pure = AHPProcessor(shrinkage_factor=1.0)
    weights_pure = processor_pure.compute_weights()
    # Deprecated field must be 0.0
    assert weights_pure.ft_pagerank == 0.0, "ft_pagerank must be 0.0 in v5"
    # Primary weight (RPR) must be positive and dominant
    print(f"Pure R weights: rpr={weights_pure.ft_reverse_pagerank:.3f}, din={weights_pure.ft_in_degree:.3f}, cdpot={weights_pure.ft_cdpot:.3f}")
    assert weights_pure.ft_reverse_pagerank > 0.0
    assert weights_pure.ft_in_degree > 0.0
    assert weights_pure.ft_cdpot > 0.0
    # Active weights must sum to ~ 1.0
    active = weights_pure.ft_reverse_pagerank + weights_pure.ft_in_degree + weights_pure.ft_cdpot
    assert abs(active - 1.0) < 0.05

    # 2. Uniform (lambda=0.0): all three active weights should be equal to ~0.333
    processor_uniform = AHPProcessor(shrinkage_factor=0.0)
    weights_uniform = processor_uniform.compute_weights()
    print(f"Uniform R weights: rpr={weights_uniform.ft_reverse_pagerank:.3f}, din={weights_uniform.ft_in_degree:.3f}, cdpot={weights_uniform.ft_cdpot:.3f}")
    assert abs(weights_uniform.ft_reverse_pagerank - 0.333) < 0.02
    assert abs(weights_uniform.ft_in_degree - 0.333) < 0.02
    assert abs(weights_uniform.ft_cdpot - 0.333) < 0.02

    # 3. Default Blend (lambda=0.7): intermediate values between pure and uniform
    processor_blend = AHPProcessor(shrinkage_factor=AHP_SHRINKAGE_LAMBDA)
    weights_blend = processor_blend.compute_weights()
    print(f"Blend (0.7) R weights: rpr={weights_blend.ft_reverse_pagerank:.3f}, din={weights_blend.ft_in_degree:.3f}, cdpot={weights_blend.ft_cdpot:.3f}")
    # RPR is highest in the v5 matrix: [1.0, 1.5, 2.0] row means RPR > DG_in > CDPot
    assert weights_blend.ft_reverse_pagerank >= weights_blend.ft_in_degree, "RPR should dominate DG_in"


    # 4. Availability Blend v2: a_qspof is now the primary SPOF weight; deprecated a_articulation=0
    print(f"Blend (0.7) A weights v2: qspof={weights_blend.a_qspof:.3f}, br={weights_blend.a_bridge_ratio:.3f}, ap_dir={weights_blend.a_ap_c_directed:.3f}, cdi={weights_blend.a_cdi:.3f}")
    # Active v2/v3 weights must be positive
    assert weights_blend.a_qspof > 0.0
    assert weights_blend.a_bridge_ratio > 0.0
    assert weights_blend.a_ap_c_directed > 0.0
    assert weights_blend.a_cdi > 0.0
    assert weights_blend.a_qos_weight > 0.0
    # Active weights must sum roughly to 1.0
    active_a = weights_blend.a_qspof + weights_blend.a_bridge_ratio + weights_blend.a_ap_c_directed + weights_blend.a_cdi + weights_blend.a_qos_weight
    assert abs(active_a - 1.0) < 0.05, f"A(v) v2 active weights should sum ~1.0, got {active_a:.4f}"

    print("\nAll AHP shrinkage tests passed!")


def test_topic_qos_matrix_reproduces_shipped_weights():
    # Proves QoSPolicy's shipped W_RELIABILITY/W_DURABILITY/W_PRIORITY constants
    # (used at runtime to compute w(topic), graph-construction Phase 3) are what
    # a consistent Saaty pairwise-comparison matrix produces — closing the gap
    # where AHPMatrices.criteria_topic_qos existed but was never read by
    # AHPProcessor.compute_weights().
    result = AHPProcessor().compute_topic_qos_weights()
    assert result["consistency_ratio"] < 0.10, "Saaty matrix must be internally consistent"
    assert result["reliability"] == pytest.approx(QoSPolicy.W_RELIABILITY, abs=0.01)
    assert result["durability"] == pytest.approx(QoSPolicy.W_DURABILITY, abs=0.01)
    assert result["priority"] == pytest.approx(QoSPolicy.W_PRIORITY, abs=0.01)

    # Non-degeneracy: an earlier version of this matrix was solved backward
    # from the target vector, which makes CR ~= 0 by construction (a matrix
    # filled in to reproduce a chosen answer is consistent almost by
    # definition) rather than by evidence of a genuine independent judgement.
    # An honestly-elicited 3x3 matrix over non-trivial pairwise ratios is
    # exceedingly unlikely to land on a CR this small by chance, so this
    # floor catches a reconstructed matrix without pinning an exact value.
    assert result["consistency_ratio"] > 0.005, (
        "CR is suspiciously close to zero for a 3x3 matrix with non-trivial "
        "off-diagonal ratios -- check that criteria_topic_qos was stated "
        "independently rather than solved backward from W_RELIABILITY/"
        "W_DURABILITY/W_PRIORITY."
    )


if __name__ == "__main__":
    test_shrinkage()
    test_topic_qos_matrix_reproduces_shipped_weights()
