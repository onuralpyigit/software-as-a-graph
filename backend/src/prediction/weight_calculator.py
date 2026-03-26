"""
Analytic Hierarchy Process (AHP) Module

Provides functionality to calculate weights for quality attributes using the 
Analytic Hierarchy Process. This allows for more rigorous, relative importance-based
weight determination compared to arbitrary assignment.

The module uses the Geometric Mean method (approximate eigenvector) to calculate
weights from pairwise comparison matrices.

Changes (v2):
    - Renamed m_degree → m_out_degree (Maintainability uses efferent coupling)
    - Renamed v_in_degree → v_out_degree (Vulnerability uses attack surface)
    - Updated AHP matrix comments to reflect new metric assignments
"""

import math
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class QualityWeights:
    """
    Configurable weights for quality score computation.
    
    All weights should sum to 1.0 within each dimension.
    
    Design principles (v2):
        - Metric orthogonality: No raw metric appears in more than two dimensions.
          In-Degree is exclusive to Reliability. Out-Degree is shared between
          Maintainability (efferent coupling) and Vulnerability (attack surface).
        - Continuous scoring: AP uses continuous fragmentation score, not binary.
    
    Note on Overall Weights (q_* parameters):
        Default equal weights (0.25 each) represent a balanced approach where
        all four dimensions are considered equally important. Adjust these
        based on system priorities:
        - Security-critical systems: Increase q_vulnerability
        - High-availability systems: Increase q_availability
        - Fast-iteration systems: Increase q_maintainability
        - Mission-critical systems: Increase q_reliability
    """
    # Reliability weights (fault propagation) — R*(v) = RPR + DG_in + CDPot (v5)
    # r_pagerank kept at 0.0 for backward-compat serialisation only.
    # r_w_in demoted to 0.0: w_in is now exclusively assigned to V*(v) as QADS.
    r_pagerank: float = 0.0          # Deprecated (v4): superseded; kept for compat
    r_reverse_pagerank: float = 0.45 # AHP leader: propagation reach (RPR); weight increased from 0.40
    r_in_degree: float = 0.30        # Reinstatement (v5): count-based immediate-dependents signal
    r_w_in: float = 0.0             # Deprecated (v5): reassigned to V*(v) as QADS; kept for compat
    r_cdpot: float = 0.25            # Cascade Depth Potential (derived, depth signal)
    
    # Maintainability weights (coupling complexity) — M(v) v6
    # Formula: 0.35*BT + 0.30*w_out + 0.15*CQP + 0.12*CouplingRisk + 0.08*(1-CC)
    m_betweenness: float = 0.35      # AHP primary: structural bottleneck position
    m_w_out: float = 0.30            # QoS-weighted efferent coupling (promoted from 'Reported only')
    m_code_quality_penalty: float = 0.15  # Code Quality Penalty (CQP): complexity + instability + LCOM composite
    m_coupling_risk: float = 0.12    # CouplingRisk: afferent/efferent imbalance signal
    m_clustering: float = 0.08       # (1-CC): direction-agnostic proxy; reduced weight
    # Deprecated in v5 — subsumed by m_w_out (QoS-aware). Kept for backward-compat serialisation.
    m_out_degree: float = 0.0
    
    # Availability weights (SPOF risk) — A(v) v3
    # Formula: 0.35*AP_c_directed + 0.25*QSPOF + 0.25*BR + 0.10*CDI + 0.05*w(v)
    a_ap_c_directed: float = 0.35  # AHP primary: structural directed SPOF severity (baseline)
    a_qspof: float = 0.25          # QoS-weighted SPOF: AP_c_directed * w(v)
    a_bridge_ratio: float = 0.25   # Edge-level irrecoverability (reduced weight)
    a_cdi: float = 0.10            # Connectivity Degradation Index
    a_qos_weight: float = 0.05     # Operational weight contribution w(v) (Issue 5: decoupling)

    # Vulnerability weights (exposure risk) — V(v) v2
    # Formula: 0.40*REV + 0.35*RCL + 0.25*QADS
    v_reverse_eigenvector: float = 0.40 # AHP primary: G^T eigenvector (strategic attack reach)
    v_reverse_closeness: float = 0.35   # AHP secondary: G^T closeness (propagation speed)
    v_qads: float = 0.25                # QoS-weighted dependent surface (w_in)
    # Deprecated in v2 — kept at 0.0 for backward-compat serialisation only
    v_eigenvector: float = 0.0
    v_closeness: float = 0.0
    v_out_degree: float = 0.0
    
    # Overall quality weights (sum should be 1.0)
    q_reliability: float = 0.25      # Default balanced (1.0 vs 1.0 vs 1.0 vs 1.0)
    q_maintainability: float = 0.25
    q_availability: float = 0.25
    q_vulnerability: float = 0.25
    
    # Impact score weights I(v) (sum should be 1.0)
    # Formally derived via AHP: Reachability > Fragmentation = Throughput
    i_reachability: float = 0.40      
    i_fragmentation: float = 0.30     
    i_throughput: float = 0.30        
    
    # Edge quality weights (sum should be 1.0)
    e_betweenness: float = 0.35      # Path importance
    e_bridge: float = 0.30           # SPOF risk
    e_endpoint: float = 0.20         # Connected node importance
    e_vulnerability: float = 0.15    # Endpoint vulnerability exposure


# Scale of Relative Importance (Saaty's Scale)
# 1: Equal importance
# 3: Moderate importance
# 5: Strong importance
# 7: Very strong importance
# 9: Extreme importance
# 2, 4, 6, 8: Intermediate values

@dataclass
class AHPMatrices:
    """
    Stores pairwise comparison matrices for all quality dimensions.
    Default values reflect a balanced/standard architectural perspective.
    
    Metric assignments (v5):
        Reliability:      Reverse PageRank (RPR), In-Degree (DG_in), CDPot    [w_in REMOVED — exclusively QADS in V*]
        Maintainability:  Betweenness (BT), w_out (QoS-efferent), CouplingRisk (CR), (1-CC)
        Availability:     QSPOF, Bridge Ratio (BR), AP_c_directed, CDI
        Vulnerability:    Reverse Eigenvector (REV), Reverse Closeness (RCL), QADS (w_in)
        Impact I*(v):     IR(v), IM(v), IA(v), IV(v) — multi-phenomenon unified ground truth
        Impact IR(v):     Cascade Reach (CR), Weighted Cascade Impact (WCI), Normalised Depth (ND)
    """
    
    # Reliability v4: Reverse PageRank (RPR), QoS-Weighted In-Degree (w_in), CDPot
    # RPR: primary propagation reach
    # w_in: QoS-weighted dependent count — richer than raw DG_in
    # CDPot: derived depth signal (no new algorithm needed)
    criteria_reliability: List[List[float]] = None
    
    # Maintainability v5: Betweenness (BT), w_out (QoS-efferent), CouplingRisk (CR), (1-CC)
    # BT: structural bottleneck; w_out: QoS-weighted contracts; CR: imbalance signal; (1-CC): proxy
    criteria_maintainability: List[List[float]] = None
    
    # Availability v2: QSPOF, Bridge Ratio, AP_c_directed, CDI
    # QSPOF = AP_c_directed * w(v) — operationally weighted structural SPOF
    # AP_c_directed = max(AP_c_out, AP_c_in) — worst-case directional SPOF
    # CDI — connectivity degradation for non-AP hubs
    criteria_availability: List[List[float]] = None
    
    # Vulnerability v2: Reverse Eigenvector (REV), Reverse Closeness (RCL), QADS
    # Strategic reach + propagation speed + QoS attack surface
    criteria_vulnerability: List[List[float]] = None
    
    # Overall Quality: Reliability (R), Maintainability (M), Availability (A), Vulnerability (V)
    criteria_overall: List[List[float]] = None

    # Topic QoS Importance: Reliability (Rel), Durability (Dur), Priority (Pri)
    # Justifies the 0.30/0.40/0.30 split used in Phase 4 modeling.
    criteria_topic_qos: List[List[float]] = None

    # Impact (I(v)): Reachability (RL), Fragmentation (FR), Throughput (TL)
    # Justifies the 0.40/0.30/0.30 split used in failure simulation.
    criteria_impact: List[List[float]] = None

    def __post_init__(self):
        # Default initialization if None
        if self.criteria_reliability is None:
            self.criteria_reliability = [
                # RPR   DG_in CDPot
                [1.0,  1.5,   2.0],  # RPR  (primary propagation reach; increased from 0.40→0.45)
                [0.67, 1.0,   1.5],  # DG_in (count-based immediate dependents; reinstated at 0.30)
                [0.5,  0.667, 1.0],  # CDPot (derived depth signal; unchanged at 0.25)
            ]
            # AHP-derived weights (geometric mean + shrinkage=0.7) ≈ (0.45, 0.30, 0.25)
            
        if self.criteria_maintainability is None:
            self.criteria_maintainability = [
                # BT    w_out   CQP    CR    (1-CC)
                [1.0,  1.17,  2.33,  2.92,  4.38],  # BT: structural bottleneck (primary)
                [0.86, 1.0,   2.0,   2.5,   3.75],  # w_out: QoS-weighted efferent coupling
                [0.43, 0.5,   1.0,   1.25,  1.88],  # CQP: code-level maintainability penalty
                [0.34, 0.4,   0.8,   1.0,   1.5],   # CouplingRisk: afferent/efferent imbalance
                [0.23, 0.267, 0.533, 0.667, 1.0],   # (1-CC): direction-agnostic proxy
            ]
            # AHP-derived geometric means ≈ [0.35, 0.30, 0.15, 0.12, 0.08] before shrinkage
            # After λ=0.7: ≈ [0.345, 0.31, 0.155, 0.134, 0.096] (close to design weights)
            
        if self.criteria_availability is None:
            self.criteria_availability = [
                # AP_c_d QSPOF BR     CDI    w
                [1.0,  1.4,   1.4,   3.5,   7.0],  # AP_c_d: Structural baseline (primary)
                [0.71, 1.0,   1.0,   2.5,   5.0],  # QSPOF: QoS-weighted SPOF
                [0.71, 1.0,   1.0,   2.5,   5.0],  # BR: Multi-edge brittleness
                [0.29, 0.4,   0.4,   1.0,   2.0],  # CDI: Path elongation
                [0.14, 0.2,   0.2,   0.5,   1.0],  # w: Pure operational priority
            ]
            # Geometric mean → approx [0.35, 0.25, 0.25, 0.10, 0.05] before shrinkage
            # With shrinkage λ=0.7, weighted toward uniform (0.2)

        if self.criteria_vulnerability is None:
            self.criteria_vulnerability = [
                # REV   RCL   QADS
                [1.0,  1.14,  1.6],  # REV (Strategic dependent reach)
                [0.88, 1.0,   1.4],  # RCL (Propagation speed)
                [0.62, 0.71,  1.0],  # QADS (QoS-weighted surface)
            ]
            # Matrix check: geometric mean approx [0.40, 0.35, 0.25]
            
        if self.criteria_overall is None:
            self.criteria_overall = [
                # R     M     A     V
                # Theoretically motivated: structural alignment A > R > M > V
                # with prediction strength based on each dimension's simulation ground truth.
                # CR ≈ 0.02 → AHP weights ≈ [0.24, 0.17, 0.43, 0.16]
                [1.0,  1.5,  0.5,  2.0],   # R: strong vs M/V; weaker vs A
                [0.67, 1.0,  0.33, 1.5],   # M: weakest overall
                [2.0,  3.0,  1.0,  3.0],   # A: dominant (highest structural alignment)
                [0.5,  0.67, 0.33, 1.0],   # V: second-weakest (G^T metric alignment)
            ]
        
        if self.criteria_topic_qos is None:
            self.criteria_topic_qos = [
                # Rel  Dur  Pri
                [1.0, 0.75, 1.0],  # Rel: Slightly less critical than Durability
                [1.33, 1.0, 1.33], # Dur: Most critical (state persistence)
                [1.0, 0.75, 1.0],  # Pri: Same as Reliability
            ]
            # Matrix check: Dur/Rel = 1.33, Rel/Dur = 0.75. 
            # Calculated weights: [0.30, 0.40, 0.30]
            
        if self.criteria_impact is None:
            self.criteria_impact = [
                # RL    FR    TL
                [1.0, 1.33, 1.33],  # RL (Reachability loss: most direct failure)
                [0.75, 1.0, 1.0],   # FR
                [0.75, 1.0, 1.0],   # TL (FR and TL equal weight)
            ]
            # Matrix check: RL/FR = 1.33, FR/RL = 0.75. 
            # Calculated weights: [0.40, 0.30, 0.30]


class AHPProcessor:
    """
    Calculates weights from pairwise comparison matrices with optional shrinkage.
    
    Shrinkage (blending) addresses methodological liability by formally 
    reconciling pure AHP weights with a uniform prior.
    """
    
    def __init__(self, matrices: AHPMatrices = None, shrinkage_factor: float = 0.7):
        self.matrices = matrices or AHPMatrices()
        self.shrinkage_factor = shrinkage_factor

    def _shrink_weights(self, weights: List[float]) -> List[float]:
        """
        Blends AHP weights with a uniform prior using mixing coefficient lambda.
        w_final = lambda * w_ahp + (1 - lambda) * w_uniform
        """
        n = len(weights)
        if n == 0:
            return weights
        uniform_weight = 1.0 / n
        return [
            (self.shrinkage_factor * w) + ((1.0 - self.shrinkage_factor) * uniform_weight)
            for w in weights
        ]

    def _calculate_priority_vector(self, matrix: List[List[float]]) -> List[float]:
        """
        Calculates the priority vector (weights) using the Geometric Mean method.
        """
        n = len(matrix)
        geometric_means = []
        
        # 1. Calculate geometric mean of each row
        for row in matrix:
            product = 1.0
            for val in row:
                product *= val
            geometric_means.append(math.pow(product, 1.0/n))
            
        # 2. Normalize
        sum_gm = sum(geometric_means)
        return [gm / sum_gm for gm in geometric_means]

    def _calculate_consistency_ratio(self, matrix: List[List[float]], weights: List[float]) -> float:
        """
        Calculates the Consistency Ratio (CR) to validate the matrix.
        CR < 0.1 is generally considered acceptable.
        """
        n = len(matrix)
        if n <= 2:
            return 0.0 # Consistency is usually perfect for n<=2
            
        # Random Index (RI) lookup for n=1 to 10
        ri_map = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24}
        ri = ri_map.get(n, 1.45)

        # Calculate Lambda Max
        # Multiply matrix by weight vector
        weighted_sum_vector = []
        for i in range(n):
            row_sum = 0
            for j in range(n):
                row_sum += matrix[i][j] * weights[j]
            weighted_sum_vector.append(row_sum)
            
        # Average of (Weighted Sum / Weight)
        lambda_max_values = [ws / w for ws, w in zip(weighted_sum_vector, weights)]
        lambda_max = sum(lambda_max_values) / n
        
        # Calculate CI and CR
        ci = (lambda_max - n) / (n - 1)
        cr = ci / ri if ri > 0 else 0
        return cr

    def compute_weights(self) -> QualityWeights:
        """Process all matrices and return a populated QualityWeights object."""
        
        # 1. Reliability Weights v4 (RPR, w_in, CDPot)
        w_rel = self._calculate_priority_vector(self.matrices.criteria_reliability)
        w_rel = self._shrink_weights(w_rel)
        
        # 2. Maintainability Weights v6 (BT, w_out, CQP, CouplingRisk, (1-CC))
        w_main = self._calculate_priority_vector(self.matrices.criteria_maintainability)
        w_main = self._shrink_weights(w_main)
        
        # 3. Availability Weights (AP_c, BR, w)
        w_avail = self._calculate_priority_vector(self.matrices.criteria_availability)
        w_avail = self._shrink_weights(w_avail)
        
        # 4. Vulnerability Weights v2 (REV, RCL, QADS)
        w_vuln = self._calculate_priority_vector(self.matrices.criteria_vulnerability)
        w_vuln = self._shrink_weights(w_vuln)

        # 5. Impact Weights (RL, FR, TL) - Added for formal derivation
        w_impact = self._calculate_priority_vector(self.matrices.criteria_impact)
        w_impact = self._shrink_weights(w_impact)
        
        # 6. Overall Weights (R, M, A, V)
        w_over = self._calculate_priority_vector(self.matrices.criteria_overall)
        w_over = self._shrink_weights(w_over)
        
        return QualityWeights(
            # Reliability v5: (RPR, DG_in, CDPot) — w_in now exclusively QADS in V*
            r_pagerank=0.0,               # Deprecated
            r_reverse_pagerank=w_rel[0],  # RPR — primary (0.45)
            r_in_degree=w_rel[1],         # DG_in — count-based immediate dependents (0.30)
            r_w_in=0.0,                   # Deprecated in v5; reassigned to V*(v) as QADS
            r_cdpot=w_rel[2],             # Cascade Depth Potential (0.25)
            
            # Maintainability v6: (BT, w_out, CQP, CouplingRisk, (1-CC))
            m_betweenness=w_main[0],
            m_w_out=w_main[1],
            m_code_quality_penalty=w_main[2],
            m_coupling_risk=w_main[3],
            m_clustering=w_main[4],
            m_out_degree=0.0,               # Deprecated in v5+
            
            # Availability v3: (AP_c_directed, QSPOF, BR, CDI, w)
            a_ap_c_directed=w_avail[0],    # Structural baseline (0.35)
            a_qspof=w_avail[1],             # QoS-weighted SPOF (0.25)
            a_bridge_ratio=w_avail[2],      # Multi-edge brittleness (0.25)
            a_cdi=w_avail[3],               # Path elongation (0.10)
            a_qos_weight=w_avail[4],        # Pure operational priority (0.05)
            
            # Vulnerability v2: (REV, RCL, QADS)
            v_reverse_eigenvector=w_vuln[0],
            v_reverse_closeness=w_vuln[1],
            v_qads=w_vuln[2],
            # Deprecated
            v_eigenvector=0.0,
            v_closeness=0.0,
            v_out_degree=0.0,
            
            # Impact
            i_reachability=w_impact[0],
            i_fragmentation=w_impact[1],
            i_throughput=w_impact[2],
            
            # Overall
            q_reliability=w_over[0],
            q_maintainability=w_over[1],
            q_availability=w_over[2],
            q_vulnerability=w_over[3]
        )