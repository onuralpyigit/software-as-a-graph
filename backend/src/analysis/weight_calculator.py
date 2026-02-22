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
    # Reliability weights (fault propagation)
    r_pagerank: float = 0.50         # AHP: (1.0, 2.0, 2.0)
    r_reverse_pagerank: float = 0.25
    r_in_degree: float = 0.25
    
    # Maintainability weights (efferent coupling complexity)
    m_betweenness: float = 0.54       # AHP: (1.0, 2.0, 3.0)
    m_out_degree: float = 0.30       
    m_clustering: float = 0.16       
    
    # Availability weights (SPOF risk)
    a_articulation: float = 0.65     # AHP: (1.0, 3.0, 5.0)
    a_bridge_ratio: float = 0.23
    a_importance: float = 0.12       

    # Vulnerability weights (exposure risk)
    v_eigenvector: float = 0.50      # AHP: (1.0, 2.0, 2.0)
    v_closeness: float = 0.25
    v_out_degree: float = 0.25       
    
    # Overall quality weights (sum should be 1.0)
    q_reliability: float = 0.25      # Default balanced (1.0 vs 1.0 vs 1.0 vs 1.0)
    q_maintainability: float = 0.25
    q_availability: float = 0.25
    q_vulnerability: float = 0.25
    
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
    
    Metric assignments (v2):
        Reliability:      PageRank (PR), Reverse PageRank (RPR), In-Degree (ID)
        Maintainability:  Betweenness (BT), Out-Degree (OD), Clustering (CC)
        Availability:     Articulation Score (AP_c), Bridge Ratio (BR), Importance (IM)
        Vulnerability:    Eigenvector (EV), Closeness (CL), Out-Degree (OD)
    """
    
    # Reliability: PageRank (PR), Reverse PageRank (RPR), In-Degree (ID)
    # PR is often slightly more important for global importance
    criteria_reliability: List[List[float]] = None
    
    # Maintainability: Betweenness (BT), Out-Degree (OD), Clustering (CC)
    # Betweenness is critical for coupling; Out-Degree = efferent coupling
    criteria_maintainability: List[List[float]] = None
    
    # Availability: Articulation Score (AP_c), Bridge Ratio (BR), Importance (IM)
    # AP_c is now continuous but still the dominant SPOF indicator
    criteria_availability: List[List[float]] = None
    
    # Vulnerability: Eigenvector (EV), Closeness (CL), Out-Degree (OD)
    # Out-Degree = attack surface (outbound traversal paths)
    criteria_vulnerability: List[List[float]] = None
    
    # Overall Quality: Reliability (R), Maintainability (M), Availability (A), Vulnerability (V)
    criteria_overall: List[List[float]] = None

    # Topic QoS Importance: Reliability (Rel), Durability (Dur), Priority (Pri)
    # Justifies the 0.30/0.40/0.30 split used in Phase 4 modeling.
    criteria_topic_qos: List[List[float]] = None

    def __post_init__(self):
        # Default initialization if None
        if self.criteria_reliability is None:
            self.criteria_reliability = [
                # PR   RPR  ID
                [1.0, 2.0, 2.0],  # PR
                [0.5, 1.0, 1.0],  # RPR
                [0.5, 1.0, 1.0],  # ID
            ]
            
        if self.criteria_maintainability is None:
            self.criteria_maintainability = [
                # BT   OD   CC
                [1.0, 2.0, 3.0],  # BT (High coupling impact)
                [0.5, 1.0, 2.0],  # OD (Efferent coupling)
                [0.33, 0.5, 1.0], # CC (Modularity indicator)
            ]
            
        if self.criteria_availability is None:
            self.criteria_availability = [
                # AP_c  BR   IM
                [1.0, 3.0, 5.0],  # AP_c (Critical SPOF, now continuous)
                [0.33, 1.0, 2.0], # BR
                [0.2, 0.5, 1.0],  # IM
            ]

        if self.criteria_vulnerability is None:
            self.criteria_vulnerability = [
                # EV   CL   OD
                [1.0, 2.0, 2.0],  # EV (Strategic importance)
                [0.5, 1.0, 1.0],  # CL (Propagation speed)
                [0.5, 1.0, 1.0],  # OD (Attack surface)
            ]
            
        if self.criteria_overall is None:
            self.criteria_overall = [
                # R    M    A    V
                [1.0, 1.0, 1.0, 1.0],  # Balanced default
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
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
        
        # 1. Reliability Weights (PR, RPR, ID)
        w_rel = self._calculate_priority_vector(self.matrices.criteria_reliability)
        w_rel = self._shrink_weights(w_rel)
        
        # 2. Maintainability Weights (BT, OD, CC)
        w_main = self._calculate_priority_vector(self.matrices.criteria_maintainability)
        w_main = self._shrink_weights(w_main)
        
        # 3. Availability Weights (AP_c, BR, IM)
        w_avail = self._calculate_priority_vector(self.matrices.criteria_availability)
        w_avail = self._shrink_weights(w_avail)
        
        # 4. Vulnerability Weights (EV, CL, OD)
        w_vuln = self._calculate_priority_vector(self.matrices.criteria_vulnerability)
        w_vuln = self._shrink_weights(w_vuln)
        
        # 5. Overall Weights (R, M, A, V)
        w_over = self._calculate_priority_vector(self.matrices.criteria_overall)
        w_over = self._shrink_weights(w_over)
        
        return QualityWeights(
            # Reliability
            r_pagerank=w_rel[0],
            r_reverse_pagerank=w_rel[1],
            r_in_degree=w_rel[2],
            
            # Maintainability
            m_betweenness=w_main[0],
            m_out_degree=w_main[1],
            m_clustering=w_main[2],
            
            # Availability
            a_articulation=w_avail[0],
            a_bridge_ratio=w_avail[1],
            a_importance=w_avail[2],
            
            # Vulnerability
            v_eigenvector=w_vuln[0],
            v_closeness=w_vuln[1],
            v_out_degree=w_vuln[2],
            
            # Overall
            q_reliability=w_over[0],
            q_maintainability=w_over[1],
            q_availability=w_over[2],
            q_vulnerability=w_over[3]
        )