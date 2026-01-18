"""
Analytic Hierarchy Process (AHP) Module

Provides functionality to calculate weights for quality attributes using the 
Analytic Hierarchy Process. This allows for more rigorous, relative importance-based
weight determination compared to arbitrary assignment.

The module uses the Geometric Mean method (approximate eigenvector) to calculate
weights from pairwise comparison matrices.
"""

import math
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class QualityWeights:
    """
    Configurable weights for quality score computation.
    
    All weights should sum to 1.0 within each dimension.
    """
    # Reliability weights (fault propagation)
    r_pagerank: float = 0.4
    r_reverse_pagerank: float = 0.35
    r_in_degree: float = 0.25
    
    # Maintainability weights (coupling complexity)
    m_betweenness: float = 0.4
    m_degree: float = 0.35
    m_clustering: float = 0.25  # Note: (1 - clustering) is used
    
    # Availability weights (SPOF risk)
    a_articulation: float = 0.5
    a_bridge_ratio: float = 0.3
    a_importance: float = 0.2  # Combined pagerank

    # Vulnerability weights (exposure risk)
    v_eigenvector: float = 0.4
    v_closeness: float = 0.3
    v_in_degree: float = 0.3
    
    # Overall quality weights
    q_reliability: float = 0.25
    q_maintainability: float = 0.25
    q_availability: float = 0.25
    q_vulnerability: float = 0.25

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
    """
    
    # Reliability: PageRank (PR), Reverse PageRank (RPR), In-Degree (ID)
    # PR is often slightly more important for global importance
    criteria_reliability: List[List[float]] = None
    
    # Maintainability: Betweenness (BT), Degree (DG), Clustering (CL)
    # Betweenness is critical for coupling
    criteria_maintainability: List[List[float]] = None
    
    # Availability: Articulation (AP), Bridge (BR), Importance (IM)
    # Articulation Points are critical SPOFs (Extreme importance)
    criteria_availability: List[List[float]] = None
    
    # Vulnerability: Eigenvector (EV), Closeness (CS), In-Degree (ID)
    criteria_vulnerability: List[List[float]] = None
    
    # Overall Quality: Reliability (R), Maintainability (M), Availability (A), Vulnerability (V)
    criteria_overall: List[List[float]] = None

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
                # BT   DG   CL
                [1.0, 2.0, 3.0],  # BT (High coupling impact)
                [0.5, 1.0, 2.0],  # DG
                [0.33, 0.5, 1.0], # CL
            ]
            
        if self.criteria_availability is None:
            self.criteria_availability = [
                # AP   BR   IM
                [1.0, 3.0, 5.0],  # AP (Critical SPOF)
                [0.33, 1.0, 2.0], # BR
                [0.2, 0.5, 1.0],  # IM
            ]

        if self.criteria_vulnerability is None:
            self.criteria_vulnerability = [
                # EV   CS   ID
                [1.0, 2.0, 2.0],  # EV
                [0.5, 1.0, 1.0],  # CS
                [0.5, 1.0, 1.0],  # ID
            ]
            
        if self.criteria_overall is None:
            self.criteria_overall = [
                # R    M    A    V
                [1.0, 1.0, 1.0, 1.0],  # Balanced default
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ]

class AHPProcessor:
    """Calculates weights from pairwise comparison matrices."""
    
    def __init__(self, matrices: AHPMatrices = None):
        self.matrices = matrices or AHPMatrices()

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
        
        # 1. Reliability Weights
        w_rel = self._calculate_priority_vector(self.matrices.criteria_reliability)
        
        # 2. Maintainability Weights
        w_main = self._calculate_priority_vector(self.matrices.criteria_maintainability)
        
        # 3. Availability Weights
        w_avail = self._calculate_priority_vector(self.matrices.criteria_availability)
        
        # 4. Vulnerability Weights
        w_vuln = self._calculate_priority_vector(self.matrices.criteria_vulnerability)
        
        # 5. Overall Weights
        w_over = self._calculate_priority_vector(self.matrices.criteria_overall)
        
        return QualityWeights(
            # Reliability
            r_pagerank=w_rel[0],
            r_reverse_pagerank=w_rel[1],
            r_in_degree=w_rel[2],
            
            # Maintainability
            m_betweenness=w_main[0],
            m_degree=w_main[1],
            m_clustering=w_main[2],
            
            # Availability
            a_articulation=w_avail[0],
            a_bridge_ratio=w_avail[1],
            a_importance=w_avail[2],
            
            # Vulnerability
            v_eigenvector=w_vuln[0],
            v_closeness=w_vuln[1],
            v_in_degree=w_vuln[2],
            
            # Overall
            q_reliability=w_over[0],
            q_maintainability=w_over[1],
            q_availability=w_over[2],
            q_vulnerability=w_over[3]
        )