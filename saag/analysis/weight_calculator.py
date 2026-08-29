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

from saag.core.models import AHP_SHRINKAGE_LAMBDA


@dataclass
class QualityWeights:
    """
    Configurable weights for quality score computation.

    All weights should sum to 1.0 within each dimension.

    Design principles (RM model, ISO/IEC 25010:2023):
        - Metric orthogonality: Each raw structural metric is assigned to
          **exactly one** quality dimension. No metric is double-counted.
        - Continuous scoring: AP uses continuous fragmentation score (AP_c),
          not a binary articulation-point flag.
        - Hierarchical Reliability: Availability is a *sub-characteristic* of
          Reliability, not a peer dimension. R(v) = r_alpha*FT(v) + (1-r_alpha)*A(v),
          where FT is fault tolerance (propagation-reach signals) and A is the
          existing 5-term availability (SPOF-risk) formula.

    Note on Overall Weights (q_* parameters):
        Default weights are **derived from the retired 4-D AHP composite**
        (A=0.43, R=0.24, M=0.17, S=0.16) by dropping Vulnerability/Security and
        renormalising: q_reliability = (0.24+0.43)/0.84 = 0.80,
        q_maintainability = 0.17/0.84 = 0.20. With only two characteristics, a
        2x2 AHP matrix is consistent by construction (CR=0 for n<=2) and would
        contribute nothing — AHP is retained for the *intra*-dimension weights
        below, not for this composite.

    Note on r_alpha:
        Likewise derived, not AHP-fitted: r_alpha = 0.24/0.67 = 0.36. It is a
        DECLARED constant (see saag.core.quality_model.RELIABILITY_ALPHA) and
        should be included in any weight-sensitivity perturbation alongside q_*.
    """
    # Fault tolerance weights (fault propagation) — FT*(v) = RPR + DG_in + CDPot (v5)
    # ft_pagerank kept at 0.0 for backward-compat serialisation only.
    ft_pagerank: float = 0.0          # Deprecated (v4): superseded; kept for compat
    ft_reverse_pagerank: float = 0.45 # AHP leader: propagation reach (RPR); weight increased from 0.40
    ft_in_degree: float = 0.30        # Reinstatement (v5): count-based immediate-dependents signal
    ft_cdpot: float = 0.25            # Cascade Depth Potential (derived, depth signal)

    # Topic Fault Tolerance weights (separate formula: FT_topic(v) = FOC + CDPot_topic,
    # see analyzer.py Topic branch of _compute_rm). Kept as its own pair of fields
    # rather than reusing ft_reverse_pagerank/ft_in_degree/ft_cdpot above, since Topic
    # nodes have no reverse-PageRank/in-degree signal to weight — but declaring them
    # here (rather than as literals in analyzer.py) means _perturb_weights and any
    # future domain/AHP derivation can reach the Topic branch too.
    ft_topic_foc: float = 0.50        # Fan-Out Criticality (message-rate x subscriber reach)
    ft_topic_cdpot: float = 0.50      # CDPot_topic: FOC gated by publisher redundancy

    # Reliability = hierarchical combination of Fault Tolerance and Availability.
    # r_alpha weights FT within R; (1 - r_alpha) weights A. See class docstring.
    r_alpha: float = 0.36

    # Maintainability weights (coupling complexity) — M(v) v6
    # Formula: 0.35*BT + 0.30*w_out + 0.15*CQP + 0.12*CouplingRisk + 0.08*(1-CC)
    m_betweenness: float = 0.35      # AHP primary: structural bottleneck position
    m_w_out: float = 0.30            # QoS-weighted efferent coupling (promoted from 'Reported only')
    m_code_quality_penalty: float = 0.15  # Code Quality Penalty (CQP): complexity + instability + LCOM composite
    m_coupling_risk: float = 0.12    # CouplingRisk: afferent/efferent imbalance signal
    m_clustering: float = 0.08       # (1-CC): direction-agnostic proxy; reduced weight
    # Deprecated in v5 — subsumed by m_w_out (QoS-aware). Kept for backward-compat serialisation.
    m_out_degree: float = 0.0
    
    # Availability weights (SPOF risk) — A(v) v4, a sub-characteristic of Reliability
    # Formula: 0.2563*AP_c_directed + 0.1998*QSPOF + 0.1998*BR + 0.2563*CDI + 0.0878*w(v)
    # v4 rebalance: CDI was previously AP-gated (0.0 for every non-articulation-point
    # node — the Application population had zero articulation points in 6/8 corpus
    # scenarios, so A(v) collapsed to ~0.05*w(v)). CDI is now computed for every node
    # in the main component (see StructuralAnalyzer._compute_continuous_ap_scores), so
    # it carries real, continuous SPOF-adjacent signal and is promoted to parity with
    # AP_c_directed. QSPOF/BR shrink proportionally; w(v) shrinks least since it was
    # already the deliberately-small operational term.
    a_ap_c_directed: float = 0.2563  # AHP co-primary: hard cut-vertex severity (binary-gated)
    a_qspof: float = 0.1998          # QoS-weighted SPOF: AP_c_directed * w(v)
    a_bridge_ratio: float = 0.1998   # Edge-level irrecoverability
    a_cdi: float = 0.2563            # AHP co-primary: continuous redundancy-deficit (ungated)
    a_qos_weight: float = 0.0878     # Operational weight contribution w(v) (Issue 5: decoupling)

    # Overall quality weights (sum should be 1.0)
    # Derived from the retired 4-D AHP composite by dropping Vulnerability/Security
    # and renormalising — see class docstring. Not AHP-fitted at this level.
    q_reliability: float = 0.80
    q_maintainability: float = 0.20

    # Impact score weights I(v) (sum should be 1.0)
    # Formally derived via AHP: Reachability > Fragmentation = Throughput > FlowDisruption
    i_reachability: float = 0.35
    i_fragmentation: float = 0.25
    i_throughput: float = 0.25
    i_flow_disruption: float = 0.15
    
    # Edge quality weights (sum should be 1.0)
    e_betweenness: float = 0.35      # Path importance
    e_bridge: float = 0.30           # SPOF risk
    e_endpoint: float = 0.20         # Connected node importance
    # Renamed from e_security: was already reused as a generic edge-weight
    # coefficient in the (retained) Maintainability formula, not exclusively
    # tied to the (retired) Security dimension.
    e_qos_weight: float = 0.15       # Operational/QoS edge-weight contribution


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

    No composite (Reliability-vs-Maintainability) matrix is stored here: with
    only two characteristics a 2x2 AHP matrix is consistent by construction
    (CR=0 for n<=2, see ``_calculate_consistency_ratio``) and would contribute
    nothing. ``QualityWeights.q_reliability``/``q_maintainability`` and
    ``r_alpha`` are DECLARED constants instead — see that class's docstring.

    Metric assignments (RM model):
        Fault Tolerance:  Reverse PageRank (RPR), In-Degree (DG_in), CDPot
        Maintainability:  Betweenness (BT), w_out (QoS-efferent), CouplingRisk (CR), (1-CC)
        Availability:     QSPOF, Bridge Ratio (BR), AP_c_directed, CDI
        Impact I*(v):     IR(v), IM(v) — multi-phenomenon unified ground truth
        Impact IR(v):     Cascade Reach (CR), Weighted Cascade Impact (WCI), Normalised Depth (ND)
    """

    # Fault Tolerance: Reverse PageRank (RPR), In-Degree (DG_in), CDPot
    # RPR: primary propagation reach
    # DG_in: count-based immediate-dependents signal
    # CDPot: derived depth signal (no new algorithm needed)
    criteria_fault_tolerance: List[List[float]] = None

    # Maintainability v5: Betweenness (BT), w_out (QoS-efferent), CouplingRisk (CR), (1-CC)
    # BT: structural bottleneck; w_out: QoS-weighted contracts; CR: imbalance signal; (1-CC): proxy
    criteria_maintainability: List[List[float]] = None

    # Availability v2: QSPOF, Bridge Ratio, AP_c_directed, CDI
    # QSPOF = AP_c_directed * w(v) — operationally weighted structural SPOF
    # AP_c_directed = max(AP_c_out, AP_c_in) — worst-case directional SPOF
    # CDI — connectivity degradation for non-AP hubs
    criteria_availability: List[List[float]] = None

    # Topic QoS Importance: Reliability (Rel), Durability (Dur), Priority (Pri)
    # Justifies the 0.24/0.62/0.14 split used in Phase 3 modeling (Intrinsic
    # Weight Computation, docs/graph-model.md §4.3), i.e. QoSPolicy.W_RELIABILITY/
    # W_DURABILITY/W_PRIORITY in saag.core.models. Enforced by
    # tests/test_ahp_shrinkage.py::test_topic_qos_matrix_reproduces_shipped_weights
    # via AHPProcessor.compute_topic_qos_weights().
    #
    # An earlier version of this matrix ([[1,0.75,1],[1.33,1,1.33],[1,0.75,1]])
    # was solved backward from the target vector (0.30, 0.40, 0.30) rather than
    # stated independently, so its near-zero CR was an artifact of construction,
    # not evidence of a genuine judgement -- a matrix filled in to reproduce a
    # chosen answer is consistent almost by definition. The matrix below states
    # three pairwise judgements independently, grounded in DDS QoS semantics
    # (durability governs whether data survives at all; reliability and
    # transport priority both govern in-flight delivery quality, with
    # reliability -- an unconditional guarantee -- weighing somewhat more than
    # priority, which only governs relative scheduling under contention), and
    # is not forced to reproduce any particular target: its CR is small but
    # genuinely nonzero (~0.016), reflecting the mild real inconsistency among
    # three independently-stated judgements rather than perfect-by-construction
    # agreement.
    criteria_topic_qos: List[List[float]] = None

    # Impact (I(v)): Reachability (RL), Fragmentation (FR), Throughput (TL), Flow Disruption (FD)
    # Justifies the 0.35/0.25/0.25/0.15 split used in composite_impact.
    criteria_impact: List[List[float]] = None

    def __post_init__(self):
        # Default initialization if None
        if self.criteria_fault_tolerance is None:
            self.criteria_fault_tolerance = [
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
                [1.0,  1.4,   1.4,   1.0,   7.0],  # AP_c_d: hard cut-vertex severity (co-primary)
                [0.71, 1.0,   1.0,   0.71,  5.0],  # QSPOF: QoS-weighted SPOF
                [0.71, 1.0,   1.0,   0.71,  5.0],  # BR: Multi-edge brittleness
                [1.0,  1.4,   1.4,   1.0,   7.0],  # CDI: continuous redundancy deficit (co-primary)
                [0.14, 0.2,   0.2,   0.14,  1.0],  # w: Pure operational priority
            ]
            # v4: CDI judged equal importance to AP_c_d (was 1/3.5 = "weakly less
            # important"), since CDI now fires for every node in the main component
            # rather than only articulation points (see StructuralAnalyzer docstring).
            # Geometric mean → approx [0.2804, 0.1998, 0.1998, 0.2804, 0.0397] before
            # shrinkage (CR ≈ 0, matrix is symmetric by row-pair construction). With
            # shrinkage λ=0.7, weighted toward uniform (0.2):
            # [0.2563, 0.1998, 0.1998, 0.2563, 0.0878]

        if self.criteria_topic_qos is None:
            self.criteria_topic_qos = [
                # Rel   Dur   Pri
                [1.0,  1/3,  2.0],  # Rel: moderately less critical than Dur; moderately more than Pri
                [3.0,  1.0,  4.0],  # Dur: moderately more critical than Rel; much more than Pri
                [0.5,  0.25, 1.0],  # Pri: least critical (scheduling-only signal)
            ]
            # Geometric-mean priority vector ≈ [0.238, 0.625, 0.136] -> rounds to
            # the shipped (0.24, 0.62, 0.14). lambda_max ≈ 3.018, CR ≈ 0.016 —
            # small and genuinely nonzero, not the ≈0 a backward-solved matrix
            # produces (see the class-level docstring above).
            
        if self.criteria_impact is None:
            self.criteria_impact = [
                # RL    FR    TL    FD
                [1.0,  1.5,  1.5,  4.0],   # RL: most direct connectivity loss
                [0.67, 1.0,  1.0,  2.5],   # FR: graph partition severity
                [0.67, 1.0,  1.0,  2.5],   # TL: topic-weight disruption (equal to FR)
                [0.25, 0.4,  0.4,  1.0],   # FD: flow-triple breakage (secondary signal)
            ]
            # Raw AHP weights (geometric mean): ≈ [0.393, 0.25, 0.25, 0.107]
            # After λ=0.7 shrinkage toward uniform (0.25): ≈ [0.35, 0.25, 0.25, 0.15]


class AHPProcessor:
    """
    Calculates weights from pairwise comparison matrices with optional shrinkage.
    
    Shrinkage (blending) addresses methodological liability by formally 
    reconciling pure AHP weights with a uniform prior.
    """
    
    def __init__(self, matrices: AHPMatrices = None, shrinkage_factor: float = AHP_SHRINKAGE_LAMBDA):
        self.matrices = matrices or AHPMatrices()
        self.shrinkage_factor = shrinkage_factor

    def _shrink_weights(self, weights: List[float]) -> List[float]:
        """
        Blends AHP weights with a uniform prior using mixing coefficient lambda (λ).
        w_final = λ * w_ahp + (1 - λ) * w_uniform
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
        """Process all matrices and return a populated QualityWeights object.

        The composite weights (q_reliability, q_maintainability) and r_alpha are
        NOT computed here — with only two characteristics a composite AHP matrix
        is consistent by construction and contributes nothing (see AHPMatrices
        docstring). They are left at the QualityWeights dataclass defaults.
        """

        # 1. Fault Tolerance Weights v5 (RPR, DG_in, CDPot)
        w_ft = self._calculate_priority_vector(self.matrices.criteria_fault_tolerance)
        w_ft = self._shrink_weights(w_ft)

        # 2. Maintainability Weights v6 (BT, w_out, CQP, CouplingRisk, (1-CC))
        w_main = self._calculate_priority_vector(self.matrices.criteria_maintainability)
        w_main = self._shrink_weights(w_main)

        # 3. Availability Weights (AP_c, BR, w)
        w_avail = self._calculate_priority_vector(self.matrices.criteria_availability)
        w_avail = self._shrink_weights(w_avail)

        # 4. Impact Weights (RL, FR, TL) - Added for formal derivation
        w_impact = self._calculate_priority_vector(self.matrices.criteria_impact)
        w_impact = self._shrink_weights(w_impact)

        return QualityWeights(
            # Fault Tolerance v5: (RPR, DG_in, CDPot)
            ft_pagerank=0.0,               # Deprecated
            ft_reverse_pagerank=w_ft[0],   # RPR — primary (0.45)
            ft_in_degree=w_ft[1],          # DG_in — count-based immediate dependents (0.30)
            ft_cdpot=w_ft[2],              # Cascade Depth Potential (0.25)

            # Maintainability v6: (BT, w_out, CQP, CouplingRisk, (1-CC))
            m_betweenness=w_main[0],
            m_w_out=w_main[1],
            m_code_quality_penalty=w_main[2],
            m_coupling_risk=w_main[3],
            m_clustering=w_main[4],
            m_out_degree=0.0,               # Deprecated in v5+

            # Availability v4: (AP_c_directed, QSPOF, BR, CDI, w)
            a_ap_c_directed=w_avail[0],    # Hard cut-vertex severity, co-primary (0.2563)
            a_qspof=w_avail[1],             # QoS-weighted SPOF (0.1998)
            a_bridge_ratio=w_avail[2],      # Multi-edge brittleness (0.1998)
            a_cdi=w_avail[3],               # Continuous redundancy deficit, co-primary (0.2563)
            a_qos_weight=w_avail[4],        # Pure operational priority (0.0878)

            # Impact — all four criteria; i_flow_disruption used to be dropped
            # here, so the fourth AHP weight was computed and then discarded.
            i_reachability=w_impact[0],
            i_fragmentation=w_impact[1],
            i_throughput=w_impact[2],
            i_flow_disruption=w_impact[3],

            # Overall (q_reliability, q_maintainability) and r_alpha: not AHP-derived,
            # left at dataclass defaults (0.80, 0.20, 0.36) — see docstring.
        )

    def compute_topic_qos_weights(self) -> Dict[str, float]:
        """Raw (unshrunk) AHP-derived weights for Topic QoS sub-criteria.

        Not shrunk: ``saag.core.models.QoSPolicy.W_RELIABILITY``/
        ``W_DURABILITY``/``W_PRIORITY`` are the single runtime source of truth
        for w(topic) (graph-construction Phase 3); this method exists to prove
        those shipped constants are what a consistent Saaty matrix produces,
        not to feed them back into the graph-construction path. See
        ``tests/test_ahp_shrinkage.py::test_topic_qos_matrix_reproduces_shipped_weights``.
        """
        w = self._calculate_priority_vector(self.matrices.criteria_topic_qos)
        cr = self._calculate_consistency_ratio(self.matrices.criteria_topic_qos, w)
        return {
            "reliability": w[0],
            "durability": w[1],
            "priority": w[2],
            "consistency_ratio": cr,
        }
