"""
Graph Metrics Data Classes

Data structures for storing and manipulating graph analysis metrics.
Provides clean separation between raw structural metrics and quality scores.

Components:
    - StructuralMetrics: Raw topological metrics for nodes
    - EdgeMetrics: Raw topological metrics for edges
    - GraphSummary: Overall graph statistics
    - QualityScores: R, M, A, V quality scores
    - QualityLevels: Classification levels for quality dimensions
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple

from .classifier import CriticalityLevel


# =============================================================================
# Structural Metrics (Raw Graph Metrics)
# =============================================================================

@dataclass
class StructuralMetrics:
    """
    Raw topological metrics for a single graph component (node).
    
    Metrics are organized by category:
    - Centrality: Importance measures (PageRank, Betweenness, etc.)
    - Degree: Connectivity measures (in/out degree)
    - Resilience: Redundancy and fault tolerance indicators
    - Weights: Component and dependency weights
    """
    id: str
    type: str  # Application, Node, Broker, Topic
    
    # === Centrality Metrics ===
    pagerank: float = 0.0           # Global importance in dependency graph
    reverse_pagerank: float = 0.0   # Influence on failure propagation
    betweenness: float = 0.0        # Bridge/bottleneck centrality
    closeness: float = 0.0          # Average distance to all nodes
    eigenvector: float = 0.0        # Influence via neighbor importance
    
    # === Degree Metrics ===
    degree: float = 0.0             # Total degree centrality (normalized)
    in_degree: float = 0.0          # Normalized in-degree (who depends on this)
    out_degree: float = 0.0         # Normalized out-degree (this depends on)
    in_degree_raw: int = 0          # Raw in-degree count
    out_degree_raw: int = 0         # Raw out-degree count
    
    # === Resilience Metrics ===
    clustering_coefficient: float = 0.0  # Local redundancy (higher = more alternatives)
    is_articulation_point: bool = False  # Removal disconnects graph (SPOF)
    is_isolated: bool = False            # No connections
    bridge_count: int = 0                # Number of bridges touching this node
    bridge_ratio: float = 0.0            # Fraction of edges that are bridges
    
    # === Weights ===
    weight: float = 1.0                  # Intrinsic component weight
    dependency_weight_in: float = 0.0    # Sum of incoming dependency weights
    dependency_weight_out: float = 0.0   # Sum of outgoing dependency weights
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def total_degree_raw(self) -> int:
        """Raw total degree (in + out)."""
        return self.in_degree_raw + self.out_degree_raw
    
    @property
    def is_hub(self) -> bool:
        """Check if this is a hub (high in-degree)."""
        return self.in_degree > 0.5 or self.in_degree_raw > 10
    
    @property
    def is_sink(self) -> bool:
        """Check if this is a sink (no outgoing edges)."""
        return self.out_degree_raw == 0 and self.in_degree_raw > 0
    
    @property
    def is_source(self) -> bool:
        """Check if this is a source (no incoming edges)."""
        return self.in_degree_raw == 0 and self.out_degree_raw > 0


@dataclass
class EdgeMetrics:
    """
    Raw topological metrics for a single edge (dependency).
    
    Tracks edge-level importance and criticality indicators.
    """
    source: str
    target: str
    source_type: str
    target_type: str
    dependency_type: str  # app_to_app, node_to_node, app_to_broker, node_to_broker
    
    # === Metrics ===
    betweenness: float = 0.0  # Edge betweenness centrality
    is_bridge: bool = False   # Removal disconnects graph
    weight: float = 1.0       # Edge weight
    
    @property
    def key(self) -> Tuple[str, str]:
        """Unique identifier tuple for the edge."""
        return (self.source, self.target)
    
    @property
    def id(self) -> str:
        """String identifier for the edge."""
        return f"{self.source}->{self.target}"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphSummary:
    """
    Summary statistics for an analyzed graph layer.
    
    Provides high-level insights about graph structure and health.
    """
    layer: str
    nodes: int = 0
    edges: int = 0
    density: float = 0.0
    avg_degree: float = 0.0
    avg_clustering: float = 0.0
    is_connected: bool = False
    num_components: int = 0
    num_articulation_points: int = 0
    num_bridges: int = 0
    diameter: Optional[int] = None
    avg_path_length: Optional[float] = None
    
    # Breakdown by type
    node_types: Dict[str, int] = field(default_factory=dict)
    edge_types: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def spof_ratio(self) -> float:
        """Ratio of single points of failure (articulation points) to total nodes."""
        return self.num_articulation_points / self.nodes if self.nodes > 0 else 0.0
    
    @property
    def bridge_ratio(self) -> float:
        """Ratio of bridges to total edges."""
        return self.num_bridges / self.edges if self.edges > 0 else 0.0
    
    @property
    def connectivity_health(self) -> str:
        """Simple connectivity health indicator."""
        if not self.is_connected:
            return "DISCONNECTED"
        if self.spof_ratio > 0.2:
            return "FRAGILE"
        if self.spof_ratio > 0.1:
            return "MODERATE"
        return "ROBUST"


# =============================================================================
# Quality Scores (Computed from Structural Metrics)
# =============================================================================

@dataclass
class QualityScores:
    """
    Composite quality scores for R, M, A dimensions.
    
    Formulas:
        R(v) = w_pr·PR + w_rpr·RPR + w_in·InDeg      (Reliability)
        M(v) = w_bt·BC + w_dg·Deg + w_cl·(1-CC)      (Maintainability)
        A(v) = w_ap·AP + w_br·BR + w_imp·Imp         (Availability)
        V(v) = w_ev·Eig + w_cl·Close + w_in·InDeg    (Vulnerability)
        Q(v) = w_r·R + w_m·M + w_a·A                 (Overall)
    """
    reliability: float = 0.0        # Fault propagation risk
    maintainability: float = 0.0    # Change/coupling complexity
    availability: float = 0.0       # Single point of failure risk
    vulnerability: float = 0.0      # Exposure and attack surface risk
    overall: float = 0.0            # Combined criticality
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "reliability": round(self.reliability, 4),
            "maintainability": round(self.maintainability, 4),
            "availability": round(self.availability, 4),
            "vulnerability": round(self.vulnerability, 4),
            "overall": round(self.overall, 4),
        }


@dataclass
class QualityLevels:
    """
    Classified levels for each quality dimension.
    
    Levels are determined by box-plot classification based on
    score distribution, not static thresholds.
    """
    reliability: CriticalityLevel = CriticalityLevel.MINIMAL
    maintainability: CriticalityLevel = CriticalityLevel.MINIMAL
    availability: CriticalityLevel = CriticalityLevel.MINIMAL
    vulnerability: CriticalityLevel = CriticalityLevel.MINIMAL
    overall: CriticalityLevel = CriticalityLevel.MINIMAL
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "reliability": self.reliability.value,
            "maintainability": self.maintainability.value,
            "availability": self.availability.value,
            "vulnerability": self.vulnerability.value,
            "overall": self.overall.value,
        }
    
    def max_level(self) -> CriticalityLevel:
        """Return the highest criticality level across all dimensions."""
        return max(
            [self.reliability, self.maintainability, self.availability, self.vulnerability],
            key=lambda x: x.numeric
        )
    
    def requires_attention(self) -> bool:
        """Check if any dimension requires attention (CRITICAL or HIGH)."""
        return self.max_level() >= CriticalityLevel.HIGH


@dataclass
class ComponentQuality:
    """
    Complete quality assessment for a single component.
    
    Combines raw structural metrics with computed quality scores
    and classification levels.
    """
    id: str
    type: str
    scores: QualityScores
    levels: QualityLevels
    structural: StructuralMetrics
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "scores": self.scores.to_dict(),
            "levels": self.levels.to_dict(),
            "structural": self.structural.to_dict(),
        }
    
    @property
    def is_critical(self) -> bool:
        """Check if component is classified as CRITICAL overall."""
        return self.levels.overall == CriticalityLevel.CRITICAL
    
    @property
    def requires_attention(self) -> bool:
        """Check if component requires attention (CRITICAL or HIGH)."""
        return self.levels.overall >= CriticalityLevel.HIGH


@dataclass
class EdgeQuality:
    """
    Quality assessment for a single edge (dependency).
    """
    source: str
    target: str
    source_type: str
    target_type: str
    dependency_type: str
    scores: QualityScores
    level: CriticalityLevel = CriticalityLevel.MINIMAL
    structural: Optional[EdgeMetrics] = None
    
    @property
    def id(self) -> str:
        return f"{self.source}->{self.target}"
    
    @property
    def key(self) -> Tuple[str, str]:
        return (self.source, self.target)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "dependency_type": self.dependency_type,
            "scores": self.scores.to_dict(),
            "level": self.level.value,
        }
        if self.structural:
            result["structural"] = self.structural.to_dict()
        return result


@dataclass
class ClassificationSummary:
    """
    Summary of classification distribution.
    """
    total_components: int = 0
    total_edges: int = 0
    component_distribution: Dict[str, int] = field(default_factory=dict)
    edge_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_components": self.total_components,
            "total_edges": self.total_edges,
            "component_distribution": self.component_distribution,
            "edge_distribution": self.edge_distribution,
        }
    
    @property
    def critical_components(self) -> int:
        return self.component_distribution.get("critical", 0)
    
    @property
    def high_components(self) -> int:
        return self.component_distribution.get("high", 0)
    
    @property
    def requires_attention(self) -> int:
        """Components requiring attention (CRITICAL + HIGH)."""
        return self.critical_components + self.high_components
