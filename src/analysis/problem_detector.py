"""
Problem Detector

Identifies architectural smells, risks, and anti-patterns using quality
analysis results. Relies on box-plot classification rather than static thresholds.

Problem Categories:
- Availability: SPOFs, bridges, connectivity risks
- Reliability: Failure propagation hubs, cascade risks
- Maintainability: God components, high coupling, bottlenecks
- Architecture: Structural issues, design smells

Author: Software-as-a-Graph Research Project
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
from enum import Enum

from .quality_analyzer import QualityAnalysisResult, ComponentQuality, EdgeQuality
from .classifier import CriticalityLevel


class ProblemCategory(Enum):
    """Categories of detected problems."""
    AVAILABILITY = "Availability"
    RELIABILITY = "Reliability"
    MAINTAINABILITY = "Maintainability"
    ARCHITECTURE = "Architecture"


class ProblemSeverity(Enum):
    """Severity levels for problems."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    
    @property
    def priority(self) -> int:
        return {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}[self.value]


@dataclass
class DetectedProblem:
    """A detected architectural problem or risk."""
    
    entity_id: str
    entity_type: str        # Component, Edge, Layer, System
    category: str           # Availability, Reliability, Maintainability, Architecture
    severity: str           # CRITICAL, HIGH, MEDIUM, LOW
    name: str               # Short problem name
    description: str        # Detailed description
    recommendation: str     # Suggested fix/mitigation
    evidence: Dict[str, Any] = field(default_factory=dict)  # Supporting data
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @property
    def priority(self) -> int:
        """Numeric priority for sorting (higher = more urgent)."""
        return {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(self.severity, 0)


@dataclass
class ProblemSummary:
    """Summary of all detected problems."""
    
    total_problems: int
    by_severity: Dict[str, int]
    by_category: Dict[str, int]
    affected_components: int
    affected_edges: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProblemDetector:
    """
    Detects architectural problems and risks from quality analysis results.
    
    Uses box-plot classification levels (not static thresholds) to identify
    components and edges that require attention.
    """
    
    def __init__(self):
        """Initialize the problem detector."""
        pass
    
    def detect(self, quality_result: QualityAnalysisResult) -> List[DetectedProblem]:
        """
        Detect problems from quality analysis results.
        
        Args:
            quality_result: Result from QualityAnalyzer
            
        Returns:
            List of detected problems, sorted by severity
        """
        problems: List[DetectedProblem] = []
        
        # 1. Component-level problems
        problems.extend(self._detect_component_problems(quality_result.components))
        
        # 2. Edge-level problems
        problems.extend(self._detect_edge_problems(quality_result.edges))
        
        # 3. System-level problems (patterns across multiple components)
        problems.extend(self._detect_system_problems(
            quality_result.components, 
            quality_result.edges,
            quality_result
        ))
        
        # Sort by severity (CRITICAL first) then by entity_id
        problems.sort(key=lambda p: (-p.priority, p.entity_id))
        
        return problems
    
    def _detect_component_problems(
        self, 
        components: List[ComponentQuality]
    ) -> List[DetectedProblem]:
        """Detect problems at the component level."""
        problems = []
        
        for c in components:
            # === AVAILABILITY PROBLEMS ===
            
            # A1: Single Point of Failure (Articulation Point)
            if c.structural.is_articulation_point:
                severity = "CRITICAL" if c.levels.availability == CriticalityLevel.CRITICAL else "HIGH"
                problems.append(DetectedProblem(
                    entity_id=c.id,
                    entity_type=c.type,
                    category=ProblemCategory.AVAILABILITY.value,
                    severity=severity,
                    name="Single Point of Failure (SPOF)",
                    description=(
                        f"Component '{c.id}' is a structural cut-vertex (articulation point). "
                        f"Removing this component disconnects the dependency graph, "
                        f"partitioning the system into isolated clusters."
                    ),
                    recommendation=(
                        "Introduce redundancy through: (1) Backup/standby instances, "
                        "(2) Alternative routing paths, (3) Service replication, or "
                        "(4) Dependency decoupling using an event bus."
                    ),
                    evidence={
                        "is_articulation_point": True,
                        "availability_score": c.scores.availability,
                        "in_degree": c.structural.in_degree_raw,
                        "out_degree": c.structural.out_degree_raw,
                    }
                ))
            
            # A2: High Bridge Ratio (controls many critical links)
            elif c.levels.availability >= CriticalityLevel.HIGH and c.structural.bridge_ratio > 0.3:
                problems.append(DetectedProblem(
                    entity_id=c.id,
                    entity_type=c.type,
                    category=ProblemCategory.AVAILABILITY.value,
                    severity="HIGH",
                    name="Structural Bridge Controller",
                    description=(
                        f"Component '{c.id}' controls {c.structural.bridge_ratio:.0%} of its "
                        f"connected edges as bridges. Failure here partitions system clusters."
                    ),
                    recommendation=(
                        "Review network topology and add redundant paths to reduce "
                        "single-link dependencies."
                    ),
                    evidence={
                        "bridge_ratio": c.structural.bridge_ratio,
                        "bridge_count": c.structural.bridge_count,
                        "availability_score": c.scores.availability,
                    }
                ))
            
            # A3: Isolated Component
            if c.structural.is_isolated:
                problems.append(DetectedProblem(
                    entity_id=c.id,
                    entity_type=c.type,
                    category=ProblemCategory.AVAILABILITY.value,
                    severity="MEDIUM",
                    name="Isolated Component",
                    description=(
                        f"Component '{c.id}' has no incoming or outgoing dependencies. "
                        f"It may be orphaned or misconfigured."
                    ),
                    recommendation=(
                        "Verify component configuration. If intentionally isolated, "
                        "document the rationale. Otherwise, establish proper dependencies."
                    ),
                    evidence={
                        "in_degree": c.structural.in_degree_raw,
                        "out_degree": c.structural.out_degree_raw,
                    }
                ))
            
            # === RELIABILITY PROBLEMS ===
            
            # R1: Failure Propagation Hub
            if c.levels.reliability >= CriticalityLevel.HIGH:
                severity = c.levels.reliability.value.upper()
                problems.append(DetectedProblem(
                    entity_id=c.id,
                    entity_type=c.type,
                    category=ProblemCategory.RELIABILITY.value,
                    severity=severity,
                    name="Failure Propagation Hub",
                    description=(
                        f"Component '{c.id}' has high influence on downstream components "
                        f"(Reverse PageRank). Failures here propagate widely through the system."
                    ),
                    recommendation=(
                        "Implement resilience patterns: (1) Circuit breakers to isolate failures, "
                        "(2) Bulkheads to limit blast radius, (3) Retry with exponential backoff, "
                        "(4) Graceful degradation for partial failures."
                    ),
                    evidence={
                        "reliability_score": c.scores.reliability,
                        "pagerank": c.structural.pagerank,
                        "reverse_pagerank": c.structural.reverse_pagerank,
                        "dependents": c.structural.in_degree_raw,
                    }
                ))
            
            # R2: High Fan-In (Many Dependents)
            if c.structural.in_degree_raw >= 5 and c.levels.reliability >= CriticalityLevel.MEDIUM:
                if not any(p.entity_id == c.id and "Propagation" in p.name for p in problems):
                    problems.append(DetectedProblem(
                        entity_id=c.id,
                        entity_type=c.type,
                        category=ProblemCategory.RELIABILITY.value,
                        severity="MEDIUM",
                        name="High Fan-In Component",
                        description=(
                            f"Component '{c.id}' has {c.structural.in_degree_raw} direct dependents. "
                            f"Changes or failures affect many downstream consumers."
                        ),
                        recommendation=(
                            "Consider: (1) Caching to reduce load, (2) Rate limiting to prevent "
                            "overload, (3) Versioned APIs for safe evolution."
                        ),
                        evidence={
                            "in_degree": c.structural.in_degree_raw,
                            "reliability_score": c.scores.reliability,
                        }
                    ))
            
            # === MAINTAINABILITY PROBLEMS ===
            
            # M1: God Component (High Coupling)
            if c.levels.maintainability >= CriticalityLevel.HIGH:
                severity = c.levels.maintainability.value.upper()
                problems.append(DetectedProblem(
                    entity_id=c.id,
                    entity_type=c.type,
                    category=ProblemCategory.MAINTAINABILITY.value,
                    severity=severity,
                    name="God Component (High Coupling)",
                    description=(
                        f"Component '{c.id}' has high betweenness centrality and degree, "
                        f"indicating tight coupling with many parts of the system. "
                        f"Changes here have wide-reaching ripple effects."
                    ),
                    recommendation=(
                        "Refactor to reduce coupling: (1) Split responsibilities into focused services, "
                        "(2) Introduce abstraction layers, (3) Use event-driven communication "
                        "to decouple producers and consumers."
                    ),
                    evidence={
                        "maintainability_score": c.scores.maintainability,
                        "betweenness": c.structural.betweenness,
                        "total_degree": c.structural.in_degree_raw + c.structural.out_degree_raw,
                        "clustering": c.structural.clustering_coefficient,
                    }
                ))
            
            # M2: Bottleneck (High Betweenness)
            if (c.structural.betweenness > 0.2 and 
                c.levels.maintainability >= CriticalityLevel.MEDIUM and
                not any(p.entity_id == c.id and "God Component" in p.name for p in problems)):
                problems.append(DetectedProblem(
                    entity_id=c.id,
                    entity_type=c.type,
                    category=ProblemCategory.MAINTAINABILITY.value,
                    severity="MEDIUM",
                    name="Communication Bottleneck",
                    description=(
                        f"Component '{c.id}' lies on many shortest paths between other components "
                        f"(betweenness={c.structural.betweenness:.3f}). It acts as a communication "
                        f"bottleneck that many data flows must traverse."
                    ),
                    recommendation=(
                        "Consider: (1) Adding direct connections between frequently communicating "
                        "components, (2) Implementing caching at this node, (3) Load balancing "
                        "across multiple instances."
                    ),
                    evidence={
                        "betweenness": c.structural.betweenness,
                        "maintainability_score": c.scores.maintainability,
                    }
                ))
            
            # M3: Poor Modularity (Low Clustering)
            if (c.structural.clustering_coefficient < 0.1 and 
                c.structural.in_degree_raw + c.structural.out_degree_raw >= 3 and
                c.levels.maintainability >= CriticalityLevel.MEDIUM):
                problems.append(DetectedProblem(
                    entity_id=c.id,
                    entity_type=c.type,
                    category=ProblemCategory.MAINTAINABILITY.value,
                    severity="LOW",
                    name="Poor Local Modularity",
                    description=(
                        f"Component '{c.id}' has low clustering coefficient "
                        f"({c.structural.clustering_coefficient:.3f}), indicating its neighbors "
                        f"are not well connected. This suggests a hub-and-spoke pattern "
                        f"without redundant paths."
                    ),
                    recommendation=(
                        "Review module boundaries. Consider grouping related components "
                        "into cohesive modules with internal redundancy."
                    ),
                    evidence={
                        "clustering_coefficient": c.structural.clustering_coefficient,
                        "degree": c.structural.in_degree_raw + c.structural.out_degree_raw,
                    }
                ))
        
        return problems
    
    def _detect_edge_problems(
        self, 
        edges: List[EdgeQuality]
    ) -> List[DetectedProblem]:
        """Detect problems at the edge (dependency) level."""
        problems = []
        
        for e in edges:
            # E1: Critical Bridge Dependency
            if e.is_bridge and e.level >= CriticalityLevel.HIGH:
                problems.append(DetectedProblem(
                    entity_id=e.id,
                    entity_type="Dependency",
                    category=ProblemCategory.AVAILABILITY.value,
                    severity=e.level.value.upper(),
                    name="Critical Bridge Dependency",
                    description=(
                        f"Dependency '{e.id}' is a bridge edge connecting critical endpoints. "
                        f"Failure of this link partitions the system."
                    ),
                    recommendation=(
                        "Add redundant communication paths between these components. "
                        "Consider asynchronous messaging or alternative routing."
                    ),
                    evidence={
                        "is_bridge": True,
                        "edge_score": e.scores.overall,
                        "source": e.source,
                        "target": e.target,
                    }
                ))
            
            # E2: High-Weight Dependency
            elif e.level >= CriticalityLevel.HIGH and e.weight > 0.7:
                problems.append(DetectedProblem(
                    entity_id=e.id,
                    entity_type="Dependency",
                    category=ProblemCategory.RELIABILITY.value,
                    severity="HIGH",
                    name="High-Weight Critical Dependency",
                    description=(
                        f"Dependency '{e.id}' carries high weight ({e.weight:.2f}), "
                        f"indicating critical data flow or QoS requirements."
                    ),
                    recommendation=(
                        "Ensure high bandwidth, monitoring, and prioritization for this link. "
                        "Implement QoS guarantees at the infrastructure level."
                    ),
                    evidence={
                        "weight": e.weight,
                        "edge_score": e.scores.overall,
                        "type": e.type,
                    }
                ))
        
        return problems
    
    def _detect_system_problems(
        self,
        components: List[ComponentQuality],
        edges: List[EdgeQuality],
        result: QualityAnalysisResult
    ) -> List[DetectedProblem]:
        """Detect system-wide patterns and anti-patterns."""
        problems = []
        summary = result.classification_summary
        
        # S1: High SPOF Density
        spof_count = sum(1 for c in components if c.structural.is_articulation_point)
        total_components = len(components)
        
        if total_components > 0:
            spof_ratio = spof_count / total_components
            if spof_ratio > 0.2 and spof_count >= 3:
                problems.append(DetectedProblem(
                    entity_id="SYSTEM",
                    entity_type="System",
                    category=ProblemCategory.ARCHITECTURE.value,
                    severity="CRITICAL",
                    name="High SPOF Density",
                    description=(
                        f"System has {spof_count} articulation points ({spof_ratio:.0%} of components). "
                        f"This indicates fragile architecture with many single points of failure."
                    ),
                    recommendation=(
                        "Systematic architectural review required. Prioritize adding redundancy "
                        "to the most critical SPOFs identified in component-level analysis."
                    ),
                    evidence={
                        "spof_count": spof_count,
                        "total_components": total_components,
                        "spof_ratio": spof_ratio,
                    }
                ))
        
        # S2: High Critical Component Ratio
        critical_ratio = (summary.critical_components + summary.high_components) / max(total_components, 1)
        if critical_ratio > 0.3 and (summary.critical_components + summary.high_components) >= 3:
            problems.append(DetectedProblem(
                entity_id="SYSTEM",
                entity_type="System",
                category=ProblemCategory.ARCHITECTURE.value,
                severity="HIGH",
                name="Concentrated Criticality",
                description=(
                    f"{summary.critical_components} CRITICAL and {summary.high_components} HIGH "
                    f"criticality components ({critical_ratio:.0%} of system). "
                    f"Risk is concentrated rather than distributed."
                ),
                recommendation=(
                    "Review overall architecture for centralization anti-patterns. "
                    "Consider decomposition and load distribution strategies."
                ),
                evidence={
                    "critical_count": summary.critical_components,
                    "high_count": summary.high_components,
                    "ratio": critical_ratio,
                }
            ))
        
        # S3: High Bridge Density
        bridge_count = sum(1 for e in edges if e.is_bridge)
        total_edges = len(edges)
        
        if total_edges > 0:
            bridge_ratio = bridge_count / total_edges
            if bridge_ratio > 0.3 and bridge_count >= 3:
                problems.append(DetectedProblem(
                    entity_id="SYSTEM",
                    entity_type="System",
                    category=ProblemCategory.ARCHITECTURE.value,
                    severity="HIGH",
                    name="Sparse Connectivity (High Bridge Ratio)",
                    description=(
                        f"{bridge_count} of {total_edges} edges ({bridge_ratio:.0%}) are bridges. "
                        f"The system has minimal redundancy in its connection topology."
                    ),
                    recommendation=(
                        "Add redundant connections between critical components. "
                        "Consider mesh topology for core services."
                    ),
                    evidence={
                        "bridge_count": bridge_count,
                        "total_edges": total_edges,
                        "bridge_ratio": bridge_ratio,
                    }
                ))
        
        # S4: Dependency Imbalance (Star Pattern)
        if total_components >= 5:
            # Check for components with significantly higher connectivity
            degrees = [(c.id, c.structural.in_degree_raw + c.structural.out_degree_raw) 
                       for c in components]
            degrees.sort(key=lambda x: x[1], reverse=True)
            
            if degrees and len(degrees) >= 3:
                top_degree = degrees[0][1]
                median_idx = len(degrees) // 2
                median_degree = degrees[median_idx][1]
                
                if median_degree > 0 and top_degree >= 3 * median_degree:
                    problems.append(DetectedProblem(
                        entity_id="SYSTEM",
                        entity_type="System",
                        category=ProblemCategory.ARCHITECTURE.value,
                        severity="MEDIUM",
                        name="Star Topology Pattern",
                        description=(
                            f"Component '{degrees[0][0]}' has degree {top_degree}, significantly "
                            f"higher than median ({median_degree}). This indicates a star/hub topology "
                            f"where one component dominates connectivity."
                        ),
                        recommendation=(
                            "Evaluate if centralization is intentional. Consider distributing "
                            "responsibilities across multiple hubs or introducing a service mesh."
                        ),
                        evidence={
                            "hub_component": degrees[0][0],
                            "hub_degree": top_degree,
                            "median_degree": median_degree,
                        }
                    ))
        
        return problems
    
    def summarize(self, problems: List[DetectedProblem]) -> ProblemSummary:
        """
        Generate a summary of detected problems.
        """
        by_severity = {s.value: 0 for s in ProblemSeverity}
        by_category = {c.value: 0 for c in ProblemCategory}
        affected_components = set()
        affected_edges = set()
        
        for p in problems:
            by_severity[p.severity] = by_severity.get(p.severity, 0) + 1
            by_category[p.category] = by_category.get(p.category, 0) + 1
            
            if p.entity_type == "Dependency":
                affected_edges.add(p.entity_id)
            elif p.entity_type not in ("System", "Layer"):
                affected_components.add(p.entity_id)
        
        return ProblemSummary(
            total_problems=len(problems),
            by_severity=by_severity,
            by_category=by_category,
            affected_components=len(affected_components),
            affected_edges=len(affected_edges),
        )