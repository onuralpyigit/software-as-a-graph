"""
Problem Detector

Identifies architectural smells, risks, and anti-patterns using quality
analysis results. Uses box-plot classification rather than static thresholds.

Problem Categories:
    - AVAILABILITY: SPOFs, bridges, connectivity risks
    - RELIABILITY: Failure propagation hubs, cascade risks
    - MAINTAINABILITY: God components, high coupling, bottlenecks
    - ARCHITECTURE: Structural issues, design smells

Severity Levels:
    - CRITICAL: Immediate action required (system risk)
    - HIGH: Should be addressed soon (significant risk)
    - MEDIUM: Monitor and plan remediation
    - LOW: Minor concern, track for future
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import List, Dict, Any, Optional

from src.domain.models.criticality import CriticalityLevel
from src.domain.services.quality_analyzer import QualityAnalysisResult
from src.domain.models.metrics import ComponentQuality, EdgeQuality


class ProblemCategory(Enum):
    """Categories of detected architectural problems."""
    AVAILABILITY = "Availability"
    RELIABILITY = "Reliability"
    MAINTAINABILITY = "Maintainability"
    SECURITY = "Security"
    ARCHITECTURE = "Architecture"


class ProblemSeverity(Enum):
    """Severity levels for detected problems."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    
    @property
    def priority(self) -> int:
        """Numeric priority for sorting (higher = more urgent)."""
        return {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}[self.value]


@dataclass
class DetectedProblem:
    """
    A detected architectural problem or risk.
    
    Contains identification, classification, and remediation guidance.
    """
    entity_id: str
    entity_type: str        # Component, Edge, Layer, System
    category: str           # Availability, Reliability, Maintainability, Architecture
    severity: str           # CRITICAL, HIGH, MEDIUM, LOW
    name: str               # Short problem name
    description: str        # Detailed description
    recommendation: str     # Suggested fix/mitigation
    evidence: Dict[str, Any] = field(default_factory=dict)
    
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
    
    @property
    def has_critical(self) -> bool:
        return self.by_severity.get("CRITICAL", 0) > 0
    
    @property
    def requires_attention(self) -> int:
        """Count of CRITICAL and HIGH severity problems."""
        return self.by_severity.get("CRITICAL", 0) + self.by_severity.get("HIGH", 0)


class ProblemDetector:
    """
    Detects architectural problems and risks from quality analysis results.
    
    Uses box-plot classification levels (not static thresholds) to identify
    components and edges that require attention.
    
    Example:
        >>> detector = ProblemDetector()
        >>> problems = detector.detect(quality_result)
        >>> critical = [p for p in problems if p.severity == "CRITICAL"]
    """
    
    def _problem(
        self,
        entity_id: str,
        entity_type: str,
        category: ProblemCategory,
        severity: str,
        name: str,
        description: str,
        recommendation: str,
        **evidence
    ) -> DetectedProblem:
        """Factory method for creating DetectedProblem instances."""
        return DetectedProblem(
            entity_id=entity_id,
            entity_type=entity_type,
            category=category.value,
            severity=severity,
            name=name,
            description=description,
            recommendation=recommendation,
            evidence=evidence,
        )
    
    def detect(self, quality_result: QualityAnalysisResult) -> List[DetectedProblem]:
        """
        Detect all problems from quality analysis results.
        
        Args:
            quality_result: Result from QualityAnalyzer
            
        Returns:
            List of detected problems, sorted by severity (CRITICAL first)
        """
        problems: List[DetectedProblem] = []
        
        # Component-level problems
        problems.extend(self._detect_component_problems(quality_result.components))
        
        # Edge-level problems
        problems.extend(self._detect_edge_problems(quality_result.edges))
        
        # System-level problems (patterns across components)
        problems.extend(self._detect_system_problems(
            quality_result.components,
            quality_result.edges,
            quality_result
        ))
        
        # Sort by severity (CRITICAL first), then by entity_id
        problems.sort(key=lambda p: (-p.priority, p.entity_id))
        
        return problems
    
    def summarize(self, problems: List[DetectedProblem]) -> ProblemSummary:
        """
        Create a summary of detected problems.
        
        Args:
            problems: List of detected problems
            
        Returns:
            ProblemSummary with counts and breakdown
        """
        by_severity = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        by_category = {}
        affected_components = set()
        affected_edges = set()
        
        for p in problems:
            by_severity[p.severity] = by_severity.get(p.severity, 0) + 1
            by_category[p.category] = by_category.get(p.category, 0) + 1
            
            if p.entity_type == "Component":
                affected_components.add(p.entity_id)
            elif p.entity_type == "Edge":
                affected_edges.add(p.entity_id)
        
        return ProblemSummary(
            total_problems=len(problems),
            by_severity=by_severity,
            by_category=by_category,
            affected_components=len(affected_components),
            affected_edges=len(affected_edges),
        )
    
    def _detect_component_problems(
        self,
        components: List[ComponentQuality]
    ) -> List[DetectedProblem]:
        """Detect problems at the component level."""
        problems = []
        
        for c in components:
            # === AVAILABILITY PROBLEMS ===
            
            # Single Point of Failure (Articulation Point)
            if c.structural.is_articulation_point:
                severity = "CRITICAL" if c.levels.availability == CriticalityLevel.CRITICAL else "HIGH"
                problems.append(DetectedProblem(
                    entity_id=c.id,
                    entity_type="Component",
                    category=ProblemCategory.AVAILABILITY.value,
                    severity=severity,
                    name="Single Point of Failure (SPOF)",
                    description=(
                        f"'{c.id}' is an articulation point. Removing this {c.type.lower()} "
                        f"disconnects the dependency graph, partitioning the system into "
                        f"isolated clusters that cannot communicate."
                    ),
                    recommendation=(
                        "Introduce redundancy: (1) Deploy backup/standby instances, "
                        "(2) Add alternative communication paths, (3) Use an event bus "
                        "or service mesh for decoupling, (4) Implement circuit breakers."
                    ),
                    evidence={
                        "is_articulation_point": True,
                        "bridge_count": c.structural.bridge_count,
                        "availability_score": c.scores.availability,
                        "in_degree": c.structural.in_degree_raw,
                        "out_degree": c.structural.out_degree_raw,
                    }
                ))
            
            # High Bridge Ratio (many critical edges)
            elif c.structural.bridge_ratio > 0.5 and c.structural.bridge_count >= 2:
                problems.append(DetectedProblem(
                    entity_id=c.id,
                    entity_type="Component",
                    category=ProblemCategory.AVAILABILITY.value,
                    severity="HIGH",
                    name="High Bridge Ratio",
                    description=(
                        f"'{c.id}' has {c.structural.bridge_count} bridge edges "
                        f"({c.structural.bridge_ratio:.0%} of its connections). "
                        f"These edges are single paths to other parts of the system."
                    ),
                    recommendation=(
                        "Add redundant connections to components currently reachable "
                        "only through bridge edges. Consider peer-to-peer backup links."
                    ),
                    evidence={
                        "bridge_count": c.structural.bridge_count,
                        "bridge_ratio": c.structural.bridge_ratio,
                        "total_degree": c.structural.total_degree_raw,
                    }
                ))
            
            # === RELIABILITY PROBLEMS ===
            
            # Failure Propagation Hub (high reverse PageRank)
            if c.levels.reliability >= CriticalityLevel.CRITICAL:
                problems.append(DetectedProblem(
                    entity_id=c.id,
                    entity_type="Component",
                    category=ProblemCategory.RELIABILITY.value,
                    severity="CRITICAL",
                    name="Failure Propagation Hub",
                    description=(
                        f"'{c.id}' has critical reliability risk (score: {c.scores.reliability:.3f}). "
                        f"A failure here would cascade to many dependent components, "
                        f"potentially causing widespread system disruption."
                    ),
                    recommendation=(
                        "Implement: (1) Health checks and fast failure detection, "
                        "(2) Circuit breakers in dependent services, "
                        "(3) Graceful degradation patterns, "
                        "(4) Retry policies with exponential backoff."
                    ),
                    evidence={
                        "reliability_score": c.scores.reliability,
                        "pagerank": c.structural.pagerank,
                        "reverse_pagerank": c.structural.reverse_pagerank,
                        "in_degree": c.structural.in_degree_raw,
                    }
                ))
            elif c.levels.reliability == CriticalityLevel.HIGH:
                problems.append(DetectedProblem(
                    entity_id=c.id,
                    entity_type="Component",
                    category=ProblemCategory.RELIABILITY.value,
                    severity="HIGH",
                    name="High Failure Impact",
                    description=(
                        f"'{c.id}' has high reliability risk. Failures would significantly "
                        f"impact multiple downstream components."
                    ),
                    recommendation=(
                        "Add monitoring, alerting, and consider redundancy or caching "
                        "to reduce cascading failure risk."
                    ),
                    evidence={
                        "reliability_score": c.scores.reliability,
                        "in_degree": c.structural.in_degree_raw,
                    }
                ))
            
            # === MAINTAINABILITY PROBLEMS ===
            
            # God Component (extreme betweenness - everything goes through it)
            if c.levels.maintainability >= CriticalityLevel.CRITICAL:
                if c.structural.betweenness > 0.3:  # Very high betweenness
                    problems.append(DetectedProblem(
                        entity_id=c.id,
                        entity_type="Component",
                        category=ProblemCategory.MAINTAINABILITY.value,
                        severity="CRITICAL",
                        name="God Component / Central Bottleneck",
                        description=(
                            f"'{c.id}' lies on {c.structural.betweenness:.1%} of all shortest paths. "
                            f"This central position makes it a coupling hotspot where changes "
                            f"are risky and testing is complex."
                        ),
                        recommendation=(
                            "Refactor to distribute responsibilities: (1) Split into smaller "
                            "services, (2) Use domain-driven design boundaries, "
                            "(3) Introduce mediator/facade patterns, (4) Consider CQRS."
                        ),
                        evidence={
                            "betweenness": c.structural.betweenness,
                            "maintainability_score": c.scores.maintainability,
                            "degree": c.structural.degree,
                        }
                    ))
            
            # High Coupling (many dependencies)
            elif c.levels.maintainability == CriticalityLevel.HIGH:
                problems.append(DetectedProblem(
                    entity_id=c.id,
                    entity_type="Component",
                    category=ProblemCategory.MAINTAINABILITY.value,
                    severity="MEDIUM",
                    name="High Coupling",
                    description=(
                        f"'{c.id}' has high maintainability concerns due to coupling. "
                        f"Changes here require careful coordination with dependent services."
                    ),
                    recommendation=(
                        "Review dependencies and consider: (1) API versioning, "
                        "(2) Event-driven decoupling, (3) Interface segregation."
                    ),
                    evidence={
                        "maintainability_score": c.scores.maintainability,
                        "degree": c.structural.degree,
                        "clustering": c.structural.clustering_coefficient,
                    }
                ))
            
            # Low Clustering (isolated, no redundant paths)
            if (c.structural.clustering_coefficient < 0.1 and 
                c.structural.total_degree_raw >= 3 and
                not c.structural.is_isolated):
                problems.append(DetectedProblem(
                    entity_id=c.id,
                    entity_type="Component",
                    category=ProblemCategory.ARCHITECTURE.value,
                    severity="LOW",
                    name="Hub-and-Spoke Pattern",
                    description=(
                        f"'{c.id}' has low clustering ({c.structural.clustering_coefficient:.2f}) "
                        f"despite multiple connections. Its neighbors don't communicate "
                        f"directly, suggesting a hub-and-spoke anti-pattern."
                    ),
                    recommendation=(
                        "Consider adding direct links between neighbors where appropriate "
                        "to create redundant paths and reduce central dependency."
                    ),
                    evidence={
                        "clustering": c.structural.clustering_coefficient,
                        "degree": c.structural.total_degree_raw,
                    }
                ))
            
            # Isolated Component
            if c.structural.is_isolated:
                problems.append(DetectedProblem(
                    entity_id=c.id,
                    entity_type="Component",
                    category=ProblemCategory.ARCHITECTURE.value,
                    severity="MEDIUM",
                    name="Isolated Component",
                    description=(
                        f"'{c.id}' has no dependencies in this layer. "
                        f"It may be orphaned, misconfigured, or pending integration."
                    ),
                    recommendation=(
                        "Verify if this component is: (1) Correctly deployed, "
                        "(2) Missing configuration, (3) A standalone service, "
                        "(4) Pending removal."
                    ),
                    evidence={
                        "is_isolated": True,
                        "in_degree": c.structural.in_degree_raw,
                        "out_degree": c.structural.out_degree_raw,
                    }
                ))

            # === SECURITY / VULNERABILITY PROBLEMS ===

            # High Value Target (High Eigenvector + Critical Importance)
            if c.levels.vulnerability >= CriticalityLevel.CRITICAL:
                problems.append(DetectedProblem(
                    entity_id=c.id,
                    entity_type="Component",
                    category=ProblemCategory.SECURITY.value,
                    severity="CRITICAL",
                    name="High Value Target",
                    description=(
                        f"'{c.id}' has critical vulnerability metrics (score: {c.scores.vulnerability:.3f}). "
                        f"It is highly connected to other important components (Eigenvector Centrality), "
                        f"making it a strategic target for attackers to compromise the system core."
                    ),
                    recommendation=(
                        "Harden security posture: (1) Implement strict Zero Trust policies, "
                        "(2) Enable comprehensive audit logging, (3) Isolate in a secure subnet, "
                        "(4) Apply highest priority patching schedule."
                    ),
                    evidence={
                        "vulnerability_score": c.scores.vulnerability,
                        "eigenvector": c.structural.eigenvector,
                        "in_degree": c.structural.in_degree_raw,
                    }
                ))

            # High Exposure (High Closeness + High In-Degree)
            elif (c.levels.vulnerability == CriticalityLevel.HIGH and 
                  c.structural.closeness > 0.6):
                problems.append(DetectedProblem(
                    entity_id=c.id,
                    entity_type="Component",
                    category=ProblemCategory.SECURITY.value,
                    severity="HIGH",
                    name="High Exposure Surface",
                    description=(
                        f"'{c.id}' is easily reachable from many parts of the system "
                        f"(Closeness: {c.structural.closeness:.2f}) and has a large attack surface."
                    ),
                    recommendation=(
                        "Reduce attack surface: (1) Restrict incoming connections to allow-list only, "
                        "(2) Validate all inputs rigorously, (3) Use API gateways for traffic control."
                    ),
                    evidence={
                        "closeness": c.structural.closeness,
                        "in_degree": c.structural.in_degree_raw,
                    }
                ))
        
        return problems
    
    def _detect_edge_problems(
        self,
        edges: List[EdgeQuality]
    ) -> List[DetectedProblem]:
        """Detect problems at the edge level."""
        problems = []
        
        for e in edges:
            # Critical Bridge Edge
            if e.structural and e.structural.is_bridge:
                severity = "CRITICAL" if e.level >= CriticalityLevel.HIGH else "HIGH"
                problems.append(DetectedProblem(
                    entity_id=e.id,
                    entity_type="Edge",
                    category=ProblemCategory.AVAILABILITY.value,
                    severity=severity,
                    name="Bridge Edge (Critical Link)",
                    description=(
                        f"The dependency {e.source} → {e.target} is a bridge. "
                        f"This is the only path connecting parts of the system. "
                        f"If this link fails, the system partitions."
                    ),
                    recommendation=(
                        "Add redundant connections: (1) Direct backup link, "
                        "(2) Alternative path through another component, "
                        "(3) Async replication or eventual consistency pattern."
                    ),
                    evidence={
                        "is_bridge": True,
                        "dependency_type": e.dependency_type,
                        "betweenness": e.structural.betweenness if e.structural else 0,
                    }
                ))
            
            # High Betweenness Edge (bottleneck)
            elif e.structural and e.structural.betweenness > 0.2:
                if e.level >= CriticalityLevel.HIGH:
                    problems.append(DetectedProblem(
                        entity_id=e.id,
                        entity_type="Edge",
                        category=ProblemCategory.MAINTAINABILITY.value,
                        severity="MEDIUM",
                        name="Bottleneck Dependency",
                        description=(
                            f"The dependency {e.source} → {e.target} carries significant "
                            f"traffic ({e.structural.betweenness:.1%} of paths). "
                            f"This link is a potential performance bottleneck."
                        ),
                        recommendation=(
                            "Consider: (1) Caching at the consumer side, "
                            "(2) Load balancing with multiple instances, "
                            "(3) Async communication patterns."
                        ),
                        evidence={
                            "betweenness": e.structural.betweenness,
                            "dependency_type": e.dependency_type,
                        }
                    ))
        
        return problems
    
    def _detect_system_problems(
        self,
        components: List[ComponentQuality],
        edges: List[EdgeQuality],
        quality_result: QualityAnalysisResult
    ) -> List[DetectedProblem]:
        """Detect system-wide patterns and problems."""
        problems = []
        
        summary = quality_result.classification_summary
        total = summary.total_components
        
        if total == 0:
            return problems
        
        # High proportion of critical components
        critical_ratio = summary.critical_components / total
        if critical_ratio > 0.2:
            problems.append(DetectedProblem(
                entity_id="SYSTEM",
                entity_type="System",
                category=ProblemCategory.ARCHITECTURE.value,
                severity="CRITICAL",
                name="Systemic Risk Pattern",
                description=(
                    f"{summary.critical_components}/{total} components ({critical_ratio:.0%}) "
                    f"are classified as CRITICAL. This indicates fundamental architectural "
                    f"issues requiring comprehensive review."
                ),
                recommendation=(
                    "Conduct architecture review: (1) Map all critical dependencies, "
                    "(2) Identify common patterns causing criticality, "
                    "(3) Develop remediation roadmap with prioritization, "
                    "(4) Consider architecture modernization initiative."
                ),
                evidence={
                    "critical_count": summary.critical_components,
                    "total_count": total,
                    "critical_ratio": critical_ratio,
                }
            ))
        elif critical_ratio > 0.1:
            problems.append(DetectedProblem(
                entity_id="SYSTEM",
                entity_type="System",
                category=ProblemCategory.ARCHITECTURE.value,
                severity="HIGH",
                name="Elevated System Risk",
                description=(
                    f"{summary.critical_components}/{total} components ({critical_ratio:.0%}) "
                    f"are critical. While not severe, this warrants attention."
                ),
                recommendation=(
                    "Review critical components and create remediation backlog."
                ),
                evidence={
                    "critical_count": summary.critical_components,
                    "critical_ratio": critical_ratio,
                }
            ))
        
        # Many articulation points (fragile graph)
        aps = [c for c in components if c.structural.is_articulation_point]
        ap_ratio = len(aps) / total if total > 0 else 0
        if ap_ratio > 0.15 and len(aps) >= 3:
            problems.append(DetectedProblem(
                entity_id="SYSTEM",
                entity_type="System",
                category=ProblemCategory.AVAILABILITY.value,
                severity="HIGH",
                name="Fragile Topology",
                description=(
                    f"The system has {len(aps)} articulation points ({ap_ratio:.0%} of components). "
                    f"This fragile topology means multiple single-point failures exist."
                ),
                recommendation=(
                    "Increase connectivity and redundancy across the system. "
                    "Consider mesh topology patterns or event-driven architecture."
                ),
                evidence={
                    "articulation_points": len(aps),
                    "ratio": ap_ratio,
                    "ap_ids": [c.id for c in aps],
                }
            ))
        
        # Many bridges
        bridges = [e for e in edges if e.structural and e.structural.is_bridge]
        bridge_ratio = len(bridges) / len(edges) if edges else 0
        if bridge_ratio > 0.3 and len(bridges) >= 3:
            problems.append(DetectedProblem(
                entity_id="SYSTEM",
                entity_type="System",
                category=ProblemCategory.AVAILABILITY.value,
                severity="HIGH",
                name="Many Critical Links",
                description=(
                    f"System has {len(bridges)} bridge edges ({bridge_ratio:.0%} of dependencies). "
                    f"Many dependencies have no alternative paths."
                ),
                recommendation=(
                    "Add redundant connections to create alternative paths "
                    "for the most critical dependencies."
                ),
                evidence={
                    "bridge_count": len(bridges),
                    "bridge_ratio": bridge_ratio,
                }
            ))
        
        # Concentration risk (few components handle most traffic)
        if len(components) >= 5:
            sorted_by_pagerank = sorted(
                components, 
                key=lambda c: c.structural.pagerank, 
                reverse=True
            )
            top_3_pagerank = sum(c.structural.pagerank for c in sorted_by_pagerank[:3])
            if top_3_pagerank > 0.5:
                top_3_ids = [c.id for c in sorted_by_pagerank[:3]]
                problems.append(DetectedProblem(
                    entity_id="SYSTEM",
                    entity_type="System",
                    category=ProblemCategory.RELIABILITY.value,
                    severity="MEDIUM",
                    name="Concentration Risk",
                    description=(
                        f"Top 3 components account for {top_3_pagerank:.0%} of system importance. "
                        f"System heavily depends on: {', '.join(top_3_ids)}."
                    ),
                    recommendation=(
                        "Consider load distribution: (1) Partition by domain, "
                        "(2) Introduce intermediary services, "
                        "(3) Use message brokers for fan-out patterns."
                    ),
                    evidence={
                        "top_3_importance": top_3_pagerank,
                        "top_3_ids": top_3_ids,
                    }
                ))
        
        return problems