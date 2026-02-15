"""
Problem Detector

Identifies architectural smells, risks, and anti-patterns from quality
analysis results.  Uses box-plot classification levels (never static
thresholds) to decide when something is problematic.

Problem Categories:
    AVAILABILITY      — SPOFs, bridges, connectivity risks
    RELIABILITY       — Failure propagation hubs, cascade risks
    MAINTAINABILITY   — God components, high coupling, bottlenecks
    SECURITY          — High-value targets, exposure surfaces
    ARCHITECTURE      — Structural issues, design smells

Severity Levels:
    CRITICAL — Immediate action required (system risk)
    HIGH     — Should be addressed soon (significant risk)
    MEDIUM   — Monitor and plan remediation
    LOW      — Minor concern, track for future
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import List, Dict, Any

from src.core.criticality import CriticalityLevel
from .models import QualityAnalysisResult
from src.core.metrics import ComponentQuality, EdgeQuality


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class ProblemCategory(Enum):
    AVAILABILITY = "Availability"
    RELIABILITY = "Reliability"
    MAINTAINABILITY = "Maintainability"
    SECURITY = "Security"
    ARCHITECTURE = "Architecture"


class ProblemSeverity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    @property
    def priority(self) -> int:
        return {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}[self.value]


from .models import DetectedProblem, ProblemSummary


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class ProblemDetector:
    """
    Detects architectural problems from quality analysis results.

    Uses box-plot classification levels (CRITICAL/HIGH/…) — never static
    numeric thresholds — to decide when something is a problem.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, quality: QualityAnalysisResult) -> List[DetectedProblem]:
        """
        Detect all problems from quality analysis results.

        Returns problems sorted by severity (CRITICAL first).
        """
        problems: List[DetectedProblem] = []
        problems.extend(self._component_problems(quality.components))
        problems.extend(self._edge_problems(quality.edges))
        problems.extend(self._system_problems(quality.components, quality.edges, quality))
        problems.sort(key=lambda p: (-p.priority, p.entity_id))
        return problems

    def summarize(self, problems: List[DetectedProblem]) -> ProblemSummary:
        """Create aggregated summary of detected problems."""
        by_sev: Dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        by_cat: Dict[str, int] = {}
        comp_ids, edge_ids = set(), set()

        for p in problems:
            by_sev[p.severity] = by_sev.get(p.severity, 0) + 1
            by_cat[p.category] = by_cat.get(p.category, 0) + 1
            if p.entity_type == "Component":
                comp_ids.add(p.entity_id)
            elif p.entity_type == "Edge":
                edge_ids.add(p.entity_id)

        return ProblemSummary(
            total_problems=len(problems),
            by_severity=by_sev,
            by_category=by_cat,
            affected_components=len(comp_ids),
            affected_edges=len(edge_ids),
        )

    # ------------------------------------------------------------------
    # Internal — component-level problems
    # ------------------------------------------------------------------

    def _component_problems(self, components: List[ComponentQuality]) -> List[DetectedProblem]:
        problems: List[DetectedProblem] = []
        for c in components:
            problems.extend(self._availability_issues(c))
            problems.extend(self._reliability_issues(c))
            problems.extend(self._maintainability_issues(c))
            problems.extend(self._security_issues(c))
            problems.extend(self._architecture_issues(c))
        return problems

    def _availability_issues(self, c: ComponentQuality) -> List[DetectedProblem]:
        out: List[DetectedProblem] = []
        # SPOF — articulation point
        if c.structural.is_articulation_point:
            sev = "CRITICAL" if c.levels.availability == CriticalityLevel.CRITICAL else "HIGH"
            out.append(DetectedProblem(
                entity_id=c.id, entity_type="Component",
                category=ProblemCategory.AVAILABILITY.value, severity=sev,
                name="Single Point of Failure (SPOF)",
                description=(
                    f"'{c.id}' is an articulation point. Removing it partitions "
                    f"the dependency graph into isolated clusters."
                ),
                recommendation=(
                    "Introduce redundancy: backup instances, alternative paths, "
                    "event bus / service mesh for decoupling, circuit breakers."
                ),
                evidence={"is_articulation_point": True,
                          "availability_level": c.levels.availability.value},
            ))
        return out

    def _reliability_issues(self, c: ComponentQuality) -> List[DetectedProblem]:
        out: List[DetectedProblem] = []
        if c.levels.reliability >= CriticalityLevel.CRITICAL:
            out.append(DetectedProblem(
                entity_id=c.id, entity_type="Component",
                category=ProblemCategory.RELIABILITY.value, severity="CRITICAL",
                name="Critical Failure Propagation Hub",
                description=(
                    f"'{c.id}' has critical reliability risk (score: {c.scores.reliability:.3f}). "
                    f"A failure here cascades to many dependent components."
                ),
                recommendation=(
                    "Health checks, circuit breakers in dependents, graceful degradation, "
                    "retry policies with exponential backoff."
                ),
                evidence={"reliability_score": c.scores.reliability,
                          "pagerank": c.structural.pagerank,
                          "in_degree": c.structural.in_degree_raw},
            ))
        elif c.levels.reliability == CriticalityLevel.HIGH:
            out.append(DetectedProblem(
                entity_id=c.id, entity_type="Component",
                category=ProblemCategory.RELIABILITY.value, severity="HIGH",
                name="High Failure Impact",
                description=(
                    f"'{c.id}' has high reliability risk. Failures would significantly "
                    f"impact downstream components."
                ),
                recommendation="Add monitoring, alerting, and consider redundancy or caching.",
                evidence={"reliability_score": c.scores.reliability,
                          "in_degree": c.structural.in_degree_raw},
            ))
        return out

    def _maintainability_issues(self, c: ComponentQuality) -> List[DetectedProblem]:
        out: List[DetectedProblem] = []
        # God component (extreme betweenness)
        if c.levels.maintainability >= CriticalityLevel.CRITICAL and c.structural.betweenness > 0.3:
            out.append(DetectedProblem(
                entity_id=c.id, entity_type="Component",
                category=ProblemCategory.MAINTAINABILITY.value, severity="CRITICAL",
                name="God Component / Central Bottleneck",
                description=(
                    f"'{c.id}' lies on {c.structural.betweenness:.1%} of shortest paths. "
                    f"Central coupling hotspot — changes are risky and testing is complex."
                ),
                recommendation=(
                    "Extract responsibilities: decompose into smaller services, "
                    "use message bus for fan-out, apply Strangler Fig pattern."
                ),
                evidence={"betweenness": c.structural.betweenness,
                          "total_degree": c.structural.total_degree_raw},
            ))
        # Low clustering (hub-and-spoke anti-pattern)
        elif (c.levels.maintainability >= CriticalityLevel.HIGH
              and c.structural.clustering_coefficient < 0.1
              and c.structural.total_degree_raw > 3):
            out.append(DetectedProblem(
                entity_id=c.id, entity_type="Component",
                category=ProblemCategory.MAINTAINABILITY.value, severity="MEDIUM",
                name="Hub-and-Spoke Anti-Pattern",
                description=(
                    f"'{c.id}' has low clustering ({c.structural.clustering_coefficient:.2f}). "
                    f"Its neighbors don't communicate directly."
                ),
                recommendation="Add direct links between neighbors for redundant paths.",
                evidence={"clustering": c.structural.clustering_coefficient,
                          "degree": c.structural.total_degree_raw},
            ))
        return out

    def _security_issues(self, c: ComponentQuality) -> List[DetectedProblem]:
        out: List[DetectedProblem] = []
        if c.levels.vulnerability >= CriticalityLevel.CRITICAL:
            out.append(DetectedProblem(
                entity_id=c.id, entity_type="Component",
                category=ProblemCategory.SECURITY.value, severity="CRITICAL",
                name="High Value Target",
                description=(
                    f"'{c.id}' has critical vulnerability (score: {c.scores.vulnerability:.3f}). "
                    f"Highly connected to important components via Eigenvector centrality."
                ),
                recommendation=(
                    "Zero Trust policies, audit logging, network isolation, "
                    "highest-priority patching schedule."
                ),
                evidence={"vulnerability_score": c.scores.vulnerability,
                          "eigenvector": c.structural.eigenvector,
                          "in_degree": c.structural.in_degree_raw},
            ))
        elif c.levels.vulnerability == CriticalityLevel.HIGH and c.structural.closeness > 0.6:
            out.append(DetectedProblem(
                entity_id=c.id, entity_type="Component",
                category=ProblemCategory.SECURITY.value, severity="HIGH",
                name="High Exposure Surface",
                description=(
                    f"'{c.id}' is easily reachable (closeness: {c.structural.closeness:.2f}) "
                    f"with a large attack surface."
                ),
                recommendation=(
                    "Restrict incoming connections, validate all inputs, use API gateways."
                ),
                evidence={"closeness": c.structural.closeness,
                          "in_degree": c.structural.in_degree_raw},
            ))
        return out

    def _architecture_issues(self, c: ComponentQuality) -> List[DetectedProblem]:
        out: List[DetectedProblem] = []
        if c.structural.is_isolated:
            out.append(DetectedProblem(
                entity_id=c.id, entity_type="Component",
                category=ProblemCategory.ARCHITECTURE.value, severity="MEDIUM",
                name="Isolated Component",
                description=(
                    f"'{c.id}' has no dependencies in this layer. "
                    f"May be orphaned, misconfigured, or pending integration."
                ),
                recommendation="Verify deployment, configuration, and integration status.",
                evidence={"is_isolated": True},
            ))
        return out

    # ------------------------------------------------------------------
    # Internal — edge-level problems
    # ------------------------------------------------------------------

    def _edge_problems(self, edges: List[EdgeQuality]) -> List[DetectedProblem]:
        problems: List[DetectedProblem] = []
        for e in edges:
            if e.structural and e.structural.is_bridge:
                sev = "CRITICAL" if e.level >= CriticalityLevel.HIGH else "HIGH"
                problems.append(DetectedProblem(
                    entity_id=e.id, entity_type="Edge",
                    category=ProblemCategory.AVAILABILITY.value, severity=sev,
                    name="Bridge Edge (Critical Link)",
                    description=(
                        f"{e.source} → {e.target} is a bridge. "
                        f"Only path connecting parts of the system."
                    ),
                    recommendation="Add redundant connections or alternative paths.",
                    evidence={"is_bridge": True,
                              "dependency_type": e.dependency_type},
                ))
            elif (e.structural and e.structural.betweenness > 0.2
                  and e.level >= CriticalityLevel.HIGH):
                problems.append(DetectedProblem(
                    entity_id=e.id, entity_type="Edge",
                    category=ProblemCategory.MAINTAINABILITY.value, severity="MEDIUM",
                    name="Bottleneck Dependency",
                    description=(
                        f"{e.source} → {e.target} carries {e.structural.betweenness:.1%} of paths. "
                        f"Potential performance bottleneck."
                    ),
                    recommendation="Consider caching, load balancing, or async patterns.",
                    evidence={"betweenness": e.structural.betweenness,
                              "dependency_type": e.dependency_type},
                ))
        return problems

    # ------------------------------------------------------------------
    # Internal — system-level patterns
    # ------------------------------------------------------------------

    def _system_problems(
        self,
        components: List[ComponentQuality],
        edges: List[EdgeQuality],
        quality: QualityAnalysisResult,
    ) -> List[DetectedProblem]:
        problems: List[DetectedProblem] = []
        total = quality.classification_summary.total_components
        if total == 0:
            return problems

        crit_count = quality.classification_summary.component_distribution.get("critical", 0)
        crit_ratio = crit_count / total

        # Systemic risk
        if crit_ratio > 0.2:
            problems.append(DetectedProblem(
                entity_id="SYSTEM", entity_type="System",
                category=ProblemCategory.ARCHITECTURE.value, severity="CRITICAL",
                name="Systemic Risk Pattern",
                description=(
                    f"{crit_count}/{total} components ({crit_ratio:.0%}) are CRITICAL. "
                    f"Fundamental architectural issues require comprehensive review."
                ),
                recommendation="Architecture review, dependency mapping, remediation roadmap.",
                evidence={"critical_count": crit_count, "total_count": total},
            ))
        elif crit_ratio > 0.1:
            problems.append(DetectedProblem(
                entity_id="SYSTEM", entity_type="System",
                category=ProblemCategory.ARCHITECTURE.value, severity="HIGH",
                name="Elevated System Risk",
                description=f"{crit_count}/{total} components ({crit_ratio:.0%}) are critical.",
                recommendation="Review critical components and create remediation backlog.",
                evidence={"critical_count": crit_count, "critical_ratio": crit_ratio},
            ))

        # High bridge ratio
        bridges = [e for e in edges if e.structural and e.structural.is_bridge]
        if edges:
            bridge_ratio = len(bridges) / len(edges)
            if bridge_ratio > 0.3:
                problems.append(DetectedProblem(
                    entity_id="SYSTEM", entity_type="System",
                    category=ProblemCategory.AVAILABILITY.value, severity="HIGH",
                    name="Fragile Connectivity",
                    description=f"{len(bridges)}/{len(edges)} edges ({bridge_ratio:.0%}) are bridges.",
                    recommendation="Add redundant connections for critical dependencies.",
                    evidence={"bridge_count": len(bridges), "bridge_ratio": bridge_ratio},
                ))

        # Concentration risk
        if len(components) >= 5:
            top3 = sorted(components, key=lambda c: c.structural.pagerank, reverse=True)[:3]
            top3_pr = sum(c.structural.pagerank for c in top3)
            if top3_pr > 0.5:
                ids = [c.id for c in top3]
                problems.append(DetectedProblem(
                    entity_id="SYSTEM", entity_type="System",
                    category=ProblemCategory.RELIABILITY.value, severity="MEDIUM",
                    name="Concentration Risk",
                    description=(
                        f"Top 3 components hold {top3_pr:.0%} of system importance: "
                        f"{', '.join(ids)}."
                    ),
                    recommendation="Distribute load via domain partitioning or message brokers.",
                    evidence={"top_3_importance": top3_pr, "top_3_ids": ids},
                ))

        return problems