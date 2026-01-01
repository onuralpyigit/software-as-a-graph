"""
Problem Detector - Version 5.0

Detects problems affecting reliability, maintainability, and availability
in distributed pub-sub systems using graph-based analysis.

Problem Categories:
- Reliability: SPOFs, cascade risks, single paths
- Maintainability: High coupling, god components, poor modularity
- Availability: Bottlenecks, no redundancy, critical dependencies

Each problem includes:
- Description of the issue
- Symptoms that indicate the problem
- Affected components
- Impact assessment
- Recommendations for remediation

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Set

from .gds_client import GDSClient, CentralityResult
from .classifier import BoxPlotClassifier, CriticalityLevel


# =============================================================================
# Enums
# =============================================================================

class QualityAttribute(Enum):
    """Quality attributes (ISO 25010)"""
    RELIABILITY = "reliability"
    MAINTAINABILITY = "maintainability"
    AVAILABILITY = "availability"
    
    @property
    def description(self) -> str:
        return {
            QualityAttribute.RELIABILITY: "Ability to perform under stated conditions",
            QualityAttribute.MAINTAINABILITY: "Ease of modification and enhancement",
            QualityAttribute.AVAILABILITY: "Degree of system operational accessibility",
        }[self]


class ProblemType(Enum):
    """Types of problems that can be detected"""
    # Reliability problems
    SINGLE_POINT_OF_FAILURE = "spof"
    CASCADE_RISK = "cascade_risk"
    NO_REDUNDANT_PATH = "no_redundant_path"
    CRITICAL_BRIDGE = "critical_bridge"
    
    # Maintainability problems
    HIGH_COUPLING = "high_coupling"
    GOD_COMPONENT = "god_component"
    POOR_MODULARITY = "poor_modularity"
    CHANGE_PROPAGATION_RISK = "change_propagation_risk"
    
    # Availability problems
    BOTTLENECK = "bottleneck"
    NO_REDUNDANCY = "no_redundancy"
    CRITICAL_DEPENDENCY = "critical_dependency"
    OVERLOADED_COMPONENT = "overloaded_component"
    
    @property
    def quality_attribute(self) -> QualityAttribute:
        """Get the primary quality attribute affected"""
        reliability = {
            ProblemType.SINGLE_POINT_OF_FAILURE,
            ProblemType.CASCADE_RISK,
            ProblemType.NO_REDUNDANT_PATH,
            ProblemType.CRITICAL_BRIDGE,
        }
        maintainability = {
            ProblemType.HIGH_COUPLING,
            ProblemType.GOD_COMPONENT,
            ProblemType.POOR_MODULARITY,
            ProblemType.CHANGE_PROPAGATION_RISK,
        }
        
        if self in reliability:
            return QualityAttribute.RELIABILITY
        elif self in maintainability:
            return QualityAttribute.MAINTAINABILITY
        else:
            return QualityAttribute.AVAILABILITY


class ProblemSeverity(Enum):
    """Severity levels for problems"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    
    @property
    def numeric(self) -> int:
        return {
            ProblemSeverity.CRITICAL: 4,
            ProblemSeverity.HIGH: 3,
            ProblemSeverity.MEDIUM: 2,
            ProblemSeverity.LOW: 1,
        }[self]
    
    @property
    def color(self) -> str:
        return {
            ProblemSeverity.CRITICAL: "\033[91m",
            ProblemSeverity.HIGH: "\033[93m",
            ProblemSeverity.MEDIUM: "\033[94m",
            ProblemSeverity.LOW: "\033[92m",
        }[self]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Symptom:
    """
    A symptom that indicates a problem.
    
    Symptoms are observable indicators from graph analysis
    that suggest an underlying problem.
    """
    name: str
    description: str
    metric: str
    value: float
    threshold: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "metric": self.metric,
            "value": round(self.value, 4),
            "threshold": round(self.threshold, 4),
        }


@dataclass
class Problem:
    """
    A detected problem in the system.
    
    Contains full context about the problem including
    symptoms, affected components, impact, and recommendations.
    """
    problem_type: ProblemType
    severity: ProblemSeverity
    title: str
    description: str
    affected_components: List[str]
    symptoms: List[Symptom]
    impact: str
    recommendation: str
    quality_attributes: List[QualityAttribute]
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.problem_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "affected_components": self.affected_components,
            "symptoms": [s.to_dict() for s in self.symptoms],
            "impact": self.impact,
            "recommendation": self.recommendation,
            "quality_attributes": [qa.value for qa in self.quality_attributes],
            "metrics": self.metrics,
        }


@dataclass
class ProblemDetectionResult:
    """
    Complete result from problem detection.
    """
    timestamp: str
    problems: List[Problem]
    by_severity: Dict[ProblemSeverity, List[Problem]]
    by_quality_attribute: Dict[QualityAttribute, List[Problem]]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "problems": [p.to_dict() for p in self.problems],
            "summary": self.summary,
            "by_severity": {
                sev.value: [p.to_dict() for p in problems]
                for sev, problems in self.by_severity.items()
            },
            "by_quality_attribute": {
                qa.value: [p.to_dict() for p in problems]
                for qa, problems in self.by_quality_attribute.items()
            },
        }
    
    @property
    def critical_count(self) -> int:
        return len(self.by_severity.get(ProblemSeverity.CRITICAL, []))
    
    @property
    def total_count(self) -> int:
        return len(self.problems)
    
    def get_by_component(self, component_id: str) -> List[Problem]:
        """Get all problems affecting a specific component"""
        return [
            p for p in self.problems 
            if component_id in p.affected_components
        ]


# =============================================================================
# Problem Detector
# =============================================================================

class ProblemDetector:
    """
    Detects problems in distributed pub-sub systems.
    
    Uses graph analysis to identify:
    - Reliability issues (SPOFs, cascade risks)
    - Maintainability issues (coupling, god components)
    - Availability issues (bottlenecks, no redundancy)
    
    Example:
        with GDSClient(uri, user, password) as gds:
            detector = ProblemDetector(gds)
            result = detector.detect_all()
            
            for problem in result.problems:
                print(f"[{problem.severity.value}] {problem.title}")
                for symptom in problem.symptoms:
                    print(f"  - {symptom.name}: {symptom.value:.2f}")
    """

    def __init__(
        self,
        gds_client: GDSClient,
        k_factor: float = 1.5,
    ):
        """
        Initialize detector.
        
        Args:
            gds_client: Connected GDS client
            k_factor: Box-plot k factor for outlier detection
        """
        self.gds = gds_client
        self.classifier = BoxPlotClassifier(k_factor=k_factor)
        self.logger = logging.getLogger(__name__)

    def detect_all(
        self,
        projection_name: str = "problem_detection",
        dependency_types: Optional[List[str]] = None,
    ) -> ProblemDetectionResult:
        """
        Run complete problem detection.
        
        Args:
            projection_name: Name for GDS projection
            dependency_types: DEPENDS_ON types to analyze
        
        Returns:
            ProblemDetectionResult with all detected problems
        """
        timestamp = datetime.now().isoformat()
        
        if dependency_types is None:
            dependency_types = ["app_to_app", "node_to_node"]
        
        self.logger.info("Starting problem detection")
        
        try:
            # Create projection
            projection_info = self.gds.create_projection(
                projection_name,
                dependency_types=dependency_types,
                include_weights=True,
            )
            
            problems: List[Problem] = []
            
            # Detect reliability problems
            problems.extend(self._detect_reliability_problems(projection_name))
            
            # Detect maintainability problems
            problems.extend(self._detect_maintainability_problems(projection_name))
            
            # Detect availability problems
            problems.extend(self._detect_availability_problems(projection_name))
            
            # Organize results
            by_severity = self._group_by_severity(problems)
            by_qa = self._group_by_quality_attribute(problems)
            summary = self._generate_summary(problems, by_severity, by_qa)
            
            return ProblemDetectionResult(
                timestamp=timestamp,
                problems=problems,
                by_severity=by_severity,
                by_quality_attribute=by_qa,
                summary=summary,
            )
        
        finally:
            self.gds.drop_projection(projection_name)

    # =========================================================================
    # Reliability Problem Detection
    # =========================================================================

    def _detect_reliability_problems(
        self, 
        projection_name: str
    ) -> List[Problem]:
        """Detect reliability-related problems"""
        problems = []
        
        # 1. Single Points of Failure (high betweenness + articulation point)
        problems.extend(self._detect_spof(projection_name))
        
        # 2. Cascade Risks (high out-degree + high betweenness)
        problems.extend(self._detect_cascade_risks(projection_name))
        
        # 3. Critical Bridges
        problems.extend(self._detect_critical_bridges())
        
        return problems

    def _detect_spof(self, projection_name: str) -> List[Problem]:
        """Detect Single Points of Failure"""
        problems = []
        
        # Get betweenness centrality
        bc_results = self.gds.betweenness(projection_name)
        if not bc_results:
            return problems
        
        # Classify to find outliers
        bc_items = [
            {"id": r.node_id, "type": r.node_type, "score": r.score}
            for r in bc_results
        ]
        classification = self.classifier.classify(bc_items, metric_name="betweenness")
        
        # Get articulation points
        articulation_points = {
            ap["node_id"] for ap in self.gds.find_articulation_points()
        }
        
        # SPOF = high betweenness + articulation point
        for item in classification.get_high_and_above():
            is_ap = item.id in articulation_points
            
            symptoms = [
                Symptom(
                    name="High Betweenness Centrality",
                    description="Component is on many shortest paths",
                    metric="betweenness",
                    value=item.score,
                    threshold=classification.stats.q3,
                ),
            ]
            
            if is_ap:
                symptoms.append(Symptom(
                    name="Articulation Point",
                    description="Removal would disconnect the graph",
                    metric="is_articulation_point",
                    value=1.0,
                    threshold=0.5,
                ))
            
            severity = ProblemSeverity.CRITICAL if is_ap else ProblemSeverity.HIGH
            
            if item.level >= CriticalityLevel.HIGH or is_ap:
                problems.append(Problem(
                    problem_type=ProblemType.SINGLE_POINT_OF_FAILURE,
                    severity=severity,
                    title=f"Single Point of Failure: {item.id}",
                    description=(
                        f"{item.item_type} '{item.id}' is a potential SPOF. "
                        f"It has high betweenness centrality ({item.score:.4f}) "
                        f"{'and is an articulation point' if is_ap else ''}"
                    ),
                    affected_components=[item.id],
                    symptoms=symptoms,
                    impact=(
                        "Failure of this component would disrupt many communication paths"
                        + (" and disconnect part of the system" if is_ap else "")
                    ),
                    recommendation=(
                        "Add redundancy through: "
                        "1) Deploying replica instances, "
                        "2) Creating alternative paths, "
                        "3) Implementing failover mechanisms"
                    ),
                    quality_attributes=[QualityAttribute.RELIABILITY],
                    metrics={
                        "betweenness": item.score,
                        "is_articulation_point": is_ap,
                        "percentile": item.percentile,
                    },
                ))
        
        return problems

    def _detect_cascade_risks(self, projection_name: str) -> List[Problem]:
        """Detect components with high cascade failure risk"""
        problems = []
        
        # Get out-degree (components that many others depend on)
        out_results = self.gds.degree(
            projection_name, 
            orientation="NATURAL"
        )
        
        # Get betweenness
        bc_results = self.gds.betweenness(projection_name)
        bc_map = {r.node_id: r.score for r in bc_results}
        
        # Classify out-degree
        out_items = [
            {"id": r.node_id, "type": r.node_type, "score": r.score}
            for r in out_results
        ]
        classification = self.classifier.classify(out_items, metric_name="out_degree")
        
        for item in classification.get_critical():
            bc_score = bc_map.get(item.id, 0)
            
            problems.append(Problem(
                problem_type=ProblemType.CASCADE_RISK,
                severity=ProblemSeverity.HIGH,
                title=f"Cascade Failure Risk: {item.id}",
                description=(
                    f"{item.item_type} '{item.id}' has high out-degree ({item.score:.1f}), "
                    f"meaning many components depend on it directly."
                ),
                affected_components=[item.id],
                symptoms=[
                    Symptom(
                        name="High Out-Degree",
                        description="Many components depend directly on this one",
                        metric="out_degree",
                        value=item.score,
                        threshold=classification.stats.upper_fence,
                    ),
                    Symptom(
                        name="Betweenness Centrality",
                        description="Position in communication paths",
                        metric="betweenness",
                        value=bc_score,
                        threshold=0,
                    ),
                ],
                impact=(
                    f"Failure could cascade to {int(item.score)} dependent components"
                ),
                recommendation=(
                    "Reduce cascade risk by: "
                    "1) Breaking dependencies into smaller units, "
                    "2) Adding circuit breakers, "
                    "3) Implementing graceful degradation"
                ),
                quality_attributes=[
                    QualityAttribute.RELIABILITY, 
                    QualityAttribute.AVAILABILITY
                ],
                metrics={
                    "out_degree": item.score,
                    "betweenness": bc_score,
                },
            ))
        
        return problems

    def _detect_critical_bridges(self) -> List[Problem]:
        """Detect critical bridge edges"""
        problems = []
        
        bridges = self.gds.find_bridges()
        
        for bridge in bridges[:10]:  # Limit to top 10
            source = bridge.get("source_id", "unknown")
            target = bridge.get("target_id", "unknown")
            
            problems.append(Problem(
                problem_type=ProblemType.CRITICAL_BRIDGE,
                severity=ProblemSeverity.MEDIUM,
                title=f"Critical Bridge: {source} → {target}",
                description=(
                    f"The connection from '{source}' to '{target}' is a bridge edge. "
                    f"If this connection fails, parts of the system become unreachable."
                ),
                affected_components=[source, target],
                symptoms=[
                    Symptom(
                        name="Bridge Edge",
                        description="Only path between components",
                        metric="is_bridge",
                        value=1.0,
                        threshold=0.5,
                    ),
                ],
                impact="Failure would partition the system",
                recommendation=(
                    "Create redundant paths between these components "
                    "or deploy an intermediate broker for resilience"
                ),
                quality_attributes=[QualityAttribute.RELIABILITY],
                metrics=bridge,
            ))
        
        return problems

    # =========================================================================
    # Maintainability Problem Detection
    # =========================================================================

    def _detect_maintainability_problems(
        self, 
        projection_name: str
    ) -> List[Problem]:
        """Detect maintainability-related problems"""
        problems = []
        
        # 1. High Coupling (high total degree)
        problems.extend(self._detect_high_coupling(projection_name))
        
        # 2. God Components (high degree + high betweenness)
        problems.extend(self._detect_god_components(projection_name))
        
        # 3. Poor Modularity (via Louvain)
        problems.extend(self._detect_poor_modularity(projection_name))
        
        return problems

    def _detect_high_coupling(self, projection_name: str) -> List[Problem]:
        """Detect highly coupled components"""
        problems = []
        
        degree_results = self.gds.degree(
            projection_name, 
            orientation="UNDIRECTED"
        )
        
        items = [
            {"id": r.node_id, "type": r.node_type, "score": r.score}
            for r in degree_results
        ]
        classification = self.classifier.classify(items, metric_name="degree")
        
        for item in classification.get_critical():
            problems.append(Problem(
                problem_type=ProblemType.HIGH_COUPLING,
                severity=ProblemSeverity.MEDIUM,
                title=f"High Coupling: {item.id}",
                description=(
                    f"{item.item_type} '{item.id}' has very high coupling "
                    f"(degree = {item.score:.1f}), making it difficult to modify independently."
                ),
                affected_components=[item.id],
                symptoms=[
                    Symptom(
                        name="High Degree Centrality",
                        description="Many direct connections",
                        metric="degree",
                        value=item.score,
                        threshold=classification.stats.upper_fence,
                    ),
                ],
                impact="Changes to this component may require coordinated changes elsewhere",
                recommendation=(
                    "Reduce coupling by: "
                    "1) Extracting interfaces, "
                    "2) Using intermediary topics, "
                    "3) Applying the dependency inversion principle"
                ),
                quality_attributes=[QualityAttribute.MAINTAINABILITY],
                metrics={"degree": item.score},
            ))
        
        return problems

    def _detect_god_components(self, projection_name: str) -> List[Problem]:
        """Detect god components (too many responsibilities)"""
        problems = []
        
        # Get both degree and betweenness
        degree_results = self.gds.degree(projection_name, orientation="UNDIRECTED")
        bc_results = self.gds.betweenness(projection_name)
        
        degree_map = {r.node_id: r.score for r in degree_results}
        bc_map = {r.node_id: r.score for r in bc_results}
        
        # Classify both
        degree_items = [
            {"id": r.node_id, "type": r.node_type, "score": r.score}
            for r in degree_results
        ]
        bc_items = [
            {"id": r.node_id, "type": r.node_type, "score": r.score}
            for r in bc_results
        ]
        
        degree_class = self.classifier.classify(degree_items, metric_name="degree")
        bc_class = self.classifier.classify(bc_items, metric_name="betweenness")
        
        # God component = high in both
        high_degree = {item.id for item in degree_class.get_high_and_above()}
        high_bc = {item.id for item in bc_class.get_high_and_above()}
        
        god_components = high_degree & high_bc
        
        for comp_id in god_components:
            degree_score = degree_map.get(comp_id, 0)
            bc_score = bc_map.get(comp_id, 0)
            
            problems.append(Problem(
                problem_type=ProblemType.GOD_COMPONENT,
                severity=ProblemSeverity.HIGH,
                title=f"God Component: {comp_id}",
                description=(
                    f"Component '{comp_id}' has too many responsibilities "
                    f"(high degree={degree_score:.1f}, high betweenness={bc_score:.4f})"
                ),
                affected_components=[comp_id],
                symptoms=[
                    Symptom(
                        name="High Degree",
                        description="Many connections",
                        metric="degree",
                        value=degree_score,
                        threshold=degree_class.stats.q3,
                    ),
                    Symptom(
                        name="High Betweenness",
                        description="Central to communication",
                        metric="betweenness",
                        value=bc_score,
                        threshold=bc_class.stats.q3,
                    ),
                ],
                impact="This component is a maintenance bottleneck and single point of failure",
                recommendation=(
                    "Split into smaller, focused components using: "
                    "1) Single Responsibility Principle, "
                    "2) Domain-Driven Design, "
                    "3) Microservices decomposition patterns"
                ),
                quality_attributes=[
                    QualityAttribute.MAINTAINABILITY,
                    QualityAttribute.RELIABILITY,
                ],
                metrics={
                    "degree": degree_score,
                    "betweenness": bc_score,
                },
            ))
        
        return problems

    def _detect_poor_modularity(self, projection_name: str) -> List[Problem]:
        """Detect poor modularity using community detection"""
        problems = []
        
        try:
            _, stats = self.gds.louvain(projection_name)
            
            # Check for problematic modularity
            if stats["num_communities"] == 1:
                problems.append(Problem(
                    problem_type=ProblemType.POOR_MODULARITY,
                    severity=ProblemSeverity.MEDIUM,
                    title="Poor Modularity: Single Cluster",
                    description=(
                        "The system appears to be a single tightly-coupled cluster "
                        "with no clear module boundaries."
                    ),
                    affected_components=[],
                    symptoms=[
                        Symptom(
                            name="Single Community",
                            description="No natural module boundaries",
                            metric="num_communities",
                            value=1,
                            threshold=2,
                        ),
                    ],
                    impact="Difficult to understand, modify, and test independently",
                    recommendation=(
                        "Introduce module boundaries by: "
                        "1) Identifying bounded contexts, "
                        "2) Using topic namespacing, "
                        "3) Deploying separate broker instances per domain"
                    ),
                    quality_attributes=[QualityAttribute.MAINTAINABILITY],
                    metrics=stats,
                ))
            
            # Check for very unequal community sizes
            if stats["max_size"] > 5 * stats["avg_size"] and stats["num_communities"] > 1:
                problems.append(Problem(
                    problem_type=ProblemType.POOR_MODULARITY,
                    severity=ProblemSeverity.LOW,
                    title="Unbalanced Modularity",
                    description=(
                        f"One module is much larger than others "
                        f"(max={stats['max_size']}, avg={stats['avg_size']:.1f})"
                    ),
                    affected_components=[],
                    symptoms=[
                        Symptom(
                            name="Unbalanced Modules",
                            description="Largest module is >5x average",
                            metric="max_community_size",
                            value=stats["max_size"],
                            threshold=stats["avg_size"] * 5,
                        ),
                    ],
                    impact="The large module may become a maintenance burden",
                    recommendation="Consider splitting the largest module into sub-modules",
                    quality_attributes=[QualityAttribute.MAINTAINABILITY],
                    metrics=stats,
                ))
        
        except Exception as e:
            self.logger.warning(f"Louvain analysis failed: {e}")
        
        return problems

    # =========================================================================
    # Availability Problem Detection
    # =========================================================================

    def _detect_availability_problems(
        self, 
        projection_name: str
    ) -> List[Problem]:
        """Detect availability-related problems"""
        problems = []
        
        # 1. Bottlenecks (high PageRank)
        problems.extend(self._detect_bottlenecks(projection_name))
        
        # 2. Connectivity issues
        problems.extend(self._detect_connectivity_issues(projection_name))
        
        return problems

    def _detect_bottlenecks(self, projection_name: str) -> List[Problem]:
        """Detect availability bottlenecks"""
        problems = []
        
        pr_results = self.gds.pagerank(projection_name)
        
        items = [
            {"id": r.node_id, "type": r.node_type, "score": r.score}
            for r in pr_results
        ]
        classification = self.classifier.classify(items, metric_name="pagerank")
        
        for item in classification.get_critical():
            problems.append(Problem(
                problem_type=ProblemType.BOTTLENECK,
                severity=ProblemSeverity.HIGH,
                title=f"Availability Bottleneck: {item.id}",
                description=(
                    f"{item.item_type} '{item.id}' has very high PageRank ({item.score:.4f}), "
                    f"indicating it receives dependencies from many important components."
                ),
                affected_components=[item.id],
                symptoms=[
                    Symptom(
                        name="High PageRank",
                        description="Many important dependencies point here",
                        metric="pagerank",
                        value=item.score,
                        threshold=classification.stats.upper_fence,
                    ),
                ],
                impact="This component's availability directly affects system availability",
                recommendation=(
                    "Improve availability by: "
                    "1) Deploying multiple replicas, "
                    "2) Implementing load balancing, "
                    "3) Adding caching layers, "
                    "4) Using asynchronous processing"
                ),
                quality_attributes=[QualityAttribute.AVAILABILITY],
                metrics={"pagerank": item.score},
            ))
        
        return problems

    def _detect_connectivity_issues(self, projection_name: str) -> List[Problem]:
        """Detect connectivity and isolation issues"""
        problems = []
        
        try:
            _, stats = self.gds.weakly_connected_components(projection_name)
            
            if not stats["is_connected"]:
                problems.append(Problem(
                    problem_type=ProblemType.NO_REDUNDANCY,
                    severity=ProblemSeverity.HIGH,
                    title="Disconnected System Components",
                    description=(
                        f"The system has {stats['num_components']} disconnected parts. "
                        f"These cannot communicate with each other."
                    ),
                    affected_components=[],
                    symptoms=[
                        Symptom(
                            name="Multiple Components",
                            description="System is not fully connected",
                            metric="num_components",
                            value=stats["num_components"],
                            threshold=1,
                        ),
                    ],
                    impact="Parts of the system cannot communicate",
                    recommendation=(
                        "Connect system parts through: "
                        "1) Adding bridge topics, "
                        "2) Deploying gateway brokers, "
                        "3) Reviewing system architecture"
                    ),
                    quality_attributes=[QualityAttribute.AVAILABILITY],
                    metrics=stats,
                ))
        
        except Exception as e:
            self.logger.warning(f"WCC analysis failed: {e}")
        
        return problems

    # =========================================================================
    # Utilities
    # =========================================================================

    def _group_by_severity(
        self, 
        problems: List[Problem]
    ) -> Dict[ProblemSeverity, List[Problem]]:
        """Group problems by severity"""
        result: Dict[ProblemSeverity, List[Problem]] = {
            sev: [] for sev in ProblemSeverity
        }
        for problem in problems:
            result[problem.severity].append(problem)
        return result

    def _group_by_quality_attribute(
        self, 
        problems: List[Problem]
    ) -> Dict[QualityAttribute, List[Problem]]:
        """Group problems by quality attribute"""
        result: Dict[QualityAttribute, List[Problem]] = {
            qa: [] for qa in QualityAttribute
        }
        for problem in problems:
            for qa in problem.quality_attributes:
                result[qa].append(problem)
        return result

    def _generate_summary(
        self,
        problems: List[Problem],
        by_severity: Dict[ProblemSeverity, List[Problem]],
        by_qa: Dict[QualityAttribute, List[Problem]],
    ) -> Dict[str, Any]:
        """Generate summary statistics"""
        return {
            "total_problems": len(problems),
            "by_severity": {
                sev.value: len(probs) for sev, probs in by_severity.items()
            },
            "by_quality_attribute": {
                qa.value: len(probs) for qa, probs in by_qa.items()
            },
            "critical_count": len(by_severity.get(ProblemSeverity.CRITICAL, [])),
            "affected_components": list(set(
                comp for p in problems for comp in p.affected_components
            )),
        }
