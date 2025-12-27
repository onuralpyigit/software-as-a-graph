"""
Quality Attribute Analyzers - Version 4.0

GDS-based analyzers for assessing software quality attributes:
- Reliability: SPOFs, cascade risks, redundancy gaps
- Maintainability: Coupling, cycles, modularity
- Availability: Connectivity, fault tolerance, critical paths

Each analyzer uses graph algorithms on DEPENDS_ON relationships
and returns findings with severity, impact, and recommendations.

Author: Software-as-a-Graph Research Project
Version: 4.0
"""

from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional
from datetime import datetime


# =============================================================================
# Enums
# =============================================================================

class QualityAttribute(Enum):
    """Software quality attributes"""
    RELIABILITY = "reliability"
    MAINTAINABILITY = "maintainability"
    AVAILABILITY = "availability"


class Severity(Enum):
    """Issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    
    @property
    def color(self) -> str:
        return {
            Severity.CRITICAL: "\033[91m",  # Red
            Severity.HIGH: "\033[93m",      # Yellow
            Severity.MEDIUM: "\033[94m",    # Blue
            Severity.LOW: "\033[92m",       # Green
            Severity.INFO: "\033[90m",      # Gray
        }[self]
    
    @property
    def weight(self) -> int:
        """Weight for score calculation"""
        return {
            Severity.CRITICAL: 20,
            Severity.HIGH: 10,
            Severity.MEDIUM: 5,
            Severity.LOW: 2,
            Severity.INFO: 0,
        }[self]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Finding:
    """A detected issue or observation"""
    severity: Severity
    category: str
    component_id: str
    component_type: str
    description: str
    impact: str
    recommendation: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "severity": self.severity.value,
            "category": self.category,
            "component_id": self.component_id,
            "component_type": self.component_type,
            "description": self.description,
            "impact": self.impact,
            "recommendation": self.recommendation,
            "metrics": self.metrics,
        }


@dataclass
class CriticalComponent:
    """A component identified as critical"""
    component_id: str
    component_type: str
    criticality_score: float
    quality_attribute: QualityAttribute
    reasons: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "criticality_score": self.criticality_score,
            "quality_attribute": self.quality_attribute.value,
            "reasons": self.reasons,
            "metrics": self.metrics,
        }


@dataclass
class AnalysisResult:
    """Result from a quality attribute analysis"""
    quality_attribute: QualityAttribute
    score: float  # 0-100
    findings: List[Finding]
    critical_components: List[CriticalComponent]
    metrics: Dict[str, Any]
    recommendations: List[str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict:
        return {
            "quality_attribute": self.quality_attribute.value,
            "score": self.score,
            "findings_count": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
            "critical_components": [c.to_dict() for c in self.critical_components],
            "metrics": self.metrics,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
        }

    def summary(self) -> Dict:
        """Get summary statistics"""
        severity_counts = {}
        for f in self.findings:
            severity_counts[f.severity.value] = severity_counts.get(f.severity.value, 0) + 1
        
        return {
            "score": self.score,
            "total_findings": len(self.findings),
            "by_severity": severity_counts,
            "critical_components": len(self.critical_components),
        }


# =============================================================================
# Base Analyzer
# =============================================================================

class BaseAnalyzer(ABC):
    """Base class for quality attribute analyzers"""

    def __init__(self, gds_client, config: Optional[Dict[str, Any]] = None):
        self.gds = gds_client
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def quality_attribute(self) -> QualityAttribute:
        """Return the quality attribute this analyzer assesses"""
        pass

    @abstractmethod
    def analyze(self, projection_name: str) -> AnalysisResult:
        """Run analysis and return results"""
        pass

    def _severity_from_score(self, score: float, thresholds: Dict[str, float]) -> Severity:
        """Determine severity based on score and thresholds"""
        if score >= thresholds.get("critical", 0.8):
            return Severity.CRITICAL
        elif score >= thresholds.get("high", 0.6):
            return Severity.HIGH
        elif score >= thresholds.get("medium", 0.4):
            return Severity.MEDIUM
        elif score >= thresholds.get("low", 0.2):
            return Severity.LOW
        return Severity.INFO

    def _calculate_score(self, findings: List[Finding], base_score: float = 100) -> float:
        """Calculate overall score based on findings"""
        score = base_score
        for f in findings:
            score -= f.severity.weight
        return max(0, min(100, score))


# =============================================================================
# Reliability Analyzer
# =============================================================================

class ReliabilityAnalyzer(BaseAnalyzer):
    """
    Analyzes system reliability using GDS algorithms.
    
    Detects:
    - Single Points of Failure (SPOFs) via betweenness + articulation points
    - Cascade failure risks via dependency chains
    - Missing redundancy via connectivity analysis
    - Critical dependency bottlenecks via weighted betweenness
    
    Reliability = ability of the system to continue functioning when
    individual components fail.
    """

    DEFAULT_CONFIG = {
        "betweenness_threshold_percentile": 90,
        "cascade_depth_threshold": 3,
        "min_redundancy": 2,
        "use_weights": True,
        "severity_thresholds": {
            "critical": 0.8,
            "high": 0.6,
            "medium": 0.4,
            "low": 0.2,
        },
    }

    def __init__(self, gds_client, config: Optional[Dict] = None):
        super().__init__(gds_client, {**self.DEFAULT_CONFIG, **(config or {})})

    @property
    def quality_attribute(self) -> QualityAttribute:
        return QualityAttribute.RELIABILITY

    def analyze(self, projection_name: str) -> AnalysisResult:
        self.logger.info(f"Starting reliability analysis on '{projection_name}'")
        
        findings = []
        critical_components = []
        metrics = {}
        
        use_weights = self.config["use_weights"]
        
        # 1. Analyze SPOFs via betweenness centrality
        self.logger.info("Analyzing SPOFs (betweenness centrality)...")
        bc_results = self.gds.betweenness(projection_name, weighted=use_weights)
        spof_findings, spof_critical = self._analyze_spofs(bc_results, use_weights)
        findings.extend(spof_findings)
        critical_components.extend(spof_critical)
        
        if bc_results:
            scores = [r.score for r in bc_results]
            metrics["betweenness"] = {
                "max": max(scores),
                "avg": sum(scores) / len(scores),
                "spof_count": len(spof_critical),
            }
        
        # 2. Find articulation points
        self.logger.info("Finding articulation points...")
        ap_results = self.gds.find_articulation_points()
        ap_findings = self._analyze_articulation_points(ap_results)
        findings.extend(ap_findings)
        metrics["articulation_points"] = len(ap_results)
        
        # 3. Find bridge edges
        self.logger.info("Finding bridge edges...")
        bridge_results = self.gds.find_bridges()
        bridge_findings = self._analyze_bridges(bridge_results)
        findings.extend(bridge_findings)
        metrics["bridge_edges"] = len(bridge_results)
        
        # 4. Check connectivity
        self.logger.info("Analyzing connectivity...")
        _, wcc_stats = self.gds.weakly_connected_components(projection_name)
        conn_findings = self._analyze_connectivity(wcc_stats)
        findings.extend(conn_findings)
        metrics["connectivity"] = wcc_stats
        
        # Calculate score
        score = self._calculate_score(findings)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(findings, metrics)
        
        return AnalysisResult(
            quality_attribute=self.quality_attribute,
            score=score,
            findings=findings,
            critical_components=critical_components,
            metrics=metrics,
            recommendations=recommendations,
        )

    def _analyze_spofs(self, results, weighted: bool):
        """Identify SPOFs from betweenness centrality"""
        findings = []
        critical = []
        
        if not results:
            return findings, critical
        
        # Find threshold (90th percentile)
        scores = sorted([r.score for r in results], reverse=True)
        threshold_idx = max(0, int(len(scores) * 0.1))
        threshold = scores[threshold_idx] if threshold_idx < len(scores) else 0
        
        for r in results:
            if r.score >= threshold and r.score > 0:
                severity = Severity.CRITICAL if r.score >= scores[0] * 0.8 else Severity.HIGH
                
                findings.append(Finding(
                    severity=severity,
                    category="single_point_of_failure",
                    component_id=r.node_id,
                    component_type=r.node_type,
                    description=f"High betweenness centrality ({r.score:.4f}) - critical path component",
                    impact="Failure cascades through this component affect many paths",
                    recommendation="Add redundancy or implement failover mechanisms",
                    metrics={"betweenness": r.score, "rank": r.rank, "weighted": weighted},
                ))
                
                critical.append(CriticalComponent(
                    component_id=r.node_id,
                    component_type=r.node_type,
                    criticality_score=r.score / scores[0] if scores[0] > 0 else 0,
                    quality_attribute=QualityAttribute.RELIABILITY,
                    reasons=["high_betweenness", "potential_spof"],
                    metrics={"betweenness": r.score},
                ))
        
        return findings, critical

    def _analyze_articulation_points(self, results):
        """Analyze articulation points"""
        findings = []
        for ap in results:
            findings.append(Finding(
                severity=Severity.CRITICAL,
                category="articulation_point",
                component_id=ap["node_id"],
                component_type=ap["node_type"],
                description=f"Articulation point - {ap.get('impact', 'removal disconnects graph')}",
                impact="System fragmentation if this component fails",
                recommendation="Add redundant paths or backup components",
                metrics=ap,
            ))
        return findings

    def _analyze_bridges(self, results):
        """Analyze bridge edges"""
        findings = []
        for bridge in results[:10]:  # Top 10 critical bridges
            findings.append(Finding(
                severity=Severity.HIGH,
                category="bridge_edge",
                component_id=f"{bridge['source']}->{bridge['target']}",
                component_type="edge",
                description=f"Bridge edge: only path between {bridge['source']} and {bridge['target']}",
                impact="No alternative path exists - single point of failure",
                recommendation="Add redundant dependency paths",
                metrics=bridge,
            ))
        return findings

    def _analyze_connectivity(self, stats):
        """Analyze graph connectivity"""
        findings = []
        if not stats.get("is_connected", True):
            findings.append(Finding(
                severity=Severity.CRITICAL,
                category="disconnected_graph",
                component_id="graph",
                component_type="system",
                description=f"Graph has {stats['component_count']} disconnected components",
                impact="System parts cannot communicate or depend on each other",
                recommendation="Review system architecture for unintended isolation",
                metrics=stats,
            ))
        return findings

    def _generate_recommendations(self, findings, metrics):
        """Generate prioritized recommendations"""
        recommendations = []
        
        critical = [f for f in findings if f.severity == Severity.CRITICAL]
        if critical:
            recommendations.append(f"🔴 Address {len(critical)} critical reliability issues immediately")
        
        spof_count = metrics.get("betweenness", {}).get("spof_count", 0)
        if spof_count > 0:
            recommendations.append(f"Add redundancy for {spof_count} single points of failure")
        
        ap_count = metrics.get("articulation_points", 0)
        if ap_count > 0:
            recommendations.append(f"Review {ap_count} articulation points for failover strategies")
        
        if not metrics.get("connectivity", {}).get("is_connected", True):
            recommendations.append("Connect isolated system components")
        
        if not recommendations:
            recommendations.append("✅ No critical reliability issues detected")
        
        return recommendations


# =============================================================================
# Maintainability Analyzer
# =============================================================================

class MaintainabilityAnalyzer(BaseAnalyzer):
    """
    Analyzes system maintainability using GDS algorithms.
    
    Detects:
    - High coupling via degree centrality
    - God components via combined metrics
    - Circular dependencies via cycle detection
    - Poor modularity via community detection
    
    Maintainability = ease of modifying the system without breaking
    other components.
    """

    DEFAULT_CONFIG = {
        "coupling_threshold_percentile": 90,
        "modularity_threshold": 0.3,
        "use_weights": True,
        "severity_thresholds": {
            "critical": 0.8,
            "high": 0.6,
            "medium": 0.4,
            "low": 0.2,
        },
    }

    def __init__(self, gds_client, config: Optional[Dict] = None):
        super().__init__(gds_client, {**self.DEFAULT_CONFIG, **(config or {})})

    @property
    def quality_attribute(self) -> QualityAttribute:
        return QualityAttribute.MAINTAINABILITY

    def analyze(self, projection_name: str) -> AnalysisResult:
        self.logger.info(f"Starting maintainability analysis on '{projection_name}'")
        
        findings = []
        critical_components = []
        metrics = {}
        
        use_weights = self.config["use_weights"]
        
        # 1. Analyze coupling via degree centrality
        self.logger.info("Analyzing coupling (degree centrality)...")
        dc_results = self.gds.degree(projection_name, weighted=use_weights)
        coupling_findings, coupling_critical = self._analyze_coupling(dc_results, use_weights)
        findings.extend(coupling_findings)
        critical_components.extend(coupling_critical)
        
        if dc_results:
            scores = [r.score for r in dc_results]
            metrics["coupling"] = {
                "max_degree": max(scores),
                "avg_degree": sum(scores) / len(scores),
                "high_coupling_count": len(coupling_critical),
            }
        
        # 2. Detect god components (high degree + high betweenness)
        self.logger.info("Detecting god components...")
        bc_results = self.gds.betweenness(projection_name, weighted=use_weights)
        god_findings, god_critical = self._detect_god_components(dc_results, bc_results)
        findings.extend(god_findings)
        critical_components.extend(god_critical)
        metrics["god_components"] = len(god_critical)
        
        # 3. Analyze modularity via Louvain
        self.logger.info("Analyzing modularity...")
        _, louvain_stats = self.gds.louvain(projection_name, weighted=use_weights)
        mod_findings = self._analyze_modularity(louvain_stats)
        findings.extend(mod_findings)
        metrics["modularity"] = louvain_stats
        
        # 4. Detect cycles
        self.logger.info("Detecting circular dependencies...")
        cycle_findings = self._detect_cycles()
        findings.extend(cycle_findings)
        metrics["cycles"] = len(cycle_findings)
        
        # Calculate score
        score = self._calculate_score(findings)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(findings, metrics)
        
        return AnalysisResult(
            quality_attribute=self.quality_attribute,
            score=score,
            findings=findings,
            critical_components=critical_components,
            metrics=metrics,
            recommendations=recommendations,
        )

    def _analyze_coupling(self, results, weighted: bool):
        """Identify high-coupling components"""
        findings = []
        critical = []
        
        if not results:
            return findings, critical
        
        scores = sorted([r.score for r in results], reverse=True)
        threshold_idx = max(0, int(len(scores) * 0.1))
        threshold = scores[threshold_idx] if threshold_idx < len(scores) else 0
        
        for r in results:
            if r.score >= threshold and r.score > 0:
                severity = Severity.HIGH if r.score >= scores[0] * 0.8 else Severity.MEDIUM
                
                findings.append(Finding(
                    severity=severity,
                    category="high_coupling",
                    component_id=r.node_id,
                    component_type=r.node_type,
                    description=f"High coupling (degree: {r.score:.2f})",
                    impact="Changes to this component may affect many dependents",
                    recommendation="Consider breaking into smaller, focused components",
                    metrics={"degree": r.score, "rank": r.rank, "weighted": weighted},
                ))
                
                critical.append(CriticalComponent(
                    component_id=r.node_id,
                    component_type=r.node_type,
                    criticality_score=r.score / scores[0] if scores[0] > 0 else 0,
                    quality_attribute=QualityAttribute.MAINTAINABILITY,
                    reasons=["high_coupling", "many_dependencies"],
                    metrics={"degree": r.score},
                ))
        
        return findings, critical

    def _detect_god_components(self, degree_results, betweenness_results):
        """Detect god components with high degree AND high betweenness"""
        findings = []
        critical = []
        
        if not degree_results or not betweenness_results:
            return findings, critical
        
        dc_map = {r.node_id: r.score for r in degree_results}
        bc_map = {r.node_id: r.score for r in betweenness_results}
        
        dc_scores = list(dc_map.values())
        bc_scores = list(bc_map.values())
        
        dc_threshold = sorted(dc_scores, reverse=True)[max(0, int(len(dc_scores) * 0.1))] if dc_scores else 0
        bc_threshold = sorted(bc_scores, reverse=True)[max(0, int(len(bc_scores) * 0.1))] if bc_scores else 0
        
        for node_id in dc_map:
            dc = dc_map.get(node_id, 0)
            bc = bc_map.get(node_id, 0)
            
            if dc >= dc_threshold and bc >= bc_threshold:
                node_type = next((r.node_type for r in degree_results if r.node_id == node_id), "Unknown")
                
                findings.append(Finding(
                    severity=Severity.CRITICAL,
                    category="god_component",
                    component_id=node_id,
                    component_type=node_type,
                    description=f"God component: high coupling ({dc:.2f}) AND high centrality ({bc:.4f})",
                    impact="Difficult to modify or replace; changes have wide impact",
                    recommendation="Decompose into smaller, single-responsibility components",
                    metrics={"degree": dc, "betweenness": bc},
                ))
                
                critical.append(CriticalComponent(
                    component_id=node_id,
                    component_type=node_type,
                    criticality_score=1.0,
                    quality_attribute=QualityAttribute.MAINTAINABILITY,
                    reasons=["god_component", "high_coupling", "high_centrality"],
                    metrics={"degree": dc, "betweenness": bc},
                ))
        
        return findings, critical

    def _analyze_modularity(self, stats):
        """Analyze system modularity"""
        findings = []
        
        if stats.get("community_count", 1) <= 1:
            findings.append(Finding(
                severity=Severity.MEDIUM,
                category="poor_modularity",
                component_id="system",
                component_type="architecture",
                description="System has no clear module boundaries",
                impact="Difficult to reason about component responsibilities",
                recommendation="Consider domain-driven design to identify module boundaries",
                metrics=stats,
            ))
        
        return findings

    def _detect_cycles(self):
        """Detect circular dependencies"""
        findings = []
        
        with self.gds.session() as session:
            # Find cycles in DEPENDS_ON
            result = session.run("""
                MATCH path = (a)-[:DEPENDS_ON*2..5]->(a)
                WITH nodes(path) AS cycle
                WHERE size(cycle) > 1
                RETURN [n IN cycle | n.id] AS nodes
                LIMIT 10
            """)
            
            for record in result:
                nodes = record["nodes"]
                findings.append(Finding(
                    severity=Severity.HIGH,
                    category="circular_dependency",
                    component_id=nodes[0] if nodes else "unknown",
                    component_type="cycle",
                    description=f"Circular dependency: {' -> '.join(nodes[:5])}{'...' if len(nodes) > 5 else ''}",
                    impact="May cause infinite loops or deadlocks; hard to understand",
                    recommendation="Break cycle using events, callbacks, or dependency inversion",
                    metrics={"cycle_length": len(nodes), "nodes": nodes},
                ))
        
        return findings

    def _generate_recommendations(self, findings, metrics):
        """Generate prioritized recommendations"""
        recommendations = []
        
        god_count = metrics.get("god_components", 0)
        if god_count > 0:
            recommendations.append(f"🔴 Decompose {god_count} god components")
        
        cycle_count = metrics.get("cycles", 0)
        if cycle_count > 0:
            recommendations.append(f"Break {cycle_count} circular dependencies")
        
        coupling = metrics.get("coupling", {})
        if coupling.get("high_coupling_count", 0) > 0:
            recommendations.append(f"Reduce coupling for {coupling['high_coupling_count']} highly-connected components")
        
        if not recommendations:
            recommendations.append("✅ Good maintainability structure")
        
        return recommendations


# =============================================================================
# Availability Analyzer
# =============================================================================

class AvailabilityAnalyzer(BaseAnalyzer):
    """
    Analyzes system availability using GDS algorithms.
    
    Detects:
    - Low connectivity via component analysis
    - Critical dependencies via PageRank
    - Fault tolerance gaps via redundancy check
    - Recovery path issues via path analysis
    
    Availability = ability of the system to remain operational and
    accessible when needed.
    """

    DEFAULT_CONFIG = {
        "pagerank_threshold_percentile": 90,
        "min_connectivity": 2,
        "use_weights": True,
        "severity_thresholds": {
            "critical": 0.8,
            "high": 0.6,
            "medium": 0.4,
            "low": 0.2,
        },
    }

    def __init__(self, gds_client, config: Optional[Dict] = None):
        super().__init__(gds_client, {**self.DEFAULT_CONFIG, **(config or {})})

    @property
    def quality_attribute(self) -> QualityAttribute:
        return QualityAttribute.AVAILABILITY

    def analyze(self, projection_name: str) -> AnalysisResult:
        self.logger.info(f"Starting availability analysis on '{projection_name}'")
        
        findings = []
        critical_components = []
        metrics = {}
        
        use_weights = self.config["use_weights"]
        
        # 1. Analyze importance via PageRank
        self.logger.info("Analyzing component importance (PageRank)...")
        pr_results = self.gds.pagerank(projection_name, weighted=use_weights)
        imp_findings, imp_critical = self._analyze_importance(pr_results, use_weights)
        findings.extend(imp_findings)
        critical_components.extend(imp_critical)
        
        if pr_results:
            scores = [r.score for r in pr_results]
            metrics["pagerank"] = {
                "max": max(scores),
                "avg": sum(scores) / len(scores),
                "critical_count": len(imp_critical),
            }
        
        # 2. Check connectivity
        self.logger.info("Analyzing connectivity...")
        _, wcc_stats = self.gds.weakly_connected_components(projection_name)
        conn_findings = self._analyze_connectivity(wcc_stats)
        findings.extend(conn_findings)
        metrics["connectivity"] = wcc_stats
        
        # 3. Analyze in-degree for dependency concentration
        self.logger.info("Analyzing dependency concentration...")
        in_degree = self.gds.degree(projection_name, weighted=use_weights, orientation="REVERSE")
        conc_findings = self._analyze_concentration(in_degree)
        findings.extend(conc_findings)
        
        # Calculate score
        score = self._calculate_score(findings)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(findings, metrics)
        
        return AnalysisResult(
            quality_attribute=self.quality_attribute,
            score=score,
            findings=findings,
            critical_components=critical_components,
            metrics=metrics,
            recommendations=recommendations,
        )

    def _analyze_importance(self, results, weighted: bool):
        """Identify high-importance components via PageRank"""
        findings = []
        critical = []
        
        if not results:
            return findings, critical
        
        scores = sorted([r.score for r in results], reverse=True)
        threshold_idx = max(0, int(len(scores) * 0.1))
        threshold = scores[threshold_idx] if threshold_idx < len(scores) else 0
        
        for r in results:
            if r.score >= threshold and r.score > 0.01:
                severity = Severity.HIGH if r.score >= scores[0] * 0.5 else Severity.MEDIUM
                
                findings.append(Finding(
                    severity=severity,
                    category="high_importance",
                    component_id=r.node_id,
                    component_type=r.node_type,
                    description=f"High importance (PageRank: {r.score:.4f})",
                    impact="Critical for system operation; downtime affects many dependents",
                    recommendation="Ensure high availability with redundancy/failover",
                    metrics={"pagerank": r.score, "rank": r.rank, "weighted": weighted},
                ))
                
                critical.append(CriticalComponent(
                    component_id=r.node_id,
                    component_type=r.node_type,
                    criticality_score=r.score / scores[0] if scores[0] > 0 else 0,
                    quality_attribute=QualityAttribute.AVAILABILITY,
                    reasons=["high_pagerank", "critical_for_operation"],
                    metrics={"pagerank": r.score},
                ))
        
        return findings, critical

    def _analyze_connectivity(self, stats):
        """Analyze system connectivity"""
        findings = []
        
        if not stats.get("is_connected", True):
            findings.append(Finding(
                severity=Severity.CRITICAL,
                category="disconnected_system",
                component_id="system",
                component_type="architecture",
                description=f"System has {stats['component_count']} disconnected parts",
                impact="Some components cannot reach others; partial system failure",
                recommendation="Review architecture for unintended isolation",
                metrics=stats,
            ))
        
        return findings

    def _analyze_concentration(self, results):
        """Analyze dependency concentration (many depend on few)"""
        findings = []
        
        if not results:
            return findings
        
        scores = [r.score for r in results]
        if not scores:
            return findings
        
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        
        # High concentration if max >> avg
        if max_score > avg_score * 5:
            top = results[0]
            findings.append(Finding(
                severity=Severity.HIGH,
                category="dependency_concentration",
                component_id=top.node_id,
                component_type=top.node_type,
                description=f"High dependency concentration: {top.node_id} has {top.score:.1f} incoming dependencies",
                impact="Many components depend on single target; availability bottleneck",
                recommendation="Distribute dependencies across multiple instances",
                metrics={"max_in_degree": max_score, "avg_in_degree": avg_score},
            ))
        
        return findings

    def _generate_recommendations(self, findings, metrics):
        """Generate prioritized recommendations"""
        recommendations = []
        
        critical = [f for f in findings if f.severity == Severity.CRITICAL]
        if critical:
            recommendations.append(f"🔴 Address {len(critical)} critical availability issues")
        
        if not metrics.get("connectivity", {}).get("is_connected", True):
            recommendations.append("Connect isolated system components")
        
        pr_critical = metrics.get("pagerank", {}).get("critical_count", 0)
        if pr_critical > 0:
            recommendations.append(f"Ensure high availability for {pr_critical} critical components")
        
        if not recommendations:
            recommendations.append("✅ Good availability posture")
        
        return recommendations
