#!/usr/bin/env python3
"""
GDS-Based Quality Attribute Analyzers
=====================================

Simplified analyzers using Neo4j Graph Data Science for:
- Reliability: SPOFs, cascade risks, redundancy
- Maintainability: Coupling, communities, cycles
- Availability: Connectivity, paths, fault tolerance

These analyzers run algorithms directly in Neo4j via GDS,
eliminating the need to load graphs into Python memory.

Author: Software-as-a-Graph Research Project
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .gds_client import GDSClient, CentralityResult, CommunityResult


# ============================================================================
# Enums and Data Classes
# ============================================================================

class Severity(Enum):
    """Issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class QualityAttribute(Enum):
    """Quality attributes analyzed"""
    RELIABILITY = "reliability"
    MAINTAINABILITY = "maintainability"
    AVAILABILITY = "availability"


@dataclass
class Finding:
    """A single analysis finding"""
    severity: Severity
    category: str
    component_id: str
    component_type: str
    description: str
    impact: str
    recommendation: str
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CriticalComponent:
    """A critical component identified by analysis"""
    component_id: str
    component_type: str
    criticality_score: float
    quality_attribute: QualityAttribute
    reasons: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Result from a quality attribute analysis"""
    quality_attribute: QualityAttribute
    score: float  # 0-100
    findings: List[Finding]
    critical_components: List[CriticalComponent]
    metrics: Dict[str, Any]
    recommendations: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'quality_attribute': self.quality_attribute.value,
            'score': round(self.score, 2),
            'findings_count': len(self.findings),
            'critical_count': len(self.critical_components),
            'findings': [
                {
                    'severity': f.severity.value,
                    'category': f.category,
                    'component': f.component_id,
                    'description': f.description,
                    'impact': f.impact,
                    'recommendation': f.recommendation,
                    'metrics': f.metrics
                }
                for f in self.findings
            ],
            'critical_components': [
                {
                    'id': c.component_id,
                    'type': c.component_type,
                    'score': round(c.criticality_score, 4),
                    'reasons': c.reasons,
                    'metrics': c.metrics
                }
                for c in self.critical_components
            ],
            'metrics': self.metrics,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp
        }


# ============================================================================
# Base Analyzer
# ============================================================================

class BaseGDSAnalyzer(ABC):
    """Base class for GDS-based analyzers"""
    
    def __init__(self, gds_client: GDSClient, config: Optional[Dict[str, Any]] = None):
        self.gds = gds_client
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @property
    @abstractmethod
    def quality_attribute(self) -> QualityAttribute:
        """Return the quality attribute this analyzer handles"""
        pass
    
    @abstractmethod
    def analyze(self, projection_name: str) -> AnalysisResult:
        """Run analysis on the given projection"""
        pass
    
    def _severity_from_score(self, score: float, thresholds: Dict[str, float]) -> Severity:
        """Convert score to severity based on thresholds"""
        if score >= thresholds.get('critical', 0.8):
            return Severity.CRITICAL
        elif score >= thresholds.get('high', 0.6):
            return Severity.HIGH
        elif score >= thresholds.get('medium', 0.4):
            return Severity.MEDIUM
        return Severity.LOW


# ============================================================================
# Reliability Analyzer
# ============================================================================

class ReliabilityAnalyzer(BaseGDSAnalyzer):
    """
    Analyzes system reliability using GDS algorithms.
    
    Detects:
    - Single Points of Failure (SPOFs) via articulation point analysis
    - Cascade failure risks via betweenness centrality
    - Missing redundancy via connectivity analysis
    - Critical dependency chains via path analysis
    """
    
    DEFAULT_CONFIG = {
        'betweenness_threshold_percentile': 90,
        'spof_min_connections': 2,
        'severity_thresholds': {
            'critical': 0.8,
            'high': 0.6,
            'medium': 0.4
        }
    }
    
    def __init__(self, gds_client: GDSClient, config: Optional[Dict[str, Any]] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(gds_client, merged_config)
    
    @property
    def quality_attribute(self) -> QualityAttribute:
        return QualityAttribute.RELIABILITY
    
    def analyze(self, projection_name: str) -> AnalysisResult:
        """Run reliability analysis"""
        self.logger.info(f"Starting reliability analysis on '{projection_name}'...")
        
        findings = []
        critical_components = []
        metrics = {}
        
        # 1. Analyze betweenness centrality for bottlenecks
        self.logger.info("Analyzing betweenness centrality...")
        bc_results = self.gds.betweenness_centrality(projection_name)
        bc_findings, bc_critical = self._analyze_betweenness(bc_results)
        findings.extend(bc_findings)
        critical_components.extend(bc_critical)
        
        if bc_results:
            scores = [r.score for r in bc_results]
            metrics['betweenness'] = {
                'max': max(scores),
                'avg': sum(scores) / len(scores),
                'high_centrality_count': len([s for s in scores if s > 0.1])
            }
        
        # 2. Find articulation points (SPOFs)
        self.logger.info("Finding Single Points of Failure...")
        spofs = self.gds.find_articulation_points()
        spof_findings, spof_critical = self._analyze_spofs(spofs)
        findings.extend(spof_findings)
        critical_components.extend(spof_critical)
        metrics['spof_count'] = len(spofs)
        
        # 3. Find bridge edges
        self.logger.info("Finding bridge edges...")
        bridges = self.gds.find_bridge_edges()
        bridge_findings = self._analyze_bridges(bridges)
        findings.extend(bridge_findings)
        metrics['bridge_count'] = len(bridges)
        
        # 4. Check connectivity
        self.logger.info("Analyzing connectivity...")
        components, comp_stats = self.gds.weakly_connected_components(projection_name)
        if not comp_stats['is_connected']:
            findings.append(Finding(
                severity=Severity.HIGH,
                category='disconnected_graph',
                component_id='system',
                component_type='System',
                description=f"Graph has {comp_stats['component_count']} disconnected components",
                impact="Some components cannot communicate with others",
                recommendation="Review network topology and add missing connections",
                metrics=comp_stats
            ))
        metrics['component_count'] = comp_stats['component_count']
        metrics['is_connected'] = comp_stats['is_connected']
        
        # Calculate reliability score
        score = self._calculate_score(findings, metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(findings, metrics)
        
        # Sort by severity
        findings.sort(key=lambda f: list(Severity).index(f.severity))
        critical_components.sort(key=lambda c: -c.criticality_score)
        
        self.logger.info(f"Reliability analysis complete. Score: {score:.1f}/100")
        
        return AnalysisResult(
            quality_attribute=QualityAttribute.RELIABILITY,
            score=score,
            findings=findings,
            critical_components=critical_components[:20],
            metrics=metrics,
            recommendations=recommendations
        )
    
    def _analyze_betweenness(self, results: List[CentralityResult]) -> Tuple[List[Finding], List[CriticalComponent]]:
        """Analyze betweenness centrality results"""
        findings = []
        critical = []
        
        if not results:
            return findings, critical
        
        # Calculate threshold (90th percentile)
        scores = sorted([r.score for r in results], reverse=True)
        threshold_idx = max(0, int(len(scores) * 0.1))
        threshold = scores[threshold_idx] if threshold_idx < len(scores) else 0
        
        for result in results:
            if result.score > threshold and result.score > 0.05:
                severity = self._severity_from_score(
                    result.score,
                    self.config['severity_thresholds']
                )
                
                findings.append(Finding(
                    severity=severity,
                    category='reliability_bottleneck',
                    component_id=result.node_id,
                    component_type=result.node_type,
                    description=f"High betweenness centrality ({result.score:.4f})",
                    impact="Acts as critical bridge - failure impacts many paths",
                    recommendation="Add redundant paths or load balancing",
                    metrics={'betweenness': result.score, 'rank': result.rank}
                ))
                
                critical.append(CriticalComponent(
                    component_id=result.node_id,
                    component_type=result.node_type,
                    criticality_score=result.score,
                    quality_attribute=QualityAttribute.RELIABILITY,
                    reasons=['high_betweenness', 'traffic_bottleneck'],
                    metrics={'betweenness': result.score, 'rank': result.rank}
                ))
        
        return findings, critical
    
    def _analyze_spofs(self, spofs: List[Dict[str, Any]]) -> Tuple[List[Finding], List[CriticalComponent]]:
        """Analyze articulation points"""
        findings = []
        critical = []
        
        for spof in spofs:
            # Normalize criticality to 0-1
            max_criticality = max([s['criticality_score'] for s in spofs]) if spofs else 1
            norm_score = spof['criticality_score'] / max_criticality if max_criticality > 0 else 0
            
            severity = Severity.CRITICAL if norm_score > 0.7 else Severity.HIGH
            
            findings.append(Finding(
                severity=severity,
                category='single_point_of_failure',
                component_id=spof['node_id'],
                component_type=spof['node_type'],
                description=f"SPOF: {spof['connections']} connections, {spof['critical_paths']} critical paths",
                impact="Removal disconnects or severely impacts the system",
                recommendation="Add redundant component or failover mechanism",
                metrics=spof
            ))
            
            critical.append(CriticalComponent(
                component_id=spof['node_id'],
                component_type=spof['node_type'],
                criticality_score=norm_score,
                quality_attribute=QualityAttribute.RELIABILITY,
                reasons=['articulation_point', 'single_point_of_failure'],
                metrics=spof
            ))
        
        return findings, critical
    
    def _analyze_bridges(self, bridges: List[Dict[str, Any]]) -> List[Finding]:
        """Analyze bridge edges"""
        findings = []
        
        for bridge in bridges[:10]:  # Limit to top 10
            findings.append(Finding(
                severity=Severity.MEDIUM,
                category='bridge_edge',
                component_id=f"{bridge['source']}->{bridge['target']}",
                component_type='Edge',
                description=f"Bridge edge: {bridge['dependency_type']}",
                impact="Only connection between components - no redundant path",
                recommendation="Add alternative dependency path",
                metrics=bridge
            ))
        
        return findings
    
    def _calculate_score(self, findings: List[Finding], metrics: Dict[str, Any]) -> float:
        """Calculate overall reliability score"""
        score = 100.0
        
        # Deduct for SPOFs
        spof_count = metrics.get('spof_count', 0)
        score -= min(30, spof_count * 5)
        
        # Deduct for bridges
        bridge_count = metrics.get('bridge_count', 0)
        score -= min(15, bridge_count * 2)
        
        # Deduct for disconnected graph
        if not metrics.get('is_connected', True):
            score -= 20
        
        # Deduct for critical findings
        critical_count = len([f for f in findings if f.severity == Severity.CRITICAL])
        high_count = len([f for f in findings if f.severity == Severity.HIGH])
        score -= critical_count * 10
        score -= high_count * 5
        
        return max(0, min(100, score))
    
    def _generate_recommendations(self, findings: List[Finding], metrics: Dict[str, Any]) -> List[str]:
        """Generate prioritized recommendations"""
        recommendations = []
        
        critical_count = len([f for f in findings if f.severity == Severity.CRITICAL])
        if critical_count > 0:
            recommendations.append(f"🔴 Address {critical_count} critical reliability issues immediately")
        
        if metrics.get('spof_count', 0) > 0:
            recommendations.append(f"Add redundancy for {metrics['spof_count']} single points of failure")
        
        if metrics.get('bridge_count', 0) > 3:
            recommendations.append(f"Add redundant paths for {metrics['bridge_count']} bridge connections")
        
        if not metrics.get('is_connected', True):
            recommendations.append("Connect disconnected components to ensure system-wide communication")
        
        if not recommendations:
            recommendations.append("✅ No critical reliability issues detected")
        
        return recommendations


# ============================================================================
# Maintainability Analyzer
# ============================================================================

class MaintainabilityAnalyzer(BaseGDSAnalyzer):
    """
    Analyzes system maintainability using GDS algorithms.
    
    Detects:
    - High coupling via degree centrality
    - God components via combined metrics
    - Circular dependencies via cycle detection
    - Poor modularity via community detection
    """
    
    DEFAULT_CONFIG = {
        'coupling_threshold': 10,
        'god_component_threshold': 15,
        'modularity_threshold': 0.3,
        'severity_thresholds': {
            'critical': 0.8,
            'high': 0.6,
            'medium': 0.4
        }
    }
    
    def __init__(self, gds_client: GDSClient, config: Optional[Dict[str, Any]] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(gds_client, merged_config)
    
    @property
    def quality_attribute(self) -> QualityAttribute:
        return QualityAttribute.MAINTAINABILITY
    
    def analyze(self, projection_name: str) -> AnalysisResult:
        """Run maintainability analysis"""
        self.logger.info(f"Starting maintainability analysis on '{projection_name}'...")
        
        findings = []
        critical_components = []
        metrics = {}
        
        # 1. Analyze degree centrality for coupling
        self.logger.info("Analyzing coupling (degree centrality)...")
        in_degree = self.gds.degree_centrality(projection_name, orientation='REVERSE')
        out_degree = self.gds.degree_centrality(projection_name, orientation='NATURAL')
        
        coupling_findings, coupling_critical = self._analyze_coupling(in_degree, out_degree)
        findings.extend(coupling_findings)
        critical_components.extend(coupling_critical)
        
        if in_degree:
            metrics['coupling'] = {
                'max_in_degree': max(r.score for r in in_degree),
                'max_out_degree': max(r.score for r in out_degree) if out_degree else 0,
                'high_coupling_count': len([r for r in in_degree if r.score > self.config['coupling_threshold']])
            }
        
        # 2. Detect communities for modularity
        self.logger.info("Detecting communities...")
        communities, comm_stats = self.gds.louvain_communities(projection_name)
        modularity_findings = self._analyze_modularity(communities, comm_stats)
        findings.extend(modularity_findings)
        metrics['modularity'] = comm_stats.get('modularity', 0)
        metrics['community_count'] = comm_stats.get('community_count', 1)
        
        # 3. Find circular dependencies
        self.logger.info("Finding circular dependencies...")
        cycles = self.gds.find_cycles(max_length=6)
        cycle_findings = self._analyze_cycles(cycles)
        findings.extend(cycle_findings)
        metrics['cycle_count'] = len(cycles)
        
        # 4. Detect god components
        god_findings, god_critical = self._detect_god_components(in_degree, out_degree)
        findings.extend(god_findings)
        critical_components.extend(god_critical)
        
        # Calculate maintainability score
        score = self._calculate_score(findings, metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(findings, metrics)
        
        # Sort by severity
        findings.sort(key=lambda f: list(Severity).index(f.severity))
        critical_components.sort(key=lambda c: -c.criticality_score)
        
        self.logger.info(f"Maintainability analysis complete. Score: {score:.1f}/100")
        
        return AnalysisResult(
            quality_attribute=QualityAttribute.MAINTAINABILITY,
            score=score,
            findings=findings,
            critical_components=critical_components[:20],
            metrics=metrics,
            recommendations=recommendations
        )
    
    def _analyze_coupling(self, in_degree: List[CentralityResult], 
                          out_degree: List[CentralityResult]) -> Tuple[List[Finding], List[CriticalComponent]]:
        """Analyze coupling via degree metrics"""
        findings = []
        critical = []
        
        # Create lookup for out-degree
        out_lookup = {r.node_id: r.score for r in out_degree}
        
        threshold = self.config['coupling_threshold']
        
        for result in in_degree:
            total_coupling = result.score + out_lookup.get(result.node_id, 0)
            
            if total_coupling > threshold:
                norm_score = min(1.0, total_coupling / (threshold * 2))
                severity = self._severity_from_score(norm_score, self.config['severity_thresholds'])
                
                findings.append(Finding(
                    severity=severity,
                    category='high_coupling',
                    component_id=result.node_id,
                    component_type=result.node_type,
                    description=f"High coupling: {result.score:.0f} in, {out_lookup.get(result.node_id, 0):.0f} out",
                    impact="Changes affect many components; hard to modify",
                    recommendation="Reduce dependencies; introduce abstraction layers",
                    metrics={
                        'in_degree': result.score,
                        'out_degree': out_lookup.get(result.node_id, 0),
                        'total_coupling': total_coupling
                    }
                ))
                
                critical.append(CriticalComponent(
                    component_id=result.node_id,
                    component_type=result.node_type,
                    criticality_score=norm_score,
                    quality_attribute=QualityAttribute.MAINTAINABILITY,
                    reasons=['high_coupling', 'many_dependencies'],
                    metrics={'total_coupling': total_coupling}
                ))
        
        return findings, critical
    
    def _analyze_modularity(self, communities: List[CommunityResult], 
                            stats: Dict[str, Any]) -> List[Finding]:
        """Analyze modularity via community detection"""
        findings = []
        
        modularity = stats.get('modularity', 0)
        threshold = self.config['modularity_threshold']
        
        if modularity < threshold:
            severity = Severity.MEDIUM if modularity > threshold / 2 else Severity.HIGH
            findings.append(Finding(
                severity=severity,
                category='low_modularity',
                component_id='system',
                component_type='System',
                description=f"Low modularity score: {modularity:.4f} (threshold: {threshold})",
                impact="Poor separation of concerns; changes have wide impact",
                recommendation="Reorganize components into more cohesive modules",
                metrics={'modularity': modularity, 'threshold': threshold}
            ))
        
        # Check for imbalanced communities
        sizes = list(stats.get('community_sizes', {}).values())
        if sizes:
            max_size = max(sizes)
            avg_size = sum(sizes) / len(sizes)
            if max_size > avg_size * 3 and max_size > 5:
                findings.append(Finding(
                    severity=Severity.MEDIUM,
                    category='imbalanced_modules',
                    component_id='system',
                    component_type='System',
                    description=f"Imbalanced community sizes (max: {max_size}, avg: {avg_size:.1f})",
                    impact="Some modules are too large; difficult to maintain",
                    recommendation="Split large modules into smaller, focused units",
                    metrics={'max_size': max_size, 'avg_size': avg_size, 'sizes': sizes}
                ))
        
        return findings
    
    def _analyze_cycles(self, cycles: List[List[str]]) -> List[Finding]:
        """Analyze circular dependencies"""
        findings = []
        
        for i, cycle in enumerate(cycles[:10]):  # Limit to 10
            severity = Severity.HIGH if len(cycle) > 3 else Severity.MEDIUM
            
            cycle_str = ' → '.join(cycle[:5])
            if len(cycle) > 5:
                cycle_str += f' → ... ({len(cycle)} total)'
            
            findings.append(Finding(
                severity=severity,
                category='circular_dependency',
                component_id=f'cycle_{i+1}',
                component_type='Cycle',
                description=f"Circular dependency: {cycle_str}",
                impact="Can cause infinite loops; hard to reason about",
                recommendation="Break cycle with async communication or redesign",
                metrics={'cycle_length': len(cycle), 'nodes': cycle}
            ))
        
        return findings
    
    def _detect_god_components(self, in_degree: List[CentralityResult],
                                out_degree: List[CentralityResult]) -> Tuple[List[Finding], List[CriticalComponent]]:
        """Detect god components with excessive responsibility"""
        findings = []
        critical = []
        
        out_lookup = {r.node_id: r.score for r in out_degree}
        threshold = self.config['god_component_threshold']
        
        for result in in_degree:
            total = result.score + out_lookup.get(result.node_id, 0)
            
            if total > threshold:
                norm_score = min(1.0, total / (threshold * 2))
                
                findings.append(Finding(
                    severity=Severity.HIGH,
                    category='god_component',
                    component_id=result.node_id,
                    component_type=result.node_type,
                    description=f"God component: {total:.0f} total connections",
                    impact="Knows too much; changes are risky; hard to test",
                    recommendation="Split into smaller, focused components",
                    metrics={'total_connections': total}
                ))
                
                critical.append(CriticalComponent(
                    component_id=result.node_id,
                    component_type=result.node_type,
                    criticality_score=norm_score,
                    quality_attribute=QualityAttribute.MAINTAINABILITY,
                    reasons=['god_component', 'excessive_responsibility'],
                    metrics={'total_connections': total}
                ))
        
        return findings, critical
    
    def _calculate_score(self, findings: List[Finding], metrics: Dict[str, Any]) -> float:
        """Calculate overall maintainability score"""
        score = 100.0
        
        # Deduct for low modularity
        modularity = metrics.get('modularity', 0.5)
        if modularity < 0.3:
            score -= 15
        elif modularity < 0.4:
            score -= 8
        
        # Deduct for cycles
        cycle_count = metrics.get('cycle_count', 0)
        score -= min(20, cycle_count * 5)
        
        # Deduct for findings
        critical_count = len([f for f in findings if f.severity == Severity.CRITICAL])
        high_count = len([f for f in findings if f.severity == Severity.HIGH])
        score -= critical_count * 10
        score -= high_count * 5
        
        return max(0, min(100, score))
    
    def _generate_recommendations(self, findings: List[Finding], metrics: Dict[str, Any]) -> List[str]:
        """Generate prioritized recommendations"""
        recommendations = []
        
        if metrics.get('cycle_count', 0) > 0:
            recommendations.append(f"Break {metrics['cycle_count']} circular dependencies")
        
        if metrics.get('modularity', 1) < 0.3:
            recommendations.append("Improve modularity by reorganizing component boundaries")
        
        high_coupling = metrics.get('coupling', {}).get('high_coupling_count', 0)
        if high_coupling > 0:
            recommendations.append(f"Reduce coupling for {high_coupling} highly-connected components")
        
        god_count = len([f for f in findings if f.category == 'god_component'])
        if god_count > 0:
            recommendations.append(f"Split {god_count} god components into smaller units")
        
        if not recommendations:
            recommendations.append("✅ Good maintainability structure")
        
        return recommendations


# ============================================================================
# Availability Analyzer
# ============================================================================

class AvailabilityAnalyzer(BaseGDSAnalyzer):
    """
    Analyzes system availability using GDS algorithms.
    
    Detects:
    - Low connectivity via component analysis
    - Fault tolerance gaps via redundancy check
    - Recovery path issues via path analysis
    - Uptime threats via PageRank importance
    """
    
    DEFAULT_CONFIG = {
        'min_connectivity': 2,
        'min_path_redundancy': 2,
        'pagerank_threshold_percentile': 90,
        'severity_thresholds': {
            'critical': 0.8,
            'high': 0.6,
            'medium': 0.4
        }
    }
    
    def __init__(self, gds_client: GDSClient, config: Optional[Dict[str, Any]] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(gds_client, merged_config)
    
    @property
    def quality_attribute(self) -> QualityAttribute:
        return QualityAttribute.AVAILABILITY
    
    def analyze(self, projection_name: str) -> AnalysisResult:
        """Run availability analysis"""
        self.logger.info(f"Starting availability analysis on '{projection_name}'...")
        
        findings = []
        critical_components = []
        metrics = {}
        
        # 1. Check connectivity
        self.logger.info("Analyzing connectivity...")
        components, comp_stats = self.gds.weakly_connected_components(projection_name)
        conn_findings = self._analyze_connectivity(comp_stats)
        findings.extend(conn_findings)
        metrics['component_count'] = comp_stats['component_count']
        metrics['largest_component'] = comp_stats['largest_component']
        metrics['is_connected'] = comp_stats['is_connected']
        
        # 2. Analyze PageRank for importance
        self.logger.info("Analyzing component importance (PageRank)...")
        pr_results = self.gds.pagerank(projection_name)
        pr_findings, pr_critical = self._analyze_importance(pr_results)
        findings.extend(pr_findings)
        critical_components.extend(pr_critical)
        
        if pr_results:
            scores = [r.score for r in pr_results]
            metrics['pagerank'] = {
                'max': max(scores),
                'avg': sum(scores) / len(scores),
                'high_importance_count': len([s for s in scores if s > 0.1])
            }
        
        # 3. Analyze closeness for recovery
        self.logger.info("Analyzing recovery paths (closeness)...")
        closeness = self.gds.closeness_centrality(projection_name)
        recovery_findings = self._analyze_recovery_paths(closeness)
        findings.extend(recovery_findings)
        
        if closeness:
            scores = [r.score for r in closeness if r.score > 0]
            if scores:
                metrics['closeness'] = {
                    'avg': sum(scores) / len(scores),
                    'min': min(scores),
                    'isolated_count': len([r for r in closeness if r.score == 0])
                }
        
        # 4. Identify uptime threats
        spofs = self.gds.find_articulation_points()
        uptime_findings, uptime_critical = self._analyze_uptime_threats(spofs, pr_results)
        findings.extend(uptime_findings)
        critical_components.extend(uptime_critical)
        
        # Calculate availability score
        score = self._calculate_score(findings, metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(findings, metrics)
        
        # Sort by severity
        findings.sort(key=lambda f: list(Severity).index(f.severity))
        critical_components.sort(key=lambda c: -c.criticality_score)
        
        self.logger.info(f"Availability analysis complete. Score: {score:.1f}/100")
        
        return AnalysisResult(
            quality_attribute=QualityAttribute.AVAILABILITY,
            score=score,
            findings=findings,
            critical_components=critical_components[:20],
            metrics=metrics,
            recommendations=recommendations
        )
    
    def _analyze_connectivity(self, stats: Dict[str, Any]) -> List[Finding]:
        """Analyze graph connectivity"""
        findings = []
        
        if not stats['is_connected']:
            findings.append(Finding(
                severity=Severity.CRITICAL,
                category='disconnected_system',
                component_id='system',
                component_type='System',
                description=f"System has {stats['component_count']} disconnected components",
                impact="Some parts cannot communicate - partial system failure",
                recommendation="Add connections between disconnected components",
                metrics=stats
            ))
        
        # Check for small isolated components
        sizes = list(stats.get('component_sizes', {}).values())
        if sizes:
            small_components = [s for s in sizes if s < 3 and s > 0]
            if small_components and len(small_components) > len(sizes) / 2:
                findings.append(Finding(
                    severity=Severity.MEDIUM,
                    category='fragmented_system',
                    component_id='system',
                    component_type='System',
                    description=f"{len(small_components)} small isolated component groups",
                    impact="Many small groups reduce overall system cohesion",
                    recommendation="Consolidate or connect small component groups",
                    metrics={'small_component_count': len(small_components)}
                ))
        
        return findings
    
    def _analyze_importance(self, results: List[CentralityResult]) -> Tuple[List[Finding], List[CriticalComponent]]:
        """Analyze component importance via PageRank"""
        findings = []
        critical = []
        
        if not results:
            return findings, critical
        
        # Find high-importance nodes
        scores = sorted([r.score for r in results], reverse=True)
        threshold_idx = max(0, int(len(scores) * 0.1))
        threshold = scores[threshold_idx] if threshold_idx < len(scores) else 0
        
        for result in results:
            if result.score > threshold and result.score > 0.05:
                severity = self._severity_from_score(
                    result.score,
                    self.config['severity_thresholds']
                )
                
                findings.append(Finding(
                    severity=severity,
                    category='high_importance_component',
                    component_id=result.node_id,
                    component_type=result.node_type,
                    description=f"High importance (PageRank: {result.score:.4f})",
                    impact="Critical for system operation; downtime has wide impact",
                    recommendation="Ensure high availability with redundancy/failover",
                    metrics={'pagerank': result.score, 'rank': result.rank}
                ))
                
                critical.append(CriticalComponent(
                    component_id=result.node_id,
                    component_type=result.node_type,
                    criticality_score=result.score,
                    quality_attribute=QualityAttribute.AVAILABILITY,
                    reasons=['high_pagerank', 'critical_for_operation'],
                    metrics={'pagerank': result.score}
                ))
        
        return findings, critical
    
    def _analyze_recovery_paths(self, results: List[CentralityResult]) -> List[Finding]:
        """Analyze recovery path availability via closeness"""
        findings = []
        
        if not results:
            return findings
        
        # Find nodes with low closeness (poor recovery paths)
        scores = [r.score for r in results if r.score > 0]
        if not scores:
            return findings
        
        avg_closeness = sum(scores) / len(scores)
        threshold = avg_closeness * 0.5
        
        poor_recovery = [r for r in results if 0 < r.score < threshold]
        
        for result in poor_recovery[:5]:  # Top 5
            findings.append(Finding(
                severity=Severity.MEDIUM,
                category='poor_recovery_paths',
                component_id=result.node_id,
                component_type=result.node_type,
                description=f"Low closeness centrality ({result.score:.4f})",
                impact="Far from other components; slow recovery if needed",
                recommendation="Add shorter paths to critical components",
                metrics={'closeness': result.score, 'avg_closeness': avg_closeness}
            ))
        
        return findings
    
    def _analyze_uptime_threats(self, spofs: List[Dict[str, Any]], 
                                  pagerank: List[CentralityResult]) -> Tuple[List[Finding], List[CriticalComponent]]:
        """Identify components that threaten system uptime"""
        findings = []
        critical = []
        
        # Create PageRank lookup
        pr_lookup = {r.node_id: r.score for r in pagerank}
        
        for spof in spofs:
            node_id = spof['node_id']
            importance = pr_lookup.get(node_id, 0)
            
            # High importance + SPOF = critical uptime threat
            if importance > 0.05:
                combined_score = (spof['criticality_score'] / 100) * 0.5 + importance * 0.5
                
                findings.append(Finding(
                    severity=Severity.CRITICAL,
                    category='uptime_threat',
                    component_id=node_id,
                    component_type=spof['node_type'],
                    description=f"Critical uptime threat: SPOF with high importance",
                    impact="Failure causes major system outage",
                    recommendation="Implement redundancy and automatic failover",
                    metrics={
                        'spof_score': spof['criticality_score'],
                        'pagerank': importance,
                        'combined': combined_score
                    }
                ))
                
                critical.append(CriticalComponent(
                    component_id=node_id,
                    component_type=spof['node_type'],
                    criticality_score=combined_score,
                    quality_attribute=QualityAttribute.AVAILABILITY,
                    reasons=['uptime_threat', 'spof_high_importance'],
                    metrics={'combined_score': combined_score}
                ))
        
        return findings, critical
    
    def _calculate_score(self, findings: List[Finding], metrics: Dict[str, Any]) -> float:
        """Calculate overall availability score"""
        score = 100.0
        
        # Major deduction for disconnected system
        if not metrics.get('is_connected', True):
            score -= 30
        
        # Deduct for findings
        critical_count = len([f for f in findings if f.severity == Severity.CRITICAL])
        high_count = len([f for f in findings if f.severity == Severity.HIGH])
        score -= critical_count * 12
        score -= high_count * 6
        
        # Bonus for good connectivity
        if metrics.get('is_connected', False) and metrics.get('component_count', 0) == 1:
            score = min(100, score + 5)
        
        return max(0, min(100, score))
    
    def _generate_recommendations(self, findings: List[Finding], metrics: Dict[str, Any]) -> List[str]:
        """Generate prioritized recommendations"""
        recommendations = []
        
        if not metrics.get('is_connected', True):
            recommendations.append("🔴 URGENT: Connect disconnected components")
        
        uptime_threats = len([f for f in findings if f.category == 'uptime_threat'])
        if uptime_threats > 0:
            recommendations.append(f"Implement failover for {uptime_threats} critical uptime threats")
        
        high_importance = metrics.get('pagerank', {}).get('high_importance_count', 0)
        if high_importance > 3:
            recommendations.append(f"Add redundancy for {high_importance} high-importance components")
        
        if not recommendations:
            recommendations.append("✅ Good availability posture")
        
        return recommendations
