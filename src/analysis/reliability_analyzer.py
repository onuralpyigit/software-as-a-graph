#!/usr/bin/env python3
"""
Reliability Analyzer
====================

Analyzes system reliability by detecting:
- Single Points of Failure (SPOFs)
- Cascade failure risks
- Missing redundancy
- Failure propagation paths
- Reliability bottlenecks

Graph Algorithms Used:
- Articulation point detection (SPOFs)
- Bridge edge detection (critical connections)
- Betweenness centrality (bottlenecks)
- Reachability analysis (impact assessment)
- k-connectivity (redundancy assessment)

Author: Software-as-a-Graph Research Project
"""

import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX is required: pip install networkx")

from .quality_attribute_analyzer import (
    BaseQualityAnalyzer,
    QualityAttribute,
    QualityAttributeResult,
    QualityIssue,
    CriticalComponent,
    CriticalEdge,
    Severity,
    IssueCategory,
    ComponentType,
    GraphAnalysisUtils
)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_RELIABILITY_CONFIG = {
    # Thresholds for SPOF detection
    'spof_component_threshold': 2,  # Min components disconnected to be SPOF
    
    # Betweenness threshold percentile for bottlenecks
    'bottleneck_percentile': 90,
    
    # Cascade risk thresholds
    'cascade_depth_threshold': 3,  # Dependency chain depth
    'cascade_breadth_threshold': 5,  # Number of dependent components
    
    # Redundancy requirements by component type
    'redundancy_requirements': {
        'broker': 2,  # Brokers should have at least 2 paths
        'application': 1,
        'topic': 1,
        'node': 2
    },
    
    # Impact score weights
    'weights': {
        'reachability_loss': 0.4,
        'dependent_count': 0.3,
        'betweenness': 0.3
    },
    
    # Severity thresholds for reliability score
    'severity_thresholds': {
        'critical': 0.8,  # Impact >= 80% is critical
        'high': 0.5,
        'medium': 0.2
    }
}


# ============================================================================
# Reliability Analyzer
# ============================================================================

class ReliabilityAnalyzer(BaseQualityAnalyzer):
    """
    Analyzes system reliability using graph-based methods.
    
    Reliability is assessed through:
    1. SPOF Detection: Articulation points that disconnect the system
    2. Cascade Risk: Components whose failure propagates widely
    3. Redundancy Analysis: Lack of backup paths/components
    4. Bottleneck Detection: High-traffic components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.config = {**DEFAULT_RELIABILITY_CONFIG, **(config or {})}
        self.logger = logging.getLogger('ReliabilityAnalyzer')
    
    @property
    def quality_attribute(self) -> QualityAttribute:
        return QualityAttribute.RELIABILITY
    
    def analyze(self, graph: nx.DiGraph) -> QualityAttributeResult:
        """
        Perform comprehensive reliability analysis.
        
        Args:
            graph: NetworkX DiGraph with DEPENDS_ON relationships
            
        Returns:
            QualityAttributeResult with reliability findings
        """
        self.logger.info("Starting reliability analysis...")
        
        issues = []
        critical_components = []
        critical_edges = []
        metrics = {}
        
        # Pre-compute graph metrics
        self.logger.info("Computing graph metrics...")
        centrality = GraphAnalysisUtils.compute_centrality_metrics(graph)
        articulation_points = GraphAnalysisUtils.find_articulation_points(graph)
        bridges = GraphAnalysisUtils.find_bridges(graph)
        k_conn = GraphAnalysisUtils.compute_k_connectivity(graph)
        
        # Store metrics
        metrics['articulation_point_count'] = len(articulation_points)
        metrics['bridge_count'] = len(bridges)
        metrics['k_connectivity'] = k_conn
        metrics['node_count'] = graph.number_of_nodes()
        metrics['edge_count'] = graph.number_of_edges()
        
        # 1. Detect SPOFs (Articulation Points)
        self.logger.info("Detecting SPOFs...")
        spof_issues, spof_components = self._detect_spofs(
            graph, articulation_points, centrality
        )
        issues.extend(spof_issues)
        critical_components.extend(spof_components)
        
        # 2. Detect Critical Bridges
        self.logger.info("Analyzing bridge edges...")
        bridge_issues, bridge_edges = self._detect_critical_bridges(
            graph, bridges, centrality
        )
        issues.extend(bridge_issues)
        critical_edges.extend(bridge_edges)
        
        # 3. Detect Cascade Failure Risks
        self.logger.info("Analyzing cascade failure risks...")
        cascade_issues, cascade_components = self._detect_cascade_risks(
            graph, centrality
        )
        issues.extend(cascade_issues)
        critical_components.extend(cascade_components)
        
        # 4. Detect Missing Redundancy
        self.logger.info("Analyzing redundancy gaps...")
        redundancy_issues = self._detect_missing_redundancy(graph, k_conn)
        issues.extend(redundancy_issues)
        
        # 5. Detect Reliability Bottlenecks
        self.logger.info("Detecting reliability bottlenecks...")
        bottleneck_issues, bottleneck_components = self._detect_bottlenecks(
            graph, centrality
        )
        issues.extend(bottleneck_issues)
        critical_components.extend(bottleneck_components)
        
        # 6. Analyze Failure Propagation Paths
        self.logger.info("Analyzing failure propagation paths...")
        propagation_issues = self._analyze_failure_propagation(graph)
        issues.extend(propagation_issues)
        
        # Calculate overall reliability score
        reliability_score = self._calculate_reliability_score(
            graph, issues, metrics
        )
        metrics['reliability_score'] = reliability_score
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, metrics)
        
        # Sort by criticality
        issues.sort(key=lambda x: x.severity)
        critical_components.sort(key=lambda x: -x.criticality_score)
        critical_edges.sort(key=lambda x: -x.criticality_score)
        
        self.logger.info(f"Reliability analysis complete. Score: {reliability_score:.1f}/100")
        
        return QualityAttributeResult(
            quality_attribute=QualityAttribute.RELIABILITY,
            score=reliability_score,
            issues=issues,
            critical_components=critical_components,
            critical_edges=critical_edges,
            metrics=metrics,
            recommendations=recommendations
        )
    
    def _detect_spofs(self, graph: nx.DiGraph, 
                      articulation_points: Set[str],
                      centrality: Dict[str, Dict[str, float]]
                      ) -> Tuple[List[QualityIssue], List[CriticalComponent]]:
        """Detect Single Points of Failure"""
        issues = []
        critical = []
        
        for ap in articulation_points:
            # Calculate impact of removing this node
            impact = self._calculate_removal_impact(graph, ap)
            
            # Determine severity based on impact
            severity = self._impact_to_severity(impact['reachability_loss'])
            
            # Get component type
            comp_type = self._get_component_type(graph, ap)
            
            # Create issue
            issue = QualityIssue(
                issue_id=self._generate_issue_id(),
                quality_attribute=QualityAttribute.RELIABILITY,
                category=IssueCategory.SINGLE_POINT_OF_FAILURE,
                severity=severity,
                affected_components=[ap],
                description=f"Component '{ap}' is a single point of failure. "
                           f"Its removal creates {impact['components_after']} disconnected components.",
                impact=f"System fragmentation: {impact['reachability_loss']*100:.1f}% reachability loss. "
                       f"{impact['affected_nodes']} components become unreachable.",
                recommendation=f"Add redundancy for '{ap}'. Consider active-passive failover "
                              f"or load balancing across multiple instances.",
                metrics={
                    'reachability_loss': impact['reachability_loss'],
                    'components_after_removal': impact['components_after'],
                    'affected_nodes': impact['affected_nodes'],
                    'betweenness': centrality.get('betweenness', {}).get(ap, 0)
                }
            )
            issues.append(issue)
            
            # Create critical component entry
            comp = CriticalComponent(
                component_id=ap,
                component_type=comp_type,
                quality_attribute=QualityAttribute.RELIABILITY,
                criticality_score=impact['reachability_loss'],
                reasons=['articulation_point', 'single_point_of_failure'],
                metrics={
                    'reachability_loss': impact['reachability_loss'],
                    'betweenness': centrality.get('betweenness', {}).get(ap, 0),
                    'in_degree': centrality.get('in_degree', {}).get(ap, 0),
                    'out_degree': centrality.get('out_degree', {}).get(ap, 0)
                },
                recommendations=[
                    f"Add redundant {comp_type.value.lower()} instance",
                    "Implement failover mechanism",
                    "Consider active-active deployment"
                ]
            )
            critical.append(comp)
        
        return issues, critical
    
    def _detect_critical_bridges(self, graph: nx.DiGraph,
                                 bridges: List[Tuple[str, str]],
                                 centrality: Dict[str, Dict[str, float]]
                                 ) -> Tuple[List[QualityIssue], List[CriticalEdge]]:
        """Detect critical bridge edges"""
        issues = []
        critical = []
        
        # Compute edge betweenness
        try:
            edge_betweenness = nx.edge_betweenness_centrality(graph)
        except:
            edge_betweenness = {}
        
        for source, target in bridges:
            # Check both directions in case of undirected analysis
            eb = edge_betweenness.get((source, target), 
                  edge_betweenness.get((target, source), 0.0))
            
            # Calculate impact - ensure minimum criticality for bridges
            impact = self._calculate_edge_removal_impact(graph, source, target)
            
            # Bridges are inherently critical - boost score if impact is low
            base_score = impact['reachability_loss']
            if base_score < 0.1:
                # All bridges are at least somewhat critical
                base_score = max(0.1, eb * 5) if eb > 0 else 0.15
            
            severity = self._impact_to_severity(base_score)
            
            # Get edge type
            edge_data = graph.get_edge_data(source, target, {})
            edge_type = edge_data.get('dependency_type', 
                        edge_data.get('type', 'unknown'))
            
            issue = QualityIssue(
                issue_id=self._generate_issue_id(),
                quality_attribute=QualityAttribute.RELIABILITY,
                category=IssueCategory.MISSING_REDUNDANCY,
                severity=severity,
                affected_components=[source, target],
                affected_edges=[(source, target)],
                description=f"Edge '{source}' → '{target}' is a bridge. "
                           f"No alternative path exists between connected components.",
                impact=f"Removing this connection causes {impact['reachability_loss']*100:.1f}% "
                       f"reachability loss.",
                recommendation="Add redundant connection path. Consider multiple routes "
                              "between these components.",
                metrics={
                    'edge_betweenness': eb,
                    'reachability_loss': impact['reachability_loss']
                }
            )
            issues.append(issue)
            
            edge = CriticalEdge(
                source=source,
                target=target,
                edge_type=edge_type,
                quality_attribute=QualityAttribute.RELIABILITY,
                criticality_score=base_score,  # Use calculated base score
                reasons=['bridge_edge', 'no_alternative_path'],
                metrics={
                    'edge_betweenness': eb,
                    'reachability_loss': impact['reachability_loss'],
                    'computed_criticality': base_score
                },
                recommendations=[
                    "Add parallel connection",
                    "Implement connection redundancy",
                    "Consider message queue for decoupling"
                ]
            )
            critical.append(edge)
        
        return issues, critical
    
    def _detect_cascade_risks(self, graph: nx.DiGraph,
                              centrality: Dict[str, Dict[str, float]]
                              ) -> Tuple[List[QualityIssue], List[CriticalComponent]]:
        """Detect components with high cascade failure risk"""
        issues = []
        critical = []
        
        depth_threshold = self.config['cascade_depth_threshold']
        breadth_threshold = self.config['cascade_breadth_threshold']
        
        for node in graph.nodes():
            # Calculate cascade metrics
            descendants = nx.descendants(graph, node)
            
            if not descendants:
                continue
            
            # Calculate max depth of influence
            max_depth = 0
            for desc in descendants:
                try:
                    path_len = nx.shortest_path_length(graph, node, desc)
                    max_depth = max(max_depth, path_len)
                except:
                    pass
            
            # Check if this is a cascade risk
            if len(descendants) >= breadth_threshold or max_depth >= depth_threshold:
                impact_score = len(descendants) / graph.number_of_nodes()
                severity = self._impact_to_severity(impact_score)
                
                # Only report significant cascade risks
                if severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM]:
                    comp_type = self._get_component_type(graph, node)
                    
                    issue = QualityIssue(
                        issue_id=self._generate_issue_id(),
                        quality_attribute=QualityAttribute.RELIABILITY,
                        category=IssueCategory.CASCADE_FAILURE_RISK,
                        severity=severity,
                        affected_components=[node] + list(descendants)[:10],
                        description=f"Component '{node}' has high cascade failure risk. "
                                   f"Failure affects {len(descendants)} downstream components "
                                   f"across {max_depth} dependency layers.",
                        impact=f"Cascade propagation: {impact_score*100:.1f}% of system affected. "
                               f"Dependency chain depth: {max_depth}",
                        recommendation="Implement circuit breakers and bulkheads. "
                                      "Add timeout mechanisms. Consider async communication.",
                        metrics={
                            'cascade_breadth': len(descendants),
                            'cascade_depth': max_depth,
                            'impact_score': impact_score
                        }
                    )
                    issues.append(issue)
                    
                    comp = CriticalComponent(
                        component_id=node,
                        component_type=comp_type,
                        quality_attribute=QualityAttribute.RELIABILITY,
                        criticality_score=impact_score,
                        reasons=['cascade_failure_risk', 'high_downstream_impact'],
                        metrics={
                            'cascade_breadth': len(descendants),
                            'cascade_depth': max_depth,
                            'betweenness': centrality.get('betweenness', {}).get(node, 0),
                            'pagerank': centrality.get('pagerank', {}).get(node, 0)
                        },
                        recommendations=[
                            "Implement circuit breaker pattern",
                            "Add health checks and monitoring",
                            "Consider bulkhead isolation"
                        ]
                    )
                    critical.append(comp)
        
        return issues, critical
    
    def _detect_missing_redundancy(self, graph: nx.DiGraph,
                                    k_connectivity: int) -> List[QualityIssue]:
        """Detect components and paths lacking redundancy"""
        issues = []
        requirements = self.config['redundancy_requirements']
        
        # System-wide redundancy check
        if k_connectivity < 2:
            issues.append(QualityIssue(
                issue_id=self._generate_issue_id(),
                quality_attribute=QualityAttribute.RELIABILITY,
                category=IssueCategory.MISSING_REDUNDANCY,
                severity=Severity.HIGH,
                affected_components=list(graph.nodes())[:10],
                description=f"System has k-connectivity of {k_connectivity}. "
                           f"Single node/edge failures can partition the system.",
                impact="Low fault tolerance. System vulnerable to single failures.",
                recommendation="Increase redundancy to achieve at least 2-connectivity. "
                              "Add alternative paths between critical components.",
                metrics={'k_connectivity': k_connectivity}
            ))
        
        # Check per-component type redundancy
        layers = GraphAnalysisUtils.get_component_layers(graph)
        
        for layer, nodes in layers.items():
            required = requirements.get(layer, 1)
            
            # Check connectivity for critical layers
            if layer in ['broker', 'node'] and len(nodes) < required:
                issues.append(QualityIssue(
                    issue_id=self._generate_issue_id(),
                    quality_attribute=QualityAttribute.RELIABILITY,
                    category=IssueCategory.MISSING_REDUNDANCY,
                    severity=Severity.HIGH if layer == 'broker' else Severity.MEDIUM,
                    affected_components=list(nodes),
                    description=f"Layer '{layer}' has {len(nodes)} components. "
                               f"Recommended minimum: {required}",
                    impact=f"Insufficient redundancy in {layer} layer.",
                    recommendation=f"Add at least {required - len(nodes)} more {layer}(s) "
                                  f"for redundancy.",
                    metrics={
                        'current_count': len(nodes),
                        'required_count': required
                    }
                ))
        
        return issues
    
    def _detect_bottlenecks(self, graph: nx.DiGraph,
                            centrality: Dict[str, Dict[str, float]]
                            ) -> Tuple[List[QualityIssue], List[CriticalComponent]]:
        """Detect reliability bottlenecks using centrality"""
        issues = []
        critical = []
        
        betweenness = centrality.get('betweenness', {})
        if not betweenness:
            return issues, critical
        
        # Calculate threshold
        threshold = self._calculate_percentile_threshold(
            list(betweenness.values()),
            self.config['bottleneck_percentile']
        )
        
        for node, bc in betweenness.items():
            if bc >= threshold and bc > 0:
                comp_type = self._get_component_type(graph, node)
                in_deg = graph.in_degree(node)
                out_deg = graph.out_degree(node)
                
                severity = Severity.HIGH if bc > threshold * 1.5 else Severity.MEDIUM
                
                issue = QualityIssue(
                    issue_id=self._generate_issue_id(),
                    quality_attribute=QualityAttribute.RELIABILITY,
                    category=IssueCategory.RELIABILITY_BOTTLENECK,
                    severity=severity,
                    affected_components=[node],
                    description=f"Component '{node}' is a reliability bottleneck. "
                               f"Betweenness centrality: {bc:.4f} (threshold: {threshold:.4f})",
                    impact=f"High traffic concentration. {in_deg} incoming, "
                           f"{out_deg} outgoing dependencies.",
                    recommendation="Distribute load across multiple instances. "
                                  "Consider caching or load balancing.",
                    metrics={
                        'betweenness': bc,
                        'threshold': threshold,
                        'in_degree': in_deg,
                        'out_degree': out_deg
                    }
                )
                issues.append(issue)
                
                comp = CriticalComponent(
                    component_id=node,
                    component_type=comp_type,
                    quality_attribute=QualityAttribute.RELIABILITY,
                    criticality_score=bc,
                    reasons=['high_betweenness', 'traffic_bottleneck'],
                    metrics={
                        'betweenness': bc,
                        'in_degree': in_deg,
                        'out_degree': out_deg,
                        'pagerank': centrality.get('pagerank', {}).get(node, 0)
                    },
                    recommendations=[
                        "Implement load balancing",
                        "Add caching layer",
                        "Consider horizontal scaling"
                    ]
                )
                critical.append(comp)
        
        return issues, critical
    
    def _analyze_failure_propagation(self, graph: nx.DiGraph) -> List[QualityIssue]:
        """Analyze failure propagation paths"""
        issues = []
        
        # Find SCCs (strongly connected components) which indicate cyclic dependencies
        sccs = GraphAnalysisUtils.find_strongly_connected_components(graph)
        
        for scc in sccs:
            if len(scc) > 2:  # Non-trivial cycles
                severity = Severity.HIGH if len(scc) > 4 else Severity.MEDIUM
                
                issue = QualityIssue(
                    issue_id=self._generate_issue_id(),
                    quality_attribute=QualityAttribute.RELIABILITY,
                    category=IssueCategory.FAILURE_PROPAGATION,
                    severity=severity,
                    affected_components=list(scc),
                    description=f"Circular dependency detected among {len(scc)} components. "
                               f"Failure can propagate in loops.",
                    impact="Failure in any component can cascade back, creating "
                           "positive feedback loops. Hard to isolate failures.",
                    recommendation="Break circular dependencies. Introduce async communication "
                                  "or event sourcing to decouple components.",
                    metrics={
                        'cycle_size': len(scc),
                        'components': list(scc)[:5]
                    }
                )
                issues.append(issue)
        
        return issues
    
    def _calculate_removal_impact(self, graph: nx.DiGraph, 
                                   node: str) -> Dict[str, Any]:
        """Calculate impact of removing a node"""
        original_components = nx.number_weakly_connected_components(graph)
        
        # Create graph without node
        test_graph = graph.copy()
        test_graph.remove_node(node)
        
        new_components = nx.number_weakly_connected_components(test_graph)
        
        # Calculate reachability loss
        original_reach = sum(len(nx.descendants(graph, n)) for n in graph.nodes())
        new_reach = sum(len(nx.descendants(test_graph, n)) for n in test_graph.nodes())
        
        reachability_loss = (original_reach - new_reach) / max(original_reach, 1)
        
        return {
            'components_after': new_components,
            'components_increase': new_components - original_components,
            'reachability_loss': reachability_loss,
            'affected_nodes': graph.number_of_nodes() - test_graph.number_of_nodes() - 1
        }
    
    def _calculate_edge_removal_impact(self, graph: nx.DiGraph,
                                        source: str, target: str) -> Dict[str, Any]:
        """Calculate impact of removing an edge"""
        test_graph = graph.copy()
        if test_graph.has_edge(source, target):
            test_graph.remove_edge(source, target)
        
        original_reach = sum(len(nx.descendants(graph, n)) for n in graph.nodes())
        new_reach = sum(len(nx.descendants(test_graph, n)) for n in test_graph.nodes())
        
        reachability_loss = (original_reach - new_reach) / max(original_reach, 1)
        
        return {
            'reachability_loss': reachability_loss
        }
    
    def _impact_to_severity(self, impact: float) -> Severity:
        """Convert impact score to severity level"""
        thresholds = self.config['severity_thresholds']
        
        if impact >= thresholds['critical']:
            return Severity.CRITICAL
        elif impact >= thresholds['high']:
            return Severity.HIGH
        elif impact >= thresholds['medium']:
            return Severity.MEDIUM
        else:
            return Severity.LOW
    
    def _calculate_reliability_score(self, graph: nx.DiGraph,
                                     issues: List[QualityIssue],
                                     metrics: Dict[str, Any]) -> float:
        """Calculate overall reliability score (0-100)"""
        # Start with perfect score
        score = 100.0
        
        # Deduct for issues
        severity_penalties = {
            Severity.CRITICAL: 15,
            Severity.HIGH: 8,
            Severity.MEDIUM: 3,
            Severity.LOW: 1
        }
        
        for issue in issues:
            score -= severity_penalties.get(issue.severity, 0)
        
        # Bonus for good k-connectivity
        k_conn = metrics.get('k_connectivity', 0)
        if k_conn >= 3:
            score += 10
        elif k_conn >= 2:
            score += 5
        
        # Penalty for high SPOF ratio
        spof_ratio = metrics.get('articulation_point_count', 0) / max(graph.number_of_nodes(), 1)
        if spof_ratio > 0.2:
            score -= 15
        elif spof_ratio > 0.1:
            score -= 8
        
        return max(0, min(100, score))
    
    def _generate_recommendations(self, issues: List[QualityIssue],
                                  metrics: Dict[str, Any]) -> List[str]:
        """Generate prioritized recommendations"""
        recommendations = []
        
        # Critical issue recommendations
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        if critical_issues:
            recommendations.append(
                f"🔴 URGENT: Address {len(critical_issues)} critical reliability issues immediately"
            )
        
        # SPOF recommendations
        spof_count = metrics.get('articulation_point_count', 0)
        if spof_count > 0:
            recommendations.append(
                f"Add redundancy for {spof_count} single points of failure"
            )
        
        # Connectivity recommendations
        k_conn = metrics.get('k_connectivity', 0)
        if k_conn < 2:
            recommendations.append(
                "Improve system connectivity to achieve at least 2-connectivity"
            )
        
        # Bridge recommendations
        bridge_count = metrics.get('bridge_count', 0)
        if bridge_count > 0:
            recommendations.append(
                f"Add redundant paths for {bridge_count} bridge connections"
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append(
                "✅ No critical reliability issues detected. Continue monitoring."
            )
        
        return recommendations