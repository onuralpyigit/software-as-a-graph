#!/usr/bin/env python3
"""
Availability Analyzer
=====================

Analyzes system availability by detecting:
- Low connectivity (k-connectivity analysis)
- Fault tolerance gaps
- Recovery path issues
- Uptime threats
- Partition risks

Graph Algorithms Used:
- k-connectivity analysis (fault tolerance)
- Min-cut computation (partition risk)
- Path analysis (recovery paths)
- Redundancy analysis (backup components)
- Component isolation detection

Author: Software-as-a-Graph Research Project
"""

import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import itertools

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

DEFAULT_AVAILABILITY_CONFIG = {
    # Connectivity requirements
    'minimum_k_connectivity': 2,        # Minimum vertex connectivity
    'minimum_edge_connectivity': 2,     # Minimum edge connectivity
    
    # Redundancy requirements by component type
    'redundancy_requirements': {
        'broker': {'minimum': 2, 'recommended': 3},
        'node': {'minimum': 2, 'recommended': 3},
        'application': {'minimum': 1, 'recommended': 2},
        'topic': {'minimum': 1, 'recommended': 1}
    },
    
    # Recovery path requirements
    'minimum_alternative_paths': 2,     # Between critical pairs
    'max_recovery_path_length': 5,      # Maximum acceptable path length
    
    # Partition analysis
    'partition_check_sample_size': 20,  # Sample size for partition analysis
    
    # Uptime factors
    'single_failure_tolerance': True,   # System should survive single failure
    'critical_component_types': ['broker', 'node'],
    
    # Scoring
    'severity_thresholds': {
        'connectivity': {'critical': 0, 'high': 1, 'medium': 2},
        'redundancy_gap': {'critical': 0, 'high': 1, 'medium': 2},
        'recovery_paths': {'critical': 0, 'high': 1, 'medium': 2}
    }
}


# ============================================================================
# Availability Metrics
# ============================================================================

@dataclass
class AvailabilityMetrics:
    """Availability metrics for the system"""
    node_connectivity: int          # k (vertex connectivity)
    edge_connectivity: int          # λ (edge connectivity)
    average_path_redundancy: float  # Average alternative paths
    component_redundancy: Dict[str, int]  # Redundancy by type
    recovery_coverage: float        # % of pairs with recovery paths
    partition_resistance: float     # Resistance to partitioning
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_connectivity': self.node_connectivity,
            'edge_connectivity': self.edge_connectivity,
            'average_path_redundancy': round(self.average_path_redundancy, 4),
            'component_redundancy': self.component_redundancy,
            'recovery_coverage': round(self.recovery_coverage, 4),
            'partition_resistance': round(self.partition_resistance, 4)
        }


# ============================================================================
# Availability Analyzer
# ============================================================================

class AvailabilityAnalyzer(BaseQualityAnalyzer):
    """
    Analyzes system availability using graph-based methods.
    
    Availability is assessed through:
    1. Connectivity Analysis: k-connectivity for fault tolerance
    2. Redundancy Analysis: Backup components and paths
    3. Recovery Paths: Alternative routes when components fail
    4. Partition Risk: Vulnerability to network splits
    5. Uptime Threats: Components that threaten system uptime
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.config = {**DEFAULT_AVAILABILITY_CONFIG, **(config or {})}
        self.logger = logging.getLogger('AvailabilityAnalyzer')
    
    @property
    def quality_attribute(self) -> QualityAttribute:
        return QualityAttribute.AVAILABILITY
    
    def analyze(self, graph: nx.DiGraph) -> QualityAttributeResult:
        """
        Perform comprehensive availability analysis.
        
        Args:
            graph: NetworkX DiGraph with DEPENDS_ON relationships
            
        Returns:
            QualityAttributeResult with availability findings
        """
        self.logger.info("Starting availability analysis...")
        
        issues = []
        critical_components = []
        critical_edges = []
        metrics_dict = {}
        
        # Compute availability metrics
        self.logger.info("Computing availability metrics...")
        avail_metrics = self._compute_availability_metrics(graph)
        metrics_dict.update(avail_metrics.to_dict())
        
        # 1. Analyze Connectivity
        self.logger.info("Analyzing connectivity...")
        conn_issues, conn_components = self._analyze_connectivity(
            graph, avail_metrics
        )
        issues.extend(conn_issues)
        critical_components.extend(conn_components)
        
        # 2. Analyze Component Redundancy
        self.logger.info("Analyzing redundancy...")
        redundancy_issues = self._analyze_redundancy(graph)
        issues.extend(redundancy_issues)
        
        # 3. Analyze Recovery Paths
        self.logger.info("Analyzing recovery paths...")
        recovery_issues, recovery_edges = self._analyze_recovery_paths(graph)
        issues.extend(recovery_issues)
        critical_edges.extend(recovery_edges)
        
        # 4. Analyze Partition Risk
        self.logger.info("Analyzing partition risk...")
        partition_issues = self._analyze_partition_risk(graph)
        issues.extend(partition_issues)
        
        # 5. Identify Uptime Threats
        self.logger.info("Identifying uptime threats...")
        uptime_issues, uptime_components = self._identify_uptime_threats(graph)
        issues.extend(uptime_issues)
        critical_components.extend(uptime_components)
        
        # 6. Analyze Fault Tolerance
        self.logger.info("Analyzing fault tolerance...")
        ft_issues = self._analyze_fault_tolerance(graph, avail_metrics)
        issues.extend(ft_issues)
        
        # Calculate overall availability score
        availability_score = self._calculate_availability_score(
            graph, issues, avail_metrics
        )
        metrics_dict['availability_score'] = availability_score
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, avail_metrics)
        
        # Sort results
        issues.sort(key=lambda x: x.severity)
        critical_components.sort(key=lambda x: -x.criticality_score)
        critical_edges.sort(key=lambda x: -x.criticality_score)
        
        self.logger.info(f"Availability analysis complete. Score: {availability_score:.1f}/100")
        
        return QualityAttributeResult(
            quality_attribute=QualityAttribute.AVAILABILITY,
            score=availability_score,
            issues=issues,
            critical_components=critical_components,
            critical_edges=critical_edges,
            metrics=metrics_dict,
            recommendations=recommendations
        )
    
    def _compute_availability_metrics(self, graph: nx.DiGraph) -> AvailabilityMetrics:
        """Compute availability-related metrics"""
        
        # Node connectivity (k)
        try:
            undirected = graph.to_undirected()
            node_conn = nx.node_connectivity(undirected) if undirected.number_of_nodes() > 1 else 0
        except:
            node_conn = 0
        
        # Edge connectivity (λ)
        try:
            edge_conn = nx.edge_connectivity(undirected) if undirected.number_of_nodes() > 1 else 0
        except:
            edge_conn = 0
        
        # Component redundancy
        layers = GraphAnalysisUtils.get_component_layers(graph)
        component_redundancy = {layer: len(nodes) for layer, nodes in layers.items()}
        
        # Path redundancy (sample-based)
        avg_path_redundancy = self._compute_average_path_redundancy(graph)
        
        # Recovery coverage
        recovery_coverage = self._compute_recovery_coverage(graph)
        
        # Partition resistance (based on min-cut)
        partition_resistance = self._compute_partition_resistance(graph)
        
        return AvailabilityMetrics(
            node_connectivity=node_conn,
            edge_connectivity=edge_conn,
            average_path_redundancy=avg_path_redundancy,
            component_redundancy=component_redundancy,
            recovery_coverage=recovery_coverage,
            partition_resistance=partition_resistance
        )
    
    def _compute_average_path_redundancy(self, graph: nx.DiGraph) -> float:
        """Compute average number of alternative paths between node pairs"""
        if graph.number_of_nodes() < 2:
            return 0.0
        
        # Sample node pairs
        nodes = list(graph.nodes())
        sample_size = min(self.config['partition_check_sample_size'], len(nodes))
        
        total_paths = 0
        pair_count = 0
        
        for i, source in enumerate(nodes[:sample_size]):
            for target in nodes[i+1:sample_size]:
                try:
                    # Count node-disjoint paths
                    paths = list(nx.node_disjoint_paths(graph.to_undirected(), source, target))
                    total_paths += len(paths)
                    pair_count += 1
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    pair_count += 1  # Count but with 0 paths
        
        return total_paths / max(pair_count, 1)
    
    def _compute_recovery_coverage(self, graph: nx.DiGraph) -> float:
        """Compute fraction of critical pairs with recovery paths"""
        if graph.number_of_nodes() < 2:
            return 1.0
        
        layers = GraphAnalysisUtils.get_component_layers(graph)
        critical_types = self.config['critical_component_types']
        
        # Get critical components
        critical_nodes = set()
        for ctype in critical_types:
            critical_nodes.update(layers.get(ctype, set()))
        
        if len(critical_nodes) < 2:
            return 1.0
        
        # Check pairs
        pairs_with_recovery = 0
        total_pairs = 0
        
        for source, target in itertools.combinations(critical_nodes, 2):
            total_pairs += 1
            try:
                paths = list(nx.node_disjoint_paths(graph.to_undirected(), source, target))
                if len(paths) >= 2:  # Has alternative path
                    pairs_with_recovery += 1
            except:
                pass
        
        return pairs_with_recovery / max(total_pairs, 1)
    
    def _compute_partition_resistance(self, graph: nx.DiGraph) -> float:
        """Compute resistance to network partitioning"""
        if graph.number_of_nodes() < 2:
            return 1.0
        
        try:
            undirected = graph.to_undirected()
            
            # Use node connectivity as base metric
            k = nx.node_connectivity(undirected)
            
            # Normalize: k=0 -> 0, k>=3 -> 1
            resistance = min(1.0, k / 3)
            
            return resistance
        except:
            return 0.0
    
    def _analyze_connectivity(self, graph: nx.DiGraph,
                              metrics: AvailabilityMetrics
                              ) -> Tuple[List[QualityIssue], List[CriticalComponent]]:
        """Analyze system connectivity for availability"""
        issues = []
        critical = []
        
        min_k = self.config['minimum_k_connectivity']
        
        # Check node connectivity
        if metrics.node_connectivity < min_k:
            severity = self._connectivity_severity(metrics.node_connectivity)
            
            # Find nodes that when removed reduce connectivity
            cut_nodes = self._find_minimum_cut_nodes(graph)
            
            issue = QualityIssue(
                issue_id=self._generate_issue_id(),
                quality_attribute=QualityAttribute.AVAILABILITY,
                category=IssueCategory.LOW_CONNECTIVITY,
                severity=severity,
                affected_components=cut_nodes[:10],
                description=f"System has low node connectivity (k={metrics.node_connectivity}). "
                           f"Minimum recommended: {min_k}.",
                impact=f"System can be disconnected by removing {metrics.node_connectivity} "
                       f"strategically chosen nodes.",
                recommendation=f"Add redundant paths to achieve {min_k}-connectivity. "
                              f"Focus on components: {', '.join(cut_nodes[:3])}",
                metrics={
                    'node_connectivity': metrics.node_connectivity,
                    'minimum_required': min_k,
                    'cut_nodes': cut_nodes[:5]
                }
            )
            issues.append(issue)
            
            # Mark cut nodes as critical
            for node in cut_nodes[:5]:
                comp_type = self._get_component_type(graph, node)
                
                comp = CriticalComponent(
                    component_id=node,
                    component_type=comp_type,
                    quality_attribute=QualityAttribute.AVAILABILITY,
                    criticality_score=1.0 - (metrics.node_connectivity / max(min_k, 1)),
                    reasons=['connectivity_critical', 'minimum_cut_member'],
                    metrics={
                        'node_connectivity': metrics.node_connectivity
                    },
                    recommendations=[
                        "Add redundant path around this component",
                        "Consider load balancing/replication",
                        "Implement failover mechanism"
                    ]
                )
                critical.append(comp)
        
        # Check edge connectivity
        if metrics.edge_connectivity < self.config['minimum_edge_connectivity']:
            severity = self._connectivity_severity(metrics.edge_connectivity)
            
            # Find bridges
            bridges = GraphAnalysisUtils.find_bridges(graph)
            
            issue = QualityIssue(
                issue_id=self._generate_issue_id(),
                quality_attribute=QualityAttribute.AVAILABILITY,
                category=IssueCategory.LOW_CONNECTIVITY,
                severity=severity,
                affected_components=[],
                affected_edges=bridges[:10],
                description=f"System has low edge connectivity (λ={metrics.edge_connectivity}). "
                           f"{len(bridges)} bridge edges exist.",
                impact=f"Removing {metrics.edge_connectivity} edges can partition the system.",
                recommendation="Add parallel connections for bridge edges.",
                metrics={
                    'edge_connectivity': metrics.edge_connectivity,
                    'bridge_count': len(bridges)
                }
            )
            issues.append(issue)
        
        return issues, critical
    
    def _analyze_redundancy(self, graph: nx.DiGraph) -> List[QualityIssue]:
        """Analyze component redundancy"""
        issues = []
        
        layers = GraphAnalysisUtils.get_component_layers(graph)
        requirements = self.config['redundancy_requirements']
        
        for comp_type, req in requirements.items():
            count = len(layers.get(comp_type, set()))
            minimum = req.get('minimum', 1)
            recommended = req.get('recommended', minimum)
            
            if count < minimum:
                severity = Severity.CRITICAL
                issue = QualityIssue(
                    issue_id=self._generate_issue_id(),
                    quality_attribute=QualityAttribute.AVAILABILITY,
                    category=IssueCategory.FAULT_TOLERANCE_GAP,
                    severity=severity,
                    affected_components=list(layers.get(comp_type, [])),
                    description=f"Insufficient {comp_type} redundancy: {count} instances, "
                               f"minimum required: {minimum}",
                    impact=f"Single {comp_type} failure will cause service disruption.",
                    recommendation=f"Deploy at least {minimum - count} additional {comp_type}(s).",
                    metrics={
                        'current_count': count,
                        'minimum_required': minimum,
                        'recommended': recommended
                    }
                )
                issues.append(issue)
            
            elif count < recommended:
                severity = Severity.MEDIUM
                issue = QualityIssue(
                    issue_id=self._generate_issue_id(),
                    quality_attribute=QualityAttribute.AVAILABILITY,
                    category=IssueCategory.FAULT_TOLERANCE_GAP,
                    severity=severity,
                    affected_components=list(layers.get(comp_type, [])),
                    description=f"Below recommended {comp_type} redundancy: {count} instances, "
                               f"recommended: {recommended}",
                    impact=f"Limited fault tolerance for {comp_type} failures.",
                    recommendation=f"Consider adding {recommended - count} more {comp_type}(s).",
                    metrics={
                        'current_count': count,
                        'minimum_required': minimum,
                        'recommended': recommended
                    }
                )
                issues.append(issue)
        
        return issues
    
    def _analyze_recovery_paths(self, graph: nx.DiGraph
                                ) -> Tuple[List[QualityIssue], List[CriticalEdge]]:
        """Analyze availability of recovery paths"""
        issues = []
        critical_edges = []
        
        min_paths = self.config['minimum_alternative_paths']
        max_length = self.config['max_recovery_path_length']
        
        layers = GraphAnalysisUtils.get_component_layers(graph)
        critical_types = self.config['critical_component_types']
        
        # Get critical components
        critical_nodes = set()
        for ctype in critical_types:
            critical_nodes.update(layers.get(ctype, set()))
        
        if len(critical_nodes) < 2:
            return issues, critical_edges
        
        # Check paths between critical pairs
        pairs_without_recovery = []
        
        for source, target in itertools.combinations(list(critical_nodes)[:20], 2):
            try:
                undirected = graph.to_undirected()
                paths = list(nx.node_disjoint_paths(undirected, source, target))
                
                if len(paths) < min_paths:
                    pairs_without_recovery.append((source, target, len(paths)))
                
                # Check path length
                if paths:
                    shortest = min(len(p) for p in paths)
                    if shortest > max_length:
                        pairs_without_recovery.append((source, target, len(paths)))
                        
            except (nx.NetworkXNoPath, nx.NetworkXError):
                pairs_without_recovery.append((source, target, 0))
        
        if pairs_without_recovery:
            severity = Severity.HIGH if any(p[2] == 0 for p in pairs_without_recovery) else Severity.MEDIUM
            
            issue = QualityIssue(
                issue_id=self._generate_issue_id(),
                quality_attribute=QualityAttribute.AVAILABILITY,
                category=IssueCategory.RECOVERY_PATH_MISSING,
                severity=severity,
                affected_components=list(set(p[0] for p in pairs_without_recovery) |
                                        set(p[1] for p in pairs_without_recovery))[:10],
                description=f"{len(pairs_without_recovery)} critical component pairs lack "
                           f"adequate recovery paths (minimum: {min_paths} paths).",
                impact="Component failure may cause extended downtime due to "
                       "lack of alternative routes.",
                recommendation="Add redundant connections between critical components. "
                              "Consider mesh topology for critical infrastructure.",
                metrics={
                    'pairs_without_recovery': len(pairs_without_recovery),
                    'minimum_paths_required': min_paths,
                    'sample_pairs': [(p[0], p[1], p[2]) for p in pairs_without_recovery[:5]]
                }
            )
            issues.append(issue)
            
            # Mark critical edges
            for source, target, path_count in pairs_without_recovery[:10]:
                # Get edge if exists
                if graph.has_edge(source, target):
                    edge_data = graph.get_edge_data(source, target, {})
                    critical_edges.append(CriticalEdge(
                        source=source,
                        target=target,
                        edge_type=edge_data.get('dependency_type', 'unknown'),
                        quality_attribute=QualityAttribute.AVAILABILITY,
                        criticality_score=1.0 - (path_count / max(min_paths, 1)),
                        reasons=['insufficient_recovery_paths'],
                        metrics={'alternative_paths': path_count},
                        recommendations=["Add parallel connection", "Implement failover"]
                    ))
        
        return issues, critical_edges
    
    def _analyze_partition_risk(self, graph: nx.DiGraph) -> List[QualityIssue]:
        """Analyze risk of network partitioning"""
        issues = []
        
        # Check for multiple weakly connected components
        components = list(nx.weakly_connected_components(graph))
        
        if len(components) > 1:
            component_sizes = [len(c) for c in components]
            
            issue = QualityIssue(
                issue_id=self._generate_issue_id(),
                quality_attribute=QualityAttribute.AVAILABILITY,
                category=IssueCategory.PARTITION_RISK,
                severity=Severity.CRITICAL,
                affected_components=[],
                description=f"System is already partitioned into {len(components)} "
                           f"disconnected components (sizes: {component_sizes}).",
                impact="Parts of the system cannot communicate. "
                       "Data consistency and coordination impossible across partitions.",
                recommendation="Connect isolated components. "
                              "Review network topology and routing.",
                metrics={
                    'partition_count': len(components),
                    'component_sizes': component_sizes
                }
            )
            issues.append(issue)
        
        # Analyze potential partitions (minimum cut)
        try:
            undirected = graph.to_undirected()
            if undirected.number_of_nodes() >= 2:
                nodes = list(undirected.nodes())
                for source in nodes[:3]:
                    for target in nodes[-3:]:
                        if source != target:
                            try:
                                cut_value, partition = nx.minimum_cut(
                                    undirected, source, target
                                )
                                
                                if cut_value <= 1:
                                    reachable, non_reachable = partition
                                    if len(reachable) > 1 and len(non_reachable) > 1:
                                        issue = QualityIssue(
                                            issue_id=self._generate_issue_id(),
                                            quality_attribute=QualityAttribute.AVAILABILITY,
                                            category=IssueCategory.PARTITION_RISK,
                                            severity=Severity.HIGH,
                                            affected_components=list(reachable)[:5] + list(non_reachable)[:5],
                                            description=f"System can be partitioned by removing "
                                                       f"only {cut_value} edge(s).",
                                            impact="Easy to partition into isolated regions.",
                                            recommendation="Add redundant cross-cutting connections.",
                                            metrics={
                                                'min_cut_value': cut_value,
                                                'partition_sizes': [len(reachable), len(non_reachable)]
                                            }
                                        )
                                        issues.append(issue)
                                        break
                            except:
                                pass
                    else:
                        continue
                    break
        except:
            pass
        
        return issues
    
    def _identify_uptime_threats(self, graph: nx.DiGraph
                                 ) -> Tuple[List[QualityIssue], List[CriticalComponent]]:
        """Identify components that threaten system uptime"""
        issues = []
        critical = []
        
        # Components whose failure causes maximum reachability loss
        uptime_threats = []
        
        for node in graph.nodes():
            # Calculate impact of this node failing
            test_graph = graph.copy()
            test_graph.remove_node(node)
            
            # Check if system becomes disconnected
            original_components = nx.number_weakly_connected_components(graph)
            new_components = nx.number_weakly_connected_components(test_graph)
            
            if new_components > original_components:
                # Calculate reachability loss
                original_reach = sum(len(nx.descendants(graph, n)) for n in graph.nodes())
                new_reach = sum(len(nx.descendants(test_graph, n)) for n in test_graph.nodes())
                
                reach_loss = (original_reach - new_reach) / max(original_reach, 1)
                
                uptime_threats.append({
                    'node': node,
                    'components_created': new_components - original_components,
                    'reachability_loss': reach_loss,
                    'type': graph.nodes[node].get('type', 'unknown')
                })
        
        # Sort by impact
        uptime_threats.sort(key=lambda x: x['reachability_loss'], reverse=True)
        
        for threat in uptime_threats[:10]:
            severity = Severity.CRITICAL if threat['reachability_loss'] > 0.5 else Severity.HIGH
            comp_type = self._get_component_type(graph, threat['node'])
            
            issue = QualityIssue(
                issue_id=self._generate_issue_id(),
                quality_attribute=QualityAttribute.AVAILABILITY,
                category=IssueCategory.UPTIME_THREAT,
                severity=severity,
                affected_components=[threat['node']],
                description=f"Component '{threat['node']}' ({threat['type']}) is a significant "
                           f"uptime threat. Failure causes {threat['reachability_loss']*100:.1f}% "
                           f"reachability loss.",
                impact=f"Creates {threat['components_created']} additional partitions on failure.",
                recommendation="Add redundancy for this component. "
                              "Implement health monitoring and automated failover.",
                metrics={
                    'reachability_loss': threat['reachability_loss'],
                    'components_created': threat['components_created']
                }
            )
            issues.append(issue)
            
            comp = CriticalComponent(
                component_id=threat['node'],
                component_type=comp_type,
                quality_attribute=QualityAttribute.AVAILABILITY,
                criticality_score=threat['reachability_loss'],
                reasons=['uptime_threat', 'high_impact_failure'],
                metrics={
                    'reachability_loss': threat['reachability_loss'],
                    'partitions_created': threat['components_created']
                },
                recommendations=[
                    "Deploy redundant instance",
                    "Implement health checks",
                    "Add automated failover"
                ]
            )
            critical.append(comp)
        
        return issues, critical
    
    def _analyze_fault_tolerance(self, graph: nx.DiGraph,
                                  metrics: AvailabilityMetrics) -> List[QualityIssue]:
        """Analyze overall fault tolerance"""
        issues = []
        
        if self.config['single_failure_tolerance']:
            # Check if system survives any single failure
            single_failure_safe = metrics.node_connectivity >= 2
            
            if not single_failure_safe:
                # Find which single failures cause problems
                problem_nodes = []
                articulation_points = GraphAnalysisUtils.find_articulation_points(graph)
                
                issue = QualityIssue(
                    issue_id=self._generate_issue_id(),
                    quality_attribute=QualityAttribute.AVAILABILITY,
                    category=IssueCategory.FAULT_TOLERANCE_GAP,
                    severity=Severity.CRITICAL,
                    affected_components=list(articulation_points)[:10],
                    description=f"System does not tolerate single failures. "
                               f"{len(articulation_points)} single points of failure exist.",
                    impact="Any failure in identified components causes service disruption.",
                    recommendation="Achieve at least 2-connectivity through redundancy.",
                    metrics={
                        'articulation_point_count': len(articulation_points),
                        'node_connectivity': metrics.node_connectivity
                    }
                )
                issues.append(issue)
        
        return issues
    
    def _find_minimum_cut_nodes(self, graph: nx.DiGraph) -> List[str]:
        """Find nodes that are part of minimum vertex cut"""
        try:
            undirected = graph.to_undirected()
            articulation = list(nx.articulation_points(undirected))
            return articulation
        except:
            return []
    
    def _connectivity_severity(self, connectivity: int) -> Severity:
        """Convert connectivity to severity"""
        thresholds = self.config['severity_thresholds']['connectivity']
        if connectivity <= thresholds['critical']:
            return Severity.CRITICAL
        elif connectivity <= thresholds['high']:
            return Severity.HIGH
        elif connectivity <= thresholds['medium']:
            return Severity.MEDIUM
        return Severity.LOW
    
    def _calculate_availability_score(self, graph: nx.DiGraph,
                                      issues: List[QualityIssue],
                                      metrics: AvailabilityMetrics) -> float:
        """Calculate overall availability score (0-100)"""
        score = 100.0
        
        # Deduct for issues
        severity_penalties = {
            Severity.CRITICAL: 18,
            Severity.HIGH: 10,
            Severity.MEDIUM: 4,
            Severity.LOW: 1
        }
        
        for issue in issues:
            score -= severity_penalties.get(issue.severity, 0)
        
        # Bonus for good connectivity
        if metrics.node_connectivity >= 3:
            score += 10
        elif metrics.node_connectivity >= 2:
            score += 5
        
        # Bonus for good recovery coverage
        if metrics.recovery_coverage >= 0.9:
            score += 5
        
        # Bonus for good partition resistance
        if metrics.partition_resistance >= 0.8:
            score += 5
        
        return max(0, min(100, score))
    
    def _generate_recommendations(self, issues: List[QualityIssue],
                                  metrics: AvailabilityMetrics) -> List[str]:
        """Generate prioritized recommendations"""
        recommendations = []
        
        # Critical issues
        critical_issues = [i for i in issues if i.severity == Severity.CRITICAL]
        if critical_issues:
            recommendations.append(
                f"🔴 URGENT: Address {len(critical_issues)} critical availability issues"
            )
        
        # Connectivity recommendations
        if metrics.node_connectivity < 2:
            recommendations.append(
                f"Improve connectivity from {metrics.node_connectivity} to at least 2"
            )
        
        # Recovery path recommendations
        if metrics.recovery_coverage < 0.8:
            recommendations.append(
                f"Improve recovery path coverage from {metrics.recovery_coverage*100:.0f}% to 80%+"
            )
        
        # Partition resistance
        if metrics.partition_resistance < 0.5:
            recommendations.append(
                "Improve partition resistance with additional cross-connections"
            )
        
        # General
        if not recommendations:
            recommendations.append(
                "✅ Good availability posture. Continue monitoring for degradation."
            )
        
        return recommendations