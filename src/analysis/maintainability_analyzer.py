#!/usr/bin/env python3
"""
Maintainability Analyzer
========================

Analyzes system maintainability by detecting:
- High coupling (afferent/efferent coupling metrics)
- God components (excessive responsibilities)
- Circular dependencies
- Modularity violations
- Complexity hotspots
- Tight coupling clusters

Graph Algorithms Used:
- Degree analysis (coupling metrics)
- Community detection (modularity)
- SCC detection (circular dependencies)
- Clique detection (tight coupling)
- Martin's coupling metrics (Ca, Ce, I, A)

Author: Software-as-a-Graph Research Project
"""

import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import math

try:
    import networkx as nx
    from networkx.algorithms import community
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

DEFAULT_MAINTAINABILITY_CONFIG = {
    # Coupling thresholds (based on Martin's metrics)
    'afferent_coupling_threshold': 10,  # Ca - incoming dependencies
    'efferent_coupling_threshold': 10,  # Ce - outgoing dependencies
    'instability_threshold': 0.8,       # I = Ce / (Ca + Ce)
    
    # God component thresholds
    'god_component_threshold': 15,      # Total connections
    'god_topic_threshold': 10,          # Publishers + subscribers
    
    # Circular dependency settings
    'max_cycle_size_to_report': 10,
    
    # Modularity settings
    'modularity_threshold': 0.3,        # Minimum acceptable modularity
    'cross_module_edge_threshold': 0.3, # Max % of cross-module edges
    
    # Complexity thresholds
    'complexity_depth_threshold': 4,    # Dependency chain depth
    'complexity_breadth_threshold': 8,  # Fan-out
    
    # Tight coupling
    'clique_size_threshold': 4,         # Min clique size to report
    
    # Scoring
    'severity_thresholds': {
        'god_component': {'critical': 25, 'high': 20, 'medium': 15},
        'coupling': {'critical': 15, 'high': 10, 'medium': 5},
        'cycle_size': {'critical': 6, 'high': 4, 'medium': 3}
    }
}


# ============================================================================
# Maintainability Metrics
# ============================================================================

@dataclass
class CouplingMetrics:
    """Robert Martin's coupling metrics for a component"""
    component_id: str
    afferent_coupling: int      # Ca - incoming dependencies
    efferent_coupling: int      # Ce - outgoing dependencies
    instability: float          # I = Ce / (Ca + Ce), 0 = stable, 1 = unstable
    abstractness: float         # A - ratio of abstract to total (approximated)
    distance_from_main: float   # D = |A + I - 1| (distance from main sequence)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'component_id': self.component_id,
            'Ca': self.afferent_coupling,
            'Ce': self.efferent_coupling,
            'I': round(self.instability, 4),
            'A': round(self.abstractness, 4),
            'D': round(self.distance_from_main, 4)
        }


# ============================================================================
# Maintainability Analyzer
# ============================================================================

class MaintainabilityAnalyzer(BaseQualityAnalyzer):
    """
    Analyzes system maintainability using graph-based methods.
    
    Maintainability is assessed through:
    1. Coupling Analysis: Martin's metrics (Ca, Ce, I, A, D)
    2. God Components: Excessive responsibility concentration
    3. Circular Dependencies: SCCs that complicate changes
    4. Modularity: Community structure and cross-module coupling
    5. Complexity: Dependency chains and fan-out
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.config = {**DEFAULT_MAINTAINABILITY_CONFIG, **(config or {})}
        self.logger = logging.getLogger('MaintainabilityAnalyzer')
    
    @property
    def quality_attribute(self) -> QualityAttribute:
        return QualityAttribute.MAINTAINABILITY
    
    def analyze(self, graph: nx.DiGraph) -> QualityAttributeResult:
        """
        Perform comprehensive maintainability analysis.
        
        Args:
            graph: NetworkX DiGraph with DEPENDS_ON relationships
            
        Returns:
            QualityAttributeResult with maintainability findings
        """
        self.logger.info("Starting maintainability analysis...")
        
        issues = []
        critical_components = []
        critical_edges = []
        metrics = {}
        
        # Pre-compute metrics
        self.logger.info("Computing coupling metrics...")
        coupling_metrics = self._compute_coupling_metrics(graph)
        
        self.logger.info("Detecting communities...")
        communities = self._detect_communities(graph)
        
        # Store summary metrics
        metrics['average_coupling'] = self._calculate_average_coupling(coupling_metrics)
        metrics['modularity'] = communities.get('modularity', 0)
        metrics['community_count'] = communities.get('community_count', 1)
        
        # 1. Analyze Coupling
        self.logger.info("Analyzing coupling issues...")
        coupling_issues, coupling_components = self._analyze_coupling(
            graph, coupling_metrics
        )
        issues.extend(coupling_issues)
        critical_components.extend(coupling_components)
        
        # 2. Detect God Components
        self.logger.info("Detecting god components...")
        god_issues, god_components = self._detect_god_components(graph)
        issues.extend(god_issues)
        critical_components.extend(god_components)
        
        # 3. Detect Circular Dependencies
        self.logger.info("Detecting circular dependencies...")
        circular_issues = self._detect_circular_dependencies(graph)
        issues.extend(circular_issues)
        
        # 4. Analyze Modularity
        self.logger.info("Analyzing modularity...")
        modularity_issues, cross_edges = self._analyze_modularity(
            graph, communities
        )
        issues.extend(modularity_issues)
        critical_edges.extend(cross_edges)
        
        # 5. Detect Complexity Hotspots
        self.logger.info("Detecting complexity hotspots...")
        complexity_issues, complexity_components = self._detect_complexity_hotspots(
            graph
        )
        issues.extend(complexity_issues)
        critical_components.extend(complexity_components)
        
        # 6. Detect Tight Coupling Clusters
        self.logger.info("Detecting tight coupling clusters...")
        cluster_issues = self._detect_tight_coupling_clusters(graph)
        issues.extend(cluster_issues)
        
        # Calculate overall maintainability score
        maintainability_score = self._calculate_maintainability_score(
            graph, issues, metrics, coupling_metrics
        )
        metrics['maintainability_score'] = maintainability_score
        metrics['coupling_details'] = [cm.to_dict() for cm in coupling_metrics[:20]]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, metrics)
        
        # Sort results
        issues.sort(key=lambda x: x.severity)
        critical_components.sort(key=lambda x: -x.criticality_score)
        critical_edges.sort(key=lambda x: -x.criticality_score)
        
        self.logger.info(f"Maintainability analysis complete. Score: {maintainability_score:.1f}/100")
        
        return QualityAttributeResult(
            quality_attribute=QualityAttribute.MAINTAINABILITY,
            score=maintainability_score,
            issues=issues,
            critical_components=critical_components,
            critical_edges=critical_edges,
            metrics=metrics,
            recommendations=recommendations
        )
    
    def _compute_coupling_metrics(self, graph: nx.DiGraph) -> List[CouplingMetrics]:
        """Compute Martin's coupling metrics for all nodes"""
        metrics = []
        
        for node in graph.nodes():
            # Afferent coupling (Ca) - incoming dependencies
            ca = graph.in_degree(node)
            
            # Efferent coupling (Ce) - outgoing dependencies
            ce = graph.out_degree(node)
            
            # Instability I = Ce / (Ca + Ce)
            total = ca + ce
            instability = ce / total if total > 0 else 0
            
            # Abstractness (approximated based on component type)
            # In pub-sub: Topics are abstract, Applications are concrete
            node_type = graph.nodes[node].get('type', '').lower()
            abstractness = 0.8 if node_type == 'topic' else 0.2
            
            # Distance from main sequence D = |A + I - 1|
            distance = abs(abstractness + instability - 1)
            
            metrics.append(CouplingMetrics(
                component_id=node,
                afferent_coupling=ca,
                efferent_coupling=ce,
                instability=instability,
                abstractness=abstractness,
                distance_from_main=distance
            ))
        
        # Sort by total coupling (descending)
        metrics.sort(key=lambda m: m.afferent_coupling + m.efferent_coupling, reverse=True)
        
        return metrics
    
    def _detect_communities(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Detect communities and compute modularity"""
        try:
            undirected = graph.to_undirected()
            
            # Use Louvain for community detection
            communities_list = list(community.louvain_communities(undirected, seed=42))
            
            # Compute modularity
            modularity_score = community.modularity(undirected, communities_list)
            
            # Map nodes to communities
            node_to_community = {}
            for idx, comm in enumerate(communities_list):
                for node in comm:
                    node_to_community[node] = idx
            
            # Count cross-community edges
            cross_edges = 0
            total_edges = graph.number_of_edges()
            
            for u, v in graph.edges():
                if node_to_community.get(u) != node_to_community.get(v):
                    cross_edges += 1
            
            cross_edge_ratio = cross_edges / max(total_edges, 1)
            
            return {
                'community_count': len(communities_list),
                'modularity': modularity_score,
                'node_to_community': node_to_community,
                'communities': [list(c) for c in communities_list],
                'cross_edge_count': cross_edges,
                'cross_edge_ratio': cross_edge_ratio
            }
            
        except Exception as e:
            self.logger.warning(f"Community detection failed: {e}")
            return {
                'community_count': 1,
                'modularity': 0,
                'node_to_community': {n: 0 for n in graph.nodes()},
                'communities': [list(graph.nodes())],
                'cross_edge_count': 0,
                'cross_edge_ratio': 0
            }
    
    def _analyze_coupling(self, graph: nx.DiGraph,
                          coupling_metrics: List[CouplingMetrics]
                          ) -> Tuple[List[QualityIssue], List[CriticalComponent]]:
        """Analyze coupling issues"""
        issues = []
        critical = []
        
        ca_threshold = self.config['afferent_coupling_threshold']
        ce_threshold = self.config['efferent_coupling_threshold']
        instability_threshold = self.config['instability_threshold']
        
        for cm in coupling_metrics:
            has_issue = False
            reasons = []
            
            # High afferent coupling (many depend on this)
            if cm.afferent_coupling >= ca_threshold:
                has_issue = True
                reasons.append(f"high_afferent_coupling (Ca={cm.afferent_coupling})")
            
            # High efferent coupling (depends on many)
            if cm.efferent_coupling >= ce_threshold:
                has_issue = True
                reasons.append(f"high_efferent_coupling (Ce={cm.efferent_coupling})")
            
            # High instability with high afferent (unstable component many depend on)
            if cm.instability >= instability_threshold and cm.afferent_coupling >= 5:
                has_issue = True
                reasons.append(f"unstable_dependency (I={cm.instability:.2f})")
            
            if has_issue:
                total_coupling = cm.afferent_coupling + cm.efferent_coupling
                severity = self._coupling_to_severity(total_coupling)
                comp_type = self._get_component_type(graph, cm.component_id)
                
                issue = QualityIssue(
                    issue_id=self._generate_issue_id(),
                    quality_attribute=QualityAttribute.MAINTAINABILITY,
                    category=IssueCategory.HIGH_COUPLING,
                    severity=severity,
                    affected_components=[cm.component_id],
                    description=f"Component '{cm.component_id}' has high coupling. "
                               f"Ca={cm.afferent_coupling}, Ce={cm.efferent_coupling}, "
                               f"I={cm.instability:.2f}",
                    impact=f"Changes to this component affect {cm.afferent_coupling} dependents. "
                           f"Changes to {cm.efferent_coupling} dependencies may break this component.",
                    recommendation="Apply dependency inversion principle. "
                                  "Consider interface segregation. "
                                  "Refactor to reduce coupling.",
                    metrics=cm.to_dict()
                )
                issues.append(issue)
                
                comp = CriticalComponent(
                    component_id=cm.component_id,
                    component_type=comp_type,
                    quality_attribute=QualityAttribute.MAINTAINABILITY,
                    criticality_score=total_coupling / 20,  # Normalize
                    reasons=reasons,
                    metrics={
                        'Ca': cm.afferent_coupling,
                        'Ce': cm.efferent_coupling,
                        'I': cm.instability,
                        'D': cm.distance_from_main
                    },
                    recommendations=[
                        "Apply Dependency Inversion Principle",
                        "Use interfaces/abstractions",
                        "Consider breaking into smaller components"
                    ]
                )
                critical.append(comp)
        
        return issues, critical
    
    def _detect_god_components(self, graph: nx.DiGraph
                               ) -> Tuple[List[QualityIssue], List[CriticalComponent]]:
        """Detect god components with excessive responsibilities"""
        issues = []
        critical = []
        
        threshold = self.config['god_component_threshold']
        topic_threshold = self.config['god_topic_threshold']
        
        for node in graph.nodes():
            in_deg = graph.in_degree(node)
            out_deg = graph.out_degree(node)
            total = in_deg + out_deg
            
            node_type = graph.nodes[node].get('type', '').lower()
            
            # Check for god topic
            if node_type == 'topic' and total >= topic_threshold:
                severity = self._god_component_severity(total)
                
                # Get publishers and subscribers
                publishers = [p for p in graph.predecessors(node)]
                subscribers = [s for s in graph.successors(node)]
                
                issue = QualityIssue(
                    issue_id=self._generate_issue_id(),
                    quality_attribute=QualityAttribute.MAINTAINABILITY,
                    category=IssueCategory.GOD_COMPONENT,
                    severity=severity,
                    affected_components=[node] + publishers[:3] + subscribers[:3],
                    description=f"Topic '{node}' is a 'God Topic' with {len(publishers)} publishers "
                               f"and {len(subscribers)} subscribers (total: {total}).",
                    impact="Changes to this topic affect many components. "
                           "Difficult to evolve. Single point of coupling.",
                    recommendation="Split into multiple focused topics. "
                                  "Consider topic hierarchy or partitioning by domain.",
                    metrics={
                        'publishers': len(publishers),
                        'subscribers': len(subscribers),
                        'total_connections': total
                    }
                )
                issues.append(issue)
                
                comp = CriticalComponent(
                    component_id=node,
                    component_type=ComponentType.TOPIC,
                    quality_attribute=QualityAttribute.MAINTAINABILITY,
                    criticality_score=total / (topic_threshold * 2),
                    reasons=['god_topic', 'high_connectivity'],
                    metrics={
                        'publishers': len(publishers),
                        'subscribers': len(subscribers),
                        'total': total
                    },
                    recommendations=[
                        "Split into domain-specific topics",
                        "Use topic hierarchy",
                        "Consider event filtering"
                    ]
                )
                critical.append(comp)
            
            # Check for god application
            elif node_type == 'application' and total >= threshold:
                severity = self._god_component_severity(total)
                
                issue = QualityIssue(
                    issue_id=self._generate_issue_id(),
                    quality_attribute=QualityAttribute.MAINTAINABILITY,
                    category=IssueCategory.GOD_COMPONENT,
                    severity=severity,
                    affected_components=[node],
                    description=f"Application '{node}' has {total} dependencies "
                               f"({in_deg} incoming, {out_deg} outgoing).",
                    impact="Monolithic component. Hard to test, deploy, and modify independently.",
                    recommendation="Apply Single Responsibility Principle. "
                                  "Consider decomposing into smaller services.",
                    metrics={
                        'in_degree': in_deg,
                        'out_degree': out_deg,
                        'total': total
                    }
                )
                issues.append(issue)
                
                comp = CriticalComponent(
                    component_id=node,
                    component_type=ComponentType.APPLICATION,
                    quality_attribute=QualityAttribute.MAINTAINABILITY,
                    criticality_score=total / (threshold * 2),
                    reasons=['god_component', 'high_coupling'],
                    metrics={
                        'in_degree': in_deg,
                        'out_degree': out_deg,
                        'total': total
                    },
                    recommendations=[
                        "Decompose into microservices",
                        "Apply Single Responsibility Principle",
                        "Use bounded contexts"
                    ]
                )
                critical.append(comp)
        
        return issues, critical
    
    def _detect_circular_dependencies(self, graph: nx.DiGraph) -> List[QualityIssue]:
        """Detect circular dependencies using SCC detection"""
        issues = []
        
        sccs = list(nx.strongly_connected_components(graph))
        max_report = self.config['max_cycle_size_to_report']
        
        for scc in sccs:
            if len(scc) > 1:  # Non-trivial SCC = cycle
                cycle_nodes = list(scc)[:max_report]
                severity = self._cycle_severity(len(scc))
                
                issue = QualityIssue(
                    issue_id=self._generate_issue_id(),
                    quality_attribute=QualityAttribute.MAINTAINABILITY,
                    category=IssueCategory.CIRCULAR_DEPENDENCY,
                    severity=severity,
                    affected_components=cycle_nodes,
                    description=f"Circular dependency detected among {len(scc)} components: "
                               f"{', '.join(cycle_nodes[:5])}{'...' if len(scc) > 5 else ''}",
                    impact="Cannot deploy or test components independently. "
                           "Changes cascade unpredictably. Violates layered architecture.",
                    recommendation="Break the cycle by introducing async communication, "
                                  "events, or dependency inversion. "
                                  "Consider redesigning component responsibilities.",
                    metrics={
                        'cycle_size': len(scc),
                        'components': cycle_nodes[:10]
                    }
                )
                issues.append(issue)
        
        return issues
    
    def _analyze_modularity(self, graph: nx.DiGraph,
                            communities: Dict[str, Any]
                            ) -> Tuple[List[QualityIssue], List[CriticalEdge]]:
        """Analyze modularity and cross-module coupling"""
        issues = []
        critical_edges = []
        
        modularity = communities.get('modularity', 0)
        cross_ratio = communities.get('cross_edge_ratio', 0)
        node_to_comm = communities.get('node_to_community', {})
        
        # Low modularity warning
        if modularity < self.config['modularity_threshold']:
            issues.append(QualityIssue(
                issue_id=self._generate_issue_id(),
                quality_attribute=QualityAttribute.MAINTAINABILITY,
                category=IssueCategory.MODULARITY_VIOLATION,
                severity=Severity.MEDIUM,
                affected_components=list(graph.nodes())[:10],
                description=f"System has low modularity score: {modularity:.3f} "
                           f"(threshold: {self.config['modularity_threshold']})",
                impact="Components are not well-organized into cohesive modules. "
                       "Hard to understand and maintain system structure.",
                recommendation="Reorganize components into cohesive modules. "
                              "Apply bounded context patterns. "
                              "Reduce cross-module dependencies.",
                metrics={
                    'modularity': modularity,
                    'community_count': communities.get('community_count', 1)
                }
            ))
        
        # High cross-module coupling
        if cross_ratio > self.config['cross_module_edge_threshold']:
            issues.append(QualityIssue(
                issue_id=self._generate_issue_id(),
                quality_attribute=QualityAttribute.MAINTAINABILITY,
                category=IssueCategory.MODULARITY_VIOLATION,
                severity=Severity.HIGH,
                affected_components=[],
                description=f"High cross-module coupling: {cross_ratio*100:.1f}% of edges "
                           f"cross module boundaries.",
                impact="Modules are not properly isolated. "
                       "Changes in one module frequently affect others.",
                recommendation="Reduce cross-module dependencies. "
                              "Use well-defined interfaces between modules. "
                              "Apply facade pattern for module boundaries.",
                metrics={
                    'cross_edge_ratio': cross_ratio,
                    'cross_edge_count': communities.get('cross_edge_count', 0)
                }
            ))
        
        # Identify critical cross-module edges
        for u, v in graph.edges():
            u_comm = node_to_comm.get(u, 0)
            v_comm = node_to_comm.get(v, 0)
            
            if u_comm != v_comm:
                edge_data = graph.get_edge_data(u, v, {})
                edge_type = edge_data.get('dependency_type', 'unknown')
                
                critical_edges.append(CriticalEdge(
                    source=u,
                    target=v,
                    edge_type=edge_type,
                    quality_attribute=QualityAttribute.MAINTAINABILITY,
                    criticality_score=0.5,  # All cross-module edges are moderately critical
                    reasons=['cross_module_dependency'],
                    metrics={
                        'source_module': u_comm,
                        'target_module': v_comm
                    },
                    recommendations=[
                        "Consider moving to same module",
                        "Use well-defined interface",
                        "Evaluate if dependency is necessary"
                    ]
                ))
        
        return issues, critical_edges[:50]  # Limit to top 50
    
    def _detect_complexity_hotspots(self, graph: nx.DiGraph
                                    ) -> Tuple[List[QualityIssue], List[CriticalComponent]]:
        """Detect complexity hotspots based on dependency chains"""
        issues = []
        critical = []
        
        depth_threshold = self.config['complexity_depth_threshold']
        breadth_threshold = self.config['complexity_breadth_threshold']
        
        for node in graph.nodes():
            # Calculate depth (longest path from this node)
            descendants = list(nx.descendants(graph, node))
            
            if not descendants:
                continue
            
            max_depth = 0
            for desc in descendants:
                try:
                    path_len = nx.shortest_path_length(graph, node, desc)
                    max_depth = max(max_depth, path_len)
                except:
                    pass
            
            # Fan-out (direct dependencies)
            fan_out = graph.out_degree(node)
            
            # Check thresholds
            is_complex = max_depth >= depth_threshold or fan_out >= breadth_threshold
            
            if is_complex:
                complexity_score = (max_depth / depth_threshold + 
                                   fan_out / breadth_threshold) / 2
                severity = Severity.HIGH if complexity_score > 1.5 else Severity.MEDIUM
                comp_type = self._get_component_type(graph, node)
                
                issue = QualityIssue(
                    issue_id=self._generate_issue_id(),
                    quality_attribute=QualityAttribute.MAINTAINABILITY,
                    category=IssueCategory.COMPLEXITY_HOTSPOT,
                    severity=severity,
                    affected_components=[node] + descendants[:5],
                    description=f"Component '{node}' is a complexity hotspot. "
                               f"Dependency depth: {max_depth}, Fan-out: {fan_out}",
                    impact="Deep dependency chains make debugging difficult. "
                           "High fan-out complicates testing and deployment.",
                    recommendation="Reduce dependency chain depth. "
                                  "Consider caching or aggregation. "
                                  "Apply facade pattern for high fan-out.",
                    metrics={
                        'depth': max_depth,
                        'fan_out': fan_out,
                        'complexity_score': complexity_score
                    }
                )
                issues.append(issue)
                
                comp = CriticalComponent(
                    component_id=node,
                    component_type=comp_type,
                    quality_attribute=QualityAttribute.MAINTAINABILITY,
                    criticality_score=min(1.0, complexity_score),
                    reasons=['complexity_hotspot', 'deep_chain' if max_depth >= depth_threshold else 'high_fanout'],
                    metrics={
                        'depth': max_depth,
                        'fan_out': fan_out,
                        'descendant_count': len(descendants)
                    },
                    recommendations=[
                        "Flatten dependency hierarchy",
                        "Introduce caching/aggregation",
                        "Consider async communication"
                    ]
                )
                critical.append(comp)
        
        return issues, critical
    
    def _detect_tight_coupling_clusters(self, graph: nx.DiGraph) -> List[QualityIssue]:
        """Detect tightly coupled clusters (cliques)"""
        issues = []
        
        threshold = self.config['clique_size_threshold']
        
        try:
            undirected = graph.to_undirected()
            cliques = list(nx.find_cliques(undirected))
            
            for clique in cliques:
                if len(clique) >= threshold:
                    severity = Severity.HIGH if len(clique) >= threshold * 1.5 else Severity.MEDIUM
                    
                    # Check if all same type (worse)
                    types = set(graph.nodes[n].get('type', 'unknown') for n in clique)
                    same_type = len(types) == 1
                    
                    issue = QualityIssue(
                        issue_id=self._generate_issue_id(),
                        quality_attribute=QualityAttribute.MAINTAINABILITY,
                        category=IssueCategory.TIGHT_COUPLING_CLUSTER,
                        severity=severity,
                        affected_components=list(clique),
                        description=f"Tightly coupled cluster of {len(clique)} components: "
                                   f"{', '.join(list(clique)[:5])}{'...' if len(clique) > 5 else ''}",
                        impact="All components in cluster depend on each other. "
                               f"{'Same-type coupling is especially problematic. ' if same_type else ''}"
                               "Testing requires all components. Hard to change independently.",
                        recommendation="Introduce abstractions/interfaces. "
                                      "Apply dependency inversion. "
                                      "Consider event-driven decoupling.",
                        metrics={
                            'cluster_size': len(clique),
                            'same_type': same_type,
                            'types': list(types)
                        }
                    )
                    issues.append(issue)
        
        except Exception as e:
            self.logger.warning(f"Clique detection failed: {e}")
        
        return issues
    
    def _coupling_to_severity(self, coupling: int) -> Severity:
        """Convert coupling value to severity"""
        thresholds = self.config['severity_thresholds']['coupling']
        if coupling >= thresholds['critical']:
            return Severity.CRITICAL
        elif coupling >= thresholds['high']:
            return Severity.HIGH
        elif coupling >= thresholds['medium']:
            return Severity.MEDIUM
        return Severity.LOW
    
    def _god_component_severity(self, connections: int) -> Severity:
        """Convert god component connections to severity"""
        thresholds = self.config['severity_thresholds']['god_component']
        if connections >= thresholds['critical']:
            return Severity.CRITICAL
        elif connections >= thresholds['high']:
            return Severity.HIGH
        elif connections >= thresholds['medium']:
            return Severity.MEDIUM
        return Severity.LOW
    
    def _cycle_severity(self, size: int) -> Severity:
        """Convert cycle size to severity"""
        thresholds = self.config['severity_thresholds']['cycle_size']
        if size >= thresholds['critical']:
            return Severity.CRITICAL
        elif size >= thresholds['high']:
            return Severity.HIGH
        elif size >= thresholds['medium']:
            return Severity.MEDIUM
        return Severity.LOW
    
    def _calculate_average_coupling(self, metrics: List[CouplingMetrics]) -> float:
        """Calculate average coupling"""
        if not metrics:
            return 0
        total = sum(m.afferent_coupling + m.efferent_coupling for m in metrics)
        return total / len(metrics)
    
    def _calculate_maintainability_score(self, graph: nx.DiGraph,
                                         issues: List[QualityIssue],
                                         metrics: Dict[str, Any],
                                         coupling_metrics: List[CouplingMetrics]) -> float:
        """Calculate overall maintainability score (0-100)"""
        score = 100.0
        
        # Deduct for issues
        severity_penalties = {
            Severity.CRITICAL: 12,
            Severity.HIGH: 6,
            Severity.MEDIUM: 2,
            Severity.LOW: 0.5
        }
        
        for issue in issues:
            score -= severity_penalties.get(issue.severity, 0)
        
        # Bonus for good modularity
        modularity = metrics.get('modularity', 0)
        if modularity >= 0.5:
            score += 10
        elif modularity >= 0.3:
            score += 5
        
        # Penalty for high average coupling
        avg_coupling = metrics.get('average_coupling', 0)
        if avg_coupling > 10:
            score -= 15
        elif avg_coupling > 5:
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
                f"🔴 URGENT: Address {len(critical_issues)} critical maintainability issues"
            )
        
        # Coupling recommendations
        coupling_issues = [i for i in issues if i.category == IssueCategory.HIGH_COUPLING]
        if coupling_issues:
            recommendations.append(
                f"Reduce coupling in {len(coupling_issues)} highly-coupled components"
            )
        
        # Circular dependency recommendations
        circular_issues = [i for i in issues if i.category == IssueCategory.CIRCULAR_DEPENDENCY]
        if circular_issues:
            recommendations.append(
                f"Break {len(circular_issues)} circular dependency cycles"
            )
        
        # God component recommendations
        god_issues = [i for i in issues if i.category == IssueCategory.GOD_COMPONENT]
        if god_issues:
            recommendations.append(
                f"Refactor {len(god_issues)} god components into smaller units"
            )
        
        # Modularity recommendations
        modularity = metrics.get('modularity', 0)
        if modularity < 0.3:
            recommendations.append(
                "Improve system modularity through better component organization"
            )
        
        if not recommendations:
            recommendations.append(
                "✅ No critical maintainability issues. Continue with regular refactoring."
            )
        
        return recommendations