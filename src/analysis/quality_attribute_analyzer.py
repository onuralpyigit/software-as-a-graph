#!/usr/bin/env python3
"""
Quality Attribute Analyzer - Base Classes and Interfaces
=========================================================

Provides base classes for analyzing quality attributes (Reliability,
Maintainability, Availability) in distributed pub-sub systems using
graph-based analysis.

Quality Attribute Mapping to Graph Properties:
- Reliability: SPOFs, cascade risks, redundancy gaps, failure propagation
- Maintainability: Coupling metrics, anti-patterns, modularity, complexity
- Availability: Uptime threats, k-connectivity, fault tolerance, recovery paths

Author: Software-as-a-Graph Research Project
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from collections import defaultdict
import json
from pathlib import Path

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX is required: pip install networkx")


# ============================================================================
# Enums and Constants
# ============================================================================

class QualityAttribute(Enum):
    """Software quality attributes analyzed"""
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

    def __lt__(self, other):
        order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]
        return order.index(self) < order.index(other)


class IssueCategory(Enum):
    """Categories of detected issues"""
    # Reliability categories
    SINGLE_POINT_OF_FAILURE = "single_point_of_failure"
    CASCADE_FAILURE_RISK = "cascade_failure_risk"
    MISSING_REDUNDANCY = "missing_redundancy"
    FAILURE_PROPAGATION = "failure_propagation"
    RELIABILITY_BOTTLENECK = "reliability_bottleneck"
    
    # Maintainability categories
    HIGH_COUPLING = "high_coupling"
    GOD_COMPONENT = "god_component"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    MODULARITY_VIOLATION = "modularity_violation"
    COMPLEXITY_HOTSPOT = "complexity_hotspot"
    TIGHT_COUPLING_CLUSTER = "tight_coupling_cluster"
    
    # Availability categories
    LOW_CONNECTIVITY = "low_connectivity"
    FAULT_TOLERANCE_GAP = "fault_tolerance_gap"
    RECOVERY_PATH_MISSING = "recovery_path_missing"
    UPTIME_THREAT = "uptime_threat"
    PARTITION_RISK = "partition_risk"


class ComponentType(Enum):
    """Types of components in pub-sub system"""
    APPLICATION = "Application"
    TOPIC = "Topic"
    BROKER = "Broker"
    NODE = "Node"


class DependencyType(Enum):
    """Types of DEPENDS_ON relationships"""
    APP_TO_APP = "app_to_app"
    APP_TO_BROKER = "app_to_broker"
    APP_TO_TOPIC = "app_to_topic"
    NODE_TO_NODE = "node_to_node"
    NODE_TO_BROKER = "node_to_broker"
    BROKER_TO_NODE = "broker_to_node"
    TOPIC_TO_BROKER = "topic_to_broker"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class QualityIssue:
    """Represents a detected quality issue"""
    issue_id: str
    quality_attribute: QualityAttribute
    category: IssueCategory
    severity: Severity
    affected_components: List[str]
    affected_edges: List[Tuple[str, str]] = field(default_factory=list)
    description: str = ""
    impact: str = ""
    recommendation: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'issue_id': self.issue_id,
            'quality_attribute': self.quality_attribute.value,
            'category': self.category.value,
            'severity': self.severity.value,
            'affected_components': self.affected_components,
            'affected_edges': [list(e) for e in self.affected_edges],
            'description': self.description,
            'impact': self.impact,
            'recommendation': self.recommendation,
            'metrics': self.metrics
        }


@dataclass
class CriticalComponent:
    """A component identified as critical for a quality attribute"""
    component_id: str
    component_type: ComponentType
    quality_attribute: QualityAttribute
    criticality_score: float
    reasons: List[str]
    metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'component_id': self.component_id,
            'component_type': self.component_type.value,
            'quality_attribute': self.quality_attribute.value,
            'criticality_score': round(self.criticality_score, 4),
            'reasons': self.reasons,
            'metrics': {k: round(v, 4) if isinstance(v, float) else v 
                       for k, v in self.metrics.items()},
            'recommendations': self.recommendations
        }


@dataclass
class CriticalEdge:
    """An edge identified as critical for a quality attribute"""
    source: str
    target: str
    edge_type: str
    quality_attribute: QualityAttribute
    criticality_score: float
    reasons: List[str]
    metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'target': self.target,
            'edge_type': self.edge_type,
            'quality_attribute': self.quality_attribute.value,
            'criticality_score': round(self.criticality_score, 4),
            'reasons': self.reasons,
            'metrics': {k: round(v, 4) if isinstance(v, float) else v 
                       for k, v in self.metrics.items()},
            'recommendations': self.recommendations
        }


@dataclass
class QualityAttributeResult:
    """Result of quality attribute analysis"""
    quality_attribute: QualityAttribute
    score: float  # 0-100 scale, higher is better
    issues: List[QualityIssue]
    critical_components: List[CriticalComponent]
    critical_edges: List[CriticalEdge]
    metrics: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'quality_attribute': self.quality_attribute.value,
            'score': round(self.score, 2),
            'issue_count': len(self.issues),
            'critical_component_count': len(self.critical_components),
            'critical_edge_count': len(self.critical_edges),
            'issues': [i.to_dict() for i in self.issues],
            'critical_components': [c.to_dict() for c in self.critical_components],
            'critical_edges': [e.to_dict() for e in self.critical_edges],
            'metrics': self.metrics,
            'recommendations': self.recommendations
        }
    
    def get_issues_by_severity(self, severity: Severity) -> List[QualityIssue]:
        return [i for i in self.issues if i.severity == severity]
    
    def get_critical_issues(self) -> List[QualityIssue]:
        return self.get_issues_by_severity(Severity.CRITICAL)


@dataclass
class ComprehensiveAnalysisResult:
    """Combined results from all quality attribute analyses"""
    reliability: QualityAttributeResult
    maintainability: QualityAttributeResult
    availability: QualityAttributeResult
    graph_summary: Dict[str, Any]
    overall_health_score: float
    top_critical_components: List[CriticalComponent]
    top_critical_edges: List[CriticalEdge]
    prioritized_recommendations: List[str]
    antipatterns: Optional[Any] = None  # AntiPatternAnalysisResult when available
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'overall_health_score': round(self.overall_health_score, 2),
            'graph_summary': self.graph_summary,
            'reliability': self.reliability.to_dict() if self.reliability else None,
            'maintainability': self.maintainability.to_dict() if self.maintainability else None,
            'availability': self.availability.to_dict() if self.availability else None,
            'top_critical_components': [c.to_dict() for c in self.top_critical_components],
            'top_critical_edges': [e.to_dict() for e in self.top_critical_edges],
            'prioritized_recommendations': self.prioritized_recommendations
        }
        if self.antipatterns:
            result['antipatterns'] = self.antipatterns.to_dict() if hasattr(self.antipatterns, 'to_dict') else self.antipatterns
        return result


# ============================================================================
# Base Analyzer Class
# ============================================================================

class BaseQualityAnalyzer(ABC):
    """
    Abstract base class for quality attribute analyzers.
    
    Subclasses implement specific analysis for:
    - ReliabilityAnalyzer
    - MaintainabilityAnalyzer  
    - AvailabilityAnalyzer
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        self._issue_counter = 0
    
    @property
    @abstractmethod
    def quality_attribute(self) -> QualityAttribute:
        """Return the quality attribute this analyzer handles"""
        pass
    
    @abstractmethod
    def analyze(self, graph: nx.DiGraph) -> QualityAttributeResult:
        """
        Perform quality attribute analysis on the graph.
        
        Args:
            graph: NetworkX DiGraph with DEPENDS_ON relationships
            
        Returns:
            QualityAttributeResult with findings
        """
        pass
    
    def _generate_issue_id(self) -> str:
        """Generate unique issue ID"""
        self._issue_counter += 1
        prefix = self.quality_attribute.value[:3].upper()
        return f"{prefix}-{self._issue_counter:04d}"
    
    def _get_component_type(self, graph: nx.DiGraph, node: str) -> ComponentType:
        """Get the type of a component from graph node attributes"""
        node_data = graph.nodes.get(node, {})
        type_str = node_data.get('type', node_data.get('label', 'Unknown'))
        
        type_mapping = {
            'application': ComponentType.APPLICATION,
            'app': ComponentType.APPLICATION,
            'topic': ComponentType.TOPIC,
            'broker': ComponentType.BROKER,
            'node': ComponentType.NODE,
            'infrastructure': ComponentType.NODE
        }
        
        return type_mapping.get(type_str.lower(), ComponentType.APPLICATION)
    
    def _calculate_percentile_threshold(self, values: List[float], 
                                        percentile: float = 90) -> float:
        """Calculate threshold value at given percentile"""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * percentile / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]
    
    def _normalize_score(self, value: float, min_val: float = 0, 
                        max_val: float = 1) -> float:
        """Normalize a value to 0-1 range"""
        if max_val <= min_val:
            return 0.0
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


# ============================================================================
# Graph Analysis Utilities
# ============================================================================

class GraphAnalysisUtils:
    """
    Utility functions for graph analysis across quality attributes.
    """
    
    @staticmethod
    def get_component_layers(graph: nx.DiGraph) -> Dict[str, Set[str]]:
        """
        Group nodes by their layer/type.
        
        Returns:
            Dict mapping layer name to set of node IDs
        """
        layers = defaultdict(set)
        for node in graph.nodes():
            node_data = graph.nodes[node]
            layer = node_data.get('type', node_data.get('layer', 'unknown'))
            layers[layer.lower()].add(node)
        return dict(layers)
    
    @staticmethod
    def get_dependency_types(graph: nx.DiGraph) -> Dict[str, List[Tuple[str, str]]]:
        """
        Group edges by dependency type.
        
        Returns:
            Dict mapping dependency type to list of (source, target) tuples
        """
        dep_types = defaultdict(list)
        for u, v, data in graph.edges(data=True):
            dep_type = data.get('dependency_type', data.get('type', 'unknown'))
            dep_types[dep_type].append((u, v))
        return dict(dep_types)
    
    @staticmethod
    def calculate_reachability_loss(graph: nx.DiGraph, 
                                    failed_nodes: Set[str]) -> float:
        """
        Calculate what fraction of reachability is lost when nodes fail.
        
        Args:
            graph: Original graph
            failed_nodes: Set of failed node IDs
            
        Returns:
            Fraction of reachability lost (0-1)
        """
        if not graph.nodes():
            return 0.0
        
        # Original reachability
        original_pairs = 0
        for node in graph.nodes():
            original_pairs += len(nx.descendants(graph, node))
        
        if original_pairs == 0:
            return 0.0
        
        # Create graph without failed nodes
        surviving = graph.copy()
        surviving.remove_nodes_from(failed_nodes)
        
        # New reachability
        new_pairs = 0
        for node in surviving.nodes():
            new_pairs += len(nx.descendants(surviving, node))
        
        return (original_pairs - new_pairs) / original_pairs
    
    @staticmethod
    def find_critical_paths(graph: nx.DiGraph, 
                           source_layer: str = 'application',
                           target_layer: str = 'broker') -> List[List[str]]:
        """
        Find critical paths between layers.
        
        Returns:
            List of paths as node ID lists
        """
        paths = []
        layers = GraphAnalysisUtils.get_component_layers(graph)
        
        sources = layers.get(source_layer, set())
        targets = layers.get(target_layer, set())
        
        for source in sources:
            for target in targets:
                try:
                    path = nx.shortest_path(graph, source, target)
                    paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        return paths
    
    @staticmethod
    def compute_centrality_metrics(graph: nx.DiGraph) -> Dict[str, Dict[str, float]]:
        """
        Compute multiple centrality metrics for all nodes.
        
        Returns:
            Dict with metric name -> {node: score}
        """
        metrics = {}
        
        # Betweenness centrality
        try:
            metrics['betweenness'] = nx.betweenness_centrality(graph)
        except:
            metrics['betweenness'] = {}
        
        # PageRank
        try:
            metrics['pagerank'] = nx.pagerank(graph)
        except:
            metrics['pagerank'] = {}
        
        # In/Out degree centrality
        try:
            metrics['in_degree'] = dict(graph.in_degree())
            metrics['out_degree'] = dict(graph.out_degree())
        except:
            metrics['in_degree'] = {}
            metrics['out_degree'] = {}
        
        # Closeness centrality
        try:
            metrics['closeness'] = nx.closeness_centrality(graph)
        except:
            metrics['closeness'] = {}
        
        return metrics
    
    @staticmethod
    def find_articulation_points(graph: nx.DiGraph) -> Set[str]:
        """Find articulation points (cut vertices)"""
        try:
            undirected = graph.to_undirected()
            return set(nx.articulation_points(undirected))
        except:
            return set()
    
    @staticmethod
    def find_bridges(graph: nx.DiGraph) -> List[Tuple[str, str]]:
        """Find bridge edges"""
        try:
            undirected = graph.to_undirected()
            return list(nx.bridges(undirected))
        except:
            return []
    
    @staticmethod
    def find_strongly_connected_components(graph: nx.DiGraph) -> List[Set[str]]:
        """Find strongly connected components (cycles)"""
        try:
            return [c for c in nx.strongly_connected_components(graph) if len(c) > 1]
        except:
            return []
    
    @staticmethod
    def compute_k_connectivity(graph: nx.DiGraph) -> int:
        """Compute vertex connectivity (k)"""
        try:
            undirected = graph.to_undirected()
            return nx.node_connectivity(undirected)
        except:
            return 0
    
    @staticmethod
    def detect_communities(graph: nx.DiGraph) -> Dict[str, int]:
        """
        Detect communities using Louvain method.
        
        Returns:
            Dict mapping node to community ID
        """
        try:
            from networkx.algorithms import community
            undirected = graph.to_undirected()
            communities = community.louvain_communities(undirected, seed=42)
            node_to_community = {}
            for idx, comm in enumerate(communities):
                for node in comm:
                    node_to_community[node] = idx
            return node_to_community
        except:
            return {node: 0 for node in graph.nodes()}


# ============================================================================
# Issue Formatter
# ============================================================================

class IssueFormatter:
    """Formats quality issues for different output formats"""
    
    @staticmethod
    def to_terminal(issues: List[QualityIssue], max_issues: int = 20) -> str:
        """Format issues for terminal output"""
        if not issues:
            return "No issues detected."
        
        lines = []
        severity_symbols = {
            Severity.CRITICAL: "🔴",
            Severity.HIGH: "🟠",
            Severity.MEDIUM: "🟡",
            Severity.LOW: "🟢",
            Severity.INFO: "ℹ️"
        }
        
        for issue in sorted(issues, key=lambda x: x.severity)[:max_issues]:
            symbol = severity_symbols.get(issue.severity, "•")
            lines.append(f"\n{symbol} [{issue.severity.value.upper()}] {issue.category.value}")
            lines.append(f"   Components: {', '.join(issue.affected_components[:5])}")
            if len(issue.affected_components) > 5:
                lines.append(f"              (+{len(issue.affected_components)-5} more)")
            lines.append(f"   Impact: {issue.impact}")
            lines.append(f"   Recommendation: {issue.recommendation}")
        
        if len(issues) > max_issues:
            lines.append(f"\n... and {len(issues) - max_issues} more issues")
        
        return "\n".join(lines)
    
    @staticmethod
    def to_html(issues: List[QualityIssue]) -> str:
        """Format issues as HTML table"""
        if not issues:
            return "<p>No issues detected.</p>"
        
        html = """
<table class="issues-table">
<thead>
<tr>
    <th>Severity</th>
    <th>Category</th>
    <th>Components</th>
    <th>Impact</th>
    <th>Recommendation</th>
</tr>
</thead>
<tbody>
"""
        
        for issue in sorted(issues, key=lambda x: x.severity):
            severity_class = f"severity-{issue.severity.value}"
            components = ", ".join(issue.affected_components[:3])
            if len(issue.affected_components) > 3:
                components += f" (+{len(issue.affected_components)-3})"
            
            html += f"""
<tr class="{severity_class}">
    <td><span class="badge {severity_class}">{issue.severity.value.upper()}</span></td>
    <td>{issue.category.value}</td>
    <td>{components}</td>
    <td>{issue.impact}</td>
    <td>{issue.recommendation}</td>
</tr>
"""
        
        html += "</tbody></table>"
        return html


if __name__ == "__main__":
    # Simple test
    print("Quality Attribute Analyzer Module loaded successfully")
    print(f"Quality Attributes: {[qa.value for qa in QualityAttribute]}")
    print(f"Issue Categories: {[ic.value for ic in IssueCategory]}")
