#!/usr/bin/env python3
"""
Anti-Pattern Detector for Distributed Pub-Sub Systems
======================================================

Detects architectural anti-patterns that affect:
- Reliability (SPOFs, cascade risks)
- Maintainability (god topics, tight coupling, circular dependencies)
- Availability (bottlenecks, partition risks)

Anti-Patterns Detected:
1. God Topic - Topic with excessive publishers/subscribers
2. Single Point of Failure - Articulation points
3. Circular Dependencies - SCCs in the dependency graph
4. Bottleneck Broker - Broker with disproportionate load
5. Chatty Publisher - Application with too many topic connections
6. Message Overload - Topic/broker receiving too many messages
7. Orphan Components - Disconnected components
8. Deep Dependency Chain - Excessively long dependency paths
9. Hub Antipattern - Single component connecting many others
10. Tight Coupling Cluster - Group of tightly interconnected components

Author: Software-as-a-Graph Research Project
"""

import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX is required: pip install networkx")


# ============================================================================
# Enums and Types
# ============================================================================

class AntiPatternType(Enum):
    """Types of architectural anti-patterns"""
    GOD_TOPIC = "god_topic"
    SINGLE_POINT_OF_FAILURE = "single_point_of_failure"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    BOTTLENECK_BROKER = "bottleneck_broker"
    CHATTY_PUBLISHER = "chatty_publisher"
    MESSAGE_OVERLOAD = "message_overload"
    ORPHAN_COMPONENT = "orphan_component"
    DEEP_DEPENDENCY_CHAIN = "deep_dependency_chain"
    HUB_ANTIPATTERN = "hub_antipattern"
    TIGHT_COUPLING_CLUSTER = "tight_coupling_cluster"
    STAR_TOPOLOGY = "star_topology"
    SINGLE_CONSUMER = "single_consumer"
    SINGLE_PRODUCER = "single_producer"


class AntiPatternSeverity(Enum):
    """Severity levels for anti-patterns"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class AntiPattern:
    """Detected anti-pattern"""
    pattern_type: AntiPatternType
    severity: AntiPatternSeverity
    affected_components: List[str]
    affected_edges: List[Tuple[str, str]] = field(default_factory=list)
    description: str = ""
    impact: str = ""
    recommendation: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    quality_attributes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_type': self.pattern_type.value,
            'severity': self.severity.value,
            'affected_components': self.affected_components,
            'affected_edges': [list(e) for e in self.affected_edges],
            'description': self.description,
            'impact': self.impact,
            'recommendation': self.recommendation,
            'metrics': self.metrics,
            'quality_attributes': self.quality_attributes
        }


@dataclass
class AntiPatternAnalysisResult:
    """Result of anti-pattern analysis"""
    patterns: List[AntiPattern]
    summary: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'patterns': [p.to_dict() for p in self.patterns],
            'summary': self.summary,
            'recommendations': self.recommendations
        }
    
    def get_by_type(self, pattern_type: AntiPatternType) -> List[AntiPattern]:
        return [p for p in self.patterns if p.pattern_type == pattern_type]
    
    def get_by_severity(self, severity: AntiPatternSeverity) -> List[AntiPattern]:
        return [p for p in self.patterns if p.severity == severity]


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_ANTIPATTERN_CONFIG = {
    # God Topic thresholds
    'god_topic_threshold': {
        'connections': 10,  # Total pub+sub connections
        'publishers': 5,    # Max publishers
        'subscribers': 8    # Max subscribers
    },
    
    # Bottleneck thresholds
    'bottleneck_threshold': {
        'topic_ratio': 3.0,      # Topics per broker vs average
        'app_ratio': 3.0,        # Apps connected vs average
        'betweenness_percentile': 90
    },
    
    # Chatty publisher threshold
    'chatty_publisher_threshold': 8,  # Max topics per publisher
    
    # Deep dependency chain threshold
    'deep_chain_threshold': 5,  # Max dependency chain length
    
    # Hub antipattern threshold
    'hub_threshold': {
        'connections': 10,  # Total connections
        'ratio': 3.0        # Ratio vs average degree
    },
    
    # Tight coupling cluster
    'tight_coupling_threshold': {
        'min_size': 4,      # Minimum cluster size
        'density': 0.8      # Edge density within cluster
    },
    
    # Orphan detection
    'orphan_include_types': ['Application', 'Topic'],
    
    # Star topology
    'star_center_ratio': 5.0  # Center degree vs periphery
}


# ============================================================================
# Anti-Pattern Detector
# ============================================================================

class AntiPatternDetector:
    """
    Detects architectural anti-patterns in pub-sub system graphs.
    
    Uses graph algorithms to identify patterns that may cause:
    - Reliability issues (SPOFs, cascades)
    - Maintainability issues (coupling, complexity)
    - Availability issues (bottlenecks, single points)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**DEFAULT_ANTIPATTERN_CONFIG, **(config or {})}
        self.logger = logging.getLogger('AntiPatternDetector')
    
    def analyze(self, graph: nx.DiGraph) -> AntiPatternAnalysisResult:
        """
        Detect all anti-patterns in the graph.
        
        Args:
            graph: NetworkX DiGraph with system components
            
        Returns:
            AntiPatternAnalysisResult with all detected patterns
        """
        self.logger.info("Starting anti-pattern detection...")
        
        patterns = []
        
        # Run all detectors
        patterns.extend(self._detect_god_topics(graph))
        patterns.extend(self._detect_spofs(graph))
        patterns.extend(self._detect_circular_dependencies(graph))
        patterns.extend(self._detect_bottleneck_brokers(graph))
        patterns.extend(self._detect_chatty_publishers(graph))
        patterns.extend(self._detect_orphan_components(graph))
        patterns.extend(self._detect_deep_chains(graph))
        patterns.extend(self._detect_hub_antipattern(graph))
        patterns.extend(self._detect_tight_coupling_clusters(graph))
        patterns.extend(self._detect_star_topology(graph))
        patterns.extend(self._detect_single_producer_consumer(graph))
        
        # Generate summary
        summary = self._generate_summary(patterns)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(patterns)
        
        self.logger.info(f"Anti-pattern detection complete. Found {len(patterns)} patterns.")
        
        return AntiPatternAnalysisResult(
            patterns=patterns,
            summary=summary,
            recommendations=recommendations
        )
    
    def _detect_god_topics(self, graph: nx.DiGraph) -> List[AntiPattern]:
        """Detect god topics with too many connections"""
        patterns = []
        thresholds = self.config['god_topic_threshold']
        
        for node in graph.nodes():
            if graph.nodes[node].get('type') != 'Topic':
                continue
            
            # Count publishers and subscribers
            publishers = set()
            subscribers = set()
            
            for pred in graph.predecessors(node):
                if graph.nodes[pred].get('type') == 'Application':
                    # Check edge type
                    edge_data = graph.get_edge_data(pred, node, {})
                    if 'publish' in str(edge_data).lower():
                        publishers.add(pred)
                    else:
                        publishers.add(pred)  # Default to publisher for predecessor
            
            for succ in graph.successors(node):
                if graph.nodes[succ].get('type') == 'Application':
                    subscribers.add(succ)
            
            # Also count from in/out degree as backup
            in_deg = graph.in_degree(node)
            out_deg = graph.out_degree(node)
            total = in_deg + out_deg
            
            # Detect god topic
            if total >= thresholds['connections']:
                severity = AntiPatternSeverity.CRITICAL if total >= thresholds['connections'] * 1.5 \
                          else AntiPatternSeverity.HIGH
                
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.GOD_TOPIC,
                    severity=severity,
                    affected_components=[node],
                    description=f"Topic '{node}' has {total} connections "
                               f"({in_deg} publishers, {out_deg} subscribers). "
                               f"Threshold: {thresholds['connections']}",
                    impact="High coupling point. Changes to this topic affect many components. "
                           "Single topic failure impacts multiple applications.",
                    recommendation="Split into multiple focused topics by domain or functionality. "
                                  "Consider topic hierarchies or namespacing.",
                    metrics={
                        'total_connections': total,
                        'in_degree': in_deg,
                        'out_degree': out_deg,
                        'publisher_count': len(publishers),
                        'subscriber_count': len(subscribers)
                    },
                    quality_attributes=['maintainability', 'reliability']
                ))
        
        return patterns
    
    def _detect_spofs(self, graph: nx.DiGraph) -> List[AntiPattern]:
        """Detect single points of failure using articulation points"""
        patterns = []
        
        try:
            undirected = graph.to_undirected()
            aps = set(nx.articulation_points(undirected))
        except:
            return patterns
        
        for ap in aps:
            node_type = graph.nodes[ap].get('type', 'Unknown')
            
            # Calculate impact
            temp_graph = graph.copy()
            temp_graph.remove_node(ap)
            
            original_components = nx.number_weakly_connected_components(graph)
            new_components = nx.number_weakly_connected_components(temp_graph)
            
            fragments = new_components - original_components + 1
            
            # More severe for infrastructure components
            if node_type in ['Broker', 'Node']:
                severity = AntiPatternSeverity.CRITICAL
            elif fragments >= 3:
                severity = AntiPatternSeverity.HIGH
            else:
                severity = AntiPatternSeverity.MEDIUM
            
            patterns.append(AntiPattern(
                pattern_type=AntiPatternType.SINGLE_POINT_OF_FAILURE,
                severity=severity,
                affected_components=[ap],
                description=f"{node_type} '{ap}' is an articulation point. "
                           f"Its removal fragments the system into {fragments} components.",
                impact=f"System partitioning on failure. {fragments-1} component groups "
                       f"become isolated.",
                recommendation=f"Add redundant {node_type.lower()}. Implement failover "
                              f"or load balancing.",
                metrics={
                    'fragments_on_removal': fragments,
                    'node_type': node_type
                },
                quality_attributes=['reliability', 'availability']
            ))
        
        return patterns
    
    def _detect_circular_dependencies(self, graph: nx.DiGraph) -> List[AntiPattern]:
        """Detect circular dependencies using SCCs"""
        patterns = []
        
        try:
            sccs = [c for c in nx.strongly_connected_components(graph) if len(c) > 1]
        except:
            return patterns
        
        for scc in sccs:
            components = sorted(scc)
            size = len(components)
            
            # Get edges within the cycle
            cycle_edges = []
            for u in scc:
                for v in graph.successors(u):
                    if v in scc:
                        cycle_edges.append((u, v))
            
            severity = AntiPatternSeverity.CRITICAL if size >= 4 \
                      else AntiPatternSeverity.HIGH if size >= 3 \
                      else AntiPatternSeverity.MEDIUM
            
            patterns.append(AntiPattern(
                pattern_type=AntiPatternType.CIRCULAR_DEPENDENCY,
                severity=severity,
                affected_components=components,
                affected_edges=cycle_edges,
                description=f"Circular dependency detected among {size} components: "
                           f"{', '.join(components[:5])}{'...' if size > 5 else ''}",
                impact="Deadlock risk. Difficult to test and modify. "
                       "Changes can trigger infinite loops.",
                recommendation="Break the cycle by introducing abstractions or "
                              "event-driven communication. Apply dependency inversion.",
                metrics={
                    'cycle_size': size,
                    'edge_count': len(cycle_edges)
                },
                quality_attributes=['maintainability', 'reliability']
            ))
        
        return patterns
    
    def _detect_bottleneck_brokers(self, graph: nx.DiGraph) -> List[AntiPattern]:
        """Detect broker bottlenecks"""
        patterns = []
        thresholds = self.config['bottleneck_threshold']
        
        brokers = [n for n in graph.nodes() if graph.nodes[n].get('type') == 'Broker']
        if not brokers:
            return patterns
        
        # Calculate average connections
        broker_degrees = {b: graph.degree(b) for b in brokers}
        avg_degree = sum(broker_degrees.values()) / len(brokers) if brokers else 0
        
        # Calculate betweenness
        try:
            betweenness = nx.betweenness_centrality(graph)
        except:
            betweenness = {}
        
        for broker in brokers:
            degree = broker_degrees[broker]
            bc = betweenness.get(broker, 0)
            
            # Check if bottleneck
            is_bottleneck = False
            reasons = []
            
            if avg_degree > 0 and degree / avg_degree >= thresholds['topic_ratio']:
                is_bottleneck = True
                reasons.append(f"Degree {degree} is {degree/avg_degree:.1f}x average")
            
            # Check betweenness percentile
            if betweenness:
                sorted_bc = sorted(betweenness.values())
                threshold_bc = sorted_bc[int(len(sorted_bc) * thresholds['betweenness_percentile'] / 100)] \
                              if len(sorted_bc) > 0 else 0
                if bc >= threshold_bc and bc > 0:
                    is_bottleneck = True
                    reasons.append(f"High betweenness centrality: {bc:.4f}")
            
            if is_bottleneck:
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.BOTTLENECK_BROKER,
                    severity=AntiPatternSeverity.HIGH,
                    affected_components=[broker],
                    description=f"Broker '{broker}' is a bottleneck. {'; '.join(reasons)}",
                    impact="Performance bottleneck. Single point of overload. "
                           "Scaling issues under load.",
                    recommendation="Distribute load across multiple brokers. "
                                  "Consider topic partitioning or sharding.",
                    metrics={
                        'degree': degree,
                        'avg_broker_degree': avg_degree,
                        'betweenness': bc,
                        'degree_ratio': degree / avg_degree if avg_degree > 0 else 0
                    },
                    quality_attributes=['availability', 'reliability']
                ))
        
        return patterns
    
    def _detect_chatty_publishers(self, graph: nx.DiGraph) -> List[AntiPattern]:
        """Detect applications publishing to too many topics"""
        patterns = []
        threshold = self.config['chatty_publisher_threshold']
        
        apps = [n for n in graph.nodes() if graph.nodes[n].get('type') == 'Application']
        
        for app in apps:
            # Count topics this app connects to
            topics = set()
            for neighbor in list(graph.successors(app)) + list(graph.predecessors(app)):
                if graph.nodes[neighbor].get('type') == 'Topic':
                    topics.add(neighbor)
            
            if len(topics) >= threshold:
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.CHATTY_PUBLISHER,
                    severity=AntiPatternSeverity.MEDIUM,
                    affected_components=[app] + list(topics)[:10],
                    description=f"Application '{app}' connects to {len(topics)} topics "
                               f"(threshold: {threshold})",
                    impact="High coupling. Difficult to reason about data flows. "
                           "Changes require broad testing.",
                    recommendation="Consider aggregating related topics. "
                                  "Split application by bounded context.",
                    metrics={
                        'topic_count': len(topics),
                        'threshold': threshold
                    },
                    quality_attributes=['maintainability']
                ))
        
        return patterns
    
    def _detect_orphan_components(self, graph: nx.DiGraph) -> List[AntiPattern]:
        """Detect disconnected components"""
        patterns = []
        include_types = self.config['orphan_include_types']
        
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'Unknown')
            if node_type not in include_types:
                continue
            
            # Check if orphan (no connections)
            in_deg = graph.in_degree(node)
            out_deg = graph.out_degree(node)
            
            if in_deg == 0 and out_deg == 0:
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.ORPHAN_COMPONENT,
                    severity=AntiPatternSeverity.LOW,
                    affected_components=[node],
                    description=f"{node_type} '{node}' has no connections.",
                    impact="Unused component. Potential configuration error or dead code.",
                    recommendation="Remove if unused, or connect to appropriate topics/brokers.",
                    metrics={
                        'node_type': node_type
                    },
                    quality_attributes=['maintainability']
                ))
        
        return patterns
    
    def _detect_deep_chains(self, graph: nx.DiGraph) -> List[AntiPattern]:
        """Detect excessively long dependency chains"""
        patterns = []
        threshold = self.config['deep_chain_threshold']
        
        # Find longest paths
        try:
            longest_path = nx.dag_longest_path(graph)
            path_length = len(longest_path)
        except nx.NetworkXUnfeasible:
            # Graph has cycles - find longest simple paths instead
            path_length = 0
            longest_path = []
            for source in graph.nodes():
                for target in graph.nodes():
                    if source != target:
                        try:
                            for path in nx.all_simple_paths(graph, source, target, cutoff=threshold+2):
                                if len(path) > path_length:
                                    path_length = len(path)
                                    longest_path = path
                        except nx.NetworkXNoPath:
                            continue
        
        if path_length > threshold:
            patterns.append(AntiPattern(
                pattern_type=AntiPatternType.DEEP_DEPENDENCY_CHAIN,
                severity=AntiPatternSeverity.MEDIUM if path_length <= threshold + 2 \
                        else AntiPatternSeverity.HIGH,
                affected_components=longest_path,
                description=f"Dependency chain of length {path_length} detected. "
                           f"Path: {' → '.join(longest_path[:6])}{'...' if path_length > 6 else ''}",
                impact="High latency. Cascade failure risk. "
                       "Difficult to trace and debug issues.",
                recommendation="Reduce dependency depth. Consider direct connections "
                              "or intermediate aggregation.",
                metrics={
                    'chain_length': path_length,
                    'threshold': threshold
                },
                quality_attributes=['reliability', 'maintainability']
            ))
        
        return patterns
    
    def _detect_hub_antipattern(self, graph: nx.DiGraph) -> List[AntiPattern]:
        """Detect hub nodes with disproportionately many connections"""
        patterns = []
        thresholds = self.config['hub_threshold']
        
        if graph.number_of_nodes() < 3:
            return patterns
        
        degrees = dict(graph.degree())
        avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
        
        for node, degree in degrees.items():
            if degree >= thresholds['connections'] and \
               (avg_degree == 0 or degree / avg_degree >= thresholds['ratio']):
                node_type = graph.nodes[node].get('type', 'Unknown')
                
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.HUB_ANTIPATTERN,
                    severity=AntiPatternSeverity.HIGH if degree >= thresholds['connections'] * 1.5 \
                            else AntiPatternSeverity.MEDIUM,
                    affected_components=[node],
                    description=f"{node_type} '{node}' is a hub with {degree} connections "
                               f"({degree/avg_degree:.1f}x average)",
                    impact="Central point of failure. Bottleneck for changes and traffic.",
                    recommendation="Distribute responsibilities. Apply microservices patterns. "
                                  "Consider event sourcing or CQRS.",
                    metrics={
                        'degree': degree,
                        'avg_degree': avg_degree,
                        'ratio': degree / avg_degree if avg_degree > 0 else 0
                    },
                    quality_attributes=['maintainability', 'reliability', 'availability']
                ))
        
        return patterns
    
    def _detect_tight_coupling_clusters(self, graph: nx.DiGraph) -> List[AntiPattern]:
        """Detect tightly coupled component clusters"""
        patterns = []
        thresholds = self.config['tight_coupling_threshold']
        
        try:
            undirected = graph.to_undirected()
            cliques = list(nx.find_cliques(undirected))
        except:
            return patterns
        
        for clique in cliques:
            if len(clique) >= thresholds['min_size']:
                # Calculate density
                subgraph = graph.subgraph(clique)
                possible_edges = len(clique) * (len(clique) - 1)
                actual_edges = subgraph.number_of_edges()
                density = actual_edges / possible_edges if possible_edges > 0 else 0
                
                if density >= thresholds['density']:
                    patterns.append(AntiPattern(
                        pattern_type=AntiPatternType.TIGHT_COUPLING_CLUSTER,
                        severity=AntiPatternSeverity.MEDIUM if len(clique) <= 5 \
                                else AntiPatternSeverity.HIGH,
                        affected_components=list(clique),
                        description=f"Tightly coupled cluster of {len(clique)} components: "
                                   f"{', '.join(sorted(clique)[:5])}{'...' if len(clique) > 5 else ''}",
                        impact="High coordination cost. Changes ripple through cluster. "
                               "Difficult to test in isolation.",
                        recommendation="Introduce abstractions. Apply facade pattern. "
                                      "Consider bounded contexts.",
                        metrics={
                            'cluster_size': len(clique),
                            'edge_density': density
                        },
                        quality_attributes=['maintainability']
                    ))
        
        return patterns
    
    def _detect_star_topology(self, graph: nx.DiGraph) -> List[AntiPattern]:
        """Detect star topology patterns (one center, many leaves)"""
        patterns = []
        ratio_threshold = self.config['star_center_ratio']
        
        # Find potential star centers
        degrees = dict(graph.degree())
        if not degrees:
            return patterns
        
        max_degree = max(degrees.values())
        avg_degree = sum(degrees.values()) / len(degrees)
        
        for node, degree in degrees.items():
            if degree == max_degree and degree >= 5:
                # Check if neighbors are mostly leaves
                neighbors = set(graph.predecessors(node)) | set(graph.successors(node))
                leaf_count = sum(1 for n in neighbors if degrees[n] <= 2)
                
                if leaf_count / len(neighbors) >= 0.7 and degree / avg_degree >= ratio_threshold:
                    node_type = graph.nodes[node].get('type', 'Unknown')
                    
                    patterns.append(AntiPattern(
                        pattern_type=AntiPatternType.STAR_TOPOLOGY,
                        severity=AntiPatternSeverity.MEDIUM,
                        affected_components=[node] + list(neighbors)[:10],
                        description=f"Star topology detected around {node_type} '{node}' "
                                   f"with {len(neighbors)} periphery nodes",
                        impact="Central point of failure. Scalability limitation.",
                        recommendation="Consider hierarchical organization or "
                                      "peer-to-peer communication for some flows.",
                        metrics={
                            'center_degree': degree,
                            'periphery_count': len(neighbors),
                            'leaf_ratio': leaf_count / len(neighbors)
                        },
                        quality_attributes=['availability', 'reliability']
                    ))
        
        return patterns
    
    def _detect_single_producer_consumer(self, graph: nx.DiGraph) -> List[AntiPattern]:
        """Detect topics with single producer or consumer"""
        patterns = []
        
        topics = [n for n in graph.nodes() if graph.nodes[n].get('type') == 'Topic']
        
        for topic in topics:
            in_deg = graph.in_degree(topic)
            out_deg = graph.out_degree(topic)
            
            if in_deg == 1:
                producer = list(graph.predecessors(topic))[0]
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.SINGLE_PRODUCER,
                    severity=AntiPatternSeverity.LOW,
                    affected_components=[topic, producer],
                    description=f"Topic '{topic}' has single producer: '{producer}'",
                    impact="Producer failure stops all data flow to topic.",
                    recommendation="Consider redundant producers or health monitoring.",
                    metrics={
                        'producer': producer,
                        'subscriber_count': out_deg
                    },
                    quality_attributes=['reliability']
                ))
            
            if out_deg == 1:
                consumer = list(graph.successors(topic))[0]
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.SINGLE_CONSUMER,
                    severity=AntiPatternSeverity.LOW,
                    affected_components=[topic, consumer],
                    description=f"Topic '{topic}' has single consumer: '{consumer}'",
                    impact="Consumer failure leads to message backlog or loss.",
                    recommendation="Consider consumer groups or competing consumers.",
                    metrics={
                        'consumer': consumer,
                        'publisher_count': in_deg
                    },
                    quality_attributes=['reliability', 'availability']
                ))
        
        return patterns
    
    def _generate_summary(self, patterns: List[AntiPattern]) -> Dict[str, Any]:
        """Generate analysis summary"""
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        by_qa = defaultdict(int)
        
        for pattern in patterns:
            by_type[pattern.pattern_type.value] += 1
            by_severity[pattern.severity.value] += 1
            for qa in pattern.quality_attributes:
                by_qa[qa] += 1
        
        return {
            'total_patterns': len(patterns),
            'by_type': dict(by_type),
            'by_severity': dict(by_severity),
            'by_quality_attribute': dict(by_qa),
            'critical_count': by_severity.get('critical', 0),
            'high_count': by_severity.get('high', 0)
        }
    
    def _generate_recommendations(self, patterns: List[AntiPattern]) -> List[str]:
        """Generate prioritized recommendations"""
        recommendations = []
        
        # Count by severity
        critical = [p for p in patterns if p.severity == AntiPatternSeverity.CRITICAL]
        high = [p for p in patterns if p.severity == AntiPatternSeverity.HIGH]
        
        if critical:
            recommendations.append(
                f"🔴 URGENT: {len(critical)} critical anti-patterns require immediate attention"
            )
        
        if high:
            recommendations.append(
                f"⚠️ HIGH: {len(high)} high-severity patterns should be addressed soon"
            )
        
        # Type-specific recommendations
        type_counts = defaultdict(int)
        for p in patterns:
            type_counts[p.pattern_type] += 1
        
        if type_counts[AntiPatternType.SINGLE_POINT_OF_FAILURE] > 0:
            recommendations.append(
                f"Add redundancy for {type_counts[AntiPatternType.SINGLE_POINT_OF_FAILURE]} "
                f"single points of failure"
            )
        
        if type_counts[AntiPatternType.CIRCULAR_DEPENDENCY] > 0:
            recommendations.append(
                f"Break {type_counts[AntiPatternType.CIRCULAR_DEPENDENCY]} circular dependencies"
            )
        
        if type_counts[AntiPatternType.GOD_TOPIC] > 0:
            recommendations.append(
                f"Split {type_counts[AntiPatternType.GOD_TOPIC]} god topics into focused topics"
            )
        
        if not recommendations:
            recommendations.append("✅ No significant anti-patterns detected")
        
        return recommendations


if __name__ == "__main__":
    # Test with sample graph
    G = nx.DiGraph()
    
    # Add nodes
    G.add_nodes_from([
        ('broker1', {'type': 'Broker'}),
        ('topic1', {'type': 'Topic'}),
        ('topic2', {'type': 'Topic'}),
        ('app1', {'type': 'Application'}),
        ('app2', {'type': 'Application'}),
        ('app3', {'type': 'Application'}),
        ('app4', {'type': 'Application'}),
        ('app5', {'type': 'Application'}),
        ('node1', {'type': 'Node'})
    ])
    
    # Create god topic
    G.add_edges_from([
        ('app1', 'topic1'), ('app2', 'topic1'), ('app3', 'topic1'),
        ('app4', 'topic1'), ('app5', 'topic1'),
        ('topic1', 'app2'), ('topic1', 'app3'), ('topic1', 'app4'),
        ('topic1', 'app5')
    ])
    
    # Create circular dependency
    G.add_edges_from([
        ('app1', 'app2'), ('app2', 'app3'), ('app3', 'app1')
    ])
    
    # Connect to broker
    G.add_edges_from([
        ('topic1', 'broker1'), ('topic2', 'broker1'),
        ('broker1', 'node1')
    ])
    
    detector = AntiPatternDetector()
    result = detector.analyze(G)
    
    print(f"\nAnti-Pattern Analysis Results")
    print("=" * 50)
    print(f"Total patterns detected: {result.summary['total_patterns']}")
    print(f"\nBy Severity:")
    for sev, count in result.summary['by_severity'].items():
        print(f"  {sev}: {count}")
    print(f"\nBy Type:")
    for ptype, count in result.summary['by_type'].items():
        print(f"  {ptype}: {count}")
    
    print(f"\nRecommendations:")
    for rec in result.recommendations:
        print(f"  • {rec}")
