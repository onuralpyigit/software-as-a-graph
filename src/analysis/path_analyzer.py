"""
Path Analyzer

Comprehensive path analysis for distributed pub-sub systems including:
- Dependency chain analysis (application-to-application dependencies)
- Message flow path tracing (publisher -> topic -> subscriber)
- Critical path identification (paths whose failure maximizes impact)
- Path redundancy analysis (alternative paths and single points of failure)
- Path-level QoS analysis (end-to-end QoS guarantees)
- Failure propagation path analysis (how failures cascade)

This module extends the structural analysis by providing in-depth
understanding of data flow and dependency paths through the system.

Author: Software-as-a-Graph Project
Date: 2025-12-08
"""

import networkx as nx
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging
import heapq


class PathCriticalityLevel(Enum):
    """Criticality levels for paths"""
    CRITICAL = "critical"      # Single point of failure, no alternatives
    HIGH = "high"              # Limited redundancy, high impact
    MEDIUM = "medium"          # Some redundancy exists
    LOW = "low"                # Multiple alternatives available


@dataclass
class PathInfo:
    """
    Represents a path through the system graph

    Attributes:
        nodes: List of node IDs in the path
        edges: List of edge types along the path
        length: Number of hops in the path
        path_type: Type of path (dependency, message_flow, etc.)
        weight: Cumulative path weight (for weighted paths)
    """
    nodes: List[str]
    edges: List[str] = field(default_factory=list)
    length: int = 0
    path_type: str = "generic"
    weight: float = 0.0

    def __post_init__(self):
        self.length = len(self.nodes) - 1 if len(self.nodes) > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'length': self.length,
            'path_type': self.path_type,
            'weight': round(self.weight, 4)
        }


@dataclass
class MessageFlowPath:
    """
    Represents a message flow path from publisher to subscriber

    Attributes:
        publisher: Publishing application
        topic: Topic being published/subscribed
        subscriber: Subscribing application
        brokers: Brokers involved in routing
        infrastructure_nodes: Physical nodes involved
        hop_count: Total number of hops
        qos_constraints: QoS policies along the path
        bottleneck_score: Score indicating potential bottlenecks [0,1]
    """
    publisher: str
    topic: str
    subscriber: str
    brokers: List[str] = field(default_factory=list)
    infrastructure_nodes: List[str] = field(default_factory=list)
    hop_count: int = 0
    qos_constraints: Dict[str, Any] = field(default_factory=dict)
    bottleneck_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'publisher': self.publisher,
            'topic': self.topic,
            'subscriber': self.subscriber,
            'brokers': self.brokers,
            'infrastructure_nodes': self.infrastructure_nodes,
            'hop_count': self.hop_count,
            'qos_constraints': self.qos_constraints,
            'bottleneck_score': round(self.bottleneck_score, 4)
        }


@dataclass
class DependencyChain:
    """
    Represents a chain of dependencies between applications

    Attributes:
        applications: List of applications in the chain
        connecting_topics: Topics that create each dependency
        chain_length: Number of dependencies in the chain
        criticality_level: Overall criticality of the chain
        single_points_of_failure: Components that would break the chain
        redundancy_score: Score indicating path redundancy [0,1]
    """
    applications: List[str]
    connecting_topics: List[List[str]] = field(default_factory=list)
    chain_length: int = 0
    criticality_level: PathCriticalityLevel = PathCriticalityLevel.MEDIUM
    single_points_of_failure: List[str] = field(default_factory=list)
    redundancy_score: float = 0.0

    def __post_init__(self):
        self.chain_length = len(self.applications) - 1 if len(self.applications) > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'applications': self.applications,
            'connecting_topics': self.connecting_topics,
            'chain_length': self.chain_length,
            'criticality_level': self.criticality_level.value,
            'single_points_of_failure': self.single_points_of_failure,
            'redundancy_score': round(self.redundancy_score, 4)
        }


@dataclass
class PathRedundancyInfo:
    """
    Information about path redundancy between two components

    Attributes:
        source: Source component
        target: Target component
        primary_path: Shortest/main path
        alternative_paths: Other available paths
        redundancy_level: Number of independent paths
        resilience_score: Score indicating resilience to failures [0,1]
        critical_nodes: Nodes that appear in ALL paths (single points of failure)
    """
    source: str
    target: str
    primary_path: List[str] = field(default_factory=list)
    alternative_paths: List[List[str]] = field(default_factory=list)
    redundancy_level: int = 0
    resilience_score: float = 0.0
    critical_nodes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'source': self.source,
            'target': self.target,
            'primary_path': self.primary_path,
            'alternative_paths': self.alternative_paths,
            'redundancy_level': self.redundancy_level,
            'resilience_score': round(self.resilience_score, 4),
            'critical_nodes': self.critical_nodes
        }


@dataclass
class FailurePropagationPath:
    """
    Represents how a failure propagates through the system

    Attributes:
        origin: The component where failure originates
        affected_components: Components affected in order of propagation
        propagation_depth: How many hops the failure propagates
        impact_radius: Total number of affected components
        cascade_probability: Estimated probability of cascade [0,1]
    """
    origin: str
    affected_components: List[str] = field(default_factory=list)
    propagation_depth: int = 0
    impact_radius: int = 0
    cascade_probability: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'origin': self.origin,
            'affected_components': self.affected_components,
            'propagation_depth': self.propagation_depth,
            'impact_radius': self.impact_radius,
            'cascade_probability': round(self.cascade_probability, 4)
        }


@dataclass
class PathAnalysisResult:
    """
    Comprehensive path analysis result
    """
    # Summary metrics
    total_paths_analyzed: int = 0
    avg_path_length: float = 0.0
    max_path_length: int = 0

    # Dependency chains
    dependency_chains: List[DependencyChain] = field(default_factory=list)
    longest_chain: Optional[DependencyChain] = None
    critical_chains: List[DependencyChain] = field(default_factory=list)

    # Message flow paths
    message_flows: List[MessageFlowPath] = field(default_factory=list)
    bottleneck_flows: List[MessageFlowPath] = field(default_factory=list)

    # Redundancy analysis
    redundancy_info: Dict[str, PathRedundancyInfo] = field(default_factory=dict)
    avg_redundancy_level: float = 0.0
    low_redundancy_pairs: List[Tuple[str, str]] = field(default_factory=list)

    # Failure propagation
    failure_propagations: Dict[str, FailurePropagationPath] = field(default_factory=dict)
    high_impact_origins: List[str] = field(default_factory=list)

    # Critical paths
    critical_paths: List[PathInfo] = field(default_factory=list)
    path_criticality_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'summary': {
                'total_paths_analyzed': self.total_paths_analyzed,
                'avg_path_length': round(self.avg_path_length, 2),
                'max_path_length': self.max_path_length,
                'avg_redundancy_level': round(self.avg_redundancy_level, 2),
                'critical_chains_count': len(self.critical_chains),
                'bottleneck_flows_count': len(self.bottleneck_flows),
                'high_impact_origins_count': len(self.high_impact_origins)
            },
            'dependency_chains': {
                'total': len(self.dependency_chains),
                'longest': self.longest_chain.to_dict() if self.longest_chain else None,
                'critical': [c.to_dict() for c in self.critical_chains[:10]],
                'all_chains': [c.to_dict() for c in self.dependency_chains[:50]]
            },
            'message_flows': {
                'total': len(self.message_flows),
                'bottleneck_flows': [f.to_dict() for f in self.bottleneck_flows[:10]],
                'sample_flows': [f.to_dict() for f in self.message_flows[:20]]
            },
            'redundancy': {
                'avg_redundancy_level': round(self.avg_redundancy_level, 2),
                'low_redundancy_pairs': self.low_redundancy_pairs[:20],
                'sample_info': {k: v.to_dict() for k, v in list(self.redundancy_info.items())[:20]}
            },
            'failure_propagation': {
                'high_impact_origins': self.high_impact_origins[:10],
                'propagation_details': {k: v.to_dict() for k, v in list(self.failure_propagations.items())[:20]}
            },
            'critical_paths': {
                'distribution': self.path_criticality_distribution,
                'paths': [p.to_dict() for p in self.critical_paths[:20]]
            }
        }


class PathAnalyzer:
    """
    Comprehensive path analyzer for distributed pub-sub systems

    Provides deep analysis of paths including:
    - Dependency chains between applications
    - Message flow tracing
    - Critical path identification
    - Path redundancy assessment
    - Failure propagation analysis
    """

    def __init__(self,
                 max_path_length: int = 10,
                 max_paths_per_pair: int = 5,
                 redundancy_threshold: int = 2):
        """
        Initialize the path analyzer

        Args:
            max_path_length: Maximum path length to consider
            max_paths_per_pair: Maximum alternative paths to find per source-target pair
            redundancy_threshold: Minimum paths needed for "adequate" redundancy
        """
        self.max_path_length = max_path_length
        self.max_paths_per_pair = max_paths_per_pair
        self.redundancy_threshold = redundancy_threshold
        self.logger = logging.getLogger(__name__)

    def analyze(self, graph: nx.DiGraph) -> PathAnalysisResult:
        """
        Perform comprehensive path analysis

        Args:
            graph: NetworkX directed graph

        Returns:
            PathAnalysisResult with complete path analysis
        """
        self.logger.info(f"Starting path analysis for graph with {len(graph)} nodes and {len(graph.edges())} edges")

        result = PathAnalysisResult()

        # Step 1: Analyze dependency chains
        self.logger.info("  Analyzing dependency chains...")
        result.dependency_chains = self._analyze_dependency_chains(graph)
        if result.dependency_chains:
            result.longest_chain = max(result.dependency_chains, key=lambda c: c.chain_length)
            result.critical_chains = [c for c in result.dependency_chains
                                      if c.criticality_level == PathCriticalityLevel.CRITICAL]

        # Step 2: Trace message flow paths
        self.logger.info("  Tracing message flow paths...")
        result.message_flows = self._trace_message_flows(graph)
        result.bottleneck_flows = [f for f in result.message_flows if f.bottleneck_score >= 0.6]

        # Step 3: Analyze path redundancy
        self.logger.info("  Analyzing path redundancy...")
        result.redundancy_info = self._analyze_path_redundancy(graph)
        if result.redundancy_info:
            result.avg_redundancy_level = sum(
                r.redundancy_level for r in result.redundancy_info.values()
            ) / len(result.redundancy_info)
            result.low_redundancy_pairs = [
                (r.source, r.target) for r in result.redundancy_info.values()
                if r.redundancy_level < self.redundancy_threshold
            ]

        # Step 4: Analyze failure propagation paths
        self.logger.info("  Analyzing failure propagation paths...")
        result.failure_propagations = self._analyze_failure_propagation(graph)
        if result.failure_propagations:
            sorted_origins = sorted(
                result.failure_propagations.items(),
                key=lambda x: x[1].impact_radius,
                reverse=True
            )
            result.high_impact_origins = [o[0] for o in sorted_origins[:10]]

        # Step 5: Identify critical paths
        self.logger.info("  Identifying critical paths...")
        result.critical_paths = self._identify_critical_paths(graph, result)

        # Compute summary statistics
        all_paths = []
        for chain in result.dependency_chains:
            all_paths.append(chain.chain_length)
        for flow in result.message_flows:
            all_paths.append(flow.hop_count)

        if all_paths:
            result.total_paths_analyzed = len(all_paths)
            result.avg_path_length = sum(all_paths) / len(all_paths)
            result.max_path_length = max(all_paths)

        # Compute criticality distribution
        result.path_criticality_distribution = {
            'critical': len([p for p in result.critical_paths if 'critical' in str(p.path_type).lower()]),
            'high': len(result.bottleneck_flows),
            'medium': len([c for c in result.dependency_chains if c.criticality_level == PathCriticalityLevel.MEDIUM]),
            'low': len([c for c in result.dependency_chains if c.criticality_level == PathCriticalityLevel.LOW])
        }

        self.logger.info(f"Path analysis complete. Found {len(result.dependency_chains)} dependency chains, "
                        f"{len(result.message_flows)} message flows")

        return result

    def _analyze_dependency_chains(self, graph: nx.DiGraph) -> List[DependencyChain]:
        """
        Analyze application-to-application dependency chains

        Finds chains of dependencies where App A depends on App B depends on App C, etc.
        """
        chains = []

        # Build application dependency graph from DEPENDS_ON edges
        app_dep_graph = nx.DiGraph()
        app_to_topics = defaultdict(list)  # Track which topics create each dependency

        for source, target, data in graph.edges(data=True):
            if data.get('type') == 'DEPENDS_ON':
                source_type = graph.nodes[source].get('type')
                target_type = graph.nodes[target].get('type')

                if source_type == 'Application' and target_type == 'Application':
                    app_dep_graph.add_edge(source, target)
                    topics = data.get('topics', [])
                    app_to_topics[(source, target)] = topics

        if len(app_dep_graph) == 0:
            return chains

        # Find all dependency chains using DFS
        visited_chains = set()

        for source in app_dep_graph.nodes():
            # Find all simple paths from this source
            for target in app_dep_graph.nodes():
                if source != target:
                    try:
                        paths = list(nx.all_simple_paths(
                            app_dep_graph, source, target,
                            cutoff=self.max_path_length
                        ))

                        for path in paths:
                            if len(path) >= 2:
                                # Create unique key to avoid duplicates
                                chain_key = tuple(path)
                                if chain_key not in visited_chains:
                                    visited_chains.add(chain_key)

                                    # Get connecting topics
                                    connecting = []
                                    for i in range(len(path) - 1):
                                        topics = app_to_topics.get((path[i], path[i+1]), [])
                                        connecting.append(topics)

                                    # Find single points of failure
                                    spofs = self._find_chain_spofs(app_dep_graph, path)

                                    # Calculate redundancy score
                                    redundancy = self._calculate_chain_redundancy(
                                        app_dep_graph, path[0], path[-1]
                                    )

                                    # Determine criticality
                                    if len(spofs) > 0 and redundancy < 0.3:
                                        criticality = PathCriticalityLevel.CRITICAL
                                    elif len(spofs) > 0 or redundancy < 0.5:
                                        criticality = PathCriticalityLevel.HIGH
                                    elif redundancy < 0.7:
                                        criticality = PathCriticalityLevel.MEDIUM
                                    else:
                                        criticality = PathCriticalityLevel.LOW

                                    chain = DependencyChain(
                                        applications=list(path),
                                        connecting_topics=connecting,
                                        criticality_level=criticality,
                                        single_points_of_failure=spofs,
                                        redundancy_score=redundancy
                                    )
                                    chains.append(chain)
                    except nx.NetworkXNoPath:
                        continue
                    except Exception as e:
                        self.logger.debug(f"Error finding paths: {e}")
                        continue

        # Sort by length (longest first) and criticality
        chains.sort(key=lambda c: (c.chain_length, c.criticality_level.value), reverse=True)
        return chains[:100]  # Limit to top 100 chains

    def _find_chain_spofs(self, graph: nx.DiGraph, path: List[str]) -> List[str]:
        """Find single points of failure in a dependency chain"""
        spofs = []

        if len(path) <= 2:
            return spofs

        source, target = path[0], path[-1]

        # Check each intermediate node
        for node in path[1:-1]:
            # Create graph without this node
            test_graph = graph.copy()
            test_graph.remove_node(node)

            # Check if path still exists
            try:
                if not nx.has_path(test_graph, source, target):
                    spofs.append(node)
            except:
                spofs.append(node)

        return spofs

    def _calculate_chain_redundancy(self, graph: nx.DiGraph, source: str, target: str) -> float:
        """Calculate redundancy score for a source-target pair"""
        try:
            # Count number of node-disjoint paths
            paths = list(nx.all_simple_paths(
                graph, source, target, cutoff=self.max_path_length
            ))

            if len(paths) == 0:
                return 0.0

            if len(paths) == 1:
                return 0.1

            # Check for node-disjoint paths
            disjoint_count = self._count_node_disjoint_paths(paths, source, target)

            # Normalize: 3+ disjoint paths = full redundancy
            return min(1.0, disjoint_count / 3.0)

        except:
            return 0.0

    def _count_node_disjoint_paths(self, paths: List[List[str]],
                                    source: str, target: str) -> int:
        """Count approximately node-disjoint paths"""
        if len(paths) <= 1:
            return len(paths)

        disjoint_paths = []
        used_nodes = {source, target}

        # Sort paths by length (prefer shorter)
        sorted_paths = sorted(paths, key=len)

        for path in sorted_paths:
            intermediate = set(path[1:-1])
            if not intermediate.intersection(used_nodes):
                disjoint_paths.append(path)
                used_nodes.update(intermediate)

        return max(1, len(disjoint_paths))

    def _trace_message_flows(self, graph: nx.DiGraph) -> List[MessageFlowPath]:
        """
        Trace message flow paths from publishers through topics to subscribers
        """
        flows = []

        # Find all topics
        topics = [n for n, d in graph.nodes(data=True) if d.get('type') == 'Topic']

        for topic in topics:
            # Find publishers (PUBLISHES_TO edges pointing to this topic)
            publishers = []
            for source, target, data in graph.in_edges(topic, data=True):
                if data.get('type') == 'PUBLISHES_TO':
                    publishers.append(source)

            # Find subscribers (SUBSCRIBES_TO edges pointing to this topic)
            subscribers = []
            for source, target, data in graph.in_edges(topic, data=True):
                if data.get('type') == 'SUBSCRIBES_TO':
                    subscribers.append(source)

            # Find brokers routing this topic
            brokers = []
            for source, target, data in graph.in_edges(topic, data=True):
                if data.get('type') == 'ROUTES':
                    brokers.append(source)

            # Get QoS constraints for this topic
            topic_data = graph.nodes[topic]
            qos_constraints = {}
            for key in ['reliability', 'durability', 'deadline_ms', 'transport_priority']:
                if key in topic_data:
                    qos_constraints[key] = topic_data[key]

            # Create flow for each publisher-subscriber pair
            for pub in publishers:
                for sub in subscribers:
                    if pub != sub:  # Skip if same application
                        # Find infrastructure nodes
                        infra_nodes = set()
                        for component in [pub, sub] + brokers:
                            for _, target, data in graph.out_edges(component, data=True):
                                if data.get('type') == 'RUNS_ON':
                                    infra_nodes.add(target)

                        # Calculate hop count
                        # Publisher -> Topic -> Subscriber = base 2 hops
                        # Add 1 for each broker in between
                        hop_count = 2 + len(brokers)

                        # Calculate bottleneck score based on centrality
                        bottleneck_score = self._calculate_flow_bottleneck_score(
                            graph, pub, topic, sub, brokers
                        )

                        flow = MessageFlowPath(
                            publisher=pub,
                            topic=topic,
                            subscriber=sub,
                            brokers=brokers,
                            infrastructure_nodes=list(infra_nodes),
                            hop_count=hop_count,
                            qos_constraints=qos_constraints,
                            bottleneck_score=bottleneck_score
                        )
                        flows.append(flow)

        # Sort by bottleneck score (highest first)
        flows.sort(key=lambda f: f.bottleneck_score, reverse=True)
        return flows

    def _calculate_flow_bottleneck_score(self, graph: nx.DiGraph,
                                          publisher: str, topic: str,
                                          subscriber: str, brokers: List[str]) -> float:
        """Calculate bottleneck score for a message flow"""
        try:
            # Get betweenness centrality for involved components
            betweenness = nx.betweenness_centrality(graph)

            # Score is weighted average of centralities
            scores = []

            # Topic centrality (most important for message routing)
            if topic in betweenness:
                scores.append(betweenness[topic] * 2)  # Double weight for topic

            # Broker centralities
            for broker in brokers:
                if broker in betweenness:
                    scores.append(betweenness[broker] * 1.5)  # 1.5x weight for brokers

            # Publisher/subscriber centrality
            for app in [publisher, subscriber]:
                if app in betweenness:
                    scores.append(betweenness[app])

            if scores:
                # Normalize to [0, 1]
                avg_score = sum(scores) / len(scores)
                return min(1.0, avg_score * 2)  # Scale up since betweenness is often small

            return 0.0

        except Exception as e:
            self.logger.debug(f"Error calculating bottleneck score: {e}")
            return 0.0

    def _analyze_path_redundancy(self, graph: nx.DiGraph) -> Dict[str, PathRedundancyInfo]:
        """
        Analyze path redundancy between important component pairs
        """
        redundancy_info = {}

        # Get applications (main actors)
        applications = [n for n, d in graph.nodes(data=True)
                       if d.get('type') == 'Application']

        # Analyze redundancy for application pairs that have dependencies
        dep_pairs = set()
        for source, target, data in graph.edges(data=True):
            if data.get('type') == 'DEPENDS_ON':
                source_type = graph.nodes[source].get('type')
                target_type = graph.nodes[target].get('type')
                if source_type == 'Application' and target_type == 'Application':
                    dep_pairs.add((source, target))

        # Use undirected graph for path analysis
        undirected = graph.to_undirected()

        for source, target in dep_pairs:
            try:
                # Find all simple paths
                all_paths = list(nx.all_simple_paths(
                    undirected, source, target,
                    cutoff=self.max_path_length
                ))

                if not all_paths:
                    continue

                # Sort by length
                all_paths.sort(key=len)
                primary_path = all_paths[0]
                alternative_paths = all_paths[1:self.max_paths_per_pair]

                # Count node-disjoint paths
                redundancy_level = self._count_node_disjoint_paths(all_paths, source, target)

                # Find critical nodes (appear in ALL paths)
                critical_nodes = []
                if len(all_paths) > 1:
                    common_nodes = set(all_paths[0])
                    for path in all_paths[1:]:
                        common_nodes &= set(path)
                    # Remove source and target
                    common_nodes -= {source, target}
                    critical_nodes = list(common_nodes)

                # Calculate resilience score
                resilience = self._calculate_resilience_score(
                    redundancy_level, len(critical_nodes), len(primary_path)
                )

                info = PathRedundancyInfo(
                    source=source,
                    target=target,
                    primary_path=primary_path,
                    alternative_paths=alternative_paths,
                    redundancy_level=redundancy_level,
                    resilience_score=resilience,
                    critical_nodes=critical_nodes
                )

                key = f"{source}->{target}"
                redundancy_info[key] = info

            except Exception as e:
                self.logger.debug(f"Error analyzing redundancy for {source}->{target}: {e}")
                continue

        return redundancy_info

    def _calculate_resilience_score(self, redundancy_level: int,
                                    critical_nodes: int, path_length: int) -> float:
        """Calculate resilience score based on redundancy metrics"""
        # Base score from redundancy level
        if redundancy_level >= 3:
            base_score = 1.0
        elif redundancy_level == 2:
            base_score = 0.7
        elif redundancy_level == 1:
            base_score = 0.3
        else:
            base_score = 0.0

        # Penalty for critical nodes (single points of failure)
        spof_penalty = min(0.5, critical_nodes * 0.15)

        # Slight penalty for long paths (more components to fail)
        length_penalty = min(0.2, (path_length - 2) * 0.03)

        return max(0.0, base_score - spof_penalty - length_penalty)

    def _analyze_failure_propagation(self, graph: nx.DiGraph) -> Dict[str, FailurePropagationPath]:
        """
        Analyze how failures propagate from each component
        """
        propagations = {}

        # Focus on applications and brokers (main actors)
        components = [n for n, d in graph.nodes(data=True)
                     if d.get('type') in ['Application', 'Broker']]

        for component in components:
            try:
                # Find all components that depend on this one (reverse dependency)
                affected = self._trace_failure_cascade(graph, component)

                if affected:
                    # Calculate propagation depth
                    depth = self._calculate_propagation_depth(graph, component, affected)

                    # Estimate cascade probability
                    cascade_prob = self._estimate_cascade_probability(
                        graph, component, len(affected)
                    )

                    propagation = FailurePropagationPath(
                        origin=component,
                        affected_components=affected,
                        propagation_depth=depth,
                        impact_radius=len(affected),
                        cascade_probability=cascade_prob
                    )
                    propagations[component] = propagation

            except Exception as e:
                self.logger.debug(f"Error analyzing failure propagation from {component}: {e}")
                continue

        return propagations

    def _trace_failure_cascade(self, graph: nx.DiGraph, origin: str) -> List[str]:
        """Trace which components would be affected by a failure"""
        affected = []

        # Build reverse dependency graph
        # If A DEPENDS_ON B, then B's failure affects A
        reverse_deps = nx.DiGraph()
        for source, target, data in graph.edges(data=True):
            if data.get('type') == 'DEPENDS_ON':
                reverse_deps.add_edge(target, source)  # Reverse direction

        # Also consider PUBLISHES_TO and SUBSCRIBES_TO through topics
        # If publisher fails, subscribers of its topics are affected
        for source, target, data in graph.edges(data=True):
            if data.get('type') == 'PUBLISHES_TO':
                # Find subscribers to this topic
                for sub_source, _, sub_data in graph.in_edges(target, data=True):
                    if sub_data.get('type') == 'SUBSCRIBES_TO':
                        reverse_deps.add_edge(source, sub_source)

        if origin not in reverse_deps:
            return affected

        # BFS to find all affected components
        try:
            affected = list(nx.descendants(reverse_deps, origin))
        except:
            pass

        return affected

    def _calculate_propagation_depth(self, graph: nx.DiGraph,
                                     origin: str, affected: List[str]) -> int:
        """Calculate maximum propagation depth"""
        if not affected:
            return 0

        # Build subgraph of affected components
        reverse_deps = nx.DiGraph()
        for source, target, data in graph.edges(data=True):
            if data.get('type') == 'DEPENDS_ON':
                if source in affected or source == origin:
                    if target in affected or target == origin:
                        reverse_deps.add_edge(target, source)

        # Find longest path from origin
        max_depth = 0
        for target in affected:
            try:
                if nx.has_path(reverse_deps, origin, target):
                    length = nx.shortest_path_length(reverse_deps, origin, target)
                    max_depth = max(max_depth, length)
            except:
                continue

        return max_depth

    def _estimate_cascade_probability(self, graph: nx.DiGraph,
                                      origin: str, impact_radius: int) -> float:
        """Estimate probability of cascade failure"""
        total_nodes = len(graph)
        if total_nodes == 0:
            return 0.0

        # Base probability on impact radius
        impact_ratio = impact_radius / total_nodes

        # Check if origin is articulation point (increases cascade probability)
        undirected = graph.to_undirected()
        is_ap = origin in nx.articulation_points(undirected)

        # Calculate probability
        base_prob = min(0.9, impact_ratio * 1.5)
        ap_bonus = 0.2 if is_ap else 0.0

        return min(1.0, base_prob + ap_bonus)

    def _identify_critical_paths(self, graph: nx.DiGraph,
                                  result: PathAnalysisResult) -> List[PathInfo]:
        """
        Identify the most critical paths in the system
        """
        critical_paths = []

        # Add critical dependency chains
        for chain in result.critical_chains[:10]:
            path = PathInfo(
                nodes=chain.applications,
                path_type="critical_dependency_chain",
                weight=1.0 - chain.redundancy_score
            )
            critical_paths.append(path)

        # Add bottleneck message flows
        for flow in result.bottleneck_flows[:10]:
            nodes = [flow.publisher, flow.topic] + flow.brokers + [flow.subscriber]
            path = PathInfo(
                nodes=nodes,
                path_type="bottleneck_message_flow",
                weight=flow.bottleneck_score
            )
            critical_paths.append(path)

        # Add paths with no redundancy
        for source_target, info in result.redundancy_info.items():
            if info.redundancy_level < self.redundancy_threshold:
                path = PathInfo(
                    nodes=info.primary_path,
                    path_type="low_redundancy_path",
                    weight=1.0 - info.resilience_score
                )
                critical_paths.append(path)

        # Sort by weight (criticality)
        critical_paths.sort(key=lambda p: p.weight, reverse=True)
        return critical_paths[:50]

    # =========================================================================
    # Specialized Analysis Methods
    # =========================================================================

    def find_shortest_path(self, graph: nx.DiGraph,
                          source: str, target: str) -> Optional[PathInfo]:
        """Find shortest path between two components"""
        try:
            path = nx.shortest_path(graph, source, target)

            # Get edge types
            edges = []
            for i in range(len(path) - 1):
                edge_data = graph.get_edge_data(path[i], path[i+1], {})
                edges.append(edge_data.get('type', 'Unknown'))

            return PathInfo(
                nodes=path,
                edges=edges,
                path_type="shortest_path"
            )
        except nx.NetworkXNoPath:
            return None
        except Exception as e:
            self.logger.error(f"Error finding shortest path: {e}")
            return None

    def find_all_paths(self, graph: nx.DiGraph,
                       source: str, target: str,
                       max_length: Optional[int] = None) -> List[PathInfo]:
        """Find all paths between two components"""
        cutoff = max_length or self.max_path_length
        paths = []

        try:
            for path in nx.all_simple_paths(graph, source, target, cutoff=cutoff):
                # Get edge types
                edges = []
                for i in range(len(path) - 1):
                    edge_data = graph.get_edge_data(path[i], path[i+1], {})
                    edges.append(edge_data.get('type', 'Unknown'))

                path_info = PathInfo(
                    nodes=path,
                    edges=edges,
                    path_type="all_paths"
                )
                paths.append(path_info)

                if len(paths) >= 100:  # Limit
                    break

        except Exception as e:
            self.logger.error(f"Error finding all paths: {e}")

        return paths

    def analyze_path_qos(self, graph: nx.DiGraph, path: List[str]) -> Dict[str, Any]:
        """
        Analyze QoS constraints along a path

        Returns the "weakest link" QoS for reliability, durability, etc.
        """
        qos_analysis = {
            'path': path,
            'topics_in_path': [],
            'reliability': 'reliable',  # Default strongest
            'durability': 'persistent',  # Default strongest
            'min_deadline_ms': None,
            'weakest_links': []
        }

        # Durability ordering (weakest to strongest)
        durability_order = ['volatile', 'transient_local', 'transient', 'persistent']

        current_durability_idx = len(durability_order) - 1  # Start with strongest
        current_reliability_is_reliable = True

        for node in path:
            node_data = graph.nodes.get(node, {})

            if node_data.get('type') == 'Topic':
                qos_analysis['topics_in_path'].append(node)

                # Check reliability
                reliability = node_data.get('reliability', 'reliable')
                if reliability == 'best_effort':
                    current_reliability_is_reliable = False
                    qos_analysis['weakest_links'].append({
                        'component': node,
                        'constraint': 'reliability',
                        'value': 'best_effort'
                    })

                # Check durability
                durability = node_data.get('durability', 'volatile')
                if durability in durability_order:
                    idx = durability_order.index(durability)
                    if idx < current_durability_idx:
                        current_durability_idx = idx
                        qos_analysis['weakest_links'].append({
                            'component': node,
                            'constraint': 'durability',
                            'value': durability
                        })

                # Check deadline
                deadline = node_data.get('deadline_ms')
                if deadline is not None:
                    if qos_analysis['min_deadline_ms'] is None:
                        qos_analysis['min_deadline_ms'] = deadline
                    else:
                        qos_analysis['min_deadline_ms'] = min(
                            qos_analysis['min_deadline_ms'], deadline
                        )

        qos_analysis['reliability'] = 'reliable' if current_reliability_is_reliable else 'best_effort'
        qos_analysis['durability'] = durability_order[current_durability_idx]

        return qos_analysis

    def get_path_statistics(self, result: PathAnalysisResult) -> Dict[str, Any]:
        """Generate summary statistics from path analysis result"""
        return {
            'dependency_chains': {
                'total': len(result.dependency_chains),
                'critical': len(result.critical_chains),
                'avg_length': result.avg_path_length,
                'max_length': result.max_path_length
            },
            'message_flows': {
                'total': len(result.message_flows),
                'bottlenecks': len(result.bottleneck_flows)
            },
            'redundancy': {
                'avg_level': result.avg_redundancy_level,
                'low_redundancy_pairs': len(result.low_redundancy_pairs)
            },
            'failure_propagation': {
                'analyzed_components': len(result.failure_propagations),
                'high_impact_origins': len(result.high_impact_origins)
            },
            'critical_paths': {
                'total': len(result.critical_paths),
                'distribution': result.path_criticality_distribution
            }
        }


# Example usage
if __name__ == '__main__':
    # Create example graph
    G = nx.DiGraph()

    # Add applications
    apps = ['App_A', 'App_B', 'App_C', 'App_D']
    for app in apps:
        G.add_node(app, type='Application')

    # Add topics
    topics = ['Topic_1', 'Topic_2']
    for topic in topics:
        G.add_node(topic, type='Topic', reliability='reliable', durability='volatile')

    # Add broker
    G.add_node('Broker_1', type='Broker')

    # Add edges
    G.add_edge('App_A', 'Topic_1', type='PUBLISHES_TO')
    G.add_edge('App_B', 'Topic_1', type='SUBSCRIBES_TO')
    G.add_edge('App_B', 'Topic_2', type='PUBLISHES_TO')
    G.add_edge('App_C', 'Topic_2', type='SUBSCRIBES_TO')
    G.add_edge('Broker_1', 'Topic_1', type='ROUTES')
    G.add_edge('Broker_1', 'Topic_2', type='ROUTES')

    # Add dependencies
    G.add_edge('App_B', 'App_A', type='DEPENDS_ON', topics=['Topic_1'])
    G.add_edge('App_C', 'App_B', type='DEPENDS_ON', topics=['Topic_2'])
    G.add_edge('App_D', 'App_C', type='DEPENDS_ON', topics=['Topic_2'])

    # Analyze
    analyzer = PathAnalyzer()
    result = analyzer.analyze(G)

    # Print results
    print("\n=== Path Analysis Results ===\n")
    print(f"Dependency Chains Found: {len(result.dependency_chains)}")
    print(f"Message Flows Found: {len(result.message_flows)}")
    print(f"Critical Chains: {len(result.critical_chains)}")
    print(f"Avg Path Length: {result.avg_path_length:.2f}")

    if result.longest_chain:
        print(f"\nLongest Chain: {' -> '.join(result.longest_chain.applications)}")
        print(f"  Length: {result.longest_chain.chain_length}")
        print(f"  Criticality: {result.longest_chain.criticality_level.value}")

    print("\n=== Top Message Flows ===\n")
    for i, flow in enumerate(result.message_flows[:3], 1):
        print(f"{i}. {flow.publisher} -> {flow.topic} -> {flow.subscriber}")
        print(f"   Bottleneck Score: {flow.bottleneck_score:.3f}")

    print("\n=== High Impact Failure Origins ===\n")
    for origin in result.high_impact_origins[:5]:
        prop = result.failure_propagations.get(origin)
        if prop:
            print(f"  {origin}: affects {prop.impact_radius} components")
