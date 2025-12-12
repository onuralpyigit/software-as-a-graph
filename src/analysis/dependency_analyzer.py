#!/usr/bin/env python3
"""
Multi-Layer Dependency Graph Analyzer
=====================================

Analyzes pub-sub systems by building a DEPENDS_ON relationship graph
and applying graph algorithms directly to identify:
  - Critical nodes (through multiple algorithmic lenses)
  - Critical edges (dependency relationships)
  - Anti-patterns (structural issues)

This approach provides more interpretable results than a single
criticality score by showing WHY something is critical.

Layers:
  - Application Layer: Publishers, Subscribers, Processors
  - Topic Layer: Message channels
  - Broker Layer: Message routing infrastructure
  - Infrastructure Layer: Physical/virtual nodes

Dependency Types:
  - APP_DEPENDS_ON_TOPIC: Application publishes to or subscribes from topic
  - APP_DEPENDS_ON_BROKER: Application connects through broker
  - TOPIC_DEPENDS_ON_BROKER: Topic hosted on broker
  - BROKER_DEPENDS_ON_NODE: Broker runs on infrastructure node
  - APP_DEPENDS_ON_NODE: Application runs on infrastructure node
  - APP_DEPENDS_ON_APP: Transitive dependency through message flow
  - NODE_DEPENDS_ON_NODE: Infrastructure dependencies

Author: Software-as-a-Graph Research Project
"""

import logging
from typing import Dict, List, Set, Tuple, Any, Optional, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum, auto
import json
from pathlib import Path

try:
    import networkx as nx
    from networkx.algorithms import community
except ImportError:
    raise ImportError("NetworkX is required: pip install networkx")


# ============================================================================
# Enums and Data Classes
# ============================================================================

class Layer(Enum):
    """Graph layers in pub-sub system"""
    APPLICATION = "application"
    TOPIC = "topic"
    BROKER = "broker"
    INFRASTRUCTURE = "infrastructure"


class DependencyType(Enum):
    """Types of dependencies between components"""
    PUBLISHES_TO = "publishes_to"
    SUBSCRIBES_FROM = "subscribes_from"
    HOSTED_ON = "hosted_on"
    RUNS_ON = "runs_on"
    ROUTES_THROUGH = "routes_through"
    DEPENDS_ON = "depends_on"  # Generic/transitive
    DATA_FLOW = "data_flow"  # App-to-app through topics


class AntiPatternType(Enum):
    """Types of anti-patterns detected"""
    GOD_TOPIC = "god_topic"
    SINGLE_POINT_OF_FAILURE = "single_point_of_failure"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    BROKER_BOTTLENECK = "broker_bottleneck"
    HUB_OVERLOAD = "hub_overload"
    ORPHAN_COMPONENT = "orphan_component"
    LONG_DEPENDENCY_CHAIN = "long_dependency_chain"
    TIGHT_COUPLING_CLUSTER = "tight_coupling_cluster"
    MISSING_REDUNDANCY = "missing_redundancy"
    HIDDEN_DEPENDENCY = "hidden_dependency"
    CHATTY_COMMUNICATION = "chatty_communication"
    DISTRIBUTED_MONOLITH = "distributed_monolith"


class CriticalityReason(Enum):
    """Why a node/edge is considered critical"""
    HIGH_BETWEENNESS = "high_betweenness"
    ARTICULATION_POINT = "articulation_point"
    BRIDGE_EDGE = "bridge_edge"
    HIGH_PAGERANK = "high_pagerank"
    HIGH_HUB_SCORE = "high_hub_score"
    HIGH_AUTHORITY_SCORE = "high_authority_score"
    HIGH_IN_DEGREE = "high_in_degree"
    HIGH_OUT_DEGREE = "high_out_degree"
    INNER_CORE = "inner_core"
    CROSS_COMMUNITY_BRIDGE = "cross_community_bridge"
    HIGH_LOAD = "high_load"
    CRITICAL_PATH = "critical_path"


@dataclass
class CriticalNode:
    """A node identified as critical with reasons"""
    node_id: str
    layer: str
    node_type: str
    reasons: List[CriticalityReason]
    metrics: Dict[str, float]
    impact_description: str
    recommendation: str


@dataclass
class CriticalEdge:
    """An edge identified as critical with reasons"""
    source: str
    target: str
    dependency_type: str
    reasons: List[CriticalityReason]
    metrics: Dict[str, float]
    impact_description: str
    recommendation: str


@dataclass
class AntiPattern:
    """Detected anti-pattern"""
    pattern_type: AntiPatternType
    severity: str  # 'critical', 'high', 'medium', 'low'
    affected_components: List[str]
    description: str
    impact: str
    recommendation: str
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Complete analysis result"""
    # Graph metrics
    graph_metrics: Dict[str, Any]
    
    # Layer analysis
    layer_metrics: Dict[str, Dict[str, Any]]
    
    # Critical components
    critical_nodes: List[CriticalNode]
    critical_edges: List[CriticalEdge]
    
    # Anti-patterns
    anti_patterns: List[AntiPattern]
    
    # Raw algorithm results
    algorithm_results: Dict[str, Any]
    
    # Summary
    summary: Dict[str, Any]


# ============================================================================
# Multi-Layer Dependency Graph Builder
# ============================================================================

class DependencyGraphBuilder:
    """
    Builds a multi-layer dependency graph from pub-sub system definition.
    
    Creates explicit DEPENDS_ON relationships between all components,
    including transitive dependencies through message flows.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('dependency_builder')
        self.G = nx.DiGraph()
        self.layers = {layer: set() for layer in Layer}
    
    def build_from_pubsub_data(self, data: Dict[str, Any]) -> nx.DiGraph:
        """Build dependency graph from pub-sub system JSON data"""
        self.G = nx.DiGraph()
        self.layers = {layer: set() for layer in Layer}
        
        # Add infrastructure nodes
        for node in data.get('nodes', []):
            self._add_node(
                node['id'],
                Layer.INFRASTRUCTURE,
                node.get('type', 'node'),
                node
            )
        
        # Add brokers
        for broker in data.get('brokers', []):
            self._add_node(
                broker['id'],
                Layer.BROKER,
                broker.get('type', 'broker'),
                broker
            )
            # Broker depends on its host node
            if 'node' in broker:
                self._add_dependency(
                    broker['id'], broker['node'],
                    DependencyType.RUNS_ON,
                    weight=1.0
                )
        
        # Add topics
        for topic in data.get('topics', []):
            self._add_node(
                topic['id'],
                Layer.TOPIC,
                'topic',
                topic
            )
            # Topic depends on its broker
            if 'broker' in topic:
                self._add_dependency(
                    topic['id'], topic['broker'],
                    DependencyType.HOSTED_ON,
                    weight=1.0
                )
        
        # Add applications
        for app in data.get('applications', []):
            self._add_node(
                app['id'],
                Layer.APPLICATION,
                app.get('type', 'application'),
                app
            )
            # App depends on its host node
            if 'node' in app:
                self._add_dependency(
                    app['id'], app['node'],
                    DependencyType.RUNS_ON,
                    weight=1.0
                )
        
        # Add publish/subscribe relationships
        for pub in data.get('publishes', []):
            app_id = pub.get('application') or pub.get('publisher')
            topic_id = pub.get('topic')
            if app_id and topic_id:
                # Publisher depends on topic (needs it to publish)
                self._add_dependency(
                    app_id, topic_id,
                    DependencyType.PUBLISHES_TO,
                    weight=pub.get('rate', 1.0)
                )
        
        for sub in data.get('subscribes', []):
            app_id = sub.get('application') or sub.get('subscriber')
            topic_id = sub.get('topic')
            if app_id and topic_id:
                # Subscriber depends on topic (needs it for data)
                self._add_dependency(
                    app_id, topic_id,
                    DependencyType.SUBSCRIBES_FROM,
                    weight=sub.get('rate', 1.0)
                )
        
        # Build transitive app-to-app dependencies through topics
        self._build_transitive_dependencies()
        
        # Build broker routing dependencies
        for route in data.get('routes', []):
            src_broker = route.get('from') or route.get('source')
            dst_broker = route.get('to') or route.get('target')
            if src_broker and dst_broker:
                self._add_dependency(
                    src_broker, dst_broker,
                    DependencyType.ROUTES_THROUGH,
                    weight=1.0
                )
        
        self.logger.info(f"Built dependency graph: {len(self.G.nodes())} nodes, "
                        f"{len(self.G.edges())} edges")
        
        return self.G
    
    def _add_node(self, node_id: str, layer: Layer, node_type: str, 
                  attrs: Dict[str, Any]):
        """Add a node to the graph"""
        self.G.add_node(
            node_id,
            layer=layer.value,
            node_type=node_type,
            **{k: v for k, v in attrs.items() if k != 'id'}
        )
        self.layers[layer].add(node_id)
    
    def _add_dependency(self, source: str, target: str, 
                       dep_type: DependencyType, weight: float = 1.0):
        """Add a dependency edge"""
        if source in self.G.nodes() and target in self.G.nodes():
            self.G.add_edge(
                source, target,
                dependency_type=dep_type.value,
                weight=weight
            )
    
    def _build_transitive_dependencies(self):
        """Build app-to-app dependencies through shared topics"""
        # Find all publishers and subscribers for each topic
        topic_publishers = defaultdict(set)
        topic_subscribers = defaultdict(set)
        
        for u, v, data in self.G.edges(data=True):
            dep_type = data.get('dependency_type')
            if dep_type == DependencyType.PUBLISHES_TO.value:
                topic_publishers[v].add(u)
            elif dep_type == DependencyType.SUBSCRIBES_FROM.value:
                topic_subscribers[v].add(u)
        
        # Create data flow dependencies: subscriber depends on publisher
        for topic, publishers in topic_publishers.items():
            subscribers = topic_subscribers.get(topic, set())
            for subscriber in subscribers:
                for publisher in publishers:
                    if subscriber != publisher:
                        # Subscriber depends on publisher (needs their data)
                        self._add_dependency(
                            subscriber, publisher,
                            DependencyType.DATA_FLOW,
                            weight=0.5  # Lower weight for indirect dependency
                        )
    
    def get_layer_subgraph(self, layer: Layer) -> nx.DiGraph:
        """Get subgraph for a specific layer"""
        return self.G.subgraph(self.layers[layer]).copy()
    
    def get_cross_layer_edges(self) -> List[Tuple[str, str, Dict]]:
        """Get edges that cross layer boundaries"""
        cross_edges = []
        for u, v, data in self.G.edges(data=True):
            u_layer = self.G.nodes[u].get('layer')
            v_layer = self.G.nodes[v].get('layer')
            if u_layer != v_layer:
                cross_edges.append((u, v, data))
        return cross_edges


# ============================================================================
# Graph Algorithm Analyzer
# ============================================================================

class GraphAlgorithmAnalyzer:
    """
    Applies graph algorithms to identify critical components.
    
    Each algorithm provides a different perspective on criticality:
    - Betweenness: routing importance
    - PageRank: influence importance
    - HITS: hub/authority importance
    - Articulation Points: structural importance
    - etc.
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self.G_undirected = graph.to_undirected()
        self.logger = logging.getLogger('algorithm_analyzer')
        
        # Store results
        self.results = {}
        
        # Thresholds for "high" values (90th percentile by default)
        self.percentile_threshold = 90
    
    def run_all_algorithms(self) -> Dict[str, Any]:
        """Run all recommended algorithms and store results"""
        self.logger.info("Running comprehensive graph algorithm analysis...")
        
        # Centrality algorithms
        self.results['betweenness'] = self._analyze_betweenness()
        self.results['pagerank'] = self._analyze_pagerank()
        self.results['hits'] = self._analyze_hits()
        self.results['degree'] = self._analyze_degree()
        self.results['eigenvector'] = self._analyze_eigenvector()
        self.results['closeness'] = self._analyze_closeness()
        self.results['load'] = self._analyze_load()
        
        # Structural algorithms
        self.results['articulation_points'] = self._find_articulation_points()
        self.results['bridges'] = self._find_bridges()
        self.results['sccs'] = self._find_strongly_connected_components()
        self.results['kcore'] = self._analyze_kcore()
        
        # Community detection
        self.results['communities'] = self._detect_communities()
        
        # Path analysis
        self.results['paths'] = self._analyze_paths()
        
        self.logger.info("Algorithm analysis complete")
        return self.results
    
    def _analyze_betweenness(self) -> Dict[str, Any]:
        """Betweenness centrality analysis"""
        bc = nx.betweenness_centrality(self.G, normalized=True)
        edge_bc = nx.edge_betweenness_centrality(self.G, normalized=True)
        
        threshold = self._get_threshold(bc.values())
        high_bc_nodes = {n: v for n, v in bc.items() if v >= threshold}
        
        edge_threshold = self._get_threshold(edge_bc.values())
        high_bc_edges = {e: v for e, v in edge_bc.items() if v >= edge_threshold}
        
        return {
            'node_scores': bc,
            'edge_scores': edge_bc,
            'high_nodes': high_bc_nodes,
            'high_edges': high_bc_edges,
            'threshold': threshold,
            'interpretation': "High betweenness = critical routing point"
        }
    
    def _analyze_pagerank(self) -> Dict[str, Any]:
        """PageRank analysis"""
        pr = nx.pagerank(self.G, alpha=0.85)
        threshold = self._get_threshold(pr.values())
        high_pr = {n: v for n, v in pr.items() if v >= threshold}
        
        return {
            'scores': pr,
            'high_nodes': high_pr,
            'threshold': threshold,
            'interpretation': "High PageRank = receives from important sources"
        }
    
    def _analyze_hits(self) -> Dict[str, Any]:
        """HITS algorithm (hubs and authorities)"""
        try:
            hubs, authorities = nx.hits(self.G, max_iter=100)
        except:
            # Fallback for convergence issues
            hubs = {n: 0 for n in self.G.nodes()}
            authorities = {n: 0 for n in self.G.nodes()}
        
        hub_threshold = self._get_threshold(hubs.values())
        auth_threshold = self._get_threshold(authorities.values())
        
        return {
            'hub_scores': hubs,
            'authority_scores': authorities,
            'high_hubs': {n: v for n, v in hubs.items() if v >= hub_threshold},
            'high_authorities': {n: v for n, v in authorities.items() if v >= auth_threshold},
            'interpretation': "Hubs = key data sources, Authorities = key data sinks"
        }
    
    def _analyze_degree(self) -> Dict[str, Any]:
        """Degree centrality analysis"""
        in_deg = dict(self.G.in_degree())
        out_deg = dict(self.G.out_degree())
        total_deg = {n: in_deg.get(n, 0) + out_deg.get(n, 0) for n in self.G.nodes()}
        
        in_threshold = self._get_threshold(in_deg.values())
        out_threshold = self._get_threshold(out_deg.values())
        
        return {
            'in_degree': in_deg,
            'out_degree': out_deg,
            'total_degree': total_deg,
            'high_in_degree': {n: v for n, v in in_deg.items() if v >= in_threshold},
            'high_out_degree': {n: v for n, v in out_deg.items() if v >= out_threshold},
            'interpretation': "High in-degree = many depend on it, High out-degree = depends on many"
        }
    
    def _analyze_eigenvector(self) -> Dict[str, Any]:
        """Eigenvector centrality"""
        try:
            ec = nx.eigenvector_centrality(self.G, max_iter=1000)
        except:
            try:
                ec = nx.eigenvector_centrality_numpy(self.G)
            except:
                ec = {n: 0 for n in self.G.nodes()}
        
        threshold = self._get_threshold(ec.values())
        
        return {
            'scores': ec,
            'high_nodes': {n: v for n, v in ec.items() if v >= threshold},
            'interpretation': "High eigenvector = connected to other important nodes"
        }
    
    def _analyze_closeness(self) -> Dict[str, Any]:
        """Closeness centrality"""
        cc = nx.closeness_centrality(self.G)
        threshold = self._get_threshold(cc.values())
        
        return {
            'scores': cc,
            'high_nodes': {n: v for n, v in cc.items() if v >= threshold},
            'interpretation': "High closeness = can reach others quickly"
        }
    
    def _analyze_load(self) -> Dict[str, Any]:
        """Load centrality (traffic estimation)"""
        load = nx.load_centrality(self.G, normalized=True)
        threshold = self._get_threshold(load.values())
        
        return {
            'scores': load,
            'high_nodes': {n: v for n, v in load.items() if v >= threshold},
            'interpretation': "High load = handles more traffic"
        }
    
    def _find_articulation_points(self) -> Dict[str, Any]:
        """Find articulation points (cut vertices)"""
        aps = set(nx.articulation_points(self.G_undirected))
        
        # Analyze impact of each AP
        ap_analysis = {}
        for ap in aps:
            G_copy = self.G_undirected.copy()
            G_copy.remove_node(ap)
            num_components = nx.number_connected_components(G_copy)
            
            # Find which components would be disconnected
            components = list(nx.connected_components(G_copy))
            
            ap_analysis[ap] = {
                'components_after_removal': num_components,
                'component_sizes': [len(c) for c in components],
                'layer': self.G.nodes[ap].get('layer'),
                'type': self.G.nodes[ap].get('node_type')
            }
        
        return {
            'articulation_points': list(aps),
            'count': len(aps),
            'analysis': ap_analysis,
            'interpretation': "Articulation points = single points of failure"
        }
    
    def _find_bridges(self) -> Dict[str, Any]:
        """Find bridge edges"""
        bridges = list(nx.bridges(self.G_undirected))
        
        bridge_analysis = []
        for u, v in bridges:
            edge_data = self.G.get_edge_data(u, v) or self.G.get_edge_data(v, u) or {}
            bridge_analysis.append({
                'source': u,
                'target': v,
                'source_layer': self.G.nodes[u].get('layer'),
                'target_layer': self.G.nodes[v].get('layer'),
                'dependency_type': edge_data.get('dependency_type')
            })
        
        return {
            'bridges': bridges,
            'count': len(bridges),
            'analysis': bridge_analysis,
            'interpretation': "Bridges = critical edges whose removal disconnects graph"
        }
    
    def _find_strongly_connected_components(self) -> Dict[str, Any]:
        """Find SCCs (cyclic dependencies)"""
        sccs = list(nx.strongly_connected_components(self.G))
        
        # Non-trivial SCCs (size > 1) indicate cycles
        non_trivial = [scc for scc in sccs if len(scc) > 1]
        
        # Analyze cycles
        cycles = []
        for scc in non_trivial:
            subgraph = self.G.subgraph(scc)
            try:
                cycle = list(nx.find_cycle(subgraph))
                cycles.append({
                    'nodes': list(scc),
                    'cycle_edges': cycle,
                    'size': len(scc)
                })
            except nx.NetworkXNoCycle:
                cycles.append({
                    'nodes': list(scc),
                    'cycle_edges': [],
                    'size': len(scc)
                })
        
        return {
            'all_sccs': [list(scc) for scc in sccs],
            'non_trivial_sccs': [list(scc) for scc in non_trivial],
            'cycles': cycles,
            'has_cycles': len(non_trivial) > 0,
            'interpretation': "Non-trivial SCCs = circular dependencies (potential infinite loops)"
        }
    
    def _analyze_kcore(self) -> Dict[str, Any]:
        """K-core decomposition"""
        core_numbers = nx.core_number(self.G_undirected)
        max_k = max(core_numbers.values()) if core_numbers else 0
        
        # Group by core number
        by_core = defaultdict(list)
        for node, k in core_numbers.items():
            by_core[k].append(node)
        
        return {
            'core_numbers': core_numbers,
            'max_k': max_k,
            'by_core': dict(by_core),
            'innermost_core': by_core[max_k] if max_k in by_core else [],
            'interpretation': f"K-core: innermost core (k={max_k}) = most densely connected"
        }
    
    def _detect_communities(self) -> Dict[str, Any]:
        """Community detection for identifying subsystems"""
        try:
            communities = list(community.louvain_communities(self.G_undirected))
            modularity = nx.community.modularity(self.G_undirected, communities)
        except:
            communities = [set(self.G.nodes())]
            modularity = 0
        
        # Map nodes to communities
        node_to_community = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_community[node] = i
        
        # Find cross-community edges (potential bridges between subsystems)
        cross_community_edges = []
        for u, v in self.G.edges():
            if node_to_community.get(u) != node_to_community.get(v):
                cross_community_edges.append((u, v))
        
        return {
            'communities': [list(c) for c in communities],
            'num_communities': len(communities),
            'modularity': modularity,
            'node_to_community': node_to_community,
            'cross_community_edges': cross_community_edges,
            'interpretation': "Communities = logical subsystems; cross-community edges = integration points"
        }
    
    def _analyze_paths(self) -> Dict[str, Any]:
        """Path analysis for dependency chains"""
        # Calculate depths (longest path from each node)
        depths = {}
        for node in self.G.nodes():
            try:
                descendants = nx.descendants(self.G, node)
                if descendants:
                    max_depth = max(
                        nx.shortest_path_length(self.G, node, desc)
                        for desc in descendants
                    )
                else:
                    max_depth = 0
            except:
                max_depth = 0
            depths[node] = max_depth
        
        # Find longest paths
        max_depth = max(depths.values()) if depths else 0
        deepest_nodes = [n for n, d in depths.items() if d == max_depth]
        
        # Average path length
        try:
            avg_path = nx.average_shortest_path_length(self.G)
        except:
            avg_path = float('inf')
        
        return {
            'node_depths': depths,
            'max_depth': max_depth,
            'deepest_nodes': deepest_nodes,
            'average_path_length': avg_path,
            'interpretation': "High depth = long dependency chains (cascade risk)"
        }
    
    def _get_threshold(self, values) -> float:
        """Get threshold for 'high' values (percentile-based)"""
        values = list(values)
        if not values:
            return 0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * self.percentile_threshold / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]


# ============================================================================
# Anti-Pattern Detector
# ============================================================================

class AntiPatternDetector:
    """
    Detects architectural anti-patterns using graph structure analysis.
    """
    
    def __init__(self, graph: nx.DiGraph, algorithm_results: Dict[str, Any]):
        self.G = graph
        self.results = algorithm_results
        self.logger = logging.getLogger('antipattern_detector')
        
        # Configurable thresholds
        self.config = {
            'god_topic_threshold': 10,  # Connections to be considered "god"
            'hub_overload_threshold': 15,  # Out-degree threshold
            'long_chain_threshold': 5,  # Dependency depth threshold
            'tight_coupling_min_size': 4,  # Minimum clique size
            'chatty_threshold': 5,  # Topics between two apps
        }
    
    def detect_all(self) -> List[AntiPattern]:
        """Detect all anti-patterns"""
        self.logger.info("Detecting anti-patterns...")
        
        patterns = []
        
        patterns.extend(self._detect_god_topics())
        patterns.extend(self._detect_single_points_of_failure())
        patterns.extend(self._detect_circular_dependencies())
        patterns.extend(self._detect_broker_bottlenecks())
        patterns.extend(self._detect_hub_overload())
        patterns.extend(self._detect_orphan_components())
        patterns.extend(self._detect_long_dependency_chains())
        patterns.extend(self._detect_tight_coupling_clusters())
        patterns.extend(self._detect_missing_redundancy())
        patterns.extend(self._detect_chatty_communication())
        
        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        patterns.sort(key=lambda p: severity_order.get(p.severity, 99))
        
        self.logger.info(f"Detected {len(patterns)} anti-patterns")
        return patterns
    
    def _detect_god_topics(self) -> List[AntiPattern]:
        """Detect topics with too many connections (god topic anti-pattern)"""
        patterns = []
        threshold = self.config['god_topic_threshold']
        
        for node in self.G.nodes():
            if self.G.nodes[node].get('layer') != 'topic':
                continue
            
            in_deg = self.G.in_degree(node)
            out_deg = self.G.out_degree(node)
            total = in_deg + out_deg
            
            if total >= threshold:
                severity = 'critical' if total >= threshold * 2 else 'high'
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.GOD_TOPIC,
                    severity=severity,
                    affected_components=[node],
                    description=f"Topic '{node}' has {total} connections "
                               f"({in_deg} publishers, {out_deg} subscribers)",
                    impact="Single point of failure for message flow. "
                          "Changes affect many components. Hard to evolve.",
                    recommendation="Split into multiple focused topics. "
                                  "Consider topic hierarchy or partitioning.",
                    metrics={'total_connections': total, 'in_degree': in_deg, 
                            'out_degree': out_deg}
                ))
        
        return patterns
    
    def _detect_single_points_of_failure(self) -> List[AntiPattern]:
        """Detect articulation points as SPOFs"""
        patterns = []
        
        ap_data = self.results.get('articulation_points', {})
        for ap, analysis in ap_data.get('analysis', {}).items():
            num_components = analysis.get('components_after_removal', 1)
            
            if num_components > 1:
                severity = 'critical' if num_components > 2 else 'high'
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.SINGLE_POINT_OF_FAILURE,
                    severity=severity,
                    affected_components=[ap],
                    description=f"Component '{ap}' ({analysis.get('type')}) is a single point of failure. "
                               f"Removal creates {num_components} disconnected components.",
                    impact="System fragmentation if this component fails. "
                          "No redundant path exists.",
                    recommendation="Add redundancy. Create alternative paths. "
                                  "Consider active-passive or active-active setup.",
                    metrics={'components_after_removal': num_components,
                            'component_sizes': analysis.get('component_sizes', [])}
                ))
        
        return patterns
    
    def _detect_circular_dependencies(self) -> List[AntiPattern]:
        """Detect circular dependencies from SCCs"""
        patterns = []
        
        scc_data = self.results.get('sccs', {})
        for cycle_info in scc_data.get('cycles', []):
            if cycle_info['size'] > 1:
                nodes = cycle_info['nodes']
                severity = 'high' if len(nodes) > 3 else 'medium'
                
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.CIRCULAR_DEPENDENCY,
                    severity=severity,
                    affected_components=nodes,
                    description=f"Circular dependency detected among {len(nodes)} components: "
                               f"{', '.join(nodes[:5])}{'...' if len(nodes) > 5 else ''}",
                    impact="Potential for infinite message loops. "
                          "Complex failure cascades. Hard to reason about.",
                    recommendation="Break the cycle by introducing asynchronous communication, "
                                  "event sourcing, or redesigning dependencies.",
                    metrics={'cycle_size': len(nodes)}
                ))
        
        return patterns
    
    def _detect_broker_bottlenecks(self) -> List[AntiPattern]:
        """Detect brokers with disproportionately high traffic"""
        patterns = []
        
        bc_data = self.results.get('betweenness', {})
        for node, score in bc_data.get('high_nodes', {}).items():
            if self.G.nodes[node].get('layer') != 'broker':
                continue
            
            # Check if significantly higher than average
            all_scores = list(bc_data.get('node_scores', {}).values())
            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
            
            if score > avg_score * 3:  # 3x average
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.BROKER_BOTTLENECK,
                    severity='high',
                    affected_components=[node],
                    description=f"Broker '{node}' has betweenness centrality {score:.4f}, "
                               f"which is {score/avg_score:.1f}x the average",
                    impact="This broker handles disproportionate message routing. "
                          "Performance bottleneck and failure risk.",
                    recommendation="Distribute load across multiple brokers. "
                                  "Consider broker clustering or federation.",
                    metrics={'betweenness': score, 'average': avg_score}
                ))
        
        return patterns
    
    def _detect_hub_overload(self) -> List[AntiPattern]:
        """Detect applications with too many outgoing dependencies"""
        patterns = []
        threshold = self.config['hub_overload_threshold']
        
        degree_data = self.results.get('degree', {})
        for node, out_deg in degree_data.get('out_degree', {}).items():
            if self.G.nodes[node].get('layer') != 'application':
                continue
            
            if out_deg >= threshold:
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.HUB_OVERLOAD,
                    severity='medium',
                    affected_components=[node],
                    description=f"Application '{node}' has {out_deg} outgoing dependencies",
                    impact="This application depends on many components. "
                          "High coupling, hard to test, many failure modes.",
                    recommendation="Apply interface segregation. "
                                  "Consider breaking into smaller services.",
                    metrics={'out_degree': out_deg}
                ))
        
        return patterns
    
    def _detect_orphan_components(self) -> List[AntiPattern]:
        """Detect isolated or nearly isolated components"""
        patterns = []
        
        for node in self.G.nodes():
            in_deg = self.G.in_degree(node)
            out_deg = self.G.out_degree(node)
            
            # Completely isolated
            if in_deg == 0 and out_deg == 0:
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.ORPHAN_COMPONENT,
                    severity='low',
                    affected_components=[node],
                    description=f"Component '{node}' is completely isolated (no connections)",
                    impact="Unused component. May indicate incomplete configuration "
                          "or dead code.",
                    recommendation="Verify if component is needed. "
                                  "Connect it or remove it.",
                    metrics={'in_degree': 0, 'out_degree': 0}
                ))
            # Only outgoing (no one depends on it)
            elif in_deg == 0 and self.G.nodes[node].get('layer') == 'application':
                # Check if it's a pure publisher (acceptable) or truly orphan
                is_publisher = out_deg > 0
                if not is_publisher:
                    patterns.append(AntiPattern(
                        pattern_type=AntiPatternType.ORPHAN_COMPONENT,
                        severity='low',
                        affected_components=[node],
                        description=f"Application '{node}' has no incoming dependencies",
                        impact="No other component depends on this. "
                              "May be unused or entry point.",
                        recommendation="Verify if this is intentional (entry point) "
                                      "or should be connected.",
                        metrics={'in_degree': 0, 'out_degree': out_deg}
                    ))
        
        return patterns
    
    def _detect_long_dependency_chains(self) -> List[AntiPattern]:
        """Detect components with deep dependency chains"""
        patterns = []
        threshold = self.config['long_chain_threshold']
        
        path_data = self.results.get('paths', {})
        for node, depth in path_data.get('node_depths', {}).items():
            if depth >= threshold:
                severity = 'high' if depth >= threshold * 2 else 'medium'
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.LONG_DEPENDENCY_CHAIN,
                    severity=severity,
                    affected_components=[node],
                    description=f"Component '{node}' has dependency chain depth of {depth}",
                    impact="Failures cascade through long chain. "
                          "High latency. Hard to debug.",
                    recommendation="Reduce chain length. Introduce caching. "
                                  "Consider async processing.",
                    metrics={'depth': depth}
                ))
        
        return patterns
    
    def _detect_tight_coupling_clusters(self) -> List[AntiPattern]:
        """Detect tightly coupled component clusters (cliques)"""
        patterns = []
        min_size = self.config['tight_coupling_min_size']
        
        G_undirected = self.G.to_undirected()
        cliques = list(nx.find_cliques(G_undirected))
        
        for clique in cliques:
            if len(clique) >= min_size:
                # Check if all same layer (even worse)
                layers = set(self.G.nodes[n].get('layer') for n in clique)
                same_layer = len(layers) == 1
                
                severity = 'high' if same_layer else 'medium'
                
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.TIGHT_COUPLING_CLUSTER,
                    severity=severity,
                    affected_components=clique,
                    description=f"Tightly coupled cluster of {len(clique)} components: "
                               f"{', '.join(clique[:4])}{'...' if len(clique) > 4 else ''}",
                    impact="High coupling makes changes risky. "
                          "Testing requires all components.",
                    recommendation="Introduce interfaces. Apply dependency inversion. "
                                  "Consider event-driven decoupling.",
                    metrics={'cluster_size': len(clique), 'same_layer': same_layer}
                ))
        
        return patterns
    
    def _detect_missing_redundancy(self) -> List[AntiPattern]:
        """Detect components that should have redundancy but don't"""
        patterns = []
        
        # Bridges are critical edges without redundancy
        bridge_data = self.results.get('bridges', {})
        
        # Group bridges by their purpose
        critical_bridges = []
        for bridge_info in bridge_data.get('analysis', []):
            dep_type = bridge_info.get('dependency_type')
            
            # Infrastructure dependencies without redundancy are critical
            if dep_type in ['runs_on', 'hosted_on']:
                critical_bridges.append(bridge_info)
        
        if critical_bridges:
            patterns.append(AntiPattern(
                pattern_type=AntiPatternType.MISSING_REDUNDANCY,
                severity='high',
                affected_components=[b['source'] for b in critical_bridges],
                description=f"{len(critical_bridges)} infrastructure dependencies lack redundancy",
                impact="Single point of failure at infrastructure level. "
                      "Hardware failure causes service disruption.",
                recommendation="Add redundant infrastructure. "
                              "Implement failover mechanisms.",
                metrics={'critical_bridges': len(critical_bridges)}
            ))
        
        return patterns
    
    def _detect_chatty_communication(self) -> List[AntiPattern]:
        """Detect applications with too many direct topic connections"""
        patterns = []
        threshold = self.config['chatty_threshold']
        
        # Count topic connections between app pairs
        app_to_app_topics = defaultdict(set)
        
        for u, v, data in self.G.edges(data=True):
            u_layer = self.G.nodes[u].get('layer')
            v_layer = self.G.nodes[v].get('layer')
            
            # App -> Topic connections
            if u_layer == 'application' and v_layer == 'topic':
                # Find which apps subscribe to this topic
                for successor in self.G.successors(v):
                    if self.G.nodes[successor].get('layer') == 'application':
                        if u != successor:
                            app_to_app_topics[(u, successor)].add(v)
        
        # Find chatty pairs
        for (app1, app2), topics in app_to_app_topics.items():
            if len(topics) >= threshold:
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.CHATTY_COMMUNICATION,
                    severity='medium',
                    affected_components=[app1, app2],
                    description=f"Applications '{app1}' and '{app2}' communicate "
                               f"through {len(topics)} different topics",
                    impact="High coupling through multiple channels. "
                          "Consider consolidation.",
                    recommendation="Consolidate related topics. "
                                  "Consider aggregate events.",
                    metrics={'num_topics': len(topics), 'topics': list(topics)}
                ))
        
        return patterns


# ============================================================================
# Critical Component Identifier
# ============================================================================

class CriticalComponentIdentifier:
    """
    Identifies critical nodes and edges using multiple algorithm perspectives.
    """
    
    def __init__(self, graph: nx.DiGraph, algorithm_results: Dict[str, Any]):
        self.G = graph
        self.results = algorithm_results
        self.logger = logging.getLogger('critical_identifier')
    
    def identify_critical_nodes(self) -> List[CriticalNode]:
        """Identify critical nodes with explanations"""
        critical_nodes = []
        node_reasons = defaultdict(list)
        node_metrics = defaultdict(dict)
        
        # Collect reasons from each algorithm
        
        # Betweenness
        for node in self.results.get('betweenness', {}).get('high_nodes', {}):
            node_reasons[node].append(CriticalityReason.HIGH_BETWEENNESS)
            node_metrics[node]['betweenness'] = self.results['betweenness']['node_scores'][node]
        
        # PageRank
        for node in self.results.get('pagerank', {}).get('high_nodes', {}):
            node_reasons[node].append(CriticalityReason.HIGH_PAGERANK)
            node_metrics[node]['pagerank'] = self.results['pagerank']['scores'][node]
        
        # HITS Hubs
        for node in self.results.get('hits', {}).get('high_hubs', {}):
            node_reasons[node].append(CriticalityReason.HIGH_HUB_SCORE)
            node_metrics[node]['hub_score'] = self.results['hits']['hub_scores'][node]
        
        # HITS Authorities
        for node in self.results.get('hits', {}).get('high_authorities', {}):
            node_reasons[node].append(CriticalityReason.HIGH_AUTHORITY_SCORE)
            node_metrics[node]['authority_score'] = self.results['hits']['authority_scores'][node]
        
        # Articulation Points
        for node in self.results.get('articulation_points', {}).get('articulation_points', []):
            node_reasons[node].append(CriticalityReason.ARTICULATION_POINT)
            node_metrics[node]['is_articulation_point'] = True
        
        # High In-Degree
        for node in self.results.get('degree', {}).get('high_in_degree', {}):
            node_reasons[node].append(CriticalityReason.HIGH_IN_DEGREE)
            node_metrics[node]['in_degree'] = self.results['degree']['in_degree'][node]
        
        # High Out-Degree
        for node in self.results.get('degree', {}).get('high_out_degree', {}):
            node_reasons[node].append(CriticalityReason.HIGH_OUT_DEGREE)
            node_metrics[node]['out_degree'] = self.results['degree']['out_degree'][node]
        
        # Inner Core
        inner_core = self.results.get('kcore', {}).get('innermost_core', [])
        for node in inner_core:
            node_reasons[node].append(CriticalityReason.INNER_CORE)
            node_metrics[node]['core_number'] = self.results['kcore']['core_numbers'].get(node, 0)
        
        # Cross-community bridges
        communities = self.results.get('communities', {})
        cross_edges = communities.get('cross_community_edges', [])
        cross_nodes = set()
        for u, v in cross_edges:
            cross_nodes.add(u)
            cross_nodes.add(v)
        for node in cross_nodes:
            node_reasons[node].append(CriticalityReason.CROSS_COMMUNITY_BRIDGE)
        
        # High Load
        for node in self.results.get('load', {}).get('high_nodes', {}):
            node_reasons[node].append(CriticalityReason.HIGH_LOAD)
            node_metrics[node]['load'] = self.results['load']['scores'][node]
        
        # Create CriticalNode objects
        for node, reasons in node_reasons.items():
            if len(reasons) >= 1:  # At least one reason
                critical_nodes.append(CriticalNode(
                    node_id=node,
                    layer=self.G.nodes[node].get('layer', 'unknown'),
                    node_type=self.G.nodes[node].get('node_type', 'unknown'),
                    reasons=reasons,
                    metrics=node_metrics[node],
                    impact_description=self._generate_impact_description(node, reasons),
                    recommendation=self._generate_recommendation(node, reasons)
                ))
        
        # Sort by number of reasons (more reasons = more critical)
        critical_nodes.sort(key=lambda x: len(x.reasons), reverse=True)
        
        return critical_nodes
    
    def identify_critical_edges(self) -> List[CriticalEdge]:
        """Identify critical edges with explanations"""
        critical_edges = []
        edge_reasons = defaultdict(list)
        edge_metrics = defaultdict(dict)
        
        # High betweenness edges
        for edge, score in self.results.get('betweenness', {}).get('high_edges', {}).items():
            edge_reasons[edge].append(CriticalityReason.HIGH_BETWEENNESS)
            edge_metrics[edge]['betweenness'] = score
        
        # Bridges
        for bridge in self.results.get('bridges', {}).get('bridges', []):
            edge_reasons[tuple(bridge)].append(CriticalityReason.BRIDGE_EDGE)
            edge_metrics[tuple(bridge)]['is_bridge'] = True
        
        # Cross-community edges
        for edge in self.results.get('communities', {}).get('cross_community_edges', []):
            edge_reasons[tuple(edge)].append(CriticalityReason.CROSS_COMMUNITY_BRIDGE)
        
        # Create CriticalEdge objects
        for edge, reasons in edge_reasons.items():
            if len(reasons) >= 1:
                source, target = edge
                edge_data = self.G.get_edge_data(source, target) or {}
                
                critical_edges.append(CriticalEdge(
                    source=source,
                    target=target,
                    dependency_type=edge_data.get('dependency_type', 'unknown'),
                    reasons=reasons,
                    metrics=edge_metrics[edge],
                    impact_description=self._generate_edge_impact(edge, reasons),
                    recommendation=self._generate_edge_recommendation(edge, reasons)
                ))
        
        critical_edges.sort(key=lambda x: len(x.reasons), reverse=True)
        
        return critical_edges
    
    def _generate_impact_description(self, node: str, reasons: List[CriticalityReason]) -> str:
        """Generate human-readable impact description"""
        impacts = []
        
        if CriticalityReason.ARTICULATION_POINT in reasons:
            impacts.append("Removal disconnects the system")
        if CriticalityReason.HIGH_BETWEENNESS in reasons:
            impacts.append("Critical for message routing")
        if CriticalityReason.HIGH_PAGERANK in reasons:
            impacts.append("Influential in the network")
        if CriticalityReason.HIGH_HUB_SCORE in reasons:
            impacts.append("Key data source")
        if CriticalityReason.HIGH_AUTHORITY_SCORE in reasons:
            impacts.append("Key data sink")
        if CriticalityReason.HIGH_IN_DEGREE in reasons:
            impacts.append("Many components depend on it")
        if CriticalityReason.HIGH_OUT_DEGREE in reasons:
            impacts.append("Depends on many components")
        if CriticalityReason.INNER_CORE in reasons:
            impacts.append("Part of densely connected core")
        if CriticalityReason.CROSS_COMMUNITY_BRIDGE in reasons:
            impacts.append("Connects different subsystems")
        
        return "; ".join(impacts) if impacts else "Critical component"
    
    def _generate_recommendation(self, node: str, reasons: List[CriticalityReason]) -> str:
        """Generate actionable recommendation"""
        recs = []
        
        if CriticalityReason.ARTICULATION_POINT in reasons:
            recs.append("Add redundancy to prevent single point of failure")
        if CriticalityReason.HIGH_BETWEENNESS in reasons:
            recs.append("Consider load balancing or caching")
        if CriticalityReason.HIGH_IN_DEGREE in reasons:
            recs.append("Ensure high availability for this critical dependency")
        if CriticalityReason.HIGH_OUT_DEGREE in reasons:
            recs.append("Review coupling; consider interface segregation")
        
        return "; ".join(recs) if recs else "Monitor closely"
    
    def _generate_edge_impact(self, edge: Tuple[str, str], reasons: List[CriticalityReason]) -> str:
        """Generate edge impact description"""
        impacts = []
        
        if CriticalityReason.BRIDGE_EDGE in reasons:
            impacts.append("Removal disconnects parts of the system")
        if CriticalityReason.HIGH_BETWEENNESS in reasons:
            impacts.append("Critical for message flow")
        if CriticalityReason.CROSS_COMMUNITY_BRIDGE in reasons:
            impacts.append("Connects different subsystems")
        
        return "; ".join(impacts) if impacts else "Important dependency"
    
    def _generate_edge_recommendation(self, edge: Tuple[str, str], 
                                       reasons: List[CriticalityReason]) -> str:
        """Generate edge recommendation"""
        recs = []
        
        if CriticalityReason.BRIDGE_EDGE in reasons:
            recs.append("Add redundant connection path")
        if CriticalityReason.HIGH_BETWEENNESS in reasons:
            recs.append("Consider caching or replication")
        
        return "; ".join(recs) if recs else "Monitor this dependency"


# ============================================================================
# Main Analyzer
# ============================================================================

class MultiLayerDependencyAnalyzer:
    """
    Main analyzer that orchestrates all analysis components.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('dependency_analyzer')
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
    
    def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        Perform complete analysis of pub-sub system.
        
        Args:
            data: Pub-sub system definition (JSON-compatible dict)
        
        Returns:
            AnalysisResult with all findings
        """
        self.logger.info("Starting multi-layer dependency analysis...")
        
        # Build dependency graph
        builder = DependencyGraphBuilder()
        G = builder.build_from_pubsub_data(data)
        
        # Run graph algorithms
        algorithm_analyzer = GraphAlgorithmAnalyzer(G)
        algorithm_results = algorithm_analyzer.run_all_algorithms()
        
        # Identify critical components
        critical_identifier = CriticalComponentIdentifier(G, algorithm_results)
        critical_nodes = critical_identifier.identify_critical_nodes()
        critical_edges = critical_identifier.identify_critical_edges()
        
        # Detect anti-patterns
        pattern_detector = AntiPatternDetector(G, algorithm_results)
        anti_patterns = pattern_detector.detect_all()
        
        # Compute layer metrics
        layer_metrics = self._compute_layer_metrics(G, builder)
        
        # Graph-level metrics
        graph_metrics = {
            'nodes': len(G.nodes()),
            'edges': len(G.edges()),
            'density': nx.density(G),
            'is_connected': nx.is_weakly_connected(G),
            'num_components': nx.number_weakly_connected_components(G),
            'avg_clustering': nx.average_clustering(G.to_undirected()),
        }
        
        # Summary
        summary = {
            'total_critical_nodes': len(critical_nodes),
            'critical_by_layer': self._count_by_layer(critical_nodes),
            'total_critical_edges': len(critical_edges),
            'total_anti_patterns': len(anti_patterns),
            'anti_patterns_by_severity': self._count_by_severity(anti_patterns),
            'anti_patterns_by_type': self._count_by_type(anti_patterns),
            'top_critical_nodes': [n.node_id for n in critical_nodes[:5]],
            'most_severe_anti_patterns': [
                {'type': p.pattern_type.value, 'components': p.affected_components[:3]}
                for p in anti_patterns if p.severity in ['critical', 'high']
            ][:5]
        }
        
        self.logger.info(f"Analysis complete: {len(critical_nodes)} critical nodes, "
                        f"{len(critical_edges)} critical edges, "
                        f"{len(anti_patterns)} anti-patterns")
        
        return AnalysisResult(
            graph_metrics=graph_metrics,
            layer_metrics=layer_metrics,
            critical_nodes=critical_nodes,
            critical_edges=critical_edges,
            anti_patterns=anti_patterns,
            algorithm_results=algorithm_results,
            summary=summary
        )
    
    def _compute_layer_metrics(self, G: nx.DiGraph, 
                               builder: DependencyGraphBuilder) -> Dict[str, Dict]:
        """Compute per-layer metrics"""
        metrics = {}
        
        for layer in Layer:
            layer_nodes = [n for n in G.nodes() 
                         if G.nodes[n].get('layer') == layer.value]
            
            if not layer_nodes:
                continue
            
            subgraph = G.subgraph(layer_nodes)
            
            # Cross-layer edges
            incoming_cross = sum(1 for u, v in G.in_edges(layer_nodes)
                               if G.nodes[u].get('layer') != layer.value)
            outgoing_cross = sum(1 for u, v in G.out_edges(layer_nodes)
                               if G.nodes[v].get('layer') != layer.value)
            
            metrics[layer.value] = {
                'node_count': len(layer_nodes),
                'internal_edges': len(subgraph.edges()),
                'incoming_cross_layer': incoming_cross,
                'outgoing_cross_layer': outgoing_cross,
                'density': nx.density(subgraph) if len(layer_nodes) > 1 else 0,
            }
        
        return metrics
    
    def _count_by_layer(self, nodes: List[CriticalNode]) -> Dict[str, int]:
        """Count critical nodes by layer"""
        counts = defaultdict(int)
        for node in nodes:
            counts[node.layer] += 1
        return dict(counts)
    
    def _count_by_severity(self, patterns: List[AntiPattern]) -> Dict[str, int]:
        """Count anti-patterns by severity"""
        counts = defaultdict(int)
        for pattern in patterns:
            counts[pattern.severity] += 1
        return dict(counts)
    
    def _count_by_type(self, patterns: List[AntiPattern]) -> Dict[str, int]:
        """Count anti-patterns by type"""
        counts = defaultdict(int)
        for pattern in patterns:
            counts[pattern.pattern_type.value] += 1
        return dict(counts)
    
    def analyze_file(self, filepath: str) -> AnalysisResult:
        """Analyze from JSON file"""
        with open(filepath) as f:
            data = json.load(f)
        return self.analyze(data)
    
    def export_results(self, result: AnalysisResult, output_path: str):
        """Export results to JSON"""
        
        def serialize(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif hasattr(obj, '__dict__'):
                return {k: serialize(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {str(k): serialize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [serialize(i) for i in obj]
            elif isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, float):
                if obj == float('inf'):
                    return "inf"
                return round(obj, 6)
            return obj
        
        serialized = serialize(result)
        
        with open(output_path, 'w') as f:
            json.dump(serialized, f, indent=2)
        
        self.logger.info(f"Results exported to {output_path}")


# ============================================================================
# Report Generator
# ============================================================================

class ReportGenerator:
    """Generates human-readable reports from analysis results"""
    
    @staticmethod
    def generate_text_report(result: AnalysisResult) -> str:
        """Generate text report"""
        lines = []
        
        lines.append("=" * 70)
        lines.append("  MULTI-LAYER DEPENDENCY ANALYSIS REPORT")
        lines.append("=" * 70)
        
        # Graph Overview
        lines.append("\n📊 GRAPH OVERVIEW")
        lines.append("-" * 40)
        gm = result.graph_metrics
        lines.append(f"  Nodes: {gm['nodes']}")
        lines.append(f"  Edges: {gm['edges']}")
        lines.append(f"  Density: {gm['density']:.4f}")
        lines.append(f"  Connected: {gm['is_connected']}")
        lines.append(f"  Components: {gm['num_components']}")
        
        # Layer Breakdown
        lines.append("\n📑 LAYER BREAKDOWN")
        lines.append("-" * 40)
        for layer, metrics in result.layer_metrics.items():
            lines.append(f"  {layer.upper()}:")
            lines.append(f"    Nodes: {metrics['node_count']}")
            lines.append(f"    Internal edges: {metrics['internal_edges']}")
            lines.append(f"    Cross-layer in/out: {metrics['incoming_cross_layer']}/{metrics['outgoing_cross_layer']}")
        
        # Critical Nodes
        lines.append("\n🎯 CRITICAL NODES")
        lines.append("-" * 40)
        lines.append(f"  Total: {len(result.critical_nodes)}")
        
        for node in result.critical_nodes[:10]:
            reasons_str = ", ".join(r.value for r in node.reasons)
            lines.append(f"\n  • {node.node_id} ({node.layer}/{node.node_type})")
            lines.append(f"    Reasons: {reasons_str}")
            lines.append(f"    Impact: {node.impact_description}")
            lines.append(f"    Recommendation: {node.recommendation}")
        
        # Critical Edges
        lines.append("\n🔗 CRITICAL EDGES")
        lines.append("-" * 40)
        lines.append(f"  Total: {len(result.critical_edges)}")
        
        for edge in result.critical_edges[:10]:
            reasons_str = ", ".join(r.value for r in edge.reasons)
            lines.append(f"\n  • {edge.source} → {edge.target} ({edge.dependency_type})")
            lines.append(f"    Reasons: {reasons_str}")
            lines.append(f"    Impact: {edge.impact_description}")
        
        # Anti-Patterns
        lines.append("\n⚠️  ANTI-PATTERNS DETECTED")
        lines.append("-" * 40)
        lines.append(f"  Total: {len(result.anti_patterns)}")
        
        by_severity = result.summary['anti_patterns_by_severity']
        lines.append(f"  By severity: Critical={by_severity.get('critical', 0)}, "
                    f"High={by_severity.get('high', 0)}, "
                    f"Medium={by_severity.get('medium', 0)}, "
                    f"Low={by_severity.get('low', 0)}")
        
        for pattern in result.anti_patterns[:10]:
            lines.append(f"\n  [{pattern.severity.upper()}] {pattern.pattern_type.value}")
            lines.append(f"    Components: {', '.join(pattern.affected_components[:5])}")
            lines.append(f"    Description: {pattern.description}")
            lines.append(f"    Impact: {pattern.impact}")
            lines.append(f"    Recommendation: {pattern.recommendation}")
        
        # Summary
        lines.append("\n" + "=" * 70)
        lines.append("  SUMMARY")
        lines.append("=" * 70)
        lines.append(f"  • {len(result.critical_nodes)} critical nodes identified")
        lines.append(f"  • {len(result.critical_edges)} critical edges identified")
        lines.append(f"  • {len(result.anti_patterns)} anti-patterns detected")
        
        if result.anti_patterns:
            critical_patterns = [p for p in result.anti_patterns if p.severity == 'critical']
            if critical_patterns:
                lines.append(f"\n  ⚠️  {len(critical_patterns)} CRITICAL anti-patterns require immediate attention!")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    @staticmethod
    def generate_html_report(result: AnalysisResult) -> str:
        """Generate HTML report"""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Dependency Analysis Report</title>
    <style>
        :root {
            --critical: #e74c3c;
            --high: #e67e22;
            --medium: #f1c40f;
            --low: #27ae60;
            --dark: #2c3e50;
            --light: #ecf0f1;
        }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            margin: 0;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { text-align: center; color: #3498db; }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h2 { color: #3498db; margin-top: 0; }
        .metric { display: inline-block; margin: 10px 20px 10px 0; }
        .metric-value { font-size: 24px; font-weight: bold; color: #3498db; }
        .metric-label { color: #999; }
        .severity-critical { color: var(--critical); }
        .severity-high { color: var(--high); }
        .severity-medium { color: var(--medium); }
        .severity-low { color: var(--low); }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }
        th { background: rgba(52, 152, 219, 0.2); }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin: 2px;
        }
        .badge-reason { background: rgba(52, 152, 219, 0.3); }
        .badge-critical { background: var(--critical); }
        .badge-high { background: var(--high); }
        .badge-medium { background: var(--medium); }
        .badge-low { background: var(--low); }
    </style>
</head>
<body>
<div class="container">
    <h1>🔍 Multi-Layer Dependency Analysis Report</h1>
"""
        
        # Summary Cards
        gm = result.graph_metrics
        html += f"""
    <div class="card">
        <h2>📊 Graph Overview</h2>
        <div class="metric">
            <div class="metric-value">{gm['nodes']}</div>
            <div class="metric-label">Nodes</div>
        </div>
        <div class="metric">
            <div class="metric-value">{gm['edges']}</div>
            <div class="metric-label">Edges</div>
        </div>
        <div class="metric">
            <div class="metric-value">{gm['density']:.4f}</div>
            <div class="metric-label">Density</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(result.critical_nodes)}</div>
            <div class="metric-label">Critical Nodes</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(result.anti_patterns)}</div>
            <div class="metric-label">Anti-Patterns</div>
        </div>
    </div>
"""
        
        # Critical Nodes Table
        html += """
    <div class="card">
        <h2>🎯 Critical Nodes</h2>
        <table>
            <tr><th>Node</th><th>Layer</th><th>Type</th><th>Reasons</th><th>Impact</th></tr>
"""
        for node in result.critical_nodes[:15]:
            reasons_html = "".join(f'<span class="badge badge-reason">{r.value}</span>' 
                                  for r in node.reasons)
            html += f"""
            <tr>
                <td><strong>{node.node_id}</strong></td>
                <td>{node.layer}</td>
                <td>{node.node_type}</td>
                <td>{reasons_html}</td>
                <td>{node.impact_description}</td>
            </tr>
"""
        html += "</table></div>"
        
        # Anti-Patterns
        html += """
    <div class="card">
        <h2>⚠️ Anti-Patterns Detected</h2>
        <table>
            <tr><th>Severity</th><th>Type</th><th>Components</th><th>Description</th><th>Recommendation</th></tr>
"""
        for pattern in result.anti_patterns[:15]:
            severity_class = f"severity-{pattern.severity}"
            components_str = ", ".join(pattern.affected_components[:3])
            if len(pattern.affected_components) > 3:
                components_str += f" (+{len(pattern.affected_components)-3} more)"
            html += f"""
            <tr>
                <td><span class="badge badge-{pattern.severity}">{pattern.severity.upper()}</span></td>
                <td>{pattern.pattern_type.value}</td>
                <td>{components_str}</td>
                <td>{pattern.description}</td>
                <td>{pattern.recommendation}</td>
            </tr>
"""
        html += "</table></div>"
        
        html += """
</div>
</body>
</html>"""
        
        return html


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Demo with sample data
    sample_data = {
        "nodes": [
            {"id": "N1", "type": "edge_server"},
            {"id": "N2", "type": "cloud_server"},
            {"id": "N3", "type": "cloud_server"}
        ],
        "brokers": [
            {"id": "B1", "type": "mqtt", "node": "N1"},
            {"id": "B2", "type": "kafka", "node": "N2"}
        ],
        "topics": [
            {"id": "T1", "name": "sensor/data", "broker": "B1"},
            {"id": "T2", "name": "processed/data", "broker": "B2"},
            {"id": "T3", "name": "alerts", "broker": "B2"},
            {"id": "T4", "name": "commands", "broker": "B1"}
        ],
        "applications": [
            {"id": "A1", "type": "sensor", "node": "N1"},
            {"id": "A2", "type": "sensor", "node": "N1"},
            {"id": "A3", "type": "processor", "node": "N2"},
            {"id": "A4", "type": "dashboard", "node": "N3"},
            {"id": "A5", "type": "controller", "node": "N2"}
        ],
        "publishes": [
            {"application": "A1", "topic": "T1"},
            {"application": "A2", "topic": "T1"},
            {"application": "A3", "topic": "T2"},
            {"application": "A3", "topic": "T3"},
            {"application": "A5", "topic": "T4"}
        ],
        "subscribes": [
            {"application": "A3", "topic": "T1"},
            {"application": "A4", "topic": "T2"},
            {"application": "A4", "topic": "T3"},
            {"application": "A1", "topic": "T4"},
            {"application": "A2", "topic": "T4"}
        ],
        "routes": [
            {"from": "B1", "to": "B2"}
        ]
    }
    
    # Run analysis
    analyzer = MultiLayerDependencyAnalyzer()
    result = analyzer.analyze(sample_data)
    
    # Generate reports
    print(ReportGenerator.generate_text_report(result))
    
    # Export
    Path("demo_output").mkdir(exist_ok=True)
    analyzer.export_results(result, "demo_output/dependency_analysis.json")
    
    with open("demo_output/dependency_analysis.html", "w") as f:
        f.write(ReportGenerator.generate_html_report(result))
    
    print("\n✅ Reports exported to demo_output/")
