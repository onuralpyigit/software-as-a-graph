#!/usr/bin/env python3
"""
Multi-Layer Dependency Graph Analyzer
=====================================

Analyzes pub-sub systems by building DEPENDS_ON relationship graphs at
multiple layers (application, infrastructure) and applying graph algorithms
DIRECTLY to identify critical nodes, edges, and anti-patterns.

This approach provides interpretable results showing WHY something is critical
rather than relying on a single composite score.

Key Features:
  - Multi-layer graph construction (App, Topic, Broker, Node layers)
  - DEPENDS_ON derivation (app-to-app, node-to-node, cross-layer)
  - Direct algorithm application with clear interpretation
  - Anti-pattern detection from structural analysis
  - Critical node/edge identification with multiple perspectives

Layers:
  - Application Layer: Apps connected via DEPENDS_ON through topics
  - Infrastructure Layer: Nodes connected via hosted application dependencies
  - Cross-Layer: Application-to-infrastructure mappings

Anti-Patterns Detected:
  - God Topic: Topic with excessive pub/sub connections
  - Single Point of Failure: Articulation points in dependency graph
  - Circular Dependency: Cycles in app-to-app dependencies
  - Hidden Coupling: Indirect dependencies through shared infrastructure
  - Hub Overload: Components with excessive fan-out
  - Long Dependency Chain: Deep transitive dependency paths
  - Broker Bottleneck: Broker routing too many critical topics
  - Missing Redundancy: Bridge edges with no alternative paths

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
except ImportError:
    raise ImportError("NetworkX is required: pip install networkx")


# ============================================================================
# Enums
# ============================================================================

class Layer(Enum):
    """Graph layers in pub-sub system"""
    APPLICATION = "application"
    TOPIC = "topic"
    BROKER = "broker"
    INFRASTRUCTURE = "infrastructure"


class DependencyType(Enum):
    """Types of DEPENDS_ON relationships"""
    APP_TO_APP = "app_to_app"              # Subscriber depends on publisher
    APP_TO_TOPIC = "app_to_topic"          # App depends on topic
    APP_TO_BROKER = "app_to_broker"        # App depends on broker for routing
    TOPIC_TO_BROKER = "topic_to_broker"    # Topic hosted on broker
    BROKER_TO_NODE = "broker_to_node"      # Broker runs on node
    APP_TO_NODE = "app_to_node"            # App runs on node
    NODE_TO_NODE = "node_to_node"          # Infrastructure dependency


class AntiPatternType(Enum):
    """Types of anti-patterns detected"""
    GOD_TOPIC = "god_topic"
    SINGLE_POINT_OF_FAILURE = "single_point_of_failure"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    HIDDEN_COUPLING = "hidden_coupling"
    HUB_OVERLOAD = "hub_overload"
    LONG_DEPENDENCY_CHAIN = "long_dependency_chain"
    BROKER_BOTTLENECK = "broker_bottleneck"
    MISSING_REDUNDANCY = "missing_redundancy"
    ORPHAN_COMPONENT = "orphan_component"
    TIGHT_COUPLING_CLUSTER = "tight_coupling_cluster"


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
    CROSS_LAYER_BRIDGE = "cross_layer_bridge"
    MANY_DEPENDENTS = "many_dependents"
    DEEP_IN_CHAIN = "deep_in_chain"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DependsOnEdge:
    """A DEPENDS_ON relationship between components"""
    source: str
    target: str
    dep_type: DependencyType
    weight: float = 1.0
    via_topics: List[str] = field(default_factory=list)
    via_apps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'target': self.target,
            'type': self.dep_type.value,
            'weight': self.weight,
            'via_topics': self.via_topics,
            'via_apps': self.via_apps
        }


@dataclass
class CriticalNode:
    """A node identified as critical with algorithmic evidence"""
    node_id: str
    node_type: str
    layer: Layer
    reasons: List[CriticalityReason]
    
    # Algorithm scores (raw values)
    betweenness: float = 0.0
    pagerank: float = 0.0
    hub_score: float = 0.0
    authority_score: float = 0.0
    in_degree: int = 0
    out_degree: int = 0
    k_core: int = 0
    
    # Derived metrics
    is_articulation_point: bool = False
    dependent_count: int = 0  # How many components depend on this
    dependency_depth: int = 0  # How deep in dependency chain
    
    # Interpretation
    impact_description: str = ""
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'layer': self.layer.value,
            'reasons': [r.value for r in self.reasons],
            'metrics': {
                'betweenness': round(self.betweenness, 4),
                'pagerank': round(self.pagerank, 4),
                'hub_score': round(self.hub_score, 4),
                'authority_score': round(self.authority_score, 4),
                'in_degree': self.in_degree,
                'out_degree': self.out_degree,
                'k_core': self.k_core,
                'is_articulation_point': self.is_articulation_point,
                'dependent_count': self.dependent_count,
                'dependency_depth': self.dependency_depth
            },
            'impact': self.impact_description,
            'recommendation': self.recommendation
        }


@dataclass
class CriticalEdge:
    """An edge identified as critical with algorithmic evidence"""
    source: str
    target: str
    edge_type: str
    reasons: List[CriticalityReason]
    
    # Algorithm scores
    betweenness: float = 0.0
    is_bridge: bool = False
    
    # Context
    source_layer: Layer = Layer.APPLICATION
    target_layer: Layer = Layer.APPLICATION
    
    # Interpretation
    impact_description: str = ""
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'target': self.target,
            'edge_type': self.edge_type,
            'reasons': [r.value for r in self.reasons],
            'metrics': {
                'betweenness': round(self.betweenness, 4),
                'is_bridge': self.is_bridge
            },
            'source_layer': self.source_layer.value,
            'target_layer': self.target_layer.value,
            'impact': self.impact_description,
            'recommendation': self.recommendation
        }


@dataclass
class AntiPattern:
    """A detected anti-pattern in the system"""
    pattern_type: AntiPatternType
    severity: str  # 'critical', 'high', 'medium', 'low'
    affected_components: List[str]
    description: str
    impact: str
    recommendation: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.pattern_type.value,
            'severity': self.severity,
            'affected_components': self.affected_components,
            'description': self.description,
            'impact': self.impact,
            'recommendation': self.recommendation,
            'metrics': self.metrics
        }


@dataclass
class LayerAnalysisResult:
    """Results for a single layer's analysis"""
    layer: Layer
    node_count: int
    edge_count: int
    density: float
    
    # Algorithm results
    articulation_points: List[str]
    bridges: List[Tuple[str, str]]
    strongly_connected_components: int
    max_k_core: int
    
    # Critical components
    critical_nodes: List[CriticalNode]
    critical_edges: List[CriticalEdge]


@dataclass
class MultiLayerAnalysisResult:
    """Complete multi-layer analysis result"""
    # Summary
    total_nodes: int
    total_edges: int
    layers_analyzed: List[str]
    
    # Layer-specific results
    application_layer: Optional[LayerAnalysisResult]
    infrastructure_layer: Optional[LayerAnalysisResult]
    full_graph: Optional[LayerAnalysisResult]
    
    # Cross-layer analysis
    cross_layer_dependencies: List[DependsOnEdge]
    layer_coupling: Dict[str, float]
    
    # Critical components (aggregated across layers)
    critical_nodes: List[CriticalNode]
    critical_edges: List[CriticalEdge]
    
    # Anti-patterns
    anti_patterns: List[AntiPattern]
    
    # Algorithm raw results (for transparency)
    algorithm_results: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'summary': {
                'total_nodes': self.total_nodes,
                'total_edges': self.total_edges,
                'layers_analyzed': self.layers_analyzed
            },
            'critical_nodes': [n.to_dict() for n in self.critical_nodes],
            'critical_edges': [e.to_dict() for e in self.critical_edges],
            'anti_patterns': [p.to_dict() for p in self.anti_patterns],
            'layer_coupling': self.layer_coupling,
            'algorithm_results': self.algorithm_results
        }


# ============================================================================
# Multi-Layer Graph Builder
# ============================================================================

class MultiLayerGraphBuilder:
    """
    Builds multi-layer DEPENDS_ON graphs from pub-sub system data.
    
    Creates separate graphs for:
    - Full system graph (all nodes and edges)
    - Application layer graph (apps + app-to-app dependencies)
    - Infrastructure layer graph (nodes + node-to-node dependencies)
    """
    
    def __init__(self):
        self.logger = logging.getLogger('multi_layer_builder')
        
        # Full graph with all components
        self.full_graph = nx.DiGraph()
        
        # Layer-specific graphs
        self.app_graph = nx.DiGraph()        # Apps + Topics
        self.infra_graph = nx.DiGraph()      # Nodes + Brokers
        self.app_depends_graph = nx.DiGraph()  # App-to-app DEPENDS_ON only
        self.node_depends_graph = nx.DiGraph() # Node-to-node DEPENDS_ON only
        
        # Index structures
        self._app_to_node: Dict[str, str] = {}
        self._broker_to_node: Dict[str, str] = {}
        self._topic_to_broker: Dict[str, str] = {}
        self._app_publishes: Dict[str, Set[str]] = defaultdict(set)
        self._app_subscribes: Dict[str, Set[str]] = defaultdict(set)
        self._topic_publishers: Dict[str, Set[str]] = defaultdict(set)
        self._topic_subscribers: Dict[str, Set[str]] = defaultdict(set)
        
        # Derived edges
        self.depends_on_edges: List[DependsOnEdge] = []
    
    def build_from_dict(self, data: Dict[str, Any]) -> 'MultiLayerGraphBuilder':
        """Build graphs from dictionary data"""
        self._reset()
        
        # Add all nodes
        self._add_nodes(data)
        
        # Add base edges and build indexes
        self._add_base_edges(data)
        
        # Derive DEPENDS_ON relationships
        self._derive_dependencies()
        
        # Build layer-specific graphs
        self._build_layer_graphs()
        
        return self
    
    def build_from_file(self, filepath: str) -> 'MultiLayerGraphBuilder':
        """Build graphs from JSON file"""
        with open(filepath) as f:
            data = json.load(f)
        return self.build_from_dict(data)
    
    def _reset(self):
        """Reset all internal state"""
        self.full_graph = nx.DiGraph()
        self.app_graph = nx.DiGraph()
        self.infra_graph = nx.DiGraph()
        self.app_depends_graph = nx.DiGraph()
        self.node_depends_graph = nx.DiGraph()
        
        self._app_to_node.clear()
        self._broker_to_node.clear()
        self._topic_to_broker.clear()
        self._app_publishes.clear()
        self._app_subscribes.clear()
        self._topic_publishers.clear()
        self._topic_subscribers.clear()
        self.depends_on_edges.clear()
    
    def _add_nodes(self, data: Dict[str, Any]):
        """Add all nodes to full graph"""
        # Applications
        for app in data.get('applications', []):
            self.full_graph.add_node(
                app['id'],
                layer=Layer.APPLICATION.value,
                type='Application',
                **{k: v for k, v in app.items() if k not in ['id', 'type']}
            )
        
        # Topics
        for topic in data.get('topics', []):
            self.full_graph.add_node(
                topic['id'],
                layer=Layer.TOPIC.value,
                type='Topic',
                **{k: v for k, v in topic.items() if k not in ['id', 'type']}
            )
        
        # Brokers
        for broker in data.get('brokers', []):
            self.full_graph.add_node(
                broker['id'],
                layer=Layer.BROKER.value,
                type='Broker',
                **{k: v for k, v in broker.items() if k not in ['id', 'type']}
            )
        
        # Infrastructure nodes
        for node in data.get('nodes', []):
            self.full_graph.add_node(
                node['id'],
                layer=Layer.INFRASTRUCTURE.value,
                type='Node',
                **{k: v for k, v in node.items() if k not in ['id', 'type']}
            )
        
        self.logger.info(f"Added {self.full_graph.number_of_nodes()} nodes")
    
    def _add_base_edges(self, data: Dict[str, Any]):
        """Add base edges and build index structures"""
        edges = data.get('relationships', data.get('edges', {}))
        
        # PUBLISHES_TO
        for pub in edges.get('publishes_to', []):
            app_id = pub.get('from', pub.get('source'))
            topic_id = pub.get('to', pub.get('target'))
            if app_id and topic_id:
                self.full_graph.add_edge(app_id, topic_id, type='PUBLISHES_TO')
                self._app_publishes[app_id].add(topic_id)
                self._topic_publishers[topic_id].add(app_id)
        
        # SUBSCRIBES_TO
        for sub in edges.get('subscribes_to', []):
            app_id = sub.get('from', sub.get('source'))
            topic_id = sub.get('to', sub.get('target'))
            if app_id and topic_id:
                self.full_graph.add_edge(app_id, topic_id, type='SUBSCRIBES_TO')
                self._app_subscribes[app_id].add(topic_id)
                self._topic_subscribers[topic_id].add(app_id)
        
        # ROUTES
        for route in edges.get('routes', []):
            broker_id = route.get('from', route.get('source'))
            topic_id = route.get('to', route.get('target'))
            if broker_id and topic_id:
                self.full_graph.add_edge(broker_id, topic_id, type='ROUTES')
                self._topic_to_broker[topic_id] = broker_id
        
        # RUNS_ON
        for runs in edges.get('runs_on', []):
            comp_id = runs.get('from', runs.get('source'))
            node_id = runs.get('to', runs.get('target'))
            if comp_id and node_id:
                self.full_graph.add_edge(comp_id, node_id, type='RUNS_ON')
                
                # Index by component type
                if comp_id in [n for n, d in self.full_graph.nodes(data=True) 
                              if d.get('type') == 'Application']:
                    self._app_to_node[comp_id] = node_id
                elif comp_id in [n for n, d in self.full_graph.nodes(data=True) 
                                if d.get('type') == 'Broker']:
                    self._broker_to_node[comp_id] = node_id
        
        # CONNECTS_TO (infrastructure)
        for conn in edges.get('connects_to', []):
            src = conn.get('from', conn.get('source'))
            tgt = conn.get('to', conn.get('target'))
            if src and tgt:
                self.full_graph.add_edge(src, tgt, type='CONNECTS_TO')
        
        self.logger.info(f"Added {self.full_graph.number_of_edges()} base edges")
    
    def _derive_dependencies(self):
        """Derive all DEPENDS_ON relationships"""
        self.logger.info("Deriving DEPENDS_ON relationships...")
        
        # 1. APP_TO_APP: subscriber depends on publisher via shared topic
        self._derive_app_to_app()
        
        # 2. APP_TO_BROKER: app depends on broker for its topics
        self._derive_app_to_broker()
        
        # 3. NODE_TO_NODE: infrastructure dependencies through app dependencies
        self._derive_node_to_node()
        
        self.logger.info(f"Derived {len(self.depends_on_edges)} DEPENDS_ON edges")
        
        # Summarize by type
        type_counts = defaultdict(int)
        for edge in self.depends_on_edges:
            type_counts[edge.dep_type.value] += 1
        
        for dep_type, count in type_counts.items():
            self.logger.info(f"  {dep_type}: {count}")
    
    def _derive_app_to_app(self):
        """
        Derive APP_TO_APP: subscriber DEPENDS_ON publisher via shared topic.
        
        For each topic, all subscribers depend on all publishers.
        """
        for topic_id, subscribers in self._topic_subscribers.items():
            publishers = self._topic_publishers.get(topic_id, set())
            
            for subscriber in subscribers:
                for publisher in publishers:
                    if subscriber != publisher:
                        # Check if edge already exists (multiple shared topics)
                        existing = self._find_edge(subscriber, publisher, DependencyType.APP_TO_APP)
                        
                        if existing:
                            if topic_id not in existing.via_topics:
                                existing.via_topics.append(topic_id)
                                existing.weight = 1.0 + (len(existing.via_topics) - 1) * 0.2
                        else:
                            self.depends_on_edges.append(DependsOnEdge(
                                source=subscriber,
                                target=publisher,
                                dep_type=DependencyType.APP_TO_APP,
                                weight=1.0,
                                via_topics=[topic_id]
                            ))
    
    def _derive_app_to_broker(self):
        """
        Derive APP_TO_BROKER: app DEPENDS_ON broker that routes its topics.
        """
        for app_id in set(self._app_publishes.keys()) | set(self._app_subscribes.keys()):
            topics = self._app_publishes[app_id] | self._app_subscribes[app_id]
            
            brokers_used: Dict[str, List[str]] = defaultdict(list)
            for topic_id in topics:
                broker_id = self._topic_to_broker.get(topic_id)
                if broker_id:
                    brokers_used[broker_id].append(topic_id)
            
            for broker_id, broker_topics in brokers_used.items():
                self.depends_on_edges.append(DependsOnEdge(
                    source=app_id,
                    target=broker_id,
                    dep_type=DependencyType.APP_TO_BROKER,
                    weight=1.0 + (len(broker_topics) - 1) * 0.1,
                    via_topics=broker_topics
                ))
    
    def _derive_node_to_node(self):
        """
        Derive NODE_TO_NODE: if app on node A depends on app on node B,
        then node A depends on node B.
        """
        node_deps: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        
        for edge in self.depends_on_edges:
            if edge.dep_type == DependencyType.APP_TO_APP:
                source_node = self._app_to_node.get(edge.source)
                target_node = self._app_to_node.get(edge.target)
                
                if source_node and target_node and source_node != target_node:
                    node_deps[(source_node, target_node)].append(edge.source)
        
        for (source_node, target_node), apps in node_deps.items():
            self.depends_on_edges.append(DependsOnEdge(
                source=source_node,
                target=target_node,
                dep_type=DependencyType.NODE_TO_NODE,
                weight=1.0 + (len(set(apps)) - 1) * 0.15,
                via_apps=list(set(apps))
            ))
    
    def _find_edge(self, source: str, target: str, 
                   dep_type: DependencyType) -> Optional[DependsOnEdge]:
        """Find existing edge of given type"""
        for edge in self.depends_on_edges:
            if (edge.source == source and edge.target == target and 
                edge.dep_type == dep_type):
                return edge
        return None
    
    def _build_layer_graphs(self):
        """Build layer-specific graphs"""
        # Application layer: apps + app-to-app dependencies
        apps = [n for n, d in self.full_graph.nodes(data=True) 
               if d.get('type') == 'Application']
        
        for app in apps:
            self.app_depends_graph.add_node(app, **self.full_graph.nodes[app])
        
        for edge in self.depends_on_edges:
            if edge.dep_type == DependencyType.APP_TO_APP:
                self.app_depends_graph.add_edge(
                    edge.source, edge.target,
                    type='DEPENDS_ON',
                    dep_type=edge.dep_type.value,
                    weight=edge.weight,
                    via_topics=edge.via_topics
                )
        
        # Infrastructure layer: nodes + node-to-node dependencies
        nodes = [n for n, d in self.full_graph.nodes(data=True) 
                if d.get('type') == 'Node']
        
        for node in nodes:
            self.node_depends_graph.add_node(node, **self.full_graph.nodes[node])
        
        for edge in self.depends_on_edges:
            if edge.dep_type == DependencyType.NODE_TO_NODE:
                self.node_depends_graph.add_edge(
                    edge.source, edge.target,
                    type='DEPENDS_ON',
                    dep_type=edge.dep_type.value,
                    weight=edge.weight,
                    via_apps=edge.via_apps
                )
        
        self.logger.info(f"App dependency graph: {self.app_depends_graph.number_of_nodes()} nodes, "
                        f"{self.app_depends_graph.number_of_edges()} edges")
        self.logger.info(f"Node dependency graph: {self.node_depends_graph.number_of_nodes()} nodes, "
                        f"{self.node_depends_graph.number_of_edges()} edges")


# ============================================================================
# Algorithm Applicator
# ============================================================================

class DirectAlgorithmApplicator:
    """
    Applies graph algorithms directly to identify critical components.
    
    No composite score - returns raw algorithm results and interprets them.
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self.G_undirected = graph.to_undirected()
        self.logger = logging.getLogger('algorithm_applicator')
        
        # Algorithm results cache
        self._results: Dict[str, Any] = {}
    
    def apply_all_algorithms(self) -> Dict[str, Any]:
        """Apply all algorithms and return raw results"""
        self.logger.info(f"Applying algorithms to graph with "
                        f"{self.G.number_of_nodes()} nodes, "
                        f"{self.G.number_of_edges()} edges")
        
        results = {}
        
        # Centrality algorithms
        results['betweenness'] = self._compute_betweenness()
        results['pagerank'] = self._compute_pagerank()
        results['hits'] = self._compute_hits()
        results['degree'] = self._compute_degree()
        results['closeness'] = self._compute_closeness()
        
        # Structural algorithms
        results['articulation_points'] = self._find_articulation_points()
        results['bridges'] = self._find_bridges()
        results['k_core'] = self._compute_k_core()
        results['sccs'] = self._find_strongly_connected_components()
        
        # Dependency-specific
        results['dependency_depth'] = self._compute_dependency_depth()
        results['dependent_count'] = self._compute_dependent_counts()
        
        self._results = results
        return results
    
    def _compute_betweenness(self) -> Dict[str, float]:
        """Compute betweenness centrality"""
        if self.G.number_of_edges() == 0:
            return {n: 0.0 for n in self.G.nodes()}
        return nx.betweenness_centrality(self.G, normalized=True)
    
    def _compute_pagerank(self) -> Dict[str, float]:
        """Compute PageRank"""
        try:
            return nx.pagerank(self.G, alpha=0.85)
        except:
            return {n: 1.0/self.G.number_of_nodes() for n in self.G.nodes()}
    
    def _compute_hits(self) -> Dict[str, Dict[str, float]]:
        """Compute HITS hub and authority scores"""
        try:
            hubs, authorities = nx.hits(self.G, max_iter=100)
            return {'hubs': hubs, 'authorities': authorities}
        except:
            default = {n: 0.5 for n in self.G.nodes()}
            return {'hubs': default, 'authorities': default}
    
    def _compute_degree(self) -> Dict[str, Dict[str, int]]:
        """Compute in-degree and out-degree"""
        return {
            'in_degree': dict(self.G.in_degree()),
            'out_degree': dict(self.G.out_degree())
        }
    
    def _compute_closeness(self) -> Dict[str, float]:
        """Compute closeness centrality"""
        return nx.closeness_centrality(self.G)
    
    def _find_articulation_points(self) -> List[str]:
        """Find articulation points (cut vertices)"""
        if self.G_undirected.number_of_edges() == 0:
            return []
        return list(nx.articulation_points(self.G_undirected))
    
    def _find_bridges(self) -> List[Tuple[str, str]]:
        """Find bridge edges (cut edges)"""
        if self.G_undirected.number_of_edges() == 0:
            return []
        return list(nx.bridges(self.G_undirected))
    
    def _compute_k_core(self) -> Dict[str, int]:
        """Compute k-core numbers"""
        if self.G_undirected.number_of_edges() == 0:
            return {n: 0 for n in self.G.nodes()}
        return nx.core_number(self.G_undirected)
    
    def _find_strongly_connected_components(self) -> List[Set[str]]:
        """Find strongly connected components"""
        return [set(c) for c in nx.strongly_connected_components(self.G)]
    
    def _compute_dependency_depth(self) -> Dict[str, int]:
        """Compute dependency depth (longest path from any root)"""
        depths = {n: 0 for n in self.G.nodes()}
        
        # Find roots (nodes with no incoming edges)
        roots = [n for n in self.G.nodes() if self.G.in_degree(n) == 0]
        
        if not roots:
            return depths
        
        # BFS from each root
        for root in roots:
            try:
                lengths = nx.single_source_shortest_path_length(self.G, root)
                for node, depth in lengths.items():
                    depths[node] = max(depths[node], depth)
            except:
                pass
        
        return depths
    
    def _compute_dependent_counts(self) -> Dict[str, int]:
        """Count how many nodes transitively depend on each node"""
        counts = {}
        for node in self.G.nodes():
            # Nodes that can reach this node (depend on it)
            ancestors = nx.ancestors(self.G, node)
            counts[node] = len(ancestors)
        return counts
    
    def get_edge_betweenness(self) -> Dict[Tuple[str, str], float]:
        """Compute edge betweenness centrality"""
        if self.G.number_of_edges() == 0:
            return {}
        return nx.edge_betweenness_centrality(self.G, normalized=True)


# ============================================================================
# Critical Component Identifier
# ============================================================================

class CriticalComponentIdentifier:
    """
    Identifies critical nodes and edges based on algorithm results.
    
    Uses thresholds and multiple indicators to determine criticality.
    """
    
    def __init__(self, graph: nx.DiGraph, algorithm_results: Dict[str, Any]):
        self.G = graph
        self.results = algorithm_results
        self.logger = logging.getLogger('critical_identifier')
        
        # Configurable thresholds (percentiles)
        self.thresholds = {
            'betweenness': 0.75,      # Top 25% by betweenness
            'pagerank': 0.75,         # Top 25% by pagerank
            'hub': 0.75,              # Top 25% by hub score
            'authority': 0.75,        # Top 25% by authority score
            'in_degree': 0.80,        # Top 20% by in-degree
            'out_degree': 0.80,       # Top 20% by out-degree
            'k_core': 0.70,           # Top 30% by k-core
            'dependent_count': 0.75,  # Top 25% by dependent count
        }
    
    def identify_critical_nodes(self) -> List[CriticalNode]:
        """Identify critical nodes with reasons"""
        self.logger.info("Identifying critical nodes...")
        
        critical_nodes = []
        
        # Get thresholds
        bc_thresh = self._get_threshold('betweenness', self.thresholds['betweenness'])
        pr_thresh = self._get_threshold('pagerank', self.thresholds['pagerank'])
        hub_thresh = self._get_threshold_nested('hits', 'hubs', self.thresholds['hub'])
        auth_thresh = self._get_threshold_nested('hits', 'authorities', self.thresholds['authority'])
        in_thresh = self._get_threshold_nested('degree', 'in_degree', self.thresholds['in_degree'])
        out_thresh = self._get_threshold_nested('degree', 'out_degree', self.thresholds['out_degree'])
        kcore_thresh = self._get_threshold('k_core', self.thresholds['k_core'])
        dep_thresh = self._get_threshold('dependent_count', self.thresholds['dependent_count'])
        
        # Articulation points
        articulation_points = set(self.results.get('articulation_points', []))
        
        for node in self.G.nodes():
            reasons = []
            
            # Check each algorithm
            bc = self.results.get('betweenness', {}).get(node, 0)
            pr = self.results.get('pagerank', {}).get(node, 0)
            hub = self.results.get('hits', {}).get('hubs', {}).get(node, 0)
            auth = self.results.get('hits', {}).get('authorities', {}).get(node, 0)
            in_deg = self.results.get('degree', {}).get('in_degree', {}).get(node, 0)
            out_deg = self.results.get('degree', {}).get('out_degree', {}).get(node, 0)
            kcore = self.results.get('k_core', {}).get(node, 0)
            dep_count = self.results.get('dependent_count', {}).get(node, 0)
            dep_depth = self.results.get('dependency_depth', {}).get(node, 0)
            is_ap = node in articulation_points
            
            # Determine reasons
            if is_ap:
                reasons.append(CriticalityReason.ARTICULATION_POINT)
            if bc >= bc_thresh and bc > 0:
                reasons.append(CriticalityReason.HIGH_BETWEENNESS)
            if pr >= pr_thresh:
                reasons.append(CriticalityReason.HIGH_PAGERANK)
            if hub >= hub_thresh:
                reasons.append(CriticalityReason.HIGH_HUB_SCORE)
            if auth >= auth_thresh:
                reasons.append(CriticalityReason.HIGH_AUTHORITY_SCORE)
            if in_deg >= in_thresh:
                reasons.append(CriticalityReason.HIGH_IN_DEGREE)
            if out_deg >= out_thresh:
                reasons.append(CriticalityReason.HIGH_OUT_DEGREE)
            if kcore >= kcore_thresh:
                reasons.append(CriticalityReason.INNER_CORE)
            if dep_count >= dep_thresh:
                reasons.append(CriticalityReason.MANY_DEPENDENTS)
            
            # Only include nodes with at least one reason
            if reasons:
                node_data = self.G.nodes[node]
                layer = self._get_layer(node_data.get('layer', node_data.get('type', 'unknown')))
                
                impact, recommendation = self._generate_interpretation(
                    reasons, node_data.get('type', 'Unknown'), in_deg, out_deg, dep_count
                )
                
                critical_nodes.append(CriticalNode(
                    node_id=node,
                    node_type=node_data.get('type', 'Unknown'),
                    layer=layer,
                    reasons=reasons,
                    betweenness=bc,
                    pagerank=pr,
                    hub_score=hub,
                    authority_score=auth,
                    in_degree=in_deg,
                    out_degree=out_deg,
                    k_core=kcore,
                    is_articulation_point=is_ap,
                    dependent_count=dep_count,
                    dependency_depth=dep_depth,
                    impact_description=impact,
                    recommendation=recommendation
                ))
        
        # Sort by number of reasons (more reasons = more critical)
        critical_nodes.sort(key=lambda x: len(x.reasons), reverse=True)
        
        self.logger.info(f"Identified {len(critical_nodes)} critical nodes")
        return critical_nodes
    
    def identify_critical_edges(self) -> List[CriticalEdge]:
        """Identify critical edges with reasons"""
        self.logger.info("Identifying critical edges...")
        
        critical_edges = []
        
        # Get edge betweenness
        applicator = DirectAlgorithmApplicator(self.G)
        edge_bc = applicator.get_edge_betweenness()
        
        # Get bridges
        bridges = set(self.results.get('bridges', []))
        # Also add reverse for undirected consideration
        bridges_undirected = bridges | {(b, a) for a, b in bridges}
        
        # Threshold for edge betweenness
        if edge_bc:
            bc_values = list(edge_bc.values())
            bc_thresh = sorted(bc_values)[int(len(bc_values) * 0.75)] if bc_values else 0
        else:
            bc_thresh = 0
        
        for edge in self.G.edges():
            reasons = []
            u, v = edge
            
            bc = edge_bc.get(edge, 0)
            is_bridge = edge in bridges_undirected
            
            if is_bridge:
                reasons.append(CriticalityReason.BRIDGE_EDGE)
            if bc >= bc_thresh and bc > 0:
                reasons.append(CriticalityReason.HIGH_BETWEENNESS)
            
            # Check if cross-layer
            u_layer = self._get_layer(self.G.nodes[u].get('layer', ''))
            v_layer = self._get_layer(self.G.nodes[v].get('layer', ''))
            if u_layer != v_layer:
                reasons.append(CriticalityReason.CROSS_LAYER_BRIDGE)
            
            if reasons:
                edge_data = self.G.edges[edge]
                
                impact, recommendation = self._generate_edge_interpretation(
                    reasons, edge_data.get('type', 'Unknown'), is_bridge
                )
                
                critical_edges.append(CriticalEdge(
                    source=u,
                    target=v,
                    edge_type=edge_data.get('type', 'Unknown'),
                    reasons=reasons,
                    betweenness=bc,
                    is_bridge=is_bridge,
                    source_layer=u_layer,
                    target_layer=v_layer,
                    impact_description=impact,
                    recommendation=recommendation
                ))
        
        # Sort by number of reasons
        critical_edges.sort(key=lambda x: len(x.reasons), reverse=True)
        
        self.logger.info(f"Identified {len(critical_edges)} critical edges")
        return critical_edges
    
    def _get_threshold(self, metric: str, percentile: float) -> float:
        """Get threshold value at percentile"""
        values = list(self.results.get(metric, {}).values())
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * percentile)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]
    
    def _get_threshold_nested(self, outer: str, inner: str, percentile: float) -> float:
        """Get threshold from nested dict"""
        values = list(self.results.get(outer, {}).get(inner, {}).values())
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * percentile)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]
    
    def _get_layer(self, layer_str: str) -> Layer:
        """Convert string to Layer enum"""
        layer_map = {
            'application': Layer.APPLICATION,
            'Application': Layer.APPLICATION,
            'topic': Layer.TOPIC,
            'Topic': Layer.TOPIC,
            'broker': Layer.BROKER,
            'Broker': Layer.BROKER,
            'infrastructure': Layer.INFRASTRUCTURE,
            'Node': Layer.INFRASTRUCTURE,
            'node': Layer.INFRASTRUCTURE,
        }
        return layer_map.get(layer_str, Layer.APPLICATION)
    
    def _generate_interpretation(self, reasons: List[CriticalityReason], 
                                  node_type: str, in_deg: int, out_deg: int,
                                  dep_count: int) -> Tuple[str, str]:
        """Generate human-readable impact and recommendation"""
        impacts = []
        recommendations = []
        
        if CriticalityReason.ARTICULATION_POINT in reasons:
            impacts.append("Single point of failure - removal disconnects the system")
            recommendations.append("Add redundant paths or backup components")
        
        if CriticalityReason.HIGH_BETWEENNESS in reasons:
            impacts.append("Critical routing point - many paths go through this component")
            recommendations.append("Consider load distribution or caching")
        
        if CriticalityReason.HIGH_HUB_SCORE in reasons:
            impacts.append(f"Major data source with {out_deg} outgoing connections")
            recommendations.append("Ensure high availability for this publisher")
        
        if CriticalityReason.HIGH_AUTHORITY_SCORE in reasons:
            impacts.append(f"Key data consumer with {in_deg} incoming connections")
            recommendations.append("Monitor for processing bottlenecks")
        
        if CriticalityReason.MANY_DEPENDENTS in reasons:
            impacts.append(f"{dep_count} components transitively depend on this")
            recommendations.append("Prioritize reliability and monitoring")
        
        if CriticalityReason.INNER_CORE in reasons:
            impacts.append("Part of the tightly connected core topology")
        
        impact = "; ".join(impacts) if impacts else "Critical component identified"
        recommendation = "; ".join(recommendations) if recommendations else "Review and monitor"
        
        return impact, recommendation
    
    def _generate_edge_interpretation(self, reasons: List[CriticalityReason],
                                       edge_type: str, is_bridge: bool) -> Tuple[str, str]:
        """Generate interpretation for edges"""
        impacts = []
        recommendations = []
        
        if CriticalityReason.BRIDGE_EDGE in reasons:
            impacts.append("Bridge edge - only connection between graph regions")
            recommendations.append("Add redundant connection path")
        
        if CriticalityReason.HIGH_BETWEENNESS in reasons:
            impacts.append("High-traffic relationship")
            recommendations.append("Monitor for latency and reliability")
        
        if CriticalityReason.CROSS_LAYER_BRIDGE in reasons:
            impacts.append("Cross-layer dependency")
            recommendations.append("Ensure proper error handling across layers")
        
        impact = "; ".join(impacts) if impacts else "Critical relationship"
        recommendation = "; ".join(recommendations) if recommendations else "Monitor"
        
        return impact, recommendation


# ============================================================================
# Anti-Pattern Detector
# ============================================================================

class AntiPatternDetector:
    """
    Detects architectural anti-patterns from graph structure.
    """
    
    def __init__(self, builder: MultiLayerGraphBuilder, 
                 algorithm_results: Dict[str, Any]):
        self.builder = builder
        self.G = builder.full_graph
        self.results = algorithm_results
        self.logger = logging.getLogger('antipattern_detector')
        
        # Configurable thresholds
        self.config = {
            'god_topic_threshold': 10,
            'hub_overload_threshold': 15,
            'long_chain_threshold': 5,
            'tight_coupling_min_size': 4,
            'broker_bottleneck_threshold': 10,
        }
    
    def detect_all(self) -> List[AntiPattern]:
        """Detect all anti-patterns"""
        self.logger.info("Detecting anti-patterns...")
        
        patterns = []
        
        patterns.extend(self._detect_god_topics())
        patterns.extend(self._detect_single_points_of_failure())
        patterns.extend(self._detect_circular_dependencies())
        patterns.extend(self._detect_hub_overload())
        patterns.extend(self._detect_long_dependency_chains())
        patterns.extend(self._detect_broker_bottlenecks())
        patterns.extend(self._detect_missing_redundancy())
        patterns.extend(self._detect_hidden_coupling())
        patterns.extend(self._detect_orphan_components())
        
        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        patterns.sort(key=lambda p: severity_order.get(p.severity, 99))
        
        self.logger.info(f"Detected {len(patterns)} anti-patterns")
        return patterns
    
    def _detect_god_topics(self) -> List[AntiPattern]:
        """Detect topics with too many connections"""
        patterns = []
        threshold = self.config['god_topic_threshold']
        
        for topic_id, pubs in self.builder._topic_publishers.items():
            subs = self.builder._topic_subscribers.get(topic_id, set())
            total = len(pubs) + len(subs)
            
            if total >= threshold:
                severity = 'critical' if total >= threshold * 2 else 'high'
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.GOD_TOPIC,
                    severity=severity,
                    affected_components=[topic_id],
                    description=f"Topic '{topic_id}' has {total} connections "
                               f"({len(pubs)} publishers, {len(subs)} subscribers)",
                    impact="Single point of failure for message flow. "
                          "Changes affect many components.",
                    recommendation="Split into multiple focused topics. "
                                  "Consider topic hierarchy or partitioning.",
                    metrics={'total_connections': total, 
                            'publishers': len(pubs),
                            'subscribers': len(subs)}
                ))
        
        return patterns
    
    def _detect_single_points_of_failure(self) -> List[AntiPattern]:
        """Detect articulation points as SPOFs"""
        patterns = []
        
        articulation_points = self.results.get('articulation_points', [])
        
        for ap in articulation_points:
            node_data = self.G.nodes.get(ap, {})
            node_type = node_data.get('type', 'Unknown')
            
            # Calculate impact by removing node
            G_copy = self.G.to_undirected().copy()
            G_copy.remove_node(ap)
            components = list(nx.connected_components(G_copy))
            
            severity = 'critical' if len(components) > 2 else 'high'
            
            patterns.append(AntiPattern(
                pattern_type=AntiPatternType.SINGLE_POINT_OF_FAILURE,
                severity=severity,
                affected_components=[ap],
                description=f"{node_type} '{ap}' is a single point of failure. "
                           f"Removal creates {len(components)} disconnected components.",
                impact="System fragmentation if this component fails.",
                recommendation="Add redundancy. Create alternative paths.",
                metrics={'components_after_removal': len(components),
                        'component_sizes': [len(c) for c in components]}
            ))
        
        return patterns
    
    def _detect_circular_dependencies(self) -> List[AntiPattern]:
        """Detect circular dependencies in app layer"""
        patterns = []
        
        # Find cycles in app dependency graph
        app_graph = self.builder.app_depends_graph
        
        try:
            cycles = list(nx.simple_cycles(app_graph))
        except:
            cycles = []
        
        for cycle in cycles:
            if len(cycle) > 1:
                severity = 'high' if len(cycle) > 3 else 'medium'
                
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.CIRCULAR_DEPENDENCY,
                    severity=severity,
                    affected_components=cycle,
                    description=f"Circular dependency among {len(cycle)} apps: "
                               f"{' -> '.join(cycle[:5])}{'...' if len(cycle) > 5 else ''} -> {cycle[0]}",
                    impact="Potential for infinite loops. Complex failure cascades.",
                    recommendation="Break cycle by introducing async communication or redesigning.",
                    metrics={'cycle_length': len(cycle)}
                ))
        
        return patterns
    
    def _detect_hub_overload(self) -> List[AntiPattern]:
        """Detect components with excessive connections"""
        patterns = []
        threshold = self.config['hub_overload_threshold']
        
        out_degree = self.results.get('degree', {}).get('out_degree', {})
        
        for node, degree in out_degree.items():
            if degree >= threshold:
                node_data = self.G.nodes.get(node, {})
                node_type = node_data.get('type', 'Unknown')
                
                severity = 'high' if degree >= threshold * 2 else 'medium'
                
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.HUB_OVERLOAD,
                    severity=severity,
                    affected_components=[node],
                    description=f"{node_type} '{node}' has {degree} outgoing connections",
                    impact="Bottleneck risk. Single component affects many.",
                    recommendation="Consider splitting responsibilities or adding load balancing.",
                    metrics={'out_degree': degree}
                ))
        
        return patterns
    
    def _detect_long_dependency_chains(self) -> List[AntiPattern]:
        """Detect deep dependency chains"""
        patterns = []
        threshold = self.config['long_chain_threshold']
        
        depths = self.results.get('dependency_depth', {})
        
        for node, depth in depths.items():
            if depth >= threshold:
                node_data = self.G.nodes.get(node, {})
                
                severity = 'high' if depth >= threshold * 2 else 'medium'
                
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.LONG_DEPENDENCY_CHAIN,
                    severity=severity,
                    affected_components=[node],
                    description=f"Component '{node}' has dependency depth of {depth}",
                    impact="Failures cascade through long chain. High latency.",
                    recommendation="Reduce chain length. Introduce caching.",
                    metrics={'depth': depth}
                ))
        
        return patterns
    
    def _detect_broker_bottlenecks(self) -> List[AntiPattern]:
        """Detect brokers routing too many critical topics"""
        patterns = []
        threshold = self.config['broker_bottleneck_threshold']
        
        for broker_id, topics in self._get_broker_topics().items():
            if len(topics) >= threshold:
                severity = 'high' if len(topics) >= threshold * 2 else 'medium'
                
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.BROKER_BOTTLENECK,
                    severity=severity,
                    affected_components=[broker_id],
                    description=f"Broker '{broker_id}' routes {len(topics)} topics",
                    impact="Broker failure affects many message flows.",
                    recommendation="Distribute topics across multiple brokers.",
                    metrics={'topic_count': len(topics), 'topics': list(topics)[:10]}
                ))
        
        return patterns
    
    def _detect_missing_redundancy(self) -> List[AntiPattern]:
        """Detect bridge edges indicating missing redundancy"""
        patterns = []
        
        bridges = self.results.get('bridges', [])
        
        # Group bridges by type of connection
        critical_bridges = []
        for u, v in bridges:
            u_type = self.G.nodes.get(u, {}).get('type', 'Unknown')
            v_type = self.G.nodes.get(v, {}).get('type', 'Unknown')
            
            # Infrastructure bridges are most critical
            if 'Node' in [u_type, v_type] or 'Broker' in [u_type, v_type]:
                critical_bridges.append((u, v))
        
        if critical_bridges:
            patterns.append(AntiPattern(
                pattern_type=AntiPatternType.MISSING_REDUNDANCY,
                severity='high',
                affected_components=[f"{u}->{v}" for u, v in critical_bridges],
                description=f"{len(critical_bridges)} infrastructure edges have no redundant path",
                impact="Single edge failure can partition the system.",
                recommendation="Add redundant network paths.",
                metrics={'bridge_count': len(critical_bridges)}
            ))
        
        return patterns
    
    def _detect_hidden_coupling(self) -> List[AntiPattern]:
        """Detect hidden coupling through shared infrastructure"""
        patterns = []
        
        # Find apps that don't directly communicate but share infrastructure
        for node_id, hosted_apps in self._get_node_apps().items():
            if len(hosted_apps) < 2:
                continue
            
            # Check for apps that don't have direct dependencies
            app_graph = self.builder.app_depends_graph
            uncoupled_pairs = []
            
            apps = list(hosted_apps)
            for i in range(len(apps)):
                for j in range(i + 1, len(apps)):
                    app1, app2 = apps[i], apps[j]
                    
                    # Check if they have direct dependency
                    has_direct = (app_graph.has_edge(app1, app2) or 
                                 app_graph.has_edge(app2, app1))
                    
                    if not has_direct:
                        uncoupled_pairs.append((app1, app2))
            
            if len(uncoupled_pairs) >= 2:
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.HIDDEN_COUPLING,
                    severity='medium',
                    affected_components=[node_id] + list(hosted_apps),
                    description=f"Node '{node_id}' hosts {len(hosted_apps)} apps with hidden coupling",
                    impact="Infrastructure failure affects seemingly independent apps.",
                    recommendation="Review deployment strategy. Consider isolation.",
                    metrics={'hosted_apps': len(hosted_apps),
                            'uncoupled_pairs': len(uncoupled_pairs)}
                ))
        
        return patterns
    
    def _detect_orphan_components(self) -> List[AntiPattern]:
        """Detect isolated or disconnected components"""
        patterns = []
        
        for node in self.G.nodes():
            in_deg = self.G.in_degree(node)
            out_deg = self.G.out_degree(node)
            
            if in_deg == 0 and out_deg == 0:
                node_data = self.G.nodes[node]
                
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.ORPHAN_COMPONENT,
                    severity='low',
                    affected_components=[node],
                    description=f"Component '{node}' ({node_data.get('type')}) is completely isolated",
                    impact="May be unused or misconfigured.",
                    recommendation="Verify if intentional or connect properly.",
                    metrics={'in_degree': in_deg, 'out_degree': out_deg}
                ))
        
        return patterns
    
    def _get_broker_topics(self) -> Dict[str, Set[str]]:
        """Get topics routed by each broker"""
        broker_topics = defaultdict(set)
        for topic_id, broker_id in self.builder._topic_to_broker.items():
            broker_topics[broker_id].add(topic_id)
        return broker_topics
    
    def _get_node_apps(self) -> Dict[str, Set[str]]:
        """Get apps running on each node"""
        node_apps = defaultdict(set)
        for app_id, node_id in self.builder._app_to_node.items():
            node_apps[node_id].add(app_id)
        return node_apps


# ============================================================================
# Main Analyzer
# ============================================================================

class MultiLayerDependencyAnalyzer:
    """
    Main analyzer for multi-layer dependency graph analysis.
    
    Workflow:
    1. Build multi-layer DEPENDS_ON graph from pub-sub data
    2. Apply graph algorithms directly to each layer
    3. Identify critical nodes and edges from algorithm results
    4. Detect anti-patterns from graph structure
    """
    
    def __init__(self):
        self.logger = logging.getLogger('multi_layer_analyzer')
        self.builder: Optional[MultiLayerGraphBuilder] = None
        self.algorithm_results: Dict[str, Dict[str, Any]] = {}
    
    def analyze_from_file(self, filepath: str) -> MultiLayerAnalysisResult:
        """Run analysis from JSON file"""
        self.logger.info(f"Loading data from {filepath}")
        
        with open(filepath) as f:
            data = json.load(f)
        
        return self.analyze_from_dict(data)
    
    def analyze_from_dict(self, data: Dict[str, Any]) -> MultiLayerAnalysisResult:
        """Run analysis from dictionary data"""
        self.logger.info("Starting multi-layer dependency analysis...")
        
        # Step 1: Build multi-layer graphs
        self.builder = MultiLayerGraphBuilder()
        self.builder.build_from_dict(data)
        
        # Step 2: Apply algorithms to each layer
        self.algorithm_results = {}
        
        # Full graph analysis
        if self.builder.full_graph.number_of_nodes() > 0:
            self.logger.info("Analyzing full graph...")
            full_applicator = DirectAlgorithmApplicator(self.builder.full_graph)
            self.algorithm_results['full'] = full_applicator.apply_all_algorithms()
        
        # App dependency graph analysis
        if self.builder.app_depends_graph.number_of_nodes() > 0:
            self.logger.info("Analyzing app dependency layer...")
            app_applicator = DirectAlgorithmApplicator(self.builder.app_depends_graph)
            self.algorithm_results['app_layer'] = app_applicator.apply_all_algorithms()
        
        # Node dependency graph analysis
        if self.builder.node_depends_graph.number_of_nodes() > 0:
            self.logger.info("Analyzing infrastructure layer...")
            node_applicator = DirectAlgorithmApplicator(self.builder.node_depends_graph)
            self.algorithm_results['infra_layer'] = node_applicator.apply_all_algorithms()
        
        # Step 3: Identify critical components
        critical_nodes = []
        critical_edges = []
        
        # From full graph
        if 'full' in self.algorithm_results:
            identifier = CriticalComponentIdentifier(
                self.builder.full_graph,
                self.algorithm_results['full']
            )
            critical_nodes.extend(identifier.identify_critical_nodes())
            critical_edges.extend(identifier.identify_critical_edges())
        
        # Step 4: Detect anti-patterns
        detector = AntiPatternDetector(
            self.builder,
            self.algorithm_results.get('full', {})
        )
        anti_patterns = detector.detect_all()
        
        # Step 5: Compute layer coupling
        layer_coupling = self._compute_layer_coupling()
        
        # Build result
        result = MultiLayerAnalysisResult(
            total_nodes=self.builder.full_graph.number_of_nodes(),
            total_edges=self.builder.full_graph.number_of_edges(),
            layers_analyzed=list(self.algorithm_results.keys()),
            application_layer=self._build_layer_result(
                self.builder.app_depends_graph,
                self.algorithm_results.get('app_layer', {}),
                Layer.APPLICATION
            ),
            infrastructure_layer=self._build_layer_result(
                self.builder.node_depends_graph,
                self.algorithm_results.get('infra_layer', {}),
                Layer.INFRASTRUCTURE
            ),
            full_graph=None,  # Could add if needed
            cross_layer_dependencies=[e for e in self.builder.depends_on_edges
                                      if e.dep_type in [DependencyType.APP_TO_BROKER,
                                                        DependencyType.APP_TO_NODE]],
            layer_coupling=layer_coupling,
            critical_nodes=critical_nodes,
            critical_edges=critical_edges,
            anti_patterns=anti_patterns,
            algorithm_results=self.algorithm_results
        )
        
        self.logger.info(f"Analysis complete: {len(critical_nodes)} critical nodes, "
                        f"{len(critical_edges)} critical edges, "
                        f"{len(anti_patterns)} anti-patterns")
        
        return result
    
    def _build_layer_result(self, graph: nx.DiGraph, 
                            algo_results: Dict[str, Any],
                            layer: Layer) -> Optional[LayerAnalysisResult]:
        """Build result for a single layer"""
        if graph.number_of_nodes() == 0:
            return None
        
        identifier = CriticalComponentIdentifier(graph, algo_results)
        
        return LayerAnalysisResult(
            layer=layer,
            node_count=graph.number_of_nodes(),
            edge_count=graph.number_of_edges(),
            density=nx.density(graph),
            articulation_points=algo_results.get('articulation_points', []),
            bridges=algo_results.get('bridges', []),
            strongly_connected_components=len(algo_results.get('sccs', [])),
            max_k_core=max(algo_results.get('k_core', {}).values()) if algo_results.get('k_core') else 0,
            critical_nodes=identifier.identify_critical_nodes(),
            critical_edges=identifier.identify_critical_edges()
        )
    
    def _compute_layer_coupling(self) -> Dict[str, float]:
        """Compute coupling between layers"""
        coupling = {}
        
        # App-to-infrastructure coupling
        cross_layer_edges = len([e for e in self.builder.depends_on_edges
                                if e.dep_type == DependencyType.APP_TO_NODE])
        
        app_count = self.builder.app_depends_graph.number_of_nodes()
        node_count = self.builder.node_depends_graph.number_of_nodes()
        
        if app_count > 0 and node_count > 0:
            max_coupling = app_count * node_count
            coupling['app_to_infra'] = cross_layer_edges / max_coupling
        else:
            coupling['app_to_infra'] = 0.0
        
        return coupling
    
    def get_summary(self, result: MultiLayerAnalysisResult) -> str:
        """Generate human-readable summary"""
        lines = [
            "=" * 70,
            "MULTI-LAYER DEPENDENCY ANALYSIS SUMMARY",
            "=" * 70,
            "",
            f"Total Nodes: {result.total_nodes}",
            f"Total Edges: {result.total_edges}",
            f"Layers Analyzed: {', '.join(result.layers_analyzed)}",
            "",
            "--- Critical Nodes (by algorithm evidence) ---",
        ]
        
        # Group by reason count
        for node in result.critical_nodes[:10]:
            reasons_str = ", ".join(r.value for r in node.reasons)
            lines.append(f"  {node.node_id} ({node.node_type}): {reasons_str}")
        
        lines.extend([
            "",
            f"--- Anti-Patterns ({len(result.anti_patterns)} detected) ---",
        ])
        
        for pattern in result.anti_patterns[:10]:
            lines.append(f"  [{pattern.severity.upper()}] {pattern.pattern_type.value}: "
                        f"{pattern.description[:60]}...")
        
        lines.extend([
            "",
            "--- Layer Coupling ---",
        ])
        
        for layer_pair, coupling in result.layer_coupling.items():
            lines.append(f"  {layer_pair}: {coupling:.4f}")
        
        return "\n".join(lines)


# ============================================================================
# Convenience Functions
# ============================================================================

def analyze_multi_layer(filepath: str) -> MultiLayerAnalysisResult:
    """Convenience function for multi-layer analysis"""
    analyzer = MultiLayerDependencyAnalyzer()
    return analyzer.analyze_from_file(filepath)


def analyze_multi_layer_dict(data: Dict[str, Any]) -> MultiLayerAnalysisResult:
    """Convenience function for multi-layer analysis from dict"""
    analyzer = MultiLayerDependencyAnalyzer()
    return analyzer.analyze_from_dict(data)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Multi-Layer Dependency Graph Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python multi_layer_analyzer.py --input system.json
    python multi_layer_analyzer.py --input system.json --output results.json
    python multi_layer_analyzer.py --input system.json --anti-patterns-only
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                        help='Input JSON file')
    parser.add_argument('--output', '-o',
                        help='Output JSON file for results')
    parser.add_argument('--anti-patterns-only', action='store_true',
                        help='Only show anti-patterns')
    parser.add_argument('--critical-only', action='store_true',
                        help='Only show critical nodes/edges')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run analysis
    analyzer = MultiLayerDependencyAnalyzer()
    result = analyzer.analyze_from_file(args.input)
    
    # Output
    if args.anti_patterns_only:
        print("\n=== ANTI-PATTERNS DETECTED ===\n")
        for pattern in result.anti_patterns:
            print(f"[{pattern.severity.upper()}] {pattern.pattern_type.value}")
            print(f"  Description: {pattern.description}")
            print(f"  Impact: {pattern.impact}")
            print(f"  Recommendation: {pattern.recommendation}")
            print()
    elif args.critical_only:
        print("\n=== CRITICAL NODES ===\n")
        for node in result.critical_nodes[:20]:
            print(f"{node.node_id} ({node.node_type})")
            print(f"  Reasons: {', '.join(r.value for r in node.reasons)}")
            print(f"  Impact: {node.impact_description}")
            print()
        
        print("\n=== CRITICAL EDGES ===\n")
        for edge in result.critical_edges[:20]:
            print(f"{edge.source} -> {edge.target} ({edge.edge_type})")
            print(f"  Reasons: {', '.join(r.value for r in edge.reasons)}")
            print()
    else:
        print(analyzer.get_summary(result))
    
    # Save to file
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())