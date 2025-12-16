#!/usr/bin/env python3
"""
Relationship Analyzer - Advanced Graph Algorithms for Pub-Sub Analysis
=======================================================================

This module provides advanced graph algorithms focused on relationship analysis
in distributed publish-subscribe systems. It extends beyond node-centric metrics
to provide edge-centric, motif-based, and flow-aware analysis.

Key Capabilities:
  - Edge-centric criticality analysis (edge betweenness, Simmelian strength)
  - HITS-based role analysis (hub/authority alignment for pub-sub)
  - Network motif detection (fan-out, fan-in, chains, diamonds)
  - Dependency chain analysis (transitive depth, fan-out/fan-in ratios)
  - Multi-layer correlation analysis
  - Flow-aware centrality (weighted by message frequency/QoS)
  - Ensemble criticality scoring

Research Foundation:
  - Composite Criticality Score: C_score(v) = α·C_B^norm(v) + β·AP(v) + γ·I(v)
  - Extended with relationship-aware metrics for comprehensive analysis

Author: Software-as-a-Graph Research Project
"""

import logging
import math
from typing import Dict, List, Any, Optional, Tuple, Set, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from functools import lru_cache

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX is required: pip install networkx")


# ============================================================================
# Enums and Data Classes
# ============================================================================

class RelationshipType(Enum):
    """Types of relationships in pub-sub systems"""
    PUBLISHES_TO = "PUBLISHES_TO"
    SUBSCRIBES_TO = "SUBSCRIBES_TO"
    ROUTES = "ROUTES"
    RUNS_ON = "RUNS_ON"
    CONNECTS_TO = "CONNECTS_TO"
    DEPENDS_ON = "DEPENDS_ON"


class MotifType(Enum):
    """Network motif types relevant to pub-sub systems"""
    FAN_OUT = "fan_out"           # One publisher → Many subscribers via topic
    FAN_IN = "fan_in"             # Many publishers → One topic → Few subscribers
    CHAIN = "chain"               # App1 → Topic1 → App2 → Topic2 → App3
    DIAMOND = "diamond"           # Redundant paths through different topics/brokers
    STAR = "star"                 # Central hub with many connections
    TRIANGLE = "triangle"         # Three-node cycle


class ComponentRole(Enum):
    """Component role based on HITS analysis"""
    PURE_PUBLISHER = "pure_publisher"      # High hub, low authority
    PURE_SUBSCRIBER = "pure_subscriber"    # Low hub, high authority
    RELAY = "relay"                        # Both high hub and authority
    PERIPHERAL = "peripheral"              # Both low hub and authority


@dataclass
class EdgeCriticalityResult:
    """Result of edge criticality analysis"""
    edge: Tuple[str, str]
    edge_type: str
    betweenness_centrality: float
    simmelian_strength: int
    is_bridge: bool
    flow_weight: float
    criticality_score: float
    criticality_level: str
    interpretation: str


@dataclass
class HITSRoleResult:
    """Result of HITS-based role analysis"""
    node_id: str
    node_type: str
    hub_score: float
    authority_score: float
    role: ComponentRole
    hub_rank: int
    authority_rank: int
    role_alignment: float  # How well the role fits the node type
    interpretation: str


@dataclass
class MotifInstance:
    """Instance of a detected motif"""
    motif_type: MotifType
    nodes: List[str]
    edges: List[Tuple[str, str]]
    central_node: Optional[str]
    criticality_impact: float
    description: str


@dataclass
class DependencyChainResult:
    """Result of dependency chain analysis"""
    node_id: str
    transitive_depth: int           # How many hops until no more dependencies
    upstream_count: int             # Components this node depends on
    downstream_count: int           # Components that depend on this node
    dependency_ratio: float         # downstream / upstream
    chain_criticality: float        # Based on position in dependency hierarchy
    longest_upstream_chain: List[str]
    longest_downstream_chain: List[str]


@dataclass
class LayerCorrelationResult:
    """Result of multi-layer correlation analysis"""
    layer1: str
    layer2: str
    centrality_correlation: float
    coupling_coefficient: float
    misaligned_components: List[Tuple[str, str, str]]  # (app, node, reason)
    interpretation: str


@dataclass
class EnsembleCriticalityResult:
    """Combined criticality from multiple algorithm perspectives"""
    node_id: str
    node_type: str
    
    # Individual scores
    betweenness_score: float
    pagerank_score: float
    hits_hub_score: float
    hits_authority_score: float
    is_articulation_point: bool
    kcore_number: int
    closeness_score: float
    
    # Derived indicators
    role: ComponentRole
    is_structural_critical: bool    # AP or high BC
    is_flow_critical: bool          # High PageRank or HITS
    is_connectivity_critical: bool  # High closeness or k-core
    
    # Ensemble score
    ensemble_score: float
    ensemble_level: str
    confidence: float  # Agreement between algorithms
    
    # Explanations
    primary_reasons: List[str]
    secondary_indicators: List[str]


@dataclass 
class RelationshipAnalysisResult:
    """Complete result of relationship analysis"""
    # Summary
    total_nodes: int
    total_edges: int
    analysis_timestamp: str
    
    # Edge Analysis
    edge_criticality: List[EdgeCriticalityResult]
    critical_edges: List[Tuple[str, str]]
    bridge_edges: List[Tuple[str, str]]
    
    # HITS Analysis
    hits_roles: Dict[str, HITSRoleResult]
    top_hubs: List[str]
    top_authorities: List[str]
    
    # Motif Analysis
    motifs: List[MotifInstance]
    motif_summary: Dict[str, int]
    
    # Dependency Analysis
    dependency_chains: Dict[str, DependencyChainResult]
    deepest_chains: List[str]
    highest_fan_out: List[str]
    
    # Layer Analysis
    layer_correlations: List[LayerCorrelationResult]
    
    # Ensemble Criticality
    ensemble_criticality: Dict[str, EnsembleCriticalityResult]
    
    # Recommendations
    recommendations: List[Dict[str, Any]]


# ============================================================================
# Edge-Centric Analysis
# ============================================================================

class EdgeCriticalityAnalyzer:
    """
    Analyzes edge (relationship) criticality in pub-sub graphs.
    
    Key metrics:
    - Edge betweenness centrality: How often an edge lies on shortest paths
    - Simmelian strength: Triangular embeddedness (structural reinforcement)
    - Bridge detection: Edges whose removal disconnects the graph
    - Flow weight: Message frequency or QoS-based importance
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self.G_undirected = graph.to_undirected()
        self.logger = logging.getLogger('edge_criticality')
        
        # Cache computed values
        self._edge_bc: Optional[Dict] = None
        self._bridges: Optional[Set] = None
    
    def analyze_all_edges(self, 
                          weight_attr: Optional[str] = None,
                          critical_threshold: float = 0.7) -> List[EdgeCriticalityResult]:
        """
        Analyze criticality of all edges in the graph.
        
        Args:
            weight_attr: Edge attribute to use as flow weight
            critical_threshold: Percentile threshold for critical classification
            
        Returns:
            List of EdgeCriticalityResult for all edges
        """
        self.logger.info(f"Analyzing {self.G.number_of_edges()} edges...")
        
        # Compute edge betweenness centrality
        edge_bc = self._compute_edge_betweenness()
        
        # Find bridges
        bridges = self._find_bridges()
        
        # Compute Simmelian strength for all edges
        simmelian = self._compute_simmelian_strength()
        
        # Get flow weights
        flow_weights = self._get_flow_weights(weight_attr)
        
        # Determine threshold for critical edges
        bc_values = list(edge_bc.values())
        if bc_values:
            bc_threshold = sorted(bc_values)[int(len(bc_values) * critical_threshold)]
        else:
            bc_threshold = 0.0
        
        results = []
        for edge in self.G.edges():
            u, v = edge
            edge_data = self.G.edges[edge]
            edge_type = edge_data.get('type', edge_data.get('edge_type', 'Unknown'))
            
            bc = edge_bc.get(edge, 0.0)
            sim = simmelian.get(edge, 0)
            is_bridge = edge in bridges or (v, u) in bridges
            flow = flow_weights.get(edge, 1.0)
            
            # Composite criticality score
            # Weighted: BC (40%), bridge (30%), flow (20%), Simmelian inverse (10%)
            bridge_score = 1.0 if is_bridge else 0.0
            sim_score = 1.0 / (1.0 + sim)  # Lower Simmelian = less redundancy = more critical
            
            criticality = (
                0.4 * bc +
                0.3 * bridge_score +
                0.2 * min(flow, 1.0) +
                0.1 * sim_score
            )
            
            # Determine level
            if criticality >= 0.7 or is_bridge:
                level = "CRITICAL"
            elif criticality >= 0.5:
                level = "HIGH"
            elif criticality >= 0.3:
                level = "MEDIUM"
            else:
                level = "LOW"
            
            # Generate interpretation
            interpretation = self._interpret_edge(bc, sim, is_bridge, flow, edge_type)
            
            results.append(EdgeCriticalityResult(
                edge=edge,
                edge_type=edge_type,
                betweenness_centrality=bc,
                simmelian_strength=sim,
                is_bridge=is_bridge,
                flow_weight=flow,
                criticality_score=criticality,
                criticality_level=level,
                interpretation=interpretation
            ))
        
        # Sort by criticality score descending
        results.sort(key=lambda x: x.criticality_score, reverse=True)
        
        self.logger.info(f"Found {sum(1 for r in results if r.criticality_level == 'CRITICAL')} critical edges")
        return results
    
    def _compute_edge_betweenness(self) -> Dict[Tuple[str, str], float]:
        """Compute edge betweenness centrality"""
        if self._edge_bc is None:
            self._edge_bc = nx.edge_betweenness_centrality(self.G, normalized=True)
        return self._edge_bc
    
    def _find_bridges(self) -> Set[Tuple[str, str]]:
        """Find bridge edges (cut edges)"""
        if self._bridges is None:
            self._bridges = set(nx.bridges(self.G_undirected))
        return self._bridges
    
    def _compute_simmelian_strength(self) -> Dict[Tuple[str, str], int]:
        """
        Compute Simmelian strength for each edge.
        Simmelian strength = number of triangles the edge participates in.
        Higher strength indicates more structural redundancy.
        """
        simmelian = {}
        for u, v in self.G.edges():
            # Count common neighbors (triangles)
            u_neighbors = set(self.G_undirected.neighbors(u))
            v_neighbors = set(self.G_undirected.neighbors(v))
            common = u_neighbors & v_neighbors
            simmelian[(u, v)] = len(common)
        return simmelian
    
    def _get_flow_weights(self, weight_attr: Optional[str]) -> Dict[Tuple[str, str], float]:
        """Get flow weights for edges"""
        weights = {}
        for edge in self.G.edges():
            if weight_attr and weight_attr in self.G.edges[edge]:
                weights[edge] = self.G.edges[edge][weight_attr]
            else:
                # Default weight based on edge type
                edge_type = self.G.edges[edge].get('type', '')
                if 'PUBLISHES' in edge_type or 'SUBSCRIBES' in edge_type:
                    weights[edge] = 1.0  # Message flow edges
                elif 'ROUTES' in edge_type:
                    weights[edge] = 0.8  # Broker routing
                else:
                    weights[edge] = 0.5  # Infrastructure edges
        return weights
    
    def _interpret_edge(self, bc: float, sim: int, is_bridge: bool, 
                        flow: float, edge_type: str) -> str:
        """Generate human-readable interpretation"""
        parts = []
        
        if is_bridge:
            parts.append("SINGLE POINT OF FAILURE - removal disconnects graph")
        
        if bc > 0.5:
            parts.append(f"High traffic route (BC={bc:.3f})")
        elif bc > 0.1:
            parts.append(f"Moderate traffic route (BC={bc:.3f})")
        
        if sim == 0:
            parts.append("No redundant paths (isolated connection)")
        elif sim >= 3:
            parts.append(f"Well-reinforced ({sim} triangles)")
        
        if 'PUBLISHES' in edge_type:
            parts.append("Publication relationship")
        elif 'SUBSCRIBES' in edge_type:
            parts.append("Subscription relationship")
        elif 'ROUTES' in edge_type:
            parts.append("Broker routing path")
        
        return "; ".join(parts) if parts else "Standard connection"


# ============================================================================
# HITS-Based Role Analysis
# ============================================================================

class HITSRoleAnalyzer:
    """
    Analyzes component roles using HITS (Hyperlink-Induced Topic Search).
    
    In pub-sub systems:
    - Publishers are natural HUBS (many outgoing PUBLISHES edges)
    - Subscribers are natural AUTHORITIES (receive from important sources)
    - Topics can be both (receive from publishers, feed subscribers)
    
    This analyzer identifies role alignment and misalignment.
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self.logger = logging.getLogger('hits_role')
        
        # Cached HITS scores
        self._hubs: Optional[Dict[str, float]] = None
        self._authorities: Optional[Dict[str, float]] = None
    
    def analyze_roles(self, max_iter: int = 100) -> Dict[str, HITSRoleResult]:
        """
        Analyze all node roles using HITS algorithm.
        
        Args:
            max_iter: Maximum iterations for HITS convergence
            
        Returns:
            Dictionary mapping node IDs to HITSRoleResult
        """
        self.logger.info("Running HITS role analysis...")
        
        # Compute HITS scores
        try:
            self._hubs, self._authorities = nx.hits(self.G, max_iter=max_iter)
        except nx.PowerIterationFailedConvergence:
            self.logger.warning("HITS did not converge, using normalized values")
            self._hubs = {n: 1.0/self.G.number_of_nodes() for n in self.G.nodes()}
            self._authorities = {n: 1.0/self.G.number_of_nodes() for n in self.G.nodes()}
        
        # Rank nodes
        hub_ranking = self._rank_scores(self._hubs)
        auth_ranking = self._rank_scores(self._authorities)
        
        # Determine thresholds for role classification
        hub_threshold = self._get_threshold(list(self._hubs.values()), 0.75)
        auth_threshold = self._get_threshold(list(self._authorities.values()), 0.75)
        
        results = {}
        for node in self.G.nodes():
            node_data = self.G.nodes[node]
            node_type = node_data.get('type', 'Unknown')
            
            hub_score = self._hubs[node]
            auth_score = self._authorities[node]
            
            # Determine role
            role = self._determine_role(hub_score, auth_score, hub_threshold, auth_threshold)
            
            # Calculate role alignment
            alignment = self._calculate_alignment(role, node_type)
            
            # Generate interpretation
            interpretation = self._interpret_role(node_type, role, hub_score, auth_score, alignment)
            
            results[node] = HITSRoleResult(
                node_id=node,
                node_type=node_type,
                hub_score=hub_score,
                authority_score=auth_score,
                role=role,
                hub_rank=hub_ranking[node],
                authority_rank=auth_ranking[node],
                role_alignment=alignment,
                interpretation=interpretation
            )
        
        return results
    
    def get_top_hubs(self, n: int = 10) -> List[str]:
        """Get top N hub nodes"""
        if self._hubs is None:
            self.analyze_roles()
        return sorted(self._hubs.keys(), key=lambda x: self._hubs[x], reverse=True)[:n]
    
    def get_top_authorities(self, n: int = 10) -> List[str]:
        """Get top N authority nodes"""
        if self._authorities is None:
            self.analyze_roles()
        return sorted(self._authorities.keys(), key=lambda x: self._authorities[x], reverse=True)[:n]
    
    def _rank_scores(self, scores: Dict[str, float]) -> Dict[str, int]:
        """Convert scores to ranks (1 = highest)"""
        sorted_nodes = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return {node: rank + 1 for rank, node in enumerate(sorted_nodes)}
    
    def _get_threshold(self, values: List[float], percentile: float) -> float:
        """Get threshold at given percentile"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * percentile)
        return sorted_values[min(idx, len(sorted_values) - 1)]
    
    def _determine_role(self, hub: float, auth: float, 
                        hub_thresh: float, auth_thresh: float) -> ComponentRole:
        """Determine component role based on HITS scores"""
        high_hub = hub >= hub_thresh
        high_auth = auth >= auth_thresh
        
        if high_hub and high_auth:
            return ComponentRole.RELAY
        elif high_hub:
            return ComponentRole.PURE_PUBLISHER
        elif high_auth:
            return ComponentRole.PURE_SUBSCRIBER
        else:
            return ComponentRole.PERIPHERAL
    
    def _calculate_alignment(self, role: ComponentRole, node_type: str) -> float:
        """
        Calculate how well the HITS role aligns with expected behavior.
        
        Expected alignments:
        - Applications with role='pub' should be PURE_PUBLISHER
        - Applications with role='sub' should be PURE_SUBSCRIBER
        - Applications with role='pubsub' should be RELAY
        - Topics should have high authority (receive from publishers)
        - Brokers should be RELAY (route messages)
        """
        # Perfect alignment = 1.0, misalignment = 0.0
        
        if node_type == 'Application':
            # Check application role attribute
            app_role = self.G.nodes.get(node_type, {}).get('role', 'pubsub')
            if app_role == 'pub' and role == ComponentRole.PURE_PUBLISHER:
                return 1.0
            elif app_role == 'sub' and role == ComponentRole.PURE_SUBSCRIBER:
                return 1.0
            elif app_role == 'pubsub' and role == ComponentRole.RELAY:
                return 1.0
            elif role == ComponentRole.PERIPHERAL:
                return 0.3  # Low activity
            else:
                return 0.6  # Partial alignment
        
        elif node_type == 'Topic':
            # Topics should have high authority (many publishers feed them)
            if role in [ComponentRole.PURE_SUBSCRIBER, ComponentRole.RELAY]:
                return 1.0
            else:
                return 0.5
        
        elif node_type == 'Broker':
            # Brokers should be relays
            if role == ComponentRole.RELAY:
                return 1.0
            else:
                return 0.6
        
        else:
            # Infrastructure nodes - expect peripheral
            if role == ComponentRole.PERIPHERAL:
                return 1.0
            else:
                return 0.7
    
    def _interpret_role(self, node_type: str, role: ComponentRole,
                        hub: float, auth: float, alignment: float) -> str:
        """Generate interpretation of role analysis"""
        parts = []
        
        role_desc = {
            ComponentRole.PURE_PUBLISHER: "Primary data source",
            ComponentRole.PURE_SUBSCRIBER: "Primary data consumer",
            ComponentRole.RELAY: "Data transformer/relay",
            ComponentRole.PERIPHERAL: "Low-activity component"
        }
        parts.append(role_desc[role])
        
        if alignment < 0.5:
            parts.append(f"⚠️ Role misalignment for {node_type}")
        
        parts.append(f"Hub rank indicates publishing influence")
        parts.append(f"Authority rank indicates consumption importance")
        
        return "; ".join(parts)


# ============================================================================
# Network Motif Detection
# ============================================================================

class MotifDetector:
    """
    Detects network motifs (recurring patterns) in pub-sub graphs.
    
    Key motifs:
    - Fan-out: One publisher → Topic → Many subscribers (broadcast)
    - Fan-in: Many publishers → Topic → Few subscribers (aggregation)
    - Chain: Sequential data flow through topics
    - Diamond: Redundant paths (good for fault tolerance)
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self.logger = logging.getLogger('motif_detector')
    
    def detect_all_motifs(self) -> List[MotifInstance]:
        """Detect all motif instances in the graph"""
        self.logger.info("Detecting network motifs...")
        
        motifs = []
        
        # Detect each motif type
        motifs.extend(self._detect_fan_out())
        motifs.extend(self._detect_fan_in())
        motifs.extend(self._detect_chains())
        motifs.extend(self._detect_diamonds())
        motifs.extend(self._detect_stars())
        
        self.logger.info(f"Found {len(motifs)} motif instances")
        return motifs
    
    def get_motif_summary(self, motifs: List[MotifInstance]) -> Dict[str, int]:
        """Summarize motif counts by type"""
        summary = defaultdict(int)
        for motif in motifs:
            summary[motif.motif_type.value] += 1
        return dict(summary)
    
    def _detect_fan_out(self, min_subscribers: int = 3) -> List[MotifInstance]:
        """
        Detect fan-out patterns: Publisher → Topic → Many Subscribers
        
        Critical because:
        - Publisher failure affects many downstream consumers
        - Topic is a broadcast bottleneck
        """
        motifs = []
        
        # Find topics with many subscribers
        for node in self.G.nodes():
            if self.G.nodes[node].get('type') != 'Topic':
                continue
            
            # Get publishers (incoming PUBLISHES edges)
            publishers = [u for u, v in self.G.in_edges(node) 
                         if self.G.edges[u, v].get('type') == 'PUBLISHES_TO']
            
            # Get subscribers (outgoing SUBSCRIBES edges - reversed in directed graph)
            subscribers = [v for u, v in self.G.out_edges(node)
                          if self.G.edges[u, v].get('type') == 'SUBSCRIBES_TO']
            
            # Also check incoming SUBSCRIBES (depends on edge direction convention)
            subscribers.extend([u for u, v in self.G.in_edges(node)
                               if self.G.edges[u, v].get('type') == 'SUBSCRIBES_TO'])
            
            subscribers = list(set(subscribers))
            
            if len(subscribers) >= min_subscribers and len(publishers) > 0:
                all_nodes = publishers + [node] + subscribers
                edges = [(p, node) for p in publishers] + [(node, s) for s in subscribers]
                
                # Criticality: more subscribers = higher impact
                criticality = min(1.0, len(subscribers) / 10.0)
                
                motifs.append(MotifInstance(
                    motif_type=MotifType.FAN_OUT,
                    nodes=all_nodes,
                    edges=edges,
                    central_node=node,
                    criticality_impact=criticality,
                    description=f"Topic {node} broadcasts from {len(publishers)} "
                               f"publishers to {len(subscribers)} subscribers"
                ))
        
        return motifs
    
    def _detect_fan_in(self, min_publishers: int = 3) -> List[MotifInstance]:
        """
        Detect fan-in patterns: Many Publishers → Topic → Few Subscribers
        
        Critical because:
        - Topic aggregates data from many sources
        - Subscribers receive combined/processed data
        """
        motifs = []
        
        for node in self.G.nodes():
            if self.G.nodes[node].get('type') != 'Topic':
                continue
            
            publishers = [u for u, v in self.G.in_edges(node)
                         if self.G.edges[u, v].get('type') == 'PUBLISHES_TO']
            
            subscribers = [v for u, v in self.G.out_edges(node)
                          if self.G.edges[u, v].get('type') == 'SUBSCRIBES_TO']
            subscribers.extend([u for u, v in self.G.in_edges(node)
                               if self.G.edges[u, v].get('type') == 'SUBSCRIBES_TO'])
            subscribers = list(set(subscribers))
            
            # Fan-in: many publishers, few subscribers
            if len(publishers) >= min_publishers and 0 < len(subscribers) <= 2:
                all_nodes = publishers + [node] + subscribers
                edges = [(p, node) for p in publishers] + [(node, s) for s in subscribers]
                
                criticality = min(1.0, len(publishers) / 10.0)
                
                motifs.append(MotifInstance(
                    motif_type=MotifType.FAN_IN,
                    nodes=all_nodes,
                    edges=edges,
                    central_node=node,
                    criticality_impact=criticality,
                    description=f"Topic {node} aggregates from {len(publishers)} "
                               f"publishers to {len(subscribers)} subscribers"
                ))
        
        return motifs
    
    def _detect_chains(self, min_length: int = 3) -> List[MotifInstance]:
        """
        Detect chain patterns: App → Topic → App → Topic → App
        
        Critical because:
        - Sequential dependencies create latency
        - Single failure breaks the chain
        """
        motifs = []
        
        # Find applications that both subscribe and publish
        relay_apps = []
        for node in self.G.nodes():
            if self.G.nodes[node].get('type') != 'Application':
                continue
            
            has_sub = any(self.G.edges[e].get('type') == 'SUBSCRIBES_TO' 
                         for e in self.G.in_edges(node))
            has_pub = any(self.G.edges[e].get('type') == 'PUBLISHES_TO'
                         for e in self.G.out_edges(node))
            
            if has_sub and has_pub:
                relay_apps.append(node)
        
        # Find chains through relay apps
        visited_chains = set()
        for start_app in relay_apps:
            chain = self._trace_chain(start_app, set())
            if len(chain) >= min_length:
                chain_key = tuple(sorted(chain))
                if chain_key not in visited_chains:
                    visited_chains.add(chain_key)
                    
                    criticality = min(1.0, len(chain) / 10.0)
                    
                    motifs.append(MotifInstance(
                        motif_type=MotifType.CHAIN,
                        nodes=chain,
                        edges=[],  # Complex to enumerate
                        central_node=chain[len(chain)//2] if chain else None,
                        criticality_impact=criticality,
                        description=f"Data processing chain of length {len(chain)}"
                    ))
        
        return motifs
    
    def _trace_chain(self, start: str, visited: Set[str], max_depth: int = 10) -> List[str]:
        """Trace a chain starting from given node"""
        if start in visited or len(visited) > max_depth:
            return []
        
        visited.add(start)
        chain = [start]
        
        # Find downstream via publish → subscribe pattern
        for _, topic in self.G.out_edges(start):
            if self.G.nodes[topic].get('type') != 'Topic':
                continue
            
            for _, subscriber in self.G.out_edges(topic):
                if self.G.nodes[subscriber].get('type') == 'Application':
                    sub_chain = self._trace_chain(subscriber, visited, max_depth)
                    if len(sub_chain) > len(chain) - 1:
                        chain = [start] + sub_chain
        
        return chain
    
    def _detect_diamonds(self) -> List[MotifInstance]:
        """
        Detect diamond patterns: Multiple paths between same source/sink
        
        Positive pattern - indicates redundancy
        """
        motifs = []
        
        # Find nodes with multiple paths to same destination
        applications = [n for n in self.G.nodes() 
                       if self.G.nodes[n].get('type') == 'Application']
        
        for source in applications:
            destinations = defaultdict(list)
            
            # Find all 2-hop paths through topics
            for _, topic in self.G.out_edges(source):
                if self.G.nodes.get(topic, {}).get('type') != 'Topic':
                    continue
                
                for _, dest in self.G.out_edges(topic):
                    if dest != source and self.G.nodes.get(dest, {}).get('type') == 'Application':
                        destinations[dest].append(topic)
            
            # Diamond exists if multiple paths to same destination
            for dest, topics in destinations.items():
                if len(topics) >= 2:
                    all_nodes = [source] + topics + [dest]
                    
                    # Diamonds are positive (redundancy) so low criticality
                    criticality = 0.3
                    
                    motifs.append(MotifInstance(
                        motif_type=MotifType.DIAMOND,
                        nodes=all_nodes,
                        edges=[],
                        central_node=None,
                        criticality_impact=criticality,
                        description=f"Redundant paths from {source} to {dest} "
                                   f"via {len(topics)} topics"
                    ))
        
        return motifs
    
    def _detect_stars(self, min_connections: int = 5) -> List[MotifInstance]:
        """
        Detect star patterns: Central node with many connections
        
        Critical because:
        - Central node is a hub/bottleneck
        - Failure affects many components
        """
        motifs = []
        
        for node in self.G.nodes():
            degree = self.G.degree(node)
            if degree >= min_connections:
                neighbors = list(self.G.predecessors(node)) + list(self.G.successors(node))
                
                criticality = min(1.0, degree / 20.0)
                
                motifs.append(MotifInstance(
                    motif_type=MotifType.STAR,
                    nodes=[node] + neighbors,
                    edges=list(self.G.in_edges(node)) + list(self.G.out_edges(node)),
                    central_node=node,
                    criticality_impact=criticality,
                    description=f"Star pattern centered on {node} with {degree} connections"
                ))
        
        return motifs


# ============================================================================
# Dependency Chain Analysis
# ============================================================================

class DependencyChainAnalyzer:
    """
    Analyzes dependency chains and hierarchies in pub-sub systems.
    
    Key metrics:
    - Transitive depth: How deep in the dependency tree
    - Fan-out/fan-in ratio: Foundational vs. leaf component
    - Chain criticality: Based on position and connectivity
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self.logger = logging.getLogger('dependency_chain')
    
    def analyze_all(self) -> Dict[str, DependencyChainResult]:
        """Analyze dependency chains for all nodes"""
        self.logger.info("Analyzing dependency chains...")
        
        results = {}
        
        for node in self.G.nodes():
            # Get upstream (what this node depends on)
            upstream = set(nx.ancestors(self.G, node))
            
            # Get downstream (what depends on this node)
            downstream = set(nx.descendants(self.G, node))
            
            # Calculate transitive depth
            depth = self._calculate_depth(node)
            
            # Calculate ratio
            upstream_count = len(upstream)
            downstream_count = len(downstream)
            ratio = downstream_count / max(upstream_count, 1)
            
            # Find longest chains
            longest_up = self._find_longest_path_to(node)
            longest_down = self._find_longest_path_from(node)
            
            # Calculate chain criticality
            criticality = self._calculate_chain_criticality(
                depth, upstream_count, downstream_count, 
                len(longest_up), len(longest_down)
            )
            
            results[node] = DependencyChainResult(
                node_id=node,
                transitive_depth=depth,
                upstream_count=upstream_count,
                downstream_count=downstream_count,
                dependency_ratio=ratio,
                chain_criticality=criticality,
                longest_upstream_chain=longest_up,
                longest_downstream_chain=longest_down
            )
        
        return results
    
    def get_deepest_chains(self, results: Dict[str, DependencyChainResult], 
                           n: int = 5) -> List[str]:
        """Get nodes with deepest dependency chains"""
        sorted_nodes = sorted(
            results.keys(),
            key=lambda x: results[x].transitive_depth,
            reverse=True
        )
        return sorted_nodes[:n]
    
    def get_highest_fan_out(self, results: Dict[str, DependencyChainResult],
                            n: int = 5) -> List[str]:
        """Get nodes with highest downstream dependency count"""
        sorted_nodes = sorted(
            results.keys(),
            key=lambda x: results[x].downstream_count,
            reverse=True
        )
        return sorted_nodes[:n]
    
    def _calculate_depth(self, node: str) -> int:
        """Calculate maximum depth in dependency tree"""
        try:
            # Depth = longest path from any root to this node
            roots = [n for n in self.G.nodes() if self.G.in_degree(n) == 0]
            max_depth = 0
            
            for root in roots:
                try:
                    path = nx.shortest_path(self.G, root, node)
                    max_depth = max(max_depth, len(path) - 1)
                except nx.NetworkXNoPath:
                    continue
            
            return max_depth
        except:
            return 0
    
    def _find_longest_path_to(self, node: str) -> List[str]:
        """Find longest path leading to this node"""
        try:
            # Get all ancestors
            ancestors = nx.ancestors(self.G, node)
            if not ancestors:
                return [node]
            
            # Find the ancestor with longest path
            longest = [node]
            for ancestor in ancestors:
                try:
                    path = nx.shortest_path(self.G, ancestor, node)
                    if len(path) > len(longest):
                        longest = path
                except nx.NetworkXNoPath:
                    continue
            
            return longest
        except:
            return [node]
    
    def _find_longest_path_from(self, node: str) -> List[str]:
        """Find longest path from this node"""
        try:
            descendants = nx.descendants(self.G, node)
            if not descendants:
                return [node]
            
            longest = [node]
            for descendant in descendants:
                try:
                    path = nx.shortest_path(self.G, node, descendant)
                    if len(path) > len(longest):
                        longest = path
                except nx.NetworkXNoPath:
                    continue
            
            return longest
        except:
            return [node]
    
    def _calculate_chain_criticality(self, depth: int, upstream: int, 
                                     downstream: int, up_chain: int,
                                     down_chain: int) -> float:
        """
        Calculate criticality based on dependency chain position.
        
        High criticality:
        - Many downstream dependents
        - Central position in long chains
        - High fan-out ratio
        """
        # Downstream impact (normalized to 0-1)
        downstream_score = min(1.0, downstream / 20.0)
        
        # Chain position (middle of chain is most critical)
        total_chain = up_chain + down_chain
        if total_chain > 0:
            position_score = 1.0 - abs(0.5 - up_chain / total_chain)
        else:
            position_score = 0.0
        
        # Depth score (deeper = more dependent)
        depth_score = min(1.0, depth / 10.0)
        
        # Composite
        criticality = (
            0.5 * downstream_score +
            0.3 * position_score +
            0.2 * depth_score
        )
        
        return criticality


# ============================================================================
# Multi-Layer Correlation Analysis
# ============================================================================

class LayerCorrelationAnalyzer:
    """
    Analyzes correlations between graph layers (Application, Topic, Broker, Node).
    
    Identifies:
    - Centrality correlation between layers
    - Coupling coefficients
    - Misaligned components (critical app on non-critical infrastructure)
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self.logger = logging.getLogger('layer_correlation')
    
    def analyze_correlations(self) -> List[LayerCorrelationResult]:
        """Analyze correlations between all layer pairs"""
        self.logger.info("Analyzing layer correlations...")
        
        # Get nodes by type
        layers = self._group_by_type()
        
        # Compute centrality for all nodes
        betweenness = nx.betweenness_centrality(self.G, normalized=True)
        
        results = []
        layer_types = list(layers.keys())
        
        for i, layer1 in enumerate(layer_types):
            for layer2 in layer_types[i+1:]:
                correlation = self._compute_centrality_correlation(
                    layers[layer1], layers[layer2], betweenness
                )
                coupling = self._compute_coupling(layer1, layer2, layers)
                misaligned = self._find_misaligned(
                    layers[layer1], layers[layer2], betweenness
                )
                
                interpretation = self._interpret_correlation(
                    layer1, layer2, correlation, coupling, len(misaligned)
                )
                
                results.append(LayerCorrelationResult(
                    layer1=layer1,
                    layer2=layer2,
                    centrality_correlation=correlation,
                    coupling_coefficient=coupling,
                    misaligned_components=misaligned,
                    interpretation=interpretation
                ))
        
        return results
    
    def _group_by_type(self) -> Dict[str, List[str]]:
        """Group nodes by their type"""
        groups = defaultdict(list)
        for node in self.G.nodes():
            node_type = self.G.nodes[node].get('type', 'Unknown')
            groups[node_type].append(node)
        return dict(groups)
    
    def _compute_centrality_correlation(self, layer1_nodes: List[str],
                                         layer2_nodes: List[str],
                                         centrality: Dict[str, float]) -> float:
        """
        Compute correlation between centrality scores of connected nodes
        in different layers.
        """
        pairs = []
        
        for n1 in layer1_nodes:
            for n2 in layer2_nodes:
                if self.G.has_edge(n1, n2) or self.G.has_edge(n2, n1):
                    pairs.append((centrality.get(n1, 0), centrality.get(n2, 0)))
        
        if len(pairs) < 2:
            return 0.0
        
        # Simple Pearson correlation
        x_vals = [p[0] for p in pairs]
        y_vals = [p[1] for p in pairs]
        
        return self._pearson(x_vals, y_vals)
    
    def _compute_coupling(self, layer1: str, layer2: str,
                          layers: Dict[str, List[str]]) -> float:
        """
        Compute coupling coefficient between two layers.
        Higher coupling = more interconnected.
        """
        layer1_nodes = set(layers.get(layer1, []))
        layer2_nodes = set(layers.get(layer2, []))
        
        if not layer1_nodes or not layer2_nodes:
            return 0.0
        
        cross_edges = 0
        for u, v in self.G.edges():
            if (u in layer1_nodes and v in layer2_nodes) or \
               (u in layer2_nodes and v in layer1_nodes):
                cross_edges += 1
        
        max_edges = len(layer1_nodes) * len(layer2_nodes)
        return cross_edges / max_edges if max_edges > 0 else 0.0
    
    def _find_misaligned(self, layer1_nodes: List[str], layer2_nodes: List[str],
                         centrality: Dict[str, float],
                         threshold: float = 0.3) -> List[Tuple[str, str, str]]:
        """
        Find misaligned components where high-criticality node in one layer
        connects to low-criticality node in another.
        """
        misaligned = []
        
        for n1 in layer1_nodes:
            c1 = centrality.get(n1, 0)
            
            for n2 in layer2_nodes:
                if not (self.G.has_edge(n1, n2) or self.G.has_edge(n2, n1)):
                    continue
                
                c2 = centrality.get(n2, 0)
                
                # Check for significant mismatch
                if abs(c1 - c2) > threshold:
                    if c1 > c2:
                        reason = f"High-criticality {n1} depends on low-criticality {n2}"
                    else:
                        reason = f"Low-criticality {n1} supports high-criticality {n2}"
                    misaligned.append((n1, n2, reason))
        
        return misaligned
    
    def _pearson(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        n = len(x)
        if n < 2:
            return 0.0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        den_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
        den_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
        
        if den_x * den_y == 0:
            return 0.0
        
        return num / (den_x * den_y)
    
    def _interpret_correlation(self, layer1: str, layer2: str,
                               correlation: float, coupling: float,
                               misaligned_count: int) -> str:
        """Generate interpretation of correlation analysis"""
        parts = []
        
        if correlation > 0.7:
            parts.append(f"Strong centrality alignment between {layer1} and {layer2}")
        elif correlation > 0.3:
            parts.append(f"Moderate centrality alignment")
        else:
            parts.append(f"Weak centrality alignment - potential hidden risks")
        
        if coupling > 0.5:
            parts.append("Tightly coupled layers")
        elif coupling > 0.1:
            parts.append("Moderately coupled")
        else:
            parts.append("Loosely coupled layers")
        
        if misaligned_count > 0:
            parts.append(f"⚠️ {misaligned_count} misaligned component pairs")
        
        return "; ".join(parts)


# ============================================================================
# Ensemble Criticality Scorer
# ============================================================================

class EnsembleCriticalityScorer:
    """
    Combines multiple algorithm perspectives into ensemble criticality scores.
    
    Algorithms combined:
    - Betweenness centrality (routing importance)
    - PageRank (influence importance)
    - HITS (hub/authority importance)
    - Articulation points (structural importance)
    - K-core (connectivity importance)
    - Closeness (accessibility importance)
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self.G_undirected = graph.to_undirected()
        self.logger = logging.getLogger('ensemble_criticality')
    
    def score_all(self, 
                  weights: Optional[Dict[str, float]] = None) -> Dict[str, EnsembleCriticalityResult]:
        """
        Calculate ensemble criticality scores for all nodes.
        
        Args:
            weights: Custom weights for each algorithm (sums to 1.0)
            
        Returns:
            Dictionary mapping node IDs to EnsembleCriticalityResult
        """
        self.logger.info("Computing ensemble criticality scores...")
        
        # Default weights
        if weights is None:
            weights = {
                'betweenness': 0.25,
                'pagerank': 0.20,
                'hits_hub': 0.10,
                'hits_auth': 0.10,
                'articulation': 0.20,
                'kcore': 0.10,
                'closeness': 0.05
            }
        
        # Compute all metrics
        betweenness = nx.betweenness_centrality(self.G, normalized=True)
        pagerank = nx.pagerank(self.G, alpha=0.85)
        
        try:
            hubs, authorities = nx.hits(self.G, max_iter=100)
        except:
            hubs = {n: 0.5 for n in self.G.nodes()}
            authorities = {n: 0.5 for n in self.G.nodes()}
        
        articulation_points = set(nx.articulation_points(self.G_undirected))
        kcore = nx.core_number(self.G_undirected)
        closeness = nx.closeness_centrality(self.G)
        
        # Normalize k-core values
        max_kcore = max(kcore.values()) if kcore else 1
        kcore_norm = {n: v / max_kcore for n, v in kcore.items()}
        
        # Determine thresholds
        bc_thresh = self._get_percentile(list(betweenness.values()), 0.75)
        pr_thresh = self._get_percentile(list(pagerank.values()), 0.75)
        hub_thresh = self._get_percentile(list(hubs.values()), 0.75)
        auth_thresh = self._get_percentile(list(authorities.values()), 0.75)
        
        results = {}
        
        for node in self.G.nodes():
            node_type = self.G.nodes[node].get('type', 'Unknown')
            
            # Get individual scores
            bc = betweenness.get(node, 0)
            pr = pagerank.get(node, 0)
            hub = hubs.get(node, 0)
            auth = authorities.get(node, 0)
            is_ap = node in articulation_points
            kc = kcore_norm.get(node, 0)
            cc = closeness.get(node, 0)
            
            # Determine role
            role = self._determine_role(hub, auth, hub_thresh, auth_thresh)
            
            # Determine criticality indicators
            is_structural = is_ap or bc > bc_thresh
            is_flow = pr > pr_thresh or hub > hub_thresh or auth > auth_thresh
            is_connectivity = cc > 0.5 or kc > 0.5
            
            # Compute ensemble score
            ap_score = 1.0 if is_ap else 0.0
            ensemble = (
                weights['betweenness'] * bc +
                weights['pagerank'] * pr +
                weights['hits_hub'] * hub +
                weights['hits_auth'] * auth +
                weights['articulation'] * ap_score +
                weights['kcore'] * kc +
                weights['closeness'] * cc
            )
            
            # Determine level
            if ensemble >= 0.6 or is_ap:
                level = "CRITICAL"
            elif ensemble >= 0.4:
                level = "HIGH"
            elif ensemble >= 0.2:
                level = "MEDIUM"
            else:
                level = "LOW"
            
            # Calculate confidence (agreement between algorithms)
            high_scores = sum([
                bc > bc_thresh,
                pr > pr_thresh,
                hub > hub_thresh,
                auth > auth_thresh,
                is_ap,
                kc > 0.5,
                cc > 0.5
            ])
            confidence = high_scores / 7.0
            
            # Generate reasons
            primary, secondary = self._generate_reasons(
                bc, pr, hub, auth, is_ap, kc, cc,
                bc_thresh, pr_thresh, hub_thresh, auth_thresh
            )
            
            results[node] = EnsembleCriticalityResult(
                node_id=node,
                node_type=node_type,
                betweenness_score=bc,
                pagerank_score=pr,
                hits_hub_score=hub,
                hits_authority_score=auth,
                is_articulation_point=is_ap,
                kcore_number=kcore.get(node, 0),
                closeness_score=cc,
                role=role,
                is_structural_critical=is_structural,
                is_flow_critical=is_flow,
                is_connectivity_critical=is_connectivity,
                ensemble_score=ensemble,
                ensemble_level=level,
                confidence=confidence,
                primary_reasons=primary,
                secondary_indicators=secondary
            )
        
        return results
    
    def _get_percentile(self, values: List[float], p: float) -> float:
        """Get value at percentile p"""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * p)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]
    
    def _determine_role(self, hub: float, auth: float,
                        hub_thresh: float, auth_thresh: float) -> ComponentRole:
        """Determine component role based on HITS"""
        high_hub = hub >= hub_thresh
        high_auth = auth >= auth_thresh
        
        if high_hub and high_auth:
            return ComponentRole.RELAY
        elif high_hub:
            return ComponentRole.PURE_PUBLISHER
        elif high_auth:
            return ComponentRole.PURE_SUBSCRIBER
        else:
            return ComponentRole.PERIPHERAL
    
    def _generate_reasons(self, bc, pr, hub, auth, is_ap, kc, cc,
                          bc_thresh, pr_thresh, hub_thresh, auth_thresh) -> Tuple[List[str], List[str]]:
        """Generate primary and secondary criticality reasons"""
        primary = []
        secondary = []
        
        if is_ap:
            primary.append("Articulation point - single point of failure")
        
        if bc > bc_thresh:
            primary.append(f"High betweenness ({bc:.3f}) - routing bottleneck")
        elif bc > bc_thresh * 0.5:
            secondary.append(f"Moderate betweenness ({bc:.3f})")
        
        if pr > pr_thresh:
            primary.append(f"High PageRank ({pr:.3f}) - influential node")
        elif pr > pr_thresh * 0.5:
            secondary.append(f"Moderate PageRank ({pr:.3f})")
        
        if hub > hub_thresh:
            primary.append(f"High hub score ({hub:.3f}) - key data source")
        
        if auth > auth_thresh:
            primary.append(f"High authority ({auth:.3f}) - key data sink")
        
        if kc > 0.7:
            secondary.append(f"High k-core ({kc:.2f}) - core topology")
        
        if cc > 0.6:
            secondary.append(f"High closeness ({cc:.3f}) - well-connected")
        
        return primary, secondary


# ============================================================================
# Main Relationship Analyzer
# ============================================================================

class RelationshipAnalyzer:
    """
    Main class for comprehensive relationship analysis.
    
    Combines all analysis components:
    - Edge criticality
    - HITS role analysis
    - Motif detection
    - Dependency chains
    - Layer correlations
    - Ensemble criticality
    """
    
    def __init__(self, graph: nx.DiGraph):
        """
        Initialize the relationship analyzer.
        
        Args:
            graph: NetworkX directed graph with node 'type' attributes
        """
        self.G = graph
        self.logger = logging.getLogger('relationship_analyzer')
        
        # Component analyzers
        self.edge_analyzer = EdgeCriticalityAnalyzer(graph)
        self.hits_analyzer = HITSRoleAnalyzer(graph)
        self.motif_detector = MotifDetector(graph)
        self.chain_analyzer = DependencyChainAnalyzer(graph)
        self.layer_analyzer = LayerCorrelationAnalyzer(graph)
        self.ensemble_scorer = EnsembleCriticalityScorer(graph)
    
    def analyze(self,
                include_edge_analysis: bool = True,
                include_hits: bool = True,
                include_motifs: bool = True,
                include_chains: bool = True,
                include_layers: bool = True,
                include_ensemble: bool = True) -> RelationshipAnalysisResult:
        """
        Run comprehensive relationship analysis.
        
        Args:
            include_edge_analysis: Analyze edge criticality
            include_hits: Run HITS role analysis
            include_motifs: Detect network motifs
            include_chains: Analyze dependency chains
            include_layers: Analyze layer correlations
            include_ensemble: Compute ensemble criticality
            
        Returns:
            RelationshipAnalysisResult with all analysis outputs
        """
        from datetime import datetime
        
        self.logger.info(f"Starting relationship analysis on graph with "
                        f"{self.G.number_of_nodes()} nodes and "
                        f"{self.G.number_of_edges()} edges")
        
        # Edge Analysis
        edge_results = []
        critical_edges = []
        bridge_edges = []
        if include_edge_analysis:
            self.logger.info("Running edge criticality analysis...")
            edge_results = self.edge_analyzer.analyze_all_edges()
            critical_edges = [(r.edge[0], r.edge[1]) for r in edge_results 
                             if r.criticality_level == "CRITICAL"]
            bridge_edges = [(r.edge[0], r.edge[1]) for r in edge_results if r.is_bridge]
        
        # HITS Analysis
        hits_roles = {}
        top_hubs = []
        top_authorities = []
        if include_hits:
            self.logger.info("Running HITS role analysis...")
            hits_roles = self.hits_analyzer.analyze_roles()
            top_hubs = self.hits_analyzer.get_top_hubs(10)
            top_authorities = self.hits_analyzer.get_top_authorities(10)
        
        # Motif Detection
        motifs = []
        motif_summary = {}
        if include_motifs:
            self.logger.info("Detecting network motifs...")
            motifs = self.motif_detector.detect_all_motifs()
            motif_summary = self.motif_detector.get_motif_summary(motifs)
        
        # Dependency Chain Analysis
        chains = {}
        deepest = []
        highest_fan_out = []
        if include_chains:
            self.logger.info("Analyzing dependency chains...")
            chains = self.chain_analyzer.analyze_all()
            deepest = self.chain_analyzer.get_deepest_chains(chains, 5)
            highest_fan_out = self.chain_analyzer.get_highest_fan_out(chains, 5)
        
        # Layer Correlation Analysis
        layer_correlations = []
        if include_layers:
            self.logger.info("Analyzing layer correlations...")
            layer_correlations = self.layer_analyzer.analyze_correlations()
        
        # Ensemble Criticality
        ensemble = {}
        if include_ensemble:
            self.logger.info("Computing ensemble criticality...")
            ensemble = self.ensemble_scorer.score_all()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            edge_results, hits_roles, motifs, chains, layer_correlations, ensemble
        )
        
        self.logger.info("Relationship analysis complete")
        
        return RelationshipAnalysisResult(
            total_nodes=self.G.number_of_nodes(),
            total_edges=self.G.number_of_edges(),
            analysis_timestamp=datetime.now().isoformat(),
            edge_criticality=edge_results,
            critical_edges=critical_edges,
            bridge_edges=bridge_edges,
            hits_roles=hits_roles,
            top_hubs=top_hubs,
            top_authorities=top_authorities,
            motifs=motifs,
            motif_summary=motif_summary,
            dependency_chains=chains,
            deepest_chains=deepest,
            highest_fan_out=highest_fan_out,
            layer_correlations=layer_correlations,
            ensemble_criticality=ensemble,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, edges, hits, motifs, chains, 
                                   layers, ensemble) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Bridge edge recommendations
        bridge_count = sum(1 for e in edges if e.is_bridge)
        if bridge_count > 0:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Redundancy',
                'issue': f'{bridge_count} bridge edges detected',
                'recommendation': 'Add redundant connections to eliminate single points of failure',
                'affected_edges': [(e.edge[0], e.edge[1]) for e in edges if e.is_bridge]
            })
        
        # Fan-out pattern recommendations
        fan_out_count = sum(1 for m in motifs if m.motif_type == MotifType.FAN_OUT)
        if fan_out_count > 5:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Architecture',
                'issue': f'{fan_out_count} fan-out patterns (broadcast bottlenecks)',
                'recommendation': 'Consider partitioning high-fanout topics or adding intermediate aggregation',
                'details': 'Fan-out patterns create single points of failure for many subscribers'
            })
        
        # Chain pattern recommendations
        long_chains = [n for n, c in chains.items() 
                      if c.transitive_depth > 5] if chains else []
        if long_chains:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Dependencies',
                'issue': f'{len(long_chains)} nodes with deep dependency chains (>5 hops)',
                'recommendation': 'Review dependency structure to reduce cascading failure risk',
                'affected_nodes': long_chains[:5]
            })
        
        # Layer misalignment recommendations
        total_misaligned = sum(len(l.misaligned_components) for l in layers)
        if total_misaligned > 0:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Infrastructure',
                'issue': f'{total_misaligned} layer misalignments detected',
                'recommendation': 'Ensure critical applications run on appropriately provisioned infrastructure',
                'details': 'High-criticality components should not depend on low-criticality infrastructure'
            })
        
        # High ensemble criticality recommendations
        critical_count = sum(1 for e in ensemble.values() 
                            if e.ensemble_level == 'CRITICAL') if ensemble else 0
        if critical_count > 5:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Risk',
                'issue': f'{critical_count} components with CRITICAL ensemble score',
                'recommendation': 'Prioritize redundancy and monitoring for these components',
                'affected_nodes': [n for n, e in ensemble.items() 
                                  if e.ensemble_level == 'CRITICAL'][:10]
            })
        
        return recommendations
    
    def to_dict(self, result: RelationshipAnalysisResult) -> Dict[str, Any]:
        """Convert analysis result to dictionary for JSON serialization"""
        return {
            'summary': {
                'total_nodes': result.total_nodes,
                'total_edges': result.total_edges,
                'timestamp': result.analysis_timestamp
            },
            'edge_analysis': {
                'total_analyzed': len(result.edge_criticality),
                'critical_edges': len(result.critical_edges),
                'bridge_edges': len(result.bridge_edges),
                'top_critical': [
                    {
                        'edge': e.edge,
                        'type': e.edge_type,
                        'score': e.criticality_score,
                        'is_bridge': e.is_bridge
                    }
                    for e in result.edge_criticality[:10]
                ]
            },
            'hits_analysis': {
                'top_hubs': result.top_hubs,
                'top_authorities': result.top_authorities,
                'roles': {
                    node: {
                        'hub_score': r.hub_score,
                        'authority_score': r.authority_score,
                        'role': r.role.value
                    }
                    for node, r in list(result.hits_roles.items())[:20]
                }
            },
            'motif_analysis': {
                'summary': result.motif_summary,
                'total_motifs': len(result.motifs),
                'examples': [
                    {
                        'type': m.motif_type.value,
                        'central_node': m.central_node,
                        'criticality': m.criticality_impact
                    }
                    for m in result.motifs[:10]
                ]
            },
            'dependency_analysis': {
                'deepest_chains': result.deepest_chains,
                'highest_fan_out': result.highest_fan_out
            },
            'layer_analysis': [
                {
                    'layers': f"{l.layer1} <-> {l.layer2}",
                    'correlation': l.centrality_correlation,
                    'coupling': l.coupling_coefficient,
                    'misaligned': len(l.misaligned_components)
                }
                for l in result.layer_correlations
            ],
            'ensemble_criticality': {
                'by_level': {
                    level: sum(1 for e in result.ensemble_criticality.values() 
                              if e.ensemble_level == level)
                    for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
                },
                'top_critical': [
                    {
                        'node': n,
                        'type': e.node_type,
                        'score': e.ensemble_score,
                        'level': e.ensemble_level,
                        'reasons': e.primary_reasons
                    }
                    for n, e in sorted(
                        result.ensemble_criticality.items(),
                        key=lambda x: x[1].ensemble_score,
                        reverse=True
                    )[:10]
                ]
            },
            'recommendations': result.recommendations
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def analyze_relationships(graph: nx.DiGraph, **kwargs) -> RelationshipAnalysisResult:
    """
    Convenience function for relationship analysis.
    
    Args:
        graph: NetworkX directed graph
        **kwargs: Options passed to RelationshipAnalyzer.analyze()
        
    Returns:
        RelationshipAnalysisResult
    """
    analyzer = RelationshipAnalyzer(graph)
    return analyzer.analyze(**kwargs)


def get_algorithm_recommendations() -> List[Dict[str, Any]]:
    """
    Get recommendations for which algorithms to apply and why.
    
    Returns:
        List of algorithm recommendations with pub-sub applications
    """
    return [
        {
            'algorithm': 'Betweenness Centrality',
            'category': 'Centrality',
            'priority': 1,
            'purpose': 'Find nodes on shortest paths between others',
            'pub_sub_application': 'Identify critical routing points. High BC brokers/topics are bottlenecks.',
            'complexity': 'O(VE)'
        },
        {
            'algorithm': 'Edge Betweenness',
            'category': 'Centrality',
            'priority': 1,
            'purpose': 'Find critical edges/relationships',
            'pub_sub_application': 'Identify critical message pathways. High-BC edges are single points of failure.',
            'complexity': 'O(VE)'
        },
        {
            'algorithm': 'HITS (Hubs & Authorities)',
            'category': 'Centrality',
            'priority': 1,
            'purpose': 'Find hubs (many outlinks) and authorities (many inlinks)',
            'pub_sub_application': 'PERFECT for pub-sub! Publishers are hubs, subscribers are authorities.',
            'complexity': 'O(k·E)'
        },
        {
            'algorithm': 'PageRank',
            'category': 'Centrality',
            'priority': 1,
            'purpose': 'Importance based on recursive neighbor importance',
            'pub_sub_application': 'Find influential topics/apps. High PageRank = receives from important sources.',
            'complexity': 'O(k·E)'
        },
        {
            'algorithm': 'Articulation Points',
            'category': 'Structural',
            'priority': 1,
            'purpose': 'Find nodes whose removal disconnects graph',
            'pub_sub_application': 'Single points of failure. Must-have for reliability analysis.',
            'complexity': 'O(V+E)'
        },
        {
            'algorithm': 'Bridges',
            'category': 'Structural',
            'priority': 1,
            'purpose': 'Find edges whose removal disconnects graph',
            'pub_sub_application': 'Critical connections. If a PUBLISHES edge is a bridge, that relationship is critical.',
            'complexity': 'O(V+E)'
        },
        {
            'algorithm': 'K-Core Decomposition',
            'category': 'Structural',
            'priority': 2,
            'purpose': 'Find hierarchical dense subgraphs',
            'pub_sub_application': 'Identify core vs periphery. High k-core = tightly coupled backbone.',
            'complexity': 'O(V+E)'
        },
        {
            'algorithm': 'Network Motifs',
            'category': 'Pattern',
            'priority': 2,
            'purpose': 'Detect recurring subgraph patterns',
            'pub_sub_application': 'Fan-out, fan-in, chains, diamonds reveal architectural patterns.',
            'complexity': 'O(V^k) for k-node motifs'
        },
        {
            'algorithm': 'Dependency Chains',
            'category': 'Flow',
            'priority': 2,
            'purpose': 'Analyze transitive dependencies',
            'pub_sub_application': 'Find cascading failure risks. Deep chains = high latency and fragility.',
            'complexity': 'O(V+E)'
        },
        {
            'algorithm': 'Layer Correlation',
            'category': 'Multi-layer',
            'priority': 3,
            'purpose': 'Correlate metrics across graph layers',
            'pub_sub_application': 'Ensure critical apps run on robust infrastructure.',
            'complexity': 'O(V²)'
        }
    ]


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI entry point for relationship analysis"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description='Analyze relationships in pub-sub system graphs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', '-i', required=True,
                        help='Input graph file (JSON)')
    parser.add_argument('--output', '-o',
                        help='Output file for results (JSON)')
    parser.add_argument('--algorithms', action='store_true',
                        help='Print algorithm recommendations')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.algorithms:
        print("\n=== Graph Algorithm Recommendations for Pub-Sub Analysis ===\n")
        for rec in get_algorithm_recommendations():
            print(f"[{rec['priority']}] {rec['algorithm']} ({rec['category']})")
            print(f"    Purpose: {rec['purpose']}")
            print(f"    Pub-Sub: {rec['pub_sub_application']}")
            print(f"    Complexity: {rec['complexity']}\n")
        return 0
    
    # Load graph
    with open(args.input) as f:
        data = json.load(f)
    
    # Build NetworkX graph from data
    G = nx.DiGraph()
    
    # Add nodes
    for app in data.get('applications', []):
        G.add_node(app['id'], type='Application', **app)
    for broker in data.get('brokers', []):
        G.add_node(broker['id'], type='Broker', **broker)
    for topic in data.get('topics', []):
        G.add_node(topic['id'], type='Topic', **topic)
    for node in data.get('nodes', []):
        G.add_node(node['id'], type='Node', **node)
    
    # Add edges
    edges = data.get('edges', data.get('relationships', {}))
    for pub in edges.get('publishes_to', []):
        G.add_edge(pub['from'], pub['to'], type='PUBLISHES_TO')
    for sub in edges.get('subscribes_to', []):
        G.add_edge(sub['from'], sub['to'], type='SUBSCRIBES_TO')
    for route in edges.get('routes', []):
        G.add_edge(route['from'], route['to'], type='ROUTES')
    for runs in edges.get('runs_on', []):
        G.add_edge(runs['from'], runs['to'], type='RUNS_ON')
    
    # Run analysis
    analyzer = RelationshipAnalyzer(G)
    result = analyzer.analyze()
    
    # Output results
    result_dict = analyzer.to_dict(result)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        print(f"Results written to {args.output}")
    else:
        print(json.dumps(result_dict, indent=2, default=str))
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())