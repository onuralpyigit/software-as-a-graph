#!/usr/bin/env python3
"""
Advanced Graph Algorithms for Pub-Sub System Analysis
======================================================

This module provides a comprehensive suite of graph algorithms specifically
tailored for analyzing distributed publish-subscribe systems. It goes beyond
basic centrality measures to provide deep insights into system structure,
message flow patterns, and potential vulnerabilities.

Algorithm Categories:
1. CENTRALITY MEASURES - Identify critical nodes from different perspectives
2. STRUCTURAL ANALYSIS - Find structural vulnerabilities and patterns
3. COMMUNITY DETECTION - Identify subsystems and clusters
4. PATH & FLOW ANALYSIS - Analyze message routing and dependencies
5. SIMILARITY & COUPLING - Measure component relationships
6. ROBUSTNESS ANALYSIS - Assess system resilience
7. LAYER-AWARE ANALYSIS - Multi-layer graph algorithms

Research Applications:
- Critical component identification
- Bottleneck detection
- Redundancy planning
- Failure impact prediction
- Architecture refactoring recommendations

Author: Software-as-a-Graph Research Project
Version: 1.0
"""

import math
import logging
from typing import Dict, List, Tuple, Set, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import heapq

try:
    import networkx as nx
    from networkx.algorithms import community
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: NetworkX not available")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ============================================================================
# Data Classes for Results
# ============================================================================

@dataclass
class CentralityResult:
    """Result from centrality analysis"""
    algorithm: str
    scores: Dict[str, float]
    top_k: List[Tuple[str, float]]
    statistics: Dict[str, float]
    interpretation: str


@dataclass
class CommunityResult:
    """Result from community detection"""
    algorithm: str
    communities: List[Set[str]]
    modularity: float
    node_to_community: Dict[str, int]
    statistics: Dict[str, Any]


@dataclass
class PathAnalysisResult:
    """Result from path analysis"""
    critical_paths: List[List[str]]
    path_statistics: Dict[str, Any]
    bottlenecks: List[str]
    redundancy_score: float


@dataclass
class RobustnessResult:
    """Result from robustness analysis"""
    attack_type: str
    failure_sequence: List[str]
    connectivity_curve: List[float]
    critical_threshold: float
    auc_robustness: float


@dataclass 
class AlgorithmRecommendation:
    """Recommendation for algorithm application"""
    algorithm: str
    category: str
    purpose: str
    pub_sub_application: str
    complexity: str
    priority: int  # 1=high, 2=medium, 3=low


# ============================================================================
# Centrality Algorithms
# ============================================================================

class CentralityAnalyzer:
    """
    Comprehensive centrality analysis for pub-sub graphs
    
    Implements multiple centrality measures, each capturing different
    aspects of node importance in the message flow network.
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self.logger = logging.getLogger('centrality')
    
    def betweenness_centrality(self, normalized: bool = True, 
                                weight: str = None) -> CentralityResult:
        """
        Betweenness Centrality: Nodes on shortest paths between others
        
        Pub-Sub Application:
        - High BC nodes are critical message routing points
        - Brokers/topics with high BC are potential bottlenecks
        - Failure of high BC nodes disrupts many message flows
        
        Formula: CB(v) = Σ(σst(v)/σst) for all s≠v≠t
        """
        scores = nx.betweenness_centrality(self.G, normalized=normalized, weight=weight)
        
        return self._create_result(
            "betweenness_centrality",
            scores,
            "Measures how often a node lies on shortest paths. "
            "High values indicate critical routing points that, if failed, "
            "would disrupt message flow between many publisher-subscriber pairs."
        )
    
    def pagerank(self, alpha: float = 0.85, max_iter: int = 100) -> CentralityResult:
        """
        PageRank: Importance based on incoming connections from important nodes
        
        Pub-Sub Application:
        - Topics with high PageRank receive messages from important publishers
        - Applications with high PageRank subscribe to important topics
        - Captures "influence propagation" through the pub-sub network
        
        Formula: PR(v) = (1-α)/N + α·Σ(PR(u)/out_degree(u)) for u→v
        """
        scores = nx.pagerank(self.G, alpha=alpha, max_iter=max_iter)
        
        return self._create_result(
            "pagerank",
            scores,
            "Measures node importance based on the importance of nodes pointing to it. "
            "In pub-sub: identifies topics/applications that are central to the overall "
            "information flow, receiving data from or feeding into important components."
        )
    
    def eigenvector_centrality(self, max_iter: int = 100) -> CentralityResult:
        """
        Eigenvector Centrality: Importance based on neighbor importance
        
        Pub-Sub Application:
        - Nodes connected to other important nodes
        - Captures "prestige" in the network
        - Good for finding influential hubs
        
        Formula: xi = (1/λ)·Σ(xj·Aij) where A is adjacency matrix
        """
        try:
            scores = nx.eigenvector_centrality(self.G, max_iter=max_iter)
        except nx.PowerIterationFailedConvergence:
            # Fall back to numpy-based calculation
            scores = nx.eigenvector_centrality_numpy(self.G)
        
        return self._create_result(
            "eigenvector_centrality",
            scores,
            "Measures influence based on connections to other influential nodes. "
            "High values indicate components that are well-connected to other "
            "important parts of the system - central to the 'core' architecture."
        )
    
    def closeness_centrality(self, wf_improved: bool = True) -> CentralityResult:
        """
        Closeness Centrality: How quickly a node can reach all others
        
        Pub-Sub Application:
        - High closeness = efficient message propagation point
        - Good locations for monitoring or aggregation services
        - Low latency potential for reaching subscribers
        
        Formula: CC(v) = (N-1) / Σd(v,u) for all u≠v
        """
        scores = nx.closeness_centrality(self.G, wf_improved=wf_improved)
        
        return self._create_result(
            "closeness_centrality",
            scores,
            "Measures how close a node is to all other nodes. "
            "In pub-sub: identifies optimal locations for monitoring services, "
            "aggregators, or components that need low-latency access to the entire system."
        )
    
    def katz_centrality(self, alpha: float = 0.1, beta: float = 1.0) -> CentralityResult:
        """
        Katz Centrality: Considers all paths, not just shortest
        
        Pub-Sub Application:
        - Accounts for redundant paths in message delivery
        - Better for systems with multiple routing options
        - Captures "reachability influence"
        
        Formula: Katz(v) = Σ(αk · (Ak)vi) for k=1 to ∞
        """
        try:
            scores = nx.katz_centrality(self.G, alpha=alpha, beta=beta)
        except nx.PowerIterationFailedConvergence:
            scores = nx.katz_centrality_numpy(self.G, alpha=alpha, beta=beta)
        
        return self._create_result(
            "katz_centrality",
            scores,
            "Measures influence considering all paths, not just shortest. "
            "Useful for pub-sub systems with redundant routing, capturing "
            "the full 'sphere of influence' of each component."
        )
    
    def hits(self) -> Tuple[CentralityResult, CentralityResult]:
        """
        HITS Algorithm: Hubs and Authorities
        
        Pub-Sub Application (HIGHLY RELEVANT):
        - HUBS: Publishers that feed many important topics
        - AUTHORITIES: Subscribers that consume from important sources
        - Natural fit for pub-sub's asymmetric relationships
        
        This is particularly valuable for pub-sub because:
        - Publishers are natural "hubs" (many outgoing PUBLISHES edges)
        - Topics are natural "authorities" (receive from publishers, feed subscribers)
        """
        hubs, authorities = nx.hits(self.G)
        
        hub_result = self._create_result(
            "hits_hubs",
            hubs,
            "Hub score: measures how well a node points to good authorities. "
            "In pub-sub: identifies key PUBLISHERS that feed important topics. "
            "High hub score = influential data source."
        )
        
        auth_result = self._create_result(
            "hits_authorities",
            authorities,
            "Authority score: measures how well a node is pointed to by good hubs. "
            "In pub-sub: identifies key TOPICS/SUBSCRIBERS that receive from important sources. "
            "High authority = important information sink."
        )
        
        return hub_result, auth_result
    
    def load_centrality(self, normalized: bool = True) -> CentralityResult:
        """
        Load Centrality: Fraction of shortest paths through node
        
        Similar to betweenness but normalized differently.
        Good for capacity planning.
        """
        scores = nx.load_centrality(self.G, normalized=normalized)
        
        return self._create_result(
            "load_centrality",
            scores,
            "Measures the load placed on a node based on message routing. "
            "Useful for capacity planning - identifies nodes that will handle "
            "the most message traffic and may need scaling."
        )
    
    def harmonic_centrality(self) -> CentralityResult:
        """
        Harmonic Centrality: Variant of closeness for disconnected graphs
        
        Formula: HC(v) = Σ(1/d(v,u)) for all u≠v
        """
        scores = nx.harmonic_centrality(self.G)
        
        return self._create_result(
            "harmonic_centrality",
            scores,
            "Harmonic mean of distances to other nodes. "
            "Works well for graphs with multiple components. "
            "Identifies nodes that are 'locally central' even if the graph is fragmented."
        )
    
    def degree_centrality_analysis(self) -> Dict[str, CentralityResult]:
        """
        Comprehensive degree analysis: in-degree, out-degree, total
        
        Pub-Sub Application:
        - In-degree: How many sources feed this node (subscriber behavior)
        - Out-degree: How many targets this node feeds (publisher behavior)
        - Total: Overall connectivity
        """
        in_deg = nx.in_degree_centrality(self.G)
        out_deg = nx.out_degree_centrality(self.G)
        total_deg = {n: (in_deg.get(n, 0) + out_deg.get(n, 0)) / 2 
                     for n in self.G.nodes()}
        
        return {
            'in_degree': self._create_result(
                "in_degree_centrality", in_deg,
                "Measures incoming connections. High in-degree nodes are "
                "major consumers/subscribers in the pub-sub network."
            ),
            'out_degree': self._create_result(
                "out_degree_centrality", out_deg,
                "Measures outgoing connections. High out-degree nodes are "
                "major producers/publishers in the pub-sub network."
            ),
            'total_degree': self._create_result(
                "total_degree_centrality", total_deg,
                "Combined connectivity measure. High total degree indicates "
                "highly connected components that are central to the topology."
            )
        }
    
    def _create_result(self, algorithm: str, scores: Dict[str, float],
                       interpretation: str) -> CentralityResult:
        """Create standardized result object"""
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        values = list(scores.values())
        stats = {
            'mean': sum(values) / len(values) if values else 0,
            'max': max(values) if values else 0,
            'min': min(values) if values else 0,
            'std': self._std(values) if values else 0
        }
        
        return CentralityResult(
            algorithm=algorithm,
            scores=scores,
            top_k=sorted_scores[:10],
            statistics=stats,
            interpretation=interpretation
        )
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def run_all(self) -> Dict[str, CentralityResult]:
        """Run all centrality algorithms and return results"""
        self.logger.info("Running comprehensive centrality analysis...")
        
        results = {}
        
        # Basic centralities
        results['betweenness'] = self.betweenness_centrality()
        results['pagerank'] = self.pagerank()
        results['closeness'] = self.closeness_centrality()
        
        # Advanced centralities
        try:
            results['eigenvector'] = self.eigenvector_centrality()
        except Exception as e:
            self.logger.warning(f"Eigenvector centrality failed: {e}")
        
        try:
            results['katz'] = self.katz_centrality()
        except Exception as e:
            self.logger.warning(f"Katz centrality failed: {e}")
        
        # HITS (particularly good for pub-sub)
        hubs, authorities = self.hits()
        results['hits_hubs'] = hubs
        results['hits_authorities'] = authorities
        
        # Degree analysis
        degree_results = self.degree_centrality_analysis()
        results.update(degree_results)
        
        # Load and harmonic
        results['load'] = self.load_centrality()
        results['harmonic'] = self.harmonic_centrality()
        
        self.logger.info(f"Completed {len(results)} centrality analyses")
        
        return results


# ============================================================================
# Structural Analysis Algorithms
# ============================================================================

class StructuralAnalyzer:
    """
    Structural analysis for identifying vulnerabilities and patterns
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self.G_undirected = graph.to_undirected()
        self.logger = logging.getLogger('structural')
    
    def find_articulation_points(self) -> Dict[str, Any]:
        """
        Articulation Points (Cut Vertices)
        
        Nodes whose removal disconnects the graph.
        
        Pub-Sub Application:
        - Single points of failure
        - Critical for maintaining message flow connectivity
        - Priority candidates for redundancy
        """
        aps = list(nx.articulation_points(self.G_undirected))
        
        # Analyze each AP's impact
        ap_analysis = {}
        for ap in aps:
            # Count components if removed
            G_copy = self.G_undirected.copy()
            G_copy.remove_node(ap)
            num_components = nx.number_connected_components(G_copy)
            
            ap_analysis[ap] = {
                'type': self.G.nodes[ap].get('type', 'Unknown'),
                'components_after_removal': num_components,
                'degree': self.G.degree(ap)
            }
        
        return {
            'articulation_points': aps,
            'count': len(aps),
            'percentage': len(aps) / len(self.G.nodes()) * 100 if self.G.nodes() else 0,
            'analysis': ap_analysis,
            'interpretation': "Articulation points are critical single points of failure. "
                            "Their removal would disconnect parts of the system, "
                            "breaking message flow between publishers and subscribers."
        }
    
    def find_bridges(self) -> Dict[str, Any]:
        """
        Bridges (Cut Edges)
        
        Edges whose removal disconnects the graph.
        
        Pub-Sub Application:
        - Critical connections between components
        - Single points of failure at the relationship level
        - Important for understanding dependency chains
        """
        bridges = list(nx.bridges(self.G_undirected))
        
        # Analyze each bridge
        bridge_analysis = []
        for u, v in bridges:
            edge_data = self.G.get_edge_data(u, v) or self.G.get_edge_data(v, u) or {}
            bridge_analysis.append({
                'from': u,
                'to': v,
                'type': edge_data.get('type', 'Unknown'),
                'from_type': self.G.nodes[u].get('type', 'Unknown'),
                'to_type': self.G.nodes[v].get('type', 'Unknown')
            })
        
        return {
            'bridges': bridges,
            'count': len(bridges),
            'percentage': len(bridges) / len(self.G.edges()) * 100 if self.G.edges() else 0,
            'analysis': bridge_analysis,
            'interpretation': "Bridges are critical edges whose failure would disconnect "
                            "parts of the system. In pub-sub, these represent irreplaceable "
                            "connections in the message flow chain."
        }
    
    def k_core_decomposition(self) -> Dict[str, Any]:
        """
        K-Core Decomposition
        
        Find densely connected subgraphs where each node has at least k neighbors.
        
        Pub-Sub Application:
        - Identify tightly coupled subsystems
        - Find the "core" vs "periphery" of the architecture
        - High k-core = critical, well-connected component cluster
        """
        core_numbers = nx.core_number(self.G_undirected)
        max_k = max(core_numbers.values()) if core_numbers else 0
        
        # Group nodes by core number
        cores = defaultdict(list)
        for node, k in core_numbers.items():
            cores[k].append(node)
        
        # Find the innermost core
        innermost_core = cores[max_k] if max_k in cores else []
        
        return {
            'core_numbers': core_numbers,
            'max_k': max_k,
            'cores_by_k': dict(cores),
            'innermost_core': innermost_core,
            'interpretation': f"K-core decomposition reveals the hierarchical structure. "
                            f"The innermost core (k={max_k}) contains {len(innermost_core)} nodes "
                            f"that form the most densely connected backbone of the system."
        }
    
    def find_cliques(self, min_size: int = 3) -> Dict[str, Any]:
        """
        Clique Detection
        
        Find fully connected subgraphs.
        
        Pub-Sub Application:
        - Identify components with full mutual dependencies
        - Potential tight coupling anti-pattern
        - Candidates for refactoring
        """
        cliques = list(nx.find_cliques(self.G_undirected))
        cliques = [c for c in cliques if len(c) >= min_size]
        cliques.sort(key=len, reverse=True)
        
        return {
            'cliques': cliques[:20],  # Top 20
            'count': len(cliques),
            'max_size': len(cliques[0]) if cliques else 0,
            'interpretation': "Cliques represent groups of components that are all directly "
                            "connected to each other. Large cliques may indicate tight coupling "
                            "that could be refactored for better maintainability."
        }
    
    def strongly_connected_components(self) -> Dict[str, Any]:
        """
        Strongly Connected Components (SCCs)
        
        Maximal sets where every node can reach every other.
        
        Pub-Sub Application:
        - Identify cyclic dependencies
        - Find components that form feedback loops
        - Important for understanding message circulation patterns
        """
        sccs = list(nx.strongly_connected_components(self.G))
        sccs.sort(key=len, reverse=True)
        
        # Analyze non-trivial SCCs (size > 1)
        non_trivial = [scc for scc in sccs if len(scc) > 1]
        
        return {
            'components': sccs,
            'count': len(sccs),
            'non_trivial_count': len(non_trivial),
            'largest_scc': list(sccs[0]) if sccs else [],
            'interpretation': "Strongly connected components where all nodes can reach each other. "
                            "Non-trivial SCCs (size > 1) indicate cyclic dependencies in message flow, "
                            "which may cause issues like infinite loops or complex failure cascades."
        }
    
    def weakly_connected_components(self) -> Dict[str, Any]:
        """
        Weakly Connected Components
        
        Components connected ignoring edge direction.
        """
        wccs = list(nx.weakly_connected_components(self.G))
        wccs.sort(key=len, reverse=True)
        
        return {
            'components': [list(c) for c in wccs],
            'count': len(wccs),
            'largest_wcc_size': len(wccs[0]) if wccs else 0,
            'is_connected': len(wccs) == 1,
            'interpretation': "Weakly connected components show isolated subsystems. "
                            "Multiple components indicate the system has disconnected parts "
                            "that cannot communicate even indirectly."
        }
    
    def cycle_detection(self, max_cycles: int = 100) -> Dict[str, Any]:
        """
        Cycle Detection
        
        Find cycles in the directed graph.
        
        Pub-Sub Application:
        - Circular dependencies
        - Potential for message loops
        - Important for preventing infinite message propagation
        """
        try:
            cycles = list(nx.simple_cycles(self.G))
            cycles = cycles[:max_cycles]  # Limit for performance
            cycles.sort(key=len)
        except Exception as e:
            self.logger.warning(f"Cycle detection error: {e}")
            cycles = []
        
        return {
            'cycles': cycles,
            'count': len(cycles),
            'has_cycles': len(cycles) > 0,
            'shortest_cycle': cycles[0] if cycles else [],
            'interpretation': "Cycles in directed graphs represent circular dependencies. "
                            "In pub-sub, these could cause infinite message loops where "
                            "A publishes to B which publishes back to A."
        }


# ============================================================================
# Community Detection Algorithms
# ============================================================================

class CommunityDetector:
    """
    Community detection for identifying subsystems and clusters
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self.G_undirected = graph.to_undirected()
        self.logger = logging.getLogger('community')
    
    def louvain_communities(self, resolution: float = 1.0) -> CommunityResult:
        """
        Louvain Algorithm
        
        Fast, scalable community detection optimizing modularity.
        
        Pub-Sub Application:
        - Identify logical subsystems
        - Group related publishers/topics/subscribers
        - Useful for microservice boundary identification
        """
        communities = list(community.louvain_communities(
            self.G_undirected, 
            resolution=resolution
        ))
        
        return self._create_result("louvain", communities)
    
    def label_propagation(self) -> CommunityResult:
        """
        Label Propagation Algorithm
        
        Fast, semi-synchronous community detection.
        
        Pub-Sub Application:
        - Quick clustering of related components
        - Good for initial architecture understanding
        """
        communities = list(community.label_propagation_communities(self.G_undirected))
        
        return self._create_result("label_propagation", communities)
    
    def girvan_newman(self, k: int = None) -> CommunityResult:
        """
        Girvan-Newman Algorithm
        
        Hierarchical decomposition based on edge betweenness.
        
        Pub-Sub Application:
        - Identifies communities by removing "bridge" connections
        - Good for understanding how subsystems connect
        - More interpretable but slower than Louvain
        """
        comp = community.girvan_newman(self.G_undirected)
        
        # Get k communities or iterate to find best
        if k:
            for _ in range(k - 1):
                communities = next(comp)
        else:
            # Find partition with best modularity
            best_communities = None
            best_modularity = -1
            
            for i, partition in enumerate(comp):
                if i > 10:  # Limit iterations
                    break
                communities = list(partition)
                mod = nx.community.modularity(self.G_undirected, communities)
                if mod > best_modularity:
                    best_modularity = mod
                    best_communities = communities
            
            communities = best_communities or [set(self.G.nodes())]
        
        return self._create_result("girvan_newman", list(communities))
    
    def greedy_modularity(self) -> CommunityResult:
        """
        Greedy Modularity Optimization
        
        Agglomerative community detection.
        """
        communities = list(community.greedy_modularity_communities(self.G_undirected))
        
        return self._create_result("greedy_modularity", communities)
    
    def _create_result(self, algorithm: str, communities: List[Set[str]]) -> CommunityResult:
        """Create standardized community result"""
        # Calculate modularity
        try:
            mod = nx.community.modularity(self.G_undirected, communities)
        except:
            mod = 0.0
        
        # Create node to community mapping
        node_to_comm = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_comm[node] = i
        
        # Statistics
        sizes = [len(c) for c in communities]
        stats = {
            'num_communities': len(communities),
            'avg_size': sum(sizes) / len(sizes) if sizes else 0,
            'max_size': max(sizes) if sizes else 0,
            'min_size': min(sizes) if sizes else 0,
            'size_distribution': sizes
        }
        
        return CommunityResult(
            algorithm=algorithm,
            communities=communities,
            modularity=mod,
            node_to_community=node_to_comm,
            statistics=stats
        )
    
    def analyze_community_composition(self, result: CommunityResult) -> Dict[str, Any]:
        """Analyze what types of nodes are in each community"""
        composition = []
        
        for i, comm in enumerate(result.communities):
            type_counts = defaultdict(int)
            for node in comm:
                node_type = self.G.nodes[node].get('type', 'Unknown')
                type_counts[node_type] += 1
            
            composition.append({
                'community_id': i,
                'size': len(comm),
                'types': dict(type_counts),
                'members': list(comm)
            })
        
        return {
            'communities': composition,
            'interpretation': "Community composition shows how different node types cluster. "
                            "Ideally, each community represents a logical subsystem with "
                            "related publishers, topics, and subscribers."
        }


# ============================================================================
# Path and Flow Analysis
# ============================================================================

class PathFlowAnalyzer:
    """
    Path and flow analysis for understanding message routing
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self.logger = logging.getLogger('path_flow')
    
    def critical_path_analysis(self) -> Dict[str, Any]:
        """
        Critical Path Analysis
        
        Find the longest paths (critical paths) in the message flow.
        
        Pub-Sub Application:
        - Identify end-to-end message latency chains
        - Find the longest dependency sequences
        - Critical for latency optimization
        """
        # Find all paths between publishers and subscribers
        publishers = [n for n, d in self.G.nodes(data=True) 
                     if d.get('type') == 'Application' and self.G.out_degree(n) > 0]
        subscribers = [n for n, d in self.G.nodes(data=True)
                      if d.get('type') == 'Application' and self.G.in_degree(n) > 0]
        
        longest_paths = []
        
        for pub in publishers[:10]:  # Limit for performance
            for sub in subscribers[:10]:
                if pub != sub:
                    try:
                        paths = list(nx.all_simple_paths(self.G, pub, sub, cutoff=10))
                        for path in paths:
                            if len(path) > 2:
                                longest_paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
        
        # Sort by length
        longest_paths.sort(key=len, reverse=True)
        
        return {
            'critical_paths': longest_paths[:10],
            'max_length': len(longest_paths[0]) if longest_paths else 0,
            'interpretation': "Critical paths show the longest message flow chains. "
                            "These determine end-to-end latency and are important for "
                            "performance optimization."
        }
    
    def shortest_path_analysis(self) -> Dict[str, Any]:
        """
        All-Pairs Shortest Paths Analysis
        
        Analyze the shortest paths between all node pairs.
        """
        # Calculate average path length
        try:
            avg_path_length = nx.average_shortest_path_length(self.G)
        except nx.NetworkXError:
            # Graph is not connected
            avg_path_length = float('inf')
        
        # Diameter (longest shortest path)
        try:
            diameter = nx.diameter(self.G.to_undirected())
        except:
            diameter = float('inf')
        
        return {
            'average_path_length': avg_path_length,
            'diameter': diameter,
            'interpretation': f"Average path length of {avg_path_length:.2f} means messages "
                            f"travel through approximately this many hops on average."
        }
    
    def find_bottlenecks(self) -> Dict[str, Any]:
        """
        Bottleneck Detection
        
        Find nodes that appear on many shortest paths.
        
        Pub-Sub Application:
        - Identify routing bottlenecks
        - Find potential throughput limitations
        - Critical for capacity planning
        """
        # Use betweenness as a proxy for bottleneck potential
        betweenness = nx.betweenness_centrality(self.G)
        
        # Also consider degree
        in_degree = dict(self.G.in_degree())
        out_degree = dict(self.G.out_degree())
        
        # Combined bottleneck score
        bottleneck_scores = {}
        for node in self.G.nodes():
            # High betweenness + high degree = bottleneck
            bc = betweenness.get(node, 0)
            total_degree = in_degree.get(node, 0) + out_degree.get(node, 0)
            max_degree = max(sum(in_degree.values()), 1)
            
            bottleneck_scores[node] = bc * 0.6 + (total_degree / max_degree) * 0.4
        
        # Sort by score
        sorted_bottlenecks = sorted(bottleneck_scores.items(), 
                                   key=lambda x: x[1], reverse=True)
        
        return {
            'bottleneck_scores': bottleneck_scores,
            'top_bottlenecks': sorted_bottlenecks[:10],
            'interpretation': "Bottlenecks are nodes that handle disproportionate message traffic. "
                            "They may need scaling, redundancy, or architectural changes."
        }
    
    def message_flow_analysis(self) -> Dict[str, Any]:
        """
        Analyze message flow patterns
        
        Track how messages flow from publishers through topics to subscribers.
        """
        # Find all publisher → topic → subscriber chains
        flow_chains = []
        
        for node in self.G.nodes():
            if self.G.nodes[node].get('type') == 'Topic':
                # Find publishers to this topic
                publishers = [pred for pred in self.G.predecessors(node)
                             if self.G.nodes[pred].get('type') == 'Application']
                
                # Find subscribers from this topic
                subscribers = [succ for succ in self.G.successors(node)
                              if self.G.nodes[succ].get('type') == 'Application']
                
                if publishers and subscribers:
                    flow_chains.append({
                        'topic': node,
                        'publishers': publishers,
                        'subscribers': subscribers,
                        'fanout': len(subscribers),
                        'fanin': len(publishers)
                    })
        
        # Sort by fanout (most distributed topics first)
        flow_chains.sort(key=lambda x: x['fanout'], reverse=True)
        
        return {
            'flow_chains': flow_chains,
            'total_topics_analyzed': len(flow_chains),
            'max_fanout': max(fc['fanout'] for fc in flow_chains) if flow_chains else 0,
            'max_fanin': max(fc['fanin'] for fc in flow_chains) if flow_chains else 0,
            'interpretation': "Flow analysis shows how messages distribute from publishers "
                            "through topics to subscribers. High fanout indicates topics "
                            "serving many consumers."
        }
    
    def dependency_depth_analysis(self) -> Dict[str, Any]:
        """
        Analyze dependency chain depths
        
        Pub-Sub Application:
        - Find components with deep dependency chains
        - Identify potential cascade failure risks
        """
        depths = {}
        
        for node in self.G.nodes():
            # Calculate depth from this node (max path length to any descendant)
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
        
        sorted_depths = sorted(depths.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'dependency_depths': depths,
            'deepest_chains': sorted_depths[:10],
            'max_depth': max(depths.values()) if depths else 0,
            'interpretation': "Dependency depth shows how far failures can cascade. "
                            "Deep chains mean a failure at the root affects many downstream components."
        }


# ============================================================================
# Similarity and Coupling Analysis
# ============================================================================

class SimilarityCouplingAnalyzer:
    """
    Analyze similarity and coupling between components
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self.G_undirected = graph.to_undirected()
        self.logger = logging.getLogger('similarity')
    
    def jaccard_similarity(self, node1: str, node2: str) -> float:
        """
        Jaccard Similarity: Common neighbors / total unique neighbors
        
        Pub-Sub Application:
        - Find components with similar connectivity patterns
        - Identify redundant or duplicate functionality
        """
        neighbors1 = set(self.G_undirected.neighbors(node1))
        neighbors2 = set(self.G_undirected.neighbors(node2))
        
        intersection = len(neighbors1 & neighbors2)
        union = len(neighbors1 | neighbors2)
        
        return intersection / union if union > 0 else 0
    
    def find_similar_components(self, threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        """Find pairs of components with high similarity"""
        similar_pairs = []
        nodes = list(self.G.nodes())
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                sim = self.jaccard_similarity(nodes[i], nodes[j])
                if sim >= threshold:
                    similar_pairs.append((nodes[i], nodes[j], sim))
        
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        return similar_pairs
    
    def coupling_analysis(self) -> Dict[str, Any]:
        """
        Analyze coupling between components
        
        Pub-Sub Application:
        - Identify tightly coupled components
        - Find candidates for decoupling
        - Measure overall system coupling
        """
        # Afferent coupling (Ca): incoming dependencies
        # Efferent coupling (Ce): outgoing dependencies
        
        coupling_metrics = {}
        
        for node in self.G.nodes():
            ca = self.G.in_degree(node)   # Afferent
            ce = self.G.out_degree(node)  # Efferent
            
            # Instability = Ce / (Ca + Ce)
            total = ca + ce
            instability = ce / total if total > 0 else 0
            
            coupling_metrics[node] = {
                'afferent_coupling': ca,
                'efferent_coupling': ce,
                'total_coupling': total,
                'instability': instability,
                'type': self.G.nodes[node].get('type', 'Unknown')
            }
        
        # Find highly coupled components
        high_coupling = sorted(
            coupling_metrics.items(),
            key=lambda x: x[1]['total_coupling'],
            reverse=True
        )[:10]
        
        # Average instability
        avg_instability = sum(m['instability'] for m in coupling_metrics.values()) / len(coupling_metrics)
        
        return {
            'coupling_metrics': coupling_metrics,
            'highly_coupled': high_coupling,
            'average_instability': avg_instability,
            'interpretation': "Coupling analysis reveals dependencies between components. "
                            "High coupling indicates tight dependencies; instability > 0.5 "
                            "suggests the component depends more on others than others depend on it."
        }
    
    def common_neighbors_analysis(self) -> Dict[str, Any]:
        """
        Analyze common neighbor patterns
        
        Find components that share many connections (potential redundancy).
        """
        common_neighbors = []
        nodes = list(self.G.nodes())
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                cn = list(nx.common_neighbors(self.G_undirected, nodes[i], nodes[j]))
                if len(cn) > 1:  # At least 2 common neighbors
                    common_neighbors.append({
                        'node1': nodes[i],
                        'node2': nodes[j],
                        'common_neighbors': cn,
                        'count': len(cn)
                    })
        
        common_neighbors.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            'common_neighbor_pairs': common_neighbors[:20],
            'interpretation': "Components sharing many neighbors may have similar roles "
                            "or represent redundancy in the architecture."
        }


# ============================================================================
# Robustness Analysis
# ============================================================================

class RobustnessAnalyzer:
    """
    Analyze system robustness and resilience to failures
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self.G_undirected = graph.to_undirected()
        self.logger = logging.getLogger('robustness')
    
    def targeted_attack_simulation(self, strategy: str = 'degree') -> RobustnessResult:
        """
        Simulate targeted attacks on high-value nodes
        
        Strategies:
        - 'degree': Remove highest degree nodes first
        - 'betweenness': Remove highest betweenness nodes first
        - 'pagerank': Remove highest PageRank nodes first
        
        Pub-Sub Application:
        - Understand how system degrades under targeted failures
        - Identify critical failure thresholds
        """
        G_copy = self.G_undirected.copy()
        original_size = len(G_copy)
        
        failure_sequence = []
        connectivity_curve = [1.0]  # Start fully connected
        
        while len(G_copy) > 0:
            # Calculate metric for remaining nodes
            if strategy == 'degree':
                scores = dict(G_copy.degree())
            elif strategy == 'betweenness':
                scores = nx.betweenness_centrality(G_copy)
            elif strategy == 'pagerank':
                scores = nx.pagerank(G_copy.to_directed() if G_copy.is_directed() else G_copy)
            else:
                scores = dict(G_copy.degree())
            
            # Remove highest scoring node
            if not scores:
                break
            target = max(scores.items(), key=lambda x: x[1])[0]
            G_copy.remove_node(target)
            failure_sequence.append(target)
            
            # Measure connectivity
            if len(G_copy) > 0:
                largest_cc = max(nx.connected_components(G_copy), key=len)
                connectivity = len(largest_cc) / original_size
            else:
                connectivity = 0
            
            connectivity_curve.append(connectivity)
            
            # Stop if mostly disconnected
            if connectivity < 0.1:
                break
        
        # Find critical threshold (where connectivity drops below 50%)
        critical_threshold = 1.0
        for i, conn in enumerate(connectivity_curve):
            if conn < 0.5:
                critical_threshold = i / original_size
                break
        
        # Calculate AUC (area under curve) as robustness measure
        auc = sum(connectivity_curve) / len(connectivity_curve)
        
        return RobustnessResult(
            attack_type=f'targeted_{strategy}',
            failure_sequence=failure_sequence,
            connectivity_curve=connectivity_curve,
            critical_threshold=critical_threshold,
            auc_robustness=auc
        )
    
    def random_failure_simulation(self, num_simulations: int = 10) -> RobustnessResult:
        """
        Simulate random failures
        
        Compare with targeted attacks to understand resilience.
        """
        import random
        
        all_curves = []
        
        for _ in range(num_simulations):
            G_copy = self.G_undirected.copy()
            original_size = len(G_copy)
            nodes = list(G_copy.nodes())
            random.shuffle(nodes)
            
            connectivity_curve = [1.0]
            
            for node in nodes:
                if node in G_copy:
                    G_copy.remove_node(node)
                    
                    if len(G_copy) > 0:
                        largest_cc = max(nx.connected_components(G_copy), key=len)
                        connectivity = len(largest_cc) / original_size
                    else:
                        connectivity = 0
                    
                    connectivity_curve.append(connectivity)
                    
                    if connectivity < 0.1:
                        break
            
            all_curves.append(connectivity_curve)
        
        # Average the curves
        max_len = max(len(c) for c in all_curves)
        avg_curve = []
        for i in range(max_len):
            values = [c[i] for c in all_curves if i < len(c)]
            avg_curve.append(sum(values) / len(values))
        
        # Find critical threshold
        critical_threshold = 1.0
        for i, conn in enumerate(avg_curve):
            if conn < 0.5:
                critical_threshold = i / len(self.G_undirected)
                break
        
        auc = sum(avg_curve) / len(avg_curve)
        
        return RobustnessResult(
            attack_type='random',
            failure_sequence=[],  # Random, so no specific sequence
            connectivity_curve=avg_curve,
            critical_threshold=critical_threshold,
            auc_robustness=auc
        )
    
    def compare_robustness(self) -> Dict[str, Any]:
        """
        Compare robustness under different failure scenarios
        """
        results = {
            'random': self.random_failure_simulation(),
            'targeted_degree': self.targeted_attack_simulation('degree'),
            'targeted_betweenness': self.targeted_attack_simulation('betweenness'),
        }
        
        comparison = {
            name: {
                'critical_threshold': r.critical_threshold,
                'auc_robustness': r.auc_robustness
            }
            for name, r in results.items()
        }
        
        # Interpretation
        random_auc = results['random'].auc_robustness
        targeted_auc = results['targeted_degree'].auc_robustness
        
        vulnerability = (random_auc - targeted_auc) / random_auc if random_auc > 0 else 0
        
        return {
            'results': results,
            'comparison': comparison,
            'vulnerability_to_targeted_attacks': vulnerability,
            'interpretation': f"System is {vulnerability*100:.1f}% more vulnerable to targeted "
                            f"attacks than random failures. Higher values indicate the system "
                            f"has critical single points of failure."
        }


# ============================================================================
# Layer-Aware Analysis
# ============================================================================

class LayerAwareAnalyzer:
    """
    Multi-layer graph analysis for pub-sub systems
    
    Layers:
    - Application Layer (publishers/subscribers)
    - Topic Layer (message channels)
    - Broker Layer (message routing)
    - Infrastructure Layer (physical/virtual nodes)
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self.logger = logging.getLogger('layer_aware')
        
        # Separate nodes by layer
        self.layers = {
            'Application': [],
            'Topic': [],
            'Broker': [],
            'Node': []
        }
        
        for node, data in graph.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            if node_type in self.layers:
                self.layers[node_type].append(node)
    
    def cross_layer_dependencies(self) -> Dict[str, Any]:
        """
        Analyze dependencies between layers
        """
        layer_edges = defaultdict(int)
        
        for u, v, data in self.G.edges(data=True):
            u_type = self.G.nodes[u].get('type', 'Unknown')
            v_type = self.G.nodes[v].get('type', 'Unknown')
            layer_edges[(u_type, v_type)] += 1
        
        return {
            'cross_layer_edges': dict(layer_edges),
            'layer_sizes': {k: len(v) for k, v in self.layers.items()},
            'interpretation': "Cross-layer dependencies show how different system layers interact. "
                            "Heavy dependencies between layers indicate tight coupling."
        }
    
    def layer_centrality(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate centrality within and across layers
        """
        results = {}
        
        for layer_name, nodes in self.layers.items():
            if not nodes:
                continue
            
            # Subgraph for this layer
            subgraph = self.G.subgraph(nodes)
            
            # Calculate centralities
            if len(subgraph.nodes()) > 1:
                bc = nx.betweenness_centrality(subgraph)
                avg_bc = sum(bc.values()) / len(bc) if bc else 0
            else:
                avg_bc = 0
            
            # Cross-layer importance (connections to other layers)
            cross_layer_edges = sum(
                1 for u, v in self.G.edges()
                if (u in nodes and v not in nodes) or (v in nodes and u not in nodes)
            )
            
            results[layer_name] = {
                'size': len(nodes),
                'internal_edges': len(subgraph.edges()),
                'cross_layer_edges': cross_layer_edges,
                'avg_internal_betweenness': avg_bc
            }
        
        return results
    
    def layer_impact_analysis(self) -> Dict[str, Any]:
        """
        Analyze the impact of each layer on overall system
        """
        impacts = {}
        
        for layer_name, nodes in self.layers.items():
            if not nodes:
                continue
            
            # Simulate removal of entire layer
            G_copy = self.G.copy()
            for node in nodes:
                if node in G_copy:
                    G_copy.remove_node(node)
            
            # Measure remaining connectivity
            remaining_nodes = len(G_copy.nodes())
            original_nodes = len(self.G.nodes())
            
            if remaining_nodes > 0 and G_copy.to_undirected().number_of_edges() > 0:
                try:
                    components = list(nx.weakly_connected_components(G_copy))
                    largest_component_ratio = len(max(components, key=len)) / remaining_nodes
                except:
                    largest_component_ratio = 0
            else:
                largest_component_ratio = 0
            
            impacts[layer_name] = {
                'nodes_in_layer': len(nodes),
                'remaining_after_removal': remaining_nodes,
                'connectivity_after_removal': largest_component_ratio,
                'impact_score': 1 - largest_component_ratio
            }
        
        return {
            'layer_impacts': impacts,
            'most_critical_layer': max(impacts.items(), 
                                       key=lambda x: x[1]['impact_score'])[0] if impacts else None,
            'interpretation': "Layer impact shows how critical each layer is to overall connectivity. "
                            "Removing a high-impact layer would fragment the system."
        }


# ============================================================================
# Algorithm Recommendations
# ============================================================================

def get_algorithm_recommendations() -> List[AlgorithmRecommendation]:
    """
    Get recommendations for which algorithms to apply and why
    """
    return [
        # Centrality - High Priority
        AlgorithmRecommendation(
            algorithm="Betweenness Centrality",
            category="Centrality",
            purpose="Find nodes on shortest paths between others",
            pub_sub_application="Identify critical routing points. High BC brokers/topics are "
                               "bottlenecks. Failure disrupts many message flows.",
            complexity="O(VE)",
            priority=1
        ),
        AlgorithmRecommendation(
            algorithm="HITS (Hubs & Authorities)",
            category="Centrality", 
            purpose="Find hubs (many outlinks) and authorities (many inlinks)",
            pub_sub_application="PERFECT for pub-sub! Publishers are hubs, topics/subscribers "
                               "are authorities. Captures asymmetric pub-sub relationships.",
            complexity="O(k·E) where k is iterations",
            priority=1
        ),
        AlgorithmRecommendation(
            algorithm="PageRank",
            category="Centrality",
            purpose="Importance based on recursive neighbor importance",
            pub_sub_application="Find influential topics/apps. High PageRank = receives from "
                               "important sources. Good for identifying key data flows.",
            complexity="O(k·E)",
            priority=1
        ),
        
        # Structural - High Priority
        AlgorithmRecommendation(
            algorithm="Articulation Points",
            category="Structural",
            purpose="Find nodes whose removal disconnects graph",
            pub_sub_application="Single points of failure. Critical for redundancy planning. "
                               "Must-have for reliability analysis.",
            complexity="O(V+E)",
            priority=1
        ),
        AlgorithmRecommendation(
            algorithm="Bridges",
            category="Structural",
            purpose="Find edges whose removal disconnects graph",
            pub_sub_application="Critical connections. If a PUBLISHES/SUBSCRIBES edge is a bridge, "
                               "that relationship is a single point of failure.",
            complexity="O(V+E)",
            priority=1
        ),
        AlgorithmRecommendation(
            algorithm="K-Core Decomposition",
            category="Structural",
            purpose="Find hierarchical dense subgraphs",
            pub_sub_application="Identify core vs periphery. High k-core = tightly coupled backbone. "
                               "Useful for understanding architecture layers.",
            complexity="O(V+E)",
            priority=2
        ),
        AlgorithmRecommendation(
            algorithm="Strongly Connected Components",
            category="Structural",
            purpose="Find cyclic dependencies",
            pub_sub_application="Detect circular message flows (A→B→C→A). Can cause infinite loops "
                               "or complex cascade failures. Important anti-pattern detection.",
            complexity="O(V+E)",
            priority=1
        ),
        
        # Community Detection - Medium Priority
        AlgorithmRecommendation(
            algorithm="Louvain Communities",
            category="Community",
            purpose="Fast modularity optimization clustering",
            pub_sub_application="Identify logical subsystems. Group related pub/sub/topics. "
                               "Useful for microservice boundary identification.",
            complexity="O(V·log²V)",
            priority=2
        ),
        AlgorithmRecommendation(
            algorithm="Girvan-Newman",
            category="Community",
            purpose="Hierarchical decomposition via edge betweenness",
            pub_sub_application="Understand how subsystems connect. More interpretable than Louvain. "
                               "Good for architecture documentation.",
            complexity="O(E²V)",
            priority=3
        ),
        
        # Path Analysis - High Priority
        AlgorithmRecommendation(
            algorithm="Critical Path Analysis",
            category="Path",
            purpose="Find longest dependency chains",
            pub_sub_application="Determines end-to-end latency. Long chains = high latency risk. "
                               "Priority for latency-sensitive systems.",
            complexity="O(V·E)",
            priority=1
        ),
        AlgorithmRecommendation(
            algorithm="All-Pairs Shortest Paths",
            category="Path",
            purpose="Distance between all node pairs",
            pub_sub_application="Understand message hop counts. Average path length indicates "
                               "overall system diameter. Important for latency estimation.",
            complexity="O(V³) or O(VE + V²logV)",
            priority=2
        ),
        
        # Robustness - Medium Priority
        AlgorithmRecommendation(
            algorithm="Targeted Attack Simulation",
            category="Robustness",
            purpose="Simulate sequential node failures",
            pub_sub_application="Understand how system degrades. Compare random vs targeted failures. "
                               "Essential for disaster recovery planning.",
            complexity="O(V·(V+E))",
            priority=2
        ),
        AlgorithmRecommendation(
            algorithm="Percolation Analysis",
            category="Robustness",
            purpose="Find failure threshold",
            pub_sub_application="At what failure rate does system fragment? Critical for SLA planning "
                               "and redundancy requirements.",
            complexity="O(V·E)",
            priority=2
        ),
        
        # Similarity - Lower Priority
        AlgorithmRecommendation(
            algorithm="Jaccard Similarity",
            category="Similarity",
            purpose="Find nodes with similar connections",
            pub_sub_application="Identify redundant components. Find candidates for consolidation "
                               "or load balancing.",
            complexity="O(V²·D) where D is avg degree",
            priority=3
        ),
        AlgorithmRecommendation(
            algorithm="Coupling Analysis",
            category="Similarity",
            purpose="Measure afferent/efferent dependencies",
            pub_sub_application="Martin's stability metrics. High coupling = hard to change. "
                               "Important for maintainability assessment.",
            complexity="O(V+E)",
            priority=2
        ),
        
        # Advanced
        AlgorithmRecommendation(
            algorithm="Eigenvector Centrality",
            category="Centrality",
            purpose="Influence based on influential neighbors",
            pub_sub_application="Find components connected to important nodes. Captures 'prestige' "
                               "in the network topology.",
            complexity="O(k·E)",
            priority=2
        ),
        AlgorithmRecommendation(
            algorithm="Closeness Centrality",
            category="Centrality",
            purpose="Average distance to all other nodes",
            pub_sub_application="Optimal locations for monitoring. Low closeness = efficient "
                               "message propagation point.",
            complexity="O(VE)",
            priority=2
        ),
    ]


def print_recommendations():
    """Print formatted algorithm recommendations"""
    recommendations = get_algorithm_recommendations()
    
    print("\n" + "="*80)
    print("  GRAPH ALGORITHM RECOMMENDATIONS FOR PUB-SUB ANALYSIS")
    print("="*80)
    
    # Group by priority
    by_priority = defaultdict(list)
    for rec in recommendations:
        by_priority[rec.priority].append(rec)
    
    priority_names = {1: "HIGH PRIORITY", 2: "MEDIUM PRIORITY", 3: "LOWER PRIORITY"}
    
    for priority in sorted(by_priority.keys()):
        print(f"\n{'='*40}")
        print(f"  {priority_names[priority]}")
        print(f"{'='*40}")
        
        for rec in by_priority[priority]:
            print(f"\n  📊 {rec.algorithm}")
            print(f"     Category: {rec.category}")
            print(f"     Complexity: {rec.complexity}")
            print(f"     Purpose: {rec.purpose}")
            print(f"     Pub-Sub Application:")
            # Word wrap the application text
            words = rec.pub_sub_application.split()
            line = "       "
            for word in words:
                if len(line) + len(word) > 75:
                    print(line)
                    line = "       " + word
                else:
                    line += " " + word if line.strip() else word
            if line.strip():
                print(line)


# ============================================================================
# Main Analysis Runner
# ============================================================================

class ComprehensiveGraphAnalyzer:
    """
    Unified interface for running all graph algorithms
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        self.logger = logging.getLogger('comprehensive')
        
        # Initialize analyzers
        self.centrality = CentralityAnalyzer(graph)
        self.structural = StructuralAnalyzer(graph)
        self.community = CommunityDetector(graph)
        self.path_flow = PathFlowAnalyzer(graph)
        self.similarity = SimilarityCouplingAnalyzer(graph)
        self.robustness = RobustnessAnalyzer(graph)
        self.layer_aware = LayerAwareAnalyzer(graph)
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run all analyses and return comprehensive results"""
        self.logger.info("Starting comprehensive graph analysis...")
        
        results = {
            'graph_info': {
                'nodes': len(self.G.nodes()),
                'edges': len(self.G.edges()),
                'density': nx.density(self.G)
            },
            'centrality': {},
            'structural': {},
            'community': {},
            'path_flow': {},
            'similarity': {},
            'robustness': {},
            'layer_analysis': {}
        }
        
        # Centrality
        self.logger.info("Running centrality analysis...")
        results['centrality'] = self.centrality.run_all()
        
        # Structural
        self.logger.info("Running structural analysis...")
        results['structural']['articulation_points'] = self.structural.find_articulation_points()
        results['structural']['bridges'] = self.structural.find_bridges()
        results['structural']['k_core'] = self.structural.k_core_decomposition()
        results['structural']['sccs'] = self.structural.strongly_connected_components()
        results['structural']['cycles'] = self.structural.cycle_detection()
        
        # Community
        self.logger.info("Running community detection...")
        results['community']['louvain'] = self.community.louvain_communities()
        
        # Path/Flow
        self.logger.info("Running path and flow analysis...")
        results['path_flow']['critical_paths'] = self.path_flow.critical_path_analysis()
        results['path_flow']['bottlenecks'] = self.path_flow.find_bottlenecks()
        results['path_flow']['message_flow'] = self.path_flow.message_flow_analysis()
        results['path_flow']['dependency_depth'] = self.path_flow.dependency_depth_analysis()
        
        # Similarity
        self.logger.info("Running coupling analysis...")
        results['similarity']['coupling'] = self.similarity.coupling_analysis()
        
        # Robustness
        self.logger.info("Running robustness analysis...")
        results['robustness'] = self.robustness.compare_robustness()
        
        # Layer-aware
        self.logger.info("Running layer-aware analysis...")
        results['layer_analysis']['cross_layer'] = self.layer_aware.cross_layer_dependencies()
        results['layer_analysis']['layer_centrality'] = self.layer_aware.layer_centrality()
        results['layer_analysis']['layer_impact'] = self.layer_aware.layer_impact_analysis()
        
        self.logger.info("Comprehensive analysis complete!")
        
        return results
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable summary"""
        report = []
        report.append("=" * 70)
        report.append("  COMPREHENSIVE GRAPH ANALYSIS REPORT")
        report.append("=" * 70)
        
        # Graph Info
        report.append(f"\n📊 Graph Overview")
        report.append(f"   Nodes: {results['graph_info']['nodes']}")
        report.append(f"   Edges: {results['graph_info']['edges']}")
        report.append(f"   Density: {results['graph_info']['density']:.4f}")
        
        # Key Findings - Centrality
        report.append(f"\n🎯 Key Central Components")
        if 'betweenness' in results['centrality']:
            bc = results['centrality']['betweenness']
            report.append(f"   Top Betweenness: {bc.top_k[0][0]} ({bc.top_k[0][1]:.4f})")
        if 'hits_hubs' in results['centrality']:
            hubs = results['centrality']['hits_hubs']
            report.append(f"   Top Hub: {hubs.top_k[0][0]} ({hubs.top_k[0][1]:.4f})")
        if 'hits_authorities' in results['centrality']:
            auth = results['centrality']['hits_authorities']
            report.append(f"   Top Authority: {auth.top_k[0][0]} ({auth.top_k[0][1]:.4f})")
        
        # Structural
        report.append(f"\n⚠️  Structural Vulnerabilities")
        aps = results['structural']['articulation_points']
        report.append(f"   Articulation Points: {aps['count']}")
        bridges = results['structural']['bridges']
        report.append(f"   Bridges: {bridges['count']}")
        cycles = results['structural']['cycles']
        report.append(f"   Cycles Detected: {cycles['count']}")
        
        # Community
        report.append(f"\n🏘️  Communities")
        louvain = results['community']['louvain']
        report.append(f"   Communities Found: {louvain.statistics['num_communities']}")
        report.append(f"   Modularity: {louvain.modularity:.4f}")
        
        # Robustness
        report.append(f"\n🛡️  Robustness")
        rob = results['robustness']
        report.append(f"   Vulnerability to Targeted Attacks: {rob['vulnerability_to_targeted_attacks']*100:.1f}%")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print_recommendations()
