"""
Edge Criticality Analyzer

Analyzes edge criticality in distributed pub-sub systems to identify:
- Bridge edges (critical links whose removal partitions the network)
- High betweenness edges (communication bottlenecks)
- Vulnerable connections
- Message flow dependencies

This module extends the node-centric analysis approach by focusing on
the criticality of edges/relationships in the system.

Author: Software-as-a-Graph Project
Date: 2025-11-16
"""

import networkx as nx
from typing import Dict, List, Tuple, Set, Any
from dataclasses import dataclass
from enum import Enum
import logging


class EdgeCriticalityLevel(Enum):
    """Criticality levels for edges"""
    CRITICAL = "critical"      # Bridge edges, partitions network
    HIGH = "high"              # Very high betweenness, major bottleneck
    MEDIUM = "medium"          # Moderate betweenness
    LOW = "low"                # Low betweenness


@dataclass
class EdgeCriticalityScore:
    """
    Comprehensive edge criticality score
    
    Attributes:
        source: Source node
        target: Target node
        edge_type: Type of relationship
        betweenness_centrality: Edge betweenness centrality [0,1]
        is_bridge: Whether edge is a bridge
        creates_disconnection: Whether removal causes disconnection
        components_after_removal: Number of components after removal
        criticality_level: Overall criticality classification
        composite_score: Combined criticality score [0,1]
    """
    source: str
    target: str
    edge_type: str
    betweenness_centrality: float
    is_bridge: bool
    creates_disconnection: bool
    components_after_removal: int
    criticality_level: EdgeCriticalityLevel
    composite_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'source': self.source,
            'target': self.target,
            'edge_type': self.edge_type,
            'betweenness_centrality': round(self.betweenness_centrality, 4),
            'is_bridge': self.is_bridge,
            'creates_disconnection': self.creates_disconnection,
            'components_after_removal': self.components_after_removal,
            'criticality_level': self.criticality_level.value,
            'composite_score': round(self.composite_score, 4)
        }


class EdgeCriticalityAnalyzer:
    """
    Analyzes edge criticality in distributed systems
    
    Implements edge-centric analysis to complement node-centric approaches,
    identifying critical connections and communication bottlenecks.
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        """
        Initialize edge criticality analyzer
        
        Args:
            alpha: Weight for betweenness centrality (default: 0.5)
            beta: Weight for bridge/disconnection (default: 0.5)
            
        Note: alpha + beta should equal 1.0 for normalized scoring
        """
        self.alpha = alpha
        self.beta = beta
        self.logger = logging.getLogger(__name__)
        
        # Validate weights
        if abs(alpha + beta - 1.0) > 0.001:
            self.logger.warning(f"Weights sum to {alpha + beta}, not 1.0. Scores may not be normalized.")
    
    def analyze(self, graph: nx.DiGraph) -> Dict[str, EdgeCriticalityScore]:
        """
        Perform comprehensive edge criticality analysis
        
        Args:
            graph: NetworkX directed graph
            
        Returns:
            Dictionary mapping edge tuples (source, target) to EdgeCriticalityScore
        """
        self.logger.info(f"Analyzing edge criticality for graph with {len(graph.edges())} edges")
        
        # Step 1: Compute edge betweenness centrality
        edge_betweenness = self._compute_edge_betweenness(graph)
        
        # Step 2: Identify bridges
        bridges = self._identify_bridges(graph)
        
        # Step 3: Compute impact of edge removal
        edge_impacts = self._compute_edge_impacts(graph)
        
        # Step 4: Calculate composite scores
        scores = self._calculate_composite_scores(
            graph, edge_betweenness, bridges, edge_impacts
        )
        
        self.logger.info(f"Computed criticality scores for {len(scores)} edges")
        return scores
    
    def _compute_edge_betweenness(self, graph: nx.DiGraph) -> Dict[Tuple[str, str], float]:
        """
        Compute normalized edge betweenness centrality
        
        Edge betweenness measures how often an edge appears on shortest paths
        between all pairs of nodes.
        
        Args:
            graph: NetworkX directed graph
            
        Returns:
            Dictionary mapping edge tuples to betweenness scores [0,1]
        """
        self.logger.debug("Computing edge betweenness centrality...")
        
        try:
            # Compute edge betweenness (normalized)
            betweenness = nx.edge_betweenness_centrality(graph, normalized=True)
            
            # Ensure all edges are present
            for edge in graph.edges():
                if edge not in betweenness:
                    betweenness[edge] = 0.0
            
            return betweenness
            
        except Exception as e:
            self.logger.error(f"Error computing edge betweenness: {e}")
            # Return zero scores for all edges
            return {edge: 0.0 for edge in graph.edges()}
    
    def _identify_bridges(self, graph: nx.DiGraph) -> Set[Tuple[str, str]]:
        """
        Identify bridge edges (critical links)
        
        A bridge is an edge whose removal increases the number of connected components.
        For directed graphs, we check both directed and undirected bridges.
        
        Args:
            graph: NetworkX directed graph
            
        Returns:
            Set of bridge edge tuples
        """
        self.logger.debug("Identifying bridge edges...")
        
        bridges = set()
        
        try:
            # For directed graphs, convert to undirected for bridge detection
            G_undirected = graph.to_undirected()
            
            # Find bridges in undirected graph
            undirected_bridges = set(nx.bridges(G_undirected))
            
            # Map back to directed edges
            for u, v in undirected_bridges:
                # Check both directions
                if graph.has_edge(u, v):
                    bridges.add((u, v))
                if graph.has_edge(v, u):
                    bridges.add((v, u))
            
            self.logger.debug(f"Found {len(bridges)} bridge edges")
            return bridges
            
        except Exception as e:
            self.logger.error(f"Error identifying bridges: {e}")
            return set()
    
    def _compute_edge_impacts(self, graph: nx.DiGraph) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Compute impact of removing each edge
        
        Measures connectivity changes when edges are removed.
        
        Args:
            graph: NetworkX directed graph
            
        Returns:
            Dictionary mapping edge tuples to impact metrics
        """
        self.logger.debug("Computing edge removal impacts...")
        
        impacts = {}
        original_components = nx.number_weakly_connected_components(graph)
        
        try:
            # For each edge, simulate removal
            for edge in graph.edges():
                u, v = edge
                
                # Create test graph without this edge
                G_test = graph.copy()
                G_test.remove_edge(u, v)
                
                # Measure impact
                new_components = nx.number_weakly_connected_components(G_test)
                creates_disconnection = new_components > original_components
                
                impacts[edge] = {
                    'components_after_removal': new_components,
                    'creates_disconnection': creates_disconnection,
                    'component_increase': new_components - original_components
                }
            
            return impacts
            
        except Exception as e:
            self.logger.error(f"Error computing edge impacts: {e}")
            return {}
    
    def _calculate_composite_scores(
        self,
        graph: nx.DiGraph,
        edge_betweenness: Dict[Tuple[str, str], float],
        bridges: Set[Tuple[str, str]],
        edge_impacts: Dict[Tuple[str, str], Dict[str, Any]]
    ) -> Dict[Tuple[str, str], EdgeCriticalityScore]:
        """
        Calculate composite criticality scores for all edges
        
        Combines:
        - Edge betweenness centrality (α weight)
        - Bridge/disconnection indicator (β weight)
        
        Args:
            graph: NetworkX directed graph
            edge_betweenness: Edge betweenness scores
            bridges: Set of bridge edges
            edge_impacts: Edge removal impacts
            
        Returns:
            Dictionary mapping edge tuples to EdgeCriticalityScore
        """
        self.logger.debug("Calculating composite criticality scores...")
        
        scores = {}
        
        for edge in graph.edges():
            u, v = edge
            edge_data = graph.edges[edge]
            edge_type = edge_data.get('type', 'Unknown')
            
            # Get metrics
            betweenness = edge_betweenness.get(edge, 0.0)
            is_bridge = edge in bridges
            impact = edge_impacts.get(edge, {})
            creates_disconnection = impact.get('creates_disconnection', False)
            components_after = impact.get('components_after_removal', 1)
            
            # Calculate composite score
            # C_edge(e) = α * BC_norm(e) + β * Bridge(e)
            bridge_indicator = 1.0 if is_bridge or creates_disconnection else 0.0
            composite = (self.alpha * betweenness) + (self.beta * bridge_indicator)
            
            # Determine criticality level
            if is_bridge or creates_disconnection:
                level = EdgeCriticalityLevel.CRITICAL
            elif betweenness >= 0.1:
                level = EdgeCriticalityLevel.HIGH
            elif betweenness >= 0.05:
                level = EdgeCriticalityLevel.MEDIUM
            else:
                level = EdgeCriticalityLevel.LOW
            
            # Create score object
            score = EdgeCriticalityScore(
                source=u,
                target=v,
                edge_type=edge_type,
                betweenness_centrality=betweenness,
                is_bridge=is_bridge,
                creates_disconnection=creates_disconnection,
                components_after_removal=components_after,
                criticality_level=level,
                composite_score=composite
            )
            
            scores[edge] = score
        
        return scores
    
    def get_top_critical_edges(
        self,
        scores: Dict[Tuple[str, str], EdgeCriticalityScore],
        n: int = 10,
        min_score: float = 0.0
    ) -> List[EdgeCriticalityScore]:
        """
        Get top N most critical edges
        
        Args:
            scores: Edge criticality scores
            n: Number of top edges to return
            min_score: Minimum composite score threshold
            
        Returns:
            List of top N EdgeCriticalityScore objects, sorted by composite score
        """
        # Filter and sort
        filtered = [
            score for score in scores.values()
            if score.composite_score >= min_score
        ]
        
        sorted_scores = sorted(
            filtered,
            key=lambda x: x.composite_score,
            reverse=True
        )
        
        return sorted_scores[:n]
    
    def get_bridges(
        self,
        scores: Dict[Tuple[str, str], EdgeCriticalityScore]
    ) -> List[EdgeCriticalityScore]:
        """
        Get all bridge edges
        
        Args:
            scores: Edge criticality scores
            
        Returns:
            List of EdgeCriticalityScore objects for bridge edges
        """
        return [
            score for score in scores.values()
            if score.is_bridge
        ]
    
    def get_edges_by_type(
        self,
        scores: Dict[Tuple[str, str], EdgeCriticalityScore],
        edge_type: str
    ) -> List[EdgeCriticalityScore]:
        """
        Get edges of a specific type
        
        Args:
            scores: Edge criticality scores
            edge_type: Type of edges to filter (e.g., 'PUBLISHES_TO', 'RUNS_ON')
            
        Returns:
            List of EdgeCriticalityScore objects for specified type
        """
        return [
            score for score in scores.values()
            if score.edge_type == edge_type
        ]
    
    def summarize_edge_criticality(
        self,
        scores: Dict[Tuple[str, str], EdgeCriticalityScore]
    ) -> Dict[str, Any]:
        """
        Generate summary statistics for edge criticality
        
        Args:
            scores: Edge criticality scores
            
        Returns:
            Dictionary with summary statistics
        """
        if not scores:
            return {
                'total_edges': 0,
                'critical_edges': 0,
                'high_edges': 0,
                'medium_edges': 0,
                'low_edges': 0,
                'bridge_count': 0,
                'avg_betweenness': 0.0,
                'max_betweenness': 0.0
            }
        
        # Count by level
        level_counts = {
            EdgeCriticalityLevel.CRITICAL: 0,
            EdgeCriticalityLevel.HIGH: 0,
            EdgeCriticalityLevel.MEDIUM: 0,
            EdgeCriticalityLevel.LOW: 0
        }
        
        bridge_count = 0
        betweenness_values = []
        
        for score in scores.values():
            level_counts[score.criticality_level] += 1
            if score.is_bridge:
                bridge_count += 1
            betweenness_values.append(score.betweenness_centrality)
        
        return {
            'total_edges': len(scores),
            'critical_edges': level_counts[EdgeCriticalityLevel.CRITICAL],
            'high_edges': level_counts[EdgeCriticalityLevel.HIGH],
            'medium_edges': level_counts[EdgeCriticalityLevel.MEDIUM],
            'low_edges': level_counts[EdgeCriticalityLevel.LOW],
            'bridge_count': bridge_count,
            'bridge_percentage': round(100 * bridge_count / len(scores), 2),
            'avg_betweenness': round(sum(betweenness_values) / len(betweenness_values), 4),
            'max_betweenness': round(max(betweenness_values), 4),
            'min_betweenness': round(min(betweenness_values), 4)
        }
    
    def generate_edge_recommendations(
        self,
        scores: Dict[Tuple[str, str], EdgeCriticalityScore],
        top_n: int = 5
    ) -> List[Dict[str, str]]:
        """
        Generate recommendations for critical edges
        
        Args:
            scores: Edge criticality scores
            top_n: Number of top edges to generate recommendations for
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Get top critical edges
        top_edges = self.get_top_critical_edges(scores, n=top_n)
        
        for score in top_edges:
            if score.is_bridge:
                recommendations.append({
                    'priority': 'CRITICAL',
                    'type': 'Bridge Edge',
                    'edge': f'{score.source} → {score.target}',
                    'issue': 'This connection is a bridge - its failure will partition the network',
                    'recommendation': 'Add redundant communication path or establish failover mechanism',
                    'impact': 'Network fragmentation',
                    'score': score.composite_score
                })
            elif score.criticality_level == EdgeCriticalityLevel.HIGH:
                recommendations.append({
                    'priority': 'HIGH',
                    'type': 'Communication Bottleneck',
                    'edge': f'{score.source} → {score.target}',
                    'issue': f'High betweenness ({score.betweenness_centrality:.3f}) indicates traffic bottleneck',
                    'recommendation': 'Consider load balancing or increasing communication capacity',
                    'impact': 'Latency and throughput degradation',
                    'score': score.composite_score
                })
            elif score.criticality_level == EdgeCriticalityLevel.MEDIUM:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'type': 'Moderate Bottleneck',
                    'edge': f'{score.source} → {score.target}',
                    'issue': f'Moderate betweenness ({score.betweenness_centrality:.3f})',
                    'recommendation': 'Monitor performance and consider optimization',
                    'impact': 'Potential performance issues under load',
                    'score': score.composite_score
                })
        
        return recommendations


# Example usage
if __name__ == '__main__':
    # Create example graph
    G = nx.DiGraph()
    
    # Add nodes
    nodes = ['A', 'B', 'C', 'D', 'E', 'F']
    G.add_nodes_from(nodes)
    
    # Add edges creating a bridge
    edges = [
        ('A', 'B'), ('B', 'C'),  # Cluster 1
        ('C', 'D'),              # Bridge
        ('D', 'E'), ('E', 'F'), ('F', 'D')  # Cluster 2
    ]
    G.add_edges_from(edges)
    
    # Analyze
    analyzer = EdgeCriticalityAnalyzer()
    scores = analyzer.analyze(G)
    
    # Print results
    print("\n=== Edge Criticality Analysis ===\n")
    
    summary = analyzer.summarize_edge_criticality(scores)
    print(f"Total Edges: {summary['total_edges']}")
    print(f"Bridges: {summary['bridge_count']}")
    print(f"Critical Edges: {summary['critical_edges']}")
    print(f"Avg Betweenness: {summary['avg_betweenness']:.4f}")
    
    print("\n=== Top 5 Critical Edges ===\n")
    top_edges = analyzer.get_top_critical_edges(scores, n=5)
    for i, score in enumerate(top_edges, 1):
        print(f"{i}. {score.source} → {score.target}")
        print(f"   Score: {score.composite_score:.4f}")
        print(f"   Bridge: {score.is_bridge}")
        print(f"   Betweenness: {score.betweenness_centrality:.4f}")
        print()
    
    print("=== Recommendations ===\n")
    recommendations = analyzer.generate_edge_recommendations(scores, top_n=3)
    for rec in recommendations:
        print(f"[{rec['priority']}] {rec['type']}")
        print(f"Edge: {rec['edge']}")
        print(f"Issue: {rec['issue']}")
        print(f"Recommendation: {rec['recommendation']}")
        print()
