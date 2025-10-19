"""
Composite Criticality Scorer

Implements the formal definition:
C_score(v) = α·C_B^norm(v) + β·AP(v) + γ·I(v)

where:
- C_B^norm(v) ∈ [0,1] is normalized betweenness centrality
- AP(v) ∈ {0,1} indicates if v is an articulation point
- I(v) ∈ [0,1] is the impact score measuring reachability loss
- α, β, γ are weight parameters (default: α=0.4, β=0.3, γ=0.3)

The impact score is defined as:
I(v) = 1 - |R(G-v)| / |R(G)|

where R(G) represents the set of all reachable vertex pairs in graph G.
"""

import networkx as nx
import numpy as np
from typing import Dict, Tuple, List, Set
from dataclasses import dataclass
from enum import Enum
import logging


class CriticalityLevel(Enum):
    """Criticality classification levels"""
    CRITICAL = "CRITICAL"  # Top tier - immediate attention
    HIGH = "HIGH"  # Second tier - high priority
    MEDIUM = "MEDIUM"  # Third tier - monitor closely
    LOW = "LOW"  # Fourth tier - routine monitoring
    MINIMAL = "MINIMAL"  # Bottom tier - low concern


@dataclass
class CompositeCriticalityScore:
    """Complete criticality assessment for a component"""
    component: str
    component_type: str
    
    # Core metrics
    betweenness_centrality_norm: float  # C_B^norm(v)
    is_articulation_point: float  # AP(v)
    impact_score: float  # I(v)
    
    # Composite score
    composite_score: float  # C_score(v)
    criticality_level: CriticalityLevel
    
    # Additional context
    degree_centrality: float
    closeness_centrality: float
    pagerank: float
    qos_score: float
    
    # Detailed impact metrics
    components_affected: int
    reachability_loss_percentage: float
    
    # Weights used
    weights: Dict[str, float]
    
    def __repr__(self):
        return (f"CriticalityScore(component={self.component}, "
                f"score={self.composite_score:.3f}, "
                f"level={self.criticality_level.value})")


class CompositeCriticalityScorer:
    """
    Calculates composite criticality scores for graph components
    
    Implements the formal mathematical model with configurable weights
    """
    
    def __init__(self, 
                 alpha: float = 0.4,  # Weight for betweenness centrality
                 beta: float = 0.3,   # Weight for articulation points
                 gamma: float = 0.3,  # Weight for impact score
                 qos_enabled: bool = True):
        """
        Initialize the scorer
        
        Args:
            alpha: Weight for betweenness centrality (default: 0.4)
            beta: Weight for articulation point indicator (default: 0.3)
            gamma: Weight for impact score (default: 0.3)
            qos_enabled: Whether to incorporate QoS scores
        """
        # Validate weights sum to 1.0
        total = alpha + beta + gamma
        if not np.isclose(total, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.qos_enabled = qos_enabled
        
        self.weights = {
            'betweenness': alpha,
            'articulation_point': beta,
            'impact': gamma
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Caches for expensive computations
        self._centrality_cache = {}
        self._reachability_cache = {}
        self._articulation_cache = {}
    
    def calculate_all_scores(self, 
                            graph: nx.DiGraph,
                            qos_scores: Dict[str, float] = None) -> Dict[str, CompositeCriticalityScore]:
        """
        Calculate composite criticality scores for all components
        
        Args:
            graph: NetworkX directed graph
            qos_scores: Optional QoS scores for components
        
        Returns:
            Dictionary mapping component names to CompositeCriticalityScore objects
        """
        self.logger.info(f"Calculating composite criticality scores for {len(graph)} components...")
        
        # Clear caches
        self._centrality_cache.clear()
        self._reachability_cache.clear()
        self._articulation_cache.clear()
        
        # Step 1: Calculate normalized betweenness centrality
        betweenness = self._calculate_normalized_betweenness(graph)
        
        # Step 2: Identify articulation points
        articulation_points = self._identify_articulation_points(graph)
        
        # Step 3: Calculate impact scores
        impact_scores = self._calculate_impact_scores(graph)
        
        # Step 4: Calculate additional centrality metrics
        degree = self._calculate_degree_centrality(graph)
        closeness = self._calculate_closeness_centrality(graph)
        pagerank = self._calculate_pagerank(graph)
        
        # Step 5: Calculate composite scores
        scores = {}
        for node in graph.nodes():
            # Get component type
            node_type = graph.nodes[node].get('type', 'Unknown')
            
            # Get QoS score if available
            qos_score = 0.0
            if qos_scores and node in qos_scores:
                qos_score = qos_scores[node]
            elif self.qos_enabled and 'qos_score' in graph.nodes[node]:
                qos_score = graph.nodes[node]['qos_score']
            
            # Calculate composite score
            bc_norm = betweenness.get(node, 0.0)
            ap_indicator = 1.0 if node in articulation_points else 0.0
            impact = impact_scores.get(node, 0.0)
            
            composite = (
                self.alpha * bc_norm +
                self.beta * ap_indicator +
                self.gamma * impact
            )
            
            # Adjust for QoS if enabled and topic
            if self.qos_enabled and node_type == 'Topic' and qos_score > 0:
                # QoS acts as a multiplier for topics
                composite = composite * (1 + qos_score * 0.5)  # Up to 50% boost
                composite = min(1.0, composite)  # Cap at 1.0
            
            # Determine criticality level
            level = self._determine_criticality_level(composite)
            
            # Calculate components affected
            components_affected = self._count_affected_components(graph, node)
            reachability_loss = impact * 100  # Convert to percentage
            
            scores[node] = CompositeCriticalityScore(
                component=node,
                component_type=node_type,
                betweenness_centrality_norm=bc_norm,
                is_articulation_point=ap_indicator,
                impact_score=impact,
                composite_score=composite,
                criticality_level=level,
                degree_centrality=degree.get(node, 0.0),
                closeness_centrality=closeness.get(node, 0.0),
                pagerank=pagerank.get(node, 0.0),
                qos_score=qos_score,
                components_affected=components_affected,
                reachability_loss_percentage=reachability_loss,
                weights=self.weights.copy()
            )
        
        self.logger.info(f"Calculated scores for {len(scores)} components")
        return scores
    
    def _calculate_normalized_betweenness(self, graph: nx.DiGraph) -> Dict[str, float]:
        """
        Calculate normalized betweenness centrality C_B^norm(v) ∈ [0,1]
        
        Args:
            graph: NetworkX directed graph
        
        Returns:
            Dictionary mapping nodes to normalized betweenness centrality
        """
        if 'betweenness' in self._centrality_cache:
            return self._centrality_cache['betweenness']
        
        self.logger.debug("Computing betweenness centrality...")
        
        # Calculate betweenness centrality
        betweenness = nx.betweenness_centrality(graph, normalized=True)
        
        # Ensure all values are in [0,1]
        max_bc = max(betweenness.values()) if betweenness else 1.0
        if max_bc > 0:
            normalized = {k: v / max_bc for k, v in betweenness.items()}
        else:
            normalized = {k: 0.0 for k in betweenness}
        
        self._centrality_cache['betweenness'] = normalized
        return normalized
    
    def _identify_articulation_points(self, graph: nx.DiGraph) -> Set[str]:
        """
        Identify articulation points AP(v) ∈ {0,1}
        
        An articulation point is a node whose removal increases the number
        of connected components.
        
        Args:
            graph: NetworkX directed graph
        
        Returns:
            Set of node names that are articulation points
        """
        if 'articulation_points' in self._articulation_cache:
            return self._articulation_cache['articulation_points']
        
        self.logger.debug("Identifying articulation points...")
        
        # Convert to undirected for articulation point detection
        undirected = graph.to_undirected()
        
        # Find articulation points
        articulation_points = set(nx.articulation_points(undirected))
        
        self._articulation_cache['articulation_points'] = articulation_points
        self.logger.debug(f"Found {len(articulation_points)} articulation points")
        
        return articulation_points
    
    def _calculate_impact_scores(self, graph: nx.DiGraph) -> Dict[str, float]:
        """
        Calculate impact scores I(v) = 1 - |R(G-v)| / |R(G)|
        
        Where R(G) is the set of reachable vertex pairs in graph G.
        
        Args:
            graph: NetworkX directed graph
        
        Returns:
            Dictionary mapping nodes to impact scores [0,1]
        """
        if 'impact_scores' in self._reachability_cache:
            return self._reachability_cache['impact_scores']
        
        self.logger.debug("Calculating impact scores...")
        
        # Calculate baseline reachability R(G)
        baseline_reachability = self._count_reachable_pairs(graph)
        
        if baseline_reachability == 0:
            # Graph has no reachable pairs - all nodes have zero impact
            impact_scores = {node: 0.0 for node in graph.nodes()}
            self._reachability_cache['impact_scores'] = impact_scores
            return impact_scores
        
        # Calculate impact for each node
        impact_scores = {}
        
        for node in graph.nodes():
            # Create graph without this node
            graph_without_node = graph.copy()
            graph_without_node.remove_node(node)
            
            # Calculate reachability without node
            reachability_without = self._count_reachable_pairs(graph_without_node)
            
            # Calculate impact score
            impact = 1.0 - (reachability_without / baseline_reachability)
            impact_scores[node] = max(0.0, min(1.0, impact))  # Clamp to [0,1]
        
        self._reachability_cache['impact_scores'] = impact_scores
        return impact_scores
    
    def _count_reachable_pairs(self, graph: nx.DiGraph) -> int:
        """
        Count the number of reachable vertex pairs |R(G)|
        
        Args:
            graph: NetworkX directed graph
        
        Returns:
            Number of reachable pairs
        """
        count = 0
        nodes = list(graph.nodes())
        
        for source in nodes:
            # Find all nodes reachable from source
            reachable = nx.descendants(graph, source)
            count += len(reachable)
        
        return count
    
    def _calculate_degree_centrality(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate degree centrality for all nodes"""
        if 'degree' in self._centrality_cache:
            return self._centrality_cache['degree']
        
        degree = nx.degree_centrality(graph)
        self._centrality_cache['degree'] = degree
        return degree
    
    def _calculate_closeness_centrality(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate closeness centrality for all nodes"""
        if 'closeness' in self._centrality_cache:
            return self._centrality_cache['closeness']
        
        try:
            closeness = nx.closeness_centrality(graph)
        except:
            # If graph is not connected, calculate for each component
            closeness = {}
            for component in nx.weakly_connected_components(graph):
                subgraph = graph.subgraph(component)
                comp_closeness = nx.closeness_centrality(subgraph)
                closeness.update(comp_closeness)
        
        self._centrality_cache['closeness'] = closeness
        return closeness
    
    def _calculate_pagerank(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate PageRank for all nodes"""
        if 'pagerank' in self._centrality_cache:
            return self._centrality_cache['pagerank']
        
        try:
            pagerank = nx.pagerank(graph)
        except:
            pagerank = {node: 1.0 / len(graph) for node in graph.nodes()}
        
        self._centrality_cache['pagerank'] = pagerank
        return pagerank
    
    def _count_affected_components(self, graph: nx.DiGraph, node: str) -> int:
        """Count how many components would be affected by removing this node"""
        # Components that depend on this node
        dependents = list(graph.successors(node))
        
        # Components this node depends on
        dependencies = list(graph.predecessors(node))
        
        # Total unique affected components
        affected = set(dependents + dependencies)
        return len(affected)
    
    def _determine_criticality_level(self, score: float) -> CriticalityLevel:
        """
        Determine criticality level from composite score
        
        Args:
            score: Composite criticality score [0,1]
        
        Returns:
            CriticalityLevel enum
        """
        if score >= 0.8:
            return CriticalityLevel.CRITICAL
        elif score >= 0.6:
            return CriticalityLevel.HIGH
        elif score >= 0.4:
            return CriticalityLevel.MEDIUM
        elif score >= 0.2:
            return CriticalityLevel.LOW
        else:
            return CriticalityLevel.MINIMAL
    
    def get_critical_components(self, 
                               scores: Dict[str, CompositeCriticalityScore],
                               threshold: float = 0.6) -> List[CompositeCriticalityScore]:
        """
        Get components with criticality above threshold
        
        Args:
            scores: Dictionary of criticality scores
            threshold: Minimum score to be considered critical
        
        Returns:
            Sorted list of critical components (highest first)
        """
        critical = [
            score for score in scores.values()
            if score.composite_score >= threshold
        ]
        
        return sorted(critical, key=lambda x: x.composite_score, reverse=True)
    
    def get_top_critical(self, 
                        scores: Dict[str, CompositeCriticalityScore],
                        n: int = 10) -> List[CompositeCriticalityScore]:
        """
        Get top N critical components
        
        Args:
            scores: Dictionary of criticality scores
            n: Number of top components to return
        
        Returns:
            Sorted list of top N critical components
        """
        all_scores = list(scores.values())
        sorted_scores = sorted(all_scores, key=lambda x: x.composite_score, reverse=True)
        return sorted_scores[:n]
    
    def summarize_criticality(self, 
                             scores: Dict[str, CompositeCriticalityScore]) -> Dict:
        """
        Generate summary statistics of criticality distribution
        
        Args:
            scores: Dictionary of criticality scores
        
        Returns:
            Dictionary with summary statistics
        """
        if not scores:
            return {
                'total_components': 0,
                'critical_count': 0,
                'high_count': 0,
                'medium_count': 0,
                'low_count': 0,
                'minimal_count': 0
            }
        
        all_scores = list(scores.values())
        composite_values = [s.composite_score for s in all_scores]
        
        # Count by level
        level_counts = {
            CriticalityLevel.CRITICAL: 0,
            CriticalityLevel.HIGH: 0,
            CriticalityLevel.MEDIUM: 0,
            CriticalityLevel.LOW: 0,
            CriticalityLevel.MINIMAL: 0
        }
        
        for score in all_scores:
            level_counts[score.criticality_level] += 1
        
        return {
            'total_components': len(scores),
            'critical_count': level_counts[CriticalityLevel.CRITICAL],
            'high_count': level_counts[CriticalityLevel.HIGH],
            'medium_count': level_counts[CriticalityLevel.MEDIUM],
            'low_count': level_counts[CriticalityLevel.LOW],
            'minimal_count': level_counts[CriticalityLevel.MINIMAL],
            'avg_score': np.mean(composite_values),
            'std_score': np.std(composite_values),
            'min_score': np.min(composite_values),
            'max_score': np.max(composite_values),
            'articulation_points': sum(1 for s in all_scores if s.is_articulation_point > 0),
            'high_impact_count': sum(1 for s in all_scores if s.impact_score > 0.5)
        }
