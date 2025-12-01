"""
Composite Criticality Scorer

Implements the formal criticality scoring model:
C_score(v) = α·C_B^norm(v) + β·AP(v) + γ·I(v)

where:
- C_B^norm(v) ∈ [0,1] is normalized betweenness centrality
- AP(v) ∈ {0,1} indicates if v is an articulation point
- I(v) ∈ [0,1] is the impact score measuring reachability loss
- α, β, γ are weight parameters (default: α=0.4, β=0.3, γ=0.3)

The impact score is defined as:
I(v) = 1 - |R(G-v)| / |R(G)|

where R(G) represents the set of all reachable vertex pairs in graph G.

Research targets:
- Spearman correlation ≥ 0.7 with failure simulations
- F1-score ≥ 0.9 for critical component identification
- Precision ≥ 0.9, Recall ≥ 0.85
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict


class CriticalityLevel(Enum):
    """Criticality classification levels based on composite score thresholds"""
    CRITICAL = "CRITICAL"    # ≥ 0.8: Immediate attention required
    HIGH = "HIGH"            # ≥ 0.6: High priority monitoring
    MEDIUM = "MEDIUM"        # ≥ 0.4: Regular monitoring
    LOW = "LOW"              # ≥ 0.2: Routine checks
    MINIMAL = "MINIMAL"      # < 0.2: Low concern


@dataclass
class CompositeCriticalityScore:
    """
    Complete criticality assessment for a component.
    
    Contains all metrics used in the composite scoring formula
    plus additional context for analysis and reporting.
    """
    component: str
    component_type: str
    
    # Core formula components
    betweenness_centrality_norm: float  # C_B^norm(v) ∈ [0,1]
    is_articulation_point: float        # AP(v) ∈ {0,1}
    impact_score: float                 # I(v) ∈ [0,1]
    
    # Composite result
    composite_score: float              # C_score(v)
    criticality_level: CriticalityLevel
    
    # Additional centrality metrics for context
    degree_centrality: float = 0.0
    closeness_centrality: float = 0.0
    pagerank: float = 0.0
    
    # QoS adjustment (for topics)
    qos_score: float = 0.0
    
    # Impact analysis details
    components_affected: int = 0
    reachability_loss_percentage: float = 0.0
    
    # Weights used
    weights: Dict[str, float] = field(default_factory=dict)
    
    def __repr__(self):
        return (f"CriticalityScore({self.component}, "
                f"score={self.composite_score:.3f}, "
                f"level={self.criticality_level.value})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'component': self.component,
            'component_type': self.component_type,
            'composite_score': round(self.composite_score, 4),
            'criticality_level': self.criticality_level.value,
            'betweenness_centrality_norm': round(self.betweenness_centrality_norm, 4),
            'is_articulation_point': bool(self.is_articulation_point),
            'impact_score': round(self.impact_score, 4),
            'degree_centrality': round(self.degree_centrality, 4),
            'closeness_centrality': round(self.closeness_centrality, 4),
            'pagerank': round(self.pagerank, 4),
            'qos_score': round(self.qos_score, 4),
            'components_affected': self.components_affected,
            'reachability_loss_percentage': round(self.reachability_loss_percentage, 2),
            'weights': self.weights
        }


class CompositeCriticalityScorer:
    """
    Calculates composite criticality scores for graph components.
    
    Implements the formal mathematical model with configurable weights
    and optional QoS-aware adjustments for topic nodes.
    """
    
    # Default weight parameters
    DEFAULT_ALPHA = 0.4  # Betweenness centrality weight
    DEFAULT_BETA = 0.3   # Articulation point weight
    DEFAULT_GAMMA = 0.3  # Impact score weight
    
    # Criticality thresholds
    THRESHOLDS = {
        CriticalityLevel.CRITICAL: 0.8,
        CriticalityLevel.HIGH: 0.6,
        CriticalityLevel.MEDIUM: 0.4,
        CriticalityLevel.LOW: 0.2
    }
    
    def __init__(self,
                 alpha: float = DEFAULT_ALPHA,
                 beta: float = DEFAULT_BETA,
                 gamma: float = DEFAULT_GAMMA,
                 qos_enabled: bool = True):
        """
        Initialize the scorer with weight parameters.
        
        Args:
            alpha: Weight for normalized betweenness centrality
            beta: Weight for articulation point indicator
            gamma: Weight for impact score
            qos_enabled: Whether to apply QoS adjustments
        """
        # Validate weights
        if not np.isclose(alpha + beta + gamma, 1.0, atol=0.01):
            raise ValueError(f"Weights must sum to 1.0, got {alpha + beta + gamma}")
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.qos_enabled = qos_enabled
        
        self.weights = {'alpha': alpha, 'beta': beta, 'gamma': gamma}
        
        # Cache for centrality calculations
        self._centrality_cache: Dict[str, Dict] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_all_scores(self,
                            graph: nx.DiGraph,
                            qos_scores: Optional[Dict[str, float]] = None
                            ) -> Dict[str, CompositeCriticalityScore]:
        """
        Calculate composite criticality scores for all nodes.
        
        Args:
            graph: NetworkX directed graph
            qos_scores: Optional QoS scores for topics
        
        Returns:
            Dictionary mapping node IDs to CompositeCriticalityScore objects
        """
        self.logger.info(f"Calculating criticality scores for {graph.number_of_nodes()} nodes...")
        
        # Clear cache
        self._centrality_cache.clear()
        
        # Step 1: Calculate normalized betweenness centrality
        betweenness = self._calculate_normalized_betweenness(graph)
        
        # Step 2: Identify articulation points
        articulation_points = self._identify_articulation_points(graph)
        
        # Step 3: Calculate additional centrality metrics
        degree = self._calculate_degree_centrality(graph)
        closeness = self._calculate_closeness_centrality(graph)
        pagerank = self._calculate_pagerank(graph)
        
        # Step 4: Calculate impact scores and composite scores
        scores = {}
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            node_type = node_data.get('type', 'Unknown')
            
            # Get QoS score if available
            qos_score = 0.0
            if qos_scores and node in qos_scores:
                qos_score = qos_scores[node]
            elif self.qos_enabled and 'qos_score' in node_data:
                qos_score = node_data['qos_score']
            
            # Core formula components
            bc_norm = betweenness.get(node, 0.0)
            ap_indicator = 1.0 if node in articulation_points else 0.0
            
            # Calculate impact score
            impact, affected, loss_pct = self._calculate_impact_score(graph, node)
            
            # Calculate composite score
            composite = (
                self.alpha * bc_norm +
                self.beta * ap_indicator +
                self.gamma * impact
            )
            
            # Apply QoS adjustment for topics
            if self.qos_enabled and node_type == 'Topic' and qos_score > 0:
                composite = composite * (1 + qos_score * 0.5)  # Up to 50% boost
                composite = min(1.0, composite)
            
            # Determine criticality level
            level = self._determine_criticality_level(composite)
            
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
                components_affected=affected,
                reachability_loss_percentage=loss_pct,
                weights=self.weights.copy()
            )
        
        self.logger.info(f"Calculated scores for {len(scores)} components")
        return scores
    
    def _calculate_normalized_betweenness(self, graph: nx.DiGraph) -> Dict[str, float]:
        """
        Calculate normalized betweenness centrality C_B^norm(v) ∈ [0,1]
        """
        if 'betweenness' in self._centrality_cache:
            return self._centrality_cache['betweenness']
        
        self.logger.debug("Computing betweenness centrality...")
        
        betweenness = nx.betweenness_centrality(graph, normalized=True)
        
        # Additional normalization to ensure max is 1.0
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
        """
        if 'articulation_points' in self._centrality_cache:
            return self._centrality_cache['articulation_points']
        
        self.logger.debug("Identifying articulation points...")
        
        # Convert to undirected for articulation point detection
        undirected = graph.to_undirected()
        aps = set(nx.articulation_points(undirected))
        
        self._centrality_cache['articulation_points'] = aps
        return aps
    
    def _calculate_impact_score(self, 
                               graph: nx.DiGraph, 
                               node: str) -> Tuple[float, int, float]:
        """
        Calculate impact score I(v) for removing a node.
        
        I(v) = 1 - |R(G-v)| / |R(G)|
        
        Returns:
            Tuple of (impact_score, components_affected, reachability_loss_percentage)
        """
        # Calculate original reachability
        original_reachable = set()
        for source in graph.nodes():
            try:
                descendants = nx.descendants(graph, source)
                for target in descendants:
                    if source != target:
                        original_reachable.add((source, target))
            except:
                pass
        
        # Calculate reachability without the node
        G_minus_v = graph.copy()
        G_minus_v.remove_node(node)
        
        new_reachable = set()
        for source in G_minus_v.nodes():
            try:
                descendants = nx.descendants(G_minus_v, source)
                for target in descendants:
                    if source != target:
                        new_reachable.add((source, target))
            except:
                pass
        
        # Calculate loss
        lost_reachability = original_reachable - new_reachable
        affected_nodes = set(p[0] for p in lost_reachability) | set(p[1] for p in lost_reachability)
        components_affected = len(affected_nodes)
        
        if len(original_reachable) > 0:
            reachability_loss_pct = (len(lost_reachability) / len(original_reachable)) * 100
            impact_score = len(lost_reachability) / len(original_reachable)
        else:
            reachability_loss_pct = 0.0
            impact_score = 0.0
        
        return min(1.0, impact_score), components_affected, reachability_loss_pct
    
    def _calculate_degree_centrality(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate degree centrality"""
        if 'degree' in self._centrality_cache:
            return self._centrality_cache['degree']
        
        degree = nx.degree_centrality(graph)
        self._centrality_cache['degree'] = degree
        return degree
    
    def _calculate_closeness_centrality(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate closeness centrality"""
        if 'closeness' in self._centrality_cache:
            return self._centrality_cache['closeness']
        
        closeness = nx.closeness_centrality(graph)
        self._centrality_cache['closeness'] = closeness
        return closeness
    
    def _calculate_pagerank(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate PageRank"""
        if 'pagerank' in self._centrality_cache:
            return self._centrality_cache['pagerank']
        
        try:
            pagerank = nx.pagerank(graph, alpha=0.85)
        except:
            pagerank = {n: 0.0 for n in graph.nodes()}
        
        self._centrality_cache['pagerank'] = pagerank
        return pagerank
    
    def _determine_criticality_level(self, score: float) -> CriticalityLevel:
        """Determine criticality level from composite score"""
        if score >= self.THRESHOLDS[CriticalityLevel.CRITICAL]:
            return CriticalityLevel.CRITICAL
        elif score >= self.THRESHOLDS[CriticalityLevel.HIGH]:
            return CriticalityLevel.HIGH
        elif score >= self.THRESHOLDS[CriticalityLevel.MEDIUM]:
            return CriticalityLevel.MEDIUM
        elif score >= self.THRESHOLDS[CriticalityLevel.LOW]:
            return CriticalityLevel.LOW
        else:
            return CriticalityLevel.MINIMAL
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_top_critical(self,
                        scores: Dict[str, CompositeCriticalityScore],
                        n: int = 10) -> List[CompositeCriticalityScore]:
        """Get top N most critical components"""
        sorted_scores = sorted(scores.values(), 
                              key=lambda s: s.composite_score, 
                              reverse=True)
        return sorted_scores[:n]
    
    def get_critical_components(self,
                               scores: Dict[str, CompositeCriticalityScore],
                               threshold: float = 0.6) -> List[CompositeCriticalityScore]:
        """Get all components above criticality threshold"""
        return [s for s in scores.values() if s.composite_score >= threshold]
    
    def get_by_level(self,
                    scores: Dict[str, CompositeCriticalityScore],
                    level: CriticalityLevel) -> List[CompositeCriticalityScore]:
        """Get all components at a specific criticality level"""
        return [s for s in scores.values() if s.criticality_level == level]
    
    def get_by_type(self,
                   scores: Dict[str, CompositeCriticalityScore],
                   component_type: str) -> List[CompositeCriticalityScore]:
        """Get all scores for a specific component type"""
        return [s for s in scores.values() if s.component_type == component_type]
    
    def summarize_criticality(self,
                             scores: Dict[str, CompositeCriticalityScore]) -> Dict[str, Any]:
        """Generate summary statistics for criticality scores"""
        all_scores = [s.composite_score for s in scores.values()]
        
        return {
            'total_components': len(scores),
            'avg_score': round(np.mean(all_scores), 4) if all_scores else 0,
            'std_score': round(np.std(all_scores), 4) if all_scores else 0,
            'max_score': round(max(all_scores), 4) if all_scores else 0,
            'min_score': round(min(all_scores), 4) if all_scores else 0,
            'by_level': {
                level.value: sum(1 for s in scores.values() 
                                if s.criticality_level == level)
                for level in CriticalityLevel
            },
            'articulation_points': sum(1 for s in scores.values() 
                                       if s.is_articulation_point),
            'weights_used': self.weights
        }