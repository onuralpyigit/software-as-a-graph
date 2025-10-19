"""
Centrality Analyzer

Computes various centrality metrics to identify the most important nodes in the graph.
Supports multiple centrality measures including degree, betweenness, closeness, 
eigenvector, PageRank, and more.

Centrality metrics help identify:
- Hub nodes (high degree centrality)
- Bottlenecks (high betweenness centrality)
- Core components (high closeness centrality)
- Influential nodes (high eigenvector/PageRank)
"""

import networkx as nx
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np


class CentralityType(Enum):
    """Types of centrality metrics"""
    DEGREE = "degree"
    IN_DEGREE = "in_degree"
    OUT_DEGREE = "out_degree"
    BETWEENNESS = "betweenness"
    CLOSENESS = "closeness"
    EIGENVECTOR = "eigenvector"
    PAGERANK = "pagerank"
    KATZ = "katz"
    HARMONIC = "harmonic"
    LOAD = "load"
    CURRENT_FLOW_BETWEENNESS = "current_flow_betweenness"
    CURRENT_FLOW_CLOSENESS = "current_flow_closeness"
    COMMUNICABILITY_BETWEENNESS = "communicability_betweenness"


@dataclass
class CentralityScore:
    """Centrality score for a single node"""
    node: str
    score: float
    rank: int
    normalized_score: float
    percentile: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'node': self.node,
            'score': round(self.score, 6),
            'rank': self.rank,
            'normalized_score': round(self.normalized_score, 6),
            'percentile': round(self.percentile, 2)
        }


@dataclass
class CentralityAnalysisResult:
    """Complete centrality analysis results"""
    centrality_type: CentralityType
    scores: Dict[str, CentralityScore]
    top_nodes: List[str]
    statistics: Dict[str, float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'centrality_type': self.centrality_type.value,
            'scores': {node: score.to_dict() for node, score in self.scores.items()},
            'top_nodes': self.top_nodes,
            'statistics': {k: round(v, 6) for k, v in self.statistics.items()}
        }


class CentralityAnalyzer:
    """
    Analyzer for computing various centrality metrics
    
    Supports computation of:
    - Degree centrality (in, out, total)
    - Betweenness centrality
    - Closeness centrality
    - Eigenvector centrality
    - PageRank
    - Katz centrality
    - Harmonic centrality
    - Load centrality
    - Current flow metrics (for weighted graphs)
    - Communicability betweenness
    """
    
    def __init__(self):
        """Initialize the centrality analyzer"""
        self.logger = logging.getLogger(__name__)
        self._cache = {}
    
    def analyze_all_centralities(self,
                                 graph: nx.DiGraph,
                                 include_expensive: bool = False) -> Dict[CentralityType, CentralityAnalysisResult]:
        """
        Compute all centrality metrics
        
        Args:
            graph: NetworkX directed graph
            include_expensive: Whether to include computationally expensive metrics
        
        Returns:
            Dictionary mapping centrality type to results
        """
        self.logger.info("Computing all centrality metrics...")
        
        results = {}
        
        # Fast metrics (always computed)
        results[CentralityType.DEGREE] = self.compute_degree_centrality(graph)
        results[CentralityType.IN_DEGREE] = self.compute_in_degree_centrality(graph)
        results[CentralityType.OUT_DEGREE] = self.compute_out_degree_centrality(graph)
        results[CentralityType.BETWEENNESS] = self.compute_betweenness_centrality(graph)
        results[CentralityType.CLOSENESS] = self.compute_closeness_centrality(graph)
        results[CentralityType.PAGERANK] = self.compute_pagerank(graph)
        results[CentralityType.HARMONIC] = self.compute_harmonic_centrality(graph)
        
        # Try eigenvector (may fail for some graphs)
        try:
            results[CentralityType.EIGENVECTOR] = self.compute_eigenvector_centrality(graph)
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError) as e:
            self.logger.warning(f"Eigenvector centrality failed: {e}")
        
        # Expensive metrics (optional)
        if include_expensive:
            try:
                results[CentralityType.KATZ] = self.compute_katz_centrality(graph)
            except Exception as e:
                self.logger.warning(f"Katz centrality failed: {e}")
            
            try:
                results[CentralityType.LOAD] = self.compute_load_centrality(graph)
            except Exception as e:
                self.logger.warning(f"Load centrality failed: {e}")
            
            # Current flow metrics require connected graph
            if nx.is_weakly_connected(graph):
                try:
                    results[CentralityType.CURRENT_FLOW_BETWEENNESS] = \
                        self.compute_current_flow_betweenness(graph)
                except Exception as e:
                    self.logger.warning(f"Current flow betweenness failed: {e}")
                
                try:
                    results[CentralityType.CURRENT_FLOW_CLOSENESS] = \
                        self.compute_current_flow_closeness(graph)
                except Exception as e:
                    self.logger.warning(f"Current flow closeness failed: {e}")
            
            try:
                results[CentralityType.COMMUNICABILITY_BETWEENNESS] = \
                    self.compute_communicability_betweenness(graph)
            except Exception as e:
                self.logger.warning(f"Communicability betweenness failed: {e}")
        
        self.logger.info(f"Computed {len(results)} centrality metrics")
        return results
    
    def compute_degree_centrality(self, graph: nx.DiGraph) -> CentralityAnalysisResult:
        """
        Compute degree centrality (total degree)
        
        Degree centrality measures the number of connections a node has.
        High degree = hub node.
        """
        self.logger.debug("Computing degree centrality...")
        
        # Compute raw scores
        raw_scores = dict(graph.degree())
        
        return self._create_result(
            CentralityType.DEGREE,
            raw_scores,
            graph
        )
    
    def compute_in_degree_centrality(self, graph: nx.DiGraph) -> CentralityAnalysisResult:
        """
        Compute in-degree centrality
        
        In-degree centrality measures the number of incoming edges.
        High in-degree = popular/consumed node.
        """
        self.logger.debug("Computing in-degree centrality...")
        
        # Compute raw scores
        raw_scores = dict(graph.in_degree())
        
        return self._create_result(
            CentralityType.IN_DEGREE,
            raw_scores,
            graph
        )
    
    def compute_out_degree_centrality(self, graph: nx.DiGraph) -> CentralityAnalysisResult:
        """
        Compute out-degree centrality
        
        Out-degree centrality measures the number of outgoing edges.
        High out-degree = influential/producing node.
        """
        self.logger.debug("Computing out-degree centrality...")
        
        # Compute raw scores
        raw_scores = dict(graph.out_degree())
        
        return self._create_result(
            CentralityType.OUT_DEGREE,
            raw_scores,
            graph
        )
    
    def compute_betweenness_centrality(self, 
                                      graph: nx.DiGraph,
                                      k: Optional[int] = None,
                                      normalized: bool = True) -> CentralityAnalysisResult:
        """
        Compute betweenness centrality
        
        Betweenness centrality measures how often a node lies on the shortest path
        between other nodes. High betweenness = bottleneck/broker node.
        
        Args:
            graph: NetworkX graph
            k: Sample k nodes for approximation (None = exact)
            normalized: Whether to normalize scores
        """
        self.logger.debug("Computing betweenness centrality...")
        
        # Compute raw scores
        raw_scores = nx.betweenness_centrality(
            graph,
            k=k,
            normalized=normalized,
            weight='weight'
        )
        
        return self._create_result(
            CentralityType.BETWEENNESS,
            raw_scores,
            graph
        )
    
    def compute_closeness_centrality(self, 
                                    graph: nx.DiGraph,
                                    distance: Optional[str] = None) -> CentralityAnalysisResult:
        """
        Compute closeness centrality
        
        Closeness centrality measures the average distance from a node to all other nodes.
        High closeness = central/core node.
        
        Args:
            graph: NetworkX graph
            distance: Edge attribute to use as distance (None = hop count)
        """
        self.logger.debug("Computing closeness centrality...")
        
        # Compute raw scores
        raw_scores = nx.closeness_centrality(
            graph,
            distance=distance
        )
        
        return self._create_result(
            CentralityType.CLOSENESS,
            raw_scores,
            graph
        )
    
    def compute_eigenvector_centrality(self,
                                      graph: nx.DiGraph,
                                      max_iter: int = 1000,
                                      tol: float = 1e-6) -> CentralityAnalysisResult:
        """
        Compute eigenvector centrality
        
        Eigenvector centrality measures influence based on connections to other
        influential nodes. High eigenvector = well-connected to important nodes.
        
        Args:
            graph: NetworkX graph
            max_iter: Maximum iterations
            tol: Convergence tolerance
        """
        self.logger.debug("Computing eigenvector centrality...")
        
        # Compute raw scores
        raw_scores = nx.eigenvector_centrality(
            graph,
            max_iter=max_iter,
            tol=tol,
            weight='weight'
        )
        
        return self._create_result(
            CentralityType.EIGENVECTOR,
            raw_scores,
            graph
        )
    
    def compute_pagerank(self,
                        graph: nx.DiGraph,
                        alpha: float = 0.85,
                        max_iter: int = 100,
                        tol: float = 1e-6) -> CentralityAnalysisResult:
        """
        Compute PageRank
        
        PageRank measures importance based on the structure of incoming links.
        Originally developed for web page ranking.
        
        Args:
            graph: NetworkX graph
            alpha: Damping parameter (0.85 = 85% follow links, 15% random jump)
            max_iter: Maximum iterations
            tol: Convergence tolerance
        """
        self.logger.debug("Computing PageRank...")
        
        # Compute raw scores
        raw_scores = nx.pagerank(
            graph,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            weight='weight'
        )
        
        return self._create_result(
            CentralityType.PAGERANK,
            raw_scores,
            graph
        )
    
    def compute_katz_centrality(self,
                               graph: nx.DiGraph,
                               alpha: float = 0.1,
                               beta: float = 1.0,
                               max_iter: int = 1000,
                               tol: float = 1e-6) -> CentralityAnalysisResult:
        """
        Compute Katz centrality
        
        Katz centrality generalizes degree centrality by considering not just
        immediate neighbors but all nodes reachable via paths, with distant
        nodes weighted less.
        
        Args:
            graph: NetworkX graph
            alpha: Attenuation factor (< 1/λ_max)
            beta: Weight attributed to immediate neighbors
            max_iter: Maximum iterations
            tol: Convergence tolerance
        """
        self.logger.debug("Computing Katz centrality...")
        
        # Compute raw scores
        raw_scores = nx.katz_centrality(
            graph,
            alpha=alpha,
            beta=beta,
            max_iter=max_iter,
            tol=tol,
            weight='weight'
        )
        
        return self._create_result(
            CentralityType.KATZ,
            raw_scores,
            graph
        )
    
    def compute_harmonic_centrality(self,
                                   graph: nx.DiGraph,
                                   distance: Optional[str] = None) -> CentralityAnalysisResult:
        """
        Compute harmonic centrality
        
        Harmonic centrality is a variant of closeness centrality that works better
        for disconnected graphs. It sums the inverse distances to all other nodes.
        
        Args:
            graph: NetworkX graph
            distance: Edge attribute to use as distance
        """
        self.logger.debug("Computing harmonic centrality...")
        
        # Compute raw scores
        raw_scores = nx.harmonic_centrality(
            graph,
            distance=distance
        )
        
        return self._create_result(
            CentralityType.HARMONIC,
            raw_scores,
            graph
        )
    
    def compute_load_centrality(self,
                               graph: nx.DiGraph,
                               normalized: bool = True) -> CentralityAnalysisResult:
        """
        Compute load centrality
        
        Load centrality measures the fraction of shortest paths that pass through
        each node. Similar to betweenness but counts node occurrences rather than
        edge occurrences.
        
        Args:
            graph: NetworkX graph
            normalized: Whether to normalize scores
        """
        self.logger.debug("Computing load centrality...")
        
        # Compute raw scores
        raw_scores = nx.load_centrality(
            graph,
            normalized=normalized,
            weight='weight'
        )
        
        return self._create_result(
            CentralityType.LOAD,
            raw_scores,
            graph
        )
    
    def compute_current_flow_betweenness(self,
                                        graph: nx.DiGraph,
                                        normalized: bool = True) -> CentralityAnalysisResult:
        """
        Compute current flow betweenness centrality
        
        Current flow betweenness treats the graph as an electrical network and
        measures how much current flows through each node. Requires connected graph.
        
        Args:
            graph: NetworkX graph (must be connected)
            normalized: Whether to normalize scores
        """
        self.logger.debug("Computing current flow betweenness...")
        
        # Convert to undirected for current flow
        undirected = graph.to_undirected()
        
        # Compute raw scores
        raw_scores = nx.current_flow_betweenness_centrality(
            undirected,
            normalized=normalized,
            weight='weight'
        )
        
        return self._create_result(
            CentralityType.CURRENT_FLOW_BETWEENNESS,
            raw_scores,
            graph
        )
    
    def compute_current_flow_closeness(self,
                                      graph: nx.DiGraph) -> CentralityAnalysisResult:
        """
        Compute current flow closeness centrality
        
        Current flow closeness is a variant of closeness based on effective resistance
        in electrical networks. Requires connected graph.
        
        Args:
            graph: NetworkX graph (must be connected)
        """
        self.logger.debug("Computing current flow closeness...")
        
        # Convert to undirected for current flow
        undirected = graph.to_undirected()
        
        # Compute raw scores
        raw_scores = nx.current_flow_closeness_centrality(
            undirected,
            weight='weight'
        )
        
        return self._create_result(
            CentralityType.CURRENT_FLOW_CLOSENESS,
            raw_scores,
            graph
        )
    
    def compute_communicability_betweenness(self,
                                           graph: nx.DiGraph) -> CentralityAnalysisResult:
        """
        Compute communicability betweenness centrality
        
        Communicability betweenness measures the importance of a node in terms of
        the communicability between other pairs of nodes.
        
        Args:
            graph: NetworkX graph
        """
        self.logger.debug("Computing communicability betweenness...")
        
        # Convert to undirected
        undirected = graph.to_undirected()
        
        # Compute raw scores
        raw_scores = nx.communicability_betweenness_centrality(undirected)
        
        return self._create_result(
            CentralityType.COMMUNICABILITY_BETWEENNESS,
            raw_scores,
            graph
        )
    
    def compute_composite_centrality(self,
                                    graph: nx.DiGraph,
                                    weights: Optional[Dict[CentralityType, float]] = None) -> Dict[str, float]:
        """
        Compute a composite centrality score combining multiple metrics
        
        Args:
            graph: NetworkX graph
            weights: Weights for each centrality type (must sum to 1.0)
        
        Returns:
            Dictionary of composite scores
        """
        self.logger.info("Computing composite centrality...")
        
        # Default weights
        if weights is None:
            weights = {
                CentralityType.DEGREE: 0.2,
                CentralityType.BETWEENNESS: 0.3,
                CentralityType.CLOSENESS: 0.2,
                CentralityType.PAGERANK: 0.3
            }
        
        # Validate weights
        total = sum(weights.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        
        # Compute each centrality
        centralities = {}
        for centrality_type in weights.keys():
            if centrality_type == CentralityType.DEGREE:
                result = self.compute_degree_centrality(graph)
            elif centrality_type == CentralityType.BETWEENNESS:
                result = self.compute_betweenness_centrality(graph)
            elif centrality_type == CentralityType.CLOSENESS:
                result = self.compute_closeness_centrality(graph)
            elif centrality_type == CentralityType.PAGERANK:
                result = self.compute_pagerank(graph)
            elif centrality_type == CentralityType.EIGENVECTOR:
                result = self.compute_eigenvector_centrality(graph)
            else:
                continue
            
            centralities[centrality_type] = result
        
        # Combine scores
        composite = {}
        for node in graph.nodes():
            score = 0.0
            for centrality_type, weight in weights.items():
                if centrality_type in centralities:
                    node_score = centralities[centrality_type].scores[node].normalized_score
                    score += weight * node_score
            composite[node] = score
        
        return composite
    
    def identify_hubs(self,
                     graph: nx.DiGraph,
                     threshold: float = 0.8) -> List[str]:
        """
        Identify hub nodes (high degree centrality)
        
        Args:
            graph: NetworkX graph
            threshold: Percentile threshold (0-1)
        
        Returns:
            List of hub node names
        """
        result = self.compute_degree_centrality(graph)
        
        hubs = [
            node for node, score in result.scores.items()
            if score.percentile >= threshold * 100
        ]
        
        return hubs
    
    def identify_bottlenecks(self,
                            graph: nx.DiGraph,
                            threshold: float = 0.8) -> List[str]:
        """
        Identify bottleneck nodes (high betweenness centrality)
        
        Args:
            graph: NetworkX graph
            threshold: Percentile threshold (0-1)
        
        Returns:
            List of bottleneck node names
        """
        result = self.compute_betweenness_centrality(graph)
        
        bottlenecks = [
            node for node, score in result.scores.items()
            if score.percentile >= threshold * 100
        ]
        
        return bottlenecks
    
    def identify_core_nodes(self,
                           graph: nx.DiGraph,
                           threshold: float = 0.8) -> List[str]:
        """
        Identify core nodes (high closeness centrality)
        
        Args:
            graph: NetworkX graph
            threshold: Percentile threshold (0-1)
        
        Returns:
            List of core node names
        """
        result = self.compute_closeness_centrality(graph)
        
        core = [
            node for node, score in result.scores.items()
            if score.percentile >= threshold * 100
        ]
        
        return core
    
    def identify_influential_nodes(self,
                                  graph: nx.DiGraph,
                                  threshold: float = 0.8) -> List[str]:
        """
        Identify influential nodes (high PageRank)
        
        Args:
            graph: NetworkX graph
            threshold: Percentile threshold (0-1)
        
        Returns:
            List of influential node names
        """
        result = self.compute_pagerank(graph)
        
        influential = [
            node for node, score in result.scores.items()
            if score.percentile >= threshold * 100
        ]
        
        return influential
    
    def compare_centralities(self,
                            graph: nx.DiGraph,
                            node: str,
                            centrality_types: Optional[List[CentralityType]] = None) -> Dict[CentralityType, float]:
        """
        Compare different centrality scores for a specific node
        
        Args:
            graph: NetworkX graph
            node: Node name
            centrality_types: List of centrality types to compute
        
        Returns:
            Dictionary mapping centrality type to normalized score
        """
        if centrality_types is None:
            centrality_types = [
                CentralityType.DEGREE,
                CentralityType.BETWEENNESS,
                CentralityType.CLOSENESS,
                CentralityType.PAGERANK
            ]
        
        comparison = {}
        
        for centrality_type in centrality_types:
            if centrality_type == CentralityType.DEGREE:
                result = self.compute_degree_centrality(graph)
            elif centrality_type == CentralityType.BETWEENNESS:
                result = self.compute_betweenness_centrality(graph)
            elif centrality_type == CentralityType.CLOSENESS:
                result = self.compute_closeness_centrality(graph)
            elif centrality_type == CentralityType.PAGERANK:
                result = self.compute_pagerank(graph)
            elif centrality_type == CentralityType.EIGENVECTOR:
                try:
                    result = self.compute_eigenvector_centrality(graph)
                except:
                    continue
            else:
                continue
            
            if node in result.scores:
                comparison[centrality_type] = result.scores[node].normalized_score
        
        return comparison
    
    def get_top_nodes(self,
                     graph: nx.DiGraph,
                     centrality_type: CentralityType,
                     n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N nodes by centrality score
        
        Args:
            graph: NetworkX graph
            centrality_type: Type of centrality to compute
            n: Number of top nodes to return
        
        Returns:
            List of (node, score) tuples
        """
        # Compute centrality
        if centrality_type == CentralityType.DEGREE:
            result = self.compute_degree_centrality(graph)
        elif centrality_type == CentralityType.BETWEENNESS:
            result = self.compute_betweenness_centrality(graph)
        elif centrality_type == CentralityType.CLOSENESS:
            result = self.compute_closeness_centrality(graph)
        elif centrality_type == CentralityType.PAGERANK:
            result = self.compute_pagerank(graph)
        elif centrality_type == CentralityType.EIGENVECTOR:
            result = self.compute_eigenvector_centrality(graph)
        else:
            raise ValueError(f"Unsupported centrality type: {centrality_type}")
        
        # Get top nodes
        sorted_nodes = sorted(
            result.scores.items(),
            key=lambda x: x[1].score,
            reverse=True
        )
        
        return [(node, score.score) for node, score in sorted_nodes[:n]]
    
    def _create_result(self,
                      centrality_type: CentralityType,
                      raw_scores: Dict[str, float],
                      graph: nx.DiGraph) -> CentralityAnalysisResult:
        """
        Create a CentralityAnalysisResult from raw scores
        
        Args:
            centrality_type: Type of centrality
            raw_scores: Raw centrality scores
            graph: NetworkX graph
        
        Returns:
            CentralityAnalysisResult
        """
        # Sort by score
        sorted_items = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate statistics
        scores_list = list(raw_scores.values())
        statistics = {
            'mean': float(np.mean(scores_list)),
            'median': float(np.median(scores_list)),
            'std': float(np.std(scores_list)),
            'min': float(np.min(scores_list)),
            'max': float(np.max(scores_list)),
            'sum': float(np.sum(scores_list))
        }
        
        # Normalize scores
        max_score = statistics['max']
        min_score = statistics['min']
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        # Create CentralityScore objects
        scores = {}
        for rank, (node, raw_score) in enumerate(sorted_items, 1):
            normalized_score = (raw_score - min_score) / score_range if score_range > 0 else 0.0
            percentile = (1 - (rank - 1) / len(sorted_items)) * 100 if len(sorted_items) > 0 else 0.0
            
            scores[node] = CentralityScore(
                node=node,
                score=raw_score,
                rank=rank,
                normalized_score=normalized_score,
                percentile=percentile
            )
        
        # Get top nodes
        top_nodes = [node for node, _ in sorted_items[:10]]
        
        return CentralityAnalysisResult(
            centrality_type=centrality_type,
            scores=scores,
            top_nodes=top_nodes,
            statistics=statistics
        )
    
    def generate_centrality_report(self,
                                  graph: nx.DiGraph,
                                  include_all: bool = False) -> Dict:
        """
        Generate a comprehensive centrality analysis report
        
        Args:
            graph: NetworkX graph
            include_all: Whether to include all metrics (including expensive ones)
        
        Returns:
            Dictionary with complete analysis
        """
        self.logger.info("Generating centrality report...")
        
        results = self.analyze_all_centralities(graph, include_expensive=include_all)
        
        report = {
            'summary': {
                'total_nodes': len(graph),
                'total_edges': len(graph.edges()),
                'metrics_computed': len(results)
            },
            'centralities': {
                ctype.value: result.to_dict()
                for ctype, result in results.items()
            },
            'key_nodes': {
                'hubs': self.identify_hubs(graph, threshold=0.9),
                'bottlenecks': self.identify_bottlenecks(graph, threshold=0.9),
                'core': self.identify_core_nodes(graph, threshold=0.9),
                'influential': self.identify_influential_nodes(graph, threshold=0.9)
            }
        }
        
        # Add composite centrality if we have the required metrics
        try:
            composite = self.compute_composite_centrality(graph)
            top_composite = sorted(composite.items(), key=lambda x: x[1], reverse=True)[:10]
            report['composite_centrality'] = {
                'scores': composite,
                'top_nodes': [node for node, _ in top_composite]
            }
        except Exception as e:
            self.logger.warning(f"Could not compute composite centrality: {e}")
        
        return report
