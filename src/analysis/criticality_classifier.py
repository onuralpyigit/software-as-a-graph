#!/usr/bin/env python3
"""
Criticality Classifier using Box-Plot Method
=============================================

Integrates box-plot statistical classification with quality attribute analysis
to provide adaptive, data-driven criticality labeling for components and edges.

Box Plot Classification Method:
- CRITICAL:  score > Q3 + 1.5*IQR (upper outliers)
- HIGH:      Q3 < score <= Q3 + 1.5*IQR
- MEDIUM:    Median < score <= Q3
- LOW:       Q1 < score <= Median
- MINIMAL:   score <= Q1

Advantages over static thresholds:
1. Adapts to actual score distribution
2. Identifies statistical outliers automatically
3. Fair comparison across different system sizes
4. Robust to score scaling differences

Author: Software-as-a-Graph Research Project
"""

import logging
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import math

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX is required: pip install networkx")


# ============================================================================
# Enums
# ============================================================================

class CriticalityLevel(Enum):
    """Criticality levels based on box-plot classification"""
    CRITICAL = "critical"      # Upper outliers (> Q3 + 1.5*IQR)
    HIGH = "high"              # Upper quartile range
    MEDIUM = "medium"          # Above median
    LOW = "low"                # Below median, above Q1
    MINIMAL = "minimal"        # Lower range (<= Q1)
    
    def __lt__(self, other):
        order = [CriticalityLevel.CRITICAL, CriticalityLevel.HIGH, 
                 CriticalityLevel.MEDIUM, CriticalityLevel.LOW, CriticalityLevel.MINIMAL]
        return order.index(self) < order.index(other)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BoxPlotStatistics:
    """Box plot statistics for score distribution"""
    min_value: float
    max_value: float
    mean: float
    median: float  # Q2
    std_dev: float
    q1: float      # 25th percentile
    q3: float      # 75th percentile
    iqr: float     # Interquartile range
    lower_fence: float   # Q1 - 1.5 * IQR
    upper_fence: float   # Q3 + 1.5 * IQR
    count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'min': round(self.min_value, 4),
            'max': round(self.max_value, 4),
            'mean': round(self.mean, 4),
            'median': round(self.median, 4),
            'std_dev': round(self.std_dev, 4),
            'q1': round(self.q1, 4),
            'q3': round(self.q3, 4),
            'iqr': round(self.iqr, 4),
            'lower_fence': round(self.lower_fence, 4),
            'upper_fence': round(self.upper_fence, 4),
            'count': self.count,
            'thresholds': {
                'critical': f'> {self.upper_fence:.4f}',
                'high': f'({self.q3:.4f}, {self.upper_fence:.4f}]',
                'medium': f'({self.median:.4f}, {self.q3:.4f}]',
                'low': f'({self.q1:.4f}, {self.median:.4f}]',
                'minimal': f'<= {self.q1:.4f}'
            }
        }


@dataclass
class ClassifiedComponent:
    """Component with box-plot criticality classification"""
    component_id: str
    component_type: str
    raw_score: float
    level: CriticalityLevel
    percentile: float
    z_score: float
    is_outlier: bool
    quality_attributes: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'component_id': self.component_id,
            'component_type': self.component_type,
            'raw_score': round(self.raw_score, 4),
            'level': self.level.value,
            'percentile': round(self.percentile, 2),
            'z_score': round(self.z_score, 4),
            'is_outlier': self.is_outlier,
            'quality_attributes': self.quality_attributes,
            'reasons': self.reasons,
            'metrics': {k: round(v, 4) if isinstance(v, float) else v 
                       for k, v in self.metrics.items()}
        }


@dataclass
class ClassifiedEdge:
    """Edge with box-plot criticality classification"""
    source: str
    target: str
    edge_type: str
    raw_score: float
    level: CriticalityLevel
    percentile: float
    z_score: float
    is_outlier: bool
    quality_attributes: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'target': self.target,
            'edge_type': self.edge_type,
            'raw_score': round(self.raw_score, 4),
            'level': self.level.value,
            'percentile': round(self.percentile, 2),
            'z_score': round(self.z_score, 4),
            'is_outlier': self.is_outlier,
            'quality_attributes': self.quality_attributes,
            'reasons': self.reasons,
            'metrics': {k: round(v, 4) if isinstance(v, float) else v 
                       for k, v in self.metrics.items()}
        }


@dataclass
class ClassificationResult:
    """Complete classification result"""
    components: List[ClassifiedComponent]
    edges: List[ClassifiedEdge]
    component_stats: BoxPlotStatistics
    edge_stats: Optional[BoxPlotStatistics]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'components': [c.to_dict() for c in self.components],
            'edges': [e.to_dict() for e in self.edges],
            'component_statistics': self.component_stats.to_dict(),
            'edge_statistics': self.edge_stats.to_dict() if self.edge_stats else None,
            'summary': self.summary
        }
    
    def get_by_level(self, level: CriticalityLevel) -> Tuple[List[ClassifiedComponent], List[ClassifiedEdge]]:
        """Get components and edges at a specific level"""
        comps = [c for c in self.components if c.level == level]
        edges = [e for e in self.edges if e.level == level]
        return comps, edges
    
    def get_critical(self) -> Tuple[List[ClassifiedComponent], List[ClassifiedEdge]]:
        """Get critical components and edges"""
        return self.get_by_level(CriticalityLevel.CRITICAL)
    
    def get_outliers(self) -> Tuple[List[ClassifiedComponent], List[ClassifiedEdge]]:
        """Get all outliers"""
        comps = [c for c in self.components if c.is_outlier]
        edges = [e for e in self.edges if e.is_outlier]
        return comps, edges


# ============================================================================
# Box-Plot Criticality Classifier
# ============================================================================

class BoxPlotCriticalityClassifier:
    """
    Classifies components and edges using box-plot statistical method.
    
    This classifier computes criticality scores from multiple metrics and
    then applies box-plot classification to adaptively label components
    based on the actual distribution of scores.
    """
    
    def __init__(self, 
                 iqr_multiplier: float = 1.5,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize classifier.
        
        Args:
            iqr_multiplier: Multiplier for IQR to calculate fences (1.5 = standard, 3.0 = extreme only)
            config: Additional configuration options
        """
        self.iqr_multiplier = iqr_multiplier
        self.config = config or {}
        self.logger = logging.getLogger('BoxPlotCriticalityClassifier')
        
        # Score weights for composite criticality
        self.weights = self.config.get('weights', {
            'betweenness': 0.25,
            'articulation_point': 0.30,
            'impact_score': 0.25,
            'degree_centrality': 0.10,
            'pagerank': 0.10
        })
    
    def classify_graph(self, graph: nx.DiGraph) -> ClassificationResult:
        """
        Classify all components and edges in a graph.
        
        Args:
            graph: NetworkX DiGraph with system components
            
        Returns:
            ClassificationResult with classified components and edges
        """
        self.logger.info("Starting box-plot criticality classification...")
        
        # Compute component scores
        component_scores = self._compute_component_scores(graph)
        
        # Compute edge scores
        edge_scores = self._compute_edge_scores(graph)
        
        # Compute statistics
        comp_values = list(component_scores.values())
        edge_values = [s['score'] for s in edge_scores.values()]
        
        comp_stats = self._compute_statistics(comp_values) if comp_values else None
        edge_stats = self._compute_statistics(edge_values) if edge_values else None
        
        # Classify components
        classified_components = []
        if comp_stats:
            classified_components = self._classify_components(
                graph, component_scores, comp_stats
            )
        
        # Classify edges
        classified_edges = []
        if edge_stats:
            classified_edges = self._classify_edges(
                graph, edge_scores, edge_stats
            )
        
        # Generate summary
        summary = self._generate_summary(classified_components, classified_edges)
        
        self.logger.info(f"Classification complete: {len(classified_components)} components, "
                        f"{len(classified_edges)} edges")
        
        return ClassificationResult(
            components=classified_components,
            edges=classified_edges,
            component_stats=comp_stats,
            edge_stats=edge_stats,
            summary=summary
        )
    
    def _compute_statistics(self, values: List[float]) -> Optional[BoxPlotStatistics]:
        """Compute box-plot statistics for a list of values"""
        if not values:
            return None
        
        n = len(values)
        sorted_values = sorted(values)
        
        # Basic statistics
        mean_val = sum(values) / n
        variance = sum((x - mean_val) ** 2 for x in values) / n
        std_dev = math.sqrt(variance)
        
        # Quartiles
        def percentile(data, p):
            k = (len(data) - 1) * p / 100
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return data[int(k)]
            return data[int(f)] * (c - k) + data[int(c)] * (k - f)
        
        q1 = percentile(sorted_values, 25)
        median = percentile(sorted_values, 50)
        q3 = percentile(sorted_values, 75)
        iqr = q3 - q1
        
        # Fences
        lower_fence = q1 - self.iqr_multiplier * iqr
        upper_fence = q3 + self.iqr_multiplier * iqr
        
        # Clamp to reasonable bounds
        lower_fence = max(0.0, lower_fence)
        upper_fence = min(1.0, upper_fence) if max(values) <= 1.0 else upper_fence
        
        return BoxPlotStatistics(
            min_value=min(values),
            max_value=max(values),
            mean=mean_val,
            median=median,
            std_dev=std_dev,
            q1=q1,
            q3=q3,
            iqr=iqr,
            lower_fence=lower_fence,
            upper_fence=upper_fence,
            count=n
        )
    
    def _classify_value(self, value: float, stats: BoxPlotStatistics) -> CriticalityLevel:
        """Classify a single value using box-plot thresholds"""
        if value > stats.upper_fence:
            return CriticalityLevel.CRITICAL
        elif value > stats.q3:
            return CriticalityLevel.HIGH
        elif value > stats.median:
            return CriticalityLevel.MEDIUM
        elif value > stats.q1:
            return CriticalityLevel.LOW
        else:
            return CriticalityLevel.MINIMAL
    
    def _compute_percentile(self, value: float, sorted_values: List[float]) -> float:
        """Compute percentile rank of a value"""
        if not sorted_values:
            return 0.0
        count_below = sum(1 for v in sorted_values if v < value)
        count_equal = sum(1 for v in sorted_values if v == value)
        return 100.0 * (count_below + 0.5 * count_equal) / len(sorted_values)
    
    def _compute_component_scores(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Compute composite criticality scores for all components"""
        scores = {}
        
        if graph.number_of_nodes() == 0:
            return scores
        
        # Compute centrality metrics
        try:
            betweenness = nx.betweenness_centrality(graph)
        except:
            betweenness = {n: 0.0 for n in graph.nodes()}
        
        try:
            pagerank = nx.pagerank(graph)
        except:
            pagerank = {n: 1.0/graph.number_of_nodes() for n in graph.nodes()}
        
        # Degree centrality
        in_degree = dict(graph.in_degree())
        out_degree = dict(graph.out_degree())
        max_degree = max(max(in_degree.values(), default=1), max(out_degree.values(), default=1))
        degree_centrality = {n: (in_degree[n] + out_degree[n]) / (2 * max_degree) 
                           for n in graph.nodes()}
        
        # Articulation points
        try:
            undirected = graph.to_undirected()
            aps = set(nx.articulation_points(undirected))
        except:
            aps = set()
        
        # Impact scores (reachability-based)
        impact_scores = self._compute_impact_scores(graph)
        
        # Normalize and compute composite scores
        max_bc = max(betweenness.values()) if betweenness else 1.0
        max_pr = max(pagerank.values()) if pagerank else 1.0
        max_impact = max(impact_scores.values()) if impact_scores else 1.0
        
        for node in graph.nodes():
            bc_norm = betweenness.get(node, 0) / max_bc if max_bc > 0 else 0
            pr_norm = pagerank.get(node, 0) / max_pr if max_pr > 0 else 0
            dc_norm = degree_centrality.get(node, 0)
            ap_score = 1.0 if node in aps else 0.0
            impact_norm = impact_scores.get(node, 0) / max_impact if max_impact > 0 else 0
            
            # Composite score using weights
            score = (
                self.weights.get('betweenness', 0.25) * bc_norm +
                self.weights.get('articulation_point', 0.30) * ap_score +
                self.weights.get('impact_score', 0.25) * impact_norm +
                self.weights.get('degree_centrality', 0.10) * dc_norm +
                self.weights.get('pagerank', 0.10) * pr_norm
            )
            
            scores[node] = score
        
        return scores
    
    def _compute_impact_scores(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Compute impact scores based on reachability"""
        impact = {}
        total_nodes = graph.number_of_nodes()
        
        for node in graph.nodes():
            # Count descendants (components affected by this node's failure)
            try:
                descendants = len(nx.descendants(graph, node))
            except:
                descendants = 0
            
            # Normalize
            impact[node] = descendants / (total_nodes - 1) if total_nodes > 1 else 0
        
        return impact
    
    def _compute_edge_scores(self, graph: nx.DiGraph) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Compute criticality scores for all edges"""
        scores = {}
        
        if graph.number_of_edges() == 0:
            return scores
        
        # Edge betweenness
        try:
            edge_bc = nx.edge_betweenness_centrality(graph)
        except:
            edge_bc = {e: 0.0 for e in graph.edges()}
        
        # Find bridges
        try:
            undirected = graph.to_undirected()
            bridges = set(nx.bridges(undirected))
        except:
            bridges = set()
        
        max_ebc = max(edge_bc.values()) if edge_bc else 1.0
        
        for u, v in graph.edges():
            ebc_norm = edge_bc.get((u, v), 0) / max_ebc if max_ebc > 0 else 0
            is_bridge = (u, v) in bridges or (v, u) in bridges
            
            # Composite edge score
            score = 0.6 * ebc_norm + 0.4 * (1.0 if is_bridge else 0.0)
            
            edge_data = graph.get_edge_data(u, v, {})
            
            scores[(u, v)] = {
                'score': score,
                'edge_betweenness': ebc_norm,
                'is_bridge': is_bridge,
                'edge_type': edge_data.get('dependency_type', 'unknown')
            }
        
        return scores
    
    def _classify_components(self, graph: nx.DiGraph, 
                            scores: Dict[str, float],
                            stats: BoxPlotStatistics) -> List[ClassifiedComponent]:
        """Classify all components"""
        classified = []
        sorted_scores = sorted(scores.values())
        
        # Get articulation points for reasons
        try:
            undirected = graph.to_undirected()
            aps = set(nx.articulation_points(undirected))
        except:
            aps = set()
        
        # Get betweenness for high-bc detection
        try:
            betweenness = nx.betweenness_centrality(graph)
            bc_threshold = stats.q3 if stats else 0.1
        except:
            betweenness = {}
            bc_threshold = 0.1
        
        for node, score in scores.items():
            level = self._classify_value(score, stats)
            percentile = self._compute_percentile(score, sorted_scores)
            z_score = (score - stats.mean) / stats.std_dev if stats.std_dev > 0 else 0
            is_outlier = score > stats.upper_fence or score < stats.lower_fence
            
            # Determine component type
            node_data = graph.nodes.get(node, {})
            comp_type = node_data.get('type', 'Unknown')
            
            # Build reasons
            reasons = []
            if node in aps:
                reasons.append('articulation_point')
            if betweenness.get(node, 0) > bc_threshold:
                reasons.append('high_betweenness')
            if graph.in_degree(node) > stats.q3 * graph.number_of_nodes():
                reasons.append('high_in_degree')
            if graph.out_degree(node) > stats.q3 * graph.number_of_nodes():
                reasons.append('high_out_degree')
            if is_outlier and score > stats.upper_fence:
                reasons.append('statistical_outlier')
            
            # Quality attributes affected
            qa = []
            if node in aps or level in [CriticalityLevel.CRITICAL, CriticalityLevel.HIGH]:
                qa.append('reliability')
            if betweenness.get(node, 0) > bc_threshold:
                qa.append('availability')
            if graph.degree(node) > 5:
                qa.append('maintainability')
            
            classified.append(ClassifiedComponent(
                component_id=node,
                component_type=comp_type,
                raw_score=score,
                level=level,
                percentile=percentile,
                z_score=z_score,
                is_outlier=is_outlier,
                quality_attributes=qa,
                reasons=reasons,
                metrics={
                    'betweenness': betweenness.get(node, 0),
                    'in_degree': graph.in_degree(node),
                    'out_degree': graph.out_degree(node),
                    'is_articulation_point': node in aps
                }
            ))
        
        # Sort by score descending
        classified.sort(key=lambda x: -x.raw_score)
        return classified
    
    def _classify_edges(self, graph: nx.DiGraph,
                       scores: Dict[Tuple[str, str], Dict[str, Any]],
                       stats: BoxPlotStatistics) -> List[ClassifiedEdge]:
        """Classify all edges"""
        classified = []
        sorted_scores = sorted([s['score'] for s in scores.values()])
        
        for (u, v), data in scores.items():
            score = data['score']
            level = self._classify_value(score, stats)
            percentile = self._compute_percentile(score, sorted_scores)
            z_score = (score - stats.mean) / stats.std_dev if stats.std_dev > 0 else 0
            is_outlier = score > stats.upper_fence or score < stats.lower_fence
            
            # Build reasons
            reasons = []
            if data.get('is_bridge'):
                reasons.append('bridge_edge')
            if data.get('edge_betweenness', 0) > stats.median:
                reasons.append('high_edge_betweenness')
            if is_outlier and score > stats.upper_fence:
                reasons.append('statistical_outlier')
            
            # Quality attributes
            qa = []
            if data.get('is_bridge'):
                qa.extend(['reliability', 'availability'])
            if level in [CriticalityLevel.CRITICAL, CriticalityLevel.HIGH]:
                qa.append('reliability')
            
            classified.append(ClassifiedEdge(
                source=u,
                target=v,
                edge_type=data.get('edge_type', 'unknown'),
                raw_score=score,
                level=level,
                percentile=percentile,
                z_score=z_score,
                is_outlier=is_outlier,
                quality_attributes=list(set(qa)),
                reasons=reasons,
                metrics={
                    'edge_betweenness': data.get('edge_betweenness', 0),
                    'is_bridge': data.get('is_bridge', False)
                }
            ))
        
        # Sort by score descending
        classified.sort(key=lambda x: -x.raw_score)
        return classified
    
    def _generate_summary(self, components: List[ClassifiedComponent],
                         edges: List[ClassifiedEdge]) -> Dict[str, Any]:
        """Generate classification summary"""
        comp_by_level = defaultdict(int)
        edge_by_level = defaultdict(int)
        
        for c in components:
            comp_by_level[c.level.value] += 1
        for e in edges:
            edge_by_level[e.level.value] += 1
        
        comp_outliers = sum(1 for c in components if c.is_outlier)
        edge_outliers = sum(1 for e in edges if e.is_outlier)
        
        return {
            'total_components': len(components),
            'total_edges': len(edges),
            'components_by_level': dict(comp_by_level),
            'edges_by_level': dict(edge_by_level),
            'component_outliers': comp_outliers,
            'edge_outliers': edge_outliers,
            'critical_components': [c.component_id for c in components 
                                   if c.level == CriticalityLevel.CRITICAL][:10],
            'critical_edges': [(e.source, e.target) for e in edges 
                              if e.level == CriticalityLevel.CRITICAL][:10]
        }
    
    def generate_report(self, result: ClassificationResult) -> str:
        """Generate human-readable classification report"""
        lines = [
            "=" * 70,
            "BOX-PLOT CRITICALITY CLASSIFICATION REPORT",
            "=" * 70,
            "",
            "COMPONENT STATISTICS",
            "-" * 40,
        ]
        
        if result.component_stats:
            stats = result.component_stats
            lines.extend([
                f"  Total Components:     {stats.count}",
                f"  Score Range:          [{stats.min_value:.4f}, {stats.max_value:.4f}]",
                f"  Mean Score:           {stats.mean:.4f}",
                f"  Median Score:         {stats.median:.4f}",
                f"  Standard Deviation:   {stats.std_dev:.4f}",
                "",
                "CLASSIFICATION THRESHOLDS (Box-Plot Method)",
                "-" * 40,
                f"  Q1 (25th percentile): {stats.q1:.4f}",
                f"  Q2 (Median):          {stats.median:.4f}",
                f"  Q3 (75th percentile): {stats.q3:.4f}",
                f"  IQR:                  {stats.iqr:.4f}",
                f"  Upper Fence:          {stats.upper_fence:.4f}",
                f"  Lower Fence:          {stats.lower_fence:.4f}",
                "",
                "CLASSIFICATION LEVELS",
                "-" * 40,
                f"  CRITICAL:  score > {stats.upper_fence:.4f} (outliers)",
                f"  HIGH:      {stats.q3:.4f} < score <= {stats.upper_fence:.4f}",
                f"  MEDIUM:    {stats.median:.4f} < score <= {stats.q3:.4f}",
                f"  LOW:       {stats.q1:.4f} < score <= {stats.median:.4f}",
                f"  MINIMAL:   score <= {stats.q1:.4f}",
            ])
        
        lines.extend([
            "",
            "DISTRIBUTION BY LEVEL",
            "-" * 40,
        ])
        
        summary = result.summary
        for level in CriticalityLevel:
            count = summary['components_by_level'].get(level.value, 0)
            pct = 100 * count / summary['total_components'] if summary['total_components'] > 0 else 0
            bar_len = int(30 * count / max(summary['components_by_level'].values(), default=1))
            bar = "█" * bar_len + "░" * (30 - bar_len)
            lines.append(f"  {level.value:10s}: {count:4d} ({pct:5.1f}%) {bar}")
        
        lines.extend([
            "",
            f"OUTLIERS: {summary['component_outliers']} components, "
            f"{summary['edge_outliers']} edges",
            "",
            "TOP 10 CRITICAL COMPONENTS",
            "-" * 40,
        ])
        
        for i, comp in enumerate(result.components[:10], 1):
            lines.append(
                f"  {i:2d}. {comp.component_id:30s} "
                f"Score: {comp.raw_score:.4f} "
                f"[{comp.level.value:8s}] "
                f"P{comp.percentile:.0f}"
            )
        
        lines.extend([
            "",
            "TOP 10 CRITICAL EDGES",
            "-" * 40,
        ])
        
        for i, edge in enumerate(result.edges[:10], 1):
            lines.append(
                f"  {i:2d}. {edge.source} → {edge.target} "
                f"Score: {edge.raw_score:.4f} "
                f"[{edge.level.value:8s}]"
            )
        
        lines.extend(["", "=" * 70])
        
        return "\n".join(lines)


# ============================================================================
# Integration with Quality Analyzers
# ============================================================================

def classify_quality_results(reliability_result, maintainability_result, 
                            availability_result, graph: nx.DiGraph,
                            iqr_multiplier: float = 1.5) -> ClassificationResult:
    """
    Classify components and edges from quality analyzer results using box-plot method.
    
    Args:
        reliability_result: Result from ReliabilityAnalyzer
        maintainability_result: Result from MaintainabilityAnalyzer
        availability_result: Result from AvailabilityAnalyzer
        graph: The analyzed graph
        iqr_multiplier: IQR multiplier for fence calculation
        
    Returns:
        ClassificationResult with box-plot classifications
    """
    classifier = BoxPlotCriticalityClassifier(iqr_multiplier=iqr_multiplier)
    return classifier.classify_graph(graph)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Test with sample graph
    G = nx.DiGraph()
    
    # Add nodes with types
    nodes = [
        ('broker1', {'type': 'Broker'}),
        ('broker2', {'type': 'Broker'}),
        ('topic1', {'type': 'Topic'}),
        ('topic2', {'type': 'Topic'}),
        ('topic3', {'type': 'Topic'}),
        ('app1', {'type': 'Application'}),
        ('app2', {'type': 'Application'}),
        ('app3', {'type': 'Application'}),
        ('app4', {'type': 'Application'}),
        ('app5', {'type': 'Application'}),
        ('node1', {'type': 'Node'}),
        ('node2', {'type': 'Node'})
    ]
    G.add_nodes_from(nodes)
    
    # Add edges
    edges = [
        ('app1', 'topic1'), ('app2', 'topic1'), ('app3', 'topic1'),
        ('topic1', 'app4'), ('topic1', 'app5'),
        ('app4', 'topic2'), ('topic2', 'app5'),
        ('topic1', 'broker1'), ('topic2', 'broker1'), ('topic3', 'broker2'),
        ('broker1', 'node1'), ('broker2', 'node2'),
        ('app1', 'broker1'), ('app2', 'broker1'), ('app3', 'broker2')
    ]
    G.add_edges_from(edges)
    
    # Classify
    classifier = BoxPlotCriticalityClassifier()
    result = classifier.classify_graph(G)
    
    # Print report
    print(classifier.generate_report(result))
    
    # Show outliers
    print("\n" + "=" * 70)
    print("STATISTICAL OUTLIERS")
    print("=" * 70)
    
    outlier_comps, outlier_edges = result.get_outliers()
    if outlier_comps:
        print("\nComponent Outliers:")
        for c in outlier_comps:
            print(f"  - {c.component_id}: {c.raw_score:.4f} (z={c.z_score:.2f})")
    else:
        print("\nNo component outliers detected")
    
    if outlier_edges:
        print("\nEdge Outliers:")
        for e in outlier_edges:
            print(f"  - {e.source} → {e.target}: {e.raw_score:.4f}")
    else:
        print("\nNo edge outliers detected")
