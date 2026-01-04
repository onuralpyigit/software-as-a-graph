"""
Structural Analyzer

Structural graph analysis using NetworkX library.

This module implements all graph algorithms:
- PageRank
- Betweenness Centrality
- Degree Centrality
- Articulation Points
- Bridges

All algorithms operate on NetworkX graphs built from Neo4j data.

Author: Software-as-a-Graph Research Project
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple

import networkx as nx

from src.core.graph_exporter import GraphData, ComponentData, EdgeData, COMPONENT_TYPES, LAYER_DEFINITIONS
from .classifier import (
    BoxPlotClassifier,
    ClassificationResult,
    CriticalityLevel,
    ClassifiedItem,
)

@dataclass
class CentralityMetrics:
    """Centrality metrics for a single component."""
    
    component_id: str
    component_type: str
    
    # Raw centrality scores
    pagerank: float = 0.0
    betweenness: float = 0.0
    degree: float = 0.0
    in_degree: float = 0.0
    out_degree: float = 0.0
    
    # Normalized scores (0-1 within group)
    pagerank_norm: float = 0.0
    betweenness_norm: float = 0.0
    degree_norm: float = 0.0
    
    # Composite score
    composite_score: float = 0.0
    
    # Classification
    level: CriticalityLevel = CriticalityLevel.MEDIUM
    is_outlier: bool = False
    
    # Structural flags
    is_articulation_point: bool = False
    
    # Original weight from database
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.component_id,
            "type": self.component_type,
            "pagerank": round(self.pagerank, 6),
            "betweenness": round(self.betweenness, 6),
            "degree": round(self.degree, 6),
            "in_degree": round(self.in_degree, 6),
            "out_degree": round(self.out_degree, 6),
            "pagerank_norm": round(self.pagerank_norm, 4),
            "betweenness_norm": round(self.betweenness_norm, 4),
            "degree_norm": round(self.degree_norm, 4),
            "composite_score": round(self.composite_score, 4),
            "level": self.level.value,
            "is_outlier": self.is_outlier,
            "is_articulation_point": self.is_articulation_point,
            "weight": round(self.weight, 4),
        }


@dataclass
class EdgeMetrics:
    """Criticality metrics for a single edge."""
    
    source_id: str
    target_id: str
    source_type: str
    target_type: str
    dependency_type: str
    
    # Edge properties
    weight: float = 1.0
    
    # Criticality
    criticality_score: float = 0.0
    level: CriticalityLevel = CriticalityLevel.MEDIUM
    is_outlier: bool = False
    
    # Structural flags
    is_bridge: bool = False
    connects_articulation_point: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "dependency_type": self.dependency_type,
            "weight": round(self.weight, 4),
            "criticality_score": round(self.criticality_score, 4),
            "level": self.level.value,
            "is_outlier": self.is_outlier,
            "is_bridge": self.is_bridge,
            "connects_articulation_point": self.connects_articulation_point,
        }


@dataclass
class ComponentAnalysisResult:
    """Analysis result for a single component type."""
    
    component_type: str
    timestamp: str
    components: List[CentralityMetrics] = field(default_factory=list)
    classification: Optional[ClassificationResult] = None
    articulation_points: Set[str] = field(default_factory=set)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_type": self.component_type,
            "timestamp": self.timestamp,
            "count": len(self.components),
            "components": [c.to_dict() for c in self.components],
            "classification": self.classification.to_dict() if self.classification else None,
            "articulation_points": list(self.articulation_points),
            "summary": self.summary,
        }
    
    def get_critical(self) -> List[CentralityMetrics]:
        """Get components classified as CRITICAL."""
        return [c for c in self.components if c.level == CriticalityLevel.CRITICAL]
    
    def get_high_and_above(self) -> List[CentralityMetrics]:
        """Get components classified as HIGH or CRITICAL."""
        return [c for c in self.components if c.level >= CriticalityLevel.HIGH]
    
    def top_n(self, n: int = 10) -> List[CentralityMetrics]:
        """Get top N components by composite score."""
        return sorted(self.components, key=lambda x: x.composite_score, reverse=True)[:n]


@dataclass
class LayerAnalysisResult:
    """Analysis result for a dependency layer."""
    
    layer_key: str
    layer_name: str
    timestamp: str
    components: List[CentralityMetrics] = field(default_factory=list)
    classification: Optional[ClassificationResult] = None
    articulation_points: Set[str] = field(default_factory=set)
    bridges: List[Tuple[str, str]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_key": self.layer_key,
            "layer_name": self.layer_name,
            "timestamp": self.timestamp,
            "count": len(self.components),
            "components": [c.to_dict() for c in self.components],
            "classification": self.classification.to_dict() if self.classification else None,
            "articulation_points": list(self.articulation_points),
            "bridges": [{"source": b[0], "target": b[1]} for b in self.bridges],
            "summary": self.summary,
        }
    
    def get_critical(self) -> List[CentralityMetrics]:
        return [c for c in self.components if c.level == CriticalityLevel.CRITICAL]
    
    def get_high_and_above(self) -> List[CentralityMetrics]:
        return [c for c in self.components if c.level >= CriticalityLevel.HIGH]


@dataclass
class EdgeAnalysisResult:
    """Analysis result for edges."""
    
    timestamp: str
    edges: List[EdgeMetrics] = field(default_factory=list)
    classification: Optional[ClassificationResult] = None
    bridges: List[Tuple[str, str]] = field(default_factory=set)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "count": len(self.edges),
            "edges": [e.to_dict() for e in self.edges],
            "classification": self.classification.to_dict() if self.classification else None,
            "bridges": [{"source": b[0], "target": b[1]} for b in self.bridges],
            "summary": self.summary,
        }
    
    def get_critical(self) -> List[EdgeMetrics]:
        return [e for e in self.edges if e.level == CriticalityLevel.CRITICAL]
    
    def get_bridges(self) -> List[EdgeMetrics]:
        return [e for e in self.edges if e.is_bridge]


class StructuralAnalyzer:
    """
    Graph analyzer using NetworkX algorithms.
    
    Computes centrality metrics using NetworkX and classifies
    components using box-plot statistical method.
    
    Example:
        analyzer = StructuralAnalyzer(k_factor=1.5)
        
        # Analyze by component type
        result = analyzer.analyze_component_type(graph_data, "Application")
        
        # Analyze by layer
        result = analyzer.analyze_layer(graph_data, "application")
    """
    
    # Default weights for composite score
    DEFAULT_WEIGHTS = {
        "pagerank": 0.35,
        "betweenness": 0.40,
        "degree": 0.25,
    }
    
    def __init__(
        self,
        k_factor: float = 1.5,
        weights: Optional[Dict[str, float]] = None,
        damping_factor: float = 0.85,
    ):
        """
        Initialize analyzer.
        
        Args:
            k_factor: Box-plot k factor for classification
            weights: Weights for composite score calculation
            damping_factor: PageRank damping factor
        """
        self.k_factor = k_factor
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.damping_factor = damping_factor
        self.classifier = BoxPlotClassifier(k_factor=k_factor)
        self.logger = logging.getLogger(__name__)
    
    def analyze_component_type(
        self,
        graph_data: GraphData,
        component_type: str,
        weighted: bool = True,
    ) -> ComponentAnalysisResult:
        """
        Analyze all components of a specific type.
        
        Components are analyzed only against others of the same type
        for fair comparison using box-plot classification.
        
        Args:
            graph_data: Graph data from Neo4j
            component_type: Type to analyze (Application, Broker, Node, Topic)
            weighted: Use edge weights in algorithms
        
        Returns:
            ComponentAnalysisResult with metrics and classification
        """
        if component_type not in COMPONENT_TYPES:
            raise ValueError(f"Invalid type: {component_type}. Valid: {COMPONENT_TYPES}")
        
        timestamp = datetime.now().isoformat()
        self.logger.info(f"Analyzing component type: {component_type}")
        
        # Build NetworkX graph
        G = self._build_graph(graph_data, weighted=weighted)
        
        if G.number_of_nodes() == 0:
            return self._empty_component_result(component_type, timestamp)
        
        # Filter to only components of this type
        type_nodes = [
            c.id for c in graph_data.components 
            if c.component_type == component_type
        ]
        
        if not type_nodes:
            return self._empty_component_result(component_type, timestamp)
        
        # Get subgraph for this type only
        subgraph = G.subgraph(type_nodes).copy()
        
        # Compute centrality metrics
        pagerank = self._compute_pagerank(subgraph, weighted)
        betweenness = self._compute_betweenness(subgraph, weighted)
        degree = self._compute_degree_centrality(subgraph)
        in_degree, out_degree = self._compute_in_out_degree(subgraph)
        
        # Find articulation points
        articulation_points = self._find_articulation_points(subgraph)
        
        # Get component weights
        component_weights = {c.id: c.weight for c in graph_data.components}
        
        # Normalize metrics within this type
        pagerank_norm = self._normalize(pagerank)
        betweenness_norm = self._normalize(betweenness)
        degree_norm = self._normalize(degree)
        
        # Build metrics list
        metrics_list = []
        for node_id in type_nodes:
            composite = self._compute_composite(
                pagerank_norm.get(node_id, 0),
                betweenness_norm.get(node_id, 0),
                degree_norm.get(node_id, 0),
            )
            
            metrics_list.append(CentralityMetrics(
                component_id=node_id,
                component_type=component_type,
                pagerank=pagerank.get(node_id, 0),
                betweenness=betweenness.get(node_id, 0),
                degree=degree.get(node_id, 0),
                in_degree=in_degree.get(node_id, 0),
                out_degree=out_degree.get(node_id, 0),
                pagerank_norm=pagerank_norm.get(node_id, 0),
                betweenness_norm=betweenness_norm.get(node_id, 0),
                degree_norm=degree_norm.get(node_id, 0),
                composite_score=composite,
                is_articulation_point=node_id in articulation_points,
                weight=component_weights.get(node_id, 1.0),
            ))
        
        # Classify using box-plot
        items = [{"id": m.component_id, "type": m.component_type, "score": m.composite_score}
                 for m in metrics_list]
        classification = self.classifier.classify(items, metric_name=f"{component_type}_composite")
        
        # Update metrics with classification
        level_map = {item.id: (item.level, item.is_outlier) for item in classification.items}
        for metrics in metrics_list:
            level, is_outlier = level_map.get(metrics.component_id, (CriticalityLevel.MEDIUM, False))
            metrics.level = level
            metrics.is_outlier = is_outlier
        
        # Sort by composite score
        metrics_list.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Build summary
        summary = {
            "total": len(metrics_list),
            "by_level": classification.summary(),
            "articulation_points": len(articulation_points),
            "critical_count": len([m for m in metrics_list if m.level == CriticalityLevel.CRITICAL]),
        }
        
        return ComponentAnalysisResult(
            component_type=component_type,
            timestamp=timestamp,
            components=metrics_list,
            classification=classification,
            articulation_points=articulation_points,
            summary=summary,
        )
    
    def analyze_layer(
        self,
        graph_data: GraphData,
        layer_key: str,
        weighted: bool = True,
    ) -> LayerAnalysisResult:
        """
        Analyze a specific dependency layer.
        
        Args:
            graph_data: Graph data from Neo4j
            layer_key: Layer to analyze (application, infrastructure, etc.)
            weighted: Use edge weights
        
        Returns:
            LayerAnalysisResult with metrics and classification
        """
        if layer_key not in LAYER_DEFINITIONS:
            raise ValueError(f"Invalid layer: {layer_key}. Valid: {list(LAYER_DEFINITIONS.keys())}")
        
        layer_def = LAYER_DEFINITIONS[layer_key]
        timestamp = datetime.now().isoformat()
        self.logger.info(f"Analyzing layer: {layer_key}")
        
        # Filter edges to this layer
        layer_edges = [
            e for e in graph_data.edges
            if e.dependency_type in layer_def["dependency_types"]
        ]
        
        # Get component IDs involved in these edges
        layer_node_ids = set()
        for edge in layer_edges:
            layer_node_ids.add(edge.source_id)
            layer_node_ids.add(edge.target_id)
        
        if not layer_node_ids:
            return self._empty_layer_result(layer_key, layer_def["name"], timestamp)
        
        # Filter components
        layer_components = [c for c in graph_data.components if c.id in layer_node_ids]
        
        # Build subgraph for this layer
        layer_graph_data = GraphData(
            components=layer_components,
            edges=layer_edges,
            metadata={"layer": layer_key},
        )
        
        G = self._build_graph(layer_graph_data, weighted=weighted)
        
        # Compute centrality
        pagerank = self._compute_pagerank(G, weighted)
        betweenness = self._compute_betweenness(G, weighted)
        degree = self._compute_degree_centrality(G)
        in_degree, out_degree = self._compute_in_out_degree(G)
        
        # Structural analysis
        articulation_points = self._find_articulation_points(G)
        bridges = self._find_bridges(G)
        
        # Normalize
        pagerank_norm = self._normalize(pagerank)
        betweenness_norm = self._normalize(betweenness)
        degree_norm = self._normalize(degree)
        
        # Component weights
        component_weights = {c.id: c.weight for c in graph_data.components}
        component_types = {c.id: c.component_type for c in graph_data.components}
        
        # Build metrics
        metrics_list = []
        for node_id in layer_node_ids:
            composite = self._compute_composite(
                pagerank_norm.get(node_id, 0),
                betweenness_norm.get(node_id, 0),
                degree_norm.get(node_id, 0),
            )
            
            metrics_list.append(CentralityMetrics(
                component_id=node_id,
                component_type=component_types.get(node_id, "unknown"),
                pagerank=pagerank.get(node_id, 0),
                betweenness=betweenness.get(node_id, 0),
                degree=degree.get(node_id, 0),
                in_degree=in_degree.get(node_id, 0),
                out_degree=out_degree.get(node_id, 0),
                pagerank_norm=pagerank_norm.get(node_id, 0),
                betweenness_norm=betweenness_norm.get(node_id, 0),
                degree_norm=degree_norm.get(node_id, 0),
                composite_score=composite,
                is_articulation_point=node_id in articulation_points,
                weight=component_weights.get(node_id, 1.0),
            ))
        
        # Classify
        items = [{"id": m.component_id, "type": m.component_type, "score": m.composite_score}
                 for m in metrics_list]
        classification = self.classifier.classify(items, metric_name=f"{layer_key}_composite")
        
        # Update levels
        level_map = {item.id: (item.level, item.is_outlier) for item in classification.items}
        for metrics in metrics_list:
            level, is_outlier = level_map.get(metrics.component_id, (CriticalityLevel.MEDIUM, False))
            metrics.level = level
            metrics.is_outlier = is_outlier
        
        metrics_list.sort(key=lambda x: x.composite_score, reverse=True)
        
        summary = {
            "total": len(metrics_list),
            "by_level": classification.summary(),
            "articulation_points": len(articulation_points),
            "bridges": len(bridges),
            "critical_count": len([m for m in metrics_list if m.level == CriticalityLevel.CRITICAL]),
        }
        
        return LayerAnalysisResult(
            layer_key=layer_key,
            layer_name=layer_def["name"],
            timestamp=timestamp,
            components=metrics_list,
            classification=classification,
            articulation_points=articulation_points,
            bridges=bridges,
            summary=summary,
        )
    
    def analyze_edges(
        self,
        graph_data: GraphData,
        articulation_points: Optional[Set[str]] = None,
    ) -> EdgeAnalysisResult:
        """
        Analyze edge criticality.
        
        Args:
            graph_data: Graph data from Neo4j
            articulation_points: Pre-computed articulation points (optional)
        
        Returns:
            EdgeAnalysisResult with edge metrics and classification
        """
        timestamp = datetime.now().isoformat()
        self.logger.info("Analyzing edges")
        
        if not graph_data.edges:
            return EdgeAnalysisResult(timestamp=timestamp)
        
        # Build graph
        G = self._build_graph(graph_data, weighted=True)
        
        # Find bridges and articulation points
        bridges = self._find_bridges(G)
        bridge_set = set(bridges)
        
        if articulation_points is None:
            articulation_points = self._find_articulation_points(G)
        
        # Calculate max weight for normalization
        max_weight = max(e.weight for e in graph_data.edges) or 1.0
        
        # Build edge metrics
        edge_metrics = []
        for edge in graph_data.edges:
            # Check if bridge (undirected comparison)
            is_bridge = (
                (edge.source_id, edge.target_id) in bridge_set or
                (edge.target_id, edge.source_id) in bridge_set
            )
            
            # Check if connects articulation point
            connects_ap = (
                edge.source_id in articulation_points or
                edge.target_id in articulation_points
            )
            
            # Calculate criticality score
            weight_norm = edge.weight / max_weight
            bridge_bonus = 0.4 if is_bridge else 0.0
            ap_bonus = 0.2 if connects_ap else 0.0
            criticality = 0.4 * weight_norm + bridge_bonus + ap_bonus
            
            edge_metrics.append(EdgeMetrics(
                source_id=edge.source_id,
                target_id=edge.target_id,
                source_type=edge.source_type,
                target_type=edge.target_type,
                dependency_type=edge.dependency_type,
                weight=edge.weight,
                criticality_score=criticality,
                is_bridge=is_bridge,
                connects_articulation_point=connects_ap,
            ))
        
        # Classify edges
        items = [{"id": f"{e.source_id}->{e.target_id}", "type": e.dependency_type, 
                  "score": e.criticality_score} for e in edge_metrics]
        classification = self.classifier.classify(items, metric_name="edge_criticality")
        
        # Update levels
        level_map = {item.id: (item.level, item.is_outlier) for item in classification.items}
        for em in edge_metrics:
            edge_id = f"{em.source_id}->{em.target_id}"
            level, is_outlier = level_map.get(edge_id, (CriticalityLevel.MEDIUM, False))
            em.level = level
            em.is_outlier = is_outlier
        
        edge_metrics.sort(key=lambda x: x.criticality_score, reverse=True)
        
        summary = {
            "total": len(edge_metrics),
            "by_level": classification.summary(),
            "bridges": len(bridges),
            "critical_count": len([e for e in edge_metrics if e.level == CriticalityLevel.CRITICAL]),
        }
        
        return EdgeAnalysisResult(
            timestamp=timestamp,
            edges=edge_metrics,
            classification=classification,
            bridges=bridges,
            summary=summary,
        )
    
    # =========================================================================
    # Graph Algorithms
    # =========================================================================
    
    def _build_graph(
        self,
        graph_data: GraphData,
        weighted: bool = True,
        directed: bool = True,
    ) -> nx.DiGraph:
        """Build graph from GraphData."""
        G = nx.DiGraph() if directed else nx.Graph()
        
        # Add nodes
        for comp in graph_data.components:
            G.add_node(comp.id, type=comp.component_type, weight=comp.weight)
        
        # Add edges
        for edge in graph_data.edges:
            edge_weight = edge.weight if weighted else 1.0
            G.add_edge(
                edge.source_id, 
                edge.target_id,
                weight=edge_weight,
                dependency_type=edge.dependency_type,
            )
        
        return G
    
    def _compute_pagerank(
        self,
        G: nx.DiGraph,
        weighted: bool = True,
    ) -> Dict[str, float]:
        """Compute PageRank using NetworkX."""
        if G.number_of_nodes() == 0:
            return {}
        
        try:
            weight_attr = "weight" if weighted else None
            return nx.pagerank(G, alpha=self.damping_factor, weight=weight_attr)
        except Exception as e:
            self.logger.warning(f"PageRank failed: {e}")
            # Fallback to equal distribution
            n = G.number_of_nodes()
            return {node: 1.0 / n for node in G.nodes()}
    
    def _compute_betweenness(
        self,
        G: nx.DiGraph,
        weighted: bool = True,
    ) -> Dict[str, float]:
        """Compute betweenness centrality using NetworkX."""
        if G.number_of_nodes() == 0:
            return {}
        
        try:
            weight_attr = "weight" if weighted else None
            return nx.betweenness_centrality(G, weight=weight_attr, normalized=True)
        except Exception as e:
            self.logger.warning(f"Betweenness failed: {e}")
            return {node: 0.0 for node in G.nodes()}
    
    def _compute_degree_centrality(self, G: nx.DiGraph) -> Dict[str, float]:
        """Compute degree centrality using NetworkX."""
        if G.number_of_nodes() == 0:
            return {}
        
        return nx.degree_centrality(G)
    
    def _compute_in_out_degree(
        self,
        G: nx.DiGraph,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute in-degree and out-degree centrality."""
        if G.number_of_nodes() == 0:
            return {}, {}
        
        n = G.number_of_nodes()
        if n <= 1:
            return {node: 0.0 for node in G.nodes()}, {node: 0.0 for node in G.nodes()}
        
        in_degree = {node: G.in_degree(node) / (n - 1) for node in G.nodes()}
        out_degree = {node: G.out_degree(node) / (n - 1) for node in G.nodes()}
        
        return in_degree, out_degree
    
    def _find_articulation_points(self, G: nx.DiGraph) -> Set[str]:
        """Find articulation points using NetworkX."""
        if G.number_of_nodes() == 0:
            return set()
        
        # Convert to undirected for articulation point detection
        G_undirected = G.to_undirected()
        
        try:
            return set(nx.articulation_points(G_undirected))
        except Exception as e:
            self.logger.warning(f"Articulation points failed: {e}")
            return set()
    
    def _find_bridges(self, G: nx.DiGraph) -> List[Tuple[str, str]]:
        """Find bridge edges using NetworkX."""
        if G.number_of_edges() == 0:
            return []
        
        # Convert to undirected for bridge detection
        G_undirected = G.to_undirected()
        
        try:
            return list(nx.bridges(G_undirected))
        except Exception as e:
            self.logger.warning(f"Bridges failed: {e}")
            return []
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [0, 1] using min-max scaling."""
        if not scores:
            return {}
        
        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return {k: 0.5 for k in scores}
        
        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}
    
    def _compute_composite(
        self,
        pagerank_norm: float,
        betweenness_norm: float,
        degree_norm: float,
    ) -> float:
        """Compute weighted composite score."""
        return (
            self.weights["pagerank"] * pagerank_norm +
            self.weights["betweenness"] * betweenness_norm +
            self.weights["degree"] * degree_norm
        )
    
    def _empty_component_result(
        self,
        component_type: str,
        timestamp: str,
    ) -> ComponentAnalysisResult:
        """Create empty result for edge cases."""
        return ComponentAnalysisResult(
            component_type=component_type,
            timestamp=timestamp,
            summary={"total": 0, "by_level": {}, "articulation_points": 0, "critical_count": 0},
        )
    
    def _empty_layer_result(
        self,
        layer_key: str,
        layer_name: str,
        timestamp: str,
    ) -> LayerAnalysisResult:
        """Create empty result for edge cases."""
        return LayerAnalysisResult(
            layer_key=layer_key,
            layer_name=layer_name,
            timestamp=timestamp,
            summary={"total": 0, "by_level": {}, "articulation_points": 0, "bridges": 0, "critical_count": 0},
        )
