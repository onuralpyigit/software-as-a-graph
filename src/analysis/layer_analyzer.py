"""
Layer Analyzer - Version 5.0

Multi-layer graph analysis for DEPENDS_ON relationships.

Analyzes each dependency layer separately:
- Application Layer: app_to_app dependencies
- Infrastructure Layer: node_to_node dependencies  
- Application-Broker Layer: app_to_broker dependencies
- Node-Broker Layer: node_to_broker dependencies

Each layer is analyzed independently with its own metrics and classification.

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

from .gds_client import (
    GDSClient, 
    CentralityResult, 
    ProjectionInfo,
    LAYER_DEFINITIONS,
    DEPENDENCY_TYPES,
)
from .classifier import (
    BoxPlotClassifier,
    ClassificationResult,
    CriticalityLevel,
    ClassifiedItem,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LayerMetrics:
    """
    Metrics for a single component within a layer.
    """
    component_id: str
    component_type: str
    
    # Raw scores
    pagerank: float = 0.0
    betweenness: float = 0.0
    degree: float = 0.0
    in_degree: float = 0.0
    out_degree: float = 0.0
    
    # Normalized scores (0-1)
    pagerank_norm: float = 0.0
    betweenness_norm: float = 0.0
    degree_norm: float = 0.0
    
    # Composite score
    composite_score: float = 0.0
    
    # Classification
    level: CriticalityLevel = CriticalityLevel.MEDIUM
    
    # Structural flags
    is_articulation_point: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.component_id,
            "type": self.component_type,
            "pagerank": round(self.pagerank, 6),
            "betweenness": round(self.betweenness, 6),
            "degree": round(self.degree, 6),
            "pagerank_norm": round(self.pagerank_norm, 4),
            "betweenness_norm": round(self.betweenness_norm, 4),
            "degree_norm": round(self.degree_norm, 4),
            "composite_score": round(self.composite_score, 4),
            "level": self.level.value,
            "is_articulation_point": self.is_articulation_point,
        }


@dataclass
class LayerResult:
    """
    Analysis result for a single layer.
    """
    layer_name: str
    layer_key: str
    timestamp: str
    
    # Graph info
    projection: ProjectionInfo
    
    # Metrics for all components in layer
    metrics: List[LayerMetrics] = field(default_factory=list)
    
    # Classifications
    pagerank_classification: Optional[ClassificationResult] = None
    betweenness_classification: Optional[ClassificationResult] = None
    degree_classification: Optional[ClassificationResult] = None
    composite_classification: Optional[ClassificationResult] = None
    
    # Summary
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "layer_key": self.layer_key,
            "timestamp": self.timestamp,
            "projection": self.projection.to_dict(),
            "component_count": len(self.metrics),
            "metrics": [m.to_dict() for m in self.metrics],
            "classifications": {
                "pagerank": self.pagerank_classification.to_dict() if self.pagerank_classification else None,
                "betweenness": self.betweenness_classification.to_dict() if self.betweenness_classification else None,
                "degree": self.degree_classification.to_dict() if self.degree_classification else None,
                "composite": self.composite_classification.to_dict() if self.composite_classification else None,
            },
            "summary": self.summary,
        }
    
    def get_critical_components(self) -> List[LayerMetrics]:
        """Get components classified as CRITICAL."""
        return [m for m in self.metrics if m.level == CriticalityLevel.CRITICAL]
    
    def get_high_and_above(self) -> List[LayerMetrics]:
        """Get components classified as HIGH or CRITICAL."""
        return [m for m in self.metrics if m.level >= CriticalityLevel.HIGH]
    
    def top_n(self, n: int = 10) -> List[LayerMetrics]:
        """Get top N components by composite score."""
        return sorted(self.metrics, key=lambda m: m.composite_score, reverse=True)[:n]


@dataclass
class MultiLayerResult:
    """
    Complete multi-layer analysis result.
    """
    timestamp: str
    layers: Dict[str, LayerResult] = field(default_factory=dict)
    graph_stats: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
            "graph_stats": self.graph_stats,
            "summary": self.summary,
        }
    
    def get_all_critical(self) -> Dict[str, List[LayerMetrics]]:
        """Get critical components from all layers."""
        return {
            layer_key: layer.get_critical_components()
            for layer_key, layer in self.layers.items()
        }


# =============================================================================
# Layer Analyzer
# =============================================================================

class LayerAnalyzer:
    """
    Analyzes graph by dependency layers.
    
    Each layer represents a different type of dependency relationship,
    allowing for targeted analysis of specific system aspects.
    
    Example:
        with GDSClient(uri, user, password) as gds:
            analyzer = LayerAnalyzer(gds)
            
            # Analyze single layer
            app_result = analyzer.analyze_layer("application")
            
            # Analyze all layers
            full_result = analyzer.analyze_all_layers()
            
            # Get critical components across layers
            for layer, components in full_result.get_all_critical().items():
                print(f"{layer}: {len(components)} critical")
    """
    
    # Default weights for composite score
    DEFAULT_WEIGHTS = {
        "pagerank": 0.35,
        "betweenness": 0.40,
        "degree": 0.25,
    }
    
    def __init__(
        self,
        gds_client: GDSClient,
        k_factor: float = 1.5,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize analyzer.
        
        Args:
            gds_client: Connected GDS client
            k_factor: Box-plot k factor for classification
            weights: Weights for composite score
        """
        self.gds = gds_client
        self.classifier = BoxPlotClassifier(k_factor=k_factor)
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.logger = logging.getLogger(__name__)
    
    def analyze_layer(
        self,
        layer_key: str,
        weighted: bool = True,
    ) -> LayerResult:
        """
        Analyze a single dependency layer.
        
        Args:
            layer_key: Layer identifier (application, infrastructure, etc.)
            weighted: Use weighted algorithms
        
        Returns:
            LayerResult with analysis for this layer
        """
        if layer_key not in LAYER_DEFINITIONS:
            raise ValueError(
                f"Unknown layer: {layer_key}. "
                f"Valid layers: {list(LAYER_DEFINITIONS.keys())}"
            )
        
        layer_def = LAYER_DEFINITIONS[layer_key]
        timestamp = datetime.now().isoformat()
        projection_name = f"layer_{layer_key}"
        
        self.logger.info(f"Analyzing layer: {layer_def['name']}")
        
        try:
            # Create layer-specific projection
            projection = self.gds.create_filtered_projection(
                name=projection_name,
                component_types=layer_def["component_types"],
                dependency_types=layer_def["dependency_types"],
                include_weights=weighted,
            )
            
            if projection.relationship_count == 0:
                self.logger.warning(f"No relationships in layer {layer_key}")
                return self._empty_result(layer_key, layer_def["name"], timestamp, projection)
            
            # Compute centrality metrics
            metrics = self._compute_layer_metrics(projection_name, weighted)
            
            if not metrics:
                return self._empty_result(layer_key, layer_def["name"], timestamp, projection)
            
            # Classify by each metric
            pr_class = self._classify_metric(metrics, "pagerank")
            bc_class = self._classify_metric(metrics, "betweenness")
            dc_class = self._classify_metric(metrics, "degree")
            comp_class = self._classify_metric(metrics, "composite_score")
            
            # Assign levels to metrics based on composite
            level_map = {item.id: item.level for item in comp_class.items}
            for m in metrics:
                m.level = level_map.get(m.component_id, CriticalityLevel.MEDIUM)
            
            # Generate summary
            summary = self._generate_summary(metrics, comp_class)
            
            return LayerResult(
                layer_name=layer_def["name"],
                layer_key=layer_key,
                timestamp=timestamp,
                projection=projection,
                metrics=metrics,
                pagerank_classification=pr_class,
                betweenness_classification=bc_class,
                degree_classification=dc_class,
                composite_classification=comp_class,
                summary=summary,
            )
        
        finally:
            self.gds.drop_projection(projection_name)
    
    def analyze_all_layers(
        self,
        weighted: bool = True,
        include_full: bool = True,
    ) -> MultiLayerResult:
        """
        Analyze all dependency layers.
        
        Args:
            weighted: Use weighted algorithms
            include_full: Include full system analysis
        
        Returns:
            MultiLayerResult with all layer analyses
        """
        timestamp = datetime.now().isoformat()
        
        layers_to_analyze = ["application", "infrastructure", "app_broker", "node_broker"]
        if include_full:
            layers_to_analyze.append("full")
        
        layers = {}
        for layer_key in layers_to_analyze:
            try:
                layers[layer_key] = self.analyze_layer(layer_key, weighted=weighted)
            except Exception as e:
                self.logger.error(f"Failed to analyze layer {layer_key}: {e}")
        
        # Get overall stats
        graph_stats = self.gds.get_graph_stats()
        
        # Generate cross-layer summary
        summary = self._generate_multi_layer_summary(layers)
        
        return MultiLayerResult(
            timestamp=timestamp,
            layers=layers,
            graph_stats=graph_stats,
            summary=summary,
        )
    
    def _compute_layer_metrics(
        self,
        projection_name: str,
        weighted: bool,
    ) -> List[LayerMetrics]:
        """Compute all centrality metrics for a layer."""
        metrics_map: Dict[str, LayerMetrics] = {}
        
        # PageRank
        pr_results = self.gds.pagerank(projection_name, weighted=weighted)
        max_pr = max((r.score for r in pr_results), default=1.0) or 1.0
        
        for r in pr_results:
            metrics_map[r.node_id] = LayerMetrics(
                component_id=r.node_id,
                component_type=r.node_type,
                pagerank=r.score,
                pagerank_norm=r.score / max_pr,
            )
        
        # Betweenness
        bc_results = self.gds.betweenness(projection_name, weighted=weighted)
        max_bc = max((r.score for r in bc_results), default=1.0) or 1.0
        
        for r in bc_results:
            if r.node_id in metrics_map:
                metrics_map[r.node_id].betweenness = r.score
                metrics_map[r.node_id].betweenness_norm = r.score / max_bc
        
        # Degree
        dc_results = self.gds.degree(projection_name, weighted=weighted)
        max_dc = max((r.score for r in dc_results), default=1.0) or 1.0
        
        for r in dc_results:
            if r.node_id in metrics_map:
                metrics_map[r.node_id].degree = r.score
                metrics_map[r.node_id].degree_norm = r.score / max_dc
        
        # In-degree
        in_results = self.gds.degree(projection_name, weighted=weighted, orientation="REVERSE")
        for r in in_results:
            if r.node_id in metrics_map:
                metrics_map[r.node_id].in_degree = r.score
        
        # Out-degree
        out_results = self.gds.degree(projection_name, weighted=weighted, orientation="NATURAL")
        for r in out_results:
            if r.node_id in metrics_map:
                metrics_map[r.node_id].out_degree = r.score
        
        # Compute composite scores
        for m in metrics_map.values():
            m.composite_score = (
                self.weights["pagerank"] * m.pagerank_norm +
                self.weights["betweenness"] * m.betweenness_norm +
                self.weights["degree"] * m.degree_norm
            )
        
        # Mark articulation points
        ap_results = self.gds.find_articulation_points()
        ap_ids = {ap.node_id for ap in ap_results}
        
        for m in metrics_map.values():
            m.is_articulation_point = m.component_id in ap_ids
        
        return list(metrics_map.values())
    
    def _classify_metric(
        self,
        metrics: List[LayerMetrics],
        metric_attr: str,
    ) -> ClassificationResult:
        """Classify metrics by a specific attribute."""
        items = [
            {
                "id": m.component_id,
                "type": m.component_type,
                "score": getattr(m, metric_attr, 0.0),
            }
            for m in metrics
        ]
        return self.classifier.classify(items, metric_name=metric_attr)
    
    def _generate_summary(
        self,
        metrics: List[LayerMetrics],
        composite_class: ClassificationResult,
    ) -> Dict[str, Any]:
        """Generate layer summary."""
        return {
            "total_components": len(metrics),
            "by_type": self._count_by_type(metrics),
            "by_level": composite_class.summary(),
            "critical_count": len([m for m in metrics if m.level == CriticalityLevel.CRITICAL]),
            "articulation_points": len([m for m in metrics if m.is_articulation_point]),
            "stats": composite_class.stats.to_dict(),
        }
    
    def _count_by_type(self, metrics: List[LayerMetrics]) -> Dict[str, int]:
        """Count metrics by component type."""
        counts: Dict[str, int] = {}
        for m in metrics:
            counts[m.component_type] = counts.get(m.component_type, 0) + 1
        return counts
    
    def _generate_multi_layer_summary(
        self,
        layers: Dict[str, LayerResult],
    ) -> Dict[str, Any]:
        """Generate cross-layer summary."""
        total_critical = 0
        total_components = 0
        
        layer_summaries = {}
        for key, layer in layers.items():
            critical = len(layer.get_critical_components())
            total = len(layer.metrics)
            total_critical += critical
            total_components += total
            
            layer_summaries[key] = {
                "components": total,
                "critical": critical,
                "articulation_points": layer.summary.get("articulation_points", 0),
            }
        
        return {
            "total_layers": len(layers),
            "total_components_analyzed": total_components,
            "total_critical": total_critical,
            "layers": layer_summaries,
        }
    
    def _empty_result(
        self,
        layer_key: str,
        layer_name: str,
        timestamp: str,
        projection: ProjectionInfo,
    ) -> LayerResult:
        """Create empty layer result."""
        return LayerResult(
            layer_name=layer_name,
            layer_key=layer_key,
            timestamp=timestamp,
            projection=projection,
            metrics=[],
            summary={"total_components": 0, "by_level": {}, "critical_count": 0},
        )
