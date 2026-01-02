"""
Component Type Analyzer - Version 5.0

Analyzes components by type for fair comparison within categories.

Compares:
- Applications with other Applications
- Brokers with other Brokers
- Nodes with other Nodes
- Topics with other Topics

This avoids misleading cross-type comparisons where different component
types naturally have different score distributions.

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

from .gds_client import GDSClient, CentralityResult, COMPONENT_TYPES
from .classifier import BoxPlotClassifier, ClassificationResult, CriticalityLevel


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ComponentMetrics:
    """
    Metrics for a single component.
    """
    component_id: str
    component_type: str
    
    # Raw centrality scores
    pagerank: float = 0.0
    betweenness: float = 0.0
    degree: float = 0.0
    
    # Normalized scores (within type)
    pagerank_norm: float = 0.0
    betweenness_norm: float = 0.0
    degree_norm: float = 0.0
    
    # Component weight from database
    weight: float = 0.0
    
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
            "weight": round(self.weight, 4),
            "composite_score": round(self.composite_score, 4),
            "level": self.level.value,
            "is_articulation_point": self.is_articulation_point,
        }


@dataclass
class ComponentTypeResult:
    """
    Analysis result for a single component type.
    """
    component_type: str
    timestamp: str
    
    # All component metrics
    components: List[ComponentMetrics] = field(default_factory=list)
    
    # Classifications
    composite_classification: Optional[ClassificationResult] = None
    pagerank_classification: Optional[ClassificationResult] = None
    betweenness_classification: Optional[ClassificationResult] = None
    
    # Summary statistics
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_type": self.component_type,
            "timestamp": self.timestamp,
            "count": len(self.components),
            "components": [c.to_dict() for c in self.components],
            "classifications": {
                "composite": self.composite_classification.to_dict() if self.composite_classification else None,
                "pagerank": self.pagerank_classification.to_dict() if self.pagerank_classification else None,
                "betweenness": self.betweenness_classification.to_dict() if self.betweenness_classification else None,
            },
            "summary": self.summary,
        }
    
    def get_critical(self) -> List[ComponentMetrics]:
        """Get critical components."""
        return [c for c in self.components if c.level == CriticalityLevel.CRITICAL]
    
    def get_high_and_above(self) -> List[ComponentMetrics]:
        """Get high and critical components."""
        return [c for c in self.components if c.level >= CriticalityLevel.HIGH]
    
    def top_n(self, n: int = 10) -> List[ComponentMetrics]:
        """Get top N by composite score."""
        return sorted(self.components, key=lambda c: c.composite_score, reverse=True)[:n]


@dataclass
class AllTypesResult:
    """
    Analysis result for all component types.
    """
    timestamp: str
    by_type: Dict[str, ComponentTypeResult] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "types": {k: v.to_dict() for k, v in self.by_type.items()},
            "summary": self.summary,
        }
    
    def get_all_critical(self) -> Dict[str, List[ComponentMetrics]]:
        """Get critical components from all types."""
        return {
            comp_type: result.get_critical()
            for comp_type, result in self.by_type.items()
        }


# =============================================================================
# Component Type Analyzer
# =============================================================================

class ComponentTypeAnalyzer:
    """
    Analyzes components by type for fair within-type comparison.
    
    Key Insight:
        Applications, Brokers, Nodes, and Topics have fundamentally
        different roles in the system. Comparing across types can be
        misleading. This analyzer computes separate statistics for
        each type, enabling fair comparison.
    
    Example:
        with GDSClient(uri, user, password) as gds:
            analyzer = ComponentTypeAnalyzer(gds)
            
            # Analyze applications
            app_result = analyzer.analyze_type("Application")
            
            # Get critical applications
            for app in app_result.get_critical():
                print(f"{app.component_id}: {app.composite_score:.4f}")
            
            # Analyze all types
            all_result = analyzer.analyze_all_types()
            
            for comp_type, critical_list in all_result.get_all_critical().items():
                print(f"{comp_type}: {len(critical_list)} critical")
    """
    
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
            k_factor: Box-plot k factor
            weights: Composite score weights
        """
        self.gds = gds_client
        self.classifier = BoxPlotClassifier(k_factor=k_factor)
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.logger = logging.getLogger(__name__)
    
    def analyze_type(
        self,
        component_type: str,
        weighted: bool = True,
    ) -> ComponentTypeResult:
        """
        Analyze all components of a specific type.
        
        Args:
            component_type: Type to analyze (Application, Broker, Node, Topic)
            weighted: Use weighted algorithms
        
        Returns:
            ComponentTypeResult with analysis for this type
        """
        if component_type not in COMPONENT_TYPES:
            raise ValueError(
                f"Invalid type: {component_type}. "
                f"Valid types: {COMPONENT_TYPES}"
            )
        
        timestamp = datetime.now().isoformat()
        projection_name = f"type_{component_type.lower()}"
        
        self.logger.info(f"Analyzing component type: {component_type}")
        
        try:
            # Create full projection (we need relationships to compute centrality)
            self.gds.create_projection(projection_name, include_weights=weighted)
            
            # Compute metrics for this type only
            components = self._compute_type_metrics(
                projection_name, component_type, weighted
            )
            
            if not components:
                return self._empty_result(component_type, timestamp)
            
            # Classify within type
            composite_class = self._classify_metric(components, "composite_score")
            pr_class = self._classify_metric(components, "pagerank_norm")
            bc_class = self._classify_metric(components, "betweenness_norm")
            
            # Assign levels
            level_map = {item.id: item.level for item in composite_class.items}
            for c in components:
                c.level = level_map.get(c.component_id, CriticalityLevel.MEDIUM)
            
            # Generate summary
            summary = self._generate_summary(components, composite_class)
            
            return ComponentTypeResult(
                component_type=component_type,
                timestamp=timestamp,
                components=components,
                composite_classification=composite_class,
                pagerank_classification=pr_class,
                betweenness_classification=bc_class,
                summary=summary,
            )
        
        finally:
            self.gds.drop_projection(projection_name)
    
    def analyze_all_types(
        self,
        weighted: bool = True,
    ) -> AllTypesResult:
        """
        Analyze all component types.
        
        Args:
            weighted: Use weighted algorithms
        
        Returns:
            AllTypesResult with analysis for each type
        """
        timestamp = datetime.now().isoformat()
        
        by_type = {}
        for comp_type in COMPONENT_TYPES:
            try:
                by_type[comp_type] = self.analyze_type(comp_type, weighted=weighted)
            except Exception as e:
                self.logger.error(f"Failed to analyze {comp_type}: {e}")
        
        # Generate summary
        summary = self._generate_all_types_summary(by_type)
        
        return AllTypesResult(
            timestamp=timestamp,
            by_type=by_type,
            summary=summary,
        )
    
    def _compute_type_metrics(
        self,
        projection_name: str,
        component_type: str,
        weighted: bool,
    ) -> List[ComponentMetrics]:
        """Compute metrics for components of a specific type."""
        metrics_map: Dict[str, ComponentMetrics] = {}
        
        # Get component weights from database
        db_weights = self.gds.get_component_weights()
        
        # PageRank
        pr_results = self.gds.pagerank(projection_name, weighted=weighted)
        type_pr = [r for r in pr_results if r.node_type == component_type]
        max_pr = max((r.score for r in type_pr), default=1.0) or 1.0
        
        for r in type_pr:
            metrics_map[r.node_id] = ComponentMetrics(
                component_id=r.node_id,
                component_type=r.node_type,
                pagerank=r.score,
                pagerank_norm=r.score / max_pr,
                weight=db_weights.get(r.node_id, 0.0),
            )
        
        # Betweenness
        bc_results = self.gds.betweenness(projection_name, weighted=weighted)
        type_bc = [r for r in bc_results if r.node_type == component_type]
        max_bc = max((r.score for r in type_bc), default=1.0) or 1.0
        
        for r in type_bc:
            if r.node_id in metrics_map:
                metrics_map[r.node_id].betweenness = r.score
                metrics_map[r.node_id].betweenness_norm = r.score / max_bc
        
        # Degree
        dc_results = self.gds.degree(projection_name, weighted=weighted)
        type_dc = [r for r in dc_results if r.node_type == component_type]
        max_dc = max((r.score for r in type_dc), default=1.0) or 1.0
        
        for r in type_dc:
            if r.node_id in metrics_map:
                metrics_map[r.node_id].degree = r.score
                metrics_map[r.node_id].degree_norm = r.score / max_dc
        
        # Compute composite scores
        for m in metrics_map.values():
            m.composite_score = (
                self.weights["pagerank"] * m.pagerank_norm +
                self.weights["betweenness"] * m.betweenness_norm +
                self.weights["degree"] * m.degree_norm
            )
        
        # Mark articulation points
        ap_results = self.gds.find_articulation_points()
        ap_ids = {ap.node_id for ap in ap_results if ap.node_type == component_type}
        
        for m in metrics_map.values():
            m.is_articulation_point = m.component_id in ap_ids
        
        return list(metrics_map.values())
    
    def _classify_metric(
        self,
        components: List[ComponentMetrics],
        metric_attr: str,
    ) -> ClassificationResult:
        """Classify components by a metric."""
        items = [
            {
                "id": c.component_id,
                "type": c.component_type,
                "score": getattr(c, metric_attr, 0.0),
            }
            for c in components
        ]
        return self.classifier.classify(items, metric_name=metric_attr)
    
    def _generate_summary(
        self,
        components: List[ComponentMetrics],
        classification: ClassificationResult,
    ) -> Dict[str, Any]:
        """Generate type summary."""
        scores = [c.composite_score for c in components]
        weights = [c.weight for c in components]
        
        return {
            "count": len(components),
            "by_level": classification.summary(),
            "critical_count": len([c for c in components if c.level == CriticalityLevel.CRITICAL]),
            "articulation_points": len([c for c in components if c.is_articulation_point]),
            "score_stats": classification.stats.to_dict(),
            "weight_stats": {
                "min": round(min(weights), 4) if weights else 0,
                "max": round(max(weights), 4) if weights else 0,
                "mean": round(sum(weights) / len(weights), 4) if weights else 0,
            },
        }
    
    def _generate_all_types_summary(
        self,
        by_type: Dict[str, ComponentTypeResult],
    ) -> Dict[str, Any]:
        """Generate summary across all types."""
        summary = {
            "total_components": 0,
            "total_critical": 0,
            "by_type": {},
        }
        
        for comp_type, result in by_type.items():
            count = len(result.components)
            critical = len(result.get_critical())
            
            summary["total_components"] += count
            summary["total_critical"] += critical
            summary["by_type"][comp_type] = {
                "count": count,
                "critical": critical,
            }
        
        return summary
    
    def _empty_result(
        self,
        component_type: str,
        timestamp: str,
    ) -> ComponentTypeResult:
        """Create empty result."""
        return ComponentTypeResult(
            component_type=component_type,
            timestamp=timestamp,
            components=[],
            summary={"count": 0, "by_level": {}},
        )
