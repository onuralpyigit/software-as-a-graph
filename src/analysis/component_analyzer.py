"""
Component-Type Analyzer - Version 5.0

Analyzes components by type (Application, Topic, Node, Broker) separately.

Key Feature:
- Evaluates and compares components of the SAME TYPE
- Each type gets its own statistics and classification
- Enables fair comparison within categories

Why Analyze by Type?
- Applications, Brokers, Topics, and Nodes have different roles
- Comparing across types can be misleading
- A "high" score for a Topic may mean something different than for a Broker
- Type-specific analysis reveals role-specific criticality

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

from .gds_client import GDSClient, CentralityResult, ComponentType
from .classifier import (
    BoxPlotClassifier,
    ClassificationResult,
    CriticalityLevel,
    BoxPlotStats,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ComponentMetrics:
    """
    All computed metrics for a single component.
    
    Stores raw centrality scores and derived metrics.
    """
    component_id: str
    component_type: str
    
    # Centrality scores
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
    
    # Structural flags
    is_articulation_point: bool = False
    is_bridge_endpoint: bool = False
    
    # Community info
    community_id: int = -1
    
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
            "is_articulation_point": self.is_articulation_point,
            "is_bridge_endpoint": self.is_bridge_endpoint,
            "community_id": self.community_id,
        }


@dataclass
class ComponentTypeResult:
    """
    Analysis result for a single component type.
    
    Contains metrics and classifications for all components of one type.
    """
    component_type: str
    timestamp: str
    component_count: int
    
    # Raw metrics for each component
    metrics: List[ComponentMetrics]
    
    # Classifications by metric
    pagerank_classification: ClassificationResult
    betweenness_classification: ClassificationResult
    degree_classification: ClassificationResult
    composite_classification: ClassificationResult
    
    # Summary statistics
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_type": self.component_type,
            "timestamp": self.timestamp,
            "component_count": self.component_count,
            "metrics": [m.to_dict() for m in self.metrics],
            "classifications": {
                "pagerank": self.pagerank_classification.to_dict(),
                "betweenness": self.betweenness_classification.to_dict(),
                "degree": self.degree_classification.to_dict(),
                "composite": self.composite_classification.to_dict(),
            },
            "summary": self.summary,
        }
    
    def get_critical_components(self) -> List[ComponentMetrics]:
        """Get components classified as CRITICAL by composite score"""
        critical_ids = {
            item.id for item in self.composite_classification.get_critical()
        }
        return [m for m in self.metrics if m.component_id in critical_ids]
    
    def get_top_n(self, n: int = 10) -> List[ComponentMetrics]:
        """Get top N components by composite score"""
        return sorted(
            self.metrics, 
            key=lambda m: m.composite_score, 
            reverse=True
        )[:n]


# =============================================================================
# Component-Type Analyzer
# =============================================================================

class ComponentTypeAnalyzer:
    """
    Analyzes components by type separately.
    
    Key capability:
    - Analyzes all components of a single type
    - Computes centrality metrics within that type's subgraph
    - Classifies using box-plot method for fair comparison
    
    Example:
        with GDSClient(uri, user, password) as gds:
            analyzer = ComponentTypeAnalyzer(gds)
            
            # Analyze all applications
            app_result = analyzer.analyze("Application")
            
            # Get critical applications
            for comp in app_result.get_critical_components():
                print(f"{comp.component_id}: {comp.composite_score:.4f}")
            
            # Analyze all brokers
            broker_result = analyzer.analyze("Broker")
    """

    # Default weights for composite score
    DEFAULT_WEIGHTS = {
        "betweenness": 0.40,  # Bottleneck importance
        "pagerank": 0.35,     # Dependency importance
        "degree": 0.25,       # Coupling level
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
            weights: Weights for composite score calculation
        """
        self.gds = gds_client
        self.classifier = BoxPlotClassifier(k_factor=k_factor)
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.logger = logging.getLogger(__name__)

    def analyze(
        self,
        component_type: str,
        weighted: bool = True,
        include_structural: bool = True,
    ) -> ComponentTypeResult:
        """
        Analyze all components of a specific type.
        
        Creates a subgraph projection for just this component type,
        computes centrality metrics, and classifies using box-plot method.
        
        Args:
            component_type: Type to analyze (Application, Broker, Node, Topic)
            weighted: Use weighted algorithms
            include_structural: Include articulation point detection
        
        Returns:
            ComponentTypeResult with all metrics and classifications
        """
        timestamp = datetime.now().isoformat()
        
        self.logger.info(f"Analyzing component type: {component_type}")
        
        # Validate component type
        if component_type not in GDSClient.VALID_COMPONENT_TYPES:
            raise ValueError(
                f"Invalid component type: {component_type}. "
                f"Valid types: {GDSClient.VALID_COMPONENT_TYPES}"
            )
        
        # Create type-specific projection
        projection_name = f"type_{component_type.lower()}"
        
        try:
            projection_info = self.gds.create_component_type_projection(
                projection_name, 
                component_type,
                include_weights=weighted,
            )
            
            # If no relationships, fall back to full projection
            if projection_info.relationship_count == 0:
                self.logger.info(
                    f"No intra-type relationships for {component_type}, "
                    f"using full projection"
                )
                self.gds.drop_projection(projection_name)
                projection_info = self.gds.create_projection(
                    projection_name,
                    component_types=[component_type],
                    include_weights=weighted,
                )
            
            # Compute metrics
            metrics = self._compute_metrics(
                projection_name, 
                component_type, 
                weighted,
                include_structural,
            )
            
            if not metrics:
                return self._empty_result(component_type, timestamp)
            
            # Classify by each metric
            pagerank_class = self._classify_by_metric(
                metrics, "pagerank", "pagerank"
            )
            betweenness_class = self._classify_by_metric(
                metrics, "betweenness", "betweenness"
            )
            degree_class = self._classify_by_metric(
                metrics, "degree", "degree"
            )
            composite_class = self._classify_by_metric(
                metrics, "composite_score", "composite"
            )
            
            # Generate summary
            summary = self._generate_summary(
                metrics, 
                pagerank_class, 
                betweenness_class, 
                degree_class,
                composite_class,
            )
            
            return ComponentTypeResult(
                component_type=component_type,
                timestamp=timestamp,
                component_count=len(metrics),
                metrics=metrics,
                pagerank_classification=pagerank_class,
                betweenness_classification=betweenness_class,
                degree_classification=degree_class,
                composite_classification=composite_class,
                summary=summary,
            )
        
        finally:
            # Cleanup projection
            self.gds.drop_projection(projection_name)

    def _compute_metrics(
        self,
        projection_name: str,
        component_type: str,
        weighted: bool,
        include_structural: bool,
    ) -> List[ComponentMetrics]:
        """Compute all centrality metrics for components"""
        metrics_map: Dict[str, ComponentMetrics] = {}
        
        # PageRank
        self.logger.info("Computing PageRank...")
        pr_results = self.gds.pagerank(projection_name, weighted=weighted)
        max_pr = max((r.score for r in pr_results), default=1.0) or 1.0
        
        for result in pr_results:
            if result.node_type == component_type:
                metrics_map[result.node_id] = ComponentMetrics(
                    component_id=result.node_id,
                    component_type=result.node_type,
                    pagerank=result.score,
                    pagerank_norm=result.score / max_pr,
                )
        
        # Betweenness
        self.logger.info("Computing Betweenness...")
        bc_results = self.gds.betweenness(projection_name, weighted=weighted)
        max_bc = max((r.score for r in bc_results), default=1.0) or 1.0
        
        for result in bc_results:
            if result.node_id in metrics_map:
                metrics_map[result.node_id].betweenness = result.score
                metrics_map[result.node_id].betweenness_norm = result.score / max_bc
        
        # Degree (total)
        self.logger.info("Computing Degree...")
        dc_results = self.gds.degree(
            projection_name, weighted=weighted, orientation="UNDIRECTED"
        )
        max_dc = max((r.score for r in dc_results), default=1.0) or 1.0
        
        for result in dc_results:
            if result.node_id in metrics_map:
                metrics_map[result.node_id].degree = result.score
                metrics_map[result.node_id].degree_norm = result.score / max_dc
        
        # In-degree
        in_results = self.gds.degree(
            projection_name, weighted=weighted, orientation="REVERSE"
        )
        for result in in_results:
            if result.node_id in metrics_map:
                metrics_map[result.node_id].in_degree = result.score
        
        # Out-degree
        out_results = self.gds.degree(
            projection_name, weighted=weighted, orientation="NATURAL"
        )
        for result in out_results:
            if result.node_id in metrics_map:
                metrics_map[result.node_id].out_degree = result.score
        
        # Structural properties
        if include_structural:
            self.logger.info("Finding structural properties...")
            articulation_points = {
                ap["node_id"] for ap in self.gds.find_articulation_points()
            }
            bridges = self.gds.find_bridges()
            bridge_endpoints = set()
            for bridge in bridges:
                bridge_endpoints.add(bridge.get("source_id", ""))
                bridge_endpoints.add(bridge.get("target_id", ""))
            
            for node_id, metrics in metrics_map.items():
                metrics.is_articulation_point = node_id in articulation_points
                metrics.is_bridge_endpoint = node_id in bridge_endpoints
        
        # Compute composite scores
        for metrics in metrics_map.values():
            metrics.composite_score = self._compute_composite_score(metrics)
        
        return list(metrics_map.values())

    def _compute_composite_score(self, metrics: ComponentMetrics) -> float:
        """Compute composite criticality score"""
        score = 0.0
        score += self.weights.get("betweenness", 0.4) * metrics.betweenness_norm
        score += self.weights.get("pagerank", 0.35) * metrics.pagerank_norm
        score += self.weights.get("degree", 0.25) * metrics.degree_norm
        
        # Bonus for structural criticality
        if metrics.is_articulation_point:
            score += 0.1
        if metrics.is_bridge_endpoint:
            score += 0.05
        
        return min(1.0, score)

    def _classify_by_metric(
        self,
        metrics: List[ComponentMetrics],
        metric_attr: str,
        metric_name: str,
    ) -> ClassificationResult:
        """Classify components by a specific metric"""
        items = [
            {
                "id": m.component_id,
                "type": m.component_type,
                "score": getattr(m, metric_attr),
            }
            for m in metrics
        ]
        return self.classifier.classify(items, metric_name=metric_name)

    def _generate_summary(
        self,
        metrics: List[ComponentMetrics],
        pr_class: ClassificationResult,
        bc_class: ClassificationResult,
        dc_class: ClassificationResult,
        comp_class: ClassificationResult,
    ) -> Dict[str, Any]:
        """Generate summary statistics"""
        pr_scores = [m.pagerank for m in metrics]
        bc_scores = [m.betweenness for m in metrics]
        dc_scores = [m.degree for m in metrics]
        comp_scores = [m.composite_score for m in metrics]
        
        return {
            "pagerank": {
                "critical_count": pr_class.critical_count,
                "max": max(pr_scores) if pr_scores else 0,
                "mean": sum(pr_scores) / len(pr_scores) if pr_scores else 0,
            },
            "betweenness": {
                "critical_count": bc_class.critical_count,
                "max": max(bc_scores) if bc_scores else 0,
                "mean": sum(bc_scores) / len(bc_scores) if bc_scores else 0,
            },
            "degree": {
                "critical_count": dc_class.critical_count,
                "max": max(dc_scores) if dc_scores else 0,
                "mean": sum(dc_scores) / len(dc_scores) if dc_scores else 0,
            },
            "composite": {
                "critical_count": comp_class.critical_count,
                "max": max(comp_scores) if comp_scores else 0,
                "mean": sum(comp_scores) / len(comp_scores) if comp_scores else 0,
            },
            "structural": {
                "articulation_points": sum(
                    1 for m in metrics if m.is_articulation_point
                ),
                "bridge_endpoints": sum(
                    1 for m in metrics if m.is_bridge_endpoint
                ),
            },
        }

    def _empty_result(
        self, 
        component_type: str, 
        timestamp: str
    ) -> ComponentTypeResult:
        """Create empty result for when no components found"""
        empty_classification = ClassificationResult(
            metric_name="empty",
            stats=BoxPlotStats.empty(),
            items=[],
            by_level={level: [] for level in CriticalityLevel},
            summary={level: 0 for level in CriticalityLevel},
        )
        
        return ComponentTypeResult(
            component_type=component_type,
            timestamp=timestamp,
            component_count=0,
            metrics=[],
            pagerank_classification=empty_classification,
            betweenness_classification=empty_classification,
            degree_classification=empty_classification,
            composite_classification=empty_classification,
            summary={},
        )

    def analyze_all_types(
        self,
        weighted: bool = True,
    ) -> Dict[str, ComponentTypeResult]:
        """
        Analyze all component types.
        
        Returns:
            Dict mapping component type -> ComponentTypeResult
        """
        results = {}
        
        for comp_type in ["Application", "Broker", "Node", "Topic"]:
            try:
                results[comp_type] = self.analyze(comp_type, weighted=weighted)
            except Exception as e:
                self.logger.warning(f"Failed to analyze {comp_type}: {e}")
        
        return results

    def compare_across_types(
        self,
        results: Dict[str, ComponentTypeResult],
    ) -> Dict[str, Any]:
        """
        Compare analysis results across component types.
        
        Args:
            results: Results from analyze_all_types()
        
        Returns:
            Comparison summary
        """
        comparison = {
            "by_type": {},
            "most_critical_type": None,
            "highest_betweenness_type": None,
            "highest_coupling_type": None,
        }
        
        max_critical = 0
        max_bc = 0.0
        max_dc = 0.0
        
        for comp_type, result in results.items():
            if result.component_count == 0:
                continue
            
            type_summary = {
                "count": result.component_count,
                "critical_count": result.composite_classification.critical_count,
                "critical_ratio": (
                    result.composite_classification.critical_count / 
                    result.component_count
                ),
                "avg_composite": result.summary.get("composite", {}).get("mean", 0),
                "max_betweenness": result.summary.get("betweenness", {}).get("max", 0),
                "max_degree": result.summary.get("degree", {}).get("max", 0),
            }
            
            comparison["by_type"][comp_type] = type_summary
            
            # Track highest
            if type_summary["critical_count"] > max_critical:
                max_critical = type_summary["critical_count"]
                comparison["most_critical_type"] = comp_type
            
            if type_summary["max_betweenness"] > max_bc:
                max_bc = type_summary["max_betweenness"]
                comparison["highest_betweenness_type"] = comp_type
            
            if type_summary["max_degree"] > max_dc:
                max_dc = type_summary["max_degree"]
                comparison["highest_coupling_type"] = comp_type
        
        return comparison
