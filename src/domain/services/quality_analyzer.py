"""
Quality Analyzer

Computes composite quality scores for Reliability (R), Maintainability (M),
and Availability (A) based on structural metrics.

Uses Box-Plot Classification for adaptive, data-driven threshold determination.

Formulas:
    R(v) = w_pr·PR + w_rpr·RPR + w_in·InDeg         (Reliability)
    M(v) = w_bt·BC + w_dg·Deg + w_cl·(1-CC)         (Maintainability)  
    A(v) = w_ap·AP + w_br·BridgeRatio + w_imp·Imp   (Availability)
    V(v) = w_ev·Eig + w_cl·Close + w_in·InDeg       (Vulnerability)
    Q(v) = w_r·R + w_m·M + w_a·A                    (Overall Quality)

Each dimension measures different aspects:
    - Reliability: Fault propagation risk (who is affected if this fails)
    - Maintainability: Coupling complexity (how hard to change)
    - Availability: Single point of failure risk (system partition risk)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.domain.services.classifier import BoxPlotClassifier
from src.domain.models.criticality import CriticalityLevel, BoxPlotStats
from src.domain.models.metrics import (
    QualityScores, QualityLevels, ComponentQuality, EdgeQuality,
    StructuralMetrics, EdgeMetrics, ClassificationSummary
)
from src.domain.services.structural_analyzer import StructuralAnalysisResult
from src.domain.config.layers import AnalysisLayer
from src.domain.services.weight_calculator import AHPProcessor, QualityWeights


@dataclass
class QualityAnalysisResult:
    """Complete quality analysis result for a layer."""
    timestamp: str
    layer: str
    context: str
    components: List[ComponentQuality]
    edges: List[EdgeQuality]
    classification_summary: ClassificationSummary
    weights: QualityWeights = field(default_factory=QualityWeights)
    stats: Dict[str, BoxPlotStats] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "layer": self.layer,
            "context": self.context,
            "components": [c.to_dict() for c in self.components],
            "edges": [e.to_dict() for e in self.edges],
            "classification_summary": self.classification_summary.to_dict(),
        }
    
    def get_critical_components(self) -> List[ComponentQuality]:
        """Get components classified as CRITICAL."""
        return [c for c in self.components if c.levels.overall == CriticalityLevel.CRITICAL]
    
    def get_high_priority(self) -> List[ComponentQuality]:
        """Get components classified as CRITICAL or HIGH."""
        return [c for c in self.components if c.levels.overall >= CriticalityLevel.HIGH]
    
    def get_by_type(self, comp_type: str) -> List[ComponentQuality]:
        """Get components of a specific type."""
        return [c for c in self.components if c.type == comp_type]
    
    def get_critical_edges(self) -> List[EdgeQuality]:
        """Get edges classified as CRITICAL."""
        return [e for e in self.edges if e.level == CriticalityLevel.CRITICAL]
    
    def get_requiring_attention(self) -> tuple[List[ComponentQuality], List[EdgeQuality]]:
        """Get components and edges requiring attention."""
        comps = [c for c in self.components if c.requires_attention]
        edges = [e for e in self.edges if e.level >= CriticalityLevel.HIGH]
        return comps, edges


class QualityAnalyzer:
    """
    Computes quality scores and classifications for components and edges.
    
    Uses configurable weights for the composite formulas and box-plot
    classification for adaptive threshold determination.
    
    Example:
        >>> analyzer = QualityAnalyzer(k_factor=1.5)
        >>> result = analyzer.analyze(structural_result, context="Application Layer")
        >>> critical = result.get_critical_components()
    """
    
    def __init__(
        self,
        k_factor: float = 1.5,
        weights: Optional[QualityWeights] = None,
        use_ahp: bool = False
    ):
        """
        Initialize the quality analyzer.
        
        Args:
            k_factor: Box-plot IQR multiplier for outlier detection (default: 1.5)
            weights: Custom weights for quality formulas (default: balanced)
        """
        self.classifier = BoxPlotClassifier(k_factor=k_factor)

        if use_ahp:
            processor = AHPProcessor()
            self.weights = processor.compute_weights()
        else:
            self.weights = weights or QualityWeights()
    
    def analyze(
        self,
        structural_result: StructuralAnalysisResult,
        context: Optional[str] = None
    ) -> QualityAnalysisResult:
        """
        Compute quality scores and classifications from structural analysis.
        
        Args:
            structural_result: Result from StructuralAnalyzer
            context: Optional context label (e.g., layer name)
            
        Returns:
            QualityAnalysisResult with scores, levels, and summary
        """
        context = context or structural_result.layer.value
        
        # Compute quality scores for components
        component_scores = self._compute_component_scores(structural_result.components)
        
        # Classify components using box-plot method
        components = self._classify_components(component_scores)
        
        # Compute and classify edge scores
        edges = self._analyze_edges(structural_result.edges, structural_result.components)
        
        # Build classification summary
        summary = self._build_summary(components, edges)
        
        return QualityAnalysisResult(
            timestamp=datetime.now().isoformat(),
            layer=structural_result.layer.value,
            context=context,
            components=components,
            edges=edges,
            classification_summary=summary,
            weights=self.weights,
        )
    
    def _compute_component_scores(
        self,
        metrics: Dict[str, StructuralMetrics]
    ) -> List[Dict[str, Any]]:
        """Compute R, M, A, Q scores for each component."""
        
        if not metrics:
            return []
        
        # Normalize metrics across all components for fair comparison
        all_metrics = list(metrics.values())
        normalizers = self._compute_normalizers(all_metrics)
        
        results = []
        for m in all_metrics:
            scores = self._compute_scores(m, normalizers)
            results.append({
                "data": m,
                "scores": scores,
            })
        
        return results
    
    def _compute_normalizers(
        self, 
        metrics: List[StructuralMetrics]
    ) -> Dict[str, float]:
        """Compute max values for normalization."""
        if not metrics:
            return {}
        
        return {
            "pagerank": max(m.pagerank for m in metrics) or 1.0,
            "reverse_pagerank": max(m.reverse_pagerank for m in metrics) or 1.0,
            "betweenness": max(m.betweenness for m in metrics) or 1.0,
            "degree": max(m.degree for m in metrics) or 1.0,
            "in_degree": max(m.in_degree for m in metrics) or 1.0,
            "clustering": 1.0,  # Already normalized [0, 1]
            "bridge_ratio": max(m.bridge_ratio for m in metrics) or 1.0,
            "eigenvector": max(m.eigenvector for m in metrics) or 1.0,
            "closeness": max(m.closeness for m in metrics) or 1.0,
        }
    
    def _compute_scores(
        self,
        m: StructuralMetrics,
        norm: Dict[str, float]
    ) -> QualityScores:
        """
        Compute R, M, A, Q scores for a single component.
        
        Formulas:
            R(v) = w_pr·PR_norm + w_rpr·RPR_norm + w_in·InDeg_norm
            M(v) = w_bt·BC_norm + w_dg·Deg_norm + w_cl·(1 - CC)
            A(v) = w_ap·AP + w_br·BR_norm + w_imp·(PR + RPR)/2
            V(v) = w_ev·Eig + w_cl·Close + w_in·InDeg
            Q(v) = w_r·R + w_m·M + w_a·A
        """
        w = self.weights
        
        # Normalize metrics
        pr = m.pagerank / norm["pagerank"] if norm["pagerank"] > 0 else 0
        rpr = m.reverse_pagerank / norm["reverse_pagerank"] if norm["reverse_pagerank"] > 0 else 0
        bt = m.betweenness / norm["betweenness"] if norm["betweenness"] > 0 else 0
        dg = m.degree / norm["degree"] if norm["degree"] > 0 else 0
        ind = m.in_degree / norm["in_degree"] if norm["in_degree"] > 0 else 0
        cc = m.clustering_coefficient
        br = m.bridge_ratio / norm["bridge_ratio"] if norm["bridge_ratio"] > 0 else 0
        ap = 1.0 if m.is_articulation_point else 0.0
        ev = m.eigenvector / norm["eigenvector"] if norm["eigenvector"] > 0 else 0
        cl = m.closeness / norm["closeness"] if norm["closeness"] > 0 else 0
        
        # Compute R: Reliability (fault propagation risk)
        reliability = (
            w.r_pagerank * pr +
            w.r_reverse_pagerank * rpr +
            w.r_in_degree * ind
        )
        
        # Compute M: Maintainability (coupling complexity)
        # Note: (1 - CC) because higher clustering = more redundancy = easier to maintain
        maintainability = (
            w.m_betweenness * bt +
            w.m_degree * dg +
            w.m_clustering * (1 - cc)
        )
        
        # Compute A: Availability (SPOF risk)
        importance = (pr + rpr) / 2
        availability = (
            w.a_articulation * ap +
            w.a_bridge_ratio * br +
            w.a_importance * importance
        )
        
        # Compute V: Vulnerability
        vulnerability = (
            w.v_eigenvector * ev +
            w.v_closeness * cl +
            w.v_in_degree * ind
        )
        
        # Compute Q: Overall Quality Score
        overall = (
            w.q_reliability * reliability +
            w.q_maintainability * maintainability +
            w.q_availability * availability +
            w.q_vulnerability * vulnerability
        )
        
        return QualityScores(
            reliability=reliability,
            maintainability=maintainability,
            availability=availability,
            vulnerability=vulnerability,
            overall=overall,
        )
    
    def _classify_components(
        self,
        component_scores: List[Dict[str, Any]]
    ) -> List[ComponentQuality]:
        """Classify components using box-plot method for each dimension."""
        
        if not component_scores:
            return []
        
        # Prepare data for classification by dimension
        def extract(dim: str):
            return [
                {"id": item["data"].id, "score": getattr(item["scores"], dim)}
                for item in component_scores
            ]
        
        # Classify each dimension
        classifications: Dict[str, Dict[str, CriticalityLevel]] = {}
        
        for dim in ["reliability", "maintainability", "availability", "vulnerability", "overall"]:
            result = self.classifier.classify(extract(dim), metric_name=dim)
            classifications[dim] = {item.id: item.level for item in result.items}
        
        # Build ComponentQuality objects
        results = []
        for item in component_scores:
            comp_id = item["data"].id
            
            levels = QualityLevels(
                reliability=classifications["reliability"].get(comp_id, CriticalityLevel.MINIMAL),
                maintainability=classifications["maintainability"].get(comp_id, CriticalityLevel.MINIMAL),
                availability=classifications["availability"].get(comp_id, CriticalityLevel.MINIMAL),
                vulnerability=classifications["vulnerability"].get(comp_id, CriticalityLevel.MINIMAL),
                overall=classifications["overall"].get(comp_id, CriticalityLevel.MINIMAL),
            )
            
            results.append(ComponentQuality(
                id=comp_id,
                type=item["data"].type,
                scores=item["scores"],
                levels=levels,
                structural=item["data"],
            ))
        
        # Sort by overall score (highest first)
        results.sort(key=lambda x: x.scores.overall, reverse=True)
        
        return results
    
    def _analyze_edges(
        self,
        edge_metrics: Dict[tuple, EdgeMetrics],
        component_metrics: Dict[str, StructuralMetrics]
    ) -> List[EdgeQuality]:
        """Compute and classify edge quality scores."""
        
        if not edge_metrics:
            return []
        
        # Compute edge scores using configurable weights
        w = self.weights
        edges = []
        max_betweenness = max(e.betweenness for e in edge_metrics.values()) or 1.0
        
        for key, em in edge_metrics.items():
            # Get connected component metrics for endpoint analysis
            source_metrics = component_metrics.get(em.source)
            target_metrics = component_metrics.get(em.target)
            
            # Endpoint importance (average PageRank of connected nodes)
            endpoint_importance = 0.0
            if source_metrics and target_metrics:
                endpoint_importance = (
                    source_metrics.pagerank + target_metrics.pagerank
                ) / 2
            
            # Edge vulnerability (exposure through connected nodes)
            # Average of endpoint eigenvector and closeness centrality
            endpoint_vulnerability = 0.0
            if source_metrics and target_metrics:
                avg_eigenvector = (source_metrics.eigenvector + target_metrics.eigenvector) / 2
                avg_closeness = (source_metrics.closeness + target_metrics.closeness) / 2
                endpoint_vulnerability = (avg_eigenvector + avg_closeness) / 2
            
            # Normalized edge metrics
            bridge_factor = 1.0 if em.is_bridge else 0.0
            betweenness_norm = em.betweenness / max_betweenness if max_betweenness > 0 else 0
            
            # Compute overall edge score using configurable weights
            overall = (
                w.e_betweenness * betweenness_norm +
                w.e_bridge * bridge_factor +
                w.e_endpoint * endpoint_importance +
                w.e_vulnerability * endpoint_vulnerability
            )
            
            edges.append(EdgeQuality(
                source=em.source,
                target=em.target,
                source_type=em.source_type,
                target_type=em.target_type,
                dependency_type=em.dependency_type,
                scores=QualityScores(
                    reliability=endpoint_importance,
                    maintainability=betweenness_norm,
                    availability=bridge_factor,
                    vulnerability=endpoint_vulnerability,
                    overall=overall,
                ),
                structural=em,
            ))
        
        # Classify edges
        if edges:
            data = [{"id": e.id, "score": e.scores.overall} for e in edges]
            result = self.classifier.classify(data, metric_name="edge_criticality")
            level_map = {item.id: item.level for item in result.items}
            for edge in edges:
                edge.level = level_map.get(edge.id, CriticalityLevel.MINIMAL)
        
        # Sort by score
        edges.sort(key=lambda x: x.scores.overall, reverse=True)
        
        return edges

    
    def _build_summary(
        self,
        components: List[ComponentQuality],
        edges: List[EdgeQuality]
    ) -> ClassificationSummary:
        """Build classification summary statistics."""
        
        comp_dist = {level.value: 0 for level in CriticalityLevel}
        edge_dist = {level.value: 0 for level in CriticalityLevel}
        
        for c in components:
            comp_dist[c.levels.overall.value] += 1
        
        for e in edges:
            edge_dist[e.level.value] += 1
        
        return ClassificationSummary(
            total_components=len(components),
            total_edges=len(edges),
            component_distribution=comp_dist,
            edge_distribution=edge_dist,
        )