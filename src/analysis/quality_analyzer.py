"""
Quality Analyzer

Computes composite quality scores for Reliability (R), Maintainability (M),
and Availability (A) based on structural metrics.

Uses Box-Plot Classification for adaptive, data-driven threshold determination.

Formulas:
    R(v) = w_pr·PR + w_rpr·RPR + w_in·InDeg         (Reliability)
    M(v) = w_bt·BC + w_dg·Deg + w_cl·(1-CC)         (Maintainability)
    A(v) = w_ap·AP + w_br·BridgeRatio + w_imp·Imp   (Availability)
    Q(v) = w_r·R + w_m·M + w_a·A                    (Overall Quality)

Author: Software-as-a-Graph Research Project
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .structural_analyzer import StructuralAnalysisResult, StructuralMetrics, EdgeMetrics, AnalysisLayer
from .classifier import BoxPlotClassifier, CriticalityLevel, ClassificationResult, BoxPlotStats


@dataclass
class QualityScores:
    """Raw quality scores for R, M, A dimensions."""
    
    reliability: float = 0.0        # Fault propagation risk
    maintainability: float = 0.0    # Change/coupling complexity
    availability: float = 0.0       # Single point of failure risk
    overall: float = 0.0            # Combined criticality
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "reliability": round(self.reliability, 4),
            "maintainability": round(self.maintainability, 4),
            "availability": round(self.availability, 4),
            "overall": round(self.overall, 4),
        }


@dataclass
class QualityLevels:
    """Classified levels for each quality dimension."""
    
    reliability: CriticalityLevel = CriticalityLevel.MINIMAL
    maintainability: CriticalityLevel = CriticalityLevel.MINIMAL
    availability: CriticalityLevel = CriticalityLevel.MINIMAL
    overall: CriticalityLevel = CriticalityLevel.MINIMAL
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "reliability": self.reliability.value,
            "maintainability": self.maintainability.value,
            "availability": self.availability.value,
            "overall": self.overall.value,
        }
    
    def max_level(self) -> CriticalityLevel:
        """Return the highest criticality level across all dimensions."""
        return max([self.reliability, self.maintainability, self.availability], 
                   key=lambda x: x.numeric)


@dataclass
class ComponentQuality:
    """Complete quality assessment for a single component."""
    
    id: str
    type: str
    scores: QualityScores
    levels: QualityLevels
    structural: StructuralMetrics  # Original structural metrics for reference
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "scores": self.scores.to_dict(),
            "levels": self.levels.to_dict(),
            "structural_metrics": {
                "pagerank": round(self.structural.pagerank, 4),
                "betweenness": round(self.structural.betweenness, 4),
                "is_articulation_point": self.structural.is_articulation_point,
                "in_degree_raw": self.structural.in_degree_raw,
                "out_degree_raw": self.structural.out_degree_raw,
            }
        }
    
    @property
    def is_critical(self) -> bool:
        """Check if component is at CRITICAL or HIGH level."""
        return self.levels.overall >= CriticalityLevel.HIGH


@dataclass
class EdgeQuality:
    """Quality assessment for a single edge (dependency)."""
    
    source: str
    target: str
    type: str
    scores: QualityScores
    level: CriticalityLevel
    is_bridge: bool = False
    weight: float = 1.0
    
    @property
    def id(self) -> str:
        return f"{self.source}->{self.target}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "scores": self.scores.to_dict(),
            "level": self.level.value,
            "is_bridge": self.is_bridge,
            "weight": round(self.weight, 4),
        }


@dataclass
class ClassificationSummary:
    """Summary of classification results."""
    
    component_distribution: Dict[str, int]
    edge_distribution: Dict[str, int]
    critical_components: int
    high_components: int
    critical_edges: int
    statistics: Dict[str, BoxPlotStats]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_distribution": self.component_distribution,
            "edge_distribution": self.edge_distribution,
            "critical_components": self.critical_components,
            "high_components": self.high_components,
            "critical_edges": self.critical_edges,
            "statistics": {k: v.to_dict() for k, v in self.statistics.items()},
        }


@dataclass
class QualityAnalysisResult:
    """Complete result of quality analysis."""
    
    timestamp: str
    layer: str
    context: str
    components: List[ComponentQuality]
    edges: List[EdgeQuality]
    classification_summary: ClassificationSummary
    
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


class QualityAnalyzer:
    """
    Computes quality scores and classifications for components and edges.
    
    Uses configurable weights for the composite formulas and box-plot
    classification for adaptive threshold determination.
    """
    
    # Default weight configurations - aligned with quality-formulations.md Section 10
    # All weights within each formula sum to 1.0
    DEFAULT_RELIABILITY_WEIGHTS = {
        "pagerank": 0.45,           # ω_PR: Transitive influence
        "reverse_pagerank": 0.35,   # ω_RP: Failure propagation potential
        "in_degree": 0.20,          # ω_ID: Direct dependency count
    }
    
    DEFAULT_MAINTAINABILITY_WEIGHTS = {
        "betweenness": 0.45,        # ω_BT: Coupling/bottleneck indicator
        "clustering": 0.25,         # ω_CC: Modularity (inverted: 1-CC)
        "degree": 0.30,             # ω_DC: Interface complexity
    }
    
    DEFAULT_AVAILABILITY_WEIGHTS = {
        "articulation": 0.50,       # ω_AP: Cut-vertex status (SPOF)
        "bridge_ratio": 0.25,       # ω_BR: Bridge edge ratio
        "criticality": 0.25,        # ω_CR: Combined criticality factor
    }
    
    DEFAULT_OVERALL_WEIGHTS = {
        "reliability": 0.35,        # α: Equal weight for runtime stability
        "maintainability": 0.30,    # β: Development effort concern
        "availability": 0.35,       # γ: Equal weight for runtime stability
    }
    
    DEFAULT_EDGE_WEIGHTS = {
        "reliability": 0.50,        # ω_R: Edge reliability contribution
        "availability": 0.50,       # ω_A: Edge availability contribution
    }
    
    # Edge component weights (per quality-formulations.md Section 3)
    DEFAULT_EDGE_RELIABILITY_WEIGHTS = {
        "weight": 0.40,             # ω_w: Edge weight factor
        "endpoint_avg": 0.60,       # ω_ep: Endpoint average factor
    }
    
    DEFAULT_EDGE_AVAILABILITY_WEIGHTS = {
        "bridge": 0.60,             # ω_br: Bridge indicator weight
        "endpoint_avg": 0.40,       # ω_ep: Endpoint average factor
    }
    
    def __init__(
        self,
        k_factor: float = 1.5,
        reliability_weights: Optional[Dict[str, float]] = None,
        maintainability_weights: Optional[Dict[str, float]] = None,
        availability_weights: Optional[Dict[str, float]] = None,
        overall_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the quality analyzer.
        
        Args:
            k_factor: Box-plot IQR multiplier for outlier detection
            reliability_weights: Custom weights for R score
            maintainability_weights: Custom weights for M score
            availability_weights: Custom weights for A score
            overall_weights: Custom weights for Q score
        """
        self.classifier = BoxPlotClassifier(k_factor=k_factor)
        
        self.W_R = reliability_weights or self.DEFAULT_RELIABILITY_WEIGHTS.copy()
        self.W_M = maintainability_weights or self.DEFAULT_MAINTAINABILITY_WEIGHTS.copy()
        self.W_A = availability_weights or self.DEFAULT_AVAILABILITY_WEIGHTS.copy()
        self.W_Q = overall_weights or self.DEFAULT_OVERALL_WEIGHTS.copy()
        self.W_E = self.DEFAULT_EDGE_WEIGHTS.copy()
    
    def analyze(
        self, 
        structural_result: StructuralAnalysisResult,
        context: str = "System"
    ) -> QualityAnalysisResult:
        """
        Compute quality scores and classifications.
        
        Args:
            structural_result: Result from StructuralAnalyzer
            context: Analysis context label
            
        Returns:
            QualityAnalysisResult with scores, levels, and summaries
        """
        # 1. Normalize structural metrics
        normalized_metrics = self._normalize_metrics(structural_result.components)
        
        # 2. Compute component quality scores
        component_scores = []
        for comp_id, struct in structural_result.components.items():
            norm = normalized_metrics.get(comp_id, {})
            scores = self._compute_component_scores(struct, norm)
            component_scores.append({
                "data": struct,
                "scores": scores,
            })
        
        # 3. Classify components using box-plot method
        component_results = self._classify_components(component_scores)
        
        # 4. Compute and classify edge scores
        edge_results = []
        if structural_result.edges:
            comp_score_lookup = {c.id: c.scores for c in component_results}
            normalized_edges = self._normalize_edges(structural_result.edges)
            
            for edge_key, edge in structural_result.edges.items():
                if edge.source in comp_score_lookup and edge.target in comp_score_lookup:
                    scores = self._compute_edge_scores(
                        edge,
                        normalized_edges.get(edge_key, {}),
                        comp_score_lookup[edge.source],
                        comp_score_lookup[edge.target]
                    )
                    edge_results.append(EdgeQuality(
                        source=edge.source,
                        target=edge.target,
                        type=edge.dependency_type,
                        scores=scores,
                        level=CriticalityLevel.MINIMAL,
                        is_bridge=edge.is_bridge,
                        weight=edge.weight,
                    ))
            
            # Classify edges
            self._classify_edges(edge_results)
        
        # 5. Build classification summary
        summary = self._build_summary(component_results, edge_results)
        
        return QualityAnalysisResult(
            timestamp=datetime.now().isoformat(),
            layer=structural_result.layer.value,
            context=context,
            components=component_results,
            edges=edge_results,
            classification_summary=summary,
        )
    
    def _normalize_metrics(
        self, 
        components: Dict[str, StructuralMetrics]
    ) -> Dict[str, Dict[str, float]]:
        """
        Normalize structural metrics to [0, 1] range using min-max scaling.
        """
        if not components:
            return {}
        
        # Metrics to normalize
        metric_names = [
            "pagerank", "reverse_pagerank", "betweenness", "closeness",
            "degree", "in_degree", "out_degree", "clustering_coefficient",
            "bridge_ratio", "weight", "eigenvector"
        ]
        
        # Collect values for each metric
        metric_values: Dict[str, List[float]] = {m: [] for m in metric_names}
        for comp in components.values():
            for m in metric_names:
                metric_values[m].append(getattr(comp, m, 0.0))
        
        # Compute min/max for each metric
        metric_ranges: Dict[str, Tuple[float, float]] = {}
        for m, values in metric_values.items():
            min_v = min(values) if values else 0.0
            max_v = max(values) if values else 0.0
            metric_ranges[m] = (min_v, max_v)
        
        # Normalize each component
        normalized = {}
        for comp_id, comp in components.items():
            norm = {}
            for m in metric_names:
                val = getattr(comp, m, 0.0)
                min_v, max_v = metric_ranges[m]
                if max_v > min_v:
                    norm[m] = (val - min_v) / (max_v - min_v)
                else:
                    norm[m] = 0.5  # All values equal
            normalized[comp_id] = norm
        
        return normalized
    
    def _normalize_edges(
        self, 
        edges: Dict[Tuple[str, str], EdgeMetrics]
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """Normalize edge metrics to [0, 1] range."""
        if not edges:
            return {}
        
        # Collect values
        weights = [e.weight for e in edges.values()]
        betweenness = [e.betweenness for e in edges.values()]
        
        min_w, max_w = min(weights), max(weights)
        min_b, max_b = min(betweenness), max(betweenness)
        
        normalized = {}
        for key, edge in edges.items():
            norm_w = (edge.weight - min_w) / (max_w - min_w) if max_w > min_w else 0.5
            norm_b = (edge.betweenness - min_b) / (max_b - min_b) if max_b > min_b else 0.5
            normalized[key] = {"weight": norm_w, "betweenness": norm_b}
        
        return normalized
    
    def _compute_component_scores(
        self, 
        raw: StructuralMetrics, 
        norm: Dict[str, float]
    ) -> QualityScores:
        """
        Compute R, M, A, Q scores for a component.
        """
        get_n = lambda k: norm.get(k, 0.0)
        
        # Formula (1): Reliability Score
        # R(v) = w_pr·PR + w_rpr·RPR + w_in·InDeg
        r = (
            self.W_R["pagerank"] * get_n("pagerank") +
            self.W_R["reverse_pagerank"] * get_n("reverse_pagerank") +
            self.W_R["in_degree"] * get_n("in_degree")
        )
        
        # Formula (2): Maintainability Score
        # M(v) = w_bt·BC + w_dg·Deg + w_cl·(1-CC)
        # Note: (1-CC) because high clustering = good modularity = lower concern
        m = (
            self.W_M["betweenness"] * get_n("betweenness") +
            self.W_M["degree"] * get_n("degree") +
            self.W_M["clustering"] * (1.0 - get_n("clustering_coefficient"))
        )
        
        # Formula (3): Availability Score
        # A(v) = ω_AP·AP + ω_BR·BridgeRatio + ω_CR·Criticality
        # Criticality = combined importance factor (PR × Deg)
        is_ap = 1.0 if raw.is_articulation_point else 0.0
        criticality = get_n("pagerank") * get_n("degree")  # Combined criticality factor
        
        a = (
            self.W_A["articulation"] * is_ap +
            self.W_A["bridge_ratio"] * raw.bridge_ratio +
            self.W_A["criticality"] * criticality
        )
        
        # Formula (4): Overall Quality Criticality
        # Q(v) = w_r·R + w_m·M + w_a·A
        q = (
            self.W_Q["reliability"] * r +
            self.W_Q["maintainability"] * m +
            self.W_Q["availability"] * a
        )
        
        return QualityScores(
            reliability=min(r, 1.0),
            maintainability=min(m, 1.0),
            availability=min(a, 1.0),
            overall=min(q, 1.0),
        )
    
    def _compute_edge_scores(
        self,
        raw: EdgeMetrics,
        norm: Dict[str, float],
        src_scores: QualityScores,
        tgt_scores: QualityScores
    ) -> QualityScores:
        """
        Compute quality scores for an edge.
        
        Formula (5): Edge Reliability Score
        R_e(e) = ω_w·W_norm(e) + ω_ep·(R(u) + R(v))/2
        
        Formula (6): Edge Availability Score
        A_e(e) = ω_br·Bridge(e) + ω_ep·(A(u) + A(v))/2
        
        Formula (7): Overall Edge Quality
        Q_e(e) = ω_R·R_e + ω_A·A_e
        """
        W_ER = self.DEFAULT_EDGE_RELIABILITY_WEIGHTS
        W_EA = self.DEFAULT_EDGE_AVAILABILITY_WEIGHTS
        
        # Formula (5): Edge Reliability
        endpoint_r = (src_scores.reliability + tgt_scores.reliability) / 2
        r_e = W_ER["weight"] * norm.get("weight", 0.5) + W_ER["endpoint_avg"] * endpoint_r
        
        # Formula (6): Edge Availability
        endpoint_a = (src_scores.availability + tgt_scores.availability) / 2
        is_bridge = 1.0 if raw.is_bridge else 0.0
        a_e = W_EA["bridge"] * is_bridge + W_EA["endpoint_avg"] * endpoint_a
        
        # Formula (7): Overall Edge Criticality
        q_e = self.W_E["reliability"] * r_e + self.W_E["availability"] * a_e
        
        return QualityScores(
            reliability=min(r_e, 1.0),
            maintainability=0.0,  # Not applicable to edges (node-centric property)
            availability=min(a_e, 1.0),
            overall=min(q_e, 1.0),
        )
    
    def _classify_components(
        self, 
        items: List[Dict[str, Any]]
    ) -> List[ComponentQuality]:
        """
        Classify components across all quality dimensions using box-plot method.
        """
        if not items:
            return []
        
        # Prepare data for classification
        extract = lambda dim: [
            {"id": i["data"].id, "score": getattr(i["scores"], dim)} 
            for i in items
        ]
        
        # Classify each dimension
        classifications: Dict[str, Dict[str, CriticalityLevel]] = {}
        stats: Dict[str, BoxPlotStats] = {}
        
        for dim in ["reliability", "maintainability", "availability", "overall"]:
            dataset = extract(dim)
            result = self.classifier.classify(dataset, metric_name=dim)
            classifications[dim] = {item.id: item.level for item in result.items}
            stats[dim] = result.stats
        
        # Build ComponentQuality objects
        results = []
        for item in items:
            comp_id = item["data"].id
            
            levels = QualityLevels(
                reliability=classifications["reliability"].get(comp_id, CriticalityLevel.MINIMAL),
                maintainability=classifications["maintainability"].get(comp_id, CriticalityLevel.MINIMAL),
                availability=classifications["availability"].get(comp_id, CriticalityLevel.MINIMAL),
                overall=classifications["overall"].get(comp_id, CriticalityLevel.MINIMAL),
            )
            
            results.append(ComponentQuality(
                id=comp_id,
                type=item["data"].type,
                scores=item["scores"],
                levels=levels,
                structural=item["data"],
            ))
        
        # Sort by overall score descending
        results.sort(key=lambda x: x.scores.overall, reverse=True)
        
        return results
    
    def _classify_edges(self, edges: List[EdgeQuality]) -> None:
        """
        Classify edges based on overall score using box-plot method.
        Modifies edges in place.
        """
        if not edges:
            return
        
        # Prepare data
        data = [{"id": e.id, "score": e.scores.overall} for e in edges]
        result = self.classifier.classify(data, metric_name="edge_criticality")
        
        # Update edge levels
        level_map = {item.id: item.level for item in result.items}
        for edge in edges:
            edge.level = level_map.get(edge.id, CriticalityLevel.MINIMAL)
    
    def _build_summary(
        self,
        components: List[ComponentQuality],
        edges: List[EdgeQuality]
    ) -> ClassificationSummary:
        """Build classification summary statistics."""
        
        # Component distribution
        comp_dist = {level.value: 0 for level in CriticalityLevel}
        for c in components:
            comp_dist[c.levels.overall.value] += 1
        
        # Edge distribution
        edge_dist = {level.value: 0 for level in CriticalityLevel}
        for e in edges:
            edge_dist[e.level.value] += 1
        
        # Counts
        critical_comps = sum(1 for c in components if c.levels.overall == CriticalityLevel.CRITICAL)
        high_comps = sum(1 for c in components if c.levels.overall == CriticalityLevel.HIGH)
        critical_edges = sum(1 for e in edges if e.level >= CriticalityLevel.HIGH)
        
        return ClassificationSummary(
            component_distribution=comp_dist,
            edge_distribution=edge_dist,
            critical_components=critical_comps,
            high_components=high_comps,
            critical_edges=critical_edges,
            statistics={},  # Could add per-dimension stats if needed
        )