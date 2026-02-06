"""
Quality Analyzer

Computes composite quality scores for the four RMAV dimensions and an
overall quality score, then classifies every component and edge using the
Box-Plot method (adaptive, data-driven thresholds).

Formulas (per component v):
    R(v) = w_pr·PR   + w_rpr·RPR   + w_in·InDeg           (Reliability)
    M(v) = w_bt·BC   + w_dg·Deg    + w_cl·(1 – CC)        (Maintainability)
    A(v) = w_ap·AP   + w_br·Bridge + w_imp·Importance      (Availability)
    V(v) = w_ev·Eig  + w_cl·Close  + w_in·InDeg            (Vulnerability)
    Q(v) = w_R·R(v)  + w_M·M(v)    + w_A·A(v) + w_V·V(v)  (Overall)

Classification (Box-Plot):
    CRITICAL : score > Q3 + k×IQR
    HIGH     : score > Q3
    MEDIUM   : score > Median
    LOW      : score > Q1
    MINIMAL  : score ≤ Q1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.domain.services.classifier import BoxPlotClassifier
from src.domain.models.criticality import CriticalityLevel, BoxPlotStats
from src.domain.models.metrics import (
    QualityScores,
    QualityLevels,
    ComponentQuality,
    EdgeQuality,
    StructuralMetrics,
    EdgeMetrics,
    ClassificationSummary,
)
from src.domain.services.structural_analyzer import StructuralAnalysisResult
from src.domain.config.layers import AnalysisLayer
from src.domain.services.weight_calculator import AHPProcessor, QualityWeights


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class QualityAnalysisResult:
    """Complete quality analysis result for a single layer."""

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

    # -- convenience queries -------------------------------------------------

    def get_critical_components(self) -> List[ComponentQuality]:
        return [c for c in self.components if c.levels.overall == CriticalityLevel.CRITICAL]

    def get_high_priority(self) -> List[ComponentQuality]:
        return [c for c in self.components if c.levels.overall >= CriticalityLevel.HIGH]

    def get_by_type(self, comp_type: str) -> List[ComponentQuality]:
        return [c for c in self.components if c.type == comp_type]

    def get_critical_edges(self) -> List[EdgeQuality]:
        return [e for e in self.edges if e.level == CriticalityLevel.CRITICAL]

    def get_requiring_attention(self) -> tuple[List[ComponentQuality], List[EdgeQuality]]:
        comps = [c for c in self.components if c.requires_attention]
        edges = [e for e in self.edges if e.level >= CriticalityLevel.HIGH]
        return comps, edges


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class QualityAnalyzer:
    """
    Computes quality scores and box-plot classifications for components / edges.

    Uses configurable weights (manual or AHP-derived) and the BoxPlotClassifier
    for adaptive threshold determination.
    """

    def __init__(
        self,
        k_factor: float = 1.5,
        weights: Optional[QualityWeights] = None,
        use_ahp: bool = False,
    ) -> None:
        self.classifier = BoxPlotClassifier(k_factor=k_factor)
        self.weights = (
            AHPProcessor().compute_weights() if use_ahp
            else (weights or QualityWeights())
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        structural_result: StructuralAnalysisResult,
        context: Optional[str] = None,
    ) -> QualityAnalysisResult:
        """
        Run quality analysis on structural results.

        Steps:
            1. Normalise raw metrics across components
            2. Compute R, M, A, V scores per component
            3. Classify components via box-plot on each dimension + overall
            4. Analyse and classify edges
            5. Build classification summary
        """
        layer_name = structural_result.layer.value
        ctx = context or f"{layer_name} layer analysis"

        # --- Component analysis -----------------------------------------
        raw_components = list(structural_result.components.values())
        norm = self._normalize(raw_components)
        components = self._score_and_classify_components(raw_components, norm)

        # --- Edge analysis ----------------------------------------------
        raw_edges = list(structural_result.edges.values())
        edges = self._score_and_classify_edges(raw_edges)

        # --- Summary ----------------------------------------------------
        summary = self._build_summary(components, edges)

        return QualityAnalysisResult(
            timestamp=datetime.now().isoformat(),
            layer=layer_name,
            context=ctx,
            components=components,
            edges=edges,
            classification_summary=summary,
            weights=self.weights,
        )

    # ------------------------------------------------------------------
    # Component scoring
    # ------------------------------------------------------------------

    def _score_and_classify_components(
        self,
        metrics_list: List[StructuralMetrics],
        norm: Dict[str, float],
    ) -> List[ComponentQuality]:
        """Score every component, then classify using box-plot per dimension."""
        if not metrics_list:
            return []

        # Compute raw RMAV scores
        scored: List[ComponentQuality] = []
        for m in metrics_list:
            scores = self._compute_rmav(m, norm)
            scored.append(ComponentQuality(
                id=m.id,
                type=m.type,
                scores=scores,
                levels=QualityLevels(),  # placeholder, filled below
                structural=m,
            ))

        # Classify each dimension with box-plot
        dim_keys = ["reliability", "maintainability", "availability", "vulnerability", "overall"]
        level_maps: Dict[str, Dict[str, CriticalityLevel]] = {}

        for dim in dim_keys:
            data = [{"id": c.id, "score": getattr(c.scores, dim)} for c in scored]
            result = self.classifier.classify(data, metric_name=dim)
            level_maps[dim] = {item.id: item.level for item in result.items}

        # Apply classified levels
        for c in scored:
            c.levels = QualityLevels(
                reliability=level_maps["reliability"].get(c.id, CriticalityLevel.MINIMAL),
                maintainability=level_maps["maintainability"].get(c.id, CriticalityLevel.MINIMAL),
                availability=level_maps["availability"].get(c.id, CriticalityLevel.MINIMAL),
                vulnerability=level_maps["vulnerability"].get(c.id, CriticalityLevel.MINIMAL),
                overall=level_maps["overall"].get(c.id, CriticalityLevel.MINIMAL),
            )

        # Sort by overall score descending
        scored.sort(key=lambda c: c.scores.overall, reverse=True)
        return scored

    def _compute_rmav(
        self, m: StructuralMetrics, norm: Dict[str, float],
    ) -> QualityScores:
        """Compute Reliability, Maintainability, Availability, Vulnerability scores."""
        w = self.weights

        def _n(val: float, key: str) -> float:
            """Normalise a value using the precomputed max."""
            mx = norm.get(key, 0.0)
            return val / mx if mx > 0 else 0.0

        # Normalised values
        pr   = _n(m.pagerank, "pagerank")
        rpr  = _n(m.reverse_pagerank, "reverse_pagerank")
        bt   = _n(m.betweenness, "betweenness")
        cl   = _n(m.closeness, "closeness")
        ev   = _n(m.eigenvector, "eigenvector")
        id_n = _n(m.in_degree_raw, "in_degree")
        dg   = _n(m.total_degree_raw, "total_degree")
        cc   = m.clustering_coefficient
        ap   = 1.0 if m.is_articulation_point else 0.0
        imp  = (pr + rpr) / 2.0  # importance proxy

        R = w.r_pagerank * pr + w.r_reverse_pagerank * rpr + w.r_in_degree * id_n
        M = w.m_betweenness * bt + w.m_degree * dg + w.m_clustering * (1.0 - cc)
        A = w.a_articulation * ap + w.a_bridge_ratio * m.bridge_ratio + w.a_importance * imp
        V = w.v_eigenvector * ev + w.v_closeness * cl + w.v_in_degree * id_n

        Q = w.q_reliability * R + w.q_maintainability * M + w.q_availability * A + w.q_vulnerability * V

        return QualityScores(
            reliability=R,
            maintainability=M,
            availability=A,
            vulnerability=V,
            overall=Q,
        )

    # ------------------------------------------------------------------
    # Edge scoring
    # ------------------------------------------------------------------

    def _score_and_classify_edges(
        self, edges_list: List[EdgeMetrics],
    ) -> List[EdgeQuality]:
        """Score and classify edges using box-plot."""
        if not edges_list:
            return []

        edges: List[EdgeQuality] = []
        for em in edges_list:
            bridge_factor = 1.0 if em.is_bridge else 0.0
            bt_norm = em.betweenness  # already normalised by NetworkX

            overall = 0.5 * bt_norm + 0.3 * bridge_factor + 0.2 * em.weight
            endpoint_importance = bt_norm
            endpoint_vulnerability = bt_norm * 0.7 + bridge_factor * 0.3

            edges.append(EdgeQuality(
                source=em.source,
                target=em.target,
                source_type=em.source_type,
                target_type=em.target_type,
                dependency_type=em.dependency_type,
                scores=QualityScores(
                    reliability=endpoint_importance,
                    maintainability=bt_norm,
                    availability=bridge_factor,
                    vulnerability=endpoint_vulnerability,
                    overall=overall,
                ),
                structural=em,
            ))

        # Classify
        if edges:
            data = [{"id": e.id, "score": e.scores.overall} for e in edges]
            result = self.classifier.classify(data, metric_name="edge_criticality")
            level_map = {item.id: item.level for item in result.items}
            for edge in edges:
                edge.level = level_map.get(edge.id, CriticalityLevel.MINIMAL)

        edges.sort(key=lambda e: e.scores.overall, reverse=True)
        return edges

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(components: List[StructuralMetrics]) -> Dict[str, float]:
        """Compute per-metric maximums for min-max normalisation."""
        if not components:
            return {}
        return {
            "pagerank": max((c.pagerank for c in components), default=0),
            "reverse_pagerank": max((c.reverse_pagerank for c in components), default=0),
            "betweenness": max((c.betweenness for c in components), default=0),
            "closeness": max((c.closeness for c in components), default=0),
            "eigenvector": max((c.eigenvector for c in components), default=0),
            "in_degree": max((c.in_degree_raw for c in components), default=0),
            "total_degree": max((c.total_degree_raw for c in components), default=0),
        }

    @staticmethod
    def _build_summary(
        components: List[ComponentQuality],
        edges: List[EdgeQuality],
    ) -> ClassificationSummary:
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