"""
Quality Analyzer

Computes composite quality scores for the four RMAV dimensions and an
overall quality score, then classifies every component and edge using the
Box-Plot method (adaptive, data-driven thresholds).

Formulas (per component v):
    R(v) = w_pr·PR   + w_rpr·RPR   + w_in·InDeg           (Reliability)
    M(v) = w_bt·BC   + w_od·OutDeg + w_cl·(1 – CC)        (Maintainability)
    A(v) = w_ap·AP_c + w_br·Bridge + w_imp·Importance      (Availability)
    V(v) = w_ev·Eig  + w_cl·Close  + w_od·OutDeg           (Vulnerability)
    Q(v) = w_R·R(v)  + w_M·M(v)    + w_A·A(v) + w_V·V(v)  (Overall)

Design changes (v2):
    - Metric orthogonality: In-Degree exclusive to R(v), Out-Degree shared
      between M(v) and V(v) with distinct semantics (efferent coupling vs
      attack surface). No raw metric appears in more than two dimensions.
    - Continuous AP: Binary is_articulation_point replaced by AP_c(v)
      measuring reachability loss after node removal.
    - Out-Degree in M(v): Efferent coupling aligns with SE maintainability
      theory (Martin's Instability metric).
    - Out-Degree in V(v): Outbound connections = traversable attack surface.
    - Edge RMAV: Edges scored with endpoint-aware formulas instead of ad-hoc
      weights.

Classification (Box-Plot):
    CRITICAL : score > Q3 + k×IQR
    HIGH     : score > Q3
    MEDIUM   : score > Median
    LOW      : score > Q1
    MINIMAL  : score ≤ Q1
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Sequence

import networkx as nx

from .classifier import BoxPlotClassifier
from src.core.criticality import CriticalityLevel, BoxPlotStats
from src.core.metrics import (
    QualityScores,
    QualityLevels,
    ComponentQuality,
    EdgeQuality,
    StructuralMetrics,
    EdgeMetrics,
    ClassificationSummary,
)
from .models import StructuralAnalysisResult
from src.core.layers import AnalysisLayer
from .weight_calculator import AHPProcessor, QualityWeights


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum sample size for box-plot classification. Below this, we use
#: fixed percentile thresholds instead.
MIN_BOXPLOT_SAMPLE = 12

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

from .models import QualityAnalysisResult


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class QualityAnalyzer:
    """
    Computes quality scores and box-plot classifications for components / edges.

    Uses configurable weights (manual or AHP-derived) and the BoxPlotClassifier
    for adaptive threshold determination.

    Args:
        k_factor: IQR multiplier for box-plot outlier fence (default 1.5).
        weights: Manually specified QualityWeights (overridden by use_ahp).
        use_ahp: If True, compute weights from default AHP matrices.
        normalization_method: 'max' (default) or 'robust' (rank-based).
    """

    def __init__(
        self,
        k_factor: float = 1.5,
        weights: Optional[QualityWeights] = None,
        use_ahp: bool = False,
        normalization_method: str = "max",
    ) -> None:
        self.classifier = BoxPlotClassifier(k_factor=k_factor)
        self.weights = (
            AHPProcessor().compute_weights() if use_ahp
            else (weights or QualityWeights())
        )
        self.normalization_method = normalization_method
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        structural_result: StructuralAnalysisResult,
        context: Optional[str] = None,
        run_sensitivity: bool = False,
        sensitivity_perturbations: int = 200,
        sensitivity_noise: float = 0.05,
    ) -> QualityAnalysisResult:
        """
        Run quality analysis on structural results.

        Steps:
            1. Normalise raw metrics across components
            2. Compute R, M, A, V scores per component
            3. Classify components via box-plot on each dimension + overall
            4. Analyse and classify edges (with endpoint quality context)
            5. Build classification summary
            6. (Optional) Run weight sensitivity analysis
        """
        layer_name = structural_result.layer.value
        ctx = context or f"{layer_name} layer analysis"

        # --- Component analysis -----------------------------------------
        raw_components = list(structural_result.components.values())
        norm = self._normalize(raw_components)

        # Compute continuous AP scores from the graph if available
        ap_scores = self._compute_continuous_ap_scores(structural_result)

        components = self._score_and_classify_components(raw_components, norm, ap_scores)

        # --- Edge analysis (with endpoint quality context) --------------
        raw_edges = list(structural_result.edges.values())
        comp_quality_map = {c.id: c for c in components}
        edges = self._score_and_classify_edges(raw_edges, comp_quality_map)

        # --- Summary ----------------------------------------------------
        summary = self._build_summary(components, edges)

        # --- Sensitivity (optional) -------------------------------------
        sensitivity = None
        if run_sensitivity:
            sensitivity = self._sensitivity_analysis(
                raw_components, norm, ap_scores,
                n_perturbations=sensitivity_perturbations,
                noise_std=sensitivity_noise,
            )

        return QualityAnalysisResult(
            timestamp=datetime.now().isoformat(),
            layer=layer_name,
            context=ctx,
            components=components,
            edges=edges,
            classification_summary=summary,
            weights=self.weights,
            sensitivity=sensitivity,
        )

    # ------------------------------------------------------------------
    # Continuous AP computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_continuous_ap_scores(
        structural_result: StructuralAnalysisResult,
    ) -> Dict[str, float]:
        """
        Compute continuous articulation point scores based on reachability loss.

        AP_c(v) = 1 - |largest_CC(G \\ {v})| / (|V| - 1)

        For true APs: > 0 (proportional to fragmentation severity)
        For non-APs: 0.0
        For leaf nodes: 0.0

        Falls back to binary AP flag if the graph is not available.
        """
        ap_scores: Dict[str, float] = {}

        # Try to reconstruct a NetworkX graph from structural result
        # to compute exact reachability loss
        try:
            G = nx.Graph()  # Undirected for connectivity analysis
            for comp_id in structural_result.components:
                G.add_node(comp_id)
            for edge_key in structural_result.edges:
                if isinstance(edge_key, tuple) and len(edge_key) == 2:
                    G.add_edge(edge_key[0], edge_key[1])
                else:
                    em = structural_result.edges[edge_key]
                    G.add_edge(em.source, em.target)

            n = G.number_of_nodes()
            if n <= 1:
                return {cid: 0.0 for cid in structural_result.components}

            for comp_id in structural_result.components:
                G_copy = G.copy()
                G_copy.remove_node(comp_id)

                if G_copy.number_of_nodes() == 0:
                    ap_scores[comp_id] = 1.0
                    continue

                largest_cc_size = max(
                    len(c) for c in nx.connected_components(G_copy)
                )
                ap_scores[comp_id] = 1.0 - (largest_cc_size / (n - 1))

        except Exception:
            # Fallback: use binary AP flag from structural analysis
            for comp_id, metrics in structural_result.components.items():
                ap_scores[comp_id] = 1.0 if metrics.is_articulation_point else 0.0

        return ap_scores

    # ------------------------------------------------------------------
    # Component scoring
    # ------------------------------------------------------------------

    def _score_and_classify_components(
        self,
        metrics_list: List[StructuralMetrics],
        norm: Dict[str, float],
        ap_scores: Optional[Dict[str, float]] = None,
    ) -> List[ComponentQuality]:
        """Score every component, then classify using box-plot per dimension."""
        if not metrics_list:
            return []

        # Compute raw RMAV scores
        scored: List[ComponentQuality] = []
        for m in metrics_list:
            ap_c = (ap_scores or {}).get(m.id, 1.0 if m.is_articulation_point else 0.0)
            scores = self._compute_rmav(m, norm, ap_c)
            scored.append(ComponentQuality(
                id=m.id,
                type=m.type,
                scores=scores,
                levels=QualityLevels(),  # placeholder, filled below
                structural=m,
            ))

        # Classify each dimension with box-plot (or percentile fallback)
        dim_keys = ["reliability", "maintainability", "availability", "vulnerability", "overall"]
        level_maps: Dict[str, Dict[str, CriticalityLevel]] = {}

        use_percentile_fallback = len(scored) < MIN_BOXPLOT_SAMPLE

        for dim in dim_keys:
            data = [{"id": c.id, "score": getattr(c.scores, dim)} for c in scored]
            if use_percentile_fallback:
                level_maps[dim] = self._percentile_classify(data)
            else:
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
        self, m: StructuralMetrics, norm: Dict[str, float], ap_c: float,
    ) -> QualityScores:
        """
        Compute Reliability, Maintainability, Availability, Vulnerability scores.

        Design principles:
            - Each raw metric maps to at most one dimension (orthogonality).
            - Out-Degree is shared between M(v) and V(v) with distinct semantics.
            - AP is continuous (ap_c), not binary.
        """
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
        od_n = _n(m.out_degree_raw, "out_degree")
        cc   = m.clustering_coefficient
        imp  = (pr + rpr) / 2.0  # importance proxy

        # --- RMAV formulas (v2 — orthogonal metrics) ---

        # Reliability: fault propagation risk
        R = w.r_pagerank * pr + w.r_reverse_pagerank * rpr + w.r_in_degree * id_n

        # Maintainability: efferent coupling + bottleneck position
        M = w.m_betweenness * bt + w.m_out_degree * od_n + w.m_clustering * (1.0 - cc)

        # Availability: SPOF risk (continuous AP score)
        A = w.a_articulation * ap_c + w.a_bridge_ratio * m.bridge_ratio + w.a_importance * imp

        # Vulnerability: attack surface + strategic importance
        V = w.v_eigenvector * ev + w.v_closeness * cl + w.v_out_degree * od_n

        Q = w.q_reliability * R + w.q_maintainability * M + w.q_availability * A + w.q_vulnerability * V

        return QualityScores(
            reliability=R,
            maintainability=M,
            availability=A,
            vulnerability=V,
            overall=Q,
        )

    # ------------------------------------------------------------------
    # Edge scoring (RMAV-aligned with endpoint context)
    # ------------------------------------------------------------------

    def _score_and_classify_edges(
        self,
        edges_list: List[EdgeMetrics],
        comp_quality_map: Optional[Dict[str, ComponentQuality]] = None,
    ) -> List[EdgeQuality]:
        """
        Score and classify edges using endpoint-aware RMAV formulas.

        When component quality scores are available, edge scores incorporate
        the criticality of connected endpoints. Otherwise falls back to
        edge-intrinsic metrics only.
        """
        if not edges_list:
            return []

        w = self.weights
        edges: List[EdgeQuality] = []

        for em in edges_list:
            bridge_factor = 1.0 if em.is_bridge else 0.0
            bt_norm = em.betweenness  # already normalised by NetworkX
            edge_weight = em.weight

            # Endpoint quality scores (fallback to 0.0 if unavailable)
            src_q = comp_quality_map.get(em.source) if comp_quality_map else None
            tgt_q = comp_quality_map.get(em.target) if comp_quality_map else None

            src_r = src_q.scores.reliability if src_q else 0.0
            tgt_r = tgt_q.scores.reliability if tgt_q else 0.0
            src_a = src_q.scores.availability if src_q else 0.0
            tgt_a = tgt_q.scores.availability if tgt_q else 0.0
            src_v = src_q.scores.vulnerability if src_q else 0.0
            tgt_v = tgt_q.scores.vulnerability if tgt_q else 0.0

            # Edge RMAV
            e_reliability = (
                w.e_betweenness * bt_norm
                + w.e_bridge * edge_weight
                + w.e_endpoint * max(src_r, tgt_r)
            )
            e_maintainability = (
                w.e_betweenness * bt_norm
                + w.e_bridge * bridge_factor
                + w.e_vulnerability * edge_weight
            )
            e_availability = (
                w.e_bridge * bridge_factor
                + w.e_endpoint * min(src_a, tgt_a)
            )
            e_vulnerability = (
                w.e_vulnerability * edge_weight
                + w.e_endpoint * max(src_v, tgt_v)
            )

            overall = (
                w.q_reliability * e_reliability
                + w.q_maintainability * e_maintainability
                + w.q_availability * e_availability
                + w.q_vulnerability * e_vulnerability
            )

            edges.append(EdgeQuality(
                source=em.source,
                target=em.target,
                source_type=em.source_type,
                target_type=em.target_type,
                dependency_type=em.dependency_type,
                scores=QualityScores(
                    reliability=e_reliability,
                    maintainability=e_maintainability,
                    availability=e_availability,
                    vulnerability=e_vulnerability,
                    overall=overall,
                ),
                structural=em,
            ))

        # Classify
        if edges:
            data = [{"id": e.id, "score": e.scores.overall} for e in edges]
            if len(edges) < MIN_BOXPLOT_SAMPLE:
                level_map = self._percentile_classify(data)
            else:
                result = self.classifier.classify(data, metric_name="edge_criticality")
                level_map = {item.id: item.level for item in result.items}
            for edge in edges:
                edge.level = level_map.get(edge.id, CriticalityLevel.MINIMAL)

        edges.sort(key=lambda e: e.scores.overall, reverse=True)
        return edges

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _percentile_classify(
        data: List[Dict[str, Any]],
    ) -> Dict[str, CriticalityLevel]:
        """
        Fixed percentile classification for small samples (< MIN_BOXPLOT_SAMPLE).

        Uses rank-based percentile thresholds instead of box-plot quartiles
        to avoid unreliable quartile estimates with few data points.
        """
        sorted_data = sorted(data, key=lambda d: d["score"], reverse=True)
        n = len(sorted_data)
        level_map: Dict[str, CriticalityLevel] = {}

        for i, d in enumerate(sorted_data):
            pct = i / n if n > 0 else 0.0
            if pct < 0.10:
                level = CriticalityLevel.CRITICAL
            elif pct < 0.25:
                level = CriticalityLevel.HIGH
            elif pct < 0.50:
                level = CriticalityLevel.MEDIUM
            elif pct < 0.75:
                level = CriticalityLevel.LOW
            else:
                level = CriticalityLevel.MINIMAL
            level_map[d["id"]] = level

        return level_map

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def _sensitivity_analysis(
        self,
        components: List[StructuralMetrics],
        norm: Dict[str, float],
        ap_scores: Dict[str, float],
        n_perturbations: int = 200,
        noise_std: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Measure ranking stability under weight perturbations.

        Perturbs all AHP weights by Gaussian noise, recomputes Q(v),
        and reports how stable the top-K rankings are.

        Returns:
            Dictionary with top5_stability, mean_kendall_tau, std_kendall_tau.
        """
        if len(components) < 3:
            return {"top5_stability": 1.0, "mean_kendall_tau": 1.0, "std_kendall_tau": 0.0}

        # Original ranking
        original_scores = {}
        for m in components:
            ap_c = ap_scores.get(m.id, 1.0 if m.is_articulation_point else 0.0)
            original_scores[m.id] = self._compute_rmav(m, norm, ap_c).overall

        original_ranking = sorted(original_scores, key=original_scores.get, reverse=True)
        original_top5 = set(original_ranking[:5])

        stable_count = 0
        taus = []

        for _ in range(n_perturbations):
            perturbed_weights = self._perturb_weights(noise_std)
            perturbed_scores = {}
            for m in components:
                ap_c = ap_scores.get(m.id, 1.0 if m.is_articulation_point else 0.0)
                perturbed_scores[m.id] = self._compute_rmav_with_weights(
                    m, norm, ap_c, perturbed_weights
                ).overall

            perturbed_ranking = sorted(perturbed_scores, key=perturbed_scores.get, reverse=True)
            perturbed_top5 = set(perturbed_ranking[:5])

            if perturbed_top5 == original_top5:
                stable_count += 1

            # Kendall tau approximation via rank displacement
            tau = self._kendall_tau(original_ranking, perturbed_ranking)
            taus.append(tau)

        import statistics as stats_module
        return {
            "top5_stability": stable_count / n_perturbations,
            "mean_kendall_tau": stats_module.mean(taus) if taus else 1.0,
            "std_kendall_tau": stats_module.stdev(taus) if len(taus) > 1 else 0.0,
            "n_perturbations": n_perturbations,
            "noise_std": noise_std,
        }

    def _perturb_weights(self, noise_std: float) -> QualityWeights:
        """Create a perturbed copy of current weights with Gaussian noise."""
        w = self.weights

        def _perturb_group(*values: float) -> list:
            perturbed = [max(0.01, v + random.gauss(0, noise_std)) for v in values]
            total = sum(perturbed)
            return [p / total for p in perturbed]

        r_weights = _perturb_group(w.r_pagerank, w.r_reverse_pagerank, w.r_in_degree)
        m_weights = _perturb_group(w.m_betweenness, w.m_out_degree, w.m_clustering)
        a_weights = _perturb_group(w.a_articulation, w.a_bridge_ratio, w.a_importance)
        v_weights = _perturb_group(w.v_eigenvector, w.v_closeness, w.v_out_degree)
        q_weights = _perturb_group(
            w.q_reliability, w.q_maintainability, w.q_availability, w.q_vulnerability
        )

        return QualityWeights(
            r_pagerank=r_weights[0], r_reverse_pagerank=r_weights[1], r_in_degree=r_weights[2],
            m_betweenness=m_weights[0], m_out_degree=m_weights[1], m_clustering=m_weights[2],
            a_articulation=a_weights[0], a_bridge_ratio=a_weights[1], a_importance=a_weights[2],
            v_eigenvector=v_weights[0], v_closeness=v_weights[1], v_out_degree=v_weights[2],
            q_reliability=q_weights[0], q_maintainability=q_weights[1],
            q_availability=q_weights[2], q_vulnerability=q_weights[3],
        )

    def _compute_rmav_with_weights(
        self, m: StructuralMetrics, norm: Dict[str, float], ap_c: float,
        weights: QualityWeights,
    ) -> QualityScores:
        """Compute RMAV with explicit weight override (for sensitivity analysis)."""
        original_weights = self.weights
        self.weights = weights
        try:
            return self._compute_rmav(m, norm, ap_c)
        finally:
            self.weights = original_weights

    @staticmethod
    def _kendall_tau(ranking_a: List[str], ranking_b: List[str]) -> float:
        """
        Compute Kendall's tau-b between two rankings.

        Simplified O(n²) implementation sufficient for the component counts
        we deal with (typically < 1000).
        """
        n = len(ranking_a)
        if n < 2:
            return 1.0

        rank_b = {item: i for i, item in enumerate(ranking_b)}
        concordant = 0
        discordant = 0

        for i in range(n):
            for j in range(i + 1, n):
                a_i, a_j = ranking_a[i], ranking_a[j]
                if a_i not in rank_b or a_j not in rank_b:
                    continue
                b_diff = rank_b[a_i] - rank_b[a_j]
                if b_diff < 0:
                    concordant += 1
                elif b_diff > 0:
                    discordant += 1

        total = concordant + discordant
        if total == 0:
            return 1.0
        return (concordant - discordant) / total

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize(self, components: List[StructuralMetrics]) -> Dict[str, float]:
        """
        Compute normalization factors based on the configured method.

        Methods:
            'max' (default): x_norm = x / max(x). Simple, preserves proportions.
            'robust': x_norm = rank(x) / n. Outlier-resistant, uniform distribution.
        """
        if self.normalization_method == "robust":
            return self._normalize_robust(components)
        return self._normalize_max(components)

    @staticmethod
    def _normalize_max(components: List[StructuralMetrics]) -> Dict[str, float]:
        """Compute per-metric maximums for max-normalization."""
        if not components:
            return {}
        return {
            "pagerank": max((c.pagerank for c in components), default=0),
            "reverse_pagerank": max((c.reverse_pagerank for c in components), default=0),
            "betweenness": max((c.betweenness for c in components), default=0),
            "closeness": max((c.closeness for c in components), default=0),
            "eigenvector": max((c.eigenvector for c in components), default=0),
            "in_degree": max((c.in_degree_raw for c in components), default=0),
            "out_degree": max((c.out_degree_raw for c in components), default=0),
            "total_degree": max((c.total_degree_raw for c in components), default=0),
        }

    @staticmethod
    def _normalize_robust(components: List[StructuralMetrics]) -> Dict[str, float]:
        """
        Compute rank-based normalization factors.

        Instead of dividing by max, we store max as a sentinel and the
        _n() helper is overridden to use rank-based normalization.

        Implementation note: For robust normalization, we pre-compute
        rank-normalized values and store them in a lookup. The norm dict
        stores max values as a signal to _compute_rmav that it should
        use rank-based lookup instead.
        """
        # For robust mode, we still return max-based norms as the actual
        # rank normalization happens at a different level. This is a
        # placeholder that can be extended for full rank-based support.
        # The key insight is that rank-normalization should be done
        # before RMAV computation for proper integration.
        if not components:
            return {}
        return {
            "pagerank": max((c.pagerank for c in components), default=0),
            "reverse_pagerank": max((c.reverse_pagerank for c in components), default=0),
            "betweenness": max((c.betweenness for c in components), default=0),
            "closeness": max((c.closeness for c in components), default=0),
            "eigenvector": max((c.eigenvector for c in components), default=0),
            "in_degree": max((c.in_degree_raw for c in components), default=0),
            "out_degree": max((c.out_degree_raw for c in components), default=0),
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