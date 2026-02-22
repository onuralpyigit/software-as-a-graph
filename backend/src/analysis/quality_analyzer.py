"""
Quality Analyzer

Computes composite quality scores for the four RMAV dimensions and an
overall quality score, then classifies every component and edge using the
Box-Plot method (adaptive, data-driven thresholds).

Formulas (per component v):
    R(v) = w_rpr·RPR + w_win·w_in + w_cdp·CDPot          (Reliability)
    M(v) = w_bt·BC   + w_od·OutDeg + w_cl·(1 – CC)        (Maintainability)
    A(v) = w_ap·AP_c + w_br·Bridge + w_imp·Importance      (Availability)
    V(v) = w_ev·Eig  + w_cl·Close  + w_od·OutDeg           (Vulnerability)
    Q(v) = w_R·R(v)  + w_M·M(v)    + w_A·A(v) + w_V·V(v)  (Overall)

R(v) v4 metrics:
    RPR(v)   — Reverse PageRank on G^T: propagation reach
    w_in(v)  — QoS-weighted in-degree (promoted from 'Reported only')
    CDPot(v) — Cascade Depth Potential (derived, no new algorithm):
                ((RPR + DG_in) / 2) × (1 − min(DG_out / DG_in, 1))

Design changes (v4):
    - R(v) formula: PR(v) removed (correlated with RPR); DG_in replaced by w_in
      (QoS-weighted — strictly more informative than unweighted in-degree);
      CDPot added as derived depth signal without new algorithm cost.
    - pubsub_betweenness removed from R(v): it belongs in Maintainability via
      betweenness; its redistribution hack is eliminated for a cleaner formula.

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
        k_factor: float = 0.75,
        weights: Optional[QualityWeights] = None,
        use_ahp: bool = False,
        ahp_shrinkage: float = 0.7,
        normalization_method: str = "robust",
        winsorize: bool = True,
        winsorize_limit: float = 0.05,
        adapt_qos_weights: bool = True,
    ) -> None:
        self.classifier = BoxPlotClassifier(k_factor=k_factor)
        self.weights = (
            AHPProcessor(shrinkage_factor=ahp_shrinkage).compute_weights() if use_ahp
            else (weights or QualityWeights())
        )
        self.normalization_method = normalization_method
        self.winsorize = winsorize
        self.winsorize_limit = winsorize_limit
        self.adapt_qos_weights = adapt_qos_weights
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

        # --- QoS-aware weight adjustment ----------------------------------
        effective_weights = self.weights
        if self.adapt_qos_weights and hasattr(structural_result, "qos_profile"):
            effective_weights = self._derive_qos_weights(
                structural_result.qos_profile, self.weights
            )

        # Store effective weights for the duration of this analysis
        original_weights = self.weights
        self.weights = effective_weights

        try:
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
        finally:
            self.weights = original_weights

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
        self, m: StructuralMetrics, norm: Dict[str, Any], ap_c: float,
    ) -> QualityScores:
        """
        Compute Reliability, Maintainability, Availability, Vulnerability scores.

        Design principles:
            - Each raw metric maps to at most one dimension (orthogonality).
            - Out-Degree is shared between M(v) and V(v) with distinct semantics.
            - AP is continuous (ap_c), not binary.
            - pubsub_betweenness added to Reliability for pub-sub fabric importance.
        """
        w = self.weights

        def _n(val: float, key: str) -> float:
            """Look up pre-normalised (rank-based or max-based) value."""
            if isinstance(norm, dict) and key in norm:
                entry = norm[key]
                if isinstance(entry, dict):
                    # Rank-based: lookup per component id → rank score
                    return entry.get(m.id, 0.0)
                else:
                    # Max-based: divide by stored max
                    return val / entry if entry > 0 else 0.0
            return 0.0

        # Normalised values
        pr   = _n(m.pagerank,         "pagerank")
        rpr  = _n(m.reverse_pagerank, "reverse_pagerank")
        bt   = _n(m.betweenness,      "betweenness")
        cl   = _n(m.closeness,        "closeness")
        ev   = _n(m.eigenvector,      "eigenvector")
        id_n = _n(m.in_degree_raw,    "in_degree")
        od_n = _n(m.out_degree_raw,   "out_degree")
        cc   = m.clustering_coefficient
        qw   = _n(m.weight,           "weight")
        psbt = m.pubsub_betweenness  # already in [0,1]

        # --- Reliability: R(v) v4 = RPR + w_in + CDPot ---
        # CDPot(v) = ((RPR + DG_in) / 2) × (1 − min(DG_out / DG_in, 1))
        # Depth penalty: high out/in ratio → shallow fan-out cascade (less CDPot)
        # High in/out ratio → deep absorber cascade (more CDPot)
        _denom = max(id_n, 1e-9)  # avoid division by zero
        _cdpot_reach = (rpr + id_n) / 2.0
        _cdpot_depth = 1.0 - min(od_n / _denom, 1.0)
        cdpot = _cdpot_reach * _cdpot_depth

        # qw serves as the normalised w_in: QoS-derived component weight already
        # incorporates both dependent count and QoS quality of those dependencies.
        R = w.r_reverse_pagerank * rpr + getattr(w, 'r_w_in', 0.35) * qw + getattr(w, 'r_cdpot', 0.25) * cdpot

        # Maintainability: efferent coupling + bottleneck position
        M = w.m_betweenness * bt + w.m_out_degree * od_n + w.m_clustering * (1.0 - cc)

        # Availability: SPOF risk (continuous AP score)
        A = w.a_articulation * ap_c + w.a_bridge_ratio * m.bridge_ratio + w.a_qos_weight * qw

        # Vulnerability: strategic importance + propagation speed
        V = w.v_eigenvector * ev + w.v_closeness * cl

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

        r_weights = _perturb_group(
            w.r_reverse_pagerank,
            getattr(w, 'r_w_in', 0.35),
            getattr(w, 'r_cdpot', 0.25),
        )
        m_weights = _perturb_group(w.m_betweenness, w.m_out_degree, w.m_clustering)
        a_weights = _perturb_group(w.a_articulation, w.a_bridge_ratio, w.a_importance)
        v_weights = _perturb_group(w.v_eigenvector, w.v_closeness, w.v_out_degree)
        q_weights = _perturb_group(
            w.q_reliability, w.q_maintainability, w.q_availability, w.q_vulnerability
        )

        return QualityWeights(
            r_pagerank=0.0, r_reverse_pagerank=r_weights[0], r_in_degree=0.0,
            r_w_in=r_weights[1], r_cdpot=r_weights[2],
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

    def _normalize(self, components: List[StructuralMetrics]) -> Dict[str, Any]:
        """
        Compute normalization factors based on the configured method.

        Steps:
            1. (Optional) Winsorize: caps extreme outliers at (1 - limit) percentile.
            2. Scale: 'max' or 'robust' (rank-based).
        """
        if self.winsorize:
            components = self._winsorize_components(components, limit=self.winsorize_limit)

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
            "weight": max((c.weight for c in components), default=1.0),
        }

    @staticmethod
    def _normalize_robust(components: List[StructuralMetrics]) -> Dict[str, Dict[str, float]]:
        """
        True rank-based normalization. For each metric, replace raw values with
        rank(v) / N where N = number of components. This gives a uniform [0, 1]
        distribution per metric, making the score robust to outliers.

        Returns a nested dict: {metric_name: {component_id: normalised_rank_score}}
        The _n() helper in _compute_rmav detects this structure and looks up
        the pre-computed rank score instead of dividing by max.
        """
        if not components:
            return {}

        n = len(components)

        def _rank_normalise(values: List[tuple]) -> Dict[str, float]:
            """Assign rank/(n-1) with average-rank tie-breaking."""
            # values: list of (component_id, raw_value)
            sorted_vals = sorted(values, key=lambda x: x[1])
            ranks: Dict[str, float] = {}
            i = 0
            while i < n:
                j = i
                # Find extent of ties
                while j < n - 1 and sorted_vals[j][1] == sorted_vals[j + 1][1]:
                    j += 1
                avg_rank = (i + j) / 2  # 0-based average rank
                for k in range(i, j + 1):
                    cid = sorted_vals[k][0]
                    ranks[cid] = avg_rank / (n - 1) if n > 1 else 0.0
                i = j + 1
            return ranks

        metrics = {
            "pagerank":        [(c.id, c.pagerank)          for c in components],
            "reverse_pagerank":[(c.id, c.reverse_pagerank)  for c in components],
            "betweenness":     [(c.id, c.betweenness)       for c in components],
            "closeness":       [(c.id, c.closeness)         for c in components],
            "eigenvector":     [(c.id, c.eigenvector)       for c in components],
            "in_degree":       [(c.id, c.in_degree_raw)     for c in components],
            "out_degree":      [(c.id, c.out_degree_raw)    for c in components],
            "total_degree":    [(c.id, c.total_degree_raw)  for c in components],
            "weight":          [(c.id, c.weight)            for c in components],
        }
        return {name: _rank_normalise(vals) for name, vals in metrics.items()}

    @staticmethod
    def _winsorize_components(
        components: List[StructuralMetrics], 
        limit: float = 0.05
    ) -> List[StructuralMetrics]:
        """
        Apply winsorization to raw structural metrics.
        Caps values above the (1 - limit) percentile to mitigate outlier influence.
        """
        if not components or limit <= 0:
            return components

        import copy

        def _get_cap(values: List[float], p: float) -> float:
            if not values: return 0.0
            s = sorted(values)
            n = len(s)
            k_idx = (n - 1) * p
            f = int(k_idx)
            c = min(f + 1, n - 1)
            return s[f] + (k_idx - f) * (s[c] - s[f])

        # We must copy metrics because we are going to modify them
        winsorized = [copy.copy(c) for c in components]
        
        metrics_to_winsorize = [
            "pagerank", "reverse_pagerank", "betweenness", "closeness", 
            "eigenvector", "in_degree_raw", "out_degree_raw", "weight"
        ]

        for attr in metrics_to_winsorize:
            raw_vals = [getattr(c, attr) for c in components]
            cap = _get_cap(raw_vals, 1.0 - limit)
            
            for c in winsorized:
                val = getattr(c, attr)
                if val > cap:
                    setattr(c, attr, cap)
        
        return winsorized

    # ------------------------------------------------------------------
    # QoS-aware weight derivation (Fix 3)
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_qos_weights(
        qos_profile: Dict,
        base_weights: QualityWeights,
    ) -> QualityWeights:
        """
        Adjust overall quality dimension weights based on the dominant
        QoS profile of the topic set. Rules:

        - PERSISTENT / RELIABLE / CRITICAL priority → high-durability,
          mission-critical system → raise q_reliability + q_availability.
        - VOLATILE / BEST_EFFORT / LOW priority → high-churn, low-durability
          system → raise q_maintainability + q_vulnerability.
        - Mixed → keep balanced defaults.

        All four weights are re-normalised to sum to 1.0.
        """
        import copy
        w = copy.copy(base_weights)

        total = qos_profile.get("total_topics", 0)
        if total == 0:
            return w

        dur = qos_profile.get("durability", {})
        rel = qos_profile.get("reliability", {})
        pri = qos_profile.get("priority", {})

        # Fraction of high-durability topics
        persistent_frac = (
            dur.get("persistent", 0) + dur.get("transient_local", 0) + dur.get("transient", 0)
        ) / total

        # Fraction of reliable topics
        reliable_frac = rel.get("reliable", 0) / total

        # Fraction of critical/high priority topics
        critical_frac = (
            pri.get("critical", 0) + pri.get("high", 0)
        ) / total

        # Composite "reliability signal"
        rel_signal = (persistent_frac + reliable_frac + critical_frac) / 3.0

        # Adjust weights: rel_signal > 0.6 → reliability-critical system
        #                 rel_signal < 0.4 → volatile/best-effort system
        if rel_signal >= 0.6:
            # Mission-critical: emphasise reliability and availability
            delta = min(0.15, (rel_signal - 0.5) * 0.30)
            q_r = w.q_reliability + delta
            q_a = w.q_availability + delta * 0.5
            q_m = w.q_maintainability - delta * 0.75
            q_v = w.q_vulnerability - delta * 0.75
        elif rel_signal <= 0.4:
            # Volatile/best-effort: emphasise maintainability and vulnerability
            delta = min(0.15, (0.5 - rel_signal) * 0.30)
            q_r = w.q_reliability - delta * 0.75
            q_a = w.q_availability - delta * 0.75
            q_m = w.q_maintainability + delta
            q_v = w.q_vulnerability + delta * 0.5
        else:
            # Balanced — keep defaults
            return w

        # Re-normalise to sum to 1.0 (clamp to small positive to avoid negatives)
        q_r = max(0.05, q_r)
        q_a = max(0.05, q_a)
        q_m = max(0.05, q_m)
        q_v = max(0.05, q_v)
        total_q = q_r + q_a + q_m + q_v

        w.q_reliability     = q_r / total_q
        w.q_availability     = q_a / total_q
        w.q_maintainability  = q_m / total_q
        w.q_vulnerability    = q_v / total_q

        return w

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