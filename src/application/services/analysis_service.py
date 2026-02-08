"""
Analysis Service

Application service implementing IAnalysisUseCase.
Orchestrates the three-stage analysis pipeline per layer:

    1. Structural Analysis  → Raw topological metrics (centrality, degree, ...)
    2. Quality Analysis     → Composite RMAV scores + Box-Plot classification
    3. Problem Detection    → Architectural smells, risks, recommendations

Each layer filters the graph to its own DEPENDS_ON relationship types:
    app    → app_to_app
    infra  → node_to_node
    mw     → app_to_broker + node_to_broker
    system → all of the above
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.application.ports import IAnalysisUseCase, IGraphRepository

from src.domain.config.layers import (
    AnalysisLayer,
    get_layer_definition,
    get_primary_layers,
)
from src.domain.services import (
    StructuralAnalyzer,
    QualityAnalyzer,
    ProblemDetector,
)
from src.domain.models.analysis.results import (
    LayerAnalysisResult,
    MultiLayerAnalysisResult,
)


class AnalysisService(IAnalysisUseCase):
    """
    Main service for multi-layer graph analysis.

    Follows the hexagonal architecture pattern:
    - Inbound port: IAnalysisUseCase
    - Outbound port: IGraphRepository (injected)
    """

    def __init__(self, repository: IGraphRepository, use_ahp: bool = False) -> None:
        self._repo = repository
        self._use_ahp = use_ahp
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_layer(self, layer: str = "system") -> LayerAnalysisResult:
        """
        Run the full analysis pipeline for a single layer.

        Args:
            layer: One of "app", "infra", "mw", "system" (or an AnalysisLayer enum).

        Returns:
            LayerAnalysisResult containing structural metrics, quality scores,
            and detected problems.
        """
        analysis_layer = self._resolve_layer(layer)
        definition = get_layer_definition(analysis_layer)
        self._logger.info("Analyzing layer: %s (%s)", definition.name, analysis_layer.value)

        # 1. Load graph data filtered to this layer's component types
        graph_data = self._repo.get_graph_data(
            component_types=definition.component_types,
        )

        # 2. Structural Analysis — raw topological metrics
        structural = StructuralAnalyzer().analyze(graph_data, layer=analysis_layer)

        # 3. Quality Analysis — RMAV scoring + Box-Plot classification
        quality = QualityAnalyzer(use_ahp=self._use_ahp).analyze(structural)

        # 4. Problem Detection — risks, smells, recommendations
        detector = ProblemDetector()
        problems = detector.detect(quality)
        summary = detector.summarize(problems)

        # 5. Contextual data for display enrichment
        library_usage = self._repo.get_library_usage()
        node_allocations = self._repo.get_node_allocations()
        broker_routing = self._repo.get_broker_routing()

        return LayerAnalysisResult(
            layer=analysis_layer.value,
            layer_name=definition.name,
            description=definition.description,
            structural=structural,
            quality=quality,
            problems=problems,
            problem_summary=summary,
            library_usage=library_usage,
            node_allocations=node_allocations,
            broker_routing=broker_routing,
        )

    def analyze_all_layers(
        self, include_cross_layer: bool = True,
    ) -> MultiLayerAnalysisResult:
        """
        Analyse every primary layer (app, infra, mw, system).

        Returns a MultiLayerAnalysisResult containing all individual layer
        results plus optional cross-layer insights.
        """
        results: Dict[str, LayerAnalysisResult] = {}
        for al in get_primary_layers():
            results[al.value] = self.analyze_layer(al.value)

        cross = self._cross_layer_insights(results) if include_cross_layer else []

        return MultiLayerAnalysisResult(
            timestamp=datetime.now().isoformat(),
            layers=results,
            cross_layer_insights=cross,
        )

    def export_results(
        self,
        result: Union[LayerAnalysisResult, MultiLayerAnalysisResult],
        output_file: str,
    ) -> None:
        """Serialise results to a JSON file."""
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(result.to_dict(), fh, indent=2, default=str)
        self._logger.info("Results exported to %s", path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_layer(layer: Union[str, AnalysisLayer]) -> AnalysisLayer:
        """Accept both string and enum; normalise to AnalysisLayer."""
        if isinstance(layer, AnalysisLayer):
            return layer
        return AnalysisLayer.from_string(layer)

    @staticmethod
    def _cross_layer_insights(
        results: Dict[str, LayerAnalysisResult],
    ) -> List[str]:
        """
        Generate insights by comparing layers.

        These are high-level observations about the system architecture
        that only become visible when looking across multiple layers.
        """
        insights: List[str] = []

        app = results.get("app")
        infra = results.get("infra")
        mw = results.get("mw")

        # App-to-Infra density ratio
        if app and infra:
            app_n = app.structural.graph_summary.nodes
            infra_n = infra.structural.graph_summary.nodes
            if infra_n > 0:
                ratio = app_n / infra_n
                insights.append(
                    f"App / Infra ratio: {ratio:.2f} applications per node"
                )
                if ratio > 5:
                    insights.append(
                        "⚠ High application density — consider scaling infrastructure"
                    )

        # Compare critical-component counts
        for key, res in results.items():
            crit = res.problem_summary.by_severity.get("CRITICAL", 0)
            if crit > 0:
                insights.append(
                    f"Layer '{key}': {crit} CRITICAL problem(s) detected"
                )

        # Middleware bottleneck warning
        if mw:
            spofs = [
                c for c in mw.quality.components
                if c.structural.is_articulation_point
            ]
            if spofs:
                ids = ", ".join(s.id for s in spofs[:5])
                insights.append(
                    f"Middleware SPOFs: {ids} — broker redundancy recommended"
                )

        return insights