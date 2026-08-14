"""
Use case for running multi-layer system analysis.
"""

import logging
from datetime import datetime
from typing import Any, List, Optional

from saag.analysis.antipattern_detector import AntiPatternDetector
from saag.analysis.cross_layer import compute_cross_layer_insights
from saag.analysis.models import MultiLayerAnalysisResult
from saag.analysis.service import AnalysisService
from saag.prediction.service import PredictionService

logger = logging.getLogger(__name__)

#: Extra kwargs forwarded from execute(**kwargs) to PredictionService.predict_quality.
_QUALITY_KWARG_KEYS = ("run_sensitivity", "sensitivity_perturbations", "sensitivity_noise")


class MultiLayerAnalysisUseCase:
    """
    Orchestrates multi-layer structural analysis (Step 2), deterministic RM
    quality scoring, anti-pattern detection, and optional GNN prediction.
    """

    def __init__(self, repository: Any):
        self.repository = repository

    def execute(
        self,
        layers: List[str],
        use_ahp: bool = False,
        normalization_method: str = "robust",
        winsorize: bool = True,
        winsorize_limit: float = 0.05,
        gnn_model: Optional[str] = None,
        equal_weights: bool = False,
        ahp_shrinkage: float = 0.7,
        **kwargs
    ) -> MultiLayerAnalysisResult:
        # 1. Structural analysis — derives dependencies and loads the graph once.
        results_map = AnalysisService(self.repository).analyze_layers(layers)

        scorer = PredictionService(
            use_ahp=use_ahp,
            normalization_method=normalization_method,
            winsorize=winsorize,
            winsorize_limit=winsorize_limit,
            equal_weights=equal_weights,
            ahp_shrinkage=ahp_shrinkage,
        )
        detector = AntiPatternDetector()
        quality_kwargs = {k: v for k, v in kwargs.items() if k in _QUALITY_KWARG_KEYS}

        for layer, layer_res in results_map.items():
            # 2. Quality analysis (RM)
            layer_res.quality = scorer.predict_quality(layer_res.structural, **quality_kwargs)

            # 3. Anti-pattern detection
            layer_res.problems = detector.detect(layer_res.quality, layer=layer)
            layer_res.problem_summary = scorer.summarize_problems(layer_res.problems)

            # 4. Optional GNN prediction
            if gnn_model:
                self._add_gnn_prediction(layer_res, layer, gnn_model)

        # 5. Cross-layer insights
        return MultiLayerAnalysisResult(
            timestamp=datetime.now().isoformat(),
            layers=results_map,
            cross_layer_insights=compute_cross_layer_insights(results_map),
        )

    @staticmethod
    def _add_gnn_prediction(layer_res: Any, layer: str, gnn_model: str) -> None:
        """Attach GNN-derived criticality to *layer_res*; log and skip on failure."""
        try:
            from saag.prediction.gnn_service import (
                GNNService,
                extract_structural_metrics_dict,
                extract_rm_scores_dict,
            )

            gnn_svc = GNNService.from_checkpoint(gnn_model, graph=layer_res.structural.graph)
            prediction_result = gnn_svc.predict(
                graph=layer_res.structural.graph,
                structural_metrics=extract_structural_metrics_dict(layer_res.structural),
                rm_scores=extract_rm_scores_dict(layer_res.quality),
            )
            layer_res.prediction = prediction_result.to_dict()
        except Exception as e:
            logger.error(f"GNN prediction for layer {layer} failed: {e}")
