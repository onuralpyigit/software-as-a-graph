"""
Prediction Service

Orchestrates the quality prediction pipeline (Step 3).
"""

from typing import Dict, Any, List, Optional, Union
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

from .analyzer import QualityAnalyzer
from .problem_detector import ProblemDetector
from .models import QualityAnalysisResult, DetectedProblem, ProblemSummary
from saag.analysis.models import StructuralAnalysisResult


class PredictionService:
    """
    Service for running quality prediction and problem detection.

    When a GNN checkpoint is available and ``prefer_gnn=True`` (default),
    :meth:`predict_quality_with_gnn` returns GNN predictions.  The rule-based
    RMAV path is always computed and serves as regularisation input during
    training and as a fallback when no checkpoint exists.
    """

    def __init__(
        self,
        use_ahp: bool = False,
        normalization_method: str = "robust",
        winsorize: bool = True,
        winsorize_limit: float = 0.05,
        equal_weights: bool = False,
        ahp_shrinkage: float = 0.7,
        gnn_checkpoint_dir: Optional[str] = None,
        prefer_gnn: bool = True,
    ):
        self.use_ahp = use_ahp
        self.normalization_method = normalization_method
        self.winsorize = winsorize
        self.winsorize_limit = winsorize_limit
        self.equal_weights = equal_weights
        self.ahp_shrinkage = ahp_shrinkage
        self.gnn_checkpoint_dir = gnn_checkpoint_dir
        self.prefer_gnn = prefer_gnn

    def predict_quality(
        self, 
        structural_result: StructuralAnalysisResult,
        run_sensitivity: bool = False,
        sensitivity_perturbations: int = 200,
        sensitivity_noise: float = 0.05
    ) -> QualityAnalysisResult:
        """
        Run quality prediction on a structural analysis result.
        """
        analyzer = QualityAnalyzer(
            normalization_method=self.normalization_method,
            winsorize=self.winsorize,
            winsorize_limit=self.winsorize_limit,
            use_ahp=self.use_ahp,
            equal_weights=self.equal_weights,
            ahp_shrinkage=self.ahp_shrinkage,
        )
        
        return analyzer.analyze(
            structural_result,
            run_sensitivity=run_sensitivity,
            sensitivity_perturbations=sensitivity_perturbations,
            sensitivity_noise=sensitivity_noise,
        )

    @staticmethod
    def _has_checkpoint(directory: str) -> bool:
        """Return True if directory contains a loadable GNN checkpoint."""
        p = Path(directory)
        return (p / "service_config.json").exists() and (
            (p / "node_model.pt").exists() or (p / "best_model.pt").exists()
        )

    def predict_quality_with_gnn(
        self,
        structural_result: StructuralAnalysisResult,
        graph,
        simulation_results=None,
    ) -> Union[QualityAnalysisResult, Any]:
        """Return GNN predictions when a checkpoint exists, else fall back to RMAV.

        RMAV scores are always computed — they serve as the consistency
        regularisation target for the GNN and as a fallback when no
        checkpoint is present.
        """
        rmav_result = self.predict_quality(structural_result)

        if (
            self.prefer_gnn
            and self.gnn_checkpoint_dir
            and self._has_checkpoint(self.gnn_checkpoint_dir)
        ):
            try:
                from .gnn_service import GNNService
                from .data_preparation import (
                    extract_structural_metrics_dict,
                    extract_rmav_scores_dict,
                )
                gnn_svc = GNNService.from_checkpoint(self.gnn_checkpoint_dir, graph=graph)
                return gnn_svc.predict(
                    graph=graph,
                    structural_metrics=extract_structural_metrics_dict(structural_result),
                    rmav_scores=extract_rmav_scores_dict(rmav_result),
                    simulation_results=simulation_results,
                    mode="gnn",
                )
            except Exception:
                logger.warning(
                    "GNN inference failed; falling back to RMAV scores.", exc_info=True
                )

        logger.info(
            "No GNN checkpoint at '%s'; returning RMAV scores.", self.gnn_checkpoint_dir
        )
        return rmav_result

    def detect_problems(self, quality_result: QualityAnalysisResult, active_patterns: Optional[List[str]] = None) -> List[DetectedProblem]:
        """
        Detect architectural problems from quality results.
        """
        detector = ProblemDetector(active_patterns=active_patterns)
        return detector.detect(quality_result)

    def summarize_problems(self, problems: List[DetectedProblem]) -> ProblemSummary:
        """
        Summarize detected problems.
        """
        detector = ProblemDetector()
        return detector.summarize(problems)
