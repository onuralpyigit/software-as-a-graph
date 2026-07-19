"""
Prediction Service

Orchestrates the unified Prediction Step (Step 3): rule-based RMAV scoring
(always), ML/GNN scoring (when a trained checkpoint is available), and
anti-pattern detection + explanations on top of the result. This replaces
the legacy "Quality Scoring" step that used to live inside Analyze (Step 2).
"""

from typing import Dict, Any, List, Optional, Union
import logging
from pathlib import Path

from .quality_scoring_service import QualityScoringService
from saag.analysis.models import QualityAnalysisResult, DetectedProblem, ProblemSummary, StructuralAnalysisResult

logger = logging.getLogger(__name__)


class PredictionService(QualityScoringService):
    """
    Service for running the unified Prediction Step: rule-based RMAV scoring,
    ML/GNN scoring, anti-pattern detection, and explanation generation.

    When a GNN checkpoint is available and ``prefer_gnn=True`` (default),
    :meth:`predict_quality_with_gnn` returns GNN predictions.  The rule-based
    RMAV path is always computed and serves as regularisation input during
    training and as a fallback when no checkpoint exists. Anti-patterns and
    explanations are always derived from the deterministic RMAV scores,
    since they are inherently rule-based (structural thresholds).
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
        super().__init__(
            use_ahp=use_ahp,
            normalization_method=normalization_method,
            winsorize=winsorize,
            winsorize_limit=winsorize_limit,
            equal_weights=equal_weights,
            ahp_shrinkage=ahp_shrinkage,
        )
        self.gnn_checkpoint_dir = gnn_checkpoint_dir
        self.prefer_gnn = prefer_gnn

    @staticmethod
    def _has_checkpoint(directory: str) -> bool:
        """Return True if directory contains a loadable GNN checkpoint."""
        p = Path(directory)
        return (p / "service_config.json").exists() and (
            (p / "node_model.pt").exists() or (p / "best_model.pt").exists()
        )

    def predict(
        self,
        structural_result: StructuralAnalysisResult,
        layer: str = "system",
        active_patterns: Optional[List[str]] = None,
        run_sensitivity: bool = False,
        sensitivity_perturbations: int = 200,
        sensitivity_noise: float = 0.05,
    ) -> QualityAnalysisResult:
        """Unified Prediction Step (Step 3): rule-based RMAV scoring, anti-pattern
        detection, and natural-language explanation in one pass.

        GNN-augmented scoring is available separately via
        :meth:`predict_quality_with_gnn`, which also attaches the anti-patterns
        and explanation computed here onto its result.
        """
        quality_result = self.predict_quality(
            structural_result,
            run_sensitivity=run_sensitivity,
            sensitivity_perturbations=sensitivity_perturbations,
            sensitivity_noise=sensitivity_noise,
        )
        problems, problem_summary, explanation = self._detect_problems_and_explain(
            quality_result, layer=layer, active_patterns=active_patterns
        )
        quality_result.problems = problems
        quality_result.problem_summary = problem_summary
        quality_result.explanation = explanation
        quality_result.prediction_mode = "rmav"
        return quality_result

    def _detect_problems_and_explain(
        self,
        quality_result: QualityAnalysisResult,
        layer: str = "system",
        active_patterns: Optional[List[str]] = None,
    ):
        """Run anti-pattern detection and explanation generation on RMAV scores."""
        from saag.analysis.antipattern_detector import AntiPatternDetector
        from saag.analysis.smells import AntiPatternReport
        from saag.explanation.engine import ExplanationEngine

        problems = AntiPatternDetector(active_patterns=active_patterns).detect(quality_result, layer=layer)
        problem_summary = self.summarize_problems(problems)
        smell_report = AntiPatternReport(
            problems=problems,
            summary=problem_summary.to_dict() if hasattr(problem_summary, "to_dict") else problem_summary,
        )
        explanation = ExplanationEngine().explain_system(quality_result, smell_report)
        return problems, problem_summary, explanation

    def predict_quality_with_gnn(
        self,
        structural_result: StructuralAnalysisResult,
        graph,
        simulation_results=None,
        layer: str = "system",
        active_patterns: Optional[List[str]] = None,
    ) -> Union[QualityAnalysisResult, Any]:
        """Return GNN predictions when a checkpoint exists, else fall back to RMAV.

        RMAV scores are always computed — they serve as the consistency
        regularisation target for the GNN and as a fallback when no
        checkpoint is present. Anti-patterns and explanations are always
        derived from the RMAV scores and attached to whichever result
        (GNN or RMAV) is ultimately returned.
        """
        rmav_result = self.predict_quality(structural_result)
        problems, problem_summary, explanation = self._detect_problems_and_explain(
            rmav_result, layer=layer, active_patterns=active_patterns
        )
        rmav_result.problems = problems
        rmav_result.problem_summary = problem_summary
        rmav_result.explanation = explanation
        rmav_result.prediction_mode = "rmav"

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
                gnn_result = gnn_svc.predict(
                    graph=graph,
                    structural_metrics=extract_structural_metrics_dict(structural_result),
                    rmav_scores=extract_rmav_scores_dict(rmav_result),
                    simulation_results=simulation_results,
                    mode="gnn",
                )
                try:
                    gnn_result.problems = problems
                    gnn_result.problem_summary = problem_summary
                    gnn_result.explanation = explanation
                except Exception:
                    logger.debug(
                        "Could not attach anti-pattern/explanation metadata to GNN result.",
                        exc_info=True,
                    )
                return gnn_result
            except Exception:
                logger.warning(
                    "GNN inference failed; falling back to RMAV scores.", exc_info=True
                )

        logger.info(
            "No GNN checkpoint at '%s'; returning RMAV scores.", self.gnn_checkpoint_dir
        )
        return rmav_result
