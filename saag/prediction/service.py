"""
Prediction Service

Orchestrates the unified Prediction Step (Step 3): rule-based RMAV scoring
(always), ML/GNN scoring (when a trained checkpoint is available), and
anti-pattern detection + explanations on top of the result. This replaces
the legacy "Quality Scoring" step that used to live inside Analyze (Step 2).
"""

from typing import Any, List, Optional, Tuple, Union
import logging
from pathlib import Path

from saag.analysis.analyzer import QualityAnalyzer
from saag.analysis.models import (
    DetectedProblem,
    ProblemSummary,
    QualityAnalysisResult,
    StructuralAnalysisResult,
)
from saag.analysis.problem_detector import ProblemDetector

logger = logging.getLogger(__name__)


class PredictionService:
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
        self.use_ahp = use_ahp
        self.normalization_method = normalization_method
        self.winsorize = winsorize
        self.winsorize_limit = winsorize_limit
        self.equal_weights = equal_weights
        self.ahp_shrinkage = ahp_shrinkage
        self.gnn_checkpoint_dir = gnn_checkpoint_dir
        self.prefer_gnn = prefer_gnn

    # ── Rule-based RMAV scoring ───────────────────────────────────────────────

    def predict_quality(
        self,
        structural_result: StructuralAnalysisResult,
        run_sensitivity: bool = False,
        sensitivity_perturbations: int = 200,
        sensitivity_noise: float = 0.05,
    ) -> QualityAnalysisResult:
        """Run deterministic RMAV quality scoring on a structural analysis result."""
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

    def detect_problems(
        self,
        quality_result: QualityAnalysisResult,
        active_patterns: Optional[List[str]] = None,
    ) -> List[DetectedProblem]:
        """Detect architectural problems from quality results."""
        return ProblemDetector(active_patterns=active_patterns).detect(quality_result)

    def summarize_problems(self, problems: List[DetectedProblem]) -> ProblemSummary:
        """Summarize detected problems."""
        return ProblemDetector().summarize(problems)

    # ── Unified Predict step ──────────────────────────────────────────────────

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
        self._attach_problems_and_explanation(
            quality_result, layer=layer, active_patterns=active_patterns
        )
        return quality_result

    def predict_quality_with_gnn(
        self,
        structural_result: StructuralAnalysisResult,
        graph,
        simulation_results=None,
        layer: str = "system",
        active_patterns: Optional[List[str]] = None,
        run_sensitivity: bool = False,
    ) -> Union[QualityAnalysisResult, Any]:
        """Return GNN predictions when a checkpoint exists, else fall back to RMAV.

        RMAV scores are always computed — they serve as the consistency
        regularisation target for the GNN and as a fallback when no
        checkpoint is present. Anti-patterns and explanations are always
        derived from the RMAV scores and attached to whichever result
        (GNN or RMAV) is ultimately returned.
        """
        rmav_result = self.predict_quality(structural_result, run_sensitivity=run_sensitivity)
        problems, problem_summary, explanation = self._attach_problems_and_explanation(
            rmav_result, layer=layer, active_patterns=active_patterns
        )

        if not (
            self.prefer_gnn
            and self.gnn_checkpoint_dir
            and self._has_checkpoint(self.gnn_checkpoint_dir)
        ):
            logger.info(
                "No GNN checkpoint at '%s'; returning RMAV scores.", self.gnn_checkpoint_dir
            )
            return rmav_result

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
                eval_labels=simulation_results,
                mode="gnn",
            )
        except Exception:
            logger.warning("GNN inference failed; falling back to RMAV scores.", exc_info=True)
            return rmav_result

        gnn_result.problems = problems
        gnn_result.problem_summary = problem_summary
        gnn_result.explanation = explanation
        return gnn_result

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _has_checkpoint(directory: str) -> bool:
        """Return True if directory contains a loadable GNN checkpoint."""
        p = Path(directory)
        return (p / "service_config.json").exists() and (
            (p / "node_model.pt").exists() or (p / "best_model.pt").exists()
        )

    def _attach_problems_and_explanation(
        self,
        quality_result: QualityAnalysisResult,
        layer: str = "system",
        active_patterns: Optional[List[str]] = None,
    ) -> Tuple[List[DetectedProblem], ProblemSummary, Any]:
        """Run anti-pattern detection and explanation on RMAV scores, and attach
        them to *quality_result*. Returns them so a GNN result can reuse them."""
        from saag.analysis.antipattern_detector import AntiPatternDetector
        from saag.analysis.smells import AntiPatternReport
        from saag.explanation.engine import ExplanationEngine

        # An empty list is an explicit opt-out, distinct from None ("run the whole
        # catalogue"). AntiPatternDetector treats both as falsy and would run
        # everything, so the caller's opt-out has to be honoured here.
        if active_patterns is not None and len(active_patterns) == 0:
            problems = []
        else:
            problems = AntiPatternDetector(active_patterns=active_patterns).detect(
                quality_result, layer=layer
            )
        problem_summary = self.summarize_problems(problems)
        smell_report = AntiPatternReport(
            problems=problems,
            summary=problem_summary.to_dict()
            if hasattr(problem_summary, "to_dict")
            else problem_summary,
        )
        explanation = ExplanationEngine().explain_system(quality_result, smell_report)

        quality_result.problems = problems
        quality_result.problem_summary = problem_summary
        quality_result.explanation = explanation
        quality_result.prediction_mode = "rmav"
        return problems, problem_summary, explanation
