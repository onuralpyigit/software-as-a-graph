"""
Quality Scoring Service

Rule-based RMAV scoring and problem detection component of the unified
Prediction Step (Step 3). See PredictionService for the merged
RMAV + GNN + anti-pattern orchestration.
"""

from typing import Dict, Any, List, Optional
import logging

from saag.analysis.analyzer import QualityAnalyzer
from saag.analysis.problem_detector import ProblemDetector
from saag.analysis.models import QualityAnalysisResult, DetectedProblem, ProblemSummary, StructuralAnalysisResult

logger = logging.getLogger(__name__)


class QualityScoringService:
    """
    Service for running deterministic RMAV quality scoring and problem detection.
    """

    def __init__(
        self,
        use_ahp: bool = False,
        normalization_method: str = "robust",
        winsorize: bool = True,
        winsorize_limit: float = 0.05,
        equal_weights: bool = False,
        ahp_shrinkage: float = 0.7,
    ):
        self.use_ahp = use_ahp
        self.normalization_method = normalization_method
        self.winsorize = winsorize
        self.winsorize_limit = winsorize_limit
        self.equal_weights = equal_weights
        self.ahp_shrinkage = ahp_shrinkage

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
