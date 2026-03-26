"""
Prediction Service

Orchestrates the quality prediction pipeline (Step 3).
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from .analyzer import QualityAnalyzer
from .problem_detector import ProblemDetector
from .models import QualityAnalysisResult, DetectedProblem, ProblemSummary
from src.analysis.models import StructuralAnalysisResult


class PredictionService:
    """
    Service for running quality prediction and problem detection.
    Orchestrates QualityAnalyzer and ProblemDetector.
    """

    def __init__(
        self,
        use_ahp: bool = False,
        normalization_method: str = "robust",
        winsorize: bool = True,
        winsorize_limit: float = 0.05,
        equal_weights: bool = False,
    ):
        self.use_ahp = use_ahp
        self.normalization_method = normalization_method
        self.winsorize = winsorize
        self.winsorize_limit = winsorize_limit
        self.equal_weights = equal_weights

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
        )
        
        return analyzer.analyze(
            structural_result,
            run_sensitivity=run_sensitivity,
            sensitivity_perturbations=sensitivity_perturbations,
            sensitivity_noise=sensitivity_noise,
        )

    def detect_problems(self, quality_result: QualityAnalysisResult) -> List[DetectedProblem]:
        """
        Detect architectural problems from quality results.
        """
        detector = ProblemDetector()
        return detector.detect(quality_result)

    def summarize_problems(self, problems: List[DetectedProblem]) -> ProblemSummary:
        """
        Summarize detected problems.
        """
        detector = ProblemDetector()
        return detector.summarize(problems)
