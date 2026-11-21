from typing import List, Optional, Any
import warnings
from saag.analysis.models import (
    DetectedProblem,
    QualityAnalysisResult,
    StructuralAnalysisResult,
)

class PredictGraphUseCase:
    """
    Legacy use case for predicting deterministic graph quality scores (RM).

    .. deprecated::
        Prefer :class:`saag.usecases.predictive.PredictiveUseCase` for Step 3
        (Pathway B predictive ranking) or :class:`saag.usecases.diagnostic.DiagnosticUseCase`
        for Step 4 (Pathway A deterministic RM attribution and AntiPatternDetector).

    INDEPENDENCE GUARANTEE:
    This use case accepts StructuralAnalysisResult as a parameter.
    It never reads raw runtime data from the repository.
    This enforces the pre-deployment claim in code.
    """
    
    def __init__(self, prediction_service: Any):
        warnings.warn(
            "PredictGraphUseCase is deprecated; use PredictiveUseCase (Pathway B) "
            "or DiagnosticUseCase (Pathway A) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.service = prediction_service

        
    def execute(
        self, 
        layer: str, 
        structural_result: StructuralAnalysisResult,
        detect_problems: bool = True,
        **kwargs
    ) -> tuple[QualityAnalysisResult, Optional[List[DetectedProblem]]]:
        quality = self.service.predict_quality(structural_result, **kwargs)
        problems = None
        if detect_problems:
            problems = self.service.detect_problems(quality)
        return quality, problems
