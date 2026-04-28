from typing import List, Optional, Any
from saag.core.ports.graph_repository import IGraphRepository
from saag.prediction.analyzer import QualityAnalyzer
from saag.prediction.models import QualityAnalysisResult, DetectedProblem
from saag.analysis.models import StructuralAnalysisResult

class PredictGraphUseCase:
    """
    Use case for predicting graph quality scores (RMAV).
    
    INDEPENDENCE GUARANTEE:
    This use case accepts StructuralAnalysisResult as a parameter.
    It never reads raw runtime data from the repository.
    This enforces the pre-deployment claim in code.
    """
    
    def __init__(self, prediction_service: Any):
        self.service = prediction_service

        
    def execute(
        self, 
        layer: str, 
        structural_result: StructuralAnalysisResult,
        detect_problems: bool = False,
        **kwargs
    ) -> tuple[QualityAnalysisResult, Optional[List[DetectedProblem]]]:
        quality = self.service.predict_quality(structural_result, **kwargs)
        problems = None
        if detect_problems:
            problems = self.service.detect_problems(quality)
        return quality, problems
