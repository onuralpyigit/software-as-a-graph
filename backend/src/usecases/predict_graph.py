from src.prediction.analyzer import QualityAnalyzer
from src.prediction.models import QualityAnalysisResult
from src.analysis.models import StructuralAnalysisResult

class PredictGraphUseCase:
    """
    Use case for predicting graph quality scores (RMAV).
    
    INDEPENDENCE GUARANTEE:
    This use case accepts StructuralAnalysisResult as a parameter.
    It never reads raw runtime data from the repository.
    This enforces the pre-deployment claim in code.
    """
    
    def __init__(self, repository: IGraphRepository):
        self.repository = repository
        from src.prediction.service import PredictionService
        self.service = PredictionService()
        
    def execute(self, layer: str, structural_result: StructuralAnalysisResult) -> QualityAnalysisResult:
        return self.service.predict_quality(structural_result)
