from src.core.ports.graph_repository import IGraphRepository
from src.analysis.quality_analyzer import QualityAnalyzer
from src.analysis.models import StructuralAnalysisResult, QualityAnalysisResult

class PredictGraphUseCase:
    """
    Use case for predicting graph quality scores (RMAV).
    
    INDEPENDENCE GUARANTEE:
    This use case accepts StructuralAnalysisResult as a parameter.
    It never reads raw runtime data from the repository.
    This enforces the pre-deployment claim in code.
    """
    
    def __init__(self, repository: IGraphRepository):
        # We hold the repository to satisfy the general contract,
        # but we do NOT use it in execute() for runtime data.
        self.repository = repository
        # Use default analyzer settings
        self.analyzer = QualityAnalyzer()
        
    def execute(self, layer: str, structural_result: StructuralAnalysisResult) -> QualityAnalysisResult:
        # Use a context string for the analysis
        context = f"Prediction for {layer} layer (Structural-Only)"
        
        # QualityAnalyzer.analyze only takes StructuralAnalysisResult
        # and computes scores based on topological metrics.
        return self.analyzer.analyze(structural_result, context=context)
