from src.core.ports.graph_repository import IGraphRepository
from src.analysis.service import AnalysisService
from src.analysis.models import StructuralAnalysisResult

class AnalyzeGraphUseCase:
    """Use case for running structural analysis on a graph."""
    
    def __init__(self, repository: IGraphRepository):
        self.repository = repository
        self.service = AnalysisService(repository)
        
    def execute(self, layer: str) -> StructuralAnalysisResult:
        result = self.service.analyze_layer(layer)
        return result.structural
