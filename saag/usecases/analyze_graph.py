from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from saag.analysis.models import LayerAnalysisResult

class AnalyzeGraphUseCase:
    """Use case for running structural analysis on a graph."""
    
    def __init__(self, service: "AnalysisService"):
        self.service = service

        
    def execute(self, layer: str) -> "LayerAnalysisResult":
        return self.service.analyze_layer(layer)
