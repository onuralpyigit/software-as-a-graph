"""
Analysis Service

Application service implementing IAnalysisUseCase.
Orchestrates domain logic for graph analysis.
"""

from typing import Union, Any

from src.application.ports.inbound.analysis_port import IAnalysisUseCase
from src.application.ports.outbound.graph_repository import IGraphRepository
from src.domain.models import LayerAnalysisResult, MultiLayerAnalysisResult
from src.domain.config import AnalysisLayer

# Import legacy service for compatibility during migration
from src.services.analysis_service import AnalysisService as LegacyAnalysisService


class AnalysisService(IAnalysisUseCase):
    """
    Application service for graph analysis.
    
    Implements IAnalysisUseCase port and delegates to domain services.
    Uses dependency injection for repository access.
    """
    
    def __init__(self, repository: IGraphRepository):
        """
        Initialize analysis service.
        
        Args:
            repository: Graph repository for data access
        """
        self._repository = repository
        # Use legacy service internally during migration
        self._legacy = LegacyAnalysisService(repository=repository)
    
    def analyze_layer(self, layer: Union[str, AnalysisLayer]) -> LayerAnalysisResult:
        """Analyze a specific graph layer."""
        return self._legacy.analyze_layer(layer)
    
    def analyze_all_layers(self) -> MultiLayerAnalysisResult:
        """Analyze all primary graph layers."""
        return self._legacy.analyze_all_layers()
    
    def export_results(self, results: Any, output_path: str) -> None:
        """Export analysis results to a file."""
        self._legacy.export_results(results, output_path)
