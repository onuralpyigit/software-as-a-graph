"""
Analysis Use Case Port

Interface defining the contract for graph analysis operations.
"""

from abc import ABC, abstractmethod
from typing import Union, Any

from src.domain.models import LayerAnalysisResult, MultiLayerAnalysisResult
from src.domain.config import AnalysisLayer


class IAnalysisUseCase(ABC):
    """
    Inbound port for graph analysis use cases.
    
    Defines the contract for analyzing graph layers for reliability,
    maintainability, and availability concerns.
    """
    
    @abstractmethod
    def analyze_layer(self, layer: Union[str, AnalysisLayer]) -> LayerAnalysisResult:
        """
        Analyze a specific graph layer.
        
        Args:
            layer: Layer to analyze (app, infra, mw, system)
            
        Returns:
            Complete analysis result for the layer
        """
        pass
    
    @abstractmethod
    def analyze_all_layers(self) -> MultiLayerAnalysisResult:
        """
        Analyze all primary graph layers.
        
        Returns:
            Analysis results across all layers with cross-layer insights
        """
        pass
    
    @abstractmethod
    def export_results(self, results: Any, output_path: str) -> None:
        """
        Export analysis results to a file.
        
        Args:
            results: Analysis results to export
            output_path: Path to output file
        """
        pass
