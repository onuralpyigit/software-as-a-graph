"""
Inbound Ports

Interfaces defining contracts for inbound use cases.
"""

from abc import ABC, abstractmethod
from typing import Union, Any, List, Optional, Dict

from src.domain.models import LayerAnalysisResult, MultiLayerAnalysisResult
from src.domain.config import AnalysisLayer


# =============================================================================
# Analysis Use Case
# =============================================================================

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


# =============================================================================
# Simulation Use Case
# =============================================================================

class ISimulationUseCase(ABC):
    """
    Inbound port for simulation use cases.
    
    Defines the contract for running event and failure simulations.
    """
    
    @abstractmethod
    def run_event_simulation(
        self, 
        source_app: str, 
        num_messages: int = 100,
        duration: float = 10.0,
        **kwargs
    ) -> Any:
        """
        Run event simulation from a specific source application.
        
        Args:
            source_app: Name of the publisher application
            num_messages: Number of messages to simulate
            duration: Simulation duration in seconds
            
        Returns:
            Event simulation result
        """
        pass
    
    @abstractmethod
    def run_failure_simulation(
        self, 
        target_id: str, 
        layer: str = "system",
        cascade_probability: float = 1.0,
        **kwargs
    ) -> Any:
        """
        Run failure simulation for a specific component.
        
        Args:
            target_id: ID of the component to fail
            layer: Layer context for simulation
            cascade_probability: Probability of cascade propagation
            
        Returns:
            Failure simulation result
        """
        pass
    
    @abstractmethod
    def run_failure_simulation_exhaustive(
        self,
        layer: str = "system",
        cascade_probability: float = 1.0,
        **kwargs
    ) -> List[Any]:
        """
        Run exhaustive failure simulation for all components in a layer.
        
        Args:
            layer: Layer to analyze
            cascade_probability: Probability of cascade propagation
            
        Returns:
            List of failure results for each component
        """
        pass

    @abstractmethod
    def run_event_simulation_all(
        self,
        num_messages: int = 100,
        duration: float = 10.0,
        layer: str = "system",
        **kwargs
    ) -> Dict[str, Any]:
        """Run event simulation for all publishers."""
        pass

    @abstractmethod
    def run_failure_simulation_monte_carlo(
        self,
        target_id: str,
        layer: str = "system",
        cascade_probability: float = 1.0,
        n_trials: int = 100,
        **kwargs
    ) -> Any:
        """Run Monte Carlo failure simulation."""
        pass

    @abstractmethod
    def classify_edges(
        self,
        layer: str = "system",
        k_factor: float = 1.5
    ) -> List[Any]:
        """Classify edges by criticality."""
        pass
    
    @abstractmethod
    def generate_report(self, layers: Optional[List[str]] = None) -> Any:
        """
        Generate comprehensive simulation report.
        
        Args:
            layers: Layers to include in report (None = all primary)
            
        Returns:
            Simulation report
        """
        pass
    
    @abstractmethod
    def classify_components(
        self, 
        layer: str = "system",
        k_factor: float = 1.5
    ) -> List[Any]:
        """
        Classify components by criticality based on simulation results.
        
        Args:
            layer: Layer to classify
            k_factor: IQR multiplier for outlier detection
            
        Returns:
            List of component criticality assessments
        """
        pass


# =============================================================================
# Validation Use Case
# =============================================================================

class IValidationUseCase(ABC):
    """
    Inbound port for validation use cases.
    
    Defines the contract for validating graph analysis and simulation results.
    """
    
    @abstractmethod
    def validate_layers(self, layers: Optional[List[str]] = None) -> Any:
        """
        Validate analysis and simulation for specified layers.
        
        Args:
            layers: Layers to validate (None = all primary)
            
        Returns:
            Validation result with pass/fail status
        """
        pass
    
    @abstractmethod
    def validate_single_layer(self, layer: str) -> Any:
        """
        Validate a single layer.
        
        Args:
            layer: Layer to validate
            
        Returns:
            Layer validation result
        """
        pass


# =============================================================================
# Pipeline Use Case (NEW)
# =============================================================================

class IPipelineUseCase(ABC):
    """
    Inbound port for the complete analysis pipeline.
    
    Orchestrates analysis, simulation, validation, and visualization.
    """
    
    @abstractmethod
    def run_pipeline(
        self,
        layers: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        **options
    ) -> Any:
        """
        Run the complete analysis pipeline.
        
        Args:
            layers: Layers to process (None = all primary)
            output_dir: Output directory for results
            options: Additional pipeline options
            
        Returns:
            Pipeline execution result
        """
        pass
