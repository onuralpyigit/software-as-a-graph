"""
Simulation Use Case Port

Interface defining the contract for simulation operations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any


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
