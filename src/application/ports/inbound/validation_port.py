"""
Validation Use Case Port

Interface defining the contract for validation operations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any


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
