"""
Validation Service

Application service implementing IValidationUseCase.
Orchestrates validation of analysis and simulation results.
"""

from typing import List, Optional, Any

from src.application.ports.inbound.validation_port import IValidationUseCase
from src.application.ports.inbound.analysis_port import IAnalysisUseCase
from src.application.ports.inbound.simulation_port import ISimulationUseCase

# Import legacy service for compatibility during migration
from src.services.validation_service import ValidationService as LegacyValidationService


class ValidationService(IValidationUseCase):
    """
    Application service for validation.
    
    Implements IValidationUseCase port and delegates to domain services.
    Uses dependency injection for service access.
    """
    
    def __init__(
        self, 
        analysis_service: IAnalysisUseCase,
        simulation_service: ISimulationUseCase,
        targets: Optional[List[str]] = None
    ):
        """
        Initialize validation service.
        
        Args:
            analysis_service: Analysis use case
            simulation_service: Simulation use case
            targets: Optional list of validation targets
        """
        self._analysis = analysis_service
        self._simulation = simulation_service
        # Use legacy service internally during migration
        self._legacy = LegacyValidationService(
            analysis_service=analysis_service,
            simulation_service=simulation_service,
            targets=targets
        )
    
    def validate_layers(self, layers: Optional[List[str]] = None) -> Any:
        """Validate analysis and simulation for specified layers."""
        return self._legacy.validate_layers(layers)
    
    def validate_single_layer(self, layer: str) -> Any:
        """Validate a single layer."""
        return self._legacy.validate_single_layer(layer)
