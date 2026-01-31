"""
Inbound Ports (Primary/Driving Ports)

Interfaces for use cases that drive the application.
"""

from .analysis_port import IAnalysisUseCase
from .simulation_port import ISimulationUseCase
from .validation_port import IValidationUseCase

__all__ = [
    "IAnalysisUseCase",
    "ISimulationUseCase", 
    "IValidationUseCase",
]
