"""
Application Ports Package

Interfaces defining boundaries between application and adapters layers.
"""

# Inbound ports (use case interfaces)
from .inbound_ports import (
    IAnalysisUseCase,
    ISimulationUseCase,
    IValidationUseCase,
    IPipelineUseCase,
)

# Outbound ports (adapter interfaces)
from .outbound_ports import (
    IGraphRepository,
    IFileStore,
    IReporter,
    IMetricsExporter,
    IReportGenerator,
)

__all__ = [
    # Inbound
    "IAnalysisUseCase",
    "ISimulationUseCase",
    "IValidationUseCase",
    "IPipelineUseCase",
    # Outbound
    "IGraphRepository",
    "IFileStore",
    "IReporter",
    "IMetricsExporter",
    "IReportGenerator",
]
