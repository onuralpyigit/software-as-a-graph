"""
Outbound Ports (Secondary/Driven Ports)

Interfaces for infrastructure that the application drives.
"""

from .graph_repository import IGraphRepository
from .report_generator import IReportGenerator
from .metrics_exporter import IMetricsExporter

__all__ = [
    "IGraphRepository",
    "IReportGenerator",
    "IMetricsExporter",
]
