"""
Static System Analyzer - Pipeline Logic Package

This package contains the core business logic for the static analysis pipeline:
- cloner: Repository cloning functionality
- analyzer: Project analysis and topic extraction
- aggregator: Data aggregation and JSON generation
- stat: Statistical analysis for design defect detection
- structural: Structural analysis (metrics, patterns, anomaly scoring)
"""

from .cloner import ClonerService
from .analyzer import AnalyzerService
from .aggregator import AggregatorService
from .stat import StatService
from .structural import StructuralService

__all__ = [
    "ClonerService",
    "AnalyzerService", 
    "AggregatorService",
    "StatService",
    "StructuralService",
]
