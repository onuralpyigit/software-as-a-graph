"""
Compatibility shim: src.analysis.classifier

Re-exports BoxPlotClassifier and CriticalityLevel from their new locations.
"""
from src.domain.services.classifier import BoxPlotClassifier
from src.domain.models.criticality import CriticalityLevel

__all__ = ["BoxPlotClassifier", "CriticalityLevel"]
