"""
Compatibility shim: src.analysis.structural_analyzer

Re-exports StructuralAnalyzer from its new location.
"""
from src.domain.services.structural_analyzer import StructuralAnalyzer

__all__ = ["StructuralAnalyzer"]
