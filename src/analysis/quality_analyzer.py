"""
Compatibility shim: src.analysis.quality_analyzer

Re-exports QualityAnalyzer from its new location.
"""
from src.domain.services.quality_analyzer import QualityAnalyzer

__all__ = ["QualityAnalyzer"]
