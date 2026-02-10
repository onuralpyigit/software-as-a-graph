"""
Compatibility shim: src.analysis.problem_detector

Re-exports ProblemDetector from its new location.
"""
from src.domain.services.problem_detector import ProblemDetector

__all__ = ["ProblemDetector"]
