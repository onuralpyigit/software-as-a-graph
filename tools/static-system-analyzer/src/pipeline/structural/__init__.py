"""
Structural analysis module for publish-subscribe architecture.

This module provides structural metric calculation, pattern detection,
and anomaly scoring based on the formal definitions in the paper.
"""

from .service import StructuralService, StructuralConfig

__all__ = ["StructuralService", "StructuralConfig"]
