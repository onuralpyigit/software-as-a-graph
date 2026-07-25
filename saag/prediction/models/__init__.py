"""
saag/prediction/models/__init__.py
===================================
Re-exports all public symbols from the original flat models.py (now models/core.py)
so that existing imports continue to work unchanged.

New submodule:
  saag.prediction.models.baselines — HomogeneousGAT_Unweighted, _ScalarWeighted
"""

# Re-export everything from the original flat module (backward compatibility).
# Using star import to avoid having to enumerate every symbol manually —
# the original models.py is now models/core.py.
from .core import *  # noqa: F401, F403
from .core import (  # noqa: F401 — explicit imports for IDE / type checking
    QualityAnalysisResult,
    DetectedProblem,
    ProblemSummary,
    NodeCriticalityGNN,
    EdgeCriticalityGNN,
    EnsembleGNN,
    CriticalityLoss,
    ResidualMLP,
    EdgeFeatureEncoder,
    TypedEdgeEncoder,
    build_node_gnn,
    build_edge_gnn,
)
