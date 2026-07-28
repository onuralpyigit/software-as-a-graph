"""
saag/prediction/models/__init__.py
===================================
GNN architecture for criticality prediction.

  saag.prediction.models.core      — HGT node/edge models, loss, builders
  saag.prediction.models.baselines — HomogeneousGAT_Unweighted, _ScalarWeighted
"""

from .core import (
    EDGE_FEATURE_DIM,
    NODE_TYPE_TO_DIM,
    NUM_LABEL_DIMS,
    CriticalityLoss,
    EdgeCriticalityGNN,
    EdgeFeatureEncoder,
    NodeCriticalityGNN,
    ResidualMLP,
    TypedEdgeEncoder,
    build_edge_gnn,
    build_node_gnn,
)

__all__ = [
    "EDGE_FEATURE_DIM",
    "NODE_TYPE_TO_DIM",
    "NUM_LABEL_DIMS",
    "CriticalityLoss",
    "EdgeCriticalityGNN",
    "EdgeFeatureEncoder",
    "NodeCriticalityGNN",
    "ResidualMLP",
    "TypedEdgeEncoder",
    "build_edge_gnn",
    "build_node_gnn",
]
