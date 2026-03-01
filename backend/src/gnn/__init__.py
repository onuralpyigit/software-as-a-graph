"""
gnn — Graph Neural Network extension for Software-as-a-Graph
=============================================================

This package integrates heterogeneous Graph Attention Networks into the
existing six-step criticality prediction pipeline, adding:

* **Step 3.5**: GNN-based node and edge criticality prediction
* **Ensemble**: Learned convex combination of GNN + RMAV scores
* **Validation parity**: Same metrics (Spearman ρ, F1, NDCG@K) for
  direct comparison with RMAV baseline

Quick start
-----------
Training on a labelled graph::

    from src.gnn import GNNService

    service = GNNService()
    result  = service.train(
        graph              = nx_graph,
        structural_metrics = structural_dict,   # from StructuralAnalyzer
        simulation_results = sim_results,       # from SimulationService
        rmav_scores        = rmav_dict,         # from AnalysisService
    )
    print(result.top_critical_nodes(10))
    print(result.top_critical_edges(10))

Inference on a new graph (models already trained)::

    service = GNNService.from_checkpoint("output/gnn_checkpoints")
    result  = service.predict(
        graph              = new_nx_graph,
        structural_metrics = new_structural_dict,
        rmav_scores        = new_rmav_dict,
    )

See ``docs/gnn-integration.md`` for full methodology documentation.
"""

from .data_preparation import (
    GraphConversionResult,
    NODE_FEATURE_DIM,
    NODE_TYPES,
    EDGE_TYPES,
    EDGE_FEATURE_DIM,
    create_node_splits,
    extract_rmav_scores_dict,
    extract_simulation_dict,
    extract_structural_metrics_dict,
    networkx_to_hetero_data,
)
from .gnn_service import (
    GNNAnalysisResult,
    GNNCriticalityScore,
    GNNEdgeCriticalityScore,
    GNNService,
)
from .models import (
    CriticalityLoss,
    EdgeCriticalityGNN,
    EnsembleGNN,
    NodeCriticalityGNN,
    ResidualMLP,
    build_edge_gnn,
    build_node_gnn,
)
from .trainer import (
    EvalMetrics,
    GNNTrainer,
    evaluate,
)

__all__ = [
    # Data preparation
    "GraphConversionResult",
    "NODE_TYPES",
    "EDGE_TYPES",
    "NODE_FEATURE_DIM",
    "EDGE_FEATURE_DIM",
    "networkx_to_hetero_data",
    "create_node_splits",
    "extract_simulation_dict",
    "extract_structural_metrics_dict",
    "extract_rmav_scores_dict",
    # Models
    "NodeCriticalityGNN",
    "EdgeCriticalityGNN",
    "EnsembleGNN",
    "ResidualMLP",
    "CriticalityLoss",
    "build_node_gnn",
    "build_edge_gnn",
    # Trainer
    "GNNTrainer",
    "EvalMetrics",
    "evaluate",
    # Service
    "GNNService",
    "GNNAnalysisResult",
    "GNNCriticalityScore",
    "GNNEdgeCriticalityScore",
]
