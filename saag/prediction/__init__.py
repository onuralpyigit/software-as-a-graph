"""
Prediction Package (Step 3)

Rule-based RMAV scoring plus the optional GNN predictor.

The GNN symbols (``GNNService`` and friends) are resolved lazily: importing them
pulls in torch and torch_geometric, and most consumers of this package —
``PredictionService``, the CLI, the API's analysis routes — never touch the GNN.
They are still importable directly from ``saag.prediction``; the import just
does not happen until one is actually referenced.
"""

from typing import TYPE_CHECKING

from .service import PredictionService

if TYPE_CHECKING:  # pragma: no cover — for type checkers and IDEs only
    from .gnn_service import GNNService, GNNAnalysisResult, GNNCriticalityScore
    from .data_preparation import (
        extract_structural_metrics_dict,
        extract_rmav_scores_dict,
        extract_simulation_dict,
        networkx_to_hetero_data,
    )

#: Submodule -> the attributes it defines, resolved on first access.
_LAZY_MODULES = {
    ".gnn_service": ("GNNService", "GNNAnalysisResult", "GNNCriticalityScore"),
    ".data_preparation": (
        "extract_structural_metrics_dict",
        "extract_rmav_scores_dict",
        "extract_simulation_dict",
        "networkx_to_hetero_data",
    ),
}
_LAZY = {name: mod for mod, names in _LAZY_MODULES.items() for name in names}

__all__ = ["PredictionService", *_LAZY]


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value  # cache so __getattr__ runs once per symbol
    return value


def __dir__():
    return sorted(__all__)
