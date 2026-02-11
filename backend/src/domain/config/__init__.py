"""
Domain Configuration Package

Unified layer definitions for analysis and simulation.
"""

from .layers import (
    # Core types
    AnalysisLayer, SimulationLayer,
    LayerDefinition, SimulationLayerDefinition,
    # Analysis registry
    LAYER_DEFINITIONS, LAYERS,
    get_layer_definition, get_layer, get_all_layers, get_primary_layers, list_layers,
    # Simulation registry
    SIMULATION_LAYERS,
    get_simulation_layer_definition, get_simulation_layers,
    # Utilities
    DEPENDENCY_TO_LAYER, resolve_layer,
)

__all__ = [
    # Core types
    "AnalysisLayer", "SimulationLayer",
    "LayerDefinition", "SimulationLayerDefinition",
    # Analysis registry
    "LAYER_DEFINITIONS", "LAYERS",
    "get_layer_definition", "get_layer", "get_all_layers", "get_primary_layers", "list_layers",
    # Simulation registry
    "SIMULATION_LAYERS",
    "get_simulation_layer_definition", "get_simulation_layers",
    # Utilities
    "DEPENDENCY_TO_LAYER", "resolve_layer",
]
