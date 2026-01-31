"""
Domain Configuration Package

Layer definitions and configuration for analysis and simulation.
"""

from .layers import (
    AnalysisLayer, LayerDefinition, LAYER_DEFINITIONS,
    get_layer_definition, get_all_layers, get_primary_layers, list_layers,
    DEPENDENCY_TO_LAYER
)
from .simulation_layers import (
    SimulationLayer, SimulationLayerDefinition, SIMULATION_LAYERS,
    get_simulation_layer_definition, get_simulation_layers
)

__all__ = [
    # Analysis layers
    "AnalysisLayer", "LayerDefinition", "LAYER_DEFINITIONS",
    "get_layer_definition", "get_all_layers", "get_primary_layers", "list_layers",
    "DEPENDENCY_TO_LAYER",
    # Simulation layers
    "SimulationLayer", "SimulationLayerDefinition", "SIMULATION_LAYERS",
    "get_simulation_layer_definition", "get_simulation_layers",
]
