"""
Simulation Models Package

Domain models for simulation scenarios and results.
"""

from .graph import SimulationGraph
from .layers import SimulationLayer, SimulationLayerDefinition, SIMULATION_LAYERS, get_layer_definition
from .types import ComponentState, RelationType, FailureMode, CascadeRule, EventType
from .components import TopicInfo, ComponentInfo
from .metrics import LayerMetrics, ComponentCriticality, EdgeCriticality, SimulationReport

__all__ = [
    # Graph model
    "SimulationGraph",
    # Layer definitions
    "SimulationLayer",
    "SimulationLayerDefinition",
    "SIMULATION_LAYERS",
    "get_layer_definition",
    # Type enums
    "ComponentState",
    "RelationType",
    "FailureMode",
    "CascadeRule",
    "EventType",
    # Component models
    "TopicInfo",
    "ComponentInfo",
    # Metrics and results
    "LayerMetrics",
    "ComponentCriticality",
    "EdgeCriticality",
    "SimulationReport",
]