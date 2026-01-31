"""
Simulation Models Package

Domain models for simulation scenarios and results.
"""

from .graph import SimulationGraph
from .layers import SimulationLayer, SimulationLayerDefinition, SIMULATION_LAYERS, get_layer_definition
from .types import ComponentState, RelationType, FailureMode, CascadeRule, EventType
from .components import TopicInfo, ComponentInfo
from .metrics import LayerMetrics, ComponentCriticality, SimulationReport

__all__ = [
    "SimulationGraph",
    "SimulationLayer",
    "SimulationLayerDefinition",
    "SIMULATION_LAYERS",
    "get_layer_definition",
    "ComponentState",
    "RelationType",
    "FailureMode",
    "CascadeRule",
    "EventType",
    "TopicInfo",
    "ComponentInfo",
    "LayerMetrics",
    "ComponentCriticality",
    "SimulationReport",
]
