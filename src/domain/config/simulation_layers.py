"""
Simulation Graph Layers

Defines layers for multi-layer simulation using RAW structural relationships.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Dict, Any, List


class SimulationLayer(Enum):
    """Enumeration of simulation layers."""
    APP = "app"
    INFRA = "infra"
    MW = "mw"
    SYSTEM = "system"
    
    @classmethod
    def from_string(cls, value: str) -> "SimulationLayer":
        """Convert string to SimulationLayer, supporting aliases."""
        aliases = {
            "application": cls.APP,
            "infrastructure": cls.INFRA,
            "middleware": cls.MW,
            "mw-app": cls.MW,
            "mw-infra": cls.MW,
            "broker": cls.MW,
            "brokers": cls.MW,
            "complete": cls.SYSTEM,
            "all": cls.SYSTEM,
        }
        normalized = value.lower().strip()
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError:
            valid = [layer.value for layer in cls] + list(aliases.keys())
            raise ValueError(f"Unknown layer '{value}'. Valid: {valid}")


@dataclass(frozen=True)
class SimulationLayerDefinition:
    """Definition of a simulation layer."""
    name: str
    description: str
    component_types: FrozenSet[str]
    analyze_types: FrozenSet[str]
    relationships: FrozenSet[str]
    cascade_rules: FrozenSet[str]
    focus_metrics: tuple
    
    @property
    def types_to_analyze(self) -> FrozenSet[str]:
        return self.analyze_types
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "component_types": list(self.component_types),
            "analyze_types": list(self.analyze_types),
            "relationships": list(self.relationships),
            "cascade_rules": list(self.cascade_rules),
            "focus_metrics": list(self.focus_metrics),
        }


SIMULATION_LAYERS: Dict[SimulationLayer, SimulationLayerDefinition] = {
    SimulationLayer.APP: SimulationLayerDefinition(
        name="Application Layer",
        description="Simulate applications via PUBLISHES_TO/SUBSCRIBES_TO/USES relationships",
        component_types=frozenset({"Application", "Topic", "Library"}),
        analyze_types=frozenset({"Application"}),
        relationships=frozenset({"PUBLISHES_TO", "SUBSCRIBES_TO", "USES"}),
        cascade_rules=frozenset({"logical"}),
        focus_metrics=("throughput", "delivery_rate", "latency", "message_drops"),
    ),
    
    SimulationLayer.INFRA: SimulationLayerDefinition(
        name="Infrastructure Layer",
        description="Simulate nodes via RUNS_ON/CONNECTS_TO relationships",
        component_types=frozenset({"Node", "Application", "Broker"}),
        analyze_types=frozenset({"Node"}),
        relationships=frozenset({"RUNS_ON", "CONNECTS_TO"}),
        cascade_rules=frozenset({"physical", "network"}),
        focus_metrics=("reachability_loss", "fragmentation", "cascade_depth"),
    ),
    
    SimulationLayer.MW: SimulationLayerDefinition(
        name="Middleware Layer",
        description="Simulate brokers via ROUTES/PUBLISHES_TO/SUBSCRIBES_TO relationships",
        component_types=frozenset({"Broker", "Topic", "Application"}),
        analyze_types=frozenset({"Broker"}),
        relationships=frozenset({"ROUTES", "PUBLISHES_TO", "SUBSCRIBES_TO"}),
        cascade_rules=frozenset({"logical"}),
        focus_metrics=("throughput", "routing_efficiency", "message_drops"),
    ),
    
    SimulationLayer.SYSTEM: SimulationLayerDefinition(
        name="Complete System",
        description="Simulate all components across all relationship types",
        component_types=frozenset({"Application", "Broker", "Node", "Topic", "Library"}),
        analyze_types=frozenset({"Application", "Broker", "Node"}),
        relationships=frozenset({"PUBLISHES_TO", "SUBSCRIBES_TO", "ROUTES", "RUNS_ON", "CONNECTS_TO", "USES"}),
        cascade_rules=frozenset({"physical", "logical", "network"}),
        focus_metrics=("throughput", "reachability_loss", "fragmentation", "composite_impact"),
    ),
}


def get_simulation_layer_definition(layer: SimulationLayer) -> SimulationLayerDefinition:
    """Get the definition for a specific layer."""
    return SIMULATION_LAYERS[layer]


def get_simulation_layers() -> List[SimulationLayer]:
    """Get all available simulation layers."""
    return list(SimulationLayer)
