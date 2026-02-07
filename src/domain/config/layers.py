"""
Graph Layer Definitions

Unified layer definitions for analysis and simulation.
This is the **canonical** layer definition used by all CLI scripts and services.

Analysis Layers and their DEPENDS_ON relationship types:
    app    → app_to_app               Analyse Applications
    infra  → node_to_node             Analyse Nodes
    mw     → app_to_broker,           Analyse Brokers
             node_to_broker
    system → all four types           Analyse all components

Simulation Layers use RAW structural relationships (PUBLISHES_TO, RUNS_ON, etc.)
for failure cascade and event propagation simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, List, Any, Optional


# ---------------------------------------------------------------------------
# Unified Layer Enum
# ---------------------------------------------------------------------------

class AnalysisLayer(Enum):
    """
    Enumeration of graph layers for both analysis and simulation.

    Each layer represents a specific view of the distributed system:
        APP    — Service-level dependencies
        INFRA  — Network / infrastructure topology
        MW     — Middleware / broker coupling
        SYSTEM — Complete system view
    """
    APP = "app"
    INFRA = "infra"
    MW = "mw"
    SYSTEM = "system"

    @classmethod
    def from_string(cls, value: str) -> AnalysisLayer:
        """Convert a string to AnalysisLayer, supporting common aliases."""
        _ALIASES: Dict[str, AnalysisLayer] = {
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
        key = value.lower().strip()
        if key in _ALIASES:
            return _ALIASES[key]
        try:
            return cls(key)
        except ValueError:
            valid = sorted({l.value for l in cls} | set(_ALIASES))
            raise ValueError(f"Unknown layer '{value}'. Valid: {valid}")


# Backward compatibility alias
SimulationLayer = AnalysisLayer


# ---------------------------------------------------------------------------
# Analysis Layer Definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayerDefinition:
    """
    Definition of a graph analysis layer.

    Attributes:
        name:             Human-readable layer name
        description:      What this layer analyses
        component_types:  Vertex labels included when building the subgraph
        dependency_types: DEPENDS_ON subtypes included
        focus_metrics:    Key centrality metrics for this layer
        quality_focus:    Primary quality dimension (reliability / maintainability / …)
        analyze_types:    If set, only these component types appear in results.
                          Defaults to *component_types* when ``None``.
    """
    name: str
    description: str
    component_types: FrozenSet[str]
    dependency_types: FrozenSet[str]
    focus_metrics: tuple
    quality_focus: str
    analyze_types: Optional[FrozenSet[str]] = None

    @property
    def types_to_analyze(self) -> FrozenSet[str]:
        """Component types that should appear in analysis results."""
        return self.analyze_types if self.analyze_types else self.component_types

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "component_types": sorted(self.component_types),
            "analyze_types": sorted(self.types_to_analyze),
            "dependency_types": sorted(self.dependency_types),
            "focus_metrics": list(self.focus_metrics),
            "quality_focus": self.quality_focus,
        }


# ---------------------------------------------------------------------------
# Analysis Layer Registry
# ---------------------------------------------------------------------------

LAYER_DEFINITIONS: Dict[AnalysisLayer, LayerDefinition] = {
    AnalysisLayer.APP: LayerDefinition(
        name="Application Layer",
        description="Analyse Applications via app_to_app dependencies for reliability",
        component_types=frozenset({"Application"}),
        dependency_types=frozenset({"app_to_app"}),
        focus_metrics=("pagerank", "reverse_pagerank", "in_degree", "betweenness"),
        quality_focus="reliability",
    ),
    AnalysisLayer.INFRA: LayerDefinition(
        name="Infrastructure Layer",
        description="Analyse Nodes via node_to_node dependencies for availability",
        component_types=frozenset({"Node"}),
        dependency_types=frozenset({"node_to_node"}),
        focus_metrics=("betweenness", "clustering", "articulation_point", "bridges"),
        quality_focus="availability",
    ),
    AnalysisLayer.MW: LayerDefinition(
        name="Middleware Layer",
        description="Analyse Brokers via app_to_broker and node_to_broker dependencies for maintainability",
        component_types=frozenset({"Application", "Broker", "Node"}),
        dependency_types=frozenset({"app_to_broker", "node_to_broker"}),
        focus_metrics=("in_degree", "pagerank", "betweenness", "clustering"),
        quality_focus="maintainability",
        analyze_types=frozenset({"Broker"}),
    ),
    AnalysisLayer.SYSTEM: LayerDefinition(
        name="Complete System",
        description="Analyse all components across all dependency types",
        component_types=frozenset({"Application", "Broker", "Node", "Topic", "Library"}),
        dependency_types=frozenset({"app_to_app", "app_to_broker", "node_to_node", "node_to_broker"}),
        focus_metrics=("pagerank", "betweenness", "articulation_point", "clustering"),
        quality_focus="overall",
    ),
}

# Backward compatibility alias
LAYERS = LAYER_DEFINITIONS


# ---------------------------------------------------------------------------
# Simulation Layer Definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationLayerDefinition:
    """
    Definition of a simulation layer using raw structural relationships.

    Attributes:
        name:             Human-readable layer name
        description:      What this layer simulates
        component_types:  Vertex labels included when building the simulation graph
        analyze_types:    Component types to analyze in simulation results
        relationships:    Raw relationship types (PUBLISHES_TO, RUNS_ON, etc.)
        cascade_rules:    Types of cascade propagation (physical, logical, network)
        focus_metrics:    Key simulation metrics for this layer
    """
    name: str
    description: str
    component_types: FrozenSet[str]
    analyze_types: FrozenSet[str]
    relationships: FrozenSet[str]
    cascade_rules: FrozenSet[str]
    focus_metrics: tuple

    @property
    def types_to_analyze(self) -> FrozenSet[str]:
        """Component types that should appear in simulation results."""
        return self.analyze_types

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "component_types": sorted(self.component_types),
            "analyze_types": sorted(self.analyze_types),
            "relationships": sorted(self.relationships),
            "cascade_rules": sorted(self.cascade_rules),
            "focus_metrics": list(self.focus_metrics),
        }


# ---------------------------------------------------------------------------
# Simulation Layer Registry
# ---------------------------------------------------------------------------

SIMULATION_LAYERS: Dict[AnalysisLayer, SimulationLayerDefinition] = {
    AnalysisLayer.APP: SimulationLayerDefinition(
        name="Application Layer",
        description="Simulate applications via PUBLISHES_TO/SUBSCRIBES_TO/USES relationships",
        component_types=frozenset({"Application", "Topic", "Library"}),
        analyze_types=frozenset({"Application"}),
        relationships=frozenset({"PUBLISHES_TO", "SUBSCRIBES_TO", "USES"}),
        cascade_rules=frozenset({"logical"}),
        focus_metrics=("throughput", "delivery_rate", "latency", "message_drops"),
    ),
    AnalysisLayer.INFRA: SimulationLayerDefinition(
        name="Infrastructure Layer",
        description="Simulate nodes via RUNS_ON/CONNECTS_TO relationships",
        component_types=frozenset({"Node", "Application", "Broker"}),
        analyze_types=frozenset({"Node"}),
        relationships=frozenset({"RUNS_ON", "CONNECTS_TO"}),
        cascade_rules=frozenset({"physical", "network"}),
        focus_metrics=("reachability_loss", "fragmentation", "cascade_depth"),
    ),
    AnalysisLayer.MW: SimulationLayerDefinition(
        name="Middleware Layer",
        description="Simulate brokers via ROUTES/PUBLISHES_TO/SUBSCRIBES_TO relationships",
        component_types=frozenset({"Broker", "Topic", "Application"}),
        analyze_types=frozenset({"Broker"}),
        relationships=frozenset({"ROUTES", "PUBLISHES_TO", "SUBSCRIBES_TO"}),
        cascade_rules=frozenset({"logical"}),
        focus_metrics=("throughput", "routing_efficiency", "message_drops"),
    ),
    AnalysisLayer.SYSTEM: SimulationLayerDefinition(
        name="Complete System",
        description="Simulate all components across all relationship types",
        component_types=frozenset({"Application", "Broker", "Node", "Topic", "Library"}),
        analyze_types=frozenset({"Application", "Broker", "Node"}),
        relationships=frozenset({"PUBLISHES_TO", "SUBSCRIBES_TO", "ROUTES", "RUNS_ON", "CONNECTS_TO", "USES"}),
        cascade_rules=frozenset({"physical", "logical", "network"}),
        focus_metrics=("throughput", "reachability_loss", "fragmentation", "composite_impact"),
    ),
}


# ---------------------------------------------------------------------------
# Analysis Layer Helpers
# ---------------------------------------------------------------------------

def get_layer_definition(layer: AnalysisLayer) -> LayerDefinition:
    """Get the analysis definition for a specific layer."""
    return LAYER_DEFINITIONS[layer]


def get_all_layers() -> List[AnalysisLayer]:
    """All layers (including system)."""
    return list(AnalysisLayer)


def get_primary_layers() -> List[AnalysisLayer]:
    """The four primary layers used by ``analyze_all_layers``."""
    return [AnalysisLayer.APP, AnalysisLayer.INFRA, AnalysisLayer.MW, AnalysisLayer.SYSTEM]


def list_layers() -> str:
    """Return a formatted string describing all available layers."""
    lines = ["Available analysis layers:", ""]
    for layer in AnalysisLayer:
        defn = LAYER_DEFINITIONS[layer]
        deps = ", ".join(sorted(defn.dependency_types))
        analyze = ", ".join(sorted(defn.types_to_analyze))
        lines.extend([
            f"  {layer.value:8} — {defn.name}",
            f"             Analyses:     {analyze}",
            f"             Dependencies: {deps}",
            f"             Focus:        {defn.quality_focus}",
            "",
        ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simulation Layer Helpers
# ---------------------------------------------------------------------------

def get_simulation_layer_definition(layer: AnalysisLayer) -> SimulationLayerDefinition:
    """Get the simulation definition for a specific layer."""
    return SIMULATION_LAYERS[layer]


def get_simulation_layers() -> List[AnalysisLayer]:
    """Get all available simulation layers."""
    return list(AnalysisLayer)


# ---------------------------------------------------------------------------
# Lookup Utilities
# ---------------------------------------------------------------------------

# Quick lookup: dependency_type → layer
DEPENDENCY_TO_LAYER: Dict[str, AnalysisLayer] = {
    "app_to_app": AnalysisLayer.APP,
    "node_to_node": AnalysisLayer.INFRA,
    "app_to_broker": AnalysisLayer.MW,
    "node_to_broker": AnalysisLayer.MW,
}


def get_layer(layer: AnalysisLayer) -> LayerDefinition:
    """Alias for get_layer_definition for shorter imports."""
    return LAYER_DEFINITIONS[layer]


def resolve_layer(value: str) -> AnalysisLayer:
    """Alias for AnalysisLayer.from_string for shorter imports."""
    return AnalysisLayer.from_string(value)