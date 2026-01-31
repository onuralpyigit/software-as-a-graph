"""
Graph Analysis Layers

Defines the multi-layer graph model for distributed pub-sub system analysis.

Layers:
    - app: Application layer (app_to_app dependencies)
    - infra: Infrastructure layer (node_to_node dependencies)
    - mw: Middleware layer (app_to_broker + node_to_broker dependencies)
    - system: Complete system (all layers combined)

Each layer focuses on specific DEPENDS_ON relationships and component types,
enabling targeted analysis for reliability, maintainability, and availability.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Dict, Any, List


class AnalysisLayer(Enum):
    """
    Enumeration of analysis layers for multi-layer graph model.
    
    Each layer represents a specific view of the distributed system:
    - APP: Service-level dependencies (who calls whom)
    - INFRA: Network topology (physical/virtual connectivity)
    - MW: Middleware coupling (both app and node to broker)
    - SYSTEM: Complete system view
    """
    APP = "app"
    INFRA = "infra"
    MW = "mw"
    SYSTEM = "system"
    
    @classmethod
    def from_string(cls, value: str) -> "AnalysisLayer":
        """Convert string to AnalysisLayer, supporting aliases."""
        aliases = {
            # App layer aliases
            "application": cls.APP,
            # Infra layer aliases
            "infrastructure": cls.INFRA,
            # Middleware layer aliases (including legacy names)
            "middleware": cls.MW,
            "mw-app": cls.MW,  # Legacy alias
            "mw-infra": cls.MW,  # Legacy alias
            "middleware-app": cls.MW,
            "middleware-infra": cls.MW,
            "app-broker": cls.MW,
            "app_broker": cls.MW,
            "node-broker": cls.MW,
            "node_broker": cls.MW,
            # System layer aliases
            "complete": cls.SYSTEM,
            "all": cls.SYSTEM,
        }
        normalized = value.lower().strip()
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError:
            valid = [l.value for l in cls] + list(aliases.keys())
            raise ValueError(f"Unknown layer '{value}'. Valid: {valid}")


@dataclass(frozen=True)
class LayerDefinition:
    """
    Definition of a graph analysis layer.
    
    Attributes:
        name: Human-readable layer name
        description: What this layer analyzes
        component_types: Types of vertices included
        dependency_types: Types of DEPENDS_ON edges included
        focus_metrics: Key metrics for this layer
        quality_focus: Primary quality dimension (R, M, or A)
    """
    name: str
    description: str
    component_types: FrozenSet[str]
    dependency_types: FrozenSet[str]
    focus_metrics: tuple
    quality_focus: str  # "reliability", "maintainability", "availability"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "component_types": list(self.component_types),
            "dependency_types": list(self.dependency_types),
            "focus_metrics": list(self.focus_metrics),
            "quality_focus": self.quality_focus,
        }


# Layer definitions with specific DEPENDS_ON relationship types
LAYER_DEFINITIONS: Dict[AnalysisLayer, LayerDefinition] = {
    AnalysisLayer.APP: LayerDefinition(
        name="Application Layer",
        description="Service-level reliability analysis via app_to_app dependencies",
        component_types=frozenset({"Application"}),
        dependency_types=frozenset({"app_to_app"}),
        focus_metrics=("pagerank", "reverse_pagerank", "in_degree", "betweenness"),
        quality_focus="reliability",
    ),
    
    AnalysisLayer.INFRA: LayerDefinition(
        name="Infrastructure Layer",
        description="Network topology resilience via node_to_node dependencies",
        component_types=frozenset({"Node"}),
        dependency_types=frozenset({"node_to_node"}),
        focus_metrics=("betweenness", "clustering", "articulation_point", "bridges"),
        quality_focus="availability",
    ),
    
    AnalysisLayer.MW: LayerDefinition(
        name="Middleware Layer",
        description="Broker coupling analysis via app_to_broker and node_to_broker dependencies",
        component_types=frozenset({"Application", "Broker", "Node"}),
        dependency_types=frozenset({"app_to_broker", "node_to_broker"}),
        focus_metrics=("in_degree", "pagerank", "betweenness", "clustering"),
        quality_focus="maintainability",
    ),
    
    AnalysisLayer.SYSTEM: LayerDefinition(
        name="Complete System",
        description="System-wide analysis across all layers and dependency types",
        component_types=frozenset({"Application", "Broker", "Node", "Topic", "Library"}),
        dependency_types=frozenset({"app_to_app", "app_to_broker", "node_to_node", "node_to_broker"}),
        focus_metrics=("pagerank", "betweenness", "articulation_point", "clustering"),
        quality_focus="overall",
    ),
}


def get_layer_definition(layer: AnalysisLayer) -> LayerDefinition:
    """Get the definition for a specific layer."""
    return LAYER_DEFINITIONS[layer]


def get_all_layers() -> List[AnalysisLayer]:
    """Get all available analysis layers."""
    return list(AnalysisLayer)


def get_primary_layers() -> List[AnalysisLayer]:
    """Get the primary layers for standard analysis."""
    return [AnalysisLayer.APP, AnalysisLayer.INFRA, AnalysisLayer.MW, AnalysisLayer.SYSTEM]


def list_layers() -> str:
    """Return a formatted string describing all available layers."""
    lines = ["Available analysis layers:", ""]
    for layer in AnalysisLayer:
        defn = LAYER_DEFINITIONS[layer]
        dep_types = ", ".join(sorted(defn.dependency_types))
        lines.append(f"  {layer.value:8} - {defn.name}")
        lines.append(f"             Dependencies: {dep_types}")
        lines.append(f"             Focus: {defn.quality_focus}")
        lines.append("")
    return "\n".join(lines)


# Dependency type to layer mapping for quick lookups
DEPENDENCY_TO_LAYER: Dict[str, AnalysisLayer] = {
    "app_to_app": AnalysisLayer.APP,
    "node_to_node": AnalysisLayer.INFRA,
    "app_to_broker": AnalysisLayer.MW,
    "node_to_broker": AnalysisLayer.MW,
}
