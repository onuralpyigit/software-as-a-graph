"""
Graph Analysis Layers

Defines the multi-layer graph model for distributed pub-sub system analysis.

Layers:
    - app: Application layer (app_to_app dependencies)
    - infra: Infrastructure layer (node_to_node dependencies)
    - mw-app: Middleware-Application layer (app_to_broker dependencies)
    - mw-infra: Middleware-Infrastructure layer (node_to_broker dependencies)
    - system: Complete system (all layers combined)

Each layer focuses on specific DEPENDS_ON relationships and component types,
enabling targeted analysis for reliability, maintainability, and availability.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Dict, Any


class AnalysisLayer(Enum):
    """
    Enumeration of analysis layers for multi-layer graph model.
    
    Each layer represents a specific view of the distributed system:
    - APP: Service-level dependencies (who calls whom)
    - INFRA: Network topology (physical/virtual connectivity)
    - MW_APP: Application-to-middleware coupling
    - MW_INFRA: Infrastructure-to-middleware coupling
    - SYSTEM: Complete system view
    """
    APP = "app"
    INFRA = "infra"
    MW_APP = "mw-app"
    MW_INFRA = "mw-infra"
    SYSTEM = "system"
    
    @classmethod
    def from_string(cls, value: str) -> "AnalysisLayer":
        """Convert string to AnalysisLayer, supporting aliases."""
        aliases = {
            "application": cls.APP,
            "infrastructure": cls.INFRA,
            "middleware-app": cls.MW_APP,
            "middleware-application": cls.MW_APP,
            "app-broker": cls.MW_APP,
            "app_broker": cls.MW_APP,
            "middleware-infra": cls.MW_INFRA,
            "middleware-infrastructure": cls.MW_INFRA,
            "node-broker": cls.MW_INFRA,
            "node_broker": cls.MW_INFRA,
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
    
    AnalysisLayer.MW_APP: LayerDefinition(
        name="Middleware-Application Layer",
        description="Application-to-broker coupling via app_to_broker dependencies",
        component_types=frozenset({"Application", "Broker"}),
        dependency_types=frozenset({"app_to_broker"}),
        focus_metrics=("in_degree", "pagerank", "betweenness"),
        quality_focus="maintainability",
    ),
    
    AnalysisLayer.MW_INFRA: LayerDefinition(
        name="Middleware-Infrastructure Layer",
        description="Infrastructure-to-broker coupling via node_to_broker dependencies",
        component_types=frozenset({"Node", "Broker"}),
        dependency_types=frozenset({"node_to_broker"}),
        focus_metrics=("betweenness", "clustering", "articulation_point"),
        quality_focus="availability",
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


def get_all_layers() -> list[AnalysisLayer]:
    """Get all available analysis layers."""
    return list(AnalysisLayer)


def get_primary_layers() -> list[AnalysisLayer]:
    """Get the primary layers for standard analysis (excluding middleware layers)."""
    return [AnalysisLayer.APP, AnalysisLayer.INFRA, AnalysisLayer.SYSTEM]


# Dependency type to layer mapping for quick lookups
DEPENDENCY_TO_LAYER: Dict[str, AnalysisLayer] = {
    "app_to_app": AnalysisLayer.APP,
    "node_to_node": AnalysisLayer.INFRA,
    "app_to_broker": AnalysisLayer.MW_APP,
    "node_to_broker": AnalysisLayer.MW_INFRA,
}
