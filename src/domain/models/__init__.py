"""
Domain Models Package

Pure domain entities with no infrastructure dependencies.
Re-exports all domain models for convenient imports.
"""

# Core graph models
from .graph import GraphData, ComponentData, EdgeData
from .value_objects import QoSPolicy
from .entities import GraphEntity, Application, Broker, Topic, Node, Library
from .enums import VertexType, EdgeType, DependencyType, ApplicationType

# Criticality
from .criticality import (
    CriticalityLevel, BoxPlotStats, ClassifiedItem, ClassificationResult
)

# Analysis metrics
from .metrics import (
    StructuralMetrics, EdgeMetrics, GraphSummary,
    QualityScores, QualityLevels,
    ComponentQuality, EdgeQuality, ClassificationSummary
)

# Analysis results
from .results import LayerAnalysisResult, MultiLayerAnalysisResult

# Simulation models
from .simulation.graph import SimulationGraph
from .simulation.types import ComponentState, RelationType, FailureMode, CascadeRule, EventType
from .simulation.components import TopicInfo, ComponentInfo
from .simulation.metrics import LayerMetrics, ComponentCriticality, EdgeCriticality, SimulationReport

# Validation models  
from .validation.metrics import (
    ValidationTargets, CorrelationMetrics, ErrorMetrics, 
    ClassificationMetrics, RankingMetrics
)
from .validation.results import (
    ComponentComparison, ValidationGroupResult, ValidationResult,
    LayerValidationResult, PipelineResult
)

__all__ = [
    # Graph data
    "GraphData", "ComponentData", "EdgeData",
    # Value objects
    "QoSPolicy",
    # Entities
    "GraphEntity", "Application", "Broker", "Topic", "Node", "Library",
    # Enums
    "VertexType", "EdgeType", "DependencyType", "ApplicationType",
    # Criticality
    "CriticalityLevel", "BoxPlotStats", "ClassifiedItem", "ClassificationResult",
    # Analysis metrics
    "StructuralMetrics", "EdgeMetrics", "GraphSummary",
    "QualityScores", "QualityLevels",
    "ComponentQuality", "EdgeQuality", "ClassificationSummary",
    # Analysis results
    "LayerAnalysisResult", "MultiLayerAnalysisResult",
    # Simulation
    "SimulationGraph",
    "ComponentState", "RelationType", "FailureMode", "CascadeRule", "EventType",
    "TopicInfo", "ComponentInfo",
    "LayerMetrics", "ComponentCriticality", "EdgeCriticality", "SimulationReport",
    # Validation
    "ValidationTargets", "CorrelationMetrics", "ErrorMetrics",
    "ClassificationMetrics", "RankingMetrics",
    "ComponentComparison", "ValidationGroupResult", "ValidationResult",
    "LayerValidationResult", "PipelineResult",
]
