"""
Analysis Module - Version 5.0

Multi-layer graph analysis for distributed pub-sub systems.

Features:
- Multi-layer analysis (by DEPENDS_ON relationship type)
- Component-type analysis (Application, Broker, Node, Topic comparison)
- Edge criticality analysis
- Box-plot statistical classification (adaptive thresholds)
- Weighted algorithm support (PageRank, Betweenness, Degree)
- Structural analysis (Articulation Points, Bridges)

Usage:
    from src.analysis import GraphAnalyzer, CriticalityLevel
    
    with GraphAnalyzer(uri, user, password) as analyzer:
        # Full analysis
        result = analyzer.analyze_full()
        
        # Layer analysis
        app_layer = analyzer.analyze_layer("application")
        
        # Component type analysis
        brokers = analyzer.analyze_component_type("Broker")
        
        # Edge analysis
        edges = analyzer.analyze_edges()

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

__version__ = "5.0.0"

# Classification
from .classifier import (
    CriticalityLevel,
    BoxPlotStats,
    ClassifiedItem,
    ClassificationResult,
    BoxPlotClassifier,
    classify_items,
    get_level_for_score,
)

# GDS Client
from .gds_client import (
    GDSClient,
    CentralityResult,
    ProjectionInfo,
    StructuralResult,
    COMPONENT_TYPES,
    DEPENDENCY_TYPES,
    LAYER_DEFINITIONS,
)

# Layer Analyzer
from .layer_analyzer import (
    LayerAnalyzer,
    LayerMetrics,
    LayerResult,
    MultiLayerResult,
)

# Component Type Analyzer
from .component_analyzer import (
    ComponentTypeAnalyzer,
    ComponentMetrics,
    ComponentTypeResult,
    AllTypesResult,
)

# Edge Analyzer
from .edge_analyzer import (
    EdgeAnalyzer,
    EdgeMetrics,
    EdgeAnalysisResult,
)

from .quality_analyzer import (
    QualityMetrics,
    QualityAnalysisResult,
    QualityAnalyzer
)

# Main Analyzer
from .analyzer import (
    GraphAnalyzer,
    FullAnalysisResult,
    analyze_graph,
    analyze_layer,
)


__all__ = [
    # Version
    "__version__",
    
    # Classification
    "CriticalityLevel",
    "BoxPlotStats",
    "ClassifiedItem",
    "ClassificationResult",
    "BoxPlotClassifier",
    "classify_items",
    "get_level_for_score",
    
    # GDS Client
    "GDSClient",
    "CentralityResult",
    "ProjectionInfo",
    "StructuralResult",
    "COMPONENT_TYPES",
    "DEPENDENCY_TYPES",
    "LAYER_DEFINITIONS",
    
    # Layer Analyzer
    "LayerAnalyzer",
    "LayerMetrics",
    "LayerResult",
    "MultiLayerResult",
    
    # Component Type Analyzer
    "ComponentTypeAnalyzer",
    "ComponentMetrics",
    "ComponentTypeResult",
    "AllTypesResult",
    
    # Edge Analyzer
    "EdgeAnalyzer",
    "EdgeMetrics",
    "EdgeAnalysisResult",

    # Quality Analyzer
    "QualityMetrics",
    "QualityAnalysisResult",
    "QualityAnalyzer",
    
    # Main Analyzer
    "GraphAnalyzer",
    "FullAnalysisResult",
    "analyze_graph",
    "analyze_layer",
]