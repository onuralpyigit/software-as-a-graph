"""
Analysis Module - Version 6.0 (Refactored)

Multi-layer graph analysis for distributed pub-sub systems.
Uses Neo4j for data retrieval and NetworkX for graph algorithms.

Key Changes from v5.0:
- Neo4j used ONLY for data retrieval (not GDS algorithms)
- NetworkX for all graph algorithms (PageRank, Betweenness, Articulation Points)
- Simplified architecture with clear separation of concerns
- Per-component-type analysis for fair comparison

Features:
- Component-type analysis (Application, Broker, Node, Topic)
- Layer analysis (app_to_app, node_to_node, app_to_broker, node_to_broker)
- Edge criticality analysis
- Box-plot statistical classification
- NetworkX-based graph algorithms

Usage:
    from src.analysis import GraphAnalyzer
    
    with GraphAnalyzer(uri, user, password) as analyzer:
        # Analyze by component type
        apps = analyzer.analyze_component_type("Application")
        
        # Analyze by layer
        app_layer = analyzer.analyze_layer("application")
        
        # Full analysis
        result = analyzer.analyze()

Author: Software-as-a-Graph Research Project
Version: 6.0
"""

__version__ = "6.0.0"

# Classification
from .classifier import (
    CriticalityLevel,
    BoxPlotStats,
    ClassifiedItem,
    ClassificationResult,
    BoxPlotClassifier,
    classify_items,
)

# Structural Analyzer
from .structural_analyzer import (
    StructuralAnalyzer
)

# Main Analyzer Facade
from .analyzer import (
    GraphAnalyzer,
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
    
    # Structural Analyzer
    "StructuralAnalyzer",
    
    # Main Analyzer
    "GraphAnalyzer",
]