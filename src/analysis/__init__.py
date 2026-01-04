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
        result = analyzer.analyze_full()

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

# Neo4j Client (data retrieval only)
from .neo4j_client import (
    Neo4jClient,
    ComponentData,
    EdgeData,
    GraphData,
)

# NetworkX Analyzer (algorithms)
from .networkx_analyzer import (
    NetworkXAnalyzer,
    CentralityMetrics,
    ComponentAnalysisResult,
    LayerAnalysisResult,
    EdgeAnalysisResult,
)

# Main Analyzer Facade
from .analyzer import (
    GraphAnalyzer,
    FullAnalysisResult,
    analyze_graph,
)

# Constants
COMPONENT_TYPES = ["Application", "Broker", "Node", "Topic"]

DEPENDENCY_TYPES = ["app_to_app", "node_to_node", "app_to_broker", "node_to_broker"]

LAYER_DEFINITIONS = {
    "application": {
        "name": "Application Layer",
        "description": "Application-to-application dependencies",
        "component_types": ["Application"],
        "dependency_types": ["app_to_app"],
    },
    "infrastructure": {
        "name": "Infrastructure Layer",
        "description": "Node-to-node dependencies",
        "component_types": ["Node"],
        "dependency_types": ["node_to_node"],
    },
    "app_broker": {
        "name": "Application-Broker Layer",
        "description": "Application-to-broker connections",
        "component_types": ["Application", "Broker"],
        "dependency_types": ["app_to_broker"],
    },
    "node_broker": {
        "name": "Node-Broker Layer",
        "description": "Node-to-broker connections",
        "component_types": ["Node", "Broker"],
        "dependency_types": ["node_to_broker"],
    },
}


__all__ = [
    # Version
    "__version__",
    
    # Constants
    "COMPONENT_TYPES",
    "DEPENDENCY_TYPES",
    "LAYER_DEFINITIONS",
    
    # Classification
    "CriticalityLevel",
    "BoxPlotStats",
    "ClassifiedItem",
    "ClassificationResult",
    "BoxPlotClassifier",
    "classify_items",
    
    # Neo4j Client
    "Neo4jClient",
    "ComponentData",
    "EdgeData",
    "GraphData",
    
    # NetworkX Analyzer
    "NetworkXAnalyzer",
    "CentralityMetrics",
    "ComponentAnalysisResult",
    "LayerAnalysisResult",
    "EdgeAnalysisResult",
    
    # Main Analyzer
    "GraphAnalyzer",
    "FullAnalysisResult",
    "analyze_graph",
]