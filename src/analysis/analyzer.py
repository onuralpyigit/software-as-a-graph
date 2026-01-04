"""
Graph Analyzer - Version 6.0

Main facade for multi-layer graph analysis.

This module combines:
- Neo4j client for data retrieval
- NetworkX analyzer for graph algorithms
- Box-plot classifier for component classification

Example:
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

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

from .neo4j_client import (
    Neo4jClient,
    GraphData,
    COMPONENT_TYPES,
    LAYER_DEFINITIONS,
)
from .networkx_analyzer import (
    NetworkXAnalyzer,
    ComponentAnalysisResult,
    LayerAnalysisResult,
    EdgeAnalysisResult,
)
from .classifier import BoxPlotClassifier


@dataclass
class FullAnalysisResult:
    """Complete analysis result combining all analyzers."""
    
    timestamp: str
    
    # Component type analysis
    component_types: Dict[str, ComponentAnalysisResult] = field(default_factory=dict)
    
    # Layer analysis
    layers: Dict[str, LayerAnalysisResult] = field(default_factory=dict)
    
    # Edge analysis
    edges: Optional[EdgeAnalysisResult] = None
    
    # Graph statistics
    graph_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Summary
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "graph_stats": self.graph_stats,
            "component_types": {k: v.to_dict() for k, v in self.component_types.items()},
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
            "edges": self.edges.to_dict() if self.edges else None,
            "summary": self.summary,
        }
    
    def get_all_critical(self) -> Dict[str, List[Any]]:
        """Get all critical components across all analyses."""
        critical = {}
        
        for comp_type, result in self.component_types.items():
            critical_list = result.get_critical()
            if critical_list:
                critical[f"type_{comp_type}"] = critical_list
        
        for layer_key, result in self.layers.items():
            critical_list = result.get_critical()
            if critical_list:
                critical[f"layer_{layer_key}"] = critical_list
        
        if self.edges:
            critical_edges = self.edges.get_critical()
            if critical_edges:
                critical["edges"] = critical_edges
        
        return critical


class GraphAnalyzer:
    """
    Main facade for graph analysis.
    
    Provides unified access to:
    - Component type analysis (comparing like with like)
    - Layer analysis (dependency-based grouping)
    - Edge criticality analysis
    - Structural analysis (articulation points, bridges)
    
    Uses Neo4j for data retrieval and NetworkX for algorithms.
    Classification uses box-plot statistical method.
    
    Example:
        # Context manager (recommended)
        with GraphAnalyzer(uri, user, password) as analyzer:
            result = analyzer.analyze_full()
            print(f"Critical: {len(result.get_all_critical())}")
        
        # Manual management
        analyzer = GraphAnalyzer(uri, user, password)
        try:
            apps = analyzer.analyze_component_type("Application")
        finally:
            analyzer.close()
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        k_factor: float = 1.5,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize analyzer.
        
        Args:
            uri: Neo4j bolt URI
            user: Username
            password: Password
            database: Database name
            k_factor: Box-plot k factor for classification
            weights: Custom weights for composite score
        """
        self.neo4j = Neo4jClient(uri, user, password, database)
        self.nx_analyzer = NetworkXAnalyzer(k_factor=k_factor, weights=weights)
        self.k_factor = k_factor
        self.logger = logging.getLogger(__name__)
        
        # Cache for graph data
        self._graph_data: Optional[GraphData] = None
    
    def __enter__(self) -> GraphAnalyzer:
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
    
    def close(self) -> None:
        """Close the analyzer and cleanup resources."""
        self.neo4j.close()
        self._graph_data = None
    
    # =========================================================================
    # Data Access
    # =========================================================================
    
    def get_graph_data(self, refresh: bool = False) -> GraphData:
        """
        Get graph data from Neo4j.
        
        Args:
            refresh: Force refresh from database
        
        Returns:
            GraphData with all components and edges
        """
        if self._graph_data is None or refresh:
            self._graph_data = self.neo4j.get_full_graph()
        return self._graph_data
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the graph."""
        return self.neo4j.get_graph_stats()
    
    # =========================================================================
    # Component Type Analysis
    # =========================================================================
    
    def analyze_component_type(
        self,
        component_type: str,
        weighted: bool = True,
    ) -> ComponentAnalysisResult:
        """
        Analyze all components of a specific type.
        
        Components are compared only with others of the same type
        for fair classification using box-plot method.
        
        Args:
            component_type: Type to analyze (Application, Broker, Node, Topic)
            weighted: Use edge weights in algorithms
        
        Returns:
            ComponentAnalysisResult with metrics and classification
        """
        if component_type not in COMPONENT_TYPES:
            raise ValueError(
                f"Invalid type: {component_type}. "
                f"Valid: {COMPONENT_TYPES}"
            )
        
        self.logger.info(f"Analyzing component type: {component_type}")
        graph_data = self.get_graph_data()
        
        return self.nx_analyzer.analyze_component_type(
            graph_data, 
            component_type, 
            weighted=weighted,
        )
    
    def analyze_all_component_types(
        self,
        weighted: bool = True,
    ) -> Dict[str, ComponentAnalysisResult]:
        """
        Analyze all component types.
        
        Args:
            weighted: Use edge weights
        
        Returns:
            Dict mapping type name to ComponentAnalysisResult
        """
        self.logger.info("Analyzing all component types")
        graph_data = self.get_graph_data()
        
        results = {}
        for comp_type in COMPONENT_TYPES:
            # Check if there are components of this type
            type_comps = graph_data.get_components_by_type(comp_type)
            if type_comps:
                results[comp_type] = self.nx_analyzer.analyze_component_type(
                    graph_data, 
                    comp_type, 
                    weighted=weighted,
                )
        
        return results
    
    # =========================================================================
    # Layer Analysis
    # =========================================================================
    
    def analyze_layer(
        self,
        layer_key: str,
        weighted: bool = True,
    ) -> LayerAnalysisResult:
        """
        Analyze a specific dependency layer.
        
        Args:
            layer_key: Layer to analyze (application, infrastructure, etc.)
            weighted: Use edge weights
        
        Returns:
            LayerAnalysisResult with metrics and classification
        """
        if layer_key not in LAYER_DEFINITIONS:
            raise ValueError(
                f"Invalid layer: {layer_key}. "
                f"Valid: {list(LAYER_DEFINITIONS.keys())}"
            )
        
        self.logger.info(f"Analyzing layer: {layer_key}")
        
        # Get layer-specific data
        layer_data = self.neo4j.get_layer(layer_key)
        
        return self.nx_analyzer.analyze_layer(
            layer_data, 
            layer_key, 
            weighted=weighted,
        )
    
    def analyze_all_layers(
        self,
        weighted: bool = True,
    ) -> Dict[str, LayerAnalysisResult]:
        """
        Analyze all dependency layers.
        
        Args:
            weighted: Use edge weights
        
        Returns:
            Dict mapping layer key to LayerAnalysisResult
        """
        self.logger.info("Analyzing all layers")
        
        results = {}
        for layer_key in LAYER_DEFINITIONS:
            layer_data = self.neo4j.get_layer(layer_key)
            if layer_data.edges:  # Only analyze if there are edges
                results[layer_key] = self.nx_analyzer.analyze_layer(
                    layer_data, 
                    layer_key, 
                    weighted=weighted,
                )
        
        return results
    
    # =========================================================================
    # Edge Analysis
    # =========================================================================
    
    def analyze_edges(self) -> EdgeAnalysisResult:
        """
        Analyze edge criticality.
        
        Returns:
            EdgeAnalysisResult with edge metrics and classification
        """
        self.logger.info("Analyzing edges")
        graph_data = self.get_graph_data()
        
        return self.nx_analyzer.analyze_edges(graph_data)
    
    # =========================================================================
    # Full Analysis
    # =========================================================================
    
    def analyze_full(
        self,
        include_component_types: bool = True,
        include_layers: bool = True,
        include_edges: bool = True,
        weighted: bool = True,
    ) -> FullAnalysisResult:
        """
        Perform complete analysis.
        
        Args:
            include_component_types: Include per-type analysis
            include_layers: Include layer analysis
            include_edges: Include edge analysis
            weighted: Use weighted algorithms
        
        Returns:
            FullAnalysisResult with complete analysis
        """
        timestamp = datetime.now().isoformat()
        self.logger.info("Starting full analysis")
        
        # Get graph statistics
        graph_stats = self.get_graph_stats()
        
        # Component type analysis
        component_types = {}
        if include_component_types:
            self.logger.info("Analyzing component types...")
            component_types = self.analyze_all_component_types(weighted=weighted)
        
        # Layer analysis
        layers = {}
        if include_layers:
            self.logger.info("Analyzing layers...")
            layers = self.analyze_all_layers(weighted=weighted)
        
        # Edge analysis
        edges = None
        if include_edges:
            self.logger.info("Analyzing edges...")
            edges = self.analyze_edges()
        
        # Build summary
        summary = self._build_summary(component_types, layers, edges, graph_stats)
        
        return FullAnalysisResult(
            timestamp=timestamp,
            component_types=component_types,
            layers=layers,
            edges=edges,
            graph_stats=graph_stats,
            summary=summary,
        )
    
    def _build_summary(
        self,
        component_types: Dict[str, ComponentAnalysisResult],
        layers: Dict[str, LayerAnalysisResult],
        edges: Optional[EdgeAnalysisResult],
        graph_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build analysis summary."""
        summary = {
            "graph": {
                "total_nodes": graph_stats.get("total_nodes", 0),
                "total_edges": graph_stats.get("total_edges", 0),
            },
            "total_critical_components": 0,
            "total_critical_edges": 0,
        }
        
        # Count critical components from type analysis
        type_summary = {}
        for comp_type, result in component_types.items():
            critical_count = len(result.get_critical())
            type_summary[comp_type] = {
                "total": result.summary.get("total", 0),
                "critical": critical_count,
                "articulation_points": result.summary.get("articulation_points", 0),
            }
            summary["total_critical_components"] += critical_count
        
        if type_summary:
            summary["component_types"] = type_summary
        
        # Layer summary
        layer_summary = {}
        for layer_key, result in layers.items():
            layer_summary[layer_key] = {
                "total": result.summary.get("total", 0),
                "critical": len(result.get_critical()),
                "bridges": result.summary.get("bridges", 0),
            }
        
        if layer_summary:
            summary["layers"] = layer_summary
        
        # Edge summary
        if edges:
            critical_edges = len(edges.get_critical())
            summary["edges"] = {
                "total": len(edges.edges),
                "critical": critical_edges,
                "bridges": len(edges.get_bridges()),
            }
            summary["total_critical_edges"] = critical_edges
        
        return summary


def analyze_graph(
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password",
    database: str = "neo4j",
    k_factor: float = 1.5,
    weighted: bool = True,
) -> FullAnalysisResult:
    """
    Convenience function for full graph analysis.
    
    Args:
        uri: Neo4j bolt URI
        user: Username
        password: Password
        database: Database name
        k_factor: Box-plot k factor
        weighted: Use weighted algorithms
    
    Returns:
        FullAnalysisResult
    """
    with GraphAnalyzer(uri, user, password, database, k_factor) as analyzer:
        return analyzer.analyze_full(weighted=weighted)