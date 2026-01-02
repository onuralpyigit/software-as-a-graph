"""
Graph Analyzer - Version 5.0

Main facade for multi-layer graph analysis of distributed pub-sub systems.

Integrates:
- Multi-layer analysis (by dependency type)
- Component-type analysis (by Application, Broker, Node, Topic)
- Edge criticality analysis
- Box-plot statistical classification
- Weighted algorithm support

Example:
    with GraphAnalyzer(uri, user, password) as analyzer:
        # Full analysis
        result = analyzer.analyze_full()
        
        # Specific layer
        app_layer = analyzer.analyze_layer("application")
        
        # Component type
        brokers = analyzer.analyze_component_type("Broker")
        
        # Critical edges
        edges = analyzer.analyze_edges()

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

from .gds_client import GDSClient, LAYER_DEFINITIONS, COMPONENT_TYPES, DEPENDENCY_TYPES
from .classifier import BoxPlotClassifier, CriticalityLevel
from .layer_analyzer import LayerAnalyzer, LayerResult, MultiLayerResult
from .component_analyzer import ComponentTypeAnalyzer, ComponentTypeResult, AllTypesResult
from .edge_analyzer import EdgeAnalyzer, EdgeAnalysisResult


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FullAnalysisResult:
    """
    Complete analysis result combining all analyzers.
    """
    timestamp: str
    
    # Layer analysis
    layers: Optional[MultiLayerResult] = None
    
    # Component type analysis
    component_types: Optional[AllTypesResult] = None
    
    # Edge analysis
    edges: Optional[EdgeAnalysisResult] = None
    
    # Graph statistics
    graph_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Overall summary
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "layers": self.layers.to_dict() if self.layers else None,
            "component_types": self.component_types.to_dict() if self.component_types else None,
            "edges": self.edges.to_dict() if self.edges else None,
            "graph_stats": self.graph_stats,
            "summary": self.summary,
        }
    
    def get_all_critical_components(self) -> Dict[str, List[Any]]:
        """Get all critical components from all analyses."""
        critical = {}
        
        if self.component_types:
            for comp_type, result in self.component_types.by_type.items():
                critical[f"type_{comp_type}"] = result.get_critical()
        
        if self.layers:
            for layer_key, layer in self.layers.layers.items():
                critical[f"layer_{layer_key}"] = layer.get_critical_components()
        
        return critical


# =============================================================================
# Graph Analyzer (Main Facade)
# =============================================================================

class GraphAnalyzer:
    """
    Main facade for graph-based analysis.
    
    Provides unified access to:
    - Layer analysis (DEPENDS_ON relationship types)
    - Component type analysis (Application, Broker, Node, Topic)
    - Edge criticality analysis
    - Structural analysis (articulation points, bridges)
    
    All classification uses box-plot statistical method.
    
    Example:
        # Context manager (recommended)
        with GraphAnalyzer(uri, user, password) as analyzer:
            result = analyzer.analyze_full()
            print(f"Critical components: {result.summary['total_critical']}")
        
        # Manual management
        analyzer = GraphAnalyzer(uri, user, password)
        try:
            layers = analyzer.analyze_all_layers()
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
    ):
        """
        Initialize analyzer.
        
        Args:
            uri: Neo4j bolt URI
            user: Username
            password: Password
            database: Database name
            k_factor: Box-plot k factor for classification
        """
        self.gds = GDSClient(uri, user, password, database)
        self.k_factor = k_factor
        
        # Initialize sub-analyzers
        self.layer_analyzer = LayerAnalyzer(self.gds, k_factor=k_factor)
        self.component_analyzer = ComponentTypeAnalyzer(self.gds, k_factor=k_factor)
        self.edge_analyzer = EdgeAnalyzer(self.gds, k_factor=k_factor)
        
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self) -> GraphAnalyzer:
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
    
    def close(self) -> None:
        """Close the analyzer and cleanup resources."""
        self.gds.close()
    
    # =========================================================================
    # Layer Analysis
    # =========================================================================
    
    def analyze_layer(
        self,
        layer: str,
        weighted: bool = True,
    ) -> LayerResult:
        """
        Analyze a specific dependency layer.
        
        Args:
            layer: Layer key (application, infrastructure, app_broker, node_broker, full)
            weighted: Use weighted algorithms
        
        Returns:
            LayerResult with layer analysis
        """
        return self.layer_analyzer.analyze_layer(layer, weighted=weighted)
    
    def analyze_all_layers(
        self,
        weighted: bool = True,
        include_full: bool = True,
    ) -> MultiLayerResult:
        """
        Analyze all dependency layers.
        
        Args:
            weighted: Use weighted algorithms
            include_full: Include full system layer
        
        Returns:
            MultiLayerResult with all layer analyses
        """
        return self.layer_analyzer.analyze_all_layers(
            weighted=weighted, 
            include_full=include_full
        )
    
    # =========================================================================
    # Component Type Analysis
    # =========================================================================
    
    def analyze_component_type(
        self,
        component_type: str,
        weighted: bool = True,
    ) -> ComponentTypeResult:
        """
        Analyze a specific component type.
        
        Args:
            component_type: Type to analyze (Application, Broker, Node, Topic)
            weighted: Use weighted algorithms
        
        Returns:
            ComponentTypeResult with type analysis
        """
        return self.component_analyzer.analyze_type(component_type, weighted=weighted)
    
    def analyze_all_component_types(
        self,
        weighted: bool = True,
    ) -> AllTypesResult:
        """
        Analyze all component types.
        
        Args:
            weighted: Use weighted algorithms
        
        Returns:
            AllTypesResult with all type analyses
        """
        return self.component_analyzer.analyze_all_types(weighted=weighted)
    
    # =========================================================================
    # Edge Analysis
    # =========================================================================
    
    def analyze_edges(
        self,
        dependency_types: Optional[List[str]] = None,
    ) -> EdgeAnalysisResult:
        """
        Analyze edge criticality.
        
        Args:
            dependency_types: Filter to specific types (None = all)
        
        Returns:
            EdgeAnalysisResult with edge analysis
        """
        return self.edge_analyzer.analyze(dependency_types=dependency_types)
    
    def analyze_edges_by_layer(self) -> Dict[str, EdgeAnalysisResult]:
        """
        Analyze edges grouped by dependency type.
        
        Returns:
            Dict mapping dependency type to EdgeAnalysisResult
        """
        return self.edge_analyzer.analyze_by_layer()
    
    # =========================================================================
    # Full Analysis
    # =========================================================================
    
    def analyze_full(
        self,
        include_layers: bool = True,
        include_component_types: bool = True,
        include_edges: bool = True,
        weighted: bool = True,
    ) -> FullAnalysisResult:
        """
        Run complete analysis.
        
        Args:
            include_layers: Include layer analysis
            include_component_types: Include component type analysis
            include_edges: Include edge analysis
            weighted: Use weighted algorithms
        
        Returns:
            FullAnalysisResult with complete analysis
        """
        timestamp = datetime.now().isoformat()
        
        self.logger.info("Starting full analysis")
        
        # Get graph stats
        graph_stats = self.gds.get_graph_stats()
        
        # Run analyses
        layers = None
        if include_layers:
            self.logger.info("Analyzing layers...")
            layers = self.analyze_all_layers(weighted=weighted)
        
        component_types = None
        if include_component_types:
            self.logger.info("Analyzing component types...")
            component_types = self.analyze_all_component_types(weighted=weighted)
        
        edges = None
        if include_edges:
            self.logger.info("Analyzing edges...")
            edges = self.analyze_edges()
        
        # Generate summary
        summary = self._generate_full_summary(layers, component_types, edges, graph_stats)
        
        return FullAnalysisResult(
            timestamp=timestamp,
            layers=layers,
            component_types=component_types,
            edges=edges,
            graph_stats=graph_stats,
            summary=summary,
        )
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return self.gds.get_graph_stats()
    
    def get_articulation_points(self) -> List[Dict[str, Any]]:
        """Get articulation points."""
        results = self.gds.find_articulation_points()
        return [r.to_dict() for r in results]
    
    def get_bridges(self) -> List[Dict[str, Any]]:
        """Get bridge edges."""
        return self.gds.find_bridges()
    
    def _generate_full_summary(
        self,
        layers: Optional[MultiLayerResult],
        component_types: Optional[AllTypesResult],
        edges: Optional[EdgeAnalysisResult],
        graph_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate full analysis summary."""
        summary = {
            "graph": {
                "total_nodes": graph_stats.get("total_nodes", 0),
                "total_relationships": graph_stats.get("total_relationships", 0),
            },
            "total_critical_components": 0,
            "total_critical_edges": 0,
        }
        
        if layers:
            summary["layers"] = layers.summary
            summary["total_critical_components"] += layers.summary.get("total_critical", 0)
        
        if component_types:
            summary["component_types"] = component_types.summary
        
        if edges:
            critical_edges = len(edges.get_critical())
            summary["edges"] = {
                "total": len(edges.edges),
                "critical": critical_edges,
                "bridges": len(edges.get_bridges()),
            }
            summary["total_critical_edges"] = critical_edges
        
        return summary


# =============================================================================
# Convenience Functions
# =============================================================================

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


def analyze_layer(
    layer: str,
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password",
    k_factor: float = 1.5,
    weighted: bool = True,
) -> LayerResult:
    """
    Convenience function for single layer analysis.
    
    Args:
        layer: Layer to analyze
        uri: Neo4j bolt URI
        user: Username
        password: Password
        k_factor: Box-plot k factor
        weighted: Use weighted algorithms
    
    Returns:
        LayerResult
    """
    with GraphAnalyzer(uri, user, password, k_factor=k_factor) as analyzer:
        return analyzer.analyze_layer(layer, weighted=weighted)
