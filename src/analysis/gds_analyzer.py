"""
GDS Analyzer - Version 5.0

Main facade for graph-based analysis of distributed pub-sub systems.

Integrates:
- Component-type analysis (analyze by Application, Topic, Node, Broker)
- Centrality algorithms (PageRank, Betweenness, Degree)
- Box-plot classification
- Problem detection
- Anti-pattern detection
- Edge criticality analysis

Example:
    with GDSAnalyzer(uri, user, password) as analyzer:
        # Full multi-layer analysis
        result = analyzer.analyze_all()
        
        # Analyze specific component type
        app_result = analyzer.analyze_component_type("Application")
        
        # Detect problems
        problems = analyzer.detect_problems()
        
        # Detect anti-patterns
        patterns = analyzer.detect_antipatterns()

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional

from .gds_client import GDSClient, CentralityResult, ProjectionInfo
from .component_analyzer import ComponentTypeAnalyzer, ComponentTypeResult
from .classifier import BoxPlotClassifier, ClassificationResult, CriticalityLevel
from .problem_detector import ProblemDetector, ProblemDetectionResult
from .antipatterns import AntiPatternDetector, AntiPatternResult
from .edge_analyzer import EdgeAnalyzer, EdgeAnalysisResult


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LayerAnalysisResult:
    """
    Analysis result for a single layer (dependency type).
    """
    layer_name: str
    projection_info: ProjectionInfo
    pagerank: ClassificationResult
    betweenness: ClassificationResult
    degree: ClassificationResult
    composite: ClassificationResult
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer_name,
            "projection": self.projection_info.to_dict(),
            "pagerank": self.pagerank.to_dict(),
            "betweenness": self.betweenness.to_dict(),
            "degree": self.degree.to_dict(),
            "composite": self.composite.to_dict(),
        }


@dataclass
class MultiLayerAnalysisResult:
    """
    Complete multi-layer analysis result.
    """
    timestamp: str
    graph_stats: Dict[str, Any]
    
    # Analysis by component type
    by_component_type: Dict[str, ComponentTypeResult]
    
    # Analysis by dependency layer
    by_layer: Dict[str, LayerAnalysisResult]
    
    # Overall analysis (all layers combined)
    overall: Optional[LayerAnalysisResult]
    
    # Problem detection
    problems: Optional[ProblemDetectionResult]
    
    # Anti-pattern detection
    antipatterns: Optional[AntiPatternResult]
    
    # Edge analysis
    edges: Optional[EdgeAnalysisResult]
    
    # Summary
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "graph_stats": self.graph_stats,
            "by_component_type": {
                ct: result.to_dict() 
                for ct, result in self.by_component_type.items()
            },
            "by_layer": {
                layer: result.to_dict() 
                for layer, result in self.by_layer.items()
            },
            "overall": self.overall.to_dict() if self.overall else None,
            "problems": self.problems.to_dict() if self.problems else None,
            "antipatterns": self.antipatterns.to_dict() if self.antipatterns else None,
            "edges": self.edges.to_dict() if self.edges else None,
            "summary": self.summary,
        }
    
    def get_all_critical_components(self) -> List[str]:
        """Get all components classified as critical across all analyses"""
        critical = set()
        
        # From component type analysis
        for result in self.by_component_type.values():
            for comp in result.get_critical_components():
                critical.add(comp.component_id)
        
        # From layer analysis
        if self.overall:
            for item in self.overall.composite.get_critical():
                critical.add(item.id)
        
        return list(critical)


# =============================================================================
# GDS Analyzer
# =============================================================================

class GDSAnalyzer:
    """
    Main analyzer for graph-based pub-sub system analysis.
    
    Provides a unified interface for:
    - Component-type specific analysis
    - Multi-layer dependency analysis
    - Problem and anti-pattern detection
    - Edge criticality analysis
    
    Example:
        # Context manager usage
        with GDSAnalyzer("bolt://localhost:7687", "neo4j", "password") as analyzer:
            result = analyzer.analyze_all()
            print(f"Found {result.problems.critical_count} critical problems")
        
        # Direct usage
        analyzer = GDSAnalyzer(uri, user, password)
        try:
            apps = analyzer.analyze_component_type("Application")
            brokers = analyzer.analyze_component_type("Broker")
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
            user: Neo4j username
            password: Neo4j password
            database: Database name
            k_factor: Box-plot k factor for classification
        """
        self.gds = GDSClient(uri, user, password, database)
        self.k_factor = k_factor
        
        # Initialize sub-analyzers
        self.component_analyzer = ComponentTypeAnalyzer(self.gds, k_factor=k_factor)
        self.problem_detector = ProblemDetector(self.gds, k_factor=k_factor)
        self.antipattern_detector = AntiPatternDetector(self.gds, k_factor=k_factor)
        self.edge_analyzer = EdgeAnalyzer(self.gds, k_factor=k_factor)
        self.classifier = BoxPlotClassifier(k_factor=k_factor)
        
        self.logger = logging.getLogger(__name__)

    def __enter__(self) -> "GDSAnalyzer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def close(self) -> None:
        """Close the analyzer and cleanup resources"""
        self.gds.close()

    # =========================================================================
    # Component-Type Analysis
    # =========================================================================

    def analyze_component_type(
        self,
        component_type: str,
        weighted: bool = True,
    ) -> ComponentTypeResult:
        """
        Analyze all components of a specific type.
        
        Compares components within the same category using
        centrality metrics and box-plot classification.
        
        Args:
            component_type: Type to analyze (Application, Broker, Node, Topic)
            weighted: Use weighted algorithms
        
        Returns:
            ComponentTypeResult with metrics and classifications
        """
        return self.component_analyzer.analyze(component_type, weighted=weighted)

    def analyze_all_component_types(
        self,
        weighted: bool = True,
    ) -> Dict[str, ComponentTypeResult]:
        """
        Analyze all component types.
        
        Returns:
            Dict mapping component type -> ComponentTypeResult
        """
        return self.component_analyzer.analyze_all_types(weighted=weighted)

    # =========================================================================
    # Layer Analysis
    # =========================================================================

    def analyze_layer(
        self,
        layer_name: str,
        dependency_types: List[str],
        weighted: bool = True,
    ) -> LayerAnalysisResult:
        """
        Analyze a specific dependency layer.
        
        Args:
            layer_name: Name for this layer (e.g., "app_dependencies")
            dependency_types: DEPENDS_ON types to include
            weighted: Use weighted algorithms
        
        Returns:
            LayerAnalysisResult
        """
        projection_name = f"layer_{layer_name}"
        
        try:
            projection_info = self.gds.create_projection(
                projection_name,
                dependency_types=dependency_types,
                include_weights=weighted,
            )
            
            # Run centrality algorithms
            pr_results = self.gds.pagerank(projection_name, weighted=weighted)
            bc_results = self.gds.betweenness(projection_name, weighted=weighted)
            dc_results = self.gds.degree(
                projection_name, weighted=weighted, orientation="UNDIRECTED"
            )
            
            # Classify results
            pr_class = self._classify_centrality(pr_results, "pagerank")
            bc_class = self._classify_centrality(bc_results, "betweenness")
            dc_class = self._classify_centrality(dc_results, "degree")
            
            # Compute composite
            composite_class = self._compute_composite_classification(
                pr_results, bc_results, dc_results
            )
            
            return LayerAnalysisResult(
                layer_name=layer_name,
                projection_info=projection_info,
                pagerank=pr_class,
                betweenness=bc_class,
                degree=dc_class,
                composite=composite_class,
            )
        
        finally:
            self.gds.drop_projection(projection_name)

    # =========================================================================
    # Full Analysis
    # =========================================================================

    def analyze_all(
        self,
        include_component_types: bool = True,
        include_layers: bool = True,
        include_problems: bool = True,
        include_antipatterns: bool = True,
        include_edges: bool = True,
        weighted: bool = True,
    ) -> MultiLayerAnalysisResult:
        """
        Run complete multi-layer analysis.
        
        Args:
            include_component_types: Analyze by component type
            include_layers: Analyze by dependency layer
            include_problems: Run problem detection
            include_antipatterns: Run anti-pattern detection
            include_edges: Analyze edge criticality
            weighted: Use weighted algorithms
        
        Returns:
            MultiLayerAnalysisResult with all analysis results
        """
        timestamp = datetime.now().isoformat()
        
        self.logger.info("Starting multi-layer analysis")
        
        # Get graph stats
        graph_stats = self.gds.get_graph_stats()
        
        # Component type analysis
        by_component_type = {}
        if include_component_types:
            self.logger.info("Analyzing component types...")
            by_component_type = self.analyze_all_component_types(weighted=weighted)
        
        # Layer analysis
        by_layer = {}
        if include_layers:
            self.logger.info("Analyzing dependency layers...")
            
            # App-to-app layer
            if graph_stats["depends_on_by_type"].get("app_to_app", 0) > 0:
                by_layer["app_to_app"] = self.analyze_layer(
                    "app_to_app", 
                    ["app_to_app"], 
                    weighted=weighted
                )
            
            # Node-to-node layer
            if graph_stats["depends_on_by_type"].get("node_to_node", 0) > 0:
                by_layer["node_to_node"] = self.analyze_layer(
                    "node_to_node", 
                    ["node_to_node"], 
                    weighted=weighted
                )
        
        # Overall analysis (all layers combined)
        overall = None
        if include_layers:
            self.logger.info("Running overall analysis...")
            overall = self.analyze_layer(
                "overall",
                ["app_to_app", "node_to_node"],
                weighted=weighted,
            )
        
        # Problem detection
        problems = None
        if include_problems:
            self.logger.info("Detecting problems...")
            problems = self.problem_detector.detect_all()
        
        # Anti-pattern detection
        antipatterns = None
        if include_antipatterns:
            self.logger.info("Detecting anti-patterns...")
            antipatterns = self.antipattern_detector.detect_all()
        
        # Edge analysis
        edges = None
        if include_edges:
            self.logger.info("Analyzing edges...")
            edges = self.edge_analyzer.analyze()
        
        # Generate summary
        summary = self._generate_summary(
            graph_stats,
            by_component_type,
            by_layer,
            overall,
            problems,
            antipatterns,
            edges,
        )
        
        return MultiLayerAnalysisResult(
            timestamp=timestamp,
            graph_stats=graph_stats,
            by_component_type=by_component_type,
            by_layer=by_layer,
            overall=overall,
            problems=problems,
            antipatterns=antipatterns,
            edges=edges,
            summary=summary,
        )

    # =========================================================================
    # Detection Methods
    # =========================================================================

    def detect_problems(
        self,
        dependency_types: Optional[List[str]] = None,
    ) -> ProblemDetectionResult:
        """
        Detect reliability, maintainability, and availability problems.
        
        Args:
            dependency_types: DEPENDS_ON types to analyze
        
        Returns:
            ProblemDetectionResult
        """
        return self.problem_detector.detect_all(
            dependency_types=dependency_types
        )

    def detect_antipatterns(self) -> AntiPatternResult:
        """
        Detect architectural anti-patterns.
        
        Returns:
            AntiPatternResult
        """
        return self.antipattern_detector.detect_all()

    def analyze_edges(
        self,
        dependency_types: Optional[List[str]] = None,
    ) -> EdgeAnalysisResult:
        """
        Analyze edge criticality.
        
        Args:
            dependency_types: DEPENDS_ON types to analyze
        
        Returns:
            EdgeAnalysisResult
        """
        return self.edge_analyzer.analyze(dependency_types=dependency_types)

    # =========================================================================
    # Helpers
    # =========================================================================

    def _classify_centrality(
        self,
        results: List[CentralityResult],
        metric_name: str,
    ) -> ClassificationResult:
        """Classify centrality results using box-plot method"""
        items = [
            {"id": r.node_id, "type": r.node_type, "score": r.score}
            for r in results
        ]
        return self.classifier.classify(items, metric_name=metric_name)

    def _compute_composite_classification(
        self,
        pr_results: List[CentralityResult],
        bc_results: List[CentralityResult],
        dc_results: List[CentralityResult],
        weights: Optional[Dict[str, float]] = None,
    ) -> ClassificationResult:
        """Compute composite score classification"""
        if weights is None:
            weights = {
                "betweenness": 0.40,
                "pagerank": 0.35,
                "degree": 0.25,
            }
        
        # Normalize scores
        def normalize(results):
            max_score = max((r.score for r in results), default=1.0) or 1.0
            return {r.node_id: r.score / max_score for r in results}
        
        pr_norm = normalize(pr_results)
        bc_norm = normalize(bc_results)
        dc_norm = normalize(dc_results)
        
        # Get all node IDs
        all_ids = set(pr_norm.keys()) | set(bc_norm.keys()) | set(dc_norm.keys())
        
        # Build node type map
        type_map = {}
        for r in pr_results + bc_results + dc_results:
            type_map[r.node_id] = r.node_type
        
        # Compute composite scores
        items = []
        for node_id in all_ids:
            score = (
                weights["pagerank"] * pr_norm.get(node_id, 0) +
                weights["betweenness"] * bc_norm.get(node_id, 0) +
                weights["degree"] * dc_norm.get(node_id, 0)
            )
            items.append({
                "id": node_id,
                "type": type_map.get(node_id, "Unknown"),
                "score": score,
            })
        
        return self.classifier.classify(items, metric_name="composite")

    def _generate_summary(
        self,
        graph_stats: Dict[str, Any],
        by_component_type: Dict[str, ComponentTypeResult],
        by_layer: Dict[str, LayerAnalysisResult],
        overall: Optional[LayerAnalysisResult],
        problems: Optional[ProblemDetectionResult],
        antipatterns: Optional[AntiPatternResult],
        edges: Optional[EdgeAnalysisResult],
    ) -> Dict[str, Any]:
        """Generate analysis summary"""
        summary = {
            "graph": {
                "total_nodes": graph_stats.get("total_nodes", 0),
                "total_depends_on": graph_stats.get("depends_on_total", 0),
            },
            "component_types_analyzed": list(by_component_type.keys()),
            "layers_analyzed": list(by_layer.keys()),
        }
        
        # Critical counts by type
        if by_component_type:
            summary["critical_by_type"] = {
                ct: result.composite_classification.critical_count
                for ct, result in by_component_type.items()
            }
        
        # Overall critical count
        if overall:
            summary["overall_critical_count"] = overall.composite.critical_count
        
        # Problem summary
        if problems:
            summary["problems"] = {
                "total": problems.total_count,
                "critical": problems.critical_count,
            }
        
        # Anti-pattern summary
        if antipatterns:
            summary["antipatterns"] = {
                "total": antipatterns.total_count,
                "critical": antipatterns.critical_count,
            }
        
        # Edge summary
        if edges:
            summary["edges"] = {
                "total": edges.edge_count,
                "critical": len(edges.get_critical_edges()),
                "bridges": len(edges.get_bridges()),
            }
        
        return summary
