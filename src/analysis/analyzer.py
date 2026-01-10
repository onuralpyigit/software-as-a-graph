"""
Graph Analyzer

Main orchestrator for multi-layer graph analysis.
Coordinates structural analysis, quality scoring, and problem detection.

Supports analysis layers:
- Application: Service-level reliability (app-to-app dependencies)
- Infrastructure: Network topology resilience (node-to-node dependencies)
- Complete: System-wide analysis across all layers

Author: Software-as-a-Graph Research Project
"""

from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict

from .structural_analyzer import (
    StructuralAnalyzer, 
    StructuralAnalysisResult, 
    AnalysisLayer,
    LAYER_DEFINITIONS
)
from .quality_analyzer import QualityAnalyzer, QualityAnalysisResult
from .problem_detector import ProblemDetector, DetectedProblem, ProblemSummary
from .classifier import CriticalityLevel


@dataclass
class LayerAnalysisResult:
    """Complete analysis result for a single layer."""
    
    layer: str
    layer_name: str
    description: str
    structural: StructuralAnalysisResult
    quality: QualityAnalysisResult
    problems: List[DetectedProblem]
    problem_summary: ProblemSummary
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "layer_name": self.layer_name,
            "description": self.description,
            "graph_summary": self.structural.graph_summary.to_dict(),
            "quality_analysis": self.quality.to_dict(),
            "problems": [p.to_dict() for p in self.problems],
            "problem_summary": self.problem_summary.to_dict(),
        }


@dataclass
class MultiLayerAnalysisResult:
    """Complete analysis result across multiple layers."""
    
    timestamp: str
    layers: Dict[str, LayerAnalysisResult]
    cross_layer_insights: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
            "cross_layer_insights": self.cross_layer_insights,
        }
    
    def get_all_problems(self) -> List[DetectedProblem]:
        """Get all problems across all layers."""
        all_problems = []
        for layer_result in self.layers.values():
            all_problems.extend(layer_result.problems)
        return sorted(all_problems, key=lambda p: -p.priority)
    
    def get_all_critical_components(self) -> List[str]:
        """Get IDs of all critical components across layers."""
        critical = set()
        for layer_result in self.layers.values():
            for comp in layer_result.quality.get_critical_components():
                critical.add(comp.id)
        return sorted(critical)


class GraphAnalyzer:
    """
    Main analyzer for multi-layer graph analysis.
    
    Coordinates:
    - Structural analysis (graph metrics)
    - Quality analysis (R, M, A scores)
    - Problem detection (architectural issues)
    
    Supports both Neo4j database and JSON file data sources.
    """
    
    def __init__(
        self,
        uri: Optional[str] = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        damping_factor: float = 0.85,
        k_factor: float = 1.5,
    ):
        """
        Initialize the analyzer.
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
            damping_factor: PageRank damping factor
            k_factor: Box-plot IQR multiplier for classification
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.damping_factor = damping_factor
        self.k_factor = k_factor
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize analyzers
        self.structural = StructuralAnalyzer(damping_factor=damping_factor)
        self.quality = QualityAnalyzer(k_factor=k_factor)
        self.detector = ProblemDetector()
        
        # Data source
        self._client = None
        self._graph_data = None
    
    def __enter__(self):
        """Context manager entry."""
        from src.core.graph_exporter import GraphExporter
        self._client = GraphExporter(self.uri, self.user, self.password)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._client:
            self._client.close()
    
    def _load_data(self, layer: AnalysisLayer) -> Any:
        # Load from Neo4j
        layer_def = LAYER_DEFINITIONS[layer]
        return self._client.get_graph_data(
            component_types=list(layer_def["component_types"]) if layer != AnalysisLayer.COMPLETE else None,
            dependency_types=list(layer_def["dependency_types"]) if layer != AnalysisLayer.COMPLETE else None,
        )
    
    def analyze_layer(
        self, 
        layer: Union[str, AnalysisLayer] = AnalysisLayer.COMPLETE,
        context: Optional[str] = None
    ) -> LayerAnalysisResult:
        """
        Analyze a single layer of the graph.
        
        Args:
            layer: Analysis layer (application, infrastructure, complete)
            context: Optional context label for the analysis
            
        Returns:
            LayerAnalysisResult with structural, quality, and problem analysis
        """
        # Convert string to enum if needed
        if isinstance(layer, str):
            layer = AnalysisLayer(layer.lower())
        
        layer_def = LAYER_DEFINITIONS[layer]
        context = context or layer_def["name"]
        
        self.logger.info(f"Analyzing layer: {layer.value}")
        
        # 1. Load graph data for this layer
        graph_data = self._load_data(layer)
        
        # 2. Structural analysis
        structural_result = self.structural.analyze(graph_data, layer=layer)
        
        # 3. Quality analysis
        quality_result = self.quality.analyze(structural_result, context=context)
        
        # 4. Problem detection
        problems = self.detector.detect(quality_result)
        problem_summary = self.detector.summarize(problems)
        
        return LayerAnalysisResult(
            layer=layer.value,
            layer_name=layer_def["name"],
            description=layer_def["description"],
            structural=structural_result,
            quality=quality_result,
            problems=problems,
            problem_summary=problem_summary,
        )
    
    def analyze_all_layers(self) -> MultiLayerAnalysisResult:
        """
        Analyze all primary layers (Application, Infrastructure, Complete).
        
        Returns:
            MultiLayerAnalysisResult with analysis for each layer
        """
        layers_to_analyze = [
            AnalysisLayer.APPLICATION,
            AnalysisLayer.INFRASTRUCTURE,
            AnalysisLayer.COMPLETE,
        ]
        
        results: Dict[str, LayerAnalysisResult] = {}
        
        for layer in layers_to_analyze:
            try:
                results[layer.value] = self.analyze_layer(layer)
            except Exception as e:
                self.logger.warning(f"Failed to analyze layer {layer.value}: {e}")
        
        # Generate cross-layer insights
        insights = self._generate_cross_layer_insights(results)
        
        return MultiLayerAnalysisResult(
            timestamp=datetime.now().isoformat(),
            layers=results,
            cross_layer_insights=insights,
        )
    
    def _generate_cross_layer_insights(
        self, 
        results: Dict[str, LayerAnalysisResult]
    ) -> List[str]:
        """Generate insights by comparing analysis across layers."""
        insights = []
        
        app_result = results.get("application")
        infra_result = results.get("infrastructure")
        complete_result = results.get("complete")
        
        # Insight 1: Compare component counts
        if app_result and infra_result:
            app_nodes = app_result.structural.graph_summary.nodes
            infra_nodes = infra_result.structural.graph_summary.nodes
            
            if app_nodes > 0 and infra_nodes > 0:
                ratio = app_nodes / infra_nodes
                if ratio > 5:
                    insights.append(
                        f"High application density: {app_nodes} applications across {infra_nodes} nodes "
                        f"(ratio: {ratio:.1f}:1). Consider infrastructure scaling."
                    )
        
        # Insight 2: Compare SPOF counts
        if app_result and infra_result:
            app_spofs = app_result.structural.graph_summary.num_articulation_points
            infra_spofs = infra_result.structural.graph_summary.num_articulation_points
            
            if infra_spofs > app_spofs and infra_spofs > 0:
                insights.append(
                    f"Infrastructure has more SPOFs ({infra_spofs}) than application layer ({app_spofs}). "
                    f"Infrastructure topology may be the bottleneck."
                )
        
        # Insight 3: Critical component overlap
        if app_result and complete_result:
            app_critical = {c.id for c in app_result.quality.get_high_priority()}
            complete_critical = {c.id for c in complete_result.quality.get_high_priority()}
            
            app_only = app_critical - complete_critical
            if app_only:
                insights.append(
                    f"Components critical in application layer but not system-wide: {', '.join(list(app_only)[:3])}. "
                    f"May be local hotspots."
                )
        
        # Insight 4: Connectivity comparison
        if app_result and infra_result:
            app_density = app_result.structural.graph_summary.density
            infra_density = infra_result.structural.graph_summary.density
            
            if app_density > 0 and infra_density > 0:
                if app_density > 2 * infra_density:
                    insights.append(
                        f"Application layer is much denser ({app_density:.3f}) than infrastructure ({infra_density:.3f}). "
                        f"Many app dependencies share few infrastructure links."
                    )
        
        if not insights:
            insights.append("No significant cross-layer anomalies detected.")
        
        return insights
    
    def export_results(
        self, 
        results: Union[LayerAnalysisResult, MultiLayerAnalysisResult],
        output_path: str,
        include_structural: bool = False
    ) -> None:
        """
        Export analysis results to JSON file.
        
        Args:
            results: Analysis results to export
            output_path: Path to output JSON file
            include_structural: Whether to include detailed structural metrics
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict
        data = results.to_dict()
        
        # Optionally strip heavy structural data
        if not include_structural:
            if "layers" in data:
                for layer_data in data["layers"].values():
                    if "structural" in layer_data:
                        del layer_data["structural"]
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Results exported to: {path.absolute()}")
    
    # Legacy method for backward compatibility
    def analyze(
        self,
        layer: str = "complete",
        component_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Legacy analyze method for backward compatibility.
        
        Returns dict format matching previous implementation.
        """
        result = self.analyze_layer(layer)
        
        # Filter by component type if specified
        if component_type:
            result.quality.components = [
                c for c in result.quality.components if c.type == component_type
            ]
        
        return {
            "timestamp": result.quality.timestamp,
            "context": result.layer_name,
            "summary": result.quality.classification_summary.to_dict(),
            "stats": result.structural.graph_summary.to_dict(),
            "problems": [p.to_dict() for p in result.problems],
            "results": result.quality,
        }


# Convenience function for quick analysis
def analyze_graph(
    layer: str = "complete",
    output: Optional[str] = None
) -> MultiLayerAnalysisResult:
    """
    Convenience function for quick graph analysis.
    
    Args:
        source: Neo4j URI or path to JSON file
        layer: Layer to analyze ("application", "infrastructure", "complete", "all")
        output: Optional path to export results
        
    Returns:
        Analysis results
    """
    with GraphAnalyzer() as analyzer:
        if layer == "all":
            results = analyzer.analyze_all_layers()
        else:
            single_result = analyzer.analyze_layer(layer)
            results = MultiLayerAnalysisResult(
                timestamp=datetime.now().isoformat(),
                layers={single_result.layer: single_result},
                cross_layer_insights=[],
            )
        
        if output:
            analyzer.export_results(results, output)
        
        return results