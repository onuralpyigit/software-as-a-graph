"""
Graph Analyzer

Main orchestrator for multi-layer graph analysis of distributed pub-sub systems.

Coordinates:
    - Structural analysis (topological metrics via NetworkX)
    - Quality analysis (R, M, A scores)
    - Problem detection (architectural issues and risks)
    - Multi-layer comparison (app, infra, middleware, system)

Usage:
    with GraphAnalyzer(uri="bolt://localhost:7687") as analyzer:
        # Analyze single layer
        result = analyzer.analyze_layer(AnalysisLayer.APP)
        
        # Analyze all layers
        results = analyzer.analyze_all_layers()
        
        # Export results
        analyzer.export_results(results, "output/analysis.json")
"""

from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Union

from .layers import (
    AnalysisLayer, 
    LAYER_DEFINITIONS, 
    get_layer_definition,
    get_all_layers,
    get_primary_layers
)
from .structural_analyzer import StructuralAnalyzer, StructuralAnalysisResult
from .quality_analyzer import QualityAnalyzer, QualityAnalysisResult
from .problem_detector import ProblemDetector, DetectedProblem, ProblemSummary
from .classifier import CriticalityLevel


@dataclass
class LayerAnalysisResult:
    """
    Complete analysis result for a single layer.
    
    Contains structural metrics, quality scores, and detected problems.
    """
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
    
    @property
    def is_healthy(self) -> bool:
        """Check if layer has no critical issues."""
        return self.problem_summary.by_severity.get("CRITICAL", 0) == 0
    
    @property
    def requires_attention(self) -> bool:
        """Check if layer has critical or high severity issues."""
        return self.problem_summary.requires_attention > 0


@dataclass
class MultiLayerAnalysisResult:
    """
    Complete analysis result across multiple layers.
    
    Enables cross-layer comparison and system-wide insights.
    """
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
        """Get all problems across all layers, sorted by severity."""
        all_problems = []
        for layer_result in self.layers.values():
            all_problems.extend(layer_result.problems)
        return sorted(all_problems, key=lambda p: (-p.priority, p.entity_id))
    
    def get_all_critical_components(self) -> List[str]:
        """Get IDs of all critical components across layers."""
        critical = set()
        for layer_result in self.layers.values():
            for comp in layer_result.quality.get_critical_components():
                critical.add(comp.id)
        return sorted(critical)
    
    def get_layer(self, layer: Union[str, AnalysisLayer]) -> Optional[LayerAnalysisResult]:
        """Get result for a specific layer."""
        key = layer.value if isinstance(layer, AnalysisLayer) else layer
        return self.layers.get(key)
    
    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary across all layers."""
        total_components = 0
        total_problems = 0
        critical_count = 0
        
        for layer_result in self.layers.values():
            total_components += layer_result.quality.classification_summary.total_components
            total_problems += layer_result.problem_summary.total_problems
            critical_count += layer_result.problem_summary.by_severity.get("CRITICAL", 0)
        
        return {
            "layers_analyzed": len(self.layers),
            "total_components": total_components,
            "total_problems": total_problems,
            "critical_problems": critical_count,
            "cross_layer_insights": len(self.cross_layer_insights),
        }


class GraphAnalyzer:
    """
    Main analyzer for multi-layer graph analysis.
    
    Coordinates structural analysis, quality scoring, and problem detection
    across all layers of the pub-sub system graph model.
    
    Supports both Neo4j database and JSON file data sources.
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        damping_factor: float = 0.85,
        k_factor: float = 1.5,
        use_ahp: bool = False,
        repository: Optional[Any] = None  # GraphRepository
    ):
        """
        Initialize the analyzer.
        
        Args:
            uri: Neo4j connection URI (used if repository not provided)
            user: Neo4j username
            password: Neo4j password
            damping_factor: PageRank damping factor (default: 0.85)
            k_factor: Box-plot IQR multiplier for classification (default: 1.5)
            repository: Optional injected GraphRepository
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.damping_factor = damping_factor
        self.k_factor = k_factor
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize analyzers
        self.structural = StructuralAnalyzer(damping_factor=damping_factor)
        self.quality = QualityAnalyzer(k_factor=k_factor, use_ahp=use_ahp)
        self.detector = ProblemDetector()
        
        # Data source
        self._client = repository
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._client and hasattr(self._client, 'close'):
            self._client.close()
    
    def _get_client(self):
        """Lazy initialization of database client."""
        if self._client is None:
            # Fallback to creating Neo4jRepository (via adapter or core)
            # Using new adapter if possible, else core for backward compat?
            # Let's use new adapter
            try:
                from src.adapters.persistence import Neo4jGraphRepository
                self._client = Neo4jGraphRepository(self.uri, self.user, self.password)
            except ImportError:
                # Fallback to old core
                try:
                    from src.core.graph_exporter import GraphExporter
                    self._client = GraphExporter(self.uri, self.user, self.password)
                except ImportError:
                    self.logger.error("Graph repository not available. Install neo4j driver.")
                    raise
        return self._client
    
    def _load_data(self, layer: AnalysisLayer) -> Any:
        """Load graph data for the specified layer."""
        client = self._get_client()
        layer_def = get_layer_definition(layer)
        
        # Determine method to call based on client type
        # Both GraphRepository and GraphExporter support get_graph_data
        
        if layer == AnalysisLayer.SYSTEM:
            return client.get_graph_data()
        else:
            return client.get_graph_data(
                component_types=list(layer_def.component_types),
                dependency_types=list(layer_def.dependency_types),
            )
    
    def analyze_layer(
        self,
        layer: Union[str, AnalysisLayer] = AnalysisLayer.SYSTEM,
        context: Optional[str] = None
    ) -> LayerAnalysisResult:
        """
        Analyze a single layer of the graph.
        
        Args:
            layer: Analysis layer (app, infra, mw-app, mw-infra, system)
            context: Optional context label for the analysis
            
        Returns:
            LayerAnalysisResult with structural, quality, and problem analysis
        """
        # Convert string to enum if needed
        if isinstance(layer, str):
            layer = AnalysisLayer.from_string(layer)
        
        layer_def = get_layer_definition(layer)
        context = context or layer_def.name
        
        self.logger.info(f"Analyzing layer: {layer.value} ({layer_def.name})")
        
        # 1. Load graph data
        graph_data = self._load_data(layer)
        
        # 2. Structural analysis (compute metrics)
        structural_result = self.structural.analyze(graph_data, layer=layer)
        
        # 3. Quality analysis (compute R, M, A scores and classify)
        quality_result = self.quality.analyze(structural_result, context=context)
        
        # 4. Problem detection
        problems = self.detector.detect(quality_result)
        problem_summary = self.detector.summarize(problems)
        
        return LayerAnalysisResult(
            layer=layer.value,
            layer_name=layer_def.name,
            description=layer_def.description,
            structural=structural_result,
            quality=quality_result,
            problems=problems,
            problem_summary=problem_summary,
        )
    
    def analyze_all_layers(
        self,
        include_middleware: bool = False
    ) -> MultiLayerAnalysisResult:
        """
        Analyze multiple layers.
        
        Args:
            include_middleware: Include middleware layers (mw-app, mw-infra)
            
        Returns:
            MultiLayerAnalysisResult with analysis for each layer
        """
        if include_middleware:
            layers_to_analyze = get_all_layers()
        else:
            layers_to_analyze = get_primary_layers()
        
        results: Dict[str, LayerAnalysisResult] = {}
        
        for layer in layers_to_analyze:
            try:
                self.logger.info(f"Analyzing {layer.value}...")
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
        
        if len(results) < 2:
            return insights
        
        # Compare critical component counts
        layer_criticals = {
            name: len(r.quality.get_critical_components())
            for name, r in results.items()
        }
        
        max_critical_layer = max(layer_criticals, key=layer_criticals.get) if layer_criticals else None
        if max_critical_layer and layer_criticals[max_critical_layer] > 0:
            insights.append(
                f"Layer '{max_critical_layer}' has the most critical components "
                f"({layer_criticals[max_critical_layer]}). Focus remediation here first."
            )
        
        # Compare SPOF counts
        layer_spofs = {}
        for name, r in results.items():
            spofs = [c for c in r.quality.components if c.structural.is_articulation_point]
            layer_spofs[name] = len(spofs)
        
        total_spofs = sum(layer_spofs.values())
        if total_spofs > 0:
            worst_layer = max(layer_spofs, key=layer_spofs.get)
            insights.append(
                f"System has {total_spofs} single points of failure across all layers. "
                f"Layer '{worst_layer}' has the most ({layer_spofs[worst_layer]})."
            )
        
        # App vs Infra comparison
        app_result = results.get(AnalysisLayer.APP.value)
        infra_result = results.get(AnalysisLayer.INFRA.value)
        
        if app_result and infra_result:
            app_density = app_result.structural.graph_summary.density
            infra_density = infra_result.structural.graph_summary.density
            
            if app_density > 2 * infra_density:
                insights.append(
                    f"Application layer is much denser ({app_density:.3f}) than infrastructure "
                    f"({infra_density:.3f}). Consider if infrastructure can support the connectivity needs."
                )
            elif infra_density > 2 * app_density:
                insights.append(
                    f"Infrastructure is denser ({infra_density:.3f}) than application layer "
                    f"({app_density:.3f}). Good redundancy at network level."
                )
        
        # Problem concentration
        all_problems = []
        for r in results.values():
            all_problems.extend(r.problems)
        
        if all_problems:
            critical_count = sum(1 for p in all_problems if p.severity == "CRITICAL")
            if critical_count > 5:
                insights.append(
                    f"System has {critical_count} CRITICAL issues across layers. "
                    f"Recommend comprehensive architecture review."
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
            elif "structural" in data:
                del data["structural"]
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Results exported to: {path.absolute()}")


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_graph(
    layer: str = "system",
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password",
    output: Optional[str] = None,
    use_ahp: bool = False
) -> Union[LayerAnalysisResult, MultiLayerAnalysisResult]:
    """
    Convenience function for quick graph analysis.
    
    Args:
        layer: Layer to analyze ("app", "infra", "system", "all")
        uri: Neo4j connection URI
        user: Neo4j username
        password: Neo4j password
        output: Optional path to export results
        
    Returns:
        Analysis results
    """
    with GraphAnalyzer(uri=uri, user=user, password=password, use_ahp=use_ahp) as analyzer:
        if layer.lower() == "all":
            results = analyzer.analyze_all_layers()
        else:
            results = analyzer.analyze_layer(layer)
        
        if output:
            analyzer.export_results(results, output)
        
        return results