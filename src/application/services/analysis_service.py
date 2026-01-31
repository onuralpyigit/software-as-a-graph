"""
Graph Analyzer

Main orchestrator for multi-layer graph analysis of distributed pub-sub systems.
implements IAnalysisUseCase.
"""

from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Union

from src.application.ports.inbound.analysis_port import IAnalysisUseCase
from src.adapters.outbound.persistence.neo4j_repository import Neo4jGraphRepository

from src.domain.models.analysis.layers import (
    AnalysisLayer, 
    LAYER_DEFINITIONS, 
    get_layer_definition,
    get_all_layers,
    get_primary_layers
)
from src.domain.services import (
    StructuralAnalyzer, StructuralAnalysisResult,
    QualityAnalyzer, QualityAnalysisResult,
    ProblemDetector, DetectedProblem, ProblemSummary
)
from src.domain.models import CriticalityLevel
from src.domain.models.analysis.results import LayerAnalysisResult, MultiLayerAnalysisResult


class AnalysisService(IAnalysisUseCase):
    """
    Main service for graph analysis.
    Orchestrates structural analysis, quality assessment, and problem detection.
    """

    def __init__(self, repository: Neo4jGraphRepository):
        self._repository = repository
        self.logger = logging.getLogger(__name__)

    def analyze_layer(self, layer: str = "system") -> LayerAnalysisResult:
        """Analyze a single layer of the architecture."""
        try:
            # 1. Load Data
            # Convert string to AnalysisLayer enum if needed
            if isinstance(layer, str):
                analysis_layer = AnalysisLayer.from_string(layer)
            else:
                analysis_layer = layer
                
            definition = get_layer_definition(analysis_layer)
            self.logger.info(f"Analyzing layer: {definition.name} ({analysis_layer.value})")
            
            # Use repository to fetch graph data for specific component types
            # Note: repository.get_graph_data(component_types=...)
            graph_data = self._repository.get_graph_data(
                component_types=definition.component_types
            )
            
            # 2. Structural Analysis
            struct_analyzer = StructuralAnalyzer()
            struct_result = struct_analyzer.analyze(graph_data, layer=analysis_layer)
            
            # 3. Quality Analysis
            quality_analyzer = QualityAnalyzer()
            quality_result = quality_analyzer.analyze(struct_result)
            
            # 4. Problem Detection
            problem_detector = ProblemDetector()
            problems = problem_detector.detect(quality_result)
            problem_summary = problem_detector.summarize(problems)
            
            # 5. Context Gathering (for display)
            library_usage = self._repository.get_library_usage()
            node_allocations = self._repository.get_node_allocations()
            broker_routing = self._repository.get_broker_routing()

            return LayerAnalysisResult(
                layer=analysis_layer.value,
                layer_name=definition.name,
                description=definition.description,
                structural=struct_result,
                quality=quality_result,
                problems=problems,
                problem_summary=problem_summary,
                library_usage=library_usage,
                node_allocations=node_allocations,
                broker_routing=broker_routing,
            )

        except Exception as e:
            self.logger.error(f"Analysis failed for layer {layer}: {str(e)}")
            raise

    def analyze_all_layers(self, include_cross_layer: bool = True) -> MultiLayerAnalysisResult:
        """Perform analysis on all primary layers."""
        layers = get_primary_layers()
        results = {}
        
        for layer in layers:
            # layer is an AnalysisLayer enum
            results[layer.value] = self.analyze_layer(layer.value)
            
        cross_insights = []
        if include_cross_layer:
            cross_insights = self._generate_cross_layer_insights(results)

        return MultiLayerAnalysisResult(
            timestamp=datetime.now().isoformat(),
            layers=results,
            cross_layer_insights=cross_insights
        )

    def _generate_cross_layer_insights(self, layer_results: Dict[str, LayerAnalysisResult]) -> List[str]:
        """Generate insights by comparing layers."""
        insights = []
        
        # Example: Compare app vs infra complexity
        if "app" in layer_results and "infra" in layer_results:
            app_nodes = layer_results["app"].structural.graph_summary.nodes
            infra_nodes = layer_results["infra"].structural.graph_summary.nodes
            
            if infra_nodes > 0:
                ratio = app_nodes / infra_nodes
                insights.append(f"App/Infra Ratio: {ratio:.2f} (Applications per Node)")
                if ratio > 5:
                    insights.append("High application density per node detected.")
        
        return insights
        
    def export_results(self, result: Union[LayerAnalysisResult, MultiLayerAnalysisResult], output_file: str) -> None:
        """Export analysis results to JSON file."""
        output_path = Path(output_file)
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        self.logger.info(f"Results exported to {output_path}")
