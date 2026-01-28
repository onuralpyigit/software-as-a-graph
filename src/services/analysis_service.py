"""
Graph Analyzer

Main orchestrator for multi-layer graph analysis of distributed pub-sub systems.
"""

from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Union

from src.models.analysis.layers import (
    AnalysisLayer, 
    LAYER_DEFINITIONS, 
    get_layer_definition,
    get_all_layers,
    get_primary_layers
)
from src.services.analysis.structural_analyzer import StructuralAnalyzer, StructuralAnalysisResult
from src.services.analysis.quality_analyzer import QualityAnalyzer, QualityAnalysisResult
from src.services.analysis.problem_detector import ProblemDetector, DetectedProblem, ProblemSummary
from src.models.analysis.criticality import CriticalityLevel
from src.models.analysis.results import LayerAnalysisResult, MultiLayerAnalysisResult


class AnalysisService:
    """
    Service for multi-layer graph analysis.
    """
    
    def __init__(
        self,
        damping_factor: float = 0.85,
        k_factor: float = 1.5,
        use_ahp: bool = False,
        repository: Optional[Any] = None
    ):
        self.logger = logging.getLogger(__name__)
        self.damping_factor = damping_factor
        self.k_factor = k_factor
        self.use_ahp = use_ahp
        self._repository = repository
        
        # Initialize sub-analyzers
        self.structural_analyzer = StructuralAnalyzer(damping_factor=damping_factor)
        self.quality_analyzer = QualityAnalyzer(k_factor=k_factor, use_ahp=use_ahp)
        self.problem_detector = ProblemDetector()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def analyze_layer(self, layer: Union[str, AnalysisLayer]) -> LayerAnalysisResult:
        """
        Run complete analysis on a specific layer.
        """
        layer_enum = AnalysisLayer(layer) if isinstance(layer, str) else layer
        layer_def = get_layer_definition(layer_enum)
        
        self.logger.info(f"Analyzing layer: {layer_enum.value}")
        
        # 0. Load Data
        if self._repository is None:
            raise ValueError("Repository is required for analysis")
        graph_data = self._repository.get_graph_data()
        
        # 1. Structural Analysis
        structural_result = self.structural_analyzer.analyze(graph_data, layer=layer_enum)
        
        # 2. Quality Analysis
        quality_result = self.quality_analyzer.analyze(structural_result)
        
        # 3. Problem Detection
        problems = self.problem_detector.detect(quality_result)
        problem_summary = self.problem_detector.summarize(problems)
        
        return LayerAnalysisResult(
            layer=layer_enum.value,
            layer_name=layer_def.name,
            description=layer_def.description,
            structural=structural_result,
            quality=quality_result,
            problems=problems,
            problem_summary=problem_summary
        )

    def analyze_all_layers(self, include_middleware: bool = False) -> MultiLayerAnalysisResult:
        """
        Run analysis across all layers.
        """
        layers_to_analyze = get_all_layers() if include_middleware else get_primary_layers()
        
        results = {}
        for layer in layers_to_analyze:
            try:
                results[layer.value] = self.analyze_layer(layer)
            except Exception as e:
                self.logger.error(f"Failed to analyze layer {layer.value}: {e}")
                self.logger.exception(f"Exception details for {layer.value}:")
        
        # Cross-layer insights (placeholder for future implementation)
        insights = []
        if "app" in results and "infra" in results:
            app_nodes = results["app"].structural.graph_summary.nodes
            infra_nodes = results["infra"].structural.graph_summary.nodes
            if infra_nodes > app_nodes:
                insights.append(f"Infrastructure layer is more complex than application layer ({infra_nodes} vs {app_nodes} nodes).")
        
        return MultiLayerAnalysisResult(
            timestamp=datetime.now().isoformat(),
            layers=results,
            cross_layer_insights=insights
        )

    def export_results(self, results: Union[LayerAnalysisResult, MultiLayerAnalysisResult], output_path: str) -> None:
        """
        Export analysis results to a JSON file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Results exported to {output_path}")