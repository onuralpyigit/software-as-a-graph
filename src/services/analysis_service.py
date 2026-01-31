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
from src.services.analysis import (
    StructuralAnalyzer, StructuralAnalysisResult,
    QualityAnalyzer, QualityAnalysisResult,
    ProblemDetector, DetectedProblem, ProblemSummary
)
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
        graph_data = self._repository.get_graph_data(include_raw=True)
        
        # 1. Structural Analysis
        structural_result = self.structural_analyzer.analyze(graph_data, layer=layer_enum)
        
        # 2. Quality Analysis
        quality_result = self.quality_analyzer.analyze(structural_result)
        
        # 3. Problem Detection
        problems = self.problem_detector.detect(quality_result)
        problem_summary = self.problem_detector.summarize(problems)
        
        # 4. Library Usage Analysis
        library_usage = {}
        node_allocations = {}
        broker_routing = {}
        
        lib_names = {c.id: c.properties.get("name", c.id) for c in graph_data.components if c.component_type == "Library"}
        app_names = {c.id: c.properties.get("name", c.id) for c in graph_data.components if c.component_type == "Application"}
        topic_names = {c.id: c.properties.get("name", c.id) for c in graph_data.components if c.component_type == "Topic"}
        
        for edge in graph_data.edges:
            # App -> USES -> Lib
            if edge.relation_type == "USES" and edge.source_type == "Application" and edge.target_type == "Library":
                app_id = edge.source_id
                lib_id = edge.target_id
                lib_name = lib_names.get(lib_id, lib_id)
                if "version" in edge.properties:
                    lib_name += f" ({edge.properties['version']})"
                
                if app_id not in library_usage:
                    library_usage[app_id] = []
                library_usage[app_id].append(lib_name)
            
            # App/Broker -> RUNS_ON -> Node
            # We want Node related Apps (so Node is target, App is source)
            elif edge.relation_type == "RUNS_ON" and edge.target_type == "Node":
                node_id = edge.target_id
                guest_id = edge.source_id
                if edge.source_type == "Application":
                    guest_name = app_names.get(guest_id, guest_id)
                    if node_id not in node_allocations:
                        node_allocations[node_id] = []
                    node_allocations[node_id].append(guest_name)
            
            # Broker -> ROUTES -> Topic
            elif edge.relation_type == "ROUTES" and edge.source_type == "Broker" and edge.target_type == "Topic":
                broker_id = edge.source_id
                topic_id = edge.target_id
                topic_name = topic_names.get(topic_id, topic_id)
                if broker_id not in broker_routing:
                    broker_routing[broker_id] = []
                broker_routing[broker_id].append(topic_name)
        
        return LayerAnalysisResult(
            layer=layer_enum.value,
            layer_name=layer_def.name,
            description=layer_def.description,
            structural=structural_result,
            quality=quality_result,
            problems=problems,
            problem_summary=problem_summary,
            library_usage=library_usage,
            node_allocations=node_allocations,
            broker_routing=broker_routing
        )

    def analyze_all_layers(self) -> MultiLayerAnalysisResult:
        """
        Run analysis across all primary layers.
        """
        layers_to_analyze = get_primary_layers()
        
        results = {}
        for layer in layers_to_analyze:
            try:
                results[layer.value] = self.analyze_layer(layer)
            except Exception as e:
                self.logger.error(f"Failed to analyze layer {layer.value}: {e}")
                self.logger.exception(f"Exception details for {layer.value}:")
        
        # Generate cross-layer insights
        insights = self._generate_cross_layer_insights(results)
        
        return MultiLayerAnalysisResult(
            timestamp=datetime.now().isoformat(),
            layers=results,
            cross_layer_insights=insights
        )

    def _generate_cross_layer_insights(self, results: Dict[str, LayerAnalysisResult]) -> List[str]:
        """Generate insights by comparing analysis results across layers."""
        insights = []
        
        # Compare layer complexity
        if "app" in results and "infra" in results:
            app_nodes = results["app"].structural.graph_summary.nodes
            infra_nodes = results["infra"].structural.graph_summary.nodes
            if infra_nodes > app_nodes * 1.5:
                insights.append(
                    f"Infrastructure layer is significantly more complex than application layer "
                    f"({infra_nodes} vs {app_nodes} nodes)."
                )
            elif app_nodes > infra_nodes * 1.5:
                insights.append(
                    f"Application layer is significantly more complex than infrastructure layer "
                    f"({app_nodes} vs {infra_nodes} nodes)."
                )
        
        # Compare problem severity across layers
        layer_problems = {}
        for layer_name, layer_result in results.items():
            critical_count = layer_result.problem_summary.by_severity.get("CRITICAL", 0)
            high_count = layer_result.problem_summary.by_severity.get("HIGH", 0)
            layer_problems[layer_name] = critical_count + high_count
        
        if layer_problems:
            worst_layer = max(layer_problems, key=layer_problems.get)
            if layer_problems[worst_layer] > 0:
                best_layer = min(layer_problems, key=layer_problems.get)
                if layer_problems[worst_layer] > layer_problems[best_layer] * 2:
                    insights.append(
                        f"The {worst_layer} layer has significantly more critical issues "
                        f"({layer_problems[worst_layer]}) than the {best_layer} layer "
                        f"({layer_problems[best_layer]})."
                    )
        
        # Check middleware coupling
        if "mw" in results:
            mw_summary = results["mw"].structural.graph_summary
            if mw_summary.num_articulation_points > 0:
                insights.append(
                    f"Middleware layer has {mw_summary.num_articulation_points} articulation points "
                    f"(single points of failure in broker connectivity)."
                )
        
        return insights

    def export_results(self, results: Union[LayerAnalysisResult, MultiLayerAnalysisResult], output_path: str) -> None:
        """
        Export analysis results to a JSON file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Results exported to {output_path}")