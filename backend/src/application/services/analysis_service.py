from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# Implements analysis logic using domain services directly
from src.core.graph_exporter import GraphExporter
from src.domain.config.layers import AnalysisLayer
from src.domain.services.structural_analyzer import StructuralAnalyzer
from src.domain.services.quality_analyzer import QualityAnalyzer
from src.domain.services.problem_detector import ProblemDetector

logger = logging.getLogger(__name__)

from datetime import datetime
from src.application.ports.inbound_ports import IAnalysisUseCase
from src.domain.models.analysis.results import MultiLayerAnalysisResult, LayerAnalysisResult
from src.domain.config.layers import AnalysisLayer, get_layer_definition

class AnalysisService(IAnalysisUseCase):
    """
    Service for running graph analysis and retrieving analysis results.
    Orchestrates StructuralAnalyzer, QualityAnalyzer, and ProblemDetector.
    """

    def __init__(self, repository, use_ahp: bool = False):
        self.repository = repository
        self.use_ahp = use_ahp

    def analyze_all_layers(self) -> MultiLayerAnalysisResult:
        """Analyze all primary graph layers."""
        layers = ["app", "infra", "mw", "system"]
        results = {}
        for layer in layers:
            results[layer] = self.analyze_layer(layer)
        
        return MultiLayerAnalysisResult(
            timestamp=datetime.now().isoformat(),
            layers=results,
            cross_layer_insights=[]
        )

    def analyze_system(self, layer: str = "system", context: Optional[str] = None) -> Dict[str, Any]:
        """Run analysis on a specific layer or the full system and return a dict."""
        res = self.analyze_layer(layer)
        return res.to_dict()

    def analyze_layer(self, layer: str) -> LayerAnalysisResult:
        """
        Run analysis on a specific layer and return the full LayerAnalysisResult object.
        """
        try:
            layer_enum = AnalysisLayer.from_string(layer)
        except ValueError:
            layer_enum = AnalysisLayer.SYSTEM
            
        layer_def = get_layer_definition(layer_enum)
            
        graph_data = self.repository.get_graph_data()
        structural_analyzer = StructuralAnalyzer()
        struct_result = structural_analyzer.analyze(graph_data, layer=layer_enum)
        
        quality_analyzer = QualityAnalyzer()
        quality_result = quality_analyzer.analyze(struct_result)
        
        detector = ProblemDetector()
        problems = detector.detect(quality_result)
        problem_summary = detector.summarize(problems)
        
        return LayerAnalysisResult(
            layer=layer_enum.value,
            layer_name=layer_def.name,
            description=layer_def.description,
            structural=struct_result,
            quality=quality_result,
            problems=problems,
            problem_summary=problem_summary
        )

    def analyze_by_type(self, component_type: str) -> Dict[str, Any]:
        """Run analysis and filter by component type."""
        # Reuse analyze_layer("system") then filter
        result = self.analyze_layer("system")
        
        # Filter components
        filtered_components = [c for c in result.components if c.type == component_type]
        
        # Calculate averages
        if filtered_components:
            avg_reliability = sum(c.scores.reliability for c in filtered_components) / len(filtered_components)
            avg_maintainability = sum(c.scores.maintainability for c in filtered_components) / len(filtered_components)
            avg_availability = sum(c.scores.availability for c in filtered_components) / len(filtered_components)
            avg_overall = sum(c.scores.overall for c in filtered_components) / len(filtered_components)
        else:
            avg_reliability = avg_maintainability = avg_availability = avg_overall = 0.0
            
        return {
            "component_type": component_type,
            "count": len(filtered_components),
            "average_scores": {
                "reliability": avg_reliability,
                "maintainability": avg_maintainability,
                "availability": avg_availability,
                "overall": avg_overall
            },
            "components": [
                {
                    "id": c.id,
                    "scores": {
                        "reliability": c.scores.reliability,
                        "maintainability": c.scores.maintainability,
                        "availability": c.scores.availability,
                        "overall": c.scores.overall
                    },
                    "issues": [p.description for p in getattr(result, "problems", []) if p.entity_id == c.id]
                }
                for c in filtered_components
            ]
        }

    def get_critical_components(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get the most critical components based on analysis."""
        result = self.analyze_layer("system")
        
        components = sorted(
            result.components,
            key=lambda c: c.scores.overall,
            reverse=True
        )[:limit]
        
        return [
            {
                "id": c.id,
                "type": c.type,
                "criticality_level": c.levels.overall.value,
                "criticality_levels": {
                    "reliability": c.levels.reliability.value,
                    "maintainability": c.levels.maintainability.value,
                    "availability": c.levels.availability.value,
                    "vulnerability": c.levels.vulnerability.value,
                    "overall": c.levels.overall.value
                },
                "overall_score": c.scores.overall,
                "scores": {
                    "reliability": c.scores.reliability,
                    "maintainability": c.scores.maintainability,
                    "availability": c.scores.availability,
                    "vulnerability": c.scores.vulnerability
                }
            }
            for c in components
        ]

    def get_critical_edges(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get the most critical edges based on analysis."""
        result = self.analyze_layer("system")
        
        edges = sorted(
            result.edges,
            key=lambda e: e.scores.overall,
            reverse=True
        )[:limit]
        
        return [
            {
                "source": e.source,
                "target": e.target,
                "type": e.dependency_type,
                "criticality_level": e.level.value,
                "overall_score": e.scores.overall,
                "scores": {
                    "reliability": e.scores.reliability,
                    "maintainability": e.scores.maintainability,
                    "availability": e.scores.availability,
                    "vulnerability": e.scores.vulnerability
                }
            }
            for e in edges
        ]
    def export_results(self, results: Any, output_path: str) -> None:
        """Export analysis results to a JSON file."""
        import json
        from src.core.utils import CustomJSONEncoder
        
        # Use CustomJSONEncoder if available to handle any non-serializable objects
        with open(output_path, 'w') as f:
            if hasattr(results, 'to_dict'):
                json.dump(results.to_dict(), f, indent=4)
            else:
                json.dump(results, f, indent=4)
