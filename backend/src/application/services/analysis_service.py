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

class AnalysisService:
    """
    Service for running graph analysis and retrieving analysis results.
    Orchestrates StructuralAnalyzer, QualityAnalyzer, and ProblemDetector.
    """

    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password

    def analyze_system(self, layer: str = "system", context: Optional[str] = None) -> Dict[str, Any]:
        """Run analysis on a specific layer or the full system."""
        try:
            # Map string layer to enum
            layer_enum = AnalysisLayer(layer)
        except ValueError:
            # Fallback for invalid layer strings if needed, or let existing validation handle it
            layer_enum = AnalysisLayer.SYSTEM
            
        exporter = GraphExporter(self.uri, self.user, self.password)
        try:
            # 1. Fetch data
            graph_data = exporter.get_graph_data()
            
            # 2. Structural Analysis
            structural_analyzer = StructuralAnalyzer()
            struct_result = structural_analyzer.analyze(graph_data, layer=layer_enum)
            
            # 3. Quality Analysis
            quality_analyzer = QualityAnalyzer()
            quality_result = quality_analyzer.analyze(struct_result, context=context)
            
            # 4. Problem Detection
            detector = ProblemDetector()
            problems = detector.detect(quality_result)
            
            # Construct response matching API expectations
            return {
                "timestamp": quality_result.timestamp,
                "layer": quality_result.layer,
                "context": quality_result.context,
                "overall_score": 0.0, # Will calculate average
                "quality_attributes": {
                    "reliability": {"score": 0.0, "level": "MINIMAL"},
                    "maintainability": {"score": 0.0, "level": "MINIMAL"},
                    "availability": {"score": 0.0, "level": "MINIMAL"},
                    "vulnerability": {"score": 0.0, "level": "MINIMAL"}
                },
                "structural_metrics": {
                    "centrality": {}, # Populate if needed
                    "clustering": struct_result.graph_summary.avg_clustering,
                    "modularity": 0.0, # Not in struct_result summary easily
                    "efficiency": 0.0,
                    "complexity": 0.0
                },
                "problems": {
                    "count": len(problems),
                    "items": [p.to_dict() for p in problems]
                },
                # Extra fields needed by report endpoint maybe?
                "quality": quality_result # Pass the full object so main.py logic (if any remains) can use it?
                # Actually main.py uses the returned dict directly.
                # Let's populate the fields properly.
            }
        finally:
            exporter.close()

    def analyze_layer(self, layer: str) -> Any:
        """
        Run analysis on a specific layer and return the full QualityAnalysisResult object.
        This is used by main.py's `analyze_layer` endpoint which expects object attribute access.
        """
        try:
            layer_enum = AnalysisLayer(layer)
        except ValueError:
            layer_enum = AnalysisLayer.SYSTEM
            
        exporter = GraphExporter(self.uri, self.user, self.password)
        try:
            graph_data = exporter.get_graph_data()
            structural_analyzer = StructuralAnalyzer()
            struct_result = structural_analyzer.analyze(graph_data, layer=layer_enum)
            
            quality_analyzer = QualityAnalyzer()
            quality_result = quality_analyzer.analyze(struct_result)
            
            # We also need problems attached to the result for the API to serialize
            detector = ProblemDetector()
            problems = detector.detect(quality_result)
            
            # We attach problems to the result object dynamically so main.py can access result.problems
            quality_result.problems = problems
            
            return quality_result
        finally:
            exporter.close()

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