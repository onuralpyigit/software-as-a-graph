"""
Graph Analyzer Facade

Orchestrates the analysis pipeline with support for granular analysis (Type/Layer).
"""

from typing import Dict, Any, Optional
import logging

from src.core.graph_exporter import GraphExporter
from .structural_analyzer import StructuralAnalyzer
from .quality_analyzer import QualityAnalyzer
from .problem_detector import ProblemDetector

class GraphAnalyzer:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.exporter = GraphExporter(uri, user, password)
        self.structural = StructuralAnalyzer()
        self.quality = QualityAnalyzer()
        self.detector = ProblemDetector()
        self.logger = logging.getLogger(__name__)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    def close(self): self.exporter.close()

    def analyze_full_system(self) -> Dict[str, Any]:
        """Analyze the complete graph model."""
        return self._run_pipeline(self.exporter.get_graph_data())

    def analyze_by_type(self, component_type: str) -> Dict[str, Any]:
        """Analyze components of a specific type (e.g., 'Application')."""
        self.logger.info(f"Analyzing component type: {component_type}")
        # Get subgraph of (Type)->(Type)
        data = self.exporter.get_subgraph_by_component_type(component_type)
        return self._run_pipeline(data)

    def analyze_layer(self, layer: str) -> Dict[str, Any]:
        """Analyze a specific architectural layer."""
        self.logger.info(f"Analyzing layer: {layer}")
        data = self.exporter.get_layer(layer)
        return self._run_pipeline(data)

    def _run_pipeline(self, graph_data) -> Dict[str, Any]:
        # 1. Structural
        struct_res = self.structural.analyze(graph_data)
        
        # 2. Quality
        quality_res = self.quality.analyze(struct_res)
        
        # 3. Problems
        problems = self.detector.detect(quality_res)
        
        return {
            "summary": quality_res.classification_summary,
            "quality": quality_res,
            "problems": problems,
            "stats": struct_res.graph_summary
        }