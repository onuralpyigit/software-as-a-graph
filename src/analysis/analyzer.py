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

    def analyze(self) -> Dict[str, Any]:
        """Analyze the complete system using Derived Dependencies."""
        self.logger.info("Analyzing Complete System...")
        # Get full graph (Apps + Nodes + dependencies)
        graph_data = self.exporter.get_graph_data() 
        return self._run_pipeline(graph_data, context="Complete System")

    def analyze_layer(self, layer: str) -> Dict[str, Any]:
        """
        Analyze a specific architectural layer.
        
        Layers:
        - 'application': App-to-App dependencies only.
        - 'infrastructure': Node-to-Node / Network dependencies.
        """
        self.logger.info(f"Analyzing Layer: {layer}")
        data = self.exporter.get_layer(layer)
        return self._run_pipeline(data, context=f"Layer: {layer.capitalize()}")

    def analyze_by_type(self, component_type: str) -> Dict[str, Any]:
        """Analyze specific component types (e.g., just 'Application' nodes)."""
        self.logger.info(f"Analyzing Component Type: {component_type}")
        data = self.exporter.get_subgraph_by_component_type(component_type)
        return self._run_pipeline(data, context=f"Type: {component_type}")

    def _run_pipeline(self, graph_data, context: str = "") -> Dict[str, Any]:
        # 1. Structural Analysis (Topology)
        struct_res = self.structural.analyze(graph_data)
        
        # 2. Quality Analysis (Scoring & Classification)
        # Passes structural context for deep analysis
        quality_res = self.quality.analyze(struct_res, context=context)
        
        # 3. Problem Detection (Pattern Matching on Levels)
        problems = self.detector.detect(quality_res)
        
        return {
            "context": context,
            "summary": quality_res.classification_summary,
            "results": quality_res,
            "problems": problems,
            "stats": struct_res.graph_summary
        }