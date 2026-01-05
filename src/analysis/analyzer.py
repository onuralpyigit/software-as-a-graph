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
        """
        Analyze the complete system using Derived Dependencies (App-App, Node-Node).
        Good for understanding overall system flow and coupling.
        """
        self.logger.info("Analyzing system (Derived Dependencies)...")
        graph_data = self.exporter.get_graph_data()  # Uses DEPENDS_ON
        return self._run_pipeline(graph_data, context="Complete System")

    def analyze_by_type(self, component_type: str) -> Dict[str, Any]:
        """
        Analyze components of a specific type.
        
        Strategies:
        - Application/Node/Broker: Uses Derived Graph (DEPENDS_ON) to see coupling.
        - Topic: Uses Structural Graph (PUBLISHES/SUBSCRIBES) to see centrality.
        """
        self.logger.info(f"Analyzing component type: {component_type}")
        
        if component_type == "Topic":
            # Topics don't have DEPENDS_ON edges; they are the bridge. 
            # We must analyze the Structural Graph (Bipartite View)
            data = self.exporter.get_structural_graph()
            # Filter to keep only Topic nodes and their immediate edges in memory
            # (handled by StructuralAnalyzer logic or pre-filtering)
            # For this pipeline, we pass the structural graph but focus metrics on Topics.
        else:
            # For Apps/Nodes, we want to see the "Logical" dependency graph
            data = self.exporter.get_subgraph_by_component_type(component_type)
            
        return self._run_pipeline(data, context=f"Type: {component_type}")

    def analyze_layer(self, layer: str) -> Dict[str, Any]:
        """Analyze a specific architectural layer (e.g., 'infrastructure')."""
        self.logger.info(f"Analyzing layer: {layer}")
        data = self.exporter.get_layer(layer)
        return self._run_pipeline(data, context=f"Layer: {layer}")

    def _run_pipeline(self, graph_data, context: str = "") -> Dict[str, Any]:
        # 1. Structural Analysis (Math)
        struct_res = self.structural.analyze(graph_data)
        
        # 2. Criticality/Quality Analysis (Scoring & Classification)
        # We group by type inside the analyzer to ensure fair comparison
        quality_res = self.quality.analyze(struct_res)
        
        # 3. Problem Detection (Insights)
        problems = self.detector.detect(quality_res)
        
        return {
            "context": context,
            "summary": quality_res.classification_summary,
            "results": quality_res,
            "problems": problems,
            "stats": struct_res.graph_summary
        }