"""
Graph Analyzer Facade

Orchestrates the analysis pipeline for the software-as-a-graph project.
Integrates Structure, Quality, and Problem Detection modules.
Supports JSON export of analysis results.
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from src.core.graph_exporter import GraphExporter
from .structural_analyzer import StructuralAnalyzer
from .quality_analyzer import QualityAnalyzer
from .problem_detector import ProblemDetector

class GraphAnalyzer:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password", database="neo4j"):
        self.exporter = GraphExporter(uri, user, password, database)
        self.structural = StructuralAnalyzer()
        self.quality = QualityAnalyzer()
        self.detector = ProblemDetector()
        self.logger = logging.getLogger(__name__)

    def __enter__(self): 
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb): 
        self.close()
    
    def close(self): 
        self.exporter.close()

    def analyze(self) -> Dict[str, Any]:
        """Analyze the complete system using Derived Dependencies."""
        self.logger.info("Starting analysis: Complete System")
        graph_data = self.exporter.get_graph_data()
        return self._run_pipeline(graph_data, context="Complete System")

    def analyze_layer(self, layer: str) -> Dict[str, Any]:
        """
        Analyze a specific architectural layer.
        
        Supported Layers:
        - 'application': App-to-App dependencies
        - 'infrastructure': Node-to-Node dependencies
        """
        self.logger.info(f"Starting analysis: Layer '{layer}'")
        try:
            data = self.exporter.get_layer(layer)
            return self._run_pipeline(data, context=f"Layer: {layer.capitalize()}")
        except ValueError as e:
            self.logger.error(f"Layer analysis failed: {e}")
            raise

    def analyze_by_type(self, component_type: str) -> Dict[str, Any]:
        """Analyze subgraph of a specific component type (e.g., 'Application')."""
        self.logger.info(f"Starting analysis: Component Type '{component_type}'")
        data = self.exporter.get_subgraph_by_component_type(component_type)
        return self._run_pipeline(data, context=f"Type: {component_type}")

    def _run_pipeline(self, graph_data, context: str = "") -> Dict[str, Any]:
        """Executes the analysis pipeline steps."""
        # 1. Structural Analysis (Topology & Metrics)
        struct_res = self.structural.analyze(graph_data)
        
        # 2. Quality Analysis (Scoring & Box-Plot Classification)
        quality_res = self.quality.analyze(struct_res, context=context)
        
        # 3. Problem Detection (Heuristics on Criticality Levels)
        problems = self.detector.detect(quality_res)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "summary": quality_res.classification_summary,
            "stats": struct_res.graph_summary,
            "problems": [p.to_dict() for p in problems],
            "results": quality_res, # Object reference for internal use
        }

    def export_results(self, results: Dict[str, Any], output_path: str):
        """
        Export analysis results to a JSON file.
        Handles the serialization of complex dataclasses.
        """
        def serialize(obj):
            if hasattr(obj, 'to_dict'): return obj.to_dict()
            if hasattr(obj, '__dict__'): return obj.__dict__
            if hasattr(obj, 'value'): return obj.value  # Enum
            return str(obj)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a simplified dict for export (removing heavy objects if necessary)
        export_data = {
            "meta": {
                "timestamp": results["timestamp"],
                "context": results["context"],
                "graph_stats": results["stats"]
            },
            "classification_summary": results["summary"],
            "detected_problems": results["problems"],
            # Flatten detailed component lists for JSON
            "components": [c.to_dict() if hasattr(c, 'to_dict') else str(c) 
                           for c in results["results"].components],
            "edges": [e.to_dict() if hasattr(e, 'to_dict') else str(e) 
                      for e in results["results"].edges]
        }

        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2, default=serialize)
        
        self.logger.info(f"Results exported to: {path.absolute()}")