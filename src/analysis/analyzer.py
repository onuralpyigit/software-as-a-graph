"""
Graph Analyzer Facade

Orchestrates the analysis pipeline:
1. Export Data (Neo4j)
2. Structural Analysis (NetworkX)
3. Quality Assessment (RMA Formulas)
4. Problem Detection
"""

from typing import Dict, Any, List
import logging

from src.core.graph_exporter import GraphExporter
from .structural_analyzer import StructuralAnalyzer
from .quality_analyzer import QualityAnalyzer, QualityAnalysisResult
from .problem_detector import ProblemDetector, DetectedProblem

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

    def analyze_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete analysis pipeline."""
        self.logger.info("Starting Full Analysis Pipeline")
        
        # 1. Data Retrieval
        graph_data = self.exporter.get_graph_data()
        
        # 2. Structural Analysis
        struct_res = self.structural.analyze(graph_data)
        
        # 3. Quality Assessment
        quality_res = self.quality.analyze(struct_res)
        
        # 4. Problem Detection
        problems = self.detector.detect(quality_res)
        
        return {
            "timestamp": quality_res.timestamp,
            "quality": quality_res,
            "problems": problems,
            "stats": {
                "nodes": len(graph_data.components),
                "edges": len(graph_data.edges),
                "critical_components": len([c for c in quality_res.components 
                                          if c.level.value == "critical"])
            }
        }