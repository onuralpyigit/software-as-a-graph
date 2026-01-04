"""
Validation Pipeline

Integration of Analysis, Simulation, and Validation modules.
Uses Neo4j for data source.
"""

import logging
from typing import Optional, Dict

from src.core.graph_exporter import GraphExporter, COMPONENT_TYPES, LAYER_DEFINITIONS
from src.analysis.structural_analyzer import StructuralAnalyzer
from src.analysis.quality_analyzer import QualityAnalyzer
from src.simulation.simulation_graph import SimulationGraph
from src.simulation.failure_simulator import FailureSimulator, FailureScenario
from .validator import Validator, ValidationResult, ValidationTargets

class ValidationPipeline:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.exporter = GraphExporter(uri, user, password)
        self.logger = logging.getLogger(__name__)

    def run(self, targets: Optional[ValidationTargets] = None) -> ValidationResult:
        """Run full validation pipeline."""
        self.logger.info("Starting Validation Pipeline")
        
        # 1. Fetch Data
        graph_data = self.exporter.get_graph_data()
        metadata = {}
        for c in graph_data.components:
            # Determine layer
            layer = "unknown"
            for l_key, l_def in LAYER_DEFINITIONS.items():
                if c.component_type in l_def["component_types"]:
                    layer = l_key
                    break
            metadata[c.id] = {"type": c.component_type, "layer": layer}

        # 2. Predicted Scores (Analysis)
        # We use Overall Quality Score (Q) as the predictor for Impact
        self.logger.info("Running Analysis...")
        struct_analyzer = StructuralAnalyzer()
        quality_analyzer = QualityAnalyzer()
        
        struct_res = struct_analyzer.analyze(graph_data)
        quality_res = quality_analyzer.analyze(struct_res)
        
        predicted_scores = {c.id: c.scores.overall for c in quality_res.components}
        
        # 3. Actual Scores (Simulation)
        # We simulate failure of EACH node and count total system impact
        self.logger.info("Running Simulation (this may take time)...")
        sim_graph = SimulationGraph(graph_data)
        simulator = FailureSimulator(sim_graph, propagation_threshold=0.4)
        
        actual_scores = {}
        total_nodes = len(graph_data.components)
        
        for i, comp in enumerate(graph_data.components):
            scenario = FailureScenario([comp.id], f"Validate {comp.id}")
            result = simulator.simulate(scenario)
            
            # Impact Score = (Total Failed Nodes - 1) / (Total Nodes - 1)
            # We subtract 1 to ignore self-failure
            impact_count = max(0, result.total_impact - 1)
            normalized_impact = impact_count / max(1, total_nodes - 1)
            actual_scores[comp.id] = normalized_impact
            
            if i % 10 == 0:
                self.logger.debug(f"Simulated {i}/{total_nodes} nodes")

        # 4. Validate
        self.logger.info("Comparing Results...")
        validator = Validator(targets)
        result = validator.validate(predicted_scores, actual_scores, metadata)
        
        return result
        
    def close(self):
        self.exporter.close()
    
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()