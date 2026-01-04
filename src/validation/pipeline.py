"""
Validation Pipeline

Orchestrates the validation process:
1. Analysis: Uses Derived Graph (DEPENDS_ON) to predict Component Quality/Criticality.
2. Simulation: Uses Raw Structural Graph (RUNS_ON, etc.) to simulate failure cascades.
3. Comparison: Validates if the Derived Graph accurately modeled the physical dependencies.
"""

import logging
from typing import Optional, Dict

from src.core.graph_exporter import GraphExporter
from src.analysis.structural_analyzer import StructuralAnalyzer
from src.analysis.quality_analyzer import QualityAnalyzer
from src.simulation.simulation_graph import SimulationGraph
from src.simulation.failure_simulator import FailureSimulator, FailureScenario

from .validator import Validator, ValidationResult, ValidationTargets

class ValidationPipeline:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.exporter = GraphExporter(uri, user, password)
        self.logger = logging.getLogger(__name__)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    def close(self): self.exporter.close()

    def run(self, targets: Optional[ValidationTargets] = None) -> ValidationResult:
        self.logger.info("Initializing Validation Pipeline...")
        
        # --- Phase 1: Prediction (Analysis on Derived Graph) ---
        self.logger.info("[1/3] Generating Predictions (Analysis Module)...")
        # Fetch the abstract dependency graph (App -> App, etc.)
        # Note: We rely on the Importer having created DEPENDS_ON relationships
        derived_graph = self.exporter.get_graph_data()
        
        # Run Analyzers
        struct_res = StructuralAnalyzer().analyze(derived_graph)
        quality_res = QualityAnalyzer().analyze(struct_res)
        
        # Extract Predictions (Predictor = Overall Quality Score)
        # In this context, "Quality" often inversely correlates with "Risk/Impact" 
        # OR we can use PageRank/Betweenness from Structural Analysis as the predictor of Criticality.
        # Let's use the 'Overall Score' (Criticality) calculated in QualityAnalyzer.
        # Note: The QualityAnalyzer in Turn 2 computes a "Level" and "Scores". 
        # Higher "scores.overall" (Q) generally implies higher structural importance/quality.
        predicted_scores = {c.id: c.scores.overall for c in quality_res.components}
        
        # Get Component Metadata for grouping
        comp_types = {c.id: c.component_type for c in derived_graph.components}

        # --- Phase 2: Ground Truth (Simulation on Structural Graph) ---
        self.logger.info("[2/3] Generating Ground Truth (Simulation Module)...")
        # Fetch the raw physical graph (App -> Node, etc.)
        structural_graph = self.exporter.get_structural_graph()
        sim_graph = SimulationGraph(structural_graph)
        simulator = FailureSimulator(sim_graph)
        
        actual_scores = {}
        total_nodes = len(structural_graph.components)
        
        # Simulate failure for EVERY component to measure impact
        # This gives us the "Ground Truth" criticality
        for i, comp in enumerate(structural_graph.components):
            # Only simulate if we have a prediction for it
            if comp.id in predicted_scores:
                res = simulator.simulate(FailureScenario(comp.id, "Validation"))
                
                # Metric: Normalized Total Impact (Cascade Size)
                # impact = (failed_nodes - 1) / (total_nodes - 1)
                total_impact = sum(res.impact_counts.values())
                cascade_size = max(0, total_impact - 1) # Exclude self
                actual_scores[comp.id] = cascade_size / max(1, total_nodes)
                
            if i % 20 == 0:
                self.logger.info(f"      Simulated {i}/{total_nodes} scenarios...")

        # --- Phase 3: Validation ---
        self.logger.info("[3/3] Validating Model...")
        validator = Validator(targets)
        result = validator.validate(predicted_scores, actual_scores, comp_types)
        
        return result