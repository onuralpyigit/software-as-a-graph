"""
Validation Pipeline

Orchestrates the validation process:
1. Analysis (Prediction): Uses topological metrics (Formula 6) to predict criticality.
2. Simulation (Ground Truth): Uses failure injection (Formula 7) to measure actual impact.
3. Comparison: Validates model accuracy using statistical framework.

Reference: PhD Progress Report - Section 6.3 (Separation of Prediction and Validation)
"""

import logging
from typing import Optional

from src.core.graph_exporter import GraphExporter
from src.analysis.structural_analyzer import StructuralAnalyzer
from src.analysis.quality_analyzer import QualityAnalyzer
from src.simulation.simulation_graph import SimulationGraph
from src.simulation.failure_simulator import FailureSimulator, FailureScenario
from src.validation.validator import Validator, ValidationResult, ValidationTargets

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
        # Goal: Calculate Composite Criticality Score (Formula 6)
        self.logger.info("[1/3] Generating Predictions (Analysis Module)...")
        derived_graph = self.exporter.get_graph_data()
        
        # Run Analyzers to get topological metrics
        struct_res = StructuralAnalyzer().analyze(derived_graph)
        quality_res = QualityAnalyzer().analyze(struct_res)
        
        # Extract Cscore(v). 
        # Assuming QualityAnalyzer.scores.overall implements the Composite Score (Formula 6)
        # (0.35*CB + 0.25*AP + 0.20*CD + 0.20*PR)
        predicted_scores = {c.id: c.scores.overall for c in quality_res.components}
        comp_types = {c.id: c.component_type for c in derived_graph.components}

        # --- Phase 2: Ground Truth (Simulation on Structural Graph) ---
        # Goal: Calculate Actual Impact Score (Formula 7)
        self.logger.info("[2/3] Generating Ground Truth (Simulation Module)...")
        structural_graph = self.exporter.get_structural_graph()
        sim_graph = SimulationGraph(structural_graph)
        simulator = FailureSimulator(sim_graph)
        
        actual_scores = {}
        total_nodes = len(structural_graph.components)
        
        self.logger.info(f"      Simulating failures for {total_nodes} components...")
        
        for i, comp in enumerate(structural_graph.components):
            # Simulation Step
            # We fail the component and measure: 
            # Reachability Loss, Fragmentation, Cascade Extent
            res = simulator.simulate(FailureScenario(comp.id, "Validation"))
            
            # The refactored FailureResult contains 'impact_score' (Formula 7)
            # Impact(v) = 0.5*Reach + 0.3*Frag + 0.2*Cascade
            actual_scores[comp.id] = res.impact_score
            
            if (i + 1) % 50 == 0:
                self.logger.info(f"      Progress: {i + 1}/{total_nodes}...")

        # --- Phase 3: Validation ---
        # Goal: Compare Cscore vs Impact
        self.logger.info("[3/3] Validating Model...")
        validator = Validator(targets)
        result = validator.validate(predicted_scores, actual_scores, comp_types)
        
        return result