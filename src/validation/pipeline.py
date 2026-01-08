"""
Validation Pipeline

Orchestrates the validation process for specific graph layers.
1. Analysis: Runs GraphAnalyzer to get Predicted Criticality Scores.
2. Simulation: Runs Simulator (Exhaustive) to get Actual Impact Scores.
3. Validation: Compares Prediction vs Reality using Validator.
"""

import logging
from typing import Optional, Dict, Any

from src.analysis.analyzer import GraphAnalyzer
from src.simulation.simulator import Simulator
from src.validation.validator import Validator, ValidationResult, ValidationTargets

class ValidationPipeline:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        # Use Facades to ensure exact logic match with production/reporting
        self.analyzer = GraphAnalyzer(uri, user, password)
        self.simulator = Simulator(uri, user, password)
        self.logger = logging.getLogger(__name__)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    
    def close(self): 
        self.analyzer.close()
        self.simulator.close()

    def run(self, layer: str = "complete", targets: Optional[ValidationTargets] = None) -> ValidationResult:
        """
        Run the full validation pipeline for a specific layer.
        
        Args:
            layer: 'application', 'infrastructure', or 'complete'
            targets: Validation success criteria
        """
        self.logger.info(f"--- Starting Validation Pipeline: {layer.upper()} LAYER ---")
        
        # --- Phase 1: Prediction (Analysis Module) ---
        self.logger.info("[1/3] Generating Predictions (Analysis)...")
        
        # 1. Run Analysis based on layer
        if layer == "complete":
            analysis_res = self.analyzer.analyze()
        else:
            analysis_res = self.analyzer.analyze_layer(layer)
            
        # 2. Extract Predicted Scores
        # We validate the 'Overall' criticality score.
        components = analysis_res["results"].components
        
        # Map: ID -> Overall Score
        predicted_scores = {c.id: c.scores.overall for c in components}
        component_types = {c.id: c.type for c in components}
        
        self.logger.info(f"      Predictions generated for {len(predicted_scores)} components.")

        # --- Phase 2: Ground Truth (Simulation Module) ---
        self.logger.info("[2/3] Generating Ground Truth (Exhaustive Simulation)...")
        
        # 1. Run Exhaustive Simulation
        # This simulates failure for every node in the target layer and measures impact
        sim_results = self.simulator.run_exhaustive_failure_sim(
            layer=layer,
            threshold=0.5, 
            probability=0.7,
            depth=5
        )
        
        # 2. Extract Actual Impact Scores
        # Map: ID -> System Impact Score (0.0 - 1.0)
        actual_scores = {res.initial_failure: res.impact_score for res in sim_results}
        
        self.logger.info(f"      Ground truth simulated for {len(actual_scores)} components.")

        # --- Phase 3: Validation (Comparison) ---
        self.logger.info("[3/3] Validating Model Accuracy...")
        
        validator = Validator(targets)
        result = validator.validate(
            predicted_scores=predicted_scores, 
            actual_scores=actual_scores, 
            component_types=component_types,
            context=f"Layer: {layer.capitalize()}"
        )
        
        self.logger.info(f"Validation Complete. Passed: {result.overall.passed}")
        return result