"""
Validation Pipeline

Orchestrates the validation process for specific graph layers.
1. Analysis: Runs GraphAnalyzer to get Predicted Criticality Scores.
2. Simulation: Runs Simulator (Exhaustive) to get Actual Impact Scores.
3. Validation: Compares Prediction vs Reality using Validator.
"""

import logging
from typing import Optional

from src.analysis.analyzer import GraphAnalyzer
from src.simulation.simulator import Simulator
from src.validation.validator import Validator, ValidationResult, ValidationTargets

class ValidationPipeline:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        # We use the Facades to ensure we validate the exact logic used in production
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
        self.logger.info(f"Initializing Validation Pipeline for Layer: {layer.upper()}")
        
        # --- Phase 1: Prediction (Analysis Module) ---
        self.logger.info("[1/3] Generating Predictions (Analysis)...")
        
        # 1. Run Analysis based on layer
        if layer == "complete":
            analysis_res = self.analyzer.analyze()
        else:
            analysis_res = self.analyzer.analyze_layer(layer)
            
        # 2. Extract Scores
        # We validate the 'Overall' criticality score against total impact.
        # Structure: analysis_res['results'] -> QualityAnalysisResult -> components -> ComponentQuality
        components = analysis_res["results"].components
        predicted_scores = {c.id: c.scores.overall for c in components}
        component_types = {c.id: c.type for c in components}
        
        self.logger.info(f"      Obtained predictions for {len(predicted_scores)} components.")

        # --- Phase 2: Ground Truth (Simulation Module) ---
        self.logger.info("[2/3] Generating Ground Truth (Exhaustive Simulation)...")
        
        # 1. Run Exhaustive Simulation
        # This will simulate failure for every node in the target layer
        sim_results = self.simulator.run_exhaustive_failure_sim(
            layer=layer,
            threshold=0.5, # Standard validation params
            probability=0.7,
            depth=5
        )
        
        # 2. Extract Impact Scores
        actual_scores = {res.initial_failure: res.impact_score for res in sim_results}
        
        self.logger.info(f"      Obtained ground truth for {len(actual_scores)} components.")

        # --- Phase 3: Validation (Comparison) ---
        self.logger.info("[3/3] Validating Model Accuracy...")
        
        validator = Validator(targets)
        result = validator.validate(
            predicted_scores=predicted_scores, 
            actual_scores=actual_scores, 
            component_types=component_types,
            context=f"Layer: {layer.capitalize()}"
        )
        
        return result