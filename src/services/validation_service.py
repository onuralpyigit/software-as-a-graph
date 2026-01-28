"""
Validation Service

Orchestrates the validation pipeline by coordinating Analysis and Simulation results.
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.models.validation.metrics import ValidationTargets
from src.models.validation.results import ValidationResult, LayerValidationResult, PipelineResult
from src.services.validation.validator import Validator
from src.services.analysis_service import AnalysisService
from src.services.simulation_service import SimulationService

# Layer definitions (moved from pipeline.py)
LAYER_DEFINITIONS = {
    "app": "Application Layer",
    "infra": "Infrastructure Layer",
    "mw-app": "Middleware-Application Layer",
    "mw-infra": "Middleware-Infrastructure Layer",
    "system": "Complete System",
}

class ValidationService:
    """
    Application Service for Graph Validation.
    """
    
    def __init__(
        self,
        analysis_service: AnalysisService,
        simulation_service: SimulationService,
        targets: Optional[ValidationTargets] = None
    ):
        self.analysis = analysis_service
        self.simulation = simulation_service
        self.targets = targets or ValidationTargets()
        self.validator = Validator(targets=self.targets)
        self.logger = logging.getLogger(__name__)
    
    def validate_layers(self, layers: List[str] = ["app", "infra", "system"]) -> PipelineResult:
        """Run validation for multiple layers."""
        valid_layers = [l for l in layers if l in LAYER_DEFINITIONS]
        if not valid_layers:
            self.logger.warning(f"No valid layers provided from: {layers}")
            return PipelineResult(
                timestamp=datetime.now().isoformat(),
                targets=self.targets,
                all_passed=False
            )

        results = {}
        passed_count = 0
        total_components = 0
        
        for layer in valid_layers:
            self.logger.info(f"Validating layer: {layer}")
            try:
                result = self._validate_single_layer(layer)
                results[layer] = result
                if result.passed:
                    passed_count += 1
                total_components += result.predicted_components
            except Exception as e:
                self.logger.error(f"Failed to validate layer {layer}: {e}")
                results[layer] = LayerValidationResult(
                    layer=layer, layer_name=LAYER_DEFINITIONS[layer], warnings=[str(e)]
                )
        
        all_passed = (passed_count == len(valid_layers))
        
        return PipelineResult(
            timestamp=datetime.now().isoformat(),
            layers=results,
            total_components=total_components,
            layers_passed=passed_count,
            all_passed=all_passed,
            targets=self.targets
        )

    def _validate_single_layer(self, layer: str) -> LayerValidationResult:
        """Validate a single layer."""
        # 1. Analysis
        self.logger.info("Running analysis...")
        analysis_result = self.analysis.analyze_layer(layer)
        pred_scores = {c.id: c.scores.overall for c in analysis_result.quality.components}
        comp_types = {c.id: c.type for c in analysis_result.quality.components}
        comp_names = {c.id: c.structural.name for c in analysis_result.quality.components}
        
        # 2. Simulation
        self.logger.info("Running simulation...")
        # Since SimulationService is designed as a context manager but we are inside a service,
        # we might need to handle the session management. 
        # Ideally, we assume the repository is shared or handled.
        # However, checking SimulationService, it loads graph on enter.
        # We can use it as a context manager here if needed, or if it's already injected with repo
        # we can just call methods if they don't depend on 'enter' state too heavily.
        # Checking: SimulationService._load_graph is called in __enter__.
        # So we should use it as context manager or ensure graph is loaded.
        
        with self.simulation as sim:
            sim_results = sim.run_failure_simulation_exhaustive(layer=layer)
        
        actual_scores = {r.target_id: r.impact.composite_impact for r in sim_results}
        
        # 3. Validation
        self.logger.info("Validating...")
        validation_res = self.validator.validate(
            predicted_scores=pred_scores,
            actual_scores=actual_scores,
            component_types=comp_types,
            layer=layer,
            context=LAYER_DEFINITIONS[layer]
        )
        
        return LayerValidationResult(
            layer=layer,
            layer_name=LAYER_DEFINITIONS[layer],
            predicted_components=validation_res.predicted_count,
            simulated_components=validation_res.actual_count,
            matched_components=validation_res.matched_count,
            validation_result=validation_res,
            
            spearman=validation_res.overall.correlation.spearman,
            f1_score=validation_res.overall.classification.f1_score,
            precision=validation_res.overall.classification.precision,
            recall=validation_res.overall.classification.recall,
            top_5_overlap=validation_res.overall.ranking.top_5_overlap,
            rmse=validation_res.overall.error.rmse,
            
            passed=validation_res.passed,
            comparisons=validation_res.overall.components,
            warnings=validation_res.warnings,
            component_names=comp_names
        )

    def validate_from_data(self, predicted, actual) -> ValidationResult:
        """Quick validation helper."""
        return self.validator.validate(predicted, actual, context="Quick Validation")
