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
from src.models.simulation.layers import SimulationLayer, SIMULATION_LAYERS, get_layer_definition

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
        valid_layers = []
        for l in layers:
            try:
                valid_layers.append(SimulationLayer.from_string(l))
            except ValueError:
                self.logger.warning(f"Skipping unknown layer: {l}")

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
            self.logger.info(f"Validating layer: {layer.value}")
            try:
                result = self._validate_single_layer(layer)
                results[layer.value] = result
                if result.passed:
                    passed_count += 1
                total_components += result.predicted_components
            except Exception as e:
                self.logger.error(f"Failed to validate layer {layer.value}: {e}")
                self.logger.exception("Validation exception details:")
                layer_def = get_layer_definition(layer)
                results[layer.value] = LayerValidationResult(
                    layer=layer.value, layer_name=layer_def.name, warnings=[str(e)]
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

    def _validate_single_layer(self, layer: SimulationLayer) -> LayerValidationResult:
        """Validate a single layer."""
        layer_def = get_layer_definition(layer)
        
        # 1. Analysis
        self.logger.info("Running analysis...")
        # AnalysisService uses AnalysisLayer which has same string values as SimulationLayer
        analysis_result = self.analysis.analyze_layer(layer.value)
        pred_scores = {c.id: c.scores.overall for c in analysis_result.quality.components}
        comp_types = {c.id: c.type for c in analysis_result.quality.components}
        comp_names = {c.id: c.structural.name for c in analysis_result.quality.components}
        
        # 2. Simulation
        self.logger.info("Running simulation...")
        
        # SimulationService handles its own graph loading but we can use context manager if needed.
        # However, calling run_failure_simulation_exhaustive internally uses self.graph property 
        # which auto-loads the graph.
        
        sim_results = self.simulation.run_failure_simulation_exhaustive(layer=layer.value)
        
        actual_scores = {r.target_id: r.impact.composite_impact for r in sim_results}
        
        # 3. Validation
        self.logger.info("Validating...")
        validation_res = self.validator.validate(
            predicted_scores=pred_scores,
            actual_scores=actual_scores,
            component_types=comp_types,
            layer=layer.value,
            context=layer_def.name
        )
        
        return LayerValidationResult(
            layer=layer.value,
            layer_name=layer_def.name,
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
