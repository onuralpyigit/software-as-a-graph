"""
Validation Service

Application service implementing IValidationUseCase.
Orchestrates the validation pipeline by coordinating Analysis and Simulation results.
"""
import logging
from typing import List, Optional, Any
from datetime import datetime

from .models import ValidationTargets, ValidationResult, LayerValidationResult, PipelineResult
from .validator import Validator
from src.core.layers import AnalysisLayer, get_simulation_layer_definition as get_layer_definition

class ValidationService:
    """
    Application Service for Graph Validation.
    """
    
    def __init__(
        self,
        analysis_service: Any,
        simulation_service: Any,
        targets: Optional[ValidationTargets] = None,
        ndcg_k: int = 10
    ):
        self.analysis = analysis_service
        self.simulation = simulation_service
        self.targets = targets or ValidationTargets()
        self.validator = Validator(targets=self.targets, ndcg_k=ndcg_k)
        self.logger = logging.getLogger(__name__)
    
    def validate_layers(self, layers: Optional[List[str]] = None) -> PipelineResult:
        """Run validation for multiple layers."""
        if layers is None:
            layers = ["app", "infra", "mw", "system"] # Default layers

        valid_layers = []
        for l in layers:
            try:
                valid_layers.append(AnalysisLayer.from_string(l))
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
                result = self.validate_single_layer(layer.value)
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

    def validate_single_layer(self, layer: str) -> LayerValidationResult:
        """Validate a single layer."""
        # Convert string to enum if needed, but here we work with string for interface compatibility
        try:
            sim_layer = AnalysisLayer.from_string(layer)
        except ValueError:
             raise ValueError(f"Unknown layer: {layer}")
             
        layer_def = get_layer_definition(sim_layer)
        
        # 1. Analysis
        self.logger.info("Running analysis...")
        # AnalysisService uses AnalysisLayer which has same string values as SimulationLayer
        analysis_result = self.analysis.analyze_layer(sim_layer.value)
        pred_scores = {c.id: c.scores.overall for c in analysis_result.quality.components}
        pred_reliability   = {c.id: c.scores.reliability   for c in analysis_result.quality.components}
        pred_availability  = {c.id: c.scores.availability  for c in analysis_result.quality.components}
        pred_vulnerability = {c.id: c.scores.vulnerability for c in analysis_result.quality.components}
        comp_types = {c.id: c.type for c in analysis_result.quality.components}
        comp_names = {c.id: c.structural.name for c in analysis_result.quality.components}
        
        # 2. Simulation
        self.logger.info("Running simulation...")
        
        # SimulationService handles its own graph loading but we can use context manager if needed.
        # However, calling run_failure_simulation_exhaustive internally uses self.graph property 
        # which auto-loads the graph.
        
        sim_results = self.simulation.run_failure_simulation_exhaustive(layer=sim_layer.value)
        
        actual_scores        = {r.target_id: r.impact.composite_impact  for r in sim_results}
        actual_reachability  = {r.target_id: r.impact.reachability_loss for r in sim_results}
        actual_fragmentation = {r.target_id: r.impact.fragmentation     for r in sim_results}
        actual_throughput    = {r.target_id: r.impact.throughput_loss   for r in sim_results}
        
        # 3. Overall validation
        self.logger.info("Validating...")
        validation_res = self.validator.validate(
            predicted_scores=pred_scores,
            actual_scores=actual_scores,
            component_types=comp_types,
            layer=sim_layer.value,
            context=layer_def.name
        )

        # 4. Per-dimension correlation — Reliability uses IR(v) ground truth
        # (ρ(R(v), IR(v)) instead of ρ(R(v), reachability_loss))
        self.logger.info("Computing reliability-specific validation (IR(v) ground truth)...")
        dimensional_validation = {}
        from src.validation.metric_calculator import (
            calculate_correlation, calculate_ccr_at_k, calculate_cme
        )

        # Build reliability-specific ground truth IR(v)
        actual_reliability_impact = {
            r.target_id: r.impact.reliability_impact for r in sim_results
        }

        # ρ(R(v), IR(v)): reliability predictor vs cascade-specific ground truth
        try:
            common_r = sorted(set(pred_reliability) & set(actual_reliability_impact))
            if len(common_r) >= 3:
                p_r_vals = [float(pred_reliability[k])          for k in common_r]
                a_ir_vals = [float(actual_reliability_impact[k]) for k in common_r]
                r_corr = calculate_correlation(p_r_vals, a_ir_vals)
                reliability_spearman = r_corr.spearman

                # CCR@5: Cascade Capture Rate
                pred_r_dict = {k: pred_reliability[k]          for k in common_r}
                act_ir_dict = {k: actual_reliability_impact[k] for k in common_r}
                ccr5 = calculate_ccr_at_k(pred_r_dict, act_ir_dict, k=5)
                cme = calculate_cme(pred_r_dict, act_ir_dict)

                dimensional_validation["reliability"] = {
                    "spearman": round(r_corr.spearman, 4),
                    "spearman_p": round(r_corr.spearman_p, 6),
                    "ccr_5": round(ccr5, 4),
                    "cme": round(cme, 4),
                    "n": len(common_r),
                    "ground_truth": "IR(v)",
                }
                self.logger.info(
                    "Reliability dim [%s]: \u03c1(R,IR)=%.3f (n=%d), CCR@5=%.3f, CME=%.4f",
                    sim_layer.value, r_corr.spearman, len(common_r), ccr5, cme,
                )

                # False-alarm diagnostic: HIGH/CRITICAL R predicted but LOW IR
                pred_r_sorted = sorted(pred_r_dict.items(), key=lambda x: x[1], reverse=True)
                top_r_ids = {cid for cid, _ in pred_r_sorted[:5]}
                ir_sorted = sorted(act_ir_dict.items(), key=lambda x: x[1], reverse=True)
                bottom_ir_n = max(1, len(ir_sorted) // 2)
                bottom_ir_ids = {cid for cid, _ in ir_sorted[bottom_ir_n:]}
                false_alarms = top_r_ids & bottom_ir_ids
                if false_alarms:
                    self.logger.warning(
                        "Reliability false alarms (HIGH R(v) but LOW IR(v)): %s",
                        sorted(false_alarms),
                    )
            else:
                reliability_spearman = 0.0
                ccr5 = 0.0
                cme = 0.0
        except Exception as e:
            self.logger.debug("Reliability-specific validation skipped: %s", e)
            reliability_spearman = 0.0
            ccr5 = 0.0
            cme = 0.0

        # Availability and Vulnerability dimensional correlations (existing; kept for reference)
        for dim_name, pred_dim, actual_dim in [
            ("availability",   pred_availability,  actual_fragmentation),
            ("vulnerability",  pred_vulnerability, actual_throughput),
        ]:
            try:
                common = sorted(set(pred_dim) & set(actual_dim))
                if len(common) >= 3:
                    p_vals = [float(pred_dim[k]) for k in common]
                    a_vals = [float(actual_dim[k]) for k in common]
                    corr = calculate_correlation(p_vals, a_vals)
                    dimensional_validation[dim_name] = {
                        "spearman": round(corr.spearman, 4),
                        "n": len(common),
                    }
                    self.logger.info(
                        "Dim validation [%s] %s: \u03c1=%.3f (n=%d)",
                        sim_layer.value, dim_name, corr.spearman, len(common),
                    )
            except Exception as e:
                self.logger.debug("Dimensional validation skipped for %s: %s", dim_name, e)


        return LayerValidationResult(
            layer=sim_layer.value,
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
            top_10_overlap=validation_res.overall.ranking.top_10_overlap,
            rmse=validation_res.overall.error.rmse,
            reliability_spearman=reliability_spearman,
            
            passed=validation_res.passed,
            comparisons=validation_res.overall.components,
            warnings=validation_res.warnings,
            component_names=comp_names,
            dimensional_validation=dimensional_validation,
        )

    def validate_from_data(self, predicted, actual) -> ValidationResult:
        """Quick validation helper."""
        return self.validator.validate(predicted, actual, context="Quick Validation")
