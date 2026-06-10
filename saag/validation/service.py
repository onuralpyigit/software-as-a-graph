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
from saag.core.layers import AnalysisLayer, get_simulation_layer_definition as get_layer_definition

class ValidationService:
    """
    Application Service for Graph Validation.
    """
    
    def __init__(
        self,
        analysis_service: Any,
        prediction_service: Any,
        simulation_service: Any,
        targets: Optional[ValidationTargets] = None,
        ndcg_k: int = 10
    ):
        self.analysis = analysis_service
        self.prediction = prediction_service
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
        
        # 1.5 Prediction
        self.logger.info("Running prediction...")
        prediction_result = self.prediction.predict_quality(analysis_result.structural)
        
        # Enrichment for compatibility
        analysis_result.quality = prediction_result
        
        pred_scores = {c.id: c.scores.overall for c in prediction_result.components}
        pred_reliability   = {c.id: c.scores.reliability   for c in prediction_result.components}
        pred_availability  = {c.id: c.scores.availability  for c in prediction_result.components}
        pred_security      = {c.id: c.scores.security      for c in prediction_result.components}
        comp_types = {c.id: c.type for c in prediction_result.components}
        comp_names = {c.id: c.structural.name for c in prediction_result.components}
        
        # 2. Simulation
        self.logger.info("Running simulation...")
        
        # SimulationService handles its own graph loading but we can use context manager if needed.
        # However, calling run_failure_simulation_exhaustive internally uses self.graph property 
        # which auto-loads the graph.
        
        sim_results = self.simulation.run_failure_simulation_exhaustive(layer=sim_layer.value)
        
        return self.validate_single_layer_from_results(analysis_result, sim_results, layer)

    def validate_single_layer_from_results(
        self, 
        analysis_result: Any, 
        sim_results: List[Any], 
        layer: str
    ) -> LayerValidationResult:
        """
        Validate a single layer using provided analysis and simulation result objects.
        
        Args:
            analysis_result: LayerAnalysisResult
            sim_results: List[FailureResult]
            layer: Layer value string
        """
        from saag.core import SimulationLayer, get_layer_definition
        sim_layer = SimulationLayer.from_string(layer)
        layer_def = get_layer_definition(sim_layer)
        sim_val_layer = sim_layer # Use for sim_layer.value if needed
        
        # 1. Prepare data
        pred_scores       = {c.id: c.scores.overall        for c in analysis_result.quality.components}
        pred_reliability   = {c.id: c.scores.reliability    for c in analysis_result.quality.components}
        pred_maintainability = {c.id: c.scores.maintainability for c in analysis_result.quality.components}
        pred_availability  = {c.id: c.scores.availability  for c in analysis_result.quality.components}
        pred_security      = {c.id: c.scores.security      for c in analysis_result.quality.components}
        comp_types = {c.id: c.type for c in analysis_result.quality.components}
        comp_names = {c.id: c.structural.name for c in analysis_result.quality.components}
        
        from saag.validation.validator import robust_sigmoid_scale_dict

        actual_scores        = robust_sigmoid_scale_dict({r.target_id: r.impact.composite_impact  for r in sim_results})
        actual_reachability  = robust_sigmoid_scale_dict({r.target_id: r.impact.reachability_loss for r in sim_results})
        actual_fragmentation = robust_sigmoid_scale_dict({r.target_id: r.impact.fragmentation     for r in sim_results})
        actual_throughput    = robust_sigmoid_scale_dict({r.target_id: r.impact.throughput_loss   for r in sim_results})

        # 3. Overall validation
        self.logger.info("Validating...")
        validation_res = self.validator.validate(
            predicted_scores=pred_scores,
            actual_scores=actual_scores,
            impact_data=sim_results,
            component_types=comp_types,
            layer=sim_layer.value,
            context=layer_def.name
        )

        # Rule-Based Baseline Verification
        rule_based_spearman = float(validation_res.overall.correlation.spearman)
        rule_based_macro_f1 = float(validation_res.overall.classification.macro_f1)
        rule_based_ndcg_10 = float(validation_res.overall.ranking.ndcg_10)
        
        rule_based_passed = (
            rule_based_spearman >= 0.85 and
            rule_based_macro_f1 >= 0.88 and
            rule_based_ndcg_10 >= 0.90
        )
        
        rule_based_baseline_metrics = {
            "spearman": round(rule_based_spearman, 4),
            "macro_f1": round(rule_based_macro_f1, 4),
            "ndcg_10": round(rule_based_ndcg_10, 4),
            "passed": rule_based_passed,
            "targets": {"spearman": 0.85, "macro_f1": 0.88, "ndcg_10": 0.90}
        }

        # GNN Forecasting Evaluation
        gnn_forecasting_metrics = None
        gnn_result = None
        if hasattr(self.prediction, "predict_quality_with_gnn") and getattr(self.prediction, "gnn_checkpoint_dir", None):
            try:
                graph = getattr(analysis_result, "graph", None)
                if graph is not None:
                    gnn_result = self.prediction.predict_quality_with_gnn(
                        analysis_result.structural,
                        graph,
                        sim_results
                    )
            except Exception as e:
                self.logger.warning(f"GNN prediction for validation failed: {e}")

        if gnn_result is not None and getattr(gnn_result, "gnn_metrics", None) is not None:
            gnn_metrics = gnn_result.gnn_metrics
            gnn_spearman = float(gnn_metrics.spearman_rho)
            gnn_macro_f1 = float(gnn_metrics.macro_f1)
            gnn_ndcg_10 = float(gnn_metrics.ndcg_10)
            
            gnn_passed = (
                gnn_spearman >= 0.85 and
                gnn_macro_f1 >= 0.88 and
                gnn_ndcg_10 >= 0.90
            )
            
            gnn_forecasting_metrics = {
                "spearman": round(gnn_spearman, 4),
                "macro_f1": round(gnn_macro_f1, 4),
                "ndcg_10": round(gnn_ndcg_10, 4),
                "bce_loss": round(float(gnn_metrics.bce_loss), 4),
                "regression_curve": {
                    "slope": round(float(gnn_metrics.regression_slope), 4),
                    "intercept": round(float(gnn_metrics.regression_intercept), 4),
                    "r2": round(float(gnn_metrics.regression_r2), 4),
                },
                "passed": gnn_passed,
                "targets": {"spearman": 0.85, "macro_f1": 0.88, "ndcg_10": 0.90}
            }

        # 4. Per-dimension correlation — Reliability uses IR(v) ground truth
        # (ρ(R(v), IR(v)) instead of ρ(R(v), reachability_loss))
        self.logger.info("Computing reliability-specific validation (IR(v) ground truth)...")
        dimensional_validation = {}
        dimensional_scatter = {}
        confidence_intervals = {}
        from saag.validation.metric_calculator import (
            calculate_correlation, calculate_ccr_at_k, calculate_cme,
            calculate_cocr_at_k, calculate_weighted_kappa_cta, calculate_bottleneck_precision,
        )

        # Build reliability-specific ground truth IR(v)
        actual_reliability_impact = robust_sigmoid_scale_dict({
            r.target_id: r.impact.reliability_impact for r in sim_results
        })

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
                
                # Full scatter points
                data_r = []
                for cid in common_r:
                    level = "MINIMAL"
                    # Try to get level from analysis_result components
                    comp = next((c for c in analysis_result.quality.components if c.id == cid), None)
                    if comp:
                        level = getattr(comp.levels.overall, 'name', str(comp.levels.overall))
                    data_r.append((cid, float(pred_reliability[cid]), float(actual_reliability_impact[cid]), level))
                
                dimensional_scatter["reliability"] = data_r
                confidence_intervals["reliability"] = (r_corr.spearman_ci_lower, r_corr.spearman_ci_upper)
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

        # 5. Maintainability-specific validation — ρ(M(v), IM(v))
        self.logger.info("Computing maintainability-specific validation (IM(v) ground truth)...")
        maintainability_spearman = 0.0

        pred_maintainability = {c.id: c.scores.maintainability for c in analysis_result.quality.components}
        actual_maintainability_impact = robust_sigmoid_scale_dict({
            r.target_id: r.impact.maintainability_impact for r in sim_results
        })

        try:
            common_m = sorted(set(pred_maintainability) & set(actual_maintainability_impact))
            if len(common_m) >= 3:
                p_m_vals = [float(pred_maintainability[k])           for k in common_m]
                a_im_vals = [float(actual_maintainability_impact[k]) for k in common_m]
                m_corr = calculate_correlation(p_m_vals, a_im_vals)
                maintainability_spearman = m_corr.spearman

                # COCR@5
                pred_m_dict = {k: pred_maintainability[k]           for k in common_m}
                act_im_dict = {k: actual_maintainability_impact[k]  for k in common_m}
                cocr5 = calculate_cocr_at_k(pred_m_dict, act_im_dict, k=5)

                # Weighted-κ Coupling Tier Agreement
                kappa_cta = calculate_weighted_kappa_cta(pred_m_dict, act_im_dict)

                # Bottleneck Precision: needs per-component BT and w_out scores
                # BT is structural.betweenness; w_out is structural.dependency_weight_out
                pred_bt_dict = {
                    c.id: c.structural.betweenness
                    for c in analysis_result.quality.components
                    if c.id in common_m
                }
                pred_wout_dict = {
                    c.id: c.structural.dependency_weight_out
                    for c in analysis_result.quality.components
                    if c.id in common_m
                }
                bp = calculate_bottleneck_precision(pred_bt_dict, pred_wout_dict, act_im_dict)

                dimensional_validation["maintainability"] = {
                    "spearman": round(m_corr.spearman, 4),
                    "spearman_p": round(m_corr.spearman_p, 6),
                    "cocr_5": round(cocr5, 4),
                    "weighted_kappa_cta": round(kappa_cta, 4),
                    "bottleneck_precision": round(bp, 4),
                    "n": len(common_m),
                    "ground_truth": "IM(v)",
                }
                
                # Full scatter points
                data_m = []
                for cid in common_m:
                    level = "MINIMAL"
                    comp = next((c for c in analysis_result.quality.components if c.id == cid), None)
                    if comp:
                        level = getattr(comp.levels.overall, 'name', str(comp.levels.overall))
                    data_m.append((cid, float(pred_maintainability[cid]), float(actual_maintainability_impact[cid]), level))
                
                dimensional_scatter["maintainability"] = data_m
                confidence_intervals["maintainability"] = (m_corr.spearman_ci_lower, m_corr.spearman_ci_upper)
                self.logger.info(
                    "Maintainability dim [%s]: ρ(M,IM)=%.3f (n=%d), "
                    "COCR@5=%.3f, κ_CTA=%.3f, BP=%.3f",
                    sim_layer.value, m_corr.spearman, len(common_m),
                    cocr5, kappa_cta, bp,
                )
            else:
                maintainability_spearman = 0.0
        except Exception as e:
            self.logger.debug("Maintainability-specific validation skipped: %s", e)
            maintainability_spearman = 0.0

        # 5. Security-specific validation — ρ(S(v), IS(v)) + AHCR@5 + FTR + APAR + CDCC
        self.logger.info("Computing security-specific validation (IS(v) ground truth)...")
        security_spearman = 0.0

        from saag.validation.metric_calculator import (
            calculate_ahcr_at_k, calculate_ftr, calculate_apar
        )

        actual_security_impact = robust_sigmoid_scale_dict({
            r.target_id: r.impact.security_impact for r in sim_results
        })
        actual_attack_reach = robust_sigmoid_scale_dict({
            r.target_id: r.impact.attack_reach for r in sim_results
        })

        try:
            common_s = sorted(set(pred_security) & set(actual_security_impact))
            if len(common_s) >= 3:
                p_s_vals  = [float(pred_security[k]) for k in common_s]
                a_is_vals = [float(actual_security_impact[k]) for k in common_s]
                s_corr = calculate_correlation(p_s_vals, a_is_vals)
                security_spearman = s_corr.spearman

                pred_s_dict = {k: pred_security[k] for k in common_s}
                act_is_dict = {k: actual_security_impact[k] for k in common_s}
                
                # AHCR@5
                ahcr_5 = calculate_ahcr_at_k(pred_s_dict, act_is_dict, k=5)
                
                # FTR
                act_reach_dict = {k: actual_attack_reach[k] for k in common_s}
                ftr = calculate_ftr(pred_s_dict, act_reach_dict, v_threshold=0.60, reach_threshold=0.10)
                
                # APAR
                paths_all = []
                for r in sim_results:
                    if hasattr(r.impact, 'critical_paths') and r.impact.critical_paths:
                        paths_all.extend(r.impact.critical_paths)
                apar = calculate_apar(pred_s_dict, paths_all, v_threshold=0.60)
                
                # CDCC: Cross-Dimensional Contamination Check (Rank divergence S vs A)
                p_a_vals_for_cdcc = [float(pred_availability[k]) for k in common_s if k in pred_availability]
                p_s_vals_for_cdcc = [float(pred_security[k]) for k in common_s if k in pred_availability]
                
                cdcc = 0.0
                if len(p_a_vals_for_cdcc) >= 3:
                    from saag.validation.metric_calculator import spearman_correlation
                    cdcc_res, _ = spearman_correlation(p_s_vals_for_cdcc, p_a_vals_for_cdcc)
                    cdcc = cdcc_res

                dimensional_validation["security"] = {
                    "spearman": round(s_corr.spearman, 4),
                    "spearman_p": round(s_corr.spearman_p, 6),
                    "ahcr_5": round(ahcr_5, 4),
                    "ftr": round(ftr, 4),
                    "apar": round(apar, 4),
                    "cdcc": round(cdcc, 4),
                    "n": len(common_s),
                    "ground_truth": "IS(v)",
                }
                
                # Full scatter points
                data_s = []
                for cid in common_s:
                    level = "MINIMAL"
                    comp = next((c for c in analysis_result.quality.components if c.id == cid), None)
                    if comp:
                        level = getattr(comp.levels.overall, 'name', str(comp.levels.overall))
                    data_s.append((cid, float(pred_security[cid]), float(actual_security_impact[cid]), level))
                
                dimensional_scatter["security"] = data_s
                confidence_intervals["security"] = (s_corr.spearman_ci_lower, s_corr.spearman_ci_upper)
                
                self.logger.info(
                    "Security dim [%s]: ρ(S,IS)=%.3f (n=%d), "
                    "AHCR@5=%.3f, FTR=%.3f, APAR=%.3f, CDCC=%.3f",
                    sim_layer.value, s_corr.spearman, len(common_s),
                    ahcr_5, ftr, apar, cdcc
                )
            else:
                security_spearman = 0.0
        except Exception as e:
            self.logger.debug("Security-specific validation skipped: %s", e)
            security_spearman = 0.0

        # — 6. Availability-specific validation — ρ(A, IA) + SPOF_F1 + RRI + HSRR + DASA
        self.logger.info("Computing availability-specific validation (IA(v) ground truth)...")
        availability_spearman = 0.0

        from saag.validation.metric_calculator import (
            calculate_spof_f1, calculate_rri, calculate_hsrr, calculate_dasa
        )

        actual_availability_impact = robust_sigmoid_scale_dict({
            r.target_id: r.impact.availability_impact for r in sim_results
        })
        actual_ia_out = robust_sigmoid_scale_dict({r.target_id: r.impact.ia_out for r in sim_results})
        actual_ia_in  = robust_sigmoid_scale_dict({r.target_id: r.impact.ia_in  for r in sim_results})

        try:
            common_a = sorted(set(pred_availability) & set(actual_availability_impact))
            if len(common_a) >= 3:
                p_a_vals  = [float(pred_availability[k])          for k in common_a]
                a_ia_vals = [float(actual_availability_impact[k]) for k in common_a]
                a_corr = calculate_correlation(p_a_vals, a_ia_vals)
                availability_spearman = a_corr.spearman

                # AP_c_directed/structural info
                comp_map = {c.id: c for c in analysis_result.quality.components}
                pred_ap_c_dir = {
                    cid: (1.0 if comp_map[cid].structural.is_articulation_point else 0.0)
                    for cid in common_a if cid in comp_map
                }
                pred_ap_c_out = {cid: comp_map[cid].metrics.get("ap_c_out", 0.0) for cid in common_a if cid in comp_map}
                pred_ap_c_in  = {cid: comp_map[cid].metrics.get("ap_c_in", 0.0)  for cid in common_a if cid in comp_map}
                pred_qspof   = {cid: comp_map[cid].metrics.get("qspof", 0.0)    for cid in common_a if cid in comp_map}
                pred_br      = {cid: comp_map[cid].metrics.get("bridge_score", 0.0) for cid in common_a if cid in comp_map}

                pred_a_dict = {k: pred_availability[k]          for k in common_a}
                act_ia_dict = {k: actual_availability_impact[k] for k in common_a}
                act_ia_out_sub = {k: actual_ia_out[k] for k in common_a}
                act_ia_in_sub  = {k: actual_ia_in[k]  for k in common_a}

                spof_f1_res = calculate_spof_f1(pred_ap_c_dir, act_ia_dict)
                rri_val     = calculate_rri(act_ia_dict, pred_br) # RRI(IA, BR)
                hsrr_val    = calculate_hsrr(pred_qspof, act_ia_dict, pred_ap_c_dir)
                dasa_val    = calculate_dasa(pred_ap_c_out, pred_ap_c_in, act_ia_out_sub, act_ia_in_sub)

                dimensional_validation["availability"] = {
                    "spearman":      round(a_corr.spearman, 4),
                    "spearman_p":    round(a_corr.spearman_p, 6),
                    "spof_f1":       round(spof_f1_res["f1"], 4),
                    "spof_precision": round(spof_f1_res["precision"], 4),
                    "spof_recall":   round(spof_f1_res["recall"], 4),
                    "hsrr":          round(hsrr_val, 4),
                    "dasa":          round(dasa_val, 4),
                    "rri":           round(rri_val, 4),
                    "n":             len(common_a),
                    "ground_truth":  "IA(v)",
                }
                
                # Full scatter points
                data_a = []
                for cid in common_a:
                    level = "MINIMAL"
                    comp = next((c for c in analysis_result.quality.components if c.id == cid), None)
                    if comp:
                        level = getattr(comp.levels.overall, 'name', str(comp.levels.overall))
                    data_a.append((cid, float(pred_availability[cid]), float(actual_availability_impact[cid]), level))
                
                dimensional_scatter["availability"] = data_a
                confidence_intervals["availability"] = (a_corr.spearman_ci_lower, a_corr.spearman_ci_upper)
                self.logger.info(
                    "Availability dim [%s]: \u03c1(A,IA)=%.3f (n=%d), SPOF_F1=%.3f, DASA=%.3f",
                    sim_layer.value, a_corr.spearman, len(common_a),
                    spof_f1_res["f1"], dasa_val,
                )
            else:
                availability_spearman = 0.0
        except Exception as e:
            self.logger.debug("Availability-specific validation skipped: %s", e)
            availability_spearman = 0.0

        # ── 7. Composite I*(v), ρ(Q*(v), I*(v)), Predictive Gain, Orthogonality ──
        self.logger.info("Computing composite I*(v) and system health metrics...")
        composite_spearman = 0.0
        predictive_gain = 0.0
        system_health: dict = {}

        try:
            # Build composite ground truth I*(v) = equal-weighted sum of dimension ground truths
            dim_weights = dict(r=0.25, m=0.25, a=0.25, s=0.25)
            composite_i_star: dict = {}
            for cid in (
                set(actual_reliability_impact)
                & set(actual_maintainability_impact)
                & set(actual_availability_impact)
                & set(actual_security_impact)
            ):
                composite_i_star[cid] = (
                    dim_weights["r"] * actual_reliability_impact.get(cid, 0.0)
                    + dim_weights["m"] * actual_maintainability_impact.get(cid, 0.0)
                    + dim_weights["a"] * actual_availability_impact.get(cid, 0.0)
                    + dim_weights["s"] * actual_security_impact.get(cid, 0.0)
                )

            # ρ(Q*(v), I*(v))
            common_comp = sorted(set(pred_scores) & set(composite_i_star))
            best_dim_label = "None"
            best_dim_val = 0.0
            if len(common_comp) >= 3:
                comp_corr = calculate_correlation(
                    [float(pred_scores[k])      for k in common_comp],
                    [float(composite_i_star[k]) for k in common_comp],
                )
                composite_spearman = comp_corr.spearman

                # Full scatter points
                data_comp = []
                for cid in common_comp:
                    level = "MINIMAL"
                    comp = next((c for c in analysis_result.quality.components if c.id == cid), None)
                    if comp:
                        level = getattr(comp.levels.overall, 'name', str(comp.levels.overall))
                    data_comp.append((cid, float(pred_scores[cid]), float(composite_i_star[cid]), level))
                
                dimensional_scatter["composite"] = data_comp
                confidence_intervals["composite"] = (comp_corr.spearman_ci_lower, comp_corr.spearman_ci_upper)

                dims_map = {
                    "Reliability": reliability_spearman,
                    "Maintainability": maintainability_spearman,
                    "Availability": availability_spearman,
                    "Security": security_spearman
                }
                try:
                    best_dim_label = max(dims_map, key=dims_map.get)
                    best_dim_val = float(dims_map[best_dim_label])
                    predictive_gain = float(composite_spearman) - best_dim_val
                except Exception as de:
                    self.logger.error("Predictive gain calc failed: dims_map=%s, error=%s", 
                                      {k: type(v) for k, v in dims_map.items()}, de)
                    best_dim_label = "Error"
                    best_dim_val = 0.0
                    predictive_gain = 0.0

                self.logger.info(
                    "Composite [%s]: ρ(Q*,I*)=%.3f, PG=%.3f (vs %s), n=%d",
                    sim_layer.value, composite_spearman, predictive_gain, best_dim_label, len(common_comp),
                )

            # ── Pairwise inter-dimension orthogonality check ──
            pred_r_all = {c.id: c.scores.reliability    for c in analysis_result.quality.components}
            pred_m_all = {c.id: c.scores.maintainability for c in analysis_result.quality.components}
            pred_a_all = {c.id: c.scores.availability   for c in analysis_result.quality.components}
            pred_s_all = {c.id: c.scores.security       for c in analysis_result.quality.components}

            from saag.validation.metric_calculator import spearman_correlation
            pairs = [
                ("R*vsM*", pred_r_all, pred_m_all),
                ("R*vsA*", pred_r_all, pred_a_all),
                ("R*vsS*", pred_r_all, pred_s_all),
                ("M*vsA*", pred_m_all, pred_a_all),
                ("M*vsS*", pred_m_all, pred_s_all),
                ("A*vsS*", pred_a_all, pred_s_all),
            ]
            interdim_rhos: dict = {}
            for label, d1, d2 in pairs:
                common_p = sorted(set(d1) & set(d2))
                if len(common_p) >= 3:
                    rho, _ = spearman_correlation(
                        [float(d1[k]) for k in common_p],
                        [float(d2[k]) for k in common_p],
                    )
                    interdim_rhos[label] = round(rho, 4)
                    try:
                        if abs(float(rho)) > float(self.targets.max_interdim_correlation):
                            self.logger.warning(
                                "Orthogonality violation! %s ρ=%.3f > %.2f threshold",
                                label, rho, self.targets.max_interdim_correlation,
                            )
                    except TypeError as te:
                        self.logger.error("Orthogonality check failed: label=%s, rho_type=%s, target_type=%s, error=%s",
                                          label, type(rho), type(self.targets.max_interdim_correlation), te)

            # ── System Health: H_R/M/A/V, SRI, RCI (Gini) ──
            all_comps = analysis_result.quality.components
            if all_comps:
                n_c = len(all_comps)
                w_vals = [max(c.structural.weight, 1e-9) for c in all_comps]
                w_sum = sum(w_vals)

                def _h(scores_iter):
                    numerator = sum(sc * ww for sc, ww in zip(scores_iter, w_vals))
                    return 1.0 - (numerator / w_sum)

                r_scores = [float(c.scores.reliability)    for c in all_comps]
                m_scores = [float(c.scores.maintainability) for c in all_comps]
                a_scores = [float(c.scores.availability)   for c in all_comps]
                s_scores = [float(c.scores.security)       for c in all_comps]
                q_scores = [float(c.scores.overall)        for c in all_comps]

                h_r = _h(r_scores)
                h_m = _h(m_scores)
                h_a = _h(a_scores)
                h_s = _h(s_scores)

                sri = (
                    dim_weights["r"] * (1 - h_r)
                    + dim_weights["m"] * (1 - h_m)
                    + dim_weights["a"] * (1 - h_a)
                    + dim_weights["s"] * (1 - h_s)
                )

                q_sorted = sorted(q_scores)
                gini_sum = sum(
                    (2 * (i + 1) - n_c - 1) * q_sorted[i]
                    for i in range(n_c)
                )
                rci = abs(gini_sum) / (n_c * sum(q_sorted)) if sum(q_sorted) > 0 else 0.0

                system_health = {
                    "H_R": round(h_r, 4),
                    "H_M": round(h_m, 4),
                    "H_A": round(h_a, 4),
                    "H_S": round(h_s, 4),
                    "SRI": round(sri, 4),
                    "RCI": round(rci, 4),
                }

                dimensional_validation["composite"] = {
                    "spearman_q_star_i_star": round(float(composite_spearman), 4),
                    "predictive_gain":        round(float(predictive_gain), 4),
                    "best_single_dim":       str(best_dim_label),
                    "best_single_dim_rho":   round(float(best_dim_val), 4),
                    "interdim_rhos":         interdim_rhos,
                    "interdim_max_correlation": max([float(v) for v in interdim_rhos.values()]) if interdim_rhos else 0.0,
                    "system_health":         system_health,
                    "n":                     len(common_comp) if 'common_comp' in locals() else 0,
                    "ground_truth":          "I*(v)",
                }

        except Exception as e:
            self.logger.warning("Composite validation failed: %s", e)

        # ── 8. Gates G1-G9 and Node-Type Stratified Reporting ──
        gates = dict(validation_res.overall.gates)
        
        # Tier 1: G1-G4 (already in validation_res.overall.gates mostly)
        # Tier 1: G1-G4
        gates["G1_spearman"] = float(validation_res.overall.correlation.spearman) >= float(self.targets.spearman)
        gates["G2_f1"] = float(validation_res.overall.classification.f1_score) >= float(self.targets.f1_score)
        gates["G3_precision"] = float(validation_res.overall.classification.precision) >= float(self.targets.precision)
        gates["G4_top5"] = float(validation_res.overall.ranking.top_5_overlap) >= float(self.targets.top_5_overlap)

        # Tier 2: G5-G7
        try:
            target_pg = float(self.targets.predictive_gain)
            gates["G5_predictive_gain"] = float(predictive_gain) > target_pg
        except:
            gates["G5_predictive_gain"] = False
        gates["G6_kappa_cta"] = float(dimensional_validation.get("maintainability", {}).get("weighted_kappa_cta", 0.0)) >= float(self.targets.weighted_kappa_cta)
        gates["G7_cdcc"] = float(dimensional_validation.get("security", {}).get("cdcc", 1.0)) < float(self.targets.cdcc_max)

        # Tier 3: G8-G9
        gates["G8_bottleneck_precision"] = dimensional_validation.get("maintainability", {}).get("bottleneck_precision", 0.0) >= self.targets.bottleneck_precision_target
        gates["G9_ftr"] = dimensional_validation.get("security", {}).get("ftr", 1.0) <= self.targets.ftr_max

        # Node-type Stratified Reporting (G1 per node-type)
        stratified = {}
        for node_type, group_res in validation_res.by_type.items():
            # Target ρ depends on node type as per doc
            target_rho = 0.70
            if node_type == "Application": target_rho = 0.75
            elif node_type == "Broker": target_rho = 0.70
            elif node_type == "Node": target_rho = 0.65
            elif node_type == "Library": target_rho = 0.60
            
            stratified[node_type] = {
                "n": group_res.sample_size,
                "spearman": round(group_res.correlation.spearman, 4),
                "target_rho": target_rho,
                "passed": group_res.correlation.spearman >= target_rho
            }

        # --- Frequency Decile Stratified Reporting ---
        frequency_decile_stratified = {}
        topic_data = []
        graph = getattr(analysis_result, "graph", None)
        if graph is not None:
            for cid, pred_val in pred_scores.items():
                if comp_types.get(cid) == "Topic" and cid in actual_scores:
                    node_attrs = graph.nodes.get(cid, {})
                    freq = float(node_attrs.get("frequency", node_attrs.get("topic_frequency", 0.0)) or 0.0)
                    topic_data.append({
                        "id": cid,
                        "pred": float(pred_val),
                        "act": float(actual_scores[cid]),
                        "freq": freq
                    })
        
        if topic_data:
            topic_data = sorted(topic_data, key=lambda x: x["freq"])
            n_topics = len(topic_data)
            
            import numpy as np
            from saag.validation.metric_calculator import spearman_correlation
            
            for i in range(1, 11):
                start_idx = int(round((i - 1) * n_topics / 10.0))
                end_idx = int(round(i * n_topics / 10.0))
                
                decile_items = topic_data[start_idx:end_idx]
                if not decile_items:
                    continue
                
                decile_name = f"Decile {i}"
                min_freq = min(x["freq"] for x in decile_items)
                max_freq = max(x["freq"] for x in decile_items)
                
                pred_vals = [x["pred"] for x in decile_items]
                act_vals = [x["act"] for x in decile_items]
                
                spearman_val = 0.0
                p_val = 1.0
                if len(decile_items) >= 3:
                    try:
                        spearman_val, p_val = spearman_correlation(pred_vals, act_vals)
                        if np.isnan(spearman_val):
                            spearman_val = 0.0
                    except Exception:
                        spearman_val = 0.0
                
                frequency_decile_stratified[decile_name] = {
                    "n": len(decile_items),
                    "frequency_range": (round(min_freq, 2), round(max_freq, 2)),
                    "spearman": round(float(spearman_val), 4),
                    "p_value": round(float(p_val), 6),
                }

        return LayerValidationResult(
            layer=sim_layer.value,
            layer_name=layer_def.name,
            predicted_components=validation_res.predicted_count,
            simulated_components=validation_res.actual_count,
            matched_components=validation_res.matched_count,
            validation_result=validation_res,
            node_type_stratified=stratified,
            frequency_decile_stratified=frequency_decile_stratified,

            spearman=validation_res.overall.correlation.spearman,
            f1_score=validation_res.overall.classification.f1_score,
            precision=validation_res.overall.classification.precision,
            recall=validation_res.overall.classification.recall,
            top_5_overlap=validation_res.overall.ranking.top_5_overlap,
            top_10_overlap=validation_res.overall.ranking.top_10_overlap,
            rmse=validation_res.overall.error.rmse,
            reliability_spearman=reliability_spearman,
            maintainability_spearman=maintainability_spearman,
            availability_spearman=availability_spearman,
            security_spearman=security_spearman,
            composite_spearman=composite_spearman,
            predictive_gain=predictive_gain,
            system_health=system_health,

            passed=all([gates.get(g, False) for g in ["G1_spearman", "G2_f1", "G3_precision", "G4_top5"]]),
            gates=gates,
            comparisons=validation_res.overall.components,
            warnings=validation_res.warnings,
            csc_names=comp_names,
            dimensional_validation=dimensional_validation,
            dimensional_scatter=dimensional_scatter,
            confidence_intervals=confidence_intervals,
            gnn_forecasting_metrics=gnn_forecasting_metrics,
            rule_based_baseline_metrics=rule_based_baseline_metrics,
        )

    def validate_from_data(self, predicted, actual) -> ValidationResult:
        """Quick validation helper."""
        return self.validator.validate(predicted, actual, context="Quick Validation")

