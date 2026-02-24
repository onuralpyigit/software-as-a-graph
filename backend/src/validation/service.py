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
            calculate_correlation, calculate_ccr_at_k, calculate_cme,
            calculate_cocr_at_k, calculate_weighted_kappa_cta, calculate_bottleneck_precision,
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

        # 5. Maintainability-specific validation — ρ(M(v), IM(v))
        self.logger.info("Computing maintainability-specific validation (IM(v) ground truth)...")
        maintainability_spearman = 0.0

        pred_maintainability = {c.id: c.scores.maintainability for c in analysis_result.quality.components}
        actual_maintainability_impact = {
            r.target_id: r.impact.maintainability_impact for r in sim_results
        }

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

        # 6. Vulnerability-specific validation — ρ(V(v), IV(v))
        self.logger.info("Computing vulnerability-specific validation (IV(v) ground truth)...")
        vulnerability_spearman = 0.0

        from src.validation.metric_calculator import (
            calculate_ahcr_at_k, calculate_ftr, calculate_apar
        )

        actual_vulnerability_impact = {
            r.target_id: r.impact.vulnerability_impact for r in sim_results
        }
        actual_attack_reach = {
            r.target_id: r.impact.attack_reach for r in sim_results
        }

        try:
            common_v = sorted(set(pred_vulnerability) & set(actual_vulnerability_impact))
            if len(common_v) >= 3:
                p_v_vals  = [float(pred_vulnerability[k]) for k in common_v]
                a_iv_vals = [float(actual_vulnerability_impact[k]) for k in common_v]
                v_corr = calculate_correlation(p_v_vals, a_iv_vals)
                vulnerability_spearman = v_corr.spearman

                pred_v_dict = {k: pred_vulnerability[k] for k in common_v}
                act_iv_dict = {k: actual_vulnerability_impact[k] for k in common_v}
                
                # AHCR@5
                ahcr_5 = calculate_ahcr_at_k(pred_v_dict, act_iv_dict, k=5)
                
                # FTR
                act_reach_dict = {k: actual_attack_reach[k] for k in common_v}
                ftr = calculate_ftr(pred_v_dict, act_reach_dict, v_threshold=0.60, reach_threshold=0.10)
                
                # APAR
                paths_all = []
                for r in sim_results:
                    if r.impact.critical_paths:
                        paths_all.extend(r.impact.critical_paths)
                apar = calculate_apar(pred_v_dict, paths_all, v_threshold=0.60)
                
                # CDCC: Cross-Dimensional Contamination Check (Rank divergence V vs A)
                p_a_vals_for_cdcc = [float(pred_availability[k]) for k in common_v if k in pred_availability]
                p_v_vals_for_cdcc = [float(pred_vulnerability[k]) for k in common_v if k in pred_availability]
                
                cdcc = 0.0
                if len(p_a_vals_for_cdcc) >= 3:
                    from src.validation.metric_calculator import spearman_correlation
                    cdcc_res, _ = spearman_correlation(p_v_vals_for_cdcc, p_a_vals_for_cdcc)
                    cdcc = cdcc_res

                dimensional_validation["vulnerability"] = {
                    "spearman": round(v_corr.spearman, 4),
                    "spearman_p": round(v_corr.spearman_p, 6),
                    "ahcr_5": round(ahcr_5, 4),
                    "ftr": round(ftr, 4),
                    "apar": round(apar, 4),
                    "cdcc": round(cdcc, 4),
                    "n": len(common_v),
                    "ground_truth": "IV(v)",
                }
                
                self.logger.info(
                    "Vulnerability dim [%s]: ρ(V,IV)=%.3f (n=%d), "
                    "AHCR@5=%.3f, FTR=%.3f, APAR=%.3f, CDCC=%.3f",
                    sim_layer.value, v_corr.spearman, len(common_v),
                    ahcr_5, ftr, apar, cdcc
                )
            else:
                vulnerability_spearman = 0.0
        except Exception as e:
            self.logger.debug("Vulnerability-specific validation skipped: %s", e)
            vulnerability_spearman = 0.0

        # 6. Availability-specific validation — ρ(A(v), IA(v)) + SPOF_F1 + RRI
        self.logger.info("Computing availability-specific validation (IA(v) ground truth)...")
        availability_spearman = 0.0

        from src.validation.metric_calculator import (
            calculate_spof_f1, calculate_rri,
        )

        actual_availability_impact = {
            r.target_id: r.impact.availability_impact for r in sim_results
        }

        try:
            common_a = sorted(set(pred_availability) & set(actual_availability_impact))
            if len(common_a) >= 3:
                p_a_vals  = [float(pred_availability[k])          for k in common_a]
                a_ia_vals = [float(actual_availability_impact[k]) for k in common_a]
                a_corr = calculate_correlation(p_a_vals, a_ia_vals)
                availability_spearman = a_corr.spearman

                # AP_c_directed scores: use is_articulation_point flag as structural proxy
                pred_ap_c_dir = {
                    c.id: (1.0 if c.structural.is_articulation_point else 0.0)
                    for c in analysis_result.quality.components
                    if c.id in common_a
                }

                pred_a_dict = {k: pred_availability[k]          for k in common_a}
                act_ia_dict = {k: actual_availability_impact[k] for k in common_a}

                spof_f1_res = calculate_spof_f1(pred_ap_c_dir, act_ia_dict)
                rri_val     = calculate_rri(pred_a_dict, act_ia_dict, pred_ap_c_dir)

                dimensional_validation["availability"] = {
                    "spearman":      round(a_corr.spearman, 4),
                    "spearman_p":    round(a_corr.spearman_p, 6),
                    "spof_f1":       round(spof_f1_res["f1"], 4),
                    "spof_precision": round(spof_f1_res["precision"], 4),
                    "spof_recall":   round(spof_f1_res["recall"], 4),
                    "rri":           round(rri_val, 4),
                    "n":             len(common_a),
                    "ground_truth":  "IA(v)",
                }
                self.logger.info(
                    "Availability dim [%s]: \u03c1(A,IA)=%.3f (n=%d), SPOF_F1=%.3f, RRI=%.3f",
                    sim_layer.value, a_corr.spearman, len(common_a),
                    spof_f1_res["f1"], rri_val,
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
            # Default: equal weights (0.25 each) as empirically motivated baseline.
            # Once dimension-specific ρ values are known, weights can be updated via
            # w_d = ρ_actual(d) / Σ ρ_actual(d'). For now we use equal weights.
            dim_weights = dict(r=0.25, m=0.25, a=0.25, v=0.25)
            composite_i_star: dict = {}
            for cid in (
                set(actual_reliability_impact)
                & set(actual_maintainability_impact)
                & set(actual_availability_impact)
                & set(actual_vulnerability_impact)
            ):
                composite_i_star[cid] = (
                    dim_weights["r"] * actual_reliability_impact.get(cid, 0.0)
                    + dim_weights["m"] * actual_maintainability_impact.get(cid, 0.0)
                    + dim_weights["a"] * actual_availability_impact.get(cid, 0.0)
                    + dim_weights["v"] * actual_vulnerability_impact.get(cid, 0.0)
                )

            # ρ(Q*(v), I*(v))
            common_comp = sorted(set(pred_scores) & set(composite_i_star))
            if len(common_comp) >= 3:
                comp_corr = calculate_correlation(
                    [float(pred_scores[k])      for k in common_comp],
                    [float(composite_i_star[k]) for k in common_comp],
                )
                composite_spearman = comp_corr.spearman

                # Predictive gain: composite vs best single-dimension predictor
                best_dim = max(
                    reliability_spearman, maintainability_spearman,
                    availability_spearman, vulnerability_spearman,
                )
                predictive_gain = composite_spearman - best_dim

                self.logger.info(
                    "Composite [%s]: ρ(Q*,I*)=%.3f, PG=%.3f (n=%d)",
                    sim_layer.value, composite_spearman, predictive_gain, len(common_comp),
                )

            # ── Pairwise inter-dimension orthogonality check ──
            pred_r_all = {c.id: c.scores.reliability    for c in analysis_result.quality.components}
            pred_m_all = {c.id: c.scores.maintainability for c in analysis_result.quality.components}
            pred_a_all = {c.id: c.scores.availability   for c in analysis_result.quality.components}
            pred_v_all = {c.id: c.scores.vulnerability  for c in analysis_result.quality.components}

            from src.validation.metric_calculator import spearman_correlation
            pairs = [
                ("R*vsM*", pred_r_all, pred_m_all),
                ("R*vsA*", pred_r_all, pred_a_all),
                ("R*vsV*", pred_r_all, pred_v_all),
                ("M*vsA*", pred_m_all, pred_a_all),
                ("M*vsV*", pred_m_all, pred_v_all),
                ("A*vsV*", pred_a_all, pred_v_all),
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
                    if abs(rho) > self.targets.max_interdim_correlation:
                        self.logger.warning(
                            "Orthogonality violation! %s ρ=%.3f > %.2f threshold",
                            label, rho, self.targets.max_interdim_correlation,
                        )

            # ── System Health: H_R/M/A/V, SRI, RCI (Gini) ──
            all_comps = analysis_result.quality.components
            if all_comps:
                n_c = len(all_comps)
                # Importance weights w(v): use QoS weight from structural metrics
                w_vals = [max(c.structural.weight, 1e-9) for c in all_comps]
                w_sum = sum(w_vals)

                def _h(scores_iter):
                    """Importance-weighted complement of mean (system health in dim)."""
                    numerator = sum(sc * ww for sc, ww in zip(scores_iter, w_vals))
                    return 1.0 - (numerator / w_sum)

                r_scores = [c.scores.reliability    for c in all_comps]
                m_scores = [c.scores.maintainability for c in all_comps]
                a_scores = [c.scores.availability   for c in all_comps]
                v_scores = [c.scores.vulnerability  for c in all_comps]
                q_scores = [c.scores.overall        for c in all_comps]

                h_r = _h(r_scores)
                h_m = _h(m_scores)
                h_a = _h(a_scores)
                h_v = _h(v_scores)

                # SRI = Σ w_d × (1 − H_d) — weighted systemic risk index
                sri = (
                    dim_weights["r"] * (1 - h_r)
                    + dim_weights["m"] * (1 - h_m)
                    + dim_weights["a"] * (1 - h_a)
                    + dim_weights["v"] * (1 - h_v)
                )

                # RCI = Gini coefficient of Q*(v) distribution
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
                    "H_V": round(h_v, 4),
                    "SRI": round(sri, 4),
                    "RCI": round(rci, 4),
                }
                self.logger.info(
                    "System health [%s]: SRI=%.3f RCI=%.3f  H(R/M/A/V)=%.3f/%.3f/%.3f/%.3f",
                    sim_layer.value, sri, rci, h_r, h_m, h_a, h_v,
                )

            dimensional_validation["composite"] = {
                "spearman":         round(composite_spearman, 4),
                "predictive_gain":  round(predictive_gain, 4),
                "interdim_rhos":    interdim_rhos,
                "system_health":    system_health,
                "n":                len(common_comp) if 'common_comp' in dir() else 0,
                "ground_truth":     "I*(v) = 0.25×IR + 0.25×IM + 0.25×IA + 0.25×IV",
            }

        except Exception as e:
            self.logger.warning("Composite validation failed: %s", e)

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
            maintainability_spearman=maintainability_spearman,
            availability_spearman=availability_spearman,
            vulnerability_spearman=vulnerability_spearman,
            composite_spearman=composite_spearman,
            predictive_gain=predictive_gain,
            system_health=system_health,

            passed=validation_res.passed,
            comparisons=validation_res.overall.components,
            warnings=validation_res.warnings,
            component_names=comp_names,
            dimensional_validation=dimensional_validation,
        )

    def validate_from_data(self, predicted, actual) -> ValidationResult:
        """Quick validation helper."""
        return self.validator.validate(predicted, actual, context="Quick Validation")

