"""
Validation Service

Application service implementing IValidationUseCase.
Orchestrates the validation pipeline by coordinating Analysis and Simulation results.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np

from .dimensions import (
    DIMENSION_SPECS, SUBCHARACTERISTIC_SPECS, GROUND_TRUTH_FIELDS,
    DimensionInputs, DimensionResult, DimensionSpec,
)
from .metric_calculator import calculate_correlation, spearman_correlation
from .models import ValidationTargets, ValidationResult, LayerValidationResult, PipelineResult
from .validator import Validator, robust_sigmoid_scale_dict
from saag.core.layers import AnalysisLayer, get_simulation_layer_definition

#: Predictor score attributes on `component.scores`, keyed by dimension. These
#: are the two RM composite dimensions — the scope of the composite gates
#: (I*, predictive_gain, orthogonality) and SRI.
_PREDICTOR_ATTRS = ("reliability", "maintainability")

#: Reliability's sub-characteristics — reported diagnostics, not composite
#: dimensions. Kept out of _PREDICTOR_ATTRS so SRI and the composite gates
#: don't double-count Reliability's signal via its own sub-scores.
_SUBCHARACTERISTIC_ATTRS = ("fault_tolerance", "availability")

#: Pairs checked for orthogonality — two dimensions that rank components
#: identically are not measuring distinct quality attributes.
_INTERDIM_PAIRS = (
    ("R*vsM*", "reliability", "maintainability"),
)

#: Reported diagnostics, excluded from the orthogonality gate: FT-vs-A is
#: expected to correlate somewhat (both feed Reliability), and its old
#: "high = bad" reading is inverted here — a *low* rho would mean the
#: r_alpha blend is redundant, i.e. one sub-characteristic could be dropped.
_DIAGNOSTIC_PAIRS = (
    ("FT*vsA*", "fault_tolerance", "availability"),
)

#: Minimum matched components for any correlation to be meaningful.
_MIN_SAMPLE = 3


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

    # ------------------------------------------------------------------
    # Pipeline entry points
    # ------------------------------------------------------------------

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

        # Analyse every layer up front: this derives DEPENDS_ON edges once for
        # the whole run rather than once per layer.
        analyses = self._analyze_layers([layer.value for layer in valid_layers])

        results = {}
        passed_count = 0
        total_components = 0

        for layer in valid_layers:
            self.logger.info(f"Validating layer: {layer.value}")
            try:
                result = self.validate_single_layer(
                    layer.value, analysis_result=analyses.get(layer.value)
                )
                results[layer.value] = result
                if result.passed:
                    passed_count += 1
                total_components += result.predicted_components
            except Exception as e:
                self.logger.error(f"Failed to validate layer {layer.value}: {e}")
                self.logger.exception("Validation exception details:")
                results[layer.value] = LayerValidationResult(
                    layer=layer.value,
                    layer_name=get_simulation_layer_definition(layer).name,
                    warnings=[str(e)],
                    errored=True,
                )

        return PipelineResult(
            timestamp=datetime.now().isoformat(),
            layers=results,
            total_components=total_components,
            layers_passed=passed_count,
            all_passed=(passed_count == len(valid_layers)),
            targets=self.targets
        )

    def _analyze_layers(self, layers: List[str]) -> Dict[str, Any]:
        """Batch-analyse layers so DEPENDS_ON is derived once for the whole run.

        Returns an empty mapping if the analysis service cannot batch, in which
        case `validate_single_layer` analyses each layer itself.
        """
        batch = getattr(self.analysis, "analyze_layers", None)
        if batch is None:
            return {}
        try:
            results = batch(layers)
        except Exception as e:
            self.logger.warning("Batch analysis failed, analysing per layer: %s", e)
            return {}
        # Anything other than a real mapping (e.g. a test double) is ignored so
        # the per-layer path stays in control.
        return results if isinstance(results, dict) else {}

    def validate_single_layer(
        self, layer: str, analysis_result: Any = None
    ) -> LayerValidationResult:
        """Validate a single layer, analysing and simulating it first.

        ``analysis_result`` may be supplied by a caller that already analysed
        the layer; otherwise this analyses it here.
        """
        try:
            sim_layer = AnalysisLayer.from_string(layer)
        except ValueError:
            raise ValueError(f"Unknown layer: {layer}")

        # 1. Analysis
        if analysis_result is None:
            self.logger.info("Running analysis...")
            analysis_result = self.analysis.analyze_layer(sim_layer.value)

        # 2. Prediction — enriches the analysis result with Q(v) scores
        self.logger.info("Running prediction...")
        analysis_result.quality = self.prediction.predict_quality(analysis_result.structural)

        # 3. Simulation — the independent oracle
        self.logger.info("Running simulation...")
        sim_results = self.simulation.run_failure_simulation_exhaustive(layer=sim_layer.value)

        return self.validate_single_layer_from_results(analysis_result, sim_results, layer)

    # ------------------------------------------------------------------
    # Core validation
    # ------------------------------------------------------------------

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
        sim_layer = AnalysisLayer.from_string(layer)
        layer_def = get_simulation_layer_definition(sim_layer)

        components = list(analysis_result.quality.components)
        comp_map = {c.id: c for c in components}
        comp_types = {c.id: c.type for c in components}
        comp_names = {c.id: c.structural.name for c in components}
        pred_scores = {c.id: c.scores.overall for c in components}
        # Spans both the composite dimensions and the reliability sub-characteristics
        # so SUBCHARACTERISTIC_SPECS validation and the FT-vs-A diagnostic pair can
        # read them, without those extra keys entering _PREDICTOR_ATTRS-scoped logic
        # (SRI, the composite gates) and double-counting Reliability's signal.
        predictions = {
            attr: {c.id: getattr(c.scores, attr) for c in components}
            for attr in _PREDICTOR_ATTRS + _SUBCHARACTERISTIC_ATTRS
        }
        ground_truths = self._extract_ground_truths(sim_results)

        # 1. Overall validation — Q(v) against composite impact.
        # Raw impact is passed through; Validator owns the label scaling.
        self.logger.info("Validating...")
        validation_res = self.validator.validate(
            predicted_scores=pred_scores,
            actual_scores={r.target_id: r.impact.composite_impact for r in sim_results},
            component_types=comp_types,
            layer=sim_layer.value,
            context=layer_def.name,
        )

        # 2. Predictor benchmarks
        rule_based_baseline_metrics = self._rule_based_baseline(validation_res)
        gnn_forecasting_metrics = self._gnn_forecasting_metrics(analysis_result, sim_results)

        # 3. Per-dimension validation — each predictor against its own ground truth
        dimensional_validation: Dict[str, Any] = {}
        dimensional_scatter: Dict[str, Any] = {}
        confidence_intervals: Dict[str, Tuple[float, float]] = {}
        dim_rhos: Dict[str, float] = {}

        for spec in DIMENSION_SPECS:
            self.logger.info(
                "Computing %s-specific validation (%s ground truth)...",
                spec.key, spec.ground_truth_label,
            )
            result = self._validate_dimension(
                spec, predictions, ground_truths, comp_map, sim_results, sim_layer.value
            )
            dim_rhos[spec.key] = result.spearman if result else 0.0
            if result:
                dimensional_validation[spec.key] = result.entry
                dimensional_scatter[spec.key] = result.scatter
                confidence_intervals[spec.key] = result.ci

        # 3b. Sub-characteristic diagnostics (currently: availability). Reported
        # the same way as a dimension, but its rho does NOT enter dim_rhos —
        # it is not a candidate for predictive_gain's "best single dimension",
        # since it is already inside reliability_impact's blend.
        subchar_rhos: Dict[str, float] = {}
        for spec in SUBCHARACTERISTIC_SPECS:
            self.logger.info(
                "Computing %s-specific validation (%s ground truth, sub-characteristic)...",
                spec.key, spec.ground_truth_label,
            )
            result = self._validate_dimension(
                spec, predictions, ground_truths, comp_map, sim_results, sim_layer.value
            )
            subchar_rhos[spec.key] = result.spearman if result else 0.0
            if result:
                dimensional_validation[spec.key] = result.entry
                dimensional_scatter[spec.key] = result.scatter
                confidence_intervals[spec.key] = result.ci

        # 4. Composite I*(v), predictive gain, orthogonality and system health
        self.logger.info("Computing composite I*(v) and system health metrics...")
        composite = self._validate_composite(
            pred_scores, ground_truths, comp_map, dim_rhos, sim_layer.value
        )
        if composite:
            dimensional_scatter["composite"] = composite["scatter"]
            confidence_intervals["composite"] = composite["ci"]

        interdim_rhos = self._interdimension_orthogonality(predictions)
        diagnostic_rhos = self._diagnostic_orthogonality(predictions)
        system_health = self._system_health(components)
        if system_health:
            dimensional_validation["composite"] = {
                "spearman_q_star_i_star": round(composite["spearman"] if composite else 0.0, 4),
                "predictive_gain": round(composite["predictive_gain"] if composite else 0.0, 4),
                "best_single_dim": composite["best_dim"] if composite else "None",
                "best_single_dim_rho": round(composite["best_dim_rho"] if composite else 0.0, 4),
                "interdim_rhos": interdim_rhos,
                "interdim_max_correlation": max(interdim_rhos.values()) if interdim_rhos else 0.0,
                "diagnostic_rhos": diagnostic_rhos,
                "system_health": system_health,
                "n": composite["n"] if composite else 0,
                "ground_truth": "I*(v)",
            }

        # 5. Gates and stratified reporting
        composite_spearman = composite["spearman"] if composite else 0.0
        predictive_gain = composite["predictive_gain"] if composite else 0.0
        gates = self._evaluate_gates(validation_res, dimensional_validation, predictive_gain)

        return LayerValidationResult(
            layer=sim_layer.value,
            layer_name=layer_def.name,
            predicted_components=validation_res.predicted_count,
            simulated_components=validation_res.actual_count,
            matched_components=validation_res.matched_count,
            validation_result=validation_res,
            node_type_stratified=self._stratify_by_node_type(validation_res),
            frequency_decile_stratified=self._stratify_by_frequency_decile(
                analysis_result, pred_scores, ground_truths["composite_impact"], comp_types
            ),

            spearman=validation_res.overall.correlation.spearman,
            f1_score=validation_res.overall.classification.f1_score,
            precision=validation_res.overall.classification.precision,
            recall=validation_res.overall.classification.recall,
            top_5_overlap=validation_res.overall.ranking.top_5_overlap,
            top_10_overlap=validation_res.overall.ranking.top_10_overlap,
            rmse=validation_res.overall.error.rmse,
            reliability_spearman=dim_rhos["reliability"],
            maintainability_spearman=dim_rhos["maintainability"],
            availability_spearman=subchar_rhos.get("availability", 0.0),
            composite_spearman=composite_spearman,
            predictive_gain=predictive_gain,
            system_health=system_health,

            passed=all(gates.get(g, False) for g in ("G1_spearman", "G2_f1", "G3_precision", "G4_top5")),
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

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _extract_ground_truths(self, sim_results: List[Any]) -> Dict[str, Dict[str, float]]:
        """Scale every simulation impact field once, keyed by field name.

        The per-dimension truths must share a scale because the composite I*(v)
        sums them.
        """
        return {
            field: robust_sigmoid_scale_dict(
                {r.target_id: getattr(r.impact, field) for r in sim_results}
            )
            for field in GROUND_TRUTH_FIELDS
        }

    @staticmethod
    def _scatter(
        ids: List[str],
        predicted: Dict[str, float],
        actual: Dict[str, float],
        comp_map: Dict[str, Any],
    ) -> List[Tuple[str, float, float, str]]:
        """Per-component (id, predicted, actual, criticality level) points."""
        points = []
        for cid in ids:
            comp = comp_map.get(cid)
            level = (
                getattr(comp.levels.overall, "name", str(comp.levels.overall))
                if comp else "MINIMAL"
            )
            points.append((cid, float(predicted[cid]), float(actual[cid]), level))
        return points

    # ------------------------------------------------------------------
    # Predictor benchmarks
    # ------------------------------------------------------------------

    def _acceptance(self, spearman: float, macro_f1: float, ndcg_10: float) -> Dict[str, Any]:
        """Shared pass/fail shape for the rule-based and GNN predictors."""
        t = self.targets
        return {
            "spearman": round(spearman, 4),
            "macro_f1": round(macro_f1, 4),
            "ndcg_10": round(ndcg_10, 4),
            "passed": (
                spearman >= t.baseline_spearman
                and macro_f1 >= t.baseline_macro_f1
                and ndcg_10 >= t.baseline_ndcg_10
            ),
            "targets": {
                "spearman": t.baseline_spearman,
                "macro_f1": t.baseline_macro_f1,
                "ndcg_10": t.baseline_ndcg_10,
            },
        }

    def _rule_based_baseline(self, validation_res: ValidationResult) -> Dict[str, Any]:
        """Acceptance check for the deterministic RM predictor."""
        return self._acceptance(
            float(validation_res.overall.correlation.spearman),
            float(validation_res.overall.classification.macro_f1),
            float(validation_res.overall.ranking.ndcg_10),
        )

    def _gnn_forecasting_metrics(
        self, analysis_result: Any, sim_results: List[Any]
    ) -> Optional[Dict[str, Any]]:
        """Acceptance check for the GNN predictor, when a checkpoint is loaded."""
        if not hasattr(self.prediction, "predict_quality_with_gnn"):
            return None
        if not getattr(self.prediction, "gnn_checkpoint_dir", None):
            return None

        graph = getattr(analysis_result, "graph", None)
        if graph is None:
            return None

        try:
            gnn_result = self.prediction.predict_quality_with_gnn(
                analysis_result.structural, graph, sim_results
            )
        except Exception as e:
            self.logger.warning(f"GNN prediction for validation failed: {e}")
            return None

        metrics = getattr(gnn_result, "gnn_metrics", None)
        if metrics is None:
            return None

        report = self._acceptance(
            float(metrics.spearman_rho), float(metrics.macro_f1), float(metrics.ndcg_10)
        )
        report["bce_loss"] = round(float(metrics.bce_loss), 4)
        report["regression_curve"] = {
            "slope": round(float(metrics.regression_slope), 4),
            "intercept": round(float(metrics.regression_intercept), 4),
            "r2": round(float(metrics.regression_r2), 4),
        }
        return report

    # ------------------------------------------------------------------
    # Dimensional validation
    # ------------------------------------------------------------------

    def _validate_dimension(
        self,
        spec: DimensionSpec,
        predictions: Dict[str, Dict[str, float]],
        ground_truths: Dict[str, Dict[str, float]],
        comp_map: Dict[str, Any],
        sim_results: List[Any],
        layer: str,
    ) -> Optional[DimensionResult]:
        """Correlate one dimension's predictor against its ground truth.

        Returns None when fewer than three components are scored by both sides,
        which is the only case where the dimension is legitimately skipped.
        """
        predicted = predictions[spec.score_attr]
        actual = ground_truths[spec.impact_attr]
        ids = sorted(set(predicted) & set(actual))
        if len(ids) < _MIN_SAMPLE:
            self.logger.debug(
                "%s dimension skipped: only %d matched components", spec.key, len(ids)
            )
            return None

        corr = calculate_correlation(
            [float(predicted[k]) for k in ids], [float(actual[k]) for k in ids]
        )
        inputs = DimensionInputs(
            ids=ids,
            predicted={k: float(predicted[k]) for k in ids},
            actual={k: float(actual[k]) for k in ids},
            components=comp_map,
            predictions=predictions,
            ground_truths=ground_truths,
            sim_results=sim_results,
            targets=self.targets,
        )
        try:
            specialists = spec.specialists(inputs)
        except Exception as e:
            # The correlation above is still valid, so report it rather than
            # dropping the dimension — but make the missing metrics loud. Every
            # gate that reads a specialist key via `.get(key, default)`
            # (_evaluate_gates, G6-G9) already fails closed when the key is
            # absent, so a crash here cannot manufacture a passing gate — but
            # the traceback matters for telling "genuinely near-zero" apart
            # from "the specialist computation raised".
            self.logger.exception(
                "%s specialist metrics unavailable (%s); reporting ρ only", spec.key, e
            )
            specialists = {}

        self.logger.info(
            "%s dim [%s]: ρ=%.3f (n=%d), %s",
            spec.key.capitalize(), layer, corr.spearman, len(ids),
            ", ".join(f"{k}={v:.3f}" if v is not None else f"{k}=n/a" for k, v in specialists.items()),
        )

        return DimensionResult(
            spearman=corr.spearman,
            entry={
                "spearman": round(corr.spearman, 4),
                "spearman_p": round(corr.spearman_p, 6),
                **{k: (round(float(v), 4) if v is not None else None) for k, v in specialists.items()},
                "n": len(ids),
                "ground_truth": spec.ground_truth_label,
            },
            scatter=self._scatter(ids, inputs.predicted, inputs.actual, comp_map),
            ci=(corr.spearman_ci_lower, corr.spearman_ci_upper),
        )

    # ------------------------------------------------------------------
    # Composite, orthogonality and system health
    # ------------------------------------------------------------------

    def _validate_composite(
        self,
        pred_scores: Dict[str, float],
        ground_truths: Dict[str, Dict[str, float]],
        comp_map: Dict[str, Any],
        dim_rhos: Dict[str, float],
        layer: str,
    ) -> Optional[Dict[str, Any]]:
        """Validate Q*(v) against the weighted composite ground truth I*(v).

        Predictive Gain is ρ(Q*, I*) minus the best single-dimension ρ: it is the
        evidence that combining dimensions beats any one of them alone.

        I*(v) = 0.5*IR + 0.5*IM, over reliability and maintainability only.
        availability_impact is not a separate term here — it already feeds
        IR(v) (reliability_impact) via the r_alpha blend — including it again
        would double-count Reliability's ground truth.
        """
        weights = self.targets.dimension_weights
        dims = {
            "r": ground_truths["reliability_impact"],
            "m": ground_truths["maintainability_impact"],
        }
        shared = set.intersection(*(set(d) for d in dims.values())) if dims else set()
        composite_i_star = {
            cid: sum(weights[k] * dims[k][cid] for k in dims)
            for cid in shared
        }

        ids = sorted(set(pred_scores) & set(composite_i_star))
        if len(ids) < _MIN_SAMPLE:
            return None

        corr = calculate_correlation(
            [float(pred_scores[k]) for k in ids], [float(composite_i_star[k]) for k in ids]
        )
        best_dim = max(dim_rhos, key=lambda k: dim_rhos[k])
        best_dim_rho = float(dim_rhos[best_dim])
        predictive_gain = float(corr.spearman) - best_dim_rho

        self.logger.info(
            "Composite [%s]: ρ(Q*,I*)=%.3f, PG=%.3f (vs %s), n=%d",
            layer, corr.spearman, predictive_gain, best_dim, len(ids),
        )

        return {
            "spearman": corr.spearman,
            "predictive_gain": predictive_gain,
            "best_dim": best_dim.capitalize(),
            "best_dim_rho": best_dim_rho,
            "n": len(ids),
            "scatter": self._scatter(ids, pred_scores, composite_i_star, comp_map),
            "ci": (corr.spearman_ci_lower, corr.spearman_ci_upper),
        }

    def _interdimension_orthogonality(
        self, predictions: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """ρ between each pair of predictors; high values mean redundant dimensions."""
        rhos: Dict[str, float] = {}
        for label, dim_a, dim_b in _INTERDIM_PAIRS:
            d1, d2 = predictions[dim_a], predictions[dim_b]
            ids = sorted(set(d1) & set(d2))
            if len(ids) < _MIN_SAMPLE:
                continue
            rho, _ = spearman_correlation(
                [float(d1[k]) for k in ids], [float(d2[k]) for k in ids]
            )
            rhos[label] = round(rho, 4)
            if abs(rho) > float(self.targets.max_interdim_correlation):
                self.logger.warning(
                    "Orthogonality violation! %s ρ=%.3f > %.2f threshold",
                    label, rho, self.targets.max_interdim_correlation,
                )
        return rhos

    def _diagnostic_orthogonality(
        self, predictions: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """ρ between Reliability's sub-characteristics (currently FT vs A).

        Reported only — excluded from the max_interdim_correlation gate. Its
        old "high = bad" reading is inverted here: since R = r_alpha*FT +
        (1-r_alpha)*A, a *low* rho is the interesting case — it means FT and A
        genuinely disagree, so blending them (rather than using either alone)
        is doing real work. A high rho means the blend is close to redundant.
        """
        rhos: Dict[str, float] = {}
        for label, dim_a, dim_b in _DIAGNOSTIC_PAIRS:
            d1, d2 = predictions[dim_a], predictions[dim_b]
            ids = sorted(set(d1) & set(d2))
            if len(ids) < _MIN_SAMPLE:
                continue
            rho, _ = spearman_correlation(
                [float(d1[k]) for k in ids], [float(d2[k]) for k in ids]
            )
            rhos[label] = round(rho, 4)
        return rhos

    def _system_health(self, components: List[Any]) -> Dict[str, float]:
        """Weighted dimension health H_d, the system risk index SRI, and Gini RCI.

        SRI sums only H_R and H_M — the two composite dimensions. H_FT and H_A
        (Reliability's sub-characteristics) are reported alongside for
        diagnostic visibility but excluded from the sum, for the same reason
        they are excluded from _validate_composite: they already feed H_R via
        the r_alpha blend, and summing them too would double-count it.
        """
        if not components:
            return {}

        weights = [max(c.structural.weight, 1e-9) for c in components]
        weight_sum = sum(weights)

        def health(attr: str) -> float:
            scored = sum(
                float(getattr(c.scores, attr)) * w for c, w in zip(components, weights)
            )
            return 1.0 - (scored / weight_sum)

        h = {attr: health(attr) for attr in _PREDICTOR_ATTRS + _SUBCHARACTERISTIC_ATTRS}
        dim_weights = self.targets.dimension_weights
        sri = (
            dim_weights["r"] * (1 - h["reliability"])
            + dim_weights["m"] * (1 - h["maintainability"])
        )

        # Risk Concentration Index: Gini coefficient of the composite scores.
        q_sorted = sorted(float(c.scores.overall) for c in components)
        n = len(q_sorted)
        q_total = sum(q_sorted)
        gini_sum = sum((2 * (i + 1) - n - 1) * q for i, q in enumerate(q_sorted))
        rci = abs(gini_sum) / (n * q_total) if q_total > 0 else 0.0

        return {
            "H_R": round(h["reliability"], 4),
            "H_M": round(h["maintainability"], 4),
            "H_FT": round(h["fault_tolerance"], 4),
            "H_A": round(h["availability"], 4),
            "SRI": round(sri, 4),
            "RCI": round(rci, 4),
        }

    # ------------------------------------------------------------------
    # Gates and stratified reporting
    # ------------------------------------------------------------------

    def _evaluate_gates(
        self,
        validation_res: ValidationResult,
        dimensional_validation: Dict[str, Any],
        predictive_gain: float,
    ) -> Dict[str, bool]:
        """The unified gate checklist. G1-G4 decide `passed`.

        G7 (CDCC) and G9 (FTR) were retired with the Vulnerability/Security
        dimension — both specialist metrics were security-only. The gap in
        the numbering is intentional; do not renumber or reuse it.
        """
        t = self.targets
        overall = validation_res.overall
        maintainability = dimensional_validation.get("maintainability", {})

        gates = dict(overall.gates)
        gates.update({
            # Tier 1 — primary
            "G1_spearman": float(overall.correlation.spearman) >= float(t.spearman),
            "G2_f1": float(overall.classification.f1_score) >= float(t.f1_score),
            "G3_precision": float(overall.classification.precision) >= float(t.precision),
            "G4_top5": float(overall.ranking.top_5_overlap) >= float(t.top_5_overlap),
            # Tier 2 — secondary
            "G5_predictive_gain": float(predictive_gain) > float(t.predictive_gain),
            "G6_kappa_cta": float(maintainability.get("weighted_kappa_cta", 0.0)) >= float(t.weighted_kappa_cta),
            # Tier 3 — dimension specialists
            "G8_bottleneck_precision": float(maintainability.get("bottleneck_precision", 0.0)) >= float(t.bottleneck_precision_target),
        })
        return gates

    def _stratify_by_node_type(self, validation_res: ValidationResult) -> Dict[str, Dict[str, Any]]:
        """G1 re-evaluated per node type, against that type's own ρ target."""
        stratified = {}
        for node_type, group in validation_res.by_type.items():
            target_rho = self.targets.node_type_rho.get(
                node_type, self.targets.node_type_rho_default
            )
            stratified[node_type] = {
                "n": group.sample_size,
                "spearman": round(group.correlation.spearman, 4),
                "target_rho": target_rho,
                "passed": group.correlation.spearman >= target_rho,
            }
        return stratified

    def _stratify_by_frequency_decile(
        self,
        analysis_result: Any,
        pred_scores: Dict[str, float],
        actual_scores: Dict[str, float],
        comp_types: Dict[str, str],
    ) -> Dict[str, Dict[str, Any]]:
        """ρ within each Topic message-frequency decile.

        Pooled ρ can hide a predictor that only works on busy topics, so the
        Topic population is split into ten equal-count frequency bands.
        """
        graph = getattr(analysis_result, "graph", None)
        if graph is None:
            return {}

        topics = []
        for cid, pred_val in pred_scores.items():
            if comp_types.get(cid) != "Topic" or cid not in actual_scores:
                continue
            attrs = graph.nodes.get(cid, {})
            frequency = float(attrs.get("frequency", attrs.get("topic_frequency", 0.0)) or 0.0)
            topics.append((frequency, float(pred_val), float(actual_scores[cid])))

        if not topics:
            return {}

        topics.sort(key=lambda x: x[0])
        n = len(topics)
        deciles: Dict[str, Dict[str, Any]] = {}

        for i in range(1, 11):
            start = int(round((i - 1) * n / 10.0))
            end = int(round(i * n / 10.0))
            band = topics[start:end]
            if not band:
                continue

            rho, p_value = 0.0, 1.0
            if len(band) >= _MIN_SAMPLE:
                rho, p_value = spearman_correlation(
                    [x[1] for x in band], [x[2] for x in band]
                )
                if np.isnan(rho):
                    rho = 0.0

            deciles[f"Decile {i}"] = {
                "n": len(band),
                "frequency_range": (round(band[0][0], 2), round(band[-1][0], 2)),
                "spearman": round(float(rho), 4),
                "p_value": round(float(p_value), 6),
            }
        return deciles
