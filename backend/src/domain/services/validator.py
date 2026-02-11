"""
Validation Domain Service

Compares predicted quality scores Q(v) against simulated failure impact I(v)
using correlation, classification, ranking, and error metrics.

Pass/fail logic (primary + secondary gates):
    - Spearman ρ  ≥ target  (primary)
    - Spearman p  ≤ p_max   (primary — statistical significance)
    - F1-Score    ≥ target  (primary)
    - Top-5 Overlap ≥ target (primary)
    - RMSE        ≤ target  (secondary)
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime

from src.domain.models.validation.metrics import (
    ValidationTargets, CorrelationMetrics, ErrorMetrics,
    ClassificationMetrics, RankingMetrics,
)
from src.domain.models.validation.results import (
    ValidationResult, ValidationGroupResult, ComponentComparison
)
from src.domain.services.metric_calculator import (
    calculate_correlation, calculate_error, calculate_classification, calculate_ranking
)


class Validator:
    """
    Validates graph analysis predictions against simulation results.
    """

    def __init__(
        self,
        targets: Optional[ValidationTargets] = None,
        critical_percentile: float = 75.0,
    ):
        self.targets = targets or ValidationTargets()
        self.critical_percentile = critical_percentile
        self.logger = logging.getLogger(__name__)

    def validate(
        self,
        predicted_scores: Dict[str, float],
        actual_scores: Dict[str, float],
        component_types: Optional[Dict[str, str]] = None,
        layer: str = "system",
        context: str = "Validation",
    ) -> ValidationResult:
        """Main validation method.

        Steps:
            1. Align data by component ID (intersection).
            2. Compute overall validation group.
            3. Compute per-type breakdown (groups with n ≥ 3).
            4. Emit warnings for low power, missing IDs, etc.
        """
        timestamp = datetime.now().isoformat()
        warnings: List[str] = []

        # Data alignment
        pred_ids = set(predicted_scores.keys())
        actual_ids = set(actual_scores.keys())
        common_ids = pred_ids & actual_ids

        if len(common_ids) < len(pred_ids):
            warnings.append(
                f"{len(pred_ids) - len(common_ids)} predicted components not in actual scores"
            )
        if len(common_ids) < len(actual_ids):
            warnings.append(
                f"{len(actual_ids) - len(common_ids)} actual components not in predictions"
            )

        if len(common_ids) < 3:
            warnings.append("Insufficient data (n < 3)")
            return self._empty_result(
                timestamp, layer, context,
                len(pred_ids), len(actual_ids), len(common_ids), warnings,
            )

        # Low statistical power warning
        if len(common_ids) < 10:
            warnings.append(
                f"Low statistical power (n={len(common_ids)} < 10): "
                "results should be interpreted with caution; confidence intervals will be wide"
            )

        # Filter to common IDs
        pred_filtered = {k: predicted_scores[k] for k in common_ids}
        actual_filtered = {k: actual_scores[k] for k in common_ids}
        types_filtered = (
            {k: component_types.get(k, "Unknown") for k in common_ids}
            if component_types else {}
        )

        # 1. Overall validation
        overall = self._validate_group(
            "Overall", pred_filtered, actual_filtered, types_filtered
        )

        # 2. Per-type validation
        by_type: Dict[str, ValidationGroupResult] = {}
        if types_filtered:
            type_groups: Dict[str, List[str]] = {}
            for comp_id, comp_type in types_filtered.items():
                type_groups.setdefault(comp_type, []).append(comp_id)

            for comp_type, comp_ids in type_groups.items():
                if len(comp_ids) >= 3:
                    pred_type = {k: pred_filtered[k] for k in comp_ids}
                    actual_type = {k: actual_filtered[k] for k in comp_ids}
                    types_type = {k: types_filtered[k] for k in comp_ids}
                    by_type[comp_type] = self._validate_group(
                        comp_type, pred_type, actual_type, types_type
                    )

        return ValidationResult(
            timestamp=timestamp,
            layer=layer,
            context=context,
            targets=self.targets,
            overall=overall,
            by_type=by_type,
            predicted_count=len(pred_ids),
            actual_count=len(actual_ids),
            matched_count=len(common_ids),
            warnings=warnings,
        )

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    def _validate_group(
        self,
        group_name: str,
        predicted: Dict[str, float],
        actual: Dict[str, float],
        types: Dict[str, str],
    ) -> ValidationGroupResult:
        """Compute all metrics for a single validation group."""
        n = len(predicted)
        ids = list(predicted.keys())
        pred_vals = [predicted[k] for k in ids]
        actual_vals = [actual[k] for k in ids]

        # Correlation (includes bootstrap CI)
        correlation = calculate_correlation(pred_vals, actual_vals)

        # Error (includes NRMSE)
        error = calculate_error(pred_vals, actual_vals)

        # Classification
        pred_thresh = self._percentile(pred_vals, self.critical_percentile)
        actual_thresh = self._percentile(actual_vals, self.critical_percentile)
        pred_crit = [v >= pred_thresh for v in pred_vals]
        actual_crit = [v >= actual_thresh for v in actual_vals]
        classification = calculate_classification(pred_crit, actual_crit)

        # Ranking
        ranking = calculate_ranking(predicted, actual)

        # Component comparisons (sorted by error descending)
        components: List[ComponentComparison] = []
        for i, cid in enumerate(ids):
            pc, ac = pred_crit[i], actual_crit[i]
            cls = "TP" if pc and ac else "FP" if pc else "FN" if ac else "TN"
            components.append(ComponentComparison(
                id=cid,
                type=types.get(cid, "Unknown"),
                predicted=pred_vals[i],
                actual=actual_vals[i],
                error=abs(pred_vals[i] - actual_vals[i]),
                predicted_critical=pc,
                actual_critical=ac,
                classification=cls,
            ))
        components.sort(key=lambda x: x.error, reverse=True)

        # Pass/fail: primary + secondary gates
        passed = (
            correlation.spearman >= self.targets.spearman
            and correlation.spearman_p <= self.targets.spearman_p_max
            and classification.f1_score >= self.targets.f1_score
            and ranking.top_5_overlap >= self.targets.top_5_overlap
            and error.rmse <= self.targets.rmse_max
        )

        return ValidationGroupResult(
            group_name=group_name,
            sample_size=n,
            correlation=correlation,
            error=error,
            classification=classification,
            ranking=ranking,
            passed=passed,
            targets=self.targets,
            components=components,
        )

    def _percentile(self, values: List[float], p: float) -> float:
        """Compute the p-th percentile via linear interpolation.

        Uses a self-contained implementation to avoid external dependencies,
        consistent with the project's zero-dependency metric calculator design.
        """
        if not values:
            return 0.0
        s = sorted(values)
        k = (len(s) - 1) * p / 100.0
        f = int(k)
        c = min(f + 1, len(s) - 1)
        if f == c:
            return s[f]
        return s[f] * (c - k) + s[c] * (k - f)

    def _empty_result(
        self,
        timestamp: str,
        layer: str,
        context: str,
        pc: int,
        ac: int,
        mc: int,
        warnings: List[str],
    ) -> ValidationResult:
        """Construct an empty ValidationResult for degenerate inputs."""
        empty_group = ValidationGroupResult(
            group_name="Overall",
            sample_size=0,
            passed=False,
            targets=self.targets,
            correlation=CorrelationMetrics(),
            error=ErrorMetrics(),
            classification=ClassificationMetrics(),
            ranking=RankingMetrics(),
        )
        return ValidationResult(
            timestamp=timestamp,
            layer=layer,
            context=context,
            targets=self.targets,
            overall=empty_group,
            predicted_count=pc,
            actual_count=ac,
            matched_count=mc,
            warnings=warnings,
        )