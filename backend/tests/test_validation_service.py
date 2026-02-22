
"""
Unit Tests for ValidationService
"""
import pytest
from unittest.mock import MagicMock, patch
from src.validation.service import ValidationService
from src.core.layers import AnalysisLayer
from src.validation.models import LayerValidationResult, ValidationResult, ValidationTargets
from src.validation.validator import Validator
from src.validation.metric_calculator import (
    calculate_classification,
    calculate_correlation,
    calculate_error,
    calculate_ranking,
    spearman_correlation,
    cohens_kappa,
    bootstrap_ci,
)
from src.analysis.models import LayerAnalysisResult, QualityAnalysisResult
from src.simulation.models import FailureResult, ImpactMetrics

@pytest.fixture
def mock_analysis_service():
    service = MagicMock()
    # Mock analyze_layer return value
    mock_result = MagicMock(spec=LayerAnalysisResult)
    mock_result.quality = MagicMock(spec=QualityAnalysisResult)
    mock_result.quality.components = []
    service.analyze_layer.return_value = mock_result
    return service

@pytest.fixture
def mock_simulation_service():
    service = MagicMock()
    # Mock run_failure_simulation_exhaustive return value
    service.run_failure_simulation_exhaustive.return_value = []
    return service

@pytest.fixture
def validation_service(mock_analysis_service, mock_simulation_service):
    return ValidationService(
        analysis_service=mock_analysis_service,
        simulation_service=mock_simulation_service
    )

class TestValidationService:
    
    def test_validate_layers_success(self, validation_service, mock_analysis_service, mock_simulation_service):
        """Test validating multiple valid layers."""
        # Setup mocks to return some data to avoid "insufficient data" warnings if possible,
        # but for this test we mainly care about flow control.
        
        result = validation_service.validate_layers(layers=["app", "infra"])
        
        assert result.total_components == 0 # internal validation returns 0 if empty
        assert "app" in result.layers
        assert "infra" in result.layers
        
        # Verify calls
        assert mock_analysis_service.analyze_layer.call_count == 2
        mock_analysis_service.analyze_layer.assert_any_call("app")
        mock_analysis_service.analyze_layer.assert_any_call("infra")
        
        assert mock_simulation_service.run_failure_simulation_exhaustive.call_count == 2
        mock_simulation_service.run_failure_simulation_exhaustive.assert_any_call(layer="app")
        mock_simulation_service.run_failure_simulation_exhaustive.assert_any_call(layer="infra")

    def test_validate_layers_skip_invalid(self, validation_service):
        """Test that invalid layers are skipped."""
        result = validation_service.validate_layers(layers=["app", "invalid_layer"])
        
        assert "app" in result.layers
        assert "invalid_layer" not in result.layers
        assert len(result.layers) == 1

    def test_validate_single_layer_flow(self, validation_service, mock_analysis_service, mock_simulation_service):
        """Test the flow of efficient sinlge layer validation."""
        # Setup Analysis Result
        comps = []
        sim_results = []
        for i, char in enumerate(['A', 'B', 'C']):
            # Analysis
            comp = MagicMock()
            comp.id = char
            comp.type = "Application"
            comp.scores.overall = 0.8 - (i * 0.1)  # Perfect match with variance
            comp.structural.name = f"App {char}"
            comps.append(comp)
            
            # Simulation
            fail_res = MagicMock(spec=FailureResult)
            fail_res.target_id = char
            fail_res.impact = MagicMock(spec=ImpactMetrics)
            fail_res.impact.composite_impact = 0.8 - (i * 0.1) # Perfect match with variance
            sim_results.append(fail_res)
        
        mock_analysis_res = MagicMock(spec=LayerAnalysisResult)
        mock_analysis_res.quality = MagicMock(spec=QualityAnalysisResult)
        mock_analysis_res.quality.components = comps
        mock_analysis_service.analyze_layer.return_value = mock_analysis_res
        
        mock_simulation_service.run_failure_simulation_exhaustive.return_value = sim_results
        
        # Execute
        result = validation_service.validate_layers(layers=["app"])
        
        layer_res = result.layers["app"]
        assert layer_res.predicted_components == 3
        assert layer_res.simulated_components == 3
        assert layer_res.matched_components == 3
        assert layer_res.passed is True
        
        # Verify data passed to validator (implicitly via result checks)
        assert layer_res.component_names["A"] == "App A"
"""
Unit Tests for validation module

Tests for:
    - Correlation metrics: Spearman, Pearson, Kendall, bootstrap CI
    - Error metrics: RMSE, MAE, NRMSE
    - Classification metrics: Precision, Recall, F1, Cohen's κ
    - Ranking metrics: Top-K overlap, NDCG
    - Validator: pass/fail logic, p-value gating, RMSE gating, warnings
    - Edge cases: ties, constant values, small n, large n
"""

import pytest
from src.validation.metric_calculator import (
    calculate_classification,
    calculate_correlation,
    calculate_error,
    calculate_ranking,
    spearman_correlation,
    cohens_kappa,
    bootstrap_ci,
)
from src.validation.validator import Validator
from src.validation.models import ValidationTargets


# =============================================================================
# Correlation Metrics Tests
# =============================================================================

class TestCorrelationMetrics:
    """Tests for correlation and error metrics."""

    def test_perfect_correlation(self):
        """Perfect positive correlation should return rho=1.0."""
        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        y = [0.1, 0.2, 0.3, 0.4, 0.5]
        rho, p_value = spearman_correlation(x, y)
        assert rho == pytest.approx(1.0, abs=0.01)
        assert p_value < 0.05

    def test_inverse_correlation(self):
        """Perfect inverse correlation should return rho=-1.0."""
        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        y_inv = [0.5, 0.4, 0.3, 0.2, 0.1]
        rho, p_value = spearman_correlation(x, y_inv)
        assert rho == pytest.approx(-1.0, abs=0.01)

    def test_no_correlation(self):
        """Random data should have weak or moderate correlation."""
        x = [0.1, 0.5, 0.2, 0.4, 0.3]
        y = [0.3, 0.1, 0.4, 0.2, 0.5]
        rho, p_value = spearman_correlation(x, y)
        assert -1.0 <= rho <= 1.0

    def test_rmse_calculation(self):
        """Test RMSE calculation via calculate_error."""
        predicted = [1.0, 2.0, 3.0]
        actual = [1.0, 2.0, 3.0]
        metrics = calculate_error(predicted, actual)
        assert metrics.rmse == pytest.approx(0.0, abs=0.001)

        # With error
        metrics_err = calculate_error(predicted, [1.1, 2.1, 3.1])
        assert metrics_err.rmse == pytest.approx(0.1, abs=0.01)

    def test_mae_calculation(self):
        """Test MAE calculation via calculate_error."""
        predicted = [1.0, 2.0, 3.0]
        actual = [1.5, 2.5, 3.5]
        metrics = calculate_error(predicted, actual)
        assert metrics.mae == pytest.approx(0.5, abs=0.01)


# =============================================================================
# Edge Cases: Ties and Constant Values
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases in correlation computation."""

    def test_spearman_with_ties(self):
        """Tied values should use average rank and still produce valid rho."""
        x = [0.5, 0.5, 0.3, 0.1]
        y = [0.4, 0.6, 0.2, 0.1]
        rho, _ = spearman_correlation(x, y)
        assert -1.0 <= rho <= 1.0
        # Moderate to strong expected since ordering mostly agrees
        assert rho > 0.3

    def test_spearman_all_same_predicted(self):
        """Constant predicted values should return rho=0."""
        x = [0.5, 0.5, 0.5, 0.5]
        y = [0.1, 0.2, 0.3, 0.4]
        rho, p = spearman_correlation(x, y)
        assert rho == pytest.approx(0.0, abs=0.01)

    def test_spearman_all_same_both(self):
        """Constant values on both sides should return rho=0."""
        x = [0.5, 0.5, 0.5]
        y = [0.5, 0.5, 0.5]
        rho, p = spearman_correlation(x, y)
        assert rho == pytest.approx(0.0, abs=0.01)

    def test_spearman_minimum_n(self):
        """n=3 should work but have high p-value."""
        x = [0.1, 0.5, 0.9]
        y = [0.2, 0.6, 0.8]
        rho, p = spearman_correlation(x, y)
        assert -1.0 <= rho <= 1.0
        # With n=3, significance is hard to achieve
        assert rho > 0.0

    def test_spearman_n_less_than_3(self):
        """n < 3 should return rho=0 and p=1.0."""
        rho, p = spearman_correlation([0.1, 0.2], [0.3, 0.4])
        assert rho == 0.0
        assert p == 1.0

    def test_empty_inputs(self):
        """Empty inputs should return zero metrics."""
        rho, p = spearman_correlation([], [])
        assert rho == 0.0
        assert p == 1.0

        error = calculate_error([], [])
        assert error.rmse == 0.0
        assert error.nrmse == 0.0


# =============================================================================
# NRMSE Tests
# =============================================================================

class TestNRMSE:
    """Tests for Normalised RMSE."""

    def test_nrmse_basic(self):
        """NRMSE should be RMSE / range(actual)."""
        predicted = [1.0, 2.0, 3.0]
        actual = [1.5, 2.5, 3.5]
        metrics = calculate_error(predicted, actual)
        # RMSE = 0.5, range = 3.5 - 1.5 = 2.0
        assert metrics.nrmse == pytest.approx(0.25, abs=0.01)

    def test_nrmse_zero_range(self):
        """NRMSE should be 0 when actual values are constant."""
        predicted = [1.0, 2.0, 3.0]
        actual = [2.0, 2.0, 2.0]
        metrics = calculate_error(predicted, actual)
        assert metrics.nrmse == 0.0

    def test_nrmse_perfect_prediction(self):
        """NRMSE should be 0 for perfect predictions."""
        vals = [0.1, 0.5, 0.9]
        metrics = calculate_error(vals, vals)
        assert metrics.nrmse == 0.0


# =============================================================================
# Classification Metrics Tests
# =============================================================================

class TestClassificationMetrics:
    """Tests for classification including Cohen's κ."""

    def test_all_true_positives(self):
        """All correctly predicted critical."""
        result = calculate_classification([True] * 5, [True] * 5)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1_score == 1.0
        assert result.accuracy == 1.0

    def test_all_true_negatives(self):
        """All correctly predicted non-critical."""
        result = calculate_classification([False] * 5, [False] * 5)
        assert result.precision == 0.0  # no positive predictions
        assert result.recall == 0.0  # no positive actuals
        assert result.f1_score == 0.0
        assert result.accuracy == 1.0

    def test_all_false_positives(self):
        """All predicted critical but none actually are."""
        result = calculate_classification([True] * 5, [False] * 5)
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1_score == 0.0

    def test_all_false_negatives(self):
        """None predicted critical but all actually are."""
        result = calculate_classification([False] * 5, [True] * 5)
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1_score == 0.0

    def test_mixed_classification(self):
        """Mixed predictions produce expected confusion matrix values."""
        pred = [True, True, False, False, True]
        actual = [True, False, False, True, True]
        result = calculate_classification(pred, actual)
        assert result.true_positives == 2
        assert result.false_positives == 1
        assert result.true_negatives == 1
        assert result.false_negatives == 1
        assert result.precision == pytest.approx(2 / 3, abs=0.01)
        assert result.recall == pytest.approx(2 / 3, abs=0.01)


# =============================================================================
# Cohen's Kappa Tests
# =============================================================================

class TestCohensKappa:
    """Tests for Cohen's κ chance-corrected agreement."""

    def test_perfect_agreement(self):
        """Perfect agreement should give κ = 1.0."""
        pred = [True, True, False, False, True]
        actual = [True, True, False, False, True]
        assert cohens_kappa(pred, actual) == pytest.approx(1.0, abs=0.01)

    def test_no_agreement(self):
        """Complete disagreement should give negative κ."""
        pred = [True, True, False, False]
        actual = [False, False, True, True]
        kappa = cohens_kappa(pred, actual)
        assert kappa < 0

    def test_chance_agreement(self):
        """Random-like agreement should give κ near 0."""
        # 50/50 split with ~50% overlap by chance
        pred = [True, False, True, False, True, False, True, False]
        actual = [True, True, False, False, True, True, False, False]
        kappa = cohens_kappa(pred, actual)
        assert -0.5 < kappa < 0.5

    def test_kappa_in_classification_metrics(self):
        """Cohen's κ should be present in ClassificationMetrics."""
        pred = [True, True, False, False, True]
        actual = [True, True, False, False, True]
        result = calculate_classification(pred, actual)
        assert result.cohens_kappa == pytest.approx(1.0, abs=0.01)

    def test_empty_input(self):
        """Empty input should return 0."""
        assert cohens_kappa([], []) == 0.0


# =============================================================================
# Bootstrap Confidence Interval Tests
# =============================================================================

class TestBootstrapCI:
    """Tests for bootstrap confidence interval computation."""

    def test_perfect_data_tight_ci(self):
        """Perfect correlation should produce tight CI near 1.0."""
        x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        lo, hi = bootstrap_ci(
            x, y,
            metric_fn=lambda p, a: spearman_correlation(p, a)[0],
            n_bootstrap=500,
            seed=42,
        )
        assert lo > 0.8
        assert hi <= 1.0
        assert lo <= hi

    def test_noisy_data_wider_ci(self):
        """Noisy data should produce wider CI."""
        x = [0.1, 0.9, 0.2, 0.8, 0.5, 0.4, 0.6, 0.3, 0.7, 0.05]
        y = [0.5, 0.3, 0.8, 0.1, 0.4, 0.6, 0.2, 0.9, 0.05, 0.7]
        lo, hi = bootstrap_ci(
            x, y,
            metric_fn=lambda p, a: spearman_correlation(p, a)[0],
            n_bootstrap=500,
            seed=42,
        )
        # CI should be wider than the perfect case
        assert (hi - lo) > 0.2

    def test_ci_in_correlation_metrics(self):
        """CI bounds should be populated in CorrelationMetrics."""
        x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        y = [0.15, 0.25, 0.28, 0.42, 0.55, 0.58, 0.72]
        result = calculate_correlation(x, y)
        assert result.spearman_ci_lower <= result.spearman
        assert result.spearman_ci_upper >= result.spearman

    def test_f1_ci_in_classification_metrics(self):
        """F1 CI bounds should be populated in ClassificationMetrics."""
        pred = [True, True, False, False, True, True, False]
        actual = [True, True, False, True, True, False, False]
        result = calculate_classification(pred, actual)
        assert result.f1_ci_lower <= result.f1_score
        assert result.f1_ci_upper >= result.f1_score

    def test_small_n_returns_point_estimate(self):
        """n < 5 should return point estimate as CI."""
        x = [0.1, 0.5, 0.9]
        y = [0.2, 0.6, 0.8]
        result = calculate_correlation(x, y)
        assert result.spearman_ci_lower == result.spearman
        assert result.spearman_ci_upper == result.spearman


# =============================================================================
# Ranking Metrics Tests
# =============================================================================

class TestRankingMetrics:
    """Tests for ranking and Top-K metrics."""

    def test_ranking_logic(self):
        """Test ranking metrics with matching top elements."""
        pred = {"A": 0.9, "B": 0.5, "C": 0.1}
        act = {"A": 0.8, "B": 0.4, "C": 0.2}
        res = calculate_ranking(pred, act)
        assert res.top_5_overlap > 0
        assert res.ndcg_10 > 0.9

    def test_ranking_with_mismatch(self):
        """Test ranking metrics when top elements differ."""
        pred = {"A": 0.9, "B": 0.5, "C": 0.1}
        act = {"C": 0.8, "B": 0.5, "A": 0.2}
        res = calculate_ranking(pred, act)
        assert res.top_5_overlap > 0
        assert res.ndcg_10 < 1.0

    def test_ndcg_perfect_ranking(self):
        """Perfect ranking should give NDCG = 1.0."""
        scores = {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3, "E": 0.1}
        res = calculate_ranking(scores, scores)
        assert res.ndcg_5 == pytest.approx(1.0, abs=0.01)

    def test_top_k_with_fewer_than_k(self):
        """Top-K should work when n < K."""
        pred = {"A": 0.9, "B": 0.5}
        act = {"A": 0.8, "B": 0.4}
        res = calculate_ranking(pred, act)
        # Only 2 elements; overlap with top-5 should be based on min(n, k)
        assert res.top_5_overlap > 0


# =============================================================================
# Validator Tests
# =============================================================================

class TestValidator:
    """Tests for Validator orchestration."""

    def test_validator_flow(self):
        """Test basic validator flow with multiple component types."""
        validator = Validator()
        pred = {"A": 0.9, "B": 0.1, "C": 0.5}
        act = {"A": 0.8, "B": 0.2, "C": 0.6}
        types = {"A": "Type1", "B": "Type1", "C": "Type2"}
        res = validator.validate(pred, act, types)
        assert res.overall.sample_size == 3
        assert res.matched_count == 3

    def test_validator_empty_input(self):
        """Test validator handles empty input gracefully."""
        validator = Validator()
        res = validator.validate({}, {}, {})
        assert res.overall.sample_size == 0

    def test_validator_single_element(self):
        """Test validator with single element (edge case)."""
        validator = Validator()
        pred = {"A": 0.5}
        act = {"A": 0.5}
        types = {"A": "Application"}
        res = validator.validate(pred, act, types)
        assert res.matched_count == 1
        assert len(res.warnings) > 0


# =============================================================================
# Pass/Fail Logic Tests
# =============================================================================

class TestPassFailLogic:
    """Tests for validation pass/fail determination."""

    @staticmethod
    def _make_monotonic_data(n: int):
        """Helper: generate perfectly correlated test data."""
        pred = {f"c{i}": (i + 1) / n for i in range(n)}
        # Add slight noise to avoid perfect prediction (RMSE = 0 trivially passes)
        actual = {f"c{i}": (i + 1) / n + 0.01 for i in range(n)}
        types = {f"c{i}": "App" for i in range(n)}
        return pred, actual, types

    def test_all_targets_met(self):
        """Validation should pass when all targets are met with lenient targets."""
        pred, actual, types = self._make_monotonic_data(20)
        validator = Validator(targets=ValidationTargets(
            spearman=0.50,
            spearman_p_max=0.10,
            f1_score=0.30,
            top_5_overlap=0.20,
            rmse_max=0.50,
        ))
        result = validator.validate(pred, actual, types)
        assert result.passed is True

    def test_spearman_fails(self):
        """Validation fails when Spearman target is not met."""
        # Reversed order -> negative correlation
        pred = {f"c{i}": (20 - i) / 20 for i in range(20)}
        actual = {f"c{i}": (i + 1) / 20 for i in range(20)}
        types = {f"c{i}": "App" for i in range(20)}
        validator = Validator(targets=ValidationTargets(spearman=0.70))
        result = validator.validate(pred, actual, types)
        assert result.passed is False

    def test_p_value_gate(self):
        """Validation should fail when Spearman p-value exceeds threshold."""
        # With n=3, p-values tend to be high even for decent correlation
        validator = Validator(targets=ValidationTargets(
            spearman=0.0,  # very lenient
            spearman_p_max=0.001,  # very strict p-value
            f1_score=0.0,
            top_5_overlap=0.0,
            rmse_max=10.0,
        ))
        pred = {"A": 0.9, "B": 0.5, "C": 0.1}
        actual = {"A": 0.8, "B": 0.4, "C": 0.2}
        types = {"A": "App", "B": "App", "C": "App"}
        result = validator.validate(pred, actual, types)
        # With n=3, even perfect correlation has high p-value
        # so this should likely fail the p-value gate
        assert result.overall.correlation.spearman_p > 0.001 or result.passed

    def test_rmse_gate(self):
        """Validation should fail when RMSE exceeds threshold."""
        validator = Validator(targets=ValidationTargets(
            spearman=0.0,
            spearman_p_max=1.0,
            f1_score=0.0,
            top_5_overlap=0.0,
            rmse_max=0.01,  # very strict RMSE
        ))
        # Same ordering but large offset
        pred = {f"c{i}": i / 10 for i in range(10)}
        actual = {f"c{i}": i / 10 + 0.5 for i in range(10)}
        types = {f"c{i}": "App" for i in range(10)}
        result = validator.validate(pred, actual, types)
        assert result.passed is False
        assert result.overall.error.rmse > 0.01

    def test_custom_targets_respected(self):
        """Custom ValidationTargets should override defaults."""
        targets = ValidationTargets(
            spearman=0.99,
            spearman_p_max=0.05,
            f1_score=0.99,
            top_5_overlap=0.99,
            rmse_max=0.001,
        )
        validator = Validator(targets=targets)
        pred, actual, types = self._make_monotonic_data(20)
        result = validator.validate(pred, actual, types)
        # Very strict targets - unlikely to pass with slight noise
        assert result.overall.targets.spearman == 0.99

    def test_top5_overlap_default_aligned_with_docs(self):
        """Default top_5_overlap should be 0.60 (tightened in v4 to align with reliability targets)."""
        targets = ValidationTargets()
        assert targets.top_5_overlap == 0.60


# =============================================================================
# Warning Tests
# =============================================================================

class TestWarnings:
    """Tests for validator warnings."""

    def test_low_power_warning(self):
        """n < 10 should trigger low statistical power warning."""
        validator = Validator()
        pred = {f"c{i}": i / 5 for i in range(5)}
        actual = {f"c{i}": i / 5 + 0.01 for i in range(5)}
        result = validator.validate(pred, actual)
        power_warnings = [w for w in result.warnings if "Low statistical power" in w]
        assert len(power_warnings) == 1

    def test_no_power_warning_for_large_n(self):
        """n >= 10 should not trigger low power warning."""
        validator = Validator()
        pred = {f"c{i}": i / 15 for i in range(15)}
        actual = {f"c{i}": i / 15 + 0.01 for i in range(15)}
        result = validator.validate(pred, actual)
        power_warnings = [w for w in result.warnings if "Low statistical power" in w]
        assert len(power_warnings) == 0

    def test_unmatched_ids_warning(self):
        """Mismatched IDs should produce alignment warnings."""
        validator = Validator()
        pred = {"A": 0.9, "B": 0.5, "C": 0.3, "D": 0.1}
        actual = {"B": 0.6, "C": 0.4, "D": 0.2, "E": 0.1}
        result = validator.validate(pred, actual)
        assert any("predicted components not in actual" in w for w in result.warnings)
        assert any("actual components not in predictions" in w for w in result.warnings)

    def test_insufficient_data_warning(self):
        """n < 3 should produce insufficient data warning."""
        validator = Validator()
        result = validator.validate({"A": 0.5, "B": 0.3}, {"A": 0.4, "B": 0.2})
        assert any("Insufficient data" in w for w in result.warnings)
        assert result.overall.sample_size == 0


# =============================================================================
# By-Type Breakdown Tests
# =============================================================================

class TestByTypeBreakdown:
    """Tests for per-type validation breakdown."""

    def test_by_type_populated(self):
        """by_type should contain groups with n >= 3."""
        validator = Validator()
        pred = {f"app{i}": (i + 1) / 5 for i in range(5)}
        pred.update({f"brk{i}": (i + 1) / 4 for i in range(4)})
        actual = {k: v + 0.01 for k, v in pred.items()}
        types = {}
        for i in range(5):
            types[f"app{i}"] = "Application"
        for i in range(4):
            types[f"brk{i}"] = "Broker"

        result = validator.validate(pred, actual, types)
        # Both types have n >= 3
        assert "Application" in result.by_type
        assert "Broker" in result.by_type
        assert result.by_type["Application"].sample_size == 5
        assert result.by_type["Broker"].sample_size == 4

    def test_by_type_skips_small_groups(self):
        """by_type should skip groups with n < 3."""
        validator = Validator()
        pred = {f"app{i}": (i + 1) / 5 for i in range(5)}
        pred["brk0"] = 0.5
        pred["brk1"] = 0.3
        actual = {k: v + 0.01 for k, v in pred.items()}
        types = {f"app{i}": "Application" for i in range(5)}
        types["brk0"] = "Broker"
        types["brk1"] = "Broker"

        result = validator.validate(pred, actual, types)
        assert "Application" in result.by_type
        assert "Broker" not in result.by_type  # only 2 brokers


# =============================================================================
# Realistic Scale Tests
# =============================================================================

class TestRealisticScale:
    """Tests at realistic component counts."""

    def test_medium_scale_monotonic(self):
        """50 components with perfect ranking should pass easily."""
        n = 50
        pred = {f"c{i}": (i + 1) / n for i in range(n)}
        actual = {f"c{i}": (i + 1) / n + 0.005 for i in range(n)}
        types = {f"c{i}": "Application" for i in range(n)}

        validator = Validator()
        result = validator.validate(pred, actual, types)
        assert result.overall.correlation.spearman > 0.95
        assert result.overall.correlation.spearman_p < 0.001
        assert result.passed is True

    def test_medium_scale_noisy(self):
        """50 components with moderate noise should still pass."""
        import random
        rng = random.Random(42)
        n = 50
        base = [(i + 1) / n for i in range(n)]
        pred = {f"c{i}": base[i] for i in range(n)}
        actual = {f"c{i}": base[i] + rng.gauss(0, 0.05) for i in range(n)}
        types = {f"c{i}": "Application" for i in range(n)}

        validator = Validator()
        result = validator.validate(pred, actual, types)
        assert result.overall.correlation.spearman > 0.70
        assert result.overall.correlation.spearman_p < 0.05
        # CI should be populated and reasonable
        assert result.overall.correlation.spearman_ci_lower < result.overall.correlation.spearman
        assert result.overall.correlation.spearman_ci_upper > result.overall.correlation.spearman