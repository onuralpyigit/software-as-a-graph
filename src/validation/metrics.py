"""
Validation Metrics

Statistical metrics for comparing Predicted Importance (Analysis)
vs Actual Impact (Simulation).

Implements:
    - Correlation metrics (Spearman, Pearson, Kendall)
    - Error metrics (RMSE, MAE, MSE)
    - Classification metrics (Precision, Recall, F1, Accuracy)
    - Ranking metrics (Top-K overlap, NDCG)

All metrics are computed without external dependencies where possible.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Sequence


# =============================================================================
# Validation Targets
# =============================================================================

@dataclass
class ValidationTargets:
    """
    Target thresholds for validation success.
    
    Based on research methodology requirements for graph-based
    criticality prediction validation.
    """
    # Correlation targets (lower bounds)
    spearman: float = 0.70          # Strong positive rank correlation
    pearson: float = 0.65           # Linear correlation
    kendall: float = 0.50           # Concordance
    
    # Classification targets (lower bounds)
    f1_score: float = 0.80          # Balanced precision/recall
    precision: float = 0.80         # Accurate positive predictions
    recall: float = 0.80            # Detection coverage
    accuracy: float = 0.75          # Overall correctness
    
    # Ranking targets (lower bounds)
    top_5_overlap: float = 0.60     # Top 5 critical agreement
    top_10_overlap: float = 0.50    # Top 10 critical agreement
    ndcg_10: float = 0.70           # Normalized DCG at 10
    
    # Error targets (upper bounds)
    rmse_max: float = 0.25          # Maximum acceptable RMSE
    mae_max: float = 0.20           # Maximum acceptable MAE
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "spearman": self.spearman,
            "pearson": self.pearson,
            "kendall": self.kendall,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "top_5_overlap": self.top_5_overlap,
            "top_10_overlap": self.top_10_overlap,
            "rmse_max": self.rmse_max,
            "mae_max": self.mae_max,
        }


# =============================================================================
# Metric Result Classes
# =============================================================================

@dataclass
class CorrelationMetrics:
    """Correlation coefficients between predicted and actual scores."""
    spearman: float = 0.0       # Spearman's rank correlation
    spearman_p: float = 1.0     # p-value for Spearman
    pearson: float = 0.0        # Pearson's linear correlation
    pearson_p: float = 1.0      # p-value for Pearson
    kendall: float = 0.0        # Kendall's tau
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "spearman": round(self.spearman, 4),
            "spearman_p_value": round(self.spearman_p, 6),
            "pearson": round(self.pearson, 4),
            "pearson_p_value": round(self.pearson_p, 6),
            "kendall": round(self.kendall, 4),
        }


@dataclass
class ErrorMetrics:
    """Error measurements between predicted and actual scores."""
    rmse: float = 0.0           # Root Mean Square Error
    mae: float = 0.0            # Mean Absolute Error
    mse: float = 0.0            # Mean Square Error
    max_error: float = 0.0      # Maximum absolute error
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            "mse": round(self.mse, 4),
            "max_error": round(self.max_error, 4),
        }


@dataclass
class ClassificationMetrics:
    """Binary classification metrics for critical component detection."""
    precision: float = 0.0      # TP / (TP + FP)
    recall: float = 0.0         # TP / (TP + FN)
    f1_score: float = 0.0       # Harmonic mean
    accuracy: float = 0.0       # (TP + TN) / Total
    
    # Confusion matrix
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    @property
    def confusion_matrix(self) -> Dict[str, int]:
        return {
            "tp": self.true_positives,
            "fp": self.false_positives,
            "tn": self.true_negatives,
            "fn": self.false_negatives,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "accuracy": round(self.accuracy, 4),
            "confusion_matrix": self.confusion_matrix,
        }


@dataclass
class RankingMetrics:
    """Ranking agreement between predicted and actual ordering."""
    top_5_overlap: float = 0.0      # Overlap at k=5
    top_10_overlap: float = 0.0     # Overlap at k=10
    ndcg_5: float = 0.0             # NDCG at k=5
    ndcg_10: float = 0.0            # NDCG at k=10
    
    # Top-K component lists
    top_5_predicted: List[str] = field(default_factory=list)
    top_5_actual: List[str] = field(default_factory=list)
    top_5_common: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_5_overlap": round(self.top_5_overlap, 4),
            "top_10_overlap": round(self.top_10_overlap, 4),
            "ndcg_5": round(self.ndcg_5, 4),
            "ndcg_10": round(self.ndcg_10, 4),
            "top_5_agreement": {
                "predicted": self.top_5_predicted,
                "actual": self.top_5_actual,
                "common": self.top_5_common,
            },
        }


# =============================================================================
# Correlation Calculations
# =============================================================================

def _get_ranks(values: Sequence[float]) -> List[float]:
    """Convert values to ranks (1-based, averaged for ties)."""
    n = len(values)
    indexed = [(v, i) for i, v in enumerate(values)]
    indexed.sort(key=lambda x: x[0], reverse=True)  # Higher = better rank
    
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        # Find ties
        while j < n - 1 and indexed[j][0] == indexed[j + 1][0]:
            j += 1
        # Average rank for tied values
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][1]] = avg_rank
        i = j + 1
    
    return ranks


def spearman_correlation(predicted: Sequence[float], actual: Sequence[float]) -> Tuple[float, float]:
    """
    Calculate Spearman's rank correlation coefficient.
    
    Returns:
        (rho, p_value) tuple
    """
    n = len(predicted)
    if n < 3:
        return 0.0, 1.0
    
    rank_pred = _get_ranks(predicted)
    rank_actual = _get_ranks(actual)
    
    # Calculate d^2
    d_squared = sum((rp - ra) ** 2 for rp, ra in zip(rank_pred, rank_actual))
    
    # Spearman's rho formula
    rho = 1 - (6 * d_squared) / (n * (n ** 2 - 1))
    
    # Approximate p-value using t-distribution
    if abs(rho) >= 1.0:
        p_value = 0.0
    else:
        t_stat = rho * math.sqrt((n - 2) / (1 - rho ** 2))
        # Simple approximation for p-value
        p_value = 2 * (1 - _t_cdf(abs(t_stat), n - 2))
    
    return rho, p_value


def pearson_correlation(predicted: Sequence[float], actual: Sequence[float]) -> Tuple[float, float]:
    """
    Calculate Pearson's linear correlation coefficient.
    
    Returns:
        (r, p_value) tuple
    """
    n = len(predicted)
    if n < 3:
        return 0.0, 1.0
    
    mean_pred = sum(predicted) / n
    mean_actual = sum(actual) / n
    
    numerator = sum((p - mean_pred) * (a - mean_actual) for p, a in zip(predicted, actual))
    
    var_pred = sum((p - mean_pred) ** 2 for p in predicted)
    var_actual = sum((a - mean_actual) ** 2 for a in actual)
    
    denominator = math.sqrt(var_pred * var_actual)
    
    if denominator == 0:
        return 0.0, 1.0
    
    r = numerator / denominator
    
    # P-value approximation
    if abs(r) >= 1.0:
        p_value = 0.0
    else:
        t_stat = r * math.sqrt((n - 2) / (1 - r ** 2))
        p_value = 2 * (1 - _t_cdf(abs(t_stat), n - 2))
    
    return r, p_value


def kendall_correlation(predicted: Sequence[float], actual: Sequence[float]) -> float:
    """Calculate Kendall's tau rank correlation."""
    n = len(predicted)
    if n < 2:
        return 0.0
    
    concordant = 0
    discordant = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            pred_sign = _sign(predicted[i] - predicted[j])
            actual_sign = _sign(actual[i] - actual[j])
            
            if pred_sign * actual_sign > 0:
                concordant += 1
            elif pred_sign * actual_sign < 0:
                discordant += 1
    
    total = concordant + discordant
    if total == 0:
        return 0.0
    
    return (concordant - discordant) / total


def _sign(x: float) -> int:
    """Return sign of x."""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


def _t_cdf(t: float, df: int) -> float:
    """Approximate t-distribution CDF (simple approximation)."""
    # Using normal approximation for df > 30
    if df > 30:
        return 0.5 * (1 + math.erf(t / math.sqrt(2)))
    
    # Simple approximation for smaller df
    x = df / (df + t ** 2)
    return 1 - 0.5 * _incomplete_beta(df / 2, 0.5, x)


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """Simple incomplete beta approximation."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    
    # Simple numerical integration
    n_steps = 100
    dx = x / n_steps
    result = 0.0
    
    for i in range(n_steps):
        xi = (i + 0.5) * dx
        result += (xi ** (a - 1)) * ((1 - xi) ** (b - 1)) * dx
    
    # Normalize (approximate)
    beta_ab = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
    return result / beta_ab


# =============================================================================
# Error Calculations
# =============================================================================

def calculate_error_metrics(
    predicted: Sequence[float],
    actual: Sequence[float]
) -> ErrorMetrics:
    """Calculate error metrics between predicted and actual values."""
    n = len(predicted)
    if n == 0:
        return ErrorMetrics()
    
    errors = [abs(p - a) for p, a in zip(predicted, actual)]
    squared_errors = [(p - a) ** 2 for p, a in zip(predicted, actual)]
    
    mse = sum(squared_errors) / n
    mae = sum(errors) / n
    rmse = math.sqrt(mse)
    max_error = max(errors)
    
    return ErrorMetrics(
        rmse=rmse,
        mae=mae,
        mse=mse,
        max_error=max_error,
    )


# =============================================================================
# Classification Calculations
# =============================================================================

def calculate_classification_metrics(
    predicted_critical: Sequence[bool],
    actual_critical: Sequence[bool]
) -> ClassificationMetrics:
    """
    Calculate binary classification metrics.
    
    Args:
        predicted_critical: Boolean sequence of predicted critical
        actual_critical: Boolean sequence of actually critical
    """
    tp = fp = tn = fn = 0
    
    for pred, actual in zip(predicted_critical, actual_critical):
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    
    return ClassificationMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1,
        accuracy=accuracy,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )


# =============================================================================
# Ranking Calculations
# =============================================================================

def calculate_ranking_metrics(
    predicted: Dict[str, float],
    actual: Dict[str, float],
    k_values: List[int] = [5, 10]
) -> RankingMetrics:
    """
    Calculate ranking agreement metrics.
    
    Args:
        predicted: Dict mapping component ID to predicted score
        actual: Dict mapping component ID to actual score
        k_values: List of K values for Top-K overlap
    """
    # Sort by score (descending)
    pred_sorted = sorted(predicted.items(), key=lambda x: x[1], reverse=True)
    actual_sorted = sorted(actual.items(), key=lambda x: x[1], reverse=True)
    
    pred_ids = [x[0] for x in pred_sorted]
    actual_ids = [x[0] for x in actual_sorted]
    
    # Top-K overlap
    def top_k_overlap(k: int) -> Tuple[float, List[str], List[str], List[str]]:
        pred_top = set(pred_ids[:k])
        actual_top = set(actual_ids[:k])
        common = pred_top & actual_top
        overlap = len(common) / k if k > 0 else 0.0
        return overlap, pred_ids[:k], actual_ids[:k], list(common)
    
    overlap_5, pred_5, actual_5, common_5 = top_k_overlap(5)
    overlap_10, _, _, _ = top_k_overlap(10)
    
    # NDCG calculations
    ndcg_5 = _calculate_ndcg(pred_ids, actual, k=5)
    ndcg_10 = _calculate_ndcg(pred_ids, actual, k=10)
    
    return RankingMetrics(
        top_5_overlap=overlap_5,
        top_10_overlap=overlap_10,
        ndcg_5=ndcg_5,
        ndcg_10=ndcg_10,
        top_5_predicted=pred_5,
        top_5_actual=actual_5,
        top_5_common=common_5,
    )


def _calculate_ndcg(ranked_ids: List[str], relevance: Dict[str, float], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k."""
    if not ranked_ids or not relevance:
        return 0.0
    
    # DCG for predicted ranking
    dcg = 0.0
    for i, comp_id in enumerate(ranked_ids[:k]):
        rel = relevance.get(comp_id, 0.0)
        dcg += rel / math.log2(i + 2)  # i+2 because rank starts at 1
    
    # Ideal DCG (sorted by relevance)
    ideal_sorted = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_sorted))
    
    return dcg / idcg if idcg > 0 else 0.0