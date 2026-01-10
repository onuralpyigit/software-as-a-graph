"""
Validation Metrics

Statistical metrics for comparing Predicted Importance (Analysis)
vs Actual Impact (Simulation).

Implements:
- Correlation metrics (Spearman, Pearson, Kendall)
- Error metrics (RMSE, MAE)
- Classification metrics (Precision, Recall, F1, Accuracy)
- Ranking metrics (Top-K overlap, NDCG)

Author: Software-as-a-Graph Research Project
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict, field
from typing import Sequence, Dict, List, Any, Tuple


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
    # Correlation targets
    spearman: float = 0.70          # Strong positive rank correlation
    pearson: float = 0.65           # Linear correlation
    kendall: float = 0.50           # Concordance
    
    # Classification targets
    f1_score: float = 0.80          # Balanced precision/recall
    precision: float = 0.80         # Accurate positive predictions
    recall: float = 0.80            # Detection coverage
    accuracy: float = 0.75          # Overall correctness
    
    # Ranking targets
    top_5_overlap: float = 0.60     # Top 5 critical agreement
    top_10_overlap: float = 0.50    # Top 10 critical agreement
    ndcg_10: float = 0.70           # Normalized DCG at 10
    
    # Error targets (upper bounds)
    rmse_max: float = 0.25          # Maximum acceptable RMSE
    mae_max: float = 0.20           # Maximum acceptable MAE
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# =============================================================================
# Metric Result Classes
# =============================================================================

@dataclass
class CorrelationMetrics:
    """Correlation coefficients between predicted and actual scores."""
    spearman: float = 0.0       # Spearman's rank correlation
    pearson: float = 0.0        # Pearson's linear correlation
    kendall: float = 0.0        # Kendall's tau
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "spearman": round(self.spearman, 4),
            "pearson": round(self.pearson, 4),
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
    f1_score: float = 0.0       # Harmonic mean of precision and recall
    accuracy: float = 0.0       # (TP + TN) / Total
    
    # Confusion matrix components
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
    """Ranking agreement metrics between predicted and actual ordering."""
    top_5_overlap: float = 0.0      # Jaccard similarity for top 5
    top_10_overlap: float = 0.0     # Jaccard similarity for top 10
    top_k_overlap: Dict[int, float] = field(default_factory=dict)
    ndcg_5: float = 0.0             # NDCG at 5
    ndcg_10: float = 0.0            # NDCG at 10
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_5_overlap": round(self.top_5_overlap, 4),
            "top_10_overlap": round(self.top_10_overlap, 4),
            "ndcg_5": round(self.ndcg_5, 4),
            "ndcg_10": round(self.ndcg_10, 4),
        }


# =============================================================================
# Correlation Functions
# =============================================================================

def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Calculate Spearman's rank correlation coefficient.
    
    Measures monotonic relationship between predicted and actual scores.
    Range: [-1, 1] where 1 indicates perfect positive correlation.
    
    Args:
        x: Predicted scores
        y: Actual scores
        
    Returns:
        Spearman's rho coefficient
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    x_ranks = _fractional_rank(x)
    y_ranks = _fractional_rank(y)
    
    return pearson_correlation(x_ranks, y_ranks)


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Calculate Pearson's linear correlation coefficient.
    
    Measures linear relationship between predicted and actual scores.
    Range: [-1, 1] where 1 indicates perfect positive correlation.
    
    Args:
        x: Predicted scores
        y: Actual scores
        
    Returns:
        Pearson's r coefficient
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    
    if var_x == 0 or var_y == 0:
        return 0.0
    
    covariance = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    
    return covariance / math.sqrt(var_x * var_y)


def kendall_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Calculate Kendall's tau-b correlation coefficient.
    
    Measures ordinal association based on concordant/discordant pairs.
    Range: [-1, 1] where 1 indicates perfect agreement in ordering.
    
    Args:
        x: Predicted scores
        y: Actual scores
        
    Returns:
        Kendall's tau-b coefficient
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            x_diff = x[i] - x[j]
            y_diff = y[i] - y[j]
            
            product = x_diff * y_diff
            
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            else:
                if x_diff == 0:
                    ties_x += 1
                if y_diff == 0:
                    ties_y += 1
    
    # Kendall's tau-b formula (handles ties)
    n_pairs = n * (n - 1) / 2
    denominator = math.sqrt((n_pairs - ties_x) * (n_pairs - ties_y))
    
    if denominator == 0:
        return 0.0
    
    return (concordant - discordant) / denominator


def _fractional_rank(values: Sequence[float]) -> List[float]:
    """
    Compute fractional ranks for values (handles ties).
    
    Ties receive the average of their ranks.
    """
    n = len(values)
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n
    
    i = 0
    while i < n:
        j = i
        # Find all ties
        while j < n - 1 and indexed[j][0] != indexed[j + 1][0] and indexed[j][1] == indexed[j + 1][1]:
            j += 1
        
        # Average rank for tied values
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        
        i = j + 1
    
    return ranks


# =============================================================================
# Error Functions
# =============================================================================

def calculate_error_metrics(
    predicted: Sequence[float],
    actual: Sequence[float]
) -> ErrorMetrics:
    """
    Calculate error metrics between predicted and actual scores.
    
    Args:
        predicted: Predicted scores
        actual: Actual scores
        
    Returns:
        ErrorMetrics with RMSE, MAE, MSE, and max error
    """
    if len(predicted) != len(actual) or len(predicted) == 0:
        return ErrorMetrics()
    
    n = len(predicted)
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
# Classification Functions
# =============================================================================

def calculate_classification_metrics(
    predicted_critical: Sequence[bool],
    actual_critical: Sequence[bool]
) -> ClassificationMetrics:
    """
    Calculate binary classification metrics for critical component detection.
    
    Args:
        predicted_critical: Boolean sequence of predicted critical status
        actual_critical: Boolean sequence of actual critical status
        
    Returns:
        ClassificationMetrics with precision, recall, F1, accuracy
    """
    if len(predicted_critical) != len(actual_critical) or len(predicted_critical) == 0:
        return ClassificationMetrics()
    
    tp = fp = fn = tn = 0
    
    for pred, actual in zip(predicted_critical, actual_critical):
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(predicted_critical)
    
    return ClassificationMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        accuracy=accuracy,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )


# =============================================================================
# Ranking Functions
# =============================================================================

def calculate_ranking_metrics(
    predicted_scores: Dict[str, float],
    actual_scores: Dict[str, float]
) -> RankingMetrics:
    """
    Calculate ranking agreement metrics.
    
    Args:
        predicted_scores: Dict mapping component ID to predicted score
        actual_scores: Dict mapping component ID to actual score
        
    Returns:
        RankingMetrics with top-K overlap and NDCG scores
    """
    # Get common IDs
    common_ids = list(set(predicted_scores.keys()) & set(actual_scores.keys()))
    
    if not common_ids:
        return RankingMetrics()
    
    # Sort by scores (descending)
    pred_ranked = sorted(common_ids, key=lambda x: predicted_scores[x], reverse=True)
    actual_ranked = sorted(common_ids, key=lambda x: actual_scores[x], reverse=True)
    
    # Calculate top-K overlaps
    def top_k_overlap(k: int) -> float:
        k = min(k, len(common_ids))
        if k == 0:
            return 0.0
        pred_top_k = set(pred_ranked[:k])
        actual_top_k = set(actual_ranked[:k])
        return len(pred_top_k & actual_top_k) / k
    
    # Calculate NDCG
    def dcg(ranked_ids: List[str], relevance_map: Dict[str, float], k: int) -> float:
        """Discounted Cumulative Gain at k."""
        score = 0.0
        for i, comp_id in enumerate(ranked_ids[:k]):
            rel = relevance_map.get(comp_id, 0.0)
            # Use position-based discount
            score += rel / math.log2(i + 2)
        return score
    
    def ndcg(k: int) -> float:
        """Normalized DCG at k."""
        k = min(k, len(common_ids))
        if k == 0:
            return 0.0
        
        # Ideal DCG (actual scores sorted by actual scores)
        ideal_dcg = dcg(actual_ranked, actual_scores, k)
        
        # Predicted DCG (predicted ranking evaluated with actual scores)
        pred_dcg = dcg(pred_ranked, actual_scores, k)
        
        if ideal_dcg == 0:
            return 0.0
        
        return pred_dcg / ideal_dcg
    
    # Compute metrics at various k values
    top_k_overlaps = {}
    for k in [3, 5, 10, 20]:
        if k <= len(common_ids):
            top_k_overlaps[k] = top_k_overlap(k)
    
    return RankingMetrics(
        top_5_overlap=top_k_overlap(5),
        top_10_overlap=top_k_overlap(10),
        top_k_overlap=top_k_overlaps,
        ndcg_5=ndcg(5),
        ndcg_10=ndcg(10),
    )


# =============================================================================
# Combined Metrics Calculation
# =============================================================================

def calculate_all_metrics(
    predicted_scores: Dict[str, float],
    actual_scores: Dict[str, float],
    critical_threshold_percentile: float = 75.0
) -> Tuple[CorrelationMetrics, ErrorMetrics, ClassificationMetrics, RankingMetrics]:
    """
    Calculate all validation metrics.
    
    Args:
        predicted_scores: Dict mapping component ID to predicted score
        actual_scores: Dict mapping component ID to actual score
        critical_threshold_percentile: Percentile for critical classification
        
    Returns:
        Tuple of (CorrelationMetrics, ErrorMetrics, ClassificationMetrics, RankingMetrics)
    """
    # Align data
    common_ids = sorted(set(predicted_scores.keys()) & set(actual_scores.keys()))
    
    if len(common_ids) < 2:
        return (CorrelationMetrics(), ErrorMetrics(), ClassificationMetrics(), RankingMetrics())
    
    pred_values = [predicted_scores[i] for i in common_ids]
    actual_values = [actual_scores[i] for i in common_ids]
    
    # Correlation metrics
    correlation = CorrelationMetrics(
        spearman=spearman_correlation(pred_values, actual_values),
        pearson=pearson_correlation(pred_values, actual_values),
        kendall=kendall_correlation(pred_values, actual_values),
    )
    
    # Error metrics
    error = calculate_error_metrics(pred_values, actual_values)
    
    # Classification metrics (using percentile threshold)
    pred_threshold = _percentile(pred_values, critical_threshold_percentile)
    actual_threshold = _percentile(actual_values, critical_threshold_percentile)
    
    pred_critical = [p >= pred_threshold for p in pred_values]
    actual_critical = [a >= actual_threshold for a in actual_values]
    
    classification = calculate_classification_metrics(pred_critical, actual_critical)
    
    # Ranking metrics
    ranking = calculate_ranking_metrics(
        {i: predicted_scores[i] for i in common_ids},
        {i: actual_scores[i] for i in common_ids},
    )
    
    return correlation, error, classification, ranking


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Calculate percentile value."""
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * percentile / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    
    if f == c:
        return sorted_values[int(k)]
    
    return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)


# =============================================================================
# Validation Summary
# =============================================================================

@dataclass
class ValidationSummary:
    """Summary of all validation metrics with pass/fail status."""
    correlation: CorrelationMetrics
    error: ErrorMetrics
    classification: ClassificationMetrics
    ranking: RankingMetrics
    
    sample_size: int = 0
    targets: ValidationTargets = field(default_factory=ValidationTargets)
    
    @property
    def passed(self) -> bool:
        """Check if validation passes all primary targets."""
        return (
            self.correlation.spearman >= self.targets.spearman and
            self.classification.f1_score >= self.targets.f1_score and
            self.ranking.top_5_overlap >= self.targets.top_5_overlap
        )
    
    @property
    def passed_strict(self) -> bool:
        """Check if validation passes all targets (strict)."""
        return (
            self.passed and
            self.classification.precision >= self.targets.precision and
            self.classification.recall >= self.targets.recall and
            self.error.rmse <= self.targets.rmse_max
        )
    
    def get_pass_fail_details(self) -> Dict[str, Tuple[float, float, bool]]:
        """Get (actual, target, passed) for each metric."""
        return {
            "spearman": (self.correlation.spearman, self.targets.spearman, 
                        self.correlation.spearman >= self.targets.spearman),
            "f1_score": (self.classification.f1_score, self.targets.f1_score,
                        self.classification.f1_score >= self.targets.f1_score),
            "precision": (self.classification.precision, self.targets.precision,
                         self.classification.precision >= self.targets.precision),
            "recall": (self.classification.recall, self.targets.recall,
                      self.classification.recall >= self.targets.recall),
            "top_5_overlap": (self.ranking.top_5_overlap, self.targets.top_5_overlap,
                             self.ranking.top_5_overlap >= self.targets.top_5_overlap),
            "rmse": (self.error.rmse, self.targets.rmse_max,
                    self.error.rmse <= self.targets.rmse_max),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "passed": self.passed,
            "passed_strict": self.passed_strict,
            "correlation": self.correlation.to_dict(),
            "error": self.error.to_dict(),
            "classification": self.classification.to_dict(),
            "ranking": self.ranking.to_dict(),
            "pass_fail_details": {
                k: {"actual": round(v[0], 4), "target": round(v[1], 4), "passed": v[2]}
                for k, v in self.get_pass_fail_details().items()
            },
        }