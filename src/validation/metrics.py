"""
Validation Metrics - Version 5.0

Statistical metrics for validating graph-based criticality predictions
against actual failure impact scores from simulation.

Metrics:
- Correlation: Spearman (rank), Pearson (linear), Kendall (ordinal)
- Classification: F1-Score, Precision, Recall, Accuracy
- Ranking: Top-K Overlap, NDCG, Mean Reciprocal Rank
- Statistical: Bootstrap Confidence Intervals, P-values

Validation Targets (from research):
- Spearman ρ ≥ 0.70 (rank correlation)
- F1-Score ≥ 0.90 (classification accuracy)
- Precision ≥ 0.80
- Recall ≥ 0.80
- Top-5 Overlap ≥ 60%
- Top-10 Overlap ≥ 70%

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Sequence
from collections import defaultdict


# =============================================================================
# Enums
# =============================================================================

class ValidationStatus(Enum):
    """Overall validation status"""
    PASSED = "passed"
    PARTIAL = "partial"
    FAILED = "failed"
    ERROR = "error"


class MetricStatus(Enum):
    """Status of individual metric"""
    ABOVE_TARGET = "above_target"
    BELOW_TARGET = "below_target"
    NOT_CALCULATED = "not_calculated"


# =============================================================================
# Validation Targets
# =============================================================================

@dataclass
class ValidationTargets:
    """Target thresholds for validation metrics"""
    spearman: float = 0.70
    pearson: float = 0.65
    kendall: float = 0.60
    f1_score: float = 0.90
    precision: float = 0.80
    recall: float = 0.80
    accuracy: float = 0.85
    top_5_overlap: float = 0.60
    top_10_overlap: float = 0.70
    
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
        }


# =============================================================================
# Correlation Metrics
# =============================================================================

def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Calculate Spearman rank correlation coefficient.
    
    Measures how well the relationship between two variables can be 
    described by a monotonic function. Range: [-1, 1]
    
    Args:
        x: First sequence of values
        y: Second sequence of values (same length as x)
    
    Returns:
        Spearman correlation coefficient
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    
    # Calculate ranks
    rank_x = _calculate_ranks(x)
    rank_y = _calculate_ranks(y)
    
    # Calculate Pearson correlation of ranks
    return pearson_correlation(rank_x, rank_y)


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Calculate Pearson linear correlation coefficient.
    
    Measures linear relationship between variables. Range: [-1, 1]
    
    Args:
        x: First sequence of values
        y: Second sequence of values
    
    Returns:
        Pearson correlation coefficient
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # Calculate covariance and standard deviations
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    
    if std_x == 0 or std_y == 0:
        return 0.0
    
    return cov / (std_x * std_y)


def kendall_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Calculate Kendall tau-b correlation coefficient.
    
    Measures ordinal association. More robust than Spearman for
    small samples or many tied values. Range: [-1, 1]
    
    Args:
        x: First sequence of values
        y: Second sequence of values
    
    Returns:
        Kendall tau-b coefficient
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    ties_both = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            sign_x = _sign(x[i] - x[j])
            sign_y = _sign(y[i] - y[j])
            
            if sign_x == 0 and sign_y == 0:
                ties_both += 1
            elif sign_x == 0:
                ties_x += 1
            elif sign_y == 0:
                ties_y += 1
            elif sign_x == sign_y:
                concordant += 1
            else:
                discordant += 1
    
    total_pairs = n * (n - 1) // 2
    n0 = total_pairs
    n1 = ties_x + ties_both
    n2 = ties_y + ties_both
    
    denom = math.sqrt((n0 - n1) * (n0 - n2))
    if denom == 0:
        return 0.0
    
    return (concordant - discordant) / denom


def _calculate_ranks(values: Sequence[float]) -> List[float]:
    """Calculate ranks with average ranking for ties"""
    n = len(values)
    indexed = [(v, i) for i, v in enumerate(values)]
    indexed.sort(key=lambda x: x[0])
    
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and indexed[j][0] == indexed[i][0]:
            j += 1
        # Average rank for tied values
        avg_rank = (i + j + 1) / 2
        for k in range(i, j):
            ranks[indexed[k][1]] = avg_rank
        i = j
    
    return ranks


def _sign(x: float) -> int:
    """Sign function"""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


@dataclass
class CorrelationMetrics:
    """Correlation metrics between predicted and actual scores"""
    spearman: float = 0.0
    pearson: float = 0.0
    kendall: float = 0.0
    n_samples: int = 0
    spearman_pvalue: Optional[float] = None
    pearson_pvalue: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spearman": round(self.spearman, 4),
            "pearson": round(self.pearson, 4),
            "kendall": round(self.kendall, 4),
            "n_samples": self.n_samples,
            "spearman_pvalue": round(self.spearman_pvalue, 6) if self.spearman_pvalue else None,
            "pearson_pvalue": round(self.pearson_pvalue, 6) if self.pearson_pvalue else None,
        }
    
    def check_targets(self, targets: ValidationTargets) -> Dict[str, MetricStatus]:
        """Check metrics against targets"""
        return {
            "spearman": MetricStatus.ABOVE_TARGET if self.spearman >= targets.spearman 
                       else MetricStatus.BELOW_TARGET,
            "pearson": MetricStatus.ABOVE_TARGET if self.pearson >= targets.pearson 
                      else MetricStatus.BELOW_TARGET,
            "kendall": MetricStatus.ABOVE_TARGET if self.kendall >= targets.kendall 
                      else MetricStatus.BELOW_TARGET,
        }


def calculate_correlation(
    predicted: Sequence[float],
    actual: Sequence[float],
) -> CorrelationMetrics:
    """
    Calculate all correlation metrics.
    
    Args:
        predicted: Predicted criticality scores
        actual: Actual impact scores from simulation
    
    Returns:
        CorrelationMetrics with all coefficients
    """
    n = len(predicted)
    
    return CorrelationMetrics(
        spearman=spearman_correlation(predicted, actual),
        pearson=pearson_correlation(predicted, actual),
        kendall=kendall_correlation(predicted, actual),
        n_samples=n,
    )


# =============================================================================
# Classification Metrics
# =============================================================================

@dataclass
class ConfusionMatrix:
    """Binary classification confusion matrix"""
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    @property
    def total(self) -> int:
        return (self.true_positives + self.true_negatives + 
                self.false_positives + self.false_negatives)
    
    @property
    def precision(self) -> float:
        """TP / (TP + FP)"""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """TP / (TP + FN) - also known as sensitivity"""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def specificity(self) -> float:
        """TN / (TN + FP)"""
        denom = self.true_negatives + self.false_positives
        return self.true_negatives / denom if denom > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        """(TP + TN) / Total"""
        if self.total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / self.total
    
    @property
    def f1_score(self) -> float:
        """2 * (Precision * Recall) / (Precision + Recall)"""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
    
    @property
    def f_beta(self) -> float:
        """F-beta score with beta=2 (weights recall higher)"""
        beta = 2.0
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return (1 + beta**2) * p * r / (beta**2 * p + r)
    
    @property
    def mcc(self) -> float:
        """Matthews Correlation Coefficient - balanced measure"""
        tp, tn = self.true_positives, self.true_negatives
        fp, fn = self.false_positives, self.false_negatives
        
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denom == 0:
            return 0.0
        return (tp * tn - fp * fn) / denom
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "specificity": round(self.specificity, 4),
            "accuracy": round(self.accuracy, 4),
            "f1_score": round(self.f1_score, 4),
            "mcc": round(self.mcc, 4),
        }


@dataclass
class ClassificationMetrics:
    """Complete classification metrics"""
    confusion_matrix: ConfusionMatrix
    threshold: float
    critical_count_predicted: int = 0
    critical_count_actual: int = 0
    
    @property
    def precision(self) -> float:
        return self.confusion_matrix.precision
    
    @property
    def recall(self) -> float:
        return self.confusion_matrix.recall
    
    @property
    def f1_score(self) -> float:
        return self.confusion_matrix.f1_score
    
    @property
    def accuracy(self) -> float:
        return self.confusion_matrix.accuracy
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "threshold": self.threshold,
            "critical_count_predicted": self.critical_count_predicted,
            "critical_count_actual": self.critical_count_actual,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "accuracy": round(self.accuracy, 4),
        }
    
    def check_targets(self, targets: ValidationTargets) -> Dict[str, MetricStatus]:
        """Check metrics against targets"""
        return {
            "f1_score": MetricStatus.ABOVE_TARGET if self.f1_score >= targets.f1_score 
                       else MetricStatus.BELOW_TARGET,
            "precision": MetricStatus.ABOVE_TARGET if self.precision >= targets.precision 
                        else MetricStatus.BELOW_TARGET,
            "recall": MetricStatus.ABOVE_TARGET if self.recall >= targets.recall 
                     else MetricStatus.BELOW_TARGET,
            "accuracy": MetricStatus.ABOVE_TARGET if self.accuracy >= targets.accuracy 
                       else MetricStatus.BELOW_TARGET,
        }


def calculate_confusion_matrix(
    predicted_critical: Sequence[bool],
    actual_critical: Sequence[bool],
) -> ConfusionMatrix:
    """
    Calculate confusion matrix from binary classifications.
    
    Args:
        predicted_critical: Predicted critical status
        actual_critical: Actual critical status
    
    Returns:
        ConfusionMatrix
    """
    cm = ConfusionMatrix()
    
    for pred, actual in zip(predicted_critical, actual_critical):
        if pred and actual:
            cm.true_positives += 1
        elif not pred and not actual:
            cm.true_negatives += 1
        elif pred and not actual:
            cm.false_positives += 1
        else:
            cm.false_negatives += 1
    
    return cm


def calculate_classification(
    predicted_scores: Sequence[float],
    actual_scores: Sequence[float],
    threshold: Optional[float] = None,
) -> ClassificationMetrics:
    """
    Calculate classification metrics using threshold-based classification.
    
    Components are classified as "critical" if their score >= threshold.
    Default threshold is the 75th percentile of actual scores.
    
    Args:
        predicted_scores: Predicted criticality scores
        actual_scores: Actual impact scores
        threshold: Classification threshold (default: 75th percentile)
    
    Returns:
        ClassificationMetrics
    """
    if not actual_scores:
        return ClassificationMetrics(
            confusion_matrix=ConfusionMatrix(),
            threshold=threshold or 0.0
        )
    
    # Calculate threshold as 75th percentile if not provided
    if threshold is None:
        sorted_actual = sorted(actual_scores)
        idx = int(len(sorted_actual) * 0.75)
        threshold = sorted_actual[min(idx, len(sorted_actual) - 1)]
    
    # Classify
    predicted_critical = [s >= threshold for s in predicted_scores]
    actual_critical = [s >= threshold for s in actual_scores]
    
    cm = calculate_confusion_matrix(predicted_critical, actual_critical)
    
    return ClassificationMetrics(
        confusion_matrix=cm,
        threshold=threshold,
        critical_count_predicted=sum(predicted_critical),
        critical_count_actual=sum(actual_critical),
    )


# =============================================================================
# Ranking Metrics
# =============================================================================

@dataclass
class RankingMetrics:
    """Ranking comparison metrics"""
    top_5_overlap: float = 0.0
    top_10_overlap: float = 0.0
    top_k_overlaps: Dict[int, float] = field(default_factory=dict)
    ndcg: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    rank_difference_mean: float = 0.0
    rank_difference_std: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_5_overlap": round(self.top_5_overlap, 4),
            "top_10_overlap": round(self.top_10_overlap, 4),
            "top_k_overlaps": {k: round(v, 4) for k, v in self.top_k_overlaps.items()},
            "ndcg": round(self.ndcg, 4),
            "mrr": round(self.mrr, 4),
            "rank_difference_mean": round(self.rank_difference_mean, 4),
            "rank_difference_std": round(self.rank_difference_std, 4),
        }
    
    def check_targets(self, targets: ValidationTargets) -> Dict[str, MetricStatus]:
        """Check metrics against targets"""
        return {
            "top_5_overlap": MetricStatus.ABOVE_TARGET if self.top_5_overlap >= targets.top_5_overlap 
                            else MetricStatus.BELOW_TARGET,
            "top_10_overlap": MetricStatus.ABOVE_TARGET if self.top_10_overlap >= targets.top_10_overlap 
                             else MetricStatus.BELOW_TARGET,
        }


def calculate_ranking(
    predicted_ranking: Sequence[str],
    actual_ranking: Sequence[str],
    k_values: Optional[List[int]] = None,
) -> RankingMetrics:
    """
    Calculate ranking comparison metrics.
    
    Args:
        predicted_ranking: Component IDs ranked by predicted criticality
        actual_ranking: Component IDs ranked by actual impact
        k_values: Values of k for top-k overlap (default: [3, 5, 10, 20])
    
    Returns:
        RankingMetrics
    """
    if k_values is None:
        k_values = [3, 5, 10, 20]
    
    n = len(predicted_ranking)
    metrics = RankingMetrics()
    
    # Build rank maps
    pred_rank = {comp: i for i, comp in enumerate(predicted_ranking)}
    actual_rank = {comp: i for i, comp in enumerate(actual_ranking)}
    
    # Calculate top-k overlaps
    for k in k_values:
        if k <= n:
            pred_top_k = set(predicted_ranking[:k])
            actual_top_k = set(actual_ranking[:k])
            overlap = len(pred_top_k & actual_top_k) / k
            metrics.top_k_overlaps[k] = overlap
    
    metrics.top_5_overlap = metrics.top_k_overlaps.get(5, 0.0)
    metrics.top_10_overlap = metrics.top_k_overlaps.get(10, 0.0)
    
    # Calculate NDCG (Normalized Discounted Cumulative Gain)
    metrics.ndcg = _calculate_ndcg(predicted_ranking, actual_ranking)
    
    # Calculate MRR (Mean Reciprocal Rank)
    metrics.mrr = _calculate_mrr(predicted_ranking, actual_ranking)
    
    # Calculate rank differences
    rank_diffs = []
    for comp in predicted_ranking:
        if comp in actual_rank:
            diff = abs(pred_rank[comp] - actual_rank[comp])
            rank_diffs.append(diff)
    
    if rank_diffs:
        metrics.rank_difference_mean = sum(rank_diffs) / len(rank_diffs)
        if len(rank_diffs) > 1:
            variance = sum((d - metrics.rank_difference_mean) ** 2 for d in rank_diffs) / len(rank_diffs)
            metrics.rank_difference_std = math.sqrt(variance)
    
    return metrics


def _calculate_ndcg(predicted: Sequence[str], actual: Sequence[str]) -> float:
    """Calculate Normalized Discounted Cumulative Gain"""
    n = len(actual)
    if n == 0:
        return 0.0
    
    # Relevance scores based on actual ranking position
    relevance = {comp: n - i for i, comp in enumerate(actual)}
    
    # DCG for predicted ranking
    dcg = 0.0
    for i, comp in enumerate(predicted):
        rel = relevance.get(comp, 0)
        dcg += rel / math.log2(i + 2)  # +2 because log2(1) = 0
    
    # IDCG (ideal DCG) - actual ranking
    idcg = 0.0
    for i, comp in enumerate(actual):
        rel = relevance.get(comp, 0)
        idcg += rel / math.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def _calculate_mrr(predicted: Sequence[str], actual: Sequence[str]) -> float:
    """Calculate Mean Reciprocal Rank for top actual items in predicted"""
    top_actual = set(actual[:5]) if len(actual) >= 5 else set(actual)
    
    reciprocal_ranks = []
    for comp in top_actual:
        try:
            rank = list(predicted).index(comp) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

@dataclass
class BootstrapCI:
    """Bootstrap confidence interval"""
    estimate: float
    lower: float
    upper: float
    confidence: float
    n_bootstrap: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimate": round(self.estimate, 4),
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "confidence": self.confidence,
            "n_bootstrap": self.n_bootstrap,
        }
    
    @property
    def width(self) -> float:
        """Width of confidence interval"""
        return self.upper - self.lower
    
    def contains(self, value: float) -> bool:
        """Check if value falls within CI"""
        return self.lower <= value <= self.upper


def bootstrap_confidence_interval(
    x: Sequence[float],
    y: Sequence[float],
    metric_func: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> BootstrapCI:
    """
    Calculate bootstrap confidence interval for a metric.
    
    Args:
        x: First sequence
        y: Second sequence
        metric_func: Function that takes (x, y) and returns float
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level (default: 0.95)
        seed: Random seed
    
    Returns:
        BootstrapCI with point estimate and interval
    """
    rng = random.Random(seed)
    n = len(x)
    
    if n < 2:
        return BootstrapCI(
            estimate=0.0, lower=0.0, upper=0.0,
            confidence=confidence, n_bootstrap=0
        )
    
    # Point estimate
    estimate = metric_func(x, y)
    
    # Bootstrap samples
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        x_sample = [x[i] for i in indices]
        y_sample = [y[i] for i in indices]
        bootstrap_estimates.append(metric_func(x_sample, y_sample))
    
    # Calculate percentiles
    bootstrap_estimates.sort()
    alpha = 1 - confidence
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))
    
    return BootstrapCI(
        estimate=estimate,
        lower=bootstrap_estimates[lower_idx],
        upper=bootstrap_estimates[upper_idx],
        confidence=confidence,
        n_bootstrap=n_bootstrap,
    )


# =============================================================================
# Statistical Utilities
# =============================================================================

def percentile(values: Sequence[float], p: float) -> float:
    """
    Calculate percentile using linear interpolation.
    
    Args:
        values: Sequence of values
        p: Percentile (0-100)
    
    Returns:
        Value at percentile
    """
    if not values:
        return 0.0
    
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    
    k = (n - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    
    if f == c:
        return sorted_vals[int(k)]
    
    return sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)


def mean(values: Sequence[float]) -> float:
    """Calculate mean"""
    if not values:
        return 0.0
    return sum(values) / len(values)


def std_dev(values: Sequence[float], sample: bool = True) -> float:
    """
    Calculate standard deviation.
    
    Args:
        values: Sequence of values
        sample: Use sample std (n-1) or population (n)
    
    Returns:
        Standard deviation
    """
    if len(values) < 2:
        return 0.0
    
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values)
    
    if sample:
        variance /= (len(values) - 1)
    else:
        variance /= len(values)
    
    return math.sqrt(variance)


def median(values: Sequence[float]) -> float:
    """Calculate median"""
    return percentile(values, 50)


def iqr(values: Sequence[float]) -> float:
    """Calculate interquartile range (Q3 - Q1)"""
    return percentile(values, 75) - percentile(values, 25)
