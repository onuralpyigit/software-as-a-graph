"""
Validation Metrics - Version 5.0

Statistical metrics for validating graph analysis predictions
against simulation results.

Metrics:
- Correlation: Spearman, Pearson, Kendall
- Classification: F1-Score, Precision, Recall
- Ranking: Top-K Overlap, NDCG

Research Targets:
- Spearman ρ ≥ 0.70 (rank correlation)
- F1-Score ≥ 0.90 (classification)
- Precision ≥ 0.80
- Recall ≥ 0.80
- Top-5 Overlap ≥ 60%

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Sequence, Callable


# =============================================================================
# Enums
# =============================================================================

class ValidationStatus(Enum):
    """Overall validation status."""
    PASSED = "passed"
    PARTIAL = "partial"
    FAILED = "failed"
    ERROR = "error"


# =============================================================================
# Validation Targets
# =============================================================================

@dataclass
class ValidationTargets:
    """Target thresholds for validation metrics."""
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
# Statistical Utilities
# =============================================================================

def mean(values: Sequence[float]) -> float:
    """Calculate arithmetic mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def std_dev(values: Sequence[float]) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def median(values: Sequence[float]) -> float:
    """Calculate median."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    return sorted_vals[mid]


def percentile(values: Sequence[float], p: float) -> float:
    """Calculate percentile (0-100)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)


def iqr(values: Sequence[float]) -> float:
    """Calculate interquartile range."""
    if len(values) < 4:
        return 0.0
    return percentile(values, 75) - percentile(values, 25)


# =============================================================================
# Correlation Metrics
# =============================================================================

def _rank(values: Sequence[float]) -> List[float]:
    """Assign ranks to values (handles ties with average)."""
    n = len(values)
    indexed = [(v, i) for i, v in enumerate(values)]
    sorted_indexed = sorted(indexed, key=lambda x: x[0])
    
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        # Find tied values
        while j < n - 1 and sorted_indexed[j][0] == sorted_indexed[j + 1][0]:
            j += 1
        # Assign average rank
        avg_rank = (i + j + 2) / 2  # 1-indexed
        for k in range(i, j + 1):
            ranks[sorted_indexed[k][1]] = avg_rank
        i = j + 1
    
    return ranks


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Calculate Spearman rank correlation coefficient.
    
    Measures monotonic relationship between predicted and actual.
    
    Returns:
        Spearman ρ in [-1, 1]. Higher is better.
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    ranks_x = _rank(list(x))
    ranks_y = _rank(list(y))
    
    # Pearson on ranks
    return pearson_correlation(ranks_x, ranks_y)


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Calculate Pearson linear correlation coefficient.
    
    Returns:
        Pearson r in [-1, 1].
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    mean_x = mean(x)
    mean_y = mean(y)
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    
    ss_x = sum((xi - mean_x) ** 2 for xi in x)
    ss_y = sum((yi - mean_y) ** 2 for yi in y)
    
    denominator = math.sqrt(ss_x * ss_y)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def kendall_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Calculate Kendall's tau rank correlation.
    
    Returns:
        Kendall τ in [-1, 1].
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    concordant = 0
    discordant = 0
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            xi, xj = x[i], x[j]
            yi, yj = y[i], y[j]
            
            if (xi < xj and yi < yj) or (xi > xj and yi > yj):
                concordant += 1
            elif (xi < xj and yi > yj) or (xi > xj and yi < yj):
                discordant += 1
            # Ties are ignored
    
    total = concordant + discordant
    if total == 0:
        return 0.0
    
    return (concordant - discordant) / total


@dataclass
class CorrelationMetrics:
    """Correlation metrics between predicted and actual."""
    spearman: float = 0.0
    pearson: float = 0.0
    kendall: float = 0.0
    n_samples: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spearman": round(self.spearman, 4),
            "pearson": round(self.pearson, 4),
            "kendall": round(self.kendall, 4),
            "n_samples": self.n_samples,
        }


def calculate_correlation(
    predicted: Sequence[float],
    actual: Sequence[float],
) -> CorrelationMetrics:
    """Calculate all correlation metrics."""
    return CorrelationMetrics(
        spearman=spearman_correlation(predicted, actual),
        pearson=pearson_correlation(predicted, actual),
        kendall=kendall_correlation(predicted, actual),
        n_samples=len(predicted),
    )


# =============================================================================
# Classification Metrics
# =============================================================================

@dataclass
class ConfusionMatrix:
    """Binary classification confusion matrix."""
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
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
    
    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / self.total
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.true_positives,
            "tn": self.true_negatives,
            "fp": self.false_positives,
            "fn": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "accuracy": round(self.accuracy, 4),
        }


@dataclass
class ClassificationMetrics:
    """Classification metrics."""
    confusion_matrix: ConfusionMatrix
    threshold: float
    
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
            "threshold": round(self.threshold, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "accuracy": round(self.accuracy, 4),
        }


def calculate_confusion_matrix(
    predicted: Sequence[float],
    actual: Sequence[float],
    threshold: Optional[float] = None,
) -> ConfusionMatrix:
    """
    Calculate confusion matrix using threshold.
    
    Args:
        predicted: Predicted scores
        actual: Actual scores
        threshold: Classification threshold (default: 75th percentile of actual)
    
    Returns:
        ConfusionMatrix
    """
    if len(predicted) != len(actual) or len(predicted) == 0:
        return ConfusionMatrix()
    
    # Default threshold: 75th percentile of actual values
    if threshold is None:
        threshold = percentile(list(actual), 75)
    
    tp = tn = fp = fn = 0
    
    for pred, act in zip(predicted, actual):
        pred_critical = pred >= threshold
        act_critical = act >= threshold
        
        if pred_critical and act_critical:
            tp += 1
        elif not pred_critical and not act_critical:
            tn += 1
        elif pred_critical and not act_critical:
            fp += 1
        else:
            fn += 1
    
    return ConfusionMatrix(
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
    )


def calculate_classification(
    predicted: Sequence[float],
    actual: Sequence[float],
    threshold: Optional[float] = None,
) -> ClassificationMetrics:
    """Calculate classification metrics."""
    if threshold is None:
        threshold = percentile(list(actual), 75)
    
    cm = calculate_confusion_matrix(predicted, actual, threshold)
    
    return ClassificationMetrics(
        confusion_matrix=cm,
        threshold=threshold,
    )


# =============================================================================
# Ranking Metrics
# =============================================================================

@dataclass
class RankingMetrics:
    """Ranking comparison metrics."""
    top_3_overlap: float = 0.0
    top_5_overlap: float = 0.0
    top_10_overlap: float = 0.0
    ndcg: float = 0.0
    mrr: float = 0.0
    rank_difference_mean: float = 0.0
    rank_difference_std: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_3_overlap": round(self.top_3_overlap, 4),
            "top_5_overlap": round(self.top_5_overlap, 4),
            "top_10_overlap": round(self.top_10_overlap, 4),
            "ndcg": round(self.ndcg, 4),
            "mrr": round(self.mrr, 4),
            "rank_difference_mean": round(self.rank_difference_mean, 2),
            "rank_difference_std": round(self.rank_difference_std, 2),
        }


def _top_k_overlap(
    predicted_ids: Sequence[str],
    actual_ids: Sequence[str],
    k: int,
) -> float:
    """Calculate overlap of top-k items."""
    if len(predicted_ids) < k or len(actual_ids) < k:
        k = min(len(predicted_ids), len(actual_ids))
    if k == 0:
        return 0.0
    
    pred_top = set(predicted_ids[:k])
    actual_top = set(actual_ids[:k])
    
    return len(pred_top & actual_top) / k


def _ndcg(
    predicted_ids: Sequence[str],
    actual_scores: Dict[str, float],
    k: Optional[int] = None,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain."""
    if not predicted_ids or not actual_scores:
        return 0.0
    
    if k is None:
        k = len(predicted_ids)
    k = min(k, len(predicted_ids))
    
    # DCG for predicted ranking
    dcg = 0.0
    for i, cid in enumerate(predicted_ids[:k]):
        rel = actual_scores.get(cid, 0.0)
        dcg += rel / math.log2(i + 2)  # i+2 because position is 1-indexed
    
    # Ideal DCG (sorted by actual scores)
    ideal_order = sorted(actual_scores.keys(), key=lambda x: -actual_scores[x])
    idcg = 0.0
    for i, cid in enumerate(ideal_order[:k]):
        rel = actual_scores[cid]
        idcg += rel / math.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def _mrr(
    predicted_ids: Sequence[str],
    actual_ids: Sequence[str],
    k: int = 10,
) -> float:
    """Calculate Mean Reciprocal Rank."""
    if not predicted_ids or not actual_ids:
        return 0.0
    
    # Consider top-k actual as relevant
    relevant = set(actual_ids[:k])
    
    for i, cid in enumerate(predicted_ids):
        if cid in relevant:
            return 1.0 / (i + 1)
    
    return 0.0


def calculate_ranking(
    predicted_scores: Dict[str, float],
    actual_scores: Dict[str, float],
) -> RankingMetrics:
    """Calculate ranking comparison metrics."""
    if not predicted_scores or not actual_scores:
        return RankingMetrics()
    
    # Get common IDs
    common_ids = set(predicted_scores.keys()) & set(actual_scores.keys())
    if len(common_ids) < 3:
        return RankingMetrics()
    
    # Sort by scores
    pred_ranked = sorted(common_ids, key=lambda x: -predicted_scores[x])
    actual_ranked = sorted(common_ids, key=lambda x: -actual_scores[x])
    
    # Calculate overlaps
    top_3 = _top_k_overlap(pred_ranked, actual_ranked, 3)
    top_5 = _top_k_overlap(pred_ranked, actual_ranked, 5)
    top_10 = _top_k_overlap(pred_ranked, actual_ranked, 10)
    
    # NDCG
    ndcg = _ndcg(pred_ranked, {k: actual_scores[k] for k in common_ids})
    
    # MRR
    mrr = _mrr(pred_ranked, actual_ranked)
    
    # Rank differences
    pred_ranks = {cid: i + 1 for i, cid in enumerate(pred_ranked)}
    actual_ranks = {cid: i + 1 for i, cid in enumerate(actual_ranked)}
    
    diffs = [pred_ranks[cid] - actual_ranks[cid] for cid in common_ids]
    
    return RankingMetrics(
        top_3_overlap=top_3,
        top_5_overlap=top_5,
        top_10_overlap=top_10,
        ndcg=ndcg,
        mrr=mrr,
        rank_difference_mean=mean([abs(d) for d in diffs]),
        rank_difference_std=std_dev(diffs) if len(diffs) > 1 else 0.0,
    )


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

@dataclass
class BootstrapCI:
    """Bootstrap confidence interval."""
    lower: float
    upper: float
    point_estimate: float
    confidence: float = 0.95
    n_bootstrap: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "point_estimate": round(self.point_estimate, 4),
            "confidence": self.confidence,
            "n_bootstrap": self.n_bootstrap,
        }


def bootstrap_confidence_interval(
    x: Sequence[float],
    y: Sequence[float],
    statistic_fn: Callable[[Sequence[float], Sequence[float]], float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> BootstrapCI:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Args:
        x: First sequence
        y: Second sequence
        statistic_fn: Function to compute statistic
        confidence: Confidence level (default: 0.95)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed
    
    Returns:
        BootstrapCI with lower/upper bounds
    """
    if len(x) != len(y) or len(x) < 5:
        point = statistic_fn(x, y)
        return BootstrapCI(
            lower=point,
            upper=point,
            point_estimate=point,
            confidence=confidence,
            n_bootstrap=0,
        )
    
    rng = random.Random(seed)
    n = len(x)
    
    # Point estimate
    point_estimate = statistic_fn(x, y)
    
    # Bootstrap samples
    statistics = []
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        x_sample = [x[i] for i in indices]
        y_sample = [y[i] for i in indices]
        statistics.append(statistic_fn(x_sample, y_sample))
    
    # Confidence interval
    alpha = 1 - confidence
    lower = percentile(statistics, alpha / 2 * 100)
    upper = percentile(statistics, (1 - alpha / 2) * 100)
    
    return BootstrapCI(
        lower=lower,
        upper=upper,
        point_estimate=point_estimate,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
    )
