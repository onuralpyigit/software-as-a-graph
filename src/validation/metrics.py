"""
Statistical Metrics for Validation - Version 4.0

Pure Python implementations of statistical functions for validating
graph-based analysis predictions against simulation results.

Includes:
- Correlation coefficients (Spearman, Pearson, Kendall)
- Classification metrics (F1, precision, recall)
- Ranking metrics (top-k overlap, NDCG)
- Bootstrap confidence intervals

Author: Software-as-a-Graph Research Project
Version: 4.0
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class ValidationStatus(Enum):
    """Overall validation result status"""
    PASSED = "passed"           # All targets met
    PARTIAL = "partial"         # Some targets met
    FAILED = "failed"           # Most targets not met
    INSUFFICIENT = "insufficient"  # Not enough data


class MetricStatus(Enum):
    """Individual metric status"""
    MET = "met"
    NOT_MET = "not_met"
    BORDERLINE = "borderline"   # Within 5% of target


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CorrelationMetrics:
    """Correlation analysis results"""
    spearman: float
    spearman_p: float
    pearson: float
    pearson_p: float
    kendall: float
    n: int

    def to_dict(self) -> Dict:
        return {
            "spearman": {
                "coefficient": round(self.spearman, 4),
                "p_value": round(self.spearman_p, 6),
                "significant": self.spearman_p < 0.05,
            },
            "pearson": {
                "coefficient": round(self.pearson, 4),
                "p_value": round(self.pearson_p, 6),
                "significant": self.pearson_p < 0.05,
            },
            "kendall": round(self.kendall, 4),
            "sample_size": self.n,
        }


@dataclass
class ConfusionMatrix:
    """Binary classification confusion matrix"""
    tp: int  # True positives
    tn: int  # True negatives
    fp: int  # False positives
    fn: int  # False negatives
    threshold: float

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d > 0 else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    @property
    def specificity(self) -> float:
        d = self.tn + self.fp
        return self.tn / d if d > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "matrix": {
                "tp": self.tp,
                "tn": self.tn,
                "fp": self.fp,
                "fn": self.fn,
            },
            "threshold": round(self.threshold, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "specificity": round(self.specificity, 4),
        }


@dataclass
class RankingMetrics:
    """Ranking analysis metrics"""
    top_k_overlap: Dict[int, float]  # k -> overlap fraction
    ndcg: float  # Normalized Discounted Cumulative Gain
    mrr: float   # Mean Reciprocal Rank
    rank_correlation: float

    def to_dict(self) -> Dict:
        return {
            "top_k_overlap": {
                str(k): round(v, 4) for k, v in self.top_k_overlap.items()
            },
            "ndcg": round(self.ndcg, 4),
            "mrr": round(self.mrr, 4),
            "rank_correlation": round(self.rank_correlation, 4),
        }


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval"""
    metric: str
    estimate: float
    ci_lower: float
    ci_upper: float
    confidence: float
    std_error: float
    n_iterations: int

    def to_dict(self) -> Dict:
        return {
            "metric": self.metric,
            "estimate": round(self.estimate, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "confidence": self.confidence,
            "std_error": round(self.std_error, 4),
            "iterations": self.n_iterations,
        }


@dataclass
class ValidationTargets:
    """Target metrics for validation"""
    spearman: float = 0.70
    f1: float = 0.90
    precision: float = 0.80
    recall: float = 0.80
    top_5: float = 0.60
    top_10: float = 0.70

    def to_dict(self) -> Dict:
        return {
            "spearman": self.spearman,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "top_5_overlap": self.top_5,
            "top_10_overlap": self.top_10,
        }


# =============================================================================
# Statistical Functions
# =============================================================================

def _rank_data(data: List[float]) -> List[float]:
    """Rank data with average rank for ties"""
    indexed = [(val, i) for i, val in enumerate(data)]
    indexed.sort(key=lambda x: x[0])
    
    ranks = [0.0] * len(data)
    i = 0
    while i < len(indexed):
        j = i
        # Find all items with same value (ties)
        while j < len(indexed) and indexed[j][0] == indexed[i][0]:
            j += 1
        # Assign average rank to ties
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k][1]] = avg_rank
        i = j
    
    return ranks


def spearman(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Calculate Spearman rank correlation coefficient.
    
    Returns (coefficient, p-value)
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    
    rank_x = _rank_data(x)
    rank_y = _rank_data(y)
    
    return pearson(rank_x, rank_y)


def pearson(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Calculate Pearson correlation coefficient.
    
    Returns (coefficient, p-value)
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    
    # Means
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # Compute correlation
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denom_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    denom_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    
    if denom_x == 0 or denom_y == 0:
        return 0.0, 1.0
    
    r = num / (denom_x * denom_y)
    r = max(-1.0, min(1.0, r))  # Clamp to [-1, 1]
    
    # Calculate p-value using t-distribution approximation
    if abs(r) == 1.0:
        p = 0.0
    else:
        t = r * math.sqrt((n - 2) / (1 - r * r))
        # Approximate p-value using normal distribution for large n
        p = 2 * (1 - _normal_cdf(abs(t) / math.sqrt(n / 2)))
    
    return r, p


def kendall(x: List[float], y: List[float]) -> float:
    """
    Calculate Kendall's tau-b rank correlation.
    """
    n = len(x)
    if n < 2:
        return 0.0
    
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            
            if dx == 0 and dy == 0:
                ties_x += 1
                ties_y += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
                concordant += 1
            else:
                discordant += 1
    
    n_pairs = n * (n - 1) / 2
    denom = math.sqrt((n_pairs - ties_x) * (n_pairs - ties_y))
    
    if denom == 0:
        return 0.0
    
    return (concordant - discordant) / denom


def _normal_cdf(x: float) -> float:
    """Approximate normal CDF using error function approximation"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile (0-100)"""
    if not data:
        return 0.0
    
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    
    if f == c:
        return sorted_data[int(k)]
    
    return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)


def std_dev(data: List[float]) -> float:
    """Calculate standard deviation"""
    if len(data) < 2:
        return 0.0
    
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)


# =============================================================================
# Metric Calculation Functions
# =============================================================================

def calculate_correlation(
    predicted: List[float],
    actual: List[float],
) -> CorrelationMetrics:
    """Calculate all correlation metrics"""
    s_r, s_p = spearman(predicted, actual)
    p_r, p_p = pearson(predicted, actual)
    k_t = kendall(predicted, actual)
    
    return CorrelationMetrics(
        spearman=s_r,
        spearman_p=s_p,
        pearson=p_r,
        pearson_p=p_p,
        kendall=k_t,
        n=len(predicted),
    )


def calculate_confusion(
    predicted: Dict[str, float],
    actual: Dict[str, float],
    pred_threshold: float,
    actual_threshold: float,
) -> Tuple[ConfusionMatrix, List[str], List[str]]:
    """
    Calculate confusion matrix and identify misclassified components.
    
    Returns (ConfusionMatrix, false_positives, false_negatives)
    """
    common = set(predicted.keys()) & set(actual.keys())
    
    tp = tn = fp = fn = 0
    fp_list = []
    fn_list = []
    
    for comp in common:
        pred_critical = predicted[comp] >= pred_threshold
        actual_critical = actual[comp] >= actual_threshold
        
        if pred_critical and actual_critical:
            tp += 1
        elif not pred_critical and not actual_critical:
            tn += 1
        elif pred_critical and not actual_critical:
            fp += 1
            fp_list.append(comp)
        else:
            fn += 1
            fn_list.append(comp)
    
    return (
        ConfusionMatrix(tp=tp, tn=tn, fp=fp, fn=fn, threshold=pred_threshold),
        fp_list,
        fn_list,
    )


def calculate_ranking(
    predicted: Dict[str, float],
    actual: Dict[str, float],
    k_values: List[int] = None,
) -> RankingMetrics:
    """Calculate ranking metrics"""
    if k_values is None:
        k_values = [3, 5, 10, 20]
    
    common = list(set(predicted.keys()) & set(actual.keys()))
    n = len(common)
    
    if n == 0:
        return RankingMetrics(
            top_k_overlap={k: 0.0 for k in k_values},
            ndcg=0.0,
            mrr=0.0,
            rank_correlation=0.0,
        )
    
    # Sort by score (descending)
    pred_ranked = sorted(common, key=lambda c: -predicted[c])
    actual_ranked = sorted(common, key=lambda c: -actual[c])
    
    # Top-k overlap
    top_k_overlap = {}
    for k in k_values:
        k_actual = min(k, n)
        pred_top_k = set(pred_ranked[:k_actual])
        actual_top_k = set(actual_ranked[:k_actual])
        overlap = len(pred_top_k & actual_top_k) / k_actual
        top_k_overlap[k] = overlap
    
    # NDCG (Normalized Discounted Cumulative Gain)
    # Using actual impact as relevance score
    dcg = 0.0
    idcg = 0.0
    
    for i, comp in enumerate(pred_ranked):
        rel = actual[comp]
        dcg += rel / math.log2(i + 2)  # i + 2 because log2(1) = 0
    
    for i, comp in enumerate(actual_ranked):
        rel = actual[comp]
        idcg += rel / math.log2(i + 2)
    
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    # MRR (Mean Reciprocal Rank)
    # Find first correctly identified critical component
    actual_top_10pct = set(actual_ranked[:max(1, n // 10)])
    mrr = 0.0
    for i, comp in enumerate(pred_ranked):
        if comp in actual_top_10pct:
            mrr = 1.0 / (i + 1)
            break
    
    # Rank correlation
    pred_ranks = {c: i + 1 for i, c in enumerate(pred_ranked)}
    actual_ranks = {c: i + 1 for i, c in enumerate(actual_ranked)}
    
    x = [pred_ranks[c] for c in common]
    y = [actual_ranks[c] for c in common]
    rank_corr, _ = spearman(x, y)
    
    return RankingMetrics(
        top_k_overlap=top_k_overlap,
        ndcg=ndcg,
        mrr=mrr,
        rank_correlation=rank_corr,
    )


def bootstrap_confidence_interval(
    predicted: List[float],
    actual: List[float],
    metric_fn,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> BootstrapCI:
    """
    Calculate bootstrap confidence interval for a metric.
    
    Args:
        predicted: Predicted values
        actual: Actual values
        metric_fn: Function that takes (predicted, actual) and returns float
        n_iterations: Number of bootstrap iterations
        confidence: Confidence level
        seed: Random seed
    
    Returns:
        BootstrapCI with confidence interval
    """
    rng = random.Random(seed)
    n = len(predicted)
    
    if n < 3:
        point = metric_fn(predicted, actual)
        return BootstrapCI(
            metric="metric",
            estimate=point,
            ci_lower=point,
            ci_upper=point,
            confidence=confidence,
            std_error=0.0,
            n_iterations=0,
        )
    
    # Point estimate
    point_estimate = metric_fn(predicted, actual)
    
    # Bootstrap samples
    samples = []
    for _ in range(n_iterations):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        pred_sample = [predicted[i] for i in indices]
        actual_sample = [actual[i] for i in indices]
        samples.append(metric_fn(pred_sample, actual_sample))
    
    # Calculate confidence interval
    samples.sort()
    alpha = (1 - confidence) / 2
    lower_idx = int(n_iterations * alpha)
    upper_idx = int(n_iterations * (1 - alpha))
    
    return BootstrapCI(
        metric="metric",
        estimate=point_estimate,
        ci_lower=samples[lower_idx],
        ci_upper=samples[min(upper_idx, n_iterations - 1)],
        confidence=confidence,
        std_error=std_dev(samples),
        n_iterations=n_iterations,
    )
