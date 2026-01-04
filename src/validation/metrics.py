"""
Validation Metrics

Statistical metrics for validating graph analysis predictions against simulation results.
"""

import math
from dataclasses import dataclass
from typing import List, Sequence, Dict, Any, Optional

@dataclass
class ValidationTargets:
    spearman: float = 0.70
    f1_score: float = 0.90
    top_10_overlap: float = 0.60

@dataclass
class CorrelationMetrics:
    spearman: float
    pearson: float
    kendall: float

@dataclass
class ClassificationMetrics:
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    confusion_matrix: Dict[str, int]

@dataclass
class RankingMetrics:
    top_5_overlap: float
    top_10_overlap: float
    ndcg: float

def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Calculate Spearman rank correlation."""
    if len(x) != len(y) or len(x) < 2: return 0.0
    xr, yr = _rank(x), _rank(y)
    return pearson_correlation(xr, yr)

def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Calculate Pearson correlation."""
    if len(x) != len(y) or len(x) < 2: return 0.0
    mx, my = sum(x)/len(x), sum(y)/len(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(sum((xi - mx)**2 for xi in x) * sum((yi - my)**2 for yi in y))
    return num / den if den != 0 else 0.0

def kendall_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Calculate Kendall's tau."""
    if len(x) != len(y) or len(x) < 2: return 0.0
    n = len(x)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i+1, n):
            tx, ty = x[i] - x[j], y[i] - y[j]
            if tx * ty > 0: concordant += 1
            elif tx * ty < 0: discordant += 1
    return (concordant - discordant) / (n * (n - 1) / 2) if n > 1 else 0.0

def _rank(x: Sequence[float]) -> List[float]:
    """Helper to rank data."""
    pairs = sorted([(v, i) for i, v in enumerate(x)])
    ranks = [0.0] * len(x)
    for r, (_, i) in enumerate(pairs):
        ranks[i] = r + 1
    return ranks

def calculate_classification(pred: Sequence[float], actual: Sequence[float], threshold_pct: float = 75) -> ClassificationMetrics:
    """Calculate classification metrics based on a percentile threshold."""
    if not pred: return ClassificationMetrics(0,0,0,0, {})
    
    # Determine thresholds (Top X percent are "Critical")
    p_sorted, a_sorted = sorted(pred), sorted(actual)
    k = int(len(pred) * (threshold_pct / 100))
    p_thresh = p_sorted[k] if k < len(pred) else p_sorted[-1]
    a_thresh = a_sorted[k] if k < len(actual) else a_sorted[-1]
    
    tp = fp = fn = tn = 0
    for p, a in zip(pred, actual):
        p_crit, a_crit = p >= p_thresh, a >= a_thresh
        if p_crit and a_crit: tp += 1
        elif p_crit and not a_crit: fp += 1
        elif not p_crit and a_crit: fn += 1
        else: tn += 1
        
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / len(pred)
    
    return ClassificationMetrics(prec, rec, f1, acc, {"tp": tp, "fp": fp, "fn": fn, "tn": tn})

def calculate_overlap(pred_ids: List[str], actual_ids: List[str], k: int) -> float:
    """Calculate intersection of top-K items."""
    k = min(k, len(pred_ids))
    if k == 0: return 0.0
    return len(set(pred_ids[:k]) & set(actual_ids[:k])) / k