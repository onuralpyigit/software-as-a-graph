"""
Validation Metrics

Statistical metrics for comparing Predicted Importance (Analysis) 
vs Actual Impact (Simulation).
"""

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Dict, Any, Optional, Set

@dataclass
class ValidationTargets:
    spearman: float = 0.60
    f1_score: float = 0.70
    top_10_overlap: float = 0.50

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
    ndcg_10: float

def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Calculate Spearman rank correlation."""
    if len(x) != len(y) or len(x) < 2: return 0.0
    xr, yr = _rank(x), _rank(y)
    return pearson_correlation(xr, yr)

def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Calculate Pearson correlation."""
    if len(x) != len(y) or len(x) < 2: return 0.0
    mx, my = sum(x)/len(x), sum(y)/len(y)
    
    # Variance check
    vx = sum((xi - mx)**2 for xi in x)
    vy = sum((yi - my)**2 for yi in y)
    
    if vx == 0 or vy == 0: return 0.0
    
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(vx * vy)
    return num / den

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
    
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom > 0 else 0.0

def _rank(x: Sequence[float]) -> List[float]:
    """Helper to rank data (handles ties by averaging)."""
    # pair (value, original_index)
    pairs = sorted([(v, i) for i, v in enumerate(x)])
    ranks = [0.0] * len(x)
    
    i = 0
    while i < len(x):
        j = i
        # find end of identical values
        while j < len(x) - 1 and pairs[j][0] == pairs[j+1][0]:
            j += 1
        
        # average rank
        r = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[pairs[k][1]] = r
        i = j + 1
        
    return ranks

def calculate_classification(pred: Sequence[float], actual: Sequence[float], top_percentile: float = 0.25) -> ClassificationMetrics:
    """
    Classify 'Critical' items based on being in the top X percentile.
    """
    if not pred or len(pred) != len(actual): 
        return ClassificationMetrics(0,0,0,0, {})

    n = len(pred)
    # Determine count for top percentile
    k = max(1, int(n * top_percentile))
    
    # Thresholds are the value of the k-th highest item
    p_sorted = sorted(pred, reverse=True)
    a_sorted = sorted(actual, reverse=True)
    p_thresh = p_sorted[k-1] if k <= len(p_sorted) else 0
    a_thresh = a_sorted[k-1] if k <= len(a_sorted) else 0
    
    tp = fp = fn = tn = 0
    for p, a in zip(pred, actual):
        p_crit = p >= p_thresh and p > 0 # Must be > 0 to be critical
        a_crit = a >= a_thresh and a > 0
        
        if p_crit and a_crit: tp += 1
        elif p_crit and not a_crit: fp += 1
        elif not p_crit and a_crit: fn += 1
        else: tn += 1
        
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / n
    
    return ClassificationMetrics(prec, rec, f1, acc, {"tp": tp, "fp": fp, "fn": fn, "tn": tn})

def calculate_ranking_metrics(pred_map: Dict[str, float], actual_map: Dict[str, float]) -> RankingMetrics:
    """Calculate overlap and NDCG."""
    ids = list(pred_map.keys())
    if not ids: 
        return RankingMetrics(0,0,0)
        
    # Sorted Lists of IDs
    p_ranked = sorted(ids, key=lambda x: pred_map[x], reverse=True)
    a_ranked = sorted(ids, key=lambda x: actual_map[x], reverse=True)
    
    def overlap(k):
        k = min(k, len(ids))
        if k == 0: return 0.0
        return len(set(p_ranked[:k]) & set(a_ranked[:k])) / k

    # NDCG Calculation
    def dcg(ranked_ids, rel_map, k):
        score = 0.0
        for i, uid in enumerate(ranked_ids[:k]):
            rel = rel_map.get(uid, 0.0)
            score += rel / math.log2(i + 2)
        return score
    
    k_ndcg = 10
    actual_dcg = dcg(a_ranked, actual_map, k_ndcg)
    pred_dcg = dcg(p_ranked, actual_map, k_ndcg) # Relevance is always actual score
    ndcg = pred_dcg / actual_dcg if actual_dcg > 0 else 0.0

    return RankingMetrics(overlap(5), overlap(10), ndcg)