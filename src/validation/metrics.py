"""
Validation Metrics

Statistical metrics for comparing Predicted Importance (Analysis) 
vs Actual Impact (Simulation).
"""

import math
from dataclasses import dataclass
from typing import Sequence, Dict, List, Any

@dataclass
class ValidationTargets:
    """Targets defined in the research methodology."""
    spearman: float = 0.70       # Rank correlation
    f1_score: float = 0.80       # Classification accuracy
    precision: float = 0.80      
    recall: float = 0.80         
    top_5_overlap: float = 0.60  # Identification of most critical
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
    """Helper to rank data handling ties."""
    pairs = sorted([(v, i) for i, v in enumerate(x)])
    ranks = [0.0] * len(x)
    
    i = 0
    while i < len(x):
        j = i
        while j < len(x) - 1 and pairs[j][0] == pairs[j+1][0]:
            j += 1
        
        r = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[pairs[k][1]] = r
        i = j + 1
    return ranks

def calculate_classification_metrics(pred_critical: Sequence[bool], actual_critical: Sequence[bool]) -> ClassificationMetrics:
    """Calculate Precision, Recall, F1 based on binary 'Critical' classification."""
    if len(pred_critical) != len(actual_critical) or not pred_critical:
        return ClassificationMetrics(0, 0, 0, 0, {})

    tp = fp = fn = tn = 0
    for p, a in zip(pred_critical, actual_critical):
        if p and a: tp += 1
        elif p and not a: fp += 1
        elif not p and a: fn += 1
        else: tn += 1
        
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / len(pred_critical)
    
    return ClassificationMetrics(prec, rec, f1, acc, {"tp": tp, "fp": fp, "fn": fn, "tn": tn})

def calculate_ranking_metrics(pred_map: Dict[str, float], actual_map: Dict[str, float]) -> RankingMetrics:
    """Calculate overlap and NDCG."""
    ids = list(pred_map.keys())
    if not ids: 
        return RankingMetrics(0, 0, 0)
        
    p_ranked = sorted(ids, key=lambda x: pred_map[x], reverse=True)
    a_ranked = sorted(ids, key=lambda x: actual_map[x], reverse=True)
    
    def overlap(k):
        k = min(k, len(ids))
        if k == 0: return 0.0
        return len(set(p_ranked[:k]) & set(a_ranked[:k])) / k

    def dcg(ranked_ids, rel_map, k):
        score = 0.0
        for i, uid in enumerate(ranked_ids[:k]):
            rel = rel_map.get(uid, 0.0)
            score += rel / math.log2(i + 2)
        return score
    
    k_ndcg = 10
    actual_dcg = dcg(a_ranked, actual_map, k_ndcg)
    pred_dcg = dcg(p_ranked, actual_map, k_ndcg)
    ndcg = pred_dcg / actual_dcg if actual_dcg > 0 else 0.0

    return RankingMetrics(overlap(5), overlap(10), ndcg)