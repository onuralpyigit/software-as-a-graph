"""
Validation Metric Calculator

Provides pure-function metric computation for the validation step.
"""
import math
import random
from typing import Callable, Sequence, List, Dict, Tuple, Any, Optional

from .models import CorrelationMetrics, ErrorMetrics, ClassificationMetrics, RankingMetrics


def calculate_correlation(
    predicted: Sequence[float],
    actual: Sequence[float],
    n_bootstrap: int = 1000,
    ci_confidence: float = 0.95,
    seed: int = 42,
) -> CorrelationMetrics:
    spearman_rho, spearman_p = spearman_correlation(predicted, actual)
    pearson_r, pearson_p = pearson_correlation(predicted, actual)
    kendall_tau = kendall_correlation(predicted, actual)

    ci_lower, ci_upper = 0.0, 0.0
    if len(predicted) >= 5:
        ci_lower, ci_upper = bootstrap_ci(
            predicted, actual,
            metric_fn=lambda p, a: spearman_correlation(p, a)[0],
            n_bootstrap=n_bootstrap,
            confidence=ci_confidence,
            seed=seed,
        )
    elif len(predicted) >= 3:
        ci_lower, ci_upper = spearman_rho, spearman_rho

    return CorrelationMetrics(
        spearman=spearman_rho,
        spearman_p=spearman_p,
        spearman_ci_lower=ci_lower,
        spearman_ci_upper=ci_upper,
        pearson=pearson_r,
        pearson_p=pearson_p,
        kendall=kendall_tau,
    )


def calculate_error(predicted: Sequence[float], actual: Sequence[float]) -> ErrorMetrics:
    n = len(predicted)
    if n == 0:
        return ErrorMetrics()

    errors = [abs(p - a) for p, a in zip(predicted, actual)]
    squared_errors = [(p - a) ** 2 for p, a in zip(predicted, actual)]

    mse = sum(squared_errors) / n
    mae = sum(errors) / n
    rmse = math.sqrt(mse)
    max_error = max(errors) if errors else 0.0

    actual_list = list(actual)
    actual_range = max(actual_list) - min(actual_list) if len(actual_list) > 1 else 0.0
    nrmse = rmse / actual_range if actual_range > 0 else 0.0

    return ErrorMetrics(rmse=rmse, nrmse=nrmse, mae=mae, mse=mse, max_error=max_error)


def calculate_classification(
    predicted_critical: Sequence[bool],
    actual_critical: Sequence[bool],
    n_bootstrap: int = 1000,
    ci_confidence: float = 0.95,
    seed: int = 42,
) -> ClassificationMetrics:
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

    kappa = cohens_kappa(predicted_critical, actual_critical)

    f1_ci_lower, f1_ci_upper = 0.0, 0.0
    n = len(predicted_critical)
    if n >= 5:
        f1_ci_lower, f1_ci_upper = _bootstrap_classification_ci(
            list(predicted_critical), list(actual_critical),
            n_bootstrap=n_bootstrap,
            confidence=ci_confidence,
            seed=seed,
        )
    elif n >= 3:
        f1_ci_lower, f1_ci_upper = f1, f1

    return ClassificationMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1,
        f1_ci_lower=f1_ci_lower,
        f1_ci_upper=f1_ci_upper,
        accuracy=accuracy,
        cohens_kappa=kappa,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )


def calculate_ranking(
    predicted: Dict[str, float],
    actual: Dict[str, float],
    k_values: Optional[List[int]] = None,
) -> RankingMetrics:
    if k_values is None:
        k_values = [5, 10]

    pred_sorted = sorted(predicted.items(), key=lambda x: x[1], reverse=True)
    actual_sorted = sorted(actual.items(), key=lambda x: x[1], reverse=True)

    pred_ids = [x[0] for x in pred_sorted]
    actual_ids = [x[0] for x in actual_sorted]

    def top_k_overlap(k: int) -> Tuple[float, List[str], List[str], List[str]]:
        pred_top = set(pred_ids[:k])
        actual_top = set(actual_ids[:k])
        common = pred_top & actual_top
        overlap = len(common) / k if k > 0 else 0.0
        return overlap, pred_ids[:k], actual_ids[:k], list(common)

    overlap_5, pred_5, actual_5, common_5 = top_k_overlap(5)
    overlap_10, _, _, _ = top_k_overlap(10)

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


def bootstrap_ci(
    predicted: Sequence[float],
    actual: Sequence[float],
    metric_fn: Callable[[Sequence[float], Sequence[float]], float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(predicted)
    if n < 3:
        return 0.0, 0.0

    pred_list = list(predicted)
    actual_list = list(actual)
    samples: List[float] = []

    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        p_sample = [pred_list[i] for i in indices]
        a_sample = [actual_list[i] for i in indices]
        try:
            val = metric_fn(p_sample, a_sample)
            samples.append(val)
        except (ZeroDivisionError, ValueError):
            continue

    if not samples:
        return 0.0, 0.0

    samples.sort()
    alpha = (1 - confidence) / 2
    lo_idx = max(0, int(alpha * len(samples)))
    hi_idx = min(len(samples) - 1, int((1 - alpha) * len(samples)))
    return samples[lo_idx], samples[hi_idx]


def cohens_kappa(
    predicted_critical: Sequence[bool],
    actual_critical: Sequence[bool],
) -> float:
    n = len(predicted_critical)
    if n == 0:
        return 0.0

    tp = sum(1 for p, a in zip(predicted_critical, actual_critical) if p and a)
    fp = sum(1 for p, a in zip(predicted_critical, actual_critical) if p and not a)
    fn = sum(1 for p, a in zip(predicted_critical, actual_critical) if not p and a)
    tn = sum(1 for p, a in zip(predicted_critical, actual_critical) if not p and not a)

    p_o = (tp + tn) / n
    p_e = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)

    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def spearman_correlation(predicted: Sequence[float], actual: Sequence[float]) -> Tuple[float, float]:
    n = len(predicted)
    if n < 3:
        return 0.0, 1.0
    rank_pred = _get_ranks(predicted)
    rank_actual = _get_ranks(actual)
    return pearson_correlation(rank_pred, rank_actual)


def pearson_correlation(predicted: Sequence[float], actual: Sequence[float]) -> Tuple[float, float]:
    n = len(predicted)
    if n < 3:
        return 0.0, 1.0
    mean_p = sum(predicted) / n
    mean_a = sum(actual) / n
    num = sum((p - mean_p) * (a - mean_a) for p, a in zip(predicted, actual))
    den = math.sqrt(
        sum((p - mean_p) ** 2 for p in predicted)
        * sum((a - mean_a) ** 2 for a in actual)
    )
    if den == 0:
        return 0.0, 1.0
    r = num / den
    r = max(-1.0, min(1.0, r))
    if abs(r) >= 1.0:
        return r, 0.0
    t = r * math.sqrt((n - 2) / (1 - r ** 2))
    p = 2 * (1 - _t_cdf(abs(t), n - 2))
    return r, p


def kendall_correlation(predicted: Sequence[float], actual: Sequence[float]) -> float:
    n = len(predicted)
    if n < 2:
        return 0.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign_p = predicted[i] - predicted[j]
            sign_a = actual[i] - actual[j]
            if sign_p * sign_a > 0:
                concordant += 1
            elif sign_p * sign_a < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else 0.0


def _get_ranks(values: Sequence[float]) -> List[float]:
    n = len(values)
    indexed = [(v, i) for i, v in enumerate(values)]
    indexed.sort(key=lambda x: x[0], reverse=True)
    rank = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j][0] == indexed[j + 1][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rank[indexed[k][1]] = avg_rank
        i = j + 1
    return rank


def _t_cdf(t: float, df: int) -> float:
    if df > 30:
        return 0.5 * (1 + math.erf(t / math.sqrt(2)))
    x = df / (df + t ** 2)
    return 1 - 0.5 * _incomplete_beta(df / 2, 0.5, x)


def _incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    steps = 100
    dx = x / steps
    res = 0.0
    for i in range(steps):
        xi = (i + 0.5) * dx
        res += (xi ** (a - 1)) * ((1 - xi) ** (b - 1)) * dx
    beta_ab = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
    return res / beta_ab


def _calculate_ndcg(ranked_ids: List[str], relevance: Dict[str, float], k: int) -> float:
    if not ranked_ids or not relevance:
        return 0.0
    dcg = 0.0
    for i, cid in enumerate(ranked_ids[:k]):
        rel = relevance.get(cid, 0.0)
        dcg += rel / math.log2(i + 2)
    ideal_sorted = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_sorted))
    return dcg / idcg if idcg > 0 else 0.0


def _bootstrap_classification_ci(
    pred_crit: List[bool],
    actual_crit: List[bool],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(pred_crit)
    if n < 3:
        return 0.0, 0.0

    samples: List[float] = []
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        p_sample = [pred_crit[i] for i in indices]
        a_sample = [actual_crit[i] for i in indices]
        tp = sum(1 for p, a in zip(p_sample, a_sample) if p and a)
        fp = sum(1 for p, a in zip(p_sample, a_sample) if p and not a)
        fn = sum(1 for p, a in zip(p_sample, a_sample) if not p and a)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        samples.append(f1)

    samples.sort()
    alpha = (1 - confidence) / 2
    lo_idx = max(0, int(alpha * len(samples)))
    hi_idx = min(len(samples) - 1, int((1 - alpha) * len(samples)))
    return samples[lo_idx], samples[hi_idx]
