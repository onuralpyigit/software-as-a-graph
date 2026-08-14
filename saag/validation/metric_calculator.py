"""
Validation Metric Calculator

Provides pure-function metric computation for the validation step.
"""
import math
import random
import warnings
from typing import Callable, Sequence, List, Dict, Set, Tuple, Optional

import numpy as np
from scipy import stats

from .models import CorrelationMetrics, ErrorMetrics, ClassificationMetrics, RankingMetrics


def _correlate(fn: Callable, predicted: Sequence[float], actual: Sequence[float]) -> Tuple[float, float]:
    """Run a scipy correlation, mapping degenerate input to this module's (0.0, 1.0) contract.

    scipy returns NaN for constant input; callers here (notably bootstrap resampling)
    rely on a numeric 0.0 instead.
    """
    if len(predicted) < 3 or len(actual) < 3:
        return 0.0, 1.0
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        # Constant input is a normal case here (e.g. every Topic scoring 0.0);
        # it is reported as rho = 0.0 rather than warned about.
        warnings.simplefilter("ignore")
        result = fn(predicted, actual)
    rho, p = float(result[0]), float(result[1])
    if math.isnan(rho):
        return 0.0, 1.0
    return rho, (1.0 if math.isnan(p) else p)


def _top_k_overlap(
    predicted: Dict[str, float],
    actual: Dict[str, float],
    k: int,
) -> Tuple[float, Set[str], Set[str]]:
    """Fraction of the top-K set shared by two scorings, over their common ids.

    Single definition behind CCR@K, COCR@K and AHCR@K. ``k`` is clamped to the
    number of common ids, so the rate stays in [0, 1] for small systems.
    """
    common = set(predicted) & set(actual)
    if not common or k <= 0:
        return 0.0, set(), set()
    effective_k = min(k, len(common))
    pred_top = set(sorted(common, key=lambda c: predicted[c], reverse=True)[:effective_k])
    actual_top = set(sorted(common, key=lambda c: actual[c], reverse=True)[:effective_k])
    return len(pred_top & actual_top) / effective_k, pred_top, actual_top


def _bootstrap_percentile_ci(
    statistic: Callable[[List[int]], float],
    n: int,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Shared resample-sort-percentile core for every bootstrap CI in this module.

    ``statistic`` receives a list of resampled indices and returns a float, or
    raises ZeroDivisionError/ValueError to skip that resample.
    """
    rng = random.Random(seed)
    samples: List[float] = []
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        try:
            samples.append(statistic(indices))
        except (ZeroDivisionError, ValueError):
            continue

    if not samples:
        return 0.0, 0.0

    samples.sort()
    alpha = (1 - confidence) / 2
    lo_idx = max(0, int(alpha * len(samples)))
    hi_idx = min(len(samples) - 1, int((1 - alpha) * len(samples)))
    return samples[lo_idx], samples[hi_idx]


def calculate_correlation(
    predicted: Sequence[float],
    actual: Sequence[float],
    n_bootstrap: int = 1000,
    ci_confidence: float = 0.95,
    seed: int = 42,
) -> CorrelationMetrics:
    spearman_rho, spearman_p = spearman_correlation(predicted, actual)
    pearson_r, pearson_p = _correlate(stats.pearsonr, predicted, actual)
    kendall_tau, _ = _correlate(stats.kendalltau, predicted, actual)

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
        spearman_kendall_gap=abs(spearman_rho - kendall_tau),
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


def calculate_macro_f1(tp: int, fp: int, tn: int, fn: int) -> float:
    """Computes Macro F1-score for binary classification."""
    # Positive class F1
    prec_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_pos = 2 * prec_pos * rec_pos / (prec_pos + rec_pos) if (prec_pos + rec_pos) > 0 else 0.0
    
    # Negative class F1
    prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_neg = 2 * prec_neg * rec_neg / (prec_neg + rec_neg) if (prec_neg + rec_neg) > 0 else 0.0
    
    return (f1_pos + f1_neg) / 2.0


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
    macro_f1 = calculate_macro_f1(tp, fp, tn, fn)

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
        macro_f1=macro_f1,
    )


def calculate_auc_pr(predicted_scores: Sequence[float], actual_critical: Sequence[bool]) -> float:
    """
    Calculates Area Under the Precision-Recall Curve (AUC-PR).
    Uses trapezoidal integration of sorted precision/recall points.
    """
    if not predicted_scores or not actual_critical or len(predicted_scores) != len(actual_critical):
        return 0.0

    # Sort by predicted scores descending
    sorted_pairs = sorted(zip(predicted_scores, actual_critical), key=lambda x: x[0], reverse=True)
    
    tp = 0
    fp = 0
    total_positives = sum(1 for x in actual_critical if x)
    
    if total_positives == 0:
        return 0.0

    precisions = [1.0]
    recalls = [0.0]
    
    for _, is_positive in sorted_pairs:
        if is_positive:
            tp += 1
        else:
            fp += 1
        
        precision = tp / (tp + fp)
        recall = tp / total_positives
        
        precisions.append(precision)
        recalls.append(recall)
        
    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(precisions)):
        auc += (recalls[i] - recalls[i-1]) * (precisions[i] + precisions[i-1]) / 2
        
    return auc


def calculate_ranking(
    predicted: Dict[str, float],
    actual: Dict[str, float],
    k_values: Optional[List[int]] = None,
    n_bootstrap: int = 1000,
    ci_confidence: float = 0.95,
    seed: int = 42,
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

    k5 = k_values[0]
    k_ndcg = k_values[1]

    overlap_5, pred_5, actual_5, common_5 = top_k_overlap(k5)
    overlap_ndcg, _, _, _ = top_k_overlap(k_ndcg)

    ndcg_5 = _calculate_ndcg(pred_ids, actual, k=k5)
    ndcg_ndcg = _calculate_ndcg(pred_ids, actual, k=k_ndcg)

    ci_lower, ci_upper = 0.0, 0.0
    n = len(predicted)
    if n >= k5:
        ci_lower, ci_upper = _bootstrap_ranking_ci(
            predicted, actual, k=k5,
            n_bootstrap=n_bootstrap,
            confidence=ci_confidence,
            seed=seed,
        )
    elif n >= 3:
        ci_lower, ci_upper = overlap_5, overlap_5

    return RankingMetrics(
        top_5_overlap=overlap_5,
        top_10_overlap=overlap_ndcg,
        ndcg_5=ndcg_5,
        ndcg_10=ndcg_ndcg,
        top_5_predicted=pred_5,
        top_5_actual=actual_5,
        top_5_common=common_5,
        top_5_ci_lower=ci_lower,
        top_5_ci_upper=ci_upper,
    )


def _bootstrap_ranking_ci(
    predicted: Dict[str, float],
    actual: Dict[str, float],
    k: int = 5,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    keys = list(predicted.keys())
    n = len(keys)
    if n < k:
        return 0.0, 0.0

    def _overlap(indices: List[int]) -> float:
        # Resampled ids are relabelled so that duplicate draws stay distinct
        # entries — otherwise the resampled top-K would shrink below k.
        s_pred = {i: predicted[keys[j]] for i, j in enumerate(indices)}
        s_actual = {i: actual[keys[j]] for i, j in enumerate(indices)}
        p_top = set(sorted(s_pred, key=lambda i: s_pred[i], reverse=True)[:k])
        a_top = set(sorted(s_actual, key=lambda i: s_actual[i], reverse=True)[:k])
        return len(p_top & a_top) / k

    return _bootstrap_percentile_ci(_overlap, n, n_bootstrap, confidence, seed)


def bootstrap_ci(
    predicted: Sequence[float],
    actual: Sequence[float],
    metric_fn: Callable[[Sequence[float], Sequence[float]], float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    n = len(predicted)
    if n < 3:
        return 0.0, 0.0

    pred_list = list(predicted)
    actual_list = list(actual)

    def _metric(indices: List[int]) -> float:
        return metric_fn([pred_list[i] for i in indices], [actual_list[i] for i in indices])

    return _bootstrap_percentile_ci(_metric, n, n_bootstrap, confidence, seed)


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
    """Spearman ρ and its two-sided p-value. Returns (0.0, 1.0) for n < 3 or constant input."""
    return _correlate(stats.spearmanr, predicted, actual)


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


def calculate_capture_rate_at_k(
    predicted: Dict[str, float],
    actual: Dict[str, float],
    k: int = 5,
) -> float:
    """Top-K agreement between a dimension predictor and its simulated ground truth.

    capture@K = |Top-K(pred) ∩ Top-K(actual)| / K, over the ids both dicts score.

    Two dimensions each name this metric differently but compute it identically:

    | Alias   | Dimension       | Predictor / ground truth | Target |
    |---------|-----------------|--------------------------|--------|
    | CCR@K   | Reliability     | R(v) vs IR(v)            | ≥ 0.80 |
    | COCR@K  | Maintainability | M(v) vs IM(v)            | ≥ 0.75 |
    """
    return _top_k_overlap(predicted, actual, k)[0]


# Domain-specific aliases — same statistic, different dimension and target.
calculate_ccr_at_k = calculate_capture_rate_at_k    # Cascade Capture Rate (Reliability)
calculate_cocr_at_k = calculate_capture_rate_at_k   # Change Obligation Capture Rate (Maintainability)


def calculate_cme(
    predicted: Dict[str, float],
    actual: Dict[str, float],
) -> float:
    """Cascade Magnitude Error (rank distance, normalised by system size).

    CME = mean|rank_R(v) - rank_IR(v)| / |V|

    Validates that the *scale* of predicted fault propagation matches
    the simulation-observed cascade magnitude, not just the ranking.  
    Target: CME ≤ 0.10.
    """
    common = sorted(set(predicted) & set(actual))
    n = len(common)
    if n < 2:
        return 0.0

    def _rank_map(scores: Dict[str, float]) -> Dict[str, int]:
        """Return 1-based rank for each id (lower score = higher rank number)."""
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return {cid: i + 1 for i, (cid, _) in enumerate(sorted_ids)}

    pred_ranks = _rank_map(predicted)
    actual_ranks = _rank_map(actual)

    total_rank_error = sum(
        abs(pred_ranks.get(cid, n) - actual_ranks.get(cid, n))
        for cid in common
    )
    return (total_rank_error / n) / n  # normalise by system size


def _bootstrap_classification_ci(
    pred_crit: List[bool],
    actual_crit: List[bool],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    n = len(pred_crit)
    if n < 3:
        return 0.0, 0.0

    def _f1(indices: List[int]) -> float:
        p_sample = [pred_crit[i] for i in indices]
        a_sample = [actual_crit[i] for i in indices]
        tp = sum(1 for p, a in zip(p_sample, a_sample) if p and a)
        fp = sum(1 for p, a in zip(p_sample, a_sample) if p and not a)
        fn = sum(1 for p, a in zip(p_sample, a_sample) if not p and a)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return _bootstrap_percentile_ci(_f1, n, n_bootstrap, confidence, seed)


# =============================================================================
# Maintainability-Specific Metrics  (M(v) v5 / IM(v))
# =============================================================================

def calculate_weighted_kappa_cta(
    predicted: Dict[str, float],
    actual: Dict[str, float],
) -> float:
    """Coupling Tier Agreement — weighted Cohen's κ across 3 ordered tiers.

    Partitions into High (top 25%), Medium (middle 50%), Low (bottom 25%).
    Ordinal weight matrix:  same=0.0, adjacent=0.5, extreme=1.0.
    κ_weighted = 1 − (Σ w_ij·p_ij) / (Σ w_ij·e_ij).
    Target: κ_weighted ≥ 0.55.
    """
    common = sorted(set(predicted) & set(actual))
    n = len(common)
    if n < 3:
        return 0.0

    def _tier(vals: List[float]) -> List[int]:
        sv = sorted(vals)
        t1 = sv[max(0, int(n * 0.33) - 1)]
        t2 = sv[min(n - 1, int(n * 0.66))]
        return [2 if v > t2 else (1 if v >= t1 else 0) for v in vals]

    pred_tiers = _tier([predicted[c] for c in common])
    actual_tiers = _tier([actual[c] for c in common])

    # Weight matrix: w[i][j] = 1 − |i−j| / (n_tiers − 1)
    # Tiers 0, 1, 2. Max distance = 2.
    # w[0][0]=1, w[0][1]=0.5, w[0][2]=0
    W = [[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]]
    conf = [[0] * 3 for _ in range(3)]
    for p_t, a_t in zip(pred_tiers, actual_tiers):
        conf[p_t][a_t] += 1

    row_sums = [sum(conf[i]) for i in range(3)]
    col_sums = [sum(conf[i][j] for i in range(3)) for j in range(3)]

    obs_w = sum(W[i][j] * conf[i][j] / n for i in range(3) for j in range(3))
    exp_w = sum(W[i][j] * (row_sums[i] / n) * (col_sums[j] / n) for i in range(3) for j in range(3))

    if exp_w == 1.0: # Perfect agreement expected by chance?
        return 1.0 if obs_w == 1.0 else 0.0
    
    # Standard linear weighted kappa: (obs_w - exp_w) / (1 - exp_w)
    if (1.0 - exp_w) == 0:
        return 1.0
    return max(-1.0, min(1.0, (obs_w - exp_w) / (1.0 - exp_w)))


def calculate_bottleneck_precision(
    predicted_bt: Dict[str, float],
    predicted_w_out: Dict[str, float],
    actual_im: Dict[str, float],
    bt_threshold: float = 0.60,
    w_out_threshold: float = 0.30,
    im_threshold: float = 0.50,
) -> float:
    """Bottleneck Precision (BP).

    Among BT-dominant components (BT > bt_threshold AND w_out < w_out_threshold),
    the fraction that also have high IM(v) (> im_threshold).

    BP = |{v : BT-dominant ∧ IM(v) > im_threshold}| / |{v : BT-dominant}|
    Target: BP ≥ 0.70. Returns 0.0 if no BT-dominant components exist.
    """
    common = set(predicted_bt) & set(predicted_w_out) & set(actual_im)
    if not common:
        return 0.0
    bt_dominant = [
        cid for cid in common
        if predicted_bt[cid] > bt_threshold and predicted_w_out[cid] < w_out_threshold
    ]
    if not bt_dominant:
        return 0.0
    return sum(1 for cid in bt_dominant if actual_im[cid] > im_threshold) / len(bt_dominant)


# =============================================================================
# Availability-Specific Metrics (A(v) v2)
# =============================================================================

def calculate_spof_f1(
    predicted_ap: Dict[str, float],
    actual_ia: Dict[str, float],
    ap_threshold: float = 0.0,
    ia_threshold: float = 0.50,
) -> Dict[str, float]:
    """SPOF Precision-Recall F1 (SPR).

    A component v is a "true SPOF" if IA(v) > ia_threshold (confirmed by
    failure simulation) and a "predicted SPOF" if AP_c(v) > ap_threshold
    (structural detection).

    SPR = {precision, recall, f1}  — Target F1 ≥ 0.90.

    Args:
        predicted_ap:  {component_id: AP_c_directed_score}
        actual_ia:     {component_id: IA(v) score from simulation}
        ap_threshold:  AP score threshold to classify as predicted-SPOF (default 0.0 = any AP)
        ia_threshold:  IA score threshold to classify as actual-SPOF (default 0.50)

    Returns:
        Dict with keys 'precision', 'recall', 'f1'.
    """
    common = set(predicted_ap) & set(actual_ia)
    if not common:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = fp = fn = 0
    for cid in common:
        pred_spof   = predicted_ap[cid] > ap_threshold
        actual_spof = actual_ia[cid] > ia_threshold
        if pred_spof and actual_spof:
            tp += 1
        elif pred_spof and not actual_spof:
            fp += 1
        elif not pred_spof and actual_spof:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def calculate_hsrr(
    predicted_qspof: Dict[str, float],
    actual_ia: Dict[str, float],
    ap_c_binary: Dict[str, float],
    ia_threshold: float = 0.50,
    qspof_threshold: float = 0.0,
) -> float:
    """Hidden SPOF Recovery Rate (HSRR).

    Fraction of high-availability-impact components that are not binary
    articulation points but are nevertheless caught by QSPOF or CDI.

    HSRR = |{v : AP_c=0 ∧ QSPOF > 0 ∧ IA > ia_threshold}|
          / |{v : AP_c=0 ∧ IA > ia_threshold}|
    """
    common = set(predicted_qspof) & set(actual_ia) & set(ap_c_binary)
    if not common:
        return 0.0
    
    candidates = [
        cid for cid in common
        if ap_c_binary[cid] == 0 and actual_ia[cid] > ia_threshold
    ]
    if not candidates:
        return 0.0
    
    recovered = sum(1 for cid in candidates if predicted_qspof[cid] > qspof_threshold)
    return recovered / len(candidates)


def calculate_dasa(
    ap_c_out: Dict[str, float],
    ap_c_in: Dict[str, float],
    ia_out: Dict[str, float],
    ia_in: Dict[str, float],
) -> float:
    """Directed SPOF Asymmetry Accuracy (DASA).

    Checks that the directionality of the SPOF (whether out-reachability
    or in-reachability dominates) matches the directionality observed
    in simulation.

    DASA = |{v : sign(AP_c_out - AP_c_in) = sign(ia_out - ia_in)}| / n
    Target: DASA ≥ 0.70.
    """
    common = set(ap_c_out) & set(ap_c_in) & set(ia_out) & set(ia_in)
    if not common:
        return 0.0
    
    def _sign(val):
        return 1 if val > 0 else -1 if val < 0 else 0

    matching = sum(
        1 for cid in common
        if _sign(ap_c_out[cid] - ap_c_in[cid]) == _sign(ia_out[cid] - ia_in[cid])
    )
    return matching / len(common)


def calculate_rri(
    actual_ia: Dict[str, float],
    br_scores: Dict[str, float],
    ia_threshold: float = 0.30,
) -> float:
    """Redundancy Robustness Index (RRI).

    Among components with no bridge edges (structurally redundant),
    what fraction also have low actual availability impact?

    RRI = |{v : BR(v) = 0 AND IA < ia_threshold}| / |{v : BR(v) = 0}|
    Target: RRI ≥ 0.80.
    """
    common = set(actual_ia) & set(br_scores)
    if not common:
        return 0.0

    redundant = [cid for cid in common if br_scores[cid] == 0.0]
    if not redundant:
        return 0.0

    true_negatives = sum(1 for cid in redundant if actual_ia[cid] < ia_threshold)
    return true_negatives / len(redundant)

