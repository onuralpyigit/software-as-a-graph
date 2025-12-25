#!/usr/bin/env python3
"""
Graph Validator - Validation of Graph-Based Criticality Analysis
==================================================================

Validates graph-based criticality predictions by comparing against
actual impact scores from failure simulation.

Validation Approach:
1. Correlation Analysis: Spearman rank correlation (target ≥ 0.70)
2. Classification Metrics: F1-Score (≥ 0.90), Precision/Recall (≥ 0.80)
3. Ranking Analysis: Top-k overlap for critical component identification
4. Statistical Confidence: Bootstrap confidence intervals, p-values

This module addresses the key research question: Do topological metrics
accurately predict actual system impact when components fail?

Author: Software-as-a-Graph Research Project
"""

import math
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Set


# ============================================================================
# Enums
# ============================================================================

class ValidationStatus(Enum):
    """Overall validation result status"""
    PASSED = "passed"           # All targets met
    PARTIAL = "partial"         # Some targets met
    FAILED = "failed"           # Most targets not met
    INSUFFICIENT_DATA = "insufficient_data"


class MetricStatus(Enum):
    """Individual metric status"""
    MET = "met"
    NOT_MET = "not_met"
    BORDERLINE = "borderline"   # Within 5% of target


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CorrelationMetrics:
    """Correlation analysis results"""
    spearman_coefficient: float
    spearman_p_value: float
    pearson_coefficient: float
    pearson_p_value: float
    kendall_tau: float
    sample_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'spearman': {
                'coefficient': round(self.spearman_coefficient, 4),
                'p_value': round(self.spearman_p_value, 6),
                'significant': self.spearman_p_value < 0.05
            },
            'pearson': {
                'coefficient': round(self.pearson_coefficient, 4),
                'p_value': round(self.pearson_p_value, 6),
                'significant': self.pearson_p_value < 0.05
            },
            'kendall_tau': round(self.kendall_tau, 4),
            'sample_size': self.sample_size
        }


@dataclass
class ConfusionMatrix:
    """Binary classification confusion matrix"""
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    threshold: float
    
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
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0
    
    @property
    def specificity(self) -> float:
        denom = self.true_negatives + self.false_positives
        return self.true_negatives / denom if denom > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'matrix': {
                'tp': self.true_positives,
                'tn': self.true_negatives,
                'fp': self.false_positives,
                'fn': self.false_negatives
            },
            'threshold': round(self.threshold, 4),
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1_score': round(self.f1_score, 4),
            'accuracy': round(self.accuracy, 4),
            'specificity': round(self.specificity, 4)
        }


@dataclass
class RankingMetrics:
    """Ranking analysis results"""
    top_k_overlap: Dict[int, float]  # k -> overlap percentage
    mean_rank_difference: float
    max_rank_difference: int
    kendall_tau: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'top_k_overlap': {f'top_{k}': round(v, 4) for k, v in self.top_k_overlap.items()},
            'mean_rank_difference': round(self.mean_rank_difference, 2),
            'max_rank_difference': self.max_rank_difference,
            'kendall_tau': round(self.kendall_tau, 4)
        }


@dataclass
class ComponentValidation:
    """Validation result for a single component"""
    component_id: str
    component_type: str
    predicted_score: float
    actual_impact: float
    predicted_rank: int
    actual_rank: int
    rank_difference: int
    predicted_critical: bool
    actual_critical: bool
    classification_correct: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.component_id,
            'type': self.component_type,
            'predicted_score': round(self.predicted_score, 4),
            'actual_impact': round(self.actual_impact, 4),
            'predicted_rank': self.predicted_rank,
            'actual_rank': self.actual_rank,
            'rank_diff': self.rank_difference,
            'correct': self.classification_correct
        }


@dataclass
class BootstrapResult:
    """Bootstrap confidence interval result"""
    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_iterations: int
    std_error: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric': self.metric_name,
            'estimate': round(self.point_estimate, 4),
            'ci_lower': round(self.ci_lower, 4),
            'ci_upper': round(self.ci_upper, 4),
            'confidence': self.confidence_level,
            'std_error': round(self.std_error, 4),
            'iterations': self.n_iterations
        }


@dataclass
class ValidationTargets:
    """Target metrics for validation"""
    spearman_correlation: float = 0.70
    f1_score: float = 0.90
    precision: float = 0.80
    recall: float = 0.80
    top_5_overlap: float = 0.60
    top_10_overlap: float = 0.70
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'spearman_correlation': self.spearman_correlation,
            'f1_score': self.f1_score,
            'precision': self.precision,
            'recall': self.recall,
            'top_5_overlap': self.top_5_overlap,
            'top_10_overlap': self.top_10_overlap
        }


@dataclass
class ValidationResult:
    """Complete validation result"""
    timestamp: datetime
    status: ValidationStatus
    total_components: int
    
    # Core metrics
    correlation: CorrelationMetrics
    classification: ConfusionMatrix
    ranking: RankingMetrics
    
    # Targets and achievements
    targets: ValidationTargets
    achieved: Dict[str, Tuple[float, MetricStatus]]
    
    # Component details
    component_validations: List[ComponentValidation]
    
    # Optional advanced analysis
    bootstrap_results: Optional[List[BootstrapResult]] = None
    
    # Misclassified components
    false_positives: List[str] = field(default_factory=list)
    false_negatives: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'total_components': self.total_components,
            'correlation': self.correlation.to_dict(),
            'classification': self.classification.to_dict(),
            'ranking': self.ranking.to_dict(),
            'targets': self.targets.to_dict(),
            'achieved': {
                k: {'value': round(v[0], 4), 'status': v[1].value}
                for k, v in self.achieved.items()
            },
            'misclassified': {
                'false_positives': self.false_positives[:10],
                'false_negatives': self.false_negatives[:10]
            },
            'summary': self.summary()
        }
        
        if self.bootstrap_results:
            result['bootstrap'] = [b.to_dict() for b in self.bootstrap_results]
        
        return result
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        met = sum(1 for _, (_, status) in self.achieved.items() if status == MetricStatus.MET)
        total = len(self.achieved)
        
        return {
            'metrics_met': f"{met}/{total}",
            'pass_rate': round(met / total, 2) if total > 0 else 0,
            'spearman': round(self.correlation.spearman_coefficient, 3),
            'f1_score': round(self.classification.f1_score, 3),
            'top_5_overlap': round(self.ranking.top_k_overlap.get(5, 0), 3)
        }


# ============================================================================
# Statistical Functions
# ============================================================================

def spearman_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Calculate Spearman rank correlation coefficient.
    
    Args:
        x: First variable values
        y: Second variable values
        
    Returns:
        Tuple of (coefficient, p-value)
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    
    rank_x = _rank_data(x)
    rank_y = _rank_data(y)
    
    return pearson_correlation(rank_x, rank_y)


def pearson_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Calculate Pearson correlation coefficient.
    
    Args:
        x: First variable values
        y: Second variable values
        
    Returns:
        Tuple of (coefficient, p-value)
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
    
    denominator = math.sqrt(sum_sq_x * sum_sq_y)
    
    if denominator == 0:
        return 0.0, 1.0
    
    r = numerator / denominator
    r = max(-1.0, min(1.0, r))  # Clamp to [-1, 1]
    
    # P-value using t-distribution approximation
    if abs(r) >= 1:
        p_value = 0.0
    else:
        t_stat = r * math.sqrt((n - 2) / (1 - r ** 2))
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
    
    return r, p_value


def kendall_tau(x: List[float], y: List[float]) -> float:
    """Calculate Kendall's tau rank correlation"""
    n = len(x)
    if n < 2:
        return 0.0
    
    concordant = 0
    discordant = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            x_diff = x[i] - x[j]
            y_diff = y[i] - y[j]
            product = x_diff * y_diff
            
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    
    total_pairs = n * (n - 1) / 2
    if total_pairs == 0:
        return 0.0
    
    return (concordant - discordant) / total_pairs


def _rank_data(data: List[float]) -> List[float]:
    """Assign ranks to data with average rank for ties"""
    n = len(data)
    indexed = [(val, i) for i, val in enumerate(data)]
    indexed.sort(key=lambda x: -x[0])  # Descending order
    
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j][0] == indexed[j + 1][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][1]] = avg_rank
        i = j + 1
    
    return ranks


def _normal_cdf(x: float) -> float:
    """Approximate normal CDF"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def percentile(data: List[float], p: float) -> float:
    """Calculate p-th percentile"""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


# ============================================================================
# Main Validator Class
# ============================================================================

class GraphValidator:
    """
    Validates graph-based criticality predictions against simulation results.
    
    The validator compares predicted criticality scores (from topological
    analysis) against actual impact scores (from failure simulation) to
    assess prediction accuracy.
    
    Target Metrics:
    - Spearman Correlation: ≥ 0.70
    - F1-Score: ≥ 0.90
    - Precision: ≥ 0.80
    - Recall: ≥ 0.80
    - Top-5 Overlap: ≥ 60%
    - Top-10 Overlap: ≥ 70%
    """
    
    def __init__(self,
                 targets: Optional[ValidationTargets] = None,
                 critical_threshold_percentile: float = 80,
                 seed: Optional[int] = None):
        """
        Initialize the validator.
        
        Args:
            targets: Custom validation targets (default: standard targets)
            critical_threshold_percentile: Percentile for critical classification
            seed: Random seed for bootstrap analysis
        """
        self.targets = targets or ValidationTargets()
        self.critical_threshold_percentile = critical_threshold_percentile
        self._rng = random.Random(seed)
        self.logger = logging.getLogger('GraphValidator')
    
    def validate(self,
                 predicted_scores: Dict[str, float],
                 actual_impacts: Dict[str, float],
                 component_types: Optional[Dict[str, str]] = None) -> ValidationResult:
        """
        Validate predictions against actual impacts.
        
        Args:
            predicted_scores: Component ID -> predicted criticality score
            actual_impacts: Component ID -> actual impact score from simulation
            component_types: Optional component ID -> type mapping
            
        Returns:
            ValidationResult with comprehensive metrics
        """
        # Find common components
        common = set(predicted_scores.keys()) & set(actual_impacts.keys())
        
        if len(common) < 5:
            self.logger.warning(f"Insufficient data: only {len(common)} common components")
            return self._insufficient_data_result()
        
        self.logger.info(f"Validating {len(common)} components...")
        
        # Extract aligned data
        pred_list = [predicted_scores[c] for c in common]
        actual_list = [actual_impacts[c] for c in common]
        
        # Calculate thresholds
        pred_threshold = percentile(pred_list, self.critical_threshold_percentile)
        actual_threshold = percentile(actual_list, self.critical_threshold_percentile)
        
        # Calculate correlations
        spearman_r, spearman_p = spearman_correlation(pred_list, actual_list)
        pearson_r, pearson_p = pearson_correlation(pred_list, actual_list)
        ktau = kendall_tau(pred_list, actual_list)
        
        correlation = CorrelationMetrics(
            spearman_coefficient=spearman_r,
            spearman_p_value=spearman_p,
            pearson_coefficient=pearson_r,
            pearson_p_value=pearson_p,
            kendall_tau=ktau,
            sample_size=len(common)
        )
        
        # Calculate classification metrics
        confusion, fp_list, fn_list = self._calculate_confusion(
            common, predicted_scores, actual_impacts,
            pred_threshold, actual_threshold
        )
        
        # Calculate ranking metrics
        ranking = self._calculate_ranking(
            common, predicted_scores, actual_impacts
        )
        
        # Build component validations
        component_validations = self._build_component_validations(
            common, predicted_scores, actual_impacts,
            pred_threshold, actual_threshold,
            component_types or {}
        )
        
        # Evaluate against targets
        achieved = self._evaluate_targets(correlation, confusion, ranking)
        
        # Determine overall status
        status = self._determine_status(achieved)
        
        return ValidationResult(
            timestamp=datetime.now(),
            status=status,
            total_components=len(common),
            correlation=correlation,
            classification=confusion,
            ranking=ranking,
            targets=self.targets,
            achieved=achieved,
            component_validations=component_validations,
            false_positives=fp_list,
            false_negatives=fn_list
        )
    
    def validate_with_bootstrap(self,
                                 predicted_scores: Dict[str, float],
                                 actual_impacts: Dict[str, float],
                                 n_iterations: int = 1000,
                                 confidence: float = 0.95) -> ValidationResult:
        """
        Validate with bootstrap confidence intervals.
        
        Args:
            predicted_scores: Component ID -> predicted score
            actual_impacts: Component ID -> actual impact
            n_iterations: Number of bootstrap iterations
            confidence: Confidence level for intervals
            
        Returns:
            ValidationResult with bootstrap confidence intervals
        """
        # First run standard validation
        result = self.validate(predicted_scores, actual_impacts)
        
        if result.status == ValidationStatus.INSUFFICIENT_DATA:
            return result
        
        # Run bootstrap analysis
        common = list(set(predicted_scores.keys()) & set(actual_impacts.keys()))
        n = len(common)
        
        spearman_samples = []
        f1_samples = []
        
        for _ in range(n_iterations):
            # Sample with replacement
            sample_indices = [self._rng.randint(0, n - 1) for _ in range(n)]
            sample_components = [common[i] for i in sample_indices]
            
            # Calculate metrics on sample
            pred_sample = [predicted_scores[c] for c in sample_components]
            actual_sample = [actual_impacts[c] for c in sample_components]
            
            r, _ = spearman_correlation(pred_sample, actual_sample)
            spearman_samples.append(r)
            
            # F1 for sample
            pred_thresh = percentile(pred_sample, self.critical_threshold_percentile)
            actual_thresh = percentile(actual_sample, self.critical_threshold_percentile)
            
            tp = sum(1 for p, a in zip(pred_sample, actual_sample) 
                    if p >= pred_thresh and a >= actual_thresh)
            fp = sum(1 for p, a in zip(pred_sample, actual_sample) 
                    if p >= pred_thresh and a < actual_thresh)
            fn = sum(1 for p, a in zip(pred_sample, actual_sample) 
                    if p < pred_thresh and a >= actual_thresh)
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            f1_samples.append(f1)
        
        # Calculate confidence intervals
        alpha = (1 - confidence) / 2
        
        spearman_sorted = sorted(spearman_samples)
        f1_sorted = sorted(f1_samples)
        
        lower_idx = int(n_iterations * alpha)
        upper_idx = int(n_iterations * (1 - alpha))
        
        result.bootstrap_results = [
            BootstrapResult(
                metric_name='spearman_correlation',
                point_estimate=result.correlation.spearman_coefficient,
                ci_lower=spearman_sorted[lower_idx],
                ci_upper=spearman_sorted[upper_idx],
                confidence_level=confidence,
                n_iterations=n_iterations,
                std_error=self._std_dev(spearman_samples)
            ),
            BootstrapResult(
                metric_name='f1_score',
                point_estimate=result.classification.f1_score,
                ci_lower=f1_sorted[lower_idx],
                ci_upper=f1_sorted[upper_idx],
                confidence_level=confidence,
                n_iterations=n_iterations,
                std_error=self._std_dev(f1_samples)
            )
        ]
        
        return result
    
    def _calculate_confusion(self,
                              components: Set[str],
                              predicted: Dict[str, float],
                              actual: Dict[str, float],
                              pred_threshold: float,
                              actual_threshold: float) -> Tuple[ConfusionMatrix, List[str], List[str]]:
        """Calculate confusion matrix and identify misclassified components"""
        tp = tn = fp = fn = 0
        fp_list = []
        fn_list = []
        
        for comp in components:
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
        
        return ConfusionMatrix(
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            threshold=pred_threshold
        ), fp_list, fn_list
    
    def _calculate_ranking(self,
                           components: Set[str],
                           predicted: Dict[str, float],
                           actual: Dict[str, float]) -> RankingMetrics:
        """Calculate ranking-based metrics"""
        # Sort by score (descending)
        pred_ranked = sorted(components, key=lambda c: -predicted[c])
        actual_ranked = sorted(components, key=lambda c: -actual[c])
        
        # Create rank lookups
        pred_rank = {c: i + 1 for i, c in enumerate(pred_ranked)}
        actual_rank = {c: i + 1 for i, c in enumerate(actual_ranked)}
        
        # Top-k overlap
        n = len(components)
        top_k_overlap = {}
        for k in [5, 10, 20]:
            if k <= n:
                pred_top_k = set(pred_ranked[:k])
                actual_top_k = set(actual_ranked[:k])
                overlap = len(pred_top_k & actual_top_k) / k
                top_k_overlap[k] = overlap
        
        # Rank differences
        rank_diffs = [abs(pred_rank[c] - actual_rank[c]) for c in components]
        mean_diff = sum(rank_diffs) / len(rank_diffs) if rank_diffs else 0
        max_diff = max(rank_diffs) if rank_diffs else 0
        
        # Kendall's tau
        pred_list = [predicted[c] for c in components]
        actual_list = [actual[c] for c in components]
        ktau = kendall_tau(pred_list, actual_list)
        
        return RankingMetrics(
            top_k_overlap=top_k_overlap,
            mean_rank_difference=mean_diff,
            max_rank_difference=max_diff,
            kendall_tau=ktau
        )
    
    def _build_component_validations(self,
                                      components: Set[str],
                                      predicted: Dict[str, float],
                                      actual: Dict[str, float],
                                      pred_threshold: float,
                                      actual_threshold: float,
                                      types: Dict[str, str]) -> List[ComponentValidation]:
        """Build per-component validation results"""
        pred_ranked = sorted(components, key=lambda c: -predicted[c])
        actual_ranked = sorted(components, key=lambda c: -actual[c])
        
        pred_rank = {c: i + 1 for i, c in enumerate(pred_ranked)}
        actual_rank = {c: i + 1 for i, c in enumerate(actual_ranked)}
        
        validations = []
        for comp in components:
            pred_critical = predicted[comp] >= pred_threshold
            actual_critical = actual[comp] >= actual_threshold
            
            validations.append(ComponentValidation(
                component_id=comp,
                component_type=types.get(comp, 'Unknown'),
                predicted_score=predicted[comp],
                actual_impact=actual[comp],
                predicted_rank=pred_rank[comp],
                actual_rank=actual_rank[comp],
                rank_difference=abs(pred_rank[comp] - actual_rank[comp]),
                predicted_critical=pred_critical,
                actual_critical=actual_critical,
                classification_correct=(pred_critical == actual_critical)
            ))
        
        # Sort by actual impact (most critical first)
        validations.sort(key=lambda v: -v.actual_impact)
        
        return validations
    
    def _evaluate_targets(self,
                          correlation: CorrelationMetrics,
                          confusion: ConfusionMatrix,
                          ranking: RankingMetrics) -> Dict[str, Tuple[float, MetricStatus]]:
        """Evaluate achieved metrics against targets"""
        achieved = {}
        
        # Spearman correlation
        spearman = correlation.spearman_coefficient
        achieved['spearman_correlation'] = (
            spearman,
            self._metric_status(spearman, self.targets.spearman_correlation)
        )
        
        # F1 Score
        f1 = confusion.f1_score
        achieved['f1_score'] = (f1, self._metric_status(f1, self.targets.f1_score))
        
        # Precision
        prec = confusion.precision
        achieved['precision'] = (prec, self._metric_status(prec, self.targets.precision))
        
        # Recall
        rec = confusion.recall
        achieved['recall'] = (rec, self._metric_status(rec, self.targets.recall))
        
        # Top-k overlap
        top5 = ranking.top_k_overlap.get(5, 0)
        achieved['top_5_overlap'] = (top5, self._metric_status(top5, self.targets.top_5_overlap))
        
        top10 = ranking.top_k_overlap.get(10, 0)
        achieved['top_10_overlap'] = (top10, self._metric_status(top10, self.targets.top_10_overlap))
        
        return achieved
    
    def _metric_status(self, value: float, target: float) -> MetricStatus:
        """Determine if a metric meets its target"""
        if value >= target:
            return MetricStatus.MET
        elif value >= target * 0.95:  # Within 5%
            return MetricStatus.BORDERLINE
        else:
            return MetricStatus.NOT_MET
    
    def _determine_status(self, achieved: Dict[str, Tuple[float, MetricStatus]]) -> ValidationStatus:
        """Determine overall validation status"""
        statuses = [status for _, status in achieved.values()]
        
        met_count = sum(1 for s in statuses if s == MetricStatus.MET)
        
        if met_count == len(statuses):
            return ValidationStatus.PASSED
        elif met_count >= len(statuses) * 0.5:
            return ValidationStatus.PARTIAL
        else:
            return ValidationStatus.FAILED
    
    def _insufficient_data_result(self) -> ValidationResult:
        """Create result for insufficient data"""
        return ValidationResult(
            timestamp=datetime.now(),
            status=ValidationStatus.INSUFFICIENT_DATA,
            total_components=0,
            correlation=CorrelationMetrics(0, 1, 0, 1, 0, 0),
            classification=ConfusionMatrix(0, 0, 0, 0, 0),
            ranking=RankingMetrics({}, 0, 0, 0),
            targets=self.targets,
            achieved={},
            component_validations=[]
        )
    
    def _std_dev(self, data: List[float]) -> float:
        """Calculate standard deviation"""
        if len(data) < 2:
            return 0.0
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
        return math.sqrt(variance)


# ============================================================================
# Convenience Functions
# ============================================================================

def validate_predictions(predicted: Dict[str, float],
                         actual: Dict[str, float],
                         component_types: Optional[Dict[str, str]] = None) -> ValidationResult:
    """
    Quick validation of predictions against actual impacts.
    
    Args:
        predicted: Component ID -> predicted criticality score
        actual: Component ID -> actual impact score
        component_types: Optional component type mapping
        
    Returns:
        ValidationResult
    """
    validator = GraphValidator()
    return validator.validate(predicted, actual, component_types)


def quick_validate(predicted: Dict[str, float],
                   actual: Dict[str, float]) -> Dict[str, Any]:
    """
    Quick validation returning key metrics as dictionary.
    
    Args:
        predicted: Predicted scores
        actual: Actual impacts
        
    Returns:
        Dictionary with key metrics
    """
    result = validate_predictions(predicted, actual)
    
    return {
        'status': result.status.value,
        'spearman': round(result.correlation.spearman_coefficient, 4),
        'f1_score': round(result.classification.f1_score, 4),
        'precision': round(result.classification.precision, 4),
        'recall': round(result.classification.recall, 4),
        'top_5_overlap': round(result.ranking.top_k_overlap.get(5, 0), 4),
        'n_components': result.total_components
    }