#!/usr/bin/env python3
"""
Graph Validator - Comprehensive Validation of Graph-Based Analysis
====================================================================

Validates graph-based criticality analysis by comparing predicted scores
from topological analysis against actual impact scores from failure simulation.

Validation Approach:
1. Correlation Analysis: Spearman (≥0.7 target) and Pearson correlation
2. Classification Metrics: F1-Score (≥0.9), Precision/Recall (≥0.8)
3. Ranking Analysis: Top-k overlap, mean rank difference
4. Sensitivity Analysis: Weight sensitivity, threshold sensitivity
5. Cross-Validation: K-fold validation for robustness
6. Bootstrap Analysis: Confidence intervals for metrics

Key Validation Targets (based on research methodology):
- Spearman Correlation ≥ 0.70
- F1-Score ≥ 0.90
- Precision ≥ 0.80
- Recall ≥ 0.80
- Top-5 Overlap ≥ 60%
- Top-10 Overlap ≥ 70%

Author: Software-as-a-Graph Research Project
"""

import math
import random
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
from abc import ABC, abstractmethod

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX is required: pip install networkx")


# ============================================================================
# Enums
# ============================================================================

class ValidationStatus(Enum):
    """Overall validation status"""
    PASSED = "passed"
    MARGINAL = "marginal"
    FAILED = "failed"
    
    def __str__(self):
        return self.value


class CriticalityLevel(Enum):
    """Criticality classification levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ConfusionMatrix:
    """Binary classification confusion matrix"""
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        """F1 = 2 * (precision * recall) / (precision + recall)"""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        """Accuracy = (TP + TN) / total"""
        total = self.true_positives + self.true_negatives + \
                self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0
    
    @property
    def specificity(self) -> float:
        """Specificity = TN / (TN + FP)"""
        denom = self.true_negatives + self.false_positives
        return self.true_negatives / denom if denom > 0 else 0.0
    
    @property
    def total(self) -> int:
        return self.true_positives + self.true_negatives + \
               self.false_positives + self.false_negatives
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1_score': round(self.f1_score, 4),
            'accuracy': round(self.accuracy, 4),
            'specificity': round(self.specificity, 4)
        }


@dataclass
class ComponentValidation:
    """Validation result for a single component"""
    component_id: str
    component_type: str
    
    # Predicted (from analysis)
    predicted_score: float
    predicted_rank: int
    predicted_level: str
    
    # Actual (from simulation)
    actual_impact: float
    actual_rank: int
    actual_level: str
    
    # Comparison
    rank_difference: int
    score_difference: float
    correctly_classified: bool
    
    # Additional context
    reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'component_id': self.component_id,
            'component_type': self.component_type,
            'predicted': {
                'score': round(self.predicted_score, 4),
                'rank': self.predicted_rank,
                'level': self.predicted_level
            },
            'actual': {
                'impact': round(self.actual_impact, 4),
                'rank': self.actual_rank,
                'level': self.actual_level
            },
            'comparison': {
                'rank_difference': self.rank_difference,
                'score_difference': round(self.score_difference, 4),
                'correctly_classified': self.correctly_classified
            },
            'reasons': self.reasons
        }


@dataclass
class CorrelationResult:
    """Correlation analysis result"""
    spearman_coefficient: float
    spearman_p_value: float
    pearson_coefficient: float
    pearson_p_value: float
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
            'sample_size': self.sample_size
        }


@dataclass
class RankingMetrics:
    """Ranking comparison metrics"""
    top_k_overlap: Dict[int, float]  # k -> overlap percentage
    mean_rank_difference: float
    max_rank_difference: int
    median_rank_difference: float
    kendall_tau: float  # Kendall's tau correlation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'top_k_overlap': {str(k): round(v, 4) for k, v in self.top_k_overlap.items()},
            'mean_rank_difference': round(self.mean_rank_difference, 2),
            'max_rank_difference': self.max_rank_difference,
            'median_rank_difference': round(self.median_rank_difference, 2),
            'kendall_tau': round(self.kendall_tau, 4)
        }


@dataclass
class SensitivityResult:
    """Sensitivity analysis result"""
    parameter_name: str
    original_value: float
    tested_values: List[float]
    metric_values: Dict[str, List[float]]  # metric_name -> values at each test point
    stability_score: float  # 0-1, higher is more stable
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'parameter': self.parameter_name,
            'original_value': self.original_value,
            'tested_values': self.tested_values,
            'metric_values': {k: [round(v, 4) for v in vals] 
                            for k, vals in self.metric_values.items()},
            'stability_score': round(self.stability_score, 4)
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
            'point_estimate': round(self.point_estimate, 4),
            'confidence_interval': {
                'lower': round(self.ci_lower, 4),
                'upper': round(self.ci_upper, 4),
                'level': self.confidence_level
            },
            'std_error': round(self.std_error, 4),
            'n_iterations': self.n_iterations
        }


@dataclass
class CrossValidationResult:
    """K-fold cross-validation result"""
    n_folds: int
    fold_results: List[Dict[str, float]]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_folds': self.n_folds,
            'fold_results': self.fold_results,
            'mean_metrics': {k: round(v, 4) for k, v in self.mean_metrics.items()},
            'std_metrics': {k: round(v, 4) for k, v in self.std_metrics.items()}
        }


@dataclass
class ValidationResult:
    """Complete validation results"""
    timestamp: datetime
    total_components: int
    
    # Status
    status: ValidationStatus
    
    # Correlation metrics
    correlation: CorrelationResult
    
    # Classification metrics
    confusion_matrix: ConfusionMatrix
    level_metrics: Dict[str, ConfusionMatrix]
    
    # Ranking metrics
    ranking: RankingMetrics
    
    # Component details
    component_validations: List[ComponentValidation]
    
    # Target comparison
    target_metrics: Dict[str, float]
    achieved_metrics: Dict[str, float]
    
    # Optional advanced analysis
    sensitivity_results: Optional[List[SensitivityResult]] = None
    bootstrap_results: Optional[List[BootstrapResult]] = None
    cross_validation: Optional[CrossValidationResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'timestamp': self.timestamp.isoformat(),
            'total_components': self.total_components,
            'status': self.status.value,
            'correlation': self.correlation.to_dict(),
            'classification': {
                'overall': self.confusion_matrix.to_dict(),
                'by_level': {k: v.to_dict() for k, v in self.level_metrics.items()}
            },
            'ranking': self.ranking.to_dict(),
            'targets': self.target_metrics,
            'achieved': {k: round(v, 4) for k, v in self.achieved_metrics.items()},
            'components': [c.to_dict() for c in self.component_validations]
        }
        
        if self.sensitivity_results:
            result['sensitivity'] = [s.to_dict() for s in self.sensitivity_results]
        if self.bootstrap_results:
            result['bootstrap'] = [b.to_dict() for b in self.bootstrap_results]
        if self.cross_validation:
            result['cross_validation'] = self.cross_validation.to_dict()
        
        return result
    
    def summary(self) -> str:
        """Generate text summary"""
        lines = [
            f"Validation Status: {self.status.value.upper()}",
            f"Components Validated: {self.total_components}",
            "",
            "Correlation Metrics:",
            f"  Spearman: {self.correlation.spearman_coefficient:.4f} (p={self.correlation.spearman_p_value:.4f})",
            f"  Pearson:  {self.correlation.pearson_coefficient:.4f} (p={self.correlation.pearson_p_value:.4f})",
            "",
            "Classification Metrics:",
            f"  Precision: {self.confusion_matrix.precision:.4f}",
            f"  Recall:    {self.confusion_matrix.recall:.4f}",
            f"  F1-Score:  {self.confusion_matrix.f1_score:.4f}",
            f"  Accuracy:  {self.confusion_matrix.accuracy:.4f}",
            "",
            "Ranking Metrics:",
            f"  Mean Rank Diff: {self.ranking.mean_rank_difference:.1f}",
            f"  Kendall's τ:    {self.ranking.kendall_tau:.4f}",
        ]
        
        for k, overlap in sorted(self.ranking.top_k_overlap.items()):
            lines.append(f"  Top-{k} Overlap: {overlap:.1%}")
        
        return "\n".join(lines)


# ============================================================================
# Statistical Functions
# ============================================================================

def spearman_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Calculate Spearman rank correlation coefficient.
    
    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    
    # Rank the data
    rank_x = _rank_data(x)
    rank_y = _rank_data(y)
    
    # Calculate Pearson on ranks
    return pearson_correlation(rank_x, rank_y)


def pearson_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Calculate Pearson correlation coefficient.
    
    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    
    # Calculate means
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # Calculate correlation
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
    
    denominator = math.sqrt(sum_sq_x * sum_sq_y)
    
    if denominator == 0:
        return 0.0, 1.0
    
    r = numerator / denominator
    
    # Calculate p-value using t-distribution approximation
    if abs(r) >= 1:
        p_value = 0.0
    else:
        t_stat = r * math.sqrt((n - 2) / (1 - r ** 2))
        # Approximate p-value using normal distribution for large n
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))
    
    return r, p_value


def kendall_tau(x: List[float], y: List[float]) -> float:
    """Calculate Kendall's tau correlation coefficient"""
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
    """Assign ranks to data (average rank for ties)"""
    n = len(data)
    indexed = [(val, i) for i, val in enumerate(data)]
    indexed.sort(key=lambda x: x[0])
    
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        # Find all elements with same value
        while j < n - 1 and indexed[j][0] == indexed[j + 1][0]:
            j += 1
        # Assign average rank
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][1]] = avg_rank
        i = j + 1
    
    return ranks


def _normal_cdf(x: float) -> float:
    """Approximate normal CDF using error function approximation"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def calculate_percentile(data: List[float], p: float) -> float:
    """Calculate p-th percentile of data"""
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
    Comprehensive validator for graph-based criticality analysis.
    
    Compares predicted criticality scores from topological analysis
    against actual impact scores from failure simulation.
    """
    
    # Default target thresholds based on research methodology
    DEFAULT_TARGETS = {
        'spearman_correlation': 0.70,
        'f1_score': 0.90,
        'precision': 0.80,
        'recall': 0.80,
        'top_5_overlap': 0.60,
        'top_10_overlap': 0.70
    }
    
    # Criticality thresholds for classification
    LEVEL_THRESHOLDS = {
        'critical': 0.7,
        'high': 0.5,
        'medium': 0.3,
        'low': 0.1,
        'minimal': 0.0
    }
    
    def __init__(self,
                 targets: Optional[Dict[str, float]] = None,
                 critical_threshold: float = 0.5,
                 seed: Optional[int] = None):
        """
        Initialize validator.
        
        Args:
            targets: Target metric thresholds (uses defaults if not provided)
            critical_threshold: Impact threshold to classify as "critical"
            seed: Random seed for reproducibility
        """
        self.targets = {**self.DEFAULT_TARGETS, **(targets or {})}
        self.critical_threshold = critical_threshold
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
        
        self.logger = logging.getLogger('GraphValidator')
        
        # Results cache
        self._validation_result: Optional[ValidationResult] = None
    
    def validate(self,
                graph: nx.DiGraph,
                predicted_scores: Dict[str, float],
                actual_impacts: Dict[str, float],
                component_types: Optional[Dict[str, str]] = None) -> ValidationResult:
        """
        Run full validation.
        
        Args:
            graph: NetworkX graph of the system
            predicted_scores: Component ID -> predicted criticality score
            actual_impacts: Component ID -> actual impact from simulation
            component_types: Optional component ID -> type mapping
            
        Returns:
            ValidationResult with comprehensive metrics
        """
        self.logger.info(f"Validating {len(predicted_scores)} predictions against "
                        f"{len(actual_impacts)} actual impacts")
        
        # Get common components
        common = set(predicted_scores.keys()) & set(actual_impacts.keys())
        if len(common) < 3:
            raise ValueError(f"Need at least 3 common components, got {len(common)}")
        
        # Get types
        if component_types is None:
            component_types = {node: graph.nodes[node].get('type', 'Unknown') 
                             for node in common if node in graph.nodes()}
        
        # Extract aligned scores
        pred_list = [predicted_scores[c] for c in common]
        actual_list = [actual_impacts[c] for c in common]
        
        # Calculate correlations
        correlation = self._calculate_correlations(pred_list, actual_list)
        
        # Calculate rankings
        pred_ranking = self._rank_dict(predicted_scores, common)
        actual_ranking = self._rank_dict(actual_impacts, common)
        ranking = self._calculate_ranking_metrics(pred_ranking, actual_ranking, common)
        
        # Calculate classification metrics
        confusion = self._calculate_confusion_matrix(
            predicted_scores, actual_impacts, common, self.critical_threshold
        )
        
        level_metrics = {}
        for level, threshold in self.LEVEL_THRESHOLDS.items():
            if level != 'minimal':
                level_metrics[level] = self._calculate_confusion_matrix(
                    predicted_scores, actual_impacts, common, threshold
                )
        
        # Build component validations
        component_validations = self._build_component_validations(
            common, predicted_scores, actual_impacts,
            pred_ranking, actual_ranking, component_types
        )
        
        # Calculate achieved metrics
        achieved = {
            'spearman_correlation': correlation.spearman_coefficient,
            'pearson_correlation': correlation.pearson_coefficient,
            'f1_score': confusion.f1_score,
            'precision': confusion.precision,
            'recall': confusion.recall,
            'accuracy': confusion.accuracy
        }
        
        for k, overlap in ranking.top_k_overlap.items():
            achieved[f'top_{k}_overlap'] = overlap
        
        # Determine status
        status = self._determine_status(achieved)
        
        self._validation_result = ValidationResult(
            timestamp=datetime.now(),
            total_components=len(common),
            status=status,
            correlation=correlation,
            confusion_matrix=confusion,
            level_metrics=level_metrics,
            ranking=ranking,
            component_validations=component_validations,
            target_metrics=self.targets,
            achieved_metrics=achieved
        )
        
        return self._validation_result
    
    def validate_with_simulation(self,
                                graph: nx.DiGraph,
                                predicted_scores: Dict[str, float],
                                enable_cascade: bool = False) -> ValidationResult:
        """
        Validate predictions by running failure simulation.
        
        Args:
            graph: NetworkX graph
            predicted_scores: Component ID -> predicted criticality score
            enable_cascade: Whether to enable cascade in simulation
            
        Returns:
            ValidationResult
        """
        from src.simulation import FailureSimulator
        
        simulator = FailureSimulator(seed=self.seed)
        
        # Run exhaustive simulation
        self.logger.info("Running exhaustive failure simulation...")
        batch_result = simulator.simulate_exhaustive(
            graph,
            enable_cascade=enable_cascade
        )
        
        # Extract impact scores
        actual_impacts = {}
        for sim_result in batch_result.results:
            if sim_result.primary_failures:
                comp = sim_result.primary_failures[0]
                actual_impacts[comp] = sim_result.impact_score
        
        return self.validate(graph, predicted_scores, actual_impacts)
    
    def run_sensitivity_analysis(self,
                                graph: nx.DiGraph,
                                predicted_scores: Dict[str, float],
                                actual_impacts: Dict[str, float],
                                parameters: Optional[Dict[str, Tuple[float, float, float]]] = None
                                ) -> List[SensitivityResult]:
        """
        Run sensitivity analysis on validation parameters.
        
        Args:
            graph: NetworkX graph
            predicted_scores: Predicted scores
            actual_impacts: Actual impacts
            parameters: Dict of parameter_name -> (min, max, original)
                       Defaults to critical_threshold sensitivity
                       
        Returns:
            List of SensitivityResult
        """
        if parameters is None:
            parameters = {
                'critical_threshold': (0.3, 0.7, self.critical_threshold)
            }
        
        results = []
        common = set(predicted_scores.keys()) & set(actual_impacts.keys())
        
        for param_name, (min_val, max_val, original) in parameters.items():
            test_values = [min_val + (max_val - min_val) * i / 10 for i in range(11)]
            metric_values = defaultdict(list)
            
            for test_val in test_values:
                # Run validation with modified parameter
                if param_name == 'critical_threshold':
                    confusion = self._calculate_confusion_matrix(
                        predicted_scores, actual_impacts, common, test_val
                    )
                    metric_values['f1_score'].append(confusion.f1_score)
                    metric_values['precision'].append(confusion.precision)
                    metric_values['recall'].append(confusion.recall)
            
            # Calculate stability score (1 - coefficient of variation)
            stability_scores = []
            for metric, values in metric_values.items():
                if values:
                    mean = sum(values) / len(values)
                    if mean > 0:
                        variance = sum((v - mean) ** 2 for v in values) / len(values)
                        cv = math.sqrt(variance) / mean
                        stability_scores.append(1 - min(cv, 1))
            
            stability = sum(stability_scores) / len(stability_scores) if stability_scores else 0.5
            
            results.append(SensitivityResult(
                parameter_name=param_name,
                original_value=original,
                tested_values=test_values,
                metric_values=dict(metric_values),
                stability_score=stability
            ))
        
        if self._validation_result:
            self._validation_result.sensitivity_results = results
        
        return results
    
    def run_bootstrap_analysis(self,
                              graph: nx.DiGraph,
                              predicted_scores: Dict[str, float],
                              actual_impacts: Dict[str, float],
                              n_iterations: int = 1000,
                              confidence_level: float = 0.95) -> List[BootstrapResult]:
        """
        Run bootstrap analysis to calculate confidence intervals.
        
        Args:
            graph: NetworkX graph
            predicted_scores: Predicted scores
            actual_impacts: Actual impacts
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level for intervals
            
        Returns:
            List of BootstrapResult for each metric
        """
        common = list(set(predicted_scores.keys()) & set(actual_impacts.keys()))
        n = len(common)
        
        if n < 5:
            self.logger.warning("Not enough samples for bootstrap analysis")
            return []
        
        # Store bootstrap samples
        bootstrap_metrics = defaultdict(list)
        
        for _ in range(n_iterations):
            # Resample with replacement
            sample_indices = [random.randint(0, n - 1) for _ in range(n)]
            sample_comps = [common[i] for i in sample_indices]
            
            pred_sample = [predicted_scores[c] for c in sample_comps]
            actual_sample = [actual_impacts[c] for c in sample_comps]
            
            # Calculate metrics
            spearman_r, _ = spearman_correlation(pred_sample, actual_sample)
            bootstrap_metrics['spearman_correlation'].append(spearman_r)
            
            pearson_r, _ = pearson_correlation(pred_sample, actual_sample)
            bootstrap_metrics['pearson_correlation'].append(pearson_r)
            
            # Classification metrics (need original threshold)
            sample_pred = {c: predicted_scores[c] for c in sample_comps}
            sample_actual = {c: actual_impacts[c] for c in sample_comps}
            confusion = self._calculate_confusion_matrix(
                sample_pred, sample_actual, set(sample_comps), self.critical_threshold
            )
            bootstrap_metrics['f1_score'].append(confusion.f1_score)
            bootstrap_metrics['precision'].append(confusion.precision)
            bootstrap_metrics['recall'].append(confusion.recall)
        
        # Calculate confidence intervals
        results = []
        alpha = 1 - confidence_level
        
        for metric_name, values in bootstrap_metrics.items():
            sorted_values = sorted(values)
            
            point_estimate = sum(values) / len(values)
            ci_lower = sorted_values[int(alpha / 2 * n_iterations)]
            ci_upper = sorted_values[int((1 - alpha / 2) * n_iterations)]
            
            variance = sum((v - point_estimate) ** 2 for v in values) / len(values)
            std_error = math.sqrt(variance)
            
            results.append(BootstrapResult(
                metric_name=metric_name,
                point_estimate=point_estimate,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                confidence_level=confidence_level,
                n_iterations=n_iterations,
                std_error=std_error
            ))
        
        if self._validation_result:
            self._validation_result.bootstrap_results = results
        
        return results
    
    def run_cross_validation(self,
                            graph: nx.DiGraph,
                            predicted_scores: Dict[str, float],
                            actual_impacts: Dict[str, float],
                            n_folds: int = 5) -> CrossValidationResult:
        """
        Run k-fold cross-validation.
        
        Args:
            graph: NetworkX graph
            predicted_scores: Predicted scores
            actual_impacts: Actual impacts
            n_folds: Number of folds
            
        Returns:
            CrossValidationResult
        """
        common = list(set(predicted_scores.keys()) & set(actual_impacts.keys()))
        n = len(common)
        
        if n < n_folds:
            n_folds = n
        
        # Shuffle and split
        random.shuffle(common)
        fold_size = n // n_folds
        
        fold_results = []
        
        for i in range(n_folds):
            # Create test fold
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < n_folds - 1 else n
            test_comps = set(common[start_idx:end_idx])
            
            if len(test_comps) < 3:
                continue
            
            # Calculate metrics on fold
            pred_fold = [predicted_scores[c] for c in test_comps]
            actual_fold = [actual_impacts[c] for c in test_comps]
            
            spearman_r, _ = spearman_correlation(pred_fold, actual_fold)
            
            confusion = self._calculate_confusion_matrix(
                predicted_scores, actual_impacts, test_comps, self.critical_threshold
            )
            
            fold_results.append({
                'fold': i + 1,
                'size': len(test_comps),
                'spearman': spearman_r,
                'f1_score': confusion.f1_score,
                'precision': confusion.precision,
                'recall': confusion.recall
            })
        
        # Calculate mean and std
        mean_metrics = {}
        std_metrics = {}
        
        for metric in ['spearman', 'f1_score', 'precision', 'recall']:
            values = [fr[metric] for fr in fold_results]
            if values:
                mean_metrics[metric] = sum(values) / len(values)
                variance = sum((v - mean_metrics[metric]) ** 2 for v in values) / len(values)
                std_metrics[metric] = math.sqrt(variance)
        
        result = CrossValidationResult(
            n_folds=n_folds,
            fold_results=fold_results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics
        )
        
        if self._validation_result:
            self._validation_result.cross_validation = result
        
        return result
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _calculate_correlations(self, pred: List[float], actual: List[float]) -> CorrelationResult:
        """Calculate correlation metrics"""
        spearman_r, spearman_p = spearman_correlation(pred, actual)
        pearson_r, pearson_p = pearson_correlation(pred, actual)
        
        return CorrelationResult(
            spearman_coefficient=spearman_r,
            spearman_p_value=spearman_p,
            pearson_coefficient=pearson_r,
            pearson_p_value=pearson_p,
            sample_size=len(pred)
        )
    
    def _rank_dict(self, scores: Dict[str, float], common: Set[str]) -> Dict[str, int]:
        """Rank components (1 = highest score)"""
        sorted_comps = sorted(
            [c for c in scores.keys() if c in common],
            key=lambda c: scores[c],
            reverse=True
        )
        return {comp: rank + 1 for rank, comp in enumerate(sorted_comps)}
    
    def _calculate_ranking_metrics(self,
                                   pred_ranking: Dict[str, int],
                                   actual_ranking: Dict[str, int],
                                   common: Set[str]) -> RankingMetrics:
        """Calculate ranking comparison metrics"""
        # Top-k overlap
        top_k_overlap = {}
        for k in [3, 5, 10, 15, 20]:
            if k <= len(common):
                pred_top_k = {c for c, r in pred_ranking.items() if r <= k}
                actual_top_k = {c for c, r in actual_ranking.items() if r <= k}
                overlap = len(pred_top_k & actual_top_k)
                top_k_overlap[k] = overlap / k
        
        # Rank differences
        rank_diffs = [abs(pred_ranking[c] - actual_ranking[c]) for c in common]
        mean_diff = sum(rank_diffs) / len(rank_diffs) if rank_diffs else 0
        max_diff = max(rank_diffs) if rank_diffs else 0
        sorted_diffs = sorted(rank_diffs)
        median_diff = sorted_diffs[len(sorted_diffs) // 2] if sorted_diffs else 0
        
        # Kendall's tau
        pred_list = [pred_ranking[c] for c in common]
        actual_list = [actual_ranking[c] for c in common]
        tau = kendall_tau(pred_list, actual_list)
        
        return RankingMetrics(
            top_k_overlap=top_k_overlap,
            mean_rank_difference=mean_diff,
            max_rank_difference=max_diff,
            median_rank_difference=median_diff,
            kendall_tau=tau
        )
    
    def _calculate_confusion_matrix(self,
                                   predicted: Dict[str, float],
                                   actual: Dict[str, float],
                                   common: Set[str],
                                   threshold: float) -> ConfusionMatrix:
        """Calculate confusion matrix for given threshold"""
        cm = ConfusionMatrix()
        
        for comp in common:
            pred_critical = predicted[comp] >= threshold
            actual_critical = actual[comp] >= threshold
            
            if pred_critical and actual_critical:
                cm.true_positives += 1
            elif pred_critical and not actual_critical:
                cm.false_positives += 1
            elif not pred_critical and actual_critical:
                cm.false_negatives += 1
            else:
                cm.true_negatives += 1
        
        return cm
    
    def _classify_level(self, score: float) -> str:
        """Classify score into criticality level"""
        for level, threshold in sorted(self.LEVEL_THRESHOLDS.items(), 
                                       key=lambda x: -x[1]):
            if score >= threshold:
                return level
        return 'minimal'
    
    def _build_component_validations(self,
                                    common: Set[str],
                                    predicted: Dict[str, float],
                                    actual: Dict[str, float],
                                    pred_ranking: Dict[str, int],
                                    actual_ranking: Dict[str, int],
                                    component_types: Dict[str, str]
                                    ) -> List[ComponentValidation]:
        """Build validation results for each component"""
        validations = []
        
        for comp in common:
            pred_level = self._classify_level(predicted[comp])
            actual_level = self._classify_level(actual[comp])
            
            cv = ComponentValidation(
                component_id=comp,
                component_type=component_types.get(comp, 'Unknown'),
                predicted_score=predicted[comp],
                predicted_rank=pred_ranking[comp],
                predicted_level=pred_level,
                actual_impact=actual[comp],
                actual_rank=actual_ranking[comp],
                actual_level=actual_level,
                rank_difference=abs(pred_ranking[comp] - actual_ranking[comp]),
                score_difference=abs(predicted[comp] - actual[comp]),
                correctly_classified=(pred_level == actual_level)
            )
            validations.append(cv)
        
        # Sort by actual impact
        validations.sort(key=lambda x: x.actual_impact, reverse=True)
        
        return validations
    
    def _determine_status(self, achieved: Dict[str, float]) -> ValidationStatus:
        """Determine overall validation status"""
        passed = 0
        total = 0
        
        for metric, target in self.targets.items():
            if metric in achieved:
                total += 1
                if achieved[metric] >= target:
                    passed += 1
        
        if total == 0:
            return ValidationStatus.FAILED
        
        ratio = passed / total
        if ratio >= 0.8:
            return ValidationStatus.PASSED
        elif ratio >= 0.5:
            return ValidationStatus.MARGINAL
        else:
            return ValidationStatus.FAILED
    
    # =========================================================================
    # Analysis Helpers
    # =========================================================================
    
    def get_misclassified(self) -> List[ComponentValidation]:
        """Get misclassified components"""
        if not self._validation_result:
            return []
        return [cv for cv in self._validation_result.component_validations
                if not cv.correctly_classified]
    
    def get_high_rank_difference(self, threshold: int = 5) -> List[ComponentValidation]:
        """Get components with rank difference above threshold"""
        if not self._validation_result:
            return []
        return [cv for cv in self._validation_result.component_validations
                if cv.rank_difference > threshold]
    
    def get_underestimated(self, threshold: float = 0.2) -> List[ComponentValidation]:
        """Get components where actual > predicted by threshold"""
        if not self._validation_result:
            return []
        return [cv for cv in self._validation_result.component_validations
                if cv.actual_impact - cv.predicted_score > threshold]
    
    def get_overestimated(self, threshold: float = 0.2) -> List[ComponentValidation]:
        """Get components where predicted > actual by threshold"""
        if not self._validation_result:
            return []
        return [cv for cv in self._validation_result.component_validations
                if cv.predicted_score - cv.actual_impact > threshold]
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        if not self._validation_result:
            return []
        
        recommendations = []
        result = self._validation_result
        
        # Correlation check
        if result.correlation.spearman_coefficient < 0.7:
            recommendations.append(
                f"Spearman correlation ({result.correlation.spearman_coefficient:.3f}) "
                "is below target (0.7). Consider adjusting analysis weights "
                "(α, β, γ) or incorporating additional metrics."
            )
        
        # Classification checks
        cm = result.confusion_matrix
        if cm.precision < 0.8:
            recommendations.append(
                f"Precision ({cm.precision:.3f}) is below target (0.8). "
                "The model may be over-predicting critical components. "
                "Consider raising the criticality threshold."
            )
        
        if cm.recall < 0.8:
            recommendations.append(
                f"Recall ({cm.recall:.3f}) is below target (0.8). "
                "The model may be missing critical components. "
                "Consider lowering the criticality threshold or "
                "increasing the weight of articulation point detection."
            )
        
        # Ranking checks
        if result.ranking.top_k_overlap.get(5, 0) < 0.6:
            recommendations.append(
                "Top-5 overlap is below 60%. The most critical components "
                "may not be correctly identified. Review the weighting scheme."
            )
        
        # Component-specific
        underestimated = self.get_underestimated(0.3)
        if underestimated:
            comp_list = ', '.join([cv.component_id for cv in underestimated[:3]])
            recommendations.append(
                f"Some components are significantly underestimated: {comp_list}. "
                "These may have dependencies not captured in the graph model."
            )
        
        return recommendations


# ============================================================================
# Convenience Functions
# ============================================================================

def validate_analysis(graph: nx.DiGraph,
                     predicted_scores: Dict[str, float],
                     actual_impacts: Dict[str, float],
                     **kwargs) -> ValidationResult:
    """
    Convenience function for quick validation.
    
    Args:
        graph: NetworkX graph
        predicted_scores: Component -> predicted score
        actual_impacts: Component -> actual impact
        **kwargs: Additional arguments for GraphValidator
        
    Returns:
        ValidationResult
    """
    validator = GraphValidator(**kwargs)
    return validator.validate(graph, predicted_scores, actual_impacts)


def quick_validate(graph: nx.DiGraph,
                  predicted_scores: Dict[str, float],
                  enable_cascade: bool = False,
                  seed: int = 42) -> ValidationResult:
    """
    Quick validation by running simulation automatically.
    
    Args:
        graph: NetworkX graph
        predicted_scores: Component -> predicted score
        enable_cascade: Enable cascade in simulation
        seed: Random seed
        
    Returns:
        ValidationResult
    """
    validator = GraphValidator(seed=seed)
    return validator.validate_with_simulation(
        graph, predicted_scores, enable_cascade=enable_cascade
    )


def compare_methods(graph: nx.DiGraph,
                   methods: Dict[str, Dict[str, float]],
                   actual_impacts: Dict[str, float]) -> Dict[str, ValidationResult]:
    """
    Compare multiple analysis methods against same actual impacts.
    
    Args:
        graph: NetworkX graph
        methods: Dict of method_name -> predicted_scores
        actual_impacts: Actual impact scores
        
    Returns:
        Dict of method_name -> ValidationResult
    """
    results = {}
    validator = GraphValidator()
    
    for method_name, predictions in methods.items():
        results[method_name] = validator.validate(graph, predictions, actual_impacts)
    
    return results