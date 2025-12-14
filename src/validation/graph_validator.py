#!/usr/bin/env python3
"""
Graph Validator - Validation of Graph-Based Analysis
=====================================================

Validates the graph-based modeling and analysis approach by comparing
predicted criticality scores from topological analysis with actual
impact scores from failure simulation.

Key Validation Metrics:
  - Spearman Correlation: Rank correlation between predicted and actual (target ≥ 0.7)
  - Pearson Correlation: Linear correlation
  - F1-Score: Classification accuracy for critical components (target ≥ 0.9)
  - Precision/Recall: Detection rates for critical components (target 0.8-0.9)

Usage:
    from src.validation import GraphValidator
    from src.analysis import GraphAnalyzer
    from src.simulation import GraphSimulator
    
    analyzer = GraphAnalyzer()
    analyzer.load_from_file('system.json')
    
    validator = GraphValidator(analyzer)
    result = validator.validate()
    
    print(f"Spearman: {result.spearman_correlation:.3f}")
    print(f"F1-Score: {result.f1_score:.3f}")

Author: Software-as-a-Graph Research Project
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import math

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX is required: pip install networkx")


# ============================================================================
# Enums and Data Classes
# ============================================================================

class ValidationStatus(Enum):
    """Overall validation status"""
    PASSED = "passed"
    MARGINAL = "marginal"
    FAILED = "failed"


class CriticalityThreshold(Enum):
    """Thresholds for criticality classification"""
    CRITICAL = 0.7
    HIGH = 0.5
    MEDIUM = 0.3
    LOW = 0.0


@dataclass
class ComponentValidation:
    """Validation results for a single component"""
    component_id: str
    component_type: str
    predicted_score: float
    predicted_rank: int
    predicted_level: str
    actual_impact: float
    actual_rank: int
    actual_level: str
    rank_difference: int
    score_difference: float
    correctly_classified: bool
    
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
            'rank_difference': self.rank_difference,
            'score_difference': round(self.score_difference, 4),
            'correctly_classified': self.correctly_classified
        }


@dataclass
class ConfusionMatrix:
    """Confusion matrix for classification evaluation"""
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
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
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives,
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1_score': round(self.f1_score, 4),
            'accuracy': round(self.accuracy, 4)
        }


@dataclass
class ValidationResult:
    """Complete validation results"""
    timestamp: datetime
    total_components: int
    
    # Correlation metrics
    spearman_correlation: float
    spearman_p_value: float
    pearson_correlation: float
    pearson_p_value: float
    
    # Classification metrics (for "critical" threshold)
    confusion_matrix: ConfusionMatrix
    
    # Per-level metrics
    level_metrics: Dict[str, ConfusionMatrix]
    
    # Ranking metrics
    top_k_overlap: Dict[int, float]  # k -> overlap percentage
    mean_rank_difference: float
    max_rank_difference: int
    
    # Component details
    component_validations: List[ComponentValidation]
    
    # Status
    status: ValidationStatus
    target_metrics: Dict[str, float]
    achieved_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_components': self.total_components,
            'status': self.status.value,
            'correlation': {
                'spearman': {
                    'coefficient': round(self.spearman_correlation, 4),
                    'p_value': round(self.spearman_p_value, 6)
                },
                'pearson': {
                    'coefficient': round(self.pearson_correlation, 4),
                    'p_value': round(self.pearson_p_value, 6)
                }
            },
            'classification': {
                'overall': self.confusion_matrix.to_dict(),
                'by_level': {k: v.to_dict() for k, v in self.level_metrics.items()}
            },
            'ranking': {
                'top_k_overlap': {str(k): round(v, 4) for k, v in self.top_k_overlap.items()},
                'mean_rank_difference': round(self.mean_rank_difference, 2),
                'max_rank_difference': self.max_rank_difference
            },
            'targets': self.target_metrics,
            'achieved': {k: round(v, 4) for k, v in self.achieved_metrics.items()},
            'components': [c.to_dict() for c in self.component_validations]
        }
    
    def summary(self) -> str:
        """Generate a text summary of validation results"""
        lines = [
            f"Validation Status: {self.status.value.upper()}",
            f"Components Validated: {self.total_components}",
            "",
            "Correlation Metrics:",
            f"  Spearman: {self.spearman_correlation:.3f} (p={self.spearman_p_value:.4f})",
            f"  Pearson:  {self.pearson_correlation:.3f} (p={self.pearson_p_value:.4f})",
            "",
            "Classification Metrics:",
            f"  Precision: {self.confusion_matrix.precision:.3f}",
            f"  Recall:    {self.confusion_matrix.recall:.3f}",
            f"  F1-Score:  {self.confusion_matrix.f1_score:.3f}",
            f"  Accuracy:  {self.confusion_matrix.accuracy:.3f}",
            "",
            "Ranking Metrics:",
            f"  Mean Rank Diff: {self.mean_rank_difference:.1f}",
            f"  Max Rank Diff:  {self.max_rank_difference}",
        ]
        
        for k, overlap in sorted(self.top_k_overlap.items()):
            lines.append(f"  Top-{k} Overlap: {overlap:.1%}")
        
        return "\n".join(lines)


# ============================================================================
# Statistical Functions
# ============================================================================

def spearman_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Calculate Spearman rank correlation coefficient.
    
    Args:
        x: First variable
        y: Second variable
    
    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    
    # Rank the data
    rank_x = _rank_data(x)
    rank_y = _rank_data(y)
    
    # Calculate correlation on ranks
    return pearson_correlation(rank_x, rank_y)


def pearson_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Calculate Pearson correlation coefficient.
    
    Args:
        x: First variable
        y: Second variable
    
    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    
    # Calculate means
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # Calculate covariance and standard deviations
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    
    if std_x == 0 or std_y == 0:
        return 0.0, 1.0
    
    r = cov / (std_x * std_y)
    
    # Calculate p-value using t-distribution approximation
    if abs(r) >= 1.0:
        p_value = 0.0
    else:
        t_stat = r * math.sqrt((n - 2) / (1 - r * r))
        # Approximate p-value (two-tailed)
        p_value = 2 * _t_distribution_cdf(-abs(t_stat), n - 2)
    
    return r, p_value


def _rank_data(data: List[float]) -> List[float]:
    """Rank data with average rank for ties"""
    n = len(data)
    indexed = [(val, i) for i, val in enumerate(data)]
    indexed.sort(key=lambda x: x[0])
    
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        # Find ties
        while j < n - 1 and indexed[j][0] == indexed[j + 1][0]:
            j += 1
        # Assign average rank
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][1]] = avg_rank
        i = j + 1
    
    return ranks


def _t_distribution_cdf(t: float, df: int) -> float:
    """Approximate CDF of t-distribution using normal approximation for large df"""
    if df >= 30:
        # Use normal approximation
        return _normal_cdf(t)
    else:
        # Use beta function approximation
        x = df / (df + t * t)
        return 0.5 * _incomplete_beta(df / 2, 0.5, x)


def _normal_cdf(x: float) -> float:
    """Approximate CDF of standard normal distribution"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """Approximate incomplete beta function (regularized)"""
    # Simple approximation for our use case
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    
    # Use continued fraction approximation
    result = 0.0
    term = 1.0
    for n in range(1, 100):
        term *= (a + n - 1) * x / n
        result += term
        if abs(term) < 1e-10:
            break
    
    return min(1.0, max(0.0, result * x ** a * (1 - x) ** b / a))


# ============================================================================
# Graph Validator
# ============================================================================

class GraphValidator:
    """
    Validates graph-based analysis by comparing predictions with simulations.
    
    The validator:
    1. Runs analysis to get predicted criticality scores
    2. Runs exhaustive simulation to get actual impact scores
    3. Computes correlation and classification metrics
    4. Determines if the analysis meets target thresholds
    """
    
    # Default target thresholds based on research methodology
    DEFAULT_TARGETS = {
        'spearman_correlation': 0.7,
        'f1_score': 0.9,
        'precision': 0.8,
        'recall': 0.8,
        'top_5_overlap': 0.6,
        'top_10_overlap': 0.7
    }
    
    def __init__(self,
                 analyzer: 'GraphAnalyzer',
                 simulator: Optional['GraphSimulator'] = None,
                 targets: Optional[Dict[str, float]] = None,
                 critical_threshold: float = 0.5,
                 seed: Optional[int] = None):
        """
        Initialize the validator.
        
        Args:
            analyzer: GraphAnalyzer instance with data loaded
            simulator: Optional GraphSimulator (created if not provided)
            targets: Target metric thresholds (uses defaults if not provided)
            critical_threshold: Impact threshold to classify as "critical"
            seed: Random seed for reproducibility
        """
        self.analyzer = analyzer
        self.critical_threshold = critical_threshold
        self.seed = seed
        
        # Import here to avoid circular imports
        from src.simulation import GraphSimulator
        self.simulator = simulator or GraphSimulator(seed=seed)
        
        self.targets = {**self.DEFAULT_TARGETS, **(targets or {})}
        self.logger = logging.getLogger('graph_validator')
        
        # Results cache
        self._analysis_result = None
        self._simulation_result = None
        self._validation_result = None
    
    def validate(self,
                 component_types: Optional[List[str]] = None,
                 enable_cascade: bool = False) -> ValidationResult:
        """
        Run full validation comparing analysis with simulation.
        
        Args:
            component_types: Filter validation to specific types
            enable_cascade: Enable cascade in simulation
        
        Returns:
            ValidationResult with all metrics
        """
        self.logger.info("Starting validation...")
        
        # Step 1: Run analysis
        self.logger.info("Running analysis...")
        self._analysis_result = self.analyzer.analyze()
        graph = self.analyzer.G
        
        # Step 2: Run exhaustive simulation
        self.logger.info("Running exhaustive simulation...")
        self._simulation_result = self.simulator.simulate_all_single_failures(
            graph,
            component_types=component_types,
            enable_cascade=enable_cascade
        )
        
        # Step 3: Build comparison data
        predicted, actual = self._build_comparison_data(component_types)
        
        # Step 4: Calculate metrics
        self.logger.info("Calculating validation metrics...")
        result = self._calculate_validation_metrics(predicted, actual)
        
        self._validation_result = result
        return result
    
    def _build_comparison_data(self,
                               component_types: Optional[List[str]] = None
                               ) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """Build predicted and actual data dictionaries"""
        
        # Build predicted scores from analysis
        predicted = {}
        for score in self._analysis_result.criticality_scores:
            if component_types and score.node_type not in component_types:
                continue
            predicted[score.node_id] = {
                'score': score.composite_score,
                'level': score.level.value,
                'type': score.node_type
            }
        
        # Build actual impact from simulation
        actual = {}
        for result in self._simulation_result.results:
            if len(result.failed_components) == 1:
                comp_id = result.failed_components[0]
                if comp_id in predicted:
                    actual[comp_id] = {
                        'impact': result.impact_score,
                        'reachability_loss': result.reachability_loss_pct
                    }
        
        return predicted, actual
    
    def _calculate_validation_metrics(self,
                                      predicted: Dict[str, Dict],
                                      actual: Dict[str, Dict]) -> ValidationResult:
        """Calculate all validation metrics"""
        
        # Get common components
        common = set(predicted.keys()) & set(actual.keys())
        if not common:
            raise ValueError("No common components between analysis and simulation")
        
        # Extract scores
        pred_scores = [predicted[c]['score'] for c in common]
        actual_scores = [actual[c]['impact'] for c in common]
        
        # Calculate correlations
        spearman_r, spearman_p = spearman_correlation(pred_scores, actual_scores)
        pearson_r, pearson_p = pearson_correlation(pred_scores, actual_scores)
        
        # Calculate rankings
        pred_ranking = self._rank_components(predicted)
        actual_ranking = self._rank_components_by_impact(actual)
        
        # Build component validations
        component_validations = []
        for comp_id in common:
            pred_level = self._classify_level(predicted[comp_id]['score'])
            actual_level = self._classify_level_by_impact(actual[comp_id]['impact'])
            
            cv = ComponentValidation(
                component_id=comp_id,
                component_type=predicted[comp_id]['type'],
                predicted_score=predicted[comp_id]['score'],
                predicted_rank=pred_ranking[comp_id],
                predicted_level=pred_level,
                actual_impact=actual[comp_id]['impact'],
                actual_rank=actual_ranking[comp_id],
                actual_level=actual_level,
                rank_difference=abs(pred_ranking[comp_id] - actual_ranking[comp_id]),
                score_difference=abs(predicted[comp_id]['score'] - actual[comp_id]['impact']),
                correctly_classified=(pred_level == actual_level)
            )
            component_validations.append(cv)
        
        # Sort by actual impact
        component_validations.sort(key=lambda x: x.actual_impact, reverse=True)
        
        # Calculate confusion matrix for "critical" threshold
        confusion = self._calculate_confusion_matrix(predicted, actual, self.critical_threshold)
        
        # Calculate per-level metrics
        level_metrics = {}
        for level in ['critical', 'high', 'medium', 'low']:
            threshold = getattr(CriticalityThreshold, level.upper()).value
            level_metrics[level] = self._calculate_confusion_matrix(
                predicted, actual, threshold
            )
        
        # Calculate top-k overlap
        top_k_overlap = {}
        for k in [3, 5, 10]:
            if k <= len(common):
                top_k_overlap[k] = self._calculate_top_k_overlap(
                    pred_ranking, actual_ranking, k
                )
        
        # Calculate ranking metrics
        rank_diffs = [cv.rank_difference for cv in component_validations]
        mean_rank_diff = sum(rank_diffs) / len(rank_diffs) if rank_diffs else 0
        max_rank_diff = max(rank_diffs) if rank_diffs else 0
        
        # Determine status
        achieved = {
            'spearman_correlation': spearman_r,
            'f1_score': confusion.f1_score,
            'precision': confusion.precision,
            'recall': confusion.recall,
        }
        if 5 in top_k_overlap:
            achieved['top_5_overlap'] = top_k_overlap[5]
        if 10 in top_k_overlap:
            achieved['top_10_overlap'] = top_k_overlap[10]
        
        status = self._determine_status(achieved)
        
        return ValidationResult(
            timestamp=datetime.now(),
            total_components=len(common),
            spearman_correlation=spearman_r,
            spearman_p_value=spearman_p,
            pearson_correlation=pearson_r,
            pearson_p_value=pearson_p,
            confusion_matrix=confusion,
            level_metrics=level_metrics,
            top_k_overlap=top_k_overlap,
            mean_rank_difference=mean_rank_diff,
            max_rank_difference=max_rank_diff,
            component_validations=component_validations,
            status=status,
            target_metrics=self.targets,
            achieved_metrics=achieved
        )
    
    def _rank_components(self, predicted: Dict[str, Dict]) -> Dict[str, int]:
        """Rank components by predicted score (1 = highest)"""
        sorted_comps = sorted(predicted.keys(), 
                             key=lambda c: predicted[c]['score'], 
                             reverse=True)
        return {comp: rank + 1 for rank, comp in enumerate(sorted_comps)}
    
    def _rank_components_by_impact(self, actual: Dict[str, Dict]) -> Dict[str, int]:
        """Rank components by actual impact (1 = highest)"""
        sorted_comps = sorted(actual.keys(),
                             key=lambda c: actual[c]['impact'],
                             reverse=True)
        return {comp: rank + 1 for rank, comp in enumerate(sorted_comps)}
    
    def _classify_level(self, score: float) -> str:
        """Classify score into criticality level"""
        if score >= 0.7:
            return 'critical'
        elif score >= 0.5:
            return 'high'
        elif score >= 0.3:
            return 'medium'
        else:
            return 'low'
    
    def _classify_level_by_impact(self, impact: float) -> str:
        """Classify impact into criticality level"""
        if impact >= 0.7:
            return 'critical'
        elif impact >= 0.5:
            return 'high'
        elif impact >= 0.3:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_confusion_matrix(self,
                                   predicted: Dict[str, Dict],
                                   actual: Dict[str, Dict],
                                   threshold: float) -> ConfusionMatrix:
        """Calculate confusion matrix for given threshold"""
        cm = ConfusionMatrix()
        
        common = set(predicted.keys()) & set(actual.keys())
        
        for comp_id in common:
            pred_critical = predicted[comp_id]['score'] >= threshold
            actual_critical = actual[comp_id]['impact'] >= threshold
            
            if pred_critical and actual_critical:
                cm.true_positives += 1
            elif pred_critical and not actual_critical:
                cm.false_positives += 1
            elif not pred_critical and actual_critical:
                cm.false_negatives += 1
            else:
                cm.true_negatives += 1
        
        return cm
    
    def _calculate_top_k_overlap(self,
                                pred_ranking: Dict[str, int],
                                actual_ranking: Dict[str, int],
                                k: int) -> float:
        """Calculate overlap between top-k predicted and actual"""
        pred_top_k = {c for c, r in pred_ranking.items() if r <= k}
        actual_top_k = {c for c, r in actual_ranking.items() if r <= k}
        
        overlap = len(pred_top_k & actual_top_k)
        return overlap / k if k > 0 else 0.0
    
    def _determine_status(self, achieved: Dict[str, float]) -> ValidationStatus:
        """Determine overall validation status"""
        passed_count = 0
        total_count = 0
        
        for metric, target in self.targets.items():
            if metric in achieved:
                total_count += 1
                if achieved[metric] >= target:
                    passed_count += 1
        
        if total_count == 0:
            return ValidationStatus.FAILED
        
        ratio = passed_count / total_count
        if ratio >= 0.8:
            return ValidationStatus.PASSED
        elif ratio >= 0.5:
            return ValidationStatus.MARGINAL
        else:
            return ValidationStatus.FAILED
    
    def get_misclassified(self) -> List[ComponentValidation]:
        """Get list of misclassified components"""
        if not self._validation_result:
            raise ValueError("Run validate() first")
        
        return [cv for cv in self._validation_result.component_validations
                if not cv.correctly_classified]
    
    def get_high_rank_difference(self, threshold: int = 5) -> List[ComponentValidation]:
        """Get components with rank difference above threshold"""
        if not self._validation_result:
            raise ValueError("Run validate() first")
        
        return [cv for cv in self._validation_result.component_validations
                if cv.rank_difference > threshold]
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        if not self._validation_result:
            raise ValueError("Run validate() first")
        
        result = self._validation_result
        
        # Find notable components
        most_underestimated = max(
            result.component_validations,
            key=lambda cv: cv.actual_impact - cv.predicted_score
        )
        most_overestimated = max(
            result.component_validations,
            key=lambda cv: cv.predicted_score - cv.actual_impact
        )
        
        return {
            'summary': {
                'status': result.status.value,
                'total_components': result.total_components,
                'timestamp': result.timestamp.isoformat()
            },
            'correlation': {
                'spearman': {
                    'value': round(result.spearman_correlation, 4),
                    'target': self.targets.get('spearman_correlation', 0.7),
                    'passed': result.spearman_correlation >= self.targets.get('spearman_correlation', 0.7)
                },
                'pearson': round(result.pearson_correlation, 4)
            },
            'classification': {
                'precision': {
                    'value': round(result.confusion_matrix.precision, 4),
                    'target': self.targets.get('precision', 0.8),
                    'passed': result.confusion_matrix.precision >= self.targets.get('precision', 0.8)
                },
                'recall': {
                    'value': round(result.confusion_matrix.recall, 4),
                    'target': self.targets.get('recall', 0.8),
                    'passed': result.confusion_matrix.recall >= self.targets.get('recall', 0.8)
                },
                'f1_score': {
                    'value': round(result.confusion_matrix.f1_score, 4),
                    'target': self.targets.get('f1_score', 0.9),
                    'passed': result.confusion_matrix.f1_score >= self.targets.get('f1_score', 0.9)
                }
            },
            'ranking': {
                'mean_rank_difference': round(result.mean_rank_difference, 2),
                'max_rank_difference': result.max_rank_difference,
                'top_k_overlap': {str(k): round(v, 4) for k, v in result.top_k_overlap.items()}
            },
            'notable_components': {
                'most_underestimated': {
                    'id': most_underestimated.component_id,
                    'predicted': round(most_underestimated.predicted_score, 4),
                    'actual': round(most_underestimated.actual_impact, 4),
                    'difference': round(most_underestimated.actual_impact - most_underestimated.predicted_score, 4)
                },
                'most_overestimated': {
                    'id': most_overestimated.component_id,
                    'predicted': round(most_overestimated.predicted_score, 4),
                    'actual': round(most_overestimated.actual_impact, 4),
                    'difference': round(most_overestimated.predicted_score - most_overestimated.actual_impact, 4)
                }
            },
            'recommendations': self._generate_recommendations(result)
        }
    
    def _generate_recommendations(self, result: ValidationResult) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if result.spearman_correlation < 0.7:
            recommendations.append(
                f"Spearman correlation ({result.spearman_correlation:.3f}) is below target (0.7). "
                "Consider adjusting weights (α, β, γ) or adding additional metrics."
            )
        
        if result.confusion_matrix.precision < 0.8:
            recommendations.append(
                f"Precision ({result.confusion_matrix.precision:.3f}) is below target (0.8). "
                "The model may be over-predicting critical components."
            )
        
        if result.confusion_matrix.recall < 0.8:
            recommendations.append(
                f"Recall ({result.confusion_matrix.recall:.3f}) is below target (0.8). "
                "The model may be missing some critical components."
            )
        
        if result.mean_rank_difference > 3:
            recommendations.append(
                f"Mean rank difference ({result.mean_rank_difference:.1f}) is high. "
                "Consider refining the scoring formula or adding domain-specific weights."
            )
        
        misclassified = [cv for cv in result.component_validations if not cv.correctly_classified]
        if len(misclassified) > len(result.component_validations) * 0.3:
            recommendations.append(
                f"{len(misclassified)} components ({len(misclassified)/len(result.component_validations):.0%}) "
                "were misclassified. Review the criticality thresholds."
            )
        
        if result.status == ValidationStatus.PASSED:
            recommendations.append(
                "Validation passed! The graph-based analysis approach shows strong "
                "correlation with actual failure impact."
            )
        
        return recommendations


# ============================================================================
# Convenience Functions
# ============================================================================

def validate_analysis(analyzer: 'GraphAnalyzer',
                     targets: Optional[Dict[str, float]] = None,
                     seed: Optional[int] = None) -> ValidationResult:
    """
    Convenience function to run validation.
    
    Args:
        analyzer: GraphAnalyzer with data loaded
        targets: Optional target thresholds
        seed: Random seed
    
    Returns:
        ValidationResult
    """
    validator = GraphValidator(analyzer, targets=targets, seed=seed)
    return validator.validate()


def quick_validate(filepath: str,
                  alpha: float = 0.4,
                  beta: float = 0.3,
                  gamma: float = 0.3) -> ValidationResult:
    """
    Quick validation from a JSON file.
    
    Args:
        filepath: Path to input JSON file
        alpha, beta, gamma: Scoring weights
    
    Returns:
        ValidationResult
    """
    from src.analysis import GraphAnalyzer
    
    analyzer = GraphAnalyzer(alpha=alpha, beta=beta, gamma=gamma)
    analyzer.load_from_file(filepath)
    
    return validate_analysis(analyzer, seed=42)