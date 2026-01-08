"""
Validator

Compares Predicted Scores (from Analysis) against Actual Impact (from Simulation).
Implements the Statistical Validation Framework.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.analysis.classifier import BoxPlotClassifier, CriticalityLevel
from .metrics import (
    ValidationTargets, CorrelationMetrics, ClassificationMetrics, RankingMetrics, ErrorMetrics,
    spearman_correlation, pearson_correlation, kendall_correlation, calculate_error_metrics,
    calculate_classification_metrics, calculate_ranking_metrics
)

@dataclass
class ValidationGroupResult:
    group_name: str
    sample_size: int
    correlation: CorrelationMetrics
    error: ErrorMetrics
    classification: ClassificationMetrics
    ranking: RankingMetrics
    passed: bool
    
    def to_dict(self):
        return {
            "n": self.sample_size,
            "passed": self.passed,
            "metrics": {
                "rho": round(self.correlation.spearman, 3),
                "rmse": round(self.error.rmse, 3),
                "f1": round(self.classification.f1_score, 3),
                "precision": round(self.classification.precision, 3),
                "recall": round(self.classification.recall, 3),
                "top5_overlap": round(self.ranking.top_5_overlap, 3),
                "top10_overlap": round(self.ranking.top_10_overlap, 3)
            }
        }

@dataclass
class ValidationResult:
    timestamp: str
    context: str
    targets: ValidationTargets
    overall: ValidationGroupResult
    by_type: Dict[str, ValidationGroupResult]
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "context": self.context,
            "targets": asdict(self.targets),
            "overall": self.overall.to_dict(),
            "by_type": {k: v.to_dict() for k, v in self.by_type.items()}
        }

class Validator:
    def __init__(self, targets: Optional[ValidationTargets] = None):
        self.targets = targets or ValidationTargets()
        self.logger = logging.getLogger(__name__)
        # Box-Plot Classifier used to normalize and categorize scores independently
        self.classifier = BoxPlotClassifier(k_factor=1.5)

    def validate(
        self, 
        predicted_scores: Dict[str, float], 
        actual_scores: Dict[str, float],
        component_types: Dict[str, str],
        context: str = "General"
    ) -> ValidationResult:
        """
        Main validation logic.
        Validates the entire set and then breaks down by component type.
        """
        # 1. Overall Validation
        overall = self._validate_subset("Overall", predicted_scores, actual_scores)
        
        # 2. Per-Type Validation
        by_type = {}
        unique_types = set(component_types.values())
        
        for c_type in unique_types:
            # Filter IDs by type
            type_ids = [nid for nid, t in component_types.items() if t == c_type]
            
            pred_sub = {nid: predicted_scores.get(nid, 0.0) for nid in type_ids if nid in predicted_scores}
            act_sub = {nid: actual_scores.get(nid, 0.0) for nid in type_ids if nid in actual_scores}
            
            # Only validate if we have a statistically meaningful sample
            if len(pred_sub) >= 3: 
                by_type[c_type] = self._validate_subset(c_type, pred_sub, act_sub)

        return ValidationResult(
            timestamp=datetime.now().isoformat(),
            context=context,
            targets=self.targets,
            overall=overall,
            by_type=by_type
        )

    def _validate_subset(self, name: str, pred: Dict[str, float], act: Dict[str, float]) -> ValidationGroupResult:
        # 1. Align Data
        common_ids = sorted(list(set(pred.keys()) & set(act.keys())))
        n = len(common_ids)
        
        if n < 2:
            return self._empty_result(name)

        p_vals = [pred[i] for i in common_ids]
        a_vals = [act[i] for i in common_ids]

        # 2. Correlation Analysis
        corr = CorrelationMetrics(
            spearman=spearman_correlation(p_vals, a_vals),
            pearson=pearson_correlation(p_vals, a_vals),
            kendall=kendall_correlation(p_vals, a_vals)
        )
        
        # 3. Error Analysis
        err = calculate_error_metrics(p_vals, a_vals)

        # 4. Classification Analysis (Criticality Detection)
        # Classify independently to handle scale differences
        p_items = [{"id": i, "score": pred[i]} for i in common_ids]
        a_items = [{"id": i, "score": act[i]} for i in common_ids]
        
        p_res = self.classifier.classify(p_items, metric_name="predicted")
        a_res = self.classifier.classify(a_items, metric_name="actual")
        
        p_map = {item.id: item.level for item in p_res.items}
        a_map = {item.id: item.level for item in a_res.items}
        
        # Define "Critical" as HIGH or CRITICAL level in the Box-Plot
        def is_critical(lvl): return lvl >= CriticalityLevel.HIGH
        
        p_binary = [is_critical(p_map[i]) for i in common_ids]
        a_binary = [is_critical(a_map[i]) for i in common_ids]
        
        cls = calculate_classification_metrics(p_binary, a_binary)

        # 5. Ranking Analysis
        rank = calculate_ranking_metrics(
            {i: pred[i] for i in common_ids}, 
            {i: act[i] for i in common_ids}
        )

        # 6. Pass/Fail Decision
        # Primary: Spearman (Trend), Secondary: F1 (Outlier Detection)
        passed = (
            corr.spearman >= self.targets.spearman and 
            cls.f1_score >= self.targets.f1_score
        )
        
        return ValidationGroupResult(name, n, corr, err, cls, rank, passed)

    def _empty_result(self, name):
        return ValidationGroupResult(
            name, 0,
            CorrelationMetrics(0,0,0),
            ErrorMetrics(0,0),
            ClassificationMetrics(0,0,0,0,{}),
            RankingMetrics(0,0,0),
            False
        )