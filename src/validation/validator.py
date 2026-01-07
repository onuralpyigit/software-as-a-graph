"""
Validator

Compares Predicted Scores (from Analysis) against Actual Impact (from Simulation).
Implements the Statistical Validation Framework.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.analysis.classifier import BoxPlotClassifier, CriticalityLevel
from .metrics import (
    ValidationTargets, CorrelationMetrics, ClassificationMetrics, RankingMetrics,
    spearman_correlation, pearson_correlation, kendall_correlation,
    calculate_classification_metrics, calculate_ranking_metrics
)

@dataclass
class ValidationGroupResult:
    group_name: str
    sample_size: int
    correlation: CorrelationMetrics
    classification: ClassificationMetrics
    ranking: RankingMetrics
    passed: bool
    
    def to_dict(self):
        return {
            "n": self.sample_size,
            "passed": self.passed,
            "metrics": {
                "rho": round(self.correlation.spearman, 3),
                "f1": round(self.classification.f1_score, 3),
                "precision": round(self.classification.precision, 3),
                "recall": round(self.classification.recall, 3),
                "top5_overlap": round(self.ranking.top_5_overlap, 3),
                "top10_overlap": round(self.ranking.top_10_overlap, 3),
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
            "targets": vars(self.targets),
            "overall": self.overall.to_dict(),
            "by_type": {k: v.to_dict() for k, v in self.by_type.items()}
        }

class Validator:
    def __init__(self, targets: Optional[ValidationTargets] = None):
        self.targets = targets or ValidationTargets()
        self.logger = logging.getLogger(__name__)
        # Box-Plot Classifier (Report Section 4.3.2)
        self.classifier = BoxPlotClassifier(k_factor=1.5)

    def validate(
        self, 
        predicted_scores: Dict[str, float], 
        actual_scores: Dict[str, float],
        component_types: Dict[str, str],
        context: str = "General"
    ) -> ValidationResult:
        """
        Main validation entry point.
        """
        # 1. Overall Validation (All components in scope)
        overall = self._validate_subset("Overall", predicted_scores, actual_scores)
        
        # 2. Per-Type Validation (Drill-down)
        by_type = {}
        unique_types = set(component_types.values())
        
        for c_type in unique_types:
            type_ids = [nid for nid, t in component_types.items() if t == c_type]
            pred_sub = {nid: predicted_scores.get(nid, 0.0) for nid in type_ids if nid in predicted_scores}
            act_sub = {nid: actual_scores.get(nid, 0.0) for nid in type_ids if nid in actual_scores}
            
            # Only validate if we have a statistically relevant sample size
            if len(pred_sub) >= 5: 
                by_type[c_type] = self._validate_subset(c_type, pred_sub, act_sub)

        return ValidationResult(
            timestamp=datetime.now().isoformat(),
            context=context,
            targets=self.targets,
            overall=overall,
            by_type=by_type
        )

    def _validate_subset(self, name: str, pred: Dict[str, float], act: Dict[str, float]) -> ValidationGroupResult:
        common_ids = sorted(list(set(pred.keys()) & set(act.keys())))
        n = len(common_ids)
        
        if n < 2:
            return self._empty_result(name)

        p_vals = [pred[i] for i in common_ids]
        a_vals = [act[i] for i in common_ids]

        # 1. Correlation 
        corr = CorrelationMetrics(
            spearman=spearman_correlation(p_vals, a_vals),
            pearson=pearson_correlation(p_vals, a_vals),
            kendall=kendall_correlation(p_vals, a_vals)
        )

        # 2. Classification (Box-Plot Method)
        # Classify both sets independently using their own distribution
        p_items = [{"id": i, "score": pred[i]} for i in common_ids]
        a_items = [{"id": i, "score": act[i]} for i in common_ids]
        
        p_res = self.classifier.classify(p_items, metric_name="predicted")
        a_res = self.classifier.classify(a_items, metric_name="actual")
        
        p_map = {item.id: item.level for item in p_res.items}
        a_map = {item.id: item.level for item in a_res.items}
        
        # Logic: Is it a CRITICAL/HIGH outlier?
        # We check if level >= HIGH (High or Critical)
        def is_critical(lvl): return lvl >= CriticalityLevel.HIGH
        
        p_binary = [is_critical(p_map[i]) for i in common_ids]
        a_binary = [is_critical(a_map[i]) for i in common_ids]
        
        cls = calculate_classification_metrics(p_binary, a_binary)

        # 3. Ranking
        rank = calculate_ranking_metrics(
            {i: pred[i] for i in common_ids}, 
            {i: act[i] for i in common_ids}
        )

        # Pass criteria
        # Primary Metric: Spearman Rank Correlation
        # Secondary Metric: F1 Score (Accurate identification of critical items)
        passed = (
            corr.spearman >= self.targets.spearman and 
            cls.f1_score >= self.targets.f1_score
        )
        
        return ValidationGroupResult(name, n, corr, cls, rank, passed)

    def _empty_result(self, name):
        return ValidationGroupResult(
            name, 0,
            CorrelationMetrics(0,0,0),
            ClassificationMetrics(0,0,0,0,{}),
            RankingMetrics(0,0,0),
            False
        )