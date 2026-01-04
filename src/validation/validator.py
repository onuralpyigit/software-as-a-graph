"""
Validator

Compares Predicted Scores (from Analysis) against Actual Impact (from Simulation).
Validates per component type and layer.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime

from .metrics import (
    ValidationTargets, CorrelationMetrics, ClassificationMetrics, RankingMetrics,
    spearman_correlation, pearson_correlation, kendall_correlation,
    calculate_classification, calculate_ranking_metrics
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
                "top10_overlap": round(self.ranking.top_10_overlap, 3),
                "ndcg": round(self.ranking.ndcg_10, 3)
            }
        }

@dataclass
class ValidationResult:
    timestamp: str
    targets: ValidationTargets
    overall: ValidationGroupResult
    by_type: Dict[str, ValidationGroupResult]
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "targets": vars(self.targets),
            "overall": self.overall.to_dict(),
            "by_type": {k: v.to_dict() for k, v in self.by_type.items()}
        }

class Validator:
    def __init__(self, targets: Optional[ValidationTargets] = None):
        self.targets = targets or ValidationTargets()
        self.logger = logging.getLogger(__name__)

    def validate(
        self, 
        predicted_scores: Dict[str, float], 
        actual_scores: Dict[str, float],
        component_types: Dict[str, str]
    ) -> ValidationResult:
        """
        Main validation entry point.
        component_types: map of {id: type_name}
        """
        # 1. Overall Validation
        overall = self._validate_subset("Overall", predicted_scores, actual_scores)
        
        # 2. Per-Type Validation
        by_type = {}
        unique_types = set(component_types.values())
        
        for c_type in unique_types:
            # Filter IDs for this type
            type_ids = [nid for nid, t in component_types.items() if t == c_type]
            
            # Extract subset of scores
            pred_sub = {nid: predicted_scores.get(nid, 0.0) for nid in type_ids if nid in predicted_scores}
            act_sub = {nid: actual_scores.get(nid, 0.0) for nid in type_ids if nid in actual_scores}
            
            if len(pred_sub) > 2: # Min sample size
                by_type[c_type] = self._validate_subset(c_type, pred_sub, act_sub)

        return ValidationResult(
            timestamp=datetime.now().isoformat(),
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

        # 2. Classification (Top 25% Criticality)
        cls = calculate_classification(p_vals, a_vals, top_percentile=0.25)

        # 3. Ranking
        rank = calculate_ranking_metrics(
            {i: pred[i] for i in common_ids}, 
            {i: act[i] for i in common_ids}
        )

        # Pass criteria
        passed = (
            corr.spearman >= self.targets.spearman and 
            rank.top_10_overlap >= self.targets.top_10_overlap
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