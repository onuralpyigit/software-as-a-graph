"""
Validator

Compares predicted scores (from Analysis) against actual impact scores (from Simulation).
Validates by Component Type and Layer.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from .metrics import (
    ValidationTargets, CorrelationMetrics, ClassificationMetrics, RankingMetrics,
    spearman_correlation, pearson_correlation, kendall_correlation,
    calculate_classification, calculate_overlap
)

@dataclass
class ValidationGroupResult:
    """Result for a specific group (e.g., Application components)."""
    group_name: str
    sample_size: int
    correlation: CorrelationMetrics
    classification: ClassificationMetrics
    ranking: RankingMetrics
    passed: bool

@dataclass
class ValidationResult:
    """Overall validation result container."""
    timestamp: str
    overall: ValidationGroupResult
    by_type: Dict[str, ValidationGroupResult]
    by_layer: Dict[str, ValidationGroupResult]
    targets: ValidationTargets
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall": self._group_to_dict(self.overall),
            "by_type": {k: self._group_to_dict(v) for k, v in self.by_type.items()},
            "by_layer": {k: self._group_to_dict(v) for k, v in self.by_layer.items()}
        }
    
    def _group_to_dict(self, g: ValidationGroupResult) -> Dict[str, Any]:
        return {
            "n": g.sample_size,
            "passed": g.passed,
            "spearman": round(g.correlation.spearman, 4),
            "f1": round(g.classification.f1_score, 4),
            "top10_overlap": round(g.ranking.top_10_overlap, 4)
        }

class Validator:
    def __init__(self, targets: Optional[ValidationTargets] = None):
        self.targets = targets or ValidationTargets()
        self.logger = logging.getLogger(__name__)

    def validate(
        self, 
        predicted_scores: Dict[str, float], 
        actual_scores: Dict[str, float],
        metadata: Dict[str, Dict[str, Any]]
    ) -> ValidationResult:
        """
        Validate predicted vs actual scores.
        metadata: {node_id: {'type': 'Application', 'layer': 'application'}}
        """
        from datetime import datetime
        
        # 1. Validate Overall
        overall_res = self._validate_subset("Overall", predicted_scores, actual_scores)
        
        # 2. Validate by Type
        by_type = {}
        types = set(m.get('type') for m in metadata.values())
        for t in types:
            nodes = [n for n, m in metadata.items() if m.get('type') == t]
            if len(nodes) > 2:
                pred_sub = {n: predicted_scores.get(n, 0) for n in nodes}
                act_sub = {n: actual_scores.get(n, 0) for n in nodes}
                by_type[t] = self._validate_subset(t, pred_sub, act_sub)

        # 3. Validate by Layer
        by_layer = {}
        layers = set(m.get('layer') for m in metadata.values() if m.get('layer'))
        for l in layers:
            nodes = [n for n, m in metadata.items() if m.get('layer') == l]
            if len(nodes) > 2:
                pred_sub = {n: predicted_scores.get(n, 0) for n in nodes}
                act_sub = {n: actual_scores.get(n, 0) for n in nodes}
                by_layer[l] = self._validate_subset(l, pred_sub, act_sub)

        return ValidationResult(
            timestamp=datetime.now().isoformat(),
            overall=overall_res,
            by_type=by_type,
            by_layer=by_layer,
            targets=self.targets
        )

    def _validate_subset(self, name: str, pred: Dict[str, float], act: Dict[str, float]) -> ValidationGroupResult:
        common = set(pred.keys()) & set(act.keys())
        if not common:
            return self._empty_result(name)
            
        # Align lists
        ids = sorted(list(common))
        p_vals = [pred[i] for i in ids]
        a_vals = [act[i] for i in ids]
        
        # 1. Correlation
        corr = CorrelationMetrics(
            spearman=spearman_correlation(p_vals, a_vals),
            pearson=pearson_correlation(p_vals, a_vals),
            kendall=kendall_correlation(p_vals, a_vals)
        )
        
        # 2. Classification (Top 25% considered critical)
        cls = calculate_classification(p_vals, a_vals, threshold_pct=75)
        
        # 3. Ranking
        # Sort IDs by score descending
        p_rank = sorted(ids, key=lambda x: pred[x], reverse=True)
        a_rank = sorted(ids, key=lambda x: act[x], reverse=True)
        
        rank = RankingMetrics(
            top_5_overlap=calculate_overlap(p_rank, a_rank, 5),
            top_10_overlap=calculate_overlap(p_rank, a_rank, 10),
            ndcg=0.0 # Placeholder
        )
        
        # Pass/Fail logic
        passed = (
            corr.spearman >= self.targets.spearman and 
            cls.f1_score >= self.targets.f1_score
        )
        
        return ValidationGroupResult(name, len(ids), corr, cls, rank, passed)

    def _empty_result(self, name: str) -> ValidationGroupResult:
        return ValidationGroupResult(
            name, 0, 
            CorrelationMetrics(0,0,0), 
            ClassificationMetrics(0,0,0,0,{}), 
            RankingMetrics(0,0,0), 
            False
        )