"""
Validation Metrics Domain Models
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class ValidationTargets:
    """Target thresholds for validation success."""
    spearman: float = 0.70
    pearson: float = 0.65
    kendall: float = 0.50
    f1_score: float = 0.80
    precision: float = 0.80
    recall: float = 0.80
    accuracy: float = 0.75
    top_5_overlap: float = 0.60
    top_10_overlap: float = 0.50
    ndcg_10: float = 0.70
    rmse_max: float = 0.25
    mae_max: float = 0.20
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "spearman": self.spearman,
            "pearson": self.pearson,
            "kendall": self.kendall,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "top_5_overlap": self.top_5_overlap,
            "top_10_overlap": self.top_10_overlap,
            "rmse_max": self.rmse_max,
            "mae_max": self.mae_max,
        }

@dataclass
class CorrelationMetrics:
    """Correlation coefficients."""
    spearman: float = 0.0
    spearman_p: float = 1.0
    pearson: float = 0.0
    pearson_p: float = 1.0
    kendall: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "spearman": round(self.spearman, 4),
            "spearman_p_value": round(self.spearman_p, 6),
            "pearson": round(self.pearson, 4),
            "pearson_p_value": round(self.pearson_p, 6),
            "kendall": round(self.kendall, 4),
        }

@dataclass
class ErrorMetrics:
    """Error measurements."""
    rmse: float = 0.0
    mae: float = 0.0
    mse: float = 0.0
    max_error: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            "mse": round(self.mse, 4),
            "max_error": round(self.max_error, 4),
        }

@dataclass
class ClassificationMetrics:
    """Binary classification metrics."""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    @property
    def confusion_matrix(self) -> Dict[str, int]:
        return {
            "tp": self.true_positives,
            "fp": self.false_positives,
            "tn": self.true_negatives,
            "fn": self.false_negatives,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "accuracy": round(self.accuracy, 4),
            "confusion_matrix": self.confusion_matrix,
        }

@dataclass
class RankingMetrics:
    """Ranking agreement metrics."""
    top_5_overlap: float = 0.0
    top_10_overlap: float = 0.0
    ndcg_5: float = 0.0
    ndcg_10: float = 0.0
    top_5_predicted: List[str] = field(default_factory=list)
    top_5_actual: List[str] = field(default_factory=list)
    top_5_common: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_5_overlap": round(self.top_5_overlap, 4),
            "top_10_overlap": round(self.top_10_overlap, 4),
            "ndcg_5": round(self.ndcg_5, 4),
            "ndcg_10": round(self.ndcg_10, 4),
            "top_5_agreement": {
                "predicted": self.top_5_predicted,
                "actual": self.top_5_actual,
                "common": self.top_5_common,
            },
        }
