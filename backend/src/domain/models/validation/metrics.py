"""
Validation Metrics Domain Models
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class ValidationTargets:
    """Target thresholds for validation success.

    Primary gates (block pass/fail):
        spearman, spearman_p_max, f1_score, top_5_overlap, rmse_max
    Reported (computed but do not block):
        All remaining fields.

    Default values are aligned with the documented targets in validation.md,
    SRS §6.1, and STD §8.1.
    """
    # Correlation
    spearman: float = 0.70
    spearman_p_max: float = 0.05   # statistical significance gate
    pearson: float = 0.65
    kendall: float = 0.50
    # Classification
    f1_score: float = 0.80
    precision: float = 0.80
    recall: float = 0.80
    accuracy: float = 0.75
    cohens_kappa: float = 0.60
    # Ranking
    top_5_overlap: float = 0.40    # aligned with docs (was 0.60)
    top_10_overlap: float = 0.50
    ndcg_10: float = 0.70
    # Error
    rmse_max: float = 0.25
    mae_max: float = 0.20

    def to_dict(self) -> Dict[str, float]:
        return {
            "spearman": self.spearman,
            "spearman_p_max": self.spearman_p_max,
            "pearson": self.pearson,
            "kendall": self.kendall,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "cohens_kappa": self.cohens_kappa,
            "top_5_overlap": self.top_5_overlap,
            "top_10_overlap": self.top_10_overlap,
            "rmse_max": self.rmse_max,
            "mae_max": self.mae_max,
        }


@dataclass
class CorrelationMetrics:
    """Correlation coefficients with confidence intervals."""
    spearman: float = 0.0
    spearman_p: float = 1.0
    spearman_ci_lower: float = 0.0
    spearman_ci_upper: float = 0.0
    pearson: float = 0.0
    pearson_p: float = 1.0
    kendall: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spearman": round(self.spearman, 4),
            "spearman_p_value": round(self.spearman_p, 6),
            "spearman_ci": [round(self.spearman_ci_lower, 4),
                            round(self.spearman_ci_upper, 4)],
            "pearson": round(self.pearson, 4),
            "pearson_p_value": round(self.pearson_p, 6),
            "kendall": round(self.kendall, 4),
        }


@dataclass
class ErrorMetrics:
    """Error measurements including normalised RMSE."""
    rmse: float = 0.0
    nrmse: float = 0.0
    mae: float = 0.0
    mse: float = 0.0
    max_error: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "rmse": round(self.rmse, 4),
            "nrmse": round(self.nrmse, 4),
            "mae": round(self.mae, 4),
            "mse": round(self.mse, 4),
            "max_error": round(self.max_error, 4),
        }


@dataclass
class ClassificationMetrics:
    """Binary classification metrics with chance-corrected agreement."""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    f1_ci_lower: float = 0.0
    f1_ci_upper: float = 0.0
    accuracy: float = 0.0
    cohens_kappa: float = 0.0
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
            "f1_ci": [round(self.f1_ci_lower, 4),
                       round(self.f1_ci_upper, 4)],
            "accuracy": round(self.accuracy, 4),
            "cohens_kappa": round(self.cohens_kappa, 4),
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