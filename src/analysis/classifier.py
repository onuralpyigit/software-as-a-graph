"""
Box-Plot Classifier

Statistical classification using box-plot method with adaptive thresholds.
Provides data-driven criticality classification without arbitrary cutoffs.

Classification Levels:
    CRITICAL : score > Q3 + k*IQR (upper outliers)
    HIGH     : score > Q3 (top quartile)
    MEDIUM   : score > Median (above average)
    LOW      : score > Q1 (below average)
    MINIMAL  : score <= Q1 (bottom quartile)

Author: Software-as-a-Graph Research Project
"""

from __future__ import annotations
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Sequence, Optional, Callable


class CriticalityLevel(Enum):
    """Criticality classification levels with comparison support."""
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"
    
    @property
    def numeric(self) -> int:
        """Numeric value for comparison (higher = more critical)."""
        return {"critical": 5, "high": 4, "medium": 3, "low": 2, "minimal": 1}[self.value]
    
    @property
    def symbol(self) -> str:
        """Single-character symbol for compact display."""
        return {"critical": "C", "high": "H", "medium": "M", "low": "L", "minimal": "-"}[self.value]
    
    @property
    def color_code(self) -> str:
        """ANSI color code for terminal output."""
        return {
            "critical": "\033[91m",  # Red
            "high": "\033[93m",      # Yellow
            "medium": "\033[94m",    # Blue
            "low": "\033[37m",       # White
            "minimal": "\033[90m"    # Gray
        }[self.value]
    
    def __ge__(self, other: CriticalityLevel) -> bool:
        return self.numeric >= other.numeric
    
    def __gt__(self, other: CriticalityLevel) -> bool:
        return self.numeric > other.numeric
    
    def __le__(self, other: CriticalityLevel) -> bool:
        return self.numeric <= other.numeric
    
    def __lt__(self, other: CriticalityLevel) -> bool:
        return self.numeric < other.numeric


@dataclass
class BoxPlotStats:
    """Box-plot statistics for a score distribution."""
    
    q1: float = 0.0
    median: float = 0.0
    q3: float = 0.0
    iqr: float = 0.0
    lower_fence: float = 0.0
    upper_fence: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    mean: float = 0.0
    std_dev: float = 0.0
    count: int = 0
    k_factor: float = 1.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "q1": round(self.q1, 6),
            "median": round(self.median, 6),
            "q3": round(self.q3, 6),
            "iqr": round(self.iqr, 6),
            "lower_fence": round(self.lower_fence, 6),
            "upper_fence": round(self.upper_fence, 6),
            "min": round(self.min_val, 6),
            "max": round(self.max_val, 6),
            "mean": round(self.mean, 6),
            "std_dev": round(self.std_dev, 6),
            "count": self.count,
            "k_factor": self.k_factor,
        }
    
    def describe(self) -> str:
        """Human-readable description of thresholds."""
        return (
            f"Thresholds: CRITICAL>{self.upper_fence:.3f}, "
            f"HIGH>{self.q3:.3f}, MEDIUM>{self.median:.3f}, "
            f"LOW>{self.q1:.3f}, MINIMAL≤{self.q1:.3f}"
        )


@dataclass
class ClassifiedItem:
    """A single classified item with its criticality level."""
    
    id: str
    score: float
    level: CriticalityLevel
    percentile: float = 0.0
    z_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "score": round(self.score, 6),
            "level": self.level.value,
            "percentile": round(self.percentile, 2),
            "z_score": round(self.z_score, 3),
        }


@dataclass
class ClassificationResult:
    """Result of classifying a set of items."""
    
    metric_name: str
    items: List[ClassifiedItem]
    stats: BoxPlotStats
    distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "statistics": self.stats.to_dict(),
            "distribution": self.distribution,
            "items": [item.to_dict() for item in self.items],
        }
    
    def get_by_level(self, level: CriticalityLevel) -> List[ClassifiedItem]:
        """Get all items at a specific criticality level."""
        return [item for item in self.items if item.level == level]
    
    def get_critical_and_high(self) -> List[ClassifiedItem]:
        """Get items requiring attention (CRITICAL or HIGH)."""
        return [item for item in self.items if item.level >= CriticalityLevel.HIGH]
    
    @property
    def critical_count(self) -> int:
        return self.distribution.get("critical", 0)
    
    @property
    def high_count(self) -> int:
        return self.distribution.get("high", 0)


class BoxPlotClassifier:
    """
    Classifier using box-plot statistics for adaptive threshold determination.
    
    The box-plot method determines criticality levels based on the actual
    distribution of scores rather than arbitrary fixed thresholds.
    
    Attributes:
        k_factor: Multiplier for IQR to determine outlier fence (default: 1.5)
    """
    
    def __init__(self, k_factor: float = 1.5):
        """
        Initialize the classifier.
        
        Args:
            k_factor: IQR multiplier for outlier detection. 
                      1.5 = standard outliers, 3.0 = extreme outliers
        """
        self.k_factor = k_factor
    
    def compute_stats(self, scores: Sequence[float]) -> BoxPlotStats:
        """
        Compute box-plot statistics for a sequence of scores.
        
        Args:
            scores: Sequence of numeric scores
            
        Returns:
            BoxPlotStats containing quartiles, fences, and descriptive stats
        """
        if not scores or len(scores) == 0:
            return BoxPlotStats(k_factor=self.k_factor)
        
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        
        # Handle edge cases
        if n == 1:
            val = sorted_scores[0]
            return BoxPlotStats(
                q1=val, median=val, q3=val, iqr=0.0,
                lower_fence=val, upper_fence=val,
                min_val=val, max_val=val, mean=val, std_dev=0.0,
                count=1, k_factor=self.k_factor
            )
        
        # Compute quartiles using linear interpolation
        def percentile(p: float) -> float:
            k = (n - 1) * p
            f = int(k)
            c = f + 1 if f + 1 < n else f
            return sorted_scores[f] + (k - f) * (sorted_scores[c] - sorted_scores[f])
        
        q1 = percentile(0.25)
        median = percentile(0.50)
        q3 = percentile(0.75)
        iqr = q3 - q1
        
        # Compute fences
        lower_fence = q1 - self.k_factor * iqr
        upper_fence = q3 + self.k_factor * iqr
        
        # Descriptive stats
        mean = statistics.mean(sorted_scores)
        std_dev = statistics.stdev(sorted_scores) if n > 1 else 0.0
        
        return BoxPlotStats(
            q1=q1, median=median, q3=q3, iqr=iqr,
            lower_fence=lower_fence, upper_fence=upper_fence,
            min_val=sorted_scores[0], max_val=sorted_scores[-1],
            mean=mean, std_dev=std_dev, count=n, k_factor=self.k_factor
        )
    
    def classify_score(self, score: float, stats: BoxPlotStats) -> CriticalityLevel:
        """
        Classify a single score based on box-plot statistics.
        
        Args:
            score: The score to classify
            stats: Pre-computed box-plot statistics
            
        Returns:
            CriticalityLevel for the score
        """
        if stats.count == 0:
            return CriticalityLevel.MINIMAL
        
        # Handle uniform distribution (all same values)
        if stats.iqr == 0:
            if score > stats.median:
                return CriticalityLevel.HIGH
            return CriticalityLevel.MEDIUM
        
        if score > stats.upper_fence:
            return CriticalityLevel.CRITICAL
        elif score > stats.q3:
            return CriticalityLevel.HIGH
        elif score > stats.median:
            return CriticalityLevel.MEDIUM
        elif score > stats.q1:
            return CriticalityLevel.LOW
        else:
            return CriticalityLevel.MINIMAL
    
    def classify(
        self, 
        data: List[Dict[str, Any]], 
        metric_name: str = "score",
        id_key: str = "id",
        score_key: str = "score"
    ) -> ClassificationResult:
        """
        Classify a list of items based on their scores.
        
        Args:
            data: List of dicts containing id and score
            metric_name: Name of the metric being classified
            id_key: Key for item identifier in data dicts
            score_key: Key for score value in data dicts
            
        Returns:
            ClassificationResult with classified items and statistics
        """
        if not data:
            return ClassificationResult(
                metric_name=metric_name,
                items=[],
                stats=BoxPlotStats(k_factor=self.k_factor),
                distribution={level.value: 0 for level in CriticalityLevel}
            )
        
        # Extract scores
        scores = [item[score_key] for item in data]
        stats = self.compute_stats(scores)
        
        # Classify each item
        classified_items = []
        distribution = {level.value: 0 for level in CriticalityLevel}
        
        for item in data:
            item_id = item[id_key]
            score = item[score_key]
            level = self.classify_score(score, stats)
            
            # Compute percentile and z-score
            percentile = self._compute_percentile(score, scores)
            z_score = (score - stats.mean) / stats.std_dev if stats.std_dev > 0 else 0.0
            
            classified_items.append(ClassifiedItem(
                id=item_id,
                score=score,
                level=level,
                percentile=percentile,
                z_score=z_score
            ))
            distribution[level.value] += 1
        
        # Sort by score descending (most critical first)
        classified_items.sort(key=lambda x: x.score, reverse=True)
        
        return ClassificationResult(
            metric_name=metric_name,
            items=classified_items,
            stats=stats,
            distribution=distribution
        )
    
    def _compute_percentile(self, score: float, all_scores: List[float]) -> float:
        """Compute the percentile rank of a score."""
        below = sum(1 for s in all_scores if s < score)
        equal = sum(1 for s in all_scores if s == score)
        return 100.0 * (below + 0.5 * equal) / len(all_scores)
    
    def classify_multi_dimensional(
        self,
        items: List[Dict[str, Any]],
        dimensions: List[str],
        id_key: str = "id"
    ) -> Dict[str, ClassificationResult]:
        """
        Classify items across multiple score dimensions.
        
        Args:
            items: List of dicts with id and multiple score dimensions
            dimensions: List of score dimension names to classify
            id_key: Key for item identifier
            
        Returns:
            Dict mapping dimension name to ClassificationResult
        """
        results = {}
        for dim in dimensions:
            data = [{"id": item[id_key], "score": item.get(dim, 0.0)} for item in items]
            results[dim] = self.classify(data, metric_name=dim)
        return results


def combine_levels(levels: List[CriticalityLevel], method: str = "max") -> CriticalityLevel:
    """
    Combine multiple criticality levels into a single level.
    
    Args:
        levels: List of CriticalityLevel values
        method: Combination method ('max', 'min', 'avg')
        
    Returns:
        Combined CriticalityLevel
    """
    if not levels:
        return CriticalityLevel.MINIMAL
    
    if method == "max":
        return max(levels, key=lambda x: x.numeric)
    elif method == "min":
        return min(levels, key=lambda x: x.numeric)
    elif method == "avg":
        avg_numeric = sum(l.numeric for l in levels) / len(levels)
        for level in [CriticalityLevel.CRITICAL, CriticalityLevel.HIGH, 
                      CriticalityLevel.MEDIUM, CriticalityLevel.LOW]:
            if avg_numeric >= level.numeric:
                return level
        return CriticalityLevel.MINIMAL
    else:
        raise ValueError(f"Unknown combination method: {method}")