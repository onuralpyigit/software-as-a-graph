"""
Box-Plot Classifier

Statistical classification using box-plot method with adaptive thresholds.
Provides data-driven criticality classification without arbitrary cutoffs.

Classification Levels (from score distribution):
    CRITICAL : score > Q3 + k×IQR (statistical outliers)
    HIGH     : score > Q3 (top quartile)
    MEDIUM   : score > Median (above average)
    LOW      : score > Q1 (below average)
    MINIMAL  : score ≤ Q1 (bottom quartile)

Benefits over static thresholds:
    - Adaptive: Adjusts to each dataset's distribution
    - No magic numbers: Avoids arbitrary cutoffs like "0.7 = critical"
    - Statistically grounded: Based on well-understood descriptive statistics
    - Scale-independent: Works regardless of absolute score magnitudes
"""

from __future__ import annotations
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Sequence, Callable, Optional


class CriticalityLevel(Enum):
    """
    Criticality classification levels with comparison support.
    
    Ordered from most critical (CRITICAL) to least (MINIMAL).
    Supports comparison operators for filtering and sorting.
    """
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
        return {"critical": "C", "high": "H", "medium": "M", "low": "L", "minimal": "·"}[self.value]
    
    @property
    def color(self) -> str:
        """ANSI color code for terminal output."""
        return {
            "critical": "\033[91m",  # Red
            "high": "\033[93m",      # Yellow
            "medium": "\033[94m",    # Blue
            "low": "\033[37m",       # White
            "minimal": "\033[90m"    # Gray
        }[self.value]
    
    @property
    def emoji(self) -> str:
        """Emoji indicator for the level."""
        return {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
            "minimal": "⚪"
        }[self.value]
    
    def __ge__(self, other: "CriticalityLevel") -> bool:
        return self.numeric >= other.numeric
    
    def __gt__(self, other: "CriticalityLevel") -> bool:
        return self.numeric > other.numeric
    
    def __le__(self, other: "CriticalityLevel") -> bool:
        return self.numeric <= other.numeric
    
    def __lt__(self, other: "CriticalityLevel") -> bool:
        return self.numeric < other.numeric


@dataclass
class BoxPlotStats:
    """
    Box-plot statistics for a score distribution.
    
    Provides quartiles, fences, and descriptive statistics used for
    adaptive threshold classification.
    """
    q1: float = 0.0           # 25th percentile (first quartile)
    median: float = 0.0       # 50th percentile (Q2)
    q3: float = 0.0           # 75th percentile (third quartile)
    iqr: float = 0.0          # Interquartile range (Q3 - Q1)
    lower_fence: float = 0.0  # Q1 - k×IQR
    upper_fence: float = 0.0  # Q3 + k×IQR (outlier threshold)
    min_val: float = 0.0
    max_val: float = 0.0
    mean: float = 0.0
    std_dev: float = 0.0
    count: int = 0
    k_factor: float = 1.5     # IQR multiplier for outliers
    
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
    
    def describe_thresholds(self) -> str:
        """Human-readable threshold description."""
        return (
            f"CRITICAL>{self.upper_fence:.4f}, "
            f"HIGH>{self.q3:.4f}, "
            f"MEDIUM>{self.median:.4f}, "
            f"LOW>{self.q1:.4f}"
        )


@dataclass
class ClassifiedItem:
    """A single classified item with its score and criticality level."""
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
    """Result of classifying a set of items using box-plot method."""
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
    
    @property
    def requires_attention(self) -> int:
        """Count of items requiring attention (CRITICAL + HIGH)."""
        return self.critical_count + self.high_count


class BoxPlotClassifier:
    """
    Classifier using box-plot statistics for adaptive threshold determination.
    
    The box-plot method determines criticality levels based on the actual
    distribution of scores rather than arbitrary fixed thresholds.
    
    Example:
        >>> classifier = BoxPlotClassifier(k_factor=1.5)
        >>> data = [{"id": "A1", "score": 0.9}, {"id": "A2", "score": 0.3}]
        >>> result = classifier.classify(data)
        >>> print(result.get_critical_and_high())
    """
    
    def __init__(self, k_factor: float = 1.5):
        """
        Initialize the classifier.
        
        Args:
            k_factor: IQR multiplier for outlier detection.
                      1.5 = standard outliers (default)
                      3.0 = extreme outliers only
        """
        self.k_factor = k_factor
    
    def compute_stats(self, scores: Sequence[float]) -> BoxPlotStats:
        """
        Compute box-plot statistics for a sequence of scores.
        
        Args:
            scores: Sequence of numeric scores
            
        Returns:
            BoxPlotStats with quartiles, fences, and descriptive statistics
        """
        if not scores:
            return BoxPlotStats(k_factor=self.k_factor)
        
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        
        # Handle edge case: single value
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
            c = min(f + 1, n - 1)
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
        
        Classification rules:
            CRITICAL: score > Q3 + k×IQR (upper outlier)
            HIGH:     score > Q3 (top quartile, not outlier)
            MEDIUM:   score > Median (above average)
            LOW:      score > Q1 (below average)
            MINIMAL:  score ≤ Q1 (bottom quartile)
        
        Args:
            score: Score to classify
            stats: Pre-computed box-plot statistics
            
        Returns:
            CriticalityLevel classification
        """
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
        data: Sequence[Dict[str, Any]],
        metric_name: str = "score",
        id_key: str = "id",
        score_key: str = "score",
    ) -> ClassificationResult:
        """
        Classify a collection of items based on their scores.
        
        Args:
            data: Sequence of dicts with 'id' and 'score' keys
            metric_name: Name of the metric being classified
            id_key: Key for item identifier
            score_key: Key for score value
            
        Returns:
            ClassificationResult with classified items and statistics
        """
        if not data:
            return ClassificationResult(
                metric_name=metric_name,
                items=[],
                stats=BoxPlotStats(k_factor=self.k_factor),
                distribution={level.value: 0 for level in CriticalityLevel},
            )
        
        # Extract scores and compute statistics
        scores = [item[score_key] for item in data]
        stats = self.compute_stats(scores)
        
        # Classify each item
        items: List[ClassifiedItem] = []
        distribution = {level.value: 0 for level in CriticalityLevel}
        
        for item in data:
            item_id = item[id_key]
            score = item[score_key]
            level = self.classify_score(score, stats)
            
            # Compute percentile and z-score
            percentile = sum(1 for s in scores if s <= score) / len(scores) * 100
            z_score = (score - stats.mean) / stats.std_dev if stats.std_dev > 0 else 0.0
            
            items.append(ClassifiedItem(
                id=item_id,
                score=score,
                level=level,
                percentile=percentile,
                z_score=z_score,
            ))
            distribution[level.value] += 1
        
        # Sort by score descending (most critical first)
        items.sort(key=lambda x: x.score, reverse=True)
        
        return ClassificationResult(
            metric_name=metric_name,
            items=items,
            stats=stats,
            distribution=distribution,
        )


def combine_levels(*levels: CriticalityLevel) -> CriticalityLevel:
    """
    Combine multiple criticality levels by taking the maximum.
    
    Useful for computing overall criticality from R, M, A dimensions.
    
    Args:
        *levels: Variable number of CriticalityLevel values
        
    Returns:
        Highest (most critical) level among inputs
    """
    if not levels:
        return CriticalityLevel.MINIMAL
    return max(levels, key=lambda x: x.numeric)


def weighted_combine(
    levels_weights: List[tuple[CriticalityLevel, float]]
) -> CriticalityLevel:
    """
    Combine multiple criticality levels with weights.
    
    Args:
        levels_weights: List of (level, weight) tuples
        
    Returns:
        CriticalityLevel based on weighted average
    """
    if not levels_weights:
        return CriticalityLevel.MINIMAL
    
    total_weight = sum(w for _, w in levels_weights)
    if total_weight == 0:
        return CriticalityLevel.MINIMAL
    
    weighted_sum = sum(level.numeric * weight for level, weight in levels_weights)
    avg = weighted_sum / total_weight
    
    if avg >= 4.5:
        return CriticalityLevel.CRITICAL
    elif avg >= 3.5:
        return CriticalityLevel.HIGH
    elif avg >= 2.5:
        return CriticalityLevel.MEDIUM
    elif avg >= 1.5:
        return CriticalityLevel.LOW
    else:
        return CriticalityLevel.MINIMAL