"""
Box-Plot Classifier - Version 6.0

Statistical classification using box-plot method with adaptive thresholds.

The box-plot method avoids arbitrary static thresholds by using the
actual distribution of scores to determine criticality levels.

Classification Levels:
    CRITICAL : score > Q3 + k*IQR (upper outliers)
    HIGH     : score > Q3 (top quartile)
    MEDIUM   : score > Median (above average)
    LOW      : score > Q1 (below average)
    MINIMAL  : score <= Q1 (bottom quartile)

Author: Software-as-a-Graph Research Project
Version: 6.0
"""

from __future__ import annotations
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Sequence, Optional


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
    
    q1: float
    median: float
    q3: float
    iqr: float
    lower_fence: float
    upper_fence: float
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


@dataclass
class ClassifiedItem:
    """A single classified item with its criticality level."""
    
    id: str
    item_type: str
    score: float
    level: CriticalityLevel
    is_outlier: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "type": self.item_type,
            "score": round(self.score, 6),
            "level": self.level.value,
            "is_outlier": self.is_outlier,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class ClassificationResult:
    """Complete classification result with statistics."""
    
    metric_name: str
    items: List[ClassifiedItem]
    stats: BoxPlotStats
    by_level: Dict[CriticalityLevel, List[ClassifiedItem]] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.by_level:
            self.by_level = {level: [] for level in CriticalityLevel}
            for item in self.items:
                self.by_level[item.level].append(item)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric_name,
            "stats": self.stats.to_dict(),
            "count": len(self.items),
            "by_level": {level.value: len(items) for level, items in self.by_level.items()},
            "items": [item.to_dict() for item in self.items],
        }
    
    def get_critical(self) -> List[ClassifiedItem]:
        """Get items classified as CRITICAL."""
        return self.by_level.get(CriticalityLevel.CRITICAL, [])
    
    def get_high_and_above(self) -> List[ClassifiedItem]:
        """Get items classified as HIGH or CRITICAL."""
        return (
            self.by_level.get(CriticalityLevel.CRITICAL, []) +
            self.by_level.get(CriticalityLevel.HIGH, [])
        )
    
    def get_outliers(self) -> List[ClassifiedItem]:
        """Get statistical outliers (both upper and lower)."""
        return [item for item in self.items if item.is_outlier]
    
    def summary(self) -> Dict[str, int]:
        """Get count summary by level."""
        return {level.value: len(items) for level, items in self.by_level.items()}
    
    def top_n(self, n: int = 10) -> List[ClassifiedItem]:
        """Get top N items by score."""
        return sorted(self.items, key=lambda x: x.score, reverse=True)[:n]


class BoxPlotClassifier:
    """
    Statistical classifier using box-plot method.
    
    Uses quartiles and IQR to determine adaptive thresholds
    based on the actual distribution of scores.
    
    Example:
        classifier = BoxPlotClassifier(k_factor=1.5)
        
        items = [
            {"id": "app1", "type": "Application", "score": 0.95},
            {"id": "app2", "type": "Application", "score": 0.45},
        ]
        
        result = classifier.classify(items)
        
        for item in result.get_critical():
            print(f"{item.id}: {item.score:.4f}")
    """
    
    def __init__(self, k_factor: float = 1.5):
        """
        Initialize classifier.
        
        Args:
            k_factor: IQR multiplier for outlier detection.
                     1.5 = standard (mild outliers)
                     3.0 = extreme outliers only
        """
        self.k_factor = k_factor
    
    def classify(
        self,
        items: Sequence[Dict[str, Any]],
        metric_name: str = "score",
        score_key: str = "score",
        id_key: str = "id",
        type_key: str = "type",
    ) -> ClassificationResult:
        """
        Classify items based on their scores.
        
        Args:
            items: Sequence of dicts with id, type, and score
            metric_name: Name for this classification
            score_key: Key for score value
            id_key: Key for item ID
            type_key: Key for item type
        
        Returns:
            ClassificationResult with classified items and statistics
        """
        if not items:
            return self._empty_result(metric_name)
        
        # Extract scores
        scores = [float(item.get(score_key, 0.0)) for item in items]
        
        # Calculate statistics
        stats = self.calculate_stats(scores)
        
        # Classify each item
        classified = []
        for item in items:
            score = float(item.get(score_key, 0.0))
            level, is_outlier = self.classify_score(score, stats)
            
            # Extract any additional metadata
            metadata = {k: v for k, v in item.items() 
                       if k not in {score_key, id_key, type_key}}
            
            classified.append(ClassifiedItem(
                id=str(item.get(id_key, "")),
                item_type=str(item.get(type_key, "unknown")),
                score=score,
                level=level,
                is_outlier=is_outlier,
                metadata=metadata,
            ))
        
        # Sort by score descending
        classified.sort(key=lambda x: x.score, reverse=True)
        
        return ClassificationResult(
            metric_name=metric_name,
            items=classified,
            stats=stats,
        )
    
    def classify_scores(
        self,
        scores: Dict[str, float],
        item_type: str = "component",
        metric_name: str = "score",
    ) -> ClassificationResult:
        """
        Classify a dictionary of id -> score mappings.
        
        Args:
            scores: Dict mapping IDs to scores
            item_type: Type label for all items
            metric_name: Name for classification
        
        Returns:
            ClassificationResult
        """
        items = [{"id": id_, "type": item_type, "score": score} 
                 for id_, score in scores.items()]
        return self.classify(items, metric_name=metric_name)
    
    def calculate_stats(self, scores: List[float]) -> BoxPlotStats:
        """Calculate box-plot statistics for a list of scores."""
        if not scores:
            return BoxPlotStats(
                q1=0, median=0, q3=0, iqr=0,
                lower_fence=0, upper_fence=0,
                k_factor=self.k_factor,
            )
        
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        
        # Quartiles
        q1 = self._percentile(sorted_scores, 25)
        median = self._percentile(sorted_scores, 50)
        q3 = self._percentile(sorted_scores, 75)
        
        # IQR and fences
        iqr = q3 - q1
        lower_fence = q1 - self.k_factor * iqr
        upper_fence = q3 + self.k_factor * iqr
        
        # Additional statistics
        mean = statistics.mean(scores)
        std_dev = statistics.stdev(scores) if n > 1 else 0.0
        
        return BoxPlotStats(
            q1=q1,
            median=median,
            q3=q3,
            iqr=iqr,
            lower_fence=lower_fence,
            upper_fence=upper_fence,
            min_val=min(sorted_scores),
            max_val=max(sorted_scores),
            mean=mean,
            std_dev=std_dev,
            count=n,
            k_factor=self.k_factor,
        )
    
    def classify_score(
        self,
        score: float,
        stats: BoxPlotStats,
    ) -> tuple[CriticalityLevel, bool]:
        """
        Classify a single score given statistics.
        
        Args:
            score: Score to classify
            stats: Pre-calculated box-plot statistics
        
        Returns:
            Tuple of (CriticalityLevel, is_outlier)
        """
        # Check for outliers
        is_upper_outlier = score > stats.upper_fence
        is_lower_outlier = score < stats.lower_fence
        is_outlier = is_upper_outlier or is_lower_outlier
        
        # Determine level
        if is_upper_outlier:
            level = CriticalityLevel.CRITICAL
        elif score > stats.q3:
            level = CriticalityLevel.HIGH
        elif score > stats.median:
            level = CriticalityLevel.MEDIUM
        elif score > stats.q1:
            level = CriticalityLevel.LOW
        else:
            level = CriticalityLevel.MINIMAL
        
        return level, is_outlier
    
    def _percentile(self, sorted_data: List[float], p: float) -> float:
        """Calculate percentile using linear interpolation."""
        if not sorted_data:
            return 0.0
        
        n = len(sorted_data)
        if n == 1:
            return sorted_data[0]
        
        # Linear interpolation
        k = (n - 1) * p / 100
        f = int(k)
        c = min(f + 1, n - 1)
        
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
    
    def _empty_result(self, metric_name: str) -> ClassificationResult:
        """Create empty result for edge cases."""
        return ClassificationResult(
            metric_name=metric_name,
            items=[],
            stats=BoxPlotStats(
                q1=0, median=0, q3=0, iqr=0,
                lower_fence=0, upper_fence=0,
                k_factor=self.k_factor,
            ),
        )


def classify_items(
    items: Sequence[Dict[str, Any]],
    k_factor: float = 1.5,
    metric_name: str = "score",
) -> ClassificationResult:
    """
    Convenience function to classify items.
    
    Args:
        items: Items to classify
        k_factor: Box-plot k factor
        metric_name: Name for the classification
    
    Returns:
        ClassificationResult
    """
    classifier = BoxPlotClassifier(k_factor=k_factor)
    return classifier.classify(items, metric_name=metric_name)