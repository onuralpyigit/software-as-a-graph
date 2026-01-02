"""
Box-Plot Classifier - Version 5.0

Statistical classification using box-plot method with adaptive thresholds.

The box-plot method avoids arbitrary static thresholds by using the
actual distribution of scores to determine criticality levels.

Classification Levels:
- CRITICAL: > Q3 + k*IQR (upper outliers)
- HIGH: > Q3 (top quartile)
- MEDIUM: > Median (above average)
- LOW: > Q1 (below average)
- MINIMAL: <= Q1 (bottom quartile)

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Sequence
import statistics


# =============================================================================
# Enums
# =============================================================================

class CriticalityLevel(Enum):
    """Criticality classification levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"
    
    def __ge__(self, other):
        order = [self.MINIMAL, self.LOW, self.MEDIUM, self.HIGH, self.CRITICAL]
        return order.index(self) >= order.index(other)
    
    def __gt__(self, other):
        order = [self.MINIMAL, self.LOW, self.MEDIUM, self.HIGH, self.CRITICAL]
        return order.index(self) > order.index(other)
    
    def __le__(self, other):
        return not self.__gt__(other)
    
    def __lt__(self, other):
        return not self.__ge__(other)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BoxPlotStats:
    """
    Box-plot statistics for a score distribution.
    
    Attributes:
        q1: First quartile (25th percentile)
        median: Median (50th percentile)
        q3: Third quartile (75th percentile)
        iqr: Interquartile range (Q3 - Q1)
        lower_fence: Lower fence (Q1 - k*IQR)
        upper_fence: Upper fence (Q3 + k*IQR)
        min_val: Minimum value
        max_val: Maximum value
        mean: Mean value
        count: Number of items
    """
    q1: float
    median: float
    q3: float
    iqr: float
    lower_fence: float
    upper_fence: float
    min_val: float = 0.0
    max_val: float = 0.0
    mean: float = 0.0
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
            "count": self.count,
            "k_factor": self.k_factor,
        }


@dataclass
class ClassifiedItem:
    """
    A single classified item.
    
    Attributes:
        id: Item identifier
        item_type: Type of item (e.g., "Application", "Broker")
        score: Raw score value
        level: Assigned criticality level
        is_outlier: Whether this is a statistical outlier
    """
    id: str
    item_type: str
    score: float
    level: CriticalityLevel
    is_outlier: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.item_type,
            "score": round(self.score, 6),
            "level": self.level.value,
            "is_outlier": self.is_outlier,
        }


@dataclass
class ClassificationResult:
    """
    Complete classification result.
    
    Attributes:
        metric_name: Name of the metric being classified
        items: All classified items
        stats: Box-plot statistics
        by_level: Items grouped by criticality level
    """
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
            "by_level": {
                level.value: len(items)
                for level, items in self.by_level.items()
            },
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
        """Get statistical outliers."""
        return [item for item in self.items if item.is_outlier]
    
    def summary(self) -> Dict[str, int]:
        """Get count summary by level."""
        return {
            level.value: len(items)
            for level, items in self.by_level.items()
        }


# =============================================================================
# Box-Plot Classifier
# =============================================================================

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
            {"id": "app3", "type": "Application", "score": 0.25},
        ]
        
        result = classifier.classify(items)
        
        for item in result.get_critical():
            print(f"{item.id}: {item.score:.4f}")
    """
    
    def __init__(self, k_factor: float = 1.5):
        """
        Initialize classifier.
        
        Args:
            k_factor: Multiplier for IQR to determine outlier fences.
                     Default 1.5 is standard for box plots.
                     Use 3.0 for extreme outliers only.
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
        Classify items based on their scores using box-plot method.
        
        Args:
            items: Sequence of dictionaries with id, type, and score
            metric_name: Name for this classification
            score_key: Key for score value in item dict
            id_key: Key for item ID in item dict
            type_key: Key for item type in item dict
        
        Returns:
            ClassificationResult with all items classified
        """
        if not items:
            return self._empty_result(metric_name)
        
        # Extract scores
        scores = [item.get(score_key, 0.0) for item in items]
        
        # Calculate box-plot statistics
        stats = self._calculate_stats(scores)
        
        # Classify each item
        classified_items = []
        for item in items:
            score = item.get(score_key, 0.0)
            level, is_outlier = self._classify_score(score, stats)
            
            classified_items.append(ClassifiedItem(
                id=item.get(id_key, ""),
                item_type=item.get(type_key, "unknown"),
                score=score,
                level=level,
                is_outlier=is_outlier,
            ))
        
        # Sort by score descending
        classified_items.sort(key=lambda x: x.score, reverse=True)
        
        return ClassificationResult(
            metric_name=metric_name,
            items=classified_items,
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
            scores: Dictionary mapping IDs to scores
            item_type: Type label for all items
            metric_name: Name for this classification
        
        Returns:
            ClassificationResult
        """
        items = [
            {"id": id_, "type": item_type, "score": score}
            for id_, score in scores.items()
        ]
        return self.classify(items, metric_name=metric_name)
    
    def _calculate_stats(self, scores: List[float]) -> BoxPlotStats:
        """Calculate box-plot statistics."""
        if not scores:
            return BoxPlotStats(
                q1=0, median=0, q3=0, iqr=0,
                lower_fence=0, upper_fence=0,
                k_factor=self.k_factor
            )
        
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        
        # Calculate quartiles
        q1 = self._percentile(sorted_scores, 25)
        median = self._percentile(sorted_scores, 50)
        q3 = self._percentile(sorted_scores, 75)
        
        # Calculate IQR and fences
        iqr = q3 - q1
        lower_fence = q1 - self.k_factor * iqr
        upper_fence = q3 + self.k_factor * iqr
        
        return BoxPlotStats(
            q1=q1,
            median=median,
            q3=q3,
            iqr=iqr,
            lower_fence=lower_fence,
            upper_fence=upper_fence,
            min_val=min(sorted_scores),
            max_val=max(sorted_scores),
            mean=statistics.mean(sorted_scores),
            count=n,
            k_factor=self.k_factor,
        )
    
    def _percentile(self, sorted_data: List[float], p: float) -> float:
        """Calculate percentile using linear interpolation."""
        if not sorted_data:
            return 0.0
        
        n = len(sorted_data)
        if n == 1:
            return sorted_data[0]
        
        k = (n - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < n else f
        
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
    
    def _classify_score(
        self,
        score: float,
        stats: BoxPlotStats,
    ) -> tuple[CriticalityLevel, bool]:
        """
        Classify a single score.
        
        Returns:
            Tuple of (level, is_outlier)
        """
        is_outlier = score > stats.upper_fence or score < stats.lower_fence
        
        if score > stats.upper_fence:
            return CriticalityLevel.CRITICAL, True
        elif score > stats.q3:
            return CriticalityLevel.HIGH, False
        elif score > stats.median:
            return CriticalityLevel.MEDIUM, False
        elif score > stats.q1:
            return CriticalityLevel.LOW, False
        else:
            return CriticalityLevel.MINIMAL, is_outlier
    
    def _empty_result(self, metric_name: str) -> ClassificationResult:
        """Create empty classification result."""
        return ClassificationResult(
            metric_name=metric_name,
            items=[],
            stats=BoxPlotStats(
                q1=0, median=0, q3=0, iqr=0,
                lower_fence=0, upper_fence=0,
                k_factor=self.k_factor
            ),
        )


# =============================================================================
# Utility Functions
# =============================================================================

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


def get_level_for_score(
    score: float,
    all_scores: List[float],
    k_factor: float = 1.5,
) -> CriticalityLevel:
    """
    Get criticality level for a single score given the distribution.
    
    Args:
        score: Score to classify
        all_scores: All scores in the distribution
        k_factor: Box-plot k factor
    
    Returns:
        CriticalityLevel
    """
    classifier = BoxPlotClassifier(k_factor=k_factor)
    stats = classifier._calculate_stats(all_scores)
    level, _ = classifier._classify_score(score, stats)
    return level
