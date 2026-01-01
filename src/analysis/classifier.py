"""
Box-Plot Criticality Classifier - Version 5.0

Statistical classification using the box-plot (IQR) method.

Why Box-Plot Method?
- Adapts to actual data distribution (no fixed thresholds)
- Statistically meaningful boundaries based on quartiles
- Natural outlier detection via IQR fences
- Comparable across different systems and scales

Box-Plot Statistics:
    Q1 = 25th percentile (first quartile)
    Q3 = 75th percentile (third quartile)
    IQR = Q3 - Q1 (interquartile range)
    Lower Fence = Q1 - k × IQR
    Upper Fence = Q3 + k × IQR

Classification Levels:
    CRITICAL: score > Upper Fence (statistical outliers)
    HIGH:     Q3 < score ≤ Upper Fence
    MEDIUM:   Median < score ≤ Q3
    LOW:      Q1 < score ≤ Median
    MINIMAL:  score ≤ Q1

Parameters:
    k_factor: IQR multiplier for fences
        - 1.5 = standard (detects mild outliers)
        - 3.0 = conservative (only extreme outliers)
        - 1.0 = aggressive (more outliers)

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


# =============================================================================
# Enums
# =============================================================================

class CriticalityLevel(Enum):
    """
    Criticality levels based on box-plot classification.
    
    Levels are defined by quartile boundaries:
    - CRITICAL: Upper outliers (above Q3 + k×IQR)
    - HIGH: Between Q3 and upper fence
    - MEDIUM: Between median and Q3
    - LOW: Between Q1 and median
    - MINIMAL: Below Q1
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"
    
    @property
    def numeric(self) -> int:
        """Numeric value for sorting (higher = more critical)"""
        return {
            CriticalityLevel.CRITICAL: 5,
            CriticalityLevel.HIGH: 4,
            CriticalityLevel.MEDIUM: 3,
            CriticalityLevel.LOW: 2,
            CriticalityLevel.MINIMAL: 1,
        }[self]
    
    @property
    def color(self) -> str:
        """ANSI color code for terminal output"""
        return {
            CriticalityLevel.CRITICAL: "\033[91m",  # Red
            CriticalityLevel.HIGH: "\033[93m",      # Yellow
            CriticalityLevel.MEDIUM: "\033[94m",    # Blue
            CriticalityLevel.LOW: "\033[92m",       # Green
            CriticalityLevel.MINIMAL: "\033[90m",   # Gray
        }[self]
    
    @property
    def description(self) -> str:
        """Human-readable description"""
        return {
            CriticalityLevel.CRITICAL: "Statistical outlier - requires immediate attention",
            CriticalityLevel.HIGH: "Above Q3 - significant concern",
            CriticalityLevel.MEDIUM: "Above median - moderate concern",
            CriticalityLevel.LOW: "Below median - low concern",
            CriticalityLevel.MINIMAL: "Below Q1 - minimal concern",
        }[self]
    
    def __lt__(self, other: "CriticalityLevel") -> bool:
        return self.numeric < other.numeric
    
    def __le__(self, other: "CriticalityLevel") -> bool:
        return self.numeric <= other.numeric
    
    def __gt__(self, other: "CriticalityLevel") -> bool:
        return self.numeric > other.numeric
    
    def __ge__(self, other: "CriticalityLevel") -> bool:
        return self.numeric >= other.numeric


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BoxPlotStats:
    """
    Box-plot statistics for a set of scores.
    
    Provides all quartile information needed for classification.
    """
    min_val: float
    q1: float          # 25th percentile
    median: float      # 50th percentile
    q3: float          # 75th percentile
    max_val: float
    iqr: float         # Interquartile range (Q3 - Q1)
    lower_fence: float # Q1 - k × IQR
    upper_fence: float # Q3 + k × IQR
    mean: float
    std_dev: float
    count: int
    k_factor: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "min": round(self.min_val, 4),
            "q1": round(self.q1, 4),
            "median": round(self.median, 4),
            "q3": round(self.q3, 4),
            "max": round(self.max_val, 4),
            "iqr": round(self.iqr, 4),
            "lower_fence": round(self.lower_fence, 4),
            "upper_fence": round(self.upper_fence, 4),
            "mean": round(self.mean, 4),
            "std_dev": round(self.std_dev, 4),
            "count": self.count,
            "k_factor": self.k_factor,
        }
    
    @staticmethod
    def empty(k_factor: float = 1.5) -> "BoxPlotStats":
        """Create empty stats for empty data"""
        return BoxPlotStats(
            min_val=0, q1=0, median=0, q3=0, max_val=0,
            iqr=0, lower_fence=0, upper_fence=0,
            mean=0, std_dev=0, count=0, k_factor=k_factor
        )


@dataclass
class ClassifiedItem:
    """
    A single item with its classification result.
    
    Contains the original item info plus classification metadata.
    """
    id: str
    item_type: str
    score: float
    level: CriticalityLevel
    percentile: float
    rank: int
    is_outlier: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.item_type,
            "score": round(self.score, 6),
            "level": self.level.value,
            "percentile": round(self.percentile, 2),
            "rank": self.rank,
            "is_outlier": self.is_outlier,
            "metadata": self.metadata,
        }


@dataclass
class ClassificationResult:
    """
    Complete classification result for a set of items.
    
    Includes statistics, all classified items, and convenience accessors.
    """
    metric_name: str
    stats: BoxPlotStats
    items: List[ClassifiedItem]
    by_level: Dict[CriticalityLevel, List[ClassifiedItem]]
    summary: Dict[CriticalityLevel, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric_name,
            "statistics": self.stats.to_dict(),
            "summary": {level.value: count for level, count in self.summary.items()},
            "items": [item.to_dict() for item in self.items],
            "by_level": {
                level.value: [item.to_dict() for item in items]
                for level, items in self.by_level.items()
            },
        }
    
    def get_critical(self) -> List[ClassifiedItem]:
        """Get items classified as CRITICAL"""
        return self.by_level.get(CriticalityLevel.CRITICAL, [])
    
    def get_high_and_above(self) -> List[ClassifiedItem]:
        """Get items classified as HIGH or CRITICAL"""
        return (
            self.by_level.get(CriticalityLevel.CRITICAL, []) +
            self.by_level.get(CriticalityLevel.HIGH, [])
        )
    
    def get_by_type(self, item_type: str) -> List[ClassifiedItem]:
        """Get items of a specific type"""
        return [item for item in self.items if item.item_type == item_type]
    
    def top_n(self, n: int = 10) -> List[ClassifiedItem]:
        """Get top N items by score"""
        return self.items[:n]
    
    @property
    def critical_count(self) -> int:
        """Number of critical items"""
        return self.summary.get(CriticalityLevel.CRITICAL, 0)
    
    @property
    def outlier_count(self) -> int:
        """Number of outliers"""
        return sum(1 for item in self.items if item.is_outlier)


# =============================================================================
# Box-Plot Classifier
# =============================================================================

class BoxPlotClassifier:
    """
    Classifies items using the box-plot statistical method.
    
    This classifier uses quartile-based thresholds that adapt to
    the actual data distribution, avoiding arbitrary fixed thresholds.
    
    Example:
        classifier = BoxPlotClassifier(k_factor=1.5)
        
        items = [
            {"id": "A1", "type": "Application", "score": 0.85},
            {"id": "A2", "type": "Application", "score": 0.42},
            {"id": "B1", "type": "Broker", "score": 0.95},
        ]
        
        result = classifier.classify(items, metric_name="betweenness")
        
        for item in result.get_critical():
            print(f"{item.id}: {item.score:.4f} (CRITICAL)")
    """

    def __init__(self, k_factor: float = 1.5):
        """
        Initialize classifier.
        
        Args:
            k_factor: IQR multiplier for fence calculation
                - 1.5 = standard outlier detection
                - 3.0 = conservative (fewer outliers)
                - 1.0 = aggressive (more outliers)
        """
        if k_factor <= 0:
            raise ValueError("k_factor must be positive")
        
        self.k_factor = k_factor
        self.logger = logging.getLogger(__name__)

    def calculate_stats(self, scores: List[float]) -> BoxPlotStats:
        """
        Calculate box-plot statistics for a list of scores.
        
        Args:
            scores: List of numeric scores
        
        Returns:
            BoxPlotStats with all quartile information
        """
        if not scores:
            return BoxPlotStats.empty(self.k_factor)
        
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
        
        # Calculate mean and std dev
        mean = sum(scores) / n
        variance = sum((x - mean) ** 2 for x in scores) / n
        std_dev = variance ** 0.5
        
        return BoxPlotStats(
            min_val=sorted_scores[0],
            q1=q1,
            median=median,
            q3=q3,
            max_val=sorted_scores[-1],
            iqr=iqr,
            lower_fence=lower_fence,
            upper_fence=upper_fence,
            mean=mean,
            std_dev=std_dev,
            count=n,
            k_factor=self.k_factor,
        )

    def _percentile(self, sorted_data: List[float], p: float) -> float:
        """
        Calculate percentile using linear interpolation.
        
        Args:
            sorted_data: Sorted list of values
            p: Percentile (0-100)
        
        Returns:
            Percentile value
        """
        n = len(sorted_data)
        if n == 1:
            return sorted_data[0]
        
        # Use linear interpolation method
        k = (n - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < n else f
        
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    def classify_score(
        self, 
        score: float, 
        stats: BoxPlotStats
    ) -> Tuple[CriticalityLevel, bool]:
        """
        Classify a single score based on box-plot statistics.
        
        Args:
            score: The score to classify
            stats: Box-plot statistics
        
        Returns:
            Tuple of (CriticalityLevel, is_outlier)
        """
        # Handle edge cases
        if stats.count == 0:
            return CriticalityLevel.MEDIUM, False
        
        if stats.iqr == 0:
            # All values are the same - use median comparison
            if score > stats.median:
                return CriticalityLevel.HIGH, False
            elif score < stats.median:
                return CriticalityLevel.LOW, False
            else:
                return CriticalityLevel.MEDIUM, False
        
        # Standard classification
        if score > stats.upper_fence:
            return CriticalityLevel.CRITICAL, True
        elif score > stats.q3:
            return CriticalityLevel.HIGH, False
        elif score > stats.median:
            return CriticalityLevel.MEDIUM, False
        elif score > stats.q1:
            return CriticalityLevel.LOW, False
        else:
            # Check for lower outlier
            is_lower_outlier = score < stats.lower_fence
            return CriticalityLevel.MINIMAL, is_lower_outlier

    def calculate_percentile(
        self, 
        score: float, 
        sorted_scores: List[float]
    ) -> float:
        """
        Calculate the percentile rank of a score.
        
        Args:
            score: The score to rank
            sorted_scores: Sorted list of all scores
        
        Returns:
            Percentile (0-100)
        """
        if not sorted_scores:
            return 50.0
        
        # Count how many scores are below this one
        below = sum(1 for s in sorted_scores if s < score)
        equal = sum(1 for s in sorted_scores if s == score)
        
        # Use midpoint for ties
        percentile = (below + 0.5 * equal) / len(sorted_scores) * 100
        return percentile

    def classify(
        self,
        items: List[Dict[str, Any]],
        score_key: str = "score",
        id_key: str = "id",
        type_key: str = "type",
        metric_name: str = "metric",
    ) -> ClassificationResult:
        """
        Classify a list of items using box-plot method.
        
        Args:
            items: List of dicts with at least id, type, and score
            score_key: Key for score value in each item
            id_key: Key for item ID
            type_key: Key for item type
            metric_name: Name of the metric being classified
        
        Returns:
            ClassificationResult with all classified items
        """
        if not items:
            return ClassificationResult(
                metric_name=metric_name,
                stats=BoxPlotStats.empty(self.k_factor),
                items=[],
                by_level={level: [] for level in CriticalityLevel},
                summary={level: 0 for level in CriticalityLevel},
            )
        
        # Extract and sort scores
        scores = [item[score_key] for item in items]
        sorted_scores = sorted(scores)
        stats = self.calculate_stats(scores)
        
        # Sort items by score descending for ranking
        sorted_items = sorted(items, key=lambda x: x[score_key], reverse=True)
        
        # Classify each item
        classified = []
        by_level: Dict[CriticalityLevel, List[ClassifiedItem]] = defaultdict(list)
        
        for rank, item in enumerate(sorted_items, 1):
            score = item[score_key]
            level, is_outlier = self.classify_score(score, stats)
            percentile = self.calculate_percentile(score, sorted_scores)
            
            # Preserve any extra metadata
            metadata = {k: v for k, v in item.items() 
                       if k not in {score_key, id_key, type_key}}
            
            classified_item = ClassifiedItem(
                id=item[id_key],
                item_type=item.get(type_key, "Unknown"),
                score=score,
                level=level,
                percentile=percentile,
                rank=rank,
                is_outlier=is_outlier,
                metadata=metadata,
            )
            
            classified.append(classified_item)
            by_level[level].append(classified_item)
        
        # Generate summary
        summary = {level: len(by_level[level]) for level in CriticalityLevel}
        
        self.logger.info(
            f"Classified {len(items)} items: "
            f"CRITICAL={summary[CriticalityLevel.CRITICAL]}, "
            f"HIGH={summary[CriticalityLevel.HIGH]}, "
            f"MEDIUM={summary[CriticalityLevel.MEDIUM]}"
        )
        
        return ClassificationResult(
            metric_name=metric_name,
            stats=stats,
            items=classified,
            by_level=dict(by_level),
            summary=summary,
        )

    def classify_by_type(
        self,
        items: List[Dict[str, Any]],
        score_key: str = "score",
        id_key: str = "id",
        type_key: str = "type",
        metric_name: str = "metric",
    ) -> Dict[str, ClassificationResult]:
        """
        Classify items grouped by their type.
        
        Each type gets its own statistics and classification.
        This allows comparing components within the same category.
        
        Args:
            items: List of items to classify
            score_key: Key for score value
            id_key: Key for item ID
            type_key: Key for item type
            metric_name: Name of the metric
        
        Returns:
            Dict mapping type -> ClassificationResult
        """
        # Group items by type
        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in items:
            item_type = item.get(type_key, "Unknown")
            by_type[item_type].append(item)
        
        # Classify each type separately
        results = {}
        for item_type, type_items in by_type.items():
            results[item_type] = self.classify(
                type_items,
                score_key=score_key,
                id_key=id_key,
                type_key=type_key,
                metric_name=f"{metric_name}_{item_type.lower()}",
            )
        
        return results


# =============================================================================
# Utility Functions
# =============================================================================

def merge_classifications(
    classifications: List[ClassificationResult],
    weights: Optional[Dict[str, float]] = None,
) -> ClassificationResult:
    """
    Merge multiple classifications into a composite result.
    
    Combines scores from multiple metrics (e.g., betweenness + pagerank)
    into a single composite classification.
    
    Args:
        classifications: List of classification results
        weights: Optional weights for each classification (by metric name)
    
    Returns:
        Merged ClassificationResult
    """
    if not classifications:
        return ClassificationResult(
            metric_name="composite",
            stats=BoxPlotStats.empty(),
            items=[],
            by_level={level: [] for level in CriticalityLevel},
            summary={level: 0 for level in CriticalityLevel},
        )
    
    if len(classifications) == 1:
        result = classifications[0]
        result.metric_name = "composite"
        return result
    
    # Default to equal weights
    if weights is None:
        weights = {c.metric_name: 1.0 / len(classifications) for c in classifications}
    
    # Normalize weights
    total_weight = sum(weights.get(c.metric_name, 1.0) for c in classifications)
    normalized_weights = {
        c.metric_name: weights.get(c.metric_name, 1.0) / total_weight 
        for c in classifications
    }
    
    # Combine scores for each item
    item_scores: Dict[str, Dict[str, Any]] = {}
    
    for classification in classifications:
        weight = normalized_weights[classification.metric_name]
        for item in classification.items:
            if item.id not in item_scores:
                item_scores[item.id] = {
                    "id": item.id,
                    "type": item.item_type,
                    "score": 0.0,
                    "metadata": item.metadata.copy(),
                }
            item_scores[item.id]["score"] += item.score * weight
    
    # Re-classify with composite scores
    classifier = BoxPlotClassifier()
    return classifier.classify(
        list(item_scores.values()),
        metric_name="composite",
    )
