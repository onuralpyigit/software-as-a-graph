"""
Criticality Levels and Classification Data Structures

Defines the levels of criticality and the structures for classification results.
"""

from __future__ import annotations
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional


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
        if isinstance(other, CriticalityLevel):
            return self.numeric >= other.numeric
        return NotImplemented
    
    def __gt__(self, other: "CriticalityLevel") -> bool:
        if isinstance(other, CriticalityLevel):
            return self.numeric > other.numeric
        return NotImplemented
    
    def __le__(self, other: "CriticalityLevel") -> bool:
        if isinstance(other, CriticalityLevel):
            return self.numeric <= other.numeric
        return NotImplemented
    
    def __lt__(self, other: "CriticalityLevel") -> bool:
        if isinstance(other, CriticalityLevel):
            return self.numeric < other.numeric
        return NotImplemented


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
