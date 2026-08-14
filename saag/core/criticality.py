"""
Criticality Domain Models

Defines criticality levels and classification data structures.

Criticality is the degree to which a fault reduces the system's capacity to enable
its stakeholders to achieve their goals — the Quality-in-Use outcomes of ISO/IEC
25019:2023 (SQuaRE). D1 scopes it to a component failing, slowing or degrading, directly
or transitively; D2 to the disruption of a single dependency while both endpoints
keep running, in proportion to the absence of a fallback path. It is a consequence,
carrying no estimate of how likely the fault is (D3), and relative to one system and
one layer rather than absolute (D4). Definitions D1–D4 are in docs/criticality.md.

The construct spans three quality views and they are not interchangeable: it is
computed from internal quality evidence (topology plus static code metrics),
estimates the loss of external quality attributes (ISO/IEC 25010:2023 — fault
tolerance, modifiability, availability, and confidentiality/integrity, one per RM
dimension), and is defined on Quality-in-Use. Only the first of those transitions is
measured anywhere in this project; see docs/criticality.md §7.1.

It is computed, never asserted: no entity carries a hand-assigned criticality.
Each score is a function of two inputs, an entity's position in the layer-projected
dependency graph and the QoS-derived weights w(v) / w(e) that Step 1 attaches to it
(docs/graph-model.md §4.3–§4.5), so structure says how many outcomes route through
an entity and weight says how strongly each of those outcomes was guaranteed.

Note that the ``weight`` fields of ComponentMetrics/EdgeMetrics (QoS-derived, per
entity) and the QualityWeights coefficients (AHP-derived, per model) are unrelated
despite the shared word — see docs/criticality.md §4.4.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any


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


class CompatNamespace:
    """Namespace that supports both attribute access and dict-like item access/methods."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __getitem__(self, key):
        return getattr(self, key)
    def get(self, key, default=None):
        return getattr(self, key, default)
    def keys(self):
        return self.__dict__.keys()
    def values(self):
        return self.__dict__.values()
    def items(self):
        return self.__dict__.items()
    def __iter__(self):
        return iter(self.__dict__)
    def to_dict(self) -> dict:
        return self.__dict__.copy()
    def __repr__(self) -> str:
        return f"CompatNamespace({self.__dict__})"


@dataclass
class CriticalityRanking:
    """
    Unified Data Transfer Object representing a component's criticality score.

    ``scores`` holds Reliability and Maintainability plus Reliability's two
    sub-characteristics (fault_tolerance, availability) and the composite,
    each computed over the QoS-weighted dependency graph; ``levels`` holds
    the box-plot tier per dimension, which is relative to this system's own
    distribution.
    """
    id: str
    type: str
    scores: Dict[str, float]  # reliability, maintainability, fault_tolerance, availability, overall
    levels: Dict[str, str]    # reliability, maintainability, fault_tolerance, availability, overall
    overall: float
    level: str
    provenance: str           # "rm" or "gnn"
    name: str = ""
    blast_radius: int = 0
    cascade_depth: int = 0
    is_articulation_point: bool = False

