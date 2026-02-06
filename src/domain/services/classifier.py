"""
Box-Plot Classifier

Statistical classification using box-plot method with adaptive thresholds.
Provides data-driven criticality classification without arbitrary cutoffs.

Classification levels (from the score distribution):
    CRITICAL : score > Q3 + k×IQR   (statistical outlier)
    HIGH     : Q3 < score ≤ upper    (top quartile)
    MEDIUM   : Median < score ≤ Q3   (above average)
    LOW      : Q1 < score ≤ Median   (below average)
    MINIMAL  : score ≤ Q1            (bottom quartile)

Why box-plot over static thresholds?
    • Adaptive — adjusts to each dataset's distribution
    • No magic numbers — avoids arbitrary cutoffs like "0.7 = critical"
    • Statistically grounded — based on well-understood descriptive statistics
    • Scale-independent — works regardless of absolute score magnitudes
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Sequence, Any

from src.domain.models.criticality import (
    CriticalityLevel,
    BoxPlotStats,
    ClassifiedItem,
    ClassificationResult,
)


class BoxPlotClassifier:
    """
    Adaptive threshold classifier based on box-plot statistics.

    Args:
        k_factor: IQR multiplier for outlier detection (default 1.5).
    """

    def __init__(self, k_factor: float = 1.5) -> None:
        self.k_factor = k_factor

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def compute_stats(self, scores: Sequence[float]) -> BoxPlotStats:
        """Compute box-plot statistics (quartiles, fences, descriptive stats)."""
        if not scores:
            return BoxPlotStats(k_factor=self.k_factor)

        s = sorted(scores)
        n = len(s)

        if n == 1:
            v = s[0]
            return BoxPlotStats(
                q1=v, median=v, q3=v, iqr=0.0,
                lower_fence=v, upper_fence=v,
                min_val=v, max_val=v, mean=v, std_dev=0.0,
                count=1, k_factor=self.k_factor,
            )

        def _pct(p: float) -> float:
            k = (n - 1) * p
            f = int(k)
            c = min(f + 1, n - 1)
            return s[f] + (k - f) * (s[c] - s[f])

        q1 = _pct(0.25)
        med = _pct(0.50)
        q3 = _pct(0.75)
        iqr = q3 - q1

        return BoxPlotStats(
            q1=q1, median=med, q3=q3, iqr=iqr,
            lower_fence=q1 - self.k_factor * iqr,
            upper_fence=q3 + self.k_factor * iqr,
            min_val=s[0], max_val=s[-1],
            mean=statistics.mean(s),
            std_dev=statistics.stdev(s) if n > 1 else 0.0,
            count=n, k_factor=self.k_factor,
        )

    # ------------------------------------------------------------------
    # Single-score classification
    # ------------------------------------------------------------------

    def classify_score(self, score: float, stats: BoxPlotStats) -> CriticalityLevel:
        """
        Classify one score against precomputed box-plot statistics.

            CRITICAL : score > upper fence
            HIGH     : score > Q3
            MEDIUM   : score > median
            LOW      : score > Q1
            MINIMAL  : score ≤ Q1
        """
        if score > stats.upper_fence:
            return CriticalityLevel.CRITICAL
        if score > stats.q3:
            return CriticalityLevel.HIGH
        if score > stats.median:
            return CriticalityLevel.MEDIUM
        if score > stats.q1:
            return CriticalityLevel.LOW
        return CriticalityLevel.MINIMAL

    # ------------------------------------------------------------------
    # Batch classification
    # ------------------------------------------------------------------

    def classify(
        self,
        data: Sequence[Dict[str, Any]],
        metric_name: str = "score",
        id_key: str = "id",
        score_key: str = "score",
    ) -> ClassificationResult:
        """
        Classify a collection of ``{id, score}`` dicts.

        Returns a ``ClassificationResult`` containing the classified items
        sorted by score (most critical first), box-plot statistics, and
        the level distribution.
        """
        if not data:
            return ClassificationResult(
                metric_name=metric_name,
                items=[],
                stats=BoxPlotStats(k_factor=self.k_factor),
                distribution={lv.value: 0 for lv in CriticalityLevel},
            )

        scores = [d[score_key] for d in data]
        stats = self.compute_stats(scores)

        items: List[ClassifiedItem] = []
        distribution: Dict[str, int] = {lv.value: 0 for lv in CriticalityLevel}

        for d in data:
            sid = d[id_key]
            sc = d[score_key]
            level = self.classify_score(sc, stats)
            pct = sum(1 for v in scores if v <= sc) / len(scores) * 100
            z = (sc - stats.mean) / stats.std_dev if stats.std_dev > 0 else 0.0

            items.append(ClassifiedItem(
                id=sid, score=sc, level=level,
                percentile=pct, z_score=z,
            ))
            distribution[level.value] += 1

        items.sort(key=lambda x: x.score, reverse=True)

        return ClassificationResult(
            metric_name=metric_name,
            items=items,
            stats=stats,
            distribution=distribution,
        )


# ---------------------------------------------------------------------------
# Utility combiners
# ---------------------------------------------------------------------------

def combine_levels(*levels: CriticalityLevel) -> CriticalityLevel:
    """Return the highest (most critical) level among the inputs."""
    if not levels:
        return CriticalityLevel.MINIMAL
    return max(levels, key=lambda x: x.numeric)


def weighted_combine(
    levels_weights: List[tuple[CriticalityLevel, float]],
) -> CriticalityLevel:
    """Combine multiple levels with weights (weighted average → nearest level)."""
    if not levels_weights:
        return CriticalityLevel.MINIMAL

    total_w = sum(w for _, w in levels_weights)
    if total_w == 0:
        return CriticalityLevel.MINIMAL

    avg = sum(lv.numeric * w for lv, w in levels_weights) / total_w

    if avg >= 4.5:
        return CriticalityLevel.CRITICAL
    if avg >= 3.5:
        return CriticalityLevel.HIGH
    if avg >= 2.5:
        return CriticalityLevel.MEDIUM
    if avg >= 1.5:
        return CriticalityLevel.LOW
    return CriticalityLevel.MINIMAL