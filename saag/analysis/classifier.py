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

**The population you pass is part of the measurement.** One box-plot over a mixed
population of node types puts Applications, Brokers, Topics, Nodes and Libraries
behind a single Q3 + k·IQR fence, even though their score distributions differ in
both scale and base rate. Measured on the eight-scenario corpus, stratifying by
node type instead of pooling moves **62.8%** of components to a different tier and
changes CRITICAL/HIGH membership — the flagged set an architect acts on — for
**19.0%** of them (``results/tier_pooling_check.json``). Pass ``group_key`` to
classify within type; see the same Simpson's-paradox hazard recorded for the
evaluation tables in the manuscript's Conclusion Validity discussion.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Sequence, Any

from saag.core.criticality import (
    CriticalityLevel,
    BoxPlotStats,
    ClassifiedItem,
    ClassificationResult,
)


#: Smallest group that gets its own quartiles. Below this a box-plot describes
#: the sample rather than the population, and its fence would be noise.
_MIN_GROUP = 8


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

    def compute_stats(self, scores: Sequence[float], k_factor: float = None) -> BoxPlotStats:
        """Compute box-plot statistics (quartiles, fences, descriptive stats)."""
        k = k_factor if k_factor is not None else self.k_factor

        if not scores:
            return BoxPlotStats(k_factor=k)

        s = sorted(scores)
        n = len(s)

        if n == 1:
            v = s[0]
            return BoxPlotStats(
                q1=v, median=v, q3=v, iqr=0.0,
                lower_fence=v, upper_fence=v,
                min_val=v, max_val=v, mean=v, std_dev=0.0,
                count=1, k_factor=k,
            )

        def _pct(p: float) -> float:
            k_idx = (n - 1) * p
            f = int(k_idx)
            c = min(f + 1, n - 1)
            return s[f] + (k_idx - f) * (s[c] - s[f])

        q1 = _pct(0.25)
        med = _pct(0.50)
        q3 = _pct(0.75)
        iqr = q3 - q1

        return BoxPlotStats(
            q1=q1, median=med, q3=q3, iqr=iqr,
            lower_fence=q1 - k * iqr,
            upper_fence=q3 + k * iqr,
            min_val=s[0], max_val=s[-1],
            mean=statistics.mean(s),
            std_dev=statistics.stdev(s) if n > 1 else 0.0,
            count=n, k_factor=k,
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
        k_factor: float = None,
        group_key: str = None,
    ) -> ClassificationResult:
        """
        Classify a collection of ``{id, score}`` dicts.

        Returns a ``ClassificationResult`` containing the classified items
        sorted by score (most critical first), box-plot statistics, and
        the level distribution.

        ``group_key``
            Name of a field partitioning *data* into populations that are scored
            independently — in practice the node type. Each group gets its own
            quartiles and its own fence, so a Broker is ranked against Brokers
            rather than against a population whose scale it does not share. See
            the module docstring for what pooling costs on this corpus.

            Items missing the field, and groups too small for stable quartiles
            (fewer than :data:`_MIN_GROUP`), fall back to the pooled fence rather
            than being dropped or scored against three samples. ``stats`` on the
            result stays the pooled box-plot, since a stratified run has no single
            set of quartiles to report; per-group statistics are in
            ``group_stats``.

            Default ``None`` preserves the pooled behaviour exactly.
        """
        k = k_factor if k_factor is not None else self.k_factor

        if group_key is not None:
            return self._classify_grouped(
                data, metric_name, id_key, score_key, k, group_key)

        if not data:
            return ClassificationResult(
                metric_name=metric_name,
                items=[],
                stats=BoxPlotStats(k_factor=k),
                distribution={lv.value: 0 for lv in CriticalityLevel},
            )

        scores = [d[score_key] for d in data]
        stats = self.compute_stats(scores, k_factor=k)

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

    def _classify_grouped(
        self,
        data: Sequence[Dict[str, Any]],
        metric_name: str,
        id_key: str,
        score_key: str,
        k: float,
        group_key: str,
    ) -> ClassificationResult:
        """Classify within each ``group_key`` population, sharing no fence.

        Percentile and z-score are computed within the group too: reporting a
        component's standing against a population it was not ranked against
        would contradict the level beside it.
        """
        if not data:
            return ClassificationResult(
                metric_name=metric_name,
                items=[],
                stats=BoxPlotStats(k_factor=k),
                distribution={lv.value: 0 for lv in CriticalityLevel},
            )

        buckets: Dict[Any, List[Dict[str, Any]]] = {}
        for d in data:
            buckets.setdefault(d.get(group_key), []).append(d)

        # Groups too small for stable quartiles, and items carrying no group at
        # all, are scored against the pooled distribution instead of against
        # three samples. Merging them is the lesser distortion, and it keeps
        # every input item in the output.
        pooled_scores = [d[score_key] for d in data]
        pooled_stats = self.compute_stats(pooled_scores, k_factor=k)

        fallback: List[Dict[str, Any]] = []
        groups: Dict[Any, List[Dict[str, Any]]] = {}
        for name, rows in buckets.items():
            if name is None or len(rows) < _MIN_GROUP:
                fallback.extend(rows)
            else:
                groups[name] = rows

        items: List[ClassifiedItem] = []
        distribution: Dict[str, int] = {lv.value: 0 for lv in CriticalityLevel}
        group_stats: Dict[str, BoxPlotStats] = {}

        def _emit(rows: List[Dict[str, Any]], stats: BoxPlotStats) -> None:
            scores = [r[score_key] for r in rows]
            for r in rows:
                sc = r[score_key]
                level = self.classify_score(sc, stats)
                pct = sum(1 for v in scores if v <= sc) / len(scores) * 100
                z = (sc - stats.mean) / stats.std_dev if stats.std_dev > 0 else 0.0
                items.append(ClassifiedItem(
                    id=r[id_key], score=sc, level=level, percentile=pct, z_score=z,
                ))
                distribution[level.value] += 1

        for name, rows in groups.items():
            stats = self.compute_stats([r[score_key] for r in rows], k_factor=k)
            group_stats[str(name)] = stats
            _emit(rows, stats)

        if fallback:
            _emit(fallback, pooled_stats)

        items.sort(key=lambda x: x.score, reverse=True)

        return ClassificationResult(
            metric_name=metric_name,
            items=items,
            stats=pooled_stats,
            distribution=distribution,
            group_stats=group_stats,
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
