#!/usr/bin/env python3
"""
Box Plot Statistical Classification for Criticality Scoring

This module implements adaptive criticality classification using the box plot
statistical method (IQR - Interquartile Range). Unlike fixed thresholds,
this approach adapts to the actual distribution of scores in the system.

Box Plot Classification Method:
===============================
The box plot method divides data into regions based on quartiles:

    VERY LOW    LOW       MEDIUM      HIGH      VERY HIGH
    |-----------|---------|----------|---------|-----------|
    Lower      Q1        Median      Q3       Upper
    Fence                                      Fence

Where:
- Q1 = 25th percentile (first quartile)
- Q3 = 75th percentile (third quartile)
- IQR = Q3 - Q1 (interquartile range)
- Lower Fence = Q1 - 1.5 * IQR
- Upper Fence = Q3 + 1.5 * IQR

Classification Levels:
- VERY HIGH: score > Upper Fence (statistical outliers - extremely critical)
- HIGH: Q3 < score <= Upper Fence (above 75th percentile)
- MEDIUM: Q1 < score <= Q3 (interquartile range - typical components)
- LOW: Lower Fence < score <= Q1 (below 25th percentile)
- VERY LOW: score <= Lower Fence (statistical outliers - minimal concern)

Advantages over Fixed Thresholds:
=================================
1. ADAPTIVE: Adjusts to the actual score distribution of each system
2. STATISTICALLY MEANINGFUL: Based on established statistical measures
3. OUTLIER DETECTION: Naturally identifies extreme values
4. COMPARABLE: Allows fair comparison across different system sizes
5. ROBUST: Less sensitive to score scaling differences

Author: Software-as-a-Graph Research Project
Version: 1.0
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

# Optional numpy - fallback to pure Python if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class BoxPlotCriticalityLevel(Enum):
    """
    Criticality levels based on box plot classification.

    These levels map to statistical regions in the box plot,
    providing adaptive classification based on data distribution.
    """
    VERY_HIGH = "VERY HIGH"   # Upper outliers (> Q3 + 1.5*IQR)
    HIGH = "HIGH"             # Upper quartile (Q3 to Upper Fence)
    MEDIUM = "MEDIUM"         # Interquartile range (Q1 to Q3)
    LOW = "LOW"               # Lower quartile (Lower Fence to Q1)
    VERY_LOW = "VERY LOW"     # Lower outliers (< Q1 - 1.5*IQR)


@dataclass
class BoxPlotStatistics:
    """
    Box plot statistics for a set of scores.

    Contains all statistical measures needed for classification
    and visualization of the score distribution.
    """
    # Core statistics
    min_value: float
    max_value: float
    mean: float
    median: float  # Q2
    std_dev: float

    # Quartiles
    q1: float      # 25th percentile
    q3: float      # 75th percentile
    iqr: float     # Interquartile range (Q3 - Q1)

    # Fences (classification boundaries)
    lower_fence: float   # Q1 - 1.5 * IQR
    upper_fence: float   # Q3 + 1.5 * IQR

    # Extended fences (for extreme outliers)
    lower_extreme_fence: float  # Q1 - 3 * IQR
    upper_extreme_fence: float  # Q3 + 3 * IQR

    # Count statistics
    total_count: int
    outlier_count_lower: int = 0
    outlier_count_upper: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'min': round(self.min_value, 4),
            'max': round(self.max_value, 4),
            'mean': round(self.mean, 4),
            'median': round(self.median, 4),
            'std_dev': round(self.std_dev, 4),
            'q1': round(self.q1, 4),
            'q3': round(self.q3, 4),
            'iqr': round(self.iqr, 4),
            'lower_fence': round(self.lower_fence, 4),
            'upper_fence': round(self.upper_fence, 4),
            'lower_extreme_fence': round(self.lower_extreme_fence, 4),
            'upper_extreme_fence': round(self.upper_extreme_fence, 4),
            'total_count': self.total_count,
            'outlier_count_lower': self.outlier_count_lower,
            'outlier_count_upper': self.outlier_count_upper,
            'outlier_percentage': round(
                100 * (self.outlier_count_lower + self.outlier_count_upper) / self.total_count, 2
            ) if self.total_count > 0 else 0.0
        }


@dataclass
class BoxPlotClassificationResult:
    """
    Result of box plot classification for a single component.

    Contains the classification level, score, and additional
    context for understanding the classification.
    """
    component: str
    component_type: str
    score: float
    level: BoxPlotCriticalityLevel

    # Position within the distribution
    percentile: float           # Percentile rank (0-100)
    z_score: float              # Standard deviations from mean
    is_outlier: bool            # Whether this is a statistical outlier
    outlier_type: Optional[str] = None  # 'upper', 'lower', 'extreme_upper', 'extreme_lower'

    # Distance from thresholds
    distance_to_upper_fence: float = 0.0
    distance_to_lower_fence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'component': self.component,
            'component_type': self.component_type,
            'score': round(self.score, 4),
            'level': self.level.value,
            'percentile': round(self.percentile, 2),
            'z_score': round(self.z_score, 4),
            'is_outlier': self.is_outlier,
            'outlier_type': self.outlier_type,
            'distance_to_upper_fence': round(self.distance_to_upper_fence, 4),
            'distance_to_lower_fence': round(self.distance_to_lower_fence, 4)
        }


@dataclass
class BoxPlotClassificationSummary:
    """
    Summary of box plot classification results for an entire system.
    """
    statistics: BoxPlotStatistics
    level_counts: Dict[str, int]
    level_percentages: Dict[str, float]
    components_by_level: Dict[str, List[str]]

    # Top critical and minimal components
    top_critical: List[BoxPlotClassificationResult] = field(default_factory=list)
    bottom_minimal: List[BoxPlotClassificationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'statistics': self.statistics.to_dict(),
            'level_counts': self.level_counts,
            'level_percentages': {k: round(v, 2) for k, v in self.level_percentages.items()},
            'components_by_level': self.components_by_level,
            'top_critical': [r.to_dict() for r in self.top_critical],
            'bottom_minimal': [r.to_dict() for r in self.bottom_minimal]
        }


class BoxPlotClassifier:
    """
    Box Plot Statistical Classifier for Criticality Scores.

    This classifier uses the Interquartile Range (IQR) method to
    adaptively classify component criticality based on the actual
    distribution of scores in the system.

    Usage:
        classifier = BoxPlotClassifier()

        # Classify from raw scores
        results = classifier.classify_scores(scores_dict)

        # Or classify from existing criticality scorer results
        results = classifier.classify_from_criticality_scores(criticality_scores)

        # Get summary
        summary = classifier.get_summary()

    The classifier automatically handles:
    - Edge cases (small datasets, uniform distributions)
    - Zero IQR (when Q1 == Q3)
    - Score bounds (0 to 1 range typical for criticality scores)
    """

    def __init__(self,
                 iqr_multiplier: float = 1.5,
                 clamp_to_bounds: bool = True,
                 score_bounds: Tuple[float, float] = (0.0, 1.0)):
        """
        Initialize the box plot classifier.

        Args:
            iqr_multiplier: Multiplier for IQR to calculate fences (default: 1.5)
                           Use 3.0 for detecting only extreme outliers
            clamp_to_bounds: Whether to clamp fence values to score bounds
            score_bounds: Expected (min, max) bounds for scores
        """
        self.iqr_multiplier = iqr_multiplier
        self.clamp_to_bounds = clamp_to_bounds
        self.score_bounds = score_bounds

        self.logger = logging.getLogger(__name__)

        # Cached results
        self._statistics: Optional[BoxPlotStatistics] = None
        self._results: Dict[str, BoxPlotClassificationResult] = {}

    def calculate_statistics(self, scores: List[float]) -> BoxPlotStatistics:
        """
        Calculate box plot statistics for a list of scores.

        Args:
            scores: List of numeric scores

        Returns:
            BoxPlotStatistics with all computed measures
        """
        if not scores:
            raise ValueError("Cannot calculate statistics for empty score list")

        n = len(scores)
        sorted_scores = sorted(scores)

        # Basic statistics
        min_val = sorted_scores[0]
        max_val = sorted_scores[-1]
        mean = sum(scores) / n

        # Calculate median (Q2)
        if n % 2 == 0:
            median = (sorted_scores[n//2 - 1] + sorted_scores[n//2]) / 2
        else:
            median = sorted_scores[n//2]

        # Calculate standard deviation
        variance = sum((x - mean) ** 2 for x in scores) / n
        std_dev = variance ** 0.5

        # Calculate quartiles using linear interpolation
        q1 = self._percentile(sorted_scores, 25)
        q3 = self._percentile(sorted_scores, 75)

        # Calculate IQR and fences
        iqr = q3 - q1

        # Handle zero IQR case (all values in middle 50% are same)
        if iqr == 0:
            # Use standard deviation as fallback
            if std_dev > 0:
                iqr = std_dev * 1.35  # Approximate IQR for normal distribution
            else:
                # All values are identical - use small epsilon
                iqr = 0.001

        lower_fence = q1 - self.iqr_multiplier * iqr
        upper_fence = q3 + self.iqr_multiplier * iqr

        # Extended fences for extreme outliers
        lower_extreme = q1 - 3 * iqr
        upper_extreme = q3 + 3 * iqr

        # Clamp fences to score bounds if requested
        if self.clamp_to_bounds:
            lower_fence = max(self.score_bounds[0], lower_fence)
            upper_fence = min(self.score_bounds[1], upper_fence)
            lower_extreme = max(self.score_bounds[0], lower_extreme)
            upper_extreme = min(self.score_bounds[1], upper_extreme)

        # Count outliers
        outliers_lower = sum(1 for s in scores if s < lower_fence)
        outliers_upper = sum(1 for s in scores if s > upper_fence)

        return BoxPlotStatistics(
            min_value=min_val,
            max_value=max_val,
            mean=mean,
            median=median,
            std_dev=std_dev,
            q1=q1,
            q3=q3,
            iqr=iqr,
            lower_fence=lower_fence,
            upper_fence=upper_fence,
            lower_extreme_fence=lower_extreme,
            upper_extreme_fence=upper_extreme,
            total_count=n,
            outlier_count_lower=outliers_lower,
            outlier_count_upper=outliers_upper
        )

    def _percentile(self, sorted_data: List[float], p: float) -> float:
        """
        Calculate percentile using linear interpolation.

        Args:
            sorted_data: Sorted list of values
            p: Percentile (0-100)

        Returns:
            Interpolated percentile value
        """
        n = len(sorted_data)

        if n == 1:
            return sorted_data[0]

        # Use linear interpolation (same as numpy default)
        k = (n - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < n else f

        if f == c:
            return sorted_data[f]

        d0 = sorted_data[f] * (c - k)
        d1 = sorted_data[c] * (k - f)

        return d0 + d1

    def _calculate_percentile_rank(self, score: float, sorted_scores: List[float]) -> float:
        """Calculate the percentile rank of a score within the distribution"""
        n = len(sorted_scores)
        count_below = sum(1 for s in sorted_scores if s < score)
        count_equal = sum(1 for s in sorted_scores if s == score)

        # Use average rank for ties
        percentile = 100 * (count_below + 0.5 * count_equal) / n
        return min(100, max(0, percentile))

    def classify_score(self,
                       score: float,
                       stats: BoxPlotStatistics) -> BoxPlotCriticalityLevel:
        """
        Classify a single score using box plot statistics.

        Args:
            score: The score to classify
            stats: Pre-calculated box plot statistics

        Returns:
            BoxPlotCriticalityLevel classification
        """
        if score > stats.upper_fence:
            return BoxPlotCriticalityLevel.VERY_HIGH
        elif score > stats.q3:
            return BoxPlotCriticalityLevel.HIGH
        elif score > stats.q1:
            return BoxPlotCriticalityLevel.MEDIUM
        elif score > stats.lower_fence:
            return BoxPlotCriticalityLevel.LOW
        else:
            return BoxPlotCriticalityLevel.VERY_LOW

    def classify_scores(self,
                        scores: Dict[str, float],
                        component_types: Optional[Dict[str, str]] = None
                        ) -> Dict[str, BoxPlotClassificationResult]:
        """
        Classify all scores using box plot method.

        Args:
            scores: Dictionary mapping component IDs to scores
            component_types: Optional dictionary mapping component IDs to types

        Returns:
            Dictionary mapping component IDs to BoxPlotClassificationResult
        """
        if not scores:
            self.logger.warning("No scores provided for classification")
            return {}

        component_types = component_types or {}
        score_values = list(scores.values())
        sorted_scores = sorted(score_values)

        # Calculate statistics
        self._statistics = self.calculate_statistics(score_values)
        stats = self._statistics

        self.logger.info(f"Classifying {len(scores)} components using box plot method")
        self.logger.info(f"Statistics: Q1={stats.q1:.4f}, Q3={stats.q3:.4f}, "
                        f"IQR={stats.iqr:.4f}, Fences=[{stats.lower_fence:.4f}, {stats.upper_fence:.4f}]")

        results = {}

        for component_id, score in scores.items():
            # Classify
            level = self.classify_score(score, stats)

            # Calculate percentile
            percentile = self._calculate_percentile_rank(score, sorted_scores)

            # Calculate z-score
            z_score = (score - stats.mean) / stats.std_dev if stats.std_dev > 0 else 0.0

            # Determine outlier status
            is_outlier = score > stats.upper_fence or score < stats.lower_fence
            outlier_type = None
            if score > stats.upper_extreme_fence:
                outlier_type = 'extreme_upper'
            elif score > stats.upper_fence:
                outlier_type = 'upper'
            elif score < stats.lower_extreme_fence:
                outlier_type = 'extreme_lower'
            elif score < stats.lower_fence:
                outlier_type = 'lower'

            # Calculate distances to fences
            dist_upper = stats.upper_fence - score
            dist_lower = score - stats.lower_fence

            results[component_id] = BoxPlotClassificationResult(
                component=component_id,
                component_type=component_types.get(component_id, 'Unknown'),
                score=score,
                level=level,
                percentile=percentile,
                z_score=z_score,
                is_outlier=is_outlier,
                outlier_type=outlier_type,
                distance_to_upper_fence=dist_upper,
                distance_to_lower_fence=dist_lower
            )

        self._results = results
        return results

    def classify_from_criticality_scores(self,
                                          criticality_scores: Dict[str, Any]
                                          ) -> Dict[str, BoxPlotClassificationResult]:
        """
        Classify from existing criticality scorer results.

        Accepts results from CompositeCriticalityScorer or FuzzyCriticalityScorer.

        Args:
            criticality_scores: Dictionary of criticality score objects
                               (CompositeCriticalityScore or FuzzyNodeCriticalityScore)

        Returns:
            Dictionary mapping component IDs to BoxPlotClassificationResult
        """
        scores = {}
        component_types = {}

        for component_id, score_obj in criticality_scores.items():
            # Handle different score object types
            if hasattr(score_obj, 'composite_score'):
                scores[component_id] = score_obj.composite_score
            elif hasattr(score_obj, 'fuzzy_score'):
                scores[component_id] = score_obj.fuzzy_score
            elif isinstance(score_obj, (int, float)):
                scores[component_id] = float(score_obj)
            else:
                continue

            if hasattr(score_obj, 'component_type'):
                component_types[component_id] = score_obj.component_type

        return self.classify_scores(scores, component_types)

    def classify_edge_scores(self,
                             edge_scores: Dict[Tuple[str, str], Any]
                             ) -> Dict[Tuple[str, str], BoxPlotClassificationResult]:
        """
        Classify edge criticality scores using box plot method.

        Args:
            edge_scores: Dictionary mapping (source, target) tuples to score objects

        Returns:
            Dictionary mapping edge tuples to BoxPlotClassificationResult
        """
        scores = {}
        edge_types = {}

        for edge_key, score_obj in edge_scores.items():
            # Handle different score object types
            if hasattr(score_obj, 'composite_score'):
                scores[edge_key] = score_obj.composite_score
            elif hasattr(score_obj, 'fuzzy_score'):
                scores[edge_key] = score_obj.fuzzy_score
            elif isinstance(score_obj, (int, float)):
                scores[edge_key] = float(score_obj)
            else:
                continue

            if hasattr(score_obj, 'edge_type'):
                edge_types[edge_key] = score_obj.edge_type

        if not scores:
            return {}

        # Convert edge keys to strings for internal processing
        str_scores = {f"{s}->{t}": score for (s, t), score in scores.items()}
        str_types = {f"{s}->{t}": etype for (s, t), etype in edge_types.items()}

        # Classify
        str_results = self.classify_scores(str_scores, str_types)

        # Convert back to tuple keys
        results = {}
        for edge_key in edge_scores.keys():
            str_key = f"{edge_key[0]}->{edge_key[1]}"
            if str_key in str_results:
                result = str_results[str_key]
                # Update component name to show edge format
                result.component = f"{edge_key[0]} → {edge_key[1]}"
                results[edge_key] = result

        return results

    def get_summary(self,
                    results: Optional[Dict[str, BoxPlotClassificationResult]] = None,
                    top_n: int = 10) -> BoxPlotClassificationSummary:
        """
        Generate summary of classification results.

        Args:
            results: Classification results (uses cached if None)
            top_n: Number of top/bottom components to include

        Returns:
            BoxPlotClassificationSummary
        """
        results = results or self._results
        stats = self._statistics

        if not results or not stats:
            raise ValueError("No classification results available. Run classify_scores first.")

        # Count by level
        level_counts = defaultdict(int)
        components_by_level = defaultdict(list)

        for component_id, result in results.items():
            level_counts[result.level.value] += 1
            components_by_level[result.level.value].append(component_id)

        # Ensure all levels are present
        for level in BoxPlotCriticalityLevel:
            if level.value not in level_counts:
                level_counts[level.value] = 0
                components_by_level[level.value] = []

        # Calculate percentages
        total = len(results)
        level_percentages = {
            level: 100 * count / total
            for level, count in level_counts.items()
        }

        # Get top critical and bottom minimal
        sorted_results = sorted(results.values(), key=lambda r: r.score, reverse=True)
        top_critical = sorted_results[:top_n]
        bottom_minimal = sorted_results[-top_n:][::-1]  # Reverse to show lowest first

        return BoxPlotClassificationSummary(
            statistics=stats,
            level_counts=dict(level_counts),
            level_percentages=dict(level_percentages),
            components_by_level=dict(components_by_level),
            top_critical=top_critical,
            bottom_minimal=bottom_minimal
        )

    def get_components_by_level(self,
                                 level: BoxPlotCriticalityLevel,
                                 results: Optional[Dict[str, BoxPlotClassificationResult]] = None
                                 ) -> List[BoxPlotClassificationResult]:
        """
        Get all components at a specific criticality level.

        Args:
            level: The criticality level to filter by
            results: Classification results (uses cached if None)

        Returns:
            List of BoxPlotClassificationResult at the specified level
        """
        results = results or self._results
        return [r for r in results.values() if r.level == level]

    def get_outliers(self,
                     results: Optional[Dict[str, BoxPlotClassificationResult]] = None,
                     outlier_type: Optional[str] = None
                     ) -> List[BoxPlotClassificationResult]:
        """
        Get all statistical outliers.

        Args:
            results: Classification results (uses cached if None)
            outlier_type: Filter by type ('upper', 'lower', 'extreme_upper', 'extreme_lower')

        Returns:
            List of outlier components
        """
        results = results or self._results
        outliers = [r for r in results.values() if r.is_outlier]

        if outlier_type:
            outliers = [r for r in outliers if r.outlier_type == outlier_type]

        return sorted(outliers, key=lambda r: r.score, reverse=True)

    def compare_with_fixed_thresholds(self,
                                       results: Optional[Dict[str, BoxPlotClassificationResult]] = None,
                                       fixed_thresholds: Dict[str, float] = None
                                       ) -> Dict[str, Any]:
        """
        Compare box plot classification with fixed threshold classification.

        Args:
            results: Classification results (uses cached if None)
            fixed_thresholds: Fixed thresholds dict (default: standard 0.8/0.6/0.4/0.2)

        Returns:
            Comparison statistics
        """
        results = results or self._results

        if fixed_thresholds is None:
            fixed_thresholds = {
                'CRITICAL': 0.8,
                'HIGH': 0.6,
                'MEDIUM': 0.4,
                'LOW': 0.2
            }

        # Map fixed levels to box plot levels
        level_mapping = {
            'CRITICAL': BoxPlotCriticalityLevel.VERY_HIGH,
            'HIGH': BoxPlotCriticalityLevel.HIGH,
            'MEDIUM': BoxPlotCriticalityLevel.MEDIUM,
            'LOW': BoxPlotCriticalityLevel.LOW,
            'MINIMAL': BoxPlotCriticalityLevel.VERY_LOW
        }

        agreements = 0
        disagreements = []

        for component_id, bp_result in results.items():
            score = bp_result.score

            # Classify using fixed thresholds
            if score >= fixed_thresholds['CRITICAL']:
                fixed_level = 'CRITICAL'
            elif score >= fixed_thresholds['HIGH']:
                fixed_level = 'HIGH'
            elif score >= fixed_thresholds['MEDIUM']:
                fixed_level = 'MEDIUM'
            elif score >= fixed_thresholds['LOW']:
                fixed_level = 'LOW'
            else:
                fixed_level = 'MINIMAL'

            # Compare
            expected_bp_level = level_mapping[fixed_level]
            if bp_result.level == expected_bp_level:
                agreements += 1
            else:
                disagreements.append({
                    'component': component_id,
                    'score': score,
                    'fixed_level': fixed_level,
                    'boxplot_level': bp_result.level.value,
                    'percentile': bp_result.percentile
                })

        total = len(results)

        return {
            'total_components': total,
            'agreements': agreements,
            'disagreements_count': len(disagreements),
            'agreement_rate': round(100 * agreements / total, 2) if total > 0 else 0,
            'disagreements': disagreements,
            'fixed_thresholds': fixed_thresholds,
            'boxplot_thresholds': {
                'upper_fence': self._statistics.upper_fence if self._statistics else None,
                'q3': self._statistics.q3 if self._statistics else None,
                'q1': self._statistics.q1 if self._statistics else None,
                'lower_fence': self._statistics.lower_fence if self._statistics else None
            }
        }

    def generate_report(self,
                        results: Optional[Dict[str, BoxPlotClassificationResult]] = None,
                        include_all_components: bool = False) -> str:
        """
        Generate a human-readable report of the classification.

        Args:
            results: Classification results (uses cached if None)
            include_all_components: Whether to list all components

        Returns:
            Formatted string report
        """
        results = results or self._results
        summary = self.get_summary(results)
        stats = summary.statistics

        lines = [
            "=" * 70,
            "BOX PLOT CRITICALITY CLASSIFICATION REPORT",
            "=" * 70,
            "",
            "STATISTICAL SUMMARY",
            "-" * 40,
            f"  Total Components:    {stats.total_count}",
            f"  Score Range:         [{stats.min_value:.4f}, {stats.max_value:.4f}]",
            f"  Mean Score:          {stats.mean:.4f}",
            f"  Median Score:        {stats.median:.4f}",
            f"  Standard Deviation:  {stats.std_dev:.4f}",
            "",
            "QUARTILES",
            "-" * 40,
            f"  Q1 (25th percentile): {stats.q1:.4f}",
            f"  Q2 (50th percentile): {stats.median:.4f}",
            f"  Q3 (75th percentile): {stats.q3:.4f}",
            f"  IQR (Q3 - Q1):        {stats.iqr:.4f}",
            "",
            "CLASSIFICATION THRESHOLDS",
            "-" * 40,
            f"  Upper Fence (VERY HIGH): > {stats.upper_fence:.4f}",
            f"  High Range:              ({stats.q3:.4f}, {stats.upper_fence:.4f}]",
            f"  Medium Range:            ({stats.q1:.4f}, {stats.q3:.4f}]",
            f"  Low Range:               ({stats.lower_fence:.4f}, {stats.q1:.4f}]",
            f"  Lower Fence (VERY LOW):  <= {stats.lower_fence:.4f}",
            "",
            "CLASSIFICATION DISTRIBUTION",
            "-" * 40,
        ]

        # Level distribution with visual bar
        max_count = max(summary.level_counts.values()) if summary.level_counts else 1
        for level in BoxPlotCriticalityLevel:
            count = summary.level_counts.get(level.value, 0)
            pct = summary.level_percentages.get(level.value, 0)
            bar_len = int(30 * count / max_count) if max_count > 0 else 0
            bar = "█" * bar_len + "░" * (30 - bar_len)
            lines.append(f"  {level.value:10s}: {count:4d} ({pct:5.1f}%) {bar}")

        lines.extend([
            "",
            f"OUTLIERS: {stats.outlier_count_upper} upper, {stats.outlier_count_lower} lower",
            "",
            "TOP 10 MOST CRITICAL COMPONENTS",
            "-" * 40,
        ])

        for i, result in enumerate(summary.top_critical[:10], 1):
            lines.append(
                f"  {i:2d}. {result.component:30s} "
                f"Score: {result.score:.4f} "
                f"[{result.level.value:9s}] "
                f"P{result.percentile:.0f}"
            )

        lines.extend([
            "",
            "BOTTOM 10 MINIMAL CONCERN COMPONENTS",
            "-" * 40,
        ])

        for i, result in enumerate(summary.bottom_minimal[:10], 1):
            lines.append(
                f"  {i:2d}. {result.component:30s} "
                f"Score: {result.score:.4f} "
                f"[{result.level.value:9s}] "
                f"P{result.percentile:.0f}"
            )

        if include_all_components:
            lines.extend([
                "",
                "ALL COMPONENTS BY LEVEL",
                "-" * 40,
            ])
            for level in BoxPlotCriticalityLevel:
                components = summary.components_by_level.get(level.value, [])
                if components:
                    lines.append(f"\n  {level.value}:")
                    for comp in sorted(components):
                        result = results[comp]
                        lines.append(f"    - {comp}: {result.score:.4f}")

        lines.extend([
            "",
            "=" * 70,
        ])

        return "\n".join(lines)


# =============================================================================
# Integration Functions
# =============================================================================

def classify_criticality_with_boxplot(
    criticality_scores: Dict[str, Any],
    iqr_multiplier: float = 1.5
) -> Tuple[Dict[str, BoxPlotClassificationResult], BoxPlotClassificationSummary]:
    """
    Convenience function to classify criticality scores using box plot method.

    Args:
        criticality_scores: Dictionary of criticality score objects
        iqr_multiplier: IQR multiplier for fence calculation

    Returns:
        Tuple of (results dict, summary)
    """
    classifier = BoxPlotClassifier(iqr_multiplier=iqr_multiplier)
    results = classifier.classify_from_criticality_scores(criticality_scores)
    summary = classifier.get_summary(results)
    return results, summary


def classify_edges_with_boxplot(
    edge_scores: Dict[Tuple[str, str], Any],
    iqr_multiplier: float = 1.5
) -> Tuple[Dict[Tuple[str, str], BoxPlotClassificationResult], BoxPlotClassificationSummary]:
    """
    Convenience function to classify edge criticality scores using box plot method.

    Args:
        edge_scores: Dictionary of edge criticality score objects
        iqr_multiplier: IQR multiplier for fence calculation

    Returns:
        Tuple of (results dict, summary)
    """
    classifier = BoxPlotClassifier(iqr_multiplier=iqr_multiplier)
    results = classifier.classify_edge_scores(edge_scores)
    summary = classifier.get_summary()
    return results, summary


# =============================================================================
# Main - Demonstration
# =============================================================================

def main():
    """Demonstrate box plot classification"""
    print("=" * 70)
    print("Box Plot Statistical Classification - Demonstration")
    print("=" * 70)

    # Sample criticality scores (simulating output from criticality scorer)
    sample_scores = {
        'app_broker_central': 0.92,   # Very high - central broker
        'app_gateway': 0.85,          # Very high - gateway
        'app_orders_service': 0.73,   # High
        'app_payment_processor': 0.68, # High
        'app_inventory_manager': 0.62, # High
        'app_user_service': 0.55,     # Medium
        'app_notification': 0.52,     # Medium
        'app_analytics': 0.48,        # Medium
        'app_logging': 0.45,          # Medium
        'app_cache': 0.42,            # Medium
        'app_config': 0.38,           # Low
        'app_health_check': 0.35,     # Low
        'app_metrics': 0.32,          # Low
        'app_backup': 0.28,           # Low
        'app_cleanup': 0.22,          # Low
        'app_test_helper': 0.15,      # Very low
        'app_debug_tool': 0.12,       # Very low
        'app_mock_service': 0.08,     # Very low
    }

    component_types = {k: 'Application' for k in sample_scores}

    print(f"\nSample dataset: {len(sample_scores)} components")
    print(f"Score range: [{min(sample_scores.values()):.2f}, {max(sample_scores.values()):.2f}]")

    # Create classifier and classify
    classifier = BoxPlotClassifier()
    results = classifier.classify_scores(sample_scores, component_types)

    # Generate and print report
    report = classifier.generate_report(results)
    print(report)

    # Compare with fixed thresholds
    print("\n" + "=" * 70)
    print("COMPARISON WITH FIXED THRESHOLDS")
    print("=" * 70)

    comparison = classifier.compare_with_fixed_thresholds(results)
    print(f"\nAgreement Rate: {comparison['agreement_rate']}%")
    print(f"Total Agreements: {comparison['agreements']}/{comparison['total_components']}")

    if comparison['disagreements']:
        print(f"\nDisagreements ({len(comparison['disagreements'])}):")
        for d in comparison['disagreements'][:5]:
            print(f"  - {d['component']}: Fixed={d['fixed_level']}, "
                  f"BoxPlot={d['boxplot_level']} (score={d['score']:.3f}, P{d['percentile']:.0f})")

    print("\nThreshold Comparison:")
    print(f"  Fixed CRITICAL: >= 0.80  vs  BoxPlot VERY_HIGH: > {comparison['boxplot_thresholds']['upper_fence']:.4f}")
    print(f"  Fixed HIGH:     >= 0.60  vs  BoxPlot HIGH:      > {comparison['boxplot_thresholds']['q3']:.4f}")
    print(f"  Fixed MEDIUM:   >= 0.40  vs  BoxPlot MEDIUM:    > {comparison['boxplot_thresholds']['q1']:.4f}")
    print(f"  Fixed LOW:      >= 0.20  vs  BoxPlot LOW:       > {comparison['boxplot_thresholds']['lower_fence']:.4f}")


if __name__ == "__main__":
    main()
