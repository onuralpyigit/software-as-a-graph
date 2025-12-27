"""
Box-Plot Criticality Classifier - Version 4.0

Statistical classification using the box-plot (IQR) method instead of
static thresholds. This addresses the "sharp boundary problem" where
components with nearly identical scores receive different classifications.

Box-Plot Method:
    Q1 = 25th percentile (first quartile)
    Q3 = 75th percentile (third quartile)
    IQR = Q3 - Q1 (interquartile range)
    Lower Fence = Q1 - k * IQR
    Upper Fence = Q3 + k * IQR

Classification Levels:
    CRITICAL: score > Q3 + k*IQR (upper outliers)
    HIGH:     Q3 < score <= Q3 + k*IQR  
    MEDIUM:   Median < score <= Q3
    LOW:      Q1 < score <= Median
    MINIMAL:  score <= Q1

Advantages:
    - Adapts to actual data distribution
    - Statistically meaningful boundaries
    - Natural outlier detection
    - Comparable across different systems

Author: Software-as-a-Graph Research Project
Version: 4.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


# =============================================================================
# Enums
# =============================================================================

class CriticalityLevel(Enum):
    """Criticality levels based on box-plot classification"""
    CRITICAL = "critical"   # Upper outliers (> Q3 + k*IQR)
    HIGH = "high"           # Q3 to upper fence
    MEDIUM = "medium"       # Median to Q3
    LOW = "low"             # Q1 to median
    MINIMAL = "minimal"     # Below Q1
    
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
        """ANSI color code for display"""
        return {
            CriticalityLevel.CRITICAL: "\033[91m",  # Red
            CriticalityLevel.HIGH: "\033[93m",      # Yellow
            CriticalityLevel.MEDIUM: "\033[94m",    # Blue
            CriticalityLevel.LOW: "\033[92m",       # Green
            CriticalityLevel.MINIMAL: "\033[90m",   # Gray
        }[self]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BoxPlotStats:
    """Statistical values from box-plot analysis"""
    min_val: float
    q1: float          # 25th percentile
    median: float      # 50th percentile
    q3: float          # 75th percentile
    max_val: float
    iqr: float         # Interquartile range
    lower_fence: float # Q1 - k*IQR
    upper_fence: float # Q3 + k*IQR
    mean: float
    std_dev: float
    count: int
    k_factor: float    # Multiplier for fence calculation

    def to_dict(self) -> Dict:
        return {
            "min": self.min_val,
            "q1": self.q1,
            "median": self.median,
            "q3": self.q3,
            "max": self.max_val,
            "iqr": self.iqr,
            "lower_fence": self.lower_fence,
            "upper_fence": self.upper_fence,
            "mean": self.mean,
            "std_dev": self.std_dev,
            "count": self.count,
            "k_factor": self.k_factor,
        }


@dataclass
class ClassifiedItem:
    """A classified component or edge"""
    id: str
    item_type: str
    score: float
    level: CriticalityLevel
    percentile: float
    rank: int
    is_outlier: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.item_type,
            "score": self.score,
            "level": self.level.value,
            "percentile": self.percentile,
            "rank": self.rank,
            "is_outlier": self.is_outlier,
            "metadata": self.metadata,
        }


@dataclass
class ClassificationResult:
    """Complete classification result"""
    metric_name: str
    stats: BoxPlotStats
    items: List[ClassifiedItem]
    by_level: Dict[CriticalityLevel, List[ClassifiedItem]]
    summary: Dict[str, int]

    def to_dict(self) -> Dict:
        return {
            "metric": self.metric_name,
            "statistics": self.stats.to_dict(),
            "summary": {level.value: count for level, count in self.summary.items()},
            "items": [item.to_dict() for item in self.items],
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


# =============================================================================
# Box-Plot Classifier
# =============================================================================

class BoxPlotClassifier:
    """
    Classifies items using the box-plot statistical method.
    
    This classifier uses quartile-based thresholds that adapt to
    the actual data distribution, avoiding the sharp boundary
    problem of static thresholds.
    
    Parameters:
        k_factor: IQR multiplier for fence calculation (default: 1.5)
                  - 1.5 is standard for outlier detection
                  - 3.0 is conservative (fewer outliers)
                  - 1.0 is aggressive (more outliers)
    """

    def __init__(self, k_factor: float = 1.5):
        self.k_factor = k_factor
        self.logger = logging.getLogger(__name__)

    def calculate_stats(self, scores: List[float]) -> BoxPlotStats:
        """
        Calculate box-plot statistics for a list of scores.
        
        Args:
            scores: List of numeric scores
        
        Returns:
            BoxPlotStats with all statistical measures
        """
        if not scores:
            return BoxPlotStats(
                min_val=0, q1=0, median=0, q3=0, max_val=0,
                iqr=0, lower_fence=0, upper_fence=0,
                mean=0, std_dev=0, count=0, k_factor=self.k_factor,
            )
        
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        
        # Calculate quartiles
        q1_idx = int(n * 0.25)
        median_idx = int(n * 0.50)
        q3_idx = int(n * 0.75)
        
        q1 = sorted_scores[q1_idx]
        median = sorted_scores[median_idx]
        q3 = sorted_scores[q3_idx]
        
        # IQR and fences
        iqr = q3 - q1
        lower_fence = q1 - self.k_factor * iqr
        upper_fence = q3 + self.k_factor * iqr
        
        # Mean and std dev
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

    def classify_score(self, score: float, stats: BoxPlotStats) -> Tuple[CriticalityLevel, bool]:
        """
        Classify a single score based on box-plot statistics.
        
        Args:
            score: The score to classify
            stats: Box-plot statistics
        
        Returns:
            Tuple of (CriticalityLevel, is_outlier)
        """
        if score > stats.upper_fence:
            return CriticalityLevel.CRITICAL, True
        elif score > stats.q3:
            return CriticalityLevel.HIGH, False
        elif score > stats.median:
            return CriticalityLevel.MEDIUM, False
        elif score > stats.q1:
            return CriticalityLevel.LOW, False
        else:
            return CriticalityLevel.MINIMAL, score < stats.lower_fence

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
                stats=self.calculate_stats([]),
                items=[],
                by_level={level: [] for level in CriticalityLevel},
                summary={level: 0 for level in CriticalityLevel},
            )
        
        # Extract scores and calculate statistics
        scores = [item[score_key] for item in items]
        stats = self.calculate_stats(scores)
        
        # Sort by score descending for ranking
        sorted_items = sorted(items, key=lambda x: x[score_key], reverse=True)
        
        # Classify each item
        classified = []
        by_level = defaultdict(list)
        
        for rank, item in enumerate(sorted_items, 1):
            score = item[score_key]
            level, is_outlier = self.classify_score(score, stats)
            percentile = self._calculate_percentile(score, scores)
            
            # Build metadata from extra fields
            metadata = {k: v for k, v in item.items() 
                       if k not in (score_key, id_key, type_key)}
            
            ci = ClassifiedItem(
                id=item.get(id_key, f"item_{rank}"),
                item_type=item.get(type_key, "unknown"),
                score=score,
                level=level,
                percentile=percentile,
                rank=rank,
                is_outlier=is_outlier,
                metadata=metadata,
            )
            
            classified.append(ci)
            by_level[level].append(ci)
        
        # Build summary
        summary = {level: len(items) for level, items in by_level.items()}
        
        self.logger.info(
            f"Classified {len(items)} items: "
            f"CRITICAL={summary.get(CriticalityLevel.CRITICAL, 0)}, "
            f"HIGH={summary.get(CriticalityLevel.HIGH, 0)}, "
            f"MEDIUM={summary.get(CriticalityLevel.MEDIUM, 0)}, "
            f"LOW={summary.get(CriticalityLevel.LOW, 0)}, "
            f"MINIMAL={summary.get(CriticalityLevel.MINIMAL, 0)}"
        )
        
        return ClassificationResult(
            metric_name=metric_name,
            stats=stats,
            items=classified,
            by_level=dict(by_level),
            summary=summary,
        )

    def _calculate_percentile(self, score: float, all_scores: List[float]) -> float:
        """Calculate percentile rank of a score"""
        below = sum(1 for s in all_scores if s < score)
        equal = sum(1 for s in all_scores if s == score)
        return 100 * (below + 0.5 * equal) / len(all_scores)


# =============================================================================
# GDS-Integrated Classifier
# =============================================================================

class GDSClassifier:
    """
    Criticality classifier integrated with Neo4j GDS.
    
    Retrieves scores from GDS algorithms and classifies components
    using the box-plot method.
    
    Usage:
        classifier = GDSClassifier(gds_client, k_factor=1.5)
        result = classifier.classify_by_betweenness("projection_name")
        critical = result.get_critical()
    """

    def __init__(self, gds_client, k_factor: float = 1.5):
        self.gds = gds_client
        self.classifier = BoxPlotClassifier(k_factor=k_factor)
        self.logger = logging.getLogger(__name__)

    def classify_by_pagerank(
        self,
        projection_name: str,
        weighted: bool = True,
    ) -> ClassificationResult:
        """
        Classify components by PageRank.
        
        High PageRank = receives important dependencies = critical for availability.
        """
        self.logger.info("Classifying by PageRank")
        
        results = self.gds.pagerank(projection_name, weighted=weighted)
        items = [
            {"id": r.node_id, "type": r.node_type, "score": r.score, "rank": r.rank}
            for r in results
        ]
        
        return self.classifier.classify(items, metric_name="pagerank")

    def classify_by_betweenness(
        self,
        projection_name: str,
        weighted: bool = True,
    ) -> ClassificationResult:
        """
        Classify components by Betweenness Centrality.
        
        High betweenness = on many shortest paths = bottleneck/SPOF.
        """
        self.logger.info("Classifying by Betweenness")
        
        results = self.gds.betweenness(projection_name, weighted=weighted)
        items = [
            {"id": r.node_id, "type": r.node_type, "score": r.score, "rank": r.rank}
            for r in results
        ]
        
        return self.classifier.classify(items, metric_name="betweenness")

    def classify_by_degree(
        self,
        projection_name: str,
        weighted: bool = True,
        orientation: str = "UNDIRECTED",
    ) -> ClassificationResult:
        """
        Classify components by Degree Centrality.
        
        High degree = many connections = coupling/maintainability issue.
        """
        self.logger.info(f"Classifying by {orientation} Degree")
        
        results = self.gds.degree(projection_name, weighted=weighted, orientation=orientation)
        items = [
            {"id": r.node_id, "type": r.node_type, "score": r.score, "rank": r.rank}
            for r in results
        ]
        
        return self.classifier.classify(items, metric_name=f"degree_{orientation.lower()}")

    def classify_by_composite(
        self,
        projection_name: str,
        weighted: bool = True,
        weights: Optional[Dict[str, float]] = None,
    ) -> ClassificationResult:
        """
        Classify by composite criticality score.
        
        Composite = α·BC + β·PR + γ·DC
        
        Args:
            projection_name: GDS projection name
            weighted: Use weighted algorithms
            weights: Custom metric weights (default: equal)
        
        Returns:
            ClassificationResult with composite scores
        """
        self.logger.info("Computing composite criticality scores")
        
        # Default weights
        if weights is None:
            weights = {"betweenness": 0.4, "pagerank": 0.35, "degree": 0.25}
        
        # Get individual metrics
        bc_results = self.gds.betweenness(projection_name, weighted=weighted)
        pr_results = self.gds.pagerank(projection_name, weighted=weighted)
        dc_results = self.gds.degree(projection_name, weighted=weighted)
        
        # Build score maps
        bc_scores = {r.node_id: r.score for r in bc_results}
        pr_scores = {r.node_id: r.score for r in pr_results}
        dc_scores = {r.node_id: r.score for r in dc_results}
        
        # Normalize scores to [0, 1]
        def normalize(scores: Dict[str, float]) -> Dict[str, float]:
            if not scores:
                return {}
            max_val = max(scores.values())
            if max_val == 0:
                return {k: 0 for k in scores}
            return {k: v / max_val for k, v in scores.items()}
        
        bc_norm = normalize(bc_scores)
        pr_norm = normalize(pr_scores)
        dc_norm = normalize(dc_scores)
        
        # Get all node IDs and types
        node_types = {r.node_id: r.node_type for r in bc_results}
        all_nodes = set(bc_scores.keys()) | set(pr_scores.keys()) | set(dc_scores.keys())
        
        # Calculate composite scores
        items = []
        for node_id in all_nodes:
            bc = bc_norm.get(node_id, 0)
            pr = pr_norm.get(node_id, 0)
            dc = dc_norm.get(node_id, 0)
            
            composite = (
                weights["betweenness"] * bc +
                weights["pagerank"] * pr +
                weights["degree"] * dc
            )
            
            items.append({
                "id": node_id,
                "type": node_types.get(node_id, "Unknown"),
                "score": composite,
                "betweenness": bc_scores.get(node_id, 0),
                "pagerank": pr_scores.get(node_id, 0),
                "degree": dc_scores.get(node_id, 0),
            })
        
        return self.classifier.classify(items, metric_name="composite")

    def classify_edges_by_weight(self) -> ClassificationResult:
        """
        Classify DEPENDS_ON edges by weight.
        
        High weight = critical dependency = reliability concern.
        """
        self.logger.info("Classifying edges by weight")
        
        with self.gds.session() as session:
            result = session.run("""
                MATCH (a)-[d:DEPENDS_ON]->(b)
                RETURN a.id AS source, b.id AS target,
                       d.dependency_type AS dep_type,
                       d.weight AS weight
                ORDER BY d.weight DESC
            """)
            
            items = [
                {
                    "id": f"{r['source']}->{r['target']}",
                    "type": r["dep_type"],
                    "score": r["weight"] or 1.0,
                    "source": r["source"],
                    "target": r["target"],
                }
                for r in result
            ]
        
        return self.classifier.classify(items, metric_name="edge_weight")


# =============================================================================
# Utility Functions
# =============================================================================

def merge_classifications(
    classifications: List[ClassificationResult],
    strategy: str = "max",
) -> ClassificationResult:
    """
    Merge multiple classification results.
    
    Args:
        classifications: List of ClassificationResults to merge
        strategy: How to combine levels - "max" (most critical wins),
                  "avg" (average level), "vote" (majority wins)
    
    Returns:
        Merged ClassificationResult
    """
    if not classifications:
        raise ValueError("No classifications to merge")
    
    if len(classifications) == 1:
        return classifications[0]
    
    # Collect items by ID
    items_by_id = defaultdict(list)
    for cr in classifications:
        for item in cr.items:
            items_by_id[item.id].append(item)
    
    # Merge items
    merged_items = []
    for item_id, items in items_by_id.items():
        if strategy == "max":
            # Take the most critical level
            best = max(items, key=lambda x: x.level.numeric)
        elif strategy == "avg":
            # Average the numeric levels
            avg_level = sum(i.level.numeric for i in items) / len(items)
            levels = list(CriticalityLevel)
            best_level = min(levels, key=lambda l: abs(l.numeric - avg_level))
            best = items[0]
            best = ClassifiedItem(
                id=best.id,
                item_type=best.item_type,
                score=sum(i.score for i in items) / len(items),
                level=best_level,
                percentile=sum(i.percentile for i in items) / len(items),
                rank=best.rank,
                is_outlier=any(i.is_outlier for i in items),
                metadata=best.metadata,
            )
        else:  # vote
            # Most common level
            from collections import Counter
            level_counts = Counter(i.level for i in items)
            best_level = level_counts.most_common(1)[0][0]
            best = [i for i in items if i.level == best_level][0]
        
        merged_items.append(best)
    
    # Sort by score descending
    merged_items.sort(key=lambda x: x.score, reverse=True)
    
    # Rebuild by_level and summary
    by_level = defaultdict(list)
    for item in merged_items:
        by_level[item.level].append(item)
    
    summary = {level: len(items) for level, items in by_level.items()}
    
    return ClassificationResult(
        metric_name="merged",
        stats=classifications[0].stats,  # Use first stats
        items=merged_items,
        by_level=dict(by_level),
        summary=summary,
    )
