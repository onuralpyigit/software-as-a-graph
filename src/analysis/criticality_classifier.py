#!/usr/bin/env python3
"""
Box-Plot Criticality Classifier
================================

Classifies components and edges by criticality using the box-plot method
instead of static thresholds. This addresses the "sharp boundary problem"
where components with nearly identical scores receive vastly different
classifications.

The box-plot method uses statistical quartiles:
- Q1 (25th percentile): Lower quartile
- Q2 (50th percentile): Median
- Q3 (75th percentile): Upper quartile
- IQR = Q3 - Q1: Interquartile range

Classification levels:
- CRITICAL: score > Q3 + k*IQR (upper outliers)
- HIGH: Q3 < score <= Q3 + k*IQR
- MEDIUM: Q2 < score <= Q3
- LOW: Q1 < score <= Q2
- MINIMAL: score <= Q1

Optional fuzzy boundaries provide smooth transitions between levels.

Author: Software-as-a-Graph Research Project
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import defaultdict


# ============================================================================
# Enums and Data Classes
# ============================================================================

class CriticalityLevel(Enum):
    """Criticality levels based on box-plot classification"""
    CRITICAL = "critical"    # Upper outliers (> Q3 + k*IQR)
    HIGH = "high"            # Upper quartile to upper fence
    MEDIUM = "medium"        # Median to upper quartile
    LOW = "low"              # Lower quartile to median
    MINIMAL = "minimal"      # Below lower quartile
    
    def __lt__(self, other):
        order = [CriticalityLevel.CRITICAL, CriticalityLevel.HIGH, 
                 CriticalityLevel.MEDIUM, CriticalityLevel.LOW, CriticalityLevel.MINIMAL]
        return order.index(self) < order.index(other)
    
    @property
    def numeric_value(self) -> int:
        """Return numeric value for sorting (higher = more critical)"""
        values = {
            CriticalityLevel.CRITICAL: 5,
            CriticalityLevel.HIGH: 4,
            CriticalityLevel.MEDIUM: 3,
            CriticalityLevel.LOW: 2,
            CriticalityLevel.MINIMAL: 1
        }
        return values[self]


@dataclass
class BoxPlotStatistics:
    """Statistical values from box-plot analysis"""
    min_val: float
    q1: float           # 25th percentile
    median: float       # 50th percentile (Q2)
    q3: float           # 75th percentile
    max_val: float
    iqr: float          # Interquartile range (Q3 - Q1)
    lower_fence: float  # Q1 - k*IQR
    upper_fence: float  # Q3 + k*IQR
    k_factor: float     # IQR multiplier (default 1.5)
    count: int
    mean: float
    std_dev: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'min': round(self.min_val, 6),
            'q1': round(self.q1, 6),
            'median': round(self.median, 6),
            'q3': round(self.q3, 6),
            'max': round(self.max_val, 6),
            'iqr': round(self.iqr, 6),
            'lower_fence': round(self.lower_fence, 6),
            'upper_fence': round(self.upper_fence, 6),
            'k_factor': self.k_factor,
            'count': self.count,
            'mean': round(self.mean, 6),
            'std_dev': round(self.std_dev, 6)
        }
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get classification thresholds"""
        return {
            'critical': self.upper_fence,
            'high': self.q3,
            'medium': self.median,
            'low': self.q1,
            'minimal': self.lower_fence
        }


@dataclass
class FuzzyMembership:
    """Fuzzy membership degrees for smooth transitions"""
    critical: float = 0.0
    high: float = 0.0
    medium: float = 0.0
    low: float = 0.0
    minimal: float = 0.0
    
    def dominant_level(self) -> CriticalityLevel:
        """Return the level with highest membership"""
        memberships = {
            CriticalityLevel.CRITICAL: self.critical,
            CriticalityLevel.HIGH: self.high,
            CriticalityLevel.MEDIUM: self.medium,
            CriticalityLevel.LOW: self.low,
            CriticalityLevel.MINIMAL: self.minimal
        }
        return max(memberships, key=memberships.get)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'critical': round(self.critical, 4),
            'high': round(self.high, 4),
            'medium': round(self.medium, 4),
            'low': round(self.low, 4),
            'minimal': round(self.minimal, 4)
        }


@dataclass
class ClassifiedItem:
    """A classified component or edge"""
    item_id: str
    item_type: str  # 'component' or 'edge'
    score: float
    level: CriticalityLevel
    percentile: float
    z_score: float
    fuzzy_membership: Optional[FuzzyMembership] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'id': self.item_id,
            'type': self.item_type,
            'score': round(self.score, 6),
            'level': self.level.value,
            'level_numeric': self.level.numeric_value,
            'percentile': round(self.percentile, 2),
            'z_score': round(self.z_score, 4),
            'metadata': self.metadata
        }
        if self.fuzzy_membership:
            result['fuzzy_membership'] = self.fuzzy_membership.to_dict()
        return result


@dataclass
class ClassificationResult:
    """Result of box-plot classification"""
    statistics: BoxPlotStatistics
    items: List[ClassifiedItem]
    by_level: Dict[CriticalityLevel, List[ClassifiedItem]]
    metric_name: str
    use_fuzzy: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric': self.metric_name,
            'statistics': self.statistics.to_dict(),
            'thresholds': self.statistics.get_thresholds(),
            'use_fuzzy': self.use_fuzzy,
            'distribution': {
                level.value: len(items) 
                for level, items in self.by_level.items()
            },
            'items': [item.to_dict() for item in self.items]
        }
    
    def get_critical(self) -> List[ClassifiedItem]:
        """Get items classified as CRITICAL"""
        return self.by_level.get(CriticalityLevel.CRITICAL, [])
    
    def get_high_and_above(self) -> List[ClassifiedItem]:
        """Get items classified as HIGH or CRITICAL"""
        result = self.by_level.get(CriticalityLevel.CRITICAL, []).copy()
        result.extend(self.by_level.get(CriticalityLevel.HIGH, []))
        return sorted(result, key=lambda x: -x.score)


# ============================================================================
# Box-Plot Classifier
# ============================================================================

class BoxPlotClassifier:
    """
    Classifies items using the box-plot method.
    
    The box-plot method provides data-driven thresholds based on the
    actual distribution of scores, avoiding the sharp boundary problem
    of static thresholds.
    
    Features:
    - Statistical quartile-based classification
    - Configurable IQR multiplier (k-factor)
    - Optional fuzzy membership for smooth transitions
    - Z-score and percentile computation
    - Support for multiple metrics
    """
    
    def __init__(self, 
                 k_factor: float = 1.5,
                 use_fuzzy: bool = False,
                 fuzzy_width: float = 0.1):
        """
        Initialize the classifier.
        
        Args:
            k_factor: IQR multiplier for fence calculation (default: 1.5)
                     - 1.5 = standard outliers
                     - 3.0 = extreme outliers only
            use_fuzzy: Enable fuzzy membership for smooth transitions
            fuzzy_width: Width of fuzzy transition zones (as fraction of IQR)
        """
        self.k_factor = k_factor
        self.use_fuzzy = use_fuzzy
        self.fuzzy_width = fuzzy_width
        self.logger = logging.getLogger('BoxPlotClassifier')
    
    def compute_statistics(self, scores: List[float]) -> BoxPlotStatistics:
        """
        Compute box-plot statistics for a list of scores.
        
        Args:
            scores: List of numeric scores
            
        Returns:
            BoxPlotStatistics with quartiles and fences
        """
        if not scores:
            return BoxPlotStatistics(
                min_val=0, q1=0, median=0, q3=0, max_val=0,
                iqr=0, lower_fence=0, upper_fence=0,
                k_factor=self.k_factor, count=0, mean=0, std_dev=0
            )
        
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        
        # Calculate quartiles using linear interpolation
        q1 = self._percentile(sorted_scores, 25)
        median = self._percentile(sorted_scores, 50)
        q3 = self._percentile(sorted_scores, 75)
        
        iqr = q3 - q1
        
        # Calculate fences
        lower_fence = q1 - self.k_factor * iqr
        upper_fence = q3 + self.k_factor * iqr
        
        # Calculate mean and std dev
        mean = sum(scores) / n
        variance = sum((x - mean) ** 2 for x in scores) / n
        std_dev = math.sqrt(variance)
        
        return BoxPlotStatistics(
            min_val=sorted_scores[0],
            q1=q1,
            median=median,
            q3=q3,
            max_val=sorted_scores[-1],
            iqr=iqr,
            lower_fence=lower_fence,
            upper_fence=upper_fence,
            k_factor=self.k_factor,
            count=n,
            mean=mean,
            std_dev=std_dev
        )
    
    def _percentile(self, sorted_data: List[float], p: float) -> float:
        """Calculate percentile using linear interpolation"""
        n = len(sorted_data)
        if n == 1:
            return sorted_data[0]
        
        k = (n - 1) * p / 100
        f = math.floor(k)
        c = math.ceil(k)
        
        if f == c:
            return sorted_data[int(k)]
        
        return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)
    
    def classify_score(self, 
                       score: float, 
                       stats: BoxPlotStatistics) -> Tuple[CriticalityLevel, Optional[FuzzyMembership]]:
        """
        Classify a single score based on box-plot statistics.
        
        Args:
            score: The score to classify
            stats: Pre-computed box-plot statistics
            
        Returns:
            Tuple of (CriticalityLevel, optional FuzzyMembership)
        """
        # Crisp classification
        if score > stats.upper_fence:
            level = CriticalityLevel.CRITICAL
        elif score > stats.q3:
            level = CriticalityLevel.HIGH
        elif score > stats.median:
            level = CriticalityLevel.MEDIUM
        elif score > stats.q1:
            level = CriticalityLevel.LOW
        else:
            level = CriticalityLevel.MINIMAL
        
        # Fuzzy membership if enabled
        fuzzy = None
        if self.use_fuzzy:
            fuzzy = self._compute_fuzzy_membership(score, stats)
        
        return level, fuzzy
    
    def _compute_fuzzy_membership(self, 
                                   score: float, 
                                   stats: BoxPlotStatistics) -> FuzzyMembership:
        """
        Compute fuzzy membership degrees for smooth transitions.
        
        Uses trapezoidal membership functions with overlapping regions
        at the boundaries to avoid sharp transitions.
        """
        # Width of transition zone
        width = stats.iqr * self.fuzzy_width if stats.iqr > 0 else 0.1
        
        # Define membership functions for each level
        # Each returns a value in [0, 1]
        
        def critical_membership(x: float) -> float:
            """Critical: ramps up from upper_fence"""
            if x >= stats.upper_fence + width:
                return 1.0
            elif x > stats.upper_fence - width:
                return (x - (stats.upper_fence - width)) / (2 * width)
            return 0.0
        
        def high_membership(x: float) -> float:
            """High: between Q3 and upper_fence"""
            center = (stats.q3 + stats.upper_fence) / 2
            if stats.q3 - width < x <= stats.q3 + width:
                return (x - (stats.q3 - width)) / (2 * width)
            elif stats.q3 + width < x < stats.upper_fence - width:
                return 1.0
            elif stats.upper_fence - width <= x < stats.upper_fence + width:
                return 1.0 - (x - (stats.upper_fence - width)) / (2 * width)
            return 0.0
        
        def medium_membership(x: float) -> float:
            """Medium: between median and Q3"""
            if stats.median - width < x <= stats.median + width:
                return (x - (stats.median - width)) / (2 * width)
            elif stats.median + width < x < stats.q3 - width:
                return 1.0
            elif stats.q3 - width <= x < stats.q3 + width:
                return 1.0 - (x - (stats.q3 - width)) / (2 * width)
            return 0.0
        
        def low_membership(x: float) -> float:
            """Low: between Q1 and median"""
            if stats.q1 - width < x <= stats.q1 + width:
                return (x - (stats.q1 - width)) / (2 * width)
            elif stats.q1 + width < x < stats.median - width:
                return 1.0
            elif stats.median - width <= x < stats.median + width:
                return 1.0 - (x - (stats.median - width)) / (2 * width)
            return 0.0
        
        def minimal_membership(x: float) -> float:
            """Minimal: below Q1"""
            if x <= stats.q1 - width:
                return 1.0
            elif x < stats.q1 + width:
                return 1.0 - (x - (stats.q1 - width)) / (2 * width)
            return 0.0
        
        # Compute memberships
        memberships = FuzzyMembership(
            critical=max(0, min(1, critical_membership(score))),
            high=max(0, min(1, high_membership(score))),
            medium=max(0, min(1, medium_membership(score))),
            low=max(0, min(1, low_membership(score))),
            minimal=max(0, min(1, minimal_membership(score)))
        )
        
        # Normalize so memberships sum to 1
        total = (memberships.critical + memberships.high + memberships.medium + 
                 memberships.low + memberships.minimal)
        if total > 0:
            memberships.critical /= total
            memberships.high /= total
            memberships.medium /= total
            memberships.low /= total
            memberships.minimal /= total
        
        return memberships
    
    def classify_items(self,
                       items: List[Dict[str, Any]],
                       score_key: str = 'score',
                       id_key: str = 'id',
                       type_key: str = 'type',
                       item_type: str = 'component',
                       metric_name: str = 'criticality') -> ClassificationResult:
        """
        Classify a list of items based on their scores.
        
        Args:
            items: List of dictionaries containing scores
            score_key: Key for the score value in each item
            id_key: Key for the item identifier
            type_key: Key for the item type (optional)
            item_type: Default item type if not in data
            metric_name: Name of the metric being classified
            
        Returns:
            ClassificationResult with all classified items
        """
        self.logger.info(f"Classifying {len(items)} items by {metric_name}...")
        
        # Extract scores
        scores = [item.get(score_key, 0) for item in items if score_key in item]
        
        if not scores:
            self.logger.warning("No scores found in items")
            return ClassificationResult(
                statistics=self.compute_statistics([]),
                items=[],
                by_level={level: [] for level in CriticalityLevel},
                metric_name=metric_name,
                use_fuzzy=self.use_fuzzy
            )
        
        # Compute statistics
        stats = self.compute_statistics(scores)
        self.logger.info(f"Statistics: Q1={stats.q1:.4f}, median={stats.median:.4f}, "
                        f"Q3={stats.q3:.4f}, IQR={stats.iqr:.4f}")
        
        # Classify each item
        classified_items = []
        by_level = {level: [] for level in CriticalityLevel}
        
        for item in items:
            if score_key not in item:
                continue
            
            score = item[score_key]
            item_id = item.get(id_key, str(len(classified_items)))
            itype = item.get(type_key, item_type)
            
            # Classify
            level, fuzzy = self.classify_score(score, stats)
            
            # Calculate percentile and z-score
            percentile = self._calculate_percentile(score, scores)
            z_score = (score - stats.mean) / stats.std_dev if stats.std_dev > 0 else 0
            
            # Create classified item
            classified = ClassifiedItem(
                item_id=item_id,
                item_type=itype,
                score=score,
                level=level,
                percentile=percentile,
                z_score=z_score,
                fuzzy_membership=fuzzy,
                metadata={k: v for k, v in item.items() 
                         if k not in [score_key, id_key, type_key]}
            )
            
            classified_items.append(classified)
            by_level[level].append(classified)
        
        # Sort items by score (descending)
        classified_items.sort(key=lambda x: -x.score)
        for level in by_level:
            by_level[level].sort(key=lambda x: -x.score)
        
        # Log distribution
        dist = {level.value: len(items) for level, items in by_level.items()}
        self.logger.info(f"Distribution: {dist}")
        
        return ClassificationResult(
            statistics=stats,
            items=classified_items,
            by_level=by_level,
            metric_name=metric_name,
            use_fuzzy=self.use_fuzzy
        )
    
    def _calculate_percentile(self, score: float, all_scores: List[float]) -> float:
        """Calculate the percentile rank of a score"""
        below = sum(1 for s in all_scores if s < score)
        equal = sum(1 for s in all_scores if s == score)
        return 100 * (below + 0.5 * equal) / len(all_scores)


# ============================================================================
# GDS-Integrated Classifier
# ============================================================================

class GDSCriticalityClassifier:
    """
    Criticality classifier integrated with Neo4j GDS.
    
    Retrieves scores from GDS algorithms and classifies components
    and edges using the box-plot method.
    """
    
    def __init__(self, 
                 gds_client,
                 k_factor: float = 1.5,
                 use_fuzzy: bool = False,
                 fuzzy_width: float = 0.1):
        """
        Initialize the GDS-integrated classifier.
        
        Args:
            gds_client: GDSClient instance
            k_factor: IQR multiplier for outlier detection
            use_fuzzy: Enable fuzzy membership
            fuzzy_width: Width of fuzzy transition zones
        """
        self.gds = gds_client
        self.classifier = BoxPlotClassifier(
            k_factor=k_factor,
            use_fuzzy=use_fuzzy,
            fuzzy_width=fuzzy_width
        )
        self.logger = logging.getLogger('GDSCriticalityClassifier')
    
    def classify_by_betweenness(self, 
                                 projection_name: str,
                                 weighted: bool = True) -> ClassificationResult:
        """
        Classify components by betweenness centrality.
        
        High betweenness = component is on many shortest paths = critical bottleneck.
        """
        self.logger.info("Classifying by betweenness centrality...")
        
        results = self.gds.betweenness_centrality(projection_name, weighted=weighted)
        
        items = [
            {
                'id': r.node_id,
                'type': r.node_type,
                'score': r.score,
                'rank': r.rank
            }
            for r in results
        ]
        
        return self.classifier.classify_items(
            items,
            metric_name='betweenness_centrality'
        )
    
    def classify_by_pagerank(self,
                              projection_name: str,
                              weighted: bool = True) -> ClassificationResult:
        """
        Classify components by PageRank.
        
        High PageRank = component receives important dependencies = critical for availability.
        """
        self.logger.info("Classifying by PageRank...")
        
        results = self.gds.pagerank(projection_name, weighted=weighted)
        
        items = [
            {
                'id': r.node_id,
                'type': r.node_type,
                'score': r.score,
                'rank': r.rank
            }
            for r in results
        ]
        
        return self.classifier.classify_items(
            items,
            metric_name='pagerank'
        )
    
    def classify_by_degree(self,
                            projection_name: str,
                            weighted: bool = True,
                            orientation: str = 'UNDIRECTED') -> ClassificationResult:
        """
        Classify components by degree centrality.
        
        High degree = component has many dependencies = potential maintainability issue.
        """
        self.logger.info(f"Classifying by {orientation.lower()} degree centrality...")
        
        results = self.gds.degree_centrality(
            projection_name, 
            weighted=weighted,
            orientation=orientation
        )
        
        items = [
            {
                'id': r.node_id,
                'type': r.node_type,
                'score': r.score,
                'rank': r.rank
            }
            for r in results
        ]
        
        return self.classifier.classify_items(
            items,
            metric_name=f'degree_centrality_{orientation.lower()}'
        )
    
    def classify_by_weighted_importance(self) -> ClassificationResult:
        """
        Classify components by weighted incoming dependency importance.
        """
        self.logger.info("Classifying by weighted importance...")
        
        results = self.gds.get_weighted_node_importance(top_k=None)
        
        items = [
            {
                'id': r['node_id'],
                'type': r['node_type'],
                'score': r['total_weight'],
                'dependency_count': r['dependency_count'],
                'avg_weight': r['avg_weight'],
                'max_weight': r['max_weight']
            }
            for r in results
        ]
        
        return self.classifier.classify_items(
            items,
            metric_name='weighted_importance'
        )
    
    def classify_by_composite_score(self,
                                     projection_name: str,
                                     weighted: bool = True,
                                     weights: Optional[Dict[str, float]] = None) -> ClassificationResult:
        """
        Classify components by a composite criticality score.
        
        Composite score combines multiple metrics:
        C_score = α·BC + β·PR + γ·DC + δ·AP
        
        Where:
        - BC: Normalized betweenness centrality
        - PR: Normalized PageRank
        - DC: Normalized degree centrality
        - AP: Articulation point indicator (1.0 if AP, 0.0 otherwise)
        
        Args:
            projection_name: GDS graph projection name
            weighted: Use weighted algorithms
            weights: Custom weights for each metric (default: equal weights)
            
        Returns:
            ClassificationResult with composite scores
        """
        self.logger.info("Computing composite criticality scores...")
        
        # Default weights
        if weights is None:
            weights = {
                'betweenness': 0.3,
                'pagerank': 0.25,
                'degree': 0.25,
                'articulation': 0.2
            }
        
        # Get individual metrics
        bc_results = self.gds.betweenness_centrality(projection_name, weighted=weighted)
        pr_results = self.gds.pagerank(projection_name, weighted=weighted)
        dc_results = self.gds.degree_centrality(projection_name, weighted=weighted, orientation='UNDIRECTED')
        ap_results = self.gds.find_articulation_points()
        
        # Create lookups
        bc_lookup = {r.node_id: r.score for r in bc_results}
        pr_lookup = {r.node_id: r.score for r in pr_results}
        dc_lookup = {r.node_id: r.score for r in dc_results}
        ap_set = {r['node_id'] for r in ap_results}
        
        # Normalize scores to [0, 1]
        def normalize(lookup: Dict[str, float]) -> Dict[str, float]:
            if not lookup:
                return {}
            max_val = max(lookup.values())
            if max_val == 0:
                return {k: 0.0 for k in lookup}
            return {k: v / max_val for k, v in lookup.items()}
        
        bc_norm = normalize(bc_lookup)
        pr_norm = normalize(pr_lookup)
        dc_norm = normalize(dc_lookup)
        
        # Compute composite scores
        all_nodes = set(bc_lookup.keys()) | set(pr_lookup.keys()) | set(dc_lookup.keys())
        
        items = []
        for node_id in all_nodes:
            bc = bc_norm.get(node_id, 0)
            pr = pr_norm.get(node_id, 0)
            dc = dc_norm.get(node_id, 0)
            ap = 1.0 if node_id in ap_set else 0.0
            
            composite = (
                weights['betweenness'] * bc +
                weights['pagerank'] * pr +
                weights['degree'] * dc +
                weights['articulation'] * ap
            )
            
            # Get node type from one of the results
            node_type = 'Unknown'
            for r in bc_results:
                if r.node_id == node_id:
                    node_type = r.node_type
                    break
            
            items.append({
                'id': node_id,
                'type': node_type,
                'score': composite,
                'betweenness_norm': bc,
                'pagerank_norm': pr,
                'degree_norm': dc,
                'is_articulation_point': ap > 0,
                'weights_used': weights
            })
        
        return self.classifier.classify_items(
            items,
            metric_name='composite_criticality'
        )
    
    def classify_edges_by_weight(self,
                                  percentile_threshold: float = 50) -> ClassificationResult:
        """
        Classify edges by their weight property.
        
        Higher weight = more critical dependency.
        """
        self.logger.info("Classifying edges by weight...")
        
        # Get all edges with weights
        with self.gds.session() as session:
            query = """
            MATCH (a)-[r:DEPENDS_ON]->(b)
            WHERE r.weight IS NOT NULL
            RETURN a.id AS source,
                   b.id AS target,
                   labels(a)[0] AS sourceType,
                   labels(b)[0] AS targetType,
                   r.dependency_type AS depType,
                   r.weight AS weight
            ORDER BY r.weight DESC
            """
            
            items = []
            for record in session.run(query):
                items.append({
                    'id': f"{record['source']}->{record['target']}",
                    'type': 'edge',
                    'score': record['weight'],
                    'source': record['source'],
                    'target': record['target'],
                    'source_type': record['sourceType'],
                    'target_type': record['targetType'],
                    'dependency_type': record['depType']
                })
        
        return self.classifier.classify_items(
            items,
            item_type='edge',
            metric_name='edge_weight'
        )
    
    def get_comprehensive_classification(self,
                                          projection_name: str,
                                          weighted: bool = True) -> Dict[str, ClassificationResult]:
        """
        Run comprehensive classification across all metrics.
        
        Returns:
            Dictionary mapping metric names to ClassificationResults
        """
        self.logger.info("Running comprehensive classification...")
        
        results = {}
        
        # Component classifications
        results['betweenness'] = self.classify_by_betweenness(projection_name, weighted)
        results['pagerank'] = self.classify_by_pagerank(projection_name, weighted)
        results['degree'] = self.classify_by_degree(projection_name, weighted)
        results['composite'] = self.classify_by_composite_score(projection_name, weighted)
        
        # Edge classification
        results['edge_weight'] = self.classify_edges_by_weight()
        
        # Weighted importance if applicable
        if weighted:
            results['weighted_importance'] = self.classify_by_weighted_importance()
        
        return results


# ============================================================================
# Utility Functions
# ============================================================================

def merge_classifications(classifications: Dict[str, ClassificationResult],
                          weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """
    Merge multiple classification results into a unified ranking.
    
    Args:
        classifications: Dictionary of metric_name -> ClassificationResult
        weights: Optional weights for each metric (default: equal weights)
        
    Returns:
        List of items with merged criticality assessment
    """
    if weights is None:
        weights = {name: 1.0 / len(classifications) for name in classifications}
    
    # Collect all items and their scores per metric
    item_scores = defaultdict(lambda: {'scores': {}, 'levels': {}, 'type': 'Unknown'})
    
    for metric_name, result in classifications.items():
        for item in result.items:
            item_scores[item.item_id]['scores'][metric_name] = item.score
            item_scores[item.item_id]['levels'][metric_name] = item.level
            item_scores[item.item_id]['type'] = item.item_type
    
    # Compute merged scores
    merged = []
    for item_id, data in item_scores.items():
        # Weighted average of normalized scores
        total_weight = 0
        weighted_score = 0
        
        for metric_name, score in data['scores'].items():
            if metric_name in weights:
                # Normalize score using the classification's statistics
                if metric_name in classifications:
                    stats = classifications[metric_name].statistics
                    if stats.max_val > stats.min_val:
                        norm_score = (score - stats.min_val) / (stats.max_val - stats.min_val)
                    else:
                        norm_score = 0.5
                else:
                    norm_score = score
                
                weighted_score += weights[metric_name] * norm_score
                total_weight += weights[metric_name]
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0
        
        # Determine dominant level
        level_counts = defaultdict(int)
        for level in data['levels'].values():
            level_counts[level] += 1
        
        dominant_level = max(level_counts, key=level_counts.get) if level_counts else CriticalityLevel.MINIMAL
        
        merged.append({
            'id': item_id,
            'type': data['type'],
            'merged_score': round(final_score, 4),
            'dominant_level': dominant_level.value,
            'scores_by_metric': {k: round(v, 4) for k, v in data['scores'].items()},
            'levels_by_metric': {k: v.value for k, v in data['levels'].items()}
        })
    
    # Sort by merged score
    merged.sort(key=lambda x: -x['merged_score'])
    
    return merged
