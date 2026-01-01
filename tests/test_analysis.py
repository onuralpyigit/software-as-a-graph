#!/usr/bin/env python3
"""
Test Suite for Analysis Module - Version 5.0

Comprehensive tests for:
- Box-plot classification
- Component-type analysis
- Problem detection
- Anti-pattern detection
- Edge criticality analysis

Run with: python -m pytest tests/test_analysis.py -v
Or:       python tests/test_analysis.py

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis import (
    # Classifier
    BoxPlotClassifier,
    CriticalityLevel,
    BoxPlotStats,
    ClassifiedItem,
    ClassificationResult,
    # Problem Detection
    ProblemType,
    ProblemSeverity,
    QualityAttribute,
    Problem,
    Symptom,
    # Anti-Pattern Detection
    AntiPatternType,
    PatternSeverity,
    AntiPattern,
    # GDS Client (enums only, don't need connection)
)

from src.analysis.gds_client import DependencyType, ComponentType


# =============================================================================
# Test: Enums
# =============================================================================

class TestEnums(unittest.TestCase):
    """Test enum definitions and properties"""
    
    def test_criticality_level_values(self):
        """Test CriticalityLevel enum values"""
        self.assertEqual(CriticalityLevel.CRITICAL.value, "critical")
        self.assertEqual(CriticalityLevel.HIGH.value, "high")
        self.assertEqual(CriticalityLevel.MEDIUM.value, "medium")
        self.assertEqual(CriticalityLevel.LOW.value, "low")
        self.assertEqual(CriticalityLevel.MINIMAL.value, "minimal")
    
    def test_criticality_level_numeric(self):
        """Test CriticalityLevel numeric ordering"""
        self.assertEqual(CriticalityLevel.CRITICAL.numeric, 5)
        self.assertEqual(CriticalityLevel.HIGH.numeric, 4)
        self.assertEqual(CriticalityLevel.MEDIUM.numeric, 3)
        self.assertEqual(CriticalityLevel.LOW.numeric, 2)
        self.assertEqual(CriticalityLevel.MINIMAL.numeric, 1)
    
    def test_criticality_level_comparison(self):
        """Test CriticalityLevel comparison operators"""
        self.assertTrue(CriticalityLevel.CRITICAL > CriticalityLevel.HIGH)
        self.assertTrue(CriticalityLevel.HIGH > CriticalityLevel.MEDIUM)
        self.assertTrue(CriticalityLevel.MEDIUM > CriticalityLevel.LOW)
        self.assertTrue(CriticalityLevel.LOW > CriticalityLevel.MINIMAL)
        
        self.assertTrue(CriticalityLevel.MINIMAL < CriticalityLevel.LOW)
        self.assertTrue(CriticalityLevel.MEDIUM >= CriticalityLevel.MEDIUM)
        self.assertTrue(CriticalityLevel.HIGH <= CriticalityLevel.CRITICAL)
    
    def test_criticality_level_has_color(self):
        """Test CriticalityLevel has color property"""
        for level in CriticalityLevel:
            self.assertIsInstance(level.color, str)
            self.assertTrue(level.color.startswith("\033["))
    
    def test_criticality_level_has_description(self):
        """Test CriticalityLevel has description property"""
        for level in CriticalityLevel:
            self.assertIsInstance(level.description, str)
            self.assertGreater(len(level.description), 0)
    
    def test_dependency_type_values(self):
        """Test DependencyType enum values"""
        self.assertEqual(DependencyType.APP_TO_APP.value, "app_to_app")
        self.assertEqual(DependencyType.NODE_TO_NODE.value, "node_to_node")
        self.assertEqual(DependencyType.APP_TO_BROKER.value, "app_to_broker")
        self.assertEqual(DependencyType.NODE_TO_BROKER.value, "node_to_broker")
    
    def test_component_type_values(self):
        """Test ComponentType enum values"""
        self.assertEqual(ComponentType.APPLICATION.value, "Application")
        self.assertEqual(ComponentType.BROKER.value, "Broker")
        self.assertEqual(ComponentType.TOPIC.value, "Topic")
        self.assertEqual(ComponentType.NODE.value, "Node")
    
    def test_problem_type_values(self):
        """Test ProblemType enum has expected values"""
        # Reliability problems
        self.assertIn(ProblemType.SINGLE_POINT_OF_FAILURE, ProblemType)
        self.assertIn(ProblemType.CASCADE_RISK, ProblemType)
        self.assertIn(ProblemType.CRITICAL_BRIDGE, ProblemType)
        
        # Maintainability problems
        self.assertIn(ProblemType.HIGH_COUPLING, ProblemType)
        self.assertIn(ProblemType.GOD_COMPONENT, ProblemType)
        self.assertIn(ProblemType.POOR_MODULARITY, ProblemType)
        
        # Availability problems
        self.assertIn(ProblemType.BOTTLENECK, ProblemType)
        self.assertIn(ProblemType.NO_REDUNDANCY, ProblemType)
    
    def test_problem_type_quality_attribute(self):
        """Test ProblemType maps to correct QualityAttribute"""
        self.assertEqual(
            ProblemType.SINGLE_POINT_OF_FAILURE.quality_attribute,
            QualityAttribute.RELIABILITY
        )
        self.assertEqual(
            ProblemType.HIGH_COUPLING.quality_attribute,
            QualityAttribute.MAINTAINABILITY
        )
        self.assertEqual(
            ProblemType.BOTTLENECK.quality_attribute,
            QualityAttribute.AVAILABILITY
        )
    
    def test_problem_severity_numeric(self):
        """Test ProblemSeverity numeric ordering"""
        self.assertEqual(ProblemSeverity.CRITICAL.numeric, 4)
        self.assertEqual(ProblemSeverity.HIGH.numeric, 3)
        self.assertEqual(ProblemSeverity.MEDIUM.numeric, 2)
        self.assertEqual(ProblemSeverity.LOW.numeric, 1)
    
    def test_antipattern_type_values(self):
        """Test AntiPatternType enum has expected values"""
        self.assertIn(AntiPatternType.GOD_TOPIC, AntiPatternType)
        self.assertIn(AntiPatternType.BOTTLENECK_BROKER, AntiPatternType)
        self.assertIn(AntiPatternType.CHATTY_APPLICATION, AntiPatternType)
        self.assertIn(AntiPatternType.HUB_AND_SPOKE, AntiPatternType)
        self.assertIn(AntiPatternType.CIRCULAR_DEPENDENCY, AntiPatternType)
        self.assertIn(AntiPatternType.ORPHAN_COMPONENT, AntiPatternType)
    
    def test_antipattern_type_has_description(self):
        """Test AntiPatternType has description property"""
        for pattern_type in AntiPatternType:
            self.assertIsInstance(pattern_type.description, str)
            self.assertGreater(len(pattern_type.description), 0)
    
    def test_quality_attribute_values(self):
        """Test QualityAttribute enum values"""
        self.assertEqual(QualityAttribute.RELIABILITY.value, "reliability")
        self.assertEqual(QualityAttribute.MAINTAINABILITY.value, "maintainability")
        self.assertEqual(QualityAttribute.AVAILABILITY.value, "availability")
    
    def test_quality_attribute_has_description(self):
        """Test QualityAttribute has description property"""
        for qa in QualityAttribute:
            self.assertIsInstance(qa.description, str)
            self.assertGreater(len(qa.description), 0)


# =============================================================================
# Test: Box-Plot Statistics
# =============================================================================

class TestBoxPlotStats(unittest.TestCase):
    """Test box-plot statistics calculation"""
    
    def setUp(self):
        self.classifier = BoxPlotClassifier(k_factor=1.5)
    
    def test_calculate_stats_basic(self):
        """Test basic statistics calculation"""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        stats = self.classifier.calculate_stats(scores)
        
        self.assertEqual(stats.min_val, 1.0)
        self.assertEqual(stats.max_val, 10.0)
        self.assertEqual(stats.count, 10)
        self.assertAlmostEqual(stats.mean, 5.5, places=2)
    
    def test_calculate_stats_quartiles(self):
        """Test quartile calculation"""
        # 20 values for clear quartiles
        scores = list(range(1, 21))  # 1-20
        stats = self.classifier.calculate_stats(scores)
        
        # Q1 should be around 5.75, median around 10.5, Q3 around 15.25
        self.assertAlmostEqual(stats.q1, 5.75, places=1)
        self.assertAlmostEqual(stats.median, 10.5, places=1)
        self.assertAlmostEqual(stats.q3, 15.25, places=1)
    
    def test_calculate_stats_iqr(self):
        """Test IQR calculation"""
        scores = list(range(1, 21))
        stats = self.classifier.calculate_stats(scores)
        
        expected_iqr = stats.q3 - stats.q1
        self.assertAlmostEqual(stats.iqr, expected_iqr, places=4)
    
    def test_calculate_stats_fences(self):
        """Test fence calculation with k=1.5"""
        scores = list(range(1, 21))
        stats = self.classifier.calculate_stats(scores)
        
        expected_lower_fence = stats.q1 - 1.5 * stats.iqr
        expected_upper_fence = stats.q3 + 1.5 * stats.iqr
        
        self.assertAlmostEqual(stats.lower_fence, expected_lower_fence, places=4)
        self.assertAlmostEqual(stats.upper_fence, expected_upper_fence, places=4)
    
    def test_calculate_stats_single_value(self):
        """Test statistics with single value"""
        scores = [5.0]
        stats = self.classifier.calculate_stats(scores)
        
        self.assertEqual(stats.min_val, 5.0)
        self.assertEqual(stats.max_val, 5.0)
        self.assertEqual(stats.median, 5.0)
        self.assertEqual(stats.q1, 5.0)
        self.assertEqual(stats.q3, 5.0)
        self.assertEqual(stats.iqr, 0.0)
        self.assertEqual(stats.count, 1)
    
    def test_calculate_stats_two_values(self):
        """Test statistics with two values"""
        scores = [1.0, 10.0]
        stats = self.classifier.calculate_stats(scores)
        
        self.assertEqual(stats.min_val, 1.0)
        self.assertEqual(stats.max_val, 10.0)
        self.assertAlmostEqual(stats.median, 5.5, places=2)
        self.assertEqual(stats.count, 2)
    
    def test_calculate_stats_empty(self):
        """Test statistics with empty data"""
        stats = self.classifier.calculate_stats([])
        self.assertEqual(stats.count, 0)
    
    def test_boxplot_stats_to_dict(self):
        """Test BoxPlotStats serialization"""
        stats = BoxPlotStats(
            min_val=1.0, q1=2.5, median=5.0, q3=7.5, max_val=10.0,
            iqr=5.0, lower_fence=-5.0, upper_fence=15.0,
            mean=5.0, std_dev=2.5, count=10, k_factor=1.5
        )
        
        d = stats.to_dict()
        self.assertIn("min", d)
        self.assertIn("q1", d)
        self.assertIn("median", d)
        self.assertIn("q3", d)
        self.assertIn("max", d)
        self.assertIn("iqr", d)
        self.assertIn("upper_fence", d)
        self.assertIn("count", d)
    
    def test_boxplot_stats_empty_factory(self):
        """Test BoxPlotStats.empty() factory method"""
        stats = BoxPlotStats.empty(k_factor=2.0)
        self.assertEqual(stats.count, 0)
        self.assertEqual(stats.k_factor, 2.0)


# =============================================================================
# Test: Classification
# =============================================================================

class TestClassification(unittest.TestCase):
    """Test box-plot classification"""
    
    def setUp(self):
        self.classifier = BoxPlotClassifier(k_factor=1.5)
    
    def test_classify_score_critical(self):
        """Test score classification as CRITICAL (outlier)"""
        # Create stats with upper fence at 100
        stats = BoxPlotStats(
            min_val=0, q1=25, median=50, q3=75, max_val=100,
            iqr=50, lower_fence=-50, upper_fence=150,
            mean=50, std_dev=25, count=20, k_factor=1.5
        )
        
        # Score above upper fence -> CRITICAL
        level, is_outlier = self.classifier.classify_score(200, stats)
        self.assertEqual(level, CriticalityLevel.CRITICAL)
        self.assertTrue(is_outlier)
    
    def test_classify_score_high(self):
        """Test score classification as HIGH"""
        stats = BoxPlotStats(
            min_val=0, q1=25, median=50, q3=75, max_val=100,
            iqr=50, lower_fence=-50, upper_fence=150,
            mean=50, std_dev=25, count=20, k_factor=1.5
        )
        
        # Score between Q3 and upper fence -> HIGH
        level, is_outlier = self.classifier.classify_score(100, stats)
        self.assertEqual(level, CriticalityLevel.HIGH)
        self.assertFalse(is_outlier)
    
    def test_classify_score_medium(self):
        """Test score classification as MEDIUM"""
        stats = BoxPlotStats(
            min_val=0, q1=25, median=50, q3=75, max_val=100,
            iqr=50, lower_fence=-50, upper_fence=150,
            mean=50, std_dev=25, count=20, k_factor=1.5
        )
        
        # Score between median and Q3 -> MEDIUM
        level, is_outlier = self.classifier.classify_score(60, stats)
        self.assertEqual(level, CriticalityLevel.MEDIUM)
        self.assertFalse(is_outlier)
    
    def test_classify_score_low(self):
        """Test score classification as LOW"""
        stats = BoxPlotStats(
            min_val=0, q1=25, median=50, q3=75, max_val=100,
            iqr=50, lower_fence=-50, upper_fence=150,
            mean=50, std_dev=25, count=20, k_factor=1.5
        )
        
        # Score between Q1 and median -> LOW
        level, is_outlier = self.classifier.classify_score(40, stats)
        self.assertEqual(level, CriticalityLevel.LOW)
        self.assertFalse(is_outlier)
    
    def test_classify_score_minimal(self):
        """Test score classification as MINIMAL"""
        stats = BoxPlotStats(
            min_val=0, q1=25, median=50, q3=75, max_val=100,
            iqr=50, lower_fence=-50, upper_fence=150,
            mean=50, std_dev=25, count=20, k_factor=1.5
        )
        
        # Score below Q1 -> MINIMAL
        level, is_outlier = self.classifier.classify_score(10, stats)
        self.assertEqual(level, CriticalityLevel.MINIMAL)
        self.assertFalse(is_outlier)
    
    def test_classify_score_lower_outlier(self):
        """Test score classification as lower outlier"""
        stats = BoxPlotStats(
            min_val=0, q1=25, median=50, q3=75, max_val=100,
            iqr=50, lower_fence=-50, upper_fence=150,
            mean=50, std_dev=25, count=20, k_factor=1.5
        )
        
        # Score below lower fence -> MINIMAL but outlier
        level, is_outlier = self.classifier.classify_score(-100, stats)
        self.assertEqual(level, CriticalityLevel.MINIMAL)
        self.assertTrue(is_outlier)
    
    def test_classify_items_list(self):
        """Test classification of item list"""
        items = [
            {"id": "app1", "type": "Application", "score": 100},
            {"id": "app2", "type": "Application", "score": 50},
            {"id": "app3", "type": "Application", "score": 25},
            {"id": "app4", "type": "Application", "score": 10},
            {"id": "app5", "type": "Application", "score": 5},
        ]
        
        result = self.classifier.classify(items, metric_name="test")
        
        self.assertIsInstance(result, ClassificationResult)
        self.assertEqual(len(result.items), 5)
        self.assertEqual(result.metric_name, "test")
        
        # Items should be sorted by score descending
        self.assertEqual(result.items[0].id, "app1")
        self.assertEqual(result.items[0].rank, 1)
        self.assertEqual(result.items[4].id, "app5")
        self.assertEqual(result.items[4].rank, 5)
    
    def test_classify_returns_summary(self):
        """Test classification returns level summary"""
        items = [
            {"id": f"item{i}", "type": "Test", "score": i * 10}
            for i in range(1, 21)
        ]
        
        result = self.classifier.classify(items, metric_name="test")
        
        self.assertIn(CriticalityLevel.CRITICAL, result.summary)
        self.assertIn(CriticalityLevel.HIGH, result.summary)
        self.assertIn(CriticalityLevel.MEDIUM, result.summary)
        self.assertIn(CriticalityLevel.LOW, result.summary)
        self.assertIn(CriticalityLevel.MINIMAL, result.summary)
        
        # Total should equal item count
        total = sum(result.summary.values())
        self.assertEqual(total, 20)
    
    def test_classify_by_level_grouping(self):
        """Test classification groups items by level"""
        items = [
            {"id": f"item{i}", "type": "Test", "score": i * 10}
            for i in range(1, 21)
        ]
        
        result = self.classifier.classify(items, metric_name="test")
        
        # by_level should contain all levels
        for level in CriticalityLevel:
            self.assertIn(level, result.by_level)
        
        # Items in each level should match summary count
        for level, level_items in result.by_level.items():
            self.assertEqual(len(level_items), result.summary[level])
    
    def test_classify_empty_list(self):
        """Test classification of empty list"""
        result = self.classifier.classify([], metric_name="test")
        
        self.assertEqual(len(result.items), 0)
        self.assertEqual(result.stats.count, 0)
        for level in CriticalityLevel:
            self.assertEqual(result.summary[level], 0)
    
    def test_classify_preserves_metadata(self):
        """Test classification preserves extra metadata"""
        items = [
            {"id": "app1", "type": "Application", "score": 100, "extra": "data1"},
            {"id": "app2", "type": "Application", "score": 50, "extra": "data2"},
        ]
        
        result = self.classifier.classify(items, metric_name="test")
        
        self.assertEqual(result.items[0].metadata.get("extra"), "data1")
        self.assertEqual(result.items[1].metadata.get("extra"), "data2")
    
    def test_classify_calculates_percentile(self):
        """Test classification calculates percentiles"""
        items = [
            {"id": f"item{i}", "type": "Test", "score": i}
            for i in range(1, 11)
        ]
        
        result = self.classifier.classify(items, metric_name="test")
        
        # Highest score should have highest percentile
        self.assertGreater(result.items[0].percentile, 80)
        # Lowest score should have lowest percentile
        self.assertLess(result.items[-1].percentile, 20)
    
    def test_classify_by_type_separates_types(self):
        """Test classify_by_type separates component types"""
        items = [
            {"id": "app1", "type": "Application", "score": 100},
            {"id": "app2", "type": "Application", "score": 50},
            {"id": "broker1", "type": "Broker", "score": 75},
            {"id": "broker2", "type": "Broker", "score": 25},
        ]
        
        results = self.classifier.classify_by_type(items, metric_name="test")
        
        self.assertIn("Application", results)
        self.assertIn("Broker", results)
        self.assertEqual(len(results["Application"].items), 2)
        self.assertEqual(len(results["Broker"].items), 2)
    
    def test_different_k_factors(self):
        """Test different k_factor values affect classification"""
        items = [
            {"id": f"item{i}", "type": "Test", "score": i * 10}
            for i in range(1, 21)
        ]
        
        # Conservative (fewer outliers)
        classifier_conservative = BoxPlotClassifier(k_factor=3.0)
        result_conservative = classifier_conservative.classify(items, metric_name="test")
        
        # Aggressive (more outliers)
        classifier_aggressive = BoxPlotClassifier(k_factor=1.0)
        result_aggressive = classifier_aggressive.classify(items, metric_name="test")
        
        # Aggressive should have equal or more outliers
        self.assertGreaterEqual(
            result_aggressive.outlier_count,
            result_conservative.outlier_count
        )


# =============================================================================
# Test: ClassificationResult
# =============================================================================

class TestClassificationResult(unittest.TestCase):
    """Test ClassificationResult data class"""
    
    def setUp(self):
        self.classifier = BoxPlotClassifier(k_factor=1.5)
        self.items = [
            {"id": f"item{i}", "type": "Test", "score": i * 10}
            for i in range(1, 21)
        ]
        self.result = self.classifier.classify(self.items, metric_name="test")
    
    def test_get_critical(self):
        """Test get_critical() returns critical items"""
        critical = self.result.get_critical()
        for item in critical:
            self.assertEqual(item.level, CriticalityLevel.CRITICAL)
    
    def test_get_high_and_above(self):
        """Test get_high_and_above() returns HIGH and CRITICAL"""
        high_and_above = self.result.get_high_and_above()
        for item in high_and_above:
            self.assertIn(item.level, [CriticalityLevel.HIGH, CriticalityLevel.CRITICAL])
    
    def test_get_by_type(self):
        """Test get_by_type() filters by type"""
        items = [
            {"id": "app1", "type": "Application", "score": 100},
            {"id": "broker1", "type": "Broker", "score": 75},
        ]
        result = self.classifier.classify(items, metric_name="test")
        
        apps = result.get_by_type("Application")
        self.assertEqual(len(apps), 1)
        self.assertEqual(apps[0].id, "app1")
    
    def test_top_n(self):
        """Test top_n() returns top items"""
        top_5 = self.result.top_n(5)
        self.assertEqual(len(top_5), 5)
        self.assertEqual(top_5[0].rank, 1)
        self.assertEqual(top_5[4].rank, 5)
    
    def test_critical_count_property(self):
        """Test critical_count property"""
        self.assertEqual(
            self.result.critical_count,
            len(self.result.get_critical())
        )
    
    def test_to_dict_serialization(self):
        """Test to_dict() returns complete serialization"""
        d = self.result.to_dict()
        
        self.assertIn("metric", d)
        self.assertIn("statistics", d)
        self.assertIn("summary", d)
        self.assertIn("items", d)
        self.assertIn("by_level", d)
        
        self.assertEqual(d["metric"], "test")
        self.assertEqual(len(d["items"]), 20)


# =============================================================================
# Test: ClassifiedItem
# =============================================================================

class TestClassifiedItem(unittest.TestCase):
    """Test ClassifiedItem data class"""
    
    def test_to_dict(self):
        """Test ClassifiedItem serialization"""
        item = ClassifiedItem(
            id="test_id",
            item_type="Application",
            score=0.75,
            level=CriticalityLevel.HIGH,
            percentile=85.0,
            rank=2,
            is_outlier=False,
            metadata={"extra": "data"}
        )
        
        d = item.to_dict()
        
        self.assertEqual(d["id"], "test_id")
        self.assertEqual(d["type"], "Application")
        self.assertEqual(d["level"], "high")
        self.assertEqual(d["percentile"], 85.0)
        self.assertEqual(d["rank"], 2)
        self.assertFalse(d["is_outlier"])
        self.assertEqual(d["metadata"]["extra"], "data")


# =============================================================================
# Test: Data Classes (Problem/AntiPattern)
# =============================================================================

class TestProblemDataClasses(unittest.TestCase):
    """Test problem-related data classes"""
    
    def test_symptom_to_dict(self):
        """Test Symptom serialization"""
        symptom = Symptom(
            name="High Betweenness",
            description="Component has high betweenness centrality",
            metric="betweenness",
            value=0.85,
            threshold=0.50
        )
        
        d = symptom.to_dict()
        
        self.assertEqual(d["name"], "High Betweenness")
        self.assertEqual(d["metric"], "betweenness")
        self.assertAlmostEqual(d["value"], 0.85, places=2)
        self.assertAlmostEqual(d["threshold"], 0.50, places=2)
    
    def test_problem_to_dict(self):
        """Test Problem serialization"""
        problem = Problem(
            problem_type=ProblemType.SINGLE_POINT_OF_FAILURE,
            severity=ProblemSeverity.CRITICAL,
            title="SPOF Detected",
            description="Component is a single point of failure",
            affected_components=["broker1"],
            symptoms=[],
            impact="System failure if component fails",
            recommendation="Add redundancy",
            quality_attributes=[QualityAttribute.RELIABILITY],
            metrics={"betweenness": 0.9}
        )
        
        d = problem.to_dict()
        
        self.assertEqual(d["type"], "spof")
        self.assertEqual(d["severity"], "critical")
        self.assertEqual(d["title"], "SPOF Detected")
        self.assertIn("broker1", d["affected_components"])
    
    def test_antipattern_to_dict(self):
        """Test AntiPattern serialization"""
        pattern = AntiPattern(
            pattern_type=AntiPatternType.GOD_TOPIC,
            severity=PatternSeverity.HIGH,
            affected_components=["topic1"],
            description="Topic has too many connections",
            impact="Single point of failure for messaging",
            recommendation="Split into multiple topics",
            quality_attributes=["reliability", "maintainability"],
            metrics={"connections": 50}
        )
        
        d = pattern.to_dict()
        
        self.assertEqual(d["type"], "god_topic")
        self.assertEqual(d["severity"], "high")
        self.assertIn("topic1", d["affected_components"])


# =============================================================================
# Test: Integration
# =============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests (without Neo4j connection)"""
    
    def test_full_classification_workflow(self):
        """Test complete classification workflow"""
        # Simulate realistic component data
        items = [
            {"id": "app_order_service", "type": "Application", "score": 0.95},
            {"id": "app_payment_service", "type": "Application", "score": 0.85},
            {"id": "app_notification_service", "type": "Application", "score": 0.45},
            {"id": "app_logging_service", "type": "Application", "score": 0.25},
            {"id": "app_cache_service", "type": "Application", "score": 0.15},
            {"id": "broker_main", "type": "Broker", "score": 0.90},
            {"id": "broker_backup", "type": "Broker", "score": 0.30},
            {"id": "topic_orders", "type": "Topic", "score": 0.80},
            {"id": "topic_payments", "type": "Topic", "score": 0.70},
            {"id": "topic_logs", "type": "Topic", "score": 0.20},
        ]
        
        classifier = BoxPlotClassifier(k_factor=1.5)
        
        # Overall classification
        overall_result = classifier.classify(items, metric_name="composite")
        
        # Verify structure
        self.assertEqual(len(overall_result.items), 10)
        self.assertIsNotNone(overall_result.stats)
        
        # Verify sorting (highest score first)
        self.assertEqual(overall_result.items[0].id, "app_order_service")
        
        # Type-specific classification
        by_type = classifier.classify_by_type(items, metric_name="composite")
        
        self.assertIn("Application", by_type)
        self.assertIn("Broker", by_type)
        self.assertIn("Topic", by_type)
        
        # Verify each type is analyzed separately
        self.assertEqual(len(by_type["Application"].items), 5)
        self.assertEqual(len(by_type["Broker"].items), 2)
        self.assertEqual(len(by_type["Topic"].items), 3)
    
    def test_classification_result_export(self):
        """Test that classification result can be exported as JSON"""
        import json
        
        items = [
            {"id": f"item{i}", "type": "Test", "score": i * 0.1}
            for i in range(1, 11)
        ]
        
        classifier = BoxPlotClassifier(k_factor=1.5)
        result = classifier.classify(items, metric_name="test")
        
        # Should be JSON serializable
        json_str = json.dumps(result.to_dict(), default=str)
        self.assertIsInstance(json_str, str)
        
        # Should be parseable back
        parsed = json.loads(json_str)
        self.assertEqual(parsed["metric"], "test")
        self.assertEqual(len(parsed["items"]), 10)


# =============================================================================
# Main
# =============================================================================

def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEnums))
    suite.addTests(loader.loadTestsFromTestCase(TestBoxPlotStats))
    suite.addTests(loader.loadTestsFromTestCase(TestClassification))
    suite.addTests(loader.loadTestsFromTestCase(TestClassificationResult))
    suite.addTests(loader.loadTestsFromTestCase(TestClassifiedItem))
    suite.addTests(loader.loadTestsFromTestCase(TestProblemDataClasses))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
