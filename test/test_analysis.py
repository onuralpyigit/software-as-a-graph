"""
Tests for src.analysis module
==============================

Tests box-plot classification, enums, data classes, and anti-pattern detection.
Note: GDS-based tests require Neo4j and are marked with @pytest.mark.neo4j
"""

import pytest
import math


class TestCriticalityLevel:
    """Tests for CriticalityLevel enum"""

    def test_level_values(self):
        """Test criticality level values"""
        from src.analysis import CriticalityLevel
        
        assert CriticalityLevel.CRITICAL.value == "critical"
        assert CriticalityLevel.HIGH.value == "high"
        assert CriticalityLevel.MEDIUM.value == "medium"
        assert CriticalityLevel.LOW.value == "low"
        assert CriticalityLevel.MINIMAL.value == "minimal"

    def test_level_numeric(self):
        """Test numeric values for sorting"""
        from src.analysis import CriticalityLevel
        
        assert CriticalityLevel.CRITICAL.numeric > CriticalityLevel.HIGH.numeric
        assert CriticalityLevel.HIGH.numeric > CriticalityLevel.MEDIUM.numeric
        assert CriticalityLevel.MEDIUM.numeric > CriticalityLevel.LOW.numeric
        assert CriticalityLevel.LOW.numeric > CriticalityLevel.MINIMAL.numeric

    def test_level_color(self):
        """Test ANSI color codes"""
        from src.analysis import CriticalityLevel
        
        # Each level should have a color code
        for level in CriticalityLevel:
            assert level.color.startswith("\033[")


class TestBoxPlotStats:
    """Tests for BoxPlotStats data class"""

    def test_create_stats(self):
        """Create BoxPlotStats instance"""
        from src.analysis import BoxPlotStats
        
        stats = BoxPlotStats(
            min_val=0.0,
            q1=0.25,
            median=0.5,
            q3=0.75,
            max_val=1.0,
            iqr=0.5,
            lower_fence=-0.5,
            upper_fence=1.5,
            mean=0.5,
            std_dev=0.2,
            count=100,
            k_factor=1.5,
        )
        
        assert stats.median == 0.5
        assert stats.iqr == 0.5
        assert stats.count == 100

    def test_stats_to_dict(self):
        """Convert stats to dictionary"""
        from src.analysis import BoxPlotStats
        
        stats = BoxPlotStats(
            min_val=0.0, q1=0.25, median=0.5, q3=0.75, max_val=1.0,
            iqr=0.5, lower_fence=-0.5, upper_fence=1.5,
            mean=0.5, std_dev=0.2, count=100, k_factor=1.5,
        )
        
        d = stats.to_dict()
        
        assert "median" in d
        assert "iqr" in d
        assert "count" in d
        assert d["median"] == 0.5


class TestClassifiedItem:
    """Tests for ClassifiedItem data class"""

    def test_create_item(self):
        """Create ClassifiedItem instance"""
        from src.analysis import ClassifiedItem, CriticalityLevel
        
        item = ClassifiedItem(
            id="A1",
            item_type="Application",
            score=0.85,
            level=CriticalityLevel.CRITICAL,
            percentile=95.0,
            rank=1,
            is_outlier=True,
            metadata={"name": "TestApp"},
        )
        
        assert item.id == "A1"
        assert item.level == CriticalityLevel.CRITICAL
        assert item.is_outlier is True

    def test_item_to_dict(self):
        """Convert item to dictionary"""
        from src.analysis import ClassifiedItem, CriticalityLevel
        
        item = ClassifiedItem(
            id="B1", item_type="Broker", score=0.7,
            level=CriticalityLevel.HIGH, percentile=80.0,
            rank=2, is_outlier=False,
        )
        
        d = item.to_dict()
        
        assert d["id"] == "B1"
        assert d["level"] == "high"
        assert d["is_outlier"] is False


class TestBoxPlotClassifier:
    """Tests for BoxPlotClassifier class"""

    def test_create_classifier(self):
        """Create classifier with default k-factor"""
        from src.analysis import BoxPlotClassifier
        
        classifier = BoxPlotClassifier()
        
        assert classifier.k_factor == 1.5

    def test_create_classifier_custom_k(self):
        """Create classifier with custom k-factor"""
        from src.analysis import BoxPlotClassifier
        
        classifier = BoxPlotClassifier(k_factor=3.0)
        
        assert classifier.k_factor == 3.0

    def test_calculate_stats_empty(self):
        """Calculate stats for empty list"""
        from src.analysis import BoxPlotClassifier
        
        classifier = BoxPlotClassifier()
        stats = classifier.calculate_stats([])
        
        assert stats.count == 0
        assert stats.median == 0

    def test_calculate_stats_simple(self):
        """Calculate stats for simple list"""
        from src.analysis import BoxPlotClassifier
        
        classifier = BoxPlotClassifier(k_factor=1.5)
        scores = list(range(1, 101))  # 1-100
        stats = classifier.calculate_stats(scores)
        
        assert stats.count == 100
        assert stats.min_val == 1
        assert stats.max_val == 100
        assert 25 <= stats.q1 <= 26
        assert 50 <= stats.median <= 51
        assert 75 <= stats.q3 <= 76

    def test_calculate_iqr(self):
        """Calculate IQR correctly"""
        from src.analysis import BoxPlotClassifier
        
        classifier = BoxPlotClassifier(k_factor=1.5)
        scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        stats = classifier.calculate_stats(scores)
        
        assert stats.iqr == stats.q3 - stats.q1

    def test_calculate_fences(self):
        """Calculate fences correctly"""
        from src.analysis import BoxPlotClassifier
        
        classifier = BoxPlotClassifier(k_factor=1.5)
        scores = list(range(1, 101))
        stats = classifier.calculate_stats(scores)
        
        expected_lower = stats.q1 - 1.5 * stats.iqr
        expected_upper = stats.q3 + 1.5 * stats.iqr
        
        assert abs(stats.lower_fence - expected_lower) < 0.01
        assert abs(stats.upper_fence - expected_upper) < 0.01

    def test_classify_score_critical(self):
        """Classify score as critical (outlier)"""
        from src.analysis import BoxPlotClassifier, BoxPlotStats, CriticalityLevel
        
        classifier = BoxPlotClassifier()
        stats = BoxPlotStats(
            min_val=0, q1=25, median=50, q3=75, max_val=100,
            iqr=50, lower_fence=-50, upper_fence=150,
            mean=50, std_dev=25, count=100, k_factor=1.5,
        )
        
        level, is_outlier = classifier.classify_score(200, stats)
        
        assert level == CriticalityLevel.CRITICAL
        assert is_outlier is True

    def test_classify_score_high(self):
        """Classify score as high"""
        from src.analysis import BoxPlotClassifier, BoxPlotStats, CriticalityLevel
        
        classifier = BoxPlotClassifier()
        stats = BoxPlotStats(
            min_val=0, q1=25, median=50, q3=75, max_val=100,
            iqr=50, lower_fence=-50, upper_fence=150,
            mean=50, std_dev=25, count=100, k_factor=1.5,
        )
        
        level, is_outlier = classifier.classify_score(80, stats)
        
        assert level == CriticalityLevel.HIGH
        assert is_outlier is False

    def test_classify_score_medium(self):
        """Classify score as medium"""
        from src.analysis import BoxPlotClassifier, BoxPlotStats, CriticalityLevel
        
        classifier = BoxPlotClassifier()
        stats = BoxPlotStats(
            min_val=0, q1=25, median=50, q3=75, max_val=100,
            iqr=50, lower_fence=-50, upper_fence=150,
            mean=50, std_dev=25, count=100, k_factor=1.5,
        )
        
        level, is_outlier = classifier.classify_score(60, stats)
        
        assert level == CriticalityLevel.MEDIUM
        assert is_outlier is False

    def test_classify_score_low(self):
        """Classify score as low"""
        from src.analysis import BoxPlotClassifier, BoxPlotStats, CriticalityLevel
        
        classifier = BoxPlotClassifier()
        stats = BoxPlotStats(
            min_val=0, q1=25, median=50, q3=75, max_val=100,
            iqr=50, lower_fence=-50, upper_fence=150,
            mean=50, std_dev=25, count=100, k_factor=1.5,
        )
        
        level, is_outlier = classifier.classify_score(30, stats)
        
        assert level == CriticalityLevel.LOW
        assert is_outlier is False

    def test_classify_score_minimal(self):
        """Classify score as minimal"""
        from src.analysis import BoxPlotClassifier, BoxPlotStats, CriticalityLevel
        
        classifier = BoxPlotClassifier()
        stats = BoxPlotStats(
            min_val=0, q1=25, median=50, q3=75, max_val=100,
            iqr=50, lower_fence=-50, upper_fence=150,
            mean=50, std_dev=25, count=100, k_factor=1.5,
        )
        
        level, is_outlier = classifier.classify_score(10, stats)
        
        assert level == CriticalityLevel.MINIMAL
        assert is_outlier is False

    def test_classify_items(self):
        """Classify a list of items"""
        from src.analysis import BoxPlotClassifier, CriticalityLevel
        
        classifier = BoxPlotClassifier()
        items = [
            {"id": "A1", "type": "App", "score": 0.95},
            {"id": "A2", "type": "App", "score": 0.80},
            {"id": "A3", "type": "App", "score": 0.60},
            {"id": "A4", "type": "App", "score": 0.40},
            {"id": "A5", "type": "App", "score": 0.20},
        ]
        
        result = classifier.classify(items, metric_name="test")
        
        assert result.metric_name == "test"
        assert len(result.items) == 5
        assert result.stats.count == 5

    def test_classify_items_sorted(self):
        """Classified items are sorted by score descending"""
        from src.analysis import BoxPlotClassifier
        
        classifier = BoxPlotClassifier()
        items = [
            {"id": "A1", "type": "App", "score": 0.3},
            {"id": "A2", "type": "App", "score": 0.9},
            {"id": "A3", "type": "App", "score": 0.5},
        ]
        
        result = classifier.classify(items)
        
        # First item should have highest score
        assert result.items[0].score >= result.items[1].score
        assert result.items[1].score >= result.items[2].score

    def test_classify_items_by_level(self):
        """Items are grouped by level"""
        from src.analysis import BoxPlotClassifier, CriticalityLevel
        
        classifier = BoxPlotClassifier()
        items = [
            {"id": f"A{i}", "type": "App", "score": i * 0.1}
            for i in range(1, 11)
        ]
        
        result = classifier.classify(items)
        
        # All levels should be represented
        total_by_level = sum(len(items) for items in result.by_level.values())
        assert total_by_level == 10

    def test_classify_result_to_dict(self):
        """Convert classification result to dictionary"""
        from src.analysis import BoxPlotClassifier
        
        classifier = BoxPlotClassifier()
        items = [
            {"id": "A1", "type": "App", "score": 0.9},
            {"id": "A2", "type": "App", "score": 0.5},
        ]
        
        result = classifier.classify(items, metric_name="betweenness")
        d = result.to_dict()
        
        assert d["metric"] == "betweenness"
        assert "statistics" in d
        assert "items" in d
        assert "summary" in d


class TestAntiPatternEnums:
    """Tests for anti-pattern enums"""

    def test_antipattern_types(self):
        """Test AntiPatternType enum"""
        from src.analysis import AntiPatternType
        
        assert AntiPatternType.GOD_TOPIC.value == "god_topic"
        assert AntiPatternType.SINGLE_POINT_OF_FAILURE.value == "spof"
        assert AntiPatternType.BOTTLENECK_BROKER.value == "bottleneck_broker"

    def test_pattern_severity(self):
        """Test PatternSeverity enum"""
        from src.analysis import PatternSeverity
        
        assert PatternSeverity.CRITICAL.value == "critical"
        assert PatternSeverity.HIGH.value == "high"
        assert PatternSeverity.MEDIUM.value == "medium"
        assert PatternSeverity.LOW.value == "low"

    def test_severity_color(self):
        """Test severity colors"""
        from src.analysis import PatternSeverity
        
        for severity in PatternSeverity:
            assert severity.color.startswith("\033[")


class TestAntiPatternDataClasses:
    """Tests for anti-pattern data classes"""

    def test_create_antipattern(self):
        """Create AntiPattern instance"""
        from src.analysis import AntiPattern, AntiPatternType, PatternSeverity
        
        pattern = AntiPattern(
            pattern_type=AntiPatternType.GOD_TOPIC,
            severity=PatternSeverity.HIGH,
            affected_components=["T1", "T2"],
            description="Topic handles too many connections",
            impact="Single point of failure risk",
            recommendation="Split into multiple topics",
            quality_attributes=["reliability", "maintainability"],
            metrics={"connections": 50},
        )
        
        assert pattern.pattern_type == AntiPatternType.GOD_TOPIC
        assert pattern.severity == PatternSeverity.HIGH
        assert len(pattern.affected_components) == 2

    def test_antipattern_to_dict(self):
        """Convert anti-pattern to dictionary"""
        from src.analysis import AntiPattern, AntiPatternType, PatternSeverity
        
        pattern = AntiPattern(
            pattern_type=AntiPatternType.BOTTLENECK_BROKER,
            severity=PatternSeverity.CRITICAL,
            affected_components=["B1"],
            description="Broker overloaded",
            impact="Performance degradation",
            recommendation="Add more brokers",
            quality_attributes=["availability"],
        )
        
        d = pattern.to_dict()
        
        assert d["type"] == "bottleneck_broker"
        assert d["severity"] == "critical"
        assert "B1" in d["affected_components"]


class TestQualityAttributeEnums:
    """Tests for quality attribute enums"""

    def test_quality_attributes(self):
        """Test QualityAttribute enum"""
        from src.analysis import QualityAttribute
        
        assert QualityAttribute.RELIABILITY.value == "reliability"
        assert QualityAttribute.MAINTAINABILITY.value == "maintainability"
        assert QualityAttribute.AVAILABILITY.value == "availability"

    def test_severity_enum(self):
        """Test Severity enum"""
        from src.analysis import Severity
        
        assert Severity.CRITICAL.value == "critical"
        assert Severity.INFO.value == "info"

    def test_severity_weight(self):
        """Test severity weights"""
        from src.analysis import Severity
        
        assert Severity.CRITICAL.weight > Severity.HIGH.weight
        assert Severity.HIGH.weight > Severity.MEDIUM.weight
        assert Severity.MEDIUM.weight > Severity.LOW.weight
        assert Severity.LOW.weight > Severity.INFO.weight


class TestFindingDataClass:
    """Tests for Finding data class"""

    def test_create_finding(self):
        """Create Finding instance"""
        from src.analysis import Finding, Severity
        
        finding = Finding(
            severity=Severity.HIGH,
            category="spof",
            component_id="B1",
            component_type="Broker",
            description="Single broker handling all traffic",
            impact="System unavailable if broker fails",
            recommendation="Add broker redundancy",
            metrics={"connections": 100},
        )
        
        assert finding.severity == Severity.HIGH
        assert finding.component_id == "B1"

    def test_finding_to_dict(self):
        """Convert finding to dictionary"""
        from src.analysis import Finding, Severity
        
        finding = Finding(
            severity=Severity.MEDIUM,
            category="coupling",
            component_id="A1",
            component_type="Application",
            description="High coupling",
            impact="Maintenance difficulty",
            recommendation="Reduce dependencies",
        )
        
        d = finding.to_dict()
        
        assert d["severity"] == "medium"
        assert d["component_id"] == "A1"


class TestMergeClassifications:
    """Tests for merge_classifications utility"""

    def test_merge_empty(self):
        """Merge empty list"""
        from src.analysis import merge_classifications
        
        result = merge_classifications([])
        
        assert result is None or len(result.items) == 0

    def test_merge_single(self):
        """Merge single classification"""
        from src.analysis import BoxPlotClassifier, merge_classifications
        
        classifier = BoxPlotClassifier()
        items = [{"id": "A1", "type": "App", "score": 0.5}]
        classification = classifier.classify(items, metric_name="test")
        
        result = merge_classifications([classification])
        
        assert len(result.items) == 1

    def test_merge_multiple(self):
        """Merge multiple classifications"""
        from src.analysis import BoxPlotClassifier, merge_classifications
        
        classifier = BoxPlotClassifier()
        
        items1 = [
            {"id": "A1", "type": "App", "score": 0.8},
            {"id": "A2", "type": "App", "score": 0.4},
        ]
        items2 = [
            {"id": "A1", "type": "App", "score": 0.6},
            {"id": "A2", "type": "App", "score": 0.9},
        ]
        
        class1 = classifier.classify(items1, metric_name="metric1")
        class2 = classifier.classify(items2, metric_name="metric2")
        
        result = merge_classifications([class1, class2])
        
        # Should have both items
        assert len(result.items) == 2


class TestClassificationResult:
    """Tests for ClassificationResult methods"""

    def test_top_critical(self):
        """Get top critical items"""
        from src.analysis import BoxPlotClassifier
        
        classifier = BoxPlotClassifier()
        items = [
            {"id": f"A{i}", "type": "App", "score": i * 0.1}
            for i in range(1, 11)
        ]
        
        result = classifier.classify(items)
        critical = result.top_critical()
        
        # Should return critical and high items
        assert isinstance(critical, list)

    def test_get_by_type(self):
        """Get items by component type"""
        from src.analysis import BoxPlotClassifier
        
        classifier = BoxPlotClassifier()
        items = [
            {"id": "A1", "type": "Application", "score": 0.9},
            {"id": "B1", "type": "Broker", "score": 0.8},
            {"id": "A2", "type": "Application", "score": 0.7},
        ]
        
        result = classifier.classify(items)
        apps = result.get_by_type("Application")
        
        assert len(apps) == 2
        assert all(item.item_type == "Application" for item in apps)
