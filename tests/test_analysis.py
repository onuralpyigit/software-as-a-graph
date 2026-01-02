#!/usr/bin/env python3
"""
Test Suite for src.analysis Module - Version 5.0

Tests for graph analysis including:
- Box-plot classification
- Centrality algorithms (mock/demo mode)
- Layer analysis
- Component type analysis
- Edge analysis

Run with pytest:
    pytest tests/test_analysis.py -v

Or standalone:
    python tests/test_analysis.py

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


# =============================================================================
# Test Data
# =============================================================================

def create_test_scores() -> List[Dict[str, Any]]:
    """Create test score data for classification tests."""
    return [
        {"id": "comp_01", "type": "Application", "score": 0.95},  # Outlier
        {"id": "comp_02", "type": "Application", "score": 0.88},
        {"id": "comp_03", "type": "Application", "score": 0.75},
        {"id": "comp_04", "type": "Broker", "score": 0.72},
        {"id": "comp_05", "type": "Application", "score": 0.65},
        {"id": "comp_06", "type": "Broker", "score": 0.55},
        {"id": "comp_07", "type": "Node", "score": 0.45},
        {"id": "comp_08", "type": "Node", "score": 0.35},
        {"id": "comp_09", "type": "Application", "score": 0.25},
        {"id": "comp_10", "type": "Node", "score": 0.15},
        {"id": "comp_11", "type": "Application", "score": 0.08},
    ]


def create_edge_test_data() -> List[Dict[str, Any]]:
    """Create test edge data."""
    return [
        {"source": "app_1", "target": "app_2", "weight": 5.5, "type": "app_to_app"},
        {"source": "app_2", "target": "app_3", "weight": 3.2, "type": "app_to_app"},
        {"source": "app_1", "target": "broker_1", "weight": 4.0, "type": "app_to_broker"},
        {"source": "node_1", "target": "node_2", "weight": 2.5, "type": "node_to_node"},
        {"source": "node_1", "target": "broker_1", "weight": 1.8, "type": "node_to_broker"},
    ]


# =============================================================================
# Test: CriticalityLevel Enum
# =============================================================================

class TestCriticalityLevel:
    """Tests for CriticalityLevel enum."""

    def test_level_values(self):
        """Test enum values."""
        from src.analysis import CriticalityLevel
        
        assert CriticalityLevel.CRITICAL.value == "critical"
        assert CriticalityLevel.HIGH.value == "high"
        assert CriticalityLevel.MEDIUM.value == "medium"
        assert CriticalityLevel.LOW.value == "low"
        assert CriticalityLevel.MINIMAL.value == "minimal"

    def test_level_ordering(self):
        """Test level comparison."""
        from src.analysis import CriticalityLevel
        
        assert CriticalityLevel.CRITICAL > CriticalityLevel.HIGH
        assert CriticalityLevel.HIGH > CriticalityLevel.MEDIUM
        assert CriticalityLevel.MEDIUM > CriticalityLevel.LOW
        assert CriticalityLevel.LOW > CriticalityLevel.MINIMAL
        
        assert CriticalityLevel.CRITICAL >= CriticalityLevel.HIGH
        assert CriticalityLevel.HIGH >= CriticalityLevel.HIGH
        
        assert CriticalityLevel.MINIMAL < CriticalityLevel.LOW
        assert CriticalityLevel.LOW <= CriticalityLevel.MEDIUM


# =============================================================================
# Test: BoxPlotStats
# =============================================================================

class TestBoxPlotStats:
    """Tests for BoxPlotStats data class."""

    def test_to_dict(self):
        """Test serialization."""
        from src.analysis import BoxPlotStats
        
        stats = BoxPlotStats(
            q1=0.25,
            median=0.50,
            q3=0.75,
            iqr=0.50,
            lower_fence=-0.50,
            upper_fence=1.50,
            min_val=0.10,
            max_val=0.95,
            mean=0.52,
            count=10,
            k_factor=1.5,
        )
        
        data = stats.to_dict()
        
        assert data["q1"] == 0.25
        assert data["median"] == 0.50
        assert data["q3"] == 0.75
        assert data["iqr"] == 0.50
        assert data["count"] == 10


# =============================================================================
# Test: BoxPlotClassifier
# =============================================================================

class TestBoxPlotClassifier:
    """Tests for BoxPlotClassifier."""

    def test_basic_classification(self):
        """Test basic classification."""
        from src.analysis import BoxPlotClassifier, CriticalityLevel
        
        classifier = BoxPlotClassifier(k_factor=1.5)
        items = create_test_scores()
        
        result = classifier.classify(items, metric_name="test_score")
        
        assert result.metric_name == "test_score"
        assert len(result.items) == len(items)
        assert result.stats.count == len(items)

    def test_classification_levels(self):
        """Test that items are classified into different levels."""
        from src.analysis import BoxPlotClassifier, CriticalityLevel
        
        classifier = BoxPlotClassifier(k_factor=1.5)
        items = create_test_scores()
        
        result = classifier.classify(items)
        
        # Check distribution
        total = sum(len(items) for items in result.by_level.values())
        assert total == len(items)
        
        # Highest score should be HIGH or CRITICAL
        assert result.items[0].level >= CriticalityLevel.HIGH
        
        # Lowest score should be LOW or MINIMAL
        assert result.items[-1].level <= CriticalityLevel.LOW

    def test_get_critical(self):
        """Test getting critical items."""
        from src.analysis import BoxPlotClassifier
        
        classifier = BoxPlotClassifier(k_factor=1.5)
        items = create_test_scores()
        
        result = classifier.classify(items)
        critical = result.get_critical()
        
        # All critical items should have high scores
        for item in critical:
            assert item.score > result.stats.upper_fence

    def test_get_high_and_above(self):
        """Test getting high and critical items."""
        from src.analysis import BoxPlotClassifier, CriticalityLevel
        
        classifier = BoxPlotClassifier(k_factor=1.5)
        items = create_test_scores()
        
        result = classifier.classify(items)
        high_plus = result.get_high_and_above()
        
        for item in high_plus:
            assert item.level >= CriticalityLevel.HIGH

    def test_empty_input(self):
        """Test classification with empty input."""
        from src.analysis import BoxPlotClassifier
        
        classifier = BoxPlotClassifier()
        result = classifier.classify([])
        
        assert len(result.items) == 0
        assert result.stats.count == 0

    def test_single_item(self):
        """Test classification with single item."""
        from src.analysis import BoxPlotClassifier
        
        classifier = BoxPlotClassifier()
        items = [{"id": "single", "type": "test", "score": 0.5}]
        
        result = classifier.classify(items)
        
        assert len(result.items) == 1

    def test_k_factor_effect(self):
        """Test that k_factor affects outlier detection."""
        from src.analysis import BoxPlotClassifier
        
        items = create_test_scores()
        
        # Lower k = more outliers
        classifier_strict = BoxPlotClassifier(k_factor=1.0)
        result_strict = classifier_strict.classify(items)
        
        # Higher k = fewer outliers
        classifier_lenient = BoxPlotClassifier(k_factor=3.0)
        result_lenient = classifier_lenient.classify(items)
        
        strict_outliers = len(result_strict.get_outliers())
        lenient_outliers = len(result_lenient.get_outliers())
        
        assert strict_outliers >= lenient_outliers

    def test_classify_scores_dict(self):
        """Test classifying a dictionary of scores."""
        from src.analysis import BoxPlotClassifier
        
        classifier = BoxPlotClassifier()
        scores = {
            "comp_1": 0.9,
            "comp_2": 0.5,
            "comp_3": 0.2,
        }
        
        result = classifier.classify_scores(scores, item_type="Application")
        
        assert len(result.items) == 3
        for item in result.items:
            assert item.item_type == "Application"

    def test_statistics_calculation(self):
        """Test that statistics are calculated correctly."""
        from src.analysis import BoxPlotClassifier
        
        classifier = BoxPlotClassifier(k_factor=1.5)
        
        # Simple data for easy verification
        items = [
            {"id": f"item_{i}", "type": "test", "score": score}
            for i, score in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        ]
        
        result = classifier.classify(items)
        
        assert result.stats.min_val == 0.1
        assert result.stats.max_val == 0.9
        assert 0.4 < result.stats.median < 0.6  # Should be around 0.5


# =============================================================================
# Test: ClassificationResult
# =============================================================================

class TestClassificationResult:
    """Tests for ClassificationResult."""

    def test_to_dict(self):
        """Test serialization."""
        from src.analysis import BoxPlotClassifier
        
        classifier = BoxPlotClassifier()
        items = create_test_scores()
        
        result = classifier.classify(items)
        data = result.to_dict()
        
        assert "metric" in data
        assert "stats" in data
        assert "count" in data
        assert "by_level" in data
        assert "items" in data

    def test_summary(self):
        """Test summary method."""
        from src.analysis import BoxPlotClassifier
        
        classifier = BoxPlotClassifier()
        items = create_test_scores()
        
        result = classifier.classify(items)
        summary = result.summary()
        
        assert "critical" in summary
        assert "high" in summary
        assert sum(summary.values()) == len(items)


# =============================================================================
# Test: Utility Functions
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_classify_items(self):
        """Test classify_items convenience function."""
        from src.analysis import classify_items, CriticalityLevel
        
        items = create_test_scores()
        result = classify_items(items, k_factor=1.5, metric_name="test")
        
        assert result.metric_name == "test"
        assert len(result.items) == len(items)

    def test_get_level_for_score(self):
        """Test get_level_for_score function."""
        from src.analysis import get_level_for_score, CriticalityLevel
        
        all_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # Very high score should be HIGH or CRITICAL
        level_high = get_level_for_score(0.95, all_scores)
        assert level_high >= CriticalityLevel.HIGH
        
        # Very low score should be LOW or MINIMAL
        level_low = get_level_for_score(0.05, all_scores)
        assert level_low <= CriticalityLevel.LOW


# =============================================================================
# Test: Constants
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_component_types(self):
        """Test COMPONENT_TYPES constant."""
        from src.analysis import COMPONENT_TYPES
        
        assert "Application" in COMPONENT_TYPES
        assert "Broker" in COMPONENT_TYPES
        assert "Node" in COMPONENT_TYPES
        assert "Topic" in COMPONENT_TYPES

    def test_dependency_types(self):
        """Test DEPENDENCY_TYPES constant."""
        from src.analysis import DEPENDENCY_TYPES
        
        assert "app_to_app" in DEPENDENCY_TYPES
        assert "node_to_node" in DEPENDENCY_TYPES
        assert "app_to_broker" in DEPENDENCY_TYPES
        assert "node_to_broker" in DEPENDENCY_TYPES

    def test_layer_definitions(self):
        """Test LAYER_DEFINITIONS constant."""
        from src.analysis import LAYER_DEFINITIONS
        
        assert "application" in LAYER_DEFINITIONS
        assert "infrastructure" in LAYER_DEFINITIONS
        assert "app_broker" in LAYER_DEFINITIONS
        assert "node_broker" in LAYER_DEFINITIONS
        assert "full" in LAYER_DEFINITIONS
        
        # Check structure
        for key, definition in LAYER_DEFINITIONS.items():
            assert "name" in definition
            assert "component_types" in definition
            assert "dependency_types" in definition


# =============================================================================
# Test: Data Classes
# =============================================================================

class TestDataClasses:
    """Tests for data classes."""

    def test_centrality_result(self):
        """Test CentralityResult data class."""
        from src.analysis import CentralityResult
        
        result = CentralityResult(
            node_id="app_1",
            node_type="Application",
            score=0.85,
            rank=1,
        )
        
        assert result.node_id == "app_1"
        assert result.node_type == "Application"
        assert result.score == 0.85
        assert result.rank == 1
        
        data = result.to_dict()
        assert data["id"] == "app_1"
        assert data["type"] == "Application"

    def test_projection_info(self):
        """Test ProjectionInfo data class."""
        from src.analysis import ProjectionInfo
        
        info = ProjectionInfo(
            name="test_projection",
            node_count=100,
            relationship_count=500,
            node_labels=["Application", "Broker"],
            relationship_types=["DEPENDS_ON"],
        )
        
        assert info.name == "test_projection"
        assert info.node_count == 100
        
        data = info.to_dict()
        assert data["node_count"] == 100

    def test_layer_metrics(self):
        """Test LayerMetrics data class."""
        from src.analysis import LayerMetrics, CriticalityLevel
        
        metrics = LayerMetrics(
            component_id="app_1",
            component_type="Application",
            pagerank=0.5,
            betweenness=0.3,
            degree=0.7,
            composite_score=0.5,
            level=CriticalityLevel.HIGH,
            is_articulation_point=True,
        )
        
        assert metrics.component_id == "app_1"
        assert metrics.is_articulation_point is True
        
        data = metrics.to_dict()
        assert data["level"] == "high"

    def test_edge_metrics(self):
        """Test EdgeMetrics data class."""
        from src.analysis import EdgeMetrics, CriticalityLevel
        
        edge = EdgeMetrics(
            source_id="app_1",
            target_id="app_2",
            source_type="Application",
            target_type="Application",
            dependency_type="app_to_app",
            weight=3.5,
            criticality_score=0.8,
            is_bridge=True,
            connects_critical=True,
            level=CriticalityLevel.CRITICAL,
        )
        
        assert edge.edge_key == "app_1->app_2"
        assert edge.is_bridge is True
        
        data = edge.to_dict()
        assert data["dependency_type"] == "app_to_app"


# =============================================================================
# Test: Result Classes
# =============================================================================

class TestResultClasses:
    """Tests for result data classes."""

    def test_layer_result(self):
        """Test LayerResult data class."""
        from src.analysis import LayerResult, LayerMetrics, ProjectionInfo, CriticalityLevel
        
        metrics = [
            LayerMetrics("app_1", "Application", composite_score=0.9, level=CriticalityLevel.CRITICAL),
            LayerMetrics("app_2", "Application", composite_score=0.5, level=CriticalityLevel.MEDIUM),
            LayerMetrics("app_3", "Application", composite_score=0.2, level=CriticalityLevel.LOW),
        ]
        
        result = LayerResult(
            layer_name="Application Layer",
            layer_key="application",
            timestamp="2025-01-01T00:00:00",
            projection=ProjectionInfo("test", 10, 20),
            metrics=metrics,
        )
        
        critical = result.get_critical_components()
        assert len(critical) == 1
        assert critical[0].component_id == "app_1"
        
        high_plus = result.get_high_and_above()
        assert len(high_plus) == 1
        
        top = result.top_n(2)
        assert len(top) == 2
        assert top[0].composite_score > top[1].composite_score

    def test_edge_analysis_result(self):
        """Test EdgeAnalysisResult data class."""
        from src.analysis import EdgeAnalysisResult, EdgeMetrics, CriticalityLevel
        
        edges = [
            EdgeMetrics("a", "b", "App", "App", "app_to_app", weight=5.0, 
                       is_bridge=True, level=CriticalityLevel.CRITICAL),
            EdgeMetrics("c", "d", "Node", "Node", "node_to_node", weight=2.0,
                       level=CriticalityLevel.MEDIUM),
        ]
        
        result = EdgeAnalysisResult(
            timestamp="2025-01-01T00:00:00",
            edges=edges,
        )
        
        critical = result.get_critical()
        assert len(critical) == 1
        
        bridges = result.get_bridges()
        assert len(bridges) == 1


# =============================================================================
# Standalone Test Runner
# =============================================================================

def run_tests_standalone():
    """Run tests without pytest."""
    import traceback
    
    test_classes = [
        TestCriticalityLevel,
        TestBoxPlotStats,
        TestBoxPlotClassifier,
        TestClassificationResult,
        TestUtilityFunctions,
        TestConstants,
        TestDataClasses,
        TestResultClasses,
    ]
    
    passed = failed = 0
    
    print("=" * 60)
    print("Software-as-a-Graph Analysis Module Tests")
    print("=" * 60)
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)
        
        instance = test_class()
        for method_name in [m for m in dir(instance) if m.startswith("test_")]:
            try:
                getattr(instance, method_name)()
                print(f"  ✓ {method_name}")
                passed += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
                if "--verbose" in sys.argv:
                    traceback.print_exc()
                failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    if HAS_PYTEST and "--no-pytest" not in sys.argv:
        sys.exit(pytest.main([__file__, "-v"]))
    else:
        success = run_tests_standalone()
        sys.exit(0 if success else 1)
