#!/usr/bin/env python3
"""
Test Suite for src.analysis Module - Version 6.0

Tests for graph analysis including:
- Box-plot classification
- NetworkX-based centrality algorithms
- Component type analysis
- Layer analysis
- Edge analysis

Run with pytest:
    pytest tests/test_analysis.py -v

Or standalone:
    python tests/test_analysis.py

Author: Software-as-a-Graph Research Project
Version: 6.0
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


# =============================================================================
# Test Data Fixtures
# =============================================================================

def create_test_scores() -> List[Dict[str, Any]]:
    """Create test score data for classification tests."""
    return [
        {"id": "comp_01", "type": "Application", "score": 0.95},  # Likely outlier
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


def create_mock_graph_data():
    """Create mock GraphData for testing NetworkX analyzer."""
    from src.analysis import ComponentData, EdgeData, GraphData
    
    components = [
        ComponentData(id="app_1", component_type="Application", weight=1.0),
        ComponentData(id="app_2", component_type="Application", weight=1.5),
        ComponentData(id="app_3", component_type="Application", weight=0.8),
        ComponentData(id="broker_1", component_type="Broker", weight=2.0),
        ComponentData(id="broker_2", component_type="Broker", weight=1.0),
        ComponentData(id="node_1", component_type="Node", weight=1.0),
        ComponentData(id="node_2", component_type="Node", weight=1.0),
    ]
    
    edges = [
        EdgeData("app_1", "app_2", "Application", "Application", "app_to_app", 1.5),
        EdgeData("app_2", "app_3", "Application", "Application", "app_to_app", 1.0),
        EdgeData("app_1", "app_3", "Application", "Application", "app_to_app", 0.5),
        EdgeData("app_1", "broker_1", "Application", "Broker", "app_to_broker", 2.0),
        EdgeData("app_2", "broker_1", "Application", "Broker", "app_to_broker", 1.5),
        EdgeData("app_3", "broker_2", "Application", "Broker", "app_to_broker", 1.0),
        EdgeData("node_1", "node_2", "Node", "Node", "node_to_node", 1.0),
        EdgeData("node_1", "broker_1", "Node", "Broker", "node_to_broker", 1.0),
        EdgeData("node_2", "broker_2", "Node", "Broker", "node_to_broker", 1.0),
    ]
    
    return GraphData(components=components, edges=edges)


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
    
    def test_numeric_property(self):
        """Test numeric property for comparison."""
        from src.analysis import CriticalityLevel
        
        assert CriticalityLevel.CRITICAL.numeric == 5
        assert CriticalityLevel.HIGH.numeric == 4
        assert CriticalityLevel.MEDIUM.numeric == 3
        assert CriticalityLevel.LOW.numeric == 2
        assert CriticalityLevel.MINIMAL.numeric == 1


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
            std_dev=0.25,
            count=10,
            k_factor=1.5,
        )
        
        data = stats.to_dict()
        
        assert data["q1"] == 0.25
        assert data["median"] == 0.50
        assert data["q3"] == 0.75
        assert data["iqr"] == 0.50
        assert data["count"] == 10
        assert data["k_factor"] == 1.5


# =============================================================================
# Test: BoxPlotClassifier
# =============================================================================

class TestBoxPlotClassifier:
    """Tests for BoxPlotClassifier."""

    def test_basic_classification(self):
        """Test basic classification."""
        from src.analysis import BoxPlotClassifier
        
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
        from src.analysis import BoxPlotClassifier, CriticalityLevel
        
        classifier = BoxPlotClassifier(k_factor=1.5)
        items = create_test_scores()
        
        result = classifier.classify(items)
        critical = result.get_critical()
        
        # All critical items should have CRITICAL level
        for item in critical:
            assert item.level == CriticalityLevel.CRITICAL

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
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        stats = classifier.calculate_stats(scores)
        
        assert stats.min_val == 0.1
        assert stats.max_val == 0.9
        assert 0.4 < stats.median < 0.6  # Should be around 0.5
        assert stats.count == 9

    def test_top_n(self):
        """Test getting top N items."""
        from src.analysis import BoxPlotClassifier
        
        classifier = BoxPlotClassifier()
        items = create_test_scores()
        
        result = classifier.classify(items)
        top_5 = result.top_n(5)
        
        assert len(top_5) == 5
        # Should be sorted descending
        for i in range(len(top_5) - 1):
            assert top_5[i].score >= top_5[i + 1].score


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
        from src.analysis import classify_items
        
        items = create_test_scores()
        result = classify_items(items, k_factor=1.5, metric_name="test")
        
        assert result.metric_name == "test"
        assert len(result.items) == len(items)


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
        assert len(COMPONENT_TYPES) == 4

    def test_dependency_types(self):
        """Test DEPENDENCY_TYPES constant."""
        from src.analysis import DEPENDENCY_TYPES
        
        assert "app_to_app" in DEPENDENCY_TYPES
        assert "node_to_node" in DEPENDENCY_TYPES
        assert "app_to_broker" in DEPENDENCY_TYPES
        assert "node_to_broker" in DEPENDENCY_TYPES
        assert len(DEPENDENCY_TYPES) == 4

    def test_layer_definitions(self):
        """Test LAYER_DEFINITIONS constant."""
        from src.analysis import LAYER_DEFINITIONS
        
        assert "application" in LAYER_DEFINITIONS
        assert "infrastructure" in LAYER_DEFINITIONS
        assert "app_broker" in LAYER_DEFINITIONS
        assert "node_broker" in LAYER_DEFINITIONS
        
        # Check structure
        app_layer = LAYER_DEFINITIONS["application"]
        assert "name" in app_layer
        assert "component_types" in app_layer
        assert "dependency_types" in app_layer


# =============================================================================
# Test: NetworkX Analyzer
# =============================================================================

class TestNetworkXAnalyzer:
    """Tests for NetworkXAnalyzer."""

    def test_analyze_component_type(self):
        """Test component type analysis."""
        from src.analysis import NetworkXAnalyzer
        
        graph_data = create_mock_graph_data()
        analyzer = NetworkXAnalyzer(k_factor=1.5)
        
        result = analyzer.analyze_component_type(graph_data, "Application")
        
        assert result.component_type == "Application"
        assert len(result.components) == 3  # app_1, app_2, app_3
        assert result.timestamp is not None
    
    def test_analyze_component_type_invalid(self):
        """Test invalid component type."""
        from src.analysis import NetworkXAnalyzer
        
        graph_data = create_mock_graph_data()
        analyzer = NetworkXAnalyzer()
        
        try:
            analyzer.analyze_component_type(graph_data, "Invalid")
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_analyze_layer(self):
        """Test layer analysis."""
        from src.analysis import NetworkXAnalyzer
        
        graph_data = create_mock_graph_data()
        analyzer = NetworkXAnalyzer(k_factor=1.5)
        
        result = analyzer.analyze_layer(graph_data, "application")
        
        assert result.layer_key == "application"
        assert result.layer_name == "Application Layer"
        assert len(result.components) > 0
    
    def test_analyze_layer_invalid(self):
        """Test invalid layer."""
        from src.analysis import NetworkXAnalyzer
        
        graph_data = create_mock_graph_data()
        analyzer = NetworkXAnalyzer()
        
        try:
            analyzer.analyze_layer(graph_data, "invalid_layer")
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_analyze_edges(self):
        """Test edge analysis."""
        from src.analysis import NetworkXAnalyzer
        
        graph_data = create_mock_graph_data()
        analyzer = NetworkXAnalyzer(k_factor=1.5)
        
        result = analyzer.analyze_edges(graph_data)
        
        assert len(result.edges) == len(graph_data.edges)
        assert result.timestamp is not None

    def test_metrics_calculation(self):
        """Test that metrics are calculated for components."""
        from src.analysis import NetworkXAnalyzer
        
        graph_data = create_mock_graph_data()
        analyzer = NetworkXAnalyzer()
        
        result = analyzer.analyze_component_type(graph_data, "Application")
        
        for comp in result.components:
            assert hasattr(comp, "pagerank")
            assert hasattr(comp, "betweenness")
            assert hasattr(comp, "degree")
            assert hasattr(comp, "composite_score")
            assert comp.pagerank >= 0
            assert comp.composite_score >= 0

    def test_normalization(self):
        """Test metric normalization."""
        from src.analysis import NetworkXAnalyzer
        
        graph_data = create_mock_graph_data()
        analyzer = NetworkXAnalyzer()
        
        result = analyzer.analyze_component_type(graph_data, "Application")
        
        for comp in result.components:
            # Normalized values should be in [0, 1]
            assert 0 <= comp.pagerank_norm <= 1
            assert 0 <= comp.betweenness_norm <= 1
            assert 0 <= comp.degree_norm <= 1

    def test_articulation_points(self):
        """Test articulation point detection."""
        from src.analysis import NetworkXAnalyzer
        
        graph_data = create_mock_graph_data()
        analyzer = NetworkXAnalyzer()
        
        result = analyzer.analyze_component_type(graph_data, "Application")
        
        # Check that articulation points are detected
        assert result.articulation_points is not None
        
        # Check that is_articulation_point flag is set
        for comp in result.components:
            if comp.component_id in result.articulation_points:
                assert comp.is_articulation_point


# =============================================================================
# Test: GraphData
# =============================================================================

class TestGraphData:
    """Tests for GraphData data class."""

    def test_summary(self):
        """Test summary method."""
        graph_data = create_mock_graph_data()
        summary = graph_data.summary()
        
        assert "total_components" in summary
        assert "total_edges" in summary
        assert summary["total_components"] == 7
        assert summary["total_edges"] == 9

    def test_get_components_by_type(self):
        """Test getting components by type."""
        graph_data = create_mock_graph_data()
        
        apps = graph_data.get_components_by_type("Application")
        assert len(apps) == 3
        
        brokers = graph_data.get_components_by_type("Broker")
        assert len(brokers) == 2

    def test_get_edges_by_type(self):
        """Test getting edges by type."""
        graph_data = create_mock_graph_data()
        
        app_edges = graph_data.get_edges_by_type("app_to_app")
        assert len(app_edges) == 3


# =============================================================================
# Test: CentralityMetrics
# =============================================================================

class TestCentralityMetrics:
    """Tests for CentralityMetrics data class."""

    def test_to_dict(self):
        """Test serialization."""
        from src.analysis import CentralityMetrics, CriticalityLevel
        
        metrics = CentralityMetrics(
            component_id="app_1",
            component_type="Application",
            pagerank=0.5,
            betweenness=0.3,
            degree=0.4,
            composite_score=0.4,
            level=CriticalityLevel.HIGH,
            is_articulation_point=True,
        )
        
        data = metrics.to_dict()
        
        assert data["id"] == "app_1"
        assert data["type"] == "Application"
        assert data["level"] == "high"
        assert data["is_articulation_point"] == True


# =============================================================================
# Test Runner (Standalone)
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
        TestNetworkXAnalyzer,
        TestGraphData,
        TestCentralityMetrics,
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for test_class in test_classes:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    print(f"  Running {test_class.__name__}.{method_name}...", end=" ")
                    getattr(instance, method_name)()
                    print("✓")
                    passed += 1
                except AssertionError as e:
                    print("✗")
                    failed += 1
                    errors.append(f"{test_class.__name__}.{method_name}: {e}")
                except Exception as e:
                    print("✗ (error)")
                    failed += 1
                    errors.append(f"{test_class.__name__}.{method_name}: {traceback.format_exc()}")
    
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")
    
    if errors:
        print(f"\nFailures:")
        for error in errors:
            print(f"  - {error}")
    
    return failed == 0


if __name__ == "__main__":
    if HAS_PYTEST:
        sys.exit(pytest.main([__file__, "-v"]))
    else:
        print("Running tests without pytest...\n")
        success = run_tests_standalone()
        sys.exit(0 if success else 1)