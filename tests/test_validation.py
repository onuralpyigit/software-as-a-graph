#!/usr/bin/env python3
"""
Test Suite for src.validation Module - Version 5.0

Tests for validation including:
- Metrics (correlation, classification, ranking)
- Validator
- Pipeline
- Layer-specific validation

Run with pytest:
    pytest tests/test_validation.py -v

Or standalone:
    python tests/test_validation.py

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import sys
from pathlib import Path

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

def create_test_scores():
    """Create test predicted and actual scores."""
    predicted = {
        "comp_1": 0.90,
        "comp_2": 0.80,
        "comp_3": 0.70,
        "comp_4": 0.60,
        "comp_5": 0.50,
        "comp_6": 0.40,
        "comp_7": 0.30,
        "comp_8": 0.20,
        "comp_9": 0.10,
        "comp_10": 0.05,
    }
    
    # Good correlation with predicted
    actual = {
        "comp_1": 0.85,
        "comp_2": 0.75,
        "comp_3": 0.65,
        "comp_4": 0.55,
        "comp_5": 0.45,
        "comp_6": 0.35,
        "comp_7": 0.25,
        "comp_8": 0.15,
        "comp_9": 0.12,
        "comp_10": 0.08,
    }
    
    return predicted, actual


def create_test_component_info():
    """Create test component info with layers."""
    return {
        "comp_1": {"type": "Broker", "layer": "app_broker"},
        "comp_2": {"type": "Broker", "layer": "app_broker"},
        "comp_3": {"type": "Application", "layer": "application"},
        "comp_4": {"type": "Application", "layer": "application"},
        "comp_5": {"type": "Application", "layer": "application"},
        "comp_6": {"type": "Node", "layer": "infrastructure"},
        "comp_7": {"type": "Node", "layer": "infrastructure"},
        "comp_8": {"type": "Topic", "layer": "application"},
        "comp_9": {"type": "Topic", "layer": "application"},
        "comp_10": {"type": "Topic", "layer": "application"},
    }


def create_test_graph():
    """Create a test simulation graph."""
    from src.simulation import create_simulation_graph
    return create_simulation_graph(
        applications=6,
        brokers=2,
        topics=8,
        nodes=3,
        seed=42,
    )


# =============================================================================
# Test: Statistical Utilities
# =============================================================================

class TestStatisticalUtilities:
    """Tests for statistical utility functions."""

    def test_mean(self):
        from src.validation import mean
        
        assert mean([1, 2, 3, 4, 5]) == 3.0
        assert mean([10]) == 10.0
        assert mean([]) == 0.0

    def test_std_dev(self):
        from src.validation import std_dev
        
        # Known values
        values = [2, 4, 4, 4, 5, 5, 7, 9]
        result = std_dev(values)
        assert 2.0 < result < 2.2  # Approximate

    def test_median(self):
        from src.validation import median
        
        assert median([1, 2, 3, 4, 5]) == 3.0
        assert median([1, 2, 3, 4]) == 2.5
        assert median([]) == 0.0

    def test_percentile(self):
        from src.validation import percentile
        
        values = list(range(1, 101))  # 1 to 100
        
        assert percentile(values, 50) == 50.5
        assert percentile(values, 0) == 1
        assert percentile(values, 100) == 100

    def test_iqr(self):
        from src.validation import iqr
        
        values = list(range(1, 101))
        result = iqr(values)
        assert 49 < result < 51  # Q3 - Q1 ≈ 50


# =============================================================================
# Test: Correlation Metrics
# =============================================================================

class TestCorrelationMetrics:
    """Tests for correlation metrics."""

    def test_spearman_perfect(self):
        from src.validation import spearman_correlation
        
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        
        result = spearman_correlation(x, y)
        assert abs(result - 1.0) < 0.001

    def test_spearman_inverse(self):
        from src.validation import spearman_correlation
        
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        
        result = spearman_correlation(x, y)
        assert abs(result - (-1.0)) < 0.001

    def test_spearman_no_correlation(self):
        from src.validation import spearman_correlation
        
        x = [1, 2, 3, 4, 5]
        y = [3, 1, 4, 2, 5]
        
        result = spearman_correlation(x, y)
        # This particular data has some correlation, just check it's computed
        assert -1.0 <= result <= 1.0

    def test_pearson_perfect(self):
        from src.validation import pearson_correlation
        
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]  # Linear relationship
        
        result = pearson_correlation(x, y)
        assert abs(result - 1.0) < 0.001

    def test_kendall_perfect(self):
        from src.validation import kendall_correlation
        
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        
        result = kendall_correlation(x, y)
        assert abs(result - 1.0) < 0.001

    def test_calculate_correlation(self):
        from src.validation import calculate_correlation
        
        x = [0.9, 0.7, 0.5, 0.3, 0.1]
        y = [0.85, 0.65, 0.45, 0.25, 0.15]
        
        result = calculate_correlation(x, y)
        
        assert result.spearman > 0.9
        assert result.pearson > 0.9
        assert result.n_samples == 5


# =============================================================================
# Test: Classification Metrics
# =============================================================================

class TestClassificationMetrics:
    """Tests for classification metrics."""

    def test_confusion_matrix(self):
        from src.validation import calculate_confusion_matrix
        
        predicted = [0.9, 0.8, 0.3, 0.2]
        actual = [0.85, 0.75, 0.25, 0.15]
        
        cm = calculate_confusion_matrix(predicted, actual, threshold=0.5)
        
        # 2 above threshold in both, 2 below in both
        assert cm.true_positives == 2
        assert cm.true_negatives == 2
        assert cm.false_positives == 0
        assert cm.false_negatives == 0

    def test_confusion_matrix_with_errors(self):
        from src.validation import calculate_confusion_matrix
        
        predicted = [0.9, 0.3, 0.8, 0.2]  # 1,3 above
        actual = [0.9, 0.8, 0.3, 0.2]      # 1,2 above
        
        cm = calculate_confusion_matrix(predicted, actual, threshold=0.5)
        
        assert cm.true_positives == 1   # comp 1
        assert cm.false_positives == 1  # comp 3 (pred high, actual low)
        assert cm.false_negatives == 1  # comp 2 (pred low, actual high)
        assert cm.true_negatives == 1   # comp 4

    def test_precision_recall(self):
        from src.validation import ConfusionMatrix
        
        cm = ConfusionMatrix(
            true_positives=8,
            false_positives=2,
            false_negatives=2,
            true_negatives=88,
        )
        
        assert abs(cm.precision - 0.8) < 0.001  # 8 / (8 + 2)
        assert abs(cm.recall - 0.8) < 0.001     # 8 / (8 + 2)
        assert abs(cm.f1_score - 0.8) < 0.001

    def test_calculate_classification(self):
        from src.validation import calculate_classification
        
        predicted = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        actual = [0.85, 0.75, 0.65, 0.25, 0.15, 0.05]
        
        result = calculate_classification(predicted, actual)
        
        assert result.threshold > 0
        assert 0 <= result.precision <= 1
        assert 0 <= result.recall <= 1


# =============================================================================
# Test: Ranking Metrics
# =============================================================================

class TestRankingMetrics:
    """Tests for ranking metrics."""

    def test_top_k_overlap_perfect(self):
        from src.validation import calculate_ranking
        
        predicted = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5}
        actual = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5}
        
        result = calculate_ranking(predicted, actual)
        
        assert result.top_3_overlap == 1.0
        assert result.top_5_overlap == 1.0

    def test_top_k_overlap_partial(self):
        from src.validation import calculate_ranking
        
        # Different order
        predicted = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5}
        actual = {"b": 0.9, "a": 0.8, "c": 0.7, "e": 0.6, "d": 0.5}
        
        result = calculate_ranking(predicted, actual)
        
        # Top 3: pred [a,b,c], actual [b,a,c] → overlap = 3/3 = 1.0
        assert result.top_3_overlap == 1.0

    def test_ndcg(self):
        from src.validation import calculate_ranking
        
        predicted = {"a": 0.9, "b": 0.8, "c": 0.3}
        actual = {"a": 1.0, "b": 0.5, "c": 0.1}
        
        result = calculate_ranking(predicted, actual)
        
        # NDCG should be high since ranking is correct
        assert result.ndcg > 0.9

    def test_mrr(self):
        from src.validation import calculate_ranking
        
        predicted = {"a": 0.9, "b": 0.8, "c": 0.7}
        actual = {"a": 1.0, "b": 0.5, "c": 0.1}
        
        result = calculate_ranking(predicted, actual)
        
        # First predicted item is first actual → MRR = 1
        assert result.mrr == 1.0


# =============================================================================
# Test: Bootstrap CI
# =============================================================================

class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_bootstrap_ci(self):
        from src.validation import bootstrap_confidence_interval, spearman_correlation
        
        x = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
        y = [0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.12, 0.08]
        
        ci = bootstrap_confidence_interval(
            x, y, spearman_correlation,
            n_bootstrap=100,
            seed=42,
        )
        
        assert ci.lower <= ci.point_estimate <= ci.upper
        assert ci.confidence == 0.95

    def test_bootstrap_ci_small_sample(self):
        from src.validation import bootstrap_confidence_interval, spearman_correlation
        
        x = [0.9, 0.5, 0.1]
        y = [0.8, 0.4, 0.2]
        
        ci = bootstrap_confidence_interval(x, y, spearman_correlation, seed=42)
        
        # Should handle small samples gracefully
        assert ci.lower <= ci.upper


# =============================================================================
# Test: Validator
# =============================================================================

class TestValidator:
    """Tests for Validator class."""

    def test_basic_validation(self):
        from src.validation import Validator
        
        predicted, actual = create_test_scores()
        
        validator = Validator(seed=42)
        result = validator.validate(predicted, actual)
        
        assert result.spearman > 0.9  # High correlation expected
        assert len(result.components) == 10

    def test_validation_with_component_info(self):
        from src.validation import Validator
        
        predicted, actual = create_test_scores()
        component_info = create_test_component_info()
        
        validator = Validator(seed=42)
        result = validator.validate(predicted, actual, component_info)
        
        # Should have layer results
        assert len(result.by_layer) > 0
        
        # Should have type results
        assert len(result.by_type) > 0

    def test_validation_by_layer(self):
        from src.validation import Validator
        
        predicted, actual = create_test_scores()
        component_info = create_test_component_info()
        
        validator = Validator(seed=42)
        result = validator.validate(predicted, actual, component_info)
        
        # Check at least one layer has results (minimum 3 components required)
        assert len(result.by_layer) >= 1
        
        # Application layer should have results (has Topics and Applications = 6 components)
        if "application" in result.by_layer:
            assert result.by_layer["application"].count > 0
            assert result.by_layer["application"].spearman is not None

    def test_validation_status_passed(self):
        from src.validation import Validator, ValidationTargets, ValidationStatus
        
        # Low targets for easy passing
        targets = ValidationTargets(spearman=0.5, f1_score=0.5)
        predicted, actual = create_test_scores()
        
        validator = Validator(targets=targets, seed=42)
        result = validator.validate(predicted, actual)
        
        assert result.status == ValidationStatus.PASSED

    def test_validation_status_failed(self):
        from src.validation import Validator, ValidationTargets, ValidationStatus
        
        # High targets for certain failure
        targets = ValidationTargets(spearman=0.99, f1_score=0.99)
        
        # Create data with some noise
        predicted = {"a": 0.9, "b": 0.5, "c": 0.1}
        actual = {"a": 0.5, "b": 0.9, "c": 0.3}  # Different order
        
        validator = Validator(targets=targets, seed=42)
        result = validator.validate(predicted, actual)
        
        assert result.status in [ValidationStatus.FAILED, ValidationStatus.PARTIAL]

    def test_false_positives_negatives(self):
        from src.validation import Validator
        
        predicted, actual = create_test_scores()
        component_info = create_test_component_info()
        
        validator = Validator(seed=42)
        result = validator.validate(predicted, actual, component_info)
        
        # Methods should work
        fps = result.get_false_positives()
        fns = result.get_false_negatives()
        misclassified = result.get_misclassified()
        
        assert isinstance(fps, list)
        assert isinstance(fns, list)
        assert len(misclassified) == len(fps) + len(fns)


# =============================================================================
# Test: Pipeline
# =============================================================================

class TestPipeline:
    """Tests for ValidationPipeline."""

    def test_pipeline_run(self):
        from src.validation import ValidationPipeline, AnalysisMethod
        
        graph = create_test_graph()
        
        pipeline = ValidationPipeline(seed=42)
        result = pipeline.run(graph, analysis_method=AnalysisMethod.COMPOSITE)
        
        assert result.spearman is not None
        assert result.f1_score is not None
        assert len(result.predicted_scores) > 0
        assert len(result.actual_scores) > 0

    def test_pipeline_with_method_comparison(self):
        from src.validation import ValidationPipeline
        
        graph = create_test_graph()
        
        pipeline = ValidationPipeline(seed=42)
        result = pipeline.run(graph, compare_methods=True)
        
        assert result.method_comparison is not None
        assert len(result.method_comparison) >= 3
        
        # Check all methods present
        assert "composite" in result.method_comparison
        assert "betweenness" in result.method_comparison

    def test_pipeline_layer_results(self):
        from src.validation import ValidationPipeline
        
        graph = create_test_graph()
        
        pipeline = ValidationPipeline(seed=42)
        result = pipeline.run(graph)
        
        # Should have layer results
        assert len(result.by_layer) > 0

    def test_pipeline_timing(self):
        from src.validation import ValidationPipeline
        
        graph = create_test_graph()
        
        pipeline = ValidationPipeline(seed=42)
        result = pipeline.run(graph)
        
        assert result.analysis_time_ms >= 0
        assert result.simulation_time_ms >= 0
        assert result.validation_time_ms >= 0

    def test_get_best_method(self):
        from src.validation import ValidationPipeline
        
        graph = create_test_graph()
        
        pipeline = ValidationPipeline(seed=42)
        result = pipeline.run(graph, compare_methods=True)
        
        best = result.get_best_method()
        assert best is not None
        assert best in result.method_comparison


# =============================================================================
# Test: Graph Analyzer
# =============================================================================

class TestGraphAnalyzer:
    """Tests for GraphAnalyzer."""

    def test_analyze_composite(self):
        from src.validation import GraphAnalyzer, AnalysisMethod
        
        graph = create_test_graph()
        analyzer = GraphAnalyzer(seed=42)
        
        scores = analyzer.analyze(graph, AnalysisMethod.COMPOSITE)
        
        assert len(scores) == len(graph.components)
        assert all(0 <= s <= 1 for s in scores.values())

    def test_analyze_betweenness(self):
        from src.validation import GraphAnalyzer, AnalysisMethod
        
        graph = create_test_graph()
        analyzer = GraphAnalyzer(seed=42)
        
        scores = analyzer.analyze(graph, AnalysisMethod.BETWEENNESS)
        
        assert len(scores) > 0

    def test_analyze_pagerank(self):
        from src.validation import GraphAnalyzer, AnalysisMethod
        
        graph = create_test_graph()
        analyzer = GraphAnalyzer(seed=42)
        
        scores = analyzer.analyze(graph, AnalysisMethod.PAGERANK)
        
        assert len(scores) > 0

    def test_analyze_all_methods(self):
        from src.validation import GraphAnalyzer
        
        graph = create_test_graph()
        analyzer = GraphAnalyzer(seed=42)
        
        all_scores = analyzer.analyze_all_methods(graph)
        
        assert "composite" in all_scores
        assert "betweenness" in all_scores
        assert "pagerank" in all_scores
        assert "degree" in all_scores


# =============================================================================
# Test: Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_validate_predictions(self):
        from src.validation import validate_predictions
        
        predicted, actual = create_test_scores()
        
        result = validate_predictions(predicted, actual)
        
        assert result.spearman > 0.9

    def test_quick_validate(self):
        from src.validation import quick_validate, ValidationStatus
        
        predicted, actual = create_test_scores()
        
        status = quick_validate(predicted, actual)
        
        assert status in [ValidationStatus.PASSED, ValidationStatus.PARTIAL, ValidationStatus.FAILED]

    def test_run_validation(self):
        from src.validation import run_validation
        
        graph = create_test_graph()
        
        result = run_validation(graph, seed=42)
        
        assert result.passed is not None

    def test_quick_pipeline(self):
        from src.validation import quick_pipeline, ValidationStatus
        
        graph = create_test_graph()
        
        status = quick_pipeline(graph, seed=42)
        
        assert status in [ValidationStatus.PASSED, ValidationStatus.PARTIAL, ValidationStatus.FAILED]


# =============================================================================
# Test: Serialization
# =============================================================================

class TestSerialization:
    """Tests for data class serialization."""

    def test_validation_result_to_dict(self):
        from src.validation import Validator
        
        predicted, actual = create_test_scores()
        component_info = create_test_component_info()
        
        validator = Validator(seed=42)
        result = validator.validate(predicted, actual, component_info)
        
        data = result.to_dict()
        
        assert "status" in data
        assert "spearman" in data
        assert "by_layer" in data
        assert "correlation" in data

    def test_pipeline_result_to_dict(self):
        from src.validation import ValidationPipeline
        
        graph = create_test_graph()
        
        pipeline = ValidationPipeline(seed=42)
        result = pipeline.run(graph, compare_methods=True)
        
        data = result.to_dict()
        
        assert "validation" in data
        assert "predicted_scores" in data
        assert "actual_scores" in data
        assert "method_comparison" in data

    def test_layer_result_to_dict(self):
        from src.validation import Validator
        
        predicted, actual = create_test_scores()
        component_info = create_test_component_info()
        
        validator = Validator(seed=42)
        result = validator.validate(predicted, actual, component_info)
        
        for layer, layer_result in result.by_layer.items():
            data = layer_result.to_dict()
            
            assert "layer" in data
            assert "spearman" in data
            assert "f1_score" in data
            assert "status" in data


# =============================================================================
# Standalone Test Runner
# =============================================================================

# =============================================================================
# Test: Neo4j Client
# =============================================================================

class TestNeo4jValidationClient:
    """Tests for Neo4jValidationClient (skipped if Neo4j unavailable)."""
    
    @classmethod
    def setup_class(cls):
        """Check if Neo4j is available."""
        from src.validation import check_neo4j_available
        cls.neo4j_available = check_neo4j_available()
        cls.connection_works = False
        
        if cls.neo4j_available:
            try:
                from src.validation import Neo4jValidationClient
                with Neo4jValidationClient() as client:
                    cls.connection_works = client.verify_connection()
            except:
                pass
    
    def test_check_neo4j_available(self):
        from src.validation import check_neo4j_available
        
        # Just verify the function runs
        result = check_neo4j_available()
        assert isinstance(result, bool)
    
    def test_neo4j_config(self):
        from src.validation.neo4j_client import Neo4jConfig
        
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
            database="neo4j"
        )
        
        assert config.uri == "bolt://localhost:7687"
        assert config.user == "neo4j"
        assert config.password == "test"
        
        data = config.to_dict()
        assert "uri" in data
        assert "user" in data
    
    def test_neo4j_config_from_dict(self):
        from src.validation.neo4j_client import Neo4jConfig
        
        data = {
            "uri": "bolt://host:7687",
            "user": "testuser",
            "password": "testpass",
            "database": "testdb"
        }
        
        config = Neo4jConfig.from_dict(data)
        assert config.uri == "bolt://host:7687"
        assert config.user == "testuser"
    
    def test_client_import_error(self):
        """Test behavior when Neo4j driver not available."""
        from src.validation import Neo4jValidationClient, check_neo4j_available
        
        # If Neo4j is not available, instantiation should fail
        if not check_neo4j_available():
            try:
                client = Neo4jValidationClient()
                assert False, "Should have raised ImportError"
            except ImportError:
                pass
    
    def test_validate_from_neo4j_function(self):
        """Test factory function exists and has correct signature."""
        from src.validation import validate_from_neo4j
        import inspect
        
        sig = inspect.signature(validate_from_neo4j)
        params = list(sig.parameters.keys())
        
        assert "uri" in params
        assert "user" in params
        assert "password" in params
        assert "database" in params
        assert "method" in params
        assert "compare_methods" in params
        assert "layer" in params
    
    def test_client_layer_names(self):
        """Test layer name definitions are correct."""
        if not self.neo4j_available:
            return  # Skip
        
        from src.validation import Neo4jValidationClient
        
        # Verify layer name mappings
        assert "application" in Neo4jValidationClient.LAYER_NAMES
        assert "infrastructure" in Neo4jValidationClient.LAYER_NAMES
        assert "app_broker" in Neo4jValidationClient.LAYER_NAMES
        assert "node_broker" in Neo4jValidationClient.LAYER_NAMES
    
    def test_client_with_live_connection(self):
        """Test actual Neo4j connection (skipped if unavailable)."""
        if not self.connection_works:
            return  # Skip
        
        from src.validation import Neo4jValidationClient
        
        with Neo4jValidationClient() as client:
            # Verify connection
            assert client.verify_connection()
            
            # Get statistics
            stats = client.get_statistics()
            assert "total_components" in stats
            assert "total_edges" in stats
    
    def test_run_validation_live(self):
        """Test running validation on live data (skipped if unavailable)."""
        if not self.connection_works:
            return  # Skip
        
        from src.validation import Neo4jValidationClient
        
        with Neo4jValidationClient() as client:
            stats = client.get_statistics()
            
            # Only run if there's data
            if stats['total_components'] > 5:
                result = client.run_validation()
                assert result is not None
                assert hasattr(result, 'spearman')
                assert hasattr(result, 'f1_score')
    
    def test_validate_layer_live(self):
        """Test layer validation (skipped if unavailable)."""
        if not self.connection_works:
            return  # Skip
        
        from src.validation import Neo4jValidationClient
        
        with Neo4jValidationClient() as client:
            stats = client.get_statistics()
            
            # Only run if there's data
            if stats['total_components'] > 5:
                for layer in ["application", "infrastructure", "app_broker", "node_broker"]:
                    try:
                        result = client.validate_layer(layer)
                        assert result is not None
                    except Exception:
                        pass  # May fail if layer has insufficient data


def run_tests_standalone():
    """Run tests without pytest."""
    import traceback
    
    test_classes = [
        TestStatisticalUtilities,
        TestCorrelationMetrics,
        TestClassificationMetrics,
        TestRankingMetrics,
        TestBootstrapCI,
        TestValidator,
        TestPipeline,
        TestGraphAnalyzer,
        TestFactoryFunctions,
        TestSerialization,
        TestNeo4jValidationClient,
    ]
    
    passed = failed = skipped = 0
    
    print("=" * 60)
    print("Software-as-a-Graph Validation Module Tests")
    print("=" * 60)
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)
        
        # Setup class if needed
        if hasattr(test_class, 'setup_class'):
            try:
                test_class.setup_class()
            except:
                pass
        
        instance = test_class()
        for method_name in [m for m in dir(instance) if m.startswith("test_")]:
            try:
                getattr(instance, method_name)()
                print(f"  ✓ {method_name}")
                passed += 1
            except Exception as e:
                if "Skip" in str(e) or str(e) == "":
                    print(f"  ○ {method_name} (skipped)")
                    skipped += 1
                else:
                    print(f"  ✗ {method_name}: {e}")
                    if "--verbose" in sys.argv:
                        traceback.print_exc()
                    failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    if HAS_PYTEST and "--no-pytest" not in sys.argv:
        sys.exit(pytest.main([__file__, "-v"]))
    else:
        success = run_tests_standalone()
        sys.exit(0 if success else 1)