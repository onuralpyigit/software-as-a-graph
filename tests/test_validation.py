#!/usr/bin/env python3
"""
Test Suite for Validation Module - Version 5.0

Comprehensive tests for:
- Statistical metrics (correlation, classification, ranking)
- Validator component-level validation
- Validation pipeline
- Component-type specific validation

Run with: python -m pytest tests/test_validation.py -v
Or:       python tests/test_validation.py

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import unittest
import sys
import os
import math

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validation import (
    # Enums
    ValidationStatus,
    MetricStatus,
    # Metrics
    ValidationTargets,
    CorrelationMetrics,
    ClassificationMetrics,
    ConfusionMatrix,
    RankingMetrics,
    BootstrapCI,
    # Functions
    spearman_correlation,
    pearson_correlation,
    kendall_correlation,
    calculate_correlation,
    calculate_confusion_matrix,
    calculate_classification,
    calculate_ranking,
    bootstrap_confidence_interval,
    percentile,
    mean,
    std_dev,
    # Validator
    Validator,
    ValidationResult,
    ComponentValidation,
    validate_predictions,
    quick_validate,
    # Pipeline
    AnalysisMethod,
    GraphAnalyzer,
    ValidationPipeline,
    PipelineResult,
    run_validation,
)


# =============================================================================
# Test Data Factory
# =============================================================================

def create_test_scores():
    """Create test predicted and actual scores"""
    # Perfect correlation case
    perfect_predicted = {"a": 0.9, "b": 0.7, "c": 0.5, "d": 0.3, "e": 0.1}
    perfect_actual = {"a": 0.9, "b": 0.7, "c": 0.5, "d": 0.3, "e": 0.1}
    
    # Good correlation case
    good_predicted = {"a": 0.9, "b": 0.75, "c": 0.5, "d": 0.35, "e": 0.1}
    good_actual = {"a": 0.85, "b": 0.7, "c": 0.55, "d": 0.3, "e": 0.15}
    
    # Poor correlation case
    poor_predicted = {"a": 0.9, "b": 0.1, "c": 0.5, "d": 0.8, "e": 0.3}
    poor_actual = {"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.2, "e": 0.7}
    
    return {
        "perfect": (perfect_predicted, perfect_actual),
        "good": (good_predicted, good_actual),
        "poor": (poor_predicted, poor_actual),
    }


def create_test_graph():
    """Create a small test graph for pipeline tests"""
    from src.simulation import (
        SimulationGraph, Component, Edge,
        ComponentType, EdgeType
    )
    
    graph = SimulationGraph()
    
    # Add components
    for i in range(1, 6):
        graph.add_component(Component(
            id=f"app{i}",
            type=ComponentType.APPLICATION
        ))
    
    for i in range(1, 3):
        graph.add_component(Component(
            id=f"broker{i}",
            type=ComponentType.BROKER
        ))
    
    for i in range(1, 4):
        graph.add_component(Component(
            id=f"topic{i}",
            type=ComponentType.TOPIC
        ))
    
    graph.add_component(Component(
        id="node1",
        type=ComponentType.NODE
    ))
    
    # Add edges
    graph.add_edge(Edge(source="app1", target="topic1", edge_type=EdgeType.PUBLISHES_TO))
    graph.add_edge(Edge(source="app2", target="topic1", edge_type=EdgeType.SUBSCRIBES_TO))
    graph.add_edge(Edge(source="app3", target="topic2", edge_type=EdgeType.PUBLISHES_TO))
    graph.add_edge(Edge(source="app4", target="topic2", edge_type=EdgeType.SUBSCRIBES_TO))
    graph.add_edge(Edge(source="app5", target="topic3", edge_type=EdgeType.SUBSCRIBES_TO))
    graph.add_edge(Edge(source="broker1", target="topic1", edge_type=EdgeType.ROUTES))
    graph.add_edge(Edge(source="broker1", target="topic2", edge_type=EdgeType.ROUTES))
    graph.add_edge(Edge(source="broker2", target="topic3", edge_type=EdgeType.ROUTES))
    graph.add_edge(Edge(source="app1", target="node1", edge_type=EdgeType.RUNS_ON))
    graph.add_edge(Edge(source="broker1", target="node1", edge_type=EdgeType.RUNS_ON))
    
    return graph


# =============================================================================
# Test: Enums
# =============================================================================

class TestEnums(unittest.TestCase):
    """Test enum definitions"""
    
    def test_validation_status_values(self):
        """Test ValidationStatus enum"""
        self.assertEqual(ValidationStatus.PASSED.value, "passed")
        self.assertEqual(ValidationStatus.PARTIAL.value, "partial")
        self.assertEqual(ValidationStatus.FAILED.value, "failed")
        self.assertEqual(ValidationStatus.ERROR.value, "error")
    
    def test_metric_status_values(self):
        """Test MetricStatus enum"""
        self.assertEqual(MetricStatus.ABOVE_TARGET.value, "above_target")
        self.assertEqual(MetricStatus.BELOW_TARGET.value, "below_target")
    
    def test_analysis_method_values(self):
        """Test AnalysisMethod enum"""
        self.assertEqual(AnalysisMethod.BETWEENNESS.value, "betweenness")
        self.assertEqual(AnalysisMethod.PAGERANK.value, "pagerank")
        self.assertEqual(AnalysisMethod.COMPOSITE.value, "composite")


# =============================================================================
# Test: Correlation Metrics
# =============================================================================

class TestCorrelation(unittest.TestCase):
    """Test correlation calculations"""
    
    def test_pearson_perfect_positive(self):
        """Test Pearson with perfect positive correlation"""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        r = pearson_correlation(x, y)
        self.assertAlmostEqual(r, 1.0, places=5)
    
    def test_pearson_perfect_negative(self):
        """Test Pearson with perfect negative correlation"""
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        r = pearson_correlation(x, y)
        self.assertAlmostEqual(r, -1.0, places=5)
    
    def test_pearson_no_correlation(self):
        """Test Pearson with no correlation"""
        x = [1, 2, 3, 4, 5]
        y = [5, 5, 5, 5, 5]  # constant
        r = pearson_correlation(x, y)
        self.assertAlmostEqual(r, 0.0, places=5)
    
    def test_spearman_perfect_rank(self):
        """Test Spearman with perfect rank correlation"""
        x = [1, 2, 3, 4, 5]
        y = [10, 20, 30, 40, 50]
        rho = spearman_correlation(x, y)
        self.assertAlmostEqual(rho, 1.0, places=5)
    
    def test_spearman_reverse_rank(self):
        """Test Spearman with reverse rank correlation"""
        x = [1, 2, 3, 4, 5]
        y = [50, 40, 30, 20, 10]
        rho = spearman_correlation(x, y)
        self.assertAlmostEqual(rho, -1.0, places=5)
    
    def test_spearman_with_ties(self):
        """Test Spearman handles ties correctly"""
        x = [1, 2, 2, 4, 5]
        y = [1, 2, 2, 4, 5]
        rho = spearman_correlation(x, y)
        self.assertGreater(rho, 0.9)
    
    def test_kendall_perfect(self):
        """Test Kendall with perfect correlation"""
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        tau = kendall_correlation(x, y)
        self.assertAlmostEqual(tau, 1.0, places=5)
    
    def test_kendall_reverse(self):
        """Test Kendall with reverse correlation"""
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        tau = kendall_correlation(x, y)
        self.assertAlmostEqual(tau, -1.0, places=5)
    
    def test_calculate_correlation(self):
        """Test calculate_correlation returns all metrics"""
        x = [1, 2, 3, 4, 5]
        y = [1.1, 2.2, 2.9, 4.1, 5.0]
        
        metrics = calculate_correlation(x, y)
        
        self.assertIsInstance(metrics, CorrelationMetrics)
        self.assertGreater(metrics.spearman, 0.9)
        self.assertGreater(metrics.pearson, 0.9)
        self.assertGreater(metrics.kendall, 0.8)
        self.assertEqual(metrics.n_samples, 5)
    
    def test_empty_sequences(self):
        """Test correlation with empty sequences"""
        self.assertEqual(pearson_correlation([], []), 0.0)
        self.assertEqual(spearman_correlation([], []), 0.0)
        self.assertEqual(kendall_correlation([], []), 0.0)
    
    def test_single_element(self):
        """Test correlation with single element"""
        self.assertEqual(pearson_correlation([1], [2]), 0.0)


# =============================================================================
# Test: Classification Metrics
# =============================================================================

class TestClassification(unittest.TestCase):
    """Test classification metrics"""
    
    def test_confusion_matrix_perfect(self):
        """Test confusion matrix with perfect classification"""
        predicted = [True, True, False, False]
        actual = [True, True, False, False]
        
        cm = calculate_confusion_matrix(predicted, actual)
        
        self.assertEqual(cm.true_positives, 2)
        self.assertEqual(cm.true_negatives, 2)
        self.assertEqual(cm.false_positives, 0)
        self.assertEqual(cm.false_negatives, 0)
        self.assertEqual(cm.precision, 1.0)
        self.assertEqual(cm.recall, 1.0)
        self.assertEqual(cm.f1_score, 1.0)
        self.assertEqual(cm.accuracy, 1.0)
    
    def test_confusion_matrix_all_wrong(self):
        """Test confusion matrix with all wrong predictions"""
        predicted = [True, True, False, False]
        actual = [False, False, True, True]
        
        cm = calculate_confusion_matrix(predicted, actual)
        
        self.assertEqual(cm.true_positives, 0)
        self.assertEqual(cm.true_negatives, 0)
        self.assertEqual(cm.false_positives, 2)
        self.assertEqual(cm.false_negatives, 2)
        self.assertEqual(cm.precision, 0.0)
        self.assertEqual(cm.recall, 0.0)
        self.assertEqual(cm.f1_score, 0.0)
        self.assertEqual(cm.accuracy, 0.0)
    
    def test_confusion_matrix_mixed(self):
        """Test confusion matrix with mixed results"""
        predicted = [True, True, True, False, False]
        actual = [True, True, False, False, True]
        
        cm = calculate_confusion_matrix(predicted, actual)
        
        self.assertEqual(cm.true_positives, 2)  # predicted and actual True
        self.assertEqual(cm.true_negatives, 1)  # predicted and actual False
        self.assertEqual(cm.false_positives, 1)  # predicted True, actual False
        self.assertEqual(cm.false_negatives, 1)  # predicted False, actual True
    
    def test_calculate_classification(self):
        """Test calculate_classification function"""
        predicted = [0.9, 0.8, 0.3, 0.2, 0.1]
        actual = [0.85, 0.75, 0.35, 0.25, 0.15]
        
        metrics = calculate_classification(predicted, actual, threshold=0.5)
        
        self.assertIsInstance(metrics, ClassificationMetrics)
        self.assertEqual(metrics.threshold, 0.5)
        self.assertGreater(metrics.f1_score, 0)
    
    def test_classification_metrics_to_dict(self):
        """Test ClassificationMetrics serialization"""
        cm = ConfusionMatrix(
            true_positives=10,
            true_negatives=20,
            false_positives=5,
            false_negatives=5
        )
        metrics = ClassificationMetrics(confusion_matrix=cm, threshold=0.5)
        
        d = metrics.to_dict()
        
        self.assertIn("confusion_matrix", d)
        self.assertIn("threshold", d)
        self.assertIn("precision", d)
        self.assertIn("recall", d)
        self.assertIn("f1_score", d)


# =============================================================================
# Test: Ranking Metrics
# =============================================================================

class TestRanking(unittest.TestCase):
    """Test ranking metrics"""
    
    def test_perfect_ranking(self):
        """Test ranking with perfect agreement"""
        predicted = ["a", "b", "c", "d", "e"]
        actual = ["a", "b", "c", "d", "e"]
        
        metrics = calculate_ranking(predicted, actual)
        
        self.assertEqual(metrics.top_5_overlap, 1.0)
        self.assertAlmostEqual(metrics.ndcg, 1.0, places=5)
        self.assertEqual(metrics.rank_difference_mean, 0.0)
    
    def test_reversed_ranking(self):
        """Test ranking with reversed order"""
        predicted = ["e", "d", "c", "b", "a"]
        actual = ["a", "b", "c", "d", "e"]
        
        metrics = calculate_ranking(predicted, actual)
        
        # Same items but different order
        self.assertEqual(metrics.top_5_overlap, 1.0)
        self.assertGreater(metrics.rank_difference_mean, 0)
    
    def test_partial_overlap(self):
        """Test ranking with partial overlap"""
        predicted = ["a", "b", "f", "g", "h"]
        actual = ["a", "b", "c", "d", "e"]
        
        metrics = calculate_ranking(predicted, actual)
        
        # Only 2 of top 5 match
        self.assertEqual(metrics.top_5_overlap, 0.4)
    
    def test_top_k_overlaps(self):
        """Test different k values for overlap"""
        predicted = ["a", "b", "c", "x", "y", "z", "w", "v", "u", "t"]
        actual = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        
        metrics = calculate_ranking(predicted, actual, k_values=[3, 5, 10])
        
        self.assertEqual(metrics.top_k_overlaps[3], 1.0)  # all 3 match
        self.assertEqual(metrics.top_k_overlaps[5], 0.6)  # 3 of 5 match
        self.assertEqual(metrics.top_k_overlaps[10], 0.3)  # 3 of 10 match
    
    def test_ranking_metrics_to_dict(self):
        """Test RankingMetrics serialization"""
        metrics = RankingMetrics(
            top_5_overlap=0.8,
            top_10_overlap=0.7,
            ndcg=0.9,
            mrr=0.85,
        )
        
        d = metrics.to_dict()
        
        self.assertEqual(d["top_5_overlap"], 0.8)
        self.assertEqual(d["top_10_overlap"], 0.7)


# =============================================================================
# Test: Statistical Utilities
# =============================================================================

class TestStatisticalUtilities(unittest.TestCase):
    """Test statistical utility functions"""
    
    def test_percentile(self):
        """Test percentile calculation"""
        values = [1, 2, 3, 4, 5]
        
        self.assertEqual(percentile(values, 0), 1)
        self.assertEqual(percentile(values, 50), 3)
        self.assertEqual(percentile(values, 100), 5)
    
    def test_percentile_interpolation(self):
        """Test percentile interpolation"""
        values = [1, 2, 3, 4]
        p25 = percentile(values, 25)
        p75 = percentile(values, 75)
        
        self.assertGreater(p25, 1)
        self.assertLess(p25, 2)
        self.assertGreater(p75, 3)
        self.assertLess(p75, 4)
    
    def test_mean(self):
        """Test mean calculation"""
        values = [1, 2, 3, 4, 5]
        self.assertEqual(mean(values), 3.0)
    
    def test_std_dev(self):
        """Test standard deviation"""
        values = [1, 2, 3, 4, 5]
        sd = std_dev(values)
        self.assertAlmostEqual(sd, 1.5811, places=3)
    
    def test_empty_sequences(self):
        """Test utilities with empty sequences"""
        self.assertEqual(percentile([], 50), 0.0)
        self.assertEqual(mean([]), 0.0)
        self.assertEqual(std_dev([]), 0.0)


# =============================================================================
# Test: Bootstrap Confidence Interval
# =============================================================================

class TestBootstrap(unittest.TestCase):
    """Test bootstrap confidence intervals"""
    
    def test_bootstrap_ci_structure(self):
        """Test bootstrap CI returns correct structure"""
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [1.1, 2.2, 3.1, 4.2, 5.1, 6.2, 7.1, 8.2, 9.1, 10.2]
        
        ci = bootstrap_confidence_interval(
            x, y, pearson_correlation,
            n_bootstrap=100,
            confidence=0.95,
            seed=42
        )
        
        self.assertIsInstance(ci, BootstrapCI)
        self.assertEqual(ci.confidence, 0.95)
        self.assertEqual(ci.n_bootstrap, 100)
        self.assertLessEqual(ci.lower, ci.estimate)
        self.assertGreaterEqual(ci.upper, ci.estimate)
    
    def test_bootstrap_ci_width(self):
        """Test bootstrap CI width property"""
        ci = BootstrapCI(
            estimate=0.8,
            lower=0.7,
            upper=0.9,
            confidence=0.95,
            n_bootstrap=1000
        )
        
        self.assertAlmostEqual(ci.width, 0.2, places=5)
    
    def test_bootstrap_ci_contains(self):
        """Test bootstrap CI contains method"""
        ci = BootstrapCI(
            estimate=0.8,
            lower=0.7,
            upper=0.9,
            confidence=0.95,
            n_bootstrap=1000
        )
        
        self.assertTrue(ci.contains(0.8))
        self.assertTrue(ci.contains(0.75))
        self.assertFalse(ci.contains(0.5))


# =============================================================================
# Test: Validation Targets
# =============================================================================

class TestValidationTargets(unittest.TestCase):
    """Test validation targets"""
    
    def test_default_targets(self):
        """Test default validation targets"""
        targets = ValidationTargets()
        
        self.assertEqual(targets.spearman, 0.70)
        self.assertEqual(targets.f1_score, 0.90)
        self.assertEqual(targets.precision, 0.80)
        self.assertEqual(targets.recall, 0.80)
    
    def test_custom_targets(self):
        """Test custom validation targets"""
        targets = ValidationTargets(
            spearman=0.75,
            f1_score=0.85
        )
        
        self.assertEqual(targets.spearman, 0.75)
        self.assertEqual(targets.f1_score, 0.85)
    
    def test_to_dict(self):
        """Test targets serialization"""
        targets = ValidationTargets()
        d = targets.to_dict()
        
        self.assertIn("spearman", d)
        self.assertIn("f1_score", d)
        self.assertEqual(d["spearman"], 0.70)


# =============================================================================
# Test: Validator
# =============================================================================

class TestValidator(unittest.TestCase):
    """Test Validator class"""
    
    def setUp(self):
        self.validator = Validator()
        self.test_scores = create_test_scores()
    
    def test_validate_perfect_correlation(self):
        """Test validation with perfect correlation"""
        predicted, actual = self.test_scores["perfect"]
        
        result = self.validator.validate(predicted, actual)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertAlmostEqual(result.spearman, 1.0, places=5)
        self.assertEqual(result.status, ValidationStatus.PASSED)
    
    def test_validate_good_correlation(self):
        """Test validation with good correlation"""
        predicted, actual = self.test_scores["good"]
        
        result = self.validator.validate(predicted, actual)
        
        self.assertGreater(result.spearman, 0.8)
    
    def test_validate_with_component_types(self):
        """Test validation with component type mapping"""
        predicted = {"a": 0.9, "b": 0.7, "c": 0.5, "d": 0.3}
        actual = {"a": 0.85, "b": 0.7, "c": 0.55, "d": 0.25}
        types = {"a": "App", "b": "App", "c": "Broker", "d": "Broker"}
        
        result = self.validator.validate(predicted, actual, types)
        
        self.assertEqual(len(result.components), 4)
        for comp in result.components:
            self.assertIn(comp.component_type, ["App", "Broker"])
    
    def test_validate_returns_components(self):
        """Test that validation returns component details"""
        predicted = {"a": 0.9, "b": 0.5, "c": 0.1}
        actual = {"a": 0.85, "b": 0.55, "c": 0.15}
        
        result = self.validator.validate(predicted, actual)
        
        self.assertEqual(len(result.components), 3)
        
        for comp in result.components:
            self.assertIsInstance(comp, ComponentValidation)
            self.assertIn(comp.component_id, ["a", "b", "c"])
    
    def test_validate_misclassified(self):
        """Test getting misclassified components"""
        # Create scores where classification will differ
        predicted = {"a": 0.9, "b": 0.1, "c": 0.5}
        actual = {"a": 0.1, "b": 0.9, "c": 0.5}
        
        result = self.validator.validate(predicted, actual)
        misclassified = result.get_misclassified()
        
        # Some should be misclassified due to swapped scores
        self.assertIsInstance(misclassified, list)
    
    def test_result_to_dict(self):
        """Test ValidationResult serialization"""
        predicted, actual = self.test_scores["good"]
        result = self.validator.validate(predicted, actual)
        
        d = result.to_dict()
        
        self.assertIn("status", d)
        self.assertIn("correlation", d)
        self.assertIn("classification", d)
        self.assertIn("ranking", d)


# =============================================================================
# Test: Factory Functions
# =============================================================================

class TestFactoryFunctions(unittest.TestCase):
    """Test module factory functions"""
    
    def test_validate_predictions(self):
        """Test validate_predictions function"""
        predicted = {"a": 0.9, "b": 0.5, "c": 0.1}
        actual = {"a": 0.85, "b": 0.55, "c": 0.15}
        
        result = validate_predictions(predicted, actual)
        
        self.assertIsInstance(result, ValidationResult)
    
    def test_quick_validate(self):
        """Test quick_validate function"""
        predicted = {"a": 0.9, "b": 0.5, "c": 0.1}
        actual = {"a": 0.85, "b": 0.55, "c": 0.15}
        
        spearman, f1, passed = quick_validate(predicted, actual)
        
        self.assertIsInstance(spearman, float)
        self.assertIsInstance(f1, float)
        self.assertIsInstance(passed, bool)


# =============================================================================
# Test: Graph Analyzer
# =============================================================================

class TestGraphAnalyzer(unittest.TestCase):
    """Test GraphAnalyzer class"""
    
    def setUp(self):
        self.graph = create_test_graph()
        self.analyzer = GraphAnalyzer(seed=42)
    
    def test_analyze_betweenness(self):
        """Test betweenness centrality analysis"""
        scores = self.analyzer.analyze(self.graph, AnalysisMethod.BETWEENNESS)
        
        self.assertIsInstance(scores, dict)
        self.assertGreater(len(scores), 0)
        
        for comp_id, score in scores.items():
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_analyze_pagerank(self):
        """Test PageRank analysis"""
        scores = self.analyzer.analyze(self.graph, AnalysisMethod.PAGERANK)
        
        self.assertIsInstance(scores, dict)
        self.assertGreater(len(scores), 0)
    
    def test_analyze_degree(self):
        """Test degree centrality analysis"""
        scores = self.analyzer.analyze(self.graph, AnalysisMethod.DEGREE)
        
        self.assertIsInstance(scores, dict)
        self.assertGreater(len(scores), 0)
    
    def test_analyze_composite(self):
        """Test composite score analysis"""
        scores = self.analyzer.analyze(self.graph, AnalysisMethod.COMPOSITE)
        
        self.assertIsInstance(scores, dict)
        self.assertGreater(len(scores), 0)
    
    def test_analyze_all_methods(self):
        """Test analyzing with all methods"""
        all_scores = self.analyzer.analyze_all_methods(self.graph)
        
        self.assertIn("betweenness", all_scores)
        self.assertIn("pagerank", all_scores)
        self.assertIn("composite", all_scores)


# =============================================================================
# Test: Validation Pipeline
# =============================================================================

class TestValidationPipeline(unittest.TestCase):
    """Test ValidationPipeline class"""
    
    def setUp(self):
        self.graph = create_test_graph()
        self.pipeline = ValidationPipeline(seed=42)
    
    def test_run_pipeline(self):
        """Test running validation pipeline"""
        result = self.pipeline.run(self.graph)
        
        self.assertIsInstance(result, PipelineResult)
        self.assertIsInstance(result.validation, ValidationResult)
        self.assertGreater(len(result.predicted_scores), 0)
        self.assertGreater(len(result.actual_scores), 0)
    
    def test_run_with_compare_methods(self):
        """Test pipeline with method comparison"""
        result = self.pipeline.run(
            self.graph,
            compare_methods=True
        )
        
        self.assertIsNotNone(result.method_comparison)
        self.assertGreater(len(result.method_comparison), 0)
    
    def test_run_validates_by_type(self):
        """Test pipeline validates by component type"""
        result = self.pipeline.run(
            self.graph,
            validate_by_type=True
        )
        
        # Should have validation for at least one component type
        self.assertGreater(len(result.by_component_type), 0)
    
    def test_pipeline_result_to_dict(self):
        """Test PipelineResult serialization"""
        result = self.pipeline.run(self.graph)
        
        d = result.to_dict()
        
        self.assertIn("validation", d)
        self.assertIn("predicted_scores", d)
        self.assertIn("actual_scores", d)
        self.assertIn("analysis_method", d)
    
    def test_pipeline_result_summary(self):
        """Test PipelineResult summary"""
        result = self.pipeline.run(self.graph)
        summary = result.summary()
        
        self.assertIsInstance(summary, str)
        self.assertIn("Spearman", summary)


# =============================================================================
# Test: Integration
# =============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_full_validation_workflow(self):
        """Test complete validation workflow"""
        graph = create_test_graph()
        
        # Run pipeline
        pipeline = ValidationPipeline(seed=42)
        result = pipeline.run(
            graph,
            analysis_method=AnalysisMethod.COMPOSITE,
            compare_methods=True,
            validate_by_type=True,
        )
        
        # Verify complete result
        self.assertIsInstance(result, PipelineResult)
        self.assertIn(result.validation.status, [
            ValidationStatus.PASSED,
            ValidationStatus.PARTIAL,
            ValidationStatus.FAILED,
        ])
        
        # Verify metrics are calculated
        self.assertGreater(len(result.predicted_scores), 0)
        self.assertGreater(len(result.actual_scores), 0)
        
        # Verify method comparison
        self.assertIsNotNone(result.method_comparison)
        
        # Verify by-type validation
        self.assertGreater(len(result.by_component_type), 0)
    
    def test_custom_score_validation(self):
        """Test validation with custom scores"""
        # Create custom predicted and actual scores
        predicted = {
            "comp1": 0.9,
            "comp2": 0.7,
            "comp3": 0.5,
            "comp4": 0.3,
            "comp5": 0.1,
        }
        actual = {
            "comp1": 0.85,
            "comp2": 0.75,
            "comp3": 0.45,
            "comp4": 0.35,
            "comp5": 0.15,
        }
        component_types = {
            "comp1": "Application",
            "comp2": "Application",
            "comp3": "Broker",
            "comp4": "Topic",
            "comp5": "Node",
        }
        
        pipeline = ValidationPipeline(seed=42)
        result = pipeline.run_with_custom_scores(
            predicted, actual, component_types
        )
        
        self.assertIsInstance(result, PipelineResult)
        self.assertEqual(result.analysis_method, "custom")


# =============================================================================
# Main
# =============================================================================

def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEnums))
    suite.addTests(loader.loadTestsFromTestCase(TestCorrelation))
    suite.addTests(loader.loadTestsFromTestCase(TestClassification))
    suite.addTests(loader.loadTestsFromTestCase(TestRanking))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticalUtilities))
    suite.addTests(loader.loadTestsFromTestCase(TestBootstrap))
    suite.addTests(loader.loadTestsFromTestCase(TestValidationTargets))
    suite.addTests(loader.loadTestsFromTestCase(TestValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestFactoryFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestValidationPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
