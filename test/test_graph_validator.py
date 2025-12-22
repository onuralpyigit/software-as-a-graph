#!/usr/bin/env python3
"""
Test Suite for Graph Validation Module
========================================

Comprehensive tests for validation of graph-based criticality analysis.

Run with:
    python -m pytest tests/test_validation.py -v
    
Or directly:
    python tests/test_validation.py
"""

import sys
import unittest
import math
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

from src.validation import (
    GraphValidator,
    ValidationResult,
    ValidationStatus,
    ConfusionMatrix,
    ComponentValidation,
    spearman_correlation,
    pearson_correlation,
    kendall_tau,
    validate_analysis,
    quick_validate
)


class TestStatisticalFunctions(unittest.TestCase):
    """Test statistical functions"""
    
    def test_pearson_perfect_positive(self):
        """Test Pearson correlation with perfect positive correlation"""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        r, p = pearson_correlation(x, y)
        self.assertAlmostEqual(r, 1.0, places=4)
        self.assertLess(p, 0.05)
    
    def test_pearson_perfect_negative(self):
        """Test Pearson correlation with perfect negative correlation"""
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        r, p = pearson_correlation(x, y)
        self.assertAlmostEqual(r, -1.0, places=4)
    
    def test_pearson_no_correlation(self):
        """Test Pearson correlation with no correlation"""
        x = [1, 2, 3, 4, 5]
        y = [3, 1, 4, 2, 5]
        r, p = pearson_correlation(x, y)
        self.assertLessEqual(abs(r), 0.6)  # Weak to moderate correlation
    
    def test_spearman_perfect_positive(self):
        """Test Spearman correlation with perfect rank correlation"""
        x = [1, 2, 3, 4, 5]
        y = [10, 20, 30, 40, 50]
        r, p = spearman_correlation(x, y)
        self.assertAlmostEqual(r, 1.0, places=4)
    
    def test_spearman_with_ties(self):
        """Test Spearman correlation with tied values"""
        x = [1, 2, 2, 4, 5]
        y = [1, 2, 3, 4, 5]
        r, p = spearman_correlation(x, y)
        self.assertGreater(r, 0.8)  # Should still be high
    
    def test_kendall_tau(self):
        """Test Kendall's tau correlation"""
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        tau = kendall_tau(x, y)
        self.assertAlmostEqual(tau, 1.0, places=4)
    
    def test_kendall_tau_reversed(self):
        """Test Kendall's tau with reversed order"""
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        tau = kendall_tau(x, y)
        self.assertAlmostEqual(tau, -1.0, places=4)
    
    def test_small_sample(self):
        """Test handling of small samples"""
        x = [1, 2]
        y = [1, 2]
        r, p = pearson_correlation(x, y)
        # Should handle gracefully
        self.assertIsInstance(r, float)


class TestConfusionMatrix(unittest.TestCase):
    """Test ConfusionMatrix class"""
    
    def test_perfect_classification(self):
        """Test perfect classification metrics"""
        cm = ConfusionMatrix(
            true_positives=10,
            true_negatives=10,
            false_positives=0,
            false_negatives=0
        )
        self.assertEqual(cm.precision, 1.0)
        self.assertEqual(cm.recall, 1.0)
        self.assertEqual(cm.f1_score, 1.0)
        self.assertEqual(cm.accuracy, 1.0)
    
    def test_no_positives(self):
        """Test metrics with no positive predictions"""
        cm = ConfusionMatrix(
            true_positives=0,
            true_negatives=10,
            false_positives=0,
            false_negatives=5
        )
        self.assertEqual(cm.precision, 0.0)
        self.assertEqual(cm.recall, 0.0)
        self.assertEqual(cm.f1_score, 0.0)
    
    def test_typical_case(self):
        """Test typical confusion matrix"""
        cm = ConfusionMatrix(
            true_positives=8,
            true_negatives=85,
            false_positives=7,
            false_negatives=0
        )
        # Precision = 8/15 = 0.533
        self.assertAlmostEqual(cm.precision, 8/15, places=3)
        # Recall = 8/8 = 1.0
        self.assertEqual(cm.recall, 1.0)
        # Accuracy = 93/100
        self.assertAlmostEqual(cm.accuracy, 0.93, places=2)
    
    def test_to_dict(self):
        """Test serialization"""
        cm = ConfusionMatrix(5, 5, 2, 3)
        d = cm.to_dict()
        self.assertIn('precision', d)
        self.assertIn('recall', d)
        self.assertIn('f1_score', d)


class TestGraphValidator(unittest.TestCase):
    """Test GraphValidator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test graph
        self.graph = nx.DiGraph()
        nodes = [
            ('app1', 'Application'),
            ('app2', 'Application'),
            ('app3', 'Application'),
            ('topic1', 'Topic'),
            ('topic2', 'Topic'),
            ('broker1', 'Broker'),
        ]
        for node_id, node_type in nodes:
            self.graph.add_node(node_id, type=node_type)
        
        self.graph.add_edges_from([
            ('app1', 'topic1'),
            ('app2', 'topic1'),
            ('topic1', 'app3'),
            ('app3', 'topic2'),
            ('topic1', 'broker1'),
        ])
        
        # Create predicted scores
        self.predicted = {
            'app1': 0.3,
            'app2': 0.2,
            'app3': 0.6,
            'topic1': 0.8,
            'topic2': 0.4,
            'broker1': 0.5
        }
        
        # Create actual impacts (similar pattern)
        self.actual = {
            'app1': 0.25,
            'app2': 0.15,
            'app3': 0.55,
            'topic1': 0.75,
            'topic2': 0.35,
            'broker1': 0.45
        }
    
    def test_basic_validation(self):
        """Test basic validation workflow"""
        validator = GraphValidator(seed=42)
        result = validator.validate(self.graph, self.predicted, self.actual)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.total_components, 6)
        self.assertIsInstance(result.status, ValidationStatus)
    
    def test_correlation_calculation(self):
        """Test correlation metrics are calculated correctly"""
        validator = GraphValidator(seed=42)
        result = validator.validate(self.graph, self.predicted, self.actual)
        
        # With similar patterns, correlation should be high
        self.assertGreater(result.correlation.spearman_coefficient, 0.8)
        self.assertGreater(result.correlation.pearson_coefficient, 0.8)
    
    def test_ranking_metrics(self):
        """Test ranking metrics calculation"""
        validator = GraphValidator(seed=42)
        result = validator.validate(self.graph, self.predicted, self.actual)
        
        self.assertIn(3, result.ranking.top_k_overlap)
        self.assertIn(5, result.ranking.top_k_overlap)
        self.assertGreaterEqual(result.ranking.mean_rank_difference, 0)
    
    def test_component_validations(self):
        """Test component-level validation"""
        validator = GraphValidator(seed=42)
        result = validator.validate(self.graph, self.predicted, self.actual)
        
        self.assertEqual(len(result.component_validations), 6)
        
        for cv in result.component_validations:
            self.assertIsInstance(cv, ComponentValidation)
            self.assertIn(cv.component_id, self.predicted)
            self.assertGreaterEqual(cv.rank_difference, 0)
    
    def test_status_determination(self):
        """Test validation status is correctly determined"""
        # Test with good correlation
        validator = GraphValidator(
            targets={'spearman_correlation': 0.5},  # Low target
            seed=42
        )
        result = validator.validate(self.graph, self.predicted, self.actual)
        
        # Should pass with high correlation and low target
        self.assertIn(result.status, [ValidationStatus.PASSED, ValidationStatus.MARGINAL])
    
    def test_custom_targets(self):
        """Test custom target thresholds"""
        validator = GraphValidator(
            targets={
                'spearman_correlation': 0.99,  # Very high
                'f1_score': 0.99
            },
            seed=42
        )
        result = validator.validate(self.graph, self.predicted, self.actual)
        
        # With very high targets, likely to fail
        self.assertIn(result.status, [ValidationStatus.MARGINAL, ValidationStatus.FAILED])
    
    def test_misclassified_components(self):
        """Test getting misclassified components"""
        validator = GraphValidator(seed=42)
        validator.validate(self.graph, self.predicted, self.actual)
        
        misclassified = validator.get_misclassified()
        self.assertIsInstance(misclassified, list)
    
    def test_high_rank_difference(self):
        """Test getting components with high rank difference"""
        validator = GraphValidator(seed=42)
        validator.validate(self.graph, self.predicted, self.actual)
        
        high_diff = validator.get_high_rank_difference(threshold=0)
        self.assertIsInstance(high_diff, list)
    
    def test_generate_recommendations(self):
        """Test recommendation generation"""
        validator = GraphValidator(seed=42)
        validator.validate(self.graph, self.predicted, self.actual)
        
        recommendations = validator.generate_recommendations()
        self.assertIsInstance(recommendations, list)
    
    def test_result_serialization(self):
        """Test result serialization to dict"""
        validator = GraphValidator(seed=42)
        result = validator.validate(self.graph, self.predicted, self.actual)
        
        result_dict = result.to_dict()
        
        self.assertIn('timestamp', result_dict)
        self.assertIn('status', result_dict)
        self.assertIn('correlation', result_dict)
        self.assertIn('classification', result_dict)
        self.assertIn('ranking', result_dict)
    
    def test_result_summary(self):
        """Test result summary generation"""
        validator = GraphValidator(seed=42)
        result = validator.validate(self.graph, self.predicted, self.actual)
        
        summary = result.summary()
        
        self.assertIsInstance(summary, str)
        self.assertIn('Spearman', summary)
        self.assertIn('F1-Score', summary)


class TestSensitivityAnalysis(unittest.TestCase):
    """Test sensitivity analysis"""
    
    def setUp(self):
        self.graph = nx.DiGraph()
        for i in range(10):
            self.graph.add_node(f'node_{i}', type='Application')
            if i > 0:
                self.graph.add_edge(f'node_{i-1}', f'node_{i}')
        
        self.predicted = {f'node_{i}': i/10 for i in range(10)}
        self.actual = {f'node_{i}': (i+1)/10 for i in range(10)}
    
    def test_sensitivity_analysis(self):
        """Test running sensitivity analysis"""
        validator = GraphValidator(seed=42)
        validator.validate(self.graph, self.predicted, self.actual)
        
        results = validator.run_sensitivity_analysis(
            self.graph, self.predicted, self.actual
        )
        
        self.assertGreater(len(results), 0)
        self.assertIn('critical_threshold', results[0].parameter_name)
        self.assertGreaterEqual(results[0].stability_score, 0)


class TestBootstrapAnalysis(unittest.TestCase):
    """Test bootstrap analysis"""
    
    def setUp(self):
        self.graph = nx.DiGraph()
        for i in range(20):
            self.graph.add_node(f'node_{i}', type='Application')
            if i > 0:
                self.graph.add_edge(f'node_{i-1}', f'node_{i}')
        
        self.predicted = {f'node_{i}': i/20 for i in range(20)}
        self.actual = {f'node_{i}': (i+1)/20 for i in range(20)}
    
    def test_bootstrap_analysis(self):
        """Test running bootstrap analysis"""
        validator = GraphValidator(seed=42)
        validator.validate(self.graph, self.predicted, self.actual)
        
        results = validator.run_bootstrap_analysis(
            self.graph, self.predicted, self.actual,
            n_iterations=100  # Fewer iterations for testing
        )
        
        self.assertGreater(len(results), 0)
        
        # Check confidence interval makes sense
        for br in results:
            self.assertLessEqual(br.ci_lower, br.point_estimate)
            self.assertGreaterEqual(br.ci_upper, br.point_estimate)


class TestCrossValidation(unittest.TestCase):
    """Test cross-validation"""
    
    def setUp(self):
        self.graph = nx.DiGraph()
        for i in range(20):
            self.graph.add_node(f'node_{i}', type='Application')
            if i > 0:
                self.graph.add_edge(f'node_{i-1}', f'node_{i}')
        
        self.predicted = {f'node_{i}': i/20 for i in range(20)}
        self.actual = {f'node_{i}': (i+1)/20 for i in range(20)}
    
    def test_cross_validation(self):
        """Test running cross-validation"""
        validator = GraphValidator(seed=42)
        validator.validate(self.graph, self.predicted, self.actual)
        
        result = validator.run_cross_validation(
            self.graph, self.predicted, self.actual,
            n_folds=5
        )
        
        self.assertEqual(result.n_folds, 5)
        self.assertGreater(len(result.fold_results), 0)
        self.assertIn('spearman', result.mean_metrics)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    def setUp(self):
        self.graph = nx.DiGraph()
        for i in range(10):
            self.graph.add_node(f'node_{i}', type='Application')
            if i > 0:
                self.graph.add_edge(f'node_{i-1}', f'node_{i}')
        
        self.predicted = {f'node_{i}': i/10 for i in range(10)}
        self.actual = {f'node_{i}': (i+1)/10 for i in range(10)}
    
    def test_validate_analysis(self):
        """Test validate_analysis convenience function"""
        result = validate_analysis(
            self.graph,
            self.predicted,
            self.actual,
            seed=42
        )
        
        self.assertIsInstance(result, ValidationResult)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases"""
    
    def test_small_graph(self):
        """Test with very small graph"""
        graph = nx.DiGraph()
        graph.add_node('a', type='Application')
        graph.add_node('b', type='Application')
        graph.add_node('c', type='Application')
        graph.add_edge('a', 'b')
        graph.add_edge('b', 'c')
        
        predicted = {'a': 0.3, 'b': 0.6, 'c': 0.1}
        actual = {'a': 0.2, 'b': 0.7, 'c': 0.1}
        
        validator = GraphValidator(seed=42)
        result = validator.validate(graph, predicted, actual)
        
        self.assertIsInstance(result, ValidationResult)
    
    def test_partial_overlap(self):
        """Test when predicted and actual have partial overlap"""
        graph = nx.DiGraph()
        for i in range(5):
            graph.add_node(f'node_{i}', type='Application')
        
        predicted = {'node_0': 0.5, 'node_1': 0.3, 'node_2': 0.7}
        actual = {'node_1': 0.4, 'node_2': 0.6, 'node_3': 0.8}
        
        validator = GraphValidator(seed=42)
        # Should work with common components (node_1, node_2)
        # But need at least 3 common
        with self.assertRaises(ValueError):
            validator.validate(graph, predicted, actual)
    
    def test_identical_scores(self):
        """Test when all scores are identical"""
        graph = nx.DiGraph()
        for i in range(5):
            graph.add_node(f'node_{i}', type='Application')
            if i > 0:
                graph.add_edge(f'node_{i-1}', f'node_{i}')
        
        predicted = {f'node_{i}': 0.5 for i in range(5)}
        actual = {f'node_{i}': 0.5 for i in range(5)}
        
        validator = GraphValidator(seed=42)
        result = validator.validate(graph, predicted, actual)
        
        # Correlation undefined with no variance
        self.assertIsInstance(result, ValidationResult)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticalFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestConfusionMatrix))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestSensitivityAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestBootstrapAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestCrossValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestConvenienceFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())