#!/usr/bin/env python3
"""
Test Suite for Graph Validator
===============================

Comprehensive tests for validation of graph-based analysis against simulation.

Usage:
    python test_graph_validator.py
    python test_graph_validator.py -v

Author: Software-as-a-Graph Research Project
"""

import sys
import json
import unittest
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import GraphAnalyzer
from src.simulation import GraphSimulator
from src.validation import (
    GraphValidator,
    ValidationResult,
    ComponentValidation,
    ConfusionMatrix,
    ValidationStatus,
    CriticalityThreshold,
    validate_analysis,
    quick_validate,
    spearman_correlation,
    pearson_correlation,
)


# ============================================================================
# Test Data
# ============================================================================

SAMPLE_PUBSUB_DATA = {
    "nodes": [
        {"id": "N1", "name": "ComputeNode1", "type": "compute"},
        {"id": "N2", "name": "ComputeNode2", "type": "compute"}
    ],
    "brokers": [
        {"id": "B1", "name": "MainBroker", "node": "N1"},
        {"id": "B2", "name": "BackupBroker", "node": "N2"}
    ],
    "applications": [
        {"id": "A1", "name": "Publisher", "role": "pub", "node": "N1"},
        {"id": "A2", "name": "Processor", "role": "both", "node": "N1"},
        {"id": "A3", "name": "Subscriber1", "role": "sub", "node": "N2"},
        {"id": "A4", "name": "Subscriber2", "role": "sub", "node": "N2"}
    ],
    "topics": [
        {"id": "T1", "name": "data", "broker": "B1"},
        {"id": "T2", "name": "processed", "broker": "B1"}
    ],
    "relationships": {
        "publishes_to": [
            {"from": "A1", "to": "T1"},
            {"from": "A2", "to": "T2"}
        ],
        "subscribes_to": [
            {"from": "A2", "to": "T1"},
            {"from": "A3", "to": "T2"},
            {"from": "A4", "to": "T2"}
        ],
        "runs_on": [
            {"from": "A1", "to": "N1"},
            {"from": "A2", "to": "N1"},
            {"from": "A3", "to": "N2"},
            {"from": "A4", "to": "N2"},
            {"from": "B1", "to": "N1"},
            {"from": "B2", "to": "N2"}
        ],
        "routes": [
            {"from": "B1", "to": "T1"},
            {"from": "B1", "to": "T2"}
        ]
    }
}


def create_test_analyzer() -> GraphAnalyzer:
    """Create a test analyzer with sample data"""
    analyzer = GraphAnalyzer(alpha=0.4, beta=0.3, gamma=0.3)
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    return analyzer


# ============================================================================
# Test Classes
# ============================================================================

class TestStatisticalFunctions(unittest.TestCase):
    """Tests for statistical correlation functions"""
    
    def test_pearson_perfect_positive(self):
        """Test Pearson with perfect positive correlation"""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        r, p = pearson_correlation(x, y)
        self.assertAlmostEqual(r, 1.0, places=4)
    
    def test_pearson_perfect_negative(self):
        """Test Pearson with perfect negative correlation"""
        x = [1, 2, 3, 4, 5]
        y = [10, 8, 6, 4, 2]
        r, p = pearson_correlation(x, y)
        self.assertAlmostEqual(r, -1.0, places=4)
    
    def test_pearson_no_correlation(self):
        """Test Pearson with no correlation"""
        x = [1, 2, 3, 4, 5]
        y = [3, 1, 4, 1, 5]  # Random-ish
        r, p = pearson_correlation(x, y)
        self.assertLess(abs(r), 0.5)
    
    def test_pearson_short_list(self):
        """Test Pearson with too short list"""
        x = [1, 2]
        y = [2, 3]
        r, p = pearson_correlation(x, y)
        # Should handle gracefully
        self.assertIsInstance(r, float)
    
    def test_spearman_perfect_positive(self):
        """Test Spearman with perfect positive correlation"""
        x = [1, 2, 3, 4, 5]
        y = [10, 20, 30, 40, 50]
        r, p = spearman_correlation(x, y)
        self.assertAlmostEqual(r, 1.0, places=4)
    
    def test_spearman_with_ties(self):
        """Test Spearman handles ties"""
        x = [1, 2, 2, 3, 4]
        y = [1, 2, 3, 3, 4]
        r, p = spearman_correlation(x, y)
        self.assertGreater(r, 0.5)
    
    def test_spearman_monotonic(self):
        """Test Spearman with monotonic but non-linear relationship"""
        x = [1, 2, 3, 4, 5]
        y = [1, 4, 9, 16, 25]  # Quadratic
        r, p = spearman_correlation(x, y)
        self.assertAlmostEqual(r, 1.0, places=4)  # Perfect rank correlation


class TestConfusionMatrix(unittest.TestCase):
    """Tests for ConfusionMatrix class"""
    
    def test_precision_calculation(self):
        """Test precision calculation"""
        cm = ConfusionMatrix(
            true_positives=8,
            false_positives=2,
            true_negatives=85,
            false_negatives=5
        )
        self.assertAlmostEqual(cm.precision, 0.8, places=4)
    
    def test_recall_calculation(self):
        """Test recall calculation"""
        cm = ConfusionMatrix(
            true_positives=8,
            false_positives=2,
            true_negatives=85,
            false_negatives=5
        )
        expected_recall = 8 / (8 + 5)  # ~0.615
        self.assertAlmostEqual(cm.recall, expected_recall, places=4)
    
    def test_f1_score_calculation(self):
        """Test F1-score calculation"""
        cm = ConfusionMatrix(
            true_positives=8,
            false_positives=2,
            true_negatives=85,
            false_negatives=5
        )
        p = cm.precision
        r = cm.recall
        expected_f1 = 2 * p * r / (p + r)
        self.assertAlmostEqual(cm.f1_score, expected_f1, places=4)
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation"""
        cm = ConfusionMatrix(
            true_positives=8,
            false_positives=2,
            true_negatives=85,
            false_negatives=5
        )
        expected_accuracy = (8 + 85) / (8 + 2 + 85 + 5)
        self.assertAlmostEqual(cm.accuracy, expected_accuracy, places=4)
    
    def test_zero_denominators(self):
        """Test handling of zero denominators"""
        # No positive predictions
        cm = ConfusionMatrix(
            true_positives=0,
            false_positives=0,
            true_negatives=100,
            false_negatives=0
        )
        self.assertEqual(cm.precision, 0.0)
        self.assertEqual(cm.recall, 0.0)
        self.assertEqual(cm.f1_score, 0.0)
    
    def test_to_dict(self):
        """Test serialization to dictionary"""
        cm = ConfusionMatrix(
            true_positives=5,
            false_positives=2,
            true_negatives=90,
            false_negatives=3
        )
        d = cm.to_dict()
        self.assertIn('precision', d)
        self.assertIn('recall', d)
        self.assertIn('f1_score', d)
        self.assertIn('accuracy', d)


class TestComponentValidation(unittest.TestCase):
    """Tests for ComponentValidation class"""
    
    def test_to_dict(self):
        """Test serialization to dictionary"""
        cv = ComponentValidation(
            component_id='A1',
            component_type='Application',
            predicted_score=0.75,
            predicted_rank=2,
            predicted_level='high',
            actual_impact=0.80,
            actual_rank=1,
            actual_level='critical',
            rank_difference=1,
            score_difference=0.05,
            correctly_classified=False
        )
        d = cv.to_dict()
        
        self.assertEqual(d['component_id'], 'A1')
        self.assertEqual(d['predicted']['score'], 0.75)
        self.assertEqual(d['actual']['impact'], 0.80)
        self.assertEqual(d['rank_difference'], 1)


class TestGraphValidator(unittest.TestCase):
    """Tests for GraphValidator class"""
    
    def setUp(self):
        self.analyzer = create_test_analyzer()
        self.validator = GraphValidator(self.analyzer, seed=42)
    
    def test_init_default(self):
        """Test default initialization"""
        validator = GraphValidator(self.analyzer)
        self.assertEqual(validator.critical_threshold, 0.5)
        self.assertIn('spearman_correlation', validator.targets)
    
    def test_init_custom_targets(self):
        """Test custom target thresholds"""
        targets = {'spearman_correlation': 0.8, 'f1_score': 0.95}
        validator = GraphValidator(self.analyzer, targets=targets)
        self.assertEqual(validator.targets['spearman_correlation'], 0.8)
        self.assertEqual(validator.targets['f1_score'], 0.95)
    
    def test_validate_returns_result(self):
        """Test that validate returns ValidationResult"""
        result = self.validator.validate()
        self.assertIsInstance(result, ValidationResult)
    
    def test_validate_result_structure(self):
        """Test validation result has correct structure"""
        result = self.validator.validate()
        
        self.assertIsInstance(result.timestamp, datetime)
        self.assertGreater(result.total_components, 0)
        self.assertIsInstance(result.spearman_correlation, float)
        self.assertIsInstance(result.pearson_correlation, float)
        self.assertIsInstance(result.confusion_matrix, ConfusionMatrix)
        self.assertIsInstance(result.component_validations, list)
        self.assertIsInstance(result.status, ValidationStatus)
    
    def test_validate_correlation_range(self):
        """Test correlation values are in valid range"""
        result = self.validator.validate()
        
        self.assertGreaterEqual(result.spearman_correlation, -1.0)
        self.assertLessEqual(result.spearman_correlation, 1.0)
        self.assertGreaterEqual(result.pearson_correlation, -1.0)
        self.assertLessEqual(result.pearson_correlation, 1.0)
    
    def test_validate_metrics_range(self):
        """Test all metrics are in valid ranges"""
        result = self.validator.validate()
        
        # Confusion matrix metrics should be between 0 and 1
        self.assertGreaterEqual(result.confusion_matrix.precision, 0.0)
        self.assertLessEqual(result.confusion_matrix.precision, 1.0)
        self.assertGreaterEqual(result.confusion_matrix.recall, 0.0)
        self.assertLessEqual(result.confusion_matrix.recall, 1.0)
        self.assertGreaterEqual(result.confusion_matrix.f1_score, 0.0)
        self.assertLessEqual(result.confusion_matrix.f1_score, 1.0)
    
    def test_validate_with_component_types(self):
        """Test validation with component type filter"""
        result = self.validator.validate(component_types=['Application'])
        
        # All validated components should be Applications
        for cv in result.component_validations:
            self.assertEqual(cv.component_type, 'Application')
    
    def test_validate_top_k_overlap(self):
        """Test top-k overlap calculation"""
        result = self.validator.validate()
        
        # Should have some top-k values
        self.assertGreater(len(result.top_k_overlap), 0)
        
        # Values should be between 0 and 1
        for k, overlap in result.top_k_overlap.items():
            self.assertGreaterEqual(overlap, 0.0)
            self.assertLessEqual(overlap, 1.0)
    
    def test_status_determination(self):
        """Test validation status is determined"""
        result = self.validator.validate()
        
        self.assertIn(result.status, [
            ValidationStatus.PASSED,
            ValidationStatus.MARGINAL,
            ValidationStatus.FAILED
        ])
    
    def test_component_validations_sorted(self):
        """Test component validations are sorted by actual impact"""
        result = self.validator.validate()
        
        if len(result.component_validations) > 1:
            impacts = [cv.actual_impact for cv in result.component_validations]
            self.assertEqual(impacts, sorted(impacts, reverse=True))


class TestValidatorReporting(unittest.TestCase):
    """Tests for validator reporting functions"""
    
    def setUp(self):
        self.analyzer = create_test_analyzer()
        self.validator = GraphValidator(self.analyzer, seed=42)
        self.validator.validate()  # Run validation first
    
    def test_generate_report(self):
        """Test report generation"""
        report = self.validator.generate_report()
        
        self.assertIn('summary', report)
        self.assertIn('correlation', report)
        self.assertIn('classification', report)
        self.assertIn('ranking', report)
        self.assertIn('recommendations', report)
    
    def test_report_notable_components(self):
        """Test report includes notable components"""
        report = self.validator.generate_report()
        
        self.assertIn('notable_components', report)
        self.assertIn('most_underestimated', report['notable_components'])
        self.assertIn('most_overestimated', report['notable_components'])
    
    def test_get_misclassified(self):
        """Test getting misclassified components"""
        misclassified = self.validator.get_misclassified()
        
        self.assertIsInstance(misclassified, list)
        for cv in misclassified:
            self.assertFalse(cv.correctly_classified)
    
    def test_get_high_rank_difference(self):
        """Test getting high rank difference components"""
        high_diff = self.validator.get_high_rank_difference(threshold=2)
        
        self.assertIsInstance(high_diff, list)
        for cv in high_diff:
            self.assertGreater(cv.rank_difference, 2)


class TestValidationResultSerialization(unittest.TestCase):
    """Tests for ValidationResult serialization"""
    
    def setUp(self):
        self.analyzer = create_test_analyzer()
        self.validator = GraphValidator(self.analyzer, seed=42)
        self.result = self.validator.validate()
    
    def test_to_dict(self):
        """Test to_dict method"""
        d = self.result.to_dict()
        
        self.assertIn('timestamp', d)
        self.assertIn('total_components', d)
        self.assertIn('correlation', d)
        self.assertIn('classification', d)
        self.assertIn('ranking', d)
        self.assertIn('status', d)
    
    def test_json_serializable(self):
        """Test result can be serialized to JSON"""
        d = self.result.to_dict()
        json_str = json.dumps(d, default=str)
        self.assertIsInstance(json_str, str)
        
        # Can parse back
        parsed = json.loads(json_str)
        self.assertEqual(parsed['total_components'], self.result.total_components)
    
    def test_summary(self):
        """Test summary generation"""
        summary = self.result.summary()
        
        self.assertIsInstance(summary, str)
        self.assertIn('Spearman', summary)
        self.assertIn('Precision', summary)
        self.assertIn('Recall', summary)


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions"""
    
    def test_validate_analysis(self):
        """Test validate_analysis function"""
        analyzer = create_test_analyzer()
        result = validate_analysis(analyzer, seed=42)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertGreater(result.total_components, 0)
    
    def test_validate_analysis_with_targets(self):
        """Test validate_analysis with custom targets"""
        analyzer = create_test_analyzer()
        targets = {'spearman_correlation': 0.5}
        result = validate_analysis(analyzer, targets=targets, seed=42)
        
        self.assertIsInstance(result, ValidationResult)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases"""
    
    def test_small_system(self):
        """Test with very small system"""
        data = {
            "nodes": [{"id": "N1", "name": "Node1", "type": "compute"}],
            "brokers": [{"id": "B1", "name": "Broker1", "node": "N1"}],
            "applications": [
                {"id": "A1", "name": "App1", "role": "pub", "node": "N1"},
                {"id": "A2", "name": "App2", "role": "sub", "node": "N1"}
            ],
            "topics": [{"id": "T1", "name": "Topic1", "broker": "B1"}],
            "relationships": {
                "publishes_to": [{"from": "A1", "to": "T1"}],
                "subscribes_to": [{"from": "A2", "to": "T1"}],
                "runs_on": [
                    {"from": "A1", "to": "N1"},
                    {"from": "A2", "to": "N1"},
                    {"from": "B1", "to": "N1"}
                ],
                "routes": [{"from": "B1", "to": "T1"}]
            }
        }
        
        analyzer = GraphAnalyzer()
        analyzer.load_from_dict(data)
        
        validator = GraphValidator(analyzer, seed=42)
        result = validator.validate()
        
        self.assertIsInstance(result, ValidationResult)
    
    def test_reproducibility(self):
        """Test that results are reproducible with seed"""
        analyzer = create_test_analyzer()
        
        validator1 = GraphValidator(analyzer, seed=42)
        result1 = validator1.validate()
        
        validator2 = GraphValidator(analyzer, seed=42)
        result2 = validator2.validate()
        
        self.assertEqual(result1.spearman_correlation, result2.spearman_correlation)
        self.assertEqual(result1.total_components, result2.total_components)


class TestLevelClassification(unittest.TestCase):
    """Tests for criticality level classification"""
    
    def setUp(self):
        self.validator = GraphValidator(create_test_analyzer(), seed=42)
    
    def test_classify_critical(self):
        """Test critical classification"""
        level = self.validator._classify_level(0.75)
        self.assertEqual(level, 'critical')
    
    def test_classify_high(self):
        """Test high classification"""
        level = self.validator._classify_level(0.55)
        self.assertEqual(level, 'high')
    
    def test_classify_medium(self):
        """Test medium classification"""
        level = self.validator._classify_level(0.35)
        self.assertEqual(level, 'medium')
    
    def test_classify_low(self):
        """Test low classification"""
        level = self.validator._classify_level(0.15)
        self.assertEqual(level, 'low')
    
    def test_classify_boundaries(self):
        """Test classification at boundaries"""
        self.assertEqual(self.validator._classify_level(0.7), 'critical')
        self.assertEqual(self.validator._classify_level(0.5), 'high')
        self.assertEqual(self.validator._classify_level(0.3), 'medium')
        self.assertEqual(self.validator._classify_level(0.0), 'low')


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestStatisticalFunctions,
        TestConfusionMatrix,
        TestComponentValidation,
        TestGraphValidator,
        TestValidatorReporting,
        TestValidationResultSerialization,
        TestConvenienceFunctions,
        TestEdgeCases,
        TestLevelClassification,
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())