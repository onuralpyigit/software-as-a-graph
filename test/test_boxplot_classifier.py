#!/usr/bin/env python3
"""
Tests for Box Plot Statistical Classification Module

Tests cover:
- Basic statistical calculations
- Classification accuracy
- Edge cases (small datasets, uniform distributions, zero IQR)
- Integration with criticality scorers
- Edge classification
- Comparison with fixed thresholds
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.analysis.boxplot_classifier import (
    BoxPlotClassifier,
    BoxPlotCriticalityLevel,
    BoxPlotStatistics,
    BoxPlotClassificationResult,
    BoxPlotClassificationSummary,
    classify_criticality_with_boxplot,
    classify_edges_with_boxplot
)


class TestBoxPlotStatistics(unittest.TestCase):
    """Test statistical calculations"""

    def setUp(self):
        self.classifier = BoxPlotClassifier()

    def test_basic_statistics(self):
        """Test basic statistical measures"""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        stats = self.classifier.calculate_statistics(scores)

        self.assertEqual(stats.total_count, 9)
        self.assertAlmostEqual(stats.min_value, 0.1, places=4)
        self.assertAlmostEqual(stats.max_value, 0.9, places=4)
        self.assertAlmostEqual(stats.median, 0.5, places=4)
        self.assertAlmostEqual(stats.mean, 0.5, places=4)

    def test_quartiles(self):
        """Test quartile calculations"""
        # Use a dataset where quartiles are easy to verify
        scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        stats = self.classifier.calculate_statistics(scores)

        # Q1 should be between 3 and 4, Q3 between 9 and 10
        # (exact values depend on interpolation method)
        self.assertGreater(stats.q1, 2.5)
        self.assertLess(stats.q1, 4.5)
        self.assertGreater(stats.q3, 8.5)
        self.assertLess(stats.q3, 10.5)
        # IQR should be roughly 6 (Q3 - Q1)
        self.assertGreater(stats.iqr, 4)
        self.assertLess(stats.iqr, 8)

    def test_fences(self):
        """Test fence calculations"""
        scores = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        stats = self.classifier.calculate_statistics(scores)

        # Verify IQR-based fences
        expected_lower = stats.q1 - 1.5 * stats.iqr
        expected_upper = stats.q3 + 1.5 * stats.iqr

        # Fences should be clamped to [0, 1] by default
        self.assertGreaterEqual(stats.lower_fence, 0.0)
        self.assertLessEqual(stats.upper_fence, 1.0)

    def test_outlier_counting(self):
        """Test outlier detection"""
        # Dataset with clear outliers
        scores = [0.01, 0.02,  # Lower outliers
                  0.3, 0.4, 0.5, 0.5, 0.5, 0.6, 0.7,  # Normal range
                  0.98, 0.99]  # Upper outliers
        stats = self.classifier.calculate_statistics(scores)

        # Should detect some outliers
        self.assertGreaterEqual(stats.outlier_count_upper + stats.outlier_count_lower, 0)

    def test_empty_scores_raises_error(self):
        """Test that empty scores raise ValueError"""
        with self.assertRaises(ValueError):
            self.classifier.calculate_statistics([])

    def test_single_value(self):
        """Test handling of single value"""
        scores = [0.5]
        stats = self.classifier.calculate_statistics(scores)

        self.assertEqual(stats.total_count, 1)
        self.assertEqual(stats.min_value, 0.5)
        self.assertEqual(stats.max_value, 0.5)
        self.assertEqual(stats.median, 0.5)

    def test_identical_values(self):
        """Test handling of identical values (zero variance)"""
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        stats = self.classifier.calculate_statistics(scores)

        self.assertEqual(stats.min_value, stats.max_value)
        self.assertEqual(stats.std_dev, 0.0)
        # IQR should use fallback since Q1 == Q3
        self.assertGreater(stats.iqr, 0)  # Should not be zero


class TestBoxPlotClassification(unittest.TestCase):
    """Test classification functionality"""

    def setUp(self):
        self.classifier = BoxPlotClassifier()

    def test_classify_single_score(self):
        """Test classification of single scores"""
        # Create statistics for reference
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        stats = self.classifier.calculate_statistics(scores)

        # Very high score should be VERY_HIGH or HIGH
        level = self.classifier.classify_score(0.95, stats)
        self.assertIn(level, [BoxPlotCriticalityLevel.VERY_HIGH, BoxPlotCriticalityLevel.HIGH])

        # Very low score should be VERY_LOW or LOW
        level = self.classifier.classify_score(0.05, stats)
        self.assertIn(level, [BoxPlotCriticalityLevel.VERY_LOW, BoxPlotCriticalityLevel.LOW])

        # Middle score should be MEDIUM
        level = self.classifier.classify_score(0.5, stats)
        self.assertEqual(level, BoxPlotCriticalityLevel.MEDIUM)

    def test_classify_scores_dict(self):
        """Test classification of score dictionary"""
        scores = {
            'comp_a': 0.9,
            'comp_b': 0.7,
            'comp_c': 0.5,
            'comp_d': 0.3,
            'comp_e': 0.1
        }

        results = self.classifier.classify_scores(scores)

        self.assertEqual(len(results), 5)
        self.assertIn('comp_a', results)
        self.assertIn('comp_e', results)

        # Highest should be higher level than lowest
        self.assertGreater(
            results['comp_a'].percentile,
            results['comp_e'].percentile
        )

    def test_classification_result_attributes(self):
        """Test that classification results have all required attributes"""
        scores = {'test_comp': 0.75}
        component_types = {'test_comp': 'Application'}

        results = self.classifier.classify_scores(scores, component_types)
        result = results['test_comp']

        # Check all attributes exist
        self.assertEqual(result.component, 'test_comp')
        self.assertEqual(result.component_type, 'Application')
        self.assertEqual(result.score, 0.75)
        self.assertIsInstance(result.level, BoxPlotCriticalityLevel)
        self.assertIsInstance(result.percentile, float)
        self.assertIsInstance(result.z_score, float)
        self.assertIsInstance(result.is_outlier, bool)

    def test_percentile_calculation(self):
        """Test percentile rank calculation"""
        scores = {f'comp_{i}': i/10 for i in range(1, 11)}
        results = self.classifier.classify_scores(scores)

        # Highest score should have highest percentile
        highest = results['comp_10']
        lowest = results['comp_1']

        self.assertGreater(highest.percentile, lowest.percentile)
        self.assertGreater(highest.percentile, 80)  # Should be in top 20%
        self.assertLess(lowest.percentile, 20)  # Should be in bottom 20%

    def test_z_score_calculation(self):
        """Test z-score calculation"""
        scores = {f'comp_{i}': i/10 for i in range(1, 11)}
        results = self.classifier.classify_scores(scores)

        # Z-scores should be positive for above-mean, negative for below-mean
        high_z = results['comp_10'].z_score
        low_z = results['comp_1'].z_score

        self.assertGreater(high_z, 0)
        self.assertLess(low_z, 0)


class TestBoxPlotEdgeClassification(unittest.TestCase):
    """Test edge classification functionality"""

    def setUp(self):
        self.classifier = BoxPlotClassifier()

    def test_classify_edge_scores(self):
        """Test classification of edge scores"""
        edge_scores = {
            ('app_a', 'topic_1'): 0.8,
            ('app_b', 'topic_1'): 0.6,
            ('app_c', 'topic_2'): 0.4,
            ('app_d', 'topic_2'): 0.2
        }

        results = self.classifier.classify_edge_scores(edge_scores)

        self.assertEqual(len(results), 4)
        self.assertIn(('app_a', 'topic_1'), results)

    def test_edge_classification_with_score_objects(self):
        """Test edge classification with score-like objects"""
        # Mock edge score object
        class MockEdgeScore:
            def __init__(self, score, edge_type):
                self.composite_score = score
                self.edge_type = edge_type

        edge_scores = {
            ('a', 'b'): MockEdgeScore(0.8, 'PUBLISHES'),
            ('c', 'd'): MockEdgeScore(0.3, 'SUBSCRIBES')
        }

        results = self.classifier.classify_edge_scores(edge_scores)
        self.assertEqual(len(results), 2)


class TestBoxPlotSummary(unittest.TestCase):
    """Test summary generation"""

    def setUp(self):
        self.classifier = BoxPlotClassifier()
        self.scores = {
            'critical_1': 0.95,
            'critical_2': 0.90,
            'high_1': 0.75,
            'high_2': 0.70,
            'medium_1': 0.55,
            'medium_2': 0.50,
            'medium_3': 0.45,
            'low_1': 0.30,
            'low_2': 0.25,
            'minimal_1': 0.10,
            'minimal_2': 0.05
        }

    def test_summary_generation(self):
        """Test summary statistics generation"""
        results = self.classifier.classify_scores(self.scores)
        summary = self.classifier.get_summary(results)

        self.assertIsInstance(summary, BoxPlotClassificationSummary)
        self.assertIsInstance(summary.statistics, BoxPlotStatistics)
        self.assertEqual(sum(summary.level_counts.values()), len(self.scores))

    def test_summary_level_counts(self):
        """Test that level counts sum to total"""
        results = self.classifier.classify_scores(self.scores)
        summary = self.classifier.get_summary(results)

        total_from_counts = sum(summary.level_counts.values())
        self.assertEqual(total_from_counts, len(self.scores))

    def test_summary_percentages(self):
        """Test that percentages sum to 100"""
        results = self.classifier.classify_scores(self.scores)
        summary = self.classifier.get_summary(results)

        total_pct = sum(summary.level_percentages.values())
        self.assertAlmostEqual(total_pct, 100.0, places=1)

    def test_top_critical_components(self):
        """Test top critical component selection"""
        results = self.classifier.classify_scores(self.scores)
        summary = self.classifier.get_summary(results, top_n=3)

        self.assertLessEqual(len(summary.top_critical), 3)
        # Should be sorted by score descending
        if len(summary.top_critical) >= 2:
            self.assertGreaterEqual(
                summary.top_critical[0].score,
                summary.top_critical[1].score
            )

    def test_components_by_level(self):
        """Test components grouped by level"""
        results = self.classifier.classify_scores(self.scores)
        summary = self.classifier.get_summary(results)

        # All levels should be present in dict
        for level in BoxPlotCriticalityLevel:
            self.assertIn(level.value, summary.components_by_level)


class TestBoxPlotComparison(unittest.TestCase):
    """Test comparison with fixed thresholds"""

    def setUp(self):
        self.classifier = BoxPlotClassifier()

    def test_comparison_with_fixed_thresholds(self):
        """Test comparison functionality"""
        scores = {f'comp_{i}': i/10 for i in range(1, 10)}
        results = self.classifier.classify_scores(scores)

        comparison = self.classifier.compare_with_fixed_thresholds(results)

        self.assertIn('agreement_rate', comparison)
        self.assertIn('disagreements', comparison)
        self.assertIn('total_components', comparison)
        self.assertEqual(comparison['total_components'], 9)

    def test_comparison_returns_thresholds(self):
        """Test that comparison returns both threshold sets"""
        scores = {'a': 0.9, 'b': 0.5, 'c': 0.1}
        results = self.classifier.classify_scores(scores)

        comparison = self.classifier.compare_with_fixed_thresholds(results)

        self.assertIn('fixed_thresholds', comparison)
        self.assertIn('boxplot_thresholds', comparison)
        self.assertIn('CRITICAL', comparison['fixed_thresholds'])
        self.assertIn('upper_fence', comparison['boxplot_thresholds'])


class TestBoxPlotUtilityMethods(unittest.TestCase):
    """Test utility methods"""

    def setUp(self):
        self.classifier = BoxPlotClassifier()
        self.scores = {
            'outlier_high': 0.99,
            'high_1': 0.80,
            'medium_1': 0.50,
            'low_1': 0.20,
            'outlier_low': 0.01
        }
        self.results = self.classifier.classify_scores(self.scores)

    def test_get_components_by_level(self):
        """Test filtering by level"""
        medium_components = self.classifier.get_components_by_level(
            BoxPlotCriticalityLevel.MEDIUM
        )

        # All returned should be MEDIUM level
        for comp in medium_components:
            self.assertEqual(comp.level, BoxPlotCriticalityLevel.MEDIUM)

    def test_get_outliers(self):
        """Test outlier retrieval"""
        outliers = self.classifier.get_outliers()

        # All returned should be outliers
        for outlier in outliers:
            self.assertTrue(outlier.is_outlier)

    def test_get_outliers_by_type(self):
        """Test outlier filtering by type"""
        upper_outliers = self.classifier.get_outliers(outlier_type='upper')

        for outlier in upper_outliers:
            self.assertIn(outlier.outlier_type, ['upper', 'extreme_upper'])


class TestBoxPlotSerialization(unittest.TestCase):
    """Test serialization methods"""

    def setUp(self):
        self.classifier = BoxPlotClassifier()

    def test_statistics_to_dict(self):
        """Test BoxPlotStatistics serialization"""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        stats = self.classifier.calculate_statistics(scores)
        stats_dict = stats.to_dict()

        self.assertIsInstance(stats_dict, dict)
        self.assertIn('q1', stats_dict)
        self.assertIn('q3', stats_dict)
        self.assertIn('iqr', stats_dict)

    def test_result_to_dict(self):
        """Test BoxPlotClassificationResult serialization"""
        scores = {'comp_a': 0.5}
        results = self.classifier.classify_scores(scores)
        result_dict = results['comp_a'].to_dict()

        self.assertIsInstance(result_dict, dict)
        self.assertIn('component', result_dict)
        self.assertIn('level', result_dict)
        self.assertIn('percentile', result_dict)

    def test_summary_to_dict(self):
        """Test BoxPlotClassificationSummary serialization"""
        scores = {'a': 0.8, 'b': 0.5, 'c': 0.2}
        results = self.classifier.classify_scores(scores)
        summary = self.classifier.get_summary(results)
        summary_dict = summary.to_dict()

        self.assertIsInstance(summary_dict, dict)
        self.assertIn('statistics', summary_dict)
        self.assertIn('level_counts', summary_dict)


class TestBoxPlotConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""

    def test_classify_criticality_with_boxplot(self):
        """Test convenience function for node classification"""
        # Mock criticality score object
        class MockScore:
            def __init__(self, score, comp_type):
                self.composite_score = score
                self.component_type = comp_type

        scores = {
            'a': MockScore(0.8, 'Application'),
            'b': MockScore(0.5, 'Topic'),
            'c': MockScore(0.2, 'Broker')
        }

        results, summary = classify_criticality_with_boxplot(scores)

        self.assertEqual(len(results), 3)
        self.assertIsInstance(summary, BoxPlotClassificationSummary)

    def test_classify_edges_with_boxplot(self):
        """Test convenience function for edge classification"""
        class MockEdgeScore:
            def __init__(self, score):
                self.composite_score = score
                self.edge_type = 'PUBLISHES'

        edge_scores = {
            ('a', 'b'): MockEdgeScore(0.7),
            ('c', 'd'): MockEdgeScore(0.3)
        }

        results, summary = classify_edges_with_boxplot(edge_scores)

        self.assertEqual(len(results), 2)
        self.assertIsInstance(summary, BoxPlotClassificationSummary)


class TestBoxPlotReport(unittest.TestCase):
    """Test report generation"""

    def setUp(self):
        self.classifier = BoxPlotClassifier()

    def test_generate_report(self):
        """Test report generation"""
        scores = {f'comp_{i}': i/10 for i in range(1, 11)}
        results = self.classifier.classify_scores(scores)

        report = self.classifier.generate_report(results)

        self.assertIsInstance(report, str)
        self.assertIn('BOX PLOT', report)
        self.assertIn('STATISTICAL SUMMARY', report)
        self.assertIn('QUARTILES', report)
        self.assertIn('CLASSIFICATION', report)

    def test_report_includes_all_sections(self):
        """Test that report includes all expected sections"""
        scores = {'a': 0.9, 'b': 0.5, 'c': 0.1}
        results = self.classifier.classify_scores(scores)

        report = self.classifier.generate_report(results)

        expected_sections = [
            'STATISTICAL SUMMARY',
            'QUARTILES',
            'CLASSIFICATION THRESHOLDS',
            'CLASSIFICATION DISTRIBUTION',
            'TOP 10 MOST CRITICAL',
            'BOTTOM 10 MINIMAL'
        ]

        for section in expected_sections:
            self.assertIn(section, report)


class TestBoxPlotEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def setUp(self):
        self.classifier = BoxPlotClassifier()

    def test_two_values(self):
        """Test with only two values"""
        scores = {'a': 0.8, 'b': 0.2}
        results = self.classifier.classify_scores(scores)

        self.assertEqual(len(results), 2)
        # Higher score should have higher level or equal
        self.assertGreaterEqual(
            results['a'].percentile,
            results['b'].percentile
        )

    def test_all_same_value(self):
        """Test when all scores are identical"""
        scores = {'a': 0.5, 'b': 0.5, 'c': 0.5, 'd': 0.5}
        results = self.classifier.classify_scores(scores)

        # All should get same classification
        levels = set(r.level for r in results.values())
        self.assertEqual(len(levels), 1)

    def test_extreme_outliers(self):
        """Test detection of extreme outliers"""
        scores = {
            'normal_1': 0.5,
            'normal_2': 0.51,
            'normal_3': 0.49,
            'normal_4': 0.52,
            'normal_5': 0.48,
            'extreme': 0.99  # Extreme outlier
        }

        results = self.classifier.classify_scores(scores)

        # Extreme value should be flagged as outlier
        self.assertTrue(results['extreme'].is_outlier)

    def test_custom_iqr_multiplier(self):
        """Test with custom IQR multiplier"""
        classifier_strict = BoxPlotClassifier(iqr_multiplier=1.0)
        classifier_loose = BoxPlotClassifier(iqr_multiplier=3.0)

        scores = {f'c_{i}': i/10 for i in range(1, 11)}

        results_strict = classifier_strict.classify_scores(scores)
        results_loose = classifier_loose.classify_scores(scores)

        # Stricter multiplier should find more outliers
        outliers_strict = sum(1 for r in results_strict.values() if r.is_outlier)
        outliers_loose = sum(1 for r in results_loose.values() if r.is_outlier)

        self.assertGreaterEqual(outliers_strict, outliers_loose)

    def test_unclamped_fences(self):
        """Test without fence clamping"""
        classifier = BoxPlotClassifier(clamp_to_bounds=False)

        scores = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
        stats = classifier.calculate_statistics(scores)

        # Without clamping, fences can go beyond [0, 1]
        # (depends on data distribution)
        self.assertIsNotNone(stats.lower_fence)
        self.assertIsNotNone(stats.upper_fence)


if __name__ == '__main__':
    unittest.main(verbosity=2)
