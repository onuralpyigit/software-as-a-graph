"""
Tests for src.validation module
================================

Tests statistical metrics, Validator, and ValidationPipeline.
"""

import pytest
import math


class TestStatisticalMetrics:
    """Tests for statistical metric functions"""

    def test_spearman_correlation(self):
        """Test Spearman correlation calculation"""
        from src.validation.metrics import spearman
        
        # Perfect correlation
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        rho, p = spearman(x, y)
        
        assert abs(rho - 1.0) < 0.01
        assert p < 0.05

    def test_spearman_negative(self):
        """Test negative Spearman correlation"""
        from src.validation.metrics import spearman
        
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        rho, p = spearman(x, y)
        
        assert abs(rho - (-1.0)) < 0.01

    def test_spearman_no_correlation(self):
        """Test no correlation"""
        from src.validation.metrics import spearman
        
        x = [1, 2, 3, 4, 5]
        y = [3, 1, 4, 2, 5]  # Random-ish
        rho, p = spearman(x, y)
        
        assert abs(rho) < 0.9  # Not perfect

    def test_pearson_correlation(self):
        """Test Pearson correlation calculation"""
        from src.validation.metrics import pearson
        
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        r, p = pearson(x, y)
        
        assert abs(r - 1.0) < 0.01

    def test_kendall_tau(self):
        """Test Kendall tau calculation"""
        from src.validation.metrics import kendall
        
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        tau = kendall(x, y)
        
        assert abs(tau - 1.0) < 0.01

    def test_percentile(self):
        """Test percentile calculation"""
        from src.validation.metrics import percentile
        
        values = list(range(1, 101))  # 1-100
        
        assert percentile(values, 50) == 50
        assert percentile(values, 0) == 1
        assert percentile(values, 100) == 100

    def test_std_dev(self):
        """Test standard deviation"""
        from src.validation.metrics import std_dev
        
        values = [2, 4, 4, 4, 5, 5, 7, 9]
        sd = std_dev(values)
        
        assert 1.5 < sd < 2.5  # Known to be ~2


class TestConfusionMatrix:
    """Tests for confusion matrix calculations"""

    def test_confusion_matrix(self):
        """Test confusion matrix calculation"""
        from src.validation.metrics import calculate_confusion
        
        predicted = [1, 1, 0, 0, 1]
        actual = [1, 0, 0, 1, 1]
        
        matrix = calculate_confusion(predicted, actual)
        
        assert matrix.tp + matrix.tn + matrix.fp + matrix.fn == 5
        assert matrix.tp == 2  # Both 1s at positions 0, 4
        assert matrix.tn == 1  # Both 0s at position 2

    def test_precision_recall_f1(self):
        """Test precision, recall, F1 calculation"""
        from src.validation.metrics import calculate_confusion
        
        # Perfect classifier
        predicted = [1, 1, 0, 0]
        actual = [1, 1, 0, 0]
        
        matrix = calculate_confusion(predicted, actual)
        
        assert matrix.precision == 1.0
        assert matrix.recall == 1.0
        assert matrix.f1 == 1.0

    def test_accuracy(self):
        """Test accuracy calculation"""
        from src.validation.metrics import calculate_confusion
        
        predicted = [1, 1, 0, 0, 1]
        actual = [1, 1, 0, 0, 0]  # One wrong
        
        matrix = calculate_confusion(predicted, actual)
        
        assert matrix.accuracy == 0.8


class TestRankingMetrics:
    """Tests for ranking metric calculations"""

    def test_top_k_overlap(self):
        """Test top-k overlap calculation"""
        from src.validation.metrics import calculate_ranking
        
        predicted = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6, "E": 0.5}
        actual = {"A": 0.9, "B": 0.8, "C": 0.6, "D": 0.7, "E": 0.5}
        
        ranking = calculate_ranking(predicted, actual, k_values=[3, 5])
        
        assert ranking.top_k_overlap[3] >= 0.5  # At least some overlap
        assert ranking.top_k_overlap[5] == 1.0  # All 5 match

    def test_ndcg(self):
        """Test NDCG calculation"""
        from src.validation.metrics import calculate_ranking
        
        # Perfect ranking
        predicted = {"A": 0.9, "B": 0.7, "C": 0.5}
        actual = {"A": 0.9, "B": 0.7, "C": 0.5}
        
        ranking = calculate_ranking(predicted, actual, k_values=[3])
        
        assert ranking.ndcg > 0.9  # Should be close to 1

    def test_mrr(self):
        """Test MRR calculation"""
        from src.validation.metrics import calculate_ranking
        
        predicted = {"A": 0.9, "B": 0.7, "C": 0.5}
        actual = {"A": 0.9, "B": 0.5, "C": 0.7}
        
        ranking = calculate_ranking(predicted, actual, k_values=[3])
        
        assert 0 <= ranking.mrr <= 1


class TestValidator:
    """Tests for Validator class"""

    def test_create_validator(self):
        """Create validator instance"""
        from src.validation import Validator
        
        validator = Validator()
        
        assert validator is not None

    def test_validate_perfect(self):
        """Validate with perfect predictions"""
        from src.validation import Validator
        
        predicted = {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3, "E": 0.1}
        actual = {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3, "E": 0.1}
        
        validator = Validator()
        result = validator.validate(predicted, actual)
        
        assert result.correlation.spearman > 0.99
        assert result.classification.f1 == 1.0

    def test_validate_partial(self):
        """Validate with partial agreement"""
        from src.validation import Validator
        
        predicted = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.4, "E": 0.1}
        actual = {"A": 0.8, "B": 0.9, "C": 0.6, "D": 0.3, "E": 0.2}
        
        validator = Validator()
        result = validator.validate(predicted, actual)
        
        assert result.correlation.spearman > 0.5
        assert result.classification.f1 > 0.5

    def test_validation_result_to_dict(self):
        """Convert validation result to dictionary"""
        from src.validation import Validator
        
        predicted = {"A": 0.9, "B": 0.7, "C": 0.5}
        actual = {"A": 0.9, "B": 0.7, "C": 0.5}
        
        validator = Validator()
        result = validator.validate(predicted, actual)
        result_dict = result.to_dict()
        
        assert "correlation" in result_dict
        assert "classification" in result_dict
        assert "ranking" in result_dict

    def test_validation_status(self):
        """Check validation status determination"""
        from src.validation import Validator, ValidationStatus
        
        # Perfect predictions
        predicted = {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3, "E": 0.1}
        actual = {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3, "E": 0.1}
        
        validator = Validator()
        result = validator.validate(predicted, actual)
        
        # Should pass with perfect data
        assert result.status in [ValidationStatus.PASSED, ValidationStatus.PARTIAL]


class TestGraphAnalyzer:
    """Tests for GraphAnalyzer class"""

    def test_create_analyzer(self, medium_graph):
        """Create graph analyzer"""
        from src.validation import GraphAnalyzer
        
        analyzer = GraphAnalyzer(medium_graph)
        
        assert analyzer is not None

    def test_degree_centrality(self, medium_graph):
        """Calculate degree centrality"""
        from src.validation import GraphAnalyzer
        
        analyzer = GraphAnalyzer(medium_graph)
        scores = analyzer.degree_centrality()
        
        assert len(scores) > 0
        assert all(0 <= v <= 1 for v in scores.values())

    def test_betweenness_centrality(self, medium_graph):
        """Calculate betweenness centrality"""
        from src.validation import GraphAnalyzer
        
        analyzer = GraphAnalyzer(medium_graph)
        scores = analyzer.betweenness_centrality()
        
        assert len(scores) > 0
        assert all(0 <= v <= 1 for v in scores.values())

    def test_message_path_centrality(self, medium_graph):
        """Calculate message path centrality"""
        from src.validation import GraphAnalyzer
        
        analyzer = GraphAnalyzer(medium_graph)
        scores = analyzer.message_path_centrality()
        
        assert len(scores) > 0

    def test_composite_score(self, medium_graph):
        """Calculate composite score"""
        from src.validation import GraphAnalyzer
        
        analyzer = GraphAnalyzer(medium_graph)
        scores = analyzer.composite_score()
        
        assert len(scores) == len(medium_graph.components)
        assert all(0 <= v <= 1 for v in scores.values())

    def test_analyze_all(self, medium_graph):
        """Run all analyses"""
        from src.validation import GraphAnalyzer
        
        analyzer = GraphAnalyzer(medium_graph)
        results = analyzer.analyze_all()
        
        assert "degree" in results
        assert "betweenness" in results
        assert "message_path" in results
        assert "composite" in results


class TestValidationPipeline:
    """Tests for ValidationPipeline class"""

    def test_create_pipeline(self):
        """Create validation pipeline"""
        from src.validation import ValidationPipeline
        
        pipeline = ValidationPipeline(seed=42)
        
        assert pipeline is not None

    def test_run_pipeline(self, medium_graph):
        """Run validation pipeline"""
        from src.validation import ValidationPipeline
        
        pipeline = ValidationPipeline(seed=42)
        result = pipeline.run(medium_graph)
        
        assert result is not None
        assert result.validation is not None
        assert result.predicted_scores is not None

    def test_pipeline_timing(self, medium_graph):
        """Check pipeline timing"""
        from src.validation import ValidationPipeline
        
        pipeline = ValidationPipeline(seed=42)
        result = pipeline.run(medium_graph)
        
        assert result.timing["total_ms"] > 0

    @pytest.mark.slow
    def test_compare_methods(self, medium_graph):
        """Compare analysis methods"""
        from src.validation import ValidationPipeline
        
        pipeline = ValidationPipeline(seed=42)
        results = pipeline.compare_methods(medium_graph)
        
        assert "composite" in results
        assert "betweenness" in results
        assert "degree" in results

    def test_pipeline_result_to_dict(self, medium_graph):
        """Convert pipeline result to dictionary"""
        from src.validation import ValidationPipeline
        
        pipeline = ValidationPipeline(seed=42)
        result = pipeline.run(medium_graph)
        result_dict = result.to_dict()
        
        assert "graph" in result_dict
        assert "validation" in result_dict
        assert "timing" in result_dict
