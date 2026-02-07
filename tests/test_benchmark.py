import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.benchmark.models import BenchmarkRecord, BenchmarkSummary, BenchmarkScenario, AggregateResult
from src.benchmark.reporting import ReportGenerator
from src.benchmark.runner import BenchmarkRunner

@pytest.fixture
def mock_output_dir(tmp_path):
    return tmp_path / "benchmark_output"

@pytest.fixture
def sample_record():
    return BenchmarkRecord(
        run_id="test_run",
        timestamp="2024-01-01T12:00:00",
        scale="small",
        layer="test",
        seed=1,
        time_total=100.0,
        spearman=0.8,
        f1_score=0.9,
        passed=True
    )

class TestBenchmarkModels:
    def test_benchmark_record_defaults(self):
        record = BenchmarkRecord(run_id="test", timestamp="now", scale="S", layer="L", seed=1)
        assert record.passed is False
        assert record.nodes == 0
        
    def test_scenario_validation(self):
        with pytest.raises(ValueError):
             BenchmarkScenario(name="Bad")

    def test_aggregate_results(self):
        # Create dummy summary
        summary = BenchmarkSummary(
             timestamp="now", duration=10, total_runs=0, passed_runs=0
        )
        assert summary.overall_pass_rate == 0.0

class TestReportGenerator:
    def test_save_json(self, mock_output_dir, sample_record):
        mock_output_dir.mkdir()
        generator = ReportGenerator(mock_output_dir)
        
        summary = BenchmarkSummary(
            timestamp="now", duration=1.0, total_runs=1, passed_runs=1,
            records=[sample_record]
        )
        
        path = generator.save_json(summary)
        assert path.exists()
        assert "benchmark_results.json" in str(path)
        
    def test_generate_markdown(self, mock_output_dir, sample_record):
        mock_output_dir.mkdir()
        generator = ReportGenerator(mock_output_dir)
        
        agg = AggregateResult(scale="small", layer="test", num_runs=1)
        agg.avg_spearman = 0.8
        
        summary = BenchmarkSummary(
            timestamp="now", duration=1.0, total_runs=1, passed_runs=1,
            records=[sample_record],
            aggregates=[agg]
        )
        
        path = generator.generate_markdown(summary)
        assert path.exists()
        assert "benchmark_report.md" in str(path)
        content = path.read_text()
        assert "small" in content
        assert "0.800" in content

class TestBenchmarkRunner:
    @patch("src.benchmark.runner.GenerationService")
    def test_generate_data_success(self, mock_gen_service, mock_output_dir):
        runner = BenchmarkRunner(mock_output_dir)
        mock_gen_service.return_value.generate.return_value = {"nodes": [], "edges": []}
        
        scenario = BenchmarkScenario(name="Test", scale="tiny")
        graph_data, time_ms = runner._generate_data(scenario, 42)
        
        assert graph_data is not None
        assert time_ms >= 0
        assert mock_gen_service.call_count == 1
        
    @patch("src.benchmark.runner.GenerationService")
    def test_generate_data_failure(self, mock_gen_service, mock_output_dir):
        runner = BenchmarkRunner(mock_output_dir)
        mock_gen_service.return_value.generate.side_effect = Exception("Generation error")
        
        scenario = BenchmarkScenario(name="Test", scale="tiny")
        graph_data, _ = runner._generate_data(scenario, 42)
        
        assert graph_data is None

    @patch("src.benchmark.runner.BenchmarkRunner._generate_data")
    @patch("src.benchmark.runner.BenchmarkRunner._import_data")
    @patch("src.benchmark.runner.BenchmarkRunner._run_analysis")
    @patch("src.benchmark.runner.BenchmarkRunner._run_simulation")
    @patch("src.benchmark.runner.BenchmarkRunner._run_validation")
    def test_run_scenario_flow(self, mock_val, mock_sim, mock_an, mock_imp, mock_gen, mock_output_dir):
        # Mock all internal methods with correct return types
        # _generate_data returns (Dict or None, float)
        mock_gen.return_value = ({"nodes": [], "edges": []}, 10.0)
        mock_imp.return_value = (True, 20.0)
        
        # Analysis mock - returns (dict, float)
        mock_an.return_value = ({"graph_summary": {"nodes": 10, "edges": 20, "density": 0.5}, "quality_analysis": {"components": []}}, 5.0)
        # Simulation mock - returns (list of dicts, float)
        mock_sim.return_value = ([{"target_id": "app1", "impact": {"composite_impact": 0.5}}], 5.0)
        
        # Validation mock - returns (ValidationResult-like object, float)
        mock_val_result = MagicMock()
        mock_val_result.overall.correlation.spearman = 0.85
        mock_val_result.overall.classification.f1_score = 0.90
        mock_val_result.overall.classification.precision = 0.88
        mock_val_result.overall.classification.recall = 0.92
        mock_val_result.overall.ranking.top_5_overlap = 0.60
        mock_val_result.overall.ranking.top_10_overlap = 0.70
        mock_val_result.overall.error.rmse = 0.15
        mock_val_result.passed = True
        mock_val.return_value = (mock_val_result, 5.0)
        
        runner = BenchmarkRunner(mock_output_dir)
        scenario = BenchmarkScenario(name="Test", scale="tiny", layers=["app"], runs=1)
        
        records = runner.run_scenario(scenario)
        
        assert len(records) == 1
        assert records[0].passed is True
        assert records[0].time_analysis == 5.0
