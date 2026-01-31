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
    @patch("subprocess.run")
    def test_generate_data_success(self, mock_run, mock_output_dir):
        runner = BenchmarkRunner(mock_output_dir)
        mock_run.return_value = MagicMock(returncode=0)
        
        scenario = BenchmarkScenario(name="Test", scale="tiny")
        success, time_ms = runner._generate_data(scenario, 42, Path("out.json"))
        
        assert success is True
        assert time_ms >= 0
        assert mock_run.call_count == 1
        
    @patch("subprocess.run")
    def test_generate_data_failure(self, mock_run, mock_output_dir):
        runner = BenchmarkRunner(mock_output_dir)
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        
        scenario = BenchmarkScenario(name="Test", scale="tiny")
        success, _ = runner._generate_data(scenario, 42, Path("out.json"))
        
        assert success is False

    @patch("src.benchmark.runner.BenchmarkRunner._generate_data")
    @patch("src.benchmark.runner.BenchmarkRunner._import_data")
    @patch("src.benchmark.runner.BenchmarkRunner._run_analysis")
    @patch("src.benchmark.runner.BenchmarkRunner._run_simulation")
    @patch("src.benchmark.runner.BenchmarkRunner._run_validation")
    def test_run_scenario_flow(self, mock_val, mock_sim, mock_an, mock_imp, mock_gen, mock_output_dir):
        # Mock all subprocess wrappers
        mock_gen.return_value = (True, 10.0)
        mock_imp.return_value = (True, 20.0)
        
        # Analysis mock
        mock_an.return_value = ({"graph_summary": {"nodes": 10}}, 5.0)
        # Simulation mock
        mock_sim.return_value = ({"results": "ok"}, 5.0)
        # Validation mock
        mock_val.return_value = ({"validation_result": {"overall": {"metrics": {}}}, "summary": {"passed": True}}, 5.0)
        
        runner = BenchmarkRunner(mock_output_dir)
        scenario = BenchmarkScenario(name="Test", scale="tiny", layers=["app"], runs=1)
        
        records = runner.run_scenario(scenario)
        
        assert len(records) == 1
        assert records[0].passed is True
        assert records[0].time_analysis == 5.0
