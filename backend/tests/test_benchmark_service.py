"""
Tests for the benchmark package (models, runner, reporting) and bin/benchmark.py CLI.
"""
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "bin"))

from src.benchmark.models import (
    AggregateResult,
    BenchmarkRecord,
    BenchmarkScenario,
    BenchmarkSummary,
)
from src.benchmark.reporting import ReportGenerator
from src.benchmark.runner import BenchmarkRunner


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path / "benchmark_output"


@pytest.fixture
def sample_record():
    return BenchmarkRecord(
        run_id="test_run",
        timestamp="2024-01-01T12:00:00",
        scale="small",
        layer="app",
        seed=42,
        time_total=100.0,
        spearman=0.85,
        f1_score=0.90,
        passed=True,
    )


@pytest.fixture
def make_records():
    """Factory that builds N records with slightly varying metrics."""

    def _make(n: int = 3, scale: str = "small", layer: str = "app") -> list[BenchmarkRecord]:
        records = []
        for i in range(n):
            records.append(
                BenchmarkRecord(
                    run_id=f"{scale}_{layer}_{42 + i}",
                    timestamp=f"2024-01-01T12:0{i}:00",
                    scale=scale,
                    layer=layer,
                    seed=42 + i,
                    nodes=10 + i,
                    edges=20 + i * 2,
                    density=0.5 + i * 0.01,
                    time_analysis=100.0 + i * 10,
                    time_simulation=200.0 + i * 20,
                    time_total=400.0 + i * 30,
                    spearman=0.80 + i * 0.02,
                    f1_score=0.85 + i * 0.02,
                    precision=0.82 + i * 0.02,
                    recall=0.88 + i * 0.01,
                    top5_overlap=0.50 + i * 0.05,
                    top10_overlap=0.60 + i * 0.05,
                    rmse=0.15 - i * 0.01,
                    passed=True,
                )
            )
        return records

    return _make


# =============================================================================
# Models
# =============================================================================

class TestBenchmarkModels:
    def test_record_defaults(self):
        r = BenchmarkRecord(run_id="x", timestamp="now", scale="S", layer="L", seed=1)
        assert r.passed is False
        assert r.nodes == 0
        assert r.error is None

    def test_record_to_dict(self, sample_record):
        d = sample_record.to_dict()
        assert d["run_id"] == "test_run"
        assert d["spearman"] == 0.85

    def test_scenario_requires_scale_or_config(self):
        with pytest.raises(ValueError, match="scale.*config_path"):
            BenchmarkScenario(name="Bad")

    def test_scenario_label_from_scale(self):
        s = BenchmarkScenario(name="X", scale="tiny")
        assert s.label == "tiny"

    def test_scenario_label_custom(self):
        s = BenchmarkScenario(name="X", config_path=Path("a.yaml"))
        assert s.label == "custom"

    def test_summary_defaults(self):
        s = BenchmarkSummary(timestamp="now", duration=10, total_runs=0, passed_runs=0)
        assert s.overall_pass_rate == 0.0
        assert s.records == []

    def test_summary_to_dict_roundtrip(self, sample_record):
        s = BenchmarkSummary(
            timestamp="now",
            duration=1.0,
            total_runs=1,
            passed_runs=1,
            records=[sample_record],
        )
        d = s.to_dict()
        assert d["total_runs"] == 1
        assert len(d["records"]) == 1


# =============================================================================
# Reporting
# =============================================================================

class TestReportGenerator:
    def test_save_json(self, tmp_output, sample_record):
        tmp_output.mkdir()
        gen = ReportGenerator(tmp_output)

        summary = BenchmarkSummary(
            timestamp="now", duration=1.0, total_runs=1, passed_runs=1,
            records=[sample_record],
        )
        path = gen.save_json(summary)

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["total_runs"] == 1

    def test_generate_markdown(self, tmp_output, sample_record):
        tmp_output.mkdir()
        gen = ReportGenerator(tmp_output)

        agg = AggregateResult(scale="small", layer="app", num_runs=1)
        agg.avg_spearman = 0.85
        agg.std_spearman = 0.02

        summary = BenchmarkSummary(
            timestamp="now", duration=1.0, total_runs=1, passed_runs=1,
            records=[sample_record], aggregates=[agg],
        )
        path = gen.generate_markdown(summary)

        assert path.exists()
        content = path.read_text()
        assert "small" in content
        assert "0.850" in content
        assert "±" in content  # std deviation is included

    def test_markdown_includes_errors(self, tmp_output):
        tmp_output.mkdir()
        gen = ReportGenerator(tmp_output)

        bad_record = BenchmarkRecord(
            run_id="err", timestamp="now", scale="tiny", layer="app",
            seed=1, error="Analysis failed",
        )
        summary = BenchmarkSummary(
            timestamp="now", duration=1.0, total_runs=1, passed_runs=0,
            records=[bad_record],
        )
        path = gen.generate_markdown(summary)
        content = path.read_text()
        assert "Errors" in content
        assert "Analysis failed" in content


# =============================================================================
# Runner
# =============================================================================

class TestBenchmarkRunner:
    """Tests for BenchmarkRunner using mocked internal methods."""

    @patch("src.benchmark.runner.GenerationService")
    def test_generate_data_success(self, mock_gen_cls, tmp_output):
        mock_gen_cls.return_value.generate.return_value = {"nodes": []}
        runner = BenchmarkRunner(tmp_output)
        scenario = BenchmarkScenario(name="T", scale="tiny")

        data, ms = runner._generate_data(scenario, 42)

        assert data is not None
        assert ms >= 0

    @patch("src.benchmark.runner.GenerationService")
    def test_generate_data_failure(self, mock_gen_cls, tmp_output):
        mock_gen_cls.return_value.generate.side_effect = Exception("boom")
        runner = BenchmarkRunner(tmp_output)
        scenario = BenchmarkScenario(name="T", scale="tiny")

        data, _ = runner._generate_data(scenario, 42)

        assert data is None

    @patch("src.benchmark.runner.BenchmarkRunner._generate_data")
    @patch("src.benchmark.runner.BenchmarkRunner._import_data")
    @patch("src.benchmark.runner.BenchmarkRunner._run_analysis")
    @patch("src.benchmark.runner.BenchmarkRunner._run_simulation")
    @patch("src.benchmark.runner.BenchmarkRunner._run_validation")
    def test_run_scenario_flow(
        self, mock_val, mock_sim, mock_an, mock_imp, mock_gen, tmp_output
    ):
        mock_gen.return_value = ({"nodes": []}, 10.0)
        mock_imp.return_value = (True, 20.0)
        mock_an.return_value = (
            {
                "graph_summary": {"nodes": 10, "edges": 20, "density": 0.5},
                "quality_analysis": {"components": []},
            },
            5.0,
        )
        mock_sim.return_value = (
            [{"target_id": "app1", "impact": {"composite_impact": 0.5}}],
            5.0,
        )

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

        runner = BenchmarkRunner(tmp_output)
        scenario = BenchmarkScenario(name="T", scale="tiny", layers=["app"], runs=1)
        records = runner.run_scenario(scenario)

        assert len(records) == 1
        assert records[0].passed is True
        assert records[0].spearman == 0.85
        assert records[0].time_analysis == 5.0
        assert records[0].time_total > 0

    @patch("src.benchmark.runner.BenchmarkRunner._generate_data")
    def test_run_scenario_skips_on_generation_failure(self, mock_gen, tmp_output):
        mock_gen.return_value = (None, 0.0)

        runner = BenchmarkRunner(tmp_output)
        scenario = BenchmarkScenario(name="T", scale="tiny", layers=["app"], runs=1)
        records = runner.run_scenario(scenario)

        assert len(records) == 0

    @patch("src.benchmark.runner.BenchmarkRunner._generate_data")
    @patch("src.benchmark.runner.BenchmarkRunner._import_data")
    @patch("src.benchmark.runner.BenchmarkRunner._run_analysis")
    def test_analysis_failure_records_error(
        self, mock_an, mock_imp, mock_gen, tmp_output
    ):
        mock_gen.return_value = ({"nodes": []}, 10.0)
        mock_imp.return_value = (True, 20.0)
        mock_an.return_value = (None, 0.0)

        runner = BenchmarkRunner(tmp_output)
        scenario = BenchmarkScenario(name="T", scale="tiny", layers=["app"], runs=1)
        records = runner.run_scenario(scenario)

        assert len(records) == 1
        assert records[0].error == "Analysis failed"
        assert records[0].passed is False


class TestAggregation:
    """Tests for aggregate_results and the _aggregate_group helper."""

    def test_aggregate_empty_records(self, tmp_output):
        runner = BenchmarkRunner(tmp_output)
        summary = runner.aggregate_results(1.0)

        assert summary.total_runs == 0
        assert summary.aggregates == []

    def test_aggregate_computes_mean_and_std(self, tmp_output, make_records):
        runner = BenchmarkRunner(tmp_output)
        runner.records = make_records(n=3, scale="small", layer="app")
        summary = runner.aggregate_results(1.0)

        assert summary.total_runs == 3
        assert summary.passed_runs == 3
        assert len(summary.aggregates) == 1

        agg = summary.aggregates[0]
        assert agg.scale == "small"
        assert agg.layer == "app"
        assert agg.num_runs == 3
        assert agg.pass_rate == 100.0

        # With 3 records std should be > 0
        assert agg.std_spearman > 0
        assert agg.std_f1 > 0

        # Mean of 0.80, 0.82, 0.84 = 0.82
        assert abs(agg.avg_spearman - 0.82) < 0.001

    def test_aggregate_identifies_best_worst(self, tmp_output, make_records):
        runner = BenchmarkRunner(tmp_output)
        runner.records = make_records(n=3)
        summary = runner.aggregate_results(1.0)

        assert summary.best_config is not None
        assert summary.worst_config is not None
        assert summary.best_config != summary.worst_config

    def test_aggregate_multiple_groups(self, tmp_output, make_records):
        runner = BenchmarkRunner(tmp_output)
        runner.records = make_records(2, "small", "app") + make_records(2, "medium", "infra")
        summary = runner.aggregate_results(1.0)

        assert len(summary.aggregates) == 2
        assert set(a.scale for a in summary.aggregates) == {"small", "medium"}
        assert set(a.layer for a in summary.aggregates) == {"app", "infra"}

    def test_context_manager(self, tmp_output):
        """Runner should support context manager protocol."""
        with BenchmarkRunner(tmp_output) as runner:
            assert runner is not None
        # After exiting, close was called (container cleaned up)

