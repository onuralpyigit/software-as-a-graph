import sys
import pytest
import importlib.util
import json
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import cli.run as run_module_imported
import cli.generate_graph
import cli.import_graph
import cli.analyze_graph
import cli.simulate_graph
import cli.validate_graph
import cli.visualize_graph
import cli.export_graph
import cli.benchmark

# =============================================================================
# Shared Fixtures
# =============================================================================





# =============================================================================
# CLI Smoke Tests (Individual Scripts)
# =============================================================================

class TestGenerateGraphCLI:
    """Tests for bin/generate_graph.py"""
    
    @pytest.fixture(scope="class")
    def script_module(self):
        return cli.generate_graph

    def test_main(self, script_module):
        mock_data = {"nodes": [{"id": "n1"}]}
        
        with patch.object(sys, 'argv', ['generate_graph.py', '--scale', 'tiny', '--output', 'test_output.json']), \
             patch('tools.generation.generate_graph', return_value=mock_data) as mock_gen, \
             patch('builtins.open', mock_open()) as m_open:
            
            script_module.main()
            
            mock_gen.assert_called_once()
            m_open.assert_called()


class TestImportGraphCLI:
    """Tests for bin/import_graph.py"""
    
    @pytest.fixture(scope="class")
    def script_module(self):
        return cli.import_graph

    def test_main(self, script_module):
        mock_client = MagicMock()
        mock_client.import_topology.return_value = {
            "nodes_imported": 10,
            "edges_imported": 5,
            "duration_ms": 1.5,
            "success": True
        }
        mock_data = {"nodes": []}
        
        with patch.object(sys, 'argv', ['import_graph.py', '--input', 'test.json', '--clear']), \
             patch.object(script_module, 'Client', return_value=mock_client) as MockClient, \
             patch.object(Path, 'exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(mock_data))):
            
            script_module.main()
            
            MockClient.assert_called_once()
            mock_client.import_topology.assert_called_once_with(filepath='test.json', clear=True)


class TestAnalyzeGraphCLI:
    """Tests for bin/analyze_graph.py"""
    
    @pytest.fixture(scope="class")
    def script_module(self):
        return cli.analyze_graph

    def test_main(self, script_module):
        mock_client = MagicMock()
        mock_results = MagicMock()
        mock_client.analyze.return_value = mock_results
        
        with patch.object(sys, 'argv', ['analyze_graph.py', '--layer', 'app']), \
             patch.object(script_module, 'Client', return_value=mock_client) as MockClient:
            
            ret = script_module.main()
            
            MockClient.assert_called_once()
            mock_client.analyze.assert_called_once()


class TestSimulateGraphCLI:
    """Tests for bin/simulate_graph.py"""
    
    @pytest.fixture(scope="class")
    def script_module(self):
        return cli.simulate_graph

    def test_main(self, script_module):
        mock_client = MagicMock()
        
        # Create a mock result object that satisfies script logic
        mock_event_result = MagicMock()
        mock_event_result.metrics = MagicMock()
        mock_event_result.metrics.messages_published = 100
        mock_event_result.metrics.messages_delivered = 100
        mock_event_result.metrics.messages_dropped = 0
        mock_event_result.metrics.delivery_rate = 100.0
        mock_event_result.metrics.drop_rate = 0.0
        mock_event_result.metrics.avg_latency = 0.005
        mock_event_result.metrics.p99_latency = 0.01
        mock_event_result.metrics.throughput = 100.0
        mock_event_result.csc_names = {}
        mock_event_result.source_app = "App1"
        mock_event_result.scenario = "test"
        mock_event_result.duration = 1.0
        mock_client.simulate.return_value = mock_event_result
        
        with patch.object(sys, 'argv', ['simulate_graph.py', '--layer', 'system', '--mode', 'exhaustive']), \
             patch.object(script_module, 'Client', return_value=mock_client) as MockClient:
            
            ret = script_module.main()
            
            MockClient.assert_called_once()
            mock_client.simulate.assert_called_once_with(layer='system', mode='exhaustive')


class TestValidateGraphCLI:
    """Tests for bin/validate_graph.py"""
    
    @pytest.fixture(scope="class")
    def script_module(self):
        return cli.validate_graph

    def test_main(self, script_module):
        mock_client = MagicMock()
        
        # Create a mock result object that satisfies script logic
        mock_result = MagicMock()
        mock_result.passed = True
        mock_result.all_passed = True
        mock_result.layers = {'app': MagicMock()}
        mock_client.validate.return_value = mock_result
        
        with patch.object(sys, 'argv', ['validate_graph.py', '--layer', 'app']), \
             patch.object(script_module, 'Client', return_value=mock_client) as MockClient:
            
            ret = script_module.main()
            
            MockClient.assert_called_once()
            mock_client.validate.assert_called_once()


class TestVisualizeGraphCLI:
    """Tests for bin/visualize_graph.py"""
    
    @pytest.fixture(scope="class")
    def script_module(self):
        return cli.visualize_graph

    def test_main(self, script_module):
        mock_client = MagicMock()
        mock_client.visualize.return_value = 'test.html'
        
        with patch.object(sys, 'argv', ['visualize_graph.py', '--layer', 'app', '--output', 'test.html']), \
             patch.object(script_module, 'Client', return_value=mock_client) as MockClient, \
             patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=1024):
            
            ret = script_module.main()
            
            MockClient.assert_called_once()
            mock_client.visualize.assert_called_once()


class TestExportGraphCLI:
    """Tests for bin/export_graph.py"""
    
    @pytest.fixture(scope="class")
    def script_module(self):
        return cli.export_graph

        mock_client = MagicMock()
        mock_data = {"nodes": [], "relationships": {}}
        mock_client.export_topology.return_value = mock_data
        
        try:
            with patch.object(sys, 'argv', ['export_graph.py', '--output', 'exported.json']), \
                 patch.object(script_module, 'Client', return_value=mock_client) as MockClient, \
                 patch('builtins.open', mock_open()) as m_open:
                
                script_module.main()
                
                MockClient.assert_called_once()
                mock_client.export_topology.assert_called_once()
                m_open.assert_called()
        except FileNotFoundError:
            pytest.skip("export_graph.py not found")


# =============================================================================
# Pipeline Orchestrator Tests (bin/run.py)
# =============================================================================

@pytest.fixture
def mock_pipeline():
    with patch('cli.run.Pipeline') as mock_pipe_class:
        instance = MagicMock()
        # Mock fluent interface
        instance.analyze.return_value = instance
        instance.predict.return_value = instance
        instance.simulate.return_value = instance
        instance.validate.return_value = instance
        instance.visualize.return_value = instance
        mock_pipe_class.return_value = instance
        mock_pipe_class.from_json.return_value = instance
        yield mock_pipe_class, instance

class TestRunOrchestrator:
    """Tests for bin/run.py using saag.Pipeline"""

    def test_run_help_subprocess(self):
        import subprocess
        result = subprocess.run([sys.executable, str(PROJECT_ROOT / "bin" / "run.py"), "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()

    
    @pytest.fixture(scope="class")
    def run_module(self):
        return cli.run

    def _run_main(self, module, argv: list[str]) -> int:
        with patch.object(sys, "argv", ["run.py"] + argv), \
             patch("pathlib.Path.exists", return_value=True):
            try:
                module.main()
                return 0
            except SystemExit as e:
                return e.code

    def test_all_stages_called(self, run_module, mock_pipeline):
        mock_class, mock_inst = mock_pipeline
        ret = self._run_main(run_module, ["--all"])
        assert ret == 0
        
        mock_class.from_json.assert_called_once()
        mock_inst.analyze.assert_called_once_with(layer='system', use_ahp=False)
        mock_inst.predict.assert_called_once_with(gnn_model=None)
        mock_inst.simulate.assert_called_once_with(layer='system', mode='exhaustive')
        mock_inst.validate.assert_called_once_with(layers=['app', 'infra', 'mw', 'system'])
        mock_inst.visualize.assert_called_once()
        mock_inst.run.assert_called_once()

    def test_neo4j_args_forwarded(self, run_module, mock_pipeline):
        mock_class, mock_inst = mock_pipeline
        uri, user, pw = "bolt://db:7687", "admin", "secret"
        ret = self._run_main(run_module, ["--analyze", "--uri", uri, "--user", user, "--password", pw])
        assert ret == 0
        mock_class.assert_called_once_with(neo4j_uri=uri, user=user, password=pw)

    def test_pipeline_aborts_on_failure(self, run_module, mock_pipeline):
        mock_class, mock_inst = mock_pipeline
        mock_inst.run.side_effect = Exception("Failed")
        with pytest.raises(Exception):
            self._run_main(run_module, ["--all"])

    def test_analyze_only(self, run_module, mock_pipeline):
        mock_class, mock_inst = mock_pipeline
        self._run_main(run_module, ["--analyze"])
        assert mock_inst.analyze.call_count == 1
        assert mock_inst.predict.call_count == 0

    def test_validate_only(self, run_module, mock_pipeline):
        mock_class, mock_inst = mock_pipeline
        self._run_main(run_module, ["--validate"])
        assert mock_inst.validate.call_count == 1
        assert mock_inst.analyze.call_count == 0

    def test_no_stage_prints_help(self, run_module, mock_pipeline):
        with patch('sys.stderr'), patch('argparse.ArgumentParser.error') as mock_error:
            self._run_main(run_module, [])
            mock_error.assert_called_once()

class TestLayerHandling:
    """Tests for --layer/--layers argument mapping in run.py"""
    
    @pytest.fixture(scope="class")
    def run_module(self):
        return cli.run

    def _run_main(self, module, argv: list[str]) -> int:
        with patch.object(sys, "argv", ["run.py"] + argv), \
             patch("pathlib.Path.exists", return_value=True):
            try:
                module.main()
                return 0
            except SystemExit as e:
                return e.code

    def test_single_layer_uses_layer_flag(self, run_module, mock_pipeline):
        mock_class, mock_inst = mock_pipeline
        self._run_main(run_module, ["--analyze", "--layer", "system"])
        mock_inst.analyze.assert_called_once_with(layer="system", use_ahp=False)

    def test_visualize_multi_layer(self, run_module, mock_pipeline):
        mock_class, mock_inst = mock_pipeline
        self._run_main(run_module, ["--visualize", "--layer", "app"])
        kwargs = mock_inst.visualize.call_args[1]
        assert kwargs["layers"] == ["app"]

class TestOptionsPassthrough:
    """Tests for flag forwarding to sub-scripts in run.py"""
    
    @pytest.fixture(scope="class")
    def run_module(self):
        return cli.run

    def _run_main(self, module, argv: list[str]) -> int:
        with patch.object(sys, "argv", ["run.py"] + argv), \
             patch("pathlib.Path.exists", return_value=True):
            try:
                module.main()
                return 0
            except SystemExit as e:
                return e.code

    def test_use_ahp_forwarded(self, run_module, mock_pipeline):
        mock_class, mock_inst = mock_pipeline
        self._run_main(run_module, ["--analyze", "--use-ahp"])
        mock_inst.analyze.assert_called_once_with(layer='system', use_ahp=True)

class TestOutputPaths:
    """Tests for output directory and file paths in run.py"""
    
    @pytest.fixture(scope="class")
    def run_module(self):
        return cli.run

    def _run_main(self, module, argv: list[str]) -> int:
        with patch.object(sys, "argv", ["run.py"] + argv), \
             patch("pathlib.Path.exists", return_value=True):
            try:
                module.main()
                return 0
            except SystemExit as e:
                return e.code

    def test_custom_output_dir(self, run_module, mock_pipeline):
        mock_class, mock_inst = mock_pipeline
        self._run_main(run_module, ["--analyze", "--output", "results/v2.json"])
        # Expect save to be called because it's not visualize or all
        mock_inst.run.return_value.save.assert_called_once_with("results/v2.json")

    def test_custom_input_path(self, run_module, mock_pipeline):
        mock_class, mock_inst = mock_pipeline
        self._run_main(run_module, ["--input", "data/custom.json"])
        mock_class.from_json.assert_called_once_with("data/custom.json", clear=False, neo4j_uri="bolt://localhost:7687", password="password", user="neo4j")

    def test_generate_stage_called(self, run_module, mock_pipeline):
        mock_class, mock_inst = mock_pipeline
        with patch('cli.common.dispatcher.dispatch_generate') as mock_gen:
            ret = self._run_main(run_module, ["--generate", "--input", "test.json", "--scale", "tiny"])
            assert ret == 0
            mock_gen.assert_called_once()
            # Pipeline is then initialized from the generated file
            mock_class.from_json.assert_called_once_with("test.json", clear=False, neo4j_uri="bolt://localhost:7687", password="password", user="neo4j")


# =============================================================================
# Benchmark CLI Tests (bin/benchmark.py)
# =============================================================================

class TestBenchmarkCLI:
    """Tests for bin/benchmark.py"""
    
    @pytest.fixture(scope="class")
    def benchmark_module(self):
        return cli.benchmark

    @pytest.fixture
    def tmp_output(self, tmp_path):
        return tmp_path / "benchmark_output"

    def _run_main(self, module, argv: list[str]) -> int:
        with patch.object(sys, "argv", ["benchmark.py"] + argv):
            return module.main()
    
    @patch("tools.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("tools.benchmark.runner.BenchmarkRunner.close")
    def test_scales_flag(self, mock_close, mock_run, benchmark_module, tmp_output):
        mock_run.return_value = []
        ret = self._run_main(
            benchmark_module,
            ["--scales", "tiny,small", "--runs", "1", "--output", str(tmp_output)]
        )
        assert ret == 0
        assert mock_run.call_count == 2
        scenario_names = [call.args[0].name for call in mock_run.call_args_list]
        assert "auto-tiny" in scenario_names
        assert "auto-small" in scenario_names

    @patch("tools.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("tools.benchmark.runner.BenchmarkRunner.close")
    def test_full_suite_flag(self, mock_close, mock_run, benchmark_module, tmp_output):
        mock_run.return_value = []
        ret = self._run_main(benchmark_module, ["--full-suite", "--output", str(tmp_output)])
        assert ret == 0
        assert mock_run.call_count == 3

    @patch("tools.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("tools.benchmark.runner.BenchmarkRunner.close")
    def test_default_scenario(self, mock_close, mock_run, benchmark_module, tmp_output):
        mock_run.return_value = []
        ret = self._run_main(benchmark_module, ["--output", str(tmp_output)])
        assert ret == 0
        assert mock_run.call_count == 2
        assert mock_run.call_args_list[0].args[0].scale == "tiny"
        assert mock_run.call_args_list[1].args[0].scale == "small"

    @patch("tools.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("tools.benchmark.runner.BenchmarkRunner.close")
    def test_layers_forwarded(self, mock_close, mock_run, benchmark_module, tmp_output):
        mock_run.return_value = []
        ret = self._run_main(
            benchmark_module,
            ["--scales", "tiny", "--layers", "app,infra", "--output", str(tmp_output)]
        )
        assert ret == 0
        scenario = mock_run.call_args[0][0]
        assert scenario.layers == ["app", "infra"]

    @patch("tools.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("tools.benchmark.runner.BenchmarkRunner.close")
    def test_runs_forwarded(self, mock_close, mock_run, benchmark_module, tmp_output):
        mock_run.return_value = []
        ret = self._run_main(
            benchmark_module,
            ["--scales", "tiny", "--runs", "5", "--output", str(tmp_output)]
        )
        assert ret == 0
        scenario = mock_run.call_args[0][0]
        assert scenario.runs == 5

    @patch("tools.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("tools.benchmark.runner.BenchmarkRunner.close")
    def test_reports_generated(self, mock_close, mock_run, benchmark_module, tmp_output):
        mock_run.return_value = []
        ret = self._run_main(benchmark_module, ["--scales", "tiny", "--output", str(tmp_output)])
        assert ret == 0
        assert (tmp_output / "benchmark_results.json").exists()
        assert (tmp_output / "benchmark_report.md").exists()

    @patch("tools.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("tools.benchmark.runner.BenchmarkRunner.close")
    def test_neo4j_args_forwarded(self, mock_close, mock_run, benchmark_module, tmp_output):
        mock_run.return_value = []
        with patch("tools.benchmark.runner.create_repository") as MockCreateRepo:
            ret = self._run_main(benchmark_module, [
                "--scales", "tiny",
                "--uri", "bolt://db:7687",
                "--user", "admin",
                "--password", "secret",
                "--output", str(tmp_output),
            ])
        assert ret == 0
        MockCreateRepo.assert_called_with(
            uri="bolt://db:7687", user="admin", password="secret"
        )
