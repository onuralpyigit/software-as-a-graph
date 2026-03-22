import sys
import pytest
import importlib.util
import json
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

def load_script(path: Path, module_name: str):
    """Load a script file as a module with a unique name."""
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# =============================================================================
# Shared Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def run_module():
    return load_script(PROJECT_ROOT / "bin" / "run.py", "test_run_module")

@pytest.fixture
def mock_stages(run_module):
    mock_dispatchers = {
        name: MagicMock() for name in [
            "dispatch_generate", "dispatch_import", "dispatch_analyze", 
            "dispatch_predict", "dispatch_simulate", "dispatch_validate", 
            "dispatch_visualize", "create_repository"
        ]
    }
    
    # Store originals
    original_stages = list(run_module.STAGES)
    original_create_repo = run_module.create_repository
    
    # Create new stages list with mocks
    new_stages = []
    for flag, label, func, prepper in original_stages:
        mock_func = mock_dispatchers.get(func.__name__, func)
        new_stages.append((flag, label, mock_func, prepper))
    
    # Patch STAGES and create_repository on the module
    run_module.STAGES = new_stages
    run_module.create_repository = mock_dispatchers["create_repository"]
    
    yield mock_dispatchers
    
    # Restore
    run_module.STAGES = original_stages
    run_module.create_repository = original_create_repo

# =============================================================================
# CLI Smoke Tests (Individual Scripts)
# =============================================================================

class TestGenerateGraphCLI:
    """Tests for bin/generate_graph.py"""
    
    @pytest.fixture(scope="class")
    def script_module(self):
        return load_script(PROJECT_ROOT / "bin" / "generate_graph.py", "test_gen_class")

    def test_main(self, script_module):
        mock_data = {"nodes": [{"id": "n1"}]}
        
        with patch.object(sys, 'argv', ['generate_graph.py', '--scale', 'tiny', '--output', 'test_output.json']), \
             patch('src.generation.generate_graph', return_value=mock_data) as mock_gen, \
             patch('builtins.open', mock_open()) as m_open:
            
            script_module.main()
            
            mock_gen.assert_called_once()
            m_open.assert_called()


class TestImportGraphCLI:
    """Tests for bin/import_graph.py"""
    
    @pytest.fixture(scope="class")
    def script_module(self):
        return load_script(PROJECT_ROOT / "bin" / "import_graph.py", "test_imp_class")

    def test_main(self, script_module):
        mock_repo = MagicMock()
        mock_repo.get_statistics.return_value = {"node_count": 10}
        mock_data = {"nodes": []}
        
        with patch.object(sys, 'argv', ['import_graph.py', '--input', 'test.json', '--clear']), \
             patch.object(script_module, 'create_repository', return_value=mock_repo) as MockCreateRepo, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(mock_data))):
            
            script_module.main()
            
            MockCreateRepo.assert_called_once()
            mock_repo.save_graph.assert_called_once_with(mock_data, clear=True)
            mock_repo.close.assert_called_once()


class TestAnalyzeGraphCLI:
    """Tests for bin/analyze_graph.py"""
    
    @pytest.fixture(scope="class")
    def script_module(self):
        return load_script(PROJECT_ROOT / "bin" / "analyze_graph.py", "test_ana_class")

    def test_main(self, script_module):
        mock_repo = MagicMock()
        mock_display = MagicMock()
        mock_results = MagicMock()
        
        with patch.object(sys, 'argv', ['analyze_graph.py', '--layer', 'app']), \
             patch.object(script_module, 'create_repository', return_value=mock_repo) as MockCreateRepo, \
             patch.object(script_module, 'dispatch_analyze', return_value=mock_results) as MockDispatch, \
             patch.object(script_module, 'ConsoleDisplay', return_value=mock_display):
            
            ret = script_module.main()
            
            assert ret == 0
            MockCreateRepo.assert_called_once()
            MockDispatch.assert_called_once()
            mock_display.display_multi_layer_analysis_result.assert_called_once_with(mock_results)
            mock_repo.close.assert_called()


class TestSimulateGraphCLI:
    """Tests for bin/simulate_graph.py"""
    
    @pytest.fixture(scope="class")
    def script_module(self):
        return load_script(PROJECT_ROOT / "bin" / "simulate_graph.py", "test_sim_class")

    def test_main(self, script_module):
        mock_repo = MagicMock()
        mock_display = MagicMock()
        
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
        mock_event_result.component_names = {}
        mock_event_result.source_app = "App1"
        mock_event_result.scenario = "test"
        mock_event_result.duration = 1.0
        
        with patch.object(sys, 'argv', ['simulate_graph.py', 'event', '--source', 'App1']), \
             patch.object(script_module, 'create_repository', return_value=mock_repo) as MockCreateRepo, \
             patch.object(script_module, 'dispatch_simulate', return_value=mock_event_result) as MockDispatch, \
             patch.object(script_module, 'ConsoleDisplay', return_value=mock_display):
            
            ret = script_module.main()
            
            assert ret == 0
            MockCreateRepo.assert_called_once()
            MockDispatch.assert_called_once()
            mock_display.display_event_result.assert_called_once()
            mock_repo.close.assert_called_once()


class TestValidateGraphCLI:
    """Tests for bin/validate_graph.py"""
    
    @pytest.fixture(scope="class")
    def script_module(self):
        return load_script(PROJECT_ROOT / "bin" / "validate_graph.py", "test_val_class")

    def test_main(self, script_module):
        mock_repo = MagicMock()
        mock_display = MagicMock()
        
        # Create a mock result object that satisfies script logic
        mock_result = MagicMock()
        mock_result.passed = True
        mock_result.all_passed = True
        mock_result.layers = {'app': MagicMock()}
        
        with patch.object(sys, 'argv', ['validate_graph.py', '--layer', 'app']), \
             patch.object(script_module, 'create_repository', return_value=mock_repo) as MockCreateRepo, \
             patch.object(script_module, 'dispatch_validate', return_value=mock_result) as MockDispatch, \
             patch.object(script_module, 'ConsoleDisplay', return_value=mock_display):
            
            ret = script_module.main()
            
            assert ret == 0
            MockCreateRepo.assert_called_once()
            MockDispatch.assert_called_once()
            mock_display.display_pipeline_validation_result.assert_called_once()
            mock_repo.close.assert_called_once()


class TestVisualizeGraphCLI:
    """Tests for bin/visualize_graph.py"""
    
    @pytest.fixture(scope="class")
    def script_module(self):
        return load_script(PROJECT_ROOT / "bin" / "visualize_graph.py", "test_viz_class")

    def test_main(self, script_module):
        mock_repo = MagicMock()
        mock_display = MagicMock()
        
        with patch.object(sys, 'argv', ['visualize_graph.py', '--layer', 'app', '--output', 'test.html']), \
             patch.object(script_module, 'create_repository', return_value=mock_repo) as MockCreateRepo, \
             patch.object(script_module, 'dispatch_visualize', return_value='test.html') as MockDispatch, \
             patch.object(script_module, 'ConsoleDisplay', return_value=mock_display), \
             patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=1024):
            
            ret = script_module.main()
            
            assert ret == 0
            MockCreateRepo.assert_called_once()
            MockDispatch.assert_called_once()
            mock_repo.close.assert_called_once()


class TestExportGraphCLI:
    """Tests for bin/export_graph.py"""
    
    @pytest.fixture(scope="class")
    def script_module(self):
        return load_script(PROJECT_ROOT / "bin" / "export_graph.py", "test_exp_class")

    def test_main(self, script_module):
        mock_repo = MagicMock()
        mock_data = {"nodes": [], "relationships": {}}
        mock_repo.export_json.return_value = mock_data
        
        try:
            with patch.object(sys, 'argv', ['export_graph.py', '--output', 'exported.json']), \
                 patch.object(script_module, 'create_repository', return_value=mock_repo) as MockCreateRepo, \
                 patch('builtins.open', mock_open()) as m_open:
                
                script_module.main()
                
                MockCreateRepo.assert_called_once()
                mock_repo.export_json.assert_called_once()
                m_open.assert_called()
                mock_repo.close.assert_called_once()
        except FileNotFoundError:
            pytest.skip("export_graph.py not found")


# =============================================================================
# Pipeline Orchestrator Tests (bin/run.py)
# =============================================================================

class TestRunOrchestrator:
    """Tests for bin/run.py"""
    
    def _reload_and_run(self, module, argv: list[str]) -> int:
        return module.main()

    def test_all_stages_called(self, run_module, mock_stages):
        with patch.object(sys, "argv", ["run.py", "--all", "--scale", "tiny", "--clean"]):
            mock_stages["dispatch_generate"].return_value = {"nodes": []}
            
            ret = self._reload_and_run(run_module, [])
            assert ret == 0
            
            mock_stages["dispatch_generate"].assert_called_once()
            mock_stages["dispatch_import"].assert_called_once()
            mock_stages["dispatch_analyze"].assert_called_once()
            mock_stages["dispatch_predict"].assert_called_once()
            mock_stages["dispatch_simulate"].assert_called_once()
            mock_stages["dispatch_validate"].assert_called_once()
            mock_stages["dispatch_visualize"].assert_called_once()

    def test_generate_args(self, run_module, mock_stages):
        with patch.object(sys, "argv", ["run.py", "--all", "--scale", "tiny"]):
            mock_stages["dispatch_generate"].return_value = {"nodes": []}
            self._reload_and_run(run_module, [])
            
            # Check if scale was passed to dispatcher
            args = mock_stages["dispatch_generate"].call_args[0][0]
            assert args.scale == "tiny"
            
    def test_generate_with_config(self, run_module, mock_stages):
        with patch.object(sys, "argv", ["run.py", "--all", "--config", "conf/ros2.yaml"]):
            mock_stages["dispatch_generate"].return_value = {"nodes": []}
            self._reload_and_run(run_module, [])
            args = mock_stages["dispatch_generate"].call_args[0][0]
            assert args.config == "conf/ros2.yaml"

    def test_import_with_clean(self, run_module, mock_stages):
        with patch.object(sys, "argv", ["run.py", "--all", "--clean"]):
            self._reload_and_run(run_module, [])
            args = mock_stages["dispatch_import"].call_args[0][1]
            assert args.clear is True
            
    def test_neo4j_args_forwarded(self, run_module, mock_stages):
        uri, user, pw = "bolt://db:7687", "admin", "secret"
        with patch.object(sys, "argv", ["run.py", "--all", "--uri", uri, "--user", user, "--password", pw]):
            self._reload_and_run(run_module, [])
            mock_stages["create_repository"].assert_called_with(uri=uri, user=user, password=pw)
            args = mock_stages["dispatch_import"].call_args[0][1]
            assert args.uri == uri
            assert args.user == user
            assert args.password == pw

    def test_analyze_uses_all_flag(self, run_module, mock_stages):
        with patch.object(sys, "argv", ["run.py", "--analyze", "--all"]):
            self._reload_and_run(run_module, [])
            args = mock_stages["dispatch_analyze"].call_args[0][1]
            assert args.all is True

    def test_simulate_report_subcommand(self, run_module, mock_stages):
        with patch.object(sys, "argv", ["run.py", "--simulate"]):
            self._reload_and_run(run_module, [])
            args = mock_stages["dispatch_simulate"].call_args[0][1]
            # Default simulation in run.py is 'report'
            assert args.command == "report"

    def test_pipeline_aborts_on_failure(self, run_module, mock_stages):
        with patch.object(sys, "argv", ["run.py", "--all", "--scale", "tiny"]):
            mock_stages["dispatch_generate"].side_effect = Exception("Failed")
            ret = self._reload_and_run(run_module, [])
            assert ret == 1
            assert mock_stages["dispatch_import"].call_count == 0
            
    def test_analyze_only(self, run_module, mock_stages):
        with patch.object(sys, "argv", ["run.py", "--analyze"]):
            self._reload_and_run(run_module, [])
            assert mock_stages["dispatch_analyze"].call_count == 1

    def test_validate_only(self, run_module, mock_stages):
        with patch.object(sys, "argv", ["run.py", "--validate"]):
            self._reload_and_run(run_module, [])
            assert mock_stages["dispatch_validate"].call_count == 1

    def test_multi_stage_selection(self, run_module, mock_stages):
        with patch.object(sys, "argv", ["run.py", "--analyze", "--simulate", "--validate"]):
            self._reload_and_run(run_module, [])
            assert mock_stages["dispatch_analyze"].call_count == 1
            assert mock_stages["dispatch_simulate"].call_count == 1
            assert mock_stages["dispatch_validate"].call_count == 1

    def test_no_stage_prints_help(self, run_module, mock_stages):
        with patch.object(sys, "argv", ["run.py"]):
            ret = self._reload_and_run(run_module, [])
            assert ret == 1


class TestLayerHandling:
    """Tests for --layer/--layers argument mapping in run.py"""
    
    def _reload_and_run(self, module, argv: list[str]) -> int:
        return module.main()

    def test_single_layer_uses_layer_flag(self, run_module, mock_stages):
        with patch.object(sys, "argv", ["run.py", "--analyze", "--layer", "system"]):
            self._reload_and_run(run_module, [])
            args = mock_stages["dispatch_analyze"].call_args[0][1]
            assert args.layer == "system"

    def test_multi_layer_uses_all_flag(self, run_module, mock_stages):
        with patch.object(sys, "argv", ["run.py", "--analyze", "--layer", "app,infra"]):
            self._reload_and_run(run_module, [])
            args = mock_stages["dispatch_analyze"].call_args[0][1]
            assert args.all is True

    def test_visualize_single_layer(self, run_module, mock_stages):
        with patch.object(sys, "argv", ["run.py", "--visualize", "--layer", "app"]):
            self._reload_and_run(run_module, [])
            args = mock_stages["dispatch_visualize"].call_args[0][1]
            assert args.layer == "app"

    def test_visualize_multi_layer(self, run_module, mock_stages):
        with patch.object(sys, "argv", ["run.py", "--visualize", "--layer", "app,infra"]):
            self._reload_and_run(run_module, [])
            args = mock_stages["dispatch_visualize"].call_args[0][1]
            assert args.layers == "app,infra"


class TestOptionsPassthrough:
    """Tests for flag forwarding to sub-scripts in run.py"""

    def _reload_and_run(self, module, argv: list[str]) -> int:
        return module.main()

    def test_use_ahp_forwarded(self, run_module, mock_stages):
        with patch.object(sys, "argv", ["run.py", "--analyze", "--use-ahp"]):
            self._reload_and_run(run_module, [])
            args = mock_stages["dispatch_analyze"].call_args[0][1]
            assert args.use_ahp is True

    def test_verbose_forwarded(self, run_module, mock_stages):
        with patch.object(sys, "argv", ["run.py", "--analyze", "--simulate", "--verbose"]):
            self._reload_and_run(run_module, [])
            assert mock_stages["dispatch_analyze"].call_args[0][1].verbose is True
            assert mock_stages["dispatch_simulate"].call_args[0][1].verbose is True

    def test_open_forwarded_to_visualize(self, run_module, mock_stages):
        with patch.object(sys, "argv", ["run.py", "--visualize", "--open", "--layer", "app"]):
            self._reload_and_run(run_module, [])
            args = mock_stages["dispatch_visualize"].call_args[0][1]
            assert args.open is True


class TestOutputPaths:
    """Tests for output directory and file paths in run.py"""
    
    def _reload_and_run(self, module, argv: list[str]) -> int:
        return module.main()

    def test_custom_output_dir(self, run_module, mock_stages):
        with patch.object(sys, "argv", ["run.py", "--analyze", "--output-dir", "results/v2"]):
            self._reload_and_run(run_module, [])
            args = mock_stages["dispatch_analyze"].call_args[0][1]
            assert "results/v2" in args.output

    def test_custom_input_path(self, run_module, mock_stages):
        with patch.object(sys, "argv", ["run.py", "--generate", "--import", "--input", "data/custom.json"]):
            mock_stages["dispatch_generate"].return_value = {"nodes": []}
            self._reload_and_run(run_module, [])
            
            assert mock_stages["dispatch_generate"].call_args[0][0].output == "data/custom.json"
            assert mock_stages["dispatch_import"].call_args[0][1].input == "data/custom.json"


# =============================================================================
# Benchmark CLI Tests (bin/benchmark.py)
# =============================================================================

class TestBenchmarkCLI:
    """Tests for bin/benchmark.py"""
    
    @pytest.fixture(scope="class")
    def benchmark_module(self):
        return load_script(PROJECT_ROOT / "bin" / "benchmark.py", "test_bench_class")

    @pytest.fixture
    def tmp_output(self, tmp_path):
        return tmp_path / "benchmark_output"

    def _run_main(self, module, argv: list[str]) -> int:
        with patch.object(sys, "argv", ["benchmark.py"] + argv):
            return module.main()
    
    @patch("src.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("src.benchmark.runner.BenchmarkRunner.close")
    def test_scales_flag(self, mock_close, mock_run, benchmark_module, tmp_output):
        mock_run.return_value = []
        ret = self._run_main(
            benchmark_module,
            ["--scales", "tiny,small", "--runs", "1", "--output", str(tmp_output)]
        )
        assert ret == 0
        assert mock_run.call_count == 2
        scenario_names = [call.args[0].name for call in mock_run.call_args_list]
        assert "tiny" in scenario_names
        assert "small" in scenario_names

    @patch("src.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("src.benchmark.runner.BenchmarkRunner.close")
    def test_full_suite_flag(self, mock_close, mock_run, benchmark_module, tmp_output):
        mock_run.return_value = []
        ret = self._run_main(benchmark_module, ["--full-suite", "--output", str(tmp_output)])
        assert ret == 0
        assert mock_run.call_count == 3

    @patch("src.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("src.benchmark.runner.BenchmarkRunner.close")
    def test_default_scenario(self, mock_close, mock_run, benchmark_module, tmp_output):
        mock_run.return_value = []
        ret = self._run_main(benchmark_module, ["--output", str(tmp_output)])
        assert ret == 0
        assert mock_run.call_count == 1
        assert mock_run.call_args[0][0].scale == "medium"

    @patch("src.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("src.benchmark.runner.BenchmarkRunner.close")
    def test_layers_forwarded(self, mock_close, mock_run, benchmark_module, tmp_output):
        mock_run.return_value = []
        ret = self._run_main(
            benchmark_module,
            ["--scales", "tiny", "--layers", "app,infra", "--output", str(tmp_output)]
        )
        assert ret == 0
        scenario = mock_run.call_args[0][0]
        assert scenario.layers == ["app", "infra"]

    @patch("src.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("src.benchmark.runner.BenchmarkRunner.close")
    def test_runs_forwarded(self, mock_close, mock_run, benchmark_module, tmp_output):
        mock_run.return_value = []
        ret = self._run_main(
            benchmark_module,
            ["--scales", "tiny", "--runs", "5", "--output", str(tmp_output)]
        )
        assert ret == 0
        scenario = mock_run.call_args[0][0]
        assert scenario.runs == 5

    @patch("src.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("src.benchmark.runner.BenchmarkRunner.close")
    def test_reports_generated(self, mock_close, mock_run, benchmark_module, tmp_output):
        mock_run.return_value = []
        ret = self._run_main(benchmark_module, ["--scales", "tiny", "--output", str(tmp_output)])
        assert ret == 0
        assert (tmp_output / "benchmark_results.json").exists()
        assert (tmp_output / "benchmark_report.md").exists()

    @patch("src.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("src.benchmark.runner.BenchmarkRunner.close")
    def test_neo4j_args_forwarded(self, mock_close, mock_run, benchmark_module, tmp_output):
        mock_run.return_value = []
        with patch("src.benchmark.runner.create_repository") as MockCreateRepo:
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
