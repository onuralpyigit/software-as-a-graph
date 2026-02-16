
import sys
import pytest
import importlib
import json
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

# Ensure project root is in path
# backend/tests/ -> backend/ -> project_root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "backend"))
sys.path.append(str(PROJECT_ROOT / "bin"))

# =============================================================================
# CLI Smoke Tests (Individual Scripts)
# =============================================================================

class TestGenerateGraphCLI:
    """Tests for bin/generate_graph.py"""
    
    def test_main(self):
        mock_data = {"nodes": [{"id": "n1"}]}
        
        with patch.object(sys, 'argv', ['generate_graph.py', '--scale', 'tiny', '--output', 'test_output.json']), \
             patch('src.generation.generate_graph', return_value=mock_data) as mock_gen, \
             patch('builtins.open', mock_open()) as m_open:
            
            if 'generate_graph' in sys.modules:
                del sys.modules['generate_graph']
            import generate_graph
            importlib.reload(generate_graph)
            
            generate_graph.main()
            
            mock_gen.assert_called_once()
            m_open.assert_called()


class TestImportGraphCLI:
    """Tests for bin/import_graph.py"""
    
    def test_main(self):
        mock_repo = MagicMock()
        mock_repo.get_statistics.return_value = {"node_count": 10}
        mock_data = {"nodes": []}
        
        with patch.object(sys, 'argv', ['import_graph.py', '--input', 'test.json', '--clear']), \
             patch('src.core.create_repository', return_value=mock_repo) as MockCreateRepo, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(mock_data))):
            
            if 'import_graph' in sys.modules:
                del sys.modules['import_graph']
            import import_graph
            importlib.reload(import_graph)
            
            import_graph.main()
            
            MockCreateRepo.assert_called_once()
            mock_repo.save_graph.assert_called_once_with(mock_data, clear=True)
            mock_repo.close.assert_called_once()


class TestAnalyzeGraphCLI:
    """Tests for bin/analyze_graph.py"""
    
    def test_main(self):
        mock_repo = MagicMock()
        mock_display = MagicMock()
        mock_analysis_service = MagicMock()
        
        mock_results = MagicMock()
        mock_results.to_dict.return_value = {}
        mock_analysis_service.analyze_layer.return_value = MagicMock()
        
        with patch.object(sys, 'argv', ['analyze_graph.py', '--layer', 'app']), \
             patch('src.core.create_repository', return_value=mock_repo) as MockCreateRepo, \
             patch('src.analysis.AnalysisService', return_value=mock_analysis_service), \
             patch('src.cli.console.ConsoleDisplay', return_value=mock_display):
            
            if 'analyze_graph' in sys.modules:
                del sys.modules['analyze_graph']
            import analyze_graph
            importlib.reload(analyze_graph)
            
            ret = analyze_graph.main()
            
            assert ret == 0
            MockCreateRepo.assert_called_once()
            mock_display.display_multi_layer_analysis_result.assert_called()
            mock_repo.close.assert_called()


class TestSimulateGraphCLI:
    """Tests for bin/simulate_graph.py"""
    
    def test_main(self):
        mock_repo = MagicMock()
        mock_display = MagicMock()
        mock_sim_service = MagicMock()
        
        mock_event_result = MagicMock()
        mock_event_result.to_dict.return_value = {}
        mock_sim_service.run_event_simulation.return_value = mock_event_result
        
        with patch.object(sys, 'argv', ['simulate_graph.py', 'event', '--source', 'App1']), \
             patch('src.core.create_repository', return_value=mock_repo) as MockCreateRepo, \
             patch('src.simulation.SimulationService', return_value=mock_sim_service) as MockSimService, \
             patch('src.cli.console.ConsoleDisplay', return_value=mock_display):
            
            if 'simulate_graph' in sys.modules:
                del sys.modules['simulate_graph']
            import simulate_graph
            importlib.reload(simulate_graph)
            
            ret = simulate_graph.main()
            
            assert ret == 0
            MockCreateRepo.assert_called_once()
            MockSimService.assert_called_once_with(mock_repo)
            mock_sim_service.run_event_simulation.assert_called_once()
            mock_display.display_event_result.assert_called_once()
            mock_repo.close.assert_called_once()


class TestValidateGraphCLI:
    """Tests for bin/validate_graph.py"""
    
    def test_main(self):
        mock_repo = MagicMock()
        mock_display = MagicMock()
        mock_val_service = MagicMock()
        
        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.to_dict.return_value = {}
        mock_val_service.validate_layers.return_value = mock_result
        
        with patch.object(sys, 'argv', ['validate_graph.py', '--layer', 'app']), \
             patch('src.core.create_repository', return_value=mock_repo) as MockCreateRepo, \
             patch('src.validation.ValidationService', return_value=mock_val_service) as MockValService, \
             patch('src.cli.console.ConsoleDisplay', return_value=mock_display):
            
            if 'validate_graph' in sys.modules:
                del sys.modules['validate_graph']
            import validate_graph
            importlib.reload(validate_graph)
            
            ret = validate_graph.main()
            
            assert ret == 0
            MockCreateRepo.assert_called_once()
            assert MockValService.call_count == 1
            mock_display.display_pipeline_validation_result.assert_called_once()
            mock_repo.close.assert_called_once()


class TestVisualizeGraphCLI:
    """Tests for bin/visualize_graph.py"""
    
    def test_main(self):
        mock_repo = MagicMock()
        mock_display = MagicMock()
        mock_viz_service = MagicMock()
        
        mock_viz_service.generate_dashboard.return_value = "dashboard.html"
        
        with patch.object(sys, 'argv', ['visualize_graph.py', '--layer', 'app', '--output', 'test.html']), \
             patch('src.core.create_repository', return_value=mock_repo) as MockCreateRepo, \
             patch('src.analysis.AnalysisService', return_value=MagicMock()), \
             patch('src.simulation.SimulationService', return_value=MagicMock()), \
             patch('src.validation.ValidationService', return_value=MagicMock()), \
             patch('src.visualization.VisualizationService', return_value=mock_viz_service), \
             patch('src.cli.console.ConsoleDisplay', return_value=mock_display), \
             patch('os.path.getsize', return_value=1024):
            
            if 'visualize_graph' in sys.modules:
                del sys.modules['visualize_graph']
            import visualize_graph
            importlib.reload(visualize_graph)
            
            ret = visualize_graph.main()
            
            assert ret == 0
            MockCreateRepo.assert_called_once()
            mock_viz_service.generate_dashboard.assert_called_once()
            mock_repo.close.assert_called_once()


class TestExportGraphCLI:
    """Tests for bin/export_graph.py"""
    
    def test_main(self):
        mock_repo = MagicMock()
        mock_data = {"nodes": [], "relationships": {}}
        mock_repo.export_json.return_value = mock_data
        
        try:
            with patch.object(sys, 'argv', ['export_graph.py', '--output', 'exported.json']), \
                 patch('src.core.create_repository', return_value=mock_repo) as MockCreateRepo, \
                 patch('builtins.open', mock_open()) as m_open:
                
                if 'export_graph' in sys.modules:
                    del sys.modules['export_graph']
                import export_graph
                importlib.reload(export_graph)
                
                export_graph.main()
                
                MockCreateRepo.assert_called_once()
                mock_repo.export_json.assert_called_once()
                m_open.assert_called()
                mock_repo.close.assert_called_once()
        except ImportError:
            pytest.skip("export_graph.py not found")


# =============================================================================
# Pipeline Orchestrator Tests (bin/run.py)
# =============================================================================

class TestRunOrchestrator:
    """Tests for bin/run.py"""
    
    def _reload_and_run(self, argv: list[str], mock_run: MagicMock) -> int:
        mock_run.return_value.returncode = 0
        if 'run' in sys.modules:
            del sys.modules['run']
        import run
        importlib.reload(run)
        return run.main()

    def _extract_commands(self, mock_run: MagicMock) -> list[list[str]]:
        return [call.args[0] for call in mock_run.call_args_list]

    def _find_cmd(self, commands: list[list[str]], script_name: str) -> list[str] | None:
        return next((cmd for cmd in commands if script_name in cmd[1]), None)

    def test_all_stages_called(self):
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py", "--all", "--scale", "tiny", "--clean"]), \
             patch("sys.executable", "python"):
            
            ret = self._reload_and_run([], mock_run)
            assert ret == 0
            
            commands = self._extract_commands(mock_run)
            scripts = [cmd[1].split("/")[-1] for cmd in commands]
            
            assert "generate_graph.py" in scripts[0]
            assert "import_graph.py" in scripts[1]
            assert "analyze_graph.py" in scripts[2]
            assert "simulate_graph.py" in scripts[3]
            assert "validate_graph.py" in scripts[4]
            assert "visualize_graph.py" in scripts[5]

    def test_generate_args(self):
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py", "--all", "--scale", "tiny"]), \
             patch("sys.executable", "python"):
            
            self._reload_and_run([], mock_run)
            cmd = self._find_cmd(self._extract_commands(mock_run), "generate_graph.py")
            
            assert cmd is not None
            assert "--scale" in cmd
            assert "tiny" in cmd
            
    def test_generate_with_config(self):
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py", "--all", "--config", "conf/ros2.yaml"]), \
             patch("sys.executable", "python"):
            
            self._reload_and_run([], mock_run)
            cmd = self._find_cmd(self._extract_commands(mock_run), "generate_graph.py")
            
            assert "--config" in cmd
            assert "conf/ros2.yaml" in cmd
            assert "--scale" not in cmd

    def test_import_with_clean(self):
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py", "--all", "--clean"]), \
             patch("sys.executable", "python"):
            
            self._reload_and_run([], mock_run)
            cmd = self._find_cmd(self._extract_commands(mock_run), "import_graph.py")
            
            assert "--clear" in cmd
            
    def test_neo4j_args_forwarded(self):
        uri, user, pw = "bolt://db:7687", "admin", "secret"
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py", "--all", "--uri", uri, "--user", user, "--password", pw]), \
             patch("sys.executable", "python"):
            
            self._reload_and_run([], mock_run)
            commands = self._extract_commands(mock_run)
            
            for cmd in commands[1:]:
                assert "--uri" in cmd and uri in cmd
                assert "--user" in cmd and user in cmd
                assert "--password" in cmd and pw in cmd

    def test_analyze_uses_all_flag(self):
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py", "--all"]), \
             patch("sys.executable", "python"):
            
            self._reload_and_run([], mock_run)
            cmd = self._find_cmd(self._extract_commands(mock_run), "analyze_graph.py")
            assert "--all" in cmd

    def test_simulate_report_subcommand(self):
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py", "--all"]), \
             patch("sys.executable", "python"):
            
            self._reload_and_run([], mock_run)
            cmd = self._find_cmd(self._extract_commands(mock_run), "simulate_graph.py")
            assert "report" in cmd

    def test_pipeline_aborts_on_failure(self):
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py", "--all", "--scale", "tiny"]), \
             patch("sys.executable", "python"):
            
            mock_run.side_effect = [
                MagicMock(returncode=0),
                MagicMock(returncode=1),
            ]
            
            if 'run' in sys.modules:
                del sys.modules['run']
            import run
            importlib.reload(run)
            ret = run.main()
            
            assert ret == 1
            assert mock_run.call_count == 2
            
    def test_analyze_only(self):
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py", "--analyze"]), \
             patch("sys.executable", "python"):
            
            self._reload_and_run([], mock_run)
            commands = self._extract_commands(mock_run)
            assert len(commands) == 1
            assert "analyze_graph.py" in commands[0][1]

    def test_validate_only(self):
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py", "--validate"]), \
             patch("sys.executable", "python"):
            
            self._reload_and_run([], mock_run)
            commands = self._extract_commands(mock_run)
            assert len(commands) == 1
            assert "validate_graph.py" in commands[0][1]

    def test_multi_stage_selection(self):
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py", "--analyze", "--simulate", "--validate"]), \
             patch("sys.executable", "python"):
            
            self._reload_and_run([], mock_run)
            commands = self._extract_commands(mock_run)
            assert len(commands) == 3

    def test_no_stage_prints_help(self):
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py"]), \
             patch("sys.executable", "python"):
            
            if 'run' in sys.modules:
                del sys.modules['run']
            import run
            importlib.reload(run)
            ret = run.main()
            assert ret == 1
            assert mock_run.call_count == 0


class TestLayerHandling:
    """Tests for --layer/--layers argument mapping in run.py"""
    
    def _reload_and_run(self, argv: list[str], mock_run: MagicMock) -> int:
        mock_run.return_value.returncode = 0
        if 'run' in sys.modules:
            del sys.modules['run']
        import run
        importlib.reload(run)
        return run.main()
        
    def _extract_commands(self, mock_run: MagicMock) -> list[list[str]]:
        return [call.args[0] for call in mock_run.call_args_list]

    def _find_cmd(self, commands: list[list[str]], script_name: str) -> list[str] | None:
        return next((cmd for cmd in commands if script_name in cmd[1]), None)

    def test_single_layer_uses_layer_flag(self):
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py", "--analyze", "--layer", "system"]), \
             patch("sys.executable", "python"):
            
            self._reload_and_run([], mock_run)
            cmd = self._find_cmd(self._extract_commands(mock_run), "analyze_graph.py")
            assert "--layer" in cmd
            assert "system" in cmd

    def test_multi_layer_uses_all_flag(self):
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py", "--analyze", "--layer", "app,infra"]), \
             patch("sys.executable", "python"):
            
            self._reload_and_run([], mock_run)
            cmd = self._find_cmd(self._extract_commands(mock_run), "analyze_graph.py")
            assert "--all" in cmd

    def test_visualize_single_layer(self):
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py", "--visualize", "--layer", "app"]), \
             patch("sys.executable", "python"):
            
            self._reload_and_run([], mock_run)
            cmd = self._find_cmd(self._extract_commands(mock_run), "visualize_graph.py")
            assert "--layer" in cmd

    def test_visualize_multi_layer(self):
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py", "--visualize", "--layer", "app,infra"]), \
             patch("sys.executable", "python"):
            
            self._reload_and_run([], mock_run)
            cmd = self._find_cmd(self._extract_commands(mock_run), "visualize_graph.py")
            assert "--layers" in cmd


class TestOptionsPassthrough:
    """Tests for flag forwarding to sub-scripts in run.py"""

    def _reload_and_run(self, argv: list[str], mock_run: MagicMock) -> int:
        mock_run.return_value.returncode = 0
        if 'run' in sys.modules:
            del sys.modules['run']
        import run
        importlib.reload(run)
        return run.main()
        
    def _extract_commands(self, mock_run: MagicMock) -> list[list[str]]:
        return [call.args[0] for call in mock_run.call_args_list]

    def _find_cmd(self, commands: list[list[str]], script_name: str) -> list[str] | None:
        return next((cmd for cmd in commands if script_name in cmd[1]), None)

    def test_use_ahp_forwarded(self):
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py", "--analyze", "--use-ahp"]), \
             patch("sys.executable", "python"):
            
            self._reload_and_run([], mock_run)
            cmd = self._find_cmd(self._extract_commands(mock_run), "analyze_graph.py")
            assert "--use-ahp" in cmd

    def test_verbose_forwarded(self):
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py", "--analyze", "--simulate", "--validate", "--verbose"]), \
             patch("sys.executable", "python"):
            
            self._reload_and_run([], mock_run)
            commands = self._extract_commands(mock_run)
            for cmd in commands:
                assert "--verbose" in cmd

    def test_open_forwarded_to_visualize(self):
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py", "--visualize", "--open", "--layer", "app"]), \
             patch("sys.executable", "python"):
            
            self._reload_and_run([], mock_run)
            cmd = self._find_cmd(self._extract_commands(mock_run), "visualize_graph.py")
            assert "--open" in cmd


class TestOutputPaths:
    """Tests for output directory and file paths in run.py"""
    
    def _reload_and_run(self, argv: list[str], mock_run: MagicMock) -> int:
        mock_run.return_value.returncode = 0
        if 'run' in sys.modules:
            del sys.modules['run']
        import run
        importlib.reload(run)
        return run.main()
        
    def _extract_commands(self, mock_run: MagicMock) -> list[list[str]]:
        return [call.args[0] for call in mock_run.call_args_list]

    def test_custom_output_dir(self):
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py", "--analyze", "--simulate", "--output-dir", "results/v2"]), \
             patch("sys.executable", "python"):
            
            self._reload_and_run([], mock_run)
            commands = self._extract_commands(mock_run)
            for cmd in commands:
                output_args = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--output"]
                for out_path in output_args:
                    assert "results/v2" in out_path

    def test_custom_input_path(self):
        with patch("subprocess.run") as mock_run, \
             patch.object(sys, "argv", ["run.py", "--generate", "--import", "--input", "data/custom.json"]), \
             patch("sys.executable", "python"):
            
            self._reload_and_run([], mock_run)
            commands = self._extract_commands(mock_run)
            
            # Helper to find cmd by script name
            def find(script): return next((c for c in commands if script in c[1]), [])
            
            gen_cmd = find("generate_graph.py")
            imp_cmd = find("import_graph.py")
            
            assert any("custom.json" in a for a in gen_cmd)
            assert any("custom.json" in a for a in imp_cmd)


# =============================================================================
# Benchmark CLI Tests (bin/benchmark.py)
# =============================================================================

class TestBenchmarkCLI:
    """Tests for bin/benchmark.py"""
    
    @pytest.fixture
    def tmp_output(self, tmp_path):
        return tmp_path / "benchmark_output"

    def _run_main(self, argv: list[str]) -> int:
        with patch.object(sys, "argv", ["benchmark.py"] + argv):
            if 'benchmark' in sys.modules:
                del sys.modules['benchmark']
            import benchmark
            importlib.reload(benchmark)
            return benchmark.main()
    
    @patch("src.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("src.benchmark.runner.BenchmarkRunner.close")
    def test_scales_flag(self, mock_close, mock_run, tmp_output):
        mock_run.return_value = []
        ret = self._run_main(
            ["--scales", "tiny,small", "--runs", "1", "--output", str(tmp_output)]
        )
        assert ret == 0
        assert mock_run.call_count == 2
        scenario_names = [call.args[0].name for call in mock_run.call_args_list]
        assert "tiny" in scenario_names
        assert "small" in scenario_names

    @patch("src.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("src.benchmark.runner.BenchmarkRunner.close")
    def test_full_suite_flag(self, mock_close, mock_run, tmp_output):
        mock_run.return_value = []
        ret = self._run_main(["--full-suite", "--output", str(tmp_output)])
        assert ret == 0
        assert mock_run.call_count == 3

    @patch("src.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("src.benchmark.runner.BenchmarkRunner.close")
    def test_default_scenario(self, mock_close, mock_run, tmp_output):
        mock_run.return_value = []
        ret = self._run_main(["--output", str(tmp_output)])
        assert ret == 0
        assert mock_run.call_count == 1
        assert mock_run.call_args[0][0].scale == "medium"

    @patch("src.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("src.benchmark.runner.BenchmarkRunner.close")
    def test_layers_forwarded(self, mock_close, mock_run, tmp_output):
        mock_run.return_value = []
        ret = self._run_main(
            ["--scales", "tiny", "--layers", "app,infra", "--output", str(tmp_output)]
        )
        assert ret == 0
        scenario = mock_run.call_args[0][0]
        assert scenario.layers == ["app", "infra"]

    @patch("src.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("src.benchmark.runner.BenchmarkRunner.close")
    def test_runs_forwarded(self, mock_close, mock_run, tmp_output):
        mock_run.return_value = []
        ret = self._run_main(
            ["--scales", "tiny", "--runs", "5", "--output", str(tmp_output)]
        )
        assert ret == 0
        scenario = mock_run.call_args[0][0]
        assert scenario.runs == 5

    @patch("src.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("src.benchmark.runner.BenchmarkRunner.close")
    def test_reports_generated(self, mock_close, mock_run, tmp_output):
        mock_run.return_value = []
        ret = self._run_main(["--scales", "tiny", "--output", str(tmp_output)])
        assert ret == 0
        assert (tmp_output / "benchmark_results.json").exists()
        assert (tmp_output / "benchmark_report.md").exists()

    @patch("src.benchmark.runner.BenchmarkRunner.run_scenario")
    @patch("src.benchmark.runner.BenchmarkRunner.close")
    def test_neo4j_args_forwarded(self, mock_close, mock_run, tmp_output):
        mock_run.return_value = []
        with patch("src.benchmark.runner.create_repository") as MockCreateRepo:
            ret = self._run_main([
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
