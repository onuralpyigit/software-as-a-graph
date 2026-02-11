"""
Tests for bin/run.py — Pipeline Orchestrator

Tests verify that the orchestrator delegates to the correct sub-scripts
with the correct arguments, without executing them for real.
"""
import sys
import importlib
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project paths are available
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "bin"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_and_run(argv: list[str], mock_run: MagicMock) -> int:
    """Reload run module with patched sys.argv and execute main()."""
    mock_run.return_value.returncode = 0
    import run
    importlib.reload(run)
    return run.main()


def _extract_commands(mock_run: MagicMock) -> list[list[str]]:
    """Extract command lists from subprocess.run mock calls."""
    return [call.args[0] for call in mock_run.call_args_list]


def _find_cmd(commands: list[list[str]], script_name: str) -> list[str] | None:
    """Find the first command that invokes the given script."""
    return next((cmd for cmd in commands if script_name in cmd[1]), None)


# ---------------------------------------------------------------------------
# Full Pipeline (--all)
# ---------------------------------------------------------------------------

class TestAllStages:
    """Tests for --all flag running the complete pipeline."""

    def test_all_stages_called(self):
        """--all should invoke all 6 sub-scripts in order."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(sys, "argv", ["run.py", "--all", "--scale", "tiny", "--clean"]),
            patch("sys.executable", "python"),
        ):
            ret = _reload_and_run([], mock_run)
            assert ret == 0

            commands = _extract_commands(mock_run)
            scripts = [cmd[1].split("/")[-1] for cmd in commands]

            assert "generate_graph.py" in scripts[0]
            assert "import_graph.py" in scripts[1]
            assert "analyze_graph.py" in scripts[2]
            assert "simulate_graph.py" in scripts[3]
            assert "validate_graph.py" in scripts[4]
            assert "visualize_graph.py" in scripts[5]

    def test_generate_args(self):
        """Generate stage should receive --scale, --seed, --output."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(sys, "argv", ["run.py", "--all", "--scale", "tiny"]),
            patch("sys.executable", "python"),
        ):
            _reload_and_run([], mock_run)
            cmd = _find_cmd(_extract_commands(mock_run), "generate_graph.py")

            assert cmd is not None
            assert "--scale" in cmd
            assert "tiny" in cmd
            assert "--output" in cmd
            assert "--seed" in cmd

    def test_generate_with_config(self):
        """--config should replace --scale for generation."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(sys, "argv", ["run.py", "--all", "--config", "conf/ros2.yaml"]),
            patch("sys.executable", "python"),
        ):
            _reload_and_run([], mock_run)
            cmd = _find_cmd(_extract_commands(mock_run), "generate_graph.py")

            assert "--config" in cmd
            assert "conf/ros2.yaml" in cmd
            assert "--scale" not in cmd

    def test_import_with_clean(self):
        """--clean should forward as --clear to import_graph.py."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(sys, "argv", ["run.py", "--all", "--clean"]),
            patch("sys.executable", "python"),
        ):
            _reload_and_run([], mock_run)
            cmd = _find_cmd(_extract_commands(mock_run), "import_graph.py")

            assert cmd is not None
            assert "--clear" in cmd
            assert "--input" in cmd

    def test_neo4j_args_forwarded(self):
        """Neo4j connection args should be forwarded to all non-generate scripts."""
        uri, user, pw = "bolt://db:7687", "admin", "secret"
        with (
            patch("subprocess.run") as mock_run,
            patch.object(
                sys, "argv",
                ["run.py", "--all", "--uri", uri, "--user", user, "--password", pw],
            ),
            patch("sys.executable", "python"),
        ):
            _reload_and_run([], mock_run)
            commands = _extract_commands(mock_run)

            # Skip generate (index 0) — it doesn't need neo4j
            for cmd in commands[1:]:
                assert "--uri" in cmd and uri in cmd
                assert "--user" in cmd and user in cmd
                assert "--password" in cmd and pw in cmd

    def test_analyze_uses_all_flag(self):
        """With default multi-layer, analyze should use --all."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(sys, "argv", ["run.py", "--all"]),
            patch("sys.executable", "python"),
        ):
            _reload_and_run([], mock_run)
            cmd = _find_cmd(_extract_commands(mock_run), "analyze_graph.py")

            # Default layers are "app,infra,mw" → multi-layer → --all
            assert "--all" in cmd

    def test_simulate_report_subcommand(self):
        """Simulate should use the 'report' subcommand."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(sys, "argv", ["run.py", "--all"]),
            patch("sys.executable", "python"),
        ):
            _reload_and_run([], mock_run)
            cmd = _find_cmd(_extract_commands(mock_run), "simulate_graph.py")

            assert "report" in cmd
            assert "--layers" in cmd

    def test_pipeline_aborts_on_failure(self):
        """Pipeline should stop and return 1 if a stage fails."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(sys, "argv", ["run.py", "--all", "--scale", "tiny"]),
            patch("sys.executable", "python"),
        ):
            # First call (generate) succeeds, second (import) fails
            mock_run.side_effect = [
                MagicMock(returncode=0),
                MagicMock(returncode=1),
            ]

            import run
            importlib.reload(run)
            ret = run.main()

            assert ret == 1
            assert mock_run.call_count == 2  # stopped after import failure


# ---------------------------------------------------------------------------
# Single-Stage Execution
# ---------------------------------------------------------------------------

class TestSingleStages:
    """Tests for running individual stages."""

    def test_analyze_only(self):
        """--analyze should only invoke analyze_graph.py."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(sys, "argv", ["run.py", "--analyze"]),
            patch("sys.executable", "python"),
        ):
            _reload_and_run([], mock_run)
            commands = _extract_commands(mock_run)

            assert len(commands) == 1
            assert "analyze_graph.py" in commands[0][1]

    def test_validate_only(self):
        """--validate should only invoke validate_graph.py."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(sys, "argv", ["run.py", "--validate"]),
            patch("sys.executable", "python"),
        ):
            _reload_and_run([], mock_run)
            commands = _extract_commands(mock_run)

            assert len(commands) == 1
            assert "validate_graph.py" in commands[0][1]

    def test_multi_stage_selection(self):
        """Selecting multiple stages should invoke only those."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(sys, "argv", ["run.py", "--analyze", "--simulate", "--validate"]),
            patch("sys.executable", "python"),
        ):
            _reload_and_run([], mock_run)
            commands = _extract_commands(mock_run)

            scripts = [cmd[1].split("/")[-1] for cmd in commands]
            assert len(scripts) == 3
            assert "analyze_graph.py" in scripts[0]
            assert "simulate_graph.py" in scripts[1]
            assert "validate_graph.py" in scripts[2]

    def test_no_stage_prints_help(self):
        """Running with no stage flags should print help and return 1."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(sys, "argv", ["run.py"]),
            patch("sys.executable", "python"),
        ):
            import run
            importlib.reload(run)
            ret = run.main()

            assert ret == 1
            assert mock_run.call_count == 0


# ---------------------------------------------------------------------------
# Layer Handling
# ---------------------------------------------------------------------------

class TestLayerHandling:
    """Tests for --layer/--layers argument mapping."""

    def test_single_layer_uses_layer_flag(self):
        """Single layer should pass --layer (not --all) to analyze."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(sys, "argv", ["run.py", "--analyze", "--layer", "system"]),
            patch("sys.executable", "python"),
        ):
            _reload_and_run([], mock_run)
            cmd = _find_cmd(_extract_commands(mock_run), "analyze_graph.py")

            assert "--layer" in cmd
            assert "system" in cmd
            assert "--all" not in cmd

    def test_multi_layer_uses_all_flag(self):
        """Multiple layers should pass --all to analyze."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(sys, "argv", ["run.py", "--analyze", "--layer", "app,infra"]),
            patch("sys.executable", "python"),
        ):
            _reload_and_run([], mock_run)
            cmd = _find_cmd(_extract_commands(mock_run), "analyze_graph.py")

            assert "--all" in cmd

    def test_visualize_single_layer(self):
        """Single layer should pass --layer to visualize."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(sys, "argv", ["run.py", "--visualize", "--layer", "app"]),
            patch("sys.executable", "python"),
        ):
            _reload_and_run([], mock_run)
            cmd = _find_cmd(_extract_commands(mock_run), "visualize_graph.py")

            assert "--layer" in cmd
            assert "app" in cmd

    def test_visualize_multi_layer(self):
        """Multiple layers should pass --layers to visualize."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(sys, "argv", ["run.py", "--visualize", "--layer", "app,infra"]),
            patch("sys.executable", "python"),
        ):
            _reload_and_run([], mock_run)
            cmd = _find_cmd(_extract_commands(mock_run), "visualize_graph.py")

            assert "--layers" in cmd


# ---------------------------------------------------------------------------
# Options Passthrough
# ---------------------------------------------------------------------------

class TestOptionsPassthrough:
    """Tests for flag forwarding to sub-scripts."""

    def test_use_ahp_forwarded(self):
        """--use-ahp should be forwarded to analyze_graph.py."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(sys, "argv", ["run.py", "--analyze", "--use-ahp"]),
            patch("sys.executable", "python"),
        ):
            _reload_and_run([], mock_run)
            cmd = _find_cmd(_extract_commands(mock_run), "analyze_graph.py")

            assert "--use-ahp" in cmd

    def test_verbose_forwarded(self):
        """--verbose should be forwarded to analysis/simulation/validation."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(
                sys, "argv",
                ["run.py", "--analyze", "--simulate", "--validate", "--verbose"],
            ),
            patch("sys.executable", "python"),
        ):
            _reload_and_run([], mock_run)
            commands = _extract_commands(mock_run)

            for cmd in commands:
                assert "--verbose" in cmd

    def test_open_forwarded_to_visualize(self):
        """--open should be forwarded to visualize_graph.py."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(sys, "argv", ["run.py", "--visualize", "--open", "--layer", "app"]),
            patch("sys.executable", "python"),
        ):
            _reload_and_run([], mock_run)
            cmd = _find_cmd(_extract_commands(mock_run), "visualize_graph.py")

            assert "--open" in cmd


# ---------------------------------------------------------------------------
# Dry Run
# ---------------------------------------------------------------------------

class TestDryRun:
    """Tests for --dry-run mode."""

    def test_dry_run_no_subprocess(self):
        """--dry-run should not invoke any subprocess."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(sys, "argv", ["run.py", "--all", "--dry-run"]),
            patch("sys.executable", "python"),
        ):
            import run
            importlib.reload(run)
            ret = run.main()

            assert ret == 0
            assert mock_run.call_count == 0


# ---------------------------------------------------------------------------
# Output Path Handling
# ---------------------------------------------------------------------------

class TestOutputPaths:
    """Tests for output directory and file paths."""

    def test_custom_output_dir(self):
        """--output-dir should be used in output paths for all stages."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(
                sys, "argv",
                ["run.py", "--analyze", "--simulate", "--output-dir", "results/v2"],
            ),
            patch("sys.executable", "python"),
        ):
            _reload_and_run([], mock_run)
            commands = _extract_commands(mock_run)

            for cmd in commands:
                output_args = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--output"]
                for out_path in output_args:
                    assert "results/v2" in out_path

    def test_custom_input_path(self):
        """--input should be forwarded to both generate and import."""
        with (
            patch("subprocess.run") as mock_run,
            patch.object(
                sys, "argv",
                ["run.py", "--generate", "--import", "--input", "data/custom.json"],
            ),
            patch("sys.executable", "python"),
        ):
            _reload_and_run([], mock_run)
            commands = _extract_commands(mock_run)

            gen_cmd = _find_cmd(commands, "generate_graph.py")
            imp_cmd = _find_cmd(commands, "import_graph.py")

            # Both should reference the custom path
            assert any("custom.json" in a for a in gen_cmd)
            assert any("custom.json" in a for a in imp_cmd)