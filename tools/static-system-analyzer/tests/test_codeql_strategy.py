"""Tests for the CodeQL analysis strategy module."""

import os
import sys
import textwrap
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common.runtime_config import (
    CodeQLMethodConfig,
    CodeQLRuntimeConfig,
    AnalyzerRuntimeConfig,
    _build_codeql_config,
)
from pipeline.analyzer.strategies.codeql import (
    _render_query,
    _extract_topic_name,
    _identify_lib_package,
    _parse_codeql_csv,
    CodeQLStrategy,
)
from pipeline.analyzer.strategies.manual import ManualStrategy
from pipeline.analyzer.models import TopicEntry


# ── runtime config parsing ───────────────────────────────────────────────────

class TestCodeQLConfigParsing:

    def test_build_codeql_config_full(self):
        data = {
            "cli_path": "/opt/codeql/codeql",
            "main_method_name": "entryPoint",
            "topic_name_pattern": "(.*)_type",
            "write_methods": [
                {"class_name": "W1", "method_name": "write1", "topic_arg_index": 0},
                {"class_name": "W2", "method_name": "write2", "topic_arg_index": 2},
            ],
            "read_methods": [
                {"class_name": "R1", "method_name": "read1", "topic_arg_index": 1},
            ],
        }
        cfg = _build_codeql_config(data)
        assert cfg.cli_path == "/opt/codeql/codeql"
        assert cfg.main_method_name == "entryPoint"
        assert cfg.topic_name_pattern == "(.*)_type"
        assert len(cfg.write_methods) == 2
        assert cfg.write_methods[0].class_name == "W1"
        assert cfg.write_methods[1].topic_arg_index == 2
        assert len(cfg.read_methods) == 1
        assert cfg.read_methods[0].topic_arg_index == 1

    def test_build_codeql_config_defaults(self):
        cfg = _build_codeql_config({})
        assert cfg.cli_path == "codeql"
        assert cfg.main_method_name == "main"
        assert cfg.topic_name_pattern == "(.*)"
        assert cfg.write_methods == ()
        assert cfg.read_methods == ()

    def test_analyzer_config_includes_codeql(self):
        cfg = AnalyzerRuntimeConfig()
        assert isinstance(cfg.codeql, CodeQLRuntimeConfig)


# ── query rendering ──────────────────────────────────────────────────────────

class TestQueryRendering:

    def _make_cfg(self, **kwargs):
        return CodeQLRuntimeConfig(
            main_method_name=kwargs.get("main", "main"),
            topic_name_pattern=kwargs.get("pattern", "(.*)_class"),
            write_methods=kwargs.get("writes", (
                CodeQLMethodConfig("WriterA", "custom_write", 0),
            )),
            read_methods=kwargs.get("reads", (
                CodeQLMethodConfig("ReaderA", "custom_read", 0),
            )),
        )

    def test_query_contains_main_method(self):
        cfg = self._make_cfg(main="startHere")
        q = _render_query(cfg)
        assert 'this.hasName("startHere")' in q

    def test_query_contains_write_predicate(self):
        cfg = self._make_cfg()
        q = _render_query(cfg)
        assert 'isPubSink_0' in q
        assert '"custom_write"' in q
        assert '"WriterA"' in q

    def test_query_contains_read_predicate(self):
        cfg = self._make_cfg()
        q = _render_query(cfg)
        assert 'isSubSink_0' in q
        assert '"custom_read"' in q
        assert '"ReaderA"' in q

    def test_query_multiple_writes(self):
        cfg = self._make_cfg(writes=(
            CodeQLMethodConfig("W1", "w1", 0),
            CodeQLMethodConfig("W2", "w2", 1),
        ))
        q = _render_query(cfg)
        assert 'isPubSink_0' in q
        assert 'isPubSink_1' in q
        assert 'getArgument(1)' in q

    def test_query_empty_writes(self):
        cfg = self._make_cfg(writes=())
        q = _render_query(cfg)
        assert 'none()' in q


# ── topic name extraction ────────────────────────────────────────────────────

class TestTopicNameExtraction:

    def test_basic_pattern(self):
        assert _extract_topic_name("Alpha_class", r"(.*)_class") == "Alpha"

    def test_no_match(self):
        assert _extract_topic_name("SomeOther", r"(.*)_class") is None

    def test_complex_pattern(self):
        assert _extract_topic_name("MyTopic_type", r"(.*)_type") == "MyTopic"

    def test_passthrough_pattern(self):
        assert _extract_topic_name("Anything", r"(.*)") == "Anything"


# ── lib detection ────────────────────────────────────────────────────────────

class TestLibDetection:

    def test_lib_suffix_match(self):
        assert _identify_lib_package("com.example.helper_lib.internal", ("_lib",)) == "helper_lib"

    def test_no_lib_suffix(self):
        assert _identify_lib_package("com.example.app.core", ("_lib",)) is None

    def test_multiple_suffixes(self):
        assert _identify_lib_package("com.my_api.handler", ("_lib", "_api")) == "my_api"


# ── CSV parsing ──────────────────────────────────────────────────────────────

class TestCodeQLCSVParsing:

    def test_parse_basic_csv(self, tmp_path):
        csv_file = tmp_path / "results.csv"
        csv_file.write_text(
            'col1,col2,col3,message\n'
            '"unused","unused","unused","pub|Alpha_class|com.example.app_alpha"\n'
            '"unused","unused","unused","sub|Beta_class|com.example.app_alpha"\n'
            '"unused","unused","unused","sub|Gamma_class|com.example.helper_lib.internal"\n'
        )
        entries = _parse_codeql_csv(csv_file, "app_alpha", r"(.*)_class", ("_lib",))

        # Should get: app_alpha,Alpha,pub + app_alpha,Beta,sub + app_alpha,Gamma,sub + app_alpha,helper_lib,uses
        names = [(e.source_folder, e.name, e.role) for e in entries]
        assert ("app_alpha", "Alpha", "pub") in names
        assert ("app_alpha", "Beta", "sub") in names
        assert ("app_alpha", "Gamma", "sub") in names
        assert ("app_alpha", "helper_lib", "uses") in names
        assert len(entries) == 4

    def test_parse_empty_csv(self, tmp_path):
        csv_file = tmp_path / "results.csv"
        csv_file.write_text("")
        entries = _parse_codeql_csv(csv_file, "app", r"(.*)_class", ("_lib",))
        assert entries == []

    def test_parse_missing_file(self, tmp_path):
        csv_file = tmp_path / "nonexistent.csv"
        entries = _parse_codeql_csv(csv_file, "app", r"(.*)_class", ("_lib",))
        assert entries == []

    def test_parse_skips_invalid_roles(self, tmp_path):
        csv_file = tmp_path / "results.csv"
        csv_file.write_text(
            'col1,message\n'
            '"x","invalid|Alpha_class|pkg"\n'
        )
        entries = _parse_codeql_csv(csv_file, "app", r"(.*)_class", ("_lib",))
        assert entries == []

    def test_lib_not_self(self, tmp_path):
        """If the enclosing package is the app itself (matching _lib suffix by chance), skip uses."""
        csv_file = tmp_path / "results.csv"
        csv_file.write_text(
            'col1,message\n'
            '"x","pub|Alpha_class|com.example.my_lib.core"\n'
        )
        entries = _parse_codeql_csv(csv_file, "my_lib", r"(.*)_class", ("_lib",))
        # Should have pub entry but NOT uses (lib == self)
        names = [(e.name, e.role) for e in entries]
        assert ("Alpha", "pub") in names
        assert ("my_lib", "uses") not in names


# ── manual strategy ──────────────────────────────────────────────────────────

class TestManualStrategy:

    def test_extract_from_fixture(self, tmp_path):
        """Verify ManualStrategy works on test fixtures (XML + no imports)."""
        fixtures = Path(__file__).resolve().parent / "fixtures" / "codeql_test" / "test_platform" / "app_alpha_1.0.0"
        if not fixtures.exists():
            pytest.skip("Test fixtures not available")

        strategy = ManualStrategy()
        entries = strategy.extract(fixtures, "app_alpha")

        roles = {(e.name, e.role) for e in entries}
        assert ("Alpha", "pub") in roles
        assert ("Beta", "sub") in roles


# ── CodeQL strategy (mocked CLI) ─────────────────────────────────────────────

class TestCodeQLStrategyMocked:

    def _mock_runtime(self):
        """Return a patched runtime config for CodeQL tests."""
        codeql_cfg = CodeQLRuntimeConfig(
            cli_path="/fake/codeql",
            main_method_name="main",
            topic_name_pattern=r"(.*)_class",
            write_methods=(CodeQLMethodConfig("CustomWriter", "custom_write", 0),),
            read_methods=(CodeQLMethodConfig("CustomReader", "custom_read", 0),),
        )
        analyzer_cfg = AnalyzerRuntimeConfig(
            dependency_suffixes=("_lib",),
            codeql=codeql_cfg,
        )

        mock_rc = MagicMock()
        mock_rc.analyzer = analyzer_cfg
        return mock_rc

    @patch("pipeline.analyzer.strategies.codeql.get_runtime_config")
    @patch("pipeline.analyzer.strategies.codeql._create_database", return_value=True)
    @patch("pipeline.analyzer.strategies.codeql._run_query", return_value=True)
    @patch("pipeline.analyzer.strategies.codeql._decode_results")
    def test_strategy_full_flow(self, mock_decode, mock_query, mock_db, mock_rc, tmp_path):
        """Mock the CLI calls and verify the strategy produces correct entries."""
        mock_rc.return_value = self._mock_runtime()

        # _decode_results writes a CSV file; simulate that
        def fake_decode(cli, bqrs, csv_out, **kwargs):
            csv_out.write_text(
                'col1,col2,col3,message\n'
                '"x","y","z","pub|Alpha_class|com.app.core"\n'
                '"x","y","z","sub|Beta_class|com.app.core"\n'
                '"x","y","z","sub|Gamma_class|com.helper_lib.util"\n'
            )
            return True

        mock_decode.side_effect = fake_decode

        strategy = CodeQLStrategy()
        entries = strategy.extract(tmp_path, "app_alpha")

        tuples = [(e.source_folder, e.name, e.role) for e in entries]
        assert ("app_alpha", "Alpha", "pub") in tuples
        assert ("app_alpha", "Beta", "sub") in tuples
        assert ("app_alpha", "Gamma", "sub") in tuples
        assert ("app_alpha", "helper_lib", "uses") in tuples

    @patch("pipeline.analyzer.strategies.codeql.get_runtime_config")
    @patch("pipeline.analyzer.strategies.codeql._create_database", return_value=True)
    @patch("pipeline.analyzer.strategies.codeql._run_query", return_value=True)
    @patch("pipeline.analyzer.strategies.codeql._decode_results")
    def test_build_command_uses_makefile_C_flag(self, mock_decode, mock_query, mock_db, mock_rc, tmp_path):
        """Build command should use -C <makefile_dir> instead of sh -c cd wrapper."""
        # Create a Makefile with the include pattern so find_makefile finds it
        subdir = tmp_path / "proj" / "build"
        subdir.mkdir(parents=True)
        makefile = subdir / "Makefile"
        makefile.write_text("include/Makefile_java.mk\nall:\n\t@echo ok\n")

        codeql_cfg = CodeQLRuntimeConfig(
            cli_path="/fake/codeql",
            build_command="gmake clean all",
            main_method_name="main",
            topic_name_pattern=r"(.*)_class",
            write_methods=(CodeQLMethodConfig("W", "w", 0),),
            read_methods=(CodeQLMethodConfig("R", "r", 0),),
        )
        analyzer_cfg = AnalyzerRuntimeConfig(
            dependency_suffixes=("_lib",),
            codeql=codeql_cfg,
            makefile_include_patterns=("include/Makefile_java.mk",),
        )
        mock_rt = MagicMock()
        mock_rt.analyzer = analyzer_cfg
        mock_rc.return_value = mock_rt

        def fake_decode(cli, bqrs, csv_out, **kwargs):
            csv_out.write_text('col1,message\n"x","pub|A_class|com.app"\n')
            return True
        mock_decode.side_effect = fake_decode

        strategy = CodeQLStrategy()
        # Use tmp_path/proj as the project folder so makefile is inside it
        strategy.extract(tmp_path / "proj", "test_proj")

        # Verify _create_database was called with -C flag, not sh -c
        call_args = mock_db.call_args
        build_arg = call_args[0][3]  # 4th positional arg = build_command
        assert "-C" in build_arg
        assert "sh -c" not in build_arg
        assert str(subdir) in build_arg
        assert build_arg == f"gmake -C {subdir} clean all"

    @patch("pipeline.analyzer.strategies.codeql.get_runtime_config")
    def test_strategy_no_methods_configured(self, mock_rc, tmp_path):
        """If no write/read methods are configured, return empty."""
        codeql_cfg = CodeQLRuntimeConfig()  # no methods
        analyzer_cfg = AnalyzerRuntimeConfig(codeql=codeql_cfg)
        mock_rt = MagicMock()
        mock_rt.analyzer = analyzer_cfg
        mock_rc.return_value = mock_rt

        strategy = CodeQLStrategy()
        entries = strategy.extract(tmp_path, "app")
        assert entries == []
