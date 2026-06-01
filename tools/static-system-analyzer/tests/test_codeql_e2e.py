"""End-to-end test: run CodeQL strategy against Maven scenario projects."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from common.runtime_config import (
    CodeQLMethodConfig,
    CodeQLRuntimeConfig,
    AnalyzerRuntimeConfig,
)
from pipeline.analyzer.strategies.codeql import CodeQLStrategy
from unittest.mock import patch, MagicMock

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "codeql_test" / "test_platform"

BUILD_CMD = "gmake clean all"


def _make_runtime():
    codeql_cfg = CodeQLRuntimeConfig(
        cli_path="codeql",
        build_command=BUILD_CMD,
        main_method_name="main",
        topic_name_pattern=r"(.*)_class",
        write_methods=(CodeQLMethodConfig("CustomWriter", "custom_write", 0),),
        read_methods=(CodeQLMethodConfig("CustomReader", "custom_read", 0),),
    )
    analyzer_cfg = AnalyzerRuntimeConfig(
        dependency_suffixes=("_lib",),
        makefile_include_patterns=("include/Makefile_java.mk",),
        codeql=codeql_cfg,
    )
    mock_rc = MagicMock()
    mock_rc.analyzer = analyzer_cfg
    return mock_rc


def _run_scenario(name: str) -> set:
    folder = f"scenario_{name}_1.0.0"
    path = FIXTURES / folder
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")

    with patch("pipeline.analyzer.strategies.codeql.get_runtime_config") as mock_rc:
        mock_rc.return_value = _make_runtime()
        strategy = CodeQLStrategy()
        entries = strategy.extract(path, f"scenario_{name}")

    tuples = {(e.name, e.role) for e in entries}
    print(f"  Entries: {tuples}")
    return tuples


def check(label, result):
    status = "PASS" if result else "FAIL"
    print(f"  [{status}] {label}")
    if not result:
        raise AssertionError(f"FAILED: {label}")


def main():
    passed = 0
    failed = 0
    total = 6

    # ── 1. Direct ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. DIRECT: main → custom_write(Alpha), custom_read(Beta)")
    print("=" * 60)
    try:
        r = _run_scenario("direct")
        check("Alpha=pub", ("Alpha", "pub") in r)
        check("Beta=sub", ("Beta", "sub") in r)
        passed += 1
    except Exception as e:
        print(f"  ERROR: {e}")
        failed += 1

    # ── 2. Callback ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. CALLBACK: main → lambda → custom_write(Alpha)")
    print("=" * 60)
    try:
        r = _run_scenario("callback")
        check("Alpha=pub", ("Alpha", "pub") in r)
        passed += 1
    except Exception as e:
        print(f"  ERROR: {e}")
        failed += 1

    # ── 3. Polymorphism ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. POLYMORPHISM: main → interface → concrete → custom_write(Alpha)")
    print("=" * 60)
    try:
        r = _run_scenario("polymorphism")
        check("Alpha=pub", ("Alpha", "pub") in r)
        passed += 1
    except Exception as e:
        print(f"  ERROR: {e}")
        failed += 1

    # ── 4. Nested ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. NESTED: main → A → B → C → custom_write(Alpha)")
    print("=" * 60)
    try:
        r = _run_scenario("nested")
        check("Alpha=pub", ("Alpha", "pub") in r)
        passed += 1
    except Exception as e:
        print(f"  ERROR: {e}")
        failed += 1

    # ── 5. Factory ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("5. FACTORY: main → factory.create().custom_write(Alpha/Beta)")
    print("=" * 60)
    try:
        r = _run_scenario("factory")
        check("Alpha=pub", ("Alpha", "pub") in r)
        check("Beta=pub", ("Beta", "pub") in r)
        passed += 1
    except Exception as e:
        print(f"  ERROR: {e}")
        failed += 1

    # ── 6. Reflection ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("6. REFLECTION: main → Method.invoke() (NOT detectable)")
    print("=" * 60)
    try:
        r = _run_scenario("reflection")
        check("no entries", len(r) == 0)
        passed += 1
    except Exception as e:
        print(f"  ERROR: {e}")
        failed += 1

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
