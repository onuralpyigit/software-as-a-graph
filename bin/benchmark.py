#!/usr/bin/env python3
"""
Benchmark Suite for Graph-Based Criticality Prediction

Evaluates the Software-as-a-Graph methodology across varying system scales
and configurations.  For each (scale × layer × seed) combination the pipeline
runs:  generate → import → analyze → simulate → validate, collecting timing
and accuracy metrics.

Usage:
    python bin/benchmark.py --scales tiny,small,medium --runs 3
    python bin/benchmark.py --full-suite
    python bin/benchmark.py --config benchmarks/suite.yaml
    python bin/benchmark.py --scales small --layers app --runs 1 --verbose
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import time
from typing import List

from src.benchmark import (
    BenchmarkRunner,
    BenchmarkScenario,
    ReportGenerator,
)


# =============================================================================
# Terminal output helpers (matches run.py style)
# =============================================================================

class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"


def print_header(title: str) -> None:
    line = "=" * 60
    print(f"\n{_c(line, Colors.CYAN)}")
    print(f"{_c(f' {title}', Colors.CYAN + Colors.BOLD)}")
    print(_c(line, Colors.CYAN))


def print_step(msg: str) -> None:
    print(f"  {_c('→', Colors.BLUE)} {msg}")


def print_success(msg: str) -> None:
    print(f"  {_c('✓', Colors.GREEN)} {msg}")


def print_error(msg: str) -> None:
    print(f"  {_c('✗', Colors.RED)} {msg}")


# =============================================================================
# Scenario builders
# =============================================================================

def _build_cli_scenarios(args: argparse.Namespace) -> List[BenchmarkScenario]:
    """Build scenarios from --scales and --layers CLI flags."""
    scales = [s.strip() for s in args.scales.split(",")]
    layers = [l.strip() for l in args.layers.split(",")]

    return [
        BenchmarkScenario(
            name=f"{scale}",
            scale=scale,
            layers=layers,
            runs=args.runs,
        )
        for scale in scales
    ]


def _build_full_suite(args: argparse.Namespace) -> List[BenchmarkScenario]:
    """Build the comprehensive full-suite benchmark."""
    scales = ["tiny", "small", "medium"]
    layers = [l.strip() for l in args.layers.split(",")]

    return [
        BenchmarkScenario(
            name=f"full-{scale}",
            scale=scale,
            layers=layers,
            runs=args.runs,
        )
        for scale in scales
    ]


def _load_yaml_scenarios(config_path: Path) -> List[BenchmarkScenario]:
    """Load benchmark scenarios from a YAML configuration file."""
    try:
        import yaml
    except ImportError:
        print_error("PyYAML is required for --config.  Install with: pip install pyyaml")
        sys.exit(1)

    if not config_path.exists():
        print_error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        data = yaml.safe_load(f)

    scenarios: List[BenchmarkScenario] = []
    for item in data.get("scenarios", []):
        # Resolve optional custom graph-config path relative to the YAML file
        graph_cfg = item.get("graph_config")
        cfg_path = None
        if graph_cfg:
            cfg_path = Path(graph_cfg)
            if not cfg_path.is_absolute():
                cfg_path = config_path.parent / cfg_path

        scenarios.append(
            BenchmarkScenario(
                name=item.get("name", "Unnamed"),
                scale=item.get("scale"),
                config_path=cfg_path,
                layers=item.get("layers", ["app", "infra", "mw", "system"]),
                runs=item.get("runs", 1),
            )
        )

    return scenarios


# =============================================================================
# Progress display
# =============================================================================

def _print_scenario_result(
    idx: int,
    total: int,
    scenario: BenchmarkScenario,
    records: list,
) -> None:
    """Print a one-line progress update after a scenario completes."""
    passed = sum(1 for r in records if r.passed)
    n = len(records)
    ratio = f"{passed}/{n}"
    status = _c("pass", Colors.GREEN) if passed == n else _c(f"{ratio}", Colors.YELLOW)

    print_step(
        f"[{idx}/{total}] {scenario.name:12s}  "
        f"scale={scenario.label:8s}  "
        f"layers={len(scenario.layers)}  "
        f"runs={scenario.runs}  "
        f"result={status}"
    )


def _print_summary_table(summary) -> None:
    """Print a compact summary table to the terminal."""
    if not summary.aggregates:
        return

    print_header("Benchmark Results")

    # Header
    hdr = (
        f"  {'Scale':<10} {'Layer':<8} {'Runs':>5} {'Pass%':>7} "
        f"{'Spearman':>10} {'F1':>8} {'Total(ms)':>10}"
    )
    print(f"\n{_c(hdr, Colors.BOLD)}")
    print(f"  {'─' * 62}")

    for a in summary.aggregates:
        sp = f"{a.avg_spearman:.3f}"
        f1 = f"{a.avg_f1:.3f}"
        rate = f"{a.pass_rate:.0f}%"
        color = Colors.GREEN if a.pass_rate == 100 else Colors.YELLOW if a.pass_rate > 0 else Colors.RED
        print(
            f"  {a.scale:<10} {a.layer:<8} {a.num_runs:>5} "
            f"{_c(rate, color):>16} {sp:>10} {f1:>8} {a.avg_time_total:>10.0f}"
        )

    overall = (
        f"\n  Overall: {summary.total_runs} runs, "
        f"{summary.passed_runs} passed ({summary.overall_pass_rate:.1f}%), "
        f"ρ={summary.overall_spearman:.3f}, F1={summary.overall_f1:.3f}"
    )
    print(_c(overall, Colors.BOLD))


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Software-as-a-Graph Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --scales tiny,small,medium --runs 3     Quick multi-scale benchmark
  %(prog)s --full-suite                             Comprehensive benchmark
  %(prog)s --config benchmarks/suite.yaml           From YAML config
  %(prog)s --scales small --layers app --runs 1     Minimal single run
""",
    )

    # --- Mode ---
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--scales",
        help="Comma-separated scales (tiny,small,medium,large,xlarge)",
    )
    mode.add_argument(
        "--config", type=Path, metavar="FILE",
        help="YAML configuration file defining scenarios",
    )
    mode.add_argument(
        "--full-suite", action="store_true",
        help="Run comprehensive suite (tiny + small + medium)",
    )

    # --- Common options ---
    opts = parser.add_argument_group("Options")
    opts.add_argument(
        "--layers", default="app,infra,mw,system",
        help="Layers to benchmark, comma-separated (default: app,infra,mw,system)",
    )
    opts.add_argument(
        "--runs", type=int, default=1,
        help="Runs per scenario for variance analysis (default: 1)",
    )
    opts.add_argument(
        "--output", "-o", default="results/benchmark", metavar="DIR",
        help="Output directory for reports (default: results/benchmark)",
    )

    # --- Neo4j ---
    neo4j = parser.add_argument_group("Neo4j Connection")
    neo4j.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    neo4j.add_argument("--user", default="neo4j", help="Neo4j username")
    neo4j.add_argument("--password", default="password", help="Neo4j password")

    # --- Runtime ---
    runtime = parser.add_argument_group("Runtime")
    runtime.add_argument("--verbose", "-v", action="store_true",
                         help="Verbose output with debug logging")

    return parser


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # --- Logging ---
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Build scenario list ---
    if args.config:
        scenarios = _load_yaml_scenarios(args.config)
    elif args.full_suite:
        scenarios = _build_full_suite(args)
    elif args.scales:
        scenarios = _build_cli_scenarios(args)
    else:
        # Default: medium scale with requested layers
        scenarios = [
            BenchmarkScenario(
                name="default",
                scale="medium",
                layers=[l.strip() for l in args.layers.split(",")],
                runs=args.runs,
            )
        ]

    if not scenarios:
        print_error("No scenarios to run.")
        return 1

    # --- Print plan ---
    output_dir = Path(args.output)
    total_runs = sum(s.runs * len(s.layers) for s in scenarios)

    print_header("Software-as-a-Graph Benchmark Suite")
    print(f"  Scenarios : {len(scenarios)}")
    print(f"  Total runs: {total_runs} (scenarios × layers × repeats)")
    print(f"  Output    : {output_dir}")

    # --- Run ---
    t0 = time.time()

    with BenchmarkRunner(
        output_dir=output_dir,
        uri=args.uri,
        user=args.user,
        password=args.password,
        verbose=args.verbose,
    ) as runner:
        for i, scenario in enumerate(scenarios, 1):
            records = runner.run_scenario(scenario)
            _print_scenario_result(i, len(scenarios), scenario, records)

        duration = time.time() - t0
        summary = runner.aggregate_results(duration)

    # --- Reports ---
    reporter = ReportGenerator(output_dir)
    json_path = reporter.save_json(summary)
    md_path = reporter.generate_markdown(summary)

    # --- Terminal summary ---
    _print_summary_table(summary)

    print(f"\n  Reports saved to:")
    print_success(str(json_path))
    print_success(str(md_path))
    print(f"\n  Completed in {_c(f'{duration:.1f}s', Colors.BOLD)}\n")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{_c('Benchmark interrupted by user.', Colors.YELLOW)}")
        sys.exit(130)