#!/usr/bin/env python3
"""
Benchmark Suite for Graph-Based Criticality Prediction

Evaluates the Software-as-a-Graph methodology across varying system scales
and configurations.  For each (scale × layer × seed) combination the pipeline
runs:  generate → import → analyze → simulate → validate, collecting timing
and accuracy metrics.

Usage:
    python cli/benchmark.py --scales tiny,small,medium --runs 3
    python cli/benchmark.py --full-suite
    python cli/benchmark.py --config benchmarks/suite.yaml
    python cli/benchmark.py --scales small --layers app --runs 1 --verbose
"""
import argparse
import logging
import time
from pathlib import Path
from typing import List

from tools.benchmark import (
    BenchmarkRunner,
    BenchmarkScenario,
    ReportGenerator,
)
from cli.common.console import ConsoleDisplay
from cli.common.arguments import add_neo4j_arguments, add_runtime_arguments, setup_logging

# Helper for color-coding terminal output (initialized in main)


# =============================================================================
# Scenario builders
# =============================================================================

def _build_scenarios(args: argparse.Namespace) -> List[BenchmarkScenario]:
    """Build scenarios based on CLI arguments."""
    if args.config:
        return _load_yaml_scenarios(args.config)

    if args.full_suite:
        scales = ["tiny", "small", "medium"]
    elif args.scales:
        scales = [s.strip() for s in args.scales.split(",")]
    else:
        # Default fallback
        scales = ["tiny", "small"]

    layers = [l.strip() for l in args.layers.split(",")]

    return [
        BenchmarkScenario(
            name=f"auto-{scale}",
            scale=scale,
            layers=layers,
            runs=args.runs,
            seed=args.seed,
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
                seed=item.get("seed", 42),
            )
        )

    return scenarios


# =============================================================================
# Progress display
# =============================================================================

def _print_scenario_result(
    display: ConsoleDisplay,
    idx: int,
    total: int,
    scenario: BenchmarkScenario,
    records: list,
) -> None:
    """Print a one-line progress update after a scenario completes."""
    passed = sum(1 for r in records if r.passed)
    n = len(records)
    ratio = f"{passed}/{n}"
    status = display.colored("pass", display.Colors.GREEN) if passed == n else display.colored(f"{ratio}", display.Colors.YELLOW)
    
    # Calculate average time for successfully completed runs in this scenario
    valid_recs = [r for r in records if not r.error]
    avg_time = sum(r.time_total for r in valid_recs) / len(valid_recs) if valid_recs else 0.0

    display.print_step(
        f"[{idx}/{total}] {scenario.name:15s}  "
        f"scale={scenario.label:8s}  "
        f"runs={scenario.runs}  "
        f"avg={avg_time/1000:5.1f}s  "
        f"result={status}"
    )


def _print_summary_table(display: ConsoleDisplay, summary) -> None:
    """Print a compact summary table to the terminal."""
    if not summary.aggregates:
        return

    display.print_header("Benchmark Results Summary")

    # Header
    hdr = (
        f"  {'Scale':<8} {'Layer':<8} {'Runs':>4} {'Pass%':>6} "
        f"{'ρ-Our':>7} {'ρ-BC':>7} {'Gain%':>7} {'F1':>6} {'Dur(s)':>7}"
    )
    print(f"\n{display.colored(hdr, display.Colors.BOLD)}")
    print(f"  {'─' * 70}")

    for a in summary.aggregates:
        sp = f"{a.avg_spearman:.2f}"
        bc = f"{a.avg_spearman_bc:.2f}"
        gain = ((a.avg_spearman - a.avg_spearman_bc) / a.avg_spearman_bc * 100) if a.avg_spearman_bc > 0 else 0.0
        f1 = f"{a.avg_f1:.2f}"
        rate = f"{a.pass_rate:.0f}%"
        
        # Color gain
        gain_str = f"{gain:+.1f}%"
        gain_color = display.Colors.GREEN if gain > 5 else display.Colors.YELLOW if gain >= 0 else display.Colors.RED
        
        pass_color = display.Colors.GREEN if a.pass_rate == 100 else display.Colors.YELLOW if a.pass_rate > 0 else display.Colors.RED
        
        print(
            f"  {a.scale:<8} {a.layer:<8} {a.num_runs:>4} "
            f"{display.colored(f'{rate:>6}', pass_color)} "
            f"{sp:>7} {bc:>7} {display.colored(f'{gain_str:>7}', gain_color)} "
            f"{f1:>6} {a.avg_time_total/1000:>7.1f}"
        )

    print(f"  {'-' * 70}")
    overall = (
        f"  Overall: {summary.total_runs} runs, "
        f"{summary.passed_runs} passed ({summary.overall_pass_rate:.1f}%), "
        f"ρ={summary.overall_spearman:.3f}, F1={summary.overall_f1:.3f}"
    )
    print(f"\n{display.colored(overall, display.Colors.WHITE, bold=True)}")


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
    opts.add_argument(
        "--ndcg-k", type=int, default=10,
        help="K for NDCG@K calculation (default: 10)",
    )
    opts.add_argument(
        "--seed", type=int, default=42,
        help="Base seed for synthetic generation (default: 42)",
    )
    opts.add_argument(
        "--spearman-target", type=float, default=0.70,
        help="Spearman correlation target (default: 0.70)",
    )
    opts.add_argument(
        "--f1-target", type=float, default=0.80,
        help="F1 score target (default: 0.80)",
    )

    # --- Neo4j ---
    add_neo4j_arguments(parser)

    # --- Runtime ---
    add_runtime_arguments(parser)
    parser.add_argument("--dry-run", action="store_true", help="Print plans without executing")

    return parser


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    display = ConsoleDisplay()

    setup_logging(args)

    # --- Build scenario list ---
    scenarios = _build_scenarios(args)

    if not scenarios:
        display.print_error("No scenarios to run.")
        return 1

    # --- Print plan ---
    output_dir = Path(args.output)
    total_runs = sum(s.runs * len(s.layers) for s in scenarios)

    display.print_header("Software-as-a-Graph Benchmark Suite")
    print(f"  Scenarios : {len(scenarios)}")
    print(f"  Total runs: {total_runs} (scenarios × layers × repeats)")
    print(f"  Output    : {display.colored(str(output_dir), display.Colors.CYAN)}")
    
    if args.dry_run:
        print(f"\n  {display.colored('[DRY RUN] Plan confirmed. Exiting.', display.Colors.YELLOW)}")
        for s in scenarios:
            print(f"    - {s.name}: scale={s.label}, layers={s.layers}, runs={s.runs}")
        return 0

    # --- Run ---
    from saag.validation import ValidationTargets
    targets = ValidationTargets(
        spearman=args.spearman_target,
        f1_score=args.f1_target,
    )

    t0 = time.time()

    with BenchmarkRunner(
        output_dir=output_dir,
        uri=args.uri,
        user=args.user,
        password=args.password,
        ndcg_k=args.ndcg_k,
        verbose=args.verbose,
        targets=targets,
    ) as runner:
        for i, scenario in enumerate(scenarios, 1):
            records = runner.run_scenario(scenario)
            _print_scenario_result(display, i, len(scenarios), scenario, records)

        duration = time.time() - t0
        summary = runner.aggregate_results(duration)

    # --- Reports ---
    reporter = ReportGenerator(output_dir)
    json_path = reporter.save_json(summary)
    md_path = reporter.generate_markdown(summary)

    # --- Terminal summary ---
    _print_summary_table(display, summary)

    print(f"\n  Reports saved to:")
    display.print_success(str(json_path))
    display.print_success(str(md_path))
    print(f"\n  Completed in {display.colored(f'{duration:.1f}s', display.Colors.BOLD)}\n")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{_c('Benchmark interrupted by user.', Colors.YELLOW)}")
        sys.exit(130)