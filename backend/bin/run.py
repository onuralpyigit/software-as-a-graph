#!/usr/bin/env python3
"""
Software-as-a-Graph Pipeline Orchestrator

Executes the end-to-end analysis pipeline by delegating to specialized CLI scripts.

Pipeline Stages:
    1. Generate   → Create synthetic graph data
    2. Import     → Build graph model in Neo4j
    3. Analyze    → Compute structural metrics (centrality, RMAV)
    4. Simulate   → Run exhaustive failure simulations
    5. Validate   → Compare predictions vs simulation
    6. Visualize  → Generate interactive dashboard

Usage:
    python bin/run.py --all --scale small                      # Full pipeline
    python bin/run.py --all --config config/ros2.yaml          # With custom config
    python bin/run.py --analyze --simulate --validate          # Specific stages
    python bin/run.py --all --layer system --open              # Single layer + open dashboard
    python bin/run.py --all --dry-run                          # Preview commands
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional


# =============================================================================
# Terminal Output Helpers
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
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


def print_warning(msg: str) -> None:
    print(f"  {_c('⚠', Colors.YELLOW)} {msg}")


# =============================================================================
# Script Runner
# =============================================================================

@dataclass
class StageResult:
    """Result of a single pipeline stage execution."""
    name: str
    script: str
    duration: float = 0.0
    success: bool = False
    skipped: bool = False


def run_script(
    script_name: str,
    args: List[str],
    project_root: Path,
    *,
    dry_run: bool = False,
) -> tuple[bool, float]:
    """
    Run a bin/ script as a subprocess.

    Returns:
        (success, duration_seconds)
    """
    script_path = project_root / "bin" / script_name
    if not script_path.exists():
        print_error(f"Script not found: bin/{script_name}")
        return False, 0.0

    cmd = [sys.executable, str(script_path)] + args
    display_cmd = f"python bin/{script_name} {' '.join(args)}"

    if dry_run:
        print_step(f"{_c('[dry-run]', Colors.YELLOW)} {_c(display_cmd, Colors.GRAY)}")
        return True, 0.0

    print_step(f"Executing: {_c(display_cmd, Colors.GRAY)}")

    try:
        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(project_root), check=False)
        duration = time.time() - t0

        if result.returncode == 0:
            print_success(f"Completed in {duration:.2f}s")
            return True, duration
        else:
            print_error(f"Failed (exit code {result.returncode})")
            return False, duration

    except Exception as e:
        print_error(f"Execution error: {e}")
        return False, 0.0


# =============================================================================
# Stage Builders — each returns the argument list for its script
# =============================================================================

def _build_generate_args(args: argparse.Namespace) -> List[str]:
    """Build arguments for generate_graph.py."""
    cmd = ["--output", args.input]
    if args.config:
        cmd += ["--config", args.config]
    else:
        cmd += ["--scale", args.scale, "--seed", str(args.seed)]
    return cmd


def _build_import_args(args: argparse.Namespace, neo4j: List[str]) -> List[str]:
    """Build arguments for import_graph.py."""
    cmd = ["--input", args.input]
    if args.clean:
        cmd.append("--clear")
    return cmd + neo4j


def _build_analyze_args(args: argparse.Namespace, neo4j: List[str]) -> List[str]:
    """Build arguments for analyze_graph.py."""
    layers = [l.strip() for l in args.layers.split(",") if l.strip()]

    # Use --all for multi-layer, --layer for single
    if len(layers) > 1:
        cmd = ["--all"]
    else:
        cmd = ["--layer", layers[0]]

    cmd += ["--output", str(Path(args.output_dir) / "analysis_results.json")]
    if args.use_ahp:
        cmd.append("--use-ahp")
    if args.verbose:
        cmd.append("--verbose")
    return cmd + neo4j


def _build_simulate_args(args: argparse.Namespace, neo4j: List[str]) -> List[str]:
    """Build arguments for simulate_graph.py (report subcommand)."""
    cmd = ["report", "--layers", args.layers]
    cmd += ["--output", str(Path(args.output_dir) / "simulation_report.json")]
    if args.verbose:
        cmd.append("--verbose")
    return cmd + neo4j


def _build_validate_args(args: argparse.Namespace, neo4j: List[str]) -> List[str]:
    """Build arguments for validate_graph.py."""
    cmd = ["--layer", args.layers]
    cmd += ["--output", str(Path(args.output_dir) / "validation_results.json")]
    if args.verbose:
        cmd.append("--verbose")
    return cmd + neo4j


def _build_visualize_args(args: argparse.Namespace, neo4j: List[str]) -> List[str]:
    """Build arguments for visualize_graph.py."""
    layers = [l.strip() for l in args.layers.split(",") if l.strip()]

    if len(layers) > 1:
        cmd = ["--layers", args.layers]
    else:
        cmd = ["--layer", layers[0]]

    cmd += ["--output", str(Path(args.output_dir) / "dashboard.html")]
    if args.open:
        cmd.append("--open")
    return cmd + neo4j


# =============================================================================
# Pipeline Definition
# =============================================================================

# (flag_name, stage_label, script_file, args_builder)
STAGES = [
    ("generate",  "Generation",    "generate_graph.py",   _build_generate_args),
    ("do_import", "Import",        "import_graph.py",     _build_import_args),
    ("analyze",   "Analysis",      "analyze_graph.py",    _build_analyze_args),
    ("simulate",  "Simulation",    "simulate_graph.py",   _build_simulate_args),
    ("validate",  "Validation",    "validate_graph.py",   _build_validate_args),
    ("visualize", "Visualization", "visualize_graph.py",  _build_visualize_args),
]


def print_summary(results: List[StageResult], total_time: float) -> None:
    """Print a timing summary table for all executed stages."""
    print_header("Pipeline Summary")

    # Column widths
    w_idx, w_stage, w_time, w_status = 4, 20, 10, 8

    header = (
        f"  {'#':<{w_idx}} {'Stage':<{w_stage}} {'Time':>{w_time}} {'Status':>{w_status}}"
    )
    print(f"\n{_c(header, Colors.BOLD)}")
    print(f"  {'─' * (w_idx + w_stage + w_time + w_status + 3)}")

    for i, r in enumerate(results, 1):
        if r.skipped:
            time_str = "—"
            status = _c("skip", Colors.GRAY)
        elif r.success:
            time_str = f"{r.duration:.2f}s"
            status = _c("pass", Colors.GREEN)
        else:
            time_str = f"{r.duration:.2f}s"
            status = _c("FAIL", Colors.RED)

        print(
            f"  {i:<{w_idx}} {r.name:<{w_stage}} {time_str:>{w_time}} {status:>{w_status + 9}}"
            # +9 accounts for ANSI escape chars in status
        )

    print(f"\n  Total: {_c(f'{total_time:.2f}s', Colors.BOLD)}")


# =============================================================================
# CLI & Main
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Software-as-a-Graph Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --all --scale small                  Full pipeline, small graph
  %(prog)s --all --config config/ros2.yaml      Full pipeline, custom config
  %(prog)s --analyze --simulate --validate      Run only 3 stages
  %(prog)s --all --layer system --open          System layer + open dashboard
  %(prog)s --all --dry-run                      Preview commands only
""",
    )

    # --- Pipeline stage flags ---
    stages = parser.add_argument_group("Pipeline Stages")
    stages.add_argument("--all", "-a", action="store_true",
                        help="Run all 6 stages")
    stages.add_argument("--generate", "-g", action="store_true",
                        help="Stage 1: Generate synthetic data")
    stages.add_argument("--import", "-i", dest="do_import", action="store_true",
                        help="Stage 2: Import data into Neo4j")
    stages.add_argument("--analyze", "-A", action="store_true",
                        help="Stage 3: Structural + quality analysis")
    stages.add_argument("--simulate", "-s", action="store_true",
                        help="Stage 4: Exhaustive failure simulation")
    stages.add_argument("--validate", "-V", action="store_true",
                        help="Stage 5: Validate predictions vs simulation")
    stages.add_argument("--visualize", "-z", action="store_true",
                        help="Stage 6: Generate dashboard")

    # --- Data options ---
    data = parser.add_argument_group("Data Options")
    data.add_argument("--config", metavar="FILE",
                      help="Graph generation config (YAML)")
    data.add_argument("--scale", default="medium",
                      choices=["tiny", "small", "medium", "large", "xlarge"],
                      help="Graph scale preset (default: medium)")
    data.add_argument("--seed", type=int, default=42,
                      help="Random seed for generation (default: 42)")
    data.add_argument("--input", default="output/system.json",
                      help="Data file path — output of generate, input to import (default: output/system.json)")
    data.add_argument("--output-dir", default="output", metavar="DIR",
                      help="Output directory for all artifacts (default: output)")

    # --- Analysis options ---
    analysis = parser.add_argument_group("Analysis Options")
    analysis.add_argument("--layer", "--layers", dest="layers",
                          default="app,infra,mw",
                          help="Layers to process, comma-separated (default: app,infra,mw)")
    analysis.add_argument("--use-ahp", action="store_true",
                          help="Use AHP-derived weights for quality scoring")
    analysis.add_argument("--clean", "--clear", dest="clean", action="store_true",
                          help="Clear Neo4j database before import")

    # --- Neo4j connection ---
    neo4j = parser.add_argument_group("Neo4j Connection")
    neo4j.add_argument("--uri", default="bolt://localhost:7687",
                       help="Neo4j Bolt URI")
    neo4j.add_argument("--user", default="neo4j",
                       help="Neo4j username")
    neo4j.add_argument("--password", default="password",
                       help="Neo4j password")

    # --- Runtime options ---
    runtime = parser.add_argument_group("Runtime Options")
    runtime.add_argument("--dry-run", action="store_true",
                         help="Print commands without executing")
    runtime.add_argument("--verbose", "-v", action="store_true",
                         help="Enable verbose logging in sub-scripts")
    runtime.add_argument("--open", "-O", action="store_true",
                         help="Open dashboard in browser after visualization")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Resolve project root and ensure output directory exists
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check that at least one stage is selected
    run_all = args.all
    any_stage = run_all or any(
        getattr(args, flag) for flag, *_ in STAGES
    )
    if not any_stage:
        parser.print_help()
        print(f"\n{_c('Error: specify --all or at least one stage flag.', Colors.RED)}")
        return 1

    # Common Neo4j arguments
    neo4j_args = ["--uri", args.uri, "--user", args.user, "--password", args.password]

    # Resolve relative input path against project root
    if not Path(args.input).is_absolute():
        args.input = str(project_root / args.input)
    if not Path(args.output_dir).is_absolute():
        args.output_dir = str(project_root / args.output_dir)

    print_header("Software-as-a-Graph Pipeline")
    print(f"  Scale : {args.scale}   Layers: {args.layers}")
    print(f"  Output: {args.output_dir}")
    if args.dry_run:
        print(f"  {_c('DRY RUN — no commands will be executed', Colors.YELLOW)}")

    # --- Execute stages ---
    results: List[StageResult] = []
    t_pipeline = time.time()

    for flag_name, label, script, builder in STAGES:
        enabled = run_all or getattr(args, flag_name, False)
        stage_num = len(results) + 1

        if not enabled:
            results.append(StageResult(name=label, script=script, skipped=True))
            continue

        print_header(f"Stage {stage_num}: {label}")

        # Build the argument list — generate doesn't need neo4j args
        if flag_name == "generate":
            stage_args = builder(args)
        else:
            stage_args = builder(args, neo4j_args)

        ok, duration = run_script(script, stage_args, project_root, dry_run=args.dry_run)
        results.append(StageResult(name=label, script=script, duration=duration, success=ok))

        if not ok and not args.dry_run:
            print_error(f"Pipeline aborted at stage {stage_num} ({label}).")
            print_summary(results, time.time() - t_pipeline)
            return 1

    total_time = time.time() - t_pipeline
    print_summary(results, total_time)

    executed = [r for r in results if not r.skipped]
    if executed:
        print(f"\n  {_c('All stages completed successfully.', Colors.GREEN)}")
        print(f"  Outputs: {args.output_dir}\n")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{_c('Pipeline interrupted by user.', Colors.YELLOW)}")
        sys.exit(130)