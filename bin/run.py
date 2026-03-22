import sys
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

import argparse
import time
import logging
import json
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

from src.infrastructure import create_repository
from common.dispatcher import (
    dispatch_generate, dispatch_import, dispatch_analyze, 
    dispatch_predict, dispatch_simulate, dispatch_validate, 
    dispatch_visualize
)
from common.console import ConsoleDisplay, Colors
from common.arguments import add_neo4j_arguments


# Helper for colorizing text (keeping it for the summary table logic)
def _c(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"


# =============================================================================
# Stage Runner
# =============================================================================

@dataclass
class StageResult:
    """Result of a single pipeline stage execution."""
    name: str
    script: str  # Kept for compatibility with summary printer
    duration: float = 0.0
    success: bool = False
    skipped: bool = False


# =============================================================================
# Stage Builders — each returns an adapted Namespace for the dispatcher
# =============================================================================

def _prep_generate_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        output=args.input,
        config=args.config,
        scale=args.scale,
        seed=args.seed,
        domain=None,
        scenario=None
    )


def _prep_import_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        input=args.input,
        clear=args.clean,
        uri=args.uri,
        user=args.user,
        password=args.password
    )


def _prep_analyze_args(args: argparse.Namespace) -> argparse.Namespace:
    layers = [l.strip() for l in args.layers.split(",") if l.strip()]
    return argparse.Namespace(
        all=len(layers) > 1,
        layer=layers[0] if len(layers) == 1 else None,
        layers=args.layers if len(layers) > 1 else None,
        output=str(Path(args.output_dir) / "analysis_results.json"),
        use_ahp=args.use_ahp,
        norm='robust',
        winsorize=True,
        winsorize_limit=0.05,
        gnn_model=args.gnn_model,
        uri=args.uri,
        user=args.user,
        password=args.password,
        verbose=args.verbose
    )


def _prep_predict_args(args: argparse.Namespace) -> argparse.Namespace:
    layers = [l.strip() for l in args.layers.split(",") if l.strip()]
    checkpoint = args.gnn_model
    if not checkpoint:
        checkpoint = str(Path(args.output_dir) / "gnn_checkpoints")
    
    return argparse.Namespace(
        layer=args.layers,
        checkpoint=checkpoint,
        output=str(Path(args.output_dir) / "predictions.json"),
        structural=str(Path(args.output_dir) / "analysis_results.json"),
        rmav=None,
        simulated=None,
        uri=args.uri,
        user=args.user,
        password=args.password,
        verbose=args.verbose
    )


def _prep_simulate_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        command="report",
        layers=args.layers,
        output=str(Path(args.output_dir) / "simulation_report.json"),
        edges=False,
        uri=args.uri,
        user=args.user,
        password=args.password,
        verbose=args.verbose
    )


def _prep_validate_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        layer=args.layers,
        output=str(Path(args.output_dir) / "validation_results.json"),
        predicted=str(Path(args.output_dir) / "predictions.json"),
        actual=str(Path(args.output_dir) / "simulation_report.json"),
        uri=args.uri,
        user=args.user,
        password=args.password,
        verbose=args.verbose
    )


def _prep_visualize_args(args: argparse.Namespace) -> argparse.Namespace:
    layers = [l.strip() for l in args.layers.split(",") if l.strip()]
    return argparse.Namespace(
        all=len(layers) > 1,
        layers=args.layers if len(layers) > 1 else None,
        layer=layers[0] if len(layers) == 1 else None,
        output=str(Path(args.output_dir) / "dashboard.html"),
        no_network=False,
        no_matrix=False,
        no_validation=False,
        antipatterns=None,
        multi_seed=0,
        open=args.open,
        uri=args.uri,
        user=args.user,
        password=args.password
    )


# =============================================================================
# Pipeline Definition
# =============================================================================

# (flag_name, stage_label, dispatcher_func, args_prepper)
STAGES = [
    ("generate",  "Generation",    dispatch_generate,  _prep_generate_args),
    ("do_import", "Import",        dispatch_import,    _prep_import_args),
    ("analyze",   "Analysis",      dispatch_analyze,   _prep_analyze_args),
    ("predict",   "Prediction",    dispatch_predict,   _prep_predict_args),
    ("simulate",  "Simulation",    dispatch_simulate,  _prep_simulate_args),
    ("validate",  "Validation",    dispatch_validate,  _prep_validate_args),
    ("visualize", "Visualization", dispatch_visualize, _prep_visualize_args),
]


def print_summary(display: ConsoleDisplay, results: List[StageResult], total_time: float) -> None:
    """Print a timing summary table for all executed stages."""
    display.print_header("Pipeline Summary")

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
        )

    print(f"\n  Total: {_c(f'{total_time:.2f}s', Colors.BOLD)}")


# =============================================================================
# CLI & Main
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Software-as-a-Graph Pipeline Orchestrator (In-Process)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    stages = parser.add_argument_group("Pipeline Stages")
    stages.add_argument("--all", "-a", action="store_true", help="Run all 7 stages")
    stages.add_argument("--generate", "-g", action="store_true", help="Stage 1: Generate synthetic data")
    stages.add_argument("--import", "-i", dest="do_import", action="store_true", help="Stage 2: Import data into Neo4j")
    stages.add_argument("--analyze", "-A", action="store_true", help="Stage 3: Structural analysis")
    stages.add_argument("--predict", "-P", action="store_true", help="Stage 4: GNN Prediction")
    stages.add_argument("--simulate", "-s", action="store_true", help="Stage 5: Simulation report")
    stages.add_argument("--validate", "-V", action="store_true", help="Stage 6: Validate results")
    stages.add_argument("--visualize", "-z", action="store_true", help="Stage 7: Generate dashboard")

    data = parser.add_argument_group("Data Options")
    data.add_argument("--config", metavar="FILE", help="Graph generation config (YAML)")
    data.add_argument("--scale", default="medium", choices=["tiny", "small", "medium", "large", "xlarge"], help="Graph scale")
    data.add_argument("--seed", type=int, default=42, help="Random seed")
    data.add_argument("--input", default="output/system.json", help="Intermediate JSON path")
    data.add_argument("--output-dir", default="output", help="Output directory")

    analysis = parser.add_argument_group("Analysis Options")
    analysis.add_argument("--layer", "--layers", dest="layers", default="app,infra,mw", help="Layers to process")
    analysis.add_argument("--gnn-model", metavar="PATH", help="Path to GNN model")
    analysis.add_argument("--use-ahp", action="store_true", help="Use AHP weights")
    analysis.add_argument("--clean", "--clear", dest="clean", action="store_true", help="Clear Neo4j before import")

    # --- Neo4j Connection ---
    add_neo4j_arguments(parser)

    runtime = parser.add_argument_group("Runtime Options")
    runtime.add_argument("--dry-run", action="store_true", help="Print plans without executing")
    runtime.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    runtime.add_argument("--open", "-O", action="store_true", help="Open dashboard after visualization")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_all = args.all
    any_stage = run_all or any(getattr(args, flag) for flag, *_ in STAGES)
    if not any_stage:
        parser.print_help()
        return 1

    # Initialize display
    display = ConsoleDisplay()

    if not args.dry_run:
        try:
            repo = create_repository(uri=args.uri, user=args.user, password=args.password)
        except Exception as e:
            display.print_error(f"Failed to connect to Neo4j: {e}")
            return 1

    results: List[StageResult] = []
    t_pipeline = time.time()
    graph_data = None

    try:
        for flag_name, label, dispatch_func, prepper in STAGES:
            enabled = run_all or getattr(args, flag_name, False)
            stage_num = len(results) + 1

            if not enabled:
                results.append(StageResult(name=label, script=label.lower(), skipped=True))
                continue

            display.print_header(f"Stage {stage_num}: {label}")
            stage_args = prepper(args)
            
            if args.dry_run:
                display.print_step(f"[dry-run] Would call {dispatch_func.__name__}")
                results.append(StageResult(name=label, script=label.lower(), success=True))
                continue

            try:
                t0 = time.time()
                display.print_step(f"In-process execution of {label}...")
                
                # Special handling for in-memory data passing
                if flag_name == "generate":
                    graph_data = dispatch_func(stage_args)
                elif flag_name == "do_import":
                    dispatch_func(repo, stage_args, graph_data=graph_data)
                else:
                    dispatch_func(repo, stage_args)
                
                duration = time.time() - t0
                display.print_success(f"Completed in {duration:.2f}s")
                results.append(StageResult(name=label, script=label.lower(), duration=duration, success=True))
                
            except Exception as e:
                duration = time.time() - t0
                display.print_error(f"Failed: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                results.append(StageResult(name=label, script=label.lower(), duration=duration, success=False))
                display.print_error(f"Pipeline aborted at stage {stage_num} ({label}).")
                print_summary(display, results, time.time() - t_pipeline)
                return 1

        total_time = time.time() - t_pipeline
        print_summary(display, results, total_time)
        print(f"\n  All stages completed successfully.")

    finally:
        if repo:
            repo.close()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\nPipeline interrupted by user.")
        sys.exit(130)