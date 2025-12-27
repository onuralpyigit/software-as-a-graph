#!/usr/bin/env python3
"""
Graph Validation CLI - Version 4.0

Validates graph-based analysis by comparing predicted criticality scores
against actual impact from failure simulation.

Pipeline Steps:
1. Load graph from JSON
2. Compute predicted scores using graph metrics
3. Run failure simulation to measure actual impact
4. Validate predictions using statistical methods

Target Metrics:
- Spearman Correlation: ≥ 0.70
- F1-Score: ≥ 0.90
- Precision/Recall: ≥ 0.80
- Top-5 Overlap: ≥ 60%

Usage:
    # Validate using composite score method
    python validate_graph.py --input system.json
    
    # Compare all analysis methods
    python validate_graph.py --input system.json --compare
    
    # Use specific method with bootstrap confidence intervals
    python validate_graph.py --input system.json --method betweenness --bootstrap
    
    # Custom validation targets
    python validate_graph.py --input system.json --spearman-target 0.8 --f1-target 0.85

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.simulation import SimulationGraph
from src.validation import (
    ValidationPipeline,
    PipelineResult,
    ValidationTargets,
    ValidationStatus,
    MetricStatus,
)


# =============================================================================
# Terminal Colors
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    @classmethod
    def disable(cls):
        for attr in ['HEADER', 'BLUE', 'CYAN', 'GREEN', 'YELLOW', 'RED', 'END', 'BOLD', 'DIM']:
            setattr(cls, attr, '')


def use_colors() -> bool:
    import os
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and not os.getenv('NO_COLOR')


# =============================================================================
# Output Helpers
# =============================================================================

def print_header(text: str) -> None:
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}")


def print_section(title: str) -> None:
    print(f"\n{Colors.CYAN}{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.DIM}{'-'*50}{Colors.END}")


def print_subsection(title: str) -> None:
    print(f"\n  {Colors.BLUE}{title}{Colors.END}")


def print_kv(key: str, value, indent: int = 2) -> None:
    print(f"{' '*indent}{Colors.DIM}{key}:{Colors.END} {value}")


def print_success(text: str) -> None:
    print(f"{Colors.GREEN}✓{Colors.END} {text}")


def print_error(text: str) -> None:
    print(f"{Colors.RED}✗{Colors.END} {text}", file=sys.stderr)


def print_warning(text: str) -> None:
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")


def status_color(status: ValidationStatus) -> str:
    if status == ValidationStatus.PASSED:
        return Colors.GREEN
    elif status == ValidationStatus.PARTIAL:
        return Colors.YELLOW
    return Colors.RED


def metric_color(status: MetricStatus) -> str:
    if status == MetricStatus.MET:
        return Colors.GREEN
    elif status == MetricStatus.BORDERLINE:
        return Colors.YELLOW
    return Colors.RED


def value_color(value: float, target: float) -> str:
    if value >= target:
        return Colors.GREEN
    elif value >= target * 0.95:
        return Colors.YELLOW
    return Colors.RED


# =============================================================================
# Display Functions
# =============================================================================

def print_pipeline_result(result: PipelineResult, verbose: bool = False) -> None:
    """Print complete pipeline result"""
    v = result.validation
    
    # Status
    sc = status_color(v.status)
    print_section("VALIDATION STATUS")
    print(f"  Status: {sc}{Colors.BOLD}{v.status.value.upper()}{Colors.END}")
    print(f"  Components: {result.n_components}")
    print(f"  Connections: {result.n_connections}")
    print(f"  Message Paths: {result.n_paths}")
    print(f"  Method: {result.analysis_method}")
    
    # Correlation metrics
    print_section("CORRELATION METRICS")
    corr = v.correlation
    
    targets = v.targets
    spearman_c = value_color(corr.spearman, targets.spearman)
    print(f"  Spearman ρ:  {spearman_c}{corr.spearman:.4f}{Colors.END} (target: ≥{targets.spearman})")
    print(f"  Pearson r:   {corr.pearson:.4f}")
    print(f"  Kendall τ:   {corr.kendall:.4f}")
    
    if corr.spearman_p < 0.05:
        print(f"  {Colors.GREEN}✓ Statistically significant (p={corr.spearman_p:.4f}){Colors.END}")
    else:
        print(f"  {Colors.YELLOW}⚠ Not significant (p={corr.spearman_p:.4f}){Colors.END}")
    
    # Classification metrics
    print_section("CLASSIFICATION METRICS")
    cls = v.classification
    
    print(f"  Confusion Matrix:")
    print(f"    TP: {cls.tp:3}  |  FP: {cls.fp:3}")
    print(f"    FN: {cls.fn:3}  |  TN: {cls.tn:3}")
    print()
    
    prec_c = value_color(cls.precision, targets.precision)
    rec_c = value_color(cls.recall, targets.recall)
    f1_c = value_color(cls.f1, targets.f1)
    
    print(f"  Precision:  {prec_c}{cls.precision:.4f}{Colors.END} (target: ≥{targets.precision})")
    print(f"  Recall:     {rec_c}{cls.recall:.4f}{Colors.END} (target: ≥{targets.recall})")
    print(f"  F1-Score:   {f1_c}{cls.f1:.4f}{Colors.END} (target: ≥{targets.f1})")
    print(f"  Accuracy:   {cls.accuracy:.4f}")
    
    # Ranking metrics
    print_section("RANKING METRICS")
    rank = v.ranking
    
    for k, overlap in sorted(rank.top_k_overlap.items()):
        target = targets.top_5 if k == 5 else targets.top_10 if k == 10 else 0.5
        oc = value_color(overlap, target)
        print(f"  Top-{k:2} Overlap: {oc}{overlap:.4f}{Colors.END}")
    
    print(f"  NDCG:        {rank.ndcg:.4f}")
    print(f"  MRR:         {rank.mrr:.4f}")
    
    # Achieved vs targets
    print_section("TARGET SUMMARY")
    for metric, (value, status) in v.achieved.items():
        mc = metric_color(status)
        check = "✓" if status == MetricStatus.MET else "✗" if status == MetricStatus.NOT_MET else "~"
        print(f"  {mc}{check}{Colors.END} {metric:15} {value:.4f}")
    
    # Bootstrap confidence intervals
    if v.bootstrap:
        print_section("BOOTSTRAP CONFIDENCE INTERVALS")
        for b in v.bootstrap:
            print(f"  {b.metric}:")
            print(f"    Estimate: {b.estimate:.4f}")
            print(f"    {int(b.confidence*100)}% CI: [{b.ci_lower:.4f}, {b.ci_upper:.4f}]")
            print(f"    Std Error: {b.std_error:.4f}")
    
    # Top components (verbose)
    if verbose and v.components:
        print_section("TOP COMPONENTS (by actual impact)")
        
        print(f"\n  {'Component':<20} {'Pred':>8} {'Actual':>8} {'P-Rank':>7} {'A-Rank':>7} {'Diff':>5}")
        print(f"  {'-'*60}")
        
        for cv in v.components[:15]:
            check = "✓" if cv.correct else "✗"
            print(f"  {cv.component_id:<20} {cv.predicted_score:>8.4f} {cv.actual_impact:>8.4f} "
                  f"{cv.predicted_rank:>7} {cv.actual_rank:>7} {cv.rank_difference:>4} {check}")
    
    # Misclassified (verbose)
    if verbose:
        if v.false_positives:
            print_subsection(f"False Positives ({len(v.false_positives)})")
            print(f"    Predicted critical but not actually: {', '.join(v.false_positives[:5])}")
        
        if v.false_negatives:
            print_subsection(f"False Negatives ({len(v.false_negatives)})")
            print(f"    Actually critical but not predicted: {', '.join(v.false_negatives[:5])}")
    
    # Timing
    print_section("TIMING")
    print(f"  Analysis:    {result.analysis_time_ms:>8.0f} ms")
    print(f"  Simulation:  {result.simulation_time_ms:>8.0f} ms")
    print(f"  Validation:  {result.validation_time_ms:>8.0f} ms")
    print(f"  {Colors.BOLD}Total:       {result.total_time_ms:>8.0f} ms{Colors.END}")


def print_comparison(results: Dict[str, PipelineResult]) -> None:
    """Print method comparison"""
    print_section("METHOD COMPARISON")
    
    # Header
    print(f"\n  {'Method':<15} {'Spearman':>10} {'F1':>10} {'Prec':>10} {'Recall':>10} {'Top-5':>10} {'Status':>10}")
    print(f"  {'-'*75}")
    
    for method, result in results.items():
        v = result.validation
        sc = status_color(v.status)
        
        print(f"  {method:<15} "
              f"{v.correlation.spearman:>10.4f} "
              f"{v.classification.f1:>10.4f} "
              f"{v.classification.precision:>10.4f} "
              f"{v.classification.recall:>10.4f} "
              f"{v.ranking.top_k_overlap.get(5, 0):>10.4f} "
              f"{sc}{v.status.value:>10}{Colors.END}")
    
    # Best method
    best = max(results.keys(), key=lambda m: results[m].validation.correlation.spearman)
    print(f"\n  {Colors.GREEN}Best Method: {best}{Colors.END}")
    print(f"  Spearman: {results[best].validation.correlation.spearman:.4f}")


def export_results(result: PipelineResult, output_dir: Path, formats: List[str]) -> None:
    """Export results to files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if "json" in formats:
        json_file = output_dir / f"validation_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print_success(f"JSON exported: {json_file}")
    
    if "summary" in formats:
        summary_file = output_dir / f"validation_{timestamp}.txt"
        with open(summary_file, "w") as f:
            v = result.validation
            f.write(f"Validation Summary\n")
            f.write(f"==================\n\n")
            f.write(f"Status: {v.status.value}\n")
            f.write(f"Method: {result.analysis_method}\n")
            f.write(f"Components: {result.n_components}\n\n")
            f.write(f"Metrics:\n")
            f.write(f"  Spearman:  {v.correlation.spearman:.4f}\n")
            f.write(f"  F1-Score:  {v.classification.f1:.4f}\n")
            f.write(f"  Precision: {v.classification.precision:.4f}\n")
            f.write(f"  Recall:    {v.classification.recall:.4f}\n")
            f.write(f"  Top-5:     {v.ranking.top_k_overlap.get(5, 0):.4f}\n\n")
            f.write(f"Time: {result.total_time_ms:.0f} ms\n")
        print_success(f"Summary exported: {summary_file}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate graph-based analysis against failure simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate using composite method
    python validate_graph.py --input system.json
    
    # Compare all methods
    python validate_graph.py --input system.json --compare
    
    # Use specific method
    python validate_graph.py --input system.json --method betweenness
    
    # With bootstrap confidence intervals
    python validate_graph.py --input system.json --bootstrap --bootstrap-n 1000
    
    # Custom targets
    python validate_graph.py --input system.json --spearman-target 0.8 --f1-target 0.85
    
    # Export results
    python validate_graph.py --input system.json --output results/ --format json summary
        """,
    )
    
    # Input
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Input graph JSON file",
    )
    
    # Analysis method
    method_group = parser.add_argument_group("Analysis Method")
    method_group.add_argument(
        "--method", "-m", default="composite",
        choices=["composite", "betweenness", "degree", "message_path"],
        help="Analysis method (default: composite)",
    )
    method_group.add_argument(
        "--compare", "-c", action="store_true",
        help="Compare all analysis methods",
    )
    
    # Validation targets
    targets_group = parser.add_argument_group("Validation Targets")
    targets_group.add_argument(
        "--spearman-target", type=float, default=0.70,
        help="Spearman correlation target (default: 0.70)",
    )
    targets_group.add_argument(
        "--f1-target", type=float, default=0.90,
        help="F1-Score target (default: 0.90)",
    )
    targets_group.add_argument(
        "--precision-target", type=float, default=0.80,
        help="Precision target (default: 0.80)",
    )
    targets_group.add_argument(
        "--recall-target", type=float, default=0.80,
        help="Recall target (default: 0.80)",
    )
    
    # Simulation options
    sim_group = parser.add_argument_group("Simulation Options")
    sim_group.add_argument(
        "--cascade", action="store_true", default=True,
        help="Enable cascade propagation (default)",
    )
    sim_group.add_argument(
        "--no-cascade", dest="cascade", action="store_false",
        help="Disable cascade propagation",
    )
    sim_group.add_argument(
        "--cascade-threshold", type=float, default=0.5,
        help="Cascade threshold (default: 0.5)",
    )
    
    # Bootstrap
    bootstrap_group = parser.add_argument_group("Bootstrap Options")
    bootstrap_group.add_argument(
        "--bootstrap", "-b", action="store_true",
        help="Calculate bootstrap confidence intervals",
    )
    bootstrap_group.add_argument(
        "--bootstrap-n", type=int, default=1000,
        help="Bootstrap iterations (default: 1000)",
    )
    bootstrap_group.add_argument(
        "--confidence", type=float, default=0.95,
        help="Confidence level (default: 0.95)",
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o", type=Path,
        help="Output directory",
    )
    output_group.add_argument(
        "--format", nargs="+", default=["json"],
        choices=["json", "summary"],
        help="Output formats (default: json)",
    )
    output_group.add_argument(
        "--json", action="store_true",
        help="JSON output to stdout",
    )
    output_group.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )
    output_group.add_argument(
        "--quiet", "-q", action="store_true",
        help="Minimal output",
    )
    output_group.add_argument(
        "--no-color", action="store_true",
        help="Disable colors",
    )
    output_group.add_argument(
        "--seed", type=int,
        help="Random seed for reproducibility",
    )
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Handle colors
    if args.no_color or not use_colors():
        Colors.disable()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    
    try:
        if not args.quiet:
            print_header("Graph Validation Pipeline")
            print(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if args.compare:
                print(f"  Mode: Method comparison")
            else:
                print(f"  Method: {args.method}")
        
        # Load graph
        if not args.quiet:
            print_section("Loading Graph")
            print_kv("File", args.input)
        
        if not args.input.exists():
            print_error(f"File not found: {args.input}")
            return 1
        
        graph = SimulationGraph.from_json(args.input)
        
        if not args.quiet:
            print_success(f"Loaded: {len(graph.components)} components, {len(graph.connections)} connections")
        
        # Create targets
        targets = ValidationTargets(
            spearman=args.spearman_target,
            f1=args.f1_target,
            precision=args.precision_target,
            recall=args.recall_target,
        )
        
        # Create pipeline
        pipeline = ValidationPipeline(
            targets=targets,
            cascade_threshold=args.cascade_threshold,
            seed=args.seed,
        )
        
        if args.compare:
            # Compare methods
            if not args.quiet:
                print_section("Comparing Analysis Methods")
            
            results = pipeline.compare_methods(
                graph, enable_cascade=args.cascade
            )
            
            if args.json:
                output = {m: r.to_dict() for m, r in results.items()}
                print(json.dumps(output, indent=2))
            elif not args.quiet:
                print_comparison(results)
                
                if args.verbose:
                    for method, result in results.items():
                        print_header(f"METHOD: {method.upper()}")
                        print_pipeline_result(result, verbose=True)
            
            # Export best result
            if args.output:
                best = max(results.keys(), key=lambda m: results[m].validation.correlation.spearman)
                if not args.quiet:
                    print_section("Exporting Results")
                export_results(results[best], args.output, args.format)
            
            # Exit code based on best result
            best_result = max(results.values(), key=lambda r: r.validation.correlation.spearman)
            return 0 if best_result.validation.status == ValidationStatus.PASSED else 1
        
        else:
            # Single method
            if not args.quiet:
                print_section("Running Validation Pipeline")
            
            if args.bootstrap:
                # Run with bootstrap
                result = pipeline.run(
                    graph,
                    analysis_method=args.method,
                    enable_cascade=args.cascade,
                )
                
                # Add bootstrap to validation
                from src.validation import Validator
                validator = Validator(targets=targets, seed=args.seed)
                result.validation = validator.validate_with_bootstrap(
                    result.predicted_scores,
                    result.actual_impacts,
                    n_iterations=args.bootstrap_n,
                    confidence=args.confidence,
                )
            else:
                result = pipeline.run(
                    graph,
                    analysis_method=args.method,
                    enable_cascade=args.cascade,
                )
            
            if args.json:
                print(json.dumps(result.to_dict(), indent=2))
            elif not args.quiet:
                print_pipeline_result(result, verbose=args.verbose)
            
            # Export
            if args.output:
                if not args.quiet:
                    print_section("Exporting Results")
                export_results(result, args.output, args.format)
            
            if not args.quiet:
                print_success("\nValidation complete!")
            
            return 0 if result.validation.status == ValidationStatus.PASSED else 1
    
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        return 1
    
    except ValueError as e:
        print_error(f"Invalid input: {e}")
        return 1
    
    except KeyboardInterrupt:
        print_warning("\nValidation interrupted")
        return 130
    
    except Exception as e:
        print_error(f"Validation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())