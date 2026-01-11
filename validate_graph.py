#!/usr/bin/env python3
"""
Software-as-a-Graph Validation CLI

Validates graph-based criticality predictions against simulation results.
Compares analysis (predicted scores) with simulation (actual impact) for
each graph layer: application, infrastructure, and complete system.

Usage:
    # Validate all layers from JSON file
    python validate_graph.py --all
    
    # Validate specific layer from Neo4j
    python validate_graph.py --layer application
    
    # Validate with custom targets
    python validate_graph.py --all --spearman 0.8 --f1 0.85
    
    # Export validation report and scatter data
    python validate_graph.py --all --output report.json --scatter results/

Author: Software-as-a-Graph Research Project
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.validation import (
    ValidationPipeline,
    QuickValidator,
    ValidationTargets,
    PipelineResult,
    LayerValidationResult,
    ValidationResult,
    LAYER_DEFINITIONS,
)


# =============================================================================
# ANSI Color Codes
# =============================================================================
COLORS_ENABLED = sys.stdout.isatty()

def color(code: str) -> str:
    return code if COLORS_ENABLED else ""

RED = color("\033[91m")
GREEN = color("\033[92m")
YELLOW = color("\033[93m")
BLUE = color("\033[94m")
MAGENTA = color("\033[95m")
CYAN = color("\033[96m")
WHITE = color("\033[97m")
GRAY = color("\033[90m")
BOLD = color("\033[1m")
DIM = color("\033[2m")
RESET = color("\033[0m")


# =============================================================================
# Display Functions
# =============================================================================

def print_header(title: str, char: str = "=", width: int = 72) -> None:
    """Print a section header."""
    print(f"\n{BOLD}{char * width}{RESET}")
    print(f"{BOLD} {title}{RESET}")
    print(f"{BOLD}{char * width}{RESET}")


def print_subheader(title: str) -> None:
    """Print a subsection header."""
    print(f"\n{BOLD}{CYAN}>> {title}{RESET}")


def status_icon(passed: bool) -> str:
    """Get icon for pass/fail status."""
    return f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"


def status_text(passed: bool) -> str:
    """Get colored text for pass/fail status."""
    if passed:
        return f"{GREEN}PASSED{RESET}"
    return f"{RED}FAILED{RESET}"


def metric_value(value: float, target: float, higher_better: bool = True) -> str:
    """Get colored metric value."""
    if higher_better:
        passed = value >= target
    else:
        passed = value <= target
    
    if passed:
        return f"{GREEN}{value:>8.4f}{RESET}"
    elif (higher_better and value >= target * 0.85) or (not higher_better and value <= target * 1.15):
        return f"{YELLOW}{value:>8.4f}{RESET}"
    else:
        return f"{RED}{value:>8.4f}{RESET}"


def display_layer_validation(layer_result: LayerValidationResult) -> None:
    """Display validation result for a single layer."""
    
    v = layer_result.validation_result
    overall = v.overall
    targets = v.targets
    
    # Header
    print_header(f"{layer_result.layer_name} Validation: {status_text(layer_result.passed)}")
    
    # Summary
    print(f"\n  {CYAN}Layer:{RESET}           {layer_result.layer}")
    print(f"  {CYAN}Analysis:{RESET}        {layer_result.analysis_components} components "
          f"({layer_result.critical_count} critical, {layer_result.high_count} high)")
    print(f"  {CYAN}Simulation:{RESET}      {layer_result.simulation_components} components "
          f"(avg={layer_result.avg_impact:.3f}, max={layer_result.max_impact:.3f})")
    print(f"  {CYAN}Matched:{RESET}         {layer_result.matched_components} components")
    
    # Alignment warnings
    if layer_result.analysis_only:
        print(f"  {YELLOW}⚠ Analysis-only:{RESET} {', '.join(layer_result.analysis_only)}")
    if layer_result.simulation_only:
        print(f"  {YELLOW}⚠ Simulation-only:{RESET} {', '.join(layer_result.simulation_only)}")
    
    # Warnings from validation
    if v.warnings:
        for warning in v.warnings:
            print(f"  {YELLOW}⚠ {warning}{RESET}")
    
    # Correlation Metrics
    print_subheader("Correlation Metrics")
    print(f"\n  {'Metric':<20} {'Value':<12} {'Target':<12} {'Status':<10}")
    print(f"  {'-' * 54}")
    
    print(f"  {'Spearman (ρ)':<20} {metric_value(overall.correlation.spearman, targets.spearman)}     "
          f"≥{targets.spearman:<8.2f}   {status_icon(overall.correlation.spearman >= targets.spearman)}")
    print(f"  {'Pearson (r)':<20} {metric_value(overall.correlation.pearson, targets.pearson)}     "
          f"≥{targets.pearson:<8.2f}   {status_icon(overall.correlation.pearson >= targets.pearson)}")
    print(f"  {'Kendall (τ)':<20} {metric_value(overall.correlation.kendall, targets.kendall)}     "
          f"≥{targets.kendall:<8.2f}   {status_icon(overall.correlation.kendall >= targets.kendall)}")
    
    # Classification Metrics
    print_subheader("Classification Metrics (Critical Detection)")
    print(f"\n  {'Metric':<20} {'Value':<12} {'Target':<12} {'Status':<10}")
    print(f"  {'-' * 54}")
    
    print(f"  {'F1 Score':<20} {metric_value(overall.classification.f1_score, targets.f1_score)}     "
          f"≥{targets.f1_score:<8.2f}   {status_icon(overall.classification.f1_score >= targets.f1_score)}")
    print(f"  {'Precision':<20} {metric_value(overall.classification.precision, targets.precision)}     "
          f"≥{targets.precision:<8.2f}   {status_icon(overall.classification.precision >= targets.precision)}")
    print(f"  {'Recall':<20} {metric_value(overall.classification.recall, targets.recall)}     "
          f"≥{targets.recall:<8.2f}   {status_icon(overall.classification.recall >= targets.recall)}")
    print(f"  {'Accuracy':<20} {overall.classification.accuracy:>8.4f}")
    
    # Confusion Matrix
    cm = overall.classification.confusion_matrix
    print(f"\n  {DIM}Confusion Matrix:{RESET}")
    print(f"  {DIM}                      Actual Critical    Actual Non-Critical{RESET}")
    print(f"  {DIM}  Pred Critical         {GREEN}{cm['tp']:>5}{RESET}              {RED}{cm['fp']:>5}{RESET}{RESET}")
    print(f"  {DIM}  Pred Non-Critical     {RED}{cm['fn']:>5}{RESET}              {GREEN}{cm['tn']:>5}{RESET}{RESET}")
    
    # Ranking Metrics
    print_subheader("Ranking Metrics (Top-K Agreement)")
    print(f"\n  {'Metric':<20} {'Value':<12} {'Target':<12} {'Status':<10}")
    print(f"  {'-' * 54}")
    
    print(f"  {'Top-5 Overlap':<20} {metric_value(overall.ranking.top_5_overlap, targets.top_5_overlap)}     "
          f"≥{targets.top_5_overlap:<8.2f}   {status_icon(overall.ranking.top_5_overlap >= targets.top_5_overlap)}")
    print(f"  {'Top-10 Overlap':<20} {metric_value(overall.ranking.top_10_overlap, targets.top_10_overlap)}     "
          f"≥{targets.top_10_overlap:<8.2f}   {status_icon(overall.ranking.top_10_overlap >= targets.top_10_overlap)}")
    print(f"  {'NDCG@5':<20} {overall.ranking.ndcg_5:>8.4f}")
    print(f"  {'NDCG@10':<20} {overall.ranking.ndcg_10:>8.4f}")
    
    # Error Metrics
    print_subheader("Error Metrics (Prediction Accuracy)")
    print(f"\n  {'Metric':<20} {'Value':<12} {'Target':<12} {'Status':<10}")
    print(f"  {'-' * 54}")
    
    print(f"  {'RMSE':<20} {metric_value(overall.error.rmse, targets.rmse_max, False)}     "
          f"≤{targets.rmse_max:<8.2f}   {status_icon(overall.error.rmse <= targets.rmse_max)}")
    print(f"  {'MAE':<20} {metric_value(overall.error.mae, targets.mae_max, False)}     "
          f"≤{targets.mae_max:<8.2f}   {status_icon(overall.error.mae <= targets.mae_max)}")
    print(f"  {'Max Error':<20} {overall.error.max_error:>8.4f}")
    
    # Per-Type Breakdown
    if v.by_type:
        print_subheader("Per-Type Breakdown")
        print(f"\n  {'Type':<15} {'N':<5} {'Spearman':<10} {'F1':<10} {'Top-5':<10} {'Status':<10}")
        print(f"  {'-' * 60}")
        
        for comp_type, type_result in v.by_type.items():
            print(f"  {comp_type:<15} {type_result.sample_size:<5} "
                  f"{type_result.correlation.spearman:>8.4f}   "
                  f"{type_result.classification.f1_score:>8.4f}   "
                  f"{type_result.ranking.top_5_overlap:>8.4f}   "
                  f"{status_text(type_result.passed)}")
    
    # Component Comparison (Top errors)
    if layer_result.component_comparisons:
        print_subheader("Top Prediction Errors")
        sorted_comps = sorted(layer_result.component_comparisons, key=lambda c: c.error, reverse=True)[:5]
        
        print(f"\n  {'ID':<12} {'Type':<12} {'Predicted':<12} {'Actual':<12} {'Error':<10}")
        print(f"  {'-' * 58}")
        
        for comp in sorted_comps:
            pred_mark = f"{GREEN}●{RESET}" if comp.predicted_critical else f"{DIM}○{RESET}"
            actual_mark = f"{GREEN}●{RESET}" if comp.actual_critical else f"{DIM}○{RESET}"
            
            print(f"  {comp.id:<12} {comp.type:<12} "
                  f"{comp.predicted_score:>8.4f} {pred_mark}   "
                  f"{comp.actual_impact:>8.4f} {actual_mark}   "
                  f"{comp.error:>8.4f}")


def display_pipeline_result(result: PipelineResult) -> None:
    """Display complete pipeline result."""
    
    print_header("VALIDATION PIPELINE RESULTS", "═")
    
    print(f"\n  {CYAN}Timestamp:{RESET}      {result.timestamp}")
    print(f"  {CYAN}Components:{RESET}     {result.total_components}")
    
    # Overall summary
    status = "ALL PASSED" if result.all_passed else "SOME FAILED"
    print(f"\n  {BOLD}Overall Status: {status_text(result.all_passed)}{RESET}")
    print(f"  Layers Validated: {len(result.layers)}")
    print(f"  Layers Passed:    {result.layers_passed}")
    
    # Layer summary table
    print_subheader("Layer Summary")
    
    print(f"\n  {'Layer':<15} {'N':<6} {'Spearman':<10} {'F1':<10} {'Top-5':<10} {'Status':<10}")
    print(f"  {'-' * 61}")
    
    for layer_name, layer_result in result.layers.items():
        v = layer_result.validation_result.overall
        print(f"  {layer_name:<15} {layer_result.matched_components:<6} "
              f"{v.correlation.spearman:>8.4f}   "
              f"{v.classification.f1_score:>8.4f}   "
              f"{v.ranking.top_5_overlap:>8.4f}   "
              f"{status_text(layer_result.passed)}")
    
    # Cross-layer insights
    if result.cross_layer_insights:
        print_subheader("Cross-Layer Insights")
        for insight in result.cross_layer_insights:
            print(f"  • {insight}")
    
    # Display each layer
    for layer_name, layer_result in result.layers.items():
        display_layer_validation(layer_result)
    
    # Final summary
    print_header("FINAL SUMMARY", "-")
    
    if result.all_passed:
        print(f"\n  {GREEN}✓ All validation targets met!{RESET}")
        print(f"  The graph analysis approach accurately predicts component criticality.")
        print(f"  Predicted quality scores correlate strongly with simulated failure impact.")
    else:
        print(f"\n  {RED}✗ Some validation targets not met.{RESET}")
        print(f"  Review the metrics above to identify areas for improvement.")
        
        # Specific suggestions
        for layer_name, layer_result in result.layers.items():
            v = layer_result.validation_result.overall
            targets = layer_result.validation_result.targets
            
            if v.correlation.spearman < targets.spearman:
                print(f"\n  {YELLOW}Suggestion for {layer_name}:{RESET}")
                print(f"    Low Spearman correlation ({v.correlation.spearman:.3f} < {targets.spearman}).")
                print(f"    Consider adjusting quality formula weights or adding structural metrics.")
            
            if v.classification.f1_score < targets.f1_score:
                print(f"\n  {YELLOW}Suggestion for {layer_name}:{RESET}")
                print(f"    Low F1 score ({v.classification.f1_score:.3f} < {targets.f1_score}).")
                print(f"    Review box-plot classification thresholds (k_factor parameter).")


# =============================================================================
# Main CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Software-as-a-Graph Validation CLI - Compare analysis predictions vs simulation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate all layers from JSON file
  python validate_graph.py --all
  
  # Validate specific layer
  python validate_graph.py --layer application
  
  # Validate from Neo4j database
  python validate_graph.py --all
  
  # Validate with custom targets
  python validate_graph.py --all --spearman 0.8 --f1 0.85
  
  # Export validation report and scatter data
  python validate_graph.py --all --output report.json --scatter results/
  
  # Neo4j with custom credentials
  python validate_graph.py --uri bolt://localhost:7687 --user admin --password secret --all
        """
    )
    
    # Layers to validate
    layer_group = parser.add_argument_group("Layers to Validate")
    layer_mutex = layer_group.add_mutually_exclusive_group(required=True)
    layer_mutex.add_argument(
        "--all", "-a",
        action="store_true",
        help="Validate all layers (application, infrastructure, complete)"
    )
    layer_mutex.add_argument(
        "--layer", "-l",
        choices=list(LAYER_DEFINITIONS.keys()),
        help="Validate specific layer"
    )
    
    # Validation targets
    targets_group = parser.add_argument_group("Validation Targets (optional)")
    targets_group.add_argument(
        "--spearman",
        type=float,
        default=0.70,
        help="Target Spearman correlation (default: 0.70)"
    )
    targets_group.add_argument(
        "--f1",
        type=float,
        default=0.80,
        help="Target F1 score (default: 0.80)"
    )
    targets_group.add_argument(
        "--precision",
        type=float,
        default=0.80,
        help="Target precision (default: 0.80)"
    )
    targets_group.add_argument(
        "--recall",
        type=float,
        default=0.80,
        help="Target recall (default: 0.80)"
    )
    targets_group.add_argument(
        "--top5",
        type=float,
        default=0.60,
        help="Target top-5 overlap (default: 0.60)"
    )
    
    # Neo4j credentials
    neo4j_group = parser.add_argument_group("Neo4j Credentials")
    neo4j_group.add_argument(
        "--uri", "-u",
        default="bolt://localhost:7687",
        help="Neo4j connection URI (default: bolt://localhost:7687)"
    )
    neo4j_group.add_argument("--user", default="neo4j", help="Neo4j username (default: neo4j)")
    neo4j_group.add_argument("--password", default="password", help="Neo4j password")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Export validation report to JSON file"
    )
    output_group.add_argument(
        "--scatter",
        metavar="DIR",
        help="Export scatter plot CSV files to directory"
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (suppress formatted output)"
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # Create validation targets
    targets = ValidationTargets(
        spearman=args.spearman,
        f1_score=args.f1,
        precision=args.precision,
        recall=args.recall,
        top_5_overlap=args.top5,
    )
    
    # Determine layers to validate
    if args.all:
        layers = ["application", "infrastructure", "complete"]
    else:
        layers = [args.layer]
    
    if not args.quiet and not args.json:
        print(f"{CYAN}Layers:{RESET} {', '.join(layers)}")
    
    # Initialize pipeline
    try:
        pipeline_kwargs = {"targets": targets}
        pipeline_kwargs["uri"] = args.uri
        pipeline_kwargs["user"] = args.user
        pipeline_kwargs["password"] = args.password
        
        pipeline = ValidationPipeline(**pipeline_kwargs)
        
        # Run validation
        result = pipeline.run(layers=layers)
        
        # Display results
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        elif not args.quiet:
            display_pipeline_result(result)
        
        # Export report
        if args.output:
            pipeline.export_report(result, args.output)
            if not args.quiet and not args.json:
                print(f"\n{GREEN}Report exported to: {args.output}{RESET}")
        
        # Export scatter data
        if args.scatter:
            files = pipeline.export_scatter_csv(result, args.scatter)
            if not args.quiet and not args.json:
                print(f"\n{GREEN}Scatter data exported to: {args.scatter}/{RESET}")
                for f in files:
                    print(f"  - {f}")
        
        if result.all_passed:
            print(f"{GREEN}All validation targets met!{RESET}")

        sys.exit(0)
    
    except FileNotFoundError as e:
        print(f"{RED}Error: {e}{RESET}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        if "neo4j" in str(e).lower():
            print(f"{RED}Error: Neo4j driver not installed.{RESET}", file=sys.stderr)
            print(f"Install with: pip install neo4j", file=sys.stderr)
        else:
            print(f"{RED}Error: Missing dependency: {e}{RESET}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if "ServiceUnavailable" in str(type(e).__name__) or "Unable to retrieve routing" in str(e):
            print(f"{RED}Error: Cannot connect to Neo4j{RESET}", file=sys.stderr)
            print(f"Check that Neo4j is running and the URI is correct.", file=sys.stderr)
        elif "AuthError" in str(type(e).__name__) or "authentication" in str(e).lower():
            print(f"{RED}Error: Neo4j authentication failed{RESET}", file=sys.stderr)
        else:
            logging.exception("Validation failed")
            print(f"{RED}Error: {e}{RESET}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()