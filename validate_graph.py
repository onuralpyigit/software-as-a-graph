#!/usr/bin/env python3
"""
Graph Validation CLI

Validates the graph modeling and analysis approach by comparing
predicted criticality scores against actual failure impact.

Validation Pipeline:
    1. Graph Analysis → Predicted criticality scores Q(v)
    2. Failure Simulation → Actual impact scores I(v)
    3. Statistical Comparison → Validation metrics

Layers:
    app      : Application layer
    infra    : Infrastructure layer
    mw-app   : Middleware-Application layer
    mw-infra : Middleware-Infrastructure layer
    system   : Complete system

Usage:
    python validate_graph.py --layers app,infra,system
    python validate_graph.py --all
    python validate_graph.py --quick predicted.json actual.json

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.validation import (
    ValidationPipeline,
    PipelineResult,
    LayerValidationResult,
    QuickValidator,
    ValidationTargets,
    LAYER_DEFINITIONS,
)


# =============================================================================
# Terminal Colors
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def colored(text: str, color: str, bold: bool = False) -> str:
    """Apply color to text."""
    style = Colors.BOLD if bold else ""
    return f"{style}{color}{text}{Colors.RESET}"


def status_icon(passed: bool) -> str:
    """Get status icon for pass/fail."""
    if passed:
        return colored("✓", Colors.GREEN, bold=True)
    return colored("✗", Colors.RED, bold=True)


def status_text(passed: bool) -> str:
    """Get status text for pass/fail."""
    if passed:
        return colored("PASSED", Colors.GREEN, bold=True)
    return colored("FAILED", Colors.RED, bold=True)


def metric_color(value: float, target: float, higher_better: bool = True) -> str:
    """Get color for metric based on target."""
    if higher_better:
        passed = value >= target
    else:
        passed = value <= target
    
    if passed:
        return Colors.GREEN
    elif higher_better and value >= target * 0.9:
        return Colors.YELLOW
    elif not higher_better and value <= target * 1.1:
        return Colors.YELLOW
    return Colors.RED


# =============================================================================
# Display Functions
# =============================================================================

def print_header(title: str, char: str = "=", width: int = 78) -> None:
    """Print a formatted header."""
    print(f"\n{colored(char * width, Colors.CYAN)}")
    print(f"{colored(f' {title} '.center(width), Colors.CYAN, bold=True)}")
    print(f"{colored(char * width, Colors.CYAN)}")


def print_subheader(title: str) -> None:
    """Print a subheader."""
    print(f"\n{colored(f'>> {title}', Colors.WHITE, bold=True)}")


def display_pipeline_result(result: PipelineResult) -> None:
    """Display the full pipeline result."""
    print_header("VALIDATION PIPELINE RESULTS", "═")
    
    print(f"\n  {colored('Timestamp:', Colors.CYAN)}      {result.timestamp}")
    print(f"  {colored('Components:', Colors.CYAN)}     {result.total_components}")
    
    # Overall status
    print(f"\n  {colored('Overall Status:', Colors.WHITE, bold=True)} {status_text(result.all_passed)}")
    print(f"  Layers Validated: {len(result.layers)}")
    print(f"  Layers Passed:    {result.layers_passed}")
    
    # Layer summary table
    print_subheader("Layer Summary")
    
    print(f"\n  {'Layer':<12} {'N':<6} {'Spearman':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'Top-5':<10} {'Status':<10}")
    print(f"  {'-' * 78}")
    
    targets = result.targets
    for layer_name, layer_result in result.layers.items():
        spearman_color = metric_color(layer_result.spearman, targets.spearman)
        f1_color = metric_color(layer_result.f1_score, targets.f1_score)
        prec_color = metric_color(layer_result.precision, targets.precision)
        recall_color = metric_color(layer_result.recall, targets.recall)
        top5_color = metric_color(layer_result.top_5_overlap, targets.top_5_overlap)
        
        print(
            f"  {layer_name:<12} {layer_result.matched_components:<6} "
            f"{colored(f'{layer_result.spearman:>8.4f}', spearman_color)}   "
            f"{colored(f'{layer_result.f1_score:>8.4f}', f1_color)}   "
            f"{colored(f'{layer_result.precision:>8.4f}', prec_color)}   "
            f"{colored(f'{layer_result.recall:>8.4f}', recall_color)}   "
            f"{colored(f'{layer_result.top_5_overlap:>8.4f}', top5_color)}   "
            f"{status_text(layer_result.passed)}"
        )
    
    # Validation targets
    print_subheader("Validation Targets")
    print(f"\n  Spearman ρ ≥ {targets.spearman:.2f}  |  F1 ≥ {targets.f1_score:.2f}  |  "
          f"Precision ≥ {targets.precision:.2f}  |  Recall ≥ {targets.recall:.2f}  |  "
          f"Top-5 ≥ {targets.top_5_overlap:.2f}")
    
    # Cross-layer insights
    if result.cross_layer_insights:
        print_subheader("Cross-Layer Insights")
        for insight in result.cross_layer_insights:
            print(f"  • {insight}")
    
    # Detailed layer results
    for layer_name, layer_result in result.layers.items():
        display_layer_result(layer_result, targets)
    
    # Final summary
    print_header("FINAL SUMMARY", "-")
    
    if result.all_passed:
        print(f"\n  {colored('✓ All validation targets met!', Colors.GREEN, bold=True)}")
        print(f"\n  The graph analysis approach accurately predicts component criticality.")
        print(f"  Predicted quality scores correlate strongly with simulated failure impact.")
        print(f"\n  {colored('Conclusion:', Colors.CYAN)} The methodology is validated for production use.")
    else:
        print(f"\n  {colored('✗ Some validation targets not met.', Colors.RED, bold=True)}")
        print(f"\n  Review the metrics above to identify areas for improvement.")
        
        # Specific suggestions
        for layer_name, layer_result in result.layers.items():
            if not layer_result.passed:
                print(f"\n  {colored(f'Suggestions for {layer_name}:', Colors.YELLOW)}")
                
                if layer_result.spearman < targets.spearman:
                    print(f"    • Low Spearman correlation ({layer_result.spearman:.3f} < {targets.spearman})")
                    print(f"      Consider adjusting quality formula weights.")
                
                if layer_result.f1_score < targets.f1_score:
                    print(f"    • Low F1 score ({layer_result.f1_score:.3f} < {targets.f1_score})")
                    print(f"      Review classification thresholds.")
                
                if layer_result.precision < targets.precision:
                    print(f"    • Low precision ({layer_result.precision:.3f} < {targets.precision})")
                    print(f"      Model over-predicts critical components (too many false positives).")
                
                if layer_result.recall < targets.recall:
                    print(f"    • Low recall ({layer_result.recall:.3f} < {targets.recall})")
                    print(f"      Model misses critical components (too many false negatives).")
    
    print()


def display_layer_result(result: LayerValidationResult, targets: ValidationTargets) -> None:
    """Display detailed results for a single layer."""
    print_subheader(f"Layer: {result.layer_name} ({result.layer})")
    
    print(f"\n  {colored('Data Alignment:', Colors.CYAN)}")
    print(f"    Predicted Components:  {result.predicted_components}")
    print(f"    Simulated Components:  {result.simulated_components}")
    print(f"    Matched Components:    {result.matched_components}")
    
    if result.warnings:
        print(f"\n  {colored('Warnings:', Colors.YELLOW)}")
        for warning in result.warnings:
            print(f"    ⚠ {warning}")
    
    if not result.validation_result:
        return
    
    overall = result.validation_result.overall
    
    # Correlation metrics
    print(f"\n  {colored('Correlation Metrics:', Colors.CYAN)}")
    print(f"    {'Metric':<15} {'Value':<12} {'Target':<12} {'Status':<10}")
    print(f"    {'-' * 49}")
    
    spearman_passed = overall.correlation.spearman >= targets.spearman
    print(f"    {'Spearman ρ':<15} {overall.correlation.spearman:>8.4f}     ≥{targets.spearman:<8.2f}   {status_icon(spearman_passed)}")
    
    pearson_passed = overall.correlation.pearson >= targets.pearson
    print(f"    {'Pearson r':<15} {overall.correlation.pearson:>8.4f}     ≥{targets.pearson:<8.2f}   {status_icon(pearson_passed)}")
    
    kendall_passed = overall.correlation.kendall >= targets.kendall
    print(f"    {'Kendall τ':<15} {overall.correlation.kendall:>8.4f}     ≥{targets.kendall:<8.2f}   {status_icon(kendall_passed)}")
    
    # Classification metrics
    print(f"\n  {colored('Classification Metrics:', Colors.CYAN)}")
    print(f"    {'Metric':<15} {'Value':<12} {'Target':<12} {'Status':<10}")
    print(f"    {'-' * 49}")
    
    f1_passed = overall.classification.f1_score >= targets.f1_score
    print(f"    {'F1 Score':<15} {overall.classification.f1_score:>8.4f}     ≥{targets.f1_score:<8.2f}   {status_icon(f1_passed)}")
    
    prec_passed = overall.classification.precision >= targets.precision
    print(f"    {'Precision':<15} {overall.classification.precision:>8.4f}     ≥{targets.precision:<8.2f}   {status_icon(prec_passed)}")
    
    recall_passed = overall.classification.recall >= targets.recall
    print(f"    {'Recall':<15} {overall.classification.recall:>8.4f}     ≥{targets.recall:<8.2f}   {status_icon(recall_passed)}")
    
    print(f"    {'Accuracy':<15} {overall.classification.accuracy:>8.4f}")
    
    # Confusion matrix
    cm = overall.classification.confusion_matrix
    print(f"\n    {colored('Confusion Matrix:', Colors.DIM)}")
    print(f"                            Actual Critical    Actual Non-Critical")
    print(f"      Pred Critical           {colored(str(cm['tp']), Colors.GREEN):>3}                {colored(str(cm['fp']), Colors.RED):>3}")
    print(f"      Pred Non-Critical       {colored(str(cm['fn']), Colors.RED):>3}                {colored(str(cm['tn']), Colors.GREEN):>3}")
    
    # Ranking metrics
    print(f"\n  {colored('Ranking Metrics:', Colors.CYAN)}")
    print(f"    {'Metric':<15} {'Value':<12} {'Target':<12} {'Status':<10}")
    print(f"    {'-' * 49}")
    
    top5_passed = overall.ranking.top_5_overlap >= targets.top_5_overlap
    print(f"    {'Top-5 Overlap':<15} {overall.ranking.top_5_overlap:>8.4f}     ≥{targets.top_5_overlap:<8.2f}   {status_icon(top5_passed)}")
    
    top10_passed = overall.ranking.top_10_overlap >= targets.top_10_overlap
    print(f"    {'Top-10 Overlap':<15} {overall.ranking.top_10_overlap:>8.4f}     ≥{targets.top_10_overlap:<8.2f}   {status_icon(top10_passed)}")
    
    # Top-5 agreement
    if overall.ranking.top_5_predicted:
        print(f"\n    {colored('Top-5 Agreement:', Colors.DIM)}")
        print(f"      Predicted:  {', '.join(overall.ranking.top_5_predicted[:5])}")
        print(f"      Actual:     {', '.join(overall.ranking.top_5_actual[:5])}")
        print(f"      Common:     {', '.join(overall.ranking.top_5_common) or 'None'}")
    
    # Error metrics
    print(f"\n  {colored('Error Metrics:', Colors.CYAN)}")
    rmse_passed = overall.error.rmse <= targets.rmse_max
    print(f"    RMSE: {overall.error.rmse:.4f} {'≤' if rmse_passed else '>'} {targets.rmse_max} {status_icon(rmse_passed)}")
    print(f"    MAE:  {overall.error.mae:.4f}")
    print(f"    Max Error: {overall.error.max_error:.4f}")
    
    # Top mismatches (if available)
    if result.comparisons:
        print(f"\n  {colored('Top Mismatches (by error):', Colors.CYAN)}")
        print(f"    {'Component':<20} {'Type':<12} {'Predicted':<10} {'Actual':<10} {'Error':<10} {'Class':<6}")
        print(f"    {'-' * 70}")
        
        for comp in result.comparisons[:5]:
            cls_color = Colors.GREEN if comp.classification in ("TP", "TN") else Colors.RED
            print(
                f"    {comp.id:<20} {comp.type:<12} "
                f"{comp.predicted:>8.4f}   {comp.actual:>8.4f}   "
                f"{comp.error:>8.4f}   {colored(comp.classification, cls_color)}"
            )
    
    # Per-type breakdown
    if result.validation_result.by_type:
        print(f"\n  {colored('Per-Type Breakdown:', Colors.CYAN)}")
        print(f"    {'Type':<15} {'N':<6} {'Spearman':<10} {'F1':<10} {'Passed':<10}")
        print(f"    {'-' * 51}")
        
        for comp_type, type_result in result.validation_result.by_type.items():
            print(
                f"    {comp_type:<15} {type_result.sample_size:<6} "
                f"{type_result.correlation.spearman:>8.4f}   "
                f"{type_result.classification.f1_score:>8.4f}   "
                f"{status_icon(type_result.passed)}"
            )


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate graph modeling and analysis approach.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Validation Pipeline:
  1. Graph Analysis → Predicted criticality scores Q(v)
  2. Failure Simulation → Actual impact scores I(v)
  3. Statistical Comparison → Validation metrics

Layers:
  app      Application layer (Applications only)
  infra    Infrastructure layer (Nodes only)
  mw-app   Middleware-Application (Applications + Brokers)
  mw-infra Middleware-Infrastructure (Nodes + Brokers)
  system   Complete system (all components)

Validation Targets:
  Spearman ρ ≥ 0.70  (rank correlation)
  F1 Score ≥ 0.80    (classification accuracy)
  Precision ≥ 0.80   (positive predictive value)
  Recall ≥ 0.80      (sensitivity)
  Top-5 Overlap ≥ 0.60 (ranking agreement)

Examples:
  %(prog)s --layer app,infra,system
  %(prog)s --all
  %(prog)s --quick predicted.json actual.json
  %(prog)s --layer system --output results/validation.json
        """
    )
    
    # Action
    action_group = parser.add_argument_group("Action")
    action_mutex = action_group.add_mutually_exclusive_group(required=True)
    action_mutex.add_argument(
        "--layer", "-l",
        help="Comma-separated layers to validate (e.g., app,infra,system)"
    )
    action_mutex.add_argument(
        "--all", "-a",
        action="store_true",
        help="Validate all layers"
    )
    action_mutex.add_argument(
        "--quick", "-q",
        nargs=2,
        metavar=("PREDICTED", "ACTUAL"),
        help="Quick validation from JSON files"
    )
    
    # Neo4j connection
    neo4j_group = parser.add_argument_group("Neo4j Connection")
    neo4j_group.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j connection URI (default: bolt://localhost:7687)"
    )
    neo4j_group.add_argument(
        "--user", "-u",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    neo4j_group.add_argument(
        "--password", "-p",
        default="password",
        help="Neo4j password (default: password)"
    )
    
    # Validation targets
    targets_group = parser.add_argument_group("Validation Targets")
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
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Export results to JSON file"
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout"
    )
    output_group.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Create targets
    targets = ValidationTargets(
        spearman=args.spearman,
        f1_score=args.f1,
        precision=args.precision,
        recall=args.recall,
    )
    
    try:
        # Quick validation
        if args.quick:
            predicted_file, actual_file = args.quick
            
            validator = QuickValidator(targets=targets)
            result = validator.validate_from_files(
                predicted_file=predicted_file,
                actual_file=actual_file,
            )
            
            if args.json:
                print(json.dumps(result.to_dict(), indent=2))
            elif not args.quiet:
                # Display simplified result
                print_header("Quick Validation Result")
                print(f"\n  Files: {predicted_file} vs {actual_file}")
                print(f"  Matched: {result.matched_count} components")
                print(f"\n  Status: {status_text(result.passed)}")
                
                overall = result.overall
                print(f"\n  Spearman:  {overall.correlation.spearman:.4f}  {status_icon(overall.correlation.spearman >= targets.spearman)}")
                print(f"  F1 Score:  {overall.classification.f1_score:.4f}  {status_icon(overall.classification.f1_score >= targets.f1_score)}")
                print(f"  Precision: {overall.classification.precision:.4f}  {status_icon(overall.classification.precision >= targets.precision)}")
                print(f"  Recall:    {overall.classification.recall:.4f}  {status_icon(overall.classification.recall >= targets.recall)}")
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
                if not args.quiet:
                    print(f"\n{colored(f'Results saved to: {args.output}', Colors.GREEN)}")
            
            return 0 if result.passed else 1
        
        # Full pipeline validation
        if args.all:
            layers = list(LAYER_DEFINITIONS.keys())
        else:
            layers = [l.strip() for l in args.layer.split(",")]
        
        # Create and run pipeline
        pipeline = ValidationPipeline(
            uri=args.uri,
            user=args.user,
            password=args.password,
            targets=targets,
        )
        
        result = pipeline.run(layers=layers)
        
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        elif not args.quiet:
            display_pipeline_result(result)
        
        if args.output:
            pipeline.export_result(result, args.output)
            if not args.quiet:
                print(f"\n{colored(f'Results saved to: {args.output}', Colors.GREEN)}")
        
        return 0 if result.all_passed else 1
    
    except KeyboardInterrupt:
        print(f"\n{colored('Validation interrupted.', Colors.YELLOW)}")
        return 130
    
    except Exception as e:
        logging.exception("Validation failed")
        print(f"{colored(f'Error: {e}', Colors.RED)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())