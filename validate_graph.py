#!/usr/bin/env python3
"""
Validate Graph CLI - Version 5.0

Command-line interface for validating graph-based criticality predictions
against actual failure impact scores.

Features:
- Validate predicted vs actual criticality scores
- Component-type specific validation
- Compare multiple analysis methods
- Load data from JSON files or Neo4j
- Export detailed validation reports

Validation Targets:
- Spearman ρ ≥ 0.70 (rank correlation)
- F1-Score ≥ 0.90 (classification accuracy)

Usage:
    # Validate from JSON file
    python validate_graph.py --input graph.json
    
    # Validate from Neo4j
    python validate_graph.py --neo4j --password secret
    
    # Compare analysis methods
    python validate_graph.py --input graph.json --compare-methods
    
    # Validate specific component type
    python validate_graph.py --input graph.json --component-type Application
    
    # Custom targets
    python validate_graph.py --input graph.json --spearman-target 0.75 --f1-target 0.85

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    from src.validation import (
        ValidationPipeline,
        ValidationTargets,
        ValidationResult,
        PipelineResult,
        AnalysisMethod,
        Validator,
        run_validation,
    )
    from src.simulation import (
        SimulationGraph,
        ComponentType,
        create_simulation_graph,
    )
    HAS_VALIDATION = True
except ImportError as e:
    HAS_VALIDATION = False
    IMPORT_ERROR = str(e)


# =============================================================================
# Output Formatting
# =============================================================================

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
GRAY = "\033[90m"


def print_header(title: str) -> None:
    """Print section header"""
    print(f"\n{BOLD}{CYAN}{'=' * 70}{RESET}")
    print(f"{BOLD}{CYAN}{title.center(70)}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 70}{RESET}\n")


def print_subheader(title: str) -> None:
    """Print subsection header"""
    print(f"\n{BOLD}{title}{RESET}")
    print(f"{'-' * 50}")


def status_color(passed: bool) -> str:
    """Get color based on pass/fail status"""
    return GREEN if passed else RED


def metric_color(value: float, target: float) -> str:
    """Get color based on metric vs target"""
    if value >= target:
        return GREEN
    elif value >= target * 0.9:
        return YELLOW
    return RED


def format_metric(name: str, value: float, target: float, width: int = 20) -> str:
    """Format a metric with target comparison"""
    color = metric_color(value, target)
    status = "✓" if value >= target else "✗"
    return f"  {name:<{width}} {color}{value:.4f}{RESET} (target: ≥{target}) {color}{status}{RESET}"


# =============================================================================
# Graph Loading
# =============================================================================

def load_graph(args) -> Optional[SimulationGraph]:
    """Load graph from file or Neo4j"""
    try:
        if args.neo4j:
            from src.simulation import load_graph_from_neo4j
            
            print(f"{CYAN}Loading graph from Neo4j at {args.uri}...{RESET}")
            graph = load_graph_from_neo4j(
                uri=args.uri,
                user=args.user,
                password=args.password,
                database=args.database,
            )
        elif args.input:
            print(f"{CYAN}Loading graph from {args.input}...{RESET}")
            graph = create_simulation_graph(args.input)
        else:
            print(f"{RED}Error: Must specify --input or --neo4j{RESET}")
            return None
        
        stats = graph.get_statistics()
        print(f"  Components: {stats['total_components']}")
        print(f"  Edges: {stats['total_edges']}")
        
        return graph
        
    except Exception as e:
        print(f"{RED}Error loading graph: {e}{RESET}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None


# =============================================================================
# Validation Output
# =============================================================================

def print_validation_result(
    result: ValidationResult,
    targets: ValidationTargets,
    verbose: bool = False,
) -> None:
    """Print validation result details"""
    
    # Status
    status = result.status.value.upper()
    color = GREEN if result.passed else (YELLOW if status == "PARTIAL" else RED)
    print(f"\n  Status: {color}{BOLD}{status}{RESET}")
    print(f"  Components: {result.correlation.n_samples}")
    
    # Correlation metrics
    print_subheader("Correlation Metrics")
    print(format_metric("Spearman ρ", result.spearman, targets.spearman))
    print(format_metric("Pearson r", result.correlation.pearson, targets.pearson))
    print(format_metric("Kendall τ", result.correlation.kendall, targets.kendall))
    
    # Confidence interval
    if result.spearman_ci:
        ci = result.spearman_ci
        print(f"  {'Spearman 95% CI':<20} [{ci.lower:.4f}, {ci.upper:.4f}]")
    
    # Classification metrics
    print_subheader("Classification Metrics")
    print(format_metric("F1-Score", result.f1_score, targets.f1_score))
    print(format_metric("Precision", result.classification.precision, targets.precision))
    print(format_metric("Recall", result.classification.recall, targets.recall))
    print(format_metric("Accuracy", result.classification.accuracy, targets.accuracy))
    
    cm = result.classification.confusion_matrix
    print(f"\n  Confusion Matrix:")
    print(f"    TP={cm.true_positives:3d}  FP={cm.false_positives:3d}")
    print(f"    FN={cm.false_negatives:3d}  TN={cm.true_negatives:3d}")
    
    # Ranking metrics
    print_subheader("Ranking Metrics")
    print(format_metric("Top-5 Overlap", result.ranking.top_5_overlap, targets.top_5_overlap))
    print(format_metric("Top-10 Overlap", result.ranking.top_10_overlap, targets.top_10_overlap))
    print(f"  {'NDCG':<20} {result.ranking.ndcg:.4f}")
    print(f"  {'MRR':<20} {result.ranking.mrr:.4f}")
    print(f"  {'Avg Rank Diff':<20} {result.ranking.rank_difference_mean:.2f} ± {result.ranking.rank_difference_std:.2f}")
    
    # Verbose: component details
    if verbose:
        # False positives
        fps = result.get_false_positives()
        if fps:
            print_subheader("False Positives (predicted critical, not actually)")
            for comp in fps[:10]:
                print(f"    {comp.component_id} ({comp.component_type}): "
                      f"pred={comp.predicted_score:.4f}, actual={comp.actual_score:.4f}")
        
        # False negatives
        fns = result.get_false_negatives()
        if fns:
            print_subheader("False Negatives (actually critical, not predicted)")
            for comp in fns[:10]:
                print(f"    {comp.component_id} ({comp.component_type}): "
                      f"pred={comp.predicted_score:.4f}, actual={comp.actual_score:.4f}")
        
        # Top rank errors
        print_subheader("Top 10 Rank Errors")
        for comp in result.get_top_rank_errors(10):
            print(f"    {comp.component_id}: pred_rank={comp.predicted_rank}, "
                  f"actual_rank={comp.actual_rank}, diff={comp.rank_difference:+d}")


def print_method_comparison(
    comparison: Dict[str, Any],
    targets: ValidationTargets,
) -> None:
    """Print method comparison results"""
    print_subheader("Analysis Method Comparison")
    
    # Sort by Spearman
    sorted_methods = sorted(
        comparison.items(),
        key=lambda x: -x[1].spearman
    )
    
    print(f"\n  {'Method':<15} {'Spearman':<12} {'F1-Score':<12} {'Status':<10}")
    print(f"  {'-' * 49}")
    
    for method, comp in sorted_methods:
        sp_color = metric_color(comp.spearman, targets.spearman)
        f1_color = metric_color(comp.f1_score, targets.f1_score)
        status_clr = GREEN if comp.status.value == "passed" else (
            YELLOW if comp.status.value == "partial" else RED
        )
        
        print(f"  {method:<15} {sp_color}{comp.spearman:.4f}{RESET}       "
              f"{f1_color}{comp.f1_score:.4f}{RESET}       "
              f"{status_clr}{comp.status.value}{RESET}")
    
    # Best method
    best = sorted_methods[0]
    print(f"\n  {GREEN}Best Method: {best[0]} (ρ={best[1].spearman:.4f}){RESET}")


def print_by_component_type(
    by_type: Dict[str, ValidationResult],
    targets: ValidationTargets,
) -> None:
    """Print validation by component type"""
    print_subheader("Validation by Component Type")
    
    print(f"\n  {'Type':<15} {'Count':<8} {'Spearman':<12} {'F1-Score':<12} {'Status':<10}")
    print(f"  {'-' * 57}")
    
    for comp_type, result in sorted(by_type.items()):
        sp_color = metric_color(result.spearman, targets.spearman)
        f1_color = metric_color(result.f1_score, targets.f1_score)
        status_clr = GREEN if result.passed else RED
        
        print(f"  {comp_type:<15} {result.correlation.n_samples:<8} "
              f"{sp_color}{result.spearman:.4f}{RESET}       "
              f"{f1_color}{result.f1_score:.4f}{RESET}       "
              f"{status_clr}{result.status.value}{RESET}")


# =============================================================================
# Main Validation
# =============================================================================

def run_validation_pipeline(args, graph: SimulationGraph) -> Optional[Dict[str, Any]]:
    """Run validation pipeline"""
    print_header("VALIDATION PIPELINE")
    
    # Configure targets
    targets = ValidationTargets(
        spearman=args.spearman_target,
        f1_score=args.f1_target,
        precision=args.precision_target,
        recall=args.recall_target,
    )
    
    # Configure pipeline
    pipeline = ValidationPipeline(
        targets=targets,
        seed=args.seed,
        cascade=args.cascade,
    )
    
    # Determine analysis method
    method = AnalysisMethod.COMPOSITE
    if args.method:
        try:
            method = AnalysisMethod(args.method)
        except ValueError:
            print(f"{YELLOW}Unknown method '{args.method}', using composite{RESET}")
    
    print(f"Analysis Method: {method.value}")
    print(f"Compare Methods: {args.compare_methods}")
    print(f"Cascade Enabled: {args.cascade}")
    
    # Run pipeline
    try:
        result = pipeline.run(
            graph,
            analysis_method=method,
            compare_methods=args.compare_methods,
            validate_by_type=True,
        )
    except Exception as e:
        print(f"{RED}Error running validation: {e}{RESET}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None
    
    # Print results
    print_validation_result(result.validation, targets, args.verbose)
    
    # Method comparison
    if result.method_comparison:
        print_method_comparison(result.method_comparison, targets)
    
    # By component type
    if result.by_component_type:
        print_by_component_type(result.by_component_type, targets)
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    status_clr = GREEN if result.passed else RED
    print(f"  Overall Status: {status_clr}{BOLD}{result.validation.status.value.upper()}{RESET}")
    print(f"  Spearman ρ: {result.spearman:.4f} (target: ≥{targets.spearman})")
    print(f"  F1-Score: {result.f1_score:.4f} (target: ≥{targets.f1_score})")
    
    if result.method_comparison:
        best = result.get_best_method()
        print(f"  Best Method: {best}")
    
    return result.to_dict()


def run_custom_validation(args) -> Optional[Dict[str, Any]]:
    """Run validation with custom predicted/actual score files"""
    print_header("CUSTOM SCORE VALIDATION")
    
    # Load predicted scores
    print(f"Loading predicted scores from {args.predicted}...")
    with open(args.predicted) as f:
        predicted_data = json.load(f)
    
    # Handle different formats
    if isinstance(predicted_data, dict) and "scores" in predicted_data:
        predicted_scores = predicted_data["scores"]
    else:
        predicted_scores = predicted_data
    
    # Load actual scores
    print(f"Loading actual scores from {args.actual}...")
    with open(args.actual) as f:
        actual_data = json.load(f)
    
    if isinstance(actual_data, dict) and "scores" in actual_data:
        actual_scores = actual_data["scores"]
    else:
        actual_scores = actual_data
    
    print(f"  Predicted: {len(predicted_scores)} components")
    print(f"  Actual: {len(actual_scores)} components")
    
    # Configure targets
    targets = ValidationTargets(
        spearman=args.spearman_target,
        f1_score=args.f1_target,
    )
    
    # Load component types if provided
    component_types = None
    if args.types:
        with open(args.types) as f:
            component_types = json.load(f)
    
    # Run validation
    validator = Validator(targets=targets, seed=args.seed)
    result = validator.validate(predicted_scores, actual_scores, component_types)
    
    # Print results
    print_validation_result(result, targets, args.verbose)
    
    return result.to_dict()


def run_neo4j_validation(args) -> Optional[Dict[str, Any]]:
    """Run validation using Neo4j data"""
    print_header("NEO4J VALIDATION")
    
    try:
        from src.validation import Neo4jValidationClient
    except ImportError:
        print(f"{RED}Error: Neo4j driver not available{RESET}")
        return None
    
    targets = ValidationTargets(
        spearman=args.spearman_target,
        f1_score=args.f1_target,
    )
    
    print(f"Connecting to Neo4j at {args.uri}...")
    
    try:
        with Neo4jValidationClient(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
            targets=targets,
        ) as client:
            # Check data availability
            stats = client.get_validation_stats()
            print(f"\n  Data Availability:")
            for comp_type, count in stats["components"].items():
                analysis = stats["has_analysis_scores"].get(comp_type, 0)
                impact = stats["has_impact_scores"].get(comp_type, 0)
                print(f"    {comp_type}: {count} total, "
                      f"{analysis} with analysis, {impact} with impact")
            
            # Run validation
            if args.full_pipeline:
                print(f"\n  Running full validation pipeline...")
                result = client.run_full_validation(seed=args.seed)
                print_validation_result(result.validation, targets, args.verbose)
                
                if result.method_comparison:
                    print_method_comparison(result.method_comparison, targets)
                
                return result.to_dict()
            
            else:
                # Validate stored scores
                print(f"\n  Validating stored scores...")
                result = client.validate(
                    args.predicted_property,
                    args.actual_property,
                )
                print_validation_result(result, targets, args.verbose)
                
                # By type
                if args.by_type:
                    by_type = client.validate_by_component_type(
                        args.predicted_property,
                        args.actual_property,
                    )
                    print_by_component_type(by_type, targets)
                
                return result.to_dict()
    
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate graph-based criticality predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate from JSON file
  python validate_graph.py --input graph.json
  
  # Compare analysis methods
  python validate_graph.py --input graph.json --compare-methods
  
  # Custom targets
  python validate_graph.py --input graph.json --spearman-target 0.75 --f1-target 0.85
  
  # Validate from Neo4j
  python validate_graph.py --neo4j --password secret --full-pipeline
  
  # Validate custom score files
  python validate_graph.py --predicted pred.json --actual actual.json
  
  # Specific analysis method
  python validate_graph.py --input graph.json --method betweenness
  
  # Export results
  python validate_graph.py --input graph.json --output results.json
        """
    )
    
    # Input source
    input_group = parser.add_argument_group("Input Source")
    input_group.add_argument(
        "--input", "-i",
        help="Input graph file (JSON format)"
    )
    input_group.add_argument(
        "--neo4j",
        action="store_true",
        help="Use Neo4j database"
    )
    input_group.add_argument(
        "--predicted",
        help="File with predicted scores (for custom validation)"
    )
    input_group.add_argument(
        "--actual",
        help="File with actual scores (for custom validation)"
    )
    input_group.add_argument(
        "--types",
        help="File with component type mapping"
    )
    
    # Neo4j connection
    neo4j_group = parser.add_argument_group("Neo4j Connection")
    neo4j_group.add_argument(
        "--uri", "-u",
        default="bolt://localhost:7687",
        help="Neo4j bolt URI (default: bolt://localhost:7687)"
    )
    neo4j_group.add_argument(
        "--user", "-U",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    neo4j_group.add_argument(
        "--password", "-p",
        default="password",
        help="Neo4j password"
    )
    neo4j_group.add_argument(
        "--database", "-d",
        default="neo4j",
        help="Neo4j database name (default: neo4j)"
    )
    neo4j_group.add_argument(
        "--predicted-property",
        default="composite_score",
        help="Neo4j property with predicted scores"
    )
    neo4j_group.add_argument(
        "--actual-property",
        default="impact_score",
        help="Neo4j property with actual scores"
    )
    neo4j_group.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run full pipeline (analysis + simulation + validation)"
    )
    
    # Validation options
    val_group = parser.add_argument_group("Validation Options")
    val_group.add_argument(
        "--method", "-m",
        choices=["betweenness", "pagerank", "degree", "composite", "closeness", "eigenvector"],
        default="composite",
        help="Analysis method (default: composite)"
    )
    val_group.add_argument(
        "--compare-methods",
        action="store_true",
        help="Compare all analysis methods"
    )
    val_group.add_argument(
        "--by-type",
        action="store_true",
        default=True,
        help="Validate by component type"
    )
    val_group.add_argument(
        "--cascade",
        action="store_true",
        default=True,
        help="Enable cascade in simulation"
    )
    val_group.add_argument(
        "--no-cascade",
        action="store_false",
        dest="cascade",
        help="Disable cascade in simulation"
    )
    
    # Targets
    target_group = parser.add_argument_group("Validation Targets")
    target_group.add_argument(
        "--spearman-target",
        type=float,
        default=0.70,
        help="Spearman correlation target (default: 0.70)"
    )
    target_group.add_argument(
        "--f1-target",
        type=float,
        default=0.90,
        help="F1-score target (default: 0.90)"
    )
    target_group.add_argument(
        "--precision-target",
        type=float,
        default=0.80,
        help="Precision target (default: 0.80)"
    )
    target_group.add_argument(
        "--recall-target",
        type=float,
        default=0.80,
        help="Recall target (default: 0.80)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        help="Output file path (JSON format)"
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout"
    )
    output_group.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Check module availability
    if not HAS_VALIDATION:
        print(f"{RED}Error: Validation module not available.{RESET}")
        print(f"Import error: {IMPORT_ERROR}")
        print("Make sure you're running from the project root.")
        sys.exit(1)
    
    # Configure logging
    log_level = logging.WARNING if args.quiet else (
        logging.DEBUG if args.verbose else logging.INFO
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Determine mode and run
    result = None
    
    if args.predicted and args.actual:
        # Custom validation with score files
        result = run_custom_validation(args)
    
    elif args.neo4j:
        # Neo4j validation
        result = run_neo4j_validation(args)
    
    elif args.input:
        # File-based validation with pipeline
        graph = load_graph(args)
        if graph:
            result = run_validation_pipeline(args, graph)
    
    else:
        print(f"{RED}Error: Must specify --input, --neo4j, or --predicted/--actual{RESET}")
        parser.print_help()
        sys.exit(1)
    
    # Output results
    if result:
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n{GREEN}✓ Results saved to: {args.output}{RESET}")
        
        if args.json:
            print(json.dumps(result, indent=2, default=str))
    
    print(f"\n{GREEN}✓ Validation complete{RESET}")


if __name__ == "__main__":
    main()
