#!/usr/bin/env python3
"""
Validate Graph CLI - Version 5.0

Validates graph analysis predictions against simulation results.

Input Sources:
- JSON file (--input)
- Neo4j database (--neo4j)

Compares predicted criticality scores from topological analysis
against actual impact scores from failure simulation.

Layers Reported:
- application: app_to_app dependencies
- infrastructure: node_to_node dependencies
- app_broker: app_to_broker dependencies
- node_broker: node_to_broker dependencies

Usage:
    # Load from file
    python validate_graph.py --input graph.json
    
    # Load from Neo4j
    python validate_graph.py --neo4j
    python validate_graph.py --neo4j --uri bolt://localhost:7687
    
    # Load specific layer from Neo4j
    python validate_graph.py --neo4j --layer application
    
    # Compare all analysis methods
    python validate_graph.py --input graph.json --compare-methods
    
    # Export results
    python validate_graph.py --neo4j --output results.json

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


# =============================================================================
# Terminal Colors
# =============================================================================

class Colors:
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    RESET = "\033[0m"
    
    @classmethod
    def disable(cls):
        for attr in ["BOLD", "GREEN", "BLUE", "CYAN", "YELLOW", "RED", "GRAY", "RESET"]:
            setattr(cls, attr, "")


def use_colors() -> bool:
    import os
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and os.name != "nt"


# =============================================================================
# Output Helpers
# =============================================================================

def print_header(title: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title:^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}\n")


def print_section(title: str) -> None:
    print(f"\n{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{'-' * 40}")


def print_success(msg: str) -> None:
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")


def print_error(msg: str) -> None:
    print(f"{Colors.RED}✗{Colors.RESET} {msg}")


def print_info(msg: str) -> None:
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {msg}")


def metric_color(value: float, target: float) -> str:
    """Get color based on whether value meets target."""
    if value >= target:
        return Colors.GREEN
    elif value >= target * 0.8:
        return Colors.YELLOW
    else:
        return Colors.RED


def format_metric(name: str, value: float, target: float) -> str:
    """Format a metric with color based on target."""
    color = metric_color(value, target)
    status = "✓" if value >= target else "✗"
    return f"  {name:<20} {color}{value:.4f}{Colors.RESET} (target: ≥{target}) {status}"


def status_color(status) -> str:
    """Get color for validation status."""
    from src.validation import ValidationStatus
    
    if status == ValidationStatus.PASSED:
        return Colors.GREEN
    elif status == ValidationStatus.PARTIAL:
        return Colors.YELLOW
    else:
        return Colors.RED


# =============================================================================
# Graph Loading
# =============================================================================

def load_graph_from_file(args):
    """Load simulation graph from JSON file."""
    from src.simulation import SimulationGraph
    
    if not args.input:
        return None
    
    input_path = Path(args.input)
    if not input_path.exists():
        print_error(f"File not found: {input_path}")
        return None
    
    print_info(f"Loading graph from {input_path}")
    
    try:
        graph = SimulationGraph.from_json(str(input_path))
        summary = graph.summary()
        
        print_success(f"Loaded {summary['total_components']} components, {summary['total_edges']} edges")
        
        return graph
    except Exception as e:
        print_error(f"Failed to load graph: {e}")
        return None


def load_graph_from_neo4j(args):
    """Load simulation graph from Neo4j database."""
    from src.simulation import Neo4jSimulationClient, check_neo4j_available
    
    if not check_neo4j_available():
        print_error("Neo4j driver not installed. Install with: pip install neo4j")
        return None
    
    print_info(f"Connecting to Neo4j at {args.uri}")
    
    try:
        with Neo4jSimulationClient(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        ) as client:
            # Verify connection
            if not client.verify_connection():
                print_error("Failed to connect to Neo4j")
                return None
            
            print_success("Connected to Neo4j")
            
            # Get statistics
            stats = client.get_statistics()
            print_info(f"Components in DB: {stats['total_components']}")
            print_info(f"Edges in DB: {stats['total_edges']}")
            
            # Load graph
            if args.layer:
                print_info(f"Loading layer: {args.layer}")
                graph = client.load_layer(args.layer)
            else:
                print_info("Loading full graph")
                graph = client.load_full_graph()
            
            summary = graph.summary()
            print_success(f"Loaded {summary['total_components']} components, {summary['total_edges']} edges")
            
            return graph
            
    except Exception as e:
        print_error(f"Failed to load from Neo4j: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None


def load_graph(args):
    """Load graph from file or Neo4j."""
    if args.neo4j:
        return load_graph_from_neo4j(args)
    elif args.input:
        return load_graph_from_file(args)
    else:
        print_error("Specify --input <file> or --neo4j")
        return None


# =============================================================================
# Output Functions
# =============================================================================

def print_validation_result(result, targets, verbose: bool = False) -> None:
    """Print validation result details."""
    
    # Overall status
    color = status_color(result.status)
    print(f"\n  {Colors.BOLD}Overall Status:{Colors.RESET} {color}{result.status.value.upper()}{Colors.RESET}")
    
    # Correlation metrics
    print_section("Correlation Metrics")
    print(format_metric("Spearman ρ", result.spearman, targets.spearman))
    print(format_metric("Pearson r", result.correlation.pearson, targets.pearson))
    print(format_metric("Kendall τ", result.correlation.kendall, targets.kendall))
    
    if result.spearman_ci:
        ci = result.spearman_ci
        print(f"  {'Spearman 95% CI':<20} [{ci.lower:.4f}, {ci.upper:.4f}]")
    
    # Classification metrics
    print_section("Classification Metrics")
    print(format_metric("F1-Score", result.f1_score, targets.f1_score))
    print(format_metric("Precision", result.classification.precision, targets.precision))
    print(format_metric("Recall", result.classification.recall, targets.recall))
    print(format_metric("Accuracy", result.classification.accuracy, targets.accuracy))
    
    cm = result.classification.confusion_matrix
    print(f"\n  Confusion Matrix:")
    print(f"    TP={cm.true_positives:3d}  FP={cm.false_positives:3d}")
    print(f"    FN={cm.false_negatives:3d}  TN={cm.true_negatives:3d}")
    
    # Ranking metrics
    print_section("Ranking Metrics")
    print(format_metric("Top-5 Overlap", result.ranking.top_5_overlap, targets.top_5_overlap))
    print(format_metric("Top-10 Overlap", result.ranking.top_10_overlap, targets.top_10_overlap))
    print(f"  {'NDCG':<20} {result.ranking.ndcg:.4f}")
    print(f"  {'MRR':<20} {result.ranking.mrr:.4f}")
    
    # Verbose details
    if verbose:
        fps = result.get_false_positives()
        if fps:
            print_section("False Positives")
            for comp in fps[:5]:
                print(f"    {comp.component_id} ({comp.component_type}): "
                      f"pred={comp.predicted_score:.4f}, actual={comp.actual_score:.4f}")
        
        fns = result.get_false_negatives()
        if fns:
            print_section("False Negatives")
            for comp in fns[:5]:
                print(f"    {comp.component_id} ({comp.component_type}): "
                      f"pred={comp.predicted_score:.4f}, actual={comp.actual_score:.4f}")


def print_layer_results(result, targets) -> None:
    """Print layer-specific results."""
    print_section("Results by Layer")
    
    print(f"\n  {'Layer':<20} {'ρ':<10} {'F1':<10} {'Status':<10} {'n'}")
    print(f"  {'-' * 60}")
    
    for layer, layer_result in sorted(result.by_layer.items()):
        rho_color = metric_color(layer_result.spearman, targets.spearman)
        f1_color = metric_color(layer_result.f1_score, targets.f1_score)
        status = "PASSED" if layer_result.passed else "FAILED"
        status_c = Colors.GREEN if layer_result.passed else Colors.RED
        
        print(f"  {layer:<20} "
              f"{rho_color}{layer_result.spearman:<10.4f}{Colors.RESET} "
              f"{f1_color}{layer_result.f1_score:<10.4f}{Colors.RESET} "
              f"{status_c}{status:<10}{Colors.RESET} "
              f"{layer_result.count}")


def print_type_results(result, targets) -> None:
    """Print component type results."""
    if not result.by_type:
        return
    
    print_section("Results by Component Type")
    
    print(f"\n  {'Type':<15} {'ρ':<10} {'F1':<10} {'n'}")
    print(f"  {'-' * 45}")
    
    for comp_type, type_result in sorted(result.by_type.items()):
        rho_color = metric_color(type_result.spearman, targets.spearman)
        f1_color = metric_color(type_result.f1_score, targets.f1_score)
        
        print(f"  {comp_type:<15} "
              f"{rho_color}{type_result.spearman:<10.4f}{Colors.RESET} "
              f"{f1_color}{type_result.f1_score:<10.4f}{Colors.RESET} "
              f"{type_result.n_samples}")


def print_method_comparison(result, targets) -> None:
    """Print method comparison results."""
    if not result.method_comparison:
        return
    
    print_section("Method Comparison")
    
    print(f"\n  {'Method':<15} {'ρ':<10} {'F1':<10} {'Status'}")
    print(f"  {'-' * 50}")
    
    for method, comp in sorted(result.method_comparison.items()):
        rho_color = metric_color(comp.spearman, targets.spearman)
        f1_color = metric_color(comp.f1_score, targets.f1_score)
        status_c = status_color(comp.status)
        
        print(f"  {method:<15} "
              f"{rho_color}{comp.spearman:<10.4f}{Colors.RESET} "
              f"{f1_color}{comp.f1_score:<10.4f}{Colors.RESET} "
              f"{status_c}{comp.status.value}{Colors.RESET}")
    
    # Best method
    best = result.get_best_method()
    if best:
        print(f"\n  {Colors.BOLD}Best Method:{Colors.RESET} {best}")


def print_timing(result) -> None:
    """Print timing breakdown."""
    print_section("Timing")
    
    print(f"  {'Analysis':<15} {result.analysis_time_ms:.0f}ms")
    print(f"  {'Simulation':<15} {result.simulation_time_ms:.0f}ms")
    print(f"  {'Validation':<15} {result.validation_time_ms:.0f}ms")


# =============================================================================
# Validation Pipeline
# =============================================================================

def run_validation_pipeline(args, graph) -> Optional[Dict[str, Any]]:
    """Run validation pipeline."""
    from src.validation import (
        ValidationPipeline,
        ValidationTargets,
        AnalysisMethod,
    )
    
    print_header("VALIDATION PIPELINE")
    
    # Build targets
    targets = ValidationTargets(
        spearman=args.spearman_target,
        f1_score=args.f1_target,
        precision=args.precision_target,
        recall=args.recall_target,
        top_5_overlap=args.top5_target,
        top_10_overlap=args.top10_target,
    )
    
    # Get method
    method = AnalysisMethod(args.method)
    
    print_info(f"Analysis Method: {method.value}")
    print_info(f"Cascade: {'enabled' if args.cascade else 'disabled'}")
    
    # Create pipeline
    pipeline = ValidationPipeline(
        targets=targets,
        seed=args.seed,
        cascade=args.cascade,
    )
    
    # Run validation
    try:
        result = pipeline.run(
            graph,
            analysis_method=method,
            compare_methods=args.compare_methods,
        )
    except Exception as e:
        print_error(f"Validation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None
    
    # Print results
    print_validation_result(result.validation, targets, args.verbose)
    
    # Layer results
    if result.by_layer:
        print_layer_results(result, targets)
    
    # Type results
    if args.verbose and result.by_type:
        print_type_results(result, targets)
    
    # Method comparison
    if result.method_comparison:
        print_method_comparison(result, targets)
    
    # Timing
    print_timing(result)
    
    # Summary
    print_section("Summary")
    color = status_color(result.validation.status)
    print(f"\n  {Colors.BOLD}Status:{Colors.RESET} {color}{result.validation.status.value.upper()}{Colors.RESET}")
    print(f"  Spearman ρ: {result.spearman:.4f} (target: ≥{targets.spearman})")
    print(f"  F1-Score: {result.f1_score:.4f} (target: ≥{targets.f1_score})")
    
    if result.method_comparison:
        best = result.get_best_method()
        print(f"  Best Method: {best}")
    
    # Layer summary
    if result.by_layer:
        print(f"\n  Layer Summary:")
        passed_layers = [l for l, r in result.by_layer.items() if r.passed]
        print(f"    Passed: {len(passed_layers)}/{len(result.by_layer)}")
        for layer in passed_layers:
            print(f"      ✓ {layer}")
    
    return result.to_dict()


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate graph analysis against simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Load from file
    python validate_graph.py --input graph.json
    python validate_graph.py --input graph.json --compare-methods
    
    # Load from Neo4j
    python validate_graph.py --neo4j
    python validate_graph.py --neo4j --uri bolt://localhost:7687
    
    # Load specific layer from Neo4j
    python validate_graph.py --neo4j --layer application
    
    # Custom targets
    python validate_graph.py --input graph.json --spearman-target 0.80
    
    # Export results
    python validate_graph.py --neo4j --output results.json

Layers Reported:
    application    - app_to_app dependencies
    infrastructure - node_to_node dependencies
    app_broker     - app_to_broker dependencies
    node_broker    - node_to_broker dependencies
        """
    )
    
    # Input source group
    input_group = parser.add_argument_group("Input Source")
    input_group.add_argument(
        "--input", "-i",
        help="Input graph JSON file"
    )
    input_group.add_argument(
        "--neo4j", "-n",
        action="store_true",
        help="Load graph from Neo4j database"
    )
    
    # Neo4j connection options
    neo4j_group = parser.add_argument_group("Neo4j Connection")
    neo4j_group.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j bolt URI (default: bolt://localhost:7687)"
    )
    neo4j_group.add_argument(
        "--user",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    neo4j_group.add_argument(
        "--password",
        default="password",
        help="Neo4j password (default: password)"
    )
    neo4j_group.add_argument(
        "--database",
        default="neo4j",
        help="Neo4j database name (default: neo4j)"
    )
    neo4j_group.add_argument(
        "--layer",
        choices=["application", "infrastructure", "app_broker", "node_broker"],
        help="Load specific layer from Neo4j"
    )
    
    # Analysis options
    analysis_group = parser.add_argument_group("Analysis Options")
    analysis_group.add_argument(
        "--method", "-m",
        choices=["betweenness", "pagerank", "degree", "composite"],
        default="composite",
        help="Analysis method (default: composite)"
    )
    analysis_group.add_argument(
        "--compare-methods", "-c",
        action="store_true",
        help="Compare all analysis methods"
    )
    
    # Simulation options
    sim_group = parser.add_argument_group("Simulation Options")
    sim_group.add_argument(
        "--cascade",
        action="store_true",
        default=True,
        help="Enable cascade propagation (default)"
    )
    sim_group.add_argument(
        "--no-cascade",
        action="store_false",
        dest="cascade",
        help="Disable cascade propagation"
    )
    
    # Target thresholds
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
    target_group.add_argument(
        "--top5-target",
        type=float,
        default=0.60,
        help="Top-5 overlap target (default: 0.60)"
    )
    target_group.add_argument(
        "--top10-target",
        type=float,
        default=0.70,
        help="Top-10 overlap target (default: 0.70)"
    )
    
    # Common options
    parser.add_argument(
        "--seed", "-s",
        type=int,
        help="Random seed"
    )
    parser.add_argument(
        "--output", "-o",
        help="Export results to JSON"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colors"
    )
    
    args = parser.parse_args()
    
    # Handle colors
    if args.no_color or not use_colors():
        Colors.disable()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level)
    
    # Import check
    try:
        from src.simulation import SimulationGraph
        from src.validation import ValidationPipeline
    except ImportError as e:
        print_error(f"Import failed: {e}")
        return 1
    
    # Load graph
    graph = load_graph(args)
    if graph is None:
        return 1
    
    # Run validation
    results = run_validation_pipeline(args, graph)
    
    if results is None:
        return 1
    
    # Export
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print_success(f"Results exported to {output_path}")
    
    print_header("Validation Complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())