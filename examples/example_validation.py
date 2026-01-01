#!/usr/bin/env python3
"""
Example Validation Script - Version 5.0

Demonstrates the validation capabilities of the software-as-a-graph project.

Features Demonstrated:
1. Statistical Metrics - Correlation, classification, ranking
2. Component-Type Validation - Separate validation per type
3. Method Comparison - Compare multiple analysis methods
4. Full Pipeline - Analysis + Simulation + Validation
5. Custom Score Validation - Validate pre-computed scores

Usage:
    # Run all demos (no Neo4j required)
    python examples/example_validation.py --demo
    
    # Run with Neo4j
    python examples/example_validation.py --uri bolt://localhost:7687 --password secret
    
    # Export results
    python examples/example_validation.py --demo --output results.json

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation import (
    # Enums
    ValidationStatus,
    AnalysisMethod,
    # Metrics
    ValidationTargets,
    CorrelationMetrics,
    ClassificationMetrics,
    RankingMetrics,
    # Functions
    spearman_correlation,
    pearson_correlation,
    calculate_correlation,
    calculate_classification,
    calculate_ranking,
    bootstrap_confidence_interval,
    # Validator
    Validator,
    ValidationResult,
    validate_predictions,
    quick_validate,
    # Pipeline
    GraphAnalyzer,
    ValidationPipeline,
    PipelineResult,
    run_validation,
)


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
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}{title.center(60)}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")


def print_subheader(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")
    print(f"{'-' * 40}")


def print_success(msg: str) -> None:
    print(f"{GREEN}✓ {msg}{RESET}")


def print_info(msg: str) -> None:
    print(f"{BLUE}ℹ {msg}{RESET}")


def metric_color(value: float, target: float) -> str:
    if value >= target:
        return GREEN
    elif value >= target * 0.9:
        return YELLOW
    return RED


# =============================================================================
# Demo Graph Creation
# =============================================================================

def create_demo_graph():
    """Create a demo graph for validation examples"""
    from src.simulation import (
        SimulationGraph, Component, Edge,
        ComponentType, EdgeType, QoSPolicy
    )
    
    graph = SimulationGraph()
    
    # Applications (8)
    apps = [
        ("sensor1", "Temperature Sensor"),
        ("sensor2", "Humidity Sensor"),
        ("processor", "Data Processor"),
        ("alerter", "Alert Generator"),
        ("controller", "System Controller"),
        ("dashboard", "Dashboard"),
        ("logger", "Data Logger"),
        ("notifier", "Notification Service"),
    ]
    for app_id, name in apps:
        graph.add_component(Component(
            id=app_id, type=ComponentType.APPLICATION, name=name
        ))
    
    # Brokers (2)
    for i in range(1, 3):
        graph.add_component(Component(
            id=f"broker{i}", type=ComponentType.BROKER
        ))
    
    # Topics (5)
    for i in range(1, 6):
        graph.add_component(Component(
            id=f"topic{i}", type=ComponentType.TOPIC
        ))
    
    # Nodes (2)
    for i in range(1, 3):
        graph.add_component(Component(
            id=f"node{i}", type=ComponentType.NODE
        ))
    
    # Pub/Sub relationships
    edges = [
        ("sensor1", "topic1", EdgeType.PUBLISHES_TO),
        ("sensor2", "topic1", EdgeType.PUBLISHES_TO),
        ("processor", "topic2", EdgeType.PUBLISHES_TO),
        ("processor", "topic1", EdgeType.SUBSCRIBES_TO),
        ("alerter", "topic3", EdgeType.PUBLISHES_TO),
        ("alerter", "topic1", EdgeType.SUBSCRIBES_TO),
        ("controller", "topic4", EdgeType.PUBLISHES_TO),
        ("controller", "topic2", EdgeType.SUBSCRIBES_TO),
        ("dashboard", "topic2", EdgeType.SUBSCRIBES_TO),
        ("dashboard", "topic3", EdgeType.SUBSCRIBES_TO),
        ("logger", "topic1", EdgeType.SUBSCRIBES_TO),
        ("logger", "topic5", EdgeType.PUBLISHES_TO),
        ("notifier", "topic3", EdgeType.SUBSCRIBES_TO),
    ]
    
    for source, target, edge_type in edges:
        graph.add_edge(Edge(source=source, target=target, edge_type=edge_type))
    
    # Broker routes
    graph.add_edge(Edge(source="broker1", target="topic1", edge_type=EdgeType.ROUTES))
    graph.add_edge(Edge(source="broker1", target="topic2", edge_type=EdgeType.ROUTES))
    graph.add_edge(Edge(source="broker1", target="topic3", edge_type=EdgeType.ROUTES))
    graph.add_edge(Edge(source="broker2", target="topic4", edge_type=EdgeType.ROUTES))
    graph.add_edge(Edge(source="broker2", target="topic5", edge_type=EdgeType.ROUTES))
    
    # RUNS_ON
    graph.add_edge(Edge(source="sensor1", target="node1", edge_type=EdgeType.RUNS_ON))
    graph.add_edge(Edge(source="sensor2", target="node1", edge_type=EdgeType.RUNS_ON))
    graph.add_edge(Edge(source="broker1", target="node1", edge_type=EdgeType.RUNS_ON))
    graph.add_edge(Edge(source="processor", target="node2", edge_type=EdgeType.RUNS_ON))
    graph.add_edge(Edge(source="broker2", target="node2", edge_type=EdgeType.RUNS_ON))
    
    return graph


# =============================================================================
# Demo Functions
# =============================================================================

def demo_correlation_metrics() -> Dict[str, Any]:
    """Demo 1: Correlation Metrics"""
    print_header("DEMO 1: CORRELATION METRICS")
    
    # Create test data
    predicted = [0.9, 0.75, 0.6, 0.45, 0.3, 0.15]
    actual = [0.85, 0.7, 0.55, 0.5, 0.35, 0.2]
    
    print(f"Test Data:")
    print(f"  Predicted: {predicted}")
    print(f"  Actual:    {actual}")
    
    # Calculate correlations
    print_subheader("Correlation Coefficients")
    
    spearman = spearman_correlation(predicted, actual)
    pearson = pearson_correlation(predicted, actual)
    
    print(f"  Spearman ρ: {GREEN}{spearman:.4f}{RESET} (rank correlation)")
    print(f"  Pearson r:  {GREEN}{pearson:.4f}{RESET} (linear correlation)")
    
    # Using calculate_correlation
    metrics = calculate_correlation(predicted, actual)
    print(f"\n  Kendall τ:  {metrics.kendall:.4f}")
    print(f"  Samples:    {metrics.n_samples}")
    
    # Bootstrap confidence interval
    print_subheader("Bootstrap Confidence Interval")
    
    ci = bootstrap_confidence_interval(
        predicted, actual, spearman_correlation,
        n_bootstrap=1000, confidence=0.95, seed=42
    )
    
    print(f"  Spearman 95% CI: [{ci.lower:.4f}, {ci.upper:.4f}]")
    print(f"  Point Estimate:  {ci.estimate:.4f}")
    print(f"  CI Width:        {ci.width:.4f}")
    
    print_success("Correlation metrics demo complete")
    
    return {
        "spearman": spearman,
        "pearson": pearson,
        "kendall": metrics.kendall,
        "ci_lower": ci.lower,
        "ci_upper": ci.upper,
    }


def demo_classification_metrics() -> Dict[str, Any]:
    """Demo 2: Classification Metrics"""
    print_header("DEMO 2: CLASSIFICATION METRICS")
    
    # Create test data
    predicted_scores = [0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1]
    actual_scores = [0.85, 0.75, 0.65, 0.45, 0.35, 0.25, 0.15]
    
    print(f"Test Data:")
    print(f"  Predicted: {predicted_scores}")
    print(f"  Actual:    {actual_scores}")
    
    # Calculate with threshold
    threshold = 0.5
    print(f"  Threshold: {threshold}")
    
    metrics = calculate_classification(predicted_scores, actual_scores, threshold)
    
    print_subheader("Classification Results")
    
    cm = metrics.confusion_matrix
    print(f"\n  Confusion Matrix:")
    print(f"    TP={cm.true_positives}  FP={cm.false_positives}")
    print(f"    FN={cm.false_negatives}  TN={cm.true_negatives}")
    
    print(f"\n  Metrics:")
    print(f"    Precision: {GREEN}{cm.precision:.4f}{RESET}")
    print(f"    Recall:    {GREEN}{cm.recall:.4f}{RESET}")
    print(f"    F1-Score:  {GREEN}{cm.f1_score:.4f}{RESET}")
    print(f"    Accuracy:  {GREEN}{cm.accuracy:.4f}{RESET}")
    print(f"    MCC:       {cm.mcc:.4f}")
    
    print_success("Classification metrics demo complete")
    
    return {
        "precision": cm.precision,
        "recall": cm.recall,
        "f1_score": cm.f1_score,
        "accuracy": cm.accuracy,
    }


def demo_ranking_metrics() -> Dict[str, Any]:
    """Demo 3: Ranking Metrics"""
    print_header("DEMO 3: RANKING METRICS")
    
    # Create test rankings
    predicted_ranking = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    actual_ranking = ["a", "c", "b", "e", "d", "f", "h", "g", "j", "i"]
    
    print(f"Test Rankings:")
    print(f"  Predicted: {predicted_ranking}")
    print(f"  Actual:    {actual_ranking}")
    
    # Calculate ranking metrics
    metrics = calculate_ranking(predicted_ranking, actual_ranking)
    
    print_subheader("Ranking Metrics")
    
    print(f"  Top-5 Overlap:  {GREEN}{metrics.top_5_overlap:.2%}{RESET}")
    print(f"  Top-10 Overlap: {GREEN}{metrics.top_10_overlap:.2%}{RESET}")
    print(f"  NDCG:           {metrics.ndcg:.4f}")
    print(f"  MRR:            {metrics.mrr:.4f}")
    print(f"  Avg Rank Diff:  {metrics.rank_difference_mean:.2f}")
    
    # Top-K overlaps
    print(f"\n  Top-K Overlaps:")
    for k, overlap in metrics.top_k_overlaps.items():
        print(f"    Top-{k}: {overlap:.2%}")
    
    print_success("Ranking metrics demo complete")
    
    return {
        "top_5_overlap": metrics.top_5_overlap,
        "top_10_overlap": metrics.top_10_overlap,
        "ndcg": metrics.ndcg,
    }


def demo_validator() -> Dict[str, Any]:
    """Demo 4: Validator with Component Types"""
    print_header("DEMO 4: VALIDATOR WITH COMPONENT TYPES")
    
    # Create test data with component types
    predicted = {
        "app1": 0.9, "app2": 0.7, "app3": 0.5,
        "broker1": 0.85, "broker2": 0.65,
        "topic1": 0.6, "topic2": 0.4,
        "node1": 0.8, "node2": 0.3,
    }
    
    actual = {
        "app1": 0.85, "app2": 0.75, "app3": 0.45,
        "broker1": 0.9, "broker2": 0.6,
        "topic1": 0.55, "topic2": 0.45,
        "node1": 0.75, "node2": 0.35,
    }
    
    component_types = {
        "app1": "Application", "app2": "Application", "app3": "Application",
        "broker1": "Broker", "broker2": "Broker",
        "topic1": "Topic", "topic2": "Topic",
        "node1": "Node", "node2": "Node",
    }
    
    print(f"Components: {len(predicted)}")
    print(f"  Applications: 3")
    print(f"  Brokers: 2")
    print(f"  Topics: 2")
    print(f"  Nodes: 2")
    
    # Configure validator
    targets = ValidationTargets(spearman=0.70, f1_score=0.80)
    validator = Validator(targets=targets, seed=42)
    
    # Run validation
    result = validator.validate(predicted, actual, component_types)
    
    print_subheader("Overall Results")
    
    status_color = GREEN if result.passed else RED
    print(f"  Status: {status_color}{BOLD}{result.status.value.upper()}{RESET}")
    print(f"  Spearman ρ: {metric_color(result.spearman, 0.70)}{result.spearman:.4f}{RESET}")
    print(f"  F1-Score: {metric_color(result.f1_score, 0.80)}{result.f1_score:.4f}{RESET}")
    
    # Confidence interval
    if result.spearman_ci:
        ci = result.spearman_ci
        print(f"  95% CI: [{ci.lower:.4f}, {ci.upper:.4f}]")
    
    # By component type
    print_subheader("By Component Type")
    
    for comp_type, type_result in result.by_type.items():
        color = GREEN if type_result.status == ValidationStatus.PASSED else YELLOW
        print(f"  {comp_type}:")
        print(f"    Count: {type_result.count}")
        print(f"    Spearman: {color}{type_result.correlation.spearman:.4f}{RESET}")
        print(f"    F1-Score: {type_result.classification.f1_score:.4f}")
    
    # Misclassified components
    misclassified = result.get_misclassified()
    if misclassified:
        print_subheader(f"Misclassified Components ({len(misclassified)})")
        for comp in misclassified[:5]:
            print(f"  {comp.component_id} ({comp.component_type}): "
                  f"pred={comp.predicted_score:.2f}, actual={comp.actual_score:.2f}")
    
    print_success("Validator demo complete")
    
    return result.to_dict()


def demo_pipeline() -> Dict[str, Any]:
    """Demo 5: Full Validation Pipeline"""
    print_header("DEMO 5: VALIDATION PIPELINE")
    
    # Create demo graph
    graph = create_demo_graph()
    stats = graph.get_statistics()
    
    print(f"Graph Statistics:")
    print(f"  Components: {stats['total_components']}")
    print(f"  Edges: {stats['total_edges']}")
    
    # Configure pipeline
    targets = ValidationTargets(spearman=0.70, f1_score=0.85)
    pipeline = ValidationPipeline(targets=targets, seed=42, cascade=True)
    
    # Run pipeline
    print(f"\nRunning validation pipeline...")
    print(f"  Analysis Method: composite")
    print(f"  Cascade Enabled: True")
    
    result = pipeline.run(
        graph,
        analysis_method=AnalysisMethod.COMPOSITE,
        compare_methods=True,
        validate_by_type=True,
    )
    
    print_subheader("Pipeline Results")
    
    status_color = GREEN if result.passed else RED
    print(f"  Status: {status_color}{BOLD}{result.validation.status.value.upper()}{RESET}")
    print(f"  Spearman ρ: {result.spearman:.4f}")
    print(f"  F1-Score: {result.f1_score:.4f}")
    
    # Method comparison
    if result.method_comparison:
        print_subheader("Method Comparison")
        
        sorted_methods = sorted(
            result.method_comparison.items(),
            key=lambda x: -x[1].spearman
        )
        
        print(f"  {'Method':<15} {'Spearman':<12} {'F1-Score':<12} {'Status'}")
        print(f"  {'-' * 50}")
        
        for method, comp in sorted_methods:
            sp_color = metric_color(comp.spearman, 0.70)
            f1_color = metric_color(comp.f1_score, 0.85)
            print(f"  {method:<15} {sp_color}{comp.spearman:.4f}{RESET}       "
                  f"{f1_color}{comp.f1_score:.4f}{RESET}       {comp.status.value}")
        
        best = result.get_best_method()
        print(f"\n  {GREEN}Best Method: {best}{RESET}")
    
    # By component type
    if result.by_component_type:
        print_subheader("By Component Type")
        
        for comp_type, type_result in result.by_component_type.items():
            print(f"  {comp_type}: ρ={type_result.spearman:.4f}, "
                  f"F1={type_result.f1_score:.4f}")
    
    print_success("Pipeline demo complete")
    
    return result.to_dict()


def demo_quick_functions() -> Dict[str, Any]:
    """Demo 6: Quick Validation Functions"""
    print_header("DEMO 6: QUICK VALIDATION FUNCTIONS")
    
    # Create test data
    predicted = {"a": 0.9, "b": 0.7, "c": 0.5, "d": 0.3, "e": 0.1}
    actual = {"a": 0.85, "b": 0.75, "c": 0.45, "d": 0.35, "e": 0.15}
    
    print(f"Test Data: {len(predicted)} components")
    
    # Quick validate
    print_subheader("quick_validate()")
    
    spearman, f1, passed = quick_validate(predicted, actual)
    
    print(f"  Spearman: {spearman:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Passed: {GREEN if passed else RED}{passed}{RESET}")
    
    # validate_predictions
    print_subheader("validate_predictions()")
    
    result = validate_predictions(predicted, actual)
    
    print(f"  Status: {result.status.value}")
    print(f"  Correlation: {result.correlation.to_dict()}")
    
    # run_validation (with graph)
    print_subheader("run_validation() with Graph")
    
    graph = create_demo_graph()
    pipeline_result = run_validation(graph, method="composite", seed=42)
    
    print(f"  Status: {pipeline_result.validation.status.value}")
    print(f"  Spearman: {pipeline_result.spearman:.4f}")
    print(f"  Best Method: {pipeline_result.get_best_method()}")
    
    print_success("Quick functions demo complete")
    
    return {
        "quick_validate": {"spearman": spearman, "f1": f1, "passed": passed},
        "validate_predictions": result.to_dict(),
    }


# =============================================================================
# Live Mode (Neo4j)
# =============================================================================

def run_live_mode(args) -> Optional[Dict[str, Any]]:
    """Run validation with Neo4j"""
    print_header("LIVE MODE - NEO4J VALIDATION")
    
    try:
        from src.validation import Neo4jValidationClient
    except ImportError as e:
        print(f"{RED}Error: Neo4j driver not installed{RESET}")
        print(f"Install with: pip install neo4j")
        return None
    
    print(f"Connecting to Neo4j at {args.uri}...")
    
    try:
        with Neo4jValidationClient(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        ) as client:
            # Get stats
            stats = client.get_validation_stats()
            
            print(f"\n  Data Availability:")
            for comp_type, count in stats["components"].items():
                print(f"    {comp_type}: {count} components")
            
            total = sum(stats["components"].values())
            if total == 0:
                print(f"\n{YELLOW}Database is empty. Running demo mode instead.{RESET}")
                return run_demo_mode(args)
            
            # Run full validation
            print(f"\n  Running full validation pipeline...")
            result = client.run_full_validation(seed=args.seed)
            
            print(f"\n  Status: {result.validation.status.value}")
            print(f"  Spearman: {result.spearman:.4f}")
            print(f"  F1-Score: {result.f1_score:.4f}")
            
            return result.to_dict()
    
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")
        print(f"\n{YELLOW}Falling back to demo mode...{RESET}")
        return run_demo_mode(args)


# =============================================================================
# Demo Mode
# =============================================================================

def run_demo_mode(args) -> Dict[str, Any]:
    """Run all demos without Neo4j"""
    results = {}
    
    # Demo 1: Correlation Metrics
    results["correlation"] = demo_correlation_metrics()
    
    # Demo 2: Classification Metrics
    results["classification"] = demo_classification_metrics()
    
    # Demo 3: Ranking Metrics
    results["ranking"] = demo_ranking_metrics()
    
    # Demo 4: Validator
    results["validator"] = demo_validator()
    
    # Demo 5: Pipeline
    results["pipeline"] = demo_pipeline()
    
    # Demo 6: Quick Functions
    results["quick_functions"] = demo_quick_functions()
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validation module examples and demonstrations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all demos
  python examples/example_validation.py --demo
  
  # Run with Neo4j
  python examples/example_validation.py --uri bolt://localhost:7687 --password secret
  
  # Export results
  python examples/example_validation.py --demo --output results.json
        """
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode (no Neo4j required)"
    )
    parser.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j bolt URI"
    )
    parser.add_argument(
        "--user",
        default="neo4j",
        help="Neo4j username"
    )
    parser.add_argument(
        "--password",
        default="password",
        help="Neo4j password"
    )
    parser.add_argument(
        "--database",
        default="neo4j",
        help="Neo4j database name"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for results (JSON)"
    )
    
    args = parser.parse_args()
    
    print_header("SOFTWARE-AS-A-GRAPH VALIDATION EXAMPLES")
    print(f"Version: 5.0")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run appropriate mode
    if args.demo:
        results = run_demo_mode(args)
    else:
        results = run_live_mode(args)
        if results is None:
            results = run_demo_mode(args)
    
    # Export if requested
    if args.output and results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n{GREEN}✓ Results saved to: {output_path}{RESET}")
    
    # Summary
    print_header("EXAMPLES COMPLETE")
    print(f"{GREEN}All validation examples completed successfully!{RESET}")
    print(f"\nKey Capabilities Demonstrated:")
    print(f"  ✓ Correlation metrics (Spearman, Pearson, Kendall)")
    print(f"  ✓ Classification metrics (F1, Precision, Recall)")
    print(f"  ✓ Ranking metrics (Top-K Overlap, NDCG, MRR)")
    print(f"  ✓ Component-type specific validation")
    print(f"  ✓ Analysis method comparison")
    print(f"  ✓ Full validation pipeline")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
