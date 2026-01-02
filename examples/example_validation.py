#!/usr/bin/env python3
"""
Validation Example - Version 5.0

Demonstrates the validation module:
1. Basic validation of predictions vs actuals
2. Layer-specific validation
3. Method comparison
4. Full validation pipeline

Usage:
    python examples/example_validation.py
    python examples/example_validation.py --verbose

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Terminal Output
# =============================================================================

BOLD = "\033[1m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
GRAY = "\033[90m"
RESET = "\033[0m"


def print_header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}{title:^60}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")


def print_section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")
    print(f"{'-' * 40}")


def print_success(msg: str) -> None:
    print(f"{GREEN}✓{RESET} {msg}")


def print_info(msg: str) -> None:
    print(f"{BLUE}ℹ{RESET} {msg}")


def metric_color(value: float, target: float) -> str:
    if value >= target:
        return GREEN
    elif value >= target * 0.8:
        return YELLOW
    return RED


def status_color(status) -> str:
    from src.validation import ValidationStatus
    
    if status == ValidationStatus.PASSED:
        return GREEN
    elif status == ValidationStatus.PARTIAL:
        return YELLOW
    return RED


# =============================================================================
# Demo Functions
# =============================================================================

def demo_correlation_metrics():
    """Demo 1: Correlation metrics."""
    print_header("DEMO 1: CORRELATION METRICS")
    
    from src.validation import (
        spearman_correlation,
        pearson_correlation,
        kendall_correlation,
        calculate_correlation,
    )
    
    print("Correlation measures how well predictions match actuals.")
    print()
    
    # Perfect correlation
    print_section("Perfect Correlation")
    x = [0.9, 0.7, 0.5, 0.3, 0.1]
    y = [0.9, 0.7, 0.5, 0.3, 0.1]
    
    print(f"  Predicted: {x}")
    print(f"  Actual:    {y}")
    print(f"  Spearman:  {GREEN}{spearman_correlation(x, y):.4f}{RESET}")
    
    # Good correlation
    print_section("Good Correlation")
    x = [0.9, 0.7, 0.5, 0.3, 0.1]
    y = [0.85, 0.65, 0.55, 0.25, 0.15]  # Slightly different
    
    result = calculate_correlation(x, y)
    print(f"  Predicted: {x}")
    print(f"  Actual:    {y}")
    print(f"  Spearman:  {GREEN}{result.spearman:.4f}{RESET}")
    print(f"  Pearson:   {result.pearson:.4f}")
    print(f"  Kendall:   {result.kendall:.4f}")
    
    # Poor correlation
    print_section("Poor Correlation")
    x = [0.9, 0.7, 0.5, 0.3, 0.1]
    y = [0.3, 0.9, 0.1, 0.7, 0.5]  # Different order
    
    result = calculate_correlation(x, y)
    print(f"  Predicted: {x}")
    print(f"  Actual:    {y}")
    print(f"  Spearman:  {RED}{result.spearman:.4f}{RESET}")
    
    print_success("Correlation metrics demonstrated")


def demo_classification_metrics():
    """Demo 2: Classification metrics."""
    print_header("DEMO 2: CLASSIFICATION METRICS")
    
    from src.validation import (
        calculate_classification,
        ConfusionMatrix,
    )
    
    print("Classification determines if critical components are identified.")
    print()
    
    # Perfect classification
    print_section("Confusion Matrix Example")
    
    cm = ConfusionMatrix(
        true_positives=8,
        false_positives=2,
        false_negatives=1,
        true_negatives=89,
    )
    
    print(f"  True Positives:  {cm.true_positives} (predicted critical, actually critical)")
    print(f"  False Positives: {cm.false_positives} (predicted critical, not actually)")
    print(f"  False Negatives: {cm.false_negatives} (not predicted, but actually critical)")
    print(f"  True Negatives:  {cm.true_negatives} (not predicted, not critical)")
    print()
    print(f"  Precision: {GREEN}{cm.precision:.4f}{RESET} (TP / (TP + FP))")
    print(f"  Recall:    {GREEN}{cm.recall:.4f}{RESET} (TP / (TP + FN))")
    print(f"  F1-Score:  {GREEN}{cm.f1_score:.4f}{RESET}")
    print(f"  Accuracy:  {cm.accuracy:.4f}")
    
    # From scores
    print_section("Classification from Scores")
    
    predicted = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    actual = [0.85, 0.75, 0.65, 0.25, 0.15, 0.05]
    
    result = calculate_classification(predicted, actual)
    
    print(f"  Threshold: {result.threshold:.4f} (75th percentile of actual)")
    print(f"  Precision: {result.precision:.4f}")
    print(f"  Recall:    {result.recall:.4f}")
    print(f"  F1-Score:  {result.f1_score:.4f}")
    
    print_success("Classification metrics demonstrated")


def demo_ranking_metrics():
    """Demo 3: Ranking metrics."""
    print_header("DEMO 3: RANKING METRICS")
    
    from src.validation import calculate_ranking
    
    print("Ranking metrics compare which components are identified as most critical.")
    print()
    
    print_section("Top-K Overlap")
    
    predicted = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5, "f": 0.4}
    actual = {"a": 0.85, "c": 0.80, "b": 0.75, "d": 0.55, "f": 0.50, "e": 0.45}
    
    result = calculate_ranking(predicted, actual)
    
    pred_order = sorted(predicted.keys(), key=lambda x: -predicted[x])
    actual_order = sorted(actual.keys(), key=lambda x: -actual[x])
    
    print(f"  Predicted ranking: {pred_order}")
    print(f"  Actual ranking:    {actual_order}")
    print()
    print(f"  Top-3 Overlap: {GREEN}{result.top_3_overlap:.2%}{RESET}")
    print(f"  Top-5 Overlap: {result.top_5_overlap:.2%}")
    print(f"  NDCG:          {result.ndcg:.4f}")
    print(f"  MRR:           {result.mrr:.4f}")
    print(f"  Avg Rank Diff: {result.rank_difference_mean:.2f}")
    
    print_success("Ranking metrics demonstrated")


def demo_validator():
    """Demo 4: Validator class."""
    print_header("DEMO 4: VALIDATOR")
    
    from src.validation import Validator, ValidationTargets
    
    print("The Validator compares predicted scores against actual scores.")
    print()
    
    # Create test data
    predicted = {
        "broker_1": 0.90,
        "broker_2": 0.85,
        "app_1": 0.70,
        "app_2": 0.65,
        "app_3": 0.60,
        "node_1": 0.55,
        "node_2": 0.50,
        "topic_1": 0.40,
        "topic_2": 0.35,
        "topic_3": 0.30,
    }
    
    actual = {
        "broker_1": 0.88,
        "broker_2": 0.82,
        "app_1": 0.68,
        "app_2": 0.62,
        "app_3": 0.58,
        "node_1": 0.52,
        "node_2": 0.48,
        "topic_1": 0.38,
        "topic_2": 0.32,
        "topic_3": 0.28,
    }
    
    component_info = {
        "broker_1": {"type": "Broker", "layer": "app_broker"},
        "broker_2": {"type": "Broker", "layer": "app_broker"},
        "app_1": {"type": "Application", "layer": "application"},
        "app_2": {"type": "Application", "layer": "application"},
        "app_3": {"type": "Application", "layer": "application"},
        "node_1": {"type": "Node", "layer": "infrastructure"},
        "node_2": {"type": "Node", "layer": "infrastructure"},
        "topic_1": {"type": "Topic", "layer": "application"},
        "topic_2": {"type": "Topic", "layer": "application"},
        "topic_3": {"type": "Topic", "layer": "application"},
    }
    
    targets = ValidationTargets(spearman=0.70, f1_score=0.85)
    validator = Validator(targets=targets, seed=42)
    
    result = validator.validate(predicted, actual, component_info)
    
    print_section("Overall Results")
    
    color = status_color(result.status)
    print(f"  Status: {color}{BOLD}{result.status.value.upper()}{RESET}")
    print(f"  Spearman ρ: {metric_color(result.spearman, 0.70)}{result.spearman:.4f}{RESET}")
    print(f"  F1-Score:   {metric_color(result.f1_score, 0.85)}{result.f1_score:.4f}{RESET}")
    
    # Layer results
    print_section("Results by Layer")
    
    for layer, layer_result in result.by_layer.items():
        sp_color = metric_color(layer_result.spearman, 0.70)
        f1_color = metric_color(layer_result.f1_score, 0.85)
        print(f"  {layer_result.layer_name}:")
        print(f"    Components: {layer_result.count}")
        print(f"    Spearman:   {sp_color}{layer_result.spearman:.4f}{RESET}")
        print(f"    F1-Score:   {f1_color}{layer_result.f1_score:.4f}{RESET}")
    
    # Type results
    print_section("Results by Component Type")
    
    for comp_type, type_result in result.by_type.items():
        print(f"  {comp_type}: ρ={type_result.spearman:.4f}, F1={type_result.f1_score:.4f}")
    
    print_success("Validator demonstrated")
    
    return result.to_dict()


def demo_pipeline():
    """Demo 5: Validation Pipeline."""
    print_header("DEMO 5: VALIDATION PIPELINE")
    
    from src.simulation import create_simulation_graph
    from src.validation import ValidationPipeline, AnalysisMethod, ValidationTargets
    
    print("The pipeline integrates analysis, simulation, and validation.")
    print()
    
    # Create graph
    graph = create_simulation_graph(
        applications=8,
        brokers=2,
        topics=10,
        nodes=3,
        seed=42,
    )
    
    summary = graph.summary()
    print(f"  Graph: {summary['total_components']} components, "
          f"{summary['total_edges']} edges")
    
    # Configure pipeline
    targets = ValidationTargets(spearman=0.60, f1_score=0.70)
    pipeline = ValidationPipeline(targets=targets, seed=42, cascade=True)
    
    # Run pipeline
    print_section("Running Pipeline")
    print(f"  Analysis Method: composite")
    print(f"  Compare Methods: True")
    
    result = pipeline.run(
        graph,
        analysis_method=AnalysisMethod.COMPOSITE,
        compare_methods=True,
    )
    
    # Results
    print_section("Pipeline Results")
    
    color = status_color(result.validation.status)
    print(f"  Status: {color}{BOLD}{result.validation.status.value.upper()}{RESET}")
    print(f"  Spearman ρ: {result.spearman:.4f}")
    print(f"  F1-Score:   {result.f1_score:.4f}")
    
    # Timing
    print(f"\n  Timing:")
    print(f"    Analysis:   {result.analysis_time_ms:.0f}ms")
    print(f"    Simulation: {result.simulation_time_ms:.0f}ms")
    print(f"    Validation: {result.validation_time_ms:.0f}ms")
    
    # Method comparison
    if result.method_comparison:
        print_section("Method Comparison")
        
        sorted_methods = sorted(
            result.method_comparison.items(),
            key=lambda x: -x[1].spearman
        )
        
        print(f"  {'Method':<12} {'Spearman':<12} {'F1-Score':<12} {'Status'}")
        print(f"  {'-' * 45}")
        
        for method, comp in sorted_methods:
            sp_color = metric_color(comp.spearman, 0.60)
            f1_color = metric_color(comp.f1_score, 0.70)
            print(f"  {method:<12} {sp_color}{comp.spearman:.4f}{RESET}       "
                  f"{f1_color}{comp.f1_score:.4f}{RESET}       {comp.status.value}")
        
        best = result.get_best_method()
        print(f"\n  {GREEN}Best Method: {best}{RESET}")
    
    # Layer results
    if result.by_layer:
        print_section("Results by Layer")
        
        for layer, layer_result in result.by_layer.items():
            st_color = status_color(layer_result.status)
            print(f"  {layer_result.layer_name}:")
            print(f"    ρ={layer_result.spearman:.4f}, "
                  f"F1={layer_result.f1_score:.4f}, "
                  f"Status={st_color}{layer_result.status.value}{RESET}")
    
    print_success("Pipeline demonstrated")
    
    return result.to_dict()


def demo_cli_usage():
    """Demo 6: CLI usage examples."""
    print_header("DEMO 6: CLI USAGE")
    
    print("The validate_graph.py CLI provides these commands:")
    print()
    
    examples = [
        ("Basic validation",
         "python validate_graph.py --input graph.json"),
        ("Compare all methods",
         "python validate_graph.py --input graph.json --compare-methods"),
        ("Specific method",
         "python validate_graph.py --input graph.json --method pagerank"),
        ("Custom targets",
         "python validate_graph.py --input graph.json --spearman-target 0.80"),
        ("Export results",
         "python validate_graph.py --input graph.json --output results.json"),
        ("No cascade",
         "python validate_graph.py --input graph.json --no-cascade"),
    ]
    
    for desc, cmd in examples:
        print(f"  {desc}:")
        print(f"    {GRAY}{cmd}{RESET}")
        print()
    
    print_success("See --help for all options")


def demo_neo4j_integration():
    """Demo 7: Neo4j Integration"""
    print_header("DEMO 7: NEO4J INTEGRATION")
    
    from src.validation import check_neo4j_available
    
    print("The validation module can load graphs directly from Neo4j.")
    print()
    
    # Check if Neo4j is available
    if not check_neo4j_available():
        print(f"  {YELLOW}Neo4j driver not installed{RESET}")
        print(f"  Install with: pip install neo4j")
        print()
        print_neo4j_cli_examples()
        print_success("Neo4j integration available when driver is installed")
        return
    
    print(f"  {GREEN}✓{RESET} Neo4j driver installed")
    
    # Try to connect
    try:
        from src.validation import Neo4jValidationClient
        
        print(f"\n  Testing connection to localhost:7687...")
        
        with Neo4jValidationClient() as client:
            if client.verify_connection():
                print(f"  {GREEN}✓{RESET} Connected to Neo4j")
                
                # Get statistics
                stats = client.get_statistics()
                print(f"\n  {BOLD}Database Statistics{RESET}")
                print(f"    Components: {stats['total_components']}")
                print(f"    Edges: {stats['total_edges']}")
                
                if stats['total_components'] > 5:
                    # Run validation
                    print(f"\n  {BOLD}Running Validation{RESET}")
                    result = client.run_validation()
                    
                    status_color = GREEN if result.validation.status.value == "passed" else YELLOW
                    print(f"    Status: {status_color}{result.validation.status.value.upper()}{RESET}")
                    print(f"    Spearman ρ: {result.spearman:.4f}")
                    print(f"    F1-Score: {result.f1_score:.4f}")
                    
                    # Layer results
                    if result.by_layer:
                        print(f"\n  {BOLD}Layer Results{RESET}")
                        for layer, layer_result in result.by_layer.items():
                            status = "✓" if layer_result.passed else "✗"
                            print(f"    {status} {layer}: ρ={layer_result.spearman:.4f}")
                else:
                    print(f"\n  {YELLOW}Insufficient data for validation{RESET}")
            else:
                print(f"  {RED}✗{RESET} Could not connect to Neo4j")
                print(f"    Make sure Neo4j is running on localhost:7687")
    
    except Exception as e:
        print(f"  {YELLOW}⚠{RESET} Connection failed: {e}")
        print(f"    Make sure Neo4j is running")
    
    print()
    print_neo4j_cli_examples()
    print_success("Neo4j integration demonstrated")


def print_neo4j_cli_examples():
    """Print Neo4j CLI examples."""
    print(f"\n  {BOLD}Neo4j CLI Examples{RESET}")
    print()
    
    neo4j_examples = [
        ("Load from Neo4j",
         "python validate_graph.py --neo4j"),
        ("Custom connection",
         "python validate_graph.py --neo4j --uri bolt://host:7687 --user neo4j --password pass"),
        ("Load specific layer",
         "python validate_graph.py --neo4j --layer application"),
        ("Compare methods from Neo4j",
         "python validate_graph.py --neo4j --compare-methods"),
        ("Export results",
         "python validate_graph.py --neo4j --output results.json"),
    ]
    
    for desc, cmd in neo4j_examples:
        print(f"  {desc}:")
        print(f"    {GRAY}{cmd}{RESET}")
        print()


def demo_neo4j_python_api():
    """Demo 8: Neo4j Python API"""
    print_header("DEMO 8: NEO4J PYTHON API")
    
    print("Using Neo4j validation in Python code:")
    print()
    
    code = '''
from src.validation import (
    Neo4jValidationClient,
    validate_from_neo4j,
    AnalysisMethod,
    ValidationTargets,
)

# Method 1: Using context manager
with Neo4jValidationClient(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
) as client:
    # Run full validation
    result = client.run_validation(compare_methods=True)
    print(f"Status: {result.validation.status.value}")
    print(f"Spearman: {result.spearman:.4f}")
    
    # Validate specific layer
    app_result = client.validate_layer("application")
    print(f"App Layer: {app_result.spearman:.4f}")
    
    # Validate all layers separately
    all_layers = client.validate_all_layers()
    for layer, result in all_layers.items():
        print(f"{layer}: {result.spearman:.4f}")

# Method 2: Using factory function
result = validate_from_neo4j(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    method=AnalysisMethod.COMPOSITE,
    compare_methods=True,
    layer="app_broker",  # Optional
)

# Check results
print(f"Status: {result.validation.status.value}")
for method, comp in result.method_comparison.items():
    print(f"  {method}: ρ={comp.spearman:.4f}")

# With custom targets
targets = ValidationTargets(spearman=0.80, f1_score=0.95)
result = validate_from_neo4j(targets=targets)
'''
    
    print(f"  {GRAY}{code}{RESET}")
    print_success("Neo4j Python API demonstrated")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validation Module Demo"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print_header("Validation Module Demo")
    
    # Run demos
    demo_correlation_metrics()
    demo_classification_metrics()
    demo_ranking_metrics()
    demo_validator()
    demo_pipeline()
    demo_cli_usage()
    demo_neo4j_integration()
    demo_neo4j_python_api()
    
    print_header("Demo Complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())