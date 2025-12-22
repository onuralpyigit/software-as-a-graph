#!/usr/bin/env python3
"""
Validation Examples for Graph-Based Analysis
==============================================

Demonstrates comprehensive validation of graph-based criticality analysis:
1. Basic validation workflow
2. Correlation analysis
3. Classification metrics
4. Sensitivity analysis
5. Bootstrap confidence intervals
6. Cross-validation

Usage:
    python examples/validation_examples.py
    python examples/validation_examples.py --example basic
    python examples/validation_examples.py --example advanced

Author: Software-as-a-Graph Research Project
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

from src.validation import (
    GraphValidator,
    ValidationStatus,
    validate_analysis,
    spearman_correlation,
    pearson_correlation
)

from src.simulation import FailureSimulator


# ============================================================================
# Terminal Formatting
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(title: str):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{title.center(70)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.ENDC}\n")


def print_section(title: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{title}{Colors.ENDC}")
    print("-" * 50)


def status_color(status: ValidationStatus) -> str:
    if status == ValidationStatus.PASSED:
        return Colors.GREEN
    elif status == ValidationStatus.MARGINAL:
        return Colors.WARNING
    return Colors.RED


# ============================================================================
# Test System Generation
# ============================================================================

def create_iot_system():
    """Create an IoT pub-sub system for testing"""
    G = nx.DiGraph()
    
    # Nodes
    nodes = [
        # Infrastructure
        ('gateway_north', 'Node'),
        ('gateway_south', 'Node'),
        # Brokers
        ('mqtt_broker', 'Broker'),
        ('kafka_broker', 'Broker'),
        # Topics
        ('temperature', 'Topic'),
        ('humidity', 'Topic'),
        ('pressure', 'Topic'),
        ('alerts', 'Topic'),
        # Applications
        ('sensor_temp_1', 'Application'),
        ('sensor_temp_2', 'Application'),
        ('sensor_humid', 'Application'),
        ('sensor_pressure', 'Application'),
        ('aggregator', 'Application'),
        ('ml_predictor', 'Application'),
        ('dashboard', 'Application'),
        ('alerter', 'Application'),
    ]
    
    for node_id, node_type in nodes:
        G.add_node(node_id, type=node_type)
    
    # Edges
    edges = [
        # Infrastructure
        ('gateway_north', 'mqtt_broker'),
        ('gateway_south', 'kafka_broker'),
        ('mqtt_broker', 'kafka_broker'),
        # Publisher to topic
        ('sensor_temp_1', 'temperature'),
        ('sensor_temp_2', 'temperature'),
        ('sensor_humid', 'humidity'),
        ('sensor_pressure', 'pressure'),
        ('ml_predictor', 'alerts'),
        # Topic to subscriber
        ('temperature', 'aggregator'),
        ('temperature', 'ml_predictor'),
        ('humidity', 'aggregator'),
        ('pressure', 'aggregator'),
        ('aggregator', 'dashboard'),
        ('alerts', 'alerter'),
        ('alerts', 'dashboard'),
        # Topic routing
        ('mqtt_broker', 'temperature'),
        ('mqtt_broker', 'humidity'),
        ('kafka_broker', 'pressure'),
        ('kafka_broker', 'alerts'),
    ]
    
    G.add_edges_from(edges)
    
    return G


def calculate_criticality(graph: nx.DiGraph) -> dict:
    """Calculate criticality scores using composite scoring"""
    scores = {}
    
    # Betweenness centrality
    bc = nx.betweenness_centrality(graph)
    max_bc = max(bc.values()) if bc.values() else 1
    bc_norm = {k: v/max_bc if max_bc > 0 else 0 for k, v in bc.items()}
    
    # Articulation points
    try:
        aps = set(nx.articulation_points(graph.to_undirected()))
    except:
        aps = set()
    
    # Degree
    degrees = dict(graph.degree())
    max_deg = max(degrees.values()) if degrees.values() else 1
    
    # PageRank
    try:
        pr = nx.pagerank(graph)
    except:
        pr = {n: 1/graph.number_of_nodes() for n in graph.nodes()}
    max_pr = max(pr.values()) if pr.values() else 1
    
    # Composite score: C = 0.25*BC + 0.30*AP + 0.25*Impact + 0.10*Degree + 0.10*PR
    for node in graph.nodes():
        bc_score = bc_norm.get(node, 0)
        ap_score = 1.0 if node in aps else 0.0
        
        try:
            descendants = len(nx.descendants(graph, node))
            ancestors = len(nx.ancestors(graph, node))
            impact = (descendants + ancestors) / (2 * graph.number_of_nodes())
        except:
            impact = 0
        
        degree_score = degrees.get(node, 0) / max_deg
        pr_score = pr.get(node, 0) / max_pr
        
        scores[node] = (0.25 * bc_score + 
                       0.30 * ap_score + 
                       0.25 * impact + 
                       0.10 * degree_score + 
                       0.10 * pr_score)
    
    return scores


# ============================================================================
# Example Functions
# ============================================================================

def example_basic_validation():
    """Basic validation workflow"""
    print_header("EXAMPLE: Basic Validation Workflow")
    
    # Create system
    graph = create_iot_system()
    print(f"System: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Calculate predicted scores
    print_section("Step 1: Calculate Criticality Scores")
    predicted = calculate_criticality(graph)
    
    print("Top 5 predicted critical components:")
    sorted_pred = sorted(predicted.items(), key=lambda x: -x[1])
    for comp, score in sorted_pred[:5]:
        print(f"  {comp:25s} {score:.4f}")
    
    # Run simulation
    print_section("Step 2: Run Failure Simulation")
    simulator = FailureSimulator(seed=42)
    batch_result = simulator.simulate_exhaustive(graph, enable_cascade=False)
    
    actual = {}
    for result in batch_result.results:
        if result.primary_failures:
            actual[result.primary_failures[0]] = result.impact_score
    
    print(f"Simulated {len(actual)} component failures")
    
    # Validate
    print_section("Step 3: Validate Predictions")
    validator = GraphValidator(seed=42)
    result = validator.validate(graph, predicted, actual)
    
    # Print results
    status_col = status_color(result.status)
    print(f"\nValidation Status: {status_col}{Colors.BOLD}{result.status.value.upper()}{Colors.ENDC}")
    print(f"Components: {result.total_components}")
    
    print(f"\nCorrelation:")
    print(f"  Spearman ρ = {result.correlation.spearman_coefficient:.4f}")
    print(f"  Pearson r  = {result.correlation.pearson_coefficient:.4f}")
    
    print(f"\nClassification:")
    cm = result.confusion_matrix
    print(f"  Precision = {cm.precision:.4f}")
    print(f"  Recall    = {cm.recall:.4f}")
    print(f"  F1-Score  = {cm.f1_score:.4f}")
    
    print(f"\nRanking:")
    for k, overlap in sorted(result.ranking.top_k_overlap.items()):
        print(f"  Top-{k} Overlap = {overlap:.1%}")
    
    return result


def example_correlation_analysis():
    """Detailed correlation analysis"""
    print_header("EXAMPLE: Correlation Analysis")
    
    graph = create_iot_system()
    predicted = calculate_criticality(graph)
    
    # Simulate
    simulator = FailureSimulator(seed=42)
    batch_result = simulator.simulate_exhaustive(graph)
    actual = {r.primary_failures[0]: r.impact_score 
             for r in batch_result.results if r.primary_failures}
    
    # Validate
    validator = GraphValidator(seed=42)
    result = validator.validate(graph, predicted, actual)
    
    print_section("Correlation Metrics")
    corr = result.correlation
    
    print(f"\nSpearman Rank Correlation:")
    print(f"  Coefficient: {corr.spearman_coefficient:.4f}")
    print(f"  P-value: {corr.spearman_p_value:.6f}")
    print(f"  Significant (p<0.05): {'Yes' if corr.spearman_p_value < 0.05 else 'No'}")
    print(f"  Target (≥0.70): {'✓ PASS' if corr.spearman_coefficient >= 0.7 else '✗ FAIL'}")
    
    print(f"\nPearson Correlation:")
    print(f"  Coefficient: {corr.pearson_coefficient:.4f}")
    print(f"  P-value: {corr.pearson_p_value:.6f}")
    
    print(f"\nKendall's Tau:")
    print(f"  Coefficient: {result.ranking.kendall_tau:.4f}")
    
    # Show ranking comparison
    print_section("Ranking Comparison")
    print("\nComponent rankings (top 10):")
    print(f"{'Component':25s} {'Pred Rank':>10s} {'Actual Rank':>12s} {'Diff':>6s}")
    print("-" * 55)
    
    sorted_by_actual = sorted(result.component_validations, 
                             key=lambda x: x.actual_rank)
    for cv in sorted_by_actual[:10]:
        diff_str = f"{cv.rank_difference:+d}" if cv.rank_difference > 0 else "0"
        print(f"{cv.component_id:25s} {cv.predicted_rank:>10d} {cv.actual_rank:>12d} {diff_str:>6s}")
    
    return result


def example_sensitivity_analysis():
    """Sensitivity analysis example"""
    print_header("EXAMPLE: Sensitivity Analysis")
    
    graph = create_iot_system()
    predicted = calculate_criticality(graph)
    
    simulator = FailureSimulator(seed=42)
    batch_result = simulator.simulate_exhaustive(graph)
    actual = {r.primary_failures[0]: r.impact_score 
             for r in batch_result.results if r.primary_failures}
    
    validator = GraphValidator(seed=42)
    validator.validate(graph, predicted, actual)
    
    print_section("Critical Threshold Sensitivity")
    print("\nTesting how F1-score varies with critical threshold...")
    
    results = validator.run_sensitivity_analysis(graph, predicted, actual)
    
    for sr in results:
        print(f"\nParameter: {sr.parameter_name}")
        print(f"Original value: {sr.original_value:.2f}")
        print(f"Stability score: {sr.stability_score:.2%}")
        
        print(f"\nThreshold vs Metrics:")
        print(f"{'Threshold':>10s} {'F1-Score':>10s} {'Precision':>10s} {'Recall':>10s}")
        print("-" * 45)
        
        f1_values = sr.metric_values.get('f1_score', [])
        precision_values = sr.metric_values.get('precision', [])
        recall_values = sr.metric_values.get('recall', [])
        
        for i, threshold in enumerate(sr.tested_values):
            f1 = f1_values[i] if i < len(f1_values) else 0
            prec = precision_values[i] if i < len(precision_values) else 0
            rec = recall_values[i] if i < len(recall_values) else 0
            print(f"{threshold:>10.2f} {f1:>10.4f} {prec:>10.4f} {rec:>10.4f}")
    
    return results


def example_bootstrap_analysis():
    """Bootstrap confidence intervals"""
    print_header("EXAMPLE: Bootstrap Confidence Intervals")
    
    graph = create_iot_system()
    predicted = calculate_criticality(graph)
    
    simulator = FailureSimulator(seed=42)
    batch_result = simulator.simulate_exhaustive(graph)
    actual = {r.primary_failures[0]: r.impact_score 
             for r in batch_result.results if r.primary_failures}
    
    validator = GraphValidator(seed=42)
    validator.validate(graph, predicted, actual)
    
    print_section("Bootstrap Analysis")
    print("Running 500 bootstrap iterations...")
    
    results = validator.run_bootstrap_analysis(
        graph, predicted, actual,
        n_iterations=500,
        confidence_level=0.95
    )
    
    print(f"\n95% Confidence Intervals:")
    print(f"{'Metric':25s} {'Point Est':>12s} {'95% CI':>25s} {'Std Error':>12s}")
    print("-" * 75)
    
    for br in results:
        ci_str = f"[{br.ci_lower:.4f}, {br.ci_upper:.4f}]"
        print(f"{br.metric_name:25s} {br.point_estimate:>12.4f} {ci_str:>25s} {br.std_error:>12.4f}")
    
    return results


def example_cross_validation():
    """K-fold cross-validation"""
    print_header("EXAMPLE: K-Fold Cross-Validation")
    
    graph = create_iot_system()
    predicted = calculate_criticality(graph)
    
    simulator = FailureSimulator(seed=42)
    batch_result = simulator.simulate_exhaustive(graph)
    actual = {r.primary_failures[0]: r.impact_score 
             for r in batch_result.results if r.primary_failures}
    
    validator = GraphValidator(seed=42)
    validator.validate(graph, predicted, actual)
    
    print_section("5-Fold Cross-Validation")
    
    result = validator.run_cross_validation(
        graph, predicted, actual,
        n_folds=5
    )
    
    print(f"\nFold Results:")
    print(f"{'Fold':>6s} {'Size':>6s} {'Spearman':>10s} {'F1-Score':>10s}")
    print("-" * 35)
    
    for fr in result.fold_results:
        print(f"{fr['fold']:>6d} {fr['size']:>6d} {fr['spearman']:>10.4f} {fr['f1_score']:>10.4f}")
    
    print(f"\nAggregated Results:")
    print(f"{'Metric':15s} {'Mean':>10s} {'Std':>10s}")
    print("-" * 35)
    
    for metric in ['spearman', 'f1_score', 'precision', 'recall']:
        mean = result.mean_metrics.get(metric, 0)
        std = result.std_metrics.get(metric, 0)
        print(f"{metric:15s} {mean:>10.4f} {std:>10.4f}")
    
    return result


def example_full_validation():
    """Complete validation workflow"""
    print_header("EXAMPLE: Complete Validation Report")
    
    graph = create_iot_system()
    print(f"System: IoT Monitoring ({graph.number_of_nodes()} nodes)")
    
    # Calculate scores
    predicted = calculate_criticality(graph)
    
    # Simulate
    simulator = FailureSimulator(seed=42)
    batch_result = simulator.simulate_exhaustive(graph)
    actual = {r.primary_failures[0]: r.impact_score 
             for r in batch_result.results if r.primary_failures}
    
    # Full validation
    validator = GraphValidator(seed=42)
    result = validator.validate(graph, predicted, actual)
    
    # Advanced analysis
    print_section("Running Advanced Analysis")
    print("  - Sensitivity analysis...")
    validator.run_sensitivity_analysis(graph, predicted, actual)
    print("  - Bootstrap analysis (200 iterations)...")
    validator.run_bootstrap_analysis(graph, predicted, actual, n_iterations=200)
    print("  - Cross-validation (5 folds)...")
    validator.run_cross_validation(graph, predicted, actual, n_folds=5)
    
    # Summary
    print_section("Validation Summary")
    
    status_col = status_color(result.status)
    print(f"\n{Colors.BOLD}Overall Status: {status_col}{result.status.value.upper()}{Colors.ENDC}")
    
    print(f"\nTarget Achievement:")
    targets = validator.targets
    achieved = result.achieved_metrics
    
    for metric, target in targets.items():
        value = achieved.get(metric, 0)
        passed = value >= target
        symbol = f"{Colors.GREEN}✓{Colors.ENDC}" if passed else f"{Colors.RED}✗{Colors.ENDC}"
        print(f"  {symbol} {metric}: {value:.4f} (target: ≥{target})")
    
    # Recommendations
    recommendations = validator.generate_recommendations()
    if recommendations:
        print_section("Recommendations")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")
    
    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Validation examples')
    parser.add_argument('--example', 
                       choices=['basic', 'correlation', 'sensitivity', 
                               'bootstrap', 'crossval', 'full', 'all'],
                       default='all',
                       help='Which example to run')
    parser.add_argument('--export', help='Export results to JSON')
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}Graph Analysis Validation Examples{Colors.ENDC}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {}
    
    if args.example in ['basic', 'all']:
        results['basic'] = example_basic_validation()
    
    if args.example in ['correlation', 'all']:
        results['correlation'] = example_correlation_analysis()
    
    if args.example in ['sensitivity', 'all']:
        results['sensitivity'] = example_sensitivity_analysis()
    
    if args.example in ['bootstrap', 'all']:
        results['bootstrap'] = example_bootstrap_analysis()
    
    if args.example in ['crossval', 'all']:
        results['crossval'] = example_cross_validation()
    
    if args.example == 'full':
        results['full'] = example_full_validation()
    
    # Export
    if args.export:
        export_data = {}
        for name, result in results.items():
            if hasattr(result, 'to_dict'):
                export_data[name] = result.to_dict()
        
        with open(args.export, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"\n{Colors.GREEN}✓{Colors.ENDC} Results exported to {args.export}")
    
    print(f"\n{Colors.GREEN}All examples completed successfully!{Colors.ENDC}\n")


if __name__ == '__main__':
    main()