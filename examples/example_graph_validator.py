#!/usr/bin/env python3
"""
Graph Validator Examples
========================

Demonstrates how to use the GraphValidator to validate the graph-based
modeling and analysis approach by comparing predicted criticality with
actual failure impact.

Examples:
1. Basic validation workflow
2. Custom target thresholds
3. Analyzing validation results
4. Comparing different scoring weights
5. Generating validation reports

Usage:
    python example_graph_validator.py

Author: Software-as-a-Graph Research Project
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import GraphAnalyzer
from src.simulation import GraphSimulator
from src.validation import (
    GraphValidator,
    ValidationResult,
    ValidationStatus,
    validate_analysis,
    quick_validate,
)


# ============================================================================
# Sample Data
# ============================================================================

SAMPLE_PUBSUB_DATA = {
    "nodes": [
        {"id": "N1", "name": "ComputeNode1", "type": "compute"},
        {"id": "N2", "name": "ComputeNode2", "type": "compute"},
        {"id": "N3", "name": "EdgeNode", "type": "edge"}
    ],
    "brokers": [
        {"id": "B1", "name": "MainBroker", "node": "N1"},
        {"id": "B2", "name": "BackupBroker", "node": "N2"}
    ],
    "applications": [
        {"id": "A1", "name": "SensorReader", "role": "pub", "node": "N3"},
        {"id": "A2", "name": "DataProcessor", "role": "both", "node": "N1"},
        {"id": "A3", "name": "Analytics", "role": "sub", "node": "N1"},
        {"id": "A4", "name": "Dashboard", "role": "sub", "node": "N2"},
        {"id": "A5", "name": "Alerting", "role": "sub", "node": "N2"},
        {"id": "A6", "name": "Logger", "role": "sub", "node": "N1"}
    ],
    "topics": [
        {"id": "T1", "name": "sensor/data", "broker": "B1"},
        {"id": "T2", "name": "processed/data", "broker": "B1"},
        {"id": "T3", "name": "alerts", "broker": "B2"}
    ],
    "relationships": {
        "publishes_to": [
            {"from": "A1", "to": "T1"},
            {"from": "A2", "to": "T2"},
            {"from": "A2", "to": "T3"}
        ],
        "subscribes_to": [
            {"from": "A2", "to": "T1"},
            {"from": "A3", "to": "T2"},
            {"from": "A4", "to": "T2"},
            {"from": "A5", "to": "T3"},
            {"from": "A6", "to": "T1"},
            {"from": "A6", "to": "T2"}
        ],
        "runs_on": [
            {"from": "A1", "to": "N3"},
            {"from": "A2", "to": "N1"},
            {"from": "A3", "to": "N1"},
            {"from": "A4", "to": "N2"},
            {"from": "A5", "to": "N2"},
            {"from": "A6", "to": "N1"},
            {"from": "B1", "to": "N1"},
            {"from": "B2", "to": "N2"}
        ],
        "routes": [
            {"from": "B1", "to": "T1"},
            {"from": "B1", "to": "T2"},
            {"from": "B2", "to": "T3"}
        ]
    }
}


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def print_subheader(title: str):
    """Print a formatted subheader"""
    print(f"\n── {title} ──")


# ============================================================================
# Example Functions
# ============================================================================

def example_1_basic_validation():
    """Example 1: Basic validation workflow"""
    print_header("Example 1: Basic Validation Workflow")
    
    # Step 1: Create analyzer with data
    print_subheader("Step 1: Setup Analyzer")
    analyzer = GraphAnalyzer(alpha=0.4, beta=0.3, gamma=0.3)
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    print(f"Loaded system with {len(SAMPLE_PUBSUB_DATA['applications'])} applications")
    
    # Step 2: Create validator
    print_subheader("Step 2: Create Validator")
    validator = GraphValidator(analyzer, seed=42)
    print(f"Target thresholds: {validator.targets}")
    
    # Step 3: Run validation
    print_subheader("Step 3: Run Validation")
    result = validator.validate()
    
    # Step 4: Review results
    print_subheader("Step 4: Results")
    print(f"\nValidation Status: {result.status.value.upper()}")
    print(f"Components Validated: {result.total_components}")
    print(f"\nCorrelation Metrics:")
    print(f"  Spearman: {result.spearman_correlation:.4f} (target: 0.70)")
    print(f"  Pearson:  {result.pearson_correlation:.4f}")
    print(f"\nClassification Metrics:")
    print(f"  Precision: {result.confusion_matrix.precision:.4f}")
    print(f"  Recall:    {result.confusion_matrix.recall:.4f}")
    print(f"  F1-Score:  {result.confusion_matrix.f1_score:.4f}")
    
    return result


def example_2_custom_targets():
    """Example 2: Custom target thresholds"""
    print_header("Example 2: Custom Target Thresholds")
    
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    
    # Define stricter targets
    strict_targets = {
        'spearman_correlation': 0.8,
        'f1_score': 0.95,
        'precision': 0.9,
        'recall': 0.9,
        'top_5_overlap': 0.8,
    }
    
    # Define relaxed targets
    relaxed_targets = {
        'spearman_correlation': 0.5,
        'f1_score': 0.7,
        'precision': 0.6,
        'recall': 0.6,
        'top_5_overlap': 0.4,
    }
    
    print_subheader("Strict Targets")
    strict_validator = GraphValidator(analyzer, targets=strict_targets, seed=42)
    strict_result = strict_validator.validate()
    print(f"Status: {strict_result.status.value}")
    print(f"Spearman: {strict_result.spearman_correlation:.4f} (target: 0.8)")
    
    print_subheader("Relaxed Targets")
    relaxed_validator = GraphValidator(analyzer, targets=relaxed_targets, seed=42)
    relaxed_result = relaxed_validator.validate()
    print(f"Status: {relaxed_result.status.value}")
    print(f"Spearman: {relaxed_result.spearman_correlation:.4f} (target: 0.5)")
    
    return strict_result, relaxed_result


def example_3_analyzing_results():
    """Example 3: Analyzing validation results in detail"""
    print_header("Example 3: Analyzing Validation Results")
    
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    
    validator = GraphValidator(analyzer, seed=42)
    result = validator.validate()
    
    print_subheader("Component Comparison")
    print(f"\n{'Component':<12} {'Predicted':>10} {'Actual':>10} {'Rank Δ':>8} {'Match':>6}")
    print("-" * 50)
    
    for cv in result.component_validations:
        match = "✓" if cv.correctly_classified else "✗"
        print(f"{cv.component_id:<12} {cv.predicted_score:>10.4f} {cv.actual_impact:>10.2%} "
              f"{cv.rank_difference:>8} {match:>6}")
    
    print_subheader("Misclassified Components")
    misclassified = validator.get_misclassified()
    if misclassified:
        for cv in misclassified:
            print(f"  {cv.component_id}: predicted {cv.predicted_level}, "
                  f"actual {cv.actual_level}")
    else:
        print("  No misclassified components!")
    
    print_subheader("High Rank Differences (>2)")
    high_diff = validator.get_high_rank_difference(threshold=2)
    if high_diff:
        for cv in high_diff:
            print(f"  {cv.component_id}: rank diff = {cv.rank_difference}")
    else:
        print("  No components with high rank difference!")
    
    return result


def example_4_comparing_weights():
    """Example 4: Comparing different scoring weights"""
    print_header("Example 4: Comparing Scoring Weight Configurations")
    
    # Different weight configurations to test
    configs = [
        ("Balanced (0.33, 0.33, 0.33)", 0.33, 0.33, 0.33),
        ("Betweenness-heavy (0.6, 0.2, 0.2)", 0.6, 0.2, 0.2),
        ("AP-heavy (0.2, 0.6, 0.2)", 0.2, 0.6, 0.2),
        ("Impact-heavy (0.2, 0.2, 0.6)", 0.2, 0.2, 0.6),
        ("Default (0.4, 0.3, 0.3)", 0.4, 0.3, 0.3),
    ]
    
    print(f"\n{'Configuration':<35} {'Spearman':>10} {'F1':>10} {'Status':>10}")
    print("-" * 70)
    
    results = []
    for name, alpha, beta, gamma in configs:
        analyzer = GraphAnalyzer(alpha=alpha, beta=beta, gamma=gamma)
        analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
        
        validator = GraphValidator(analyzer, seed=42)
        result = validator.validate()
        
        print(f"{name:<35} {result.spearman_correlation:>10.4f} "
              f"{result.confusion_matrix.f1_score:>10.4f} {result.status.value:>10}")
        
        results.append((name, result))
    
    # Find best configuration
    best = max(results, key=lambda x: x[1].spearman_correlation)
    print(f"\nBest configuration: {best[0]} (Spearman = {best[1].spearman_correlation:.4f})")
    
    return results


def example_5_generating_reports():
    """Example 5: Generating validation reports"""
    print_header("Example 5: Generating Validation Reports")
    
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    
    validator = GraphValidator(analyzer, seed=42)
    result = validator.validate()
    
    # Generate report
    report = validator.generate_report()
    
    print_subheader("Report Summary")
    print(f"  Status: {report['summary']['status']}")
    print(f"  Components: {report['summary']['total_components']}")
    
    print_subheader("Correlation Analysis")
    corr = report['correlation']
    spearman_passed = "✓ PASS" if corr['spearman']['passed'] else "✗ FAIL"
    print(f"  Spearman: {corr['spearman']['value']:.4f} (target: {corr['spearman']['target']}) {spearman_passed}")
    print(f"  Pearson:  {corr['pearson']:.4f}")
    
    print_subheader("Classification Analysis")
    for metric in ['precision', 'recall', 'f1_score']:
        data = report['classification'][metric]
        passed = "✓ PASS" if data['passed'] else "✗ FAIL"
        print(f"  {metric.capitalize():12s}: {data['value']:.4f} (target: {data['target']}) {passed}")
    
    print_subheader("Notable Components")
    notable = report['notable_components']
    print(f"  Most Underestimated: {notable['most_underestimated']['id']} "
          f"(diff: {notable['most_underestimated']['difference']:.4f})")
    print(f"  Most Overestimated:  {notable['most_overestimated']['id']} "
          f"(diff: {notable['most_overestimated']['difference']:.4f})")
    
    print_subheader("Recommendations")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    return report


def example_6_convenience_functions():
    """Example 6: Using convenience functions"""
    print_header("Example 6: Convenience Functions")
    
    # Create analyzer
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    
    print_subheader("validate_analysis()")
    result = validate_analysis(analyzer, seed=42)
    print(f"Status: {result.status.value}")
    print(f"Spearman: {result.spearman_correlation:.4f}")
    
    print_subheader("Result Summary")
    print(result.summary())
    
    return result


def example_7_filtering_by_type():
    """Example 7: Filtering validation by component type"""
    print_header("Example 7: Filtering by Component Type")
    
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    
    validator = GraphValidator(analyzer, seed=42)
    
    print_subheader("All Components")
    all_result = validator.validate()
    print(f"Components: {all_result.total_components}")
    print(f"Spearman: {all_result.spearman_correlation:.4f}")
    
    print_subheader("Applications Only")
    app_result = validator.validate(component_types=['Application'])
    print(f"Components: {app_result.total_components}")
    print(f"Spearman: {app_result.spearman_correlation:.4f}")
    
    print_subheader("Infrastructure Only (Nodes + Brokers)")
    infra_result = validator.validate(component_types=['Node', 'Broker'])
    print(f"Components: {infra_result.total_components}")
    print(f"Spearman: {infra_result.spearman_correlation:.4f}")
    
    return all_result, app_result, infra_result


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print(" GRAPH VALIDATOR EXAMPLES")
    print("=" * 60)
    
    # Run examples
    example_1_basic_validation()
    example_2_custom_targets()
    example_3_analyzing_results()
    example_4_comparing_weights()
    example_5_generating_reports()
    example_6_convenience_functions()
    example_7_filtering_by_type()
    
    print("\n" + "=" * 60)
    print(" ALL EXAMPLES COMPLETED")
    print("=" * 60)
    print()


if __name__ == '__main__':
    main()