#!/usr/bin/env python3
"""
Pub-Sub System Validation Tool
================================

Validates graph-based criticality analysis by comparing predicted scores
from topological analysis against actual impact scores from failure simulation.

Validation Approach:
1. Run structural analysis to get predicted criticality scores
2. Run exhaustive failure simulation to get actual impact scores
3. Compare using correlation, classification, and ranking metrics
4. Optionally run sensitivity, bootstrap, and cross-validation analysis

Target Metrics:
- Spearman Correlation: ≥ 0.70
- F1-Score: ≥ 0.90
- Precision/Recall: ≥ 0.80
- Top-5 Overlap: ≥ 60%
- Top-10 Overlap: ≥ 70%

Usage:
    # Basic validation
    python validate_graph.py --input system.json
    
    # With custom weights
    python validate_graph.py --input system.json --alpha 0.4 --beta 0.3 --gamma 0.3
    
    # Full analysis with sensitivity and bootstrap
    python validate_graph.py --input system.json --full-analysis
    
    # Export results
    python validate_graph.py --input system.json --output results/ --format json html csv

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import sys
import logging
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

import networkx as nx

from src.validation import (
    GraphValidator,
    ValidationResult,
    ValidationStatus,
    spearman_correlation,
    pearson_correlation
)

from src.simulation import (
    FailureSimulator,
    BatchSimulationResult
)


# ============================================================================
# Terminal Colors
# ============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    _enabled = True
    
    @classmethod
    def disable(cls):
        cls._enabled = False
        for attr in ['HEADER', 'BLUE', 'CYAN', 'GREEN', 'WARNING', 'RED',
                     'ENDC', 'BOLD', 'DIM']:
            setattr(cls, attr, '')


def print_header(text: str):
    """Print header"""
    width = 70
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * width}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(width)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * width}{Colors.ENDC}")


def print_section(title: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{title}{Colors.ENDC}")
    print("-" * 50)


def print_success(msg: str):
    print(f"{Colors.GREEN}✓{Colors.ENDC} {msg}")


def print_error(msg: str):
    print(f"{Colors.RED}✗{Colors.ENDC} {msg}")


def print_warning(msg: str):
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {msg}")


def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ{Colors.ENDC} {msg}")


def metric_color(achieved: float, target: float) -> str:
    """Get color based on whether metric meets target"""
    if achieved >= target:
        return Colors.GREEN
    elif achieved >= target * 0.8:
        return Colors.WARNING
    else:
        return Colors.RED


def status_color(status: ValidationStatus) -> str:
    """Get color for validation status"""
    if status == ValidationStatus.PASSED:
        return Colors.GREEN
    elif status == ValidationStatus.MARGINAL:
        return Colors.WARNING
    else:
        return Colors.RED


# ============================================================================
# Graph Loading
# ============================================================================

def load_graph_from_json(filepath: str) -> nx.DiGraph:
    """Load graph from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    # Add nodes (infrastructure)
    for node in data.get('nodes', []):
        node_id = node.get('id', node.get('name'))
        G.add_node(node_id, type='Node', **node)
        
    # Add brokers 
    for broker in data.get('brokers', []):
        broker_id = broker.get('id', broker.get('name'))
        G.add_node(broker_id, type='Broker', **broker)
        
    # Add topics
    for topic in data.get('topics', []):
        topic_id = topic.get('id', topic.get('name'))
        G.add_node(topic_id, type='Topic', **topic)
        
    # Add applications
    for app in data.get('applications', []):
        app_id = app.get('id', app.get('name'))
        G.add_node(app_id, type='Application', **app)
        
    # Process relationships
    relationships = data.get('relationships', {})

    # PUBLISHES_TO relationships
    for rel in relationships.get('publishes_to', data.get('publishes', [])):
        app_id = rel.get('from', rel.get('source', rel.get('app')))
        topic_id = rel.get('to', rel.get('target', rel.get('topic')))
        G.add_edge(app_id, topic_id, type='PUBLISHES_TO', **rel)

    # SUBSCRIBES_TO relationships
    for rel in relationships.get('subscribes_to', data.get('subscribes', [])):
        app_id = rel.get('from', rel.get('source', rel.get('app')))
        topic_id = rel.get('to', rel.get('target', rel.get('topic')))
        G.add_edge(topic_id, app_id, type='SUBSCRIBES_TO', **rel)

    # ROUTES relationships
    for rel in relationships.get('routes', data.get('routes', [])):
        topic_id = rel.get('to', rel.get('target', rel.get('topic')))
        broker_id = rel.get('from', rel.get('source', rel.get('broker')))
        G.add_edge(broker_id, topic_id, type='ROUTES', **rel)

    # RUNS_ON relationships
    for rel in relationships.get('runs_on', data.get('runs', [])):
        comp_id = rel.get('from', rel.get('source', rel.get('component')))
        node_id = rel.get('to', rel.get('target', rel.get('node')))
        G.add_edge(comp_id, node_id, type='RUNS_ON', **rel)

    # CONNECTS_TO relationships
    for rel in relationships.get('connects_to', data.get('connects', [])):
        src_id = rel.get('from', rel.get('source', rel.get('source_component')))
        dst_id = rel.get('to', rel.get('target', rel.get('target_component')))
        G.add_edge(src_id, dst_id, type='CONNECTS_TO', **rel)
    
    return G


# ============================================================================
# Analysis Functions
# ============================================================================

def calculate_criticality_scores(graph: nx.DiGraph,
                                alpha: float = 0.25,
                                beta: float = 0.30,
                                gamma: float = 0.25,
                                delta: float = 0.10,
                                epsilon: float = 0.10) -> Dict[str, float]:
    """
    Calculate composite criticality scores for all components.
    
    C_score = α×BC + β×AP + γ×I + δ×DC + ε×PR
    
    Where:
    - BC: Betweenness centrality (normalized)
    - AP: Articulation point (1.0 if AP, 0.0 otherwise)
    - I: Impact score (based on reachability)
    - DC: Degree centrality
    - PR: PageRank
    
    Args:
        graph: NetworkX graph
        alpha: Weight for betweenness centrality
        beta: Weight for articulation points
        gamma: Weight for impact score
        delta: Weight for degree centrality
        epsilon: Weight for PageRank
        
    Returns:
        Dict of component_id -> criticality_score
    """
    scores = {}
    
    if graph.number_of_nodes() == 0:
        return scores
    
    # Calculate metrics
    bc = nx.betweenness_centrality(graph)
    
    # Normalize betweenness
    max_bc = max(bc.values()) if bc.values() else 1
    bc_norm = {k: v / max_bc if max_bc > 0 else 0 for k, v in bc.items()}
    
    # Articulation points (on undirected version)
    try:
        aps = set(nx.articulation_points(graph.to_undirected()))
    except:
        aps = set()
    
    # Degree centrality
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    max_degree = max(max(in_degree.values(), default=1), max(out_degree.values(), default=1))
    
    # PageRank
    try:
        pr = nx.pagerank(graph, alpha=0.85)
    except:
        pr = {n: 1.0 / graph.number_of_nodes() for n in graph.nodes()}
    
    max_pr = max(pr.values()) if pr.values() else 1
    pr_norm = {k: v / max_pr if max_pr > 0 else 0 for k, v in pr.items()}
    
    # Calculate composite score for each node
    for node in graph.nodes():
        bc_score = bc_norm.get(node, 0)
        ap_score = 1.0 if node in aps else 0.0
        
        # Impact score based on reachability
        try:
            descendants = len(nx.descendants(graph, node))
            ancestors = len(nx.ancestors(graph, node))
            impact = (descendants + ancestors) / (2 * graph.number_of_nodes())
        except:
            impact = 0.0
        
        # Degree centrality
        degree_score = (in_degree.get(node, 0) + out_degree.get(node, 0)) / (2 * max_degree)
        
        # PageRank
        pr_score = pr_norm.get(node, 0)
        
        # Composite score
        composite = (alpha * bc_score + 
                    beta * ap_score + 
                    gamma * impact + 
                    delta * degree_score + 
                    epsilon * pr_score)
        
        scores[node] = min(1.0, composite)
    
    return scores


def run_failure_simulation(graph: nx.DiGraph,
                          enable_cascade: bool = False,
                          seed: Optional[int] = None) -> Dict[str, float]:
    """
    Run exhaustive failure simulation to get actual impact scores.
    
    Args:
        graph: NetworkX graph
        enable_cascade: Enable cascade propagation
        seed: Random seed
        
    Returns:
        Dict of component_id -> impact_score
    """
    simulator = FailureSimulator(seed=seed)
    batch_result = simulator.simulate_exhaustive(graph, enable_cascade=enable_cascade)
    
    impacts = {}
    for result in batch_result.results:
        if result.primary_failures:
            comp = result.primary_failures[0]
            impacts[comp] = result.impact_score
    
    return impacts


# ============================================================================
# Output Functions
# ============================================================================

def print_validation_result(result: ValidationResult, targets: Dict[str, float], verbose: bool = False):
    """Print validation results to terminal"""
    
    # Status
    print_section("Validation Status")
    status_col = status_color(result.status)
    print(f"\n  Status: {status_col}{Colors.BOLD}{result.status.value.upper()}{Colors.ENDC}")
    print(f"  Components Validated: {result.total_components}")
    print(f"  Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Correlation metrics
    print_section("Correlation Metrics")
    
    corr = result.correlation
    spearman_target = targets.get('spearman_correlation', 0.7)
    spearman_col = metric_color(corr.spearman_coefficient, spearman_target)
    
    print(f"\n  Spearman ρ: {spearman_col}{corr.spearman_coefficient:.4f}{Colors.ENDC} "
          f"(target: ≥{spearman_target})")
    print(f"    p-value: {corr.spearman_p_value:.6f} "
          f"{'✓ significant' if corr.spearman_p_value < 0.05 else '✗ not significant'}")
    
    pearson_col = metric_color(corr.pearson_coefficient, 0.5)
    print(f"  Pearson r: {pearson_col}{corr.pearson_coefficient:.4f}{Colors.ENDC}")
    print(f"    p-value: {corr.pearson_p_value:.6f}")
    
    # Classification metrics
    print_section("Classification Metrics")
    
    cm = result.confusion_matrix
    
    precision_target = targets.get('precision', 0.8)
    precision_col = metric_color(cm.precision, precision_target)
    print(f"\n  Precision: {precision_col}{cm.precision:.4f}{Colors.ENDC} "
          f"(target: ≥{precision_target})")
    
    recall_target = targets.get('recall', 0.8)
    recall_col = metric_color(cm.recall, recall_target)
    print(f"  Recall:    {recall_col}{cm.recall:.4f}{Colors.ENDC} "
          f"(target: ≥{recall_target})")
    
    f1_target = targets.get('f1_score', 0.9)
    f1_col = metric_color(cm.f1_score, f1_target)
    print(f"  F1-Score:  {f1_col}{cm.f1_score:.4f}{Colors.ENDC} "
          f"(target: ≥{f1_target})")
    
    print(f"  Accuracy:  {cm.accuracy:.4f}")
    
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {cm.true_positives:3d}  FP: {cm.false_positives:3d}")
    print(f"    FN: {cm.false_negatives:3d}  TN: {cm.true_negatives:3d}")
    
    # Ranking metrics
    print_section("Ranking Metrics")
    
    ranking = result.ranking
    print(f"\n  Mean Rank Difference: {ranking.mean_rank_difference:.1f}")
    print(f"  Max Rank Difference:  {ranking.max_rank_difference}")
    print(f"  Kendall's τ:          {ranking.kendall_tau:.4f}")
    
    print(f"\n  Top-k Overlap:")
    for k, overlap in sorted(ranking.top_k_overlap.items()):
        target_key = f'top_{k}_overlap'
        target = targets.get(target_key, 0.5)
        col = metric_color(overlap, target)
        print(f"    Top-{k:2d}: {col}{overlap:.1%}{Colors.ENDC} (target: ≥{target:.0%})")
    
    # Component analysis
    if verbose:
        print_section("Component Analysis")
        
        # Top predicted vs actual
        sorted_by_pred = sorted(result.component_validations,
                               key=lambda x: x.predicted_score, reverse=True)[:5]
        sorted_by_actual = sorted(result.component_validations,
                                 key=lambda x: x.actual_impact, reverse=True)[:5]
        
        print(f"\n  Top 5 by Predicted Score:")
        for cv in sorted_by_pred:
            match = "✓" if cv.correctly_classified else "✗"
            print(f"    {cv.component_id:25s} pred={cv.predicted_score:.3f} "
                  f"actual={cv.actual_impact:.3f} {match}")
        
        print(f"\n  Top 5 by Actual Impact:")
        for cv in sorted_by_actual:
            match = "✓" if cv.correctly_classified else "✗"
            print(f"    {cv.component_id:25s} pred={cv.predicted_score:.3f} "
                  f"actual={cv.actual_impact:.3f} {match}")
        
        # Misclassified
        misclassified = [cv for cv in result.component_validations 
                        if not cv.correctly_classified]
        if misclassified:
            print(f"\n  Misclassified Components ({len(misclassified)}):")
            for cv in misclassified[:5]:
                print(f"    {cv.component_id:25s} pred_level={cv.predicted_level} "
                      f"actual_level={cv.actual_level}")
    
    # Advanced analysis results
    if result.sensitivity_results:
        print_section("Sensitivity Analysis")
        for sr in result.sensitivity_results:
            print(f"\n  Parameter: {sr.parameter_name}")
            print(f"  Original: {sr.original_value:.2f}")
            print(f"  Stability: {sr.stability_score:.2%}")
    
    if result.bootstrap_results:
        print_section("Bootstrap Confidence Intervals")
        for br in result.bootstrap_results:
            print(f"\n  {br.metric_name}:")
            print(f"    Point Estimate: {br.point_estimate:.4f}")
            print(f"    {br.confidence_level:.0%} CI: [{br.ci_lower:.4f}, {br.ci_upper:.4f}]")
    
    if result.cross_validation:
        print_section("Cross-Validation Results")
        cv = result.cross_validation
        print(f"\n  {cv.n_folds}-Fold Cross-Validation:")
        for metric, mean in cv.mean_metrics.items():
            std = cv.std_metrics.get(metric, 0)
            print(f"    {metric}: {mean:.4f} ± {std:.4f}")


def export_json(result: ValidationResult, filepath: Path):
    """Export results to JSON"""
    with open(filepath, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)


def export_csv(result: ValidationResult, output_dir: Path):
    """Export results to CSV files"""
    # Summary metrics
    summary_path = output_dir / 'validation_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'target', 'passed'])
        
        metrics = [
            ('spearman_correlation', result.correlation.spearman_coefficient, 0.7),
            ('pearson_correlation', result.correlation.pearson_coefficient, 0.5),
            ('f1_score', result.confusion_matrix.f1_score, 0.9),
            ('precision', result.confusion_matrix.precision, 0.8),
            ('recall', result.confusion_matrix.recall, 0.8),
        ]
        
        for metric, value, target in metrics:
            writer.writerow([metric, f"{value:.4f}", f"{target:.2f}", value >= target])
    
    # Component details
    components_path = output_dir / 'validation_components.csv'
    with open(components_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['component_id', 'type', 'predicted_score', 'predicted_rank',
                        'predicted_level', 'actual_impact', 'actual_rank', 'actual_level',
                        'rank_diff', 'score_diff', 'correct'])
        
        for cv in result.component_validations:
            writer.writerow([
                cv.component_id, cv.component_type,
                f"{cv.predicted_score:.4f}", cv.predicted_rank, cv.predicted_level,
                f"{cv.actual_impact:.4f}", cv.actual_rank, cv.actual_level,
                cv.rank_difference, f"{cv.score_difference:.4f}", cv.correctly_classified
            ])
    
    return summary_path, components_path


def export_html(result: ValidationResult, filepath: Path, targets: Dict[str, float]):
    """Export results to HTML report"""
    
    status_class = {
        ValidationStatus.PASSED: 'passed',
        ValidationStatus.MARGINAL: 'marginal',
        ValidationStatus.FAILED: 'failed'
    }[result.status]
    
    # Build component rows
    component_rows = []
    for cv in result.component_validations[:50]:  # Limit to 50
        correct_class = 'correct' if cv.correctly_classified else 'incorrect'
        component_rows.append(f"""
            <tr class="{correct_class}">
                <td>{cv.component_id}</td>
                <td>{cv.component_type}</td>
                <td>{cv.predicted_score:.4f}</td>
                <td>{cv.predicted_rank}</td>
                <td>{cv.actual_impact:.4f}</td>
                <td>{cv.actual_rank}</td>
                <td>{cv.rank_difference}</td>
                <td>{'✓' if cv.correctly_classified else '✗'}</td>
            </tr>
        """)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Graph Analysis Validation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 3px solid #2196F3; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        .passed {{ color: #4CAF50; }}
        .marginal {{ color: #FF9800; }}
        .failed {{ color: #F44336; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f5f5f5; }}
        tr.incorrect {{ background: #ffebee; }}
        tr.correct {{ background: #e8f5e9; }}
        .badge {{ padding: 3px 10px; border-radius: 12px; font-size: 0.85em; font-weight: 500; }}
        .badge-success {{ background: #e8f5e9; color: #2e7d32; }}
        .badge-danger {{ background: #ffebee; color: #c62828; }}
        .badge-warning {{ background: #fff3e0; color: #ef6c00; }}
        .progress {{ background: #e9ecef; border-radius: 4px; height: 24px; overflow: hidden; }}
        .progress-bar {{ height: 100%; transition: width 0.3s; }}
        .progress-bar.success {{ background: #4CAF50; }}
        .progress-bar.warning {{ background: #FF9800; }}
        .progress-bar.danger {{ background: #F44336; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Graph Analysis Validation Report</h1>
        <p>Generated: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="card">
            <h2>Summary</h2>
            <div class="metric">
                <div class="metric-value {status_class}">{result.status.value.upper()}</div>
                <div class="metric-label">Status</div>
            </div>
            <div class="metric">
                <div class="metric-value">{result.total_components}</div>
                <div class="metric-label">Components</div>
            </div>
            <div class="metric">
                <div class="metric-value">{result.correlation.spearman_coefficient:.3f}</div>
                <div class="metric-label">Spearman ρ</div>
            </div>
            <div class="metric">
                <div class="metric-value">{result.confusion_matrix.f1_score:.3f}</div>
                <div class="metric-label">F1-Score</div>
            </div>
        </div>
        
        <div class="card">
            <h2>Correlation Analysis</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Target</th>
                    <th>P-Value</th>
                    <th>Status</th>
                </tr>
                <tr>
                    <td>Spearman Correlation</td>
                    <td>{result.correlation.spearman_coefficient:.4f}</td>
                    <td>≥ 0.70</td>
                    <td>{result.correlation.spearman_p_value:.6f}</td>
                    <td><span class="badge {'badge-success' if result.correlation.spearman_coefficient >= 0.7 else 'badge-danger'}">
                        {'PASS' if result.correlation.spearman_coefficient >= 0.7 else 'FAIL'}</span></td>
                </tr>
                <tr>
                    <td>Pearson Correlation</td>
                    <td>{result.correlation.pearson_coefficient:.4f}</td>
                    <td>-</td>
                    <td>{result.correlation.pearson_p_value:.6f}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Kendall's τ</td>
                    <td>{result.ranking.kendall_tau:.4f}</td>
                    <td>-</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
            </table>
        </div>
        
        <div class="card">
            <h2>Classification Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Target</th>
                    <th>Progress</th>
                    <th>Status</th>
                </tr>
                <tr>
                    <td>Precision</td>
                    <td>{result.confusion_matrix.precision:.4f}</td>
                    <td>≥ 0.80</td>
                    <td><div class="progress"><div class="progress-bar {'success' if result.confusion_matrix.precision >= 0.8 else 'warning' if result.confusion_matrix.precision >= 0.6 else 'danger'}" 
                        style="width: {result.confusion_matrix.precision*100:.0f}%"></div></div></td>
                    <td><span class="badge {'badge-success' if result.confusion_matrix.precision >= 0.8 else 'badge-danger'}">
                        {'PASS' if result.confusion_matrix.precision >= 0.8 else 'FAIL'}</span></td>
                </tr>
                <tr>
                    <td>Recall</td>
                    <td>{result.confusion_matrix.recall:.4f}</td>
                    <td>≥ 0.80</td>
                    <td><div class="progress"><div class="progress-bar {'success' if result.confusion_matrix.recall >= 0.8 else 'warning' if result.confusion_matrix.recall >= 0.6 else 'danger'}" 
                        style="width: {result.confusion_matrix.recall*100:.0f}%"></div></div></td>
                    <td><span class="badge {'badge-success' if result.confusion_matrix.recall >= 0.8 else 'badge-danger'}">
                        {'PASS' if result.confusion_matrix.recall >= 0.8 else 'FAIL'}</span></td>
                </tr>
                <tr>
                    <td>F1-Score</td>
                    <td>{result.confusion_matrix.f1_score:.4f}</td>
                    <td>≥ 0.90</td>
                    <td><div class="progress"><div class="progress-bar {'success' if result.confusion_matrix.f1_score >= 0.9 else 'warning' if result.confusion_matrix.f1_score >= 0.7 else 'danger'}" 
                        style="width: {result.confusion_matrix.f1_score*100:.0f}%"></div></div></td>
                    <td><span class="badge {'badge-success' if result.confusion_matrix.f1_score >= 0.9 else 'badge-danger'}">
                        {'PASS' if result.confusion_matrix.f1_score >= 0.9 else 'FAIL'}</span></td>
                </tr>
                <tr>
                    <td>Accuracy</td>
                    <td>{result.confusion_matrix.accuracy:.4f}</td>
                    <td>-</td>
                    <td><div class="progress"><div class="progress-bar success" 
                        style="width: {result.confusion_matrix.accuracy*100:.0f}%"></div></div></td>
                    <td>-</td>
                </tr>
            </table>
            
            <h3>Confusion Matrix</h3>
            <table style="width: 300px;">
                <tr><th></th><th>Pred Critical</th><th>Pred Non-Critical</th></tr>
                <tr><th>Actual Critical</th><td>{result.confusion_matrix.true_positives}</td><td>{result.confusion_matrix.false_negatives}</td></tr>
                <tr><th>Actual Non-Critical</th><td>{result.confusion_matrix.false_positives}</td><td>{result.confusion_matrix.true_negatives}</td></tr>
            </table>
        </div>
        
        <div class="card">
            <h2>Ranking Analysis</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Target</th>
                    <th>Status</th>
                </tr>
                {''.join([f'''<tr>
                    <td>Top-{k} Overlap</td>
                    <td>{overlap:.1%}</td>
                    <td>≥ {targets.get(f"top_{k}_overlap", 0.5):.0%}</td>
                    <td><span class="badge {'badge-success' if overlap >= targets.get(f"top_{k}_overlap", 0.5) else 'badge-danger'}">
                        {'PASS' if overlap >= targets.get(f"top_{k}_overlap", 0.5) else 'FAIL'}</span></td>
                </tr>''' for k, overlap in sorted(result.ranking.top_k_overlap.items())])}
                <tr>
                    <td>Mean Rank Difference</td>
                    <td>{result.ranking.mean_rank_difference:.1f}</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Max Rank Difference</td>
                    <td>{result.ranking.max_rank_difference}</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
            </table>
        </div>
        
        <div class="card">
            <h2>Component Details</h2>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Type</th>
                    <th>Predicted</th>
                    <th>Pred Rank</th>
                    <th>Actual</th>
                    <th>Actual Rank</th>
                    <th>Rank Diff</th>
                    <th>Correct</th>
                </tr>
                {''.join(component_rows)}
            </table>
        </div>
    </div>
</body>
</html>
"""
    
    with open(filepath, 'w') as f:
        f.write(html)


# ============================================================================
# Main CLI
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description='Validate graph-based criticality analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic validation
    python validate_graph.py --input system.json
    
    # Custom scoring weights
    python validate_graph.py --input system.json --alpha 0.4 --beta 0.3 --gamma 0.3
    
    # Full analysis with sensitivity and bootstrap
    python validate_graph.py --input system.json --full-analysis
    
    # Custom targets
    python validate_graph.py --input system.json --target-spearman 0.8 --target-f1 0.85
    
    # Export results
    python validate_graph.py --input system.json --output results/ --format json html csv

Target Metrics (defaults):
    - Spearman Correlation: ≥ 0.70
    - F1-Score: ≥ 0.90
    - Precision: ≥ 0.80
    - Recall: ≥ 0.80
    - Top-5 Overlap: ≥ 60%
    - Top-10 Overlap: ≥ 70%
        """
    )
    
    # Input
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('--input', '-i', required=True,
                            help='Input JSON file with pub-sub system')
    
    # Scoring weights
    weights_group = parser.add_argument_group('Scoring Weights')
    weights_group.add_argument('--alpha', type=float, default=0.25,
                              help='Betweenness centrality weight (default: 0.25)')
    weights_group.add_argument('--beta', type=float, default=0.30,
                              help='Articulation point weight (default: 0.30)')
    weights_group.add_argument('--gamma', type=float, default=0.25,
                              help='Impact score weight (default: 0.25)')
    weights_group.add_argument('--delta', type=float, default=0.10,
                              help='Degree centrality weight (default: 0.10)')
    weights_group.add_argument('--epsilon', type=float, default=0.10,
                              help='PageRank weight (default: 0.10)')
    
    # Target thresholds
    targets_group = parser.add_argument_group('Target Thresholds')
    targets_group.add_argument('--target-spearman', type=float, default=0.70,
                              help='Target Spearman correlation (default: 0.70)')
    targets_group.add_argument('--target-f1', type=float, default=0.90,
                              help='Target F1-score (default: 0.90)')
    targets_group.add_argument('--target-precision', type=float, default=0.80,
                              help='Target precision (default: 0.80)')
    targets_group.add_argument('--target-recall', type=float, default=0.80,
                              help='Target recall (default: 0.80)')
    targets_group.add_argument('--target-top5', type=float, default=0.60,
                              help='Target top-5 overlap (default: 0.60)')
    targets_group.add_argument('--target-top10', type=float, default=0.70,
                              help='Target top-10 overlap (default: 0.70)')
    targets_group.add_argument('--critical-threshold', type=float, default=0.50,
                              help='Impact threshold for "critical" (default: 0.50)')
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument('--cascade', action='store_true',
                               help='Enable cascade in simulation')
    analysis_group.add_argument('--full-analysis', action='store_true',
                               help='Run sensitivity, bootstrap, and cross-validation')
    analysis_group.add_argument('--sensitivity', action='store_true',
                               help='Run sensitivity analysis')
    analysis_group.add_argument('--bootstrap', action='store_true',
                               help='Run bootstrap analysis')
    analysis_group.add_argument('--cross-validation', action='store_true',
                               help='Run k-fold cross-validation')
    analysis_group.add_argument('--n-folds', type=int, default=5,
                               help='Number of folds for cross-validation (default: 5)')
    analysis_group.add_argument('--bootstrap-iterations', type=int, default=1000,
                               help='Bootstrap iterations (default: 1000)')
    
    # Output
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output', '-o',
                             help='Output directory for results')
    output_group.add_argument('--format', nargs='+', default=['json'],
                             choices=['json', 'html', 'csv'],
                             help='Output formats (default: json)')
    output_group.add_argument('--verbose', '-v', action='store_true',
                             help='Verbose output')
    output_group.add_argument('--quiet', '-q', action='store_true',
                             help='Minimal output')
    output_group.add_argument('--no-color', action='store_true',
                             help='Disable colored output')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup
    if args.no_color:
        Colors.disable()
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING)
    
    if not args.quiet:
        print_header("GRAPH ANALYSIS VALIDATION")
        print(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Input: {args.input}")
    
    try:
        # Load graph
        if not args.quiet:
            print_section("Loading Graph")
        
        graph = load_graph_from_json(args.input)
        print_success(f"Loaded graph: {graph.number_of_nodes()} nodes, "
                     f"{graph.number_of_edges()} edges")
        
        # Build targets
        targets = {
            'spearman_correlation': args.target_spearman,
            'f1_score': args.target_f1,
            'precision': args.target_precision,
            'recall': args.target_recall,
            'top_5_overlap': args.target_top5,
            'top_10_overlap': args.target_top10
        }
        
        # Calculate predicted scores
        if not args.quiet:
            print_section("Calculating Criticality Scores")
            print(f"\n  Weights: α={args.alpha}, β={args.beta}, γ={args.gamma}, "
                  f"δ={args.delta}, ε={args.epsilon}")
        
        predicted = calculate_criticality_scores(
            graph,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            delta=args.delta,
            epsilon=args.epsilon
        )
        print_success(f"Calculated scores for {len(predicted)} components")
        
        # Run failure simulation
        if not args.quiet:
            print_section("Running Failure Simulation")
            print_info(f"Cascade: {'enabled' if args.cascade else 'disabled'}")
        
        actual = run_failure_simulation(graph, enable_cascade=args.cascade, seed=args.seed)
        print_success(f"Simulated failures for {len(actual)} components")
        
        # Run validation
        if not args.quiet:
            print_section("Validating Results")
        
        validator = GraphValidator(
            targets=targets,
            critical_threshold=args.critical_threshold,
            seed=args.seed
        )
        
        result = validator.validate(graph, predicted, actual)
        print_success("Validation complete")
        
        # Run advanced analysis if requested
        if args.full_analysis or args.sensitivity:
            if not args.quiet:
                print_info("Running sensitivity analysis...")
            validator.run_sensitivity_analysis(graph, predicted, actual)
        
        if args.full_analysis or args.bootstrap:
            if not args.quiet:
                print_info(f"Running bootstrap analysis ({args.bootstrap_iterations} iterations)...")
            validator.run_bootstrap_analysis(
                graph, predicted, actual,
                n_iterations=args.bootstrap_iterations
            )
        
        if args.full_analysis or args.cross_validation:
            if not args.quiet:
                print_info(f"Running {args.n_folds}-fold cross-validation...")
            validator.run_cross_validation(
                graph, predicted, actual,
                n_folds=args.n_folds
            )
        
        # Print results
        if not args.quiet:
            print_validation_result(result, targets, verbose=args.verbose)
        
        # Generate recommendations
        recommendations = validator.generate_recommendations()
        if recommendations and not args.quiet:
            print_section("Recommendations")
            for i, rec in enumerate(recommendations, 1):
                print(f"\n  {i}. {rec}")
        
        # Export results
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if 'json' in args.format:
                json_path = output_dir / 'validation_results.json'
                export_json(result, json_path)
                print_success(f"Exported JSON: {json_path}")
            
            if 'html' in args.format:
                html_path = output_dir / 'validation_report.html'
                export_html(result, html_path, targets)
                print_success(f"Exported HTML: {html_path}")
            
            if 'csv' in args.format:
                csv_paths = export_csv(result, output_dir)
                for path in csv_paths:
                    print_success(f"Exported CSV: {path}")
        
        # Return exit code based on status
        if result.status == ValidationStatus.PASSED:
            return 0
        elif result.status == ValidationStatus.MARGINAL:
            return 0  # Still success, but with warnings
        else:
            return 1
    
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        return 1
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}")
        return 1
    except ValueError as e:
        print_error(f"Validation error: {e}")
        return 1
    except Exception as e:
        print_error(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())