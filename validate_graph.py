#!/usr/bin/env python3
"""
Graph Validation CLI
====================

Validates the graph-based modeling and analysis approach by comparing
predicted criticality scores with actual impact from failure simulation.

Usage:
    # Basic validation
    python validate_graph.py
    
    # With custom scoring weights
    python validate_graph.py --alpha 0.5 --beta 0.25 --gamma 0.25
    
    # Custom target thresholds
    python validate_graph.py --target-spearman 0.8 --target-f1 0.85
    
    # Export detailed results
    python validate_graph.py --output-dir results/ --format json html csv

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from src.analysis import GraphAnalyzer, NEO4J_AVAILABLE
from src.simulation import GraphSimulator
from src.validation import (
    GraphValidator,
    ValidationResult,
    ValidationStatus,
    validate_analysis,
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
        cls.HEADER = ''
        cls.BLUE = ''
        cls.CYAN = ''
        cls.GREEN = ''
        cls.WARNING = ''
        cls.RED = ''
        cls.ENDC = ''
        cls.BOLD = ''
        cls.DIM = ''
        cls._enabled = False


def print_header(text: str):
    """Print a header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text:^60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")


def print_section(text: str):
    """Print a section header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}── {text} ──{Colors.ENDC}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.ENDC} {text}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {text}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.ENDC} {text}")


def status_color(status: ValidationStatus) -> str:
    """Get color for validation status"""
    if status == ValidationStatus.PASSED:
        return Colors.GREEN
    elif status == ValidationStatus.MARGINAL:
        return Colors.WARNING
    else:
        return Colors.RED


def metric_color(achieved: float, target: float) -> str:
    """Get color based on whether metric meets target"""
    if achieved >= target:
        return Colors.GREEN
    elif achieved >= target * 0.8:
        return Colors.WARNING
    else:
        return Colors.RED


# ============================================================================
# Output Formatters
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
    
    spearman_target = targets.get('spearman_correlation', 0.7)
    spearman_col = metric_color(result.spearman_correlation, spearman_target)
    print(f"\n  Spearman Correlation: {spearman_col}{result.spearman_correlation:.4f}{Colors.ENDC} "
          f"(target: {spearman_target})")
    print(f"    p-value: {result.spearman_p_value:.6f}")
    
    pearson_col = metric_color(result.pearson_correlation, 0.5)
    print(f"  Pearson Correlation:  {pearson_col}{result.pearson_correlation:.4f}{Colors.ENDC}")
    print(f"    p-value: {result.pearson_p_value:.6f}")
    
    # Classification metrics
    print_section("Classification Metrics")
    
    cm = result.confusion_matrix
    
    precision_target = targets.get('precision', 0.8)
    precision_col = metric_color(cm.precision, precision_target)
    print(f"\n  Precision: {precision_col}{cm.precision:.4f}{Colors.ENDC} (target: {precision_target})")
    
    recall_target = targets.get('recall', 0.8)
    recall_col = metric_color(cm.recall, recall_target)
    print(f"  Recall:    {recall_col}{cm.recall:.4f}{Colors.ENDC} (target: {recall_target})")
    
    f1_target = targets.get('f1_score', 0.9)
    f1_col = metric_color(cm.f1_score, f1_target)
    print(f"  F1-Score:  {f1_col}{cm.f1_score:.4f}{Colors.ENDC} (target: {f1_target})")
    
    print(f"  Accuracy:  {cm.accuracy:.4f}")
    
    # Confusion matrix details
    if verbose:
        print(f"\n  Confusion Matrix:")
        print(f"    TP: {cm.true_positives}  FP: {cm.false_positives}")
        print(f"    FN: {cm.false_negatives}  TN: {cm.true_negatives}")
    
    # Ranking metrics
    print_section("Ranking Metrics")
    
    print(f"\n  Mean Rank Difference: {result.mean_rank_difference:.2f}")
    print(f"  Max Rank Difference:  {result.max_rank_difference}")
    
    print(f"\n  Top-K Overlap:")
    for k, overlap in sorted(result.top_k_overlap.items()):
        target_key = f'top_{k}_overlap'
        target_val = targets.get(target_key, 0.5)
        col = metric_color(overlap, target_val)
        print(f"    Top-{k}: {col}{overlap:.1%}{Colors.ENDC} (target: {target_val:.0%})")
    
    # Component comparison (top 10)
    if verbose and result.component_validations:
        print_section("Component Comparison (Top 10 by Actual Impact)")
        
        print(f"\n  {'Component':<15} {'Predicted':>10} {'Actual':>10} {'Rank Δ':>8} {'Match':>6}")
        print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*8} {'-'*6}")
        
        for cv in result.component_validations[:10]:
            match = "✓" if cv.correctly_classified else "✗"
            match_col = Colors.GREEN if cv.correctly_classified else Colors.RED
            print(f"  {cv.component_id:<15} {cv.predicted_score:>10.4f} {cv.actual_impact:>10.2%} "
                  f"{cv.rank_difference:>8} {match_col}{match:>6}{Colors.ENDC}")


def print_recommendations(recommendations: List[str]):
    """Print recommendations"""
    print_section("Recommendations")
    for rec in recommendations:
        if "passed" in rec.lower():
            print(f"\n  {Colors.GREEN}✓{Colors.ENDC} {rec}")
        elif "below target" in rec.lower() or "high" in rec.lower():
            print(f"\n  {Colors.WARNING}⚠{Colors.ENDC} {rec}")
        else:
            print(f"\n  • {rec}")


# ============================================================================
# Export Functions
# ============================================================================

def export_json(result: ValidationResult, report: Dict, filepath: Path):
    """Export results to JSON"""
    data = {
        'validation_result': result.to_dict(),
        'report': report
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print_success(f"JSON exported to {filepath}")


def export_csv(result: ValidationResult, filepath: Path):
    """Export component comparison to CSV"""
    with open(filepath, 'w') as f:
        # Header
        f.write("component,type,predicted_score,predicted_rank,predicted_level,"
                "actual_impact,actual_rank,actual_level,rank_diff,correctly_classified\n")
        
        for cv in result.component_validations:
            f.write(f"{cv.component_id},{cv.component_type},{cv.predicted_score:.4f},"
                   f"{cv.predicted_rank},{cv.predicted_level},{cv.actual_impact:.4f},"
                   f"{cv.actual_rank},{cv.actual_level},{cv.rank_difference},"
                   f"{cv.correctly_classified}\n")
    
    print_success(f"CSV exported to {filepath}")


def export_html(result: ValidationResult, report: Dict, filepath: Path):
    """Export results to HTML report"""
    
    status_class = {
        ValidationStatus.PASSED: 'success',
        ValidationStatus.MARGINAL: 'warning',
        ValidationStatus.FAILED: 'danger'
    }.get(result.status, 'info')
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Validation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1, h2, h3 {{ color: #333; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        .success {{ color: #28a745; }}
        .warning {{ color: #fd7e14; }}
        .danger {{ color: #dc3545; }}
        .info {{ color: #17a2b8; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f8f9fa; }}
        .progress {{ background: #e9ecef; border-radius: 4px; height: 20px; overflow: hidden; }}
        .progress-bar {{ height: 100%; }}
        .badge {{ padding: 3px 8px; border-radius: 4px; font-size: 0.85em; }}
        .badge-success {{ background: #d4edda; color: #155724; }}
        .badge-danger {{ background: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <h1>Graph Analysis Validation Report</h1>
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
            <div class="metric-value">{result.spearman_correlation:.3f}</div>
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
            <tr><th>Metric</th><th>Value</th><th>P-Value</th><th>Target</th><th>Status</th></tr>
            <tr>
                <td>Spearman Correlation</td>
                <td>{result.spearman_correlation:.4f}</td>
                <td>{result.spearman_p_value:.6f}</td>
                <td>≥ 0.70</td>
                <td><span class="badge {'badge-success' if result.spearman_correlation >= 0.7 else 'badge-danger'}">
                    {'PASS' if result.spearman_correlation >= 0.7 else 'FAIL'}</span></td>
            </tr>
            <tr>
                <td>Pearson Correlation</td>
                <td>{result.pearson_correlation:.4f}</td>
                <td>{result.pearson_p_value:.6f}</td>
                <td>-</td>
                <td>-</td>
            </tr>
        </table>
    </div>
    
    <div class="card">
        <h2>Classification Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th><th>Target</th><th>Status</th></tr>
            <tr>
                <td>Precision</td>
                <td>{result.confusion_matrix.precision:.4f}</td>
                <td>≥ 0.80</td>
                <td><span class="badge {'badge-success' if result.confusion_matrix.precision >= 0.8 else 'badge-danger'}">
                    {'PASS' if result.confusion_matrix.precision >= 0.8 else 'FAIL'}</span></td>
            </tr>
            <tr>
                <td>Recall</td>
                <td>{result.confusion_matrix.recall:.4f}</td>
                <td>≥ 0.80</td>
                <td><span class="badge {'badge-success' if result.confusion_matrix.recall >= 0.8 else 'badge-danger'}">
                    {'PASS' if result.confusion_matrix.recall >= 0.8 else 'FAIL'}</span></td>
            </tr>
            <tr>
                <td>F1-Score</td>
                <td>{result.confusion_matrix.f1_score:.4f}</td>
                <td>≥ 0.90</td>
                <td><span class="badge {'badge-success' if result.confusion_matrix.f1_score >= 0.9 else 'badge-danger'}">
                    {'PASS' if result.confusion_matrix.f1_score >= 0.9 else 'FAIL'}</span></td>
            </tr>
            <tr>
                <td>Accuracy</td>
                <td>{result.confusion_matrix.accuracy:.4f}</td>
                <td>-</td>
                <td>-</td>
            </tr>
        </table>
        
        <h3>Confusion Matrix</h3>
        <table style="width: auto;">
            <tr><th></th><th>Predicted Critical</th><th>Predicted Non-Critical</th></tr>
            <tr>
                <th>Actually Critical</th>
                <td class="success">{result.confusion_matrix.true_positives} (TP)</td>
                <td class="danger">{result.confusion_matrix.false_negatives} (FN)</td>
            </tr>
            <tr>
                <th>Actually Non-Critical</th>
                <td class="warning">{result.confusion_matrix.false_positives} (FP)</td>
                <td class="success">{result.confusion_matrix.true_negatives} (TN)</td>
            </tr>
        </table>
    </div>
    
    <div class="card">
        <h2>Component Comparison</h2>
        <table>
            <tr>
                <th>Component</th>
                <th>Type</th>
                <th>Predicted Score</th>
                <th>Actual Impact</th>
                <th>Rank Δ</th>
                <th>Classification</th>
            </tr>
"""
    
    for cv in result.component_validations[:20]:
        match_class = 'success' if cv.correctly_classified else 'danger'
        match_text = '✓ Correct' if cv.correctly_classified else '✗ Misclassified'
        html += f"""
            <tr>
                <td>{cv.component_id}</td>
                <td>{cv.component_type}</td>
                <td>{cv.predicted_score:.4f}</td>
                <td>{cv.actual_impact:.2%}</td>
                <td>{cv.rank_difference}</td>
                <td class="{match_class}">{match_text}</td>
            </tr>
"""
    
    html += """
        </table>
    </div>
    
    <div class="card">
        <h2>Recommendations</h2>
        <ul>
"""
    
    for rec in report.get('recommendations', []):
        html += f"            <li>{rec}</li>\n"
    
    html += """
        </ul>
    </div>
</body>
</html>
"""
    
    with open(filepath, 'w') as f:
        f.write(html)
    
    print_success(f"HTML report exported to {filepath}")


# ============================================================================
# Main Functions
# ============================================================================

def run_validation(args) -> int:
    """Run the validation"""
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger('validate_graph')
    
    # Handle colors
    if not sys.stdout.isatty() or args.no_color:
        Colors.disable()
    
    # Print header
    if not args.quiet:
        print_header("GRAPH ANALYSIS VALIDATION")
    
    try:
        # Load data
        if not args.quiet:
            print_section("Loading Data")
        
        analyzer = GraphAnalyzer(
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma
        )
        
        if not NEO4J_AVAILABLE:
            print_error("Neo4j driver not installed. Install with: pip install neo4j")
            return 1
        
        analyzer.load_from_neo4j(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            database=args.neo4j_database
        )
        if not args.quiet:
            print_success(f"Loaded from Neo4j: {args.neo4j_uri}")
        
        # Build target thresholds
        targets = {
            'spearman_correlation': args.target_spearman,
            'f1_score': args.target_f1,
            'precision': args.target_precision,
            'recall': args.target_recall,
            'top_5_overlap': args.target_top5,
            'top_10_overlap': args.target_top10,
        }
        
        # Create validator
        simulator = GraphSimulator(seed=args.seed)
        validator = GraphValidator(
            analyzer=analyzer,
            simulator=simulator,
            targets=targets,
            critical_threshold=args.critical_threshold,
            seed=args.seed
        )
        
        # Run validation
        if not args.quiet:
            print_section("Running Validation")
            print("\n  1. Analyzing system...")
        
        result = validator.validate(
            component_types=args.component_types,
            enable_cascade=args.cascade
        )
        
        if not args.quiet:
            print("  2. Running exhaustive simulation...")
            print("  3. Calculating metrics...")
            print_success("Validation completed")
        
        # Generate report
        report = validator.generate_report()
        
        # Print results
        if not args.quiet:
            print_validation_result(result, targets, verbose=args.verbose)
            print_recommendations(report['recommendations'])
        
        # Export results
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if not args.quiet:
                print_section("Exporting Results")
            
            for fmt in args.format:
                if fmt == 'json':
                    export_json(result, report, output_dir / 'validation_results.json')
                elif fmt == 'html':
                    export_html(result, report, output_dir / 'validation_report.html')
                elif fmt == 'csv':
                    export_csv(result, output_dir / 'component_comparison.csv')
        
        # JSON to stdout
        if args.json_output:
            print(json.dumps(result.to_dict(), indent=2, default=str))
        
        # Return based on status
        if result.status == ValidationStatus.PASSED:
            return 0
        elif result.status == ValidationStatus.MARGINAL:
            return 0  # Still success, just marginal
        else:
            return 1  # Failed
        
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        return 1
    except ValueError as e:
        print_error(f"Invalid input: {e}")
        return 1
    except Exception as e:
        print_error(f"Validation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Validate graph-based analysis by comparing predictions with simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic validation
    python validate_graph.py
    
    # With custom scoring weights
    python validate_graph.py --alpha 0.5 --beta 0.25 --gamma 0.25
    
    # Custom target thresholds
    python validate_graph.py --target-spearman 0.8 --target-f1 0.85
    
    # Filter by component type
    python validate_graph.py --component-types Application Broker
    
    # Export results
    python validate_graph.py --output-dir results/ --format json html csv

Target Metrics (defaults):
    - Spearman Correlation: ≥ 0.70
    - F1-Score: ≥ 0.90
    - Precision: ≥ 0.80
    - Recall: ≥ 0.80
    - Top-5 Overlap: ≥ 60%
    - Top-10 Overlap: ≥ 70%
        """
    )
    
    # Neo4j options
    neo4j_group = parser.add_argument_group('Neo4j Connection')
    neo4j_group.add_argument('--neo4j-uri', default='bolt://localhost:7687',
                             help='Neo4j URI (default: bolt://localhost:7687)')
    neo4j_group.add_argument('--neo4j-user', default='neo4j',
                             help='Neo4j username (default: neo4j)')
    neo4j_group.add_argument('--neo4j-password', default='password',
                             help='Neo4j password (default: password)')
    neo4j_group.add_argument('--neo4j-database', default='neo4j',
                             help='Neo4j database name (default: neo4j)')
    
    # Analysis weights
    weight_group = parser.add_argument_group('Scoring Weights')
    weight_group.add_argument('--alpha', type=float, default=0.4,
                              help='Betweenness centrality weight (default: 0.4)')
    weight_group.add_argument('--beta', type=float, default=0.3,
                              help='Articulation point weight (default: 0.3)')
    weight_group.add_argument('--gamma', type=float, default=0.3,
                              help='Impact score weight (default: 0.3)')
    
    # Target thresholds
    target_group = parser.add_argument_group('Target Thresholds')
    target_group.add_argument('--target-spearman', type=float, default=0.7,
                              help='Target Spearman correlation (default: 0.7)')
    target_group.add_argument('--target-f1', type=float, default=0.9,
                              help='Target F1-score (default: 0.9)')
    target_group.add_argument('--target-precision', type=float, default=0.8,
                              help='Target precision (default: 0.8)')
    target_group.add_argument('--target-recall', type=float, default=0.8,
                              help='Target recall (default: 0.8)')
    target_group.add_argument('--target-top5', type=float, default=0.6,
                              help='Target top-5 overlap (default: 0.6)')
    target_group.add_argument('--target-top10', type=float, default=0.7,
                              help='Target top-10 overlap (default: 0.7)')
    target_group.add_argument('--critical-threshold', type=float, default=0.5,
                              help='Impact threshold for "critical" classification (default: 0.5)')
    
    # Validation options
    val_group = parser.add_argument_group('Validation Options')
    val_group.add_argument('--component-types', nargs='+',
                           help='Filter validation to specific types (e.g., Application Broker)')
    val_group.add_argument('--cascade', action='store_true',
                           help='Enable cascade in simulation')
    val_group.add_argument('--seed', type=int, default=42,
                           help='Random seed for reproducibility (default: 42)')
    
    # Output options
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output-dir', '-o',
                              help='Output directory for exports')
    output_group.add_argument('--format', '-f', nargs='+',
                              choices=['json', 'html', 'csv'],
                              default=['json', 'html'],
                              help='Export formats (default: json html)')
    
    # General options
    parser.add_argument('--json-output', action='store_true',
                        help='Print JSON results to stdout')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')
    
    args = parser.parse_args()
    
    return run_validation(args)


if __name__ == '__main__':
    sys.exit(main())