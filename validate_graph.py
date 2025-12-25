#!/usr/bin/env python3
"""
Graph Validation CLI
=====================

Validates graph-based criticality analysis by comparing predicted scores
from topological analysis against actual impact scores from failure simulation.

Validation Pipeline:
1. Load graph from Neo4j
2. Run analysis to get predicted criticality scores
3. Run failure simulation to get actual impact scores
4. Compare using correlation, classification, and ranking metrics

Target Metrics:
- Spearman Correlation: ≥ 0.70
- F1-Score: ≥ 0.90
- Precision/Recall: ≥ 0.80
- Top-5 Overlap: ≥ 60%
- Top-10 Overlap: ≥ 70%

Usage:
    # Basic validation
    python validate_graph.py
    
    # With specific analysis method
    python validate_graph.py --method composite
    
    # Compare multiple methods
    python validate_graph.py --compare
    
    # With bootstrap confidence intervals
    python validate_graph.py --bootstrap
    
    # Export results
    python validate_graph.py --output results/ --format json

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from src.validation import (
    IntegratedValidator,
    IntegratedValidationResult,
    ValidationTargets,
    ValidationStatus,
    MetricStatus
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
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    @classmethod
    def disable(cls):
        for attr in ['HEADER', 'BLUE', 'CYAN', 'GREEN', 'YELLOW', 'RED',
                     'ENDC', 'BOLD', 'DIM']:
            setattr(cls, attr, '')


def print_header(text: str):
    """Print formatted header"""
    width = 70
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*width}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^{width}}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*width}{Colors.ENDC}")


def print_section(text: str):
    """Print section header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.DIM}{'-'*50}{Colors.ENDC}")


def print_success(text: str):
    print(f"{Colors.GREEN}✓{Colors.ENDC} {text}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠{Colors.ENDC} {text}")


def print_error(text: str):
    print(f"{Colors.RED}✗{Colors.ENDC} {text}")


def print_info(text: str):
    print(f"{Colors.BLUE}ℹ{Colors.ENDC} {text}")


def status_color(status: ValidationStatus) -> str:
    """Get color for validation status"""
    colors = {
        ValidationStatus.PASSED: Colors.GREEN,
        ValidationStatus.PARTIAL: Colors.YELLOW,
        ValidationStatus.FAILED: Colors.RED,
        ValidationStatus.INSUFFICIENT_DATA: Colors.DIM
    }
    return colors.get(status, '')


def metric_symbol(status: MetricStatus) -> str:
    """Get symbol for metric status"""
    symbols = {
        MetricStatus.MET: f"{Colors.GREEN}✓{Colors.ENDC}",
        MetricStatus.BORDERLINE: f"{Colors.YELLOW}○{Colors.ENDC}",
        MetricStatus.NOT_MET: f"{Colors.RED}✗{Colors.ENDC}"
    }
    return symbols.get(status, '?')


# ============================================================================
# Output Functions
# ============================================================================

def print_validation_result(result: IntegratedValidationResult, verbose: bool = False):
    """Print validation result"""
    v = result.validation
    
    # Status
    sc = status_color(v.status)
    print_section("VALIDATION STATUS")
    print(f"  Status: {sc}{Colors.BOLD}{v.status.value.upper()}{Colors.ENDC}")
    print(f"  Components Validated: {v.total_components}")
    print(f"  Analysis Method: {result.analysis_method}")
    
    # Correlation metrics
    print_section("CORRELATION METRICS")
    corr = v.correlation
    
    spearman_color = Colors.GREEN if corr.spearman_coefficient >= 0.7 else Colors.YELLOW if corr.spearman_coefficient >= 0.5 else Colors.RED
    print(f"  Spearman ρ:  {spearman_color}{corr.spearman_coefficient:.4f}{Colors.ENDC} (p={corr.spearman_p_value:.4f})")
    print(f"  Pearson r:   {corr.pearson_coefficient:.4f} (p={corr.pearson_p_value:.4f})")
    print(f"  Kendall τ:   {corr.kendall_tau:.4f}")
    
    if corr.spearman_p_value < 0.05:
        print(f"  {Colors.GREEN}✓ Statistically significant (p < 0.05){Colors.ENDC}")
    else:
        print(f"  {Colors.YELLOW}⚠ Not statistically significant{Colors.ENDC}")
    
    # Classification metrics
    print_section("CLASSIFICATION METRICS")
    cls = v.classification
    
    print(f"  Confusion Matrix:")
    print(f"    TP: {cls.true_positives:3}  |  FP: {cls.false_positives:3}")
    print(f"    FN: {cls.false_negatives:3}  |  TN: {cls.true_negatives:3}")
    print()
    
    prec_color = Colors.GREEN if cls.precision >= 0.8 else Colors.YELLOW if cls.precision >= 0.6 else Colors.RED
    rec_color = Colors.GREEN if cls.recall >= 0.8 else Colors.YELLOW if cls.recall >= 0.6 else Colors.RED
    f1_color = Colors.GREEN if cls.f1_score >= 0.9 else Colors.YELLOW if cls.f1_score >= 0.7 else Colors.RED
    
    print(f"  Precision:  {prec_color}{cls.precision:.4f}{Colors.ENDC}")
    print(f"  Recall:     {rec_color}{cls.recall:.4f}{Colors.ENDC}")
    print(f"  F1-Score:   {f1_color}{cls.f1_score:.4f}{Colors.ENDC}")
    print(f"  Accuracy:   {cls.accuracy:.4f}")
    
    # Ranking metrics
    print_section("RANKING METRICS")
    rank = v.ranking
    
    for k, overlap in sorted(rank.top_k_overlap.items()):
        target = 0.6 if k == 5 else 0.7 if k == 10 else 0.5
        overlap_color = Colors.GREEN if overlap >= target else Colors.YELLOW if overlap >= target * 0.8 else Colors.RED
        bar_len = int(overlap * 30)
        bar = '█' * bar_len + '░' * (30 - bar_len)
        print(f"  Top-{k:2} Overlap: {overlap_color}{bar}{Colors.ENDC} {overlap:.1%}")
    
    print(f"\n  Mean Rank Difference: {rank.mean_rank_difference:.1f}")
    print(f"  Max Rank Difference:  {rank.max_rank_difference}")
    
    # Target achievement
    print_section("TARGET ACHIEVEMENT")
    
    met = 0
    total = len(v.achieved)
    
    for metric_name, (value, status) in v.achieved.items():
        target_val = getattr(v.targets, metric_name, 0)
        symbol = metric_symbol(status)
        
        if status == MetricStatus.MET:
            met += 1
        
        # Format metric name nicely
        display_name = metric_name.replace('_', ' ').title()
        print(f"  {symbol} {display_name:20} {value:.4f} (target: {target_val})")
    
    print(f"\n  Overall: {met}/{total} targets met")
    
    # Bootstrap results
    if v.bootstrap_results:
        print_section("BOOTSTRAP CONFIDENCE INTERVALS")
        for br in v.bootstrap_results:
            print(f"  {br.metric_name}:")
            print(f"    Point estimate: {br.point_estimate:.4f}")
            print(f"    {br.confidence_level:.0%} CI: [{br.ci_lower:.4f}, {br.ci_upper:.4f}]")
            print(f"    Std error: {br.std_error:.4f}")
    
    # Misclassified components
    if verbose and (v.false_positives or v.false_negatives):
        print_section("MISCLASSIFIED COMPONENTS")
        
        if v.false_positives:
            print(f"\n  {Colors.YELLOW}False Positives (predicted critical but not):{Colors.ENDC}")
            for comp in v.false_positives[:5]:
                print(f"    • {comp}")
            if len(v.false_positives) > 5:
                print(f"    ... and {len(v.false_positives) - 5} more")
        
        if v.false_negatives:
            print(f"\n  {Colors.RED}False Negatives (missed critical components):{Colors.ENDC}")
            for comp in v.false_negatives[:5]:
                print(f"    • {comp}")
            if len(v.false_negatives) > 5:
                print(f"    ... and {len(v.false_negatives) - 5} more")
    
    # Component details
    if verbose and v.component_validations:
        print_section("TOP COMPONENTS (by actual impact)")
        
        print(f"\n  {'Component':<20} {'Pred':>8} {'Actual':>8} {'P-Rank':>7} {'A-Rank':>7} {'Diff':>5}")
        print(f"  {'-'*60}")
        
        for cv in v.component_validations[:10]:
            correct = "✓" if cv.classification_correct else "✗"
            print(f"  {cv.component_id:<20} {cv.predicted_score:>8.4f} {cv.actual_impact:>8.4f} "
                  f"{cv.predicted_rank:>7} {cv.actual_rank:>7} {cv.rank_difference:>4} {correct}")
    
    # Timing
    print_section("TIMING")
    print(f"  Analysis:    {result.analysis_time_ms:>8.0f} ms")
    print(f"  Simulation:  {result.simulation_time_ms:>8.0f} ms")
    print(f"  Validation:  {result.validation_time_ms:>8.0f} ms")
    print(f"  {Colors.BOLD}Total:       {result.total_time_ms:>8.0f} ms{Colors.ENDC}")


def print_comparison(results: Dict[str, IntegratedValidationResult]):
    """Print method comparison"""
    print_section("METHOD COMPARISON")
    
    # Header
    print(f"\n  {'Method':<15} {'Spearman':>10} {'F1':>10} {'Prec':>10} {'Recall':>10} {'Top-5':>10} {'Status':>10}")
    print(f"  {'-'*75}")
    
    for method, result in results.items():
        v = result.validation
        sc = status_color(v.status)
        
        print(f"  {method:<15} "
              f"{v.correlation.spearman_coefficient:>10.4f} "
              f"{v.classification.f1_score:>10.4f} "
              f"{v.classification.precision:>10.4f} "
              f"{v.classification.recall:>10.4f} "
              f"{v.ranking.top_k_overlap.get(5, 0):>10.4f} "
              f"{sc}{v.status.value:>10}{Colors.ENDC}")
    
    # Find best method
    best_method = max(results.keys(), 
                      key=lambda m: results[m].validation.correlation.spearman_coefficient)
    best_result = results[best_method]
    
    print(f"\n  {Colors.GREEN}Best Method: {best_method}{Colors.ENDC}")
    print(f"  Spearman: {best_result.validation.correlation.spearman_coefficient:.4f}")


def export_results(result: IntegratedValidationResult, output_dir: str, formats: List[str]):
    """Export results to files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if 'json' in formats:
        json_file = output_path / f"validation_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print_success(f"JSON exported: {json_file}")
    
    if 'csv' in formats:
        csv_file = output_path / f"validation_components_{timestamp}.csv"
        with open(csv_file, 'w') as f:
            f.write("component_id,component_type,predicted_score,actual_impact,"
                   "predicted_rank,actual_rank,rank_diff,correct\n")
            for cv in result.validation.component_validations:
                f.write(f"{cv.component_id},{cv.component_type},"
                       f"{cv.predicted_score:.6f},{cv.actual_impact:.6f},"
                       f"{cv.predicted_rank},{cv.actual_rank},"
                       f"{cv.rank_difference},{cv.classification_correct}\n")
        print_success(f"CSV exported: {csv_file}")
    
    if 'summary' in formats:
        summary_file = output_path / f"validation_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            v = result.validation
            f.write(f"Validation Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Timestamp: {result.timestamp}\n")
            f.write(f"Status: {v.status.value.upper()}\n")
            f.write(f"Components: {v.total_components}\n\n")
            
            f.write(f"Correlation:\n")
            f.write(f"  Spearman: {v.correlation.spearman_coefficient:.4f}\n")
            f.write(f"  Pearson:  {v.correlation.pearson_coefficient:.4f}\n\n")
            
            f.write(f"Classification:\n")
            f.write(f"  Precision: {v.classification.precision:.4f}\n")
            f.write(f"  Recall:    {v.classification.recall:.4f}\n")
            f.write(f"  F1-Score:  {v.classification.f1_score:.4f}\n\n")
            
            f.write(f"Ranking:\n")
            for k, overlap in sorted(v.ranking.top_k_overlap.items()):
                f.write(f"  Top-{k} Overlap: {overlap:.1%}\n")
        
        print_success(f"Summary exported: {summary_file}")


# ============================================================================
# CLI
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description='Validate graph-based criticality analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic validation with composite analysis
    python validate_graph.py
    
    # Use specific analysis method
    python validate_graph.py --method betweenness
    
    # Compare all methods
    python validate_graph.py --compare
    
    # With bootstrap confidence intervals
    python validate_graph.py --bootstrap
    
    # Custom targets
    python validate_graph.py --spearman-target 0.75 --f1-target 0.85
    
    # Export results
    python validate_graph.py --output results/ --format json csv summary
        """
    )
    
    # Neo4j connection
    conn_group = parser.add_argument_group('Neo4j Connection')
    conn_group.add_argument('--uri', default='bolt://localhost:7687',
                           help='Neo4j URI (default: bolt://localhost:7687)')
    conn_group.add_argument('--user', '-u', default='neo4j',
                           help='Neo4j username (default: neo4j)')
    conn_group.add_argument('--password', '-p', default='password',
                           help='Neo4j password (default: password)')
    conn_group.add_argument('--database', '-d', default='neo4j',
                           help='Neo4j database (default: neo4j)')
    
    # Validation options
    val_group = parser.add_argument_group('Validation Options')
    val_group.add_argument('--method', '-m',
                          choices=['composite', 'betweenness', 'pagerank', 'degree'],
                          default='composite',
                          help='Analysis method (default: composite)')
    val_group.add_argument('--compare', action='store_true',
                          help='Compare all analysis methods')
    val_group.add_argument('--cascade', action='store_true', default=True,
                          help='Enable cascade in simulation (default: True)')
    val_group.add_argument('--no-cascade', action='store_true',
                          help='Disable cascade in simulation')
    val_group.add_argument('--bootstrap', action='store_true',
                          help='Run bootstrap confidence intervals')
    val_group.add_argument('--bootstrap-iterations', type=int, default=1000,
                          help='Number of bootstrap iterations (default: 1000)')
    
    # Target customization
    target_group = parser.add_argument_group('Target Metrics')
    target_group.add_argument('--spearman-target', type=float, default=0.70,
                             help='Spearman correlation target (default: 0.70)')
    target_group.add_argument('--f1-target', type=float, default=0.90,
                             help='F1-score target (default: 0.90)')
    target_group.add_argument('--precision-target', type=float, default=0.80,
                             help='Precision target (default: 0.80)')
    target_group.add_argument('--recall-target', type=float, default=0.80,
                             help='Recall target (default: 0.80)')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output', '-o',
                             help='Output directory for results')
    output_group.add_argument('--format', nargs='+',
                             choices=['json', 'csv', 'summary'],
                             default=['json'],
                             help='Output formats')
    
    # Display options
    display_group = parser.add_argument_group('Display Options')
    display_group.add_argument('--verbose', '-v', action='store_true',
                              help='Verbose output with component details')
    display_group.add_argument('--quiet', '-q', action='store_true',
                              help='Minimal output')
    display_group.add_argument('--no-color', action='store_true',
                              help='Disable colored output')
    display_group.add_argument('--json-stdout', action='store_true',
                              help='Output JSON to stdout')
    display_group.add_argument('--seed', type=int,
                              help='Random seed for reproducibility')
    display_group.add_argument('--debug', action='store_true',
                              help='Enable debug logging')
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING)
    
    # Handle colors
    if not sys.stdout.isatty() or args.no_color:
        Colors.disable()
    
    # Print header
    if not args.quiet:
        print_header("GRAPH VALIDATION")
        print(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Neo4j: {args.uri}")
        if args.compare:
            print(f"  Mode: Method comparison")
        else:
            print(f"  Method: {args.method}")
    
    try:
        # Create custom targets
        targets = ValidationTargets(
            spearman_correlation=args.spearman_target,
            f1_score=args.f1_target,
            precision=args.precision_target,
            recall=args.recall_target
        )
        
        # Create validator
        validator = IntegratedValidator(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
            targets=targets
        )
        
        enable_cascade = args.cascade and not args.no_cascade
        
        if args.compare:
            # Compare methods
            if not args.quiet:
                print_section("COMPARING ANALYSIS METHODS")
                print_info("Running validation for all methods...")
            
            results = validator.compare_methods(
                enable_cascade=enable_cascade,
                seed=args.seed
            )
            
            if args.json_stdout:
                output = {method: r.to_dict() for method, r in results.items()}
                print(json.dumps(output, indent=2))
            else:
                print_comparison(results)
                
                # Also print detailed results for each
                if args.verbose:
                    for method, result in results.items():
                        print_header(f"METHOD: {method.upper()}")
                        print_validation_result(result, verbose=True)
            
            # Export best result
            if args.output:
                best_method = max(results.keys(),
                                  key=lambda m: results[m].validation.correlation.spearman_coefficient)
                if not args.quiet:
                    print_section("EXPORTING RESULTS")
                export_results(results[best_method], args.output, args.format)
            
            # Exit code based on best result
            best_result = max(results.values(),
                             key=lambda r: r.validation.correlation.spearman_coefficient)
            if best_result.validation.status == ValidationStatus.PASSED:
                sys.exit(0)
            elif best_result.validation.status == ValidationStatus.PARTIAL:
                sys.exit(1)
            else:
                sys.exit(2)
        
        else:
            # Single method validation
            if not args.quiet:
                print_section("RUNNING VALIDATION")
                print_info(f"Analysis method: {args.method}")
                print_info(f"Cascade: {'enabled' if enable_cascade else 'disabled'}")
            
            result = validator.run_validation(
                analysis_method=args.method,
                enable_cascade=enable_cascade,
                bootstrap=args.bootstrap,
                seed=args.seed
            )
            
            if args.json_stdout:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print_validation_result(result, verbose=args.verbose)
            
            # Export if requested
            if args.output:
                if not args.quiet:
                    print_section("EXPORTING RESULTS")
                export_results(result, args.output, args.format)
            
            if not args.quiet:
                sc = status_color(result.validation.status)
                print(f"\n{sc}{Colors.BOLD}Validation {result.validation.status.value.upper()}{Colors.ENDC}")
            
            # Exit code based on status
            if result.validation.status == ValidationStatus.PASSED:
                sys.exit(0)
            elif result.validation.status == ValidationStatus.PARTIAL:
                sys.exit(1)
            else:
                sys.exit(2)
    
    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        print_warning("Install with: pip install neo4j")
        sys.exit(1)
    
    except Exception as e:
        print_error(f"Validation failed: {e}")
        if args.debug or args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()