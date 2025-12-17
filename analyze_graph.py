#!/usr/bin/env python3
"""
Pub-Sub System Graph Analyzer
=============================

Main entry point for analyzing distributed publish-subscribe systems.
Derives DEPENDS_ON relationships directly from base pub-sub relationships
without storing them in input files.

Usage:
    # Load from Neo4j database
    python analyze_graph.py --neo4j-uri bolt://localhost:7687
    
    # With custom weights
    python analyze_graph.py --alpha 0.5 --beta 0.25 --gamma 0.25
    
    # Export to different formats
    python analyze_graph.py --output-dir results/ --format json html
    
    # Verbose output
    python analyze_graph.py --verbose

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from src.analysis import (
    GraphAnalyzer,
    AnalysisResult,
    DependencyType,
    CriticalityLevel,
    NEO4J_AVAILABLE,
)


# ============================================================================
# Colors for Terminal Output
# ============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    
    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output)"""
        cls.HEADER = ''
        cls.BLUE = ''
        cls.CYAN = ''
        cls.GREEN = ''
        cls.WARNING = ''
        cls.FAIL = ''
        cls.ENDC = ''
        cls.BOLD = ''


# ============================================================================
# Output Formatters
# ============================================================================

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")


def print_section(text: str):
    """Print section header"""
    print(f"\n{Colors.CYAN}▶ {text}{Colors.ENDC}")


def print_metric(name: str, value: Any, indent: int = 2):
    """Print a metric"""
    spaces = ' ' * indent
    print(f"{spaces}{Colors.BLUE}{name}:{Colors.ENDC} {value}")


def print_critical(text: str):
    """Print critical item"""
    print(f"  {Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning"""
    print(f"  {Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_success(text: str):
    """Print success"""
    print(f"  {Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info"""
    print(f"  {Colors.BLUE}• {text}{Colors.ENDC}")


# ============================================================================
# Export Functions
# ============================================================================

def export_json(result: AnalysisResult, output_path: Path):
    """Export results to JSON"""
    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    print_success(f"Exported JSON: {output_path}")


def export_html(result: AnalysisResult, output_path: Path):
    """Export results to HTML report"""
    data = result.to_dict()
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph Analysis Report</title>
    <style>
        :root {{
            --primary: #2563eb;
            --critical: #dc2626;
            --high: #ea580c;
            --medium: #ca8a04;
            --low: #16a34a;
            --bg: #f8fafc;
            --card: #ffffff;
            --border: #e2e8f0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            margin: 0;
            padding: 20px;
            color: #1e293b;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: var(--primary); border-bottom: 3px solid var(--primary); padding-bottom: 10px; }}
        h2 {{ color: #475569; margin-top: 30px; }}
        .card {{
            background: var(--card);
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .metric {{
            background: var(--bg);
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: var(--primary); }}
        .metric-label {{ color: #64748b; font-size: 0.9em; }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 500;
        }}
        .badge-critical {{ background: #fef2f2; color: var(--critical); }}
        .badge-high {{ background: #fff7ed; color: var(--high); }}
        .badge-medium {{ background: #fefce8; color: var(--medium); }}
        .badge-low {{ background: #f0fdf4; color: var(--low); }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{ background: var(--bg); font-weight: 600; }}
        tr:hover {{ background: var(--bg); }}
        .recommendation {{
            background: #f0f9ff;
            border-left: 4px solid var(--primary);
            padding: 15px;
            margin: 10px 0;
        }}
        .timestamp {{ color: #94a3b8; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Graph Analysis Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Graph Summary</h2>
        <div class="card">
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{data['graph_summary']['total_nodes']}</div>
                    <div class="metric-label">Total Nodes</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{data['graph_summary']['total_edges']}</div>
                    <div class="metric-label">DEPENDS_ON Edges</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{data['graph_summary']['density']}</div>
                    <div class="metric-label">Density</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{'Connected' if data['graph_summary']['is_connected'] else 'Disconnected'}</div>
                    <div class="metric-label">Connectivity</div>
                </div>
            </div>
        </div>
        
        <h2>DEPENDS_ON Relationships</h2>
        <div class="card">
            <div class="metrics">
                {''.join(f'<div class="metric"><div class="metric-value">{count}</div><div class="metric-label">{dtype}</div></div>' 
                         for dtype, count in data['depends_on']['by_type'].items())}
            </div>
        </div>
        
        <h2>Criticality Analysis</h2>
        <div class="card">
            <div class="metrics">
                {''.join(f'<div class="metric"><div class="metric-value">{count}</div><div class="metric-label">{level.upper()}</div></div>' 
                         for level, count in data['criticality']['by_level'].items())}
            </div>
            
            <h3>Top Critical Components</h3>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Type</th>
                    <th>Score</th>
                    <th>Level</th>
                    <th>Reasons</th>
                </tr>
                {''.join(f'''<tr>
                    <td>{score['node_id']}</td>
                    <td>{score['type']}</td>
                    <td>{score['composite_score']:.4f}</td>
                    <td><span class="badge badge-{score['level']}">{score['level'].upper()}</span></td>
                    <td>
                        Betweenness: {score['betweenness']:.4f}<br>
                        Articulation Point: {'Yes' if score['is_articulation_point'] else 'No'}<br>
                        Impact Score: {score['impact_score']:.4f}
                    </td>
                </tr>''' for score in data['criticality']['scores'][:10])}
        </div>
        
        <h2>Structural Analysis</h2>
        <div class="card">
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{data['structural']['articulation_point_count']}</div>
                    <div class="metric-label">Articulation Points</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{data['structural']['bridge_count']}</div>
                    <div class="metric-label">Bridges</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{data['structural']['weakly_connected_components']}</div>
                    <div class="metric-label">Components</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{'Yes' if data['structural']['has_cycles'] else 'No'}</div>
                    <div class="metric-label">Circular Dependencies</div>
                </div>
            </div>
        </div>
        
        <h2>Recommendations</h2>
        <div class="card">
            {''.join(f'<div class="recommendation">{rec}</div>' for rec in data['recommendations'])}
        </div>
    </div>
</body>
</html>"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    print_success(f"Exported HTML: {output_path}")


def export_csv(result: AnalysisResult, output_dir: Path):
    """Export criticality scores to CSV"""
    data = result.to_dict()
    
    # Criticality scores CSV
    csv_path = output_dir / 'criticality_scores.csv'
    with open(csv_path, 'w') as f:
        f.write('node_id,type,betweenness,is_articulation_point,impact_score,composite_score,level\n')
        for s in data['criticality']['scores']:
            f.write(f"{s['node_id']},{s['type']},{s['betweenness']},{s['is_articulation_point']},"
                    f"{s['impact_score']},{s['composite_score']},{s['level']}\n")
    print_success(f"Exported CSV: {csv_path}")
    
    # Dependencies CSV
    deps_path = output_dir / 'depends_on_edges.csv'
    with open(deps_path, 'w') as f:
        f.write('source,target,type,weight,via_topics,via_apps\n')
        for e in data['depends_on']['edges']:
            topics = ';'.join(e['via_topics']) if e['via_topics'] else ''
            apps = ';'.join(e['via_apps']) if e['via_apps'] else ''
            f.write(f"{e['source']},{e['target']},{e['type']},{e['weight']},{topics},{apps}\n")
    print_success(f"Exported CSV: {deps_path}")


# ============================================================================
# Main Analysis Function
# ============================================================================

def run_analysis(args) -> int:
    """Run the analysis pipeline"""
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger('analyze_graph')
    
    # Check if output is TTY for colors
    if not sys.stdout.isatty() or args.no_color:
        Colors.disable()
    
    # Print header
    if not args.quiet:
        print_header("PUB-SUB SYSTEM GRAPH ANALYZER")
        print(f"\n  Source: Neo4j at {args.neo4j_uri}")
        print(f"  Weights: α={args.alpha}, β={args.beta}, γ={args.gamma}")
    
    try:
        # Create analyzer
        analyzer = GraphAnalyzer(
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma
        )
        
        # Load data
        if not args.quiet:
            print_section("Loading Data")
        
        if not NEO4J_AVAILABLE:
            logger.error("Neo4j driver not installed. Install with: pip install neo4j")
            return 1
        
        analyzer.load_from_neo4j(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            database=args.neo4j_database
        )
        if not args.quiet:
            print_success(f"Loaded from Neo4j: {args.neo4j_uri}")
        
        # Run analysis
        if not args.quiet:
            print_section("Running Analysis")
        
        start_time = datetime.now()
        result = analyzer.analyze()
        duration = (datetime.now() - start_time).total_seconds()
        
        if not args.quiet:
            print_success(f"Analysis completed in {duration:.2f}s")
        
        # Print summary
        if not args.quiet:
            print_summary(result)
        
        # Export results
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if not args.quiet:
                print_section("Exporting Results")
            
            for fmt in args.format:
                if fmt == 'json':
                    export_json(result, output_dir / 'analysis_results.json')
                elif fmt == 'html':
                    export_html(result, output_dir / 'analysis_report.html')
                elif fmt == 'csv':
                    export_csv(result, output_dir)
        else:
            # Print JSON to stdout if no output dir
            if args.json_output:
                print(json.dumps(result.to_dict(), indent=2, default=str))
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in input file: {e}")
        return 1
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def print_summary(result: AnalysisResult):
    """Print analysis summary to terminal"""
    data = result.to_dict()
    
    # Graph Summary
    print_section("Graph Summary")
    summary = data['graph_summary']
    print_metric("Total Nodes", summary['total_nodes'])
    print_metric("Total DEPENDS_ON Edges", summary['total_edges'])
    print_metric("Density", f"{summary['density']:.4f}")
    print_metric("Connected", "Yes" if summary['is_connected'] else "No")
    
    print("\n  Nodes by Type:")
    for ntype, count in summary['nodes_by_type'].items():
        print_info(f"{ntype}: {count}")
    
    # DEPENDS_ON Summary
    print_section("DEPENDS_ON Relationships")
    deps = data['depends_on']
    print_metric("Total Dependencies", deps['total'])
    for dtype, count in deps['by_type'].items():
        print_info(f"{dtype}: {count}")
    
    # Criticality Summary
    print_section("Criticality Analysis")
    crit = data['criticality']
    for level, count in crit['by_level'].items():
        if level == 'critical':
            print_critical(f"{level.upper()}: {count}")
        elif level == 'high':
            print_warning(f"{level.upper()}: {count}")
        else:
            print_info(f"{level.upper()}: {count}")
    
    # Top critical components
    print("\n  Top 5 Critical Components:")
    for i, score in enumerate(crit['scores'][:5], 1):
        level_color = Colors.FAIL if score['level'] == 'critical' else Colors.WARNING
        print(f"    {i}. {level_color}{score['node_id']}{Colors.ENDC} "
              f"({score['type']}) - Score: {score['composite_score']:.4f}")
    
    # Structural
    print_section("Structural Analysis")
    struct = data['structural']
    
    if struct['articulation_point_count'] > 0:
        print_critical(f"Articulation Points (SPOFs): {struct['articulation_point_count']}")
    else:
        print_success("No articulation points found")
    
    if struct['bridge_count'] > 0:
        print_warning(f"Bridge Edges: {struct['bridge_count']}")
    else:
        print_success("No bridge edges found")
    
    if struct['has_cycles']:
        print_warning(f"Circular dependencies detected: {len(struct['cycles'])} cycles")
    else:
        print_success("No circular dependencies")
    
    # Recommendations
    if data['recommendations']:
        print_section("Recommendations")
        for rec in data['recommendations']:
            print_warning(rec)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Analyze pub-sub system graphs using DEPENDS_ON relationships',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Load from Neo4j database
    python analyze_graph.py --neo4j-uri bolt://localhost:7687
    
    # Export to multiple formats
    python analyze_graph.py --output-dir results/ --format json html csv
    
    # Custom criticality weights
    python analyze_graph.py --alpha 0.5 --beta 0.25 --gamma 0.25
    
    # JSON output to stdout
    python analyze_graph.py --json-output --quiet
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
    
    # Output options
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output-dir', '-o',
                              help='Output directory for exports')
    output_group.add_argument('--format', '-f', nargs='+', 
                              choices=['json', 'html', 'csv'],
                              default=['json', 'html'],
                              help='Export formats (default: json html)')
    
    # Criticality weights
    weights_group = parser.add_argument_group('Criticality Weights')
    weights_group.add_argument('--alpha', type=float, default=0.4,
                               help='Weight for betweenness centrality (default: 0.4)')
    weights_group.add_argument('--beta', type=float, default=0.3,
                               help='Weight for articulation point indicator (default: 0.3)')
    weights_group.add_argument('--gamma', type=float, default=0.3,
                               help='Weight for impact score (default: 0.3)')
    
    # Output options
    parser.add_argument('--json-output', action='store_true',
                        help='Print JSON results to stdout')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')
    
    args = parser.parse_args()
    
    # Validate weights
    if abs(args.alpha + args.beta + args.gamma - 1.0) > 0.001:
        parser.error("Weights (alpha + beta + gamma) must sum to 1.0")
    
    return run_analysis(args)


if __name__ == '__main__':
    sys.exit(main())