#!/usr/bin/env python3
"""
Analyze Graph Script

Command-line interface for comprehensive pub-sub system analysis using
the modular architecture.

Features:
- Load graphs from JSON files or Neo4j database
- Multi-metric centrality analysis
- QoS-aware criticality scoring
- Failure simulation and impact assessment
- Multi-layer dependency analysis
- Multiple output formats (JSON, CSV, HTML)
- Configurable analysis parameters

Architecture:
  GraphBuilder → GraphModel → GraphExporter → NetworkX
                                                  ↓
                                         AnalysisOrchestrator
                                                  ↓
                                           Analysis Results

Usage:
    # Analyze from JSON
    python analyze_graph.py --input system.json
    
    # Analyze from Neo4j
    python analyze_graph.py --neo4j --uri bolt://localhost:7687 \\
        --user neo4j --password password
    
    # With failure simulation and custom weights
    python analyze_graph.py --input system.json --simulate \\
        --alpha 0.5 --beta 0.3 --gamma 0.2
    
    # Export to multiple formats
    python analyze_graph.py --input system.json \\
        --export-json --export-csv --export-html
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Tuple
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.graph_builder import GraphBuilder
from src.core.graph_exporter import GraphExporter
from src.core.graph_model import GraphModel
from src.orchestration.analysis_orchestrator import AnalysisOrchestrator


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging
    
    Args:
        verbose: Enable DEBUG level logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('analysis.log', mode='w')
        ]
    )


def load_graph_from_json(filepath: str) -> Tuple[GraphModel, 'nx.DiGraph']:
    """
    Load graph from JSON file
    
    Args:
        filepath: Path to JSON configuration file
    
    Returns:
        Tuple of (GraphModel, NetworkX DiGraph)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading graph from JSON: {filepath}")
    
    # Build GraphModel from JSON
    builder = GraphBuilder()
    model = builder.build_from_json(filepath)
    
    # Convert to NetworkX for analysis
    exporter = GraphExporter()
    graph = exporter.export_to_networkx(model)
    
    summary = model.summary()
    logger.info(f"✓ Loaded graph: {summary['total_nodes']} nodes, {summary['total_edges']} edges")
    
    return model, graph


def load_graph_from_neo4j(
    uri: str,
    username: str,
    password: str,
    database: str = "neo4j"
) -> Tuple[GraphModel, 'nx.DiGraph']:
    """
    Load graph from Neo4j database
    
    Args:
        uri: Neo4j connection URI
        username: Database username
        password: Database password
        database: Database name
    
    Returns:
        Tuple of (GraphModel, NetworkX DiGraph)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading graph from Neo4j: {uri}")
    
    # Build GraphModel from Neo4j
    builder = GraphBuilder()
    model = builder.build_from_neo4j(
        uri=uri,
        auth=(username, password),
        database=database
    )
    
    # Convert to NetworkX for analysis
    exporter = GraphExporter()
    graph = exporter.export_to_networkx(model)
    
    summary = model.summary()
    logger.info(f"✓ Loaded graph: {summary['total_nodes']} nodes, {summary['total_edges']} edges")
    
    return model, graph


def save_analysis_results(
    results: dict,
    output_dir: Path,
    export_json: bool = True,
    export_csv: bool = False,
    export_html: bool = False
) -> None:
    """
    Save analysis results to various formats
    
    Args:
        results: Analysis results dictionary
        output_dir: Output directory
        export_json: Export as JSON
        export_csv: Export criticality scores as CSV
        export_html: Export as HTML report
    """
    logger = logging.getLogger(__name__)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON export
    if export_json:
        json_path = output_dir / 'analysis_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"✓ JSON results: {json_path}")
        
        # Also save criticality scores separately
        if 'criticality_scores' in results and 'scores' in results['criticality_scores']:
            scores_path = output_dir / 'criticality_scores.json'
            with open(scores_path, 'w') as f:
                json.dump(results['criticality_scores']['scores'], f, indent=2, default=str)
            logger.info(f"✓ Criticality scores: {scores_path}")
    
    # CSV export
    if export_csv:
        try:
            import pandas as pd
            
            if 'criticality_scores' in results and 'scores' in results['criticality_scores']:
                # Convert scores to DataFrame
                scores_data = []
                for component, score in results['criticality_scores']['scores'].items():
                    row = {'component': component}
                    row.update(score)
                    scores_data.append(row)
                
                df = pd.DataFrame(scores_data)
                csv_path = output_dir / 'criticality_scores.csv'
                df.to_csv(csv_path, index=False)
                logger.info(f"✓ CSV export: {csv_path}")
        except ImportError:
            logger.warning("pandas not available, skipping CSV export")
    
    # HTML export
    if export_html:
        html_path = output_dir / 'analysis_report.html'
        generate_html_report(results, html_path)
        logger.info(f"✓ HTML report: {html_path}")


def generate_html_report(results: dict, output_path: Path) -> None:
    """
    Generate HTML report from analysis results
    
    Args:
        results: Analysis results dictionary
        output_path: Path to save HTML file
    """
    from datetime import datetime
    
    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pub-Sub System Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .metric-card .value {{
            font-size: 28px;
            font-weight: bold;
            color: #3498db;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .critical {{ color: #e74c3c; font-weight: bold; }}
        .high {{ color: #e67e22; font-weight: bold; }}
        .medium {{ color: #f39c12; }}
        .low {{ color: #27ae60; }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
        .badge-critical {{ background: #e74c3c; color: white; }}
        .badge-high {{ background: #e67e22; color: white; }}
        .badge-medium {{ background: #f39c12; color: white; }}
        .badge-low {{ background: #27ae60; color: white; }}
        .recommendation {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Pub-Sub System Analysis Report</h1>
        <p class="timestamp">Generated: {results.get('timestamp', datetime.now().isoformat())}</p>
        
        <h2>📊 System Overview</h2>
        <div class="metric-grid">
"""
    
    # Add system metrics
    summary = results.get('graph_summary', {})
    metrics = [
        ('Total Nodes', summary.get('total_nodes', 'N/A')),
        ('Total Edges', summary.get('total_edges', 'N/A')),
        ('Graph Density', f"{summary.get('density', 0):.4f}"),
        ('Connected', '✓' if summary.get('is_connected') else '✗')
    ]
    
    for label, value in metrics:
        html += f"""
            <div class="metric-card">
                <h3>{label}</h3>
                <div class="value">{value}</div>
            </div>
"""
    
    html += """
        </div>
        
        <h2>⚠️ Critical Components</h2>
"""
    
    # Add critical components table
    critical_scores = results.get('criticality_scores', {})
    critical_comps = critical_scores.get('critical_components', [])
    
    if critical_comps:
        html += """
        <table>
            <thead>
                <tr>
                    <th>Component</th>
                    <th>Type</th>
                    <th>Criticality Score</th>
                    <th>Level</th>
                    <th>Articulation Point</th>
                    <th>Components Affected</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for comp in critical_comps[:15]:  # Top 15
            level = comp.get('level', 'UNKNOWN')
            badge_class = {
                'CRITICAL': 'badge-critical',
                'HIGH': 'badge-high',
                'MEDIUM': 'badge-medium',
                'LOW': 'badge-low'
            }.get(level, 'badge-medium')
            
            html += f"""
                <tr>
                    <td><strong>{comp.get('component', 'N/A')}</strong></td>
                    <td>{comp.get('type', 'N/A')}</td>
                    <td>{comp.get('composite_score', 0):.3f}</td>
                    <td><span class="badge {badge_class}">{level}</span></td>
                    <td>{'✓' if comp.get('is_articulation_point') else ''}</td>
                    <td>{comp.get('components_affected', 'N/A')}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
"""
    else:
        html += "<p>No critical components identified.</p>"
    
    # Add QoS analysis if available
    if 'qos_analysis' in results:
        qos = results['qos_analysis']
        html += f"""
        <h2>🎯 QoS Analysis</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>High Priority Topics</h3>
                <div class="value">{len(qos.get('high_priority_topics', []))}</div>
            </div>
            <div class="metric-card">
                <h3>High Priority Applications</h3>
                <div class="value">{len(qos.get('high_priority_applications', []))}</div>
            </div>
            <div class="metric-card">
                <h3>Compatibility Issues</h3>
                <div class="value">{len(qos.get('compatibility_issues', []))}</div>
            </div>
        </div>
"""
    
    # Add failure simulation if available
    if 'failure_simulation' in results and results['failure_simulation']:
        failure = results['failure_simulation']
        html += f"""
        <h2>💥 Failure Simulation</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>Resilience Score</h3>
                <div class="value">{failure.get('resilience_score', 0):.3f}</div>
            </div>
            <div class="metric-card">
                <h3>Avg Components Affected</h3>
                <div class="value">{failure.get('avg_components_affected', 0):.1f}</div>
            </div>
        </div>
"""
    
    # Add recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        html += """
        <h2>💡 Recommendations</h2>
"""
        
        for i, rec in enumerate(recommendations[:10], 1):
            priority = rec.get('priority', 'MEDIUM')
            html += f"""
        <div class="recommendation">
            <strong>[{priority}] {rec.get('type', 'Recommendation')}</strong><br>
            <strong>Component:</strong> {rec.get('component', 'N/A')}<br>
            <strong>Issue:</strong> {rec.get('issue', 'N/A')}<br>
            <strong>Action:</strong> {rec.get('recommendation', 'N/A')}
        </div>
"""
    
    # Add execution time
    exec_time = results.get('execution_time', {})
    html += f"""
        <h2>⏱️ Execution Time</h2>
        <p><strong>Total Analysis Time:</strong> {exec_time.get('total', 0):.2f} seconds</p>
    </div>
</body>
</html>
"""
    
    # Write file
    with open(output_path, 'w') as f:
        f.write(html)


def print_analysis_summary(results: dict) -> None:
    """
    Print formatted analysis summary to console
    
    Args:
        results: Analysis results dictionary
    """
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    # System overview
    summary = results.get('graph_summary', {})
    print(f"\n📊 System Overview:")
    print(f"   Nodes:      {summary.get('total_nodes', 'N/A')}")
    print(f"   Edges:      {summary.get('total_edges', 'N/A')}")
    print(f"   Density:    {summary.get('density', 0):.4f}")
    print(f"   Connected:  {summary.get('is_connected', 'N/A')}")
    
    # Critical components
    critical_scores = results.get('criticality_scores', {})
    print(f"\n⚠️  Critical Components:")
    print(f"   Total Analyzed:        {critical_scores.get('total_components', 'N/A')}")
    print(f"   Critical (>0.7):       {critical_scores.get('critical_count', 'N/A')}")
    print(f"   High (0.5-0.7):        {critical_scores.get('high_count', 'N/A')}")
    print(f"   Articulation Points:   {critical_scores.get('articulation_points', 'N/A')}")
    
    # Top critical components
    critical_comps = critical_scores.get('critical_components', [])
    if critical_comps:
        print(f"\n   Top 5 Most Critical:")
        for i, comp in enumerate(critical_comps[:5], 1):
            print(f"   {i}. {comp.get('component', 'N/A')} "
                  f"(Score: {comp.get('composite_score', 0):.3f}, "
                  f"Type: {comp.get('type', 'N/A')})")
    
    # QoS analysis
    if 'qos_analysis' in results:
        qos = results['qos_analysis']
        print(f"\n🎯 QoS Analysis:")
        print(f"   High Priority Topics:       {len(qos.get('high_priority_topics', []))}")
        print(f"   High Priority Applications: {len(qos.get('high_priority_applications', []))}")
        print(f"   Compatibility Issues:       {len(qos.get('compatibility_issues', []))}")
    
    # Failure simulation
    if 'failure_simulation' in results and results['failure_simulation']:
        failure = results['failure_simulation']
        print(f"\n💥 Failure Simulation:")
        print(f"   Resilience Score:          {failure.get('resilience_score', 0):.3f}")
        print(f"   Avg Components Affected:   {failure.get('avg_components_affected', 0):.1f}")
    
    # Recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        print(f"\n💡 Top 3 Recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"   {i}. [{rec.get('priority', 'N/A')}] {rec.get('type', 'N/A')}")
            print(f"      Component: {rec.get('component', 'N/A')}")
            print(f"      Action: {rec.get('recommendation', 'N/A')}")
    
    # Execution time
    exec_time = results.get('execution_time', {})
    total_time = exec_time.get('total', 0)
    print(f"\n⏱️  Execution Time: {total_time:.2f}s")
    
    print("\n" + "=" * 70)


def main() -> int:
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Analyze pub-sub system graph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze from JSON
  python analyze_graph.py --input system.json

  # Analyze from Neo4j
  python analyze_graph.py --neo4j --uri bolt://localhost:7687 \\
      --user neo4j --password password

  # With failure simulation
  python analyze_graph.py --input system.json --simulate

  # Custom criticality weights
  python analyze_graph.py --input system.json \\
      --alpha 0.5 --beta 0.3 --gamma 0.2

  # Export to multiple formats
  python analyze_graph.py --input system.json \\
      --export-json --export-csv --export-html

  # Disable QoS analysis
  python analyze_graph.py --input system.json --no-qos

  # Verbose logging
  python analyze_graph.py --input system.json --verbose
        """
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', type=str,
                            help='Input JSON file')
    input_group.add_argument('--neo4j', action='store_true',
                            help='Load from Neo4j database')
    
    # Neo4j connection
    parser.add_argument('--uri', type=str, default='bolt://localhost:7687',
                       help='Neo4j URI (default: bolt://localhost:7687)')
    parser.add_argument('--user', type=str, default='neo4j',
                       help='Neo4j username (default: neo4j)')
    parser.add_argument('--password', type=str, default='password',
                       help='Neo4j password (default: password)')
    parser.add_argument('--database', type=str, default='neo4j',
                       help='Neo4j database (default: neo4j)')
    
    # Analysis options
    parser.add_argument('--simulate', action='store_true',
                       help='Enable failure simulation')
    parser.add_argument('--no-qos', action='store_true',
                       help='Disable QoS analysis')
    parser.add_argument('--alpha', type=float, default=0.4,
                       help='Betweenness centrality weight (default: 0.4)')
    parser.add_argument('--beta', type=float, default=0.3,
                       help='Articulation point weight (default: 0.3)')
    parser.add_argument('--gamma', type=float, default=0.3,
                       help='Impact score weight (default: 0.3)')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--export-json', action='store_true', default=True,
                       help='Export results as JSON (default: enabled)')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export criticality scores as CSV')
    parser.add_argument('--export-html', action='store_true',
                       help='Export HTML report')
    parser.add_argument('--no-summary', action='store_true',
                       help='Skip console summary')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        print("\n" + "=" * 70)
        print("COMPLEX SYSTEM ANALYSIS")
        print("=" * 70)
        
        # Load graph
        start_time = time.time()
        
        if args.neo4j:
            model, graph = load_graph_from_neo4j(
                args.uri, args.user, args.password, args.database
            )
        else:
            model, graph = load_graph_from_json(args.input)
        
        # Validate graph
        if len(graph) == 0:
            logger.error("Graph is empty! Cannot perform analysis.")
            return 1
        
        logger.info(f"Graph loaded in {time.time() - start_time:.2f}s")
        
        # Initialize orchestrator
        logger.info("\nInitializing AnalysisOrchestrator...")
        orchestrator = AnalysisOrchestrator(
            output_dir=args.output,
            enable_qos=not args.no_qos,
            criticality_weights={
                'alpha': args.alpha,
                'beta': args.beta,
                'gamma': args.gamma
            }
        )
        
        # Run analysis
        logger.info("\nRunning comprehensive analysis...")
        results = orchestrator.analyze_graph(
            graph=graph,
            graph_model=model,
            enable_simulation=args.simulate
        )
        
        logger.info(f"\n✓ Analysis complete in {results['execution_time']['total']:.2f}s")
        
        # Export results
        output_dir = Path(args.output)
        save_analysis_results(
            results,
            output_dir,
            export_json=args.export_json,
            export_csv=args.export_csv,
            export_html=args.export_html
        )
        
        # Print summary
        if not args.no_summary:
            print_analysis_summary(results)
        
        logger.info(f"\n✓ All results saved to {args.output}/")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install required packages: pip install neo4j networkx pandas")
        return 1
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
