#!/usr/bin/env python3
"""
Analyze Graph Script - Enhanced Version

Command-line interface for comprehensive pub-sub system analysis using
the modular architecture with enhanced error handling, validation, and reporting.

Features:
- Load graphs from JSON files or Neo4j database
- Multi-metric centrality analysis (13+ metrics)
- QoS-aware criticality scoring with configurable weights
- Failure simulation and impact assessment
- Multi-layer dependency analysis
- Anti-pattern detection (SPOF, God Topics, etc.)
- Multiple output formats (JSON, CSV, HTML, Markdown)
- Configurable analysis parameters
- Progress reporting and validation
- Comprehensive error handling

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
        --export-json --export-csv --export-html --export-md
    
    # With anti-pattern detection
    python analyze_graph.py --input system.json --detect-antipatterns
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.core.graph_builder import GraphBuilder
    from src.core.graph_exporter import GraphExporter
    from src.core.graph_model import GraphModel
    from src.orchestration.analysis_orchestrator import AnalysisOrchestrator
    import networkx as nx
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("Please ensure all required modules are installed:")
    print("  pip install networkx")
    print("  pip install neo4j  # Optional, for Neo4j support")
    print("  pip install pandas  # Optional, for CSV export")
    sys.exit(1)


# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def setup_logging(verbose: bool = False, log_file: str = 'analysis.log') -> logging.Logger:
    """
    Configure logging with enhanced formatting
    
    Args:
        verbose: Enable DEBUG level logging
        log_file: Path to log file
    
    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)


def validate_graph_structure(graph: nx.DiGraph, model: GraphModel, logger: logging.Logger) -> Tuple[bool, List[str]]:
    """
    Validate graph structure and detect potential issues
    
    Args:
        graph: NetworkX graph
        model: GraphModel instance
        logger: Logger instance
    
    Returns:
        Tuple of (is_valid, list of warnings)
    """
    warnings = []
    
    # Check if graph is empty
    if len(graph) == 0:
        logger.error("Graph is empty! Cannot perform analysis.")
        return False, ["Graph contains no nodes"]
    
    # Check connectivity
    if graph.is_directed():
        if not nx.is_weakly_connected(graph):
            num_components = nx.number_weakly_connected_components(graph)
            warnings.append(f"Graph has {num_components} disconnected components")
    else:
        if not nx.is_connected(graph):
            num_components = nx.number_connected_components(graph)
            warnings.append(f"Graph has {num_components} disconnected components")
    
    # Check for self-loops
    self_loops = list(nx.selfloop_edges(graph))
    if self_loops:
        warnings.append(f"Graph contains {len(self_loops)} self-loops")
    
    # Check node types distribution
    node_types = {}
    for node, data in graph.nodes(data=True):
        node_type = data.get('type', 'Unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    logger.info(f"Node type distribution: {node_types}")
    
    # Check for isolated nodes
    isolated = list(nx.isolates(graph))
    if isolated:
        warnings.append(f"Graph contains {len(isolated)} isolated nodes: {isolated[:5]}")
    
    # Check edge count
    if len(graph.edges()) == 0:
        warnings.append("Graph contains no edges")
    
    return True, warnings


def load_graph_from_json(filepath: str, logger: logging.Logger) -> Tuple[GraphModel, nx.DiGraph]:
    """
    Load graph from JSON file with enhanced error handling
    
    Args:
        filepath: Path to JSON configuration file
        logger: Logger instance
    
    Returns:
        Tuple of (GraphModel, NetworkX DiGraph)
    
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
        Exception: For other loading errors
    """
    logger.info(f"Loading graph from JSON: {filepath}")
    
    # Validate file exists
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    # Validate file is readable
    if not Path(filepath).is_file():
        raise ValueError(f"Path is not a file: {filepath}")
    
    try:
        # Load and validate JSON
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger.debug(f"Loaded JSON with keys: {list(data.keys())}")
        
        # Build graph using GraphBuilder
        builder = GraphBuilder()
        model = builder.build_from_json(filepath)
        
        logger.info(f"✓ Loaded {len(model.applications)} applications, "
                   f"{len(model.topics)} topics, "
                   f"{len(model.brokers)} brokers, "
                   f"{len(model.nodes)} nodes")
        
        # Convert to NetworkX
        exporter = GraphExporter()
        graph = exporter.export_to_networkx(model)
        
        logger.info(f"✓ Created NetworkX graph: {len(graph)} nodes, {len(graph.edges())} edges")
        
        return model, graph
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading graph: {e}")
        raise


def load_graph_from_neo4j(uri: str, user: str, password: str, database: str,
                          logger: logging.Logger) -> Tuple[GraphModel, nx.DiGraph]:
    """
    Load graph from Neo4j database with enhanced error handling
    
    Args:
        uri: Neo4j URI
        user: Username
        password: Password
        database: Database name
        logger: Logger instance
    
    Returns:
        Tuple of (GraphModel, NetworkX DiGraph)
    
    Raises:
        ImportError: If neo4j package is not installed
        Exception: For connection or query errors
    """
    logger.info(f"Connecting to Neo4j: {uri}, database: {database}")
    
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise ImportError(
            "neo4j package is required for Neo4j support. "
            "Install with: pip install neo4j"
        )
    
    try:
        # Build graph using GraphBuilder
        builder = GraphBuilder()
        model = builder.build_from_neo4j(
            uri=uri,
            auth=(user, password),
            database=database
        )
        
        logger.info(f"✓ Loaded {len(model.applications)} applications, "
                   f"{len(model.topics)} topics, "
                   f"{len(model.brokers)} brokers, "
                   f"{len(model.nodes)} nodes from Neo4j")
        
        # Convert to NetworkX
        exporter = GraphExporter()
        graph = exporter.export_to_networkx(model)
        
        logger.info(f"✓ Created NetworkX graph: {len(graph)} nodes, {len(graph.edges())} edges")
        
        return model, graph
        
    except Exception as e:
        logger.error(f"Error connecting to Neo4j: {e}")
        logger.error("Please verify:")
        logger.error("  - Neo4j is running")
        logger.error("  - URI, username, and password are correct")
        logger.error("  - Database name is correct")
        logger.error("  - Network connectivity")
        raise


def export_results(results: Dict[str, Any], output_dir: Path, 
                  export_json: bool, export_csv: bool, export_html: bool,
                  export_md: bool, logger: logging.Logger) -> Dict[str, str]:
    """
    Export analysis results in multiple formats
    
    Args:
        results: Analysis results dictionary
        output_dir: Output directory
        export_json: Export as JSON
        export_csv: Export criticality scores as CSV
        export_html: Export as HTML report
        export_md: Export as Markdown report
        logger: Logger instance
    
    Returns:
        Dictionary mapping format to output file path
    """
    output_files = {}
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export JSON
    if export_json:
        json_path = output_dir / "analysis_results.json"
        try:
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            output_files['json'] = str(json_path)
            logger.info(f"✓ Exported JSON: {json_path}")
        except Exception as e:
            logger.error(f"Error exporting JSON: {e}")
    
    # Export CSV (criticality scores)
    if export_csv:
        csv_path = output_dir / "criticality_scores.csv"
        try:
            import pandas as pd
            
            # Extract criticality scores
            if 'criticality_scores' in results:
                scores_data = []
                for comp in results['criticality_scores'].get('all_scores', []):
                    scores_data.append({
                        'Component': comp.get('component'),
                        'Type': comp.get('type'),
                        'Composite Score': comp.get('composite_score'),
                        'Betweenness': comp.get('betweenness_centrality'),
                        'Is Articulation Point': comp.get('is_articulation_point'),
                        'Impact Score': comp.get('impact_score'),
                        'Criticality Level': comp.get('criticality_level')
                    })
                
                df = pd.DataFrame(scores_data)
                df = df.sort_values('Composite Score', ascending=False)
                df.to_csv(csv_path, index=False)
                
                output_files['csv'] = str(csv_path)
                logger.info(f"✓ Exported CSV: {csv_path}")
            else:
                logger.warning("No criticality scores found for CSV export")
                
        except ImportError:
            logger.warning("pandas not installed, skipping CSV export. Install with: pip install pandas")
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
    
    # Export HTML
    if export_html:
        html_path = output_dir / "analysis_report.html"
        try:
            html_content = generate_html_report(results)
            with open(html_path, 'w') as f:
                f.write(html_content)
            output_files['html'] = str(html_path)
            logger.info(f"✓ Exported HTML: {html_path}")
        except Exception as e:
            logger.error(f"Error exporting HTML: {e}")
    
    # Export Markdown
    if export_md:
        md_path = output_dir / "analysis_report.md"
        try:
            md_content = generate_markdown_report(results)
            with open(md_path, 'w') as f:
                f.write(md_content)
            output_files['markdown'] = str(md_path)
            logger.info(f"✓ Exported Markdown: {md_path}")
        except Exception as e:
            logger.error(f"Error exporting Markdown: {e}")
    
    return output_files


def generate_html_report(results: Dict[str, Any]) -> str:
    """
    Generate HTML report from analysis results
    
    Args:
        results: Analysis results dictionary
    
    Returns:
        HTML content as string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pub-Sub System Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
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
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        h3 {{
            color: #7f8c8d;
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
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .critical {{ color: #e74c3c; font-weight: bold; }}
        .high {{ color: #e67e22; font-weight: bold; }}
        .medium {{ color: #f39c12; }}
        .low {{ color: #27ae60; }}
        .recommendation {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-style: italic;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .badge-critical {{ background: #e74c3c; color: white; }}
        .badge-high {{ background: #e67e22; color: white; }}
        .badge-medium {{ background: #f39c12; color: white; }}
        .badge-low {{ background: #27ae60; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Pub-Sub System Analysis Report</h1>
        <p class="timestamp">Generated: {timestamp}</p>
"""
    
    # Graph summary
    if 'graph_summary' in results:
        summary = results['graph_summary']
        html += """
        <h2>📈 Graph Summary</h2>
        <div class="metric-grid">
"""
        html += f"""
            <div class="metric-card">
                <div class="metric-value">{summary.get('total_nodes', 0)}</div>
                <div class="metric-label">Total Nodes</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('total_edges', 0)}</div>
                <div class="metric-label">Total Edges</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('applications', 0)}</div>
                <div class="metric-label">Applications</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('topics', 0)}</div>
                <div class="metric-label">Topics</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('brokers', 0)}</div>
                <div class="metric-label">Brokers</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get('nodes', 0)}</div>
                <div class="metric-label">Infrastructure Nodes</div>
            </div>
"""
        html += """
        </div>
"""
    
    # Critical components
    if 'criticality_scores' in results:
        crit = results['criticality_scores']
        critical_comps = crit.get('critical_components', [])
        
        html += f"""
        <h2>⚠️ Critical Components ({len(critical_comps)})</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Component</th>
                    <th>Type</th>
                    <th>Score</th>
                    <th>Level</th>
                    <th>Articulation Point</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for i, comp in enumerate(critical_comps[:10], 1):
            level = comp.get('criticality_level', 'LOW')
            badge_class = f"badge-{level.lower()}"
            is_ap = "✓" if comp.get('is_articulation_point') else ""
            
            html += f"""
                <tr>
                    <td>{i}</td>
                    <td><strong>{comp.get('component', 'N/A')}</strong></td>
                    <td>{comp.get('type', 'N/A')}</td>
                    <td>{comp.get('composite_score', 0):.3f}</td>
                    <td><span class="badge {badge_class}">{level}</span></td>
                    <td>{is_ap}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
"""
    
    # Recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        html += f"""
        <h2>💡 Recommendations ({len(recommendations)})</h2>
"""
        for i, rec in enumerate(recommendations[:5], 1):
            html += f"""
        <div class="recommendation">
            <h3>{i}. [{rec.get('priority', 'N/A')}] {rec.get('type', 'N/A')}</h3>
            <p><strong>Component:</strong> {rec.get('component', 'N/A')}</p>
            <p><strong>Issue:</strong> {rec.get('issue', 'N/A')}</p>
            <p><strong>Recommendation:</strong> {rec.get('recommendation', 'N/A')}</p>
        </div>
"""
    
    # Execution time
    if 'execution_time' in results:
        exec_time = results['execution_time']
        html += f"""
        <h2>⏱️ Performance</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{exec_time.get('total', 0):.2f}s</div>
                <div class="metric-label">Total Execution Time</div>
            </div>
"""
        if 'analysis' in exec_time:
            html += f"""
            <div class="metric-card">
                <div class="metric-value">{exec_time.get('analysis', 0):.2f}s</div>
                <div class="metric-label">Analysis Time</div>
            </div>
"""
        html += """
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    return html


def generate_markdown_report(results: Dict[str, Any]) -> str:
    """
    Generate Markdown report from analysis results
    
    Args:
        results: Analysis results dictionary
    
    Returns:
        Markdown content as string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md = f"""# Pub-Sub System Analysis Report

**Generated:** {timestamp}

---

## Graph Summary

"""
    
    if 'graph_summary' in results:
        summary = results['graph_summary']
        md += f"""
| Metric | Value |
|--------|-------|
| Total Nodes | {summary.get('total_nodes', 0)} |
| Total Edges | {summary.get('total_edges', 0)} |
| Applications | {summary.get('applications', 0)} |
| Topics | {summary.get('topics', 0)} |
| Brokers | {summary.get('brokers', 0)} |
| Infrastructure Nodes | {summary.get('nodes', 0)} |

"""
    
    # Critical components
    if 'criticality_scores' in results:
        crit = results['criticality_scores']
        critical_comps = crit.get('critical_components', [])
        
        md += f"""## Critical Components ({len(critical_comps)})

| Rank | Component | Type | Score | Level | AP |
|------|-----------|------|-------|-------|-------|
"""
        
        for i, comp in enumerate(critical_comps[:10], 1):
            is_ap = "✓" if comp.get('is_articulation_point') else ""
            md += f"| {i} | **{comp.get('component', 'N/A')}** | {comp.get('type', 'N/A')} | {comp.get('composite_score', 0):.3f} | {comp.get('criticality_level', 'LOW')} | {is_ap} |\n"
    
    # Recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        md += f"\n## Recommendations ({len(recommendations)})\n\n"
        
        for i, rec in enumerate(recommendations[:5], 1):
            md += f"""### {i}. [{rec.get('priority', 'N/A')}] {rec.get('type', 'N/A')}

- **Component:** {rec.get('component', 'N/A')}
- **Issue:** {rec.get('issue', 'N/A')}
- **Recommendation:** {rec.get('recommendation', 'N/A')}

"""
    
    # Performance
    if 'execution_time' in results:
        exec_time = results['execution_time']
        md += f"""## Performance

- **Total Execution Time:** {exec_time.get('total', 0):.2f}s
"""
        if 'analysis' in exec_time:
            md += f"- **Analysis Time:** {exec_time.get('analysis', 0):.2f}s\n"
    
    return md


def print_summary(results: Dict[str, Any], elapsed_time: float) -> None:
    """
    Print comprehensive analysis summary to console
    
    Args:
        results: Analysis results dictionary
        elapsed_time: Total execution time in seconds
    """
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}ANALYSIS COMPLETE{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
    
    # Graph summary
    if 'graph_summary' in results:
        summary = results['graph_summary']
        print(f"\n{Colors.OKBLUE}📊 Graph Summary:{Colors.ENDC}")
        print(f"   Total Nodes:            {summary.get('total_nodes', 0)}")
        print(f"   Total Edges:            {summary.get('total_edges', 0)}")
        print(f"   Applications:           {summary.get('applications', 0)}")
        print(f"   Topics:                 {summary.get('topics', 0)}")
        print(f"   Brokers:                {summary.get('brokers', 0)}")
        print(f"   Infrastructure Nodes:   {summary.get('nodes', 0)}")
        
        if 'articulation_points' in summary:
            print(f"   Articulation Points:    {summary.get('articulation_points', 0)}")
        if 'avg_degree' in summary:
            print(f"   Average Degree:         {summary.get('avg_degree', 0):.2f}")
    
    # Top critical components
    if 'criticality_scores' in results:
        crit = results['criticality_scores']
        critical_comps = crit.get('critical_components', [])
        
        print(f"\n{Colors.WARNING}⚠️  Top 5 Critical Components:{Colors.ENDC}")
        for i, comp in enumerate(critical_comps[:5], 1):
            comp_name = comp.get('component', 'N/A')
            comp_score = comp.get('composite_score', 0)
            comp_type = comp.get('type', 'N/A')
            comp_level = comp.get('criticality_level', 'LOW')
            
            # Color based on level
            if comp_level == 'CRITICAL':
                color = Colors.FAIL
            elif comp_level == 'HIGH':
                color = Colors.WARNING
            else:
                color = Colors.OKGREEN
            
            print(f"   {i}. {color}{comp_name}{Colors.ENDC} "
                  f"(Score: {comp_score:.3f}, Type: {comp_type})")
    
    # QoS analysis
    if 'qos_analysis' in results:
        qos = results['qos_analysis']
        print(f"\n{Colors.OKCYAN}🎯 QoS Analysis:{Colors.ENDC}")
        print(f"   High Priority Topics:       {len(qos.get('high_priority_topics', []))}")
        print(f"   High Priority Applications: {len(qos.get('high_priority_applications', []))}")
        print(f"   Compatibility Issues:       {len(qos.get('compatibility_issues', []))}")
    
    # Failure simulation
    if 'failure_simulation' in results and results['failure_simulation']:
        failure = results['failure_simulation']
        print(f"\n{Colors.FAIL}💥 Failure Simulation:{Colors.ENDC}")
        print(f"   Resilience Score:          {failure.get('resilience_score', 0):.3f}")
        print(f"   Avg Components Affected:   {failure.get('avg_components_affected', 0):.1f}")
        
        if 'max_cascade_depth' in failure:
            print(f"   Max Cascade Depth:         {failure.get('max_cascade_depth', 0)}")
    
    # Anti-patterns
    if 'anti_patterns' in results:
        antipatterns = results['anti_patterns']
        if antipatterns:
            print(f"\n{Colors.WARNING}🚨 Detected Anti-Patterns:{Colors.ENDC}")
            for pattern_name, instances in antipatterns.items():
                if instances:
                    print(f"   • {pattern_name}: {len(instances)} instance(s)")
    
    # Recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        print(f"\n{Colors.OKGREEN}💡 Top 3 Recommendations:{Colors.ENDC}")
        for i, rec in enumerate(recommendations[:3], 1):
            priority = rec.get('priority', 'N/A')
            rec_type = rec.get('type', 'N/A')
            component = rec.get('component', 'N/A')
            recommendation = rec.get('recommendation', 'N/A')
            
            print(f"   {i}. [{priority}] {rec_type}")
            print(f"      Component: {component}")
            print(f"      Action: {recommendation}")
    
    # Execution time
    print(f"\n{Colors.OKBLUE}⏱️  Execution Time: {elapsed_time:.2f}s{Colors.ENDC}")
    
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}\n")


def main() -> int:
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Analyze pub-sub system graph - Enhanced Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze from JSON
  python analyze_graph.py --input system.json

  # Analyze from Neo4j
  python analyze_graph.py --neo4j --uri bolt://localhost:7687 \\
      --user neo4j --password password

  # With failure simulation and anti-pattern detection
  python analyze_graph.py --input system.json --simulate --detect-antipatterns

  # Custom criticality weights
  python analyze_graph.py --input system.json \\
      --alpha 0.5 --beta 0.3 --gamma 0.2

  # Export to multiple formats
  python analyze_graph.py --input system.json \\
      --export-json --export-csv --export-html --export-md

  # Disable QoS analysis for faster execution
  python analyze_graph.py --input system.json --no-qos

  # Verbose logging with custom output directory
  python analyze_graph.py --input system.json --verbose \\
      --output results/analysis_$(date +%Y%m%d)

For more information, see docs/ANALYZE_GRAPH_README.md
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
                       help='Disable QoS analysis (faster execution)')
    parser.add_argument('--detect-antipatterns', action='store_true',
                       help='Enable anti-pattern detection')
    
    # Criticality weights
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
    parser.add_argument('--export-md', action='store_true',
                       help='Export Markdown report')
    parser.add_argument('--no-summary', action='store_true',
                       help='Skip console summary')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging (DEBUG level)')
    parser.add_argument('--log-file', type=str, default='analysis.log',
                       help='Log file path (default: analysis.log)')
    parser.add_argument('--no-color', action='store_true',
                       help='Disable colored output')
    
    args = parser.parse_args()
    
    # Disable colors if requested
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith('_'):
                setattr(Colors, attr, '')
    
    # Setup logging
    logger = setup_logging(args.verbose, args.log_file)
    
    try:
        # Print header
        print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}COMPREHENSIVE PUB-SUB SYSTEM ANALYSIS{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")
        
        # Load graph
        start_time = time.time()
        
        logger.info("Loading graph...")
        if args.neo4j:
            model, graph = load_graph_from_neo4j(
                args.uri, args.user, args.password, args.database, logger
            )
        else:
            model, graph = load_graph_from_json(args.input, logger)
        
        # Validate graph
        logger.info("Validating graph structure...")
        is_valid, warnings = validate_graph_structure(graph, model, logger)
        
        if not is_valid:
            logger.error("Graph validation failed!")
            return 1
        
        if warnings:
            logger.warning(f"Graph validation completed with {len(warnings)} warning(s):")
            for warning in warnings:
                logger.warning(f"  • {warning}")
        else:
            logger.info("✓ Graph validation passed")
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir.absolute()}")
        
        # Validate weights sum to ~1.0
        weight_sum = args.alpha + args.beta + args.gamma
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Criticality weights sum to {weight_sum:.3f} (expected ~1.0)")
            logger.warning("Weights will be normalized automatically")
        
        # Initialize analysis orchestrator
        logger.info("Initializing analysis orchestrator...")
        orchestrator = AnalysisOrchestrator(
            output_dir=str(output_dir),
            enable_qos=not args.no_qos,
            criticality_weights={
                'alpha': args.alpha,
                'beta': args.beta,
                'gamma': args.gamma
            }
        )
        
        # Run analysis
        logger.info(f"Running analysis (QoS: {not args.no_qos}, Simulation: {args.simulate})...")
        analysis_start = time.time()
        
        results = orchestrator.analyze_graph(
            graph=graph,
            graph_model=model,
            enable_simulation=args.simulate
        )
        
        analysis_time = time.time() - analysis_start
        logger.info(f"✓ Analysis completed in {analysis_time:.2f}s")
        
        # Anti-pattern detection
        if args.detect_antipatterns:
            logger.info("Detecting anti-patterns...")
            # This would integrate with anti-pattern detection module
            # For now, adding a placeholder
            results['anti_patterns'] = {}
        
        # Export results
        logger.info("Exporting results...")
        output_files = export_results(
            results, output_dir,
            args.export_json, args.export_csv,
            args.export_html, args.export_md,
            logger
        )
        
        # Print summary
        total_time = time.time() - start_time
        results['execution_time'] = {
            'total': total_time,
            'analysis': analysis_time,
            'loading': start_time,
        }
        
        if not args.no_summary:
            print_summary(results, total_time)
        
        # Print output files
        if output_files:
            print(f"{Colors.OKGREEN}📁 Output Files:{Colors.ENDC}")
            for format_type, filepath in output_files.items():
                print(f"   • {format_type.upper()}: {filepath}")
            print()
        
        print(f"{Colors.OKGREEN}✓ Analysis completed successfully!{Colors.ENDC}\n")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\nAnalysis interrupted by user")
        return 130
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        return 1
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
