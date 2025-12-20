#!/usr/bin/env python3
"""
Enhanced Graph Analyzer - Comprehensive Pub-Sub System Analysis
================================================================

Main entry point for analyzing distributed publish-subscribe systems using
advanced graph algorithms including relationship analysis, motif detection,
dependency chains, and ensemble criticality scoring.

Features:
  - Node-centric criticality (betweenness, PageRank, articulation points)
  - Edge-centric analysis (edge betweenness, bridges, Simmelian strength)
  - HITS-based role analysis (hubs and authorities)
  - Network motif detection (fan-out, fan-in, chains, diamonds)
  - Dependency chain analysis (transitive depth, cascade risk)
  - Multi-layer correlation analysis
  - Ensemble criticality scoring (combining multiple algorithms)

Usage:
    # Basic analysis from JSON file
    python analyze_graph.py --input system.json
    
    # With relationship analysis
    python analyze_graph.py --input system.json --relationship-analysis
    
    # Full analysis with all features
    python analyze_graph.py --input system.json --full
    
    # Export to multiple formats
    python analyze_graph.py --input system.json --output-dir results/ --format json html
    
    # Custom criticality weights
    python analyze_graph.py --input system.json --alpha 0.5 --beta 0.25 --gamma 0.25

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    import networkx as nx
except ImportError:
    print("ERROR: NetworkX is required. Install with: pip install networkx")
    sys.exit(1)


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
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    _enabled = True
    
    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output)"""
        cls._enabled = False
        cls.HEADER = ''
        cls.BLUE = ''
        cls.CYAN = ''
        cls.GREEN = ''
        cls.WARNING = ''
        cls.FAIL = ''
        cls.ENDC = ''
        cls.BOLD = ''
        cls.DIM = ''


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
    print(f"\n{Colors.CYAN}{Colors.BOLD}▸ {text}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-'*50}{Colors.ENDC}")


def print_subsection(text: str):
    """Print subsection header"""
    print(f"\n  {Colors.BLUE}{text}{Colors.ENDC}")


def print_success(text: str):
    """Print success message"""
    print(f"  {Colors.GREEN}✓{Colors.ENDC} {text}")


def print_warning(text: str):
    """Print warning message"""
    print(f"  {Colors.WARNING}⚠{Colors.ENDC} {text}")


def print_critical(text: str):
    """Print critical message"""
    print(f"  {Colors.FAIL}✗{Colors.ENDC} {Colors.BOLD}{text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message"""
    print(f"    {Colors.DIM}•{Colors.ENDC} {text}")


def print_metric(label: str, value: Any, indent: int = 2):
    """Print metric with label"""
    spaces = "  " * indent
    print(f"{spaces}{Colors.BOLD}{label}:{Colors.ENDC} {value}")


# ============================================================================
# Graph Builder
# ============================================================================

def build_graph_from_dict(data: Dict[str, Any]) -> nx.DiGraph:
    """
    Build NetworkX DiGraph from dictionary data.
    
    Args:
        data: Dictionary with applications, brokers, topics, nodes, and edges
        
    Returns:
        NetworkX directed graph
    """
    G = nx.DiGraph()
    
    # Add nodes - filter out 'type' from kwargs if present to avoid conflicts
    for app in data.get('applications', []):
        attrs = {k: v for k, v in app.items() if k != 'type'}
        G.add_node(app['id'], type='Application', **attrs)
    
    for broker in data.get('brokers', []):
        attrs = {k: v for k, v in broker.items() if k != 'type'}
        G.add_node(broker['id'], type='Broker', **attrs)
    
    for topic in data.get('topics', []):
        attrs = {k: v for k, v in topic.items() if k != 'type'}
        G.add_node(topic['id'], type='Topic', **attrs)
    
    for node in data.get('nodes', []):
        attrs = {k: v for k, v in node.items() if k != 'type'}
        G.add_node(node['id'], type='Node', **attrs)
    
    # Add edges (support both 'edges' and 'relationships' keys)
    edges = data.get('edges', data.get('relationships', {}))
    
    for pub in edges.get('publishes_to', []):
        G.add_edge(pub['from'], pub['to'], type='PUBLISHES_TO')
    
    for sub in edges.get('subscribes_to', []):
        G.add_edge(sub['from'], sub['to'], type='SUBSCRIBES_TO')
    
    for route in edges.get('routes', []):
        G.add_edge(route['from'], route['to'], type='ROUTES')
    
    for runs in edges.get('runs_on', []):
        G.add_edge(runs['from'], runs['to'], type='RUNS_ON')
    
    for conn in edges.get('connects_to', []):
        G.add_edge(conn['from'], conn['to'], type='CONNECTS_TO')
    
    return G


def build_graph_from_file(filepath: str) -> nx.DiGraph:
    """Load graph from JSON file"""
    with open(filepath) as f:
        data = json.load(f)
    return build_graph_from_dict(data)


# ============================================================================
# Analysis Result Classes
# ============================================================================

@dataclass
class NodeCriticalityScore:
    """Criticality score for a single node"""
    node_id: str
    node_type: str
    betweenness_centrality: float
    is_articulation_point: bool
    impact_score: float
    composite_score: float
    criticality_level: str
    pagerank: float = 0.0
    closeness: float = 0.0
    degree: int = 0
    reasons: List[str] = None
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


@dataclass
class StructuralAnalysisResult:
    """Results of structural analysis"""
    articulation_points: List[str]
    bridges: List[tuple]
    strongly_connected_components: int
    weakly_connected_components: int
    density: float
    diameter: Optional[int]
    average_clustering: float
    k_core_max: int


@dataclass
class GraphAnalysisResult:
    """Complete analysis result"""
    # Metadata
    timestamp: str
    input_file: str
    
    # Graph summary
    total_nodes: int
    total_edges: int
    nodes_by_type: Dict[str, int]
    edges_by_type: Dict[str, int]
    
    # Criticality scores
    criticality_scores: List[NodeCriticalityScore]
    criticality_by_level: Dict[str, int]
    
    # Structural analysis
    structural: StructuralAnalysisResult
    
    # Relationship analysis (optional)
    relationship_analysis: Optional[Dict[str, Any]] = None
    
    # Recommendations
    recommendations: List[Dict[str, Any]] = None


# ============================================================================
# Core Analysis Functions
# ============================================================================

def analyze_structure(G: nx.DiGraph) -> StructuralAnalysisResult:
    """Analyze structural properties of the graph"""
    G_undirected = G.to_undirected()
    
    # Articulation points
    aps = list(nx.articulation_points(G_undirected))
    
    # Bridges
    bridges = list(nx.bridges(G_undirected))
    
    # Connected components
    sccs = nx.number_strongly_connected_components(G)
    wccs = nx.number_weakly_connected_components(G)
    
    # Density
    density = nx.density(G)
    
    # Diameter (only if connected)
    try:
        if nx.is_weakly_connected(G):
            diameter = nx.diameter(G_undirected)
        else:
            diameter = None
    except:
        diameter = None
    
    # Clustering
    avg_clustering = nx.average_clustering(G_undirected)
    
    # K-core
    kcore = nx.core_number(G_undirected)
    k_max = max(kcore.values()) if kcore else 0
    
    return StructuralAnalysisResult(
        articulation_points=aps,
        bridges=bridges,
        strongly_connected_components=sccs,
        weakly_connected_components=wccs,
        density=density,
        diameter=diameter,
        average_clustering=avg_clustering,
        k_core_max=k_max
    )


def calculate_criticality_scores(G: nx.DiGraph, 
                                  alpha: float = 0.4,
                                  beta: float = 0.3,
                                  gamma: float = 0.3) -> List[NodeCriticalityScore]:
    """
    Calculate composite criticality scores for all nodes.
    
    Formula: C_score(v) = α·C_B^norm(v) + β·AP(v) + γ·I(v)
    
    Args:
        G: NetworkX directed graph
        alpha: Weight for betweenness centrality
        beta: Weight for articulation point indicator
        gamma: Weight for impact score
        
    Returns:
        List of NodeCriticalityScore sorted by composite score
    """
    G_undirected = G.to_undirected()
    
    # Compute metrics
    betweenness = nx.betweenness_centrality(G, normalized=True)
    pagerank = nx.pagerank(G, alpha=0.85)
    closeness = nx.closeness_centrality(G)
    articulation_points = set(nx.articulation_points(G_undirected))
    
    # Normalize betweenness
    bc_values = list(betweenness.values())
    bc_max = max(bc_values) if bc_values else 1.0
    bc_normalized = {n: v / bc_max if bc_max > 0 else 0 for n, v in betweenness.items()}
    
    scores = []
    
    for node in G.nodes():
        node_data = G.nodes[node]
        node_type = node_data.get('type', 'Unknown')
        
        # Core metrics
        bc_norm = bc_normalized.get(node, 0)
        is_ap = node in articulation_points
        ap_indicator = 1.0 if is_ap else 0.0
        
        # Impact score (based on reachability)
        descendants = nx.descendants(G, node)
        impact = len(descendants) / max(G.number_of_nodes() - 1, 1)
        
        # Composite score
        composite = (
            alpha * bc_norm +
            beta * ap_indicator +
            gamma * impact
        )
        
        # Determine level
        if composite >= 0.6 or is_ap:
            level = "CRITICAL"
        elif composite >= 0.4:
            level = "HIGH"
        elif composite >= 0.2:
            level = "MEDIUM"
        else:
            level = "LOW"
        
        # Generate reasons
        reasons = []
        if is_ap:
            reasons.append("Articulation point - single point of failure")
        if bc_norm > 0.5:
            reasons.append(f"High betweenness ({bc_norm:.3f}) - routing bottleneck")
        if impact > 0.3:
            reasons.append(f"High impact ({impact:.3f}) - affects many components")
        if pagerank.get(node, 0) > 0.1:
            reasons.append(f"High PageRank ({pagerank.get(node, 0):.3f}) - influential")
        if not reasons:
            reasons.append("Standard component")
        
        scores.append(NodeCriticalityScore(
            node_id=node,
            node_type=node_type,
            betweenness_centrality=bc_norm,
            is_articulation_point=is_ap,
            impact_score=impact,
            composite_score=composite,
            criticality_level=level,
            pagerank=pagerank.get(node, 0),
            closeness=closeness.get(node, 0),
            degree=G.degree(node),
            reasons=reasons
        ))
    
    # Sort by composite score descending
    scores.sort(key=lambda x: x.composite_score, reverse=True)
    
    return scores


def generate_recommendations(G: nx.DiGraph, 
                             structural: StructuralAnalysisResult,
                             scores: List[NodeCriticalityScore]) -> List[Dict[str, Any]]:
    """Generate actionable recommendations based on analysis"""
    recommendations = []
    
    # Articulation points
    if structural.articulation_points:
        recommendations.append({
            'priority': 'CRITICAL',
            'category': 'Single Points of Failure',
            'issue': f'{len(structural.articulation_points)} articulation points detected',
            'recommendation': 'Add redundant connections to eliminate single points of failure',
            'affected_nodes': structural.articulation_points[:5],
            'risk_reduction': 'High'
        })
    
    # Bridges
    if structural.bridges:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Network Topology',
            'issue': f'{len(structural.bridges)} bridge edges detected',
            'recommendation': 'Add redundant network paths between disconnected regions',
            'affected_edges': [list(b) for b in structural.bridges[:5]],
            'risk_reduction': 'High'
        })
    
    # High criticality nodes
    critical_nodes = [s for s in scores if s.criticality_level == 'CRITICAL']
    if len(critical_nodes) > 5:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Risk Concentration',
            'issue': f'{len(critical_nodes)} nodes with CRITICAL criticality',
            'recommendation': 'Implement redundancy and enhanced monitoring for critical components',
            'affected_nodes': [n.node_id for n in critical_nodes[:5]],
            'risk_reduction': 'Medium'
        })
    
    # Low connectivity
    if structural.density < 0.1:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Architecture',
            'issue': f'Low graph density ({structural.density:.4f})',
            'recommendation': 'Consider adding alternative message paths for resilience',
            'risk_reduction': 'Medium'
        })
    
    # Disconnected components
    if structural.weakly_connected_components > 1:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Connectivity',
            'issue': f'{structural.weakly_connected_components} disconnected components',
            'recommendation': 'Review if isolation is intentional or add bridging connections',
            'risk_reduction': 'High'
        })
    
    return recommendations


# ============================================================================
# Export Functions
# ============================================================================

def export_json(result: GraphAnalysisResult, filepath: Path):
    """Export results to JSON file"""
    def serialize(obj):
        if hasattr(obj, '__dict__'):
            return {k: serialize(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, list):
            return [serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return obj
    
    data = serialize(result)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print_success(f"Exported JSON: {filepath}")


def export_html(result: GraphAnalysisResult, filepath: Path):
    """Export results to HTML report"""
    
    # Count levels
    level_counts = result.criticality_by_level
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph Analysis Report</title>
    <style>
        :root {{
            --critical: #dc3545;
            --high: #fd7e14;
            --medium: #ffc107;
            --low: #28a745;
            --bg: #f8f9fa;
            --card-bg: #ffffff;
            --text: #212529;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: var(--bg);
            color: var(--text);
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: var(--text);
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: white;
            margin: 0;
        }}
        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: var(--card-bg);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .card h3 {{
            margin-top: 0;
            font-size: 0.9rem;
            color: #666;
        }}
        .card .value {{
            font-size: 2rem;
            font-weight: bold;
        }}
        .critical {{ color: var(--critical); }}
        .high {{ color: var(--high); }}
        .medium {{ color: var(--medium); }}
        .low {{ color: var(--low); }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f1f3f4;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }}
        .badge-critical {{ background: var(--critical); color: white; }}
        .badge-high {{ background: var(--high); color: white; }}
        .badge-medium {{ background: var(--medium); color: black; }}
        .badge-low {{ background: var(--low); color: white; }}
        .recommendation {{
            background: var(--card-bg);
            border-left: 4px solid var(--high);
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 0 10px 10px 0;
        }}
        .recommendation.critical {{
            border-left-color: var(--critical);
        }}
        .recommendation h4 {{
            margin-top: 0;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Graph Analysis Report</h1>
            <p class="timestamp">Generated: {result.timestamp}</p>
            <p>Input: {result.input_file}</p>
        </div>
        
        <h2>Overview</h2>
        <div class="cards">
            <div class="card">
                <h3>Total Nodes</h3>
                <div class="value">{result.total_nodes}</div>
            </div>
            <div class="card">
                <h3>Total Edges</h3>
                <div class="value">{result.total_edges}</div>
            </div>
            <div class="card">
                <h3>Critical Components</h3>
                <div class="value critical">{level_counts.get('CRITICAL', 0)}</div>
            </div>
            <div class="card">
                <h3>Articulation Points</h3>
                <div class="value high">{len(result.structural.articulation_points)}</div>
            </div>
        </div>
        
        <h2>Criticality Distribution</h2>
        <div class="cards">
            <div class="card">
                <h3>CRITICAL</h3>
                <div class="value critical">{level_counts.get('CRITICAL', 0)}</div>
            </div>
            <div class="card">
                <h3>HIGH</h3>
                <div class="value high">{level_counts.get('HIGH', 0)}</div>
            </div>
            <div class="card">
                <h3>MEDIUM</h3>
                <div class="value medium">{level_counts.get('MEDIUM', 0)}</div>
            </div>
            <div class="card">
                <h3>LOW</h3>
                <div class="value low">{level_counts.get('LOW', 0)}</div>
            </div>
        </div>
        
        <h2>Top Critical Components</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Component</th>
                    <th>Type</th>
                    <th>Score</th>
                    <th>Level</th>
                    <th>Primary Reason</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for i, score in enumerate(result.criticality_scores[:20], 1):
        level_class = score.criticality_level.lower()
        reason = score.reasons[0] if score.reasons else "N/A"
        html += f"""
                <tr>
                    <td>{i}</td>
                    <td><strong>{score.node_id}</strong></td>
                    <td>{score.node_type}</td>
                    <td>{score.composite_score:.4f}</td>
                    <td><span class="badge badge-{level_class}">{score.criticality_level}</span></td>
                    <td>{reason}</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
        
        <h2>Structural Analysis</h2>
        <div class="cards">
            <div class="card">
                <h3>Graph Density</h3>
                <div class="value">{density:.4f}</div>
            </div>
            <div class="card">
                <h3>Avg Clustering</h3>
                <div class="value">{clustering:.4f}</div>
            </div>
            <div class="card">
                <h3>Max K-Core</h3>
                <div class="value">{kcore}</div>
            </div>
            <div class="card">
                <h3>Bridge Edges</h3>
                <div class="value">{bridges}</div>
            </div>
        </div>
""".format(
        density=result.structural.density,
        clustering=result.structural.average_clustering,
        kcore=result.structural.k_core_max,
        bridges=len(result.structural.bridges)
    )
    
    if result.recommendations:
        html += """
        <h2>Recommendations</h2>
"""
        for rec in result.recommendations:
            priority_class = "critical" if rec['priority'] == 'CRITICAL' else ""
            html += f"""
        <div class="recommendation {priority_class}">
            <h4>[{rec['priority']}] {rec['category']}</h4>
            <p><strong>Issue:</strong> {rec['issue']}</p>
            <p><strong>Recommendation:</strong> {rec['recommendation']}</p>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    with open(filepath, 'w') as f:
        f.write(html)
    
    print_success(f"Exported HTML: {filepath}")


def export_csv(result: GraphAnalysisResult, output_dir: Path):
    """Export results to CSV files"""
    # Criticality scores CSV
    scores_path = output_dir / 'criticality_scores.csv'
    with open(scores_path, 'w') as f:
        f.write('node_id,type,betweenness,is_articulation_point,impact,composite_score,level,pagerank,closeness,degree\n')
        for s in result.criticality_scores:
            f.write(f"{s.node_id},{s.node_type},{s.betweenness_centrality:.6f},"
                   f"{s.is_articulation_point},{s.impact_score:.6f},"
                   f"{s.composite_score:.6f},{s.criticality_level},"
                   f"{s.pagerank:.6f},{s.closeness:.6f},{s.degree}\n")
    print_success(f"Exported CSV: {scores_path}")
    
    # Structural analysis CSV
    struct_path = output_dir / 'structural_analysis.csv'
    with open(struct_path, 'w') as f:
        f.write('metric,value\n')
        f.write(f"articulation_points,{len(result.structural.articulation_points)}\n")
        f.write(f"bridges,{len(result.structural.bridges)}\n")
        f.write(f"density,{result.structural.density:.6f}\n")
        f.write(f"diameter,{result.structural.diameter or 'N/A'}\n")
        f.write(f"avg_clustering,{result.structural.average_clustering:.6f}\n")
        f.write(f"k_core_max,{result.structural.k_core_max}\n")
    print_success(f"Exported CSV: {struct_path}")


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
    
    # Check TTY for colors
    if not sys.stdout.isatty() or args.no_color:
        Colors.disable()
    
    # Print header
    if not args.quiet:
        print_header("ENHANCED GRAPH ANALYZER")
        print(f"\n  Input: {args.input}")
        print(f"  Weights: α={args.alpha}, β={args.beta}, γ={args.gamma}")
        if args.relationship_analysis or args.full:
            print(f"  Relationship Analysis: Enabled")
    
    try:
        # Load graph
        if not args.quiet:
            print_section("Loading Graph")
        
        G = build_graph_from_file(args.input)
        
        if not args.quiet:
            print_success(f"Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Validate graph
        if G.number_of_nodes() == 0:
            logger.error("Graph is empty")
            return 1
        
        # Count by type
        nodes_by_type = {}
        for node in G.nodes():
            ntype = G.nodes[node].get('type', 'Unknown')
            nodes_by_type[ntype] = nodes_by_type.get(ntype, 0) + 1
        
        edges_by_type = {}
        for u, v in G.edges():
            etype = G.edges[u, v].get('type', 'Unknown')
            edges_by_type[etype] = edges_by_type.get(etype, 0) + 1
        
        if not args.quiet:
            print_subsection("Nodes by Type:")
            for ntype, count in nodes_by_type.items():
                print_info(f"{ntype}: {count}")
        
        # Structural analysis
        if not args.quiet:
            print_section("Structural Analysis")
        
        start_time = datetime.now()
        structural = analyze_structure(G)
        
        if not args.quiet:
            print_metric("Articulation Points", len(structural.articulation_points))
            print_metric("Bridges", len(structural.bridges))
            print_metric("Density", f"{structural.density:.4f}")
            print_metric("Max K-Core", structural.k_core_max)
        
        # Criticality scoring
        if not args.quiet:
            print_section("Criticality Analysis")
        
        scores = calculate_criticality_scores(G, args.alpha, args.beta, args.gamma)
        
        # Count by level
        level_counts = {}
        for s in scores:
            level_counts[s.criticality_level] = level_counts.get(s.criticality_level, 0) + 1
        
        if not args.quiet:
            for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                count = level_counts.get(level, 0)
                if level == 'CRITICAL' and count > 0:
                    print_critical(f"{level}: {count}")
                elif level == 'HIGH' and count > 0:
                    print_warning(f"{level}: {count}")
                else:
                    print_info(f"{level}: {count}")
            
            print_subsection("Top 5 Critical Components:")
            for i, s in enumerate(scores[:5], 1):
                level_color = Colors.FAIL if s.criticality_level == 'CRITICAL' else Colors.WARNING
                print(f"    {i}. {level_color}{s.node_id}{Colors.ENDC} "
                      f"({s.node_type}) - Score: {s.composite_score:.4f}")
        
        # Relationship analysis
        relationship_results = None
        if args.relationship_analysis or args.full:
            if not args.quiet:
                print_section("Relationship Analysis")
            
            try:
                from src.analysis.relationship_analyzer import RelationshipAnalyzer
                
                rel_analyzer = RelationshipAnalyzer(G)
                rel_result = rel_analyzer.analyze()
                relationship_results = rel_analyzer.to_dict(rel_result)
                
                if not args.quiet:
                    print_success(f"Analyzed {len(rel_result.edge_criticality)} edges")
                    print_metric("Critical Edges", len(rel_result.critical_edges))
                    print_metric("Bridge Edges", len(rel_result.bridge_edges))
                    print_metric("Network Motifs", len(rel_result.motifs))
                    
                    print_subsection("Top Hubs (Data Sources):")
                    for hub in rel_result.top_hubs[:5]:
                        print_info(hub)
                    
                    print_subsection("Top Authorities (Data Sinks):")
                    for auth in rel_result.top_authorities[:5]:
                        print_info(auth)
                    
                    if rel_result.motif_summary:
                        print_subsection("Motif Summary:")
                        for motif_type, count in rel_result.motif_summary.items():
                            print_info(f"{motif_type}: {count}")
                    
            except ImportError as e:
                logger.warning(f"Relationship analyzer not available: {e}")
        
        # Generate recommendations
        recommendations = generate_recommendations(G, structural, scores)
        
        if relationship_results and 'recommendations' in relationship_results:
            recommendations.extend(relationship_results['recommendations'])
        
        if not args.quiet and recommendations:
            print_section("Recommendations")
            for rec in recommendations[:5]:
                if rec['priority'] == 'CRITICAL':
                    print_critical(f"[{rec['priority']}] {rec['category']}: {rec['issue']}")
                else:
                    print_warning(f"[{rec['priority']}] {rec['category']}: {rec['issue']}")
                print_info(rec['recommendation'])
        
        # Create result object
        duration = (datetime.now() - start_time).total_seconds()
        
        result = GraphAnalysisResult(
            timestamp=datetime.now().isoformat(),
            input_file=str(args.input),
            total_nodes=G.number_of_nodes(),
            total_edges=G.number_of_edges(),
            nodes_by_type=nodes_by_type,
            edges_by_type=edges_by_type,
            criticality_scores=scores,
            criticality_by_level=level_counts,
            structural=structural,
            relationship_analysis=relationship_results,
            recommendations=recommendations
        )
        
        if not args.quiet:
            print_section("Summary")
            print_success(f"Analysis completed in {duration:.2f}s")
        
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
        
        # Print JSON to stdout if requested
        if args.json_output:
            def serialize(obj):
                if hasattr(obj, '__dict__'):
                    return {k: serialize(v) for k, v in obj.__dict__.items()}
                elif isinstance(obj, list):
                    return [serialize(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: serialize(v) for k, v in obj.items()}
                elif isinstance(obj, tuple):
                    return list(obj)
                else:
                    return obj
            print(json.dumps(serialize(result), indent=2, default=str))
        
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


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Analyze pub-sub system graphs with advanced graph algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis
    python analyze_graph.py --input system.json
    
    # With relationship analysis
    python analyze_graph.py --input system.json --relationship-analysis
    
    # Full analysis with all features
    python analyze_graph.py --input system.json --full
    
    # Export to multiple formats
    python analyze_graph.py --input system.json --output-dir results/ --format json html csv
    
    # Custom criticality weights
    python analyze_graph.py --input system.json --alpha 0.5 --beta 0.25 --gamma 0.25

Algorithm Recommendations:
    Run with --algorithms flag to see recommended graph algorithms for pub-sub analysis.
        """
    )
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('--input', '-i',
                             help='Input graph file (JSON)')
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument('--alpha', type=float, default=0.4,
                                help='Weight for betweenness centrality (default: 0.4)')
    analysis_group.add_argument('--beta', type=float, default=0.3,
                                help='Weight for articulation point indicator (default: 0.3)')
    analysis_group.add_argument('--gamma', type=float, default=0.3,
                                help='Weight for impact score (default: 0.3)')
    analysis_group.add_argument('--relationship-analysis', '-r', action='store_true',
                                help='Enable relationship analysis (edges, HITS, motifs)')
    analysis_group.add_argument('--full', '-f', action='store_true',
                                help='Enable all analysis features')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output-dir', '-o',
                              help='Output directory for results')
    output_group.add_argument('--format', nargs='+', default=['json'],
                              choices=['json', 'html', 'csv'],
                              help='Output formats (default: json)')
    output_group.add_argument('--json-output', '-j', action='store_true',
                              help='Print JSON to stdout')
    
    # Display options
    display_group = parser.add_argument_group('Display Options')
    display_group.add_argument('--verbose', '-v', action='store_true',
                               help='Verbose output')
    display_group.add_argument('--quiet', '-q', action='store_true',
                               help='Quiet mode (minimal output)')
    display_group.add_argument('--no-color', action='store_true',
                               help='Disable colored output')
    
    # Info options
    info_group = parser.add_argument_group('Information')
    info_group.add_argument('--algorithms', action='store_true',
                            help='Print algorithm recommendations')
    
    args = parser.parse_args()
    
    # Print algorithm recommendations if requested
    if args.algorithms:
        print("\n" + "="*70)
        print("  GRAPH ALGORITHM RECOMMENDATIONS FOR PUB-SUB ANALYSIS")
        print("="*70 + "\n")
        
        try:
            from src.analysis.relationship_analyzer import get_algorithm_recommendations
            
            for rec in get_algorithm_recommendations():
                print(f"[Priority {rec['priority']}] {rec['algorithm']} ({rec['category']})")
                print(f"    Purpose: {rec['purpose']}")
                print(f"    Pub-Sub: {rec['pub_sub_application']}")
                print(f"    Complexity: {rec['complexity']}\n")
        except ImportError:
            print("Relationship analyzer module not available.")
        
        return 0
    
    # Validate input is provided for analysis
    if not args.input:
        parser.error("--input/-i is required when not using --algorithms")
    
    return run_analysis(args)


if __name__ == '__main__':
    sys.exit(main())