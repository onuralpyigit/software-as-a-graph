#!/usr/bin/env python3
"""
Graph Analyzer CLI
===========================

Comprehensive analysis of distributed pub-sub systems for:
- Reliability (SPOFs, cascade risks, redundancy)
- Maintainability (coupling, anti-patterns, modularity)
- Availability (connectivity, fault tolerance, recovery)

Supports loading from:
- Neo4j database

Usage:
    # Basic analysis
    python analyze_graph.py
    
    # Load from Neo4j database
    python analyze_graph.py --uri bolt://localhost:7687
    
    # Full analysis with all exports
    python analyze_graph.py --full --output results/
    
    # Specific quality attributes
    python analyze_graph.py --reliability --maintainability

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    import networkx as nx
except ImportError:
    print("ERROR: NetworkX is required. Install with: pip install networkx")
    sys.exit(1)

# Import analyzers
from src.analysis.quality_attribute_analyzer import (
    QualityAttribute,
    ComprehensiveAnalysisResult,
    QualityIssue,
    CriticalComponent,
    CriticalEdge,
    Severity,
    IssueCategory
)
from src.analysis.reliability_analyzer import ReliabilityAnalyzer
from src.analysis.maintainability_analyzer import MaintainabilityAnalyzer
from src.analysis.availability_analyzer import AvailabilityAnalyzer
from src.analysis.antipattern_detector import AntiPatternDetector, AntiPatternAnalysisResult
from src.analysis.criticality_classifier import (
    BoxPlotCriticalityClassifier,
    ClassificationResult,
    CriticalityLevel
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
        """Disable colors for non-TTY output"""
        cls.HEADER = ''
        cls.BLUE = ''
        cls.CYAN = ''
        cls.GREEN = ''
        cls.WARNING = ''
        cls.RED = ''
        cls.ENDC = ''
        cls.BOLD = ''
        cls.DIM = ''


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
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.ENDC} {text}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.ENDC} {text}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {text}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ{Colors.ENDC} {text}")


# ============================================================================
# Graph Building
# ============================================================================

class DependsOnGraphBuilder:
    """
    Builds a DEPENDS_ON graph from pub-sub system data.
    
    Derives dependency relationships from:
    - PUBLISHES_TO / SUBSCRIBES_TO -> APP_TO_APP dependencies
    - RUNS_ON -> NODE_TO_BROKER, APP_TO_NODE dependencies
    - ROUTES -> TOPIC_TO_BROKER dependencies
    """
    
    def __init__(self):
        self.logger = logging.getLogger('GraphBuilder')
    
    def build_from_json(self, filepath: str) -> nx.DiGraph:
        """Build DEPENDS_ON graph from JSON file"""
        with open(filepath) as f:
            data = json.load(f)
        return self.build_from_dict(data)
    
    def build_from_dict(self, data: Dict[str, Any]) -> nx.DiGraph:
        """Build DEPENDS_ON graph from dictionary"""
        G = nx.DiGraph()
        
        # Build indexes
        app_to_node = {}
        broker_to_node = {}
        topic_to_broker = {}
        app_publishes = defaultdict(set)
        app_subscribes = defaultdict(set)
        topic_publishers = defaultdict(set)
        topic_subscribers = defaultdict(set)
        
        # Add nodes (infrastructure)
        for node in data.get('nodes', []):
            node_id = node.get('id', node.get('name'))
            G.add_node(node_id, type='Node', **node)
        
        # Add brokers
        for broker in data.get('brokers', []):
            broker_id = broker.get('id', broker.get('name'))
            G.add_node(broker_id, type='Broker', **broker)
            if 'node' in broker:
                broker_to_node[broker_id] = broker['node']
        
        # Add topics
        for topic in data.get('topics', []):
            topic_id = topic.get('id', topic.get('name'))
            G.add_node(topic_id, type='Topic', **topic)
            if 'broker' in topic:
                topic_to_broker[topic_id] = topic['broker']
        
        # Add applications
        for app in data.get('applications', []):
            app_id = app.get('id', app.get('name'))
            G.add_node(app_id, type='Application', **app)
            if 'node' in app:
                app_to_node[app_id] = app['node']
        
        # Process relationships
        relationships = data.get('relationships', {})
        
        # PUBLISHES_TO relationships
        for rel in relationships.get('publishes_to', data.get('publishes', [])):
            app_id = rel.get('from', rel.get('source', rel.get('app')))
            topic_id = rel.get('to', rel.get('target', rel.get('topic')))
            if app_id and topic_id:
                app_publishes[app_id].add(topic_id)
                topic_publishers[topic_id].add(app_id)
        
        # SUBSCRIBES_TO relationships
        for rel in relationships.get('subscribes_to', data.get('subscribes', [])):
            app_id = rel.get('from', rel.get('source', rel.get('app')))
            topic_id = rel.get('to', rel.get('target', rel.get('topic')))
            if app_id and topic_id:
                app_subscribes[app_id].add(topic_id)
                topic_subscribers[topic_id].add(app_id)
        
        # Derive DEPENDS_ON relationships
        
        # 1. APP_TO_APP: Subscriber depends on publisher via topic
        for topic_id in topic_publishers:
            publishers = topic_publishers[topic_id]
            subscribers = topic_subscribers.get(topic_id, set())
            
            for subscriber in subscribers:
                for publisher in publishers:
                    if subscriber != publisher:
                        self._add_depends_on(G, subscriber, publisher,
                                           'app_to_app',
                                           via_topic=topic_id)
        
        # 2. APP_TO_BROKER: App depends on broker for its topics
        for app_id in list(app_publishes.keys()) + list(app_subscribes.keys()):
            topics = app_publishes.get(app_id, set()) | app_subscribes.get(app_id, set())
            for topic_id in topics:
                broker_id = topic_to_broker.get(topic_id)
                if broker_id:
                    self._add_depends_on(G, app_id, broker_id,
                                       'app_to_broker',
                                       via_topic=topic_id)
        
        # 3. NODE_TO_NODE: Infrastructure dependencies via apps
        for app_id, node_id in app_to_node.items():
            # Get apps this app depends on
            for topic in app_subscribes.get(app_id, set()):
                for publisher in topic_publishers.get(topic, set()):
                    pub_node = app_to_node.get(publisher)
                    if pub_node and pub_node != node_id:
                        self._add_depends_on(G, node_id, pub_node,
                                           'node_to_node',
                                           via_apps=[app_id, publisher])
        
        # 4. BROKER_TO_NODE: Broker depends on its host
        for broker_id, node_id in broker_to_node.items():
            self._add_depends_on(G, broker_id, node_id,
                               'broker_to_node')
        
        # 5. TOPIC_TO_BROKER: Topic depends on its broker
        for topic_id, broker_id in topic_to_broker.items():
            self._add_depends_on(G, topic_id, broker_id,
                               'topic_to_broker')
        
        self.logger.info(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def _add_depends_on(self, G: nx.DiGraph, source: str, target: str,
                        dep_type: str, **attrs):
        """Add DEPENDS_ON edge if not exists"""
        if source in G and target in G:
            if not G.has_edge(source, target):
                G.add_edge(source, target, dependency_type=dep_type, **attrs)
            else:
                # Update existing edge
                G[source][target]['dependency_type'] = dep_type
                G[source][target].update(attrs)
    
    def build_from_neo4j(self, uri: str, user: str, password: str,
                         database: str = 'neo4j') -> nx.DiGraph:
        """Build DEPENDS_ON graph from Neo4j database"""
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("neo4j driver required: pip install neo4j")
        
        self.logger.info(f"Connecting to Neo4j at {uri}...")
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        G = nx.DiGraph()
        
        try:
            with driver.session(database=database) as session:
                # Load all nodes
                self.logger.info("Loading nodes from Neo4j...")
                
                result = session.run("""
                    MATCH (n)
                    WHERE n:Application OR n:Topic OR n:Broker OR n:Node
                    RETURN n.id AS id, n.name AS name, labels(n)[0] AS type,
                           properties(n) AS props
                """)
                
                for record in result:
                    node_id = record['id'] or record['name']
                    if node_id:
                        props = dict(record['props'])
                        props['type'] = record['type']
                        G.add_node(node_id, **props)
                
                self.logger.info(f"Loaded {G.number_of_nodes()} nodes")
                
                # Load DEPENDS_ON edges (if they exist)
                result = session.run("""
                    MATCH (a)-[d:DEPENDS_ON]->(b)
                    RETURN a.id AS source, b.id AS target,
                           d.dependency_type AS dep_type,
                           properties(d) AS props
                """)
                
                depends_on_count = 0
                for record in result:
                    source = record['source']
                    target = record['target']
                    if source and target and source in G and target in G:
                        props = dict(record['props']) if record['props'] else {}
                        props['dependency_type'] = record['dep_type'] or 'unknown'
                        G.add_edge(source, target, **props)
                        depends_on_count += 1
                
                # If no DEPENDS_ON edges, derive from base relationships
                if depends_on_count == 0:
                    self.logger.info("No DEPENDS_ON edges found. Deriving from base relationships...")
                    G = self._derive_from_neo4j(session, G)
                else:
                    self.logger.info(f"Loaded {depends_on_count} DEPENDS_ON edges")
                
        finally:
            driver.close()
        
        return G
    
    def _derive_from_neo4j(self, session, G: nx.DiGraph) -> nx.DiGraph:
        """Derive DEPENDS_ON from base Neo4j relationships"""
        
        # APP_TO_APP via topics
        result = session.run("""
            MATCH (sub:Application)-[:SUBSCRIBES_TO]->(t:Topic)<-[:PUBLISHES_TO]-(pub:Application)
            WHERE sub <> pub
            RETURN sub.id AS subscriber, pub.id AS publisher, t.id AS topic
        """)
        
        for record in result:
            if record['subscriber'] in G and record['publisher'] in G:
                self._add_depends_on(G, record['subscriber'], record['publisher'],
                                   'app_to_app', via_topic=record['topic'])
        
        # APP_TO_BROKER
        result = session.run("""
            MATCH (a:Application)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)-[:ROUTES]->(b:Broker)
            RETURN DISTINCT a.id AS app, b.id AS broker
        """)
        
        for record in result:
            if record['app'] in G and record['broker'] in G:
                self._add_depends_on(G, record['app'], record['broker'],
                                   'app_to_broker')
        
        # BROKER_TO_NODE
        result = session.run("""
            MATCH (b:Broker)-[:RUNS_ON]->(n:Node)
            RETURN b.id AS broker, n.id AS node
        """)
        
        for record in result:
            if record['broker'] in G and record['node'] in G:
                self._add_depends_on(G, record['broker'], record['node'],
                                   'broker_to_node')
        
        self.logger.info(f"Derived {G.number_of_edges()} DEPENDS_ON edges")
        
        return G


# ============================================================================
# Analysis Orchestrator
# ============================================================================

class QualityAnalysisOrchestrator:
    """
    Orchestrates quality attribute analysis across multiple analyzers.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger('AnalysisOrchestrator')
        
        # Initialize analyzers
        self.reliability_analyzer = ReliabilityAnalyzer(config)
        self.maintainability_analyzer = MaintainabilityAnalyzer(config)
        self.availability_analyzer = AvailabilityAnalyzer(config)
        self.antipattern_detector = AntiPatternDetector(config)
    
    def analyze(self, graph: nx.DiGraph,
                analyze_reliability: bool = True,
                analyze_maintainability: bool = True,
                analyze_availability: bool = True,
                detect_antipatterns: bool = True) -> ComprehensiveAnalysisResult:
        """
        Perform comprehensive quality analysis.
        
        Args:
            graph: NetworkX DiGraph with DEPENDS_ON relationships
            analyze_reliability: Include reliability analysis
            analyze_maintainability: Include maintainability analysis
            analyze_availability: Include availability analysis
            detect_antipatterns: Include anti-pattern detection
            
        Returns:
            ComprehensiveAnalysisResult with all findings
        """
        self.logger.info("Starting comprehensive quality analysis...")
        
        # Graph summary
        graph_summary = self._compute_graph_summary(graph)
        
        # Run enabled analyses
        reliability_result = None
        maintainability_result = None
        availability_result = None
        antipattern_result = None
        
        if analyze_reliability:
            self.logger.info("\n" + "="*50)
            self.logger.info("RELIABILITY ANALYSIS")
            self.logger.info("="*50)
            reliability_result = self.reliability_analyzer.analyze(graph)
        
        if analyze_maintainability:
            self.logger.info("\n" + "="*50)
            self.logger.info("MAINTAINABILITY ANALYSIS")
            self.logger.info("="*50)
            maintainability_result = self.maintainability_analyzer.analyze(graph)
        
        if analyze_availability:
            self.logger.info("\n" + "="*50)
            self.logger.info("AVAILABILITY ANALYSIS")
            self.logger.info("="*50)
            availability_result = self.availability_analyzer.analyze(graph)
        
        if detect_antipatterns:
            self.logger.info("\n" + "="*50)
            self.logger.info("ANTI-PATTERN DETECTION")
            self.logger.info("="*50)
            antipattern_result = self.antipattern_detector.analyze(graph)
        
        # Calculate overall health score
        scores = []
        if reliability_result:
            scores.append(reliability_result.score * 1.2)  # Weight reliability higher
        if maintainability_result:
            scores.append(maintainability_result.score)
        if availability_result:
            scores.append(availability_result.score * 1.1)  # Weight availability
        
        overall_score = sum(scores) / len(scores) if scores else 0
        overall_score = max(0, min(100, overall_score))
        
        # Aggregate critical components
        all_critical = []
        if reliability_result:
            all_critical.extend(reliability_result.critical_components)
        if maintainability_result:
            all_critical.extend(maintainability_result.critical_components)
        if availability_result:
            all_critical.extend(availability_result.critical_components)
        
        # Deduplicate and sort
        seen = set()
        unique_critical = []
        for comp in sorted(all_critical, key=lambda x: -x.criticality_score):
            if comp.component_id not in seen:
                seen.add(comp.component_id)
                unique_critical.append(comp)
        
        # Aggregate critical edges
        all_critical_edges = []
        if reliability_result:
            all_critical_edges.extend(reliability_result.critical_edges)
        if maintainability_result:
            all_critical_edges.extend(maintainability_result.critical_edges)
        if availability_result:
            all_critical_edges.extend(availability_result.critical_edges)
        
        all_critical_edges.sort(key=lambda x: -x.criticality_score)
        
        # Generate prioritized recommendations
        recommendations = self._prioritize_recommendations(
            reliability_result,
            maintainability_result,
            availability_result,
            antipattern_result
        )
        
        return ComprehensiveAnalysisResult(
            reliability=reliability_result,
            maintainability=maintainability_result,
            availability=availability_result,
            graph_summary=graph_summary,
            overall_health_score=overall_score,
            top_critical_components=unique_critical[:20],
            top_critical_edges=all_critical_edges[:20],
            prioritized_recommendations=recommendations,
            antipatterns=antipattern_result
        )
    
    def _compute_graph_summary(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Compute graph summary statistics"""
        # Node counts by type
        node_types = defaultdict(int)
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'Unknown')
            node_types[node_type] += 1
        
        # Edge counts by type
        edge_types = defaultdict(int)
        for u, v, data in graph.edges(data=True):
            edge_type = data.get('dependency_type', 'unknown')
            edge_types[edge_type] += 1
        
        return {
            'total_nodes': graph.number_of_nodes(),
            'total_edges': graph.number_of_edges(),
            'nodes_by_type': dict(node_types),
            'edges_by_type': dict(edge_types),
            'density': nx.density(graph),
            'is_connected': nx.is_weakly_connected(graph) if graph.number_of_nodes() > 0 else True,
            'connected_components': nx.number_weakly_connected_components(graph)
        }
    
    def _prioritize_recommendations(self, reliability, maintainability, availability, antipatterns=None) -> List[str]:
        """Generate prioritized recommendations across all analyses"""
        recommendations = []
        
        # Count critical issues
        critical_count = 0
        if reliability:
            critical_count += len(reliability.get_critical_issues())
        if maintainability:
            critical_count += len(maintainability.get_critical_issues())
        if availability:
            critical_count += len(availability.get_critical_issues())
        
        # Count critical antipatterns
        critical_antipatterns = 0
        if antipatterns and hasattr(antipatterns, 'summary'):
            critical_antipatterns = antipatterns.summary.get('critical_count', 0)
        
        if critical_count > 0:
            recommendations.append(
                f"🔴 URGENT: {critical_count} critical issues require immediate attention"
            )
        
        if critical_antipatterns > 0:
            recommendations.append(
                f"🔴 ARCHITECTURAL: {critical_antipatterns} critical anti-patterns detected"
            )
        
        # Add recommendations from each analyzer
        if reliability and reliability.recommendations:
            recommendations.extend(reliability.recommendations[:3])
        
        if availability and availability.recommendations:
            recommendations.extend(availability.recommendations[:2])
        
        if maintainability and maintainability.recommendations:
            recommendations.extend(maintainability.recommendations[:2])
        
        # Add antipattern recommendations
        if antipatterns and hasattr(antipatterns, 'recommendations'):
            recommendations.extend(antipatterns.recommendations[:2])
        
        return recommendations[:12]


# ============================================================================
# Output Formatters
# ============================================================================

def print_analysis_summary(result: ComprehensiveAnalysisResult, verbose: bool = False):
    """Print analysis summary to terminal"""
    
    print_section("GRAPH SUMMARY")
    gs = result.graph_summary
    print(f"  Total Nodes:  {gs['total_nodes']}")
    print(f"  Total Edges:  {gs['total_edges']}")
    print(f"  Density:      {gs['density']:.4f}")
    print(f"  Connected:    {'Yes' if gs['is_connected'] else 'No'}")
    
    print("\n  Nodes by Type:")
    for node_type, count in gs['nodes_by_type'].items():
        print(f"    {node_type}: {count}")
    
    print("\n  Edges by Type:")
    for edge_type, count in gs['edges_by_type'].items():
        print(f"    {edge_type}: {count}")
    
    # Overall Health Score
    print_section("OVERALL HEALTH SCORE")
    score = result.overall_health_score
    if score >= 80:
        color = Colors.GREEN
        status = "HEALTHY"
    elif score >= 60:
        color = Colors.WARNING
        status = "MODERATE"
    else:
        color = Colors.RED
        status = "CRITICAL"
    
    print(f"\n  {color}{Colors.BOLD}  {score:.1f}/100 - {status}  {Colors.ENDC}")
    
    # Individual Scores
    print_section("QUALITY ATTRIBUTE SCORES")
    
    def print_score(name: str, result, emoji: str):
        if result:
            score = result.score
            issues = len(result.issues)
            critical = len(result.get_critical_issues())
            
            if score >= 80:
                status_color = Colors.GREEN
            elif score >= 60:
                status_color = Colors.WARNING
            else:
                status_color = Colors.RED
            
            print(f"\n  {emoji} {name}")
            print(f"     Score: {status_color}{score:.1f}/100{Colors.ENDC}")
            print(f"     Issues: {issues} total, {critical} critical")
    
    print_score("RELIABILITY", result.reliability, "🛡️")
    print_score("MAINTAINABILITY", result.maintainability, "🔧")
    print_score("AVAILABILITY", result.availability, "⬆️")
    
    # Anti-Patterns Summary
    if result.antipatterns and hasattr(result.antipatterns, 'summary'):
        ap = result.antipatterns
        print(f"\n  🏗️ ANTI-PATTERNS")
        print(f"     Total: {ap.summary.get('total_patterns', 0)} detected")
        print(f"     Critical: {ap.summary.get('critical_count', 0)}, High: {ap.summary.get('high_count', 0)}")
    
    # Top Critical Components
    if result.top_critical_components:
        print_section("TOP CRITICAL COMPONENTS")
        for i, comp in enumerate(result.top_critical_components[:10], 1):
            qa = comp.quality_attribute.value
            print(f"  {i}. {comp.component_id} ({comp.component_type.value})")
            print(f"     Criticality: {comp.criticality_score:.3f} [{qa}]")
            print(f"     Reasons: {', '.join(comp.reasons[:3])}")
    
    # Top Critical Edges
    if result.top_critical_edges:
        print_section("TOP CRITICAL EDGES")
        for i, edge in enumerate(result.top_critical_edges[:5], 1):
            print(f"  {i}. {edge.source} → {edge.target}")
            print(f"     Type: {edge.edge_type}")
            print(f"     Criticality: {edge.criticality_score:.3f}")
    
    # Prioritized Recommendations
    print_section("PRIORITIZED RECOMMENDATIONS")
    for rec in result.prioritized_recommendations[:7]:
        print(f"  • {rec}")
    
    # Detailed Issues (if verbose)
    if verbose:
        for qa_name, qa_result in [("Reliability", result.reliability),
                                    ("Maintainability", result.maintainability),
                                    ("Availability", result.availability)]:
            if qa_result and qa_result.issues:
                print_section(f"{qa_name.upper()} ISSUES")
                for issue in qa_result.issues[:10]:
                    severity_colors = {
                        Severity.CRITICAL: Colors.RED,
                        Severity.HIGH: Colors.WARNING,
                        Severity.MEDIUM: Colors.BLUE,
                        Severity.LOW: Colors.DIM
                    }
                    color = severity_colors.get(issue.severity, '')
                    print(f"\n  {color}[{issue.severity.value.upper()}]{Colors.ENDC} {issue.category.value}")
                    print(f"    {issue.description}")
                    if issue.recommendation:
                        print(f"    → {issue.recommendation}")
        
        # Anti-pattern details
        if result.antipatterns and hasattr(result.antipatterns, 'patterns'):
            print_section("ANTI-PATTERNS DETECTED")
            for pattern in result.antipatterns.patterns[:10]:
                severity_colors = {
                    'critical': Colors.RED,
                    'high': Colors.WARNING,
                    'medium': Colors.BLUE,
                    'low': Colors.DIM
                }
                color = severity_colors.get(pattern.severity.value, '')
                print(f"\n  {color}[{pattern.severity.value.upper()}]{Colors.ENDC} {pattern.pattern_type.value}")
                print(f"    {pattern.description}")
                print(f"    Impact: {pattern.impact}")
                if pattern.recommendation:
                    print(f"    → {pattern.recommendation}")


def export_json(result: ComprehensiveAnalysisResult, filepath: Path):
    """Export results to JSON"""
    with open(filepath, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    print_success(f"Exported JSON: {filepath}")


def export_csv(result: ComprehensiveAnalysisResult, output_dir: Path):
    """Export results to CSV files"""
    import csv
    
    # Critical components CSV
    comp_file = output_dir / 'critical_components.csv'
    with open(comp_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['component_id', 'type', 'quality_attribute', 
                        'criticality_score', 'reasons'])
        for comp in result.top_critical_components:
            writer.writerow([
                comp.component_id,
                comp.component_type.value,
                comp.quality_attribute.value,
                f"{comp.criticality_score:.4f}",
                '; '.join(comp.reasons)
            ])
    print_success(f"Exported CSV: {comp_file}")
    
    # Issues CSV
    issues_file = output_dir / 'issues.csv'
    all_issues = []
    for qa_result in [result.reliability, result.maintainability, result.availability]:
        if qa_result:
            all_issues.extend(qa_result.issues)
    
    with open(issues_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['issue_id', 'quality_attribute', 'category', 
                        'severity', 'description', 'recommendation'])
        for issue in all_issues:
            writer.writerow([
                issue.issue_id,
                issue.quality_attribute.value,
                issue.category.value,
                issue.severity.value,
                issue.description,
                issue.recommendation
            ])
    print_success(f"Exported CSV: {issues_file}")


def export_html(result: ComprehensiveAnalysisResult, filepath: Path):
    """Export results to HTML report"""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Quality Analysis Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .card {{ background: white; border-radius: 10px; padding: 20px; 
                margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .score {{ font-size: 48px; font-weight: bold; }}
        .score-good {{ color: #27ae60; }}
        .score-moderate {{ color: #f39c12; }}
        .score-poor {{ color: #e74c3c; }}
        .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
        .metric-card {{ text-align: center; padding: 20px; }}
        .metric-value {{ font-size: 36px; font-weight: bold; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; 
                 font-size: 12px; font-weight: 600; }}
        .badge-critical {{ background: #fee; color: #c00; }}
        .badge-high {{ background: #fff3cd; color: #856404; }}
        .badge-medium {{ background: #cce5ff; color: #004085; }}
        .badge-low {{ background: #d4edda; color: #155724; }}
        .recommendation {{ padding: 10px; margin: 5px 0; background: #e8f4f8; 
                         border-left: 3px solid #3498db; border-radius: 4px; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🔍 Quality Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="card">
        <h2>Overall Health Score</h2>
        <div class="score {'score-good' if result.overall_health_score >= 80 else 'score-moderate' if result.overall_health_score >= 60 else 'score-poor'}">
            {result.overall_health_score:.1f}/100
        </div>
    </div>
    
    <div class="grid">
"""
    
    # Add quality attribute cards
    for name, qa_result, icon in [
        ("Reliability", result.reliability, "🛡️"),
        ("Maintainability", result.maintainability, "🔧"),
        ("Availability", result.availability, "⬆️")
    ]:
        if qa_result:
            score_class = 'score-good' if qa_result.score >= 80 else 'score-moderate' if qa_result.score >= 60 else 'score-poor'
            html += f"""
        <div class="card metric-card">
            <div class="metric-value {score_class}">{qa_result.score:.0f}</div>
            <div class="metric-label">{icon} {name}</div>
            <div style="margin-top:10px; color:#666;">
                {len(qa_result.issues)} issues found
            </div>
        </div>
"""
    
    html += """
    </div>
    
    <div class="card">
        <h2>📊 Graph Summary</h2>
        <div class="grid">
"""
    
    gs = result.graph_summary
    for label, value in [
        ("Total Nodes", gs['total_nodes']),
        ("Total Edges", gs['total_edges']),
        ("Graph Density", f"{gs['density']:.4f}")
    ]:
        html += f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:24px;">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
"""
    
    html += """
        </div>
    </div>
    
    <div class="card">
        <h2>🎯 Critical Components</h2>
        <table>
            <thead>
                <tr>
                    <th>Component</th>
                    <th>Type</th>
                    <th>Quality Attribute</th>
                    <th>Criticality</th>
                    <th>Reasons</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for comp in result.top_critical_components[:15]:
        html += f"""
                <tr>
                    <td><strong>{comp.component_id}</strong></td>
                    <td>{comp.component_type.value}</td>
                    <td>{comp.quality_attribute.value}</td>
                    <td>{comp.criticality_score:.3f}</td>
                    <td>{', '.join(comp.reasons[:3])}</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
    </div>
    
    <div class="card">
        <h2>✅ Prioritized Recommendations</h2>
"""
    
    for rec in result.prioritized_recommendations[:10]:
        html += f"""
        <div class="recommendation">{rec}</div>
"""
    
    html += """
    </div>
</div>
</body>
</html>
"""
    
    with open(filepath, 'w') as f:
        f.write(html)
    print_success(f"Exported HTML: {filepath}")


def print_boxplot_summary(result: ClassificationResult):
    """Print box-plot classification summary"""
    print_section("BOX-PLOT CLASSIFICATION SUMMARY")
    
    if result.component_stats:
        stats = result.component_stats
        print(f"\n  {Colors.BOLD}Score Distribution:{Colors.ENDC}")
        print(f"    Range:      [{stats.min_value:.4f}, {stats.max_value:.4f}]")
        print(f"    Mean:       {stats.mean:.4f}")
        print(f"    Median:     {stats.median:.4f}")
        print(f"    Std Dev:    {stats.std_dev:.4f}")
        
        print(f"\n  {Colors.BOLD}Quartiles (Box-Plot Thresholds):{Colors.ENDC}")
        print(f"    Q1 (25%):   {stats.q1:.4f}")
        print(f"    Q2 (50%):   {stats.median:.4f}")
        print(f"    Q3 (75%):   {stats.q3:.4f}")
        print(f"    IQR:        {stats.iqr:.4f}")
        
        print(f"\n  {Colors.BOLD}Classification Boundaries:{Colors.ENDC}")
        print(f"    CRITICAL:   > {stats.upper_fence:.4f} (outliers)")
        print(f"    HIGH:       ({stats.q3:.4f}, {stats.upper_fence:.4f}]")
        print(f"    MEDIUM:     ({stats.median:.4f}, {stats.q3:.4f}]")
        print(f"    LOW:        ({stats.q1:.4f}, {stats.median:.4f}]")
        print(f"    MINIMAL:    <= {stats.q1:.4f}")
    
    summary = result.summary
    print(f"\n  {Colors.BOLD}Distribution by Level:{Colors.ENDC}")
    
    level_colors = {
        'critical': Colors.RED,
        'high': Colors.WARNING,
        'medium': Colors.BLUE,
        'low': Colors.CYAN,
        'minimal': Colors.DIM
    }
    
    total = summary['total_components']
    max_count = max(summary['components_by_level'].values()) if summary['components_by_level'] else 1
    
    for level in ['critical', 'high', 'medium', 'low', 'minimal']:
        count = summary['components_by_level'].get(level, 0)
        pct = 100 * count / total if total > 0 else 0
        bar_len = int(25 * count / max_count) if max_count > 0 else 0
        bar = "█" * bar_len
        color = level_colors.get(level, '')
        print(f"    {color}{level.upper():10s}{Colors.ENDC}: {count:4d} ({pct:5.1f}%) {color}{bar}{Colors.ENDC}")
    
    print(f"\n  {Colors.BOLD}Outliers:{Colors.ENDC} {summary['component_outliers']} components, "
          f"{summary['edge_outliers']} edges")
    
    if result.components:
        print(f"\n  {Colors.BOLD}Top Critical Components (Box-Plot):{Colors.ENDC}")
        critical_comps = [c for c in result.components if c.level == CriticalityLevel.CRITICAL][:5]
        if critical_comps:
            for i, comp in enumerate(critical_comps, 1):
                print(f"    {i}. {comp.component_id} ({comp.component_type})")
                print(f"       Score: {comp.raw_score:.4f}, Percentile: P{comp.percentile:.0f}")
        else:
            # Show top HIGH if no CRITICAL
            high_comps = [c for c in result.components if c.level == CriticalityLevel.HIGH][:5]
            if high_comps:
                print(f"    (No CRITICAL outliers - showing top HIGH)")
                for i, comp in enumerate(high_comps, 1):
                    print(f"    {i}. {comp.component_id} ({comp.component_type})")
                    print(f"       Score: {comp.raw_score:.4f}, Percentile: P{comp.percentile:.0f}")


def export_classification_csv(result: ClassificationResult, output_dir: Path):
    """Export box-plot classification to CSV"""
    import csv
    
    # Components CSV
    comp_file = output_dir / 'boxplot_components.csv'
    with open(comp_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['component_id', 'type', 'raw_score', 'level', 
                        'percentile', 'z_score', 'is_outlier', 'reasons'])
        for comp in result.components:
            writer.writerow([
                comp.component_id,
                comp.component_type,
                f"{comp.raw_score:.4f}",
                comp.level.value,
                f"{comp.percentile:.1f}",
                f"{comp.z_score:.4f}",
                comp.is_outlier,
                '; '.join(comp.reasons)
            ])
    print_success(f"Exported CSV: {comp_file}")
    
    # Edges CSV
    edge_file = output_dir / 'boxplot_edges.csv'
    with open(edge_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'target', 'edge_type', 'raw_score', 
                        'level', 'percentile', 'is_outlier', 'reasons'])
        for edge in result.edges:
            writer.writerow([
                edge.source,
                edge.target,
                edge.edge_type,
                f"{edge.raw_score:.4f}",
                edge.level.value,
                f"{edge.percentile:.1f}",
                edge.is_outlier,
                '; '.join(edge.reasons)
            ])
    print_success(f"Exported CSV: {edge_file}")


# ============================================================================
# Main CLI
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description='Analyze pub-sub systems for Reliability, Maintainability, and Availability',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis
    python analyze_graph.py
    
    # Full analysis with Neo4j
    python analyze_graph.py --uri bolt://localhost:7687 --full
    
    # Export to multiple formats
    python analyze_graph.py --output results/ --format json html csv
    
    # Specific quality attributes only
    python analyze_graph.py --reliability --availability
        """
    )
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument('--uri', default='bolt://localhost:7687',
                            help='Neo4j URI (default: bolt://localhost:7687)')
    input_group.add_argument('--user', '-u', default='neo4j',
                            help='Neo4j username (default: neo4j)')
    input_group.add_argument('--password', '-p', default='password',
                            help='Neo4j password (default: password)')
    input_group.add_argument('--database', '-d', default='neo4j',
                            help='Neo4j database (default: neo4j)')
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument('--full', '-f', action='store_true',
                               help='Run full analysis (all quality attributes)')
    analysis_group.add_argument('--reliability', '-R', action='store_true',
                               help='Analyze reliability')
    analysis_group.add_argument('--maintainability', '-M', action='store_true',
                               help='Analyze maintainability')
    analysis_group.add_argument('--availability', '-A', action='store_true',
                               help='Analyze availability')
    analysis_group.add_argument('--antipatterns', '-P', action='store_true',
                               help='Detect architectural anti-patterns')
    analysis_group.add_argument('--no-antipatterns', action='store_true',
                               help='Skip anti-pattern detection')
    analysis_group.add_argument('--classify', '-C', action='store_true',
                               help='Classify components/edges using box-plot method')
    analysis_group.add_argument('--iqr-multiplier', type=float, default=1.5,
                               help='IQR multiplier for box-plot classification (default: 1.5)')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output', '-o',
                             help='Output directory for results')
    output_group.add_argument('--format', nargs='+', 
                             choices=['json', 'html', 'csv'],
                             default=['json'],
                             help='Output formats (default: json)')
    
    # Display options
    display_group = parser.add_argument_group('Display Options')
    display_group.add_argument('--verbose', '-v', action='store_true',
                              help='Verbose output with detailed issues')
    display_group.add_argument('--quiet', '-q', action='store_true',
                              help='Minimal output')
    display_group.add_argument('--no-color', action='store_true',
                              help='Disable colored output')
    display_group.add_argument('--json-stdout', action='store_true',
                              help='Output JSON to stdout')
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
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
    
    # Handle colors
    if not sys.stdout.isatty() or args.no_color:
        Colors.disable()
    
    # Determine which analyses to run
    analyze_reliability = args.full or args.reliability or (
        not args.reliability and not args.maintainability and not args.availability and not args.antipatterns
    )
    analyze_maintainability = args.full or args.maintainability or (
        not args.reliability and not args.maintainability and not args.availability and not args.antipatterns
    )
    analyze_availability = args.full or args.availability or (
        not args.reliability and not args.maintainability and not args.availability and not args.antipatterns
    )
    # Anti-patterns are detected by default unless explicitly disabled
    detect_antipatterns = not args.no_antipatterns
    
    # Print header
    if not args.quiet:
        print_header("GRAPH MODEL ANALYZER")
        print(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Neo4j: {args.uri}")
        print(f"  Analyzing: ", end="")
        analyses = []
        if analyze_reliability:
            analyses.append("Reliability")
        if analyze_maintainability:
            analyses.append("Maintainability")
        if analyze_availability:
            analyses.append("Availability")
        if detect_antipatterns:
            analyses.append("Anti-Patterns")
        print(", ".join(analyses))
    
    try:
        # Build graph
        if not args.quiet:
            print_section("LOADING GRAPH")
        
        builder = DependsOnGraphBuilder()
        
        graph = builder.build_from_neo4j(
                uri=args.uri,
                user=args.user,
                password=args.password,
                database=args.database)
        
        print_success(f"Graph loaded: {graph.number_of_nodes()} nodes, "
                     f"{graph.number_of_edges()} edges")
        
        # Run analysis
        if not args.quiet:
            print_section("RUNNING ANALYSIS")
        
        orchestrator = QualityAnalysisOrchestrator()
        result = orchestrator.analyze(
            graph,
            analyze_reliability=analyze_reliability,
            analyze_maintainability=analyze_maintainability,
            analyze_availability=analyze_availability,
            detect_antipatterns=detect_antipatterns
        )
        
        print_success("Analysis complete")
        
        # Run box-plot classification if requested
        classification_result = None
        if args.classify:
            if not args.quiet:
                print_section("BOX-PLOT CRITICALITY CLASSIFICATION")
            
            classifier = BoxPlotCriticalityClassifier(iqr_multiplier=args.iqr_multiplier)
            classification_result = classifier.classify_graph(graph)
            
            print_success(f"Classification complete: {len(classification_result.components)} components, "
                         f"{len(classification_result.edges)} edges")
            
            if not args.quiet and not args.json_stdout:
                print_boxplot_summary(classification_result)
        
        # Output results
        if args.json_stdout:
            output_dict = result.to_dict()
            if classification_result:
                output_dict['boxplot_classification'] = classification_result.to_dict()
            print(json.dumps(output_dict, indent=2, default=str))
        elif not args.quiet:
            print_analysis_summary(result, verbose=args.verbose)
        
        # Export to files
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if not args.quiet:
                print_section("EXPORTING RESULTS")
            
            if 'json' in args.format:
                output_dict = result.to_dict()
                if classification_result:
                    output_dict['boxplot_classification'] = classification_result.to_dict()
                with open(output_dir / 'analysis_results.json', 'w') as f:
                    json.dump(output_dict, f, indent=2, default=str)
                print_success(f"Exported JSON: {output_dir / 'analysis_results.json'}")
            
            if 'html' in args.format:
                export_html(result, output_dir / 'analysis_report.html')
            
            if 'csv' in args.format:
                export_csv(result, output_dir)
                if classification_result:
                    export_classification_csv(classification_result, output_dir)
        
        return 0
        
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        return 1
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}")
        return 1
    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        return 1
    except Exception as e:
        print_error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())