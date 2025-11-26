#!/usr/bin/env python3
"""
Graph Analysis Script for Pub-Sub Systems - Enhanced Production Version

Comprehensive analysis including:
- Critical nodes (articulation points, high betweenness)
- Critical edges (bridges, bottleneck links)
- Multi-layer dependency analysis
- QoS-aware criticality scoring (C_score = α·BC_norm + β·AP + γ·I)
- Failure simulation
- Anti-pattern detection (SPOF, god topics, circular dependencies)
- Support for both JSON and Neo4j input

Usage:
    # From JSON file
    python analyze_graph.py --input system.json
    
    # From Neo4j database
    python analyze_graph.py --neo4j --uri bolt://localhost:7687 \\
        --user neo4j --password password
    
    # With full analysis and exports
    python analyze_graph.py --input system.json --analyze-edges \\
        --simulate --detect-antipatterns \\
        --export-json --export-csv --export-html

Author: Software-as-a-Graph Research Project
Version: 3.0
"""

import argparse
import json
import logging
import sys
import time
import csv
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import traceback
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# Dependencies Check
# ============================================================================

try:
    import networkx as nx
except ImportError:
    print("Error: networkx is required. Install with: pip install networkx")
    sys.exit(1)

# Optional dependencies
PANDAS_AVAILABLE = False
NEO4J_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pass

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# Configuration and Constants
# ============================================================================

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
    
    @classmethod
    def disable(cls):
        """Disable colors for non-TTY output"""
        cls.HEADER = ''
        cls.OKBLUE = ''
        cls.OKCYAN = ''
        cls.OKGREEN = ''
        cls.WARNING = ''
        cls.FAIL = ''
        cls.ENDC = ''
        cls.BOLD = ''
        cls.UNDERLINE = ''


class CriticalityLevel(Enum):
    """Criticality classification levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class AntiPatternType(Enum):
    """Types of anti-patterns detected"""
    SPOF = "single_point_of_failure"
    GOD_TOPIC = "god_topic"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    TIGHT_COUPLING = "tight_coupling"
    HIDDEN_COUPLING = "hidden_coupling"
    BROKER_BOTTLENECK = "broker_bottleneck"


# Default criticality scoring weights (from research methodology)
DEFAULT_ALPHA = 0.4  # Betweenness centrality weight
DEFAULT_BETA = 0.3   # Articulation point weight
DEFAULT_GAMMA = 0.3  # Impact score weight


# Fuzzy logic support
FUZZY_AVAILABLE = False
try:
    from src.analysis.fuzzy_criticality_analyzer import (
        FuzzyCriticalityAnalyzer,
        FuzzyNodeCriticalityResult,
        FuzzyEdgeCriticalityResult,
        FuzzyCriticalityLevel,
        DefuzzificationMethod,
        analyze_graph_with_fuzzy_logic,
        create_node_criticality_fis,
        create_edge_criticality_fis
    )
    FUZZY_AVAILABLE = True
except ImportError:
    pass


class CriticalityMethod(Enum):
    """Criticality calculation methods"""
    COMPOSITE = "composite"  # Traditional weighted formula
    FUZZY = "fuzzy"          # Fuzzy logic inference


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CompositeCriticalityScore:
    """Composite criticality score with all components"""
    component: str
    component_type: str
    betweenness_centrality: float = 0.0
    betweenness_centrality_norm: float = 0.0
    is_articulation_point: bool = False
    impact_score: float = 0.0
    reachability_loss_percentage: float = 0.0
    components_affected: int = 0
    composite_score: float = 0.0
    criticality_level: CriticalityLevel = CriticalityLevel.MINIMAL
    qos_factor: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'component': self.component,
            'component_type': self.component_type,
            'betweenness_centrality': round(self.betweenness_centrality, 6),
            'betweenness_centrality_norm': round(self.betweenness_centrality_norm, 4),
            'is_articulation_point': self.is_articulation_point,
            'impact_score': round(self.impact_score, 4),
            'reachability_loss_percentage': round(self.reachability_loss_percentage, 2),
            'components_affected': self.components_affected,
            'composite_score': round(self.composite_score, 4),
            'criticality_level': self.criticality_level.value,
            'qos_factor': round(self.qos_factor, 3)
        }


@dataclass
class AntiPattern:
    """Detected anti-pattern"""
    type: AntiPatternType
    components: List[str]
    severity: str
    description: str
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value,
            'components': self.components,
            'severity': self.severity,
            'description': self.description,
            'recommendation': self.recommendation
        }


@dataclass
class AnalysisResults:
    """Complete analysis results container"""
    metadata: Dict[str, Any] = field(default_factory=dict)
    structure: Dict[str, Any] = field(default_factory=dict)
    node_analysis: Dict[str, Any] = field(default_factory=dict)
    edge_analysis: Dict[str, Any] = field(default_factory=dict)
    layer_analysis: Dict[str, Any] = field(default_factory=dict)
    anti_patterns: List[Dict] = field(default_factory=list)
    failure_simulation: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging with optional file output"""
    logger = logging.getLogger('graph_analysis')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


# Global logger
logger = logging.getLogger('graph_analysis')


# ============================================================================
# Graph Loading Functions
# ============================================================================

def validate_graph_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate graph data structure
    
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    # Check required fields
    required_fields = ['applications', 'topics', 'brokers', 'nodes']
    for field in required_fields:
        if field not in data:
            warnings.append(f"Missing field: {field}")
    
    # Check for empty collections
    for field in required_fields:
        if field in data and len(data.get(field, [])) == 0:
            warnings.append(f"Empty collection: {field}")
    
    # Check for relationship data
    relationship_fields = ['publishes', 'subscribes']
    for field in relationship_fields:
        if field not in data or len(data.get(field, [])) == 0:
            warnings.append(f"No {field} relationships defined")
    
    is_valid = len([w for w in warnings if 'Missing field' in w]) == 0
    return is_valid, warnings


def load_graph_from_json(file_path: str) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Load graph from JSON file
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Tuple of (NetworkX graph, raw data dictionary)
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    logger.info(f"Loading graph from: {file_path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Validate data
    is_valid, warnings = validate_graph_data(data)
    
    for warning in warnings:
        logger.warning(f"Validation: {warning}")
    
    if not is_valid:
        raise ValueError("Invalid graph data structure")
    
    # Build NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for node in data.get('nodes', []):
        node_id = node.get('id', node.get('name'))
        G.add_node(node_id, type='Node', **node)
    
    for broker in data.get('brokers', []):
        broker_id = broker.get('id', broker.get('name'))
        G.add_node(broker_id, type='Broker', **broker)
    
    for topic in data.get('topics', []):
        topic_id = topic.get('id', topic.get('name'))
        G.add_node(topic_id, type='Topic', **topic)
    
    for app in data.get('applications', []):
        app_id = app.get('id', app.get('name'))
        G.add_node(app_id, type='Application', **app)
    
    # Add edges - PUBLISHES
    for pub in data.get('publishes', []):
        source = pub.get('application') or pub.get('source')
        target = pub.get('topic') or pub.get('target')
        if source and target and source in G and target in G:
            G.add_edge(source, target, type='PUBLISHES', **pub)
    
    # Add edges - SUBSCRIBES
    for sub in data.get('subscribes', []):
        source = sub.get('topic') or sub.get('source')
        target = sub.get('application') or sub.get('target')
        if source and target and source in G and target in G:
            G.add_edge(source, target, type='SUBSCRIBES', **sub)
    
    # Add edges - HOSTS_ON (broker to topic)
    for topic in data.get('topics', []):
        topic_id = topic.get('id', topic.get('name'))
        broker_id = topic.get('broker')
        if broker_id and topic_id in G and broker_id in G:
            G.add_edge(topic_id, broker_id, type='HOSTS_ON')
    
    # Add edges - RUNS_ON (app/broker to node)
    for app in data.get('applications', []):
        app_id = app.get('id', app.get('name'))
        node_id = app.get('node')
        if node_id and app_id in G and node_id in G:
            G.add_edge(app_id, node_id, type='RUNS_ON')
    
    for broker in data.get('brokers', []):
        broker_id = broker.get('id', broker.get('name'))
        node_id = broker.get('node')
        if node_id and broker_id in G and node_id in G:
            G.add_edge(broker_id, node_id, type='RUNS_ON')
    
    # Add broker routes
    for source, targets in data.get('broker_routes', {}).items():
        for target in targets:
            if source in G and target in G:
                G.add_edge(source, target, type='ROUTES_TO')
    
    logger.info(f"Loaded graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    return G, data


def load_graph_from_neo4j(uri: str, user: str, password: str) -> Tuple[nx.DiGraph, Dict]:
    """
    Load graph from Neo4j database
    
    Args:
        uri: Neo4j bolt URI
        user: Username
        password: Password
    
    Returns:
        Tuple of (NetworkX graph, metadata dictionary)
    """
    if not NEO4J_AVAILABLE:
        raise ImportError("neo4j package required. Install with: pip install neo4j")
    
    logger.info(f"Connecting to Neo4j at: {uri}")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    G = nx.DiGraph()
    
    with driver.session() as session:
        # Load all nodes
        result = session.run("""
            MATCH (n)
            RETURN id(n) as id, labels(n) as labels, properties(n) as props
        """)
        
        for record in result:
            node_id = record['props'].get('id', str(record['id']))
            node_type = record['labels'][0] if record['labels'] else 'Unknown'
            G.add_node(node_id, type=node_type, **record['props'])
        
        # Load all relationships
        result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN 
                properties(a).id as source,
                properties(b).id as target,
                type(r) as rel_type,
                properties(r) as props
        """)
        
        for record in result:
            G.add_edge(
                record['source'],
                record['target'],
                type=record['rel_type'],
                **record['props']
            )
    
    driver.close()
    
    metadata = {
        'source': 'neo4j',
        'uri': uri,
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Loaded from Neo4j: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    return G, metadata


# ============================================================================
# Graph Structure Analysis
# ============================================================================

def analyze_graph_structure(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Analyze basic graph structure and metrics
    
    Args:
        G: NetworkX directed graph
    
    Returns:
        Dictionary with structure metrics
    """
    logger.info("Analyzing graph structure...")
    
    # Basic metrics
    structure = {
        'nodes': len(G.nodes()),
        'edges': len(G.edges()),
        'density': round(nx.density(G), 6),
        'is_weakly_connected': nx.is_weakly_connected(G),
        'number_weakly_connected_components': nx.number_weakly_connected_components(G)
    }
    
    # Node type distribution
    node_types = defaultdict(int)
    for _, data in G.nodes(data=True):
        node_types[data.get('type', 'Unknown')] += 1
    structure['node_types'] = dict(node_types)
    
    # Edge type distribution
    edge_types = defaultdict(int)
    for _, _, data in G.edges(data=True):
        edge_types[data.get('type', 'Unknown')] += 1
    structure['edge_types'] = dict(edge_types)
    
    # Average degree
    if len(G) > 0:
        structure['avg_in_degree'] = round(sum(d for _, d in G.in_degree()) / len(G), 2)
        structure['avg_out_degree'] = round(sum(d for _, d in G.out_degree()) / len(G), 2)
    else:
        structure['avg_in_degree'] = 0
        structure['avg_out_degree'] = 0
    
    # Check for self-loops
    structure['self_loops'] = len(list(nx.selfloop_edges(G)))
    
    # Strongly connected components (if graph is strongly connected)
    structure['is_strongly_connected'] = nx.is_strongly_connected(G)
    if not structure['is_strongly_connected']:
        structure['number_strongly_connected_components'] = (
            nx.number_strongly_connected_components(G)
        )
    
    return structure


# ============================================================================
# Node Criticality Analysis
# ============================================================================

def calculate_betweenness_centrality(G: nx.DiGraph) -> Dict[str, float]:
    """Calculate betweenness centrality for all nodes"""
    return nx.betweenness_centrality(G, normalized=True)


def find_articulation_points(G: nx.DiGraph) -> Set[str]:
    """
    Find articulation points (cut vertices) in the graph
    
    Uses undirected version for articulation point detection
    """
    undirected = G.to_undirected()
    return set(nx.articulation_points(undirected))


def calculate_impact_score(G: nx.DiGraph, node: str) -> Tuple[float, int, float]:
    """
    Calculate impact score based on reachability loss
    
    Args:
        G: Original graph
        node: Node to analyze
    
    Returns:
        Tuple of (impact_score, components_affected, reachability_loss_pct)
    """
    if node not in G:
        return 0.0, 0, 0.0
    
    # Calculate original reachability
    original_reachable = set()
    for n in G.nodes():
        if n != node:
            for target in nx.descendants(G, n):
                original_reachable.add((n, target))
    
    # Remove node and recalculate
    G_copy = G.copy()
    G_copy.remove_node(node)
    
    new_reachable = set()
    for n in G_copy.nodes():
        for target in nx.descendants(G_copy, n):
            new_reachable.add((n, target))
    
    # Calculate loss
    lost_reachability = original_reachable - new_reachable
    components_affected = len(set(pair[0] for pair in lost_reachability) | 
                              set(pair[1] for pair in lost_reachability))
    
    if len(original_reachable) > 0:
        reachability_loss_pct = (len(lost_reachability) / len(original_reachable)) * 100
    else:
        reachability_loss_pct = 0.0
    
    # Normalize impact score to 0-1 range
    impact_score = min(1.0, components_affected / max(1, len(G.nodes()) - 1))
    
    return impact_score, components_affected, reachability_loss_pct


def classify_criticality_level(score: float) -> CriticalityLevel:
    """Classify criticality based on composite score"""
    if score >= 0.8:
        return CriticalityLevel.CRITICAL
    elif score >= 0.6:
        return CriticalityLevel.HIGH
    elif score >= 0.4:
        return CriticalityLevel.MEDIUM
    elif score >= 0.2:
        return CriticalityLevel.LOW
    else:
        return CriticalityLevel.MINIMAL


def calculate_criticality_scores(
    G: nx.DiGraph,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    calculate_impact: bool = True
) -> Dict[str, CompositeCriticalityScore]:
    """
    Calculate composite criticality scores for all nodes
    
    Formula: C_score(v) = α·BC_norm(v) + β·AP(v) + γ·I(v)
    
    Args:
        G: NetworkX graph
        alpha: Weight for betweenness centrality (default 0.4)
        beta: Weight for articulation point indicator (default 0.3)
        gamma: Weight for impact score (default 0.3)
        calculate_impact: Whether to calculate full impact scores (slower)
    
    Returns:
        Dictionary mapping node ID to CompositeCriticalityScore
    """
    logger.info("Calculating criticality scores...")
    
    # Calculate betweenness centrality
    betweenness = calculate_betweenness_centrality(G)
    
    # Find articulation points
    articulation_points = find_articulation_points(G)
    
    # Normalize betweenness
    max_betweenness = max(betweenness.values()) if betweenness else 1.0
    if max_betweenness == 0:
        max_betweenness = 1.0
    
    scores = {}
    
    for node in G.nodes():
        node_data = G.nodes[node]
        node_type = node_data.get('type', 'Unknown')
        
        # Betweenness (normalized)
        bc = betweenness.get(node, 0.0)
        bc_norm = bc / max_betweenness
        
        # Articulation point indicator
        is_ap = node in articulation_points
        ap_value = 1.0 if is_ap else 0.0
        
        # Impact score (expensive calculation)
        if calculate_impact:
            impact, affected, reachability_loss = calculate_impact_score(G, node)
        else:
            impact, affected, reachability_loss = 0.0, 0, 0.0
        
        # Composite score
        composite = (alpha * bc_norm) + (beta * ap_value) + (gamma * impact)
        
        # Classification
        level = classify_criticality_level(composite)
        
        scores[node] = CompositeCriticalityScore(
            component=node,
            component_type=node_type,
            betweenness_centrality=bc,
            betweenness_centrality_norm=bc_norm,
            is_articulation_point=is_ap,
            impact_score=impact,
            reachability_loss_percentage=reachability_loss,
            components_affected=affected,
            composite_score=composite,
            criticality_level=level
        )
    
    logger.info(f"Calculated criticality for {len(scores)} nodes")
    
    return scores


def analyze_node_criticality(
    G: nx.DiGraph,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    calculate_impact: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive node criticality analysis
    
    Args:
        G: NetworkX graph
        alpha, beta, gamma: Criticality scoring weights
        calculate_impact: Whether to calculate full impact scores
    
    Returns:
        Dictionary with node analysis results
    """
    scores = calculate_criticality_scores(G, alpha, beta, gamma, calculate_impact)
    
    # Sort by composite score
    sorted_scores = sorted(
        scores.values(),
        key=lambda x: x.composite_score,
        reverse=True
    )
    
    # Statistics
    all_scores = [s.composite_score for s in sorted_scores]
    
    statistics = {
        'total_nodes': len(scores),
        'articulation_point_count': sum(1 for s in scores.values() if s.is_articulation_point),
        'articulation_point_percentage': round(
            sum(1 for s in scores.values() if s.is_articulation_point) / max(1, len(scores)) * 100, 2
        ),
        'avg_criticality': round(sum(all_scores) / max(1, len(all_scores)), 4),
        'max_criticality': round(max(all_scores) if all_scores else 0, 4),
        'min_criticality': round(min(all_scores) if all_scores else 0, 4),
        'critical_count': sum(1 for s in scores.values() if s.criticality_level == CriticalityLevel.CRITICAL),
        'high_count': sum(1 for s in scores.values() if s.criticality_level == CriticalityLevel.HIGH),
        'medium_count': sum(1 for s in scores.values() if s.criticality_level == CriticalityLevel.MEDIUM),
        'low_count': sum(1 for s in scores.values() if s.criticality_level == CriticalityLevel.LOW),
        'minimal_count': sum(1 for s in scores.values() if s.criticality_level == CriticalityLevel.MINIMAL)
    }
    
    # Top critical nodes
    top_critical = [
        {
            'node': s.component,
            'type': s.component_type,
            'score': round(s.composite_score, 4),
            'is_articulation_point': s.is_articulation_point,
            'betweenness': round(s.betweenness_centrality_norm, 4),
            'impact': round(s.impact_score, 4),
            'level': s.criticality_level.value
        }
        for s in sorted_scores[:20]  # Top 20
    ]
    
    # Criticality by type
    by_type = defaultdict(list)
    for s in sorted_scores:
        by_type[s.component_type].append(s)
    
    type_statistics = {}
    for node_type, type_scores in by_type.items():
        type_all = [s.composite_score for s in type_scores]
        type_statistics[node_type] = {
            'count': len(type_scores),
            'avg_score': round(sum(type_all) / max(1, len(type_all)), 4),
            'critical_count': sum(1 for s in type_scores if s.criticality_level == CriticalityLevel.CRITICAL),
            'high_count': sum(1 for s in type_scores if s.criticality_level == CriticalityLevel.HIGH)
        }
    
    return {
        'statistics': statistics,
        'type_statistics': type_statistics,
        'top_critical_nodes': top_critical,
        'all_scores': {node: score.to_dict() for node, score in scores.items()}
    }


# ============================================================================
# Edge Criticality Analysis
# ============================================================================

def find_bridges(G: nx.DiGraph) -> List[Tuple[str, str]]:
    """Find bridges (cut edges) in the graph"""
    undirected = G.to_undirected()
    return list(nx.bridges(undirected))


def calculate_edge_betweenness(G: nx.DiGraph) -> Dict[Tuple[str, str], float]:
    """Calculate edge betweenness centrality"""
    return nx.edge_betweenness_centrality(G, normalized=True)


# ============================================================================
# Fuzzy Logic Analysis Functions
# ============================================================================

def analyze_node_criticality_fuzzy(
    G: nx.DiGraph,
    calculate_impact: bool = True,
    defuzz_method: str = "centroid"
) -> Dict[str, Any]:
    """
    Analyze node criticality using fuzzy logic
    
    Args:
        G: NetworkX graph
        calculate_impact: Whether to calculate full impact scores
        defuzz_method: Defuzzification method (centroid, mom, bisector)
    
    Returns:
        Dictionary with fuzzy node analysis results
    """
    if not FUZZY_AVAILABLE:
        logger.warning("Fuzzy logic not available, falling back to composite scoring")
        return analyze_node_criticality(G, calculate_impact=calculate_impact)
    
    logger.info("Analyzing node criticality using fuzzy logic...")
    
    # Map defuzzification method
    method_map = {
        'centroid': DefuzzificationMethod.CENTROID,
        'mom': DefuzzificationMethod.MOM,
        'bisector': DefuzzificationMethod.BISECTOR,
        'som': DefuzzificationMethod.SOM,
        'lom': DefuzzificationMethod.LOM
    }
    defuzz = method_map.get(defuzz_method, DefuzzificationMethod.CENTROID)
    
    # Run fuzzy analysis
    node_results, _ = analyze_graph_with_fuzzy_logic(
        G, 
        calculate_impact=calculate_impact,
        defuzz_method=defuzz
    )
    
    # Convert to standard format
    sorted_results = sorted(
        node_results.values(),
        key=lambda x: x.fuzzy_criticality_score,
        reverse=True
    )
    
    # Statistics
    all_scores = [r.fuzzy_criticality_score for r in sorted_results]
    
    statistics = {
        'total_nodes': len(node_results),
        'articulation_point_count': sum(1 for r in node_results.values() if r.is_articulation_point),
        'articulation_point_percentage': round(
            sum(1 for r in node_results.values() if r.is_articulation_point) / max(1, len(node_results)) * 100, 2
        ),
        'avg_criticality': round(sum(all_scores) / max(1, len(all_scores)), 4),
        'max_criticality': round(max(all_scores) if all_scores else 0, 4),
        'min_criticality': round(min(all_scores) if all_scores else 0, 4),
        'critical_count': sum(1 for r in node_results.values() if r.criticality_level == FuzzyCriticalityLevel.CRITICAL),
        'high_count': sum(1 for r in node_results.values() if r.criticality_level == FuzzyCriticalityLevel.HIGH),
        'medium_count': sum(1 for r in node_results.values() if r.criticality_level == FuzzyCriticalityLevel.MEDIUM),
        'low_count': sum(1 for r in node_results.values() if r.criticality_level == FuzzyCriticalityLevel.LOW),
        'minimal_count': sum(1 for r in node_results.values() if r.criticality_level == FuzzyCriticalityLevel.MINIMAL),
        'analysis_method': 'fuzzy_logic',
        'defuzzification_method': defuzz_method
    }
    
    # Top critical nodes
    top_critical = [
        {
            'node': r.component,
            'type': r.component_type,
            'score': round(r.fuzzy_criticality_score, 4),
            'is_articulation_point': r.is_articulation_point,
            'betweenness': round(r.betweenness_centrality_norm, 4),
            'impact': round(r.impact_score, 4),
            'level': r.criticality_level.value,
            'membership_degrees': r.membership_degrees
        }
        for r in sorted_results[:20]
    ]
    
    # Criticality by type
    by_type = defaultdict(list)
    for r in sorted_results:
        by_type[r.component_type].append(r)
    
    type_statistics = {}
    for node_type, type_results in by_type.items():
        type_scores = [r.fuzzy_criticality_score for r in type_results]
        type_statistics[node_type] = {
            'count': len(type_results),
            'avg_score': round(sum(type_scores) / max(1, len(type_scores)), 4),
            'critical_count': sum(1 for r in type_results if r.criticality_level == FuzzyCriticalityLevel.CRITICAL),
            'high_count': sum(1 for r in type_results if r.criticality_level == FuzzyCriticalityLevel.HIGH)
        }
    
    return {
        'statistics': statistics,
        'type_statistics': type_statistics,
        'top_critical_nodes': top_critical,
        'all_scores': {node: result.to_dict() for node, result in node_results.items()}
    }


def analyze_edge_criticality_fuzzy(
    G: nx.DiGraph,
    defuzz_method: str = "centroid"
) -> Dict[str, Any]:
    """
    Analyze edge criticality using fuzzy logic
    
    Args:
        G: NetworkX graph
        defuzz_method: Defuzzification method
    
    Returns:
        Dictionary with fuzzy edge analysis results
    """
    if not FUZZY_AVAILABLE:
        logger.warning("Fuzzy logic not available, falling back to standard edge analysis")
        return analyze_edge_criticality(G)
    
    logger.info("Analyzing edge criticality using fuzzy logic...")
    
    # Map defuzzification method
    method_map = {
        'centroid': DefuzzificationMethod.CENTROID,
        'mom': DefuzzificationMethod.MOM,
        'bisector': DefuzzificationMethod.BISECTOR,
        'som': DefuzzificationMethod.SOM,
        'lom': DefuzzificationMethod.LOM
    }
    defuzz = method_map.get(defuzz_method, DefuzzificationMethod.CENTROID)
    
    # Run fuzzy analysis
    _, edge_results = analyze_graph_with_fuzzy_logic(
        G, 
        calculate_impact=False,  # Already calculated for nodes
        defuzz_method=defuzz
    )
    
    # Sort by criticality score
    sorted_results = sorted(
        edge_results.values(),
        key=lambda x: x.fuzzy_criticality_score,
        reverse=True
    )
    
    # Statistics
    all_scores = [r.fuzzy_criticality_score for r in sorted_results]
    
    statistics = {
        'total_edges': len(edge_results),
        'bridge_count': sum(1 for r in edge_results.values() if r.is_bridge),
        'bridge_percentage': round(
            sum(1 for r in edge_results.values() if r.is_bridge) / max(1, len(edge_results)) * 100, 2
        ),
        'avg_edge_criticality': round(sum(all_scores) / max(1, len(all_scores)), 4),
        'max_edge_criticality': round(max(all_scores) if all_scores else 0, 4),
        'critical_count': sum(1 for r in edge_results.values() if r.criticality_level == FuzzyCriticalityLevel.CRITICAL),
        'high_count': sum(1 for r in edge_results.values() if r.criticality_level == FuzzyCriticalityLevel.HIGH),
        'analysis_method': 'fuzzy_logic',
        'defuzzification_method': defuzz_method
    }
    
    # Top edges by criticality
    top_edges = [
        {
            'from': r.source,
            'to': r.target,
            'score': round(r.fuzzy_criticality_score, 4),
            'is_bridge': r.is_bridge,
            'edge_type': r.edge_type,
            'level': r.criticality_level.value,
            'membership_degrees': r.membership_degrees
        }
        for r in sorted_results[:20]
    ]
    
    # Bridge details
    bridges = [
        {
            'from': r.source,
            'to': r.target,
            'criticality': round(r.fuzzy_criticality_score, 4),
            'edge_type': r.edge_type,
            'level': r.criticality_level.value
        }
        for r in sorted_results if r.is_bridge
    ]
    
    return {
        'statistics': statistics,
        'top_edge_criticality': top_edges,
        'top_edge_betweenness': top_edges,  # Compatibility alias
        'bridges': bridges,
        'all_scores': {f"{e[0]}->{e[1]}": result.to_dict() for e, result in edge_results.items()}
    }


def analyze_edge_criticality(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Comprehensive edge criticality analysis
    
    Args:
        G: NetworkX graph
    
    Returns:
        Dictionary with edge analysis results
    """
    logger.info("Analyzing edge criticality...")
    
    # Find bridges
    bridges = find_bridges(G)
    
    # Calculate edge betweenness
    edge_betweenness = calculate_edge_betweenness(G)
    
    # Sort by betweenness
    sorted_edges = sorted(
        edge_betweenness.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Statistics
    all_betweenness = list(edge_betweenness.values())
    
    statistics = {
        'total_edges': len(G.edges()),
        'bridge_count': len(bridges),
        'bridge_percentage': round(len(bridges) / max(1, len(G.edges())) * 100, 2),
        'avg_edge_betweenness': round(sum(all_betweenness) / max(1, len(all_betweenness)), 6),
        'max_edge_betweenness': round(max(all_betweenness) if all_betweenness else 0, 6)
    }
    
    # Top edges by betweenness
    top_edge_betweenness = [
        {
            'from': edge[0],
            'to': edge[1],
            'score': round(score, 6),
            'is_bridge': edge in bridges or (edge[1], edge[0]) in bridges,
            'edge_type': G.edges[edge].get('type', 'Unknown') if edge in G.edges else 'Unknown'
        }
        for edge, score in sorted_edges[:20]
    ]
    
    # Bridges detail
    bridges_detail = [
        {
            'from': edge[0],
            'to': edge[1],
            'betweenness': round(edge_betweenness.get(edge, edge_betweenness.get((edge[1], edge[0]), 0)), 6),
            'edge_type': G.edges.get(edge, {}).get('type', 'Unknown')
        }
        for edge in bridges
    ]
    
    return {
        'statistics': statistics,
        'top_edge_betweenness': top_edge_betweenness,
        'bridges': bridges_detail
    }


# ============================================================================
# Layer Analysis
# ============================================================================

def analyze_layer_dependencies(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Analyze multi-layer dependencies
    
    Args:
        G: NetworkX graph
    
    Returns:
        Dictionary with layer analysis results
    """
    logger.info("Analyzing layer dependencies...")
    
    # Group nodes by type (layer)
    layers = defaultdict(list)
    for node, data in G.nodes(data=True):
        layers[data.get('type', 'Unknown')].append(node)
    
    layer_metrics = {}
    for layer_name, nodes in layers.items():
        # Create subgraph for this layer
        subgraph = G.subgraph(nodes)
        
        # Internal edges (within layer)
        internal_edges = len(subgraph.edges())
        
        # External edges (cross-layer)
        external_edges = 0
        for node in nodes:
            for _, target in G.out_edges(node):
                if target not in nodes:
                    external_edges += 1
            for source, _ in G.in_edges(node):
                if source not in nodes:
                    external_edges += 1
        
        layer_metrics[layer_name] = {
            'node_count': len(nodes),
            'internal_edges': internal_edges,
            'external_edges': external_edges // 2,  # Avoid double counting
            'internal_density': round(nx.density(subgraph), 6) if len(nodes) > 1 else 0
        }
    
    # Cross-layer dependency matrix
    cross_layer = defaultdict(lambda: defaultdict(int))
    for source, target in G.edges():
        source_type = G.nodes[source].get('type', 'Unknown')
        target_type = G.nodes[target].get('type', 'Unknown')
        if source_type != target_type:
            cross_layer[source_type][target_type] += 1
    
    return {
        'layer_metrics': layer_metrics,
        'cross_layer_dependencies': {k: dict(v) for k, v in cross_layer.items()},
        'layer_order': list(layers.keys())
    }


# ============================================================================
# Anti-Pattern Detection
# ============================================================================

def detect_anti_patterns(
    G: nx.DiGraph,
    criticality_scores: Dict[str, CompositeCriticalityScore]
) -> List[AntiPattern]:
    """
    Detect common anti-patterns in the pub-sub system
    
    Args:
        G: NetworkX graph
        criticality_scores: Pre-calculated criticality scores
    
    Returns:
        List of detected anti-patterns
    """
    logger.info("Detecting anti-patterns...")
    
    anti_patterns = []
    
    # 1. Single Points of Failure (SPOFs)
    # High criticality nodes that are articulation points
    for node, score in criticality_scores.items():
        if score.is_articulation_point and score.composite_score >= 0.6:
            anti_patterns.append(AntiPattern(
                type=AntiPatternType.SPOF,
                components=[node],
                severity='high' if score.composite_score >= 0.8 else 'medium',
                description=f"Node '{node}' ({score.component_type}) is a single point of failure "
                           f"with criticality score {score.composite_score:.3f}",
                recommendation=f"Add redundancy for {node} or distribute its responsibilities"
            ))
    
    # 2. God Topics (topics with too many subscribers)
    topics = [n for n, d in G.nodes(data=True) if d.get('type') == 'Topic']
    for topic in topics:
        subscribers = len([e for e in G.out_edges(topic) if G.edges[e].get('type') == 'SUBSCRIBES'])
        publishers = len([e for e in G.in_edges(topic) if G.edges[e].get('type') == 'PUBLISHES'])
        
        if subscribers > 10 or publishers > 5:
            anti_patterns.append(AntiPattern(
                type=AntiPatternType.GOD_TOPIC,
                components=[topic],
                severity='high' if subscribers > 20 else 'medium',
                description=f"Topic '{topic}' has {publishers} publishers and {subscribers} subscribers",
                recommendation=f"Consider splitting '{topic}' into more specific topics"
            ))
    
    # 3. Circular Dependencies
    try:
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles[:10]:  # Report up to 10 cycles
            if len(cycle) > 1:
                anti_patterns.append(AntiPattern(
                    type=AntiPatternType.CIRCULAR_DEPENDENCY,
                    components=cycle,
                    severity='high' if len(cycle) <= 3 else 'medium',
                    description=f"Circular dependency detected: {' -> '.join(cycle)} -> {cycle[0]}",
                    recommendation="Break the cycle by introducing an intermediate component or redesigning"
                ))
    except Exception as e:
        logger.warning(f"Cycle detection failed: {e}")
    
    # 4. Broker Bottlenecks
    brokers = [n for n, d in G.nodes(data=True) if d.get('type') == 'Broker']
    for broker in brokers:
        topics_hosted = len([e for e in G.in_edges(broker) if G.edges[e].get('type') == 'HOSTS_ON'])
        if topics_hosted > 15:
            anti_patterns.append(AntiPattern(
                type=AntiPatternType.BROKER_BOTTLENECK,
                components=[broker],
                severity='high' if topics_hosted > 25 else 'medium',
                description=f"Broker '{broker}' hosts {topics_hosted} topics",
                recommendation=f"Distribute topics across multiple brokers"
            ))
    
    # 5. Tight Coupling (applications with many direct dependencies)
    apps = [n for n, d in G.nodes(data=True) if d.get('type') == 'Application']
    for app in apps:
        out_topics = len([e for e in G.out_edges(app) if G.edges[e].get('type') == 'PUBLISHES'])
        in_topics = len([e for e in G.in_edges(app)])  # Subscriptions come through topics
        
        if out_topics + in_topics > 10:
            anti_patterns.append(AntiPattern(
                type=AntiPatternType.TIGHT_COUPLING,
                components=[app],
                severity='medium',
                description=f"Application '{app}' has {out_topics} publish and multiple subscribe connections",
                recommendation=f"Consider using message aggregation or reducing topic coupling"
            ))
    
    logger.info(f"Detected {len(anti_patterns)} anti-patterns")
    
    return anti_patterns


# ============================================================================
# Failure Simulation
# ============================================================================

def simulate_node_failure(G: nx.DiGraph, node: str) -> Dict[str, Any]:
    """
    Simulate failure of a single node
    
    Args:
        G: Original graph
        node: Node to fail
    
    Returns:
        Impact metrics
    """
    if node not in G:
        return {'error': f'Node {node} not found'}
    
    node_data = G.nodes[node]
    
    # Create copy without the failed node
    G_failed = G.copy()
    G_failed.remove_node(node)
    
    # Calculate impact
    original_components = nx.number_weakly_connected_components(G)
    failed_components = nx.number_weakly_connected_components(G_failed)
    
    # Find isolated nodes
    isolated = []
    if not nx.is_weakly_connected(G_failed) and len(G_failed) > 0:
        components = list(nx.weakly_connected_components(G_failed))
        largest_component = max(components, key=len)
        for component in components:
            if component != largest_component:
                isolated.extend(list(component))
    
    # Calculate affected paths
    original_paths = sum(1 for _ in nx.all_pairs_shortest_path_length(G))
    failed_paths = sum(1 for _ in nx.all_pairs_shortest_path_length(G_failed))
    
    return {
        'failed_node': node,
        'node_type': node_data.get('type', 'Unknown'),
        'components_before': original_components,
        'components_after': failed_components,
        'fragments_created': failed_components - original_components,
        'isolated_nodes': isolated,
        'isolated_count': len(isolated),
        'connectivity_loss_percentage': round(
            (1 - (failed_paths / max(1, original_paths))) * 100, 2
        ) if original_paths > 0 else 0
    }


def run_failure_simulations(
    G: nx.DiGraph,
    criticality_scores: Dict[str, CompositeCriticalityScore],
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Run failure simulations for top critical nodes
    
    Args:
        G: NetworkX graph
        criticality_scores: Pre-calculated criticality scores
        top_n: Number of top nodes to simulate
    
    Returns:
        Simulation results
    """
    logger.info(f"Running failure simulations for top {top_n} critical nodes...")
    
    # Get top critical nodes
    sorted_scores = sorted(
        criticality_scores.values(),
        key=lambda x: x.composite_score,
        reverse=True
    )[:top_n]
    
    simulations = []
    for score in sorted_scores:
        result = simulate_node_failure(G, score.component)
        result['criticality_score'] = round(score.composite_score, 4)
        result['criticality_level'] = score.criticality_level.value
        simulations.append(result)
    
    # Summary statistics
    total_isolated = sum(s['isolated_count'] for s in simulations)
    avg_fragments = sum(s['fragments_created'] for s in simulations) / max(1, len(simulations))
    
    return {
        'simulations_run': len(simulations),
        'summary': {
            'total_isolated_nodes': total_isolated,
            'avg_fragments_created': round(avg_fragments, 2),
            'max_fragments': max(s['fragments_created'] for s in simulations) if simulations else 0,
            'avg_connectivity_loss': round(
                sum(s['connectivity_loss_percentage'] for s in simulations) / max(1, len(simulations)), 2
            )
        },
        'simulations': simulations
    }


# ============================================================================
# Recommendations Generation
# ============================================================================

def generate_recommendations(
    node_analysis: Dict[str, Any],
    edge_analysis: Dict[str, Any],
    structure: Dict[str, Any],
    anti_patterns: List[AntiPattern]
) -> List[Dict[str, Any]]:
    """
    Generate actionable recommendations
    
    Args:
        node_analysis: Node criticality analysis results
        edge_analysis: Edge criticality analysis results
        structure: Graph structure analysis
        anti_patterns: Detected anti-patterns
    
    Returns:
        List of recommendations
    """
    logger.info("Generating recommendations...")
    
    recommendations = []
    priority = 1
    
    # High-criticality node recommendations
    for node in node_analysis.get('top_critical_nodes', [])[:5]:
        if node['score'] >= 0.7:
            recommendations.append({
                'priority': priority,
                'type': 'critical_node',
                'component': node['node'],
                'issue': f"High criticality score ({node['score']:.3f})",
                'recommendation': f"Add redundancy or load balancing for '{node['node']}'",
                'impact': 'high'
            })
            priority += 1
    
    # Bridge recommendations
    for bridge in edge_analysis.get('bridges', [])[:5]:
        recommendations.append({
            'priority': priority,
            'type': 'critical_edge',
            'component': f"{bridge['from']} -> {bridge['to']}",
            'issue': "Bridge edge - removal disconnects graph",
            'recommendation': f"Add alternative path between {bridge['from']} and {bridge['to']}",
            'impact': 'high'
        })
        priority += 1
    
    # Anti-pattern recommendations
    for ap in anti_patterns:
        recommendations.append({
            'priority': priority,
            'type': 'anti_pattern',
            'component': ', '.join(ap.components),
            'issue': ap.description,
            'recommendation': ap.recommendation,
            'impact': ap.severity
        })
        priority += 1
    
    # Structure-based recommendations
    if structure.get('number_weakly_connected_components', 1) > 1:
        recommendations.append({
            'priority': priority,
            'type': 'structure',
            'component': 'graph',
            'issue': f"Graph has {structure['number_weakly_connected_components']} disconnected components",
            'recommendation': "Review disconnected components - they may indicate missing relationships",
            'impact': 'medium'
        })
        priority += 1
    
    if structure.get('density', 0) > 0.3:
        recommendations.append({
            'priority': priority,
            'type': 'structure',
            'component': 'graph',
            'issue': f"High graph density ({structure['density']:.3f})",
            'recommendation': "Consider reducing coupling between components",
            'impact': 'low'
        })
        priority += 1
    
    return recommendations


# ============================================================================
# Export Functions
# ============================================================================

def export_results_json(results: Dict[str, Any], output_path: str):
    """Export results to JSON file"""
    logger.info(f"Exporting results to JSON: {output_path}")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✓ JSON export complete: {output_path}")


def export_results_csv(results: Dict[str, Any], output_dir: str):
    """Export results to CSV files"""
    logger.info(f"Exporting results to CSV: {output_dir}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export criticality scores
    if 'node_analysis' in results and 'all_scores' in results['node_analysis']:
        scores_file = output_path / 'criticality_scores.csv'
        with open(scores_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'component', 'component_type', 'composite_score', 'criticality_level',
                'betweenness_centrality_norm', 'is_articulation_point', 'impact_score',
                'reachability_loss_percentage', 'components_affected'
            ])
            writer.writeheader()
            for node, score in results['node_analysis']['all_scores'].items():
                writer.writerow(score)
        logger.info(f"  ✓ Criticality scores: {scores_file}")
    
    # Export recommendations
    if 'recommendations' in results:
        rec_file = output_path / 'recommendations.csv'
        with open(rec_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'priority', 'type', 'component', 'issue', 'recommendation', 'impact'
            ])
            writer.writeheader()
            for rec in results['recommendations']:
                writer.writerow(rec)
        logger.info(f"  ✓ Recommendations: {rec_file}")
    
    # Export edge analysis
    if 'edge_analysis' in results:
        edges_file = output_path / 'edge_betweenness.csv'
        with open(edges_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'from', 'to', 'score', 'is_bridge', 'edge_type'
            ])
            writer.writeheader()
            for edge in results['edge_analysis'].get('top_edge_betweenness', []):
                writer.writerow(edge)
        logger.info(f"  ✓ Edge analysis: {edges_file}")
    
    logger.info(f"✓ CSV export complete")


def export_results_html(results: Dict[str, Any], output_path: str):
    """Export results to HTML report"""
    logger.info(f"Exporting results to HTML: {output_path}")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pub-Sub System Analysis Report</title>
    <style>
        :root {{
            --primary: #3498db;
            --danger: #e74c3c;
            --warning: #f39c12;
            --success: #27ae60;
            --dark: #2c3e50;
            --light: #ecf0f1;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--light);
            color: var(--dark);
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        header {{
            background: linear-gradient(135deg, var(--primary), var(--dark));
            color: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 10px;
        }}
        h1 {{ font-size: 2.5rem; margin-bottom: 10px; }}
        .timestamp {{ opacity: 0.8; font-size: 0.9rem; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            color: var(--primary);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--light);
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid var(--light);
        }}
        .metric:last-child {{ border-bottom: none; }}
        .metric-value {{ font-weight: bold; color: var(--primary); }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--light);
        }}
        th {{ background: var(--primary); color: white; }}
        tr:hover {{ background: #f8f9fa; }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
        }}
        .badge-critical {{ background: var(--danger); color: white; }}
        .badge-high {{ background: var(--warning); color: white; }}
        .badge-medium {{ background: var(--primary); color: white; }}
        .badge-low {{ background: var(--success); color: white; }}
        .section {{ margin-bottom: 30px; }}
        .full-width {{ grid-column: 1 / -1; }}
        .progress-bar {{
            height: 8px;
            background: var(--light);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 5px;
        }}
        .progress-fill {{
            height: 100%;
            background: var(--primary);
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 Pub-Sub System Analysis Report</h1>
            <p class="timestamp">Generated: {results['metadata'].get('timestamp', datetime.now().isoformat())}</p>
            <p class="timestamp">Analysis Duration: {results['metadata'].get('analysis_time_seconds', 'N/A')}s</p>
        </header>
        
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="grid">
                <div class="card">
                    <h2>Graph Structure</h2>
                    <div class="metric">
                        <span>Total Nodes</span>
                        <span class="metric-value">{results['structure'].get('nodes', 0)}</span>
                    </div>
                    <div class="metric">
                        <span>Total Edges</span>
                        <span class="metric-value">{results['structure'].get('edges', 0)}</span>
                    </div>
                    <div class="metric">
                        <span>Graph Density</span>
                        <span class="metric-value">{results['structure'].get('density', 0):.4f}</span>
                    </div>
                    <div class="metric">
                        <span>Connected Components</span>
                        <span class="metric-value">{results['structure'].get('number_weakly_connected_components', 0)}</span>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Criticality Overview</h2>
                    <div class="metric">
                        <span>Critical Nodes</span>
                        <span class="metric-value" style="color: var(--danger);">
                            {results['node_analysis'].get('statistics', {}).get('critical_count', 0)}
                        </span>
                    </div>
                    <div class="metric">
                        <span>High Priority</span>
                        <span class="metric-value" style="color: var(--warning);">
                            {results['node_analysis'].get('statistics', {}).get('high_count', 0)}
                        </span>
                    </div>
                    <div class="metric">
                        <span>Articulation Points</span>
                        <span class="metric-value">
                            {results['node_analysis'].get('statistics', {}).get('articulation_point_count', 0)}
                        </span>
                    </div>
                    <div class="metric">
                        <span>Bridge Edges</span>
                        <span class="metric-value">
                            {results['edge_analysis'].get('statistics', {}).get('bridge_count', 0)}
                        </span>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Issues Detected</h2>
                    <div class="metric">
                        <span>Anti-Patterns</span>
                        <span class="metric-value" style="color: var(--danger);">
                            {len(results.get('anti_patterns', []))}
                        </span>
                    </div>
                    <div class="metric">
                        <span>Recommendations</span>
                        <span class="metric-value">
                            {len(results.get('recommendations', []))}
                        </span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="card full-width">
                <h2>🎯 Top Critical Nodes</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Component</th>
                            <th>Type</th>
                            <th>Score</th>
                            <th>Level</th>
                            <th>Articulation Point</th>
                            <th>Betweenness</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    # Add top critical nodes
    for idx, node in enumerate(results['node_analysis'].get('top_critical_nodes', [])[:10], 1):
        level_class = f"badge-{node.get('level', 'low')}"
        ap_marker = "✓" if node.get('is_articulation_point') else ""
        html += f"""
                        <tr>
                            <td>{idx}</td>
                            <td><strong>{node['node']}</strong></td>
                            <td>{node['type']}</td>
                            <td>{node['score']:.4f}</td>
                            <td><span class="badge {level_class}">{node['level'].upper()}</span></td>
                            <td style="text-align: center;">{ap_marker}</td>
                            <td>{node['betweenness']:.4f}</td>
                        </tr>
"""
    
    html += """
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="section">
            <div class="card full-width">
                <h2>🔗 Critical Edges (Bridges)</h2>
                <table>
                    <thead>
                        <tr>
                            <th>From</th>
                            <th>To</th>
                            <th>Type</th>
                            <th>Betweenness Score</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    # Add bridges
    for bridge in results['edge_analysis'].get('bridges', [])[:10]:
        html += f"""
                        <tr>
                            <td>{bridge['from']}</td>
                            <td>{bridge['to']}</td>
                            <td>{bridge.get('edge_type', 'Unknown')}</td>
                            <td>{bridge['betweenness']:.6f}</td>
                        </tr>
"""
    
    html += """
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="section">
            <div class="card full-width">
                <h2>💡 Recommendations</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Priority</th>
                            <th>Type</th>
                            <th>Component</th>
                            <th>Issue</th>
                            <th>Recommendation</th>
                            <th>Impact</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    # Add recommendations
    for rec in results.get('recommendations', [])[:15]:
        impact_class = f"badge-{rec.get('impact', 'medium')}"
        html += f"""
                        <tr>
                            <td>{rec.get('priority', '-')}</td>
                            <td>{rec.get('type', '-')}</td>
                            <td><strong>{rec.get('component', '-')}</strong></td>
                            <td>{rec.get('issue', '-')}</td>
                            <td>{rec.get('recommendation', '-')}</td>
                            <td><span class="badge {impact_class}">{rec.get('impact', '-').upper()}</span></td>
                        </tr>
"""
    
    html += """
                    </tbody>
                </table>
            </div>
        </div>
        
        <footer style="text-align: center; padding: 20px; color: #7f8c8d;">
            <p>Generated by Software-as-a-Graph Analysis Framework</p>
            <p>Research Project - Graph-Based Modeling and Analysis of Distributed Pub-Sub Systems</p>
        </footer>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"✓ HTML export complete: {output_path}")


# ============================================================================
# Output Printing
# ============================================================================

def print_summary(results: Dict[str, Any]):
    """Print colorful summary to terminal"""
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}  PUB-SUB SYSTEM ANALYSIS SUMMARY{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
    
    # Analysis method
    method = results.get('metadata', {}).get('options', {}).get('criticality_method', 'composite')
    defuzz = results.get('metadata', {}).get('options', {}).get('defuzzification_method', '')
    
    if method == 'fuzzy':
        print(f"\n{Colors.OKCYAN}🔮 Analysis Method: FUZZY LOGIC{Colors.ENDC}")
        print(f"   Defuzzification: {defuzz}")
    else:
        weights = results.get('metadata', {}).get('weights', {})
        print(f"\n{Colors.OKCYAN}📐 Analysis Method: COMPOSITE SCORING{Colors.ENDC}")
        print(f"   Weights: α={weights.get('alpha', 0.4)}, β={weights.get('beta', 0.3)}, γ={weights.get('gamma', 0.3)}")
    
    # Structure
    struct = results['structure']
    print(f"\n{Colors.OKBLUE}📊 Graph Structure{Colors.ENDC}")
    print(f"   Nodes: {struct['nodes']} | Edges: {struct['edges']}")
    print(f"   Density: {struct['density']:.4f}")
    print(f"   Connected: {'✓ Yes' if struct['is_weakly_connected'] else '✗ No'}")
    if not struct['is_weakly_connected']:
        print(f"   Components: {struct['number_weakly_connected_components']}")
    
    # Node Types
    if 'node_types' in struct:
        print(f"\n   Node Types:")
        for node_type, count in struct['node_types'].items():
            print(f"     • {node_type}: {count}")
    
    # Node Criticality
    node_stats = results['node_analysis'].get('statistics', {})
    print(f"\n{Colors.OKCYAN}🎯 Node Criticality{Colors.ENDC}")
    print(f"   Articulation Points: {node_stats.get('articulation_point_count', 0)} "
          f"({node_stats.get('articulation_point_percentage', 0):.1f}%)")
    print(f"   Avg Criticality Score: {node_stats.get('avg_criticality', 0):.4f}")
    print(f"\n   Distribution:")
    print(f"     🔴 Critical: {node_stats.get('critical_count', 0)}")
    print(f"     🟠 High:     {node_stats.get('high_count', 0)}")
    print(f"     🟡 Medium:   {node_stats.get('medium_count', 0)}")
    print(f"     🟢 Low:      {node_stats.get('low_count', 0)}")
    print(f"     ⚪ Minimal:  {node_stats.get('minimal_count', 0)}")
    
    # Top Critical
    print(f"\n   Top 5 Critical Nodes:")
    for idx, node in enumerate(results['node_analysis'].get('top_critical_nodes', [])[:5], 1):
        ap_marker = f" {Colors.FAIL}[AP]{Colors.ENDC}" if node.get('is_articulation_point') else ""
        
        # Show membership degrees for fuzzy analysis
        if method == 'fuzzy' and 'membership_degrees' in node:
            memberships = node['membership_degrees']
            # Find dominant membership
            if memberships:
                dominant = max(memberships.items(), key=lambda x: x[1])
                membership_info = f" ({dominant[0]}:{dominant[1]:.2f})"
            else:
                membership_info = ""
            print(f"     {idx}. {node['node']}{ap_marker} (score: {node['score']:.4f}){membership_info}")
        else:
            print(f"     {idx}. {node['node']}{ap_marker} (score: {node['score']:.4f})")
    
    # Edge Analysis
    if 'edge_analysis' in results and results['edge_analysis'].get('statistics'):
        edge_stats = results['edge_analysis']['statistics']
        print(f"\n{Colors.OKBLUE}🔗 Edge Criticality{Colors.ENDC}")
        print(f"   Bridges: {edge_stats.get('bridge_count', 0)} ({edge_stats.get('bridge_percentage', 0):.1f}%)")
        
        if method == 'fuzzy':
            print(f"   Critical Edges: {edge_stats.get('critical_count', 0)}")
            print(f"   High Criticality Edges: {edge_stats.get('high_count', 0)}")
            print(f"   Avg Edge Criticality: {edge_stats.get('avg_edge_criticality', 0):.4f}")
        else:
            print(f"   Avg Edge Betweenness: {edge_stats.get('avg_edge_betweenness', 0):.6f}")
        
        # Top critical edges
        top_edges = results['edge_analysis'].get('top_edge_criticality', 
                    results['edge_analysis'].get('top_edge_betweenness', []))[:5]
        if top_edges:
            print(f"\n   Top 5 Critical Edges:")
            for idx, edge in enumerate(top_edges, 1):
                bridge_marker = f" {Colors.WARNING}[BRIDGE]{Colors.ENDC}" if edge.get('is_bridge') else ""
                print(f"     {idx}. {edge['from']} → {edge['to']}{bridge_marker} "
                      f"(score: {edge['score']:.4f})")
    
    # Anti-patterns
    if results.get('anti_patterns'):
        print(f"\n{Colors.WARNING}⚠️  Anti-Patterns Detected: {len(results['anti_patterns'])}{Colors.ENDC}")
        for ap in results['anti_patterns'][:3]:
            print(f"   • [{ap['severity'].upper()}] {ap['type']}: {ap['components'][0] if ap['components'] else 'N/A'}")
    
    # Recommendations
    print(f"\n{Colors.OKGREEN}💡 Recommendations: {len(results.get('recommendations', []))}{Colors.ENDC}")
    for rec in results.get('recommendations', [])[:3]:
        print(f"   {rec['priority']}. [{rec['impact'].upper()}] {rec['type']}: {rec['component']}")
    
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}\n")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Comprehensive Pub-Sub System Graph Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis from JSON (composite scoring)
  python analyze_graph.py --input system.json
  
  # Analysis using FUZZY LOGIC (recommended for smooth transitions)
  python analyze_graph.py --input system.json --fuzzy
  
  # Fuzzy analysis with specific defuzzification method
  python analyze_graph.py --input system.json --fuzzy --defuzz mom
  
  # Full analysis with fuzzy logic and all exports
  python analyze_graph.py --input system.json --fuzzy --analyze-edges \\
      --simulate --detect-antipatterns \\
      --export-json --export-csv --export-html
  
  # From Neo4j database with fuzzy analysis
  python analyze_graph.py --neo4j --uri bolt://localhost:7687 \\
      --user neo4j --password password --fuzzy
  
  # Custom composite scoring weights
  python analyze_graph.py --input system.json \\
      --alpha 0.5 --beta 0.3 --gamma 0.2
  
  # Quick analysis (skip expensive calculations)
  python analyze_graph.py --input system.json --quick

Criticality Methods:
  composite  - Traditional weighted formula: C = α·BC + β·AP + γ·I
  fuzzy      - Fuzzy logic inference with smooth transitions (recommended)

Defuzzification Methods (for fuzzy analysis):
  centroid   - Center of gravity (default, most common)
  mom        - Mean of maximum
  bisector   - Bisector of area
  som        - Smallest of maximum
  lom        - Largest of maximum
        """
    )
    
    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', type=str,
                            help='Input JSON file path')
    input_group.add_argument('--neo4j', action='store_true',
                            help='Load from Neo4j database')
    
    # Neo4j connection
    parser.add_argument('--uri', type=str, default='bolt://localhost:7687',
                       help='Neo4j connection URI')
    parser.add_argument('--user', type=str, default='neo4j',
                       help='Neo4j username')
    parser.add_argument('--password', type=str, default='password',
                       help='Neo4j password')
    
    # Analysis options
    parser.add_argument('--analyze-edges', action='store_true',
                       help='Perform edge criticality analysis')
    parser.add_argument('--no-edge-analysis', action='store_true',
                       help='Skip edge analysis (faster)')
    parser.add_argument('--simulate', action='store_true',
                       help='Run failure simulations')
    parser.add_argument('--detect-antipatterns', action='store_true',
                       help='Detect anti-patterns')
    parser.add_argument('--quick', action='store_true',
                       help='Quick analysis (skip impact calculations)')
    
    # Criticality method selection
    parser.add_argument('--fuzzy', action='store_true',
                       help='Use fuzzy logic for criticality analysis (recommended)')
    parser.add_argument('--method', type=str, default='composite',
                       choices=['composite', 'fuzzy'],
                       help='Criticality calculation method (default: composite)')
    parser.add_argument('--defuzz', type=str, default='centroid',
                       choices=['centroid', 'mom', 'bisector', 'som', 'lom'],
                       help='Defuzzification method for fuzzy analysis (default: centroid)')
    
    # Criticality weights (for composite method)
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                       help=f'Betweenness centrality weight (default: {DEFAULT_ALPHA})')
    parser.add_argument('--beta', type=float, default=DEFAULT_BETA,
                       help=f'Articulation point weight (default: {DEFAULT_BETA})')
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA,
                       help=f'Impact score weight (default: {DEFAULT_GAMMA})')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, default='output/analysis',
                       help='Output path prefix (default: output/analysis)')
    parser.add_argument('--export-json', action='store_true',
                       help='Export results to JSON')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export results to CSV files')
    parser.add_argument('--export-html', action='store_true',
                       help='Export results to HTML report')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress output (except errors)')
    parser.add_argument('--log-file', type=str,
                       help='Log to file')
    parser.add_argument('--no-color', action='store_true',
                       help='Disable colored output')
    
    args = parser.parse_args()
    
    # Setup
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()
    
    global logger
    logger = setup_logging(args.verbose, args.log_file)
    
    if args.quiet:
        logger.setLevel(logging.ERROR)
    
    # Validate weights
    weight_sum = args.alpha + args.beta + args.gamma
    if abs(weight_sum - 1.0) > 0.01:
        logger.warning(f"Weights sum to {weight_sum:.2f}, normalizing...")
        args.alpha /= weight_sum
        args.beta /= weight_sum
        args.gamma /= weight_sum
    
    try:
        start_time = time.time()
        
        # Load graph
        if args.neo4j:
            G, metadata = load_graph_from_neo4j(args.uri, args.user, args.password)
        else:
            G, metadata = load_graph_from_json(args.input)
        
        # Initialize results
        results = {
            'metadata': {
                'source': args.input if args.input else args.uri,
                'timestamp': datetime.now().isoformat(),
                'weights': {'alpha': args.alpha, 'beta': args.beta, 'gamma': args.gamma},
                'options': {
                    'analyze_edges': not args.no_edge_analysis,
                    'simulate': args.simulate,
                    'detect_antipatterns': args.detect_antipatterns,
                    'quick_mode': args.quick,
                    'criticality_method': 'fuzzy' if (args.fuzzy or args.method == 'fuzzy') else 'composite',
                    'defuzzification_method': args.defuzz if (args.fuzzy or args.method == 'fuzzy') else None
                }
            },
            'structure': {},
            'node_analysis': {},
            'edge_analysis': {},
            'layer_analysis': {},
            'anti_patterns': [],
            'failure_simulation': {},
            'recommendations': []
        }
        
        # Determine analysis method
        use_fuzzy = args.fuzzy or args.method == 'fuzzy'
        
        if use_fuzzy and not FUZZY_AVAILABLE:
            logger.warning("Fuzzy logic requested but not available. "
                          "Ensure src/analysis/fuzzy_criticality_analyzer.py exists.")
            logger.warning("Falling back to composite scoring method.")
            use_fuzzy = False
        
        if use_fuzzy:
            logger.info(f"Using FUZZY LOGIC analysis (defuzzification: {args.defuzz})")
        else:
            logger.info(f"Using COMPOSITE SCORING analysis (α={args.alpha}, β={args.beta}, γ={args.gamma})")
        
        # 1. Structure analysis
        results['structure'] = analyze_graph_structure(G)
        
        # 2. Node criticality analysis (fuzzy or composite)
        calculate_impact = not args.quick
        
        if use_fuzzy:
            results['node_analysis'] = analyze_node_criticality_fuzzy(
                G, 
                calculate_impact=calculate_impact,
                defuzz_method=args.defuzz
            )
        else:
            results['node_analysis'] = analyze_node_criticality(
                G, args.alpha, args.beta, args.gamma, calculate_impact
            )
        
        # 3. Edge criticality analysis (fuzzy or standard)
        if not args.no_edge_analysis:
            if use_fuzzy:
                results['edge_analysis'] = analyze_edge_criticality_fuzzy(G, args.defuzz)
            else:
                results['edge_analysis'] = analyze_edge_criticality(G)
        else:
            results['edge_analysis'] = {'skipped': True, 'statistics': {}}
        
        # 4. Layer analysis
        results['layer_analysis'] = analyze_layer_dependencies(G)
        
        # 5. Anti-pattern detection
        if args.detect_antipatterns:
            criticality_scores = {
                node: CompositeCriticalityScore(
                    component=node,
                    component_type=score_data['component_type'],
                    composite_score=score_data['composite_score'],
                    is_articulation_point=score_data['is_articulation_point'],
                    criticality_level=CriticalityLevel(score_data['criticality_level'])
                )
                for node, score_data in results['node_analysis']['all_scores'].items()
            }
            anti_patterns = detect_anti_patterns(G, criticality_scores)
            results['anti_patterns'] = [ap.to_dict() for ap in anti_patterns]
        
        # 6. Failure simulation
        if args.simulate:
            criticality_scores = {
                node: CompositeCriticalityScore(
                    component=node,
                    component_type=score_data['component_type'],
                    composite_score=score_data['composite_score'],
                    is_articulation_point=score_data['is_articulation_point'],
                    criticality_level=CriticalityLevel(score_data['criticality_level'])
                )
                for node, score_data in results['node_analysis']['all_scores'].items()
            }
            results['failure_simulation'] = run_failure_simulations(G, criticality_scores)
        
        # 7. Generate recommendations
        results['recommendations'] = generate_recommendations(
            results['node_analysis'],
            results['edge_analysis'],
            results['structure'],
            [AntiPattern(**ap) for ap in results['anti_patterns']] if results['anti_patterns'] else []
        )
        
        # Record analysis time
        analysis_time = time.time() - start_time
        results['metadata']['analysis_time_seconds'] = round(analysis_time, 2)
        
        # Print summary
        if not args.quiet:
            print_summary(results)
        
        # Export results
        output_base = Path(args.output)
        
        if args.export_json:
            export_results_json(results, str(output_base) + '.json')
        
        if args.export_csv:
            export_results_csv(results, str(output_base) + '_csv')
        
        if args.export_html:
            export_results_html(results, str(output_base) + '.html')
        
        logger.info(f"✓ Analysis complete in {analysis_time:.2f}s")
        return 0
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        return 130
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid data: {e}")
        return 1
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
