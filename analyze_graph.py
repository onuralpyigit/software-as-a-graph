#!/usr/bin/env python3
"""
Graph Analysis CLI

Comprehensive analysis of pub-sub system graphs including:
- Criticality scoring: C_score(v) = α·C_B^norm(v) + β·AP(v) + γ·I(v)
- Structural analysis (articulation points, bridges, communities)
- Anti-pattern detection (SPOF, god topics, circular dependencies)
- Failure simulation and impact assessment
- Multi-layer dependency analysis
- QoS-aware analysis

Usage Examples:
    # Basic analysis from JSON file
    python analyze_graph.py --input system.json
    
    # Full analysis with exports
    python analyze_graph.py --input system.json \\
        --detect-antipatterns --simulate \\
        --export-json results.json --export-csv results.csv
    
    # Analysis from Neo4j database
    python analyze_graph.py --neo4j --uri bolt://localhost:7687 \\
        --user neo4j --password password
    
    # Custom criticality weights
    python analyze_graph.py --input system.json \\
        --alpha 0.5 --beta 0.25 --gamma 0.25

Research Target Metrics:
    - Spearman correlation ≥ 0.7 with failure simulations
    - F1-score ≥ 0.9 for critical component identification
    - Precision ≥ 0.9, Recall ≥ 0.85
"""

import argparse
import json
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Error: networkx is required. Install with: pip install networkx")
    sys.exit(1)

try:
    from src.core.graph_builder import GraphBuilder
    HAS_BUILDER = True
except ImportError:
    HAS_BUILDER = False


# =============================================================================
# Terminal Colors
# =============================================================================

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

    @classmethod
    def disable(cls):
        cls.HEADER = cls.BLUE = cls.CYAN = cls.GREEN = ''
        cls.WARNING = cls.FAIL = cls.ENDC = cls.BOLD = cls.DIM = ''


def print_header(text: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")


def print_section(text: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.DIM}{'-'*50}{Colors.ENDC}")


def print_success(text: str):
    print(f"{Colors.GREEN}✓{Colors.ENDC} {text}")


def print_error(text: str):
    print(f"{Colors.FAIL}✗{Colors.ENDC} {text}")


def print_warning(text: str):
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {text}")


def print_info(text: str):
    print(f"{Colors.CYAN}ℹ{Colors.ENDC} {text}")


def print_kv(key: str, value: Any, indent: int = 2):
    spaces = ' ' * indent
    print(f"{spaces}{Colors.DIM}{key}:{Colors.ENDC} {value}")


# =============================================================================
# Data Classes
# =============================================================================

class CriticalityLevel(Enum):
    """Criticality classification levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    MINIMAL = "MINIMAL"


@dataclass
class CriticalityScore:
    """Criticality assessment for a component"""
    component: str
    component_type: str
    betweenness_centrality_norm: float
    is_articulation_point: bool
    impact_score: float
    composite_score: float
    criticality_level: CriticalityLevel
    degree_centrality: float = 0.0
    closeness_centrality: float = 0.0
    pagerank: float = 0.0
    components_affected: int = 0
    reachability_loss_pct: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'component': self.component,
            'type': self.component_type,
            'composite_score': round(self.composite_score, 4),
            'level': self.criticality_level.value,
            'betweenness_norm': round(self.betweenness_centrality_norm, 4),
            'is_articulation_point': self.is_articulation_point,
            'impact_score': round(self.impact_score, 4),
            'degree_centrality': round(self.degree_centrality, 4),
            'closeness_centrality': round(self.closeness_centrality, 4),
            'pagerank': round(self.pagerank, 4),
            'components_affected': self.components_affected,
            'reachability_loss_pct': round(self.reachability_loss_pct, 2)
        }


@dataclass
class AntiPattern:
    """Detected anti-pattern"""
    pattern_type: str
    severity: str
    components: List[str]
    description: str
    recommendation: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'type': self.pattern_type,
            'severity': self.severity,
            'components': self.components,
            'description': self.description,
            'recommendation': self.recommendation,
            'metrics': self.metrics
        }


@dataclass 
class AnalysisResults:
    """Complete analysis results"""
    timestamp: str
    graph_summary: Dict[str, Any]
    criticality_scores: Dict[str, CriticalityScore]
    structural_analysis: Dict[str, Any]
    antipatterns: List[AntiPattern]
    simulation_results: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    execution_time: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'graph_summary': self.graph_summary,
            'criticality_analysis': {
                'scores': {k: v.to_dict() for k, v in self.criticality_scores.items()},
                'summary': self._criticality_summary()
            },
            'structural_analysis': self.structural_analysis,
            'antipatterns': [ap.to_dict() for ap in self.antipatterns],
            'simulation_results': self.simulation_results,
            'recommendations': self.recommendations,
            'execution_time': self.execution_time
        }
    
    def _criticality_summary(self) -> Dict:
        scores = list(self.criticality_scores.values())
        if not scores:
            return {}
        
        all_scores = [s.composite_score for s in scores]
        return {
            'total_components': len(scores),
            'avg_score': round(sum(all_scores) / len(all_scores), 4),
            'max_score': round(max(all_scores), 4),
            'critical_count': sum(1 for s in scores if s.criticality_level == CriticalityLevel.CRITICAL),
            'high_count': sum(1 for s in scores if s.criticality_level == CriticalityLevel.HIGH),
            'medium_count': sum(1 for s in scores if s.criticality_level == CriticalityLevel.MEDIUM),
            'low_count': sum(1 for s in scores if s.criticality_level == CriticalityLevel.LOW),
            'minimal_count': sum(1 for s in scores if s.criticality_level == CriticalityLevel.MINIMAL),
            'articulation_points': sum(1 for s in scores if s.is_articulation_point)
        }


# =============================================================================
# Graph Loading
# =============================================================================

def load_graph_from_json(filepath: str) -> Tuple[nx.DiGraph, Dict]:
    """
    Load graph from JSON file and convert to NetworkX.
    
    Returns:
        Tuple of (NetworkX DiGraph, original data dict)
    """
    with open(filepath) as f:
        data = json.load(f)
    
    G = nx.DiGraph()
    
    # Add nodes
    for node in data.get('nodes', []):
        node_attrs = {k: v for k, v in node.items() if k != 'id'}
        node_attrs['type'] = 'Node'
        if 'name' not in node_attrs:
            node_attrs['name'] = node['id']
        G.add_node(node['id'], **node_attrs)
    
    for broker in data.get('brokers', []):
        broker_attrs = {k: v for k, v in broker.items() if k != 'id'}
        broker_attrs['type'] = 'Broker'
        if 'name' not in broker_attrs:
            broker_attrs['name'] = broker['id']
        G.add_node(broker['id'], **broker_attrs)
    
    for app in data.get('applications', []):
        app_attrs = {k: v for k, v in app.items() if k != 'id'}
        app_attrs['type'] = 'Application'
        if 'name' not in app_attrs:
            app_attrs['name'] = app['id']
        G.add_node(app['id'], **app_attrs)
    
    for topic in data.get('topics', []):
        # Flatten QoS
        topic_attrs = {k: v for k, v in topic.items() if k not in ['id', 'qos']}
        topic_attrs['type'] = 'Topic'
        if 'name' not in topic_attrs:
            topic_attrs['name'] = topic['id']
        if 'qos' in topic:
            for qk, qv in topic['qos'].items():
                topic_attrs[f'qos_{qk}'] = qv
        G.add_node(topic['id'], **topic_attrs)
    
    # Add edges
    relationships = data.get('relationships', {})
    
    for rel in relationships.get('runs_on', []):
        source = rel.get('from', rel.get('source'))
        target = rel.get('to', rel.get('target'))
        if source and target:
            G.add_edge(source, target, type='RUNS_ON')
    
    for rel in relationships.get('publishes_to', []):
        source = rel.get('from', rel.get('source'))
        target = rel.get('to', rel.get('target'))
        if source and target:
            G.add_edge(source, target, type='PUBLISHES_TO', 
                      period_ms=rel.get('period_ms'), msg_size=rel.get('msg_size_bytes'))
    
    for rel in relationships.get('subscribes_to', []):
        source = rel.get('from', rel.get('source'))
        target = rel.get('to', rel.get('target'))
        if source and target:
            G.add_edge(source, target, type='SUBSCRIBES_TO')
    
    for rel in relationships.get('routes', []):
        source = rel.get('from', rel.get('source'))
        target = rel.get('to', rel.get('target'))
        if source and target:
            G.add_edge(source, target, type='ROUTES')
    
    # Derive DEPENDS_ON edges
    G = derive_dependencies(G)
    
    return G, data


def derive_dependencies(G: nx.DiGraph) -> nx.DiGraph:
    """Derive DEPENDS_ON relationships from pub/sub patterns"""
    
    # APP_TO_APP: subscriber depends on publisher via shared topic
    topics = [n for n, d in G.nodes(data=True) if d.get('type') == 'Topic']
    
    for topic in topics:
        # Find publishers and subscribers
        publishers = [s for s, t, d in G.in_edges(topic, data=True) 
                     if d.get('type') == 'PUBLISHES_TO']
        subscribers = [s for s, t, d in G.in_edges(topic, data=True) 
                      if d.get('type') == 'SUBSCRIBES_TO']
        
        # Subscribers depend on publishers
        for sub in subscribers:
            for pub in publishers:
                if sub != pub and not G.has_edge(sub, pub):
                    G.add_edge(sub, pub, type='DEPENDS_ON', 
                              dependency_type='app_to_app', via_topic=topic)
    
    return G


def load_graph_from_neo4j(uri: str, user: str, password: str, database: str = 'neo4j') -> nx.DiGraph:
    """Load graph from Neo4j database"""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise ImportError("neo4j driver required. Install with: pip install neo4j")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    G = nx.DiGraph()
    
    with driver.session(database=database) as session:
        # Load all nodes
        result = session.run("""
            MATCH (n)
            RETURN n.id AS id, labels(n)[0] AS type, properties(n) AS props
        """)
        for record in result:
            props = dict(record['props'])
            props['type'] = record['type']
            G.add_node(record['id'], **props)
        
        # Load all relationships
        result = session.run("""
            MATCH (s)-[r]->(t)
            RETURN s.id AS source, t.id AS target, type(r) AS rel_type, properties(r) AS props
        """)
        for record in result:
            props = dict(record['props']) if record['props'] else {}
            props['type'] = record['rel_type']
            G.add_edge(record['source'], record['target'], **props)
    
    driver.close()
    return G


# =============================================================================
# Criticality Analysis
# =============================================================================

DEFAULT_ALPHA = 0.4  # Betweenness centrality weight
DEFAULT_BETA = 0.3   # Articulation point weight
DEFAULT_GAMMA = 0.3  # Impact score weight


def calculate_betweenness_centrality(G: nx.DiGraph) -> Dict[str, float]:
    """Calculate betweenness centrality for all nodes"""
    return nx.betweenness_centrality(G, normalized=True)


def find_articulation_points(G: nx.DiGraph) -> Set[str]:
    """Find articulation points (cut vertices)"""
    undirected = G.to_undirected()
    return set(nx.articulation_points(undirected))


def calculate_impact_score(G: nx.DiGraph, node: str) -> Tuple[float, int, float]:
    """
    Calculate impact score I(v) for removing a node.
    
    I(v) = 1 - |R(G-v)| / |R(G)|
    
    Returns:
        Tuple of (impact_score, components_affected, reachability_loss_pct)
    """
    # Calculate original reachability
    original_reachable = set()
    for source in G.nodes():
        for target in nx.descendants(G, source):
            if source != target:
                original_reachable.add((source, target))
    
    # Calculate reachability without the node
    G_minus_v = G.copy()
    G_minus_v.remove_node(node)
    
    new_reachable = set()
    for source in G_minus_v.nodes():
        for target in nx.descendants(G_minus_v, source):
            if source != target:
                new_reachable.add((source, target))
    
    # Calculate loss
    lost_reachability = original_reachable - new_reachable
    components_affected = len(set(p[0] for p in lost_reachability) | 
                             set(p[1] for p in lost_reachability))
    
    if len(original_reachable) > 0:
        reachability_loss_pct = (len(lost_reachability) / len(original_reachable)) * 100
        impact_score = len(lost_reachability) / len(original_reachable)
    else:
        reachability_loss_pct = 0.0
        impact_score = 0.0
    
    return min(1.0, impact_score), components_affected, reachability_loss_pct


def classify_criticality(score: float) -> CriticalityLevel:
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
    calculate_impact: bool = True,
    logger: logging.Logger = None
) -> Dict[str, CriticalityScore]:
    """
    Calculate composite criticality scores for all nodes.
    
    Formula: C_score(v) = α·C_B^norm(v) + β·AP(v) + γ·I(v)
    """
    if logger:
        logger.info("Calculating criticality scores...")
    
    # Calculate centrality metrics
    betweenness = calculate_betweenness_centrality(G)
    articulation_points = find_articulation_points(G)
    degree = nx.degree_centrality(G)
    closeness = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G, alpha=0.85)
    
    # Normalize betweenness
    max_bc = max(betweenness.values()) if betweenness else 1.0
    if max_bc == 0:
        max_bc = 1.0
    
    scores = {}
    total = len(G.nodes())
    
    for i, node in enumerate(G.nodes()):
        node_data = G.nodes[node]
        node_type = node_data.get('type', 'Unknown')
        
        # Normalized betweenness
        bc_norm = betweenness.get(node, 0.0) / max_bc
        
        # Articulation point indicator
        is_ap = node in articulation_points
        ap_indicator = 1.0 if is_ap else 0.0
        
        # Impact score
        if calculate_impact:
            impact, affected, loss_pct = calculate_impact_score(G, node)
        else:
            impact, affected, loss_pct = 0.0, 0, 0.0
        
        # Composite score
        composite = alpha * bc_norm + beta * ap_indicator + gamma * impact
        
        # Classify
        level = classify_criticality(composite)
        
        scores[node] = CriticalityScore(
            component=node,
            component_type=node_type,
            betweenness_centrality_norm=bc_norm,
            is_articulation_point=is_ap,
            impact_score=impact,
            composite_score=composite,
            criticality_level=level,
            degree_centrality=degree.get(node, 0.0),
            closeness_centrality=closeness.get(node, 0.0),
            pagerank=pagerank.get(node, 0.0),
            components_affected=affected,
            reachability_loss_pct=loss_pct
        )
        
        if logger and (i + 1) % 50 == 0:
            logger.debug(f"  Processed {i + 1}/{total} nodes")
    
    return scores


# =============================================================================
# Structural Analysis
# =============================================================================

def analyze_structure(G: nx.DiGraph) -> Dict[str, Any]:
    """Perform structural analysis of the graph"""
    
    # Basic metrics
    undirected = G.to_undirected()
    
    results = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_connected': nx.is_weakly_connected(G),
        'connected_components': nx.number_weakly_connected_components(G)
    }
    
    # Node types
    node_types = defaultdict(int)
    for _, data in G.nodes(data=True):
        node_types[data.get('type', 'Unknown')] += 1
    results['node_types'] = dict(node_types)
    
    # Edge types
    edge_types = defaultdict(int)
    for _, _, data in G.edges(data=True):
        edge_types[data.get('type', 'Unknown')] += 1
    results['edge_types'] = dict(edge_types)
    
    # Degree statistics
    degrees = [d for _, d in G.degree()]
    if degrees:
        results['degree_stats'] = {
            'avg': round(sum(degrees) / len(degrees), 2),
            'max': max(degrees),
            'min': min(degrees)
        }
    
    # Articulation points
    aps = list(nx.articulation_points(undirected))
    results['articulation_points'] = {
        'count': len(aps),
        'nodes': aps[:20]  # Top 20
    }
    
    # Bridges
    bridges = list(nx.bridges(undirected))
    results['bridges'] = {
        'count': len(bridges),
        'edges': bridges[:20]
    }
    
    # Clustering
    try:
        results['avg_clustering'] = round(nx.average_clustering(undirected), 4)
    except:
        results['avg_clustering'] = 0.0
    
    # Diameter (if connected)
    if results['is_connected']:
        try:
            results['diameter'] = nx.diameter(undirected)
        except:
            results['diameter'] = None
    
    return results


# =============================================================================
# Anti-Pattern Detection
# =============================================================================

def detect_antipatterns(G: nx.DiGraph, data: Dict = None) -> List[AntiPattern]:
    """Detect architectural anti-patterns in the graph"""
    antipatterns = []
    
    # 1. Single Points of Failure (SPOF)
    antipatterns.extend(detect_spof(G))
    
    # 2. God Topics
    antipatterns.extend(detect_god_topics(G))
    
    # 3. Circular Dependencies
    antipatterns.extend(detect_circular_dependencies(G))
    
    # 4. Broker Overload
    antipatterns.extend(detect_broker_overload(G))
    
    # 5. Tight Coupling
    antipatterns.extend(detect_tight_coupling(G))
    
    # 6. Orphan Components
    antipatterns.extend(detect_orphans(G))
    
    # Check for injected anti-patterns in data
    if data and 'injected_antipatterns' in data:
        for ap in data['injected_antipatterns']:
            antipatterns.append(AntiPattern(
                pattern_type=f"INJECTED_{ap.get('type', 'unknown').upper()}",
                severity='HIGH',
                components=ap.get('components', [ap.get('component', 'unknown')]),
                description=ap.get('description', 'Injected anti-pattern'),
                recommendation='This was intentionally injected for testing',
                metrics=ap
            ))
    
    return antipatterns


def detect_spof(G: nx.DiGraph) -> List[AntiPattern]:
    """Detect Single Points of Failure"""
    antipatterns = []
    
    # Find nodes with high in-degree of DEPENDS_ON
    depends_on_count = defaultdict(int)
    for s, t, d in G.edges(data=True):
        if d.get('type') == 'DEPENDS_ON':
            depends_on_count[t] += 1
    
    # Threshold: 5+ dependents
    for node, count in depends_on_count.items():
        if count >= 5:
            node_type = G.nodes[node].get('type', 'Unknown')
            antipatterns.append(AntiPattern(
                pattern_type='SPOF',
                severity='CRITICAL' if count >= 10 else 'HIGH',
                components=[node],
                description=f"{node_type} '{node}' has {count} dependent components",
                recommendation=f"Add redundancy or redistribute responsibilities from {node}",
                metrics={'dependents': count, 'node_type': node_type}
            ))
    
    # Also check articulation points
    undirected = G.to_undirected()
    for ap in nx.articulation_points(undirected):
        if ap not in depends_on_count or depends_on_count[ap] < 5:
            node_type = G.nodes[ap].get('type', 'Unknown')
            antipatterns.append(AntiPattern(
                pattern_type='SPOF_ARTICULATION',
                severity='HIGH',
                components=[ap],
                description=f"{node_type} '{ap}' is an articulation point",
                recommendation=f"Add redundant path around {ap}",
                metrics={'node_type': node_type, 'is_articulation_point': True}
            ))
    
    return antipatterns


def detect_god_topics(G: nx.DiGraph) -> List[AntiPattern]:
    """Detect God Topics (topics with too many connections)"""
    antipatterns = []
    
    topics = [n for n, d in G.nodes(data=True) if d.get('type') == 'Topic']
    
    for topic in topics:
        # Count publishers and subscribers
        publishers = [s for s, t, d in G.in_edges(topic, data=True) 
                     if d.get('type') == 'PUBLISHES_TO']
        subscribers = [s for s, t, d in G.in_edges(topic, data=True) 
                      if d.get('type') == 'SUBSCRIBES_TO']
        
        total = len(publishers) + len(subscribers)
        
        # Threshold: 10+ total connections
        if total >= 10:
            antipatterns.append(AntiPattern(
                pattern_type='GOD_TOPIC',
                severity='HIGH' if total >= 15 else 'MEDIUM',
                components=[topic],
                description=f"Topic '{topic}' has {len(publishers)} publishers and {len(subscribers)} subscribers",
                recommendation=f"Split '{topic}' into more specific topics",
                metrics={'publishers': len(publishers), 'subscribers': len(subscribers), 'total': total}
            ))
    
    return antipatterns


def detect_circular_dependencies(G: nx.DiGraph) -> List[AntiPattern]:
    """Detect circular dependency chains"""
    antipatterns = []
    
    # Find cycles in DEPENDS_ON edges
    depends_on_graph = nx.DiGraph()
    for s, t, d in G.edges(data=True):
        if d.get('type') == 'DEPENDS_ON':
            depends_on_graph.add_edge(s, t)
    
    try:
        cycles = list(nx.simple_cycles(depends_on_graph))
        
        # Filter to meaningful cycles (length 2-6)
        for cycle in cycles:
            if 2 <= len(cycle) <= 6:
                antipatterns.append(AntiPattern(
                    pattern_type='CIRCULAR_DEPENDENCY',
                    severity='HIGH' if len(cycle) <= 3 else 'MEDIUM',
                    components=cycle,
                    description=f"Circular dependency chain: {' → '.join(cycle)} → {cycle[0]}",
                    recommendation="Break the cycle by introducing an abstraction layer",
                    metrics={'cycle_length': len(cycle)}
                ))
    except:
        pass
    
    return antipatterns


def detect_broker_overload(G: nx.DiGraph) -> List[AntiPattern]:
    """Detect overloaded brokers"""
    antipatterns = []
    
    brokers = [n for n, d in G.nodes(data=True) if d.get('type') == 'Broker']
    
    for broker in brokers:
        # Count topics routed
        topics = [t for s, t, d in G.out_edges(broker, data=True) 
                 if d.get('type') == 'ROUTES']
        
        # Check load
        load = G.nodes[broker].get('current_load', 0.0)
        
        if len(topics) >= 20 or load >= 0.8:
            antipatterns.append(AntiPattern(
                pattern_type='BROKER_OVERLOAD',
                severity='CRITICAL' if load >= 0.9 else 'HIGH',
                components=[broker],
                description=f"Broker '{broker}' routes {len(topics)} topics with {load*100:.0f}% load",
                recommendation=f"Distribute topics across additional brokers",
                metrics={'topics_routed': len(topics), 'current_load': load}
            ))
    
    return antipatterns


def detect_tight_coupling(G: nx.DiGraph) -> List[AntiPattern]:
    """Detect tightly coupled component clusters"""
    antipatterns = []
    
    # Build dependency graph
    apps = [n for n, d in G.nodes(data=True) if d.get('type') == 'Application']
    
    # Count bidirectional dependencies
    dep_pairs = defaultdict(int)
    for s, t, d in G.edges(data=True):
        if d.get('type') == 'DEPENDS_ON' and s in apps and t in apps:
            pair = tuple(sorted([s, t]))
            dep_pairs[pair] += 1
    
    # Find highly coupled pairs
    for (a, b), count in dep_pairs.items():
        if count >= 2:  # Bidirectional
            antipatterns.append(AntiPattern(
                pattern_type='TIGHT_COUPLING',
                severity='MEDIUM',
                components=[a, b],
                description=f"Bidirectional dependency between '{a}' and '{b}'",
                recommendation="Consider merging or introducing an intermediary",
                metrics={'dependency_count': count}
            ))
    
    return antipatterns


def detect_orphans(G: nx.DiGraph) -> List[AntiPattern]:
    """Detect orphan components with no connections"""
    antipatterns = []
    
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        
        if in_deg == 0 and out_deg == 0:
            node_type = G.nodes[node].get('type', 'Unknown')
            antipatterns.append(AntiPattern(
                pattern_type='ORPHAN_COMPONENT',
                severity='LOW',
                components=[node],
                description=f"{node_type} '{node}' has no connections",
                recommendation=f"Remove unused component or add necessary connections",
                metrics={'node_type': node_type}
            ))
    
    return antipatterns


# =============================================================================
# Failure Simulation
# =============================================================================

def simulate_failures(
    G: nx.DiGraph, 
    scores: Dict[str, CriticalityScore],
    top_n: int = 10
) -> Dict[str, Any]:
    """Simulate failures of top critical components"""
    
    # Sort by criticality
    sorted_scores = sorted(scores.values(), key=lambda s: s.composite_score, reverse=True)
    
    results = {
        'simulations': [],
        'summary': {}
    }
    
    for score in sorted_scores[:top_n]:
        node = score.component
        
        # Simulate removal
        G_sim = G.copy()
        G_sim.remove_node(node)
        
        # Measure impact
        original_components = nx.number_weakly_connected_components(G)
        new_components = nx.number_weakly_connected_components(G_sim)
        
        # Calculate message flow disruption
        original_paths = sum(1 for _ in nx.all_pairs_shortest_path_length(G))
        try:
            new_paths = sum(1 for _ in nx.all_pairs_shortest_path_length(G_sim))
        except:
            new_paths = 0
        
        results['simulations'].append({
            'component': node,
            'type': score.component_type,
            'criticality_score': round(score.composite_score, 4),
            'impact': {
                'components_before': original_components,
                'components_after': new_components,
                'fragmentation': new_components - original_components,
                'reachability_loss_pct': round(score.reachability_loss_pct, 2),
                'nodes_affected': score.components_affected
            }
        })
    
    # Summary statistics
    if results['simulations']:
        avg_loss = sum(s['impact']['reachability_loss_pct'] for s in results['simulations']) / len(results['simulations'])
        max_loss = max(s['impact']['reachability_loss_pct'] for s in results['simulations'])
        
        results['summary'] = {
            'simulations_run': len(results['simulations']),
            'avg_reachability_loss_pct': round(avg_loss, 2),
            'max_reachability_loss_pct': round(max_loss, 2),
            'total_fragmentation_events': sum(1 for s in results['simulations'] if s['impact']['fragmentation'] > 0)
        }
    
    return results


# =============================================================================
# Recommendations
# =============================================================================

def generate_recommendations(
    scores: Dict[str, CriticalityScore],
    antipatterns: List[AntiPattern],
    structural: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generate actionable recommendations"""
    recommendations = []
    
    # From criticality scores
    critical_components = [s for s in scores.values() 
                         if s.criticality_level == CriticalityLevel.CRITICAL]
    
    for score in critical_components[:5]:
        if score.is_articulation_point:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Redundancy',
                'component': score.component,
                'issue': f'{score.component_type} is a single point of failure',
                'recommendation': f'Add redundancy for {score.component}',
                'risk_reduction': 'High'
            })
        
        if score.impact_score > 0.5:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Fault Tolerance',
                'component': score.component,
                'issue': f'High impact score ({score.impact_score:.2f})',
                'recommendation': f'Implement fallback mechanism for {score.component}',
                'risk_reduction': 'Medium-High'
            })
    
    # From anti-patterns
    for ap in antipatterns:
        if ap.severity in ['CRITICAL', 'HIGH']:
            recommendations.append({
                'priority': ap.severity,
                'category': f'Anti-Pattern: {ap.pattern_type}',
                'component': ', '.join(ap.components[:3]),
                'issue': ap.description,
                'recommendation': ap.recommendation,
                'risk_reduction': 'Medium'
            })
    
    # From structural analysis
    if structural.get('bridges', {}).get('count', 0) > 0:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Network Topology',
            'component': 'Network',
            'issue': f'{structural["bridges"]["count"]} bridge edges detected',
            'recommendation': 'Add redundant network paths',
            'risk_reduction': 'High'
        })
    
    # Sort by priority
    priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    recommendations.sort(key=lambda r: priority_order.get(r['priority'], 4))
    
    return recommendations


# =============================================================================
# Output Functions
# =============================================================================

def print_analysis_results(results: AnalysisResults, verbose: bool = False):
    """Print formatted analysis results"""
    
    # Graph Summary
    print_section("Graph Summary")
    gs = results.graph_summary
    print_kv("Nodes", gs.get('nodes', 'N/A'))
    print_kv("Edges", gs.get('edges', 'N/A'))
    print_kv("Density", f"{gs.get('density', 0):.4f}")
    print_kv("Connected", gs.get('is_connected', 'N/A'))
    
    if 'node_types' in gs:
        print(f"\n  {Colors.DIM}Node Types:{Colors.ENDC}")
        for ntype, count in gs['node_types'].items():
            print_kv(ntype, count, indent=4)
    
    # Criticality Summary
    print_section("Criticality Analysis")
    crit_summary = results.to_dict()['criticality_analysis']['summary']
    print_kv("Total Components", crit_summary.get('total_components', 0))
    print_kv("Avg Score", f"{crit_summary.get('avg_score', 0):.4f}")
    print_kv("Max Score", f"{crit_summary.get('max_score', 0):.4f}")
    print(f"\n  {Colors.DIM}By Level:{Colors.ENDC}")
    print(f"    {Colors.FAIL}CRITICAL:{Colors.ENDC} {crit_summary.get('critical_count', 0)}")
    print(f"    {Colors.WARNING}HIGH:{Colors.ENDC} {crit_summary.get('high_count', 0)}")
    print(f"    {Colors.CYAN}MEDIUM:{Colors.ENDC} {crit_summary.get('medium_count', 0)}")
    print(f"    {Colors.GREEN}LOW:{Colors.ENDC} {crit_summary.get('low_count', 0)}")
    print(f"    {Colors.DIM}MINIMAL:{Colors.ENDC} {crit_summary.get('minimal_count', 0)}")
    print_kv("Articulation Points", crit_summary.get('articulation_points', 0))
    
    # Top Critical Components
    print(f"\n  {Colors.BOLD}Top Critical Components:{Colors.ENDC}")
    sorted_scores = sorted(results.criticality_scores.values(), 
                          key=lambda s: s.composite_score, reverse=True)
    for score in sorted_scores[:10]:
        level_color = {
            CriticalityLevel.CRITICAL: Colors.FAIL,
            CriticalityLevel.HIGH: Colors.WARNING,
            CriticalityLevel.MEDIUM: Colors.CYAN,
            CriticalityLevel.LOW: Colors.GREEN,
            CriticalityLevel.MINIMAL: Colors.DIM
        }.get(score.criticality_level, '')
        
        ap_marker = " [AP]" if score.is_articulation_point else ""
        print(f"    {level_color}{score.criticality_level.value:8s}{Colors.ENDC} "
              f"{score.component:30s} ({score.component_type:12s}) "
              f"Score: {score.composite_score:.4f}{ap_marker}")
    
    # Anti-Patterns
    if results.antipatterns:
        print_section(f"Anti-Patterns Detected ({len(results.antipatterns)})")
        for ap in results.antipatterns[:10]:
            severity_color = Colors.FAIL if ap.severity == 'CRITICAL' else Colors.WARNING
            print(f"  {severity_color}[{ap.severity}]{Colors.ENDC} {ap.pattern_type}")
            print(f"    {Colors.DIM}{ap.description}{Colors.ENDC}")
    
    # Simulation Results
    if results.simulation_results.get('simulations'):
        print_section("Failure Simulation")
        summary = results.simulation_results['summary']
        print_kv("Simulations Run", summary.get('simulations_run', 0))
        print_kv("Avg Reachability Loss", f"{summary.get('avg_reachability_loss_pct', 0):.2f}%")
        print_kv("Max Reachability Loss", f"{summary.get('max_reachability_loss_pct', 0):.2f}%")
    
    # Recommendations
    if results.recommendations:
        print_section(f"Recommendations ({len(results.recommendations)})")
        for rec in results.recommendations[:5]:
            priority_color = Colors.FAIL if rec['priority'] == 'CRITICAL' else Colors.WARNING
            print(f"  {priority_color}[{rec['priority']}]{Colors.ENDC} {rec['category']}")
            print(f"    {Colors.DIM}{rec['recommendation']}{Colors.ENDC}")
    
    # Execution Time
    print_section("Execution Time")
    for phase, duration in results.execution_time.items():
        print_kv(phase, f"{duration:.2f}s")
    total = sum(results.execution_time.values())
    print_kv("Total", f"{total:.2f}s", indent=2)


def export_to_json(results: AnalysisResults, filepath: str):
    """Export results to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)


def export_to_csv(results: AnalysisResults, filepath: str):
    """Export criticality scores to CSV"""
    import csv
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'component', 'type', 'composite_score', 'level',
            'betweenness_norm', 'is_articulation_point', 'impact_score',
            'degree_centrality', 'closeness_centrality', 'pagerank',
            'components_affected', 'reachability_loss_pct'
        ])
        
        for score in results.criticality_scores.values():
            writer.writerow([
                score.component, score.component_type,
                round(score.composite_score, 4), score.criticality_level.value,
                round(score.betweenness_centrality_norm, 4), score.is_articulation_point,
                round(score.impact_score, 4), round(score.degree_centrality, 4),
                round(score.closeness_centrality, 4), round(score.pagerank, 4),
                score.components_affected, round(score.reachability_loss_pct, 2)
            ])


# =============================================================================
# Main CLI
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Comprehensive pub-sub system graph analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input system.json
  %(prog)s --input system.json --detect-antipatterns --simulate
  %(prog)s --neo4j --uri bolt://localhost:7687 --user neo4j --password pass
        """
    )
    
    # Input options
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--input', '-i', help='Input JSON file')
    input_group.add_argument('--neo4j', action='store_true', help='Load from Neo4j')
    input_group.add_argument('--uri', default='bolt://localhost:7687', help='Neo4j URI')
    input_group.add_argument('--user', default='neo4j', help='Neo4j username')
    input_group.add_argument('--password', default='password', help='Neo4j password')
    input_group.add_argument('--database', default='neo4j', help='Neo4j database')
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis')
    analysis_group.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                               help=f'Betweenness weight (default: {DEFAULT_ALPHA})')
    analysis_group.add_argument('--beta', type=float, default=DEFAULT_BETA,
                               help=f'Articulation point weight (default: {DEFAULT_BETA})')
    analysis_group.add_argument('--gamma', type=float, default=DEFAULT_GAMMA,
                               help=f'Impact score weight (default: {DEFAULT_GAMMA})')
    analysis_group.add_argument('--skip-impact', action='store_true',
                               help='Skip impact score calculation (faster)')
    analysis_group.add_argument('--detect-antipatterns', '-d', action='store_true',
                               help='Detect anti-patterns')
    analysis_group.add_argument('--simulate', '-s', action='store_true',
                               help='Run failure simulations')
    analysis_group.add_argument('--top-n', type=int, default=10,
                               help='Number of components to simulate (default: 10)')
    
    # Export options
    export_group = parser.add_argument_group('Export')
    export_group.add_argument('--export-json', metavar='FILE', help='Export to JSON')
    export_group.add_argument('--export-csv', metavar='FILE', help='Export scores to CSV')
    
    # Verbosity
    verbosity_group = parser.add_argument_group('Verbosity')
    verbosity_group.add_argument('--verbose', '-v', action='store_true')
    verbosity_group.add_argument('--quiet', '-q', action='store_true')
    verbosity_group.add_argument('--no-color', action='store_true')
    
    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()
    
    log_level = logging.DEBUG if args.verbose else (logging.ERROR if args.quiet else logging.INFO)
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    # Validate input
    if not args.input and not args.neo4j:
        print_error("Input required. Use --input FILE or --neo4j")
        return 1
    
    if not args.quiet:
        print_header("GRAPH ANALYSIS")
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Weights: α={args.alpha}, β={args.beta}, γ={args.gamma}")
    
    execution_time = {}
    
    # Load graph
    if not args.quiet:
        print_info("Loading graph...")
    
    start = time.time()
    try:
        if args.neo4j:
            G = load_graph_from_neo4j(args.uri, args.user, args.password, args.database)
            data = {}
        else:
            G, data = load_graph_from_json(args.input)
        execution_time['load_graph'] = time.time() - start
        print_success(f"Loaded graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        print_error(f"Failed to load graph: {e}")
        return 1
    
    # Structural analysis
    if not args.quiet:
        print_info("Analyzing structure...")
    start = time.time()
    structural = analyze_structure(G)
    execution_time['structural_analysis'] = time.time() - start
    
    # Criticality scoring
    if not args.quiet:
        print_info("Calculating criticality scores...")
    start = time.time()
    scores = calculate_criticality_scores(
        G, args.alpha, args.beta, args.gamma,
        calculate_impact=not args.skip_impact,
        logger=logger if args.verbose else None
    )
    execution_time['criticality_scoring'] = time.time() - start
    
    # Anti-pattern detection
    antipatterns = []
    if args.detect_antipatterns:
        if not args.quiet:
            print_info("Detecting anti-patterns...")
        start = time.time()
        antipatterns = detect_antipatterns(G, data)
        execution_time['antipattern_detection'] = time.time() - start
    
    # Failure simulation
    simulation_results = {}
    if args.simulate:
        if not args.quiet:
            print_info("Running failure simulations...")
        start = time.time()
        simulation_results = simulate_failures(G, scores, args.top_n)
        execution_time['failure_simulation'] = time.time() - start
    
    # Generate recommendations
    recommendations = generate_recommendations(scores, antipatterns, structural)
    
    # Build results
    results = AnalysisResults(
        timestamp=datetime.now().isoformat(),
        graph_summary=structural,
        criticality_scores=scores,
        structural_analysis=structural,
        antipatterns=antipatterns,
        simulation_results=simulation_results,
        recommendations=recommendations,
        execution_time=execution_time
    )
    
    # Print results
    if not args.quiet:
        print_analysis_results(results, args.verbose)
    
    # Export
    if args.export_json:
        export_to_json(results, args.export_json)
        print_success(f"Results exported to {args.export_json}")
    
    if args.export_csv:
        export_to_csv(results, args.export_csv)
        print_success(f"Scores exported to {args.export_csv}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())