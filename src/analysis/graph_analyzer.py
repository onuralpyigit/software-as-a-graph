#!/usr/bin/env python3
"""
Graph Analyzer - DEPENDS_ON Relationship Analysis
==================================================

Analyzes pub-sub systems by deriving DEPENDS_ON relationships directly
from base relationships (PUBLISHES_TO, SUBSCRIBES_TO, RUNS_ON, ROUTES)
without storing derived relationships in JSON input files.

DEPENDS_ON Types:
  - APP_TO_APP: Subscriber depends on publisher via shared topic
  - APP_TO_BROKER: Application depends on broker for message routing
  - NODE_TO_NODE: Infrastructure dependencies via application relationships
  - NODE_TO_BROKER: Node depends on broker running on it

Author: Software-as-a-Graph Research Project
"""

import logging
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import json
from pathlib import Path

try:
    import networkx as nx
except ImportError:
    raise ImportError("NetworkX is required: pip install networkx")


# ============================================================================
# Enums and Data Classes
# ============================================================================

class DependencyType(Enum):
    """Types of DEPENDS_ON relationships"""
    APP_TO_APP = "app_to_app"
    APP_TO_BROKER = "app_to_broker"
    NODE_TO_NODE = "node_to_node"
    NODE_TO_BROKER = "node_to_broker"


class CriticalityLevel(Enum):
    """Criticality levels for components"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class DependsOnEdge:
    """Represents a DEPENDS_ON relationship"""
    source: str
    target: str
    dep_type: DependencyType
    weight: float = 1.0
    via_topics: List[str] = field(default_factory=list)
    via_apps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'target': self.target,
            'type': self.dep_type.value,
            'weight': self.weight,
            'via_topics': self.via_topics,
            'via_apps': self.via_apps
        }


@dataclass
class CriticalityScore:
    """Criticality score for a component"""
    node_id: str
    node_type: str
    betweenness: float
    is_articulation_point: bool
    impact_score: float
    composite_score: float
    level: CriticalityLevel
    reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'type': self.node_type,
            'betweenness': round(self.betweenness, 4),
            'is_articulation_point': self.is_articulation_point,
            'impact_score': round(self.impact_score, 4),
            'composite_score': round(self.composite_score, 4),
            'level': self.level.value,
            'reasons': self.reasons
        }


@dataclass
class AnalysisResult:
    """Complete analysis results"""
    graph_summary: Dict[str, Any]
    depends_on_edges: List[DependsOnEdge]
    criticality_scores: List[CriticalityScore]
    structural_analysis: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'graph_summary': self.graph_summary,
            'depends_on': {
                'total': len(self.depends_on_edges),
                'by_type': self._count_by_type(),
                'edges': [e.to_dict() for e in self.depends_on_edges]
            },
            'criticality': {
                'scores': [s.to_dict() for s in self.criticality_scores],
                'by_level': self._count_by_level()
            },
            'structural': self.structural_analysis,
            'recommendations': self.recommendations
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for edge in self.depends_on_edges:
            counts[edge.dep_type.value] += 1
        return dict(counts)
    
    def _count_by_level(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for score in self.criticality_scores:
            counts[score.level.value] += 1
        return dict(counts)


# ============================================================================
# Graph Analyzer
# ============================================================================

class GraphAnalyzer:
    """
    Analyzes pub-sub system graphs by deriving DEPENDS_ON relationships.
    
    This analyzer takes raw pub-sub data and derives all dependency
    relationships during analysis, rather than storing them in input files.
    """
    
    def __init__(self, 
                 alpha: float = 0.4,
                 beta: float = 0.3,
                 gamma: float = 0.3):
        """
        Initialize analyzer with criticality scoring weights.
        
        Args:
            alpha: Weight for betweenness centrality (default: 0.4)
            beta: Weight for articulation point indicator (default: 0.3)
            gamma: Weight for impact score (default: 0.3)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.logger = logging.getLogger('graph_analyzer')
        
        # Internal data structures
        self.raw_data: Dict[str, Any] = {}
        self.G: Optional[nx.DiGraph] = None
        self.depends_on_edges: List[DependsOnEdge] = []
        
        # Index structures for fast lookup
        self._app_to_node: Dict[str, str] = {}
        self._broker_to_node: Dict[str, str] = {}
        self._topic_to_broker: Dict[str, str] = {}
        self._app_publishes: Dict[str, Set[str]] = defaultdict(set)
        self._app_subscribes: Dict[str, Set[str]] = defaultdict(set)
        self._topic_publishers: Dict[str, Set[str]] = defaultdict(set)
        self._topic_subscribers: Dict[str, Set[str]] = defaultdict(set)
    
    def load_from_file(self, filepath: str) -> 'GraphAnalyzer':
        """Load pub-sub data from JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {filepath}")
        
        with open(path) as f:
            self.raw_data = json.load(f)
        
        self._build_indexes()
        return self
    
    def load_from_dict(self, data: Dict[str, Any]) -> 'GraphAnalyzer':
        """Load pub-sub data from dictionary."""
        self.raw_data = data
        self._build_indexes()
        return self
    
    def load_from_neo4j(self, 
                        uri: str = "bolt://localhost:7687",
                        user: str = "neo4j",
                        password: str = "password",
                        database: str = "neo4j") -> 'GraphAnalyzer':
        """
        Load pub-sub data from Neo4j database.
        
        Args:
            uri: Neo4j bolt URI (default: bolt://localhost:7687)
            user: Database username (default: neo4j)
            password: Database password (default: password)
            database: Database name (default: neo4j)
        
        Returns:
            Self for method chaining
        
        Raises:
            ImportError: If neo4j driver is not installed
        """
        from .neo4j_loader import Neo4jGraphLoader
        
        self.logger.info(f"Loading data from Neo4j at {uri}...")
        
        with Neo4jGraphLoader(uri, user, password, database) as loader:
            self.raw_data = loader.load()
        
        self._build_indexes()
        return self
    
    def _build_indexes(self):
        """Build lookup indexes from raw data."""
        self._app_to_node.clear()
        self._broker_to_node.clear()
        self._topic_to_broker.clear()
        self._app_publishes.clear()
        self._app_subscribes.clear()
        self._topic_publishers.clear()
        self._topic_subscribers.clear()
        
        # Index applications to nodes
        for app in self.raw_data.get('applications', []):
            if 'node' in app:
                self._app_to_node[app['id']] = app['node']
        
        # Index brokers to nodes
        for broker in self.raw_data.get('brokers', []):
            if 'node' in broker:
                self._broker_to_node[broker['id']] = broker['node']
        
        # Index topics to brokers
        for topic in self.raw_data.get('topics', []):
            if 'broker' in topic:
                self._topic_to_broker[topic['id']] = topic['broker']
        
        # Index publish/subscribe relationships
        relationships = self.raw_data.get('relationships', {})
        
        for rel in relationships.get('publishes_to', []):
            app_id = rel.get('from', rel.get('source'))
            topic_id = rel.get('to', rel.get('target'))
            if app_id and topic_id:
                self._app_publishes[app_id].add(topic_id)
                self._topic_publishers[topic_id].add(app_id)
        
        for rel in relationships.get('subscribes_to', []):
            app_id = rel.get('from', rel.get('source'))
            topic_id = rel.get('to', rel.get('target'))
            if app_id and topic_id:
                self._app_subscribes[app_id].add(topic_id)
                self._topic_subscribers[topic_id].add(app_id)
    
    def derive_depends_on(self) -> List[DependsOnEdge]:
        """
        Derive all DEPENDS_ON relationships from base relationships.
        
        This is the core method that creates the dependency graph without
        storing relationships in the input JSON.
        
        Returns:
            List of DependsOnEdge objects
        """
        self.depends_on_edges = []
        
        # 1. APP_TO_APP: subscriber depends on publisher via shared topic
        self._derive_app_to_app()
        
        # 2. APP_TO_BROKER: application depends on broker for its topics
        self._derive_app_to_broker()
        
        # 3. NODE_TO_NODE: infrastructure dependencies via application relationships
        self._derive_node_to_node()
        
        # 4. NODE_TO_BROKER: node depends on broker running on it (if apps need that broker)
        self._derive_node_to_broker()
        
        self.logger.info(f"Derived {len(self.depends_on_edges)} DEPENDS_ON relationships")
        return self.depends_on_edges
    
    def _derive_app_to_app(self):
        """
        Derive APP_TO_APP dependencies.
        
        A subscriber depends on publishers of the topics it subscribes to.
        Weight increases with number of shared topics.
        """
        for topic_id, subscribers in self._topic_subscribers.items():
            publishers = self._topic_publishers.get(topic_id, set())
            
            for subscriber in subscribers:
                for publisher in publishers:
                    if subscriber != publisher:
                        # Check if edge already exists
                        existing = self._find_edge(subscriber, publisher, DependencyType.APP_TO_APP)
                        if existing:
                            existing.via_topics.append(topic_id)
                            existing.weight = 1.0 + (len(existing.via_topics) - 1) * 0.2
                        else:
                            self.depends_on_edges.append(DependsOnEdge(
                                source=subscriber,
                                target=publisher,
                                dep_type=DependencyType.APP_TO_APP,
                                weight=1.0,
                                via_topics=[topic_id]
                            ))
    
    def _derive_app_to_broker(self):
        """
        Derive APP_TO_BROKER dependencies.
        
        An application depends on brokers that route topics it uses.
        """
        for app_id in set(self._app_publishes.keys()) | set(self._app_subscribes.keys()):
            # Get all topics this app uses
            topics = self._app_publishes[app_id] | self._app_subscribes[app_id]
            
            # Find brokers for these topics
            brokers_used: Dict[str, List[str]] = defaultdict(list)
            for topic_id in topics:
                broker_id = self._topic_to_broker.get(topic_id)
                if broker_id:
                    brokers_used[broker_id].append(topic_id)
            
            # Create dependencies
            for broker_id, broker_topics in brokers_used.items():
                self.depends_on_edges.append(DependsOnEdge(
                    source=app_id,
                    target=broker_id,
                    dep_type=DependencyType.APP_TO_BROKER,
                    weight=1.0 + (len(broker_topics) - 1) * 0.1,
                    via_topics=broker_topics
                ))
    
    def _derive_node_to_node(self):
        """
        Derive NODE_TO_NODE dependencies.
        
        If an app on node A depends on an app on node B, then node A
        depends on node B (transitively through app dependencies).
        """
        # First, group app-to-app dependencies by source/target nodes
        node_deps: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        
        for edge in self.depends_on_edges:
            if edge.dep_type == DependencyType.APP_TO_APP:
                source_node = self._app_to_node.get(edge.source)
                target_node = self._app_to_node.get(edge.target)
                
                if source_node and target_node and source_node != target_node:
                    node_deps[(source_node, target_node)].append(edge.source)
        
        # Create node-to-node dependencies
        for (source_node, target_node), apps in node_deps.items():
            self.depends_on_edges.append(DependsOnEdge(
                source=source_node,
                target=target_node,
                dep_type=DependencyType.NODE_TO_NODE,
                weight=1.0 + (len(set(apps)) - 1) * 0.15,
                via_apps=list(set(apps))
            ))
    
    def _derive_node_to_broker(self):
        """
        Derive NODE_TO_BROKER dependencies.
        
        A node depends on a broker if apps on that node use topics
        routed by that broker (and the broker is on a different node).
        """
        node_broker_deps: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        
        for edge in self.depends_on_edges:
            if edge.dep_type == DependencyType.APP_TO_BROKER:
                app_node = self._app_to_node.get(edge.source)
                broker_node = self._broker_to_node.get(edge.target)
                
                if app_node and broker_node and app_node != broker_node:
                    node_broker_deps[(app_node, edge.target)].append(edge.source)
        
        for (node_id, broker_id), apps in node_broker_deps.items():
            self.depends_on_edges.append(DependsOnEdge(
                source=node_id,
                target=broker_id,
                dep_type=DependencyType.NODE_TO_BROKER,
                weight=1.0 + (len(set(apps)) - 1) * 0.1,
                via_apps=list(set(apps))
            ))
    
    def _find_edge(self, source: str, target: str, dep_type: DependencyType) -> Optional[DependsOnEdge]:
        """Find existing edge between source and target."""
        for edge in self.depends_on_edges:
            if edge.source == source and edge.target == target and edge.dep_type == dep_type:
                return edge
        return None
    
    def build_dependency_graph(self) -> nx.DiGraph:
        """
        Build NetworkX graph from derived dependencies.
        
        Returns:
            NetworkX DiGraph with nodes and DEPENDS_ON edges
        """
        if not self.depends_on_edges:
            self.derive_depends_on()
        
        self.G = nx.DiGraph()
        
        # Add all component nodes (filter out 'type' from data to avoid conflict)
        for node in self.raw_data.get('nodes', []):
            attrs = {k: v for k, v in node.items() if k not in ('id', 'type')}
            self.G.add_node(node['id'], type='Node', **attrs)
        
        for broker in self.raw_data.get('brokers', []):
            attrs = {k: v for k, v in broker.items() if k not in ('id', 'type')}
            self.G.add_node(broker['id'], type='Broker', **attrs)
        
        for app in self.raw_data.get('applications', []):
            attrs = {k: v for k, v in app.items() if k not in ('id', 'type')}
            self.G.add_node(app['id'], type='Application', **attrs)
        
        for topic in self.raw_data.get('topics', []):
            attrs = {k: v for k, v in topic.items() if k not in ('id', 'type')}
            self.G.add_node(topic['id'], type='Topic', **attrs)
        
        # Add DEPENDS_ON edges
        for edge in self.depends_on_edges:
            self.G.add_edge(
                edge.source,
                edge.target,
                type='DEPENDS_ON',
                dependency_type=edge.dep_type.value,
                weight=edge.weight,
                via_topics=edge.via_topics,
                via_apps=edge.via_apps
            )
        
        return self.G
    
    def analyze(self) -> AnalysisResult:
        """
        Run complete analysis.
        
        Returns:
            AnalysisResult with all analysis data
        """
        # Derive dependencies and build graph
        self.derive_depends_on()
        self.build_dependency_graph()
        
        # Graph summary
        graph_summary = self._compute_graph_summary()
        
        # Structural analysis
        structural = self._analyze_structure()
        
        # Criticality scoring
        criticality_scores = self._compute_criticality_scores(structural)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(criticality_scores, structural)
        
        return AnalysisResult(
            graph_summary=graph_summary,
            depends_on_edges=self.depends_on_edges,
            criticality_scores=criticality_scores,
            structural_analysis=structural,
            recommendations=recommendations
        )
    
    def _compute_graph_summary(self) -> Dict[str, Any]:
        """Compute graph summary statistics."""
        node_types = defaultdict(int)
        for _, data in self.G.nodes(data=True):
            node_types[data.get('type', 'Unknown')] += 1
        
        edge_types = defaultdict(int)
        for _, _, data in self.G.edges(data=True):
            edge_types[data.get('dependency_type', 'unknown')] += 1
        
        return {
            'total_nodes': self.G.number_of_nodes(),
            'total_edges': self.G.number_of_edges(),
            'nodes_by_type': dict(node_types),
            'edges_by_type': dict(edge_types),
            'density': round(nx.density(self.G), 4),
            'is_connected': nx.is_weakly_connected(self.G) if self.G.number_of_nodes() > 0 else True
        }
    
    def _analyze_structure(self) -> Dict[str, Any]:
        """Analyze structural properties."""
        G_undirected = self.G.to_undirected()
        
        # Articulation points
        articulation_points = list(nx.articulation_points(G_undirected))
        
        # Bridges
        bridges = list(nx.bridges(G_undirected))
        
        # Connected components
        if self.G.number_of_nodes() > 0:
            wccs = list(nx.weakly_connected_components(self.G))
            sccs = list(nx.strongly_connected_components(self.G))
        else:
            wccs, sccs = [], []
        
        # Betweenness centrality
        betweenness = nx.betweenness_centrality(self.G, normalized=True)
        
        # Find cycles (potential circular dependencies)
        try:
            cycles = list(nx.simple_cycles(self.G))
            cycles = [c for c in cycles if len(c) <= 10][:20]  # Limit
        except:
            cycles = []
        
        return {
            'articulation_points': articulation_points,
            'articulation_point_count': len(articulation_points),
            'bridges': [(b[0], b[1]) for b in bridges],
            'bridge_count': len(bridges),
            'weakly_connected_components': len(wccs),
            'strongly_connected_components': len(sccs),
            'has_cycles': len(cycles) > 0,
            'cycles': cycles[:10],
            'betweenness_centrality': betweenness
        }
    
    def _compute_criticality_scores(self, structural: Dict[str, Any]) -> List[CriticalityScore]:
        """
        Compute criticality scores using the composite formula:
        C_score = α·BC + β·AP + γ·I
        
        Where:
        - BC: Normalized betweenness centrality
        - AP: Articulation point indicator (1 if AP, 0 otherwise)
        - I: Impact score (based on reachability)
        """
        scores = []
        betweenness = structural['betweenness_centrality']
        articulation_points = set(structural['articulation_points'])
        
        # Normalize betweenness
        max_bc = max(betweenness.values()) if betweenness else 1.0
        if max_bc == 0:
            max_bc = 1.0
        
        for node_id in self.G.nodes():
            node_data = self.G.nodes[node_id]
            node_type = node_data.get('type', 'Unknown')
            
            # Betweenness (normalized)
            bc = betweenness.get(node_id, 0) / max_bc
            
            # Articulation point indicator
            ap = 1.0 if node_id in articulation_points else 0.0
            
            # Impact score (simplified: based on out-degree and reachability)
            impact = self._compute_impact_score(node_id)
            
            # Composite score
            composite = self.alpha * bc + self.beta * ap + self.gamma * impact
            
            # Determine level
            level = self._determine_level(composite, ap)
            
            # Reasons
            reasons = self._determine_reasons(bc, ap, impact, node_type)
            
            scores.append(CriticalityScore(
                node_id=node_id,
                node_type=node_type,
                betweenness=bc,
                is_articulation_point=ap == 1.0,
                impact_score=impact,
                composite_score=composite,
                level=level,
                reasons=reasons
            ))
        
        # Sort by composite score descending
        scores.sort(key=lambda x: x.composite_score, reverse=True)
        return scores
    
    def _compute_impact_score(self, node_id: str) -> float:
        """
        Compute impact score based on reachability loss.
        
        Impact = (nodes reachable from this node) / (total nodes - 1)
        """
        if self.G.number_of_nodes() <= 1:
            return 0.0
        
        reachable = nx.descendants(self.G, node_id)
        return len(reachable) / (self.G.number_of_nodes() - 1)
    
    def _determine_level(self, composite: float, ap: float) -> CriticalityLevel:
        """Determine criticality level from composite score."""
        if ap == 1.0 or composite >= 0.7:
            return CriticalityLevel.CRITICAL
        elif composite >= 0.5:
            return CriticalityLevel.HIGH
        elif composite >= 0.3:
            return CriticalityLevel.MEDIUM
        else:
            return CriticalityLevel.LOW
    
    def _determine_reasons(self, bc: float, ap: float, impact: float, 
                           node_type: str) -> List[str]:
        """Generate human-readable reasons for criticality."""
        reasons = []
        
        if ap == 1.0:
            reasons.append("Single point of failure (articulation point)")
        
        if bc >= 0.5:
            reasons.append(f"High betweenness centrality ({bc:.2f}) - routing bottleneck")
        elif bc >= 0.3:
            reasons.append(f"Moderate betweenness ({bc:.2f})")
        
        if impact >= 0.5:
            reasons.append(f"High impact ({impact:.2f}) - affects many components")
        elif impact >= 0.3:
            reasons.append(f"Moderate impact ({impact:.2f})")
        
        if node_type == 'Broker':
            reasons.append("Broker - critical infrastructure component")
        
        return reasons if reasons else ["Low criticality"]
    
    def _generate_recommendations(self, scores: List[CriticalityScore],
                                   structural: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Critical components
        critical = [s for s in scores if s.level == CriticalityLevel.CRITICAL]
        if critical:
            recommendations.append(
                f"Found {len(critical)} CRITICAL components. "
                "Consider adding redundancy for these components."
            )
        
        # Articulation points
        ap_count = structural['articulation_point_count']
        if ap_count > 0:
            recommendations.append(
                f"Found {ap_count} articulation points (SPOFs). "
                "These nodes are single points of failure."
            )
        
        # Bridges
        bridge_count = structural['bridge_count']
        if bridge_count > 0:
            recommendations.append(
                f"Found {bridge_count} bridge edges. "
                "Consider adding redundant paths for these connections."
            )
        
        # Cycles
        if structural['has_cycles']:
            recommendations.append(
                "Circular dependencies detected. "
                "Review for potential infinite loops or deadlocks."
            )
        
        # Components
        wcc = structural['weakly_connected_components']
        if wcc > 1:
            recommendations.append(
                f"Graph has {wcc} disconnected components. "
                "Some parts of the system are isolated."
            )
        
        return recommendations


# ============================================================================
# Convenience Functions
# ============================================================================

def analyze_pubsub_system(filepath: str, 
                          alpha: float = 0.4,
                          beta: float = 0.3,
                          gamma: float = 0.3) -> AnalysisResult:
    """
    Convenience function to analyze a pub-sub system.
    
    Args:
        filepath: Path to JSON file with pub-sub data
        alpha: Weight for betweenness centrality
        beta: Weight for articulation point indicator
        gamma: Weight for impact score
    
    Returns:
        AnalysisResult with complete analysis
    """
    analyzer = GraphAnalyzer(alpha=alpha, beta=beta, gamma=gamma)
    analyzer.load_from_file(filepath)
    return analyzer.analyze()


def derive_dependencies(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convenience function to derive DEPENDS_ON relationships.
    
    Args:
        data: Dictionary with pub-sub system data
    
    Returns:
        List of DEPENDS_ON relationships as dictionaries
    """
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(data)
    edges = analyzer.derive_depends_on()
    return [e.to_dict() for e in edges]