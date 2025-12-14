"""
Graph Builder - Simplified Version 3.0

Builds GraphModel instances from various data sources:
- JSON files
- YAML files
- Python dictionaries
- CSV files (vertices.csv + edges.csv)
- Neo4j databases
- NetworkX graphs

Features:
- Comprehensive validation with detailed error reporting
- JSON Schema validation
- Graph merging and filtering
- Graph comparison and diff
- Cypher query generation (single and batch)
- Statistics and summary reports
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Callable, Union
from collections import defaultdict
from datetime import datetime, timezone

from .graph_model import (
    GraphModel, Application, Broker, Topic, Node, Edge,
    QoSPolicy, EdgeType, VertexType
)

# =============================================================================
# JSON Schema
# =============================================================================

GRAPH_SCHEMA = {
    "type": "object",
    "required": ["applications", "brokers", "topics", "nodes", "edges"],
    "properties": {
        "metadata": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "generated_at": {"type": "string"},
                "generator_version": {"type": "string"},
                "scale": {"type": "string"},
                "scenario": {"type": "string"},
                "seed": {"type": ["integer", "null"]}
            }
        },
        "applications": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "role": {"type": "string", "enum": ["pub", "sub", "pubsub"]}
                }
            }
        },
        "brokers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"}
                }
            }
        },
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "size": {"type": "integer"},
                    "qos": {
                        "type": "object",
                        "properties": {
                            "durability": {"type": "string"},
                            "reliability": {"type": "string"},
                            "transport_priority": {"type": "string"}
                        }
                    }
                }
            }
        },
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"}
                }
            }
        },
        "relationships": {
            "type": "object",
            "required": ["publishes_to", "subscribes_to", "routes", "runs_on", "connects_to"],
            "properties": {
                "publishes_to": {"type": "array"},
                "subscribes_to": {"type": "array"},
                "routes": {"type": "array"},
                "runs_on": {"type": "array"},
                "connects_to": {"type": "array"}
            }
        }
    }
}


# =============================================================================
# Validation Result
# =============================================================================

class ValidationResult:
    """Result of graph validation with detailed error and warning tracking"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        self.schema_errors: List[str] = []
    
    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0 and len(self.schema_errors) == 0
    
    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
    
    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
    
    def add_info(self, msg: str) -> None:
        self.info.append(msg)
    
    def add_schema_error(self, msg: str) -> None:
        self.schema_errors.append(msg)
    
    def merge(self, other: 'ValidationResult') -> None:
        """Merge another validation result into this one"""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        self.schema_errors.extend(other.schema_errors)
    
    def summary(self) -> str:
        """Get a summary of the validation result"""
        lines = [f"Valid: {self.is_valid}"]
        if self.schema_errors:
            lines.append(f"Schema Errors ({len(self.schema_errors)}):")
            for e in self.schema_errors[:5]:
                lines.append(f"  - {e}")
            if len(self.schema_errors) > 5:
                lines.append(f"  ... and {len(self.schema_errors) - 5} more")
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for e in self.errors[:5]:
                lines.append(f"  - {e}")
            if len(self.errors) > 5:
                lines.append(f"  ... and {len(self.errors) - 5} more")
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings[:5]:
                lines.append(f"  - {w}")
            if len(self.warnings) > 5:
                lines.append(f"  ... and {len(self.warnings) - 5} more")
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export"""
        return {
            'is_valid': self.is_valid,
            'schema_errors': self.schema_errors,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info
        }
    
    def __repr__(self) -> str:
        return f"ValidationResult(valid={self.is_valid}, errors={len(self.errors)}, warnings={len(self.warnings)})"


# =============================================================================
# Graph Diff Result
# =============================================================================

class GraphDiffResult:
    """Result of comparing two graphs"""
    
    def __init__(self):
        self.added_vertices: Dict[str, List[str]] = defaultdict(list)
        self.removed_vertices: Dict[str, List[str]] = defaultdict(list)
        self.modified_vertices: Dict[str, List[Tuple[str, Dict, Dict]]] = defaultdict(list)
        self.added_edges: List[Edge] = []
        self.removed_edges: List[Edge] = []
    
    @property
    def has_changes(self) -> bool:
        return (any(self.added_vertices.values()) or 
                any(self.removed_vertices.values()) or
                any(self.modified_vertices.values()) or
                self.added_edges or self.removed_edges)
    
    def summary(self) -> str:
        """Get a summary of changes"""
        lines = ["Graph Diff Summary", "=" * 40]
        
        # Vertex changes
        for vtype in ['applications', 'brokers', 'topics', 'nodes']:
            added = len(self.added_vertices.get(vtype, []))
            removed = len(self.removed_vertices.get(vtype, []))
            modified = len(self.modified_vertices.get(vtype, []))
            if added or removed or modified:
                lines.append(f"{vtype.title()}: +{added} -{removed} ~{modified}")
        
        # Edge changes
        if self.added_edges or self.removed_edges:
            lines.append(f"Edges: +{len(self.added_edges)} -{len(self.removed_edges)}")
        
        if not self.has_changes:
            lines.append("No changes detected")
        
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'has_changes': self.has_changes,
            'added_vertices': dict(self.added_vertices),
            'removed_vertices': dict(self.removed_vertices),
            'modified_vertices': {k: [(vid, old, new) for vid, old, new in v] 
                                  for k, v in self.modified_vertices.items()},
            'added_edges': [{'from': e.source, 'to': e.target, 'type': e.edge_type} for e in self.added_edges],
            'removed_edges': [{'from': e.source, 'to': e.target, 'type': e.edge_type} for e in self.removed_edges]
        }


# =============================================================================
# Graph Builder
# =============================================================================

class GraphBuilder:
    """
    Builds GraphModel instances from various data sources
    
    Features:
    - Build from JSON, dict, CSV, Neo4j, or NetworkX
    - Validation of edge references and graph structure
    - Graph merging and filtering
    - Cypher query generation
    - Comprehensive error reporting
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation: ValidationResult = ValidationResult()
    
    # -------------------------------------------------------------------------
    # Build from Dictionary
    # -------------------------------------------------------------------------
    
    def build_from_dict(self, data: Dict) -> GraphModel:
        """
        Build GraphModel from a Python dictionary
        
        Args:
            data: Dictionary with applications, brokers, topics, nodes, edges
        
        Returns:
            GraphModel instance
        """
        self.validation = ValidationResult()
        model = GraphModel()
        
        model.metadata = data.get('metadata', {})
        
        # Build vertices
        self._build_applications(model, data.get('applications', []))
        self._build_brokers(model, data.get('brokers', []))
        self._build_topics(model, data.get('topics', []))
        self._build_nodes(model, data.get('nodes', []))
        
        # Build edges with validation
        self._build_edges(model, data.get('edges', {}))
        
        # Log summary
        stats = model.get_statistics()
        self.logger.info(
            f"Built graph: {stats['num_applications']} apps, {stats['num_brokers']} brokers, "
            f"{stats['num_topics']} topics, {stats['num_nodes']} nodes, {stats['num_relationships']} relationships"
        )
        
        return model
    
    def _build_applications(self, model: GraphModel, apps_data: List[Dict]) -> None:
        """Build Application vertices from list of dicts"""
        for i, app_data in enumerate(apps_data):
            try:
                app = Application.from_dict(app_data)
                if not app.id:
                    self.validation.add_error(f"Application[{i}]: Missing 'id' field")
                    continue
                if app.role not in ('pub', 'sub', 'pubsub'):
                    self.validation.add_warning(
                        f"Application '{app.id}': Invalid role '{app.role}', defaulting to 'pubsub'"
                    )
                    app.role = 'pubsub'
                if app.id in model.applications:
                    self.validation.add_warning(f"Application '{app.id}': Duplicate ID, overwriting")
                model.add_application(app)
            except Exception as e:
                self.validation.add_error(f"Application[{i}]: {e}")
    
    def _build_brokers(self, model: GraphModel, brokers_data: List[Dict]) -> None:
        """Build Broker vertices from list of dicts"""
        for i, broker_data in enumerate(brokers_data):
            try:
                broker = Broker.from_dict(broker_data)
                if not broker.id:
                    self.validation.add_error(f"Broker[{i}]: Missing 'id' field")
                    continue
                if broker.id in model.brokers:
                    self.validation.add_warning(f"Broker '{broker.id}': Duplicate ID, overwriting")
                model.add_broker(broker)
            except Exception as e:
                self.validation.add_error(f"Broker[{i}]: {e}")
    
    def _build_topics(self, model: GraphModel, topics_data: List[Dict]) -> None:
        """Build Topic vertices from list of dicts"""
        for i, topic_data in enumerate(topics_data):
            try:
                topic = Topic.from_dict(topic_data)
                if not topic.id:
                    self.validation.add_error(f"Topic[{i}]: Missing 'id' field")
                    continue
                if topic.id in model.topics:
                    self.validation.add_warning(f"Topic '{topic.id}': Duplicate ID, overwriting")
                model.add_topic(topic)
            except Exception as e:
                self.validation.add_error(f"Topic[{i}]: {e}")
    
    def _build_nodes(self, model: GraphModel, nodes_data: List[Dict]) -> None:
        """Build Node vertices from list of dicts"""
        for i, node_data in enumerate(nodes_data):
            try:
                node = Node.from_dict(node_data)
                if not node.id:
                    self.validation.add_error(f"Node[{i}]: Missing 'id' field")
                    continue
                if node.id in model.nodes:
                    self.validation.add_warning(f"Node '{node.id}': Duplicate ID, overwriting")
                model.add_node(node)
            except Exception as e:
                self.validation.add_error(f"Node[{i}]: {e}")
    
    def _build_edges(self, model: GraphModel, edges_data: Dict) -> None:
        """Build all edge types with validation"""
        app_ids = set(model.applications.keys())
        broker_ids = set(model.brokers.keys())
        topic_ids = set(model.topics.keys())
        node_ids = set(model.nodes.keys())
        
        # Edge configuration: (json_key, edge_type, valid_sources, valid_targets, src_desc, tgt_desc)
        edge_configs = [
            ('publishes_to', EdgeType.PUBLISHES_TO.value, app_ids, topic_ids, 'application', 'topic'),
            ('subscribes_to', EdgeType.SUBSCRIBES_TO.value, app_ids, topic_ids, 'application', 'topic'),
            ('routes', EdgeType.ROUTES.value, broker_ids, topic_ids, 'broker', 'topic'),
            ('runs_on', EdgeType.RUNS_ON.value, app_ids | broker_ids, node_ids, 'component', 'node'),
            ('connects_to', EdgeType.CONNECTS_TO.value, node_ids, node_ids, 'node', 'node'),
        ]
        
        for edge_key, edge_type, valid_sources, valid_targets, src_desc, tgt_desc in edge_configs:
            edge_list = edges_data.get(edge_key, [])
            
            for i, edge_data in enumerate(edge_list):
                source = edge_data.get('from', '')
                target = edge_data.get('to', '')
                
                if not source or not target:
                    self.validation.add_warning(f"{edge_key}[{i}]: Missing 'from' or 'to' field")
                    continue
                
                if source not in valid_sources:
                    self.validation.add_error(f"{edge_key}[{i}]: Unknown {src_desc} '{source}'")
                    continue
                
                if target not in valid_targets:
                    self.validation.add_error(f"{edge_key}[{i}]: Unknown {tgt_desc} '{target}'")
                    continue
                
                edge = Edge(source=source, target=target, edge_type=edge_type)
                model.add_edge(edge)
    
    # -------------------------------------------------------------------------
    # Build from JSON
    # -------------------------------------------------------------------------
    
    def build_from_json(self, filepath: str) -> GraphModel:
        """
        Build GraphModel from a JSON file
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            GraphModel instance
        """
        self.logger.info(f"Building graph from JSON: {filepath}")
        
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return self.build_from_dict(data)
    
    # -------------------------------------------------------------------------
    # Build from YAML
    # -------------------------------------------------------------------------
    
    def build_from_yaml(self, filepath: str) -> GraphModel:
        """
        Build GraphModel from a YAML file
        
        Args:
            filepath: Path to YAML file
        
        Returns:
            GraphModel instance
        
        Raises:
            ImportError: If pyyaml is not installed
            FileNotFoundError: If file doesn't exist
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("pyyaml package required. Install with: pip install pyyaml")
        
        self.logger.info(f"Building graph from YAML: {filepath}")
        
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return self.build_from_dict(data)
    
    # -------------------------------------------------------------------------
    # Schema Validation
    # -------------------------------------------------------------------------
    
    def validate_schema(self, data: Dict) -> ValidationResult:
        """
        Validate data against v3.0 JSON schema
        
        Args:
            data: Dictionary to validate
        
        Returns:
            ValidationResult with schema errors if any
        """
        result = ValidationResult()
        
        # Check required top-level keys
        required_keys = ['applications', 'brokers', 'topics', 'nodes', 'relationships']
        for key in required_keys:
            if key not in data:
                result.add_schema_error(f"Missing required key: '{key}'")
        
        # Validate applications
        for i, app in enumerate(data.get('applications', [])):
            if not isinstance(app, dict):
                result.add_schema_error(f"applications[{i}]: Expected object, got {type(app).__name__}")
                continue
            if 'id' not in app:
                result.add_schema_error(f"applications[{i}]: Missing required 'id' field")
            if 'role' in app and app['role'] not in ('pub', 'sub', 'pubsub'):
                result.add_schema_error(f"applications[{i}]: Invalid role '{app['role']}' (expected: pub, sub, pubsub)")
        
        # Validate brokers
        for i, broker in enumerate(data.get('brokers', [])):
            if not isinstance(broker, dict):
                result.add_schema_error(f"brokers[{i}]: Expected object, got {type(broker).__name__}")
                continue
            if 'id' not in broker:
                result.add_schema_error(f"brokers[{i}]: Missing required 'id' field")
        
        # Validate topics
        valid_durability = {'VOLATILE', 'TRANSIENT_LOCAL', 'TRANSIENT', 'PERSISTENT'}
        valid_reliability = {'BEST_EFFORT', 'RELIABLE'}
        valid_priority = {'LOW', 'MEDIUM', 'HIGH', 'URGENT'}
        
        for i, topic in enumerate(data.get('topics', [])):
            if not isinstance(topic, dict):
                result.add_schema_error(f"topics[{i}]: Expected object, got {type(topic).__name__}")
                continue
            if 'id' not in topic:
                result.add_schema_error(f"topics[{i}]: Missing required 'id' field")
            if 'qos' in topic and isinstance(topic['qos'], dict):
                qos = topic['qos']
                if 'durability' in qos and qos['durability'] not in valid_durability:
                    result.add_schema_error(f"topics[{i}].qos.durability: Invalid value '{qos['durability']}'")
                if 'reliability' in qos and qos['reliability'] not in valid_reliability:
                    result.add_schema_error(f"topics[{i}].qos.reliability: Invalid value '{qos['reliability']}'")
                if 'transport_priority' in qos and qos['transport_priority'] not in valid_priority:
                    result.add_schema_error(f"topics[{i}].qos.transport_priority: Invalid value '{qos['transport_priority']}'")
        
        # Validate nodes
        for i, node in enumerate(data.get('nodes', [])):
            if not isinstance(node, dict):
                result.add_schema_error(f"nodes[{i}]: Expected object, got {type(node).__name__}")
                continue
            if 'id' not in node:
                result.add_schema_error(f"nodes[{i}]: Missing required 'id' field")
        
        # Validate edges structure
        edges = data.get('relationships', {})
        if not isinstance(edges, dict):
            result.add_schema_error(f"relationships: Expected object, got {type(edges).__name__}")
        else:
            required_edge_types = ['publishes_to', 'subscribes_to', 'routes', 'runs_on', 'connects_to']
            for edge_type in required_edge_types:
                if edge_type not in edges:
                    result.add_schema_error(f"relationships: Missing required key '{edge_type}'")
                elif not isinstance(edges[edge_type], list):
                    result.add_schema_error(f"relationships.{edge_type}: Expected array, got {type(edges[edge_type]).__name__}")
        
        return result
    
    def build_from_dict_validated(self, data: Dict) -> Tuple[GraphModel, ValidationResult]:
        """
        Build GraphModel with schema validation
        
        Args:
            data: Dictionary with graph data
        
        Returns:
            Tuple of (GraphModel, ValidationResult including schema validation)
        """
        # First validate schema
        schema_result = self.validate_schema(data)
        
        if not schema_result.is_valid:
            # Return empty model with schema errors
            self.validation = schema_result
            return GraphModel(), schema_result
        
        # Build model
        model = self.build_from_dict(data)
        
        # Merge schema validation with build validation
        schema_result.merge(self.validation)
        
        return model, schema_result
    
    # -------------------------------------------------------------------------
    # Build from CSV
    # -------------------------------------------------------------------------
    
    def build_from_csv(self, vertices_file: str, edges_file: str) -> GraphModel:
        """
        Build GraphModel from CSV files
        
        Args:
            vertices_file: Path to vertices.csv (id, name, type, role, size, durability, reliability, transport_priority)
            edges_file: Path to edges.csv (from, to, type)
        
        Returns:
            GraphModel instance
        """
        self.logger.info(f"Building graph from CSV: {vertices_file}, {edges_file}")
        self.validation = ValidationResult()
        model = GraphModel()
        
        # Load vertices
        vertices_path = Path(vertices_file)
        if not vertices_path.exists():
            raise FileNotFoundError(f"Vertices file not found: {vertices_file}")
        
        with open(vertices_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                vertex_type = row.get('type', '').upper()
                vertex_id = row.get('id', '')
                name = row.get('name', vertex_id)
                
                if not vertex_id:
                    self.validation.add_error(f"Vertex[{i}]: Missing 'id' field")
                    continue
                
                if vertex_type == 'APPLICATION':
                    role = row.get('role', 'pubsub')
                    model.add_application(Application(id=vertex_id, name=name, role=role))
                
                elif vertex_type == 'BROKER':
                    model.add_broker(Broker(id=vertex_id, name=name))
                
                elif vertex_type == 'TOPIC':
                    size = int(row.get('size', 256)) if row.get('size') else 256
                    qos = QoSPolicy(
                        durability=row.get('durability', 'VOLATILE') or 'VOLATILE',
                        reliability=row.get('reliability', 'BEST_EFFORT') or 'BEST_EFFORT',
                        transport_priority=row.get('transport_priority', 'MEDIUM') or 'MEDIUM'
                    )
                    model.add_topic(Topic(id=vertex_id, name=name, size=size, qos=qos))
                
                elif vertex_type == 'NODE':
                    model.add_node(Node(id=vertex_id, name=name))
                
                else:
                    self.validation.add_warning(f"Vertex[{i}]: Unknown type '{vertex_type}'")
        
        # Load edges
        edges_path = Path(edges_file)
        if not edges_path.exists():
            raise FileNotFoundError(f"Edges file not found: {edges_file}")
        
        with open(edges_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                source = row.get('from', '')
                target = row.get('to', '')
                edge_type = row.get('type', '').upper()
                
                if not source or not target:
                    self.validation.add_warning(f"Edge[{i}]: Missing 'from' or 'to' field")
                    continue
                
                if edge_type not in [e.value for e in EdgeType]:
                    self.validation.add_warning(f"Edge[{i}]: Unknown type '{edge_type}'")
                    continue
                
                model.add_edge(Edge(source=source, target=target, edge_type=edge_type))
        
        return model
    
    # -------------------------------------------------------------------------
    # Build from Neo4j
    # -------------------------------------------------------------------------
    
    def build_from_neo4j(self, uri: str, auth: Tuple[str, str], database: str = "neo4j") -> GraphModel:
        """
        Build GraphModel from Neo4j database
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            auth: Tuple of (username, password)
            database: Database name (default: "neo4j")
        
        Returns:
            GraphModel instance
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("neo4j package required. Install with: pip install neo4j")
        
        self.logger.info(f"Building graph from Neo4j: {uri}")
        self.validation = ValidationResult()
        model = GraphModel()
        
        driver = GraphDatabase.driver(uri, auth=auth)
        try:
            with driver.session(database=database) as session:
                self._load_neo4j_vertices(session, model)
                self._load_neo4j_edges(session, model)
        finally:
            driver.close()
        
        return model
    
    def _load_neo4j_vertices(self, session, model: GraphModel) -> None:
        """Load all vertices from Neo4j"""
        # Applications
        for record in session.run("MATCH (a:Application) RETURN a"):
            props = dict(record['a'])
            model.add_application(Application(
                id=props.get('id', props.get('name', '')),
                name=props.get('name', ''),
                role=props.get('role', 'pubsub')
            ))
        
        # Brokers
        for record in session.run("MATCH (b:Broker) RETURN b"):
            props = dict(record['b'])
            model.add_broker(Broker(
                id=props.get('id', props.get('name', '')),
                name=props.get('name', '')
            ))
        
        # Topics
        for record in session.run("MATCH (t:Topic) RETURN t"):
            props = dict(record['t'])
            qos = QoSPolicy(
                durability=props.get('durability', 'VOLATILE'),
                reliability=props.get('reliability', 'BEST_EFFORT'),
                transport_priority=props.get('transport_priority', 'MEDIUM')
            )
            model.add_topic(Topic(
                id=props.get('id', props.get('name', '')),
                name=props.get('name', ''),
                size=props.get('size', 256),
                qos=qos
            ))
        
        # Nodes
        for record in session.run("MATCH (n:Node) RETURN n"):
            props = dict(record['n'])
            model.add_node(Node(
                id=props.get('id', props.get('name', '')),
                name=props.get('name', '')
            ))
    
    def _load_neo4j_edges(self, session, model: GraphModel) -> None:
        """Load all edges from Neo4j"""
        queries = [
            ("MATCH (a:Application)-[:PUBLISHES_TO]->(t:Topic) RETURN a.id as src, t.id as tgt", EdgeType.PUBLISHES_TO.value),
            ("MATCH (a:Application)-[:SUBSCRIBES_TO]->(t:Topic) RETURN a.id as src, t.id as tgt", EdgeType.SUBSCRIBES_TO.value),
            ("MATCH (b:Broker)-[:ROUTES]->(t:Topic) RETURN b.id as src, t.id as tgt", EdgeType.ROUTES.value),
            ("MATCH (c)-[:RUNS_ON]->(n:Node) RETURN c.id as src, n.id as tgt", EdgeType.RUNS_ON.value),
            ("MATCH (n1:Node)-[:CONNECTS_TO]->(n2:Node) RETURN n1.id as src, n2.id as tgt", EdgeType.CONNECTS_TO.value),
        ]
        
        for query, edge_type in queries:
            try:
                for record in session.run(query):
                    model.add_edge(Edge(
                        source=record['src'],
                        target=record['tgt'],
                        edge_type=edge_type
                    ))
            except Exception as e:
                self.validation.add_warning(f"Failed to load {edge_type} edges: {e}")
    
    # -------------------------------------------------------------------------
    # Build from NetworkX
    # -------------------------------------------------------------------------
    
    def build_from_networkx(self, G: Any) -> GraphModel:
        """
        Build GraphModel from a NetworkX graph
        
        Args:
            G: NetworkX DiGraph with typed nodes (type attribute: APPLICATION, BROKER, TOPIC, NODE)
        
        Returns:
            GraphModel instance
        """
        self.logger.info("Building graph from NetworkX")
        self.validation = ValidationResult()
        model = GraphModel()
        
        # Build vertices from nodes
        for node_id, attrs in G.nodes(data=True):
            node_type = attrs.get('type', '').upper()
            name = attrs.get('label', attrs.get('name', str(node_id)))
            
            if node_type == 'APPLICATION':
                model.add_application(Application(
                    id=str(node_id),
                    name=name,
                    role=attrs.get('role', 'pubsub')
                ))
            elif node_type == 'BROKER':
                model.add_broker(Broker(id=str(node_id), name=name))
            elif node_type == 'TOPIC':
                qos = QoSPolicy(
                    durability=attrs.get('durability', 'VOLATILE'),
                    reliability=attrs.get('reliability', 'BEST_EFFORT'),
                    transport_priority=attrs.get('transport_priority', 'MEDIUM')
                )
                model.add_topic(Topic(
                    id=str(node_id),
                    name=name,
                    size=attrs.get('size', 256),
                    qos=qos
                ))
            elif node_type == 'NODE':
                model.add_node(Node(id=str(node_id), name=name))
            else:
                self.validation.add_warning(f"Unknown node type '{node_type}' for '{node_id}'")
        
        # Build edges
        for src, tgt, attrs in G.edges(data=True):
            edge_type = attrs.get('edge_type', attrs.get('type', ''))
            if edge_type:
                model.add_edge(Edge(source=str(src), target=str(tgt), edge_type=edge_type))
            else:
                self.validation.add_warning(f"Edge ({src}, {tgt}) missing edge_type")
        
        return model
    
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    
    def validate(self, model: GraphModel, strict: bool = False) -> ValidationResult:
        """
        Validate a GraphModel for structural integrity
        
        Checks:
        - All edge references point to existing vertices
        - Edge types match source/target vertex types
        - No duplicate edges
        - Application role validity
        - Optionally checks for orphan topics and isolated apps
        
        Args:
            model: GraphModel to validate
            strict: If True, treat warnings as errors
        
        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult()
        
        all_ids = model.get_all_vertex_ids()
        app_ids = set(model.applications.keys())
        broker_ids = set(model.brokers.keys())
        topic_ids = set(model.topics.keys())
        node_ids = set(model.nodes.keys())
        
        # Check edge references
        seen_edges = set()
        for i, edge in enumerate(model.edges):
            edge_key = (edge.source, edge.target, edge.edge_type)
            
            # Check for duplicates
            if edge_key in seen_edges:
                result.add_warning(f"Duplicate edge: {edge.source} -[{edge.edge_type}]-> {edge.target}")
            seen_edges.add(edge_key)
            
            # Check source exists
            if edge.source not in all_ids:
                result.add_error(f"Edge[{i}]: Unknown source '{edge.source}'")
            
            # Check target exists
            if edge.target not in all_ids:
                result.add_error(f"Edge[{i}]: Unknown target '{edge.target}'")
            
            # Validate edge type matches vertex types
            if edge.edge_type == EdgeType.PUBLISHES_TO.value:
                if edge.source not in app_ids:
                    result.add_error(f"PUBLISHES_TO: Source '{edge.source}' is not an application")
                if edge.target not in topic_ids:
                    result.add_error(f"PUBLISHES_TO: Target '{edge.target}' is not a topic")
            
            elif edge.edge_type == EdgeType.SUBSCRIBES_TO.value:
                if edge.source not in app_ids:
                    result.add_error(f"SUBSCRIBES_TO: Source '{edge.source}' is not an application")
                if edge.target not in topic_ids:
                    result.add_error(f"SUBSCRIBES_TO: Target '{edge.target}' is not a topic")
            
            elif edge.edge_type == EdgeType.ROUTES.value:
                if edge.source not in broker_ids:
                    result.add_error(f"ROUTES: Source '{edge.source}' is not a broker")
                if edge.target not in topic_ids:
                    result.add_error(f"ROUTES: Target '{edge.target}' is not a topic")
            
            elif edge.edge_type == EdgeType.RUNS_ON.value:
                if edge.source not in (app_ids | broker_ids):
                    result.add_error(f"RUNS_ON: Source '{edge.source}' is not an application or broker")
                if edge.target not in node_ids:
                    result.add_error(f"RUNS_ON: Target '{edge.target}' is not a node")
            
            elif edge.edge_type == EdgeType.CONNECTS_TO.value:
                if edge.source not in node_ids:
                    result.add_error(f"CONNECTS_TO: Source '{edge.source}' is not a node")
                if edge.target not in node_ids:
                    result.add_error(f"CONNECTS_TO: Target '{edge.target}' is not a node")
        
        # Check application roles
        for app_id, app in model.applications.items():
            if app.role not in ('pub', 'sub', 'pubsub'):
                result.add_error(f"Application '{app_id}': Invalid role '{app.role}'")
        
        # Check for orphan topics
        orphans = model.get_orphan_topics()
        if orphans:
            msg = f"Found {len(orphans)} orphan topic(s) with no pub/sub connections"
            if strict:
                result.add_error(msg)
            else:
                result.add_warning(msg)
        
        # Check for isolated apps
        isolated = model.get_isolated_apps()
        if isolated:
            msg = f"Found {len(isolated)} isolated app(s) with no pub/sub connections"
            if strict:
                result.add_error(msg)
            else:
                result.add_warning(msg)
        
        # Add info about graph size
        result.add_info(f"Graph has {len(all_ids)} vertices and {len(model.edges)} edges")
        
        return result
    
    # -------------------------------------------------------------------------
    # Graph Merging
    # -------------------------------------------------------------------------
    
    def merge(self, models: List[GraphModel], prefix_ids: bool = True) -> GraphModel:
        """
        Merge multiple GraphModels into one
        
        Args:
            models: List of GraphModel instances to merge
            prefix_ids: If True, prefix IDs with model index to avoid conflicts
        
        Returns:
            Merged GraphModel
        """
        self.validation = ValidationResult()
        merged = GraphModel()
        
        for idx, model in enumerate(models):
            prefix = f"m{idx}_" if prefix_ids else ""
            
            # Merge applications
            for app_id, app in model.applications.items():
                new_id = f"{prefix}{app_id}"
                merged.add_application(Application(id=new_id, name=app.name, role=app.role))
            
            # Merge brokers
            for broker_id, broker in model.brokers.items():
                new_id = f"{prefix}{broker_id}"
                merged.add_broker(Broker(id=new_id, name=broker.name))
            
            # Merge topics
            for topic_id, topic in model.topics.items():
                new_id = f"{prefix}{topic_id}"
                merged.add_topic(Topic(id=new_id, name=topic.name, size=topic.size, qos=topic.qos))
            
            # Merge nodes
            for node_id, node in model.nodes.items():
                new_id = f"{prefix}{node_id}"
                merged.add_node(Node(id=new_id, name=node.name))
            
            # Merge edges
            for edge in model.edges:
                new_source = f"{prefix}{edge.source}"
                new_target = f"{prefix}{edge.target}"
                merged.add_edge(Edge(source=new_source, target=new_target, edge_type=edge.edge_type))
        
        self.logger.info(f"Merged {len(models)} graphs into one with {len(merged.edges)} edges")
        return merged
    
    # -------------------------------------------------------------------------
    # Graph Filtering
    # -------------------------------------------------------------------------
    
    def filter(self, model: GraphModel, 
               vertex_filter: Optional[Callable[[Any], bool]] = None,
               edge_filter: Optional[Callable[[Edge], bool]] = None) -> GraphModel:
        """
        Create a filtered copy of a GraphModel
        
        Args:
            model: Source GraphModel
            vertex_filter: Function that returns True for vertices to keep
            edge_filter: Function that returns True for edges to keep
        
        Returns:
            Filtered GraphModel
        """
        filtered = GraphModel()
        filtered.metadata = model.metadata.copy()
        
        # Filter vertices
        kept_ids = set()
        
        for app_id, app in model.applications.items():
            if vertex_filter is None or vertex_filter(app):
                filtered.add_application(app)
                kept_ids.add(app_id)
        
        for broker_id, broker in model.brokers.items():
            if vertex_filter is None or vertex_filter(broker):
                filtered.add_broker(broker)
                kept_ids.add(broker_id)
        
        for topic_id, topic in model.topics.items():
            if vertex_filter is None or vertex_filter(topic):
                filtered.add_topic(topic)
                kept_ids.add(topic_id)
        
        for node_id, node in model.nodes.items():
            if vertex_filter is None or vertex_filter(node):
                filtered.add_node(node)
                kept_ids.add(node_id)
        
        # Filter edges (only keep edges where both endpoints are kept)
        for edge in model.edges:
            if edge.source in kept_ids and edge.target in kept_ids:
                if edge_filter is None or edge_filter(edge):
                    filtered.add_edge(edge)
        
        return filtered
    
    def subgraph(self, model: GraphModel, vertex_ids: Set[str]) -> GraphModel:
        """
        Extract a subgraph containing only specified vertices
        
        Args:
            model: Source GraphModel
            vertex_ids: Set of vertex IDs to include
        
        Returns:
            Subgraph GraphModel
        """
        return self.filter(
            model,
            vertex_filter=lambda v: v.id in vertex_ids
        )
    
    # -------------------------------------------------------------------------
    # Cypher Generation
    # -------------------------------------------------------------------------
    
    def generate_cypher(self, model: GraphModel, clear_existing: bool = True) -> List[str]:
        """
        Generate Cypher queries for Neo4j import
        
        Args:
            model: GraphModel to export
            clear_existing: If True, include MATCH DELETE statement
        
        Returns:
            List of Cypher query strings
        """
        queries = []
        
        if clear_existing:
            queries.append("// Clear existing data")
            queries.append("MATCH (n) DETACH DELETE n;")
            queries.append("")
        
        queries.append("// Create constraints")
        queries.append("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Application) REQUIRE a.id IS UNIQUE;")
        queries.append("CREATE CONSTRAINT IF NOT EXISTS FOR (b:Broker) REQUIRE b.id IS UNIQUE;")
        queries.append("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE;")
        queries.append("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE;")
        queries.append("")
        
        # Create vertices
        queries.append("// Create Applications")
        for app in model.applications.values():
            queries.append(
                f"CREATE (:Application {{id: '{app.id}', name: '{self._esc(app.name)}', role: '{app.role}'}});"
            )
        queries.append("")
        
        queries.append("// Create Brokers")
        for broker in model.brokers.values():
            queries.append(
                f"CREATE (:Broker {{id: '{broker.id}', name: '{self._esc(broker.name)}'}});"
            )
        queries.append("")
        
        queries.append("// Create Topics")
        for topic in model.topics.values():
            qos = topic.qos if isinstance(topic.qos, QoSPolicy) else QoSPolicy()
            queries.append(
                f"CREATE (:Topic {{id: '{topic.id}', name: '{self._esc(topic.name)}', size: {topic.size}, "
                f"durability: '{qos.durability}', reliability: '{qos.reliability}', "
                f"transport_priority: '{qos.transport_priority}'}});"
            )
        queries.append("")
        
        queries.append("// Create Nodes")
        for node in model.nodes.values():
            queries.append(
                f"CREATE (:Node {{id: '{node.id}', name: '{self._esc(node.name)}'}});"
            )
        queries.append("")
        
        # Create edges
        queries.append("// Create relationships")
        edge_labels = {
            EdgeType.PUBLISHES_TO.value: ("Application", "Topic"),
            EdgeType.SUBSCRIBES_TO.value: ("Application", "Topic"),
            EdgeType.ROUTES.value: ("Broker", "Topic"),
            EdgeType.RUNS_ON.value: (None, "Node"),
            EdgeType.CONNECTS_TO.value: ("Node", "Node"),
        }
        
        for edge in model.edges:
            src_label, tgt_label = edge_labels.get(edge.edge_type, (None, None))
            if src_label:
                queries.append(
                    f"MATCH (a:{src_label} {{id: '{edge.source}'}}), (b:{tgt_label} {{id: '{edge.target}'}}) "
                    f"CREATE (a)-[:{edge.edge_type}]->(b);"
                )
            else:
                queries.append(
                    f"MATCH (a {{id: '{edge.source}'}}), (b:{tgt_label} {{id: '{edge.target}'}}) "
                    f"CREATE (a)-[:{edge.edge_type}]->(b);"
                )
        
        return queries
    
    def generate_cypher_batch(self, model: GraphModel) -> str:
        """
        Generate optimized batch Cypher for large graphs using UNWIND
        
        Args:
            model: GraphModel to export
        
        Returns:
            Single Cypher script string
        """
        lines = []
        
        lines.append("// Batch import using UNWIND for better performance")
        lines.append("")
        
        # Applications
        if model.applications:
            apps_data = [{'id': a.id, 'name': a.name, 'role': a.role} for a in model.applications.values()]
            lines.append("// Create Applications")
            lines.append(f"UNWIND {json.dumps(apps_data)} AS app")
            lines.append("CREATE (:Application {id: app.id, name: app.name, role: app.role});")
            lines.append("")
        
        # Brokers
        if model.brokers:
            brokers_data = [{'id': b.id, 'name': b.name} for b in model.brokers.values()]
            lines.append("// Create Brokers")
            lines.append(f"UNWIND {json.dumps(brokers_data)} AS broker")
            lines.append("CREATE (:Broker {id: broker.id, name: broker.name});")
            lines.append("")
        
        # Topics
        if model.topics:
            topics_data = []
            for t in model.topics.values():
                qos = t.qos if isinstance(t.qos, QoSPolicy) else QoSPolicy()
                topics_data.append({
                    'id': t.id, 'name': t.name, 'size': t.size,
                    'durability': qos.durability, 'reliability': qos.reliability,
                    'transport_priority': qos.transport_priority
                })
            lines.append("// Create Topics")
            lines.append(f"UNWIND {json.dumps(topics_data)} AS topic")
            lines.append("CREATE (:Topic {id: topic.id, name: topic.name, size: topic.size, "
                        "durability: topic.durability, reliability: topic.reliability, "
                        "transport_priority: topic.transport_priority});")
            lines.append("")
        
        # Nodes
        if model.nodes:
            nodes_data = [{'id': n.id, 'name': n.name} for n in model.nodes.values()]
            lines.append("// Create Nodes")
            lines.append(f"UNWIND {json.dumps(nodes_data)} AS node")
            lines.append("CREATE (:Node {id: node.id, name: node.name});")
            lines.append("")
        
        # Edges by type
        for edge_type in EdgeType:
            edges = model.get_edges_by_type(edge_type.value)
            if edges:
                edges_data = [{'src': e.source, 'tgt': e.target} for e in edges]
                lines.append(f"// Create {edge_type.value} relationships")
                lines.append(f"UNWIND {json.dumps(edges_data)} AS edge")
                lines.append(f"MATCH (a {{id: edge.src}}), (b {{id: edge.tgt}})")
                lines.append(f"CREATE (a)-[:{edge_type.value}]->(b);")
                lines.append("")
        
        return '\n'.join(lines)
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def _esc(self, s: str) -> str:
        """Escape string for Cypher"""
        if not s:
            return ''
        return s.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
    
    def get_build_summary(self, model: GraphModel) -> str:
        """Get a summary of the built graph"""
        stats = model.get_statistics()
        lines = [
            "Graph Build Summary",
            "=" * 40,
            f"Applications: {stats['num_applications']}",
            f"Brokers:      {stats['num_brokers']}",
            f"Topics:       {stats['num_topics']}",
            f"Nodes:        {stats['num_nodes']}",
            f"Total Relationships:  {stats['num_relationships']}",
            "",
            "Relationships by Type:",
        ]
        for edge_type, count in stats['edges_by_type'].items():
            lines.append(f"  {edge_type}: {count}")
        
        if self.validation.errors:
            lines.append("")
            lines.append(f"Errors: {len(self.validation.errors)}")
        if self.validation.warnings:
            lines.append(f"Warnings: {len(self.validation.warnings)}")
        
        return '\n'.join(lines)
    
    # -------------------------------------------------------------------------
    # Graph Comparison
    # -------------------------------------------------------------------------
    
    def compare(self, model1: GraphModel, model2: GraphModel) -> GraphDiffResult:
        """
        Compare two GraphModels and return differences
        
        Args:
            model1: First (baseline) GraphModel
            model2: Second (comparison) GraphModel
        
        Returns:
            GraphDiffResult with changes
        """
        diff = GraphDiffResult()
        
        # Compare applications
        self._compare_vertex_type(
            model1.applications, model2.applications, 
            'applications', diff
        )
        
        # Compare brokers
        self._compare_vertex_type(
            model1.brokers, model2.brokers,
            'brokers', diff
        )
        
        # Compare topics
        self._compare_vertex_type(
            model1.topics, model2.topics,
            'topics', diff
        )
        
        # Compare nodes
        self._compare_vertex_type(
            model1.nodes, model2.nodes,
            'nodes', diff
        )
        
        # Compare edges
        edges1 = {(e.source, e.target, e.edge_type) for e in model1.edges}
        edges2 = {(e.source, e.target, e.edge_type) for e in model2.edges}
        
        for edge_key in edges2 - edges1:
            diff.added_edges.append(Edge(source=edge_key[0], target=edge_key[1], edge_type=edge_key[2]))
        
        for edge_key in edges1 - edges2:
            diff.removed_edges.append(Edge(source=edge_key[0], target=edge_key[1], edge_type=edge_key[2]))
        
        return diff
    
    def _compare_vertex_type(self, dict1: Dict, dict2: Dict, vtype: str, diff: GraphDiffResult) -> None:
        """Compare vertices of a specific type"""
        ids1 = set(dict1.keys())
        ids2 = set(dict2.keys())
        
        # Added
        for vid in ids2 - ids1:
            diff.added_vertices[vtype].append(vid)
        
        # Removed
        for vid in ids1 - ids2:
            diff.removed_vertices[vtype].append(vid)
        
        # Modified (compare by dict representation)
        for vid in ids1 & ids2:
            v1_dict = dict1[vid].to_dict()
            v2_dict = dict2[vid].to_dict()
            if v1_dict != v2_dict:
                diff.modified_vertices[vtype].append((vid, v1_dict, v2_dict))
    
    # -------------------------------------------------------------------------
    # Additional Utility Methods
    # -------------------------------------------------------------------------
    
    def clone(self, model: GraphModel) -> GraphModel:
        """
        Create a deep copy of a GraphModel
        
        Args:
            model: Source GraphModel
        
        Returns:
            New GraphModel instance with same data
        """
        return self.build_from_dict(model.to_dict())
    
    def get_schema(self) -> Dict:
        """Get the JSON schema for v3.0 graph format"""
        return GRAPH_SCHEMA.copy()
    
    def auto_build(self, filepath: str) -> GraphModel:
        """
        Auto-detect format and build GraphModel
        
        Supports: JSON (.json), YAML (.yaml, .yml), v2.1 legacy
        
        Args:
            filepath: Path to graph file
        
        Returns:
            GraphModel instance
        """
        path = Path(filepath)
        ext = path.suffix.lower()
        
        if ext in ('.yaml', '.yml'):
            return self.build_from_yaml(filepath)
        
        # For JSON, detect version
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        version = self.detect_version(data)
        self.logger.info(f"Detected graph version: {version}")
        
        if version == '2.1':
            self.logger.info("Automatically migrating from v2.1 to v3.0")
            data = self.migrate_from_v2(data)
        
        return self.build_from_dict(data)
    
    def export_validation_report(self, model: GraphModel, filepath: str) -> None:
        """
        Generate and export a validation report
        
        Args:
            model: GraphModel to validate
            filepath: Output path for report (JSON or TXT)
        """
        result = self.validate(model)
        path = Path(filepath)
        
        if path.suffix.lower() == '.json':
            report = result.to_dict()
            report['model_stats'] = model.get_statistics()
            report['generated_at'] = datetime.now(timezone.utc).isoformat()
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("Graph Validation Report\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
                f.write(result.summary())
                f.write("\n\n")
                f.write(self.get_build_summary(model))
        
        self.logger.info(f"Validation report exported to: {filepath}")