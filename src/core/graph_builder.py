"""
Graph Builder

Constructs GraphModel instances from various data sources including:
- JSON files
- CSV files
- Neo4j databases
- ROS2 DDS introspection
- Kafka metadata
- Direct Python dictionaries

Automatically derives relationships (DEPENDS_ON, CONNECTS_TO) from
explicit relationships (PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON).
"""

import json
import csv
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
import logging
from datetime import datetime

from .graph_model import (
    GraphModel, ApplicationNode, TopicNode, BrokerNode, InfrastructureNode,
    QoSPolicy, QoSDurability, QoSReliability, ApplicationType, MessagePattern,
    PublishesEdge, SubscribesEdge, RoutesEdge, RunsOnEdge, ConnectsToEdge, DependsOnEdge
)


class GraphBuilder:
    """
    Builds GraphModel instances from various data sources with robust error handling
    
    Key Features:
    - Flexible edge field mapping (from/to vs source/target)
    - Graceful handling of missing/invalid data
    - Comprehensive validation
    - Detailed error messages
    """
    
    def __init__(self):
        """Initialize the graph builder"""
        self.logger = logging.getLogger(__name__)
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def build_from_json(self, filepath: str) -> GraphModel:
        """
        Build graph from JSON configuration file
        
        Expected JSON structure (flexible):
        {
            "applications": [...],
            "topics": [...],
            "brokers": [...],
            "nodes": [...],
            "edges": {
                "publishes": [...],    // or "publishes_to"
                "subscribes": [...],   // or "subscribes_to"
                "routes": [...],
                "runs_on": [...],
                "connects_to": [...]
            }
        }
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            GraphModel instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        self.logger.info(f"Building graph from JSON: {filepath}")
        self.errors = []
        self.warnings = []
        
        # Validate file exists
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Input file not found: {filepath}")
        
        # Load JSON
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON format in {filepath}: {e.msg}",
                e.doc, e.pos
            )
        
        # Build from dictionary
        return self.build_from_dict(data)
    
    def build_from_dict(self, data: Dict) -> GraphModel:
        """
        Build graph from Python dictionary with comprehensive error handling
        
        Args:
            data: Dictionary with system configuration
        
        Returns:
            GraphModel instance
        """
        self.logger.info("Building graph from dictionary...")
        
        model = GraphModel()
        
        # Step 1: Add nodes with error handling
        self._add_applications(model, data.get('applications', []))
        self._add_topics(model, data.get('topics', []))
        self._add_brokers(model, data.get('brokers', []))
        self._add_infrastructure_nodes(model, data.get('nodes', []))
        
        # Step 2: Add edges with flexible field mapping
        edges = data.get('relationships', {})
        self._add_publishes_edges(model, edges)
        self._add_subscribes_edges(model, edges)
        self._add_routes_edges(model, edges)
        self._add_runs_on_edges(model, edges)
        self._add_connects_to_edges(model, edges)
        
        # Step 3: Derive DEPENDS_ON relationships
        self._derive_dependencies(model)
        
        # Step 4: Report summary
        summary = model.summary()
        self.logger.info(
            f"Graph built successfully: {summary['total_nodes']} nodes, "
            f"{summary['total_edges']} edges"
        )
        
        if self.warnings:
            self.logger.warning(f"Build completed with {len(self.warnings)} warnings")
            for warning in self.warnings[:5]:  # Show first 5
                self.logger.warning(f"  - {warning}")
        
        if self.errors:
            self.logger.error(f"Build completed with {len(self.errors)} errors")
            for error in self.errors[:5]:  # Show first 5
                self.logger.error(f"  - {error}")
        
        return model
    
    # ========================================================================
    # Node Creation Methods
    # ========================================================================
    
    def _add_applications(self, model: GraphModel, apps_data: List[Dict]):
        """Add applications with error handling"""
        for idx, app_data in enumerate(apps_data):
            try:
                if 'name' not in app_data and 'id' not in app_data:
                    self.errors.append(f"Application {idx}: missing 'name' or 'id' field")
                    continue
                
                id =  app_data.get('id')
                name = app_data.get('name')
                
                # Parse application type
                app_type_str = app_data.get('type', 'PROSUMER')
                try:
                    app_type = ApplicationType[app_type_str] if app_type_str in ApplicationType.__members__ else ApplicationType.PROSUMER
                except (KeyError, ValueError):
                    app_type = ApplicationType.PROSUMER
                    self.warnings.append(f"App {name}: invalid type '{app_type_str}', using PROSUMER")
                
                app = ApplicationNode(
                    id=id,
                    name=name,
                    app_type=app_type
                )
                model.add_application(app)
                
            except Exception as e:
                self.errors.append(f"Application {idx}: {str(e)}")
    
    def _add_topics(self, model: GraphModel, topics_data: List[Dict]):
        """Add topics with error handling"""
        for idx, topic_data in enumerate(topics_data):
            try:
                if 'name' not in topic_data and 'id' not in topic_data:
                    self.errors.append(f"Topic {idx}: missing 'name' or 'id' field")
                    continue
                
                id = topic_data.get('id')
                name = topic_data.get('name')
                message_type = topic_data.get('message_type', 'unknown')
                
                # Parse QoS if present
                qos_policy = None
                if 'qos_policy' in topic_data:
                    qos_policy = self._parse_qos_policy(topic_data['qos_policy'])
                
                topic = TopicNode(
                    id=id,
                    name=name,
                    message_type=message_type,
                    qos_policy=qos_policy
                )
                model.add_topic(topic)
                
            except Exception as e:
                self.errors.append(f"Topic {idx}: {str(e)}")
    
    def _add_brokers(self, model: GraphModel, brokers_data: List[Dict]):
        """Add brokers with error handling"""
        for idx, broker_data in enumerate(brokers_data):
            try:
                if 'name' not in broker_data and 'id' not in broker_data:
                    self.errors.append(f"Broker {idx}: missing 'name' or 'id' field")
                    continue
                
                id = broker_data.get('id')
                name = broker_data.get('name')
                broker_type = broker_data.get('protocol', 'DDS')
                
                broker = BrokerNode(
                    id=id,
                    name=name,
                    broker_type=broker_type
                )
                model.add_broker(broker)
                
            except Exception as e:
                self.errors.append(f"Broker {idx}: {str(e)}")
    
    def _add_infrastructure_nodes(self, model: GraphModel, nodes_data: List[Dict]):
        """Add infrastructure nodes with error handling"""
        for idx, node_data in enumerate(nodes_data):
            try:
                if 'name' not in node_data and 'id' not in node_data:
                    self.errors.append(f"Node {idx}: missing 'name' or 'id' field")
                    continue
                
                id = node_data.get('id')
                name = node_data.get('name')
                
                node = InfrastructureNode(
                    id=id,
                    name=name
                )
                model.add_node(node)
                
            except Exception as e:
                self.errors.append(f"Infrastructure node {idx}: {str(e)}")
    
    # ========================================================================
    # Edge Creation Methods with Flexible Field Mapping
    # ========================================================================
    
    def _add_publishes_edges(self, model: GraphModel, edges: Dict):
        """Add publishes edges with flexible field mapping"""
        # Try multiple possible field names
        publishes_data = (edges.get('publishes') or 
                         edges.get('publishes_to') or 
                         [])
        
        for idx, edge_data in enumerate(publishes_data):
            try:
                # Support both from/to and source/target
                source = edge_data.get('source') or edge_data.get('from')
                target = edge_data.get('target') or edge_data.get('to')
                
                if not source or not target:
                    self.errors.append(f"Publishes edge {idx}: missing source or target")
                    continue
                
                # Validate references
                if source not in model.applications:
                    self.warnings.append(f"Publishes edge {idx}: unknown application '{source}'")
                if target not in model.topics:
                    self.warnings.append(f"Publishes edge {idx}: unknown topic '{target}'")
                
                edge = PublishesEdge(
                    source=source,
                    target=target,
                    **{k: v for k, v in edge_data.items() 
                       if k not in ['source', 'target', 'from', 'to']}
                )
                model.publishes_edges.append(edge)
                
            except Exception as e:
                self.errors.append(f"Publishes edge {idx}: {str(e)}")
    
    def _add_subscribes_edges(self, model: GraphModel, edges: Dict):
        """Add subscribes edges with flexible field mapping"""
        # Try multiple possible field names
        subscribes_data = (edges.get('subscribes') or 
                          edges.get('subscribes_to') or 
                          [])
        
        for idx, edge_data in enumerate(subscribes_data):
            try:
                # Support both from/to and source/target
                source = edge_data.get('source') or edge_data.get('from')
                target = edge_data.get('target') or edge_data.get('to')
                
                if not source or not target:
                    self.errors.append(f"Subscribes edge {idx}: missing source or target")
                    continue
                
                # Validate references
                if source not in model.applications:
                    self.warnings.append(f"Subscribes edge {idx}: unknown application '{source}'")
                if target not in model.topics:
                    self.warnings.append(f"Subscribes edge {idx}: unknown topic '{target}'")
                
                edge = SubscribesEdge(
                    source=source,
                    target=target,
                    **{k: v for k, v in edge_data.items() 
                       if k not in ['source', 'target', 'from', 'to']}
                )
                model.subscribes_edges.append(edge)
                
            except Exception as e:
                self.errors.append(f"Subscribes edge {idx}: {str(e)}")
    
    def _add_routes_edges(self, model: GraphModel, edges: Dict):
        """Add routes edges with flexible field mapping"""
        routes_data = edges.get('routes', [])
        
        for idx, edge_data in enumerate(routes_data):
            try:
                # Support both from/to and source/target
                source = edge_data.get('source') or edge_data.get('from')
                target = edge_data.get('target') or edge_data.get('to')
                
                if not source or not target:
                    self.errors.append(f"Routes edge {idx}: missing source or target")
                    continue
                
                # Validate references
                if source not in model.brokers:
                    self.warnings.append(f"Routes edge {idx}: unknown broker '{source}'")
                if target not in model.topics:
                    self.warnings.append(f"Routes edge {idx}: unknown topic '{target}'")
                
                edge = RoutesEdge(
                    source=source,
                    target=target,
                    **{k: v for k, v in edge_data.items() 
                       if k not in ['source', 'target', 'from', 'to']}
                )
                model.routes_edges.append(edge)
                
            except Exception as e:
                self.errors.append(f"Routes edge {idx}: {str(e)}")
    
    def _add_runs_on_edges(self, model: GraphModel, edges: Dict):
        """Add runs_on edges with flexible field mapping"""
        runs_on_data = edges.get('runs_on', [])
        
        for idx, edge_data in enumerate(runs_on_data):
            try:
                # Support both from/to and source/target
                source = edge_data.get('source') or edge_data.get('from')
                target = edge_data.get('target') or edge_data.get('to')
                
                if not source or not target:
                    self.errors.append(f"RunsOn edge {idx}: missing source or target")
                    continue
                
                # Validate references
                if source not in model.applications:
                    self.warnings.append(f"RunsOn edge {idx}: unknown application '{source}'")
                if target not in model.nodes:
                    self.warnings.append(f"RunsOn edge {idx}: unknown node '{target}'")
                
                edge = RunsOnEdge(
                    source=source,
                    target=target,
                    **{k: v for k, v in edge_data.items() 
                       if k not in ['source', 'target', 'from', 'to']}
                )
                model.runs_on_edges.append(edge)
                
            except Exception as e:
                self.errors.append(f"RunsOn edge {idx}: {str(e)}")
    
    def _add_connects_to_edges(self, model: GraphModel, edges: Dict):
        """Add connects_to edges with flexible field mapping"""
        connects_data = edges.get('connects_to', [])
        
        for idx, edge_data in enumerate(connects_data):
            try:
                # Support both from/to and source/target
                source = edge_data.get('source') or edge_data.get('from')
                target = edge_data.get('target') or edge_data.get('to')
                
                if not source or not target:
                    self.errors.append(f"ConnectsTo edge {idx}: missing source or target")
                    continue
                
                # Validate references
                if source not in model.brokers:
                    self.warnings.append(f"ConnectsTo edge {idx}: unknown broker '{source}'")
                if target not in model.brokers:
                    self.warnings.append(f"ConnectsTo edge {idx}: unknown broker '{target}'")
                
                edge = ConnectsToEdge(
                    source=source,
                    target=target,
                    **{k: v for k, v in edge_data.items() 
                       if k not in ['source', 'target', 'from', 'to']}
                )
                model.connects_to_edges.append(edge)
                
            except Exception as e:
                self.errors.append(f"ConnectsTo edge {idx}: {str(e)}")
    
    # ========================================================================
    # Dependency Derivation
    # ========================================================================
    
    def _derive_dependencies(self, model: GraphModel):
        """Derive DEPENDS_ON relationships from existing edges"""
        self.logger.info("Deriving DEPENDS_ON relationships...")
        self._derive_depends_on_relationships(model)
        self.logger.info("Deriving CONNECTS_TO relationships...")
        self._derive_connects_to_relationships(model)

    def _derive_depends_on_relationships(self, model: GraphModel):
        """
        Derive DEPENDS_ON relationships from PUBLISHES_TO and SUBSCRIBES_TO
        
        Logic:
        - If App1 publishes to Topic T and App2 subscribes to T, then App2 DEPENDS_ON App1
        """
        self.logger.info("Deriving DEPENDS_ON relationships...")
        
        # Map topics to their publishers and subscribers
        topic_publishers: Dict[str, List[str]] = {}
        topic_subscribers: Dict[str, List[str]] = {}
        
        for edge in model.publishes_edges:
            if edge.target not in topic_publishers:
                topic_publishers[edge.target] = []
            topic_publishers[edge.target].append(edge.source)
        
        for edge in model.subscribes_edges:
            if edge.target not in topic_subscribers:
                topic_subscribers[edge.target] = []
            topic_subscribers[edge.target].append(edge.source)
        
        # Derive application-to-application dependencies
        for topic, publishers in topic_publishers.items():
            if topic in topic_subscribers:
                subscribers = topic_subscribers[topic]
                
                for subscriber in subscribers:
                    for publisher in publishers:
                        if subscriber != publisher:
                            # Subscriber depends on publisher
                            model.depends_on_edges.append(DependsOnEdge(
                                source=subscriber,
                                target=publisher,
                                dependency_type='FUNCTIONAL',
                                strength=0.8
                            ))
        
        self.logger.info(f"Derived {len(model.depends_on_edges)} DEPENDS_ON relationships")

    def _derive_connects_to_relationships(self, model: GraphModel):
        """
        Derive CONNECTS_TO relationships between infrastructure nodes based on
        applications and brokers running on them.

        Logic:
        1. Map each application to its infrastructure node (via RUNS_ON)
        2. For each application that depends on other application, create CONNECTS_TO edges
           between the app's node and other app's node (if different)
        3. Deduplicate edges to avoid creating multiple CONNECTS_TO edges
           between the same pair of nodes
        """
        app_to_node: Dict[str, str] = {}
        
        # Map applications to their infrastructure nodes
        for edge in model.runs_on_edges:
            app_to_node[edge.source] = edge.target
        
        # Track created CONNECTS_TO edges to avoid duplicates
        created_edges: Set[Tuple[str, str]] = set()
        
        # Create CONNECTS_TO edges based on DEPENDS_ON relationships
        for dep_edge in model.depends_on_edges:
            source = dep_edge.source
            target = dep_edge.target
            
            node_source = app_to_node.get(source)
            node_target = app_to_node.get(target)

            if node_source and node_target and node_source != node_target:
                edge_key = (node_source, node_target)
                if edge_key not in created_edges:
                    model.connects_to_edges.append(ConnectsToEdge(
                        source=node_source,
                        target=node_target
                    ))
                    created_edges.add(edge_key)
        
        self.logger.info(f"Derived {len(created_edges)} CONNECTS_TO relationships")

    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _parse_qos_policy(self, data: Dict) -> QoSPolicy:
        """Parse QoS policy from dictionary with error handling"""
        try:
            # Parse durability
            durability_str = data.get('durability', 'VOLATILE')
            try:
                durability = QoSDurability[durability_str]
            except (KeyError, ValueError):
                durability = QoSDurability.VOLATILE
                self.warnings.append(f"Invalid durability '{durability_str}', using VOLATILE")
            
            # Parse reliability
            reliability_str = data.get('reliability', 'BEST_EFFORT')
            try:
                reliability = QoSReliability[reliability_str]
            except (KeyError, ValueError):
                reliability = QoSReliability.BEST_EFFORT
                self.warnings.append(f"Invalid reliability '{reliability_str}', using BEST_EFFORT")
            
            return QoSPolicy(
                durability=durability,
                reliability=reliability,
                deadline_ms=self._parse_float(data.get('deadline_ms')),
                lifespan_ms=self._parse_float(data.get('lifespan_ms')),
                transport_priority=self._parse_int(data.get('transport_priority', 0)),
                history_depth=self._parse_int(data.get('history_depth', 1))
            )
        except Exception as e:
            self.warnings.append(f"Error parsing QoS policy: {str(e)}, using defaults")
            return QoSPolicy()
    
    def _parse_float(self, value: Any) -> Optional[float]:
        """Safely parse float value"""
        if value is None or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _parse_int(self, value: Any) -> int:
        """Safely parse int value"""
        if value is None or value == '':
            return 0
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
    
    def _parse_bool(self, value: Any) -> bool:
        """Safely parse bool value"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)
