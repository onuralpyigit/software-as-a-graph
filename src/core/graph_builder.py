"""
Graph Builder

Constructs GraphModel instances from various data sources including:
- JSON files
- CSV files
- Neo4j databases
- ROS2 DDS introspection
- Kafka metadata
- Direct Python dictionaries

Automatically derives unified DEPENDS_ON relationships across all layers:
- APP_TO_APP: From topic subscription overlap
- APP_TO_BROKER: From topic routing relationships
- NODE_TO_NODE: From application dependencies across nodes
- NODE_TO_BROKER: From broker placement dependencies
"""

import json
import csv
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from collections import defaultdict
import logging
from datetime import datetime

from .graph_model import (
    GraphModel, ApplicationNode, TopicNode, BrokerNode, InfrastructureNode,
    QoSPolicy, QoSDurability, QoSReliability, QosTransportPriority, ApplicationType,
    PublishesEdge, SubscribesEdge, RoutesEdge, RunsOnEdge, ConnectsToEdge, 
    DependsOnEdge, DependencyType
)


class GraphBuilder:
    """
    Builds GraphModel instances from various data sources with robust error handling.
    
    Key Features:
    - Flexible edge field mapping (from/to vs source/target)
    - Graceful handling of missing/invalid data
    - Comprehensive validation
    - Unified DEPENDS_ON derivation across all system layers
    - Detailed error messages
    """
    
    def __init__(self):
        """Initialize the graph builder"""
        self.logger = logging.getLogger(__name__)
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    # =========================================================================
    # Public Build Methods
    # =========================================================================
    
    def build_from_json(self, filepath: str) -> GraphModel:
        """
        Build graph from JSON configuration file.
        
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
                "connects_to": [...]    // network topology
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
        Build graph from dictionary data.
        
        Args:
            data: Dictionary containing graph definition
        
        Returns:
            GraphModel instance
        """
        self.logger.info("Building graph from dictionary...")
        self.errors = []
        self.warnings = []
        
        model = GraphModel()
        
        # Build nodes
        self._build_applications(model, data.get('applications', []))
        self._build_topics(model, data.get('topics', []))
        self._build_brokers(model, data.get('brokers', []))
        self._build_infrastructure_nodes(model, data.get('nodes', []))
        
        # Build explicit edges
        edges = data.get('edges', data.get('relationships', {}))
        self._build_publishes_edges(model, edges)
        self._build_subscribes_edges(model, edges)
        self._build_routes_edges(model, edges)
        self._build_runs_on_edges(model, edges)
        self._build_connects_to_edges(model, edges)
        
        # Derive unified DEPENDS_ON relationships
        self._derive_all_dependencies(model)
        
        # Log summary
        stats = model.get_statistics()
        self.logger.info(
            f"Built graph with {stats['vertices']['total']} vertices and "
            f"{stats['derived_edges']['depends_on']} derived dependencies"
        )
        
        if self.errors:
            self.logger.warning(f"Build completed with {len(self.errors)} errors")
        if self.warnings:
            self.logger.info(f"Build completed with {len(self.warnings)} warnings")
        
        return model
    
    # =========================================================================
    # Node Building Methods
    # =========================================================================
    
    def _build_applications(self, model: GraphModel, apps: List[Dict]):
        """Build application nodes"""
        for app_data in apps:
            try:
                app_id = app_data.get('id', app_data.get('name'))
                if not app_id:
                    self.errors.append("Application missing 'id' or 'name' field")
                    continue
                app_name = app_data.get('name')
                
                # Parse application type
                app_type_str = app_data.get('app_type', 'PROSUMER')
                try:
                    app_type = ApplicationType[app_type_str] if app_type_str in ApplicationType.__members__ else ApplicationType.PROSUMER
                except (KeyError, ValueError):
                    app_type = ApplicationType.PROSUMER
                    self.warnings.append(f"App {app_name}: invalid type '{app_type_str}', using PROSUMER")
                
                app = ApplicationNode(
                    id=app_id,
                    name=app_name,
                    app_type=app_type
                )
                model.add_application(app)
                
            except Exception as e:
                self.errors.append(f"Error building application: {e}")
        
        self.logger.info(f"  Built {len(model.applications)} applications")
    
    def _build_topics(self, model: GraphModel, topics: List[Dict]):
        """Build topic nodes with QoS policies"""
        for topic_data in topics:
            try:
                topic_id = topic_data.get('id', topic_data.get('name'))
                if not topic_id:
                    self.errors.append("Topic missing 'id' or 'name' field")
                    continue
                topic_name = topic_data.get('name')

                # Parse QoS if present
                qos = None
                qos_data = topic_data.get('qos')
                if qos_data:
                    qos = QoSPolicy.from_dict(qos_data)

                message_size_bytes = topic_data.get('message_size_bytes')
                message_rate_hz = topic_data.get('message_rate_hz')

                topic = TopicNode(
                    id=topic_id,
                    name=topic_name,
                    qos=qos,
                    message_size_bytes=message_size_bytes,
                    message_rate_hz=message_rate_hz
                )
                model.add_topic(topic)
                
            except Exception as e:
                self.errors.append(f"Error building topic: {e}")
        
        self.logger.info(f"  Built {len(model.topics)} topics")
    
    def _build_brokers(self, model: GraphModel, brokers: List[Dict]):
        """Build broker nodes"""
        for broker_data in brokers:
            try:
                broker_id = broker_data.get('id', broker_data.get('name'))
                if not broker_id:
                    self.errors.append("Broker missing 'id' or 'name' field")
                    continue
                broker_name = broker_data.get('name')
                broker = BrokerNode(
                    id=broker_id,
                    name=broker_name,
                    broker_type=broker_data.get('broker_type', 'generic')
                )
                model.add_broker(broker)
                
            except Exception as e:
                self.errors.append(f"Error building broker: {e}")
        
        self.logger.info(f"  Built {len(model.brokers)} brokers")
    
    def _build_infrastructure_nodes(self, model: GraphModel, nodes: List[Dict]):
        """Build infrastructure nodes"""
        for node_data in nodes:
            try:
                node_id = node_data.get('id', node_data.get('name'))
                if not node_id:
                    self.errors.append("Infrastructure node missing 'id' or 'name' field")
                    continue
                node_name = node_data.get('name')
                node = InfrastructureNode(
                    id=node_id,
                    name=node_name,
                    node_type=node_data.get('node_type', 'compute')
                )
                model.add_node(node)
                
            except Exception as e:
                self.errors.append(f"Error building infrastructure node: {e}")
        
        self.logger.info(f"  Built {len(model.nodes)} infrastructure nodes")
    
    # =========================================================================
    # Edge Building Methods
    # =========================================================================
    
    def _get_edge_endpoints(self, edge: Dict) -> Tuple[Optional[str], Optional[str]]:
        """Extract source and target from edge with flexible field names"""
        source = edge.get('source', edge.get('from'))
        target = edge.get('target', edge.get('to'))
        return source, target
    
    def _build_publishes_edges(self, model: GraphModel, edges: Dict):
        """Build PUBLISHES_TO edges"""
        edge_list = edges.get('publishes', edges.get('publishes_to', []))
        
        for edge in edge_list:
            source, target = self._get_edge_endpoints(edge)
            if source and target:
                model.publishes_edges.append(PublishesEdge(
                    source=source,
                    target=target,
                    period_ms=edge.get('period_ms'),
                    msg_size_bytes=edge.get('msg_size_bytes')
                ))
            else:
                self.warnings.append(f"PUBLISHES edge missing source/target: {edge}")
        
        self.logger.info(f"  Built {len(model.publishes_edges)} PUBLISHES_TO edges")
    
    def _build_subscribes_edges(self, model: GraphModel, edges: Dict):
        """Build SUBSCRIBES_TO edges"""
        edge_list = edges.get('subscribes', edges.get('subscribes_to', []))
        
        for edge in edge_list:
            source, target = self._get_edge_endpoints(edge)
            if source and target:
                model.subscribes_edges.append(SubscribesEdge(
                    source=source,
                    target=target
                ))
            else:
                self.warnings.append(f"SUBSCRIBES edge missing source/target: {edge}")
        
        self.logger.info(f"  Built {len(model.subscribes_edges)} SUBSCRIBES_TO edges")
    
    def _build_routes_edges(self, model: GraphModel, edges: Dict):
        """Build ROUTES edges"""
        edge_list = edges.get('routes', [])
        
        for edge in edge_list:
            source, target = self._get_edge_endpoints(edge)
            if source and target:
                model.routes_edges.append(RoutesEdge(
                    source=source,
                    target=target
                ))
            else:
                self.warnings.append(f"ROUTES edge missing source/target: {edge}")
        
        self.logger.info(f"  Built {len(model.routes_edges)} ROUTES edges")
    
    def _build_runs_on_edges(self, model: GraphModel, edges: Dict):
        """Build RUNS_ON edges"""
        edge_list = edges.get('runs_on', [])
        
        for edge in edge_list:
            source, target = self._get_edge_endpoints(edge)
            if source and target:
                model.runs_on_edges.append(RunsOnEdge(
                    source=source,
                    target=target
                ))
            else:
                self.warnings.append(f"RUNS_ON edge missing source/target: {edge}")
        
        self.logger.info(f"  Built {len(model.runs_on_edges)} RUNS_ON edges")
    
    def _build_connects_to_edges(self, model: GraphModel, edges: Dict):
        """Build explicit CONNECTS_TO edges (physical topology)"""
        edge_list = edges.get('connects_to', [])
        
        for edge in edge_list:
            source, target = self._get_edge_endpoints(edge)
            if source and target:
                model.connects_to_edges.append(ConnectsToEdge(
                    source=source,
                    target=target
                ))
            else:
                self.warnings.append(f"CONNECTS_TO edge missing source/target: {edge}")
        
        if model.connects_to_edges:
            self.logger.info(
                f"  Built {len(model.connects_to_edges)} explicit CONNECTS_TO edges"
            )
    
    # =========================================================================
    # Unified Dependency Derivation
    # =========================================================================
    
    def _derive_all_dependencies(self, model: GraphModel):
        """
        Derive all DEPENDS_ON relationships across system layers.
        
        Derivation order matters for proper dependency propagation:
        1. APP_TO_APP: From topic subscription patterns
        2. APP_TO_BROKER: From topic routing relationships
        3. NODE_TO_NODE: From application dependencies across nodes
        4. NODE_TO_BROKER: From broker placement
        """
        self.logger.info("Deriving unified DEPENDS_ON relationships...")
        
        # Track counts for logging
        initial_count = len(model.depends_on_edges)
        
        # 1. Application-to-Application dependencies (from topic overlap)
        self._derive_app_to_app_dependencies(model)
        app_to_app_count = len(model.depends_on_edges) - initial_count
        
        # 2. Application-to-Broker dependencies (from routing)
        before_broker = len(model.depends_on_edges)
        self._derive_app_to_broker_dependencies(model)
        app_to_broker_count = len(model.depends_on_edges) - before_broker
        
        # 3. Node-to-Node dependencies (from application dependencies)
        before_node = len(model.depends_on_edges)
        self._derive_node_to_node_dependencies(model)
        node_to_node_count = len(model.depends_on_edges) - before_node
        
        # 4. Node-to-Broker dependencies (from broker placement)
        before_node_broker = len(model.depends_on_edges)
        self._derive_node_to_broker_dependencies(model)
        node_to_broker_count = len(model.depends_on_edges) - before_node_broker
        
        # Log summary
        total = len(model.depends_on_edges)
        self.logger.info(f"  Derived {app_to_app_count} APP_TO_APP dependencies")
        self.logger.info(f"  Derived {app_to_broker_count} APP_TO_BROKER dependencies")
        self.logger.info(f"  Derived {node_to_node_count} NODE_TO_NODE dependencies")
        self.logger.info(f"  Derived {node_to_broker_count} NODE_TO_BROKER dependencies")
        self.logger.info(f"  Total: {total} DEPENDS_ON relationships")
    
    def _derive_app_to_app_dependencies(self, model: GraphModel):
        """
        Derive APP_TO_APP dependencies from topic subscription patterns.
        
        Rule: If App_A subscribes to Topic_T and App_B publishes to Topic_T,
              then App_A DEPENDS_ON App_B (via Topic_T)
        
        Multiple topics between the same app pair are aggregated into a single
        DEPENDS_ON edge with increased weight.
        """
        self.logger.debug("Deriving APP_TO_APP dependencies...")
        
        # Map topics to their publishers and subscribers
        topic_publishers: Dict[str, List[str]] = defaultdict(list)
        topic_subscribers: Dict[str, List[str]] = defaultdict(list)
        
        for edge in model.publishes_edges:
            topic_publishers[edge.target].append(edge.source)
        
        for edge in model.subscribes_edges:
            topic_subscribers[edge.target].append(edge.source)
        
        # Track dependencies to aggregate multiple topic connections
        # Key: (subscriber, publisher), Value: list of topics
        dependency_topics: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        
        # Find all app-to-app dependencies through topics
        for topic, publishers in topic_publishers.items():
            subscribers = topic_subscribers.get(topic, [])
            
            for subscriber in subscribers:
                for publisher in publishers:
                    if subscriber != publisher:
                        # Subscriber depends on publisher via this topic
                        key = (subscriber, publisher)
                        dependency_topics[key].append(topic)
        
        # Create DEPENDS_ON edges with aggregated topic information
        for (subscriber, publisher), topics in dependency_topics.items():
            # Weight increases with number of shared topics
            weight = 1.0 + (len(topics) - 1) * 0.2  # +0.2 for each additional topic
            
            model.depends_on_edges.append(DependsOnEdge(
                source=subscriber,
                target=publisher,
                dependency_type=DependencyType.APP_TO_APP,
                derived_from=[f"SUBSCRIBES_TO({subscriber},{t})" for t in topics] +
                             [f"PUBLISHES_TO({publisher},{t})" for t in topics],
                topics=topics,
                weight=min(weight, 2.0)  # Cap at 2.0
            ))
    
    def _derive_app_to_broker_dependencies(self, model: GraphModel):
        """
        Derive APP_TO_BROKER dependencies from topic routing.
        
        Rule: If App uses Topic_T (publish or subscribe) and Broker_B routes Topic_T,
              then App DEPENDS_ON Broker_B (for topic routing)
        
        This captures the application's operational dependency on the broker
        infrastructure for message delivery.
        """
        self.logger.debug("Deriving APP_TO_BROKER dependencies...")
        
        # Map topics to their routing brokers
        topic_brokers: Dict[str, List[str]] = defaultdict(list)
        for edge in model.routes_edges:
            topic_brokers[edge.target].append(edge.source)
        
        # Map applications to topics they use
        app_topics: Dict[str, Set[str]] = defaultdict(set)
        
        for edge in model.publishes_edges:
            app_topics[edge.source].add(edge.target)
        
        for edge in model.subscribes_edges:
            app_topics[edge.source].add(edge.target)
        
        # Track unique app-broker dependencies
        app_broker_deps: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        
        # Find app-to-broker dependencies
        for app, topics in app_topics.items():
            for topic in topics:
                brokers = topic_brokers.get(topic, [])
                for broker in brokers:
                    key = (app, broker)
                    app_broker_deps[key].append(topic)
        
        # Create DEPENDS_ON edges
        for (app, broker), topics in app_broker_deps.items():
            weight = 1.0 + (len(topics) - 1) * 0.15  # Slightly lower weight per topic
            
            model.depends_on_edges.append(DependsOnEdge(
                source=app,
                target=broker,
                dependency_type=DependencyType.APP_TO_BROKER,
                derived_from=[f"ROUTES({broker},{t})" for t in topics],
                topics=topics,
                weight=min(weight, 1.8)
            ))
    
    def _derive_node_to_node_dependencies(self, model: GraphModel):
        """
        Derive NODE_TO_NODE dependencies from application dependencies.
        
        Rule: If App_A DEPENDS_ON App_B (APP_TO_APP)
              and App_A RUNS_ON Node_X
              and App_B RUNS_ON Node_Y
              and Node_X != Node_Y
              then Node_X DEPENDS_ON Node_Y
        
        The weight of the node dependency reflects the aggregated importance
        of the underlying application dependencies.
        """
        self.logger.debug("Deriving NODE_TO_NODE dependencies...")
        
        # Map applications/brokers to their infrastructure nodes
        component_to_node: Dict[str, str] = {}
        for edge in model.runs_on_edges:
            component_to_node[edge.source] = edge.target
        
        # Track node-to-node dependencies with their sources
        # Key: (source_node, target_node), Value: list of (source_app, target_app, weight)
        node_deps: Dict[Tuple[str, str], List[Tuple[str, str, float]]] = defaultdict(list)
        
        # Get APP_TO_APP dependencies
        app_deps = model.get_depends_on_by_type(DependencyType.APP_TO_APP)
        
        for dep in app_deps:
            source_node = component_to_node.get(dep.source)
            target_node = component_to_node.get(dep.target)
            
            if source_node and target_node and source_node != target_node:
                key = (source_node, target_node)
                node_deps[key].append((dep.source, dep.target, dep.weight))
        
        # Create NODE_TO_NODE DEPENDS_ON edges
        for (source_node, target_node), app_pairs in node_deps.items():
            # Aggregate weight based on number and importance of app dependencies
            total_weight = sum(w for _, _, w in app_pairs)
            normalized_weight = min(1.0 + (total_weight - 1) * 0.3, 3.0)
            
            # Build derived_from list
            derived_from = [
                f"APP_TO_APP({src},{tgt})" for src, tgt, _ in app_pairs
            ]
            
            model.depends_on_edges.append(DependsOnEdge(
                source=source_node,
                target=target_node,
                dependency_type=DependencyType.NODE_TO_NODE,
                derived_from=derived_from,
                topics=[],  # Node-level deps don't directly reference topics
                weight=normalized_weight
            ))
    
    def _derive_node_to_broker_dependencies(self, model: GraphModel):
        """
        Derive NODE_TO_BROKER dependencies from broker placement.
        
        Rule: If App DEPENDS_ON Broker (APP_TO_BROKER)
              and App RUNS_ON Node_X
              and Broker RUNS_ON Node_Y (or is accessible from Node_X)
              then Node_X DEPENDS_ON Broker
        
        This captures infrastructure-level dependency on broker services.
        """
        self.logger.debug("Deriving NODE_TO_BROKER dependencies...")
        
        # Map components to nodes
        component_to_node: Dict[str, str] = {}
        for edge in model.runs_on_edges:
            component_to_node[edge.source] = edge.target
        
        # Track node-to-broker dependencies
        node_broker_deps: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        
        # Get APP_TO_BROKER dependencies
        app_broker_deps = model.get_depends_on_by_type(DependencyType.APP_TO_BROKER)
        
        for dep in app_broker_deps:
            app_node = component_to_node.get(dep.source)
            
            if app_node and dep.target in model.brokers:
                key = (app_node, dep.target)
                node_broker_deps[key].append(dep.source)
        
        # Create NODE_TO_BROKER DEPENDS_ON edges
        for (node, broker), apps in node_broker_deps.items():
            weight = 1.0 + (len(apps) - 1) * 0.25
            
            model.depends_on_edges.append(DependsOnEdge(
                source=node,
                target=broker,
                dependency_type=DependencyType.NODE_TO_BROKER,
                derived_from=[f"APP_TO_BROKER({app},{broker})" for app in apps],
                topics=[],
                weight=min(weight, 2.5)
            ))
    
    # =========================================================================
    # Validation Methods
    # =========================================================================
    
    def validate(self, model: GraphModel) -> Tuple[bool, List[str], List[str]]:
        """
        Validate the graph model for consistency.
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check for orphan references in edges
        all_component_ids = set(
            list(model.applications.keys()) +
            list(model.topics.keys()) +
            list(model.brokers.keys()) +
            list(model.nodes.keys())
        )
        
        # Validate PUBLISHES_TO edges
        for edge in model.publishes_edges:
            if edge.source not in model.applications.keys():
                errors.append(f"PUBLISHES_TO source '{edge.source}' not found in applications")
            if edge.target not in model.topics.keys():
                errors.append(f"PUBLISHES_TO target '{edge.target}' not found in topics")
        
        # Validate SUBSCRIBES_TO edges
        for edge in model.subscribes_edges:
            if edge.source not in model.applications.keys():
                errors.append(f"SUBSCRIBES_TO source '{edge.source}' not found in applications")
            if edge.target not in model.topics:
                errors.append(f"SUBSCRIBES_TO target '{edge.target}' not found in topics")
        
        # Validate ROUTES edges
        for edge in model.routes_edges:
            if edge.source not in model.brokers:
                errors.append(f"ROUTES source '{edge.source}' not found in brokers")
            if edge.target not in model.topics:
                errors.append(f"ROUTES target '{edge.target}' not found in topics")
        
        # Validate RUNS_ON edges
        for edge in model.runs_on_edges:
            if edge.source not in model.applications.keys() and edge.source not in model.brokers:
                errors.append(f"RUNS_ON source '{edge.source}' not found in applications or brokers")
            if edge.target not in model.nodes:
                errors.append(f"RUNS_ON target '{edge.target}' not found in infrastructure nodes")
        
        # Check for topics without publishers or subscribers
        published_topics = {e.target for e in model.publishes_edges}
        subscribed_topics = {e.target for e in model.subscribes_edges}
        
        for topic_id in model.topics:
            if topic_id not in published_topics:
                warnings.append(f"Topic '{topic_id}' has no publishers")
            if topic_id not in subscribed_topics:
                warnings.append(f"Topic '{topic_id}' has no subscribers")
        
        # Check for isolated applications
        connected_apps = set()
        for e in model.publishes_edges:
            connected_apps.add(e.source)
        for e in model.subscribes_edges:
            connected_apps.add(e.source)
        
        for app_id in model.applications:
            if app_id not in connected_apps:
                warnings.append(f"Application '{app_id}' is isolated (no pub/sub connections)")
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings
    
    # =========================================================================
    # NetworkX Conversion
    # =========================================================================
    
    def to_networkx(self, model: GraphModel, include_derived: bool = True) -> nx.DiGraph:
        """
        Convert GraphModel to NetworkX directed graph.
        
        Args:
            model: The GraphModel to convert
            include_derived: Whether to include derived DEPENDS_ON edges
        
        Returns:
            NetworkX DiGraph
        """
        G = nx.DiGraph()
        
        # Add all nodes with attributes
        for app_id, app in model.applications.items():
            G.add_node(app_id, type='Application', **app.to_dict())
        
        for topic_id, topic in model.topics.items():
            G.add_node(topic_id, type='Topic', **topic.to_dict())
        
        for broker_id, broker in model.brokers.items():
            G.add_node(broker_id, type='Broker', **broker.to_dict())
        
        for node_id, node in model.nodes.items():
            G.add_node(node_id, type='Node', **node.to_dict())
        
        # Add explicit edges
        for edge in model.publishes_edges:
            G.add_edge(edge.source, edge.target, type='PUBLISHES_TO', **edge.to_dict())
        
        for edge in model.subscribes_edges:
            G.add_edge(edge.source, edge.target, type='SUBSCRIBES_TO', **edge.to_dict())
        
        for edge in model.routes_edges:
            G.add_edge(edge.source, edge.target, type='ROUTES', **edge.to_dict())
        
        for edge in model.runs_on_edges:
            G.add_edge(edge.source, edge.target, type='RUNS_ON', **edge.to_dict())
        
        for edge in model.connects_to_edges:
            G.add_edge(edge.source, edge.target, type='CONNECTS_TO', **edge.to_dict())
        
        # Add derived DEPENDS_ON edges
        if include_derived:
            for edge in model.depends_on_edges:
                # Get edge dict and remove keys we'll set explicitly
                edge_data = edge.to_dict()
                edge_data.pop('source', None)
                edge_data.pop('target', None)
                edge_data['dependency_type'] = edge.dependency_type.value  # Ensure string value
                
                G.add_edge(
                    edge.source, 
                    edge.target, 
                    type='DEPENDS_ON',
                    **edge_data
                )
        
        return G
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_build_report(self) -> Dict[str, Any]:
        """Get a report of the build process"""
        return {
            'errors': self.errors,
            'warnings': self.warnings,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'success': len(self.errors) == 0
        }