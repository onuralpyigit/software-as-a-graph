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
    Builds GraphModel instances from various data sources
    
    Supports:
    - JSON configuration files
    - CSV data files
    - Neo4j database exports
    - ROS2 DDS introspection (if available)
    - Direct Python dictionaries
    """
    
    def __init__(self):
        """Initialize the graph builder"""
        self.logger = logging.getLogger(__name__)
        self.model = None
        
    def build_from_json(self, filepath: str) -> GraphModel:
        """
        Build graph from JSON configuration file
        
        Expected JSON structure:
        {
            "applications": [...],
            "topics": [...],
            "brokers": [...],
            "nodes": [...],
            "edges": {
                "publishes": [...],
                "subscribes": [...],
                "routes": [...],
                "runs_on": [...],
                "connects_to": [...]
            }
        }
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            GraphModel instance
        """
        self.logger.info(f"Building graph from JSON: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return self.build_from_dict(data)
    
    def build_from_dict(self, data: Dict) -> GraphModel:
        """
        Build graph from Python dictionary
        
        Args:
            data: Dictionary with system configuration
        
        Returns:
            GraphModel instance
        """
        self.logger.info("Building graph from dictionary...")
        
        model = GraphModel()
        
        # Step 1: Add nodes
        if 'applications' in data:
            for app_data in data['applications']:
                app = self._create_application_from_dict(app_data)
                model.add_application(app)
        
        if 'topics' in data:
            for topic_data in data['topics']:
                topic = self._create_topic_from_dict(topic_data)
                model.add_topic(topic)
        
        if 'brokers' in data:
            for broker_data in data['brokers']:
                broker = self._create_broker_from_dict(broker_data)
                model.add_broker(broker)
        
        if 'nodes' in data:
            for node_data in data['nodes']:
                node = self._create_node_from_dict(node_data)
                model.add_node(node)
        
        # Step 2: Add edges
        if 'edges' in data:
            edges = data['edges']
            
            if 'publishes' in edges:
                for edge_data in edges['publishes']:
                    edge = self._create_publishes_edge(edge_data)
                    model.publishes_edges.append(edge)
            
            if 'subscribes' in edges:
                for edge_data in edges['subscribes']:
                    edge = self._create_subscribes_edge(edge_data)
                    model.subscribes_edges.append(edge)
            
            if 'routes' in edges:
                for edge_data in edges['routes']:
                    edge = self._create_routes_edge(edge_data)
                    model.routes_edges.append(edge)
            
            if 'runs_on' in edges:
                for edge_data in edges['runs_on']:
                    edge = self._create_runs_on_edge(edge_data)
                    model.runs_on_edges.append(edge)
            
            if 'connects_to' in edges:
                for edge_data in edges['connects_to']:
                    edge = self._create_connects_to_edge(edge_data)
                    model.connects_to_edges.append(edge)
        
        # Step 3: Derive DEPENDS_ON relationships
        self._derive_dependencies(model)
        
        summary = model.summary()
        self.logger.info(f"Graph built: {summary['total_nodes']} nodes, {summary['total_edges']} edges")
        
        return model
    
    def build_from_csv(self, 
                       nodes_file: str,
                       edges_file: str,
                       qos_file: Optional[str] = None) -> GraphModel:
        """
        Build graph from CSV files
        
        Args:
            nodes_file: CSV with columns: name, type, [properties...]
            edges_file: CSV with columns: source, target, type, [properties...]
            qos_file: Optional CSV with QoS policies for topics
        
        Returns:
            GraphModel instance
        """
        self.logger.info(f"Building graph from CSV: {nodes_file}, {edges_file}")
        
        model = GraphModel()
        
        # Load nodes
        nodes_data = self._load_csv(nodes_file)
        for node_data in nodes_data:
            node_type = node_data.get('type', 'Application')
            
            if node_type == 'Application':
                app = self._create_application_from_dict(node_data)
                model.add_application(app)
            elif node_type == 'Topic':
                topic = self._create_topic_from_dict(node_data)
                model.add_topic(topic)
            elif node_type == 'Broker':
                broker = self._create_broker_from_dict(node_data)
                model.add_broker(broker)
            elif node_type == 'Node':
                node = self._create_node_from_dict(node_data)
                model.add_node(node)
        
        # Load QoS policies if provided
        if qos_file and Path(qos_file).exists():
            qos_data = self._load_csv(qos_file)
            self._apply_qos_policies(model, qos_data)
        
        # Load edges
        edges_data = self._load_csv(edges_file)
        for edge_data in edges_data:
            edge_type = edge_data.get('type', 'PUBLISHES_TO')
            
            if edge_type == 'PUBLISHES_TO':
                edge = self._create_publishes_edge(edge_data)
                model.publishes_edges.append(edge)
            elif edge_type == 'SUBSCRIBES_TO':
                edge = self._create_subscribes_edge(edge_data)
                model.subscribes_edges.append(edge)
            elif edge_type == 'ROUTES':
                edge = self._create_routes_edge(edge_data)
                model.routes_edges.append(edge)
            elif edge_type == 'RUNS_ON':
                edge = self._create_runs_on_edge(edge_data)
                model.runs_on_edges.append(edge)
            elif edge_type == 'CONNECTS_TO':
                edge = self._create_connects_to_edge(edge_data)
                model.connects_to_edges.append(edge)
        
        # Derive dependencies
        self._derive_dependencies(model)
        
        return model
    
    def build_from_neo4j(self, 
                        uri: str,
                        auth: Tuple[str, str],
                        database: str = "neo4j") -> GraphModel:
        """
        Build graph from Neo4j database
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            auth: Tuple of (username, password)
            database: Database name
        
        Returns:
            GraphModel instance
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("neo4j package required. Install with: pip install neo4j")
        
        self.logger.info(f"Building graph from Neo4j: {uri}")
        
        driver = GraphDatabase.driver(uri, auth=auth)
        model = GraphModel()
        
        with driver.session(database=database) as session:
            # Load nodes
            result = session.run("MATCH (n) RETURN n")
            for record in result:
                node = record['n']
                node_data = dict(node)
                node_type = node_data.get('type', 'Application')
                
                if node_type == 'Application':
                    app = self._create_application_from_dict(node_data)
                    model.add_application(app)
                elif node_type == 'Topic':
                    topic = self._create_topic_from_dict(node_data)
                    model.add_topic(topic)
                elif node_type == 'Broker':
                    broker = self._create_broker_from_dict(node_data)
                    model.add_broker(broker)
                elif node_type == 'Node':
                    node_obj = self._create_node_from_dict(node_data)
                    model.add_node(node_obj)
            
            # Load edges
            result = session.run("""
                MATCH (source)-[r]->(target)
                RETURN source.name as source, target.name as target, 
                       type(r) as type, properties(r) as props
            """)
            
            for record in result:
                edge_data = {
                    'source': record['source'],
                    'target': record['target'],
                    'type': record['type'],
                    **dict(record['props'])
                }
                
                edge_type = edge_data['type']
                
                if edge_type == 'PUBLISHES_TO':
                    edge = self._create_publishes_edge(edge_data)
                    model.publishes_edges.append(edge)
                elif edge_type == 'SUBSCRIBES_TO':
                    edge = self._create_subscribes_edge(edge_data)
                    model.subscribes_edges.append(edge)
                elif edge_type == 'ROUTES':
                    edge = self._create_routes_edge(edge_data)
                    model.routes_edges.append(edge)
                elif edge_type == 'RUNS_ON':
                    edge = self._create_runs_on_edge(edge_data)
                    model.runs_on_edges.append(edge)
                elif edge_type == 'CONNECTS_TO':
                    edge = self._create_connects_to_edge(edge_data)
                    model.connects_to_edges.append(edge)
        
        driver.close()
        
        # Derive dependencies
        self._derive_dependencies(model)
        
        return model
    
    def build_from_networkx(self, graph: nx.DiGraph) -> GraphModel:
        """
        Build GraphModel from NetworkX DiGraph
        
        Args:
            graph: NetworkX directed graph with node/edge attributes
        
        Returns:
            GraphModel instance
        """
        self.logger.info("Building graph from NetworkX...")
        
        model = GraphModel()
        
        # Add nodes
        for node_name, node_data in graph.nodes(data=True):
            node_type = node_data.get('type', 'Application')
            
            if node_type == 'Application':
                app = self._create_application_from_dict({**node_data, 'name': node_name})
                model.add_application(app)
            elif node_type == 'Topic':
                topic = self._create_topic_from_dict({**node_data, 'name': node_name})
                model.add_topic(topic)
            elif node_type == 'Broker':
                broker = self._create_broker_from_dict({**node_data, 'name': node_name})
                model.add_broker(broker)
            elif node_type == 'Node':
                node = self._create_node_from_dict({**node_data, 'name': node_name})
                model.add_node(node)
        
        # Add edges
        for source, target, edge_data in graph.edges(data=True):
            edge_type = edge_data.get('type', 'PUBLISHES_TO')
            edge_dict = {**edge_data, 'source': source, 'target': target}
            
            if edge_type == 'PUBLISHES_TO':
                edge = self._create_publishes_edge(edge_dict)
                model.publishes_edges.append(edge)
            elif edge_type == 'SUBSCRIBES_TO':
                edge = self._create_subscribes_edge(edge_dict)
                model.subscribes_edges.append(edge)
            elif edge_type == 'ROUTES':
                edge = self._create_routes_edge(edge_dict)
                model.routes_edges.append(edge)
            elif edge_type == 'RUNS_ON':
                edge = self._create_runs_on_edge(edge_dict)
                model.runs_on_edges.append(edge)
            elif edge_type == 'CONNECTS_TO':
                edge = self._create_connects_to_edge(edge_dict)
                model.connects_to_edges.append(edge)
            elif edge_type == 'DEPENDS_ON':
                edge = self._create_depends_on_edge(edge_dict)
                model.depends_on_edges.append(edge)
        
        return model
    
    def _create_application_from_dict(self, data: Dict) -> ApplicationNode:
        """Create ApplicationNode from dictionary"""
        
        # Parse application type
        app_type_str = data.get('app_type', 'Prosumer')
        if isinstance(app_type_str, str):
            app_type = ApplicationType[app_type_str.upper()] if app_type_str.upper() in ApplicationType.__members__ else ApplicationType.PROSUMER
        else:
            app_type = ApplicationType.PROSUMER
        
        return ApplicationNode(
            name=data.get('name', ''),
            app_type=app_type,
            node_host=data.get('node_host'),
            required_latency_ms=self._parse_float(data.get('required_latency_ms')),
            required_throughput_msgs_per_sec=self._parse_float(data.get('required_throughput_msgs_per_sec')),
            cpu_cores=self._parse_float(data.get('cpu_cores', 1.0)),
            memory_mb=self._parse_float(data.get('memory_mb', 512.0)),
            actual_latency_ms=self._parse_float(data.get('actual_latency_ms')),
            actual_throughput=self._parse_float(data.get('actual_throughput')),
            health_score=self._parse_float(data.get('health_score', 1.0)),
            business_domain=data.get('business_domain'),
            team_owner=data.get('team_owner'),
            version=data.get('version'),
            last_updated=self._parse_datetime(data.get('last_updated'))
        )
    
    def _create_topic_from_dict(self, data: Dict) -> TopicNode:
        """Create TopicNode from dictionary"""
        
        # Parse QoS policy
        qos_policy = self._parse_qos_policy(data)
        
        return TopicNode(
            name=data.get('name', ''),
            broker=data.get('broker'),
            qos_policy=qos_policy,
            message_rate_per_sec=self._parse_float(data.get('message_rate_per_sec', 0.0)),
            avg_message_size_bytes=self._parse_float(data.get('avg_message_size_bytes', 0.0)),
            peak_message_rate=self._parse_float(data.get('peak_message_rate', 0.0)),
            message_type=data.get('message_type'),
            schema_version=data.get('schema_version'),
            description=data.get('description'),
            created_at=self._parse_datetime(data.get('created_at'))
        )
    
    def _create_broker_from_dict(self, data: Dict) -> BrokerNode:
        """Create BrokerNode from dictionary"""
        
        return BrokerNode(
            name=data.get('name', ''),
            node_host=data.get('node_host'),
            max_connections=self._parse_int(data.get('max_connections', 1000)),
            max_throughput_msgs_per_sec=self._parse_float(data.get('max_throughput_msgs_per_sec', 10000.0)),
            replication_factor=self._parse_int(data.get('replication_factor', 1)),
            partition_count=self._parse_int(data.get('partition_count', 1)),
            current_connections=self._parse_int(data.get('current_connections', 0)),
            current_throughput=self._parse_float(data.get('current_throughput', 0.0)),
            avg_latency_ms=self._parse_float(data.get('avg_latency_ms', 0.0)),
            cpu_utilization=self._parse_float(data.get('cpu_utilization', 0.0)),
            memory_utilization=self._parse_float(data.get('memory_utilization', 0.0)),
            health_score=self._parse_float(data.get('health_score', 1.0)),
            uptime_seconds=self._parse_float(data.get('uptime_seconds', 0.0)),
            broker_type=data.get('broker_type', 'Generic'),
            version=data.get('version')
        )
    
    def _create_node_from_dict(self, data: Dict) -> InfrastructureNode:
        """Create InfrastructureNode from dictionary"""
        
        return InfrastructureNode(
            name=data.get('name', ''),
            datacenter=data.get('datacenter'),
            rack=data.get('rack'),
            zone=data.get('zone'),
            region=data.get('region'),
            total_cpu_cores=self._parse_float(data.get('total_cpu_cores', 4.0)),
            total_memory_mb=self._parse_float(data.get('total_memory_mb', 8192.0)),
            total_disk_gb=self._parse_float(data.get('total_disk_gb', 100.0)),
            network_bandwidth_mbps=self._parse_float(data.get('network_bandwidth_mbps', 1000.0)),
            cpu_utilization=self._parse_float(data.get('cpu_utilization', 0.0)),
            memory_utilization=self._parse_float(data.get('memory_utilization', 0.0)),
            disk_utilization=self._parse_float(data.get('disk_utilization', 0.0)),
            network_utilization=self._parse_float(data.get('network_utilization', 0.0)),
            health_score=self._parse_float(data.get('health_score', 1.0)),
            uptime_seconds=self._parse_float(data.get('uptime_seconds', 0.0)),
            node_type=data.get('node_type', 'VM'),
            os=data.get('os'),
            ip_address=data.get('ip_address')
        )
    
    def _parse_qos_policy(self, data: Dict) -> QoSPolicy:
        """Parse QoS policy from dictionary"""
        
        # Parse durability
        durability_str = data.get('durability', 'VOLATILE')
        if isinstance(durability_str, str):
            durability = QoSDurability[durability_str] if durability_str in QoSDurability.__members__ else QoSDurability.VOLATILE
        else:
            durability = QoSDurability.VOLATILE
        
        # Parse reliability
        reliability_str = data.get('reliability', 'BEST_EFFORT')
        if isinstance(reliability_str, str):
            reliability = QoSReliability[reliability_str] if reliability_str in QoSReliability.__members__ else QoSReliability.BEST_EFFORT
        else:
            reliability = QoSReliability.BEST_EFFORT
        
        return QoSPolicy(
            durability=durability,
            reliability=reliability,
            deadline_ms=self._parse_float(data.get('deadline_ms')),
            lifespan_ms=self._parse_float(data.get('lifespan_ms')),
            transport_priority=self._parse_int(data.get('transport_priority', 0)),
            history_depth=self._parse_int(data.get('history_depth', 1))
        )
    
    def _create_publishes_edge(self, data: Dict) -> PublishesEdge:
        """Create PublishesEdge from dictionary"""
        
        pattern_str = data.get('message_pattern', 'EVENT_DRIVEN')
        if isinstance(pattern_str, str):
            pattern = MessagePattern[pattern_str] if pattern_str in MessagePattern.__members__ else MessagePattern.EVENT_DRIVEN
        else:
            pattern = MessagePattern.EVENT_DRIVEN
        
        return PublishesEdge(
            source=data.get('source', ''),
            target=data.get('target', ''),
            message_pattern=pattern,
            message_rate_per_sec=self._parse_float(data.get('message_rate_per_sec', 0.0)),
            is_synchronous=self._parse_bool(data.get('is_synchronous', False)),
            timeout_ms=self._parse_float(data.get('timeout_ms')),
            weight=self._parse_float(data.get('weight', 1.0))
        )
    
    def _create_subscribes_edge(self, data: Dict) -> SubscribesEdge:
        """Create SubscribesEdge from dictionary"""
        
        return SubscribesEdge(
            source=data.get('source', ''),
            target=data.get('target', ''),
            filter_expression=data.get('filter_expression'),
            qos_compatible=self._parse_bool(data.get('qos_compatible', True)),
            acknowledgment_mode=data.get('acknowledgment_mode', 'AUTO'),
            weight=self._parse_float(data.get('weight', 1.0))
        )
    
    def _create_routes_edge(self, data: Dict) -> RoutesEdge:
        """Create RoutesEdge from dictionary"""
        
        return RoutesEdge(
            source=data.get('source', ''),
            target=data.get('target', ''),
            routing_weight=self._parse_float(data.get('routing_weight', 1.0)),
            partition_count=self._parse_int(data.get('partition_count', 1)),
            weight=self._parse_float(data.get('weight', 1.0))
        )
    
    def _create_runs_on_edge(self, data: Dict) -> RunsOnEdge:
        """Create RunsOnEdge from dictionary"""
        
        resource_allocation = data.get('resource_allocation', {})
        if isinstance(resource_allocation, str):
            try:
                resource_allocation = eval(resource_allocation)
            except:
                resource_allocation = {}
        
        return RunsOnEdge(
            source=data.get('source', ''),
            target=data.get('target', ''),
            resource_allocation=resource_allocation,
            weight=self._parse_float(data.get('weight', 1.0))
        )
    
    def _create_connects_to_edge(self, data: Dict) -> ConnectsToEdge:
        """Create ConnectsToEdge from dictionary"""
        
        return ConnectsToEdge(
            source=data.get('source', ''),
            target=data.get('target', ''),
            bandwidth_mbps=self._parse_float(data.get('bandwidth_mbps', 1000.0)),
            latency_ms=self._parse_float(data.get('latency_ms', 1.0)),
            packet_loss_rate=self._parse_float(data.get('packet_loss_rate', 0.0)),
            is_redundant=self._parse_bool(data.get('is_redundant', False)),
            weight=self._parse_float(data.get('weight', 1.0))
        )
    
    def _create_depends_on_edge(self, data: Dict) -> DependsOnEdge:
        """Create DependsOnEdge from dictionary"""
        
        return DependsOnEdge(
            source=data.get('source', ''),
            target=data.get('target', ''),
            dependency_type=data.get('dependency_type', 'FUNCTIONAL'),
            strength=self._parse_float(data.get('strength', 1.0)),
            is_critical=self._parse_bool(data.get('is_critical', False)),
            weight=self._parse_float(data.get('weight', 1.0))
        )
    
    def _derive_dependencies(self, model: GraphModel):
        """
        Derive DEPENDS_ON relationships from PUBLISHES_TO and SUBSCRIBES_TO
        
        Logic:
        - If App1 publishes to Topic T and App2 subscribes to T, then App2 DEPENDS_ON App1
        - Applications depend on brokers that route their topics
        - Applications depend on nodes they run on
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
        
        # Derive application-to-broker dependencies
        app_topics: Dict[str, Set[str]] = {}
        
        for edge in model.publishes_edges + model.subscribes_edges:
            if edge.source not in app_topics:
                app_topics[edge.source] = set()
            app_topics[edge.source].add(edge.target)
        
        for edge in model.routes_edges:
            broker = edge.source
            topic = edge.target
            
            # Find all apps using this topic
            for app, topics in app_topics.items():
                if topic in topics:
                    model.depends_on_edges.append(DependsOnEdge(
                        source=app,
                        target=broker,
                        dependency_type='FUNCTIONAL',
                        strength=0.9
                    ))
        
        self.logger.info(f"Derived {len(model.depends_on_edges)} DEPENDS_ON relationships")
    
    def _load_csv(self, filepath: str) -> List[Dict]:
        """Load CSV file into list of dictionaries"""
        data = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data
    
    def _apply_qos_policies(self, model: GraphModel, qos_data: List[Dict]):
        """Apply QoS policies from CSV to topics"""
        for qos_row in qos_data:
            topic_name = qos_row.get('topic')
            if topic_name and topic_name in model.topics:
                topic = model.topics[topic_name]
                topic.qos_policy = self._parse_qos_policy(qos_row)
    
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
    
    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Safely parse datetime value"""
        if value is None or value == '':
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except (ValueError, TypeError):
            return None


def create_example_json_config() -> Dict:
    """
    Create an example JSON configuration for documentation
    
    Returns:
        Example configuration dictionary
    """
    return {
        "applications": [
            {
                "name": "OrderService",
                "app_type": "Prosumer",
                "node_host": "node1",
                "required_latency_ms": 100,
                "cpu_cores": 2.0,
                "memory_mb": 2048,
                "business_domain": "Commerce"
            },
            {
                "name": "InventoryService",
                "app_type": "Prosumer",
                "node_host": "node1",
                "required_latency_ms": 50,
                "cpu_cores": 1.5,
                "memory_mb": 1024,
                "business_domain": "Inventory"
            }
        ],
        "topics": [
            {
                "name": "orders.created",
                "broker": "broker1",
                "durability": "PERSISTENT",
                "reliability": "RELIABLE",
                "deadline_ms": 1000,
                "transport_priority": 90,
                "message_rate_per_sec": 100.0
            }
        ],
        "brokers": [
            {
                "name": "broker1",
                "node_host": "node1",
                "max_throughput_msgs_per_sec": 10000,
                "broker_type": "Kafka"
            }
        ],
        "nodes": [
            {
                "name": "node1",
                "datacenter": "DC1",
                "zone": "us-east-1a",
                "total_cpu_cores": 8,
                "total_memory_mb": 16384
            }
        ],
        "edges": {
            "publishes": [
                {
                    "source": "OrderService",
                    "target": "orders.created",
                    "message_pattern": "EVENT_DRIVEN",
                    "message_rate_per_sec": 100.0
                }
            ],
            "subscribes": [
                {
                    "source": "InventoryService",
                    "target": "orders.created"
                }
            ],
            "routes": [
                {
                    "source": "broker1",
                    "target": "orders.created"
                }
            ],
            "runs_on": [
                {
                    "source": "OrderService",
                    "target": "node1"
                },
                {
                    "source": "InventoryService",
                    "target": "node1"
                },
                {
                    "source": "broker1",
                    "target": "node1"
                }
            ],
            "connects_to": []
        }
    }
