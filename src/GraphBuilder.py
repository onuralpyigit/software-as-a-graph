from neo4j import GraphDatabase
import numpy as np
import pandas as pd
import math
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from src.DatasetGenerator import DatasetGenerator

class GraphBuilder:
    def __init__(self, uri: str, user: str, password: str):
        """Initialize enhanced graph builder with Neo4j connection"""        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.rng = np.random.default_rng(12345)
        self.generator = DatasetGenerator()
        self.validation_errors = []
        
    def close(self):
        """Close database connection"""
        self.driver.close()
    
    def execute_cypher(self, query: str, parameters: Dict = None):
        """Execute Cypher query with parameters"""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def clear_graph(self):
        """Clear all nodes and relationships from the graph"""
        self.execute_cypher("MATCH (n) DETACH DELETE n")
        print("Graph cleared successfully")

    def create_constraints(self):
        """Create all necessary constraints and indexes"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Application) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Broker) REQUIRE b.id IS UNIQUE", 
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX IF NOT EXISTS FOR (a:Application) ON (a.name)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Application) ON (a.type)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Application) ON (a.criticality_score)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.name)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.criticality_score)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.status)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.zone)",
            
            # Composite indexes
            "CREATE INDEX IF NOT EXISTS FOR (a:Application) ON (a.type, a.criticality_score)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.durability, t.reliability)"
        ]
        
        for constraint in constraints:
            try:
                self.execute_cypher(constraint)
            except Exception as e:
                # Index might already exist
                pass
        
        print("✓ Constraints and indexes created")    

    def create_node_with_properties(self, node_name: str):
        """Create a compute node with enhanced properties"""
        if node_name is None:
            raise ValueError("Node name cannot be None")
            
        properties = {
            "name": node_name,
            "type": "Node",
            "status": "active",
            "cpu_capacity": float(self.rng.integers(4, 32)),
            "memory_gb": float(self.rng.integers(8, 128)),
            "network_bandwidth_mbps": float(self.rng.integers(100, 10000)),
            "created_at": datetime.now().isoformat()
        }
        
        self.execute_cypher("""
        MERGE (node:Node {name: $name})
        SET node += $properties
        """, {"name": node_name, "properties": properties})

    def create_nodes(self, num_nodes):
        # Create computing nodes with status
        self.execute_cypher("""
        UNWIND range(1, $num_nodes) AS id
        CREATE (:Node {
            id: 'N' + id,
            name: 'Node' + id,
            type: 'Node'
        });
        """, {"num_nodes": num_nodes})

        for node_id in range(1, num_nodes + 1):
            node_name = f'Node{node_id}'
            self.create_node_with_properties(node_name)

    def create_broker_with_properties(self, broker_name: str):
        """Create a broker with enhanced properties"""
        if broker_name is None:
            raise ValueError("Broker name cannot be None")
            
        properties = {
            "name": broker_name,
            "type": "Broker",
            "status": "active",
            "max_throughput_mbps": float(self.rng.integers(100, 10000)),
            "max_connections": int(self.rng.integers(1000, 100000)),
            "queue_capacity_mb": float(self.rng.integers(100, 10000)),
            "protocol": self.rng.choice(["AMQP", "MQTT", "DDS", "Kafka"]),
            "created_at": datetime.now().isoformat()
        }
        
        self.execute_cypher("""
        MERGE (broker:Broker {name: $name})
        SET broker += $properties
        """, {"name": broker_name, "properties": properties})

    def create_brokers(self, num_brokers, num_topics):
        self.execute_cypher("""
        UNWIND range(1, $num_brokers) AS id
        WITH id, 'Node' + id as nodeName
        MATCH (n:Node {name: nodeName}) // Assign brokers to the first nodes
        CREATE (b:Broker {
            id: 'B' + id,
            name: 'Broker' + id,
            type: 'Broker'
        })-[:RUNS_ON]->(n);
        """, {"num_brokers": num_brokers}) 

        for broker_id in range(1, num_brokers + 1):
            broker_name = f'Broker{broker_id}'
            self.create_broker_with_properties(broker_name)

        # Create ROUTES relationships for brokers and topics
        self.execute_cypher("""
        UNWIND range(1, $num_topics) AS topicId
        WITH 'Topic' + topicId as topicName, 'Broker' + toInteger(1 + rand() * $num_brokers) AS brokerName
        MATCH (b:Broker WHERE b.name = brokerName), (t:Topic WHERE t.name = topicName)
        MERGE (b)-[:ROUTES]->(t);
        """, {"num_topics": num_topics, "num_brokers": num_brokers})     

    def create_application_with_properites(self, app_name: str):
        if app_name is None:
            raise ValueError("Application name cannot be None")
        
        properties = {
            "name": app_name,
            "type": "Application",
            "criticality_score": float(self.rng.uniform(0.1, 1.0)),
            "business_impact": self.rng.choice(["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"]),
            "avg_cpu_usage": float(self.rng.uniform(10, 80)),
            "avg_memory_usage": float(self.rng.uniform(20, 90)),
            "created_at": datetime.now().isoformat()
        }
        
        self.execute_cypher("""
        MERGE (app:Application {name: $name})
        SET app += $properties
        """, {"name": app_name, "properties": properties})

    def create_applications(self, num_apps, num_nodes):
        self.execute_cypher("""
            UNWIND range(1, $num_apps) AS id
            WITH id, 'Node' + toInteger(1 + rand() * $num_nodes) AS nodeName
            MATCH (n:Node {name: nodeName})
            CREATE (a:Application {
                id: 'A' + id,
                name: 'App' + id,
                type: 'Application'
            })-[:RUNS_ON]->(n);
            """, {"num_apps": num_apps, "num_nodes": num_nodes})
        
        for app_id in range(1, num_apps + 1):
            app_name = f'App{app_id}'
            self.create_application_with_properites(app_name)

    def create_topic_with_properties(self, topic_name: str):
        properties = {
            "name": topic_name,
            "type": "Topic",
            "durability": self.rng.choice(["VOLATILE", "TRANSIENT_LOCAL", "TRANSIENT", "PERSISTENT"]),
            "reliability": self.rng.choice(["BEST_EFFORT", "RELIABLE"]),
            "transport_priority": self.rng.choice(["LOW", "MEDIUM", "HIGH", "URGENT"]),
            "deadline_ms": float(self.rng.integers(10, 1000)),
            "lifespan_ms": float(self.rng.integers(1000, 60000)),
            "history_depth": int(self.rng.integers(1, 100)),
            "partition_count": int(self.rng.integers(1, 16)),
            "replication_factor": int(self.rng.integers(1, 5)),
            "created_at": datetime.now().isoformat()
        }
        
        self.execute_cypher("""
            MERGE (topic:Topic {name: $name})
            SET topic += $properties
            """, {"name": topic_name, "properties": properties})
        
    def create_topics(self, num_topics, num_apps):
        # Create topics
        self.execute_cypher("""
        UNWIND range(1, $num_topics) AS id
        CREATE (:Topic {
            id: 'T' + id,
            name: 'Topic' + id,
            type: 'Topic'
        });
        """, {"num_topics": num_topics})

        # Set topic properties
        for topic_id in range(1, num_topics + 1):
            topic_name = f'Topic{topic_id}'
            self.create_topic_with_properties(topic_name)

        # Calculate reasonable publishing range based on system size
        min_pub = self.num_min_pub_apps
        max_pub = self.num_max_pub_apps

        # Calculate reasonable subscription range based on system size
        min_sub = self.num_min_sub_apps
        max_sub = self.num_max_sub_apps

        # Define PUBLISHES_TO relationships
        for topic_id in range(1, num_topics + 1):
            topic_name = f'Topic{topic_id}'
            num_publishers = random.randint(min_pub, max_pub)

            for _ in range(num_publishers):
                app_index = random.randint(1, num_apps)
                app_name = f'App{app_index}'

                # Create relationship between App and Topic
                self.execute_cypher("""
                MATCH (a:Application {name: $appName}), (t:Topic {name: $topicName})
                MERGE (a)-[r:PUBLISHES_TO]->(t)
                """, {
                    "appName": app_name,
                    "topicName": topic_name
                })

        # Create SUBSCRIBES_TO relationships
        for topic_id in range(1, num_topics + 1):
            topic_name = f'Topic{topic_id}'
            num_subscribers = random.randint(min_sub, max_sub)

            for _ in range(num_subscribers):
                app_index = random.randint(1, num_apps)
                app_name = f'App{app_index}'

                # Create relationship between App and Topic
                self.execute_cypher("""
                MATCH (a:Application {name: $appName}), (t:Topic {name: $topicName})
                MERGE (a)-[:SUBSCRIBES_TO]->(t);
                """, {
                    "appName": app_name,
                    "topicName": topic_name
                })

    def add_time_series_properties(self):
        # Simulate historical metrics
        self.execute_cypher("""
        MATCH (a:Application)
        SET a.cpu_usage_history = [x IN range(0, 23) | toFloat(rand() * 80 + 10)],
            a.memory_usage_history = [x IN range(0, 23) | toFloat(rand() * 70 + 20)],
            a.error_rate_history = [x IN range(0, 23) | toFloat(rand() * 0.05)],
            a.request_rate_history = [x IN range(0, 23) | toFloat(rand() * 1000)]
        """)

    def build_graph_from_ros2_data_file(self, data_file):
        self.clear_graph()

        # Load JSON data from file
        with open(data_file, 'r') as file:
            json_data = json.load(file)
        
        # Import graph from JSON data
        self.import_graph_from_ros2_json(json_data)

        # Derive relationships
        self.derive_relationships()

    def import_graph_from_ros2_json(self, json_data):
        # Create applications and related topics directly
        for app in json_data['nodes']:
            self.create_application_from_ros2_json(app) 

        if json_data.get('infrastructure') is not None:
            # Create nodes based on the infrastructure data
            nodes = json_data['infrastructure']['nodes']
            for node in nodes:
                self.create_nodes_from_ros2_json(node)

    def create_application_from_ros2_json(self, app):
        app_name = app['node_name']
        self.create_application_with_properites(app_name)
        self.handle_publishers(app, app_name)
        self.handle_subscribers(app, app_name)

    def handle_publishers(self, app, app_name):
        publishers = app.get('publishers', [])
        for publisher in publishers:
            msg_size = 0
            if publisher.get('msg_size') is not None:
                msg_size = publisher.get('msg_size') * 128
            else:
                msg_size = random.randint(128, 65536)

            topic_name = publisher['topic_name']
            qos = publisher.get('qos') or {}
            durability = qos.get('durability', "VOLATILE")
            reliability = qos.get('reliability', "BEST_EFFORT")
            self.execute_cypher("""
            MERGE (topic:Topic {name: $name, durability: $durability, reliability: $reliability})
            """, {"name": topic_name, "durability": durability, "reliability": reliability})

            self.execute_cypher("""
            MATCH (t:Topic {name: $topic_name}), (a:Application {name: $app_name})
            MERGE (a)-[r:PUBLISHES_TO]->(t)
            SET r.msg_type = $msg_type, r.msg_size = $msg_size, r.period_ms = $period_ms
            """, {
                "topic_name": topic_name,
                "app_name": app_name,
                "msg_type": publisher['msg_type'],
                "period_ms": publisher.get('period_ms'),
                "msg_size": msg_size
            })

    def handle_subscribers(self, app, app_name):
        subscribers = app.get('subscribers', [])
        for subscriber in subscribers:
            topic_name = subscriber['topic_name']
            qos = subscriber.get('qos') or {}
            durability = qos.get('durability', "VOLATILE")
            reliability = qos.get('reliability', "BEST_EFFORT")
            self.execute_cypher("""
            MERGE (topic:Topic {name: $name, durability: $durability, reliability: $reliability})
            """, {"name": topic_name, "durability": durability, "reliability": reliability})

            self.execute_cypher("""
            MATCH (t:Topic {name: $topic_name}), (a:Application {name: $app_name})
            MERGE (a)-[r:SUBSCRIBES_TO]->(t)
            SET r.msg_type = $msg_type
            """, {
                "topic_name": topic_name,
                "app_name": app_name,
                "msg_type": subscriber['msg_type']
            })

    def create_nodes_from_ros2_json(self, node):
        node_name = node['node_id']
        self.create_node_with_properties(node_name)

        for app_name in node['services']:
            self.execute_cypher("""
            MATCH (n:Node {name: $node_name}), (a:Application {name: $app_name})
            MERGE (a)-[:RUNS_ON]->(n);
            """, {"node_name": node_name, "app_name": app_name})

    def create_broker_from_json(self, broker):
        broker_name = broker['broker_id']
        self.create_broker_with_properties(broker_name)

        for topic_name in broker['topics']:
            self.execute_cypher("""
            MATCH (b:Broker {name: $broker_name}), (t:Topic {name: $topic_name})
            MERGE (b)-[:ROUTES]->(t);
            """, {"broker_name": broker_name, "topic_name": topic_name})

    def create_runs_on_relationships_randomly(self):
        node_names = self.get_node_names()
        app_names = self.get_application_names()

        for app_name in app_names:
            # Randomly choose a node to run the application on
            random_node_name = random.choice(node_names)

            self.execute_cypher("""
            MATCH (n:Node {name: $node_name}), (a:Application {name: $app_name})
            MERGE (a)-[:RUNS_ON]->(n);
            """, {"node_name": random_node_name, "app_name": app_name})

    def get_node_names(self):
        nodes = self.execute_cypher("MATCH (n:Node) RETURN n.name as node_name;")
        node_name_list = [record["node_name"] for record in nodes]
        return node_name_list
    
    def get_application_names(self):
        apps = self.execute_cypher("MATCH (a:Application) RETURN a.name as app_name;")
        app_name_list = [record["app_name"] for record in apps]
        return app_name_list

    def create_nodes_and_relationships(self, num_nodes, num_apps, num_topics, num_brokers):
        # Create nodes
        self.create_nodes(num_nodes)

        # Create applications
        self.create_applications(num_apps, num_nodes)

        # Create topics
        self.create_topics(num_topics, num_apps)

        # Create brokers with status and RUNS_ON relationships if num_brokers > 0
        if num_brokers > 0:
            self.create_brokers(num_brokers, num_topics)

        # Add time series data
        self.add_time_series_properties()

    def print_graph(self):
        self.print_nodes_and_relationships()
        self.print_topics()
        self.print_applications_and_relationships()
        self.print_brokers_and_relationships()
        self.print_derived_relationships()

    def print_nodes_and_relationships(self):
        nodes = self.execute_cypher("MATCH (n:Node) RETURN n.name as node_name;")
        node_name_list = [record["node_name"] for record in nodes]
        print("Nodes: ", node_name_list, "; Total nodes: ", len(node_name_list))
        
        for node_name in node_name_list:
            self.print_runs_on_relationships(node_name)

    def print_runs_on_relationships(self, node_name):
        app_name_list = self.execute_cypher("""
        MATCH (n:Node {name: $node_name})<-[:RUNS_ON]-(a:Application)
        RETURN a.name AS app_name
        """, {"node_name": node_name})
        app_name_list = [record["app_name"] for record in app_name_list]
        print("Node : ", node_name, " runs applications: ", app_name_list, "; Total apps: ", len(app_name_list))

    def print_applications_and_relationships(self):
        apps = self.execute_cypher("MATCH (a:Application) RETURN a.name as app_name;")
        app_name_list = [record["app_name"] for record in apps]
        print("Apps: ", app_name_list, "; Total apps: ", len(app_name_list))

        for app_name in app_name_list:
            self.print_publish_to_relationships(app_name)
            self.print_subscribe_to_relationships(app_name)

    def print_publish_to_relationships(self, app_name):
        topic_name_list = self.execute_cypher("""
        MATCH (a:Application {name: $app_name})-[r:PUBLISHES_TO]->(t:Topic)
        RETURN t.name AS topic_name
        """, {"app_name": app_name})
        topic_name_list = [record["topic_name"] for record in topic_name_list]
        print("App: ", app_name, " publishes topics: ", topic_name_list, "; Total topics: ", len(topic_name_list))

    def print_subscribe_to_relationships(self, app_name):
        topic_name_list = self.execute_cypher("""
        MATCH (a:Application {name: $app_name})-[r:SUBSCRIBES_TO]->(t:Topic)
        RETURN t.name AS topic_name
        """, {"app_name": app_name})
        topic_name_list = [record["topic_name"] for record in topic_name_list]
        print("App name: ", app_name, " subscribes to topics: ", topic_name_list, "; Total topics: ", len(topic_name_list))

    def print_topics(self):
        topics = self.execute_cypher("MATCH (t:Topic) RETURN t.name as topic_name;")
        topic_name_list = [record["topic_name"] for record in topics]
        print("Topics: ", topic_name_list, "; Total topics: ", len(topic_name_list))

    def print_brokers_and_relationships(self):
        brokers = self.execute_cypher("MATCH (b:Broker) RETURN b.name as broker_name;")
        broker_name_list = [record["broker_name"] for record in brokers]
        print("Brokers: ", broker_name_list, "; Total brokers: ", len(broker_name_list))

        for broker_name in broker_name_list:
            self.print_routes_relationships(broker_name)

    def print_routes_relationships(self, broker_name):
        topic_name_list = self.execute_cypher("""
        MATCH (b:Broker {name: $broker_name})-[r:ROUTES]->(t:Topic)
        RETURN t.name AS topic_name
        """, {"broker_name": broker_name})
        topic_name_list = [record["topic_name"] for record in topic_name_list]
        print("Broker: ", broker_name, " routes topics: ", topic_name_list, "; Total topics: ", len(topic_name_list))

    def print_derived_relationships(self):
        self.print_depends_on_relationships()
        self.print_connects_to_relationships()

    def print_depends_on_relationships(self):
        self.print_depends_on_relationships_application_to_application()

    def print_depends_on_relationships_application_to_application(self):
        depends_on_relationships = self.execute_cypher("""
        MATCH (a1:Application)-[:DEPENDS_ON]->(a2:Application)
        RETURN a1.name AS fromApp, a2.name AS toApp
        """)
        for record in depends_on_relationships:
            print("App: ", record["fromApp"], " depends on App: ", record["toApp"])
        print("Total DEPENDS_ON relationships between applications: ", len(depends_on_relationships))

    def print_connects_to_relationships(self):
        connects_to_relationships = self.execute_cypher("""
        MATCH (n1:Node)-[:CONNECTS_TO]->(n2:Node)
        RETURN n1.name AS fromNode, n2.name AS toNode
        """)
        for record in connects_to_relationships:
            print("Node: ", record["fromNode"], " connects to Node: ", record["toNode"])
        print("Total CONNECTS_TO relationships: ", len(connects_to_relationships))

    def import_graph_from_file(self, file_dir):
        self.clear_graph()

        dataset_file_path = file_dir + "dataset.json"
        vertices_file_path = file_dir + "nodes.csv"
        edges_file_path = file_dir + "edges.csv"    
        if dataset_file_path is not None:
            self.import_dataset_from_file(dataset_file_path)
        elif vertices_file_path is not None and edges_file_path is not None:
            self.import_vertices_from_csv(vertices_file_path)
            self.import_edges_from_csv(edges_file_path)
        else:
            raise ValueError("Either dataset file or both vertices and edges files must be provided")

    def import_vertices_from_csv(self, file_path):
        vertices_df = pd.read_csv(file_path)
        for index, row in vertices_df.iterrows():
            if row["type"] == "Topic":
                # For topics, we need to handle durability and reliability
                self.execute_cypher("""
                CREATE (n:Topic {
                    id: $id,
                    name: $name,
                    type: $type,
                    durability: $durability,
                    reliability: $reliability
                });
                """, {"id": row["id"], "name": row["name"], "type": row["type"], "durability": row["durability"], "reliability": row["reliability"]})
            else:
                # For other nodes, we can create them without durability and reliability
                self.execute_cypher("""
                CREATE (n:%s {
                    id: $id,
                    name: $name,
                    type: $type  
                });
                """ % row["type"], {"id": row["id"], "name": row["name"], "type": row["type"]})

    def import_edges_from_csv(self, file_path):
        edges_df = pd.read_csv(file_path)
        for index, row in edges_df.iterrows():
            self.execute_cypher("""
            MATCH (n1 {id: $source}), (n2 {id: $target})
            MERGE (n1)-[:%s]->(n2);
            """ % row["relationship"], {"source": row["source"], "target": row["target"]})

    def import_dataset_from_file(self, dataset_file):
        """Import dataset with enhanced properties and validation"""
        print(f"\nImporting dataset from {dataset_file}")
        
        # Load dataset
        with open(dataset_file, 'r') as f:
            self.dataset = json.load(f)

        self.import_dataset_to_neo4j()
    
    def create_publish_relationship_with_metrics(self, app_name: str, topic_name: str, 
                                                 msg_size: int, period_ms: Optional[float] = None):
        """Create PUBLISHES_TO with complete metrics"""
        if period_ms is None:
            period_ms = float(self.rng.integers(10, 1000))
            
        # Calculate message rate and bandwidth
        msg_rate_hz = 1000.0 / period_ms if period_ms > 0 else 0
        bandwidth_bps = msg_size * msg_rate_hz * 8  # bits per second
        
        # Simulate some realistic metrics
        properties = {
            "msg_size": msg_size,
            "period_ms": period_ms,
            "msg_rate_hz": msg_rate_hz,
            "bandwidth_bps": bandwidth_bps,
            "avg_latency_ms": float(self.rng.uniform(0.1, 10)),
            "max_latency_ms": float(self.rng.uniform(10, 100)),
            "min_latency_ms": float(self.rng.uniform(0.05, 0.5)),
            "error_rate": float(self.rng.uniform(0, 0.01)),
            "messages_sent": int(self.rng.integers(1000, 1000000)),
            "messages_failed": int(self.rng.integers(0, 100)),
            "last_message_at": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat()
        }
        
        self.execute_cypher("""
            MATCH (a:Application {name: $app_name}), (t:Topic {name: $topic_name})
            MERGE (a)-[r:PUBLISHES_TO]->(t)
            SET r += $properties
        """, {
            "app_name": app_name,
            "topic_name": topic_name,
            "properties": properties
        })
    
    def create_subscribe_relationship_with_metrics(self, app_name: str, topic_name: str):
        """Create SUBSCRIBES_TO with complete metrics"""
        properties = {
            "buffer_size": int(self.rng.integers(10, 1000)),
            "messages_received": int(self.rng.integers(1000, 1000000)),
            "messages_dropped": int(self.rng.integers(0, 100)),
            "avg_processing_time_ms": float(self.rng.uniform(0.1, 50)),
            "max_processing_time_ms": float(self.rng.uniform(50, 500)),
            "last_message_at": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat()
        }
        
        self.execute_cypher("""
            MATCH (a:Application {name: $app_name}), (t:Topic {name: $topic_name})
            MERGE (a)-[r:SUBSCRIBES_TO]->(t)
            SET r += $properties
        """, {
            "app_name": app_name,
            "topic_name": topic_name,
            "properties": properties
        })
    
    def add_temporal_metrics(self):
        """Add realistic temporal metrics with proper timestamps"""
        # Generate hourly metrics for the last 24 hours
        now = datetime.now()
        timestamps = [(now - timedelta(hours=i)).isoformat() for i in range(24, 0, -1)]
        
        # Applications: Add time-series metrics
        self.execute_cypher("""
            MATCH (a:Application)
            WITH a, $timestamps AS timestamps
            SET a.cpu_usage_history = [t IN $timestamps | toFloat(rand() * 60 + 20)],
                a.memory_usage_history = [t IN $timestamps | toFloat(rand() * 50 + 30)],
                a.error_rate_history = [t IN $timestamps | toFloat(rand() * 0.05)],
                a.request_rate_history = [t IN $timestamps | toFloat(rand() * 1000 + 100)],
                a.timestamp_history = $timestamps
        """, {"timestamps": timestamps})
        
        # Topics: Add throughput history
        self.execute_cypher("""
            MATCH (t:Topic)
            WITH t, $timestamps AS timestamps
            SET t.throughput_timestamps = timestamps,
                t.messages_per_sec_history = [x IN range(0, size(timestamps)-1) | toFloat(rand() * 1000 + 10)],
                t.bytes_per_sec_history = [x IN range(0, size(timestamps)-1) | toFloat(rand() * 1000000 + 10000)],
                t.active_publishers_history = [x IN range(0, size(timestamps)-1) | toInteger(rand() * 10 + 1)],
                t.active_subscribers_history = [x IN range(0, size(timestamps)-1) | toInteger(rand() * 20 + 1)]
        """, {"timestamps": timestamps})
    
    def validate_graph(self) -> Tuple[bool, List[str]]:
        """Enhanced validation with more comprehensive checks"""
        errors = []
        warnings = []
        
        # 1. Structural validation
        orphaned_topics = self.execute_cypher("""
            MATCH (t:Topic)
            WHERE NOT EXISTS((t)<-[:PUBLISHES_TO]-(:Application)) 
            AND NOT EXISTS((t)<-[:SUBSCRIBES_TO]-(:Application))
            RETURN collect(t.name) as topics
        """)
        if orphaned_topics[0]['topics']:
            warnings.append(f"Orphaned topics: {orphaned_topics[0]['topics']}")
        
        # 2. Check for topics without publishers (critical issue)
        no_publishers = self.execute_cypher("""
            MATCH (t:Topic)<-[:SUBSCRIBES_TO]-(a:Application)
            WHERE NOT EXISTS((t)<-[:PUBLISHES_TO]-(:Application))
            RETURN collect(DISTINCT t.name) as topics
        """)
        if no_publishers[0]['topics']:
            errors.append(f"Topics with subscribers but no publishers: {no_publishers[0]['topics']}")
        
        # 3. Check for undeployed applications
        undeployed = self.execute_cypher("""
            MATCH (a:Application)
            WHERE NOT EXISTS((a)-[:RUNS_ON]->(:Node))
            RETURN collect(a.name) as apps
        """)
        if undeployed[0]['apps']:
            errors.append(f"Undeployed applications: {undeployed[0]['apps']}")
        
        # 4. Check for brokers without topics (if brokers exist)
        broker_count = self.execute_cypher("MATCH (b:Broker) RETURN count(b) as count")[0]['count']
        if broker_count > 0:
            idle_brokers = self.execute_cypher("""
                MATCH (b:Broker)
                WHERE NOT EXISTS((b)-[:ROUTES]->(:Topic))
                RETURN collect(b.name) as brokers
            """)
            if idle_brokers[0]['brokers']:
                warnings.append(f"Idle brokers: {idle_brokers[0]['brokers']}")
            
            # Check for topics without brokers
            unrouted_topics = self.execute_cypher("""
                MATCH (t:Topic)
                WHERE NOT EXISTS((t)<-[:ROUTES]-(:Broker))
                RETURN collect(t.name) as topics
            """)
            if unrouted_topics[0]['topics']:
                errors.append(f"Topics not routed by any broker: {unrouted_topics[0]['topics']}")
        
        # 5. QoS validation
        qos_issues = self.execute_cypher("""
            MATCH (t:Topic)
            WHERE t.deadline_ms IS NULL 
            OR t.reliability IS NULL 
            OR t.durability IS NULL
            OR t.criticality_score IS NULL
            RETURN collect(t.name) as topics
        """)
        if qos_issues[0]['topics']:
            errors.append(f"Topics with incomplete QoS policies: {qos_issues[0]['topics']}")
        
        # 6. Check for circular dependencies maximum depth 2
        circular = self.execute_cypher("""
            MATCH (a:Application)-[:DEPENDS_ON*2]->(a)
            RETURN collect(DISTINCT a.name) as apps
        """)
        if circular[0]['apps']:
            errors.append(f"Applications with circular dependencies: {circular[0]['apps']}")
        
        # 7. Node oversubscription check
        oversubscribed = self.execute_cypher("""
            MATCH (n:Node)<-[:RUNS_ON]-(a:Application)
            WITH n, count(a) as app_count, sum(a.criticality_score) as total_criticality
            WHERE app_count > 10 OR total_criticality > 5.0
            RETURN collect(n.name + ' (' + toString(app_count) + ' apps)') as nodes
        """)
        if oversubscribed[0]['nodes']:
            warnings.append(f"Potentially oversubscribed nodes: {oversubscribed[0]['nodes']}")
        
        is_valid = len(errors) == 0
        all_issues = errors + warnings
        
        return is_valid, all_issues
    
    def generate_synthetic_dataset(self, config: Dict) -> Dict:
        self.dataset = self.generator.generate_dataset(config)
    
    def import_dataset_to_neo4j(self):    
        # Clear existing graph
        self.clear_graph()
        
        # Create constraints first
        self.create_constraints()
        
        # Print metadata
        metadata = self.dataset.get('metadata', {})
        print(f"Scenario: {metadata.get('scenario', 'Unknown')}")
        print(f"Scale: {metadata.get('scale', 'Unknown')}")
        print(f"Expected nodes: {metadata.get('statistics', {}).get('total_nodes', 'Unknown')}")
        
        # Import in batches for performance
        batch_size = 1000
        
        # Import nodes with enhanced properties
        self._import_infrastructure_nodes_batch(batch_size)
        self._import_applications_batch(batch_size)
        self._import_topics_batch(batch_size)
        self._import_brokers_batch(batch_size)
        
        # Import relationships with enhanced properties
        self._import_relationships_batch(batch_size)
        
        # Derive additional relationships
        self.derive_relationships()
        
        # Calculate and update computed properties
        self._update_application_types()
        self._calculate_topic_fanout()
        self._update_node_utilization()
        
        # Validate the imported graph
        is_valid, issues = self.validate_graph()
        if is_valid:
            print("✓ Graph validation passed")
        else:
            print("⚠ Graph validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Print statistics
        self.print_statistics()
        
        return True
    
    def _import_infrastructure_nodes_batch(self, batch_size: int):
        """Import infrastructure nodes with enhanced properties"""
        nodes = self.dataset.get('nodes', [])
        
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            
            self.execute_cypher("""
                UNWIND $batch AS item
                CREATE (n:Node {
                    id: item.id,
                    name: item.name,
                    hostname: item.name,
                    type: coalesce(item.type, 'COMPUTE'),
                    
                    // Resources
                    cpu_capacity: item.cpu_capacity,
                    memory_gb: item.memory_gb,
                    storage_gb: coalesce(item.storage_gb, 1000),
                    network_bandwidth_mbps: item.network_bandwidth_mbps,
                    
                    // Location
                    zone: coalesce(item.zone, 'default-zone'),
                    datacenter: coalesce(item.datacenter, 'default-dc'),
                    rack: coalesce(item.rack, 'default-rack'),
                    region: coalesce(item.region, 'us-east-1'),
                    
                    // Status
                    status: coalesce(item.status, 'active'),
                    health_score: coalesce(item.health_score, 1.0),
                    
                    // Utilization (will be calculated)
                    cpu_utilization: 0.0,
                    memory_utilization: 0.0,
                    network_utilization: 0.0,
                    
                    // Metadata
                    created_at: datetime(),
                    updated_at: datetime()
                })
            """, {"batch": batch})
            
            print(f"  Imported {min(i + batch_size, len(nodes))}/{len(nodes)} infrastructure nodes")
    
    def _import_applications_batch(self, batch_size: int):
        """Import applications with complete properties"""
        apps = self.dataset.get('applications', [])
        
        for i in range(0, len(apps), batch_size):
            batch = apps[i:i + batch_size]
            
            # Enhance application properties
            for app in batch:
                # Add missing QoS requirements
                if 'qos_requirements' not in app:
                    app['qos_requirements'] = self._generate_qos_requirements(
                        app.get('criticality_score', 0.5)
                    )
                
                # Add resource requirements
                if 'resource_requirements' not in app:
                    app['resource_requirements'] = self._generate_resource_requirements(
                        app.get('criticality_score', 0.5),
                        app.get('type', 'PROSUMER')
                    )
            
            self.execute_cypher("""
                UNWIND $batch AS item
                CREATE (a:Application {
                    id: item.id,
                    name: item.name,
                    type: coalesce(item.type, 'UNKNOWN'),
                    criticality_score: item.criticality_score,
                    
                    // Version info
                    version: coalesce(item.version, '1.0.0'),
                    replicas: coalesce(item.replicas, 1),
                    
                    // QoS Requirements
                    qos_latency_ms: item.qos_requirements.latency_ms,
                    qos_throughput_mbps: item.qos_requirements.throughput_mbps,
                    qos_availability_percent: item.qos_requirements.availability_percent,
                    
                    // Resource Requirements
                    cpu_cores: item.resource_requirements.cpu_cores,
                    memory_gb: item.resource_requirements.memory_gb,
                    storage_gb: item.resource_requirements.storage_gb,
                    
                    // Metadata
                    owner: coalesce(item.owner, 'system'),
                    deployment_date: datetime(),
                    last_update: datetime(),
                    
                    // Metrics (will be calculated)
                    message_rate_in: 0.0,
                    message_rate_out: 0.0,
                    error_rate: 0.0,
                    avg_processing_time_ms: 0.0
                })
            """, {"batch": batch})
            
            print(f"  Imported {min(i + batch_size, len(apps))}/{len(apps)} applications")
    
    def _import_topics_batch(self, batch_size: int):
        """Import topics with enhanced QoS properties"""
        topics = self.dataset.get('topics', [])
        
        for i in range(0, len(topics), batch_size):
            batch = topics[i:i + batch_size]
            
            self.execute_cypher("""
                UNWIND $batch AS item
                CREATE (t:Topic {
                    id: item.id,
                    name: item.name,
                    
                    // QoS Policies
                    durability: item.qos.durability,
                    reliability: item.qos.reliability,
                    transport_priority: item.qos.transport_priority,
                    deadline_ms: item.qos.deadline_ms,
                    lifespan_ms: item.qos.lifespan_ms,
                    history_depth: item.qos.history_depth,
                    
                    // Characteristics
                    criticality_score: item.criticality_score,
                    partition_count: coalesce(item.partition_count, 1),
                    message_pattern: coalesce(item.message_pattern, 'EVENT_DRIVEN'),
                    
                    // Schema
                    schema_type: coalesce(item.schema_type, 'JSON'),
                    schema_version: coalesce(item.schema_version, '1.0'),
                    
                    // Metrics (will be calculated)
                    avg_message_size_kb: 0.0,
                    messages_per_second: 0.0,
                    total_throughput_mbps: 0.0,
                    publisher_count: 0,
                    subscriber_count: 0,
                    fanout_ratio: 0.0,
                    
                    // Metadata
                    created_at: datetime(),
                    updated_at: datetime()
                })
            """, {"batch": batch})
            
            print(f"  Imported {min(i + batch_size, len(topics))}/{len(topics)} topics")
    
    def _import_brokers_batch(self, batch_size: int):
        """Import brokers with complete configuration"""
        brokers = self.dataset.get('brokers', [])
        
        for i in range(0, len(brokers), batch_size):
            batch = brokers[i:i + batch_size]
            
            self.execute_cypher("""
                UNWIND $batch AS item
                CREATE (b:Broker {
                    id: item.id,
                    name: item.name,
                    type: item.type,
                    
                    // Capacity
                    max_topics: item.max_topics,
                    max_connections: item.max_connections,
                    max_throughput_mbps: item.max_throughput_mbps,
                    
                    // Configuration
                    replication_factor: item.replication_factor,
                    partition_count: item.partition_count,
                    retention_hours: item.retention_hours,
                    
                    // Performance Metrics (will be updated)
                    avg_latency_ms: 0.0,
                    current_load_percent: 0.0,
                    uptime_percent: 100.0,
                    active_connections: 0,
                    topics_count: 0,
                    
                    // Metadata
                    version: item.version,
                    created_at: datetime(),
                    status: 'active'
                })
            """, {"batch": batch})
            
            print(f"  Imported {min(i + batch_size, len(brokers))}/{len(brokers)} brokers")
    
    def _import_relationships_batch(self, batch_size: int):
        """Import relationships with enhanced properties"""
        relationships = self.dataset.get('relationships', {})
        
        # Import RUNS_ON relationships
        runs_on = relationships.get('runs_on', [])
        for i in range(0, len(runs_on), batch_size):
            batch = runs_on[i:i + batch_size]
            
            self.execute_cypher("""
                UNWIND $batch AS rel
                MATCH (source {id: rel.from})
                MATCH (target:Node {id: rel.to})
                CREATE (source)-[r:RUNS_ON {
                    created_at: datetime(),
                    resource_allocation: coalesce(rel.resource_allocation, 1.0)
                }]->(target)
            """, {"batch": batch})
        
        print(f"  Imported {len(runs_on)} RUNS_ON relationships")
        
        # Import PUBLISHES_TO relationships with enhanced properties
        publishes = relationships.get('publishes_to', [])
        for i in range(0, len(publishes), batch_size):
            batch = publishes[i:i + batch_size]
            
            # Enhance each relationship
            for rel in batch:
                rel['pattern'] = self._detect_message_pattern(rel.get('period_ms', 0))
                rel['reliability'] = 'RELIABLE' if rel.get('msg_size', 0) > 1024 else 'BEST_EFFORT'
                rel['priority'] = random.randint(1, 10)
            
            self.execute_cypher("""
                UNWIND $batch AS rel
                MATCH (a:Application {id: rel.from})
                MATCH (t:Topic {id: rel.to})
                CREATE (a)-[r:PUBLISHES_TO {
                    msg_size: coalesce(rel.msg_size, 512),
                    period_ms: coalesce(rel.period_ms, 1000),
                    msg_rate_hz: CASE 
                        WHEN rel.period_ms > 0 THEN 1000.0 / rel.period_ms 
                        ELSE 0.0 
                    END,
                    pattern: rel.pattern,
                    reliability: rel.reliability,
                    priority: rel.priority,
                    
                    // Metrics
                    messages_sent: 0,
                    messages_failed: 0,
                    avg_latency_ms: 0.0,
                    max_latency_ms: 0.0,
                    error_rate: 0.0,
                    
                    // Metadata
                    created_at: datetime(),
                    last_message_at: datetime()
                }]->(t)
            """, {"batch": batch})
        
        print(f"  Imported {len(publishes)} PUBLISHES_TO relationships")
        
        # Import SUBSCRIBES_TO relationships with enhanced properties
        subscribes = relationships.get('subscribes_to', [])
        for i in range(0, len(subscribes), batch_size):
            batch = subscribes[i:i + batch_size]
            
            # Enhance each relationship
            for rel in batch:
                rel['ack_mode'] = random.choice(['AUTO', 'MANUAL'])
                rel['subscription_type'] = random.choice(['EXCLUSIVE', 'SHARED', 'FAILOVER'])
            
            self.execute_cypher("""
                UNWIND $batch AS rel
                MATCH (a:Application {id: rel.from})
                MATCH (t:Topic {id: rel.to})
                CREATE (a)-[r:SUBSCRIBES_TO {
                    filter_expression: coalesce(rel.filter, '*'),
                    consumption_rate: coalesce(rel.consumption_rate, 100.0),
                    acknowledgment_mode: rel.ack_mode,
                    offset_management: 'STORED',
                    subscription_type: rel.subscription_type,
                    
                    // Metrics
                    messages_received: 0,
                    messages_processed: 0,
                    messages_failed: 0,
                    avg_processing_time_ms: 0.0,
                    lag: 0,
                    
                    // Metadata
                    created_at: datetime(),
                    last_message_at: datetime()
                }]->(t)
            """, {"batch": batch})
        
        print(f"  Imported {len(subscribes)} SUBSCRIBES_TO relationships")
        
        # Import ROUTES relationships
        routes = relationships.get('routes', [])
        for i in range(0, len(routes), batch_size):
            batch = routes[i:i + batch_size]
            
            self.execute_cypher("""
                UNWIND $batch AS rel
                MATCH (b:Broker {id: rel.from})
                MATCH (t:Topic {id: rel.to})
                CREATE (b)-[r:ROUTES {
                    partition_assignment: coalesce(rel.partitions, '0-*'),
                    routing_policy: coalesce(rel.policy, 'ROUND_ROBIN'),
                    created_at: datetime()
                }]->(t)
            """, {"batch": batch})
        
        print(f"  Imported {len(routes)} ROUTES relationships")
    
    def derive_relationships(self):
        """Derive DEPENDS_ON and CONNECTS_TO with calculated metrics"""
        print("\nDeriving enhanced relationships...")
        
        # DEPENDS_ON with strength calculation
        self.execute_cypher("""
            MATCH (a1:Application)-[rp:PUBLISHES_TO]->(t:Topic)<-[rs:SUBSCRIBES_TO]-(a2:Application)
            WITH a1, a2, t, rp, rs,
                 t.criticality_score as topic_criticality,
                 rp.msg_rate_hz * rp.msg_size as throughput_bps,
                 CASE 
                    WHEN t.reliability = 'RELIABLE' AND t.durability = 'PERSISTENT' THEN 1.0
                    WHEN t.reliability = 'RELIABLE' THEN 0.75
                    WHEN t.durability = 'PERSISTENT' THEN 0.75
                    ELSE 0.5
                 END as strength
            MERGE (a2)-[rd:DEPENDS_ON]->(a1)
            SET rd.type = 'DATA',
                rd.strength = strength,
                rd.criticality = CASE
                    WHEN topic_criticality > 0.8 THEN 'CRITICAL'
                    WHEN topic_criticality > 0.5 THEN 'REQUIRED'
                    ELSE 'OPTIONAL'
                END,
                rd.topic = t.name,
                rd.topic_id = t.id,
                rd.throughput_bps = throughput_bps,
                rd.latency_requirement_ms = t.deadline_ms,
                rd.qos_reliability = t.reliability,
                rd.created_at = datetime()
        """)
        
        deps = self.execute_cypher("MATCH ()-[d:DEPENDS_ON]->() RETURN count(d) as count")[0]['count']
        print(f"  Created {deps} DEPENDS_ON relationships")
        
        # CONNECTS_TO with aggregated metrics
        self.execute_cypher("""
            MATCH (n1:Node)<-[:RUNS_ON]-(a1:Application)-[d:DEPENDS_ON]->(a2:Application)-[:RUNS_ON]->(n2:Node)
            WHERE n1 <> n2
            WITH n1, n2, 
                 COUNT(DISTINCT d) as num_dependencies,
                 SUM(d.throughput_bps) as total_throughput,
                 MAX(CASE 
                    WHEN d.criticality = 'CRITICAL' THEN 1.0
                    WHEN d.criticality = 'REQUIRED' THEN 0.7
                    ELSE 0.3
                 END) as max_criticality,
                 MIN(d.latency_requirement_ms) as min_latency_req
            MERGE (n1)-[c:CONNECTS_TO]->(n2)
            SET c.num_dependencies = num_dependencies,
                c.total_throughput_bps = total_throughput,
                c.bandwidth_gbps = total_throughput / 1000000000.0,
                c.criticality = max_criticality,
                c.latency_ms = CASE
                    WHEN min_latency_req < 10 THEN 0.5
                    WHEN min_latency_req < 100 THEN 1.0
                    ELSE 5.0
                END,
                c.packet_loss_percent = 0.01,
                c.connection_type = 'DATACENTER',
                c.reliability = 0.999,
                c.connection_strength = num_dependencies * max_criticality,
                c.created_at = datetime()
        """)
        
        conns = self.execute_cypher("MATCH ()-[c:CONNECTS_TO]->() RETURN count(c) as count")[0]['count']
        print(f"  Created {conns} CONNECTS_TO relationships")
    
    def _update_application_types(self):
        """Update application types based on pub/sub relationships"""
        self.execute_cypher("""
            MATCH (a:Application)
            OPTIONAL MATCH (a)-[:PUBLISHES_TO]->()
            WITH a, count(*) as pub_count
            OPTIONAL MATCH (a)-[:SUBSCRIBES_TO]->()
            WITH a, pub_count, count(*) as sub_count
            SET a.type = CASE 
                WHEN pub_count > 0 AND sub_count > 0 THEN 'PROSUMER'
                WHEN pub_count > 0 THEN 'PRODUCER'
                WHEN sub_count > 0 THEN 'CONSUMER'
                ELSE 'UNKNOWN'
            END
        """)
        print("  Updated application types")
    
    def _calculate_topic_fanout(self):
        """Calculate fanout ratio for topics"""
        self.execute_cypher("""
            MATCH (t:Topic)
            OPTIONAL MATCH (t)<-[:PUBLISHES_TO]-(pub:Application)
            WITH t, count(DISTINCT pub) as pub_count
            OPTIONAL MATCH (t)<-[:SUBSCRIBES_TO]-(sub:Application)
            WITH t, pub_count, count(DISTINCT sub) as sub_count
            SET t.publisher_count = pub_count,
                t.subscriber_count = sub_count,
                t.fanout_ratio = CASE 
                    WHEN pub_count > 0 THEN toFloat(sub_count) / pub_count 
                    ELSE 0.0 
                END
        """)
        print("  Calculated topic fanout metrics")
    
    def _update_node_utilization(self):
        """Calculate node resource utilization"""
        self.execute_cypher("""
            MATCH (n:Node)
            OPTIONAL MATCH (n)<-[:RUNS_ON]-(a:Application)
            WITH n, 
                 sum(a.cpu_cores) as used_cpu,
                 sum(a.memory_gb) as used_memory,
                 sum(a.qos_throughput_mbps) as used_bandwidth
            SET n.cpu_utilization = CASE 
                    WHEN n.cpu_capacity > 0 THEN used_cpu / n.cpu_capacity 
                    ELSE 0.0 
                END,
                n.memory_utilization = CASE 
                    WHEN n.memory_gb > 0 THEN used_memory / n.memory_gb 
                    ELSE 0.0 
                END,
                n.network_utilization = CASE 
                    WHEN n.network_bandwidth_mbps > 0 THEN used_bandwidth / n.network_bandwidth_mbps 
                    ELSE 0.0 
                END
        """)
        print("  Updated node utilization metrics")
    
    def validate_graph(self) -> Tuple[bool, List[str]]:
        """Comprehensive graph validation"""
        errors = []
        warnings = []
        
        # Check for orphaned topics
        orphaned = self.execute_cypher("""
            MATCH (t:Topic)
            WHERE NOT (()-[:PUBLISHES_TO]->(t)) OR NOT (()-[:SUBSCRIBES_TO]->(t))
            RETURN collect(t.name) as topics
        """)[0]['topics']
        
        if orphaned:
            warnings.append(f"Found {len(orphaned)} orphaned topics: {orphaned[:5]}")
        
        # Check for isolated applications
        isolated = self.execute_cypher("""
            MATCH (a:Application)
            WHERE NOT (a)-[:PUBLISHES_TO|SUBSCRIBES_TO]->()
            RETURN collect(a.name) as apps
        """)[0]['apps']
        
        if isolated:
            errors.append(f"Found {len(isolated)} isolated applications: {isolated[:5]}")
        
        # Check for circular dependencies (depth 2)
        circular = self.execute_cypher("""
            MATCH (a:Application)-[:DEPENDS_ON*2]->(a)
            RETURN collect(DISTINCT a.name) as apps
        """)[0]['apps']
        
        if circular:
            warnings.append(f"Circular dependencies detected: {circular[:5]}")
        
        # Check for overutilized nodes
        overutilized = self.execute_cypher("""
            MATCH (n:Node)
            WHERE n.cpu_utilization > 0.9 OR n.memory_utilization > 0.9
            RETURN collect(n.name) as nodes
        """)[0]['nodes']
        
        if overutilized:
            warnings.append(f"Overutilized nodes: {overutilized}")
        
        # Check QoS compatibility
        qos_issues = self.execute_cypher("""
            MATCH (a1:Application)-[:DEPENDS_ON]->(a2:Application)
            WHERE a1.qos_latency_ms < a2.qos_latency_ms
            RETURN count(*) as count
        """)[0]['count']
        
        if qos_issues > 0:
            warnings.append(f"Found {qos_issues} QoS compatibility issues")
        
        is_valid = len(errors) == 0
        all_issues = errors + warnings
        
        return is_valid, all_issues
    
    def print_statistics(self):
        """Print comprehensive graph statistics"""
        stats = self.execute_cypher("""
            MATCH (a:Application) WITH count(a) as apps
            MATCH (t:Topic) WITH apps, count(t) as topics
            MATCH (n:Node) WITH apps, topics, count(n) as nodes
            MATCH (b:Broker) WITH apps, topics, nodes, count(b) as brokers
            MATCH ()-[p:PUBLISHES_TO]->() WITH apps, topics, nodes, brokers, count(p) as pubs
            MATCH ()-[s:SUBSCRIBES_TO]->() WITH apps, topics, nodes, brokers, pubs, count(s) as subs
            MATCH ()-[d:DEPENDS_ON]->() WITH apps, topics, nodes, brokers, pubs, subs, count(d) as deps
            MATCH ()-[c:CONNECTS_TO]->() WITH apps, topics, nodes, brokers, pubs, subs, deps, count(c) as conns
            RETURN apps, topics, nodes, brokers, pubs, subs, deps, conns
        """)[0]
        
        print("\n📊 Graph Statistics:")
        print(f"  Nodes: {stats['nodes']}")
        print(f"  Applications: {stats['apps']}")
        print(f"  Topics: {stats['topics']}")
        print(f"  Brokers: {stats['brokers']}")
        print(f"  PUBLISHES_TO: {stats['pubs']}")
        print(f"  SUBSCRIBES_TO: {stats['subs']}")
        print(f"  DEPENDS_ON: {stats['deps']}")
        print(f"  CONNECTS_TO: {stats['conns']}")
        
        # Additional metrics
        metrics = self.execute_cypher("""
            MATCH (a:Application)
            WITH avg(a.criticality_score) as avg_app_criticality,
                 count(CASE WHEN a.type = 'PRODUCER' THEN 1 END) as producers,
                 count(CASE WHEN a.type = 'CONSUMER' THEN 1 END) as consumers,
                 count(CASE WHEN a.type = 'PROSUMER' THEN 1 END) as prosumers
            MATCH (t:Topic)
            WITH avg_app_criticality, producers, consumers, prosumers,
                 avg(t.criticality_score) as avg_topic_criticality,
                 avg(t.fanout_ratio) as avg_fanout
            MATCH (n:Node)
            WITH avg_app_criticality, producers, consumers, prosumers,
                 avg_topic_criticality, avg_fanout,
                 avg(n.cpu_utilization) as avg_cpu_util,
                 avg(n.memory_utilization) as avg_mem_util
            RETURN avg_app_criticality, producers, consumers, prosumers,
                   avg_topic_criticality, avg_fanout, avg_cpu_util, avg_mem_util
        """)[0]
        
        print("\n📈 Key Metrics:")
        print(f"  Application Types: {metrics['producers']} producers, {metrics['consumers']} consumers, {metrics['prosumers']} prosumers")
        print(f"  Avg Application Criticality: {metrics['avg_app_criticality']:.2f}")
        print(f"  Avg Topic Criticality: {metrics['avg_topic_criticality']:.2f}")
        print(f"  Avg Topic Fanout: {metrics['avg_fanout']:.2f}")
        print(f"  Avg CPU Utilization: {metrics['avg_cpu_util']:.1%}")
        print(f"  Avg Memory Utilization: {metrics['avg_mem_util']:.1%}")
    
    # Helper methods
    def _detect_message_pattern(self, period_ms: float) -> str:
        """Detect message pattern from period"""
        if period_ms == 0:
            return 'EVENT_DRIVEN'
        elif period_ms < 100:
            return 'BURST'
        elif period_ms % 1000 == 0:
            return 'PERIODIC'
        else:
            return 'RANDOM'
    
    def _generate_qos_requirements(self, criticality: float) -> Dict:
        """Generate QoS requirements based on criticality"""
        if criticality > 0.8:
            return {
                "latency_ms": random.uniform(10, 50),
                "throughput_mbps": random.uniform(50, 100),
                "availability_percent": 99.9
            }
        elif criticality > 0.5:
            return {
                "latency_ms": random.uniform(50, 200),
                "throughput_mbps": random.uniform(10, 50),
                "availability_percent": 99.0
            }
        else:
            return {
                "latency_ms": random.uniform(200, 1000),
                "throughput_mbps": random.uniform(1, 10),
                "availability_percent": 95.0
            }
    
    def _generate_resource_requirements(self, criticality: float, app_type: str) -> Dict:
        """Generate resource requirements based on criticality and type"""
        base_cpu = 0.5 if app_type == 'CONSUMER' else 1.0
        base_memory = 1.0 if app_type == 'CONSUMER' else 2.0
        
        multiplier = 1.0 + criticality * 2.0
        
        return {
            "cpu_cores": round(base_cpu * multiplier, 1),
            "memory_gb": round(base_memory * multiplier, 1),
            "storage_gb": round(10 * multiplier, 1)
        }
    
    def export_dataset_to_file(self, output_file: str):
        """Export dataset to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.dataset, f, indent=2)

        # Export methods for compatibility
    def export_graph(self, file_dir: str):
        """Export graph to CSV files"""
        vertices_file = f"{file_dir}/nodes.csv"
        edges_file = f"{file_dir}/edges.csv"

        self.export_vertices_to_csv(vertices_file)
        self.export_edges_to_csv(edges_file)

        if self.dataset is not None:
            dataset_file_path = f"{file_dir}/dataset.json"
            self.export_dataset_to_file(dataset_file_path)
    
    def export_vertices_to_csv(self, file_path: str):
        """Export vertices to CSV with enhanced properties"""
        records = self.execute_cypher("""
            MATCH (n)
            RETURN n.id AS id, 
                   n.name AS name, 
                   labels(n)[0] AS type,
                   n.criticality_score AS criticality,
                   n.durability AS durability,
                   n.reliability AS reliability,
                   CASE 
                       WHEN 'Application' IN labels(n) THEN n.qos_latency_ms
                       WHEN 'Topic' IN labels(n) THEN n.deadline_ms
                       ELSE null
                   END AS latency,
                   CASE
                       WHEN 'Node' IN labels(n) THEN n.cpu_utilization
                       ELSE null
                   END AS utilization
        """)
        
        vertices_df = pd.DataFrame(records)
        vertices_df.to_csv(file_path, index=False)
        print(f"  Exported {len(vertices_df)} vertices to {file_path}")
    
    def export_edges_to_csv(self, file_path: str):
        """Export edges to CSV with enhanced properties"""
        records = self.execute_cypher("""
            MATCH ()-[r]->()
            RETURN startNode(r).id AS source, 
                   endNode(r).id AS target, 
                   type(r) AS relationship,
                   r.strength AS strength,
                   r.criticality AS criticality,
                   r.throughput_bps AS throughput,
                   r.latency_ms AS latency
        """)
        
        edges_df = pd.DataFrame(records)
        edges_df.to_csv(file_path, index=False)
        print(f"  Exported {len(edges_df)} edges to {file_path}")