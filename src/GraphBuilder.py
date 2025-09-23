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
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.rng = np.random.default_rng(12345)
        self.generator = DatasetGenerator()
        self.validation_errors = []
        
    def close(self):
        self.driver.close()
        
    def execute_cypher(self, cypher: str, parameters: Optional[Dict] = None):
        with self.driver.session() as session:
            result = session.run(cypher, parameters or {})
            return list(result)

    def clear_graph(self):
        self.execute_cypher("MATCH (n) DETACH DELETE n;")
        self.dataset = None

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

    def derive_relationships(self):
        self.derive_depends_on_relationships()
        self.derive_connects_to_relationships()

    def derive_depends_on_relationships(self):
        # Define DEPENDS_ON relationships between applications
        self.execute_cypher("""
        MATCH (a1:Application)-[rp:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(a2:Application)
        MERGE (a2)-[rd:DEPENDS_ON {topic: t.name}]->(a1)
        SET rd.throughput = 1000*rp.msg_size/rp.period_ms, rd.latency = toInteger(rand() * 1000), rd.reliability=toInteger(rand() * 100)
        """)

    def derive_connects_to_relationships(self):
        # Create CONNECTS_TO relationships between nodes via interacting applications
        self.execute_cypher("""
        MATCH
        (n1:Node)<-[:RUNS_ON]-(a1:Application)-[:DEPENDS_ON]->(a2:Application)-[:RUNS_ON]->(n2:Node)
        WHERE n1 <> n2
        WITH DISTINCT n1, n2
        MERGE (n1)-[c:CONNECTS_TO]->(n2)
        """)

    def export_graph(self, file_dir):
        vertices_file_path = file_dir + "nodes.csv"
        self.export_vertices_to_csv(vertices_file_path)
        edges_file_path = file_dir + "edges.csv"
        self.export_edges_to_csv(edges_file_path)

        if self.dataset is not None:
            dataset_file_path = file_dir + "dataset.json"
            self.export_dataset_to_file(dataset_file_path)

    def export_vertices_to_csv(self, file_path):
        records = self.execute_cypher("""
        MATCH (n)
        RETURN n.id AS id, n.name AS name, labels(n)[0] AS type, n.durability AS durability, n.reliability AS reliability
        """)
        vertices_data = [{"id": record["id"], 
                          "name": record["name"], 
                          "type": record["type"], 
                          "durability": record["durability"],
                          "reliability": record["reliability"]} for record in records]
        vertices_df = pd.DataFrame(vertices_data)
        vertices_df.to_csv(file_path, index=False)

    def export_edges_to_csv(self, file_path):
        records = self.execute_cypher("""
        MATCH ()-[r]->()
        RETURN startNode(r).id AS source, endNode(r).id AS target, type(r) AS relationship, startNode(r).id + endNode(r).id AS id
        """)

        edges_data = [{"id": record["id"], "source": record["source"], "target": record["target"], "relationship": record["relationship"]} for record in records]
        edges_df = pd.DataFrame(edges_data)
        edges_df.to_csv(file_path, index=False)

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

    def import_dataset_from_file(self, file_path):
        # Read dataset JSON file
        with open(file_path, 'r') as file_path:
            self.dataset = json.load(file_path)
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
        if self.dataset is None:
            raise ValueError("No dataset provided for import")
        
        if 'nodes' not in self.dataset or 'applications' not in self.dataset or 'topics' not in self.dataset or 'relationships' not in self.dataset:
            raise ValueError("Dataset missing required sections (nodes, applications, topics, relationships)")
        
        """Import complete dataset to Neo4j with proper ID handling"""
        self.execute_cypher("MATCH (n) DETACH DELETE n")  # Clear existing
        
        # Import nodes with both id and name
        for node in self.dataset['nodes']:
            self.execute_cypher("""
                CREATE (n:Node {
                    id: $id,
                    name: $name,
                    cpu_capacity: $cpu,
                    memory_gb: $memory,
                    network_bandwidth_mbps: $bandwidth,
                    status: 'active',
                    type: 'Node'
                })
            """, {
                "id": node['id'],
                "name": node['name'],
                "cpu": node['cpu_capacity'],
                "memory": node['memory_gb'],
                "bandwidth": node['network_bandwidth_mbps']
            })
        
        # Import applications
        for app in self.dataset['applications']:
            self.execute_cypher("""
                CREATE (a:Application {
                    id: $id,
                    name: $name,
                    criticality_score: $criticality,
                    business_impact: $impact,
                    type: 'Application',
                    status: 'active'
                })
            """, {
                "id": app['id'],
                "name": app['name'],
                "criticality": app['criticality_score'],
                "impact": app['business_impact']
            })
        
        # Import topics with full QoS
        for topic in self.dataset['topics']:
            qos = topic['qos']
            self.execute_cypher("""
                CREATE (t:Topic {
                    id: $id,
                    name: $name,
                    durability: $durability,
                    reliability: $reliability,
                    transport_priority: $priority,
                    deadline_ms: $deadline,
                    lifespan_ms: $lifespan,
                    history_depth: $history,
                    criticality_score: $criticality,
                    type: 'Topic'
                })
            """, {
                "id": topic['id'],
                "name": topic['name'],
                "durability": qos['durability'],
                "reliability": qos['reliability'],
                "priority": qos['transport_priority'],
                "deadline": qos['deadline_ms'],
                "lifespan": qos['lifespan_ms'],
                "history": qos['history_depth'],
                "criticality": topic['criticality_score']
            })
        
        # Import brokers
        for broker in self.dataset.get('brokers', []):
            self.execute_cypher("""
                CREATE (b:Broker {
                    id: $id,
                    name: $name,
                    protocol: $protocol,
                    max_throughput_mbps: $throughput,
                    type: 'Broker',
                    status: 'active'
                })
            """, {
                "id": broker['id'],
                "name": broker['name'],
                "protocol": broker.get('protocol', 'Kafka'),
                "throughput": broker['max_throughput_mbps']
            })
            
            # Brokers should run on nodes - assign to first available node
            if self.dataset['nodes']:
                self.execute_cypher("""
                    MATCH (b:Broker {id: $broker_id}), (n:Node {id: $node_id})
                    CREATE (b)-[:RUNS_ON]->(n)
                """, {
                    "broker_id": broker['id'],
                    "node_id": self.dataset['nodes'][0]['id']  # Simplified assignment
                })
        
        # Import relationships with proper error handling
        for rel in self.dataset['relationships']['publishes_to']:
            result = self.execute_cypher("""
                MATCH (a:Application {id: $from}), (t:Topic {id: $to})
                CREATE (a)-[:PUBLISHES_TO {
                    msg_size: $size,
                    period_ms: $period,
                    msg_rate_hz: $rate,
                    bandwidth_bps: $bandwidth
                }]->(t)
                RETURN a.name as app, t.name as topic
            """, {
                "from": rel['from'],
                "to": rel['to'],
                "size": rel.get('msg_size', 1024),
                "period": rel.get('period_ms', 100),
                "rate": 1000.0 / rel.get('period_ms', 100),
                "bandwidth": rel.get('msg_size', 1024) * (1000.0 / rel.get('period_ms', 100)) * 8
            })
            
            if not result:
                print(f"Warning: Could not create PUBLISHES_TO from {rel['from']} to {rel['to']}")
        
        for rel in self.dataset['relationships']['subscribes_to']:
            self.execute_cypher("""
                MATCH (a:Application {id: $from}), (t:Topic {id: $to})
                CREATE (a)-[:SUBSCRIBES_TO {
                    buffer_size: 100,
                    created_at: datetime()
                }]->(t)
            """, {"from": rel['from'], "to": rel['to']})
        
        for rel in self.dataset['relationships']['runs_on']:
            self.execute_cypher("""
                MATCH (a:Application {id: $from}), (n:Node {id: $to})
                CREATE (a)-[:RUNS_ON]->(n)
            """, {"from": rel['from'], "to": rel['to']})
        
        for rel in self.dataset['relationships'].get('routes', []):
            self.execute_cypher("""
                MATCH (b:Broker {id: $from}), (t:Topic {id: $to})
                CREATE (b)-[:ROUTES {
                    routing_latency_ms: rand() * 10,
                    throughput_capacity_mbps: 1000
                }]->(t)
            """, {"from": rel['from'], "to": rel['to']})
        
        # Add temporal metrics
        self.add_temporal_metrics()
        
        # Derive additional relationships
        self.derive_enhanced_relationships()
        
        return True
    
    def export_dataset_to_file(self, output_file: str):
        """Export dataset to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.dataset, f, indent=2)
    
    def derive_enhanced_relationships(self):
        """Derive relationships with proper metrics"""
        # DEPENDS_ON with calculated metrics
        self.execute_cypher("""
            MATCH (a1:Application)-[rp:PUBLISHES_TO]->(t:Topic)<-[rs:SUBSCRIBES_TO]-(a2:Application)
            WITH a1, a2, t, rp, rs,
                 t.criticality_score as topic_criticality,
                 rp.msg_rate_hz * rp.msg_size as throughput_bps
            MERGE (a2)-[rd:DEPENDS_ON]->(a1)
            SET rd.topic = t.name,
                rd.throughput_bps = throughput_bps,
                rd.criticality = topic_criticality,
                rd.qos_deadline = t.deadline_ms,
                rd.qos_reliability = t.reliability,
                rd.dependency_strength = CASE 
                    WHEN t.reliability = 'RELIABLE' AND t.durability = 'PERSISTENT' THEN 1.0
                    WHEN t.reliability = 'RELIABLE' THEN 0.75
                    WHEN t.durability = 'PERSISTENT' THEN 0.75
                    ELSE 0.5
                END
        """)
        
        # CONNECTS_TO with aggregated metrics
        self.execute_cypher("""
            MATCH (n1:Node)<-[:RUNS_ON]-(a1:Application)-[d:DEPENDS_ON]->(a2:Application)-[:RUNS_ON]->(n2:Node)
            WHERE n1 <> n2
            WITH n1, n2, 
                 COUNT(DISTINCT d) as num_dependencies,
                 SUM(d.throughput_bps) as total_throughput,
                 MAX(d.criticality) as max_criticality
            MERGE (n1)-[c:CONNECTS_TO]->(n2)
            SET c.num_dependencies = num_dependencies,
                c.total_throughput_bps = total_throughput,
                c.criticality = max_criticality,
                c.connection_strength = num_dependencies * max_criticality
        """)

    def calculate_graph_metrics(self) -> Dict:
        """Calculate comprehensive graph metrics for validation"""
        metrics = {}
        
        # Basic counts
        counts = self.execute_cypher("""
            MATCH (a:Application) WITH count(a) as apps
            MATCH (t:Topic) WITH apps, count(t) as topics  
            MATCH (n:Node) WITH apps, topics, count(n) as nodes
            MATCH (b:Broker) WITH apps, topics, nodes, count(b) as brokers
            MATCH ()-[p:PUBLISHES_TO]->() WITH apps, topics, nodes, brokers, count(p) as pubs
            MATCH ()-[s:SUBSCRIBES_TO]->() WITH apps, topics, nodes, brokers, pubs, count(s) as subs
            MATCH ()-[d:DEPENDS_ON]->() WITH apps, topics, nodes, brokers, pubs, subs, count(d) as deps
            MATCH ()-[c:CONNECTS_TO]->() 
            RETURN apps, topics, nodes, brokers, pubs, subs, deps, count(c) as connects
        """)[0]
        metrics['counts'] = counts
        
        # Topology metrics
        topology = self.execute_cypher("""
            MATCH (a:Application)
            WITH avg(size((a)-[:PUBLISHES_TO]->())) as avg_pub_topics,
                avg(size((a)-[:SUBSCRIBES_TO]->())) as avg_sub_topics,
                max(size((a)-[:DEPENDS_ON]->())) as max_dependencies
            RETURN avg_pub_topics, avg_sub_topics, max_dependencies
        """)[0]
        metrics['topology'] = topology
        
        # QoS distribution
        qos_dist = self.execute_cypher("""
            MATCH (t:Topic)
            RETURN t.durability as durability, 
                t.reliability as reliability,
                count(*) as count
            ORDER BY count DESC
        """)
        metrics['qos_distribution'] = qos_dist
        
        # Criticality analysis
        criticality = self.execute_cypher("""
            MATCH (t:Topic)
            WHERE t.criticality_score IS NOT NULL
            RETURN avg(t.criticality_score) as avg_criticality,
                max(t.criticality_score) as max_criticality,
                min(t.criticality_score) as min_criticality,
                stDev(t.criticality_score) as std_criticality
        """)[0]
        metrics['criticality'] = criticality
        
        return metrics

    def print_statistics(self):
        """Print basic statistics of the graph"""
        stats = self.execute_cypher("""
            MATCH (a:Application) WITH count(a) as apps
            MATCH (t:Topic) WITH apps, count(t) as topics
            MATCH (n:Node) WITH apps, topics, count(n) as nodes
            MATCH (b:Broker) WITH apps, topics, nodes, count(b) as brokers
            MATCH ()-[p:PUBLISHES_TO]->() WITH apps, topics, nodes, brokers, count(p) as pubs
            MATCH ()-[s:SUBSCRIBES_TO]->() WITH apps, topics, nodes, brokers, pubs, count(s) as subs
            MATCH ()-[d:DEPENDS_ON]->() WITH apps, topics, nodes, brokers, pubs, subs, count(d) as deps
            RETURN apps, topics, nodes, brokers, pubs, subs, deps
        """)[0]
        
        print("Graph Statistics:")
        print(f"  Nodes: {stats['nodes']}")
        print(f"  Applications: {stats['apps']}")
        print(f"  Topics: {stats['topics']}")
        print(f"  Brokers: {stats['brokers']}")
        print(f"  PUBLISHES_TO: {stats['pubs']}")
        print(f"  SUBSCRIBES_TO: {stats['subs']}")
        print(f"  DEPENDS_ON: {stats['deps']}")

# Example usage function
def validate_and_generate_dataset():
    """Generate and validate a complete synthetic dataset"""
    builder = GraphBuilder(
        uri="bolt://localhost:7687",
        user="neo4j", 
        password="password"
    )
    
    try:
        # Configuration for a medium-scale system
        config = {
            'num_nodes': 10,
            'num_apps': 50,
            'num_topics': 100,
            'num_brokers': 3
        }
        
        print("Generating synthetic dataset...")
        builder.generate_synthetic_dataset(config)
        
        # Save dataset to file
        output_file = "../output/synthetic_dataset.json"
        builder.export_dataset_to_file(output_file)
        
        print("\nImporting dataset to Neo4j...")
        builder.import_dataset_to_neo4j()
        
        print("\nValidating graph consistency...")
        is_valid, issues = builder.validate_graph()
        
        if is_valid:
            print("✓ Graph validation passed")
        else:
            print("✗ Graph validation failed")
            
        if issues:
            print("\nValidation issues found:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Print statistics
        builder.print_statistics()
        
    finally:
        builder.close()

if __name__ == "__main__":
    validate_and_generate_dataset()