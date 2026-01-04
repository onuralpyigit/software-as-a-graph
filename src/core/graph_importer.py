"""
Graph Importer - Version 5.0

Imports graph data into Neo4j and performs advanced post-processing:
1. Derives DEPENDS_ON relationships (App->App, Node->Node, etc.)
2. Calculates Relationship Weights based on Topic QoS and Message Size.
3. Calculates Component Weights (Criticality).
"""

import logging
from typing import Dict, Any, List

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

class GraphImporter:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password", database="neo4j"):
        if not GraphDatabase: raise ImportError("pip install neo4j")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.logger = logging.getLogger(__name__)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    def close(self): self.driver.close()

    def import_graph(self, data: Dict[str, Any], clear: bool = False) -> Dict[str, int]:
        """
        Orchestrates the full import and derivation process.
        """
        if clear: self._run_query("MATCH (n) DETACH DELETE n")
        
        self._create_constraints()
        
        # 1. Import Basic Entities
        stats = {}
        stats["nodes"] = self._import_nodes(data.get("nodes", []))
        stats["brokers"] = self._import_brokers(data.get("brokers", []))
        stats["topics"] = self._import_topics(data.get("topics", []))
        stats["apps"] = self._import_apps(data.get("applications", []))
        
        # 2. Import Relationships
        rels = data.get("relationships", {})
        self._import_rels(rels.get("runs_on", []), "RUNS_ON", "Application", "Node") # Also Broker->Node handling inside
        self._import_rels(rels.get("routes", []), "ROUTES", "Broker", "Topic")
        self._import_rels(rels.get("publishes_to", []), "PUBLISHES_TO", "Application", "Topic")
        self._import_rels(rels.get("subscribes_to", []), "SUBSCRIBES_TO", "Application", "Topic")
        self._import_rels(rels.get("connects_to", []), "CONNECTS_TO", "Node", "Node")

        # 3. Derive DEPENDS_ON Relationships (The Core Logic)
        self.logger.info("Deriving DEPENDS_ON relationships...")
        stats["deps_app_app"] = self._derive_app_to_app()
        stats["deps_app_broker"] = self._derive_app_to_broker()
        stats["deps_node_node"] = self._derive_node_to_node()
        stats["deps_node_broker"] = self._derive_node_to_broker()
        
        # 4. Calculate Component Criticality (Weights)
        self._calculate_component_weights()
        
        return stats

    def _create_constraints(self):
        with self.driver.session(database=self.database) as session:
            for label in ["Application", "Broker", "Topic", "Node"]:
                session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")

    # --- Basic Import Helpers ---

    def _import_nodes(self, data): return self._batch_import(data, "MERGE (n:Node {id: row.id}) SET n.name=row.name, n.weight=0.0")
    def _import_brokers(self, data): return self._batch_import(data, "MERGE (n:Broker {id: row.id}) SET n.name=row.name, n.weight=0.0")
    def _import_apps(self, data): return self._batch_import(data, "MERGE (n:Application {id: row.id}) SET n.name=row.name, n.role=row.role, n.weight=0.0")
    
    def _import_topics(self, data):
        # Flatten QoS for Cypher
        flattened = []
        for t in data:
            qos = t.get("qos", {})
            flattened.append({
                "id": t["id"], "name": t["name"], "size": t.get("size", 256),
                "qos_d": qos.get("durability", "VOLATILE"),
                "qos_r": qos.get("reliability", "BEST_EFFORT"),
                "qos_p": qos.get("transport_priority", "MEDIUM")
            })
        
        query = """
        MERGE (t:Topic {id: row.id})
        SET t.name = row.name, t.size = row.size,
            t.qos_durability = row.qos_d,
            t.qos_reliability = row.qos_r,
            t.qos_transport_priority = row.qos_p
        """
        return self._batch_import(flattened, query)

    def _import_rels(self, data, rel_type, source_lbl, target_lbl):
        # Generic relationship importer
        query = f"""
        MATCH (a {{id: row.from}}), (b {{id: row.to}})
        MERGE (a)-[:{rel_type}]->(b)
        """
        # Special handling for RUNS_ON which can be from App or Broker
        if rel_type == "RUNS_ON":
            query = """
            MATCH (a), (b:Node {id: row.to}) 
            WHERE a.id = row.from AND (a:Application OR a:Broker)
            MERGE (a)-[:RUNS_ON]->(b)
            """
        return self._batch_import(data, query)

    def _batch_import(self, data, query, batch_size=500):
        if not data: return 0
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(data), batch_size):
                session.run(f"UNWIND $batch AS row {query}", batch=data[i:i+batch_size])
        return len(data)

    # --- Derivation Logic (The "Smart" Part) ---

    def _derive_app_to_app(self) -> int:
        """
        Derive APP_TO_APP dependencies.
        
        Rule: subscriber DEPENDS_ON publisher if they share topics.
        Weight = topic_count + sum(qos_scores) + sum(size_factors)
        """
        query = """
            // Find publisher-subscriber pairs through shared topics
            MATCH (pub:Application)-[:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(sub:Application)
            WHERE pub <> sub
            
            // Calculate per-topic contributions
            WITH pub, sub, t,
                // QoS score
                CASE t.qos_durability
                    WHEN 'PERSISTENT' THEN 0.40
                    WHEN 'TRANSIENT' THEN 0.25
                    WHEN 'TRANSIENT_LOCAL' THEN 0.20
                    ELSE 0.0
                END +
                CASE t.qos_reliability
                    WHEN 'RELIABLE' THEN 0.30
                    ELSE 0.0
                END +
                CASE t.qos_transport_priority
                    WHEN 'URGENT' THEN 0.30
                    WHEN 'HIGH' THEN 0.20
                    WHEN 'MEDIUM' THEN 0.10
                    ELSE 0.0
                END AS qos_score,
                // Size factor (normalized, capped at 0.5)
                CASE 
                    WHEN t.size IS NULL THEN 0.025
                    ELSE toFloat(t.size) / 10000.0
                END AS size_factor
            
            // Aggregate per pair
            WITH sub, pub,
                 collect(DISTINCT t.id) AS topics,
                 sum(qos_score) AS total_qos,
                 sum(CASE WHEN size_factor > 0.5 THEN 0.5 ELSE size_factor END) AS total_size
            
            // Calculate weight
            WITH sub, pub, topics,
                 size(topics) + total_qos + total_size AS weight
            
            // Create relationship
            MERGE (sub)-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(pub)
            SET d.via = topics,
                d.weight = weight
            
            RETURN count(*) AS count
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return result.single()["count"]
        
    def _derive_app_to_broker(self) -> int:
        """
        Derive APP_TO_BROKER dependencies.
        
        Rule: app DEPENDS_ON broker that routes topics the app uses.
        Weight = topic_count + sum(qos_scores) + sum(size_factors)
        """
        query = """
            // Find apps and brokers through topics
            MATCH (app:Application)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)<-[:ROUTES]-(broker:Broker)
            
            // Calculate per-topic contributions
            WITH app, broker, t,
                CASE t.qos_durability
                    WHEN 'PERSISTENT' THEN 0.40
                    WHEN 'TRANSIENT' THEN 0.25
                    WHEN 'TRANSIENT_LOCAL' THEN 0.20
                    ELSE 0.0
                END +
                CASE t.qos_reliability
                    WHEN 'RELIABLE' THEN 0.30
                    ELSE 0.0
                END +
                CASE t.qos_transport_priority
                    WHEN 'URGENT' THEN 0.30
                    WHEN 'HIGH' THEN 0.20
                    WHEN 'MEDIUM' THEN 0.10
                    ELSE 0.0
                END AS qos_score,
                CASE 
                    WHEN t.size IS NULL THEN 0.025
                    ELSE toFloat(t.size) / 10000.0
                END AS size_factor
            
            // Aggregate
            WITH app, broker,
                 collect(DISTINCT t.id) AS topics,
                 sum(qos_score) AS total_qos,
                 sum(CASE WHEN size_factor > 0.5 THEN 0.5 ELSE size_factor END) AS total_size
            
            WITH app, broker, topics,
                 size(topics) + total_qos + total_size AS weight
            
            MERGE (app)-[d:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(broker)
            SET d.via = topics,
                d.weight = weight
            
            RETURN count(*) AS count
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return result.single()["count"]
        
    def _derive_node_to_node(self) -> int:
        """
        Derive NODE_TO_NODE dependencies.
        
        Rule: node_A DEPENDS_ON node_B if app on node_A depends on app on node_B.
        Weight = sum of underlying app_to_app weights.
        """
        query = """
            // Find app dependencies that cross nodes
            MATCH (app1:Application)-[:RUNS_ON]->(n1:Node)
            MATCH (app2:Application)-[:RUNS_ON]->(n2:Node)
            MATCH (app1)-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(app2)
            WHERE n1 <> n2
            
            // Aggregate by node pair
            WITH n1, n2,
                 collect(DISTINCT {app1: app1.id, app2: app2.id}) AS app_pairs,
                 sum(d.weight) AS total_weight
            
            MERGE (n1)-[dep:DEPENDS_ON {dependency_type: 'node_to_node'}]->(n2)
            SET dep.via = [p IN app_pairs | p.app1 + '->' + p.app2],
                dep.weight = total_weight
            
            RETURN count(*) AS count
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return result.single()["count"]
        
    def _derive_node_to_broker(self) -> int:
        """
        Derive NODE_TO_BROKER dependencies.
        
        Rule: node DEPENDS_ON broker if any app on node depends on that broker.
        Weight = sum of underlying app_to_broker weights.
        """
        query = """
            // Find node-to-broker dependencies through apps
            MATCH (app:Application)-[:RUNS_ON]->(n:Node)
            MATCH (app)-[d:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(broker:Broker)
            
            // Aggregate
            WITH n, broker,
                 collect(DISTINCT app.id) AS apps,
                 sum(d.weight) AS total_weight
            
            MERGE (n)-[dep:DEPENDS_ON {dependency_type: 'node_to_broker'}]->(broker)
            SET dep.via = apps,
                dep.weight = total_weight
            
            RETURN count(*) AS count
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            return result.single()["count"]

    def _calculate_component_weights(self):
        """
        Sets 'weight' property on nodes based on importance (Degree Centrality of DEPENDS_ON).
        """
        query = """
        MATCH (n) WHERE n:Application OR n:Node OR n:Broker
        OPTIONAL MATCH (n)-[out:DEPENDS_ON]->()
        OPTIONAL MATCH ()-[inc:DEPENDS_ON]->(n)
        WITH n, coalesce(sum(out.weight), 0) + coalesce(sum(inc.weight), 0) as score
        SET n.weight = score
        """
        self._run_query(query)

    def _run_query(self, query):
        with self.driver.session(database=self.database) as session:
            session.run(query)

    def _run_count(self, query):
        with self.driver.session(database=self.database) as session:
            res = session.run(query)
            return res.single()["c"]