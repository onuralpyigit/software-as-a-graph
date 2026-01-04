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

    def _derive_app_to_app(self):
        """
        Derives App->App dependency if Sub subscribes to topic Pub publishes.
        Weight = Count(SharedTopics) + Sum(TopicQoS + TopicSizeFactor)
        """
        query = """
        MATCH (pub:Application)-[:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(sub:Application)
        WHERE pub <> sub
        WITH pub, sub, t,
             (CASE t.qos_reliability WHEN 'RELIABLE' THEN 0.3 ELSE 0.0 END +
              CASE t.qos_durability WHEN 'PERSISTENT' THEN 0.4 WHEN 'TRANSIENT' THEN 0.2 ELSE 0.0 END +
              CASE t.qos_transport_priority WHEN 'URGENT' THEN 0.3 WHEN 'HIGH' THEN 0.2 ELSE 0.0 END) as qos_score,
             (t.size / 10000.0) as size_factor
        WITH pub, sub, count(t) as shared_topics, sum(qos_score + size_factor) as weight_val
        MERGE (sub)-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(pub)
        SET d.weight = shared_topics + weight_val
        RETURN count(d) as c
        """
        return self._run_count(query)

    def _derive_app_to_broker(self):
        """
        App depends on Broker if the Broker routes a topic the App uses.
        """
        query = """
        MATCH (app:Application)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)<-[:ROUTES]-(b:Broker)
        WITH app, b, count(t) as topics
        MERGE (app)-[d:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(b)
        SET d.weight = topics * 1.0
        RETURN count(d) as c
        """
        return self._run_count(query)

    def _derive_node_to_node(self):
        """
        Node A depends on Node B if an App on A depends on an App on B.
        Weight sums the weights of the underlying app dependencies.
        """
        query = """
        MATCH (a1:Application)-[dep:DEPENDS_ON {dependency_type: 'app_to_app'}]->(a2:Application)
        MATCH (a1)-[:RUNS_ON]->(n1:Node), (a2)-[:RUNS_ON]->(n2:Node)
        WHERE n1 <> n2
        WITH n1, n2, sum(dep.weight) as total_weight
        MERGE (n1)-[d:DEPENDS_ON {dependency_type: 'node_to_node'}]->(n2)
        SET d.weight = total_weight
        RETURN count(d) as c
        """
        return self._run_count(query)

    def _derive_node_to_broker(self):
        """
        Node depends on Broker if an App on Node depends on that Broker.
        """
        query = """
        MATCH (a:Application)-[dep:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(b:Broker)
        MATCH (a)-[:RUNS_ON]->(n:Node)
        WITH n, b, sum(dep.weight) as total_weight
        MERGE (n)-[d:DEPENDS_ON {dependency_type: 'node_to_broker'}]->(b)
        SET d.weight = total_weight
        RETURN count(d) as c
        """
        return self._run_count(query)

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