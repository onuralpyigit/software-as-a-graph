"""
Graph Importer

Imports graph data into Neo4j and performs advanced post-processing:
1. Imports entities (Nodes, Brokers, Apps, Topics) and structural relationships.
2. Derives DEPENDS_ON relationships (App->App, Node->Node, etc.)
3. Calculates Relationship Weights based on Topic QoS and Message Size.
"""

import logging
from typing import Dict, Any, List
from neo4j import GraphDatabase

class GraphImporter:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password", database="neo4j"):
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
        self.logger.info(f"Starting import. Clear DB: {clear}")
        
        if clear: 
            self._run_query("MATCH (n) DETACH DELETE n")
            self._create_constraints()
        
        # 1. Import Basic Entities
        stats = {}
        stats["nodes"] = self._import_batch(data.get("nodes", []), "MERGE (n:Node {id: row.id}) SET n.name=row.name, n.weight=0.0")
        stats["brokers"] = self._import_batch(data.get("brokers", []), "MERGE (n:Broker {id: row.id}) SET n.name=row.name, n.weight=0.0")
        stats["apps"] = self._import_batch(data.get("applications", []), "MERGE (n:Application {id: row.id}) SET n.name=row.name, n.role=row.role, n.weight=0.0")
        stats["topics"] = self._import_topics(data.get("topics", []))
        
        # 2. Import Relationships
        rels = data.get("relationships", {})
        self._import_rels(rels.get("runs_on", []), "RUNS_ON") 
        self._import_rels(rels.get("routes", []), "ROUTES")
        self._import_rels(rels.get("publishes_to", []), "PUBLISHES_TO")
        self._import_rels(rels.get("subscribes_to", []), "SUBSCRIBES_TO")
        self._import_rels(rels.get("connects_to", []), "CONNECTS_TO") # Physical links

        # 3. Calculate Intrinsic Weights (Topics, Apps, Explicit Edges)
        self.logger.info("Calculating intrinsic weights...")
        self._calculate_intrinsic_weights()

        # 4. Derive DEPENDS_ON Relationships
        self.logger.info("Deriving DEPENDS_ON relationships...")
        stats.update(self._derive_dependencies())
        
        # 5. Calculate Component Criticality (Total Weights)
        self.logger.info("Calculating final component criticality...")
        self._calculate_component_weights()
        
        return stats

    def _create_constraints(self):
        with self.driver.session(database=self.database) as session:
            for label in ["Application", "Broker", "Topic", "Node"]:
                session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")

    def _import_batch(self, data: List[Dict], query: str, batch_size: int = 1000) -> int:
        if not data: return 0
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(data), batch_size):
                session.run(f"UNWIND $batch AS row {query}", batch=data[i:i+batch_size])
        return len(data)

    def _import_topics(self, data: List[Dict]) -> int:
        # Flatten QoS structure for easier Cypher ingestion
        flattened = []
        for t in data:
            qos = t.get("qos", {})
            flattened.append({
                "id": t["id"], 
                "name": t["name"], 
                "size": t.get("size", 256),
                "d": qos.get("durability", "VOLATILE"),
                "r": qos.get("reliability", "BEST_EFFORT"),
                "p": qos.get("transport_priority", "MEDIUM")
            })
        
        query = """
        MERGE (t:Topic {id: row.id})
        SET t.name = row.name, 
            t.size = row.size,
            t.qos_durability = row.d,
            t.qos_reliability = row.r,
            t.qos_transport_priority = row.p,
            t.weight = 0.0
        """
        return self._import_batch(flattened, query)

    def _import_rels(self, data: List[Dict], rel_type: str):
        query = f"""
        MATCH (a {{id: row.from}}), (b {{id: row.to}})
        MERGE (a)-[:{rel_type}]->(b)
        """
        return self._import_batch(data, query)

    def _get_qos_weight_cypher(self, topic_var: str) -> str:
        """
        Returns the Cypher fragment to calculate weight based on QoS and Size.
        
        Weight = Reliability + Durability + Priority + Size_Score
        
        Size Score uses logarithmic scaling with soft cap to prevent
        large messages from dominating the weight calculation:
            S_size = min(log2(1 + size/1024) / 10, 1.0)
        
        This normalizes typical message sizes to [0, 1] range:
            - 256 bytes  → 0.032
            - 1 KB       → 0.100
            - 8 KB       → 0.317
            - 64 KB      → 0.604
            - 250 KB     → 0.800
            - 1 MB+      → 1.000 (capped)
        """
        return f"""
        (CASE {topic_var}.qos_reliability WHEN 'RELIABLE' THEN 0.3 ELSE 0.0 END +
         CASE {topic_var}.qos_durability WHEN 'PERSISTENT' THEN 0.4 WHEN 'TRANSIENT' THEN 0.25 WHEN 'TRANSIENT_LOCAL' THEN 0.2 ELSE 0.0 END +
         CASE {topic_var}.qos_transport_priority WHEN 'URGENT' THEN 0.3 WHEN 'HIGH' THEN 0.2 WHEN 'MEDIUM' THEN 0.1 ELSE 0.0 END +
         CASE WHEN {topic_var}.size <= 0 THEN 0.0
                WHEN (log(1 + {topic_var}.size / 1024.0) / (log(2) * 10)) > 1.0 THEN 1.0
                ELSE (log(1 + {topic_var}.size / 1024.0) / (log(2) * 10))
         END)
        """

    def _calculate_intrinsic_weights(self):
        """
        Calculates weights for Topics based on QoS and Size.
        Propagates these weights to explicit edges (PUBLISHES_TO/SUBSCRIBES_TO)
        and upstream components (Apps, Brokers, Nodes).
        """
        qos_calc = self._get_qos_weight_cypher("t")

        # 1. Topic Weight
        self._run_query(f"""
            MATCH (t:Topic)
            SET t.weight = {qos_calc}
        """)

        # 2. Edge Weights (Inherit from Topic)
        self._run_query("""
            MATCH ()-[r:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
            SET r.weight = t.weight
        """)
        
        # 2b. ROUTES Edge Weights (Broker -> Topic)
        self._run_query("""
            MATCH ()-[r:ROUTES]->(t:Topic)
            SET r.weight = t.weight
        """)
        
        # 3. Application Weight (Sum of Topic Weights it interacts with)
        self._run_query("""
            MATCH (a:Application)
            OPTIONAL MATCH (a)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
            WITH a, coalesce(sum(t.weight), 0.0) as load_weight
            SET a.weight = load_weight
        """)

        # 4. Broker Weight (Sum of Routed Topic Weights)
        self._run_query("""
            MATCH (b:Broker)
            OPTIONAL MATCH (b)-[:ROUTES]->(t:Topic)
            WITH b, coalesce(sum(t.weight), 0.0) as routed_weight
            SET b.weight = routed_weight
        """)

        # 5. Node Weight (Sum of Hosted App and Broker Weights)
        self._run_query("""
            MATCH (n:Node)
            OPTIONAL MATCH (a:Application)-[:RUNS_ON]->(n)
            OPTIONAL MATCH (b:Broker)-[:RUNS_ON]->(n)
            WITH n, coalesce(sum(a.weight), 0.0) + coalesce(sum(b.weight), 0.0) as host_weight
            SET n.weight = host_weight
        """)

    def _derive_dependencies(self) -> Dict[str, int]:
        stats = {}
        qos_calc = self._get_qos_weight_cypher("t")
        
        # A. App->App
        # Logic: Sub depends on Pub via shared Topic.
        # Weight = Count(Topics) + Sum(Topic_Weight)
        query_app_app = f"""
        MATCH (pub:Application)-[:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(sub:Application)
        WHERE pub <> sub
        WITH pub, sub, t, {qos_calc} as importance
        WITH pub, sub, count(t) as shared_count, sum(importance) as importance_sum
        MERGE (sub)-[d:DEPENDS_ON {{dependency_type: 'app_to_app'}}]->(pub)
        SET d.weight = shared_count + importance_sum
        RETURN count(d) as c
        """
        stats["deps_app_app"] = self._run_count(query_app_app)

        # B. App->Broker
        # Logic: App depends on Broker if Broker routes a topic the App Publishes/Subscribes to.
        query_app_broker = f"""
        MATCH (app:Application)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)<-[:ROUTES]-(b:Broker)
        WITH app, b, t, {qos_calc} as importance
        WITH app, b, count(t) as topics, sum(importance) as importance_sum
        MERGE (app)-[d:DEPENDS_ON {{dependency_type: 'app_to_broker'}}]->(b)
        SET d.weight = topics + importance_sum
        RETURN count(d) as c
        """
        stats["deps_app_broker"] = self._run_count(query_app_broker)

        # C. Node->Node
        # Logic: Node A depends on Node B if an App on A depends on an App on B.
        # Weight aggregates the underlying App dependencies.
        query_node_node = """
        MATCH (a1:Application)-[dep:DEPENDS_ON {dependency_type: 'app_to_app'}]->(a2:Application)
        MATCH (a1)-[:RUNS_ON]->(n1:Node), (a2)-[:RUNS_ON]->(n2:Node)
        WHERE n1 <> n2
        WITH n1, n2, sum(dep.weight) as total_weight
        MERGE (n1)-[d:DEPENDS_ON {dependency_type: 'node_to_node'}]->(n2)
        SET d.weight = total_weight
        RETURN count(d) as c
        """
        stats["deps_node_node"] = self._run_count(query_node_node)

        # D. Node->Broker
        # Logic: Node depends on Broker if hosted App depends on Broker.
        query_node_broker = """
        MATCH (a:Application)-[dep:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(b:Broker)
        MATCH (a)-[:RUNS_ON]->(n:Node)
        WITH n, b, sum(dep.weight) as total_weight
        MERGE (n)-[d:DEPENDS_ON {dependency_type: 'node_to_broker'}]->(b)
        SET d.weight = total_weight
        RETURN count(d) as c
        """
        stats["deps_node_broker"] = self._run_count(query_node_broker)

        return stats

    def _calculate_component_weights(self):
        """
        Calculates the final importance weight for every component.
        Final Weight = Intrinsic Weight + Structural Centrality (In-Degree + Out-Degree Weights)
        """
        query = """
        MATCH (n) WHERE n:Application OR n:Node OR n:Broker
        OPTIONAL MATCH (n)-[out:DEPENDS_ON]->()
        OPTIONAL MATCH ()-[inc:DEPENDS_ON]->(n)
        WITH n, 
             n.weight as intrinsic_weight, 
             coalesce(sum(out.weight), 0) + coalesce(sum(inc.weight), 0) as centrality_score
        SET n.weight = intrinsic_weight + centrality_score,
            n.intrinsic_weight = intrinsic_weight,
            n.centrality_score = centrality_score
        """
        self._run_query(query)

    def _run_query(self, query):
        with self.driver.session(database=self.database) as session:
            session.run(query)

    def _run_count(self, query):
        with self.driver.session(database=self.database) as session:
            res = session.run(query)
            try:
                return res.single()["c"]
            except (TypeError, KeyError):
                return 0