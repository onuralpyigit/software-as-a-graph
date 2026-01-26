"""
Graph Importer

Imports graph data into Neo4j and performs advanced post-processing:
1. Imports entities (Nodes, Brokers, Apps, Topics, Libraries) and structural relationships.
2. Derives DEPENDS_ON relationships (App->App, Node->Node, etc.)
3. Calculates Relationship Weights based on Topic QoS and Message Size.

The importer uses the QoS scoring constants from QoSPolicy for consistent
weight calculations between Python and Cypher.
"""

import logging
from typing import Dict, Any, List, Optional

from neo4j import GraphDatabase

from .graph_model import QoSPolicy


class GraphImporter:
    """Imports graph data into Neo4j with dependency derivation and weight calculation."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.logger = logging.getLogger(__name__)

    def __enter__(self) -> "GraphImporter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self.driver.close()

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
        stats["apps"] = self._import_batch(data.get("applications", []), "MERGE (n:Application {id: row.id}) SET n.name=row.name, n.role=row.role, n.app_type=row.app_type, n.version=row.version, n.criticality=row.criticality, n.weight=0.0")
        stats["libraries"] = self._import_batch(data.get("libraries", []), "MERGE (n:Library {id: row.id}) SET n.name=row.name, n.version=row.version, n.weight=0.0")
        stats["topics"] = self._import_topics(data.get("topics", []))
        
        # 2. Import Relationships
        rels = data.get("relationships", {})
        self._import_rels(rels.get("runs_on", []), "RUNS_ON") 
        self._import_rels(rels.get("routes", []), "ROUTES")
        self._import_rels(rels.get("publishes_to", []), "PUBLISHES_TO")
        self._import_rels(rels.get("subscribes_to", []), "SUBSCRIBES_TO")
        self._import_rels(rels.get("connects_to", []), "CONNECTS_TO")
        self._import_rels(rels.get("uses", []), "USES")

        # 3. Calculate Intrinsic Weights (Topics, Apps, Libs, Explicit Edges)
        self.logger.info("Calculating intrinsic weights...")
        self._calculate_intrinsic_weights()

        # 4. Derive DEPENDS_ON Relationships
        self.logger.info("Deriving DEPENDS_ON relationships...")
        stats.update(self._derive_dependencies())
        
        # 5. Calculate Component Criticality (Total Weights)
        self.logger.info("Calculating final component criticality...")
        self._calculate_component_weights()
        
        return stats

    def _create_constraints(self) -> None:
        """Create uniqueness constraints on node IDs."""
        with self.driver.session(database=self.database) as session:
            for label in ["Application", "Broker", "Topic", "Node", "Library"]:
                session.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
                )

    def _import_batch(
        self, data: List[Dict[str, Any]], query: str, batch_size: int = 1000
    ) -> int:
        """Import data in batches using the given Cypher query."""
        if not data:
            return 0
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(data), batch_size):
                session.run(f"UNWIND $batch AS row {query}", batch=data[i : i + batch_size])
        return len(data)

    def _import_topics(self, data: List[Dict[str, Any]]) -> int:
        """Import topics with flattened QoS structure."""
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

    def _import_rels(self, data: List[Dict[str, str]], rel_type: str) -> int:
        """Import relationships of the given type."""
        query = f"""
        MATCH (a {{id: row.from}}), (b {{id: row.to}})
        MERGE (a)-[:{rel_type}]->(b)
        """
        return self._import_batch(data, query)

    def _get_qos_weight_cypher(self, topic_var: str) -> str:
        """Generate Cypher expression for QoS weight calculation.
        
        Uses the same scoring constants as QoSPolicy.calculate_weight() to ensure
        consistency between Python and Cypher calculations.
        """
        # Build CASE expressions from QoSPolicy scoring constants
        rel_scores = QoSPolicy.RELIABILITY_SCORES
        dur_scores = QoSPolicy.DURABILITY_SCORES
        pri_scores = QoSPolicy.PRIORITY_SCORES
        
        return f"""
        (CASE {topic_var}.qos_reliability WHEN 'RELIABLE' THEN {rel_scores['RELIABLE']} ELSE 0.0 END +
         CASE {topic_var}.qos_durability 
             WHEN 'PERSISTENT' THEN {dur_scores['PERSISTENT']} 
             WHEN 'TRANSIENT' THEN {dur_scores['TRANSIENT']} 
             WHEN 'TRANSIENT_LOCAL' THEN {dur_scores['TRANSIENT_LOCAL']} 
             ELSE 0.0 END +
         CASE {topic_var}.qos_transport_priority 
             WHEN 'URGENT' THEN {pri_scores['URGENT']} 
             WHEN 'HIGH' THEN {pri_scores['HIGH']} 
             WHEN 'MEDIUM' THEN {pri_scores['MEDIUM']} 
             ELSE 0.0 END +
         CASE WHEN {topic_var}.size <= 0 THEN 0.0
              WHEN (log(1 + {topic_var}.size / 1024.0) / (log(2) * 10)) > 1.0 THEN 1.0
              ELSE (log(1 + {topic_var}.size / 1024.0) / (log(2) * 10))
         END)
        """

    def _calculate_intrinsic_weights(self) -> None:
        """
        Calculates weights for Topics based on QoS and Size.
        Propagates these weights to explicit edges and upstream components.
        """
        qos_calc = self._get_qos_weight_cypher("t")

        # 1. Topic Weight
        self._run_query(f"""
            MATCH (t:Topic)
            SET t.weight = {qos_calc}
        """)

        # 2. Edge Weights (Inherit from Topic) - Covers both App and Library edges
        self._run_query("""
            MATCH ()-[r:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
            SET r.weight = t.weight
        """)
        
        # 3. ROUTES Edge Weights (Broker -> Topic)
        self._run_query("""
            MATCH ()-[r:ROUTES]->(t:Topic)
            SET r.weight = t.weight
        """)
        
        # 4. Application Weight (Sum of Topic Weights + Library Weights via USES)
        self._run_query("""
            MATCH (a:Application)
            OPTIONAL MATCH (a)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
            OPTIONAL MATCH (a)-[:USES]->(l:Library)
            WITH a, coalesce(sum(DISTINCT t.weight), 0.0) as topic_weight, 
                 coalesce(sum(DISTINCT l.weight), 0.0) as lib_weight
            SET a.weight = topic_weight + lib_weight
        """)


        # 5. Library Weight (Sum of Topic Weights it interacts with)
        self._run_query("""
            MATCH (l:Library)
            OPTIONAL MATCH (l)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
            WITH l, coalesce(sum(t.weight), 0.0) as load_weight
            SET l.weight = load_weight
        """)

        # 6. USES Edge Weights (App/Library -> Library)
        self._run_query("""
            MATCH ()-[r:USES]->(l:Library)
            SET r.weight = l.weight
        """)

        # 7. Broker Weight (Sum of Routed Topic Weights)
        self._run_query("""
            MATCH (b:Broker)
            OPTIONAL MATCH (b)-[:ROUTES]->(t:Topic)
            WITH b, coalesce(sum(t.weight), 0.0) as routed_weight
            SET b.weight = routed_weight
        """)

        # 8. Node Weight (Sum of Hosted App and Broker Weights)
        self._run_query("""
            MATCH (n:Node)
            OPTIONAL MATCH (a:Application)-[:RUNS_ON]->(n)
            OPTIONAL MATCH (b:Broker)-[:RUNS_ON]->(n)
            WITH n, coalesce(sum(a.weight), 0.0) + coalesce(sum(b.weight), 0.0) as host_weight
            SET n.weight = host_weight
        """)

    def _derive_dependencies(self) -> Dict[str, int]:
        """Derive DEPENDS_ON relationships from structural graph patterns."""
        stats = {}
        qos_calc = self._get_qos_weight_cypher("t")
        
        # A. App->App (direct)
        # Pattern: pub:App -[:PUBLISHES_TO]-> Topic <-[:SUBSCRIBES_TO]- sub:App
        query_app_app = f"""
        MATCH (pub:Application)-[:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(sub:Application)
        WHERE pub <> sub
        WITH pub, sub, t, {qos_calc} as importance
        WITH pub, sub, count(t) as shared_count, sum(importance) as importance_sum
        MERGE (sub)-[d:DEPENDS_ON {{dependency_type: 'app_to_app'}}]->(pub)
        SET d.weight = shared_count + importance_sum
        RETURN count(d) as c
        """
        count_deps_app_app = self._run_count(query_app_app)

        # B. App->Lib->App (publisher via library)
        # Pattern: appA -[:USES]-> lib -[:PUBLISHES_TO]-> Topic <-[:SUBSCRIBES_TO]- appB
        query_app_lib_app_pub = f"""
        MATCH (appA:Application)-[:USES]->(lib:Library)-[:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(appB:Application)
        WHERE appA <> appB
        WITH appA, appB, t, {qos_calc} as importance
        WITH appA, appB, count(t) as shared_count, sum(importance) as importance_sum
        MERGE (appB)-[d:DEPENDS_ON {{dependency_type: 'app_to_app'}}]->(appA)
        ON CREATE SET d.weight = shared_count + importance_sum
        ON MATCH SET d.weight = d.weight + shared_count + importance_sum
        RETURN count(d) as c
        """
        count_deps_app_lib_app_pub = self._run_count(query_app_lib_app_pub)

        # C. App->Lib->App (subscriber via library)
        # Pattern: appA -[:PUBLISHES_TO]-> Topic <-[:SUBSCRIBES_TO]- lib <-[:USES]- appB
        query_app_lib_app_sub = f"""
        MATCH (appA:Application)-[:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(lib:Library)<-[:USES]-(appB:Application)
        WHERE appA <> appB
        WITH appA, appB, t, {qos_calc} as importance
        WITH appA, appB, count(t) as shared_count, sum(importance) as importance_sum
        MERGE (appB)-[d:DEPENDS_ON {{dependency_type: 'app_to_app'}}]->(appA)
        ON CREATE SET d.weight = shared_count + importance_sum
        ON MATCH SET d.weight = d.weight + shared_count + importance_sum
        RETURN count(d) as c
        """
        count_deps_app_lib_app_sub = self._run_count(query_app_lib_app_sub)

        # D. App->Lib->Lib->App (two-level library chain, publisher side)
        # Pattern: appA -[:USES]-> lib1 -[:USES]-> lib2 -[:PUBLISHES_TO]-> Topic <-[:SUBSCRIBES_TO]- appB
        query_app_lib_lib_app_pub = f"""
        MATCH (appA:Application)-[:USES]->(lib1:Library)-[:USES]->(lib2:Library)-[:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(appB:Application)
        WHERE appA <> appB
        WITH appA, appB, t, {qos_calc} as importance
        WITH appA, appB, count(t) as shared_count, sum(importance) as importance_sum
        MERGE (appB)-[d:DEPENDS_ON {{dependency_type: 'app_to_app'}}]->(appA)
        ON CREATE SET d.weight = shared_count + importance_sum
        ON MATCH SET d.weight = d.weight + shared_count + importance_sum
        RETURN count(d) as c
        """
        count_deps_app_lib_lib_app_pub = self._run_count(query_app_lib_lib_app_pub)

        # E. App->Lib->Lib->App (two-level library chain, subscriber side)
        # Pattern: appA -[:PUBLISHES_TO]-> Topic <-[:SUBSCRIBES_TO]- lib2 <-[:USES]- lib1 <-[:USES]- appB
        query_app_lib_lib_app_sub = f"""
        MATCH (appA:Application)-[:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(lib2:Library)<-[:USES]-(lib1:Library)<-[:USES]-(appB:Application)
        WHERE appA <> appB
        WITH appA, appB, t, {qos_calc} as importance
        WITH appA, appB, count(t) as shared_count, sum(importance) as importance_sum
        MERGE (appB)-[d:DEPENDS_ON {{dependency_type: 'app_to_app'}}]->(appA)
        ON CREATE SET d.weight = shared_count + importance_sum
        ON MATCH SET d.weight = d.weight + shared_count + importance_sum
        RETURN count(d) as c
        """
        count_deps_app_lib_lib_app_sub = self._run_count(query_app_lib_lib_app_sub)

        # F. App->Lib->Lib->App (mixed: publisher via lib, subscriber via lib)
        # Pattern: appA -[:USES]-> lib1 -[:PUBLISHES_TO]-> Topic <-[:SUBSCRIBES_TO]- lib2 <-[:USES]- appB
        query_app_lib_lib_app_mixed = f"""
        MATCH (appA:Application)-[:USES]->(lib1:Library)-[:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(lib2:Library)<-[:USES]-(appB:Application)
        WHERE appA <> appB
        WITH appA, appB, t, {qos_calc} as importance
        WITH appA, appB, count(t) as shared_count, sum(importance) as importance_sum
        MERGE (appB)-[d:DEPENDS_ON {{dependency_type: 'app_to_app'}}]->(appA)
        ON CREATE SET d.weight = shared_count + importance_sum
        ON MATCH SET d.weight = d.weight + shared_count + importance_sum
        RETURN count(d) as c
        """
        count_deps_app_lib_lib_app_mixed = self._run_count(query_app_lib_lib_app_mixed)

        stats["deps_app_app"] = (
            count_deps_app_app +
            count_deps_app_lib_app_pub +
            count_deps_app_lib_app_sub +
            count_deps_app_lib_lib_app_pub +
            count_deps_app_lib_lib_app_sub +
            count_deps_app_lib_lib_app_mixed
        )

        # B. App->Broker
        query_app_broker = f"""
        MATCH (app:Application)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)<-[:ROUTES]-(b:Broker)
        WITH app, b, t, {qos_calc} as importance
        WITH app, b, count(t) as topics, sum(importance) as importance_sum
        MERGE (app)-[d:DEPENDS_ON {{dependency_type: 'app_to_broker'}}]->(b)
        SET d.weight = topics + importance_sum
        RETURN count(d) as c
        """
        count_deps_app_broker = self._run_count(query_app_broker)

        # B. App->Broker (indirect via library)
        query_app_broker_via_lib = f"""
        MATCH (app:Application)-[:USES]->(lib:Library)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)<-[:ROUTES]-(b:Broker)
        WITH app, b, t, {qos_calc} as importance
        WITH app, b, count(t) as topics, sum(importance) as importance_sum
        MERGE (app)-[d:DEPENDS_ON {{dependency_type: 'app_to_broker'}}]->(b)
        SET d.weight = topics + importance_sum
        RETURN count(d) as c
        """
        count_deps_app_broker_via_lib = self._run_count(query_app_broker_via_lib)

        stats["deps_app_broker"] = count_deps_app_broker + count_deps_app_broker_via_lib

        # C. Node->Node
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

    def _calculate_component_weights(self) -> None:
        """
        Calculates the final importance weight for every component (Apps, Nodes, Brokers, Libraries).
        """
        query = """
        MATCH (n) WHERE n:Application OR n:Node OR n:Broker OR n:Library
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

    def _run_query(self, query: str) -> None:
        """Execute a Cypher query."""
        with self.driver.session(database=self.database) as session:
            session.run(query)

    def _run_count(self, query: str) -> int:
        """Execute a Cypher query and return the count result."""
        with self.driver.session(database=self.database) as session:
            res = session.run(query)
            try:
                return res.single()["c"]
            except (TypeError, KeyError):
                return 0