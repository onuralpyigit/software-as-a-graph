from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import asdict

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.models import QoSPolicy
from src.models import GraphData, ComponentData, EdgeData

# Layer definitions for filtering
LAYER_DEFINITIONS = {
    "application": {
        "name": "Application Layer",
        "component_types": ["Application"],
        "dependency_types": ["app_to_app"],
    },
    "infrastructure": {
        "name": "Infrastructure Layer",
        "component_types": ["Node"],
        "dependency_types": ["node_to_node"],
    },
    "app_broker": {
        "name": "Application-Broker Layer",
        "component_types": ["Application", "Broker"],
        "dependency_types": ["app_to_broker"],
    },
    "complete": {
        "name": "Complete System",
        "component_types": ["Application", "Broker", "Node", "Topic", "Library"],
        "dependency_types": ["app_to_app", "node_to_node", "app_to_broker", "node_to_broker"],
    },
}

class GraphRepository:
    """
    Neo4j adapter for GraphRepository port.
    Combines functionality of GraphImporter and GraphExporter.
    """

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

    def __enter__(self) -> "GraphRepository":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self.driver.close()

    # ==========================================
    # Import Methods (Save)
    # ==========================================

    def save_graph(self, data: Dict[str, Any], clear: bool = False) -> None:
        """Import graph data into the repository."""
        self.logger.info(f"Starting import. Clear DB: {clear}")
        
        if clear: 
            self._run_query("MATCH (n) DETACH DELETE n")
            self._create_constraints()
        
        # 1. Import Basic Entities
        self._import_batch(data.get("nodes", []), "MERGE (n:Node {id: row.id}) SET n.name=row.name, n.weight=0.0")
        self._import_batch(data.get("brokers", []), "MERGE (n:Broker {id: row.id}) SET n.name=row.name, n.weight=0.0")
        self._import_batch(data.get("applications", []), "MERGE (n:Application {id: row.id}) SET n.name=row.name, n.role=row.role, n.app_type=row.app_type, n.version=row.version, n.criticality=row.criticality, n.weight=0.0")
        self._import_batch(data.get("libraries", []), "MERGE (n:Library {id: row.id}) SET n.name=row.name, n.version=row.version, n.weight=0.0")
        self._import_topics(data.get("topics", []))
        
        # 2. Import Relationships
        rels = data.get("relationships", {})
        self._import_rels(rels.get("runs_on", []), "RUNS_ON") 
        self._import_rels(rels.get("routes", []), "ROUTES")
        self._import_rels(rels.get("publishes_to", []), "PUBLISHES_TO")
        self._import_rels(rels.get("subscribes_to", []), "SUBSCRIBES_TO")
        self._import_rels(rels.get("connects_to", []), "CONNECTS_TO")
        self._import_rels(rels.get("uses", []), "USES")

        # 3. Calculate Intrinsic Weights
        self.logger.info("Calculating intrinsic weights...")
        self._calculate_intrinsic_weights()

        # 4. Derive DEPENDS_ON Relationships
        self.logger.info("Deriving DEPENDS_ON relationships...")
        self._derive_dependencies()
        
        # 5. Calculate Component Criticality
        self.logger.info("Calculating final component criticality...")
        self._calculate_component_weights()

    def _create_constraints(self) -> None:
        """Create uniqueness constraints on node IDs."""
        with self.driver.session(database=self.database) as session:
            for label in ["Application", "Broker", "Topic", "Node", "Library"]:
                session.run(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
                )

    def _import_batch(self, data: List[Dict[str, Any]], query: str, batch_size: int = 1000) -> int:
        """Import data in batches using the given Cypher query."""
        if not data:
            return 0
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(data), batch_size):
                session.run(f"UNWIND $batch AS row {query}", batch=data[i : i + batch_size])
        return len(data)

    def _import_topics(self, data: List[Dict[str, Any]]) -> int:
        """Import topics with flattened QoS structure."""
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
        """Generate Cypher expression for QoS weight calculation."""
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
        """Calculates weights for Topics based on QoS and Size."""
        qos_calc = self._get_qos_weight_cypher("t")

        # 1. Topic Weight
        self._run_query(f"MATCH (t:Topic) SET t.weight = {qos_calc}")

        # 2. Edge Weights (Inherit from Topic)
        self._run_query("MATCH ()-[r:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic) SET r.weight = t.weight")
        
        # 3. ROUTES Edge Weights
        self._run_query("MATCH ()-[r:ROUTES]->(t:Topic) SET r.weight = t.weight")
        
        # 4. Application Weight
        self._run_query("""
            MATCH (a:Application)
            OPTIONAL MATCH (a)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
            OPTIONAL MATCH (a)-[:USES]->(l:Library)
            WITH a, coalesce(sum(DISTINCT t.weight), 0.0) as topic_weight, 
                 coalesce(sum(DISTINCT l.weight), 0.0) as lib_weight
            SET a.weight = topic_weight + lib_weight
        """)

        # 5. Library Weight
        self._run_query("""
            MATCH (l:Library)
            OPTIONAL MATCH (l)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
            WITH l, coalesce(sum(t.weight), 0.0) as load_weight
            SET l.weight = load_weight
        """)

        # 6. USES Edge Weights
        self._run_query("MATCH ()-[r:USES]->(l:Library) SET r.weight = l.weight")

        # 7. Broker Weight
        self._run_query("""
            MATCH (b:Broker)
            OPTIONAL MATCH (b)-[:ROUTES]->(t:Topic)
            WITH b, coalesce(sum(t.weight), 0.0) as routed_weight
            SET b.weight = routed_weight
        """)

        # 8. Node Weight
        self._run_query("""
            MATCH (n:Node)
            OPTIONAL MATCH (a:Application)-[:RUNS_ON]->(n)
            OPTIONAL MATCH (b:Broker)-[:RUNS_ON]->(n)
            WITH n, coalesce(sum(a.weight), 0.0) + coalesce(sum(b.weight), 0.0) as host_weight
            SET n.weight = host_weight
        """)

        # 9. RUNS_ON Edge Weights
        self._run_query("MATCH (a:Application)-[r:RUNS_ON]->(n:Node) SET r.weight = a.weight")

        # 10. CONNECTS_TO Edge Weights
        self._run_query("MATCH (n1:Node)-[r:CONNECTS_TO]->(n2:Node) SET r.weight = n1.weight + n2.weight")

    def _derive_dependencies(self) -> None:
        """Derive DEPENDS_ON relationships."""
        qos_calc = self._get_qos_weight_cypher("t")
        
        # App->App (direct)
        # Pattern: pub:App -[:PUBLISHES_TO]-> Topic <-[:SUBSCRIBES_TO]- sub:App
        self._run_query(f"""
        MATCH (pub:Application)-[:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(sub:Application)
        WHERE pub <> sub
        WITH pub, sub, t, {qos_calc} as importance
        WITH pub, sub, collect(t.id) as via_ids, count(t) as shared_count, sum(importance) as importance_sum
        MERGE (sub)-[d:DEPENDS_ON {{dependency_type: 'app_to_app'}}]->(pub)
        SET d.weight = shared_count + importance_sum,
            d.via = via_ids
        RETURN count(d) as c
        """)

        # App->Lib->App (publisher via library)
        # Pattern: appA -[:USES*]-> lib -[:PUBLISHES_TO]-> Topic <-[:SUBSCRIBES_TO]- appB
        self._run_query(f"""
        MATCH (appA:Application)-[:USES]->(lib:Library)-[:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(appB:Application)
        WHERE appA <> appB
        WITH appA, appB, t, {qos_calc} as importance
        WITH appA, appB, collect(t.id) as via_ids, count(t) as shared_count, sum(importance) as importance_sum
        MERGE (appB)-[d:DEPENDS_ON {{dependency_type: 'app_to_app'}}]->(appA)
        ON CREATE SET d.weight = shared_count + importance_sum,
                      d.via = via_ids
        ON MATCH SET d.weight = d.weight + shared_count + importance_sum,
                     d.via = d.via + [x IN via_ids WHERE NOT x IN d.via]
        RETURN count(d) as c
        """)

        # App->Lib->App (subscriber via library)
        # Pattern: appA -[:PUBLISHES_TO]-> Topic <-[:SUBSCRIBES_TO]- lib <-[:USES*]-(appB:Application)
        self._run_query(f"""
        MATCH (appA:Application)-[:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(lib:Library)<-[:USES]-(appB:Application)
        WHERE appA <> appB
        WITH appA, appB, t, {qos_calc} as importance
        WITH appA, appB, collect(t.id) as via_ids, count(t) as shared_count, sum(importance) as importance_sum
        MERGE (appB)-[d:DEPENDS_ON {{dependency_type: 'app_to_app'}}]->(appA)
        ON CREATE SET d.weight = shared_count + importance_sum,
                      d.via = via_ids
        ON MATCH SET d.weight = d.weight + shared_count + importance_sum,
                     d.via = d.via + [x IN via_ids WHERE NOT x IN d.via]
        RETURN count(d) as c
        """)

        # App->Lib->Lib->App (mixed: publisher via lib, subscriber via lib)
        # Pattern: appA -[:USES]-> lib1 -[:PUBLISHES_TO]-> Topic <-[:SUBSCRIBES_TO]- lib2 <-[:USES]- appB
        self._run_query(f"""
        MATCH (appA:Application)-[:USES]->(lib1:Library)-[:PUBLISHES_TO]->(t:Topic)<-[:SUBSCRIBES_TO]-(lib2:Library)<-[:USES]-(appB:Application)
        WHERE appA <> appB
        WITH appA, appB, t, {qos_calc} as importance
        WITH appA, appB, collect(t.id) as via_ids, count(t) as shared_count, sum(importance) as importance_sum
        MERGE (appB)-[d:DEPENDS_ON {{dependency_type: 'app_to_app'}}]->(appA)
        ON CREATE SET d.weight = shared_count + importance_sum,
                      d.via = via_ids
        ON MATCH SET d.weight = d.weight + shared_count + importance_sum,
                     d.via = d.via + [x IN via_ids WHERE NOT x IN d.via]
        RETURN count(d) as c
        """)

        # App->Broker
        # Pattern: appA -[:PUBLISHES_TO|SUBSCRIBES_TO]-> Topic <-[:ROUTES]- broker
        self._run_query(f"""
        MATCH (appA:Application)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)<-[:ROUTES]-(broker:Broker)
        WHERE appA <> broker
        WITH appA, broker, t, {qos_calc} as importance
        WITH appA, broker, collect(t.id) as via_ids, count(t) as shared_count, sum(importance) as importance_sum
        MERGE (appA)-[d:DEPENDS_ON {{dependency_type: 'app_to_broker'}}]->(broker)
        ON CREATE SET d.weight = shared_count + importance_sum,
                      d.via = via_ids
        ON MATCH SET d.weight = d.weight + shared_count + importance_sum,
                     d.via = d.via + [x IN via_ids WHERE NOT x IN d.via]
        RETURN count(d) as c
        """)

        # App->Broker (indirect via library)
        # Pattern: appA -[:USES*]-> lib -[:PUBLISHES_TO| SUBSCRIBES_TO]-> Topic <-[:ROUTES]- broker
        self._run_query(f"""
        MATCH (appA:Application)-[:USES*]->(lib:Library)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)<-[:ROUTES]-(broker:Broker)
        WHERE appA <> broker
        WITH appA, broker, t, {qos_calc} as importance
        WITH appA, broker, collect(t.id) as via_ids, count(t) as shared_count, sum(importance) as importance_sum
        MERGE (appA)-[d:DEPENDS_ON {{dependency_type: 'app_to_broker'}}]->(broker)
        ON CREATE SET d.weight = shared_count + importance_sum,
                      d.via = via_ids
        ON MATCH SET d.weight = d.weight + shared_count + importance_sum,
                     d.via = d.via + [x IN via_ids WHERE NOT x IN d.via]
        RETURN count(d) as c
        """)

        # Node->Node
        # Pattern: appA -[:RUNS_ON]-> nodeA -[:DEPENDS_ON]-> nodeB <-[:RUNS_ON]- appB 
        self._run_query("""
        MATCH (a1:Application)-[dep:DEPENDS_ON {dependency_type: 'app_to_app'}]->(a2:Application)
        MATCH (a1)-[:RUNS_ON]->(n1:Node), (a2)-[:RUNS_ON]->(n2:Node)
        WHERE n1 <> n2
        WITH n1, n2, sum(dep.weight) as total_weight, collect(a1.id + "->" + a2.id) as via_ids
        MERGE (n1)-[d:DEPENDS_ON {dependency_type: 'node_to_node'}]->(n2)
        SET d.weight = total_weight,
            d.via = via_ids
        RETURN count(d) as c
        """)
        
        # Node->Broker
        # Pattern: appA -[:RUNS_ON]-> nodeA -[:DEPENDS_ON]-> broker <-[:RUNS_ON]- appB 
        self._run_query("""
        MATCH (a:Application)-[dep:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(b:Broker)
        MATCH (a)-[:RUNS_ON]->(n:Node)
        WITH n, b, sum(dep.weight) as total_weight, collect(a.id) as via_ids
        MERGE (n)-[d:DEPENDS_ON {dependency_type: 'node_to_broker'}]->(b)
        SET d.weight = total_weight,
            d.via = via_ids
        RETURN count(d) as c
        """)

    def _calculate_component_weights(self) -> None:
        """Calculates the final importance weight for components."""
        self._run_query("""
        MATCH (n) WHERE n:Application OR n:Node OR n:Broker OR n:Library
        OPTIONAL MATCH (n)-[out:DEPENDS_ON]->()
        OPTIONAL MATCH ()-[inc:DEPENDS_ON]->(n)
        WITH n, 
             n.weight as intrinsic_weight, 
             coalesce(sum(out.weight), 0) + coalesce(sum(inc.weight), 0) as centrality_score
        SET n.weight = intrinsic_weight + centrality_score
        """)

    def _run_query(self, query: str) -> None:
        """Execute a Cypher query."""
        with self.driver.session(database=self.database) as session:
            session.run(query)



    # ==========================================
    # Retrieval Methods (Getters)
    # ==========================================

    def get_graph_data(
        self,
        component_types: Optional[List[str]] = None,
        dependency_types: Optional[List[str]] = None,
        include_raw: bool = False,
    ) -> GraphData:
        """Retrieve graph data with optional type filtering."""
        
        all_component_types = ["Application", "Broker", "Node", "Topic", "Library"]
        all_dependency_types = ["app_to_app", "node_to_node", "app_to_broker", "node_to_broker"]
        
        target_comp_types = component_types or all_component_types
        target_dep_types = dependency_types or all_dependency_types
        
        components = self._get_components(target_comp_types)
        component_ids = {c.id for c in components}
        
        edges = self._get_edges(target_dep_types, component_ids)
        
        if include_raw:
            raw_edges = self._get_raw_edges(component_ids)
            edges.extend(raw_edges)
        
        return GraphData(components=components, edges=edges)

    def _get_components(self, types: List[str]) -> List[ComponentData]:
        types_str = ", ".join(f"'{t}'" for t in types)
        query = f"""
        MATCH (n)
        WHERE any(label IN labels(n) WHERE label IN [{types_str}])
        RETURN n.id as id, labels(n)[0] as type, coalesce(n.weight, 1.0) as weight, properties(n) as props
        """
        
        components = []
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            for record in result:
                props = dict(record["props"])
                props.pop("id", None)
                props.pop("weight", None)
                components.append(ComponentData(
                    id=record["id"],
                    component_type=record["type"],
                    weight=float(record["weight"]),
                    properties=props
                ))
        return components

    def _get_edges(self, types: List[str], valid_ids: Set[str]) -> List[EdgeData]:
        types_str = ", ".join(f"'{t}'" for t in types)
        query = f"""
        MATCH (source)-[r:DEPENDS_ON]->(target)
        WHERE r.dependency_type IN [{types_str}]
        RETURN source.id as source_id, target.id as target_id, labels(source)[0] as source_type,
               labels(target)[0] as target_type, r.dependency_type as dependency_type,
               coalesce(r.weight, 1.0) as weight, properties(r) as props
        """
        
        edges = []
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            for record in result:
                if valid_ids and (record["source_id"] not in valid_ids or record["target_id"] not in valid_ids):
                    continue
                
                props = dict(record["props"])
                props.pop("dependency_type", None)
                props.pop("weight", None)
                edges.append(EdgeData(
                    source_id=record["source_id"],
                    target_id=record["target_id"],
                    source_type=record["source_type"],
                    target_type=record["target_type"],
                    dependency_type=record["dependency_type"],
                    relation_type="DEPENDS_ON",
                    weight=float(record["weight"]),
                    properties=props
                ))
        return edges

    def _get_raw_edges(self, valid_ids: Set[str]) -> List[EdgeData]:
        """Retrieve raw structural edges (PUBLISHES_TO, SUBSCRIBES_TO, etc)."""
        query = """
        MATCH (source)-[r:PUBLISHES_TO|SUBSCRIBES_TO|ROUTES|RUNS_ON|CONNECTS_TO|USES]->(target)
        RETURN source.id as source_id, target.id as target_id, labels(source)[0] as source_type,
               labels(target)[0] as target_type, type(r) as relation_type,
               coalesce(r.weight, 1.0) as weight, properties(r) as props
        """
        
        edges = []
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            for record in result:
                if valid_ids and (record["source_id"] not in valid_ids or record["target_id"] not in valid_ids):
                    continue
                
                props = dict(record["props"])
                props.pop("weight", None)
                edges.append(EdgeData(
                    source_id=record["source_id"],
                    target_id=record["target_id"],
                    source_type=record["source_type"],
                    target_type=record["target_type"],
                    dependency_type=None,
                    relation_type=record["relation_type"],
                    weight=float(record["weight"]),
                    properties=props
                ))
        return edges

    def get_layer_data(self, layer: str) -> GraphData:
        """Retrieve graph data for a specific layer."""
        if layer not in LAYER_DEFINITIONS:
            raise ValueError(f"Unknown layer: {layer}")
        return self.get_graph_data(
            component_types=LAYER_DEFINITIONS[layer]["component_types"],
            dependency_types=LAYER_DEFINITIONS[layer]["dependency_types"],
        )

    def get_statistics(self) -> Dict[str, int]:
        """Retrieve counts of components and dependencies by type."""
        all_component_types = ["Application", "Broker", "Node", "Topic", "Library"]
        all_relationship_types = ["RUNS_ON", "ROUTES", "PUBLISHES_TO", "SUBSCRIBES_TO", "CONNECTS_TO", "USES"]
        all_dependency_types = ["app_to_app", "node_to_node", "app_to_broker", "node_to_broker"]
        
        stats = {}
        with self.driver.session(database=self.database) as session:
            for comp_type in all_component_types:
                result = session.run(f"MATCH (n:{comp_type}) RETURN count(n) as c")
                stats[f"{comp_type.lower()}_count"] = result.single()["c"]
            for rel_type in all_relationship_types:
                result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as c")
                stats[f"{rel_type.lower()}_count"] = result.single()["c"]
            for dep_type in all_dependency_types:
                result = session.run(f"MATCH ()-[r:DEPENDS_ON {{dependency_type: '{dep_type}'}}]->() RETURN count(r) as c")
                stats[f"{dep_type}_count"] = result.single()["c"]
        return stats

    def export_json(self) -> Dict[str, Any]:
        """Export graph as JSON (compatible with data generation)."""
        data: Dict[str, Any] = {
            "metadata": {},
            "nodes": [],
            "brokers": [],
            "topics": [],
            "applications": [],
            "libraries": [],
            "relationships": {
                "runs_on": [],
                "routes": [],
                "publishes_to": [],
                "subscribes_to": [],
                "connects_to": [],
                "uses": [],
            },
        }
        
        with self.driver.session(database=self.database) as session:
            # Nodes
            result = session.run("MATCH (n:Node) RETURN n.id as id, n.name as name")
            for record in result:
                data["nodes"].append({
                    "id": record["id"],
                    "name": record["name"] or record["id"],
                })
            
            # Brokers
            result = session.run("MATCH (b:Broker) RETURN b.id as id, b.name as name")
            for record in result:
                data["brokers"].append({
                    "id": record["id"],
                    "name": record["name"] or record["id"],
                })
            
            # Topics
            result = session.run("""
                MATCH (t:Topic)
                RETURN t.id as id, t.name as name, coalesce(t.size, 256) as size,
                       coalesce(t.qos_durability, 'VOLATILE') as durability,
                       coalesce(t.qos_reliability, 'BEST_EFFORT') as reliability,
                       coalesce(t.qos_transport_priority, 'MEDIUM') as transport_priority
            """)
            for record in result:
                data["topics"].append({
                    "id": record["id"],
                    "name": record["name"] or record["id"],
                    "size": int(record["size"]),
                    "qos": {
                        "durability": record["durability"],
                        "reliability": record["reliability"],
                        "transport_priority": record["transport_priority"],
                    },
                })
            
            # Applications
            result = session.run("""
                MATCH (a:Application)
                RETURN a.id as id, a.name as name, 
                       coalesce(a.role, 'pubsub') as role,
                       a.app_type as app_type,
                       a.criticality as criticality,
                       a.version as version
            """)
            for record in result:
                app = {
                    "id": record["id"],
                    "name": record["name"] or record["id"],
                    "role": record["role"],
                }
                if record["app_type"]:
                    app["app_type"] = record["app_type"]
                if record["criticality"] is not None:
                    app["criticality"] = record["criticality"]
                if record["version"]:
                    app["version"] = record["version"]
                data["applications"].append(app)
            
            # Libraries
            result = session.run("""
                MATCH (l:Library)
                RETURN l.id as id, l.name as name, l.version as version
            """)
            for record in result:
                lib = {
                    "id": record["id"],
                    "name": record["name"] or record["id"],
                }
                if record["version"]:
                    lib["version"] = record["version"]
                data["libraries"].append(lib)
            
            # Relationships
            rels = {
                "runs_on": "RUNS_ON", "routes": "ROUTES", 
                "publishes_to": "PUBLISHES_TO", "subscribes_to": "SUBSCRIBES_TO",
                "connects_to": "CONNECTS_TO", "uses": "USES"
            }
            
            for key, rel_type in rels.items():
                result = session.run(f"MATCH (s)-[:{rel_type}]->(t) RETURN s.id as src, t.id as tgt")
                for r in result:
                    data["relationships"][key].append({"from": r["src"], "to": r["tgt"]})
        
        return data
