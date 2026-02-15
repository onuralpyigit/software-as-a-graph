"""
Neo4j Graph Repository Adapter

Implements IGraphRepository using Neo4j as the backend.

This adapter handles all Neo4j-specific operations for the graph model
defined in docs/graph-model.md (Definition 1):
    G = (V, E, τ_V, τ_E, L, w, QoS)

Import performs the four construction phases:
    Phase 1: Entity import (V)
    Phase 2: Structural edge import (E_S)
    Phase 3: QoS-based weight computation (w)
    Phase 4: Dependency derivation (E_D, Rules 1–4)
"""

from __future__ import annotations
import logging
import os
from typing import Dict, Any, List, Optional, Set

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.core.models import GraphData, ComponentData, EdgeData, QoSPolicy, MIN_TOPIC_WEIGHT

# ---------------------------------------------------------------------------
# Layer Definitions for Neo4j Queries
# ---------------------------------------------------------------------------
# These use the canonical layer names (app, infra, mw, system) matching
# docs/graph-model.md Definition 3 and src/core/layers.py.
#
# Legacy aliases (application, infrastructure, app_broker, complete) are
# supported for backward compatibility via _LAYER_ALIASES.
# ---------------------------------------------------------------------------

LAYER_DEFINITIONS = {
    "app": {
        "name": "Application Layer",
        "component_types": ["Application"],
        "dependency_types": ["app_to_app"],
    },
    "infra": {
        "name": "Infrastructure Layer",
        "component_types": ["Node"],
        "dependency_types": ["node_to_node"],
    },
    "mw": {
        "name": "Middleware Layer",
        "component_types": ["Application", "Broker", "Node"],
        "dependency_types": ["app_to_broker", "node_to_broker"],
    },
    "system": {
        "name": "Complete System",
        "component_types": ["Application", "Broker", "Node", "Topic", "Library"],
        "dependency_types": ["app_to_app", "node_to_node", "app_to_broker", "node_to_broker"],
    },
}

# Backward compatibility aliases for legacy code that uses old layer names
_LAYER_ALIASES: Dict[str, str] = {
    "application": "app",
    "infrastructure": "infra",
    "app_broker": "mw",
    "complete": "system",
}


def _resolve_layer(layer: str) -> str:
    """Resolve a layer name, supporting canonical names and legacy aliases."""
    canonical = _LAYER_ALIASES.get(layer, layer)
    if canonical not in LAYER_DEFINITIONS:
        valid = sorted(set(LAYER_DEFINITIONS.keys()) | set(_LAYER_ALIASES.keys()))
        raise ValueError(f"Unknown layer: '{layer}'. Valid layers: {valid}")
    return canonical


def create_repository(uri=None, user=None, password=None):
    """Create a Neo4jRepository from params or environment."""
    return Neo4jRepository(
        uri=uri or os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=user or os.getenv("NEO4J_USERNAME", "neo4j"),
        password=password or os.getenv("NEO4J_PASSWORD", "password"),
    )


class Neo4jRepository:
    """
    Neo4j adapter for the graph model.
    
    Handles all Neo4j-specific operations including:
    - Graph data import with constraint management
    - Weight calculations for nodes and edges (§1.5)
    - Dependency derivation between components (Definition 2, Rules 1–4)
    - Graph data retrieval with layer filtering (Definition 3)
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        """Initialize Neo4j repository."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.logger = logging.getLogger(__name__)

    def __enter__(self) -> "Neo4jRepository":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
    
    def close(self) -> None:
        """Close Neo4j driver connection."""
        self.driver.close()

    # ==========================================
    # Import Methods (Save)
    # ==========================================

    def save_graph(self, data: Dict[str, Any], clear: bool = False) -> None:
        """
        Import graph data into the repository.
        
        Orchestrates all four construction phases:
            1. Clear + constraints (optional)
            2. Import entities (V) and structural edges (E_S)
            3. Compute intrinsic weights from QoS (w)
            4. Derive DEPENDS_ON dependencies (E_D)
        """
        self.logger.info(f"Starting import. Clear DB: {clear}")
        
        if clear: 
            self._run_query("MATCH (n) DETACH DELETE n")
            self._create_constraints()
        
        # 1. Import entities
        self._import_entities(data)
        
        # 2. Import structural relationships
        self._import_relationships(data)
        
        # 3. Compute weights (Phase 3)
        self._calculate_intrinsic_weights()
        
        # 4. Derive dependencies (Phase 4, Rules 1–4)
        self._derive_dependencies()
        
        # 5. Compute aggregate component weights
        self._calculate_aggregate_weights()

    def _run_query(self, query: str, parameters: Dict = None) -> Any:
        """Execute a Cypher query."""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return result.consume()

    def _import_batch(self, data: List[Dict], query: str) -> int:
        """Import a batch of records using UNWIND."""
        if not data:
            return 0
        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"UNWIND $rows AS row {query}",
                {"rows": data}
            )
            summary = result.consume()
            return summary.counters.nodes_created + summary.counters.relationships_created

    def _create_constraints(self) -> None:
        """Create uniqueness constraints for all entity types."""
        for label in ["Application", "Broker", "Node", "Topic", "Library"]:
            try:
                self._run_query(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
                )
            except Exception as e:
                self.logger.warning(f"Could not create constraint for {label}: {e}")

    def _import_entities(self, data: Dict[str, Any]) -> None:
        """Import all entity types (Phase 1)."""
        # Nodes
        nodes = [{"id": n["id"], "name": n.get("name", n["id"])} for n in data.get("nodes", [])]
        self._import_batch(nodes, "MERGE (n:Node {id: row.id}) SET n.name = row.name")
        
        # Brokers
        brokers = [{"id": b["id"], "name": b.get("name", b["id"])} for b in data.get("brokers", [])]
        self._import_batch(brokers, "MERGE (b:Broker {id: row.id}) SET b.name = row.name")
        
        # Topics (with QoS properties)
        topics = []
        for t in data.get("topics", []):
            qos = t.get("qos", t.get("qos_policy", {}))
            topics.append({
                "id": t["id"],
                "name": t.get("name", t["id"]),
                "size": t.get("size", 256),
                "qos_reliability": qos.get("reliability", "BEST_EFFORT"),
                "qos_durability": qos.get("durability", "VOLATILE"),
                "qos_transport_priority": qos.get("transport_priority", "MEDIUM"),
            })
        self._import_batch(topics, """
            MERGE (t:Topic {id: row.id})
            SET t.name = row.name, t.size = row.size,
                t.qos_reliability = row.qos_reliability,
                t.qos_durability = row.qos_durability,
                t.qos_transport_priority = row.qos_transport_priority
        """)
        
        # Applications
        apps = []
        for a in data.get("applications", []):
            apps.append({
                "id": a["id"],
                "name": a.get("name", a["id"]),
                "role": a.get("role", "pubsub"),
                "app_type": a.get("app_type", "service"),
                "criticality": a.get("criticality", False),
                "version": a.get("version"),
            })
        self._import_batch(apps, """
            MERGE (a:Application {id: row.id})
            SET a.name = row.name, a.role = row.role, a.app_type = row.app_type,
                a.criticality = row.criticality, a.version = row.version
        """)
        
        # Libraries
        libs = []
        for l in data.get("libraries", []):
            libs.append({
                "id": l["id"],
                "name": l.get("name", l["id"]),
                "version": l.get("version"),
            })
        self._import_batch(libs, """
            MERGE (l:Library {id: row.id})
            SET l.name = row.name, l.version = row.version
        """)

    def _import_relationships(self, data: Dict[str, Any]) -> None:
        """Import structural relationships (Phase 2)."""
        rels = data.get("relationships", {})
        
        rel_mapping = {
            "runs_on": "RUNS_ON",
            "routes": "ROUTES",
            "publishes_to": "PUBLISHES_TO",
            "subscribes_to": "SUBSCRIBES_TO",
            "connects_to": "CONNECTS_TO",
            "uses": "USES",
        }
        
        for key, rel_type in rel_mapping.items():
            items = rels.get(key, [])
            if items:
                batch = [{"from": r.get("from", r.get("source")), 
                          "to": r.get("to", r.get("target"))} for r in items]
                self._import_batch(batch, f"""
                    MATCH (a {{id: row.from}}), (b {{id: row.to}})
                    MERGE (a)-[:{rel_type}]->(b)
                """)

    def _get_qos_weight_cypher(self, topic_var: str) -> str:
        """
        Generate Cypher expression for QoS weight calculation.
        
        Implements: W_topic = max(ε, S_reliability + S_durability + S_priority + S_size)
        Uses scoring constants from QoSPolicy class to ensure consistency.
        """
        rel_scores = QoSPolicy.RELIABILITY_SCORES
        dur_scores = QoSPolicy.DURABILITY_SCORES
        pri_scores = QoSPolicy.PRIORITY_SCORES
        
        # Build the raw sum expression
        raw_sum = f"""
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
        
        # Apply minimum weight floor: max(ε, raw_sum)
        return f"""
        CASE WHEN {raw_sum} < {MIN_TOPIC_WEIGHT} THEN {MIN_TOPIC_WEIGHT}
             ELSE {raw_sum}
        END
        """

    def _calculate_intrinsic_weights(self) -> None:
        """
        Calculate weights for Topics based on QoS and Size (Phase 3).
        
        Implements §1.5 Weight Calculation:
            W_topic = max(ε, S_reliability + S_durability + S_priority + S_size)
        """
        qos_calc = self._get_qos_weight_cypher("t")

        # 1. Topic Weight
        self._run_query(f"MATCH (t:Topic) SET t.weight = {qos_calc}")

        # 2. Edge Weights (Inherit from Topic)
        self._run_query("MATCH ()-[r:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic) SET r.weight = t.weight")
        
        # 3. ROUTES Edge Weights
        self._run_query("MATCH ()-[r:ROUTES]->(t:Topic) SET r.weight = t.weight")

    def _calculate_aggregate_weights(self) -> None:
        """
        Propagate weights from topics up through the component hierarchy.
        
        Implements §1.5 Weight Propagation:
            Library:     w(l) = Σ w(t) for topics l pub/sub to
            Application: w(a) = Σ w(t) + Σ w(l) for direct topics + used libraries
            Broker:      w(b) = Σ w(t) for routed topics
            Node:        w(n) = Σ w(c) for hosted components
            USES edges:  w(e) = w(target_library)
        """
        # 1. Library Weight
        self._run_query("""
            MATCH (l:Library)
            OPTIONAL MATCH (l)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
            WITH l, coalesce(sum(t.weight), 0.0) as load_weight
            SET l.weight = load_weight
        """)

        # 2. USES Edge Weights
        self._run_query("MATCH ()-[r:USES]->(l:Library) SET r.weight = l.weight")

        # 3. Application Weight
        self._run_query("""
            MATCH (a:Application)
            OPTIONAL MATCH (a)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
            OPTIONAL MATCH (a)-[:USES]->(l:Library)
            WITH a, coalesce(sum(DISTINCT t.weight), 0.0) as topic_weight, 
                 coalesce(sum(DISTINCT l.weight), 0.0) as lib_weight
            SET a.weight = topic_weight + lib_weight
        """)

        # 4. Broker Weight
        self._run_query("""
            MATCH (b:Broker)
            OPTIONAL MATCH (b)-[:ROUTES]->(t:Topic)
            WITH b, coalesce(sum(t.weight), 0.0) as routed_weight
            SET b.weight = routed_weight
        """)

        # 5. Node Weight
        self._run_query("""
            MATCH (n:Node)
            OPTIONAL MATCH (c)-[:RUNS_ON]->(n) WHERE c:Application OR c:Broker
            WITH n, coalesce(sum(c.weight), 0.0) as hosted_weight
            SET n.weight = hosted_weight
        """)

        # 6. DEPENDS_ON Edge Weights: |shared_topics| × avg(topic_weight)
        self._run_query("""
            MATCH (a)-[d:DEPENDS_ON]->(b)
            WHERE d.dependency_type = 'app_to_app'
            OPTIONAL MATCH (a)-[:SUBSCRIBES_TO]->(t:Topic)<-[:PUBLISHES_TO]-(b)
            WITH a, b, d, collect(t.weight) as weights
            SET d.weight = CASE WHEN size(weights) > 0
                THEN size(weights) * reduce(s = 0.0, w IN weights | s + w) / size(weights)
                ELSE 1.0 END
        """)

    def _derive_dependencies(self) -> None:
        """
        Derive DEPENDS_ON relationships from structural edges (Phase 4).
        
        Implements Definition 2, Rules 1–4 from docs/graph-model.md.
        """
        # Rule 1: app_to_app — subscriber depends on publisher (via shared topic)
        # Note: This captures direct pub/sub. Library transitive paths are handled
        # by additional queries below.
        self._run_query("""
            MATCH (subscriber)-[:SUBSCRIBES_TO]->(t:Topic)<-[:PUBLISHES_TO]-(publisher)
            WHERE subscriber <> publisher
              AND (subscriber:Application OR subscriber:Library)
              AND (publisher:Application OR publisher:Library)
            WITH subscriber, publisher, count(t) as shared_count, avg(t.weight) as avg_weight
            MERGE (subscriber)-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(publisher)
            SET d.weight = shared_count * avg_weight,
                d.shared_topics = shared_count
        """)
        
        # Rule 1 (transitive): app depends on publisher via library chain
        # App-A -[USES]-> Lib-X -[SUBSCRIBES_TO]-> Topic-T <-[PUBLISHES_TO]- App-B
        self._run_query("""
            MATCH (app:Application)-[:USES*1..]->(lib)-[:SUBSCRIBES_TO]->(t:Topic)<-[:PUBLISHES_TO]-(publisher)
            WHERE app <> publisher
              AND (publisher:Application OR publisher:Library)
            WITH app, publisher, count(DISTINCT t) as shared_count, avg(t.weight) as avg_weight
            MERGE (app)-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(publisher)
            ON CREATE SET d.weight = shared_count * avg_weight, d.shared_topics = shared_count
            ON MATCH SET d.weight = CASE WHEN shared_count * avg_weight > d.weight 
                                         THEN shared_count * avg_weight ELSE d.weight END,
                         d.shared_topics = CASE WHEN shared_count > d.shared_topics 
                                                THEN shared_count ELSE d.shared_topics END
        """)
        
        # Rule 1 (transitive, reverse): app publishes via library chain
        # App-A -[SUBSCRIBES_TO]-> Topic-T <-[PUBLISHES_TO]- Lib-Y <-[USES*]- App-B
        self._run_query("""
            MATCH (subscriber)-[:SUBSCRIBES_TO]->(t:Topic)<-[:PUBLISHES_TO]-(lib)<-[:USES*1..]-(app:Application)
            WHERE subscriber <> app
              AND (subscriber:Application OR subscriber:Library)
            WITH subscriber, app, count(DISTINCT t) as shared_count, avg(t.weight) as avg_weight
            MERGE (subscriber)-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(app)
            ON CREATE SET d.weight = shared_count * avg_weight, d.shared_topics = shared_count
            ON MATCH SET d.weight = CASE WHEN shared_count * avg_weight > d.weight 
                                         THEN shared_count * avg_weight ELSE d.weight END,
                         d.shared_topics = CASE WHEN shared_count > d.shared_topics 
                                                THEN shared_count ELSE d.shared_topics END
        """)
        
        # Rule 2: app_to_broker — app depends on broker that routes its topics
        self._run_query("""
            MATCH (app)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)<-[:ROUTES]-(broker:Broker)
            WHERE app:Application OR app:Library
            WITH app, broker, count(t) as topic_count
            MERGE (app)-[d:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(broker)
            SET d.weight = topic_count
        """)
        
        # Rule 2 (transitive): app depends on broker via library chain
        self._run_query("""
            MATCH (app:Application)-[:USES*1..]->(lib)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)<-[:ROUTES]-(broker:Broker)
            WITH app, broker, count(DISTINCT t) as topic_count
            MERGE (app)-[d:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(broker)
            ON CREATE SET d.weight = topic_count
            ON MATCH SET d.weight = CASE WHEN topic_count > d.weight THEN topic_count ELSE d.weight END
        """)
        
        # Rule 3: node_to_node — lifted from component dependencies
        self._run_query("""
            MATCH (a)-[:DEPENDS_ON]->(b),
                  (a)-[:RUNS_ON]->(n1:Node),
                  (b)-[:RUNS_ON]->(n2:Node)
            WHERE n1 <> n2
            WITH n1, n2, count(*) as dep_count
            MERGE (n1)-[d:DEPENDS_ON {dependency_type: 'node_to_node'}]->(n2)
            SET d.weight = dep_count
        """)
        
        # Rule 4: node_to_broker — lifted from hosted app broker usage
        self._run_query("""
            MATCH (app)-[dep:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(broker:Broker),
                  (app)-[:RUNS_ON]->(n:Node)
            WITH n, broker, count(*) as dep_count
            MERGE (n)-[d:DEPENDS_ON {dependency_type: 'node_to_broker'}]->(broker)
            SET d.weight = dep_count
        """)

    # ==========================================
    # Query Methods (Read)
    # ==========================================

    def get_graph_data(
        self,
        component_types: Optional[List[str]] = None,
        dependency_types: Optional[List[str]] = None,
        include_raw: bool = False,
    ) -> GraphData:
        """
        Retrieve graph data with optional type filtering.
        
        Args:
            component_types: Filter to specific vertex types (e.g., ["Application"])
            dependency_types: Filter to specific dependency subtypes (e.g., ["app_to_app"])
            include_raw: Include raw structural edges in addition to DEPENDS_ON
        """
        components = []
        edges = []
        
        with self.driver.session(database=self.database) as session:
            # Fetch components
            all_types = component_types or ["Application", "Broker", "Node", "Topic", "Library"]
            for comp_type in all_types:
                result = session.run(
                    f"MATCH (n:{comp_type}) RETURN n.id as id, n.name as name, n.weight as weight, "
                    f"labels(n)[0] as type, properties(n) as props"
                )
                for record in result:
                    props = dict(record["props"])
                    # Keep name in properties for later use in analysis
                    props["name"] = record["name"]
                    for key in ["id", "weight"]:
                        props.pop(key, None)
                    components.append(ComponentData(
                        id=record["id"],
                        component_type=record["type"],
                        weight=record["weight"] or 1.0,
                        properties=props,
                    ))
            
            # Fetch DEPENDS_ON edges
            dep_filter = ""
            if dependency_types:
                types_str = ", ".join(f"'{t}'" for t in dependency_types)
                dep_filter = f" WHERE d.dependency_type IN [{types_str}]"
            
            result = session.run(
                f"MATCH (s)-[d:DEPENDS_ON]->(t){dep_filter} "
                f"RETURN s.id as src, t.id as tgt, labels(s)[0] as stype, "
                f"labels(t)[0] as ttype, d.dependency_type as dep_type, d.weight as weight"
            )
            for record in result:
                edges.append(EdgeData(
                    source_id=record["src"],
                    target_id=record["tgt"],
                    source_type=record["stype"],
                    target_type=record["ttype"],
                    dependency_type=record["dep_type"],
                    relation_type="DEPENDS_ON",
                    weight=record["weight"] or 1.0,
                ))
            
            # Optionally include raw structural edges
            if include_raw:
                for rel_type in ["PUBLISHES_TO", "SUBSCRIBES_TO", "ROUTES", 
                                 "RUNS_ON", "CONNECTS_TO", "USES"]:
                    result = session.run(
                        f"MATCH (s)-[r:{rel_type}]->(t) "
                        f"RETURN s.id as src, t.id as tgt, labels(s)[0] as stype, "
                        f"labels(t)[0] as ttype, r.weight as weight"
                    )
                    for record in result:
                        edges.append(EdgeData(
                            source_id=record["src"],
                            target_id=record["tgt"],
                            source_type=record["stype"],
                            target_type=record["ttype"],
                            dependency_type=rel_type.lower(),
                            relation_type=rel_type,
                            weight=record["weight"] or 1.0,
                        ))
        
        return GraphData(components=components, edges=edges)

    def get_layer_data(self, layer: str) -> GraphData:
        """
        Retrieve graph data for a specific architectural layer.
        
        Implements Definition 3 (Layer Projection) by filtering components
        and dependencies to those relevant for the requested layer.
        
        Args:
            layer: Layer name — canonical (app, infra, mw, system) or
                   legacy alias (application, infrastructure, app_broker, complete)
        """
        canonical = _resolve_layer(layer)
        defn = LAYER_DEFINITIONS[canonical]
        return self.get_graph_data(
            component_types=defn["component_types"],
            dependency_types=defn["dependency_types"],
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
                result = session.run(
                    f"MATCH ()-[r:DEPENDS_ON {{dependency_type: '{dep_type}'}}]->() RETURN count(r) as c"
                )
                stats[f"{dep_type}_count"] = result.single()["c"]
        return stats

    def export_json(self) -> Dict[str, Any]:
        """Export graph as JSON (compatible with data generation format)."""
        data = {
            "nodes": [], "brokers": [], "topics": [], 
            "applications": [], "libraries": [],
            "relationships": {
                "runs_on": [], "routes": [], "publishes_to": [],
                "subscribes_to": [], "connects_to": [], "uses": []
            }
        }
        
        with self.driver.session(database=self.database) as session:
            # Nodes
            result = session.run("MATCH (n:Node) RETURN n.id as id, n.name as name")
            for record in result:
                data["nodes"].append({"id": record["id"], "name": record["name"] or record["id"]})
            
            # Brokers
            result = session.run("MATCH (b:Broker) RETURN b.id as id, b.name as name")
            for record in result:
                data["brokers"].append({"id": record["id"], "name": record["name"] or record["id"]})
            
            # Topics
            result = session.run("""
                MATCH (t:Topic)
                RETURN t.id as id, t.name as name, t.size as size,
                       t.qos_reliability as reliability, t.qos_durability as durability,
                       t.qos_transport_priority as transport_priority
            """)
            for record in result:
                data["topics"].append({
                    "id": record["id"],
                    "name": record["name"] or record["id"],
                    "size": record["size"] or 256,
                    "qos": {
                        "reliability": record["reliability"] or "BEST_EFFORT",
                        "durability": record["durability"] or "VOLATILE",
                        "transport_priority": record["transport_priority"] or "MEDIUM",
                    }
                })
            
            # Applications
            result = session.run("""
                MATCH (a:Application)
                RETURN a.id as id, a.name as name, a.role as role, 
                       a.app_type as app_type, a.criticality as criticality, a.version as version
            """)
            for record in result:
                app = {
                    "id": record["id"],
                    "name": record["name"] or record["id"],
                    "role": record["role"] or "pubsub",
                    "app_type": record["app_type"] or "service",
                }
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
                lib = {"id": record["id"], "name": record["name"] or record["id"]}
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

    def get_library_usage(self) -> Dict[str, int]:
        """Get library usage counts."""
        query = """
        MATCH (l:Library)<-[:USES]-(a:Application)
        RETURN l.id as id, count(a) as usage_count
        ORDER BY usage_count DESC
        """
        usage = {}
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            for record in result:
                usage[record["id"]] = record["usage_count"]
        return usage

    def get_node_allocations(self) -> Dict[str, List[str]]:
        """Get allocation of components to nodes."""
        query = """
        MATCH (n:Node)<-[:RUNS_ON]-(c)
        RETURN n.id as node_id, collect(c.id) as components
        """
        allocations = {}
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            for record in result:
                allocations[record["node_id"]] = record["components"]
        return allocations

    def get_broker_routing(self) -> Dict[str, List[str]]:
        """Get topics routed by each broker."""
        query = """
        MATCH (b:Broker)-[:ROUTES]->(t:Topic)
        RETURN b.id as broker_id, collect(t.id) as topics
        """
        routing = {}
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            for record in result:
                routing[record["broker_id"]] = record["topics"]
        return routing

    def get_components_with_filter(
        self, 
        component_type: Optional[str] = None, 
        min_weight: Optional[float] = None, 
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get components from the graph with optional filtering."""
        with self.driver.session(database=self.database) as session:
            # Build query
            label_filter = f":{component_type}" if component_type else ""
            weight_filter = f"WHERE coalesce(n.weight, 0.0) >= {min_weight}" if min_weight is not None else ""
            
            query = f"""
            MATCH (n{label_filter})
            {weight_filter}
            RETURN n.id as id, labels(n)[0] as type, coalesce(n.weight, 1.0) as weight, properties(n) as props
            LIMIT $limit
            """
            
            result = session.run(query, limit=limit)
            components = []
            
            for record in result:
                props = dict(record["props"])
                props.pop("id", None)
                props.pop("weight", None)
                components.append({
                    "id": record["id"],
                    "type": record["type"],
                    "weight": record["weight"],
                    **props
                })
                
            return {
                "count": len(components),
                "components": components
            }

    def get_edges_with_filter(
        self, 
        dependency_type: Optional[str] = None, 
        min_weight: Optional[float] = None, 
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get edges from the graph with optional filtering."""
        with self.driver.session(database=self.database) as session:
            # Build query
            dep_filter = f"{{dependency_type: '{dependency_type}'}}" if dependency_type else ""
            weight_filter = f"WHERE coalesce(r.weight, 0.0) >= {min_weight}" if min_weight is not None else ""
            
            query = f"""
            MATCH (s)-[r:DEPENDS_ON{dep_filter}]->(t)
            {weight_filter}
            RETURN s.id as source, t.id as target, labels(s)[0] as source_type, 
                   labels(t)[0] as target_type, r.dependency_type as dependency_type, 
                   coalesce(r.weight, 1.0) as weight, properties(r) as props
            LIMIT $limit
            """
            
            result = session.run(query, limit=limit)
            edges = []
            
            for record in result:
                props = dict(record["props"])
                props.pop("weight", None)
                props.pop("dependency_type", None)
                edges.append({
                    "source": record["source"],
                    "target": record["target"],
                    "source_type": record["source_type"],
                    "target_type": record["target_type"],
                    "dependency_type": record["dependency_type"],
                    "weight": record["weight"],
                    **props
                })
                
            return {
                "count": len(edges),
                "edges": edges
            }

    def search_nodes(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search for nodes by ID or label."""
        cypher_query = """
        MATCH (n)
        WHERE (n:Application OR n:Broker OR n:Node OR n:Topic OR n:Library)
            AND (toLower(n.id) CONTAINS toLower($search_term) OR toLower(COALESCE(n.name, n.id)) CONTAINS toLower($search_term))
        RETURN n.id AS id, labels(n)[0] AS type,
                COALESCE(n.name, n.id) AS label,
                COALESCE(n.weight, 1.0) AS weight
        ORDER BY n.id
        LIMIT $limit
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher_query, search_term=query, limit=limit)
            nodes = []
            for record in result:
                nodes.append({
                    "id": record["id"],
                    "type": record["type"],
                    "label": record["label"],
                    "weight": float(record["weight"])
                })
            return nodes

    def get_node_connections(self, node_id: str, fetch_structural: bool, depth: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Fetch connections for a specific node at specified depth.
        Returns (nodes, edges) as list of dicts.
        """
        # Determine relationship types based on view
        if fetch_structural:
            rel_types = "|".join(["PUBLISHES_TO", "SUBSCRIBES_TO", "RUNS_ON", "ROUTES", "CONNECTS_TO", "USES"])
            query = f"""
            MATCH (center {{id: $node_id}})
            MATCH path = (center)-[:{rel_types}*1..{depth}]-(connected)
            WITH DISTINCT connected
            RETURN connected.id AS id, labels(connected)[0] AS type,
                    COALESCE(connected.weight, 1.0) AS weight, properties(connected) AS props
            """
            
            edges_query = f"""
            MATCH (center {{id: $node_id}})
            MATCH path = (center)-[:{rel_types}*1..{depth}]-(n)
            WITH DISTINCT n
            WITH collect(DISTINCT n.id) + [$node_id] AS node_ids
            UNWIND node_ids AS node_id
            MATCH (s {{id: node_id}})-[r:{rel_types}]->(t)
            WHERE t.id IN node_ids
            RETURN DISTINCT s.id AS source_id, t.id AS target_id,
                    labels(s)[0] AS source_type, labels(t)[0] AS target_type,
                    type(r) AS relation_type, COALESCE(r.weight, 1.0) AS weight,
                    properties(r) AS props
            """
        else:
            query = f"""
            MATCH (center {{id: $node_id}})
            MATCH path = (center)-[:DEPENDS_ON*1..{depth}]-(connected)
            WITH DISTINCT connected
            RETURN connected.id AS id, labels(connected)[0] AS type,
                    COALESCE(connected.weight, 1.0) AS weight, properties(connected) AS props
            """
            
            edges_query = f"""
            MATCH (center {{id: $node_id}})
            MATCH path = (center)-[:DEPENDS_ON*1..{depth}]-(n)
            WITH DISTINCT n
            WITH collect(DISTINCT n.id) + [$node_id] AS node_ids
            UNWIND node_ids AS node_id
            MATCH (s {{id: node_id}})-[r:DEPENDS_ON]->(t)
            WHERE t.id IN node_ids
            RETURN DISTINCT s.id AS source_id, t.id AS target_id,
                    labels(s)[0] AS source_type, labels(t)[0] AS target_type,
                    COALESCE(r.dependency_type, 'unknown') AS dependency_type,
                    COALESCE(r.weight, 1.0) AS weight, properties(r) AS props
            """
        
        # Center node query
        center_node_query = """
        MATCH (center {id: $node_id})
        RETURN center.id AS id, labels(center)[0] AS type,
                COALESCE(center.weight, 1.0) AS weight, properties(center) AS props
        """
        
        with self.driver.session(database=self.database) as session:
            # Check center node
            result = session.run(center_node_query, node_id=node_id)
            components = []
            for record in result:
                props = dict(record["props"])
                props.pop("id", None)
                props.pop("weight", None)
                components.append({
                    "id": record["id"],
                    "type": record["type"],
                    "weight": float(record["weight"]),
                    **props
                })
            
            if not components:
                return [], []

            # Fetch connected nodes
            result = session.run(query, node_id=node_id)
            for record in result:
                props = dict(record["props"])
                props.pop("id", None)
                props.pop("weight", None)
                components.append({
                    "id": record["id"],
                    "type": record["type"],
                    "weight": float(record["weight"]),
                    **props
                })
            
            # Fetch edges
            result = session.run(edges_query, node_id=node_id)
            edges = []
            for record in result:
                props = dict(record["props"])
                props.pop("weight", None)
                props.pop("dependency_type", None)
                
                edge = {
                    "source": record["source_id"],
                    "target": record["target_id"],
                    "source_type": record["source_type"],
                    "target_type": record["target_type"],
                    "relation_type": record.get("relation_type", "DEPENDS_ON"),
                    "weight": float(record["weight"]),
                    **props
                }
                
                if "dependency_type" in record.keys():
                    edge["dependency_type"] = record["dependency_type"]
                
                edges.append(edge)
                
            return components, edges

    def get_topology_data(self, node_id: Optional[str], node_limit: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Fetch topology data with drill-down support."""
        with self.driver.session(database=self.database) as session:
            components = []
            edges = []
            
            if node_id:
                # First, determine the type of the selected node
                type_query = "MATCH (n {id: $node_id}) RETURN labels(n)[0] AS type"
                result = session.run(type_query, node_id=node_id)
                record = result.single()
                
                if not record:
                    raise ValueError(f"Node {node_id} not found")
                
                node_type = record["type"]
                self.logger.info(f"Node {node_id} is of type {node_type}")
                
                if node_type == "Node":
                    # Level 2: Show Applications and Brokers running on this Node
                    self._fetch_node_topology(session, components, edges, node_id)
                elif node_type == "Application":
                    # Level 3: Show Topics and Libraries related to this Application
                    self._fetch_application_topology(session, components, edges, node_id)
                elif node_type == "Broker":
                    # Level 3 for Broker: Show Topics it routes
                    self._fetch_broker_topology(session, components, edges, node_id)
                elif node_type == "Library":
                    # Level 3 for Library: Show Libraries it uses and Applications using it
                    self._fetch_library_topology(session, components, edges, node_id)
                elif node_type == "Topic":
                    # Level 3 for Topic: Show Brokers that route it and Applications that publish/subscribe
                    self._fetch_topic_topology(session, components, edges, node_id)
                else:
                    self._fetch_single_node(session, components, node_id)
            
            else:
                # Level 1: Full topology
                self._fetch_full_topology(session, components, edges, node_limit)
                
            return components, edges

    def _process_node_result(self, result, components):
        for record in result:
            props = dict(record["props"])
            name = props.get("name", record["id"])
            props.pop("id", None)
            props.pop("weight", None)
            props.pop("name", None)
            components.append({
                "id": record["id"],
                "type": record["type"],
                "label": name,
                "weight": float(record["weight"]),
                **props
            })

    def _process_edge_result(self, result, edges, relation_type=None):
        for record in result:
            props = dict(record["props"])
            props.pop("weight", None)
            edges.append({
                "source": record["source_id"],
                "target": record["target_id"],
                "source_type": record["source_type"],
                "target_type": record["target_type"],
                "relation_type": relation_type or record.get("relation_type"),
                "weight": float(record["weight"]),
                **props
            })

    def _fetch_node_topology(self, session, components, edges, node_id):
        node_query = """
        MATCH (center:Node {id: $node_id})
        RETURN center.id AS id, labels(center)[0] AS type,
                COALESCE(center.weight, 1.0) AS weight, properties(center) AS props
        UNION
        MATCH (center:Node {id: $node_id})<-[:RUNS_ON]-(entity)
        WHERE entity:Application OR entity:Broker
        RETURN entity.id AS id, labels(entity)[0] AS type,
                COALESCE(entity.weight, 1.0) AS weight, properties(entity) AS props
        """
        self._process_node_result(session.run(node_query, node_id=node_id), components)
        
        edges_query = """
        MATCH (entity)-[r:RUNS_ON]->(center:Node {id: $node_id})
        WHERE entity:Application OR entity:Broker
        RETURN entity.id AS source_id, center.id AS target_id,
                labels(entity)[0] AS source_type, labels(center)[0] AS target_type,
                COALESCE(r.weight, 1.0) AS weight, properties(r) AS props
        """
        self._process_edge_result(session.run(edges_query, node_id=node_id), edges, "RUNS_ON")

    def _fetch_application_topology(self, session, components, edges, node_id):
        node_query = """
        MATCH (center:Application {id: $node_id})
        RETURN center.id AS id, labels(center)[0] AS type, COALESCE(center.weight, 1.0) AS weight, properties(center) AS props
        UNION
        MATCH (center:Application {id: $node_id})-[:PUBLISHES_TO|SUBSCRIBES_TO]->(topic:Topic)
        RETURN topic.id AS id, labels(topic)[0] AS type, COALESCE(topic.weight, 1.0) AS weight, properties(topic) AS props
        UNION
        MATCH (center:Application {id: $node_id})-[:USES]->(lib:Library)
        RETURN lib.id AS id, labels(lib)[0] AS type, COALESCE(lib.weight, 1.0) AS weight, properties(lib) AS props
        """
        self._process_node_result(session.run(node_query, node_id=node_id), components)
        
        edges_query = """
        MATCH (center:Application {id: $node_id})-[r:PUBLISHES_TO|SUBSCRIBES_TO|USES]->(target)
        WHERE target:Topic OR target:Library
        RETURN center.id AS source_id, target.id AS target_id, labels(center)[0] AS source_type, labels(target)[0] AS target_type,
                type(r) AS relation_type, COALESCE(r.weight, 1.0) AS weight, properties(r) AS props
        """
        self._process_edge_result(session.run(edges_query, node_id=node_id), edges)

    def _fetch_broker_topology(self, session, components, edges, node_id):
        node_query = """
        MATCH (center:Broker {id: $node_id})
        RETURN center.id AS id, labels(center)[0] AS type, COALESCE(center.weight, 1.0) AS weight, properties(center) AS props
        UNION
        MATCH (center:Broker {id: $node_id})-[:ROUTES]->(topic:Topic)
        RETURN topic.id AS id, labels(topic)[0] AS type, COALESCE(topic.weight, 1.0) AS weight, properties(topic) AS props
        """
        self._process_node_result(session.run(node_query, node_id=node_id), components)
        
        edges_query = """
        MATCH (center:Broker {id: $node_id})-[r:ROUTES]->(topic:Topic)
        RETURN center.id AS source_id, topic.id AS target_id, labels(center)[0] AS source_type, labels(topic)[0] AS target_type,
                COALESCE(r.weight, 1.0) AS weight, properties(r) AS props
        """
        self._process_edge_result(session.run(edges_query, node_id=node_id), edges, "ROUTES")

    def _fetch_library_topology(self, session, components, edges, node_id):
        node_query = """
        MATCH (center:Library {id: $node_id})
        RETURN center.id AS id, labels(center)[0] AS type, COALESCE(center.weight, 1.0) AS weight, properties(center) AS props
        UNION
        MATCH (center:Library {id: $node_id})-[:USES]->(lib:Library)
        RETURN lib.id AS id, labels(lib)[0] AS type, COALESCE(lib.weight, 1.0) AS weight, properties(lib) AS props
        UNION
        MATCH (app:Application)-[:USES]->(center:Library {id: $node_id})
        RETURN app.id AS id, labels(app)[0] AS type, COALESCE(app.weight, 1.0) AS weight, properties(app) AS props
        """
        self._process_node_result(session.run(node_query, node_id=node_id), components)
        
        edges_query = """
        MATCH (center:Library {id: $node_id})-[r:USES]->(lib:Library)
        RETURN center.id AS source_id, lib.id AS target_id, labels(center)[0] AS source_type, labels(lib)[0] AS target_type,
                COALESCE(r.weight, 1.0) AS weight, properties(r) AS props
        UNION
        MATCH (app:Application)-[r:USES]->(center:Library {id: $node_id})
        RETURN app.id AS source_id, center.id AS target_id, labels(app)[0] AS source_type, labels(center)[0] AS target_type,
                COALESCE(r.weight, 1.0) AS weight, properties(r) AS props
        """
        self._process_edge_result(session.run(edges_query, node_id=node_id), edges, "USES")

    def _fetch_topic_topology(self, session, components, edges, node_id):
        node_query = """
        MATCH (center:Topic {id: $node_id})
        RETURN center.id AS id, labels(center)[0] AS type, COALESCE(center.weight, 1.0) AS weight, properties(center) AS props
        UNION
        MATCH (broker:Broker)-[:ROUTES]->(center:Topic {id: $node_id})
        RETURN broker.id AS id, labels(broker)[0] AS type, COALESCE(broker.weight, 1.0) AS weight, properties(broker) AS props
        UNION
        MATCH (app:Application)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(center:Topic {id: $node_id})
        RETURN app.id AS id, labels(app)[0] AS type, COALESCE(app.weight, 1.0) AS weight, properties(app) AS props
        """
        self._process_node_result(session.run(node_query, node_id=node_id), components)
        
        edges_query = """
        MATCH (broker:Broker)-[r:ROUTES]->(center:Topic {id: $node_id})
        RETURN broker.id AS source_id, center.id AS target_id, labels(broker)[0] AS source_type, labels(center)[0] AS target_type,
                type(r) AS relation_type, COALESCE(r.weight, 1.0) AS weight, properties(r) AS props
        UNION
        MATCH (app:Application)-[r:PUBLISHES_TO|SUBSCRIBES_TO]->(center:Topic {id: $node_id})
        RETURN app.id AS source_id, center.id AS target_id, labels(app)[0] AS source_type, labels(center)[0] AS target_type,
                type(r) AS relation_type, COALESCE(r.weight, 1.0) AS weight, properties(r) AS props
        """
        self._process_edge_result(session.run(edges_query, node_id=node_id), edges)

    def _fetch_single_node(self, session, components, node_id):
        node_query = """
        MATCH (n {id: $node_id})
        RETURN n.id AS id, labels(n)[0] AS type, COALESCE(n.weight, 1.0) AS weight, properties(n) AS props
        """
        self._process_node_result(session.run(node_query, node_id=node_id), components)

    def _fetch_full_topology(self, session, components, edges, node_limit):
        node_query = """
        MATCH (n:Node)
        RETURN n.id AS id, labels(n)[0] AS type, COALESCE(n.weight, 1.0) AS weight, properties(n) AS props
        ORDER BY COALESCE(n.weight, 0.0) DESC
        LIMIT $limit
        """
        self._process_node_result(session.run(node_query, limit=node_limit), components)
        
        # Helper to extract node IDs
        node_ids = [c["id"] for c in components]
        
        edges_query = """
        MATCH (s:Node)-[r:CONNECTS_TO]->(t:Node)
        WHERE s.id IN $node_ids AND t.id IN $node_ids
        RETURN s.id AS source_id, t.id AS target_id, labels(s)[0] AS source_type, labels(t)[0] AS target_type,
                type(r) AS relation_type, COALESCE(r.weight, 1.0) AS weight, properties(r) AS props
        ORDER BY COALESCE(r.weight, 0.0) DESC
        """
        self._process_edge_result(session.run(edges_query, node_ids=node_ids), edges)