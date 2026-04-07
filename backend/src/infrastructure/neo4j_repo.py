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
from typing import Dict, Any, List, Optional, Set, Tuple

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from src.core.ports.graph_repository import IGraphRepository
from src.core.models import (
    ComponentData, EdgeData, GraphData, QoSPolicy, 
    MIN_TOPIC_WEIGHT, TOPIC_QOS_WEIGHT_BETA,
    APP_HYBRID_MAX_COEFF, APP_HYBRID_MEAN_COEFF,
    BROKER_HYBRID_MAX_COEFF, BROKER_HYBRID_MEAN_COEFF,
    LIB_FANOUT_GAMMA
)
from . import config

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
        "component_types": ["Application", "Library"],
        "dependency_types": ["app_to_app", "app_to_lib"],
    },
    "infra": {
        "name": "Infrastructure Layer",
        "component_types": ["Node"],
        "dependency_types": ["node_to_node"],
    },
    "mw": {
        "name": "Middleware Layer",
        "component_types": ["Application", "Broker", "Node"],
        "dependency_types": ["app_to_broker", "node_to_broker", "broker_to_broker"],
    },
    "system": {
        "name": "Complete System",
        "component_types": ["Application", "Broker", "Node", "Topic", "Library"],
        "dependency_types": ["app_to_app", "app_to_lib", "app_to_broker", "node_to_node", "node_to_broker", "broker_to_broker"],
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
        uri=uri or config.get_default_uri(),
        user=user or config.get_default_username(),
        password=password or config.get_default_password(),
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
        uri: str = None,
        user: str = None,
        password: str = None,
        database: str = None
    ):
        """Initialize Neo4j repository."""
        uri = uri or config.get_default_uri()
        user = user or config.get_default_username()
        password = password or config.get_default_password()
        database = database or config.get_default_database()
        
        self.driver = GraphDatabase.driver(
            uri, auth=(user, password),
            notifications_min_severity="WARNING",
        )
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
        Import graph data into the repository within a single transaction.
        
        Orchestrates all five construction phases:
            1. Import entities (V) — Apps, Brokers, Nodes, Topics, Libraries
            2. Import structural relationships (E_S) — USES, RUNS_ON, etc.
            3. Compute intrinsic weights from QoS (Phase 3)
            4. Derive DEPENDS_ON dependencies (Phase 4)
            5. Compute aggregate component weights (Phase 5)
        """
        self.logger.info(f"Starting import. Clear DB: {clear}")
        
        # 0. Schema: Constraints run in their own transaction (Neo4j requirement)
        self._create_constraints()
        
        with self.driver.session(database=self.database) as session:
            session.execute_write(self._save_graph_tx, data, clear)

    def _save_graph_tx(self, tx: Any, data: Dict[str, Any], clear: bool) -> None:
        """Internal unit of work for save_graph transaction."""
        try:
            # 0. Clear
            if clear: 
                tx.run("MATCH (n) DETACH DELETE n")
            
            # 1. Import entities
            self._import_entities(data, tx)
            
            # 2. Import structural relationships
            self._import_relationships(data, tx)
            
            # 3. Compute intrinsic weights
            self._calculate_intrinsic_weights(tx)
            
            # 4. Derive dependencies
            self._derive_dependencies(tx)
            
            # 5. Compute aggregate component weights (Phase 5)
            self._calculate_aggregate_weights(tx)
            
            self.logger.info("Import completed successfully.")
            
        except Exception as e:
            self.logger.error(f"Import failed during phase orchestration: {e}")
            self.logger.critical(
                "Database may be in an inconsistent state. "
                "Recommendation: Re-run with clear=True to ensure reproducibility."
            )
            # Transaction will be rolled back by the session.execute_write context manager
            raise

    def _run_query(self, query: str, parameters: Dict = None, tx: Any = None) -> Any:
        """Execute a Cypher query, optionally within an existing transaction."""
        if tx:
            result = tx.run(query, parameters or {})
            return result.consume()
            
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return result.consume()

    def _import_batch(self, data: List[Dict], query: str, tx: Any = None) -> int:
        """Import a batch of records using UNWIND, optionally within a transaction."""
        if not data:
            return 0
            
        if tx:
            result = tx.run(
                f"UNWIND $rows AS row {query}",
                {"rows": data}
            )
            summary = result.consume()
            return summary.counters.nodes_created + summary.counters.relationships_created

        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"UNWIND $rows AS row {query}",
                {"rows": data}
            )
            summary = result.consume()
            return summary.counters.nodes_created + summary.counters.relationships_created

    def _create_constraints(self, tx: Any = None) -> None:
        """Create uniqueness constraints for all entity types."""
        for label in ["Application", "Broker", "Node", "Topic", "Library"]:
            try:
                self._run_query(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE",
                    tx=tx
                )
            except Exception as e:
                self.logger.warning(f"Could not create constraint for {label}: {e}")

    def _import_entities(self, data: Dict[str, Any], tx: Any = None) -> None:
        """Import all entity types (Phase 1)."""
        # 0. Import metadata provenance
        self._import_metadata(data.get("metadata", {}), tx=tx)
        
        # 1. Import components
        self._import_nodes(data.get("nodes", []), tx=tx)
        self._import_brokers(data.get("brokers", []), tx=tx)
        self._import_topics(data.get("topics", []), tx=tx)
        self._import_applications(data.get("applications", []), tx=tx)
        self._import_libraries(data.get("libraries", []), tx=tx)

    def _import_nodes(self, nodes_data: List[Dict[str, Any]], tx: Any = None) -> None:
        """Import compute nodes with infrastructure metadata."""
        nodes = []
        for n in nodes_data:
            nodes.append({
                "id": n["id"],
                "name": n.get("name", n["id"]),
                "ip_address": n.get("ip_address", ""),
                "cpu_cores": n.get("cpu_cores", 0),
                "memory_gb": n.get("memory_gb", 0),
                "os_type": n.get("os_type", "linux"),
            })
        self._import_batch(nodes, """
            MERGE (n:Node {id: row.id})
            SET n.name = row.name,
                n.ip_address = row.ip_address,
                n.cpu_cores = row.cpu_cores,
                n.memory_gb = row.memory_gb,
                n.os_type = row.os_type
        """, tx=tx)

    def _import_metadata(self, metadata: Dict[str, Any], tx: Any = None) -> None:
        """
        Store graph metadata (scale, seed, etc.) in a singleton :Metadata node.
        """
        if not metadata:
            return
            
        # Flatten scale object for storage
        scale = metadata.get("scale", {})
        params = {
            "seed": metadata.get("seed"),
            "generation_mode": metadata.get("generation_mode", "unknown"),
            "domain": metadata.get("domain"),
            "scenario": metadata.get("scenario"),
            "scale_apps": scale.get("apps", 0),
            "scale_topics": scale.get("topics", 0),
            "scale_brokers": scale.get("brokers", 0),
            "scale_nodes": scale.get("nodes", 0),
            "scale_libs": scale.get("libs", 0),
        }
        
        query = """
        MERGE (m:Metadata)
        SET m += $params
        """
        self._run_query(query, {"params": params}, tx=tx)

    def _import_brokers(self, brokers_data: List[Dict[str, Any]], tx: Any = None) -> None:
        """Import message brokers with middleware metadata."""
        brokers = []
        for b in brokers_data:
            brokers.append({
                "id": b["id"],
                "name": b.get("name", b["id"]),
                "type": b.get("type", "mqtt"),
                "max_connections": b.get("max_connections", 0),
                "host": b.get("host", ""),
            })
        self._import_batch(brokers, """
            MERGE (b:Broker {id: row.id})
            SET b.name = row.name,
                b.type = row.type,
                b.max_connections = row.max_connections,
                b.host = row.host
        """, tx=tx)

    def _import_topics(self, topics_data: List[Dict[str, Any]], tx: Any = None) -> None:
        """Import topics with QoS policies."""
        topics = []
        for t in topics_data:
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
        """, tx=tx)

    def _import_applications(self, apps_data: List[Dict[str, Any]], tx: Any = None) -> None:
        """Import applications with code metrics and hierarchy."""
        apps = []
        for a in apps_data:
            cm = a.get("code_metrics") or {}
            sh = a.get("system_hierarchy") or {}
            size = cm.get("size", {})
            complexity = cm.get("complexity", {})
            cohesion = cm.get("cohesion", {})
            coupling = cm.get("coupling", {})
            # SonarQube / Quality metrics
            quality = cm.get("quality", cm)  # Support both nested and flat for flexibility
            
            apps.append({
                "id": a["id"],
                "name": a.get("name", a["id"]),
                "role": a.get("role", "pubsub"),
                "app_type": a.get("app_type", "service"),
                "criticality": a.get("criticality", "LOW"),
                "version": a.get("version"),
                # System hierarchy
                "component_name": sh.get("component_name", ""),
                "config_item_name": sh.get("config_item_name", ""),
                "domain_name": sh.get("domain_name", ""),
                "system_name": sh.get("system_name", ""),
                # Code metrics — size
                "cm_total_loc": size.get("total_loc", 0),
                "cm_total_classes": size.get("total_classes", 0),
                "cm_total_methods": size.get("total_methods", 0),
                "cm_total_fields": size.get("total_fields", 0),
                # Code metrics — complexity
                "cm_total_wmc": complexity.get("total_wmc", 0),
                "cm_avg_wmc": float(complexity.get("avg_wmc", 0.0)),
                "cm_max_wmc": complexity.get("max_wmc", 0),
                # Code metrics — cohesion
                "cm_avg_lcom": float(cohesion.get("avg_lcom", 0.0)),
                "cm_max_lcom": float(cohesion.get("max_lcom", 0.0)),
                # Code metrics — coupling
                "cm_avg_cbo": float(coupling.get("avg_cbo", 0.0)),
                "cm_max_cbo": coupling.get("max_cbo", 0),
                "cm_avg_rfc": float(coupling.get("avg_rfc", 0.0)),
                "cm_max_rfc": coupling.get("max_rfc", 0),
                "cm_avg_fanin": float(coupling.get("avg_fanin", 0.0)),
                "cm_max_fanin": coupling.get("max_fanin", 0),
                "cm_avg_fanout": float(coupling.get("avg_fanout", 0.0)),
                "cm_max_fanout": coupling.get("max_fanout", 0),
                # Quality Metrics (SonarQube)
                "sqale_debt_ratio": float(quality.get("sqale_debt_ratio", 0.0)),
                "bugs": int(quality.get("bugs", 0)),
                "vulnerabilities": int(quality.get("vulnerabilities", 0)),
                "duplicated_lines_density": float(quality.get("duplicated_lines_density", 0.0)),
                # Analysis-compatible aliases
                "loc": size.get("total_loc", 0),
                "cyclomatic_complexity": float(complexity.get("avg_wmc", 0.0)),
                "coupling_afferent": int(coupling.get("avg_fanin", 0)),
                "coupling_efferent": int(coupling.get("avg_fanout", 0)),
                "lcom": float(cohesion.get("avg_lcom", 0.0)),
            })

        self._import_batch(apps, """
            MERGE (a:Application {id: row.id})
            SET a.name = row.name, a.role = row.role, a.app_type = row.app_type,
                a.criticality = row.criticality, a.version = row.version,
                a.component_name = row.component_name, a.config_item_name = row.config_item_name,
                a.domain_name = row.domain_name, a.system_name = row.system_name,
                a.cm_total_loc = row.cm_total_loc, a.cm_total_classes = row.cm_total_classes,
                a.cm_total_methods = row.cm_total_methods, a.cm_total_fields = row.cm_total_fields,
                a.cm_total_wmc = row.cm_total_wmc, a.cm_avg_wmc = row.cm_avg_wmc, a.cm_max_wmc = row.cm_max_wmc,
                a.cm_avg_lcom = row.cm_avg_lcom, a.cm_max_lcom = row.cm_max_lcom,
                a.cm_avg_cbo = row.cm_avg_cbo, a.cm_max_cbo = row.cm_max_cbo,
                a.cm_avg_rfc = row.cm_avg_rfc, a.cm_max_rfc = row.cm_max_rfc,
                a.cm_avg_fanin = row.cm_avg_fanin, a.cm_max_fanin = row.cm_max_fanin,
                a.cm_avg_fanout = row.cm_avg_fanout, a.cm_max_fanout = row.cm_max_fanout,
                a.sqale_debt_ratio = row.sqale_debt_ratio, a.bugs = row.bugs,
                a.vulnerabilities = row.vulnerabilities, a.duplicated_lines_density = row.duplicated_lines_density,
                a.loc = row.loc,
                a.cyclomatic_complexity = row.cyclomatic_complexity,
                a.coupling_afferent = row.coupling_afferent,
                a.coupling_efferent = row.coupling_efferent,
                a.lcom = row.lcom
        """, tx=tx)

    def _import_libraries(self, libs_data: List[Dict[str, Any]], tx: Any = None) -> None:
        """Import libraries with code metrics and hierarchy."""
        libs = []
        for l in libs_data:
            cm = l.get("code_metrics") or {}
            sh = l.get("system_hierarchy") or {}
            size = cm.get("size", {})
            complexity = cm.get("complexity", {})
            cohesion = cm.get("cohesion", {})
            coupling = cm.get("coupling", {})
            quality = cm.get("quality", cm)
            
            libs.append({
                "id": l["id"],
                "name": l.get("name", l["id"]),
                "version": l.get("version"),
                # System hierarchy
                "component_name": sh.get("component_name", ""),
                "config_item_name": sh.get("config_item_name", ""),
                "domain_name": sh.get("domain_name", ""),
                "system_name": sh.get("system_name", ""),
                # Code metrics — size
                "cm_total_loc": size.get("total_loc", 0),
                "cm_total_classes": size.get("total_classes", 0),
                "cm_total_methods": size.get("total_methods", 0),
                "cm_total_fields": size.get("total_fields", 0),
                # Code metrics — complexity
                "cm_total_wmc": complexity.get("total_wmc", 0),
                "cm_avg_wmc": float(complexity.get("avg_wmc", 0.0)),
                "cm_max_wmc": complexity.get("max_wmc", 0),
                # Code metrics — cohesion
                "cm_avg_lcom": float(cohesion.get("avg_lcom", 0.0)),
                "cm_max_lcom": float(cohesion.get("max_lcom", 0.0)),
                # Code metrics — coupling
                "cm_avg_cbo": float(coupling.get("avg_cbo", 0.0)),
                "cm_max_cbo": coupling.get("max_cbo", 0),
                "cm_avg_rfc": float(coupling.get("avg_rfc", 0.0)),
                "cm_max_rfc": coupling.get("max_rfc", 0),
                "cm_avg_fanin": float(coupling.get("avg_fanin", 0.0)),
                "cm_max_fanin": coupling.get("max_fanin", 0),
                "cm_avg_fanout": float(coupling.get("avg_fanout", 0.0)),
                "cm_max_fanout": coupling.get("max_fanout", 0),
                # Quality Metrics (SonarQube)
                "sqale_debt_ratio": float(quality.get("sqale_debt_ratio", 0.0)),
                "bugs": int(quality.get("bugs", 0)),
                "vulnerabilities": int(quality.get("vulnerabilities", 0)),
                "duplicated_lines_density": float(quality.get("duplicated_lines_density", 0.0)),
                # Analysis-compatible aliases
                "loc": size.get("total_loc", 0),
                "cyclomatic_complexity": float(complexity.get("avg_wmc", 0.0)),
                "coupling_afferent": int(coupling.get("avg_fanin", 0)),
                "coupling_efferent": int(coupling.get("avg_fanout", 0)),
                "lcom": float(cohesion.get("avg_lcom", 0.0)),
            })
        self._import_batch(libs, """
            MERGE (l:Library {id: row.id})
            SET l.name = row.name, l.version = row.version,
                l.component_name = row.component_name, l.config_item_name = row.config_item_name,
                l.domain_name = row.domain_name, l.system_name = row.system_name,
                l.cm_total_loc = row.cm_total_loc, l.cm_total_classes = row.cm_total_classes,
                l.cm_total_methods = row.cm_total_methods, l.cm_total_fields = row.cm_total_fields,
                l.cm_total_wmc = row.cm_total_wmc, l.cm_avg_wmc = row.cm_avg_wmc, l.cm_max_wmc = row.cm_max_wmc,
                l.cm_avg_lcom = row.cm_avg_lcom, l.cm_max_lcom = row.cm_max_lcom,
                l.cm_avg_cbo = row.cm_avg_cbo, l.cm_max_cbo = row.cm_max_cbo,
                l.cm_avg_rfc = row.cm_avg_rfc, l.cm_max_rfc = row.cm_max_rfc,
                l.cm_avg_fanin = row.cm_avg_fanin, l.cm_max_fanin = row.cm_max_fanin,
                l.cm_avg_fanout = row.cm_avg_fanout, l.cm_max_fanout = row.cm_max_fanout,
                l.sqale_debt_ratio = row.sqale_debt_ratio, l.bugs = row.bugs,
                l.vulnerabilities = row.vulnerabilities, l.duplicated_lines_density = row.duplicated_lines_density,
                l.loc = row.loc,
                l.cyclomatic_complexity = row.cyclomatic_complexity,
                l.coupling_afferent = row.coupling_afferent,
                l.coupling_efferent = row.coupling_efferent,
                l.lcom = row.lcom
        """, tx=tx)


    def _import_relationships(self, data: Dict[str, Any], tx: Any = None) -> None:
        """
        Import structural relationships (Phase 2) with entity validation.
        
        Validates that referenced source and target entities exist before
        creating edges. Missing entities raise ValueError to trigger rollback.
        """
        rels = data.get("relationships", {})
        
        # Mapping: key -> (RelationType, SourceLabels, TargetLabels)
        # Labels are used for performance optimization (matching indexed constraints)
        rel_config = {
            "runs_on": ("RUNS_ON", "Application|Broker", "Node"),
            "routes": ("ROUTES", "Broker", "Topic"),
            "publishes_to": ("PUBLISHES_TO", "Application|Library", "Topic"),
            "subscribes_to": ("SUBSCRIBES_TO", "Application|Library", "Topic"),
            "connects_to": ("CONNECTS_TO", "Node", "Node"),
            "uses": ("USES", "Application|Library", "Library"),
        }
        
        for key, (rel_type, src_labels, tgt_labels) in rel_config.items():
            items = rels.get(key, [])
            if not items:
                continue

            batch = [{"from": r.get("from", r.get("source")), 
                      "to": r.get("to", r.get("target"))} for r in items]
            
            # 1. Validate existence
            # OPTIONAL MATCH allows identifying precisely which rows refer to missing nodes
            validation_query = """
                UNWIND $rows AS row
                OPTIONAL MATCH (src {id: row.from})
                OPTIONAL MATCH (tgt {id: row.to})
                WITH row, src, tgt
                WHERE src IS NULL OR tgt IS NULL
                RETURN row.from as src_id, src IS NOT NULL as src_exists,
                       row.to as tgt_id, tgt IS NOT NULL as tgt_exists
                LIMIT 100
            """
            res = self._run_query(validation_query, {"rows": batch}, tx=tx)
            # res for a tx run returns a Result object or something we can iterate if it's run via tx.run
            # Wait, my _run_query returns result.consume() which is a ResultSummary.
            # I should use tx.run directly here to get the records.
            
            # Re-running validation manually since _run_query consumes the result
            if tx:
                res = tx.run(validation_query, {"rows": batch})
                errors = []
                for record in res:
                    if not record["src_exists"]:
                        errors.append(f"Source entity missing (id='{record['src_id']}')")
                    if not record["tgt_exists"]:
                        errors.append(f"Target entity missing (id='{record['tgt_id']}')")
                
                if errors:
                    example_errs = "; ".join(errors[:5])
                    raise ValueError(
                        f"Structural integrity violation in '{key}' ('{rel_type}') relationship: "
                        f"Referenced entities must exist. Found {len(errors)} errors including: {example_errs}"
                    )

            # 2. Create edges using label-optimized match
            query = f"""
                MATCH (a:{src_labels} {{id: row.from}}), (b:{tgt_labels} {{id: row.to}})
                MERGE (a)-[:{rel_type}]->(b)
            """
            self._import_batch(batch, query, tx=tx)

        # Phase 2 post-step: Fan-out augmentation for Topic
        self._run_query("""
            MATCH (t:Topic)
            OPTIONAL MATCH (sub)-[:SUBSCRIBES_TO]->(t) WHERE sub:Application OR sub:Library
            WITH t, count(DISTINCT sub) as sub_count
            OPTIONAL MATCH (pub)-[:PUBLISHES_TO]->(t) WHERE pub:Application OR pub:Library
            WITH t, sub_count, count(DISTINCT pub) as pub_count
            SET t.subscriber_count = sub_count,
                t.publisher_count = pub_count
        """, tx=tx)

    def _get_qos_weight_cypher(self, topic_var: str) -> str:
        """
        Generate Cypher expression for QoS weight calculation.
        
        Implements: W_topic = max(ε, β * QoS_score + (1-β) * S_size)
        Uses scoring constants from QoSPolicy class to ensure consistency.
        """
        rel_scores = QoSPolicy.RELIABILITY_SCORES
        dur_scores = QoSPolicy.DURABILITY_SCORES
        pri_scores = QoSPolicy.PRIORITY_SCORES
        
        beta = TOPIC_QOS_WEIGHT_BETA
        one_minus_beta = round(1.0 - beta, 4)
        
        # Build the QoS score expression
        qos_score = f"""
        ({QoSPolicy.W_RELIABILITY} * CASE {topic_var}.qos_reliability WHEN 'RELIABLE' THEN {rel_scores['RELIABLE']} ELSE 0.0 END +
         {QoSPolicy.W_DURABILITY} * CASE {topic_var}.qos_durability 
             WHEN 'PERSISTENT' THEN {dur_scores['PERSISTENT']} 
             WHEN 'TRANSIENT' THEN {dur_scores['TRANSIENT']} 
             WHEN 'TRANSIENT_LOCAL' THEN {dur_scores['TRANSIENT_LOCAL']} 
             ELSE 0.0 END +
         {QoSPolicy.W_PRIORITY} * CASE {topic_var}.qos_transport_priority 
             WHEN 'URGENT' THEN {pri_scores['URGENT']} 
             WHEN 'HIGH' THEN {pri_scores['HIGH']} 
             WHEN 'MEDIUM' THEN {pri_scores['MEDIUM']} 
             ELSE 0.0 END)
        """
        
        # Build the size norm expression (capped at 1.0)
        size_norm = f"""
        CASE WHEN {topic_var}.size <= 0 THEN 0.0
              WHEN (log(1 + {topic_var}.size / 1024.0) / (log(2) * 50.0)) > 1.0 THEN 1.0
              ELSE (log(1 + {topic_var}.size / 1024.0) / (log(2) * 50.0))
         END
        """
        
        weighted_sum = f"({beta} * {qos_score} + {one_minus_beta} * {size_norm})"
        
        # Apply minimum weight floor: max(ε, weighted_sum)
        return f"""
        CASE WHEN {weighted_sum} < {MIN_TOPIC_WEIGHT} THEN {MIN_TOPIC_WEIGHT}
             ELSE {weighted_sum}
        END
        """

    def _calculate_intrinsic_weights(self, tx: Any = None) -> None:
        """
        Step 3: Compute intrinsic weights for Topic nodes.
        
        Implements Equation 1 (Topic Weight):
            w(topic) = β * QoS_score + (1-β) * Size_Norm
        Where β = TOPIC_QOS_WEIGHT_BETA (0.85).
        """
        qos_calc = self._get_qos_weight_cypher("t")

        # 1. Topic Weight
        self._run_query(f"MATCH (t:Topic) SET t.weight = {qos_calc}", tx=tx)

        # 2. Edge Weights (Inherit from Topic)
        self._run_query("MATCH ()-[r:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic) SET r.weight = t.weight", tx=tx)
        
        # 3. ROUTES Edge Weights
        self._run_query("MATCH ()-[r:ROUTES]->(t:Topic) SET r.weight = t.weight", tx=tx)

    def _calculate_aggregate_weights(self, tx: Any = None) -> None:
        """
        Step 5: Compute aggregate weights for secondary infrastructure components.
        
        Propagates weights from topics up the dependency chain using max/hybrid 
        blending. This ensures that a node or broker's importance reflects 
        the most critical data it carries.
        
        Aggregation Rules:
            Application: w(a) = max w(t) for direct topics
            Library:     w(l) = max w(app) for consuming apps
            Broker:      w(b) = 0.70 * max(w(t)) + 0.30 * mean(w(t))
            Node:        w(n) = max w(c) for hosted components
            app_to_lib:  w(e) = w(App)
            broker_to_broker: w(e) = w(Node)
        """
        # 1. Application Weight (hybrid: 0.80 * max + 0.20 * mean)
        self._run_query(f"""
            MATCH (a:Application)
            OPTIONAL MATCH (a)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
            WITH a, max(t.weight) as max_w, avg(t.weight) as mean_w
            SET a.weight = coalesce({APP_HYBRID_MAX_COEFF} * max_w + {APP_HYBRID_MEAN_COEFF} * mean_w, 0.01)
        """, tx=tx)

        # 2. Library Weight (propagated + fan-out multiplier)
        # Formula: min(1.0, base_w * (1 + γ * log2(1 + DG_in)))
        # Reflects simultaneous blast semantics: shared libraries are higher priority.
        self._run_query(f"""
            MATCH (l:Library)
            OPTIONAL MATCH (l)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
            WITH l, max(t.weight) as t_max
            OPTIONAL MATCH (app:Application)-[:USES]->(l)
            WITH l, t_max, max(app.weight) as a_max, count(app) as dg_in
            WITH l, 
                 CASE WHEN coalesce(t_max, 0.0) > coalesce(a_max, 0.0) 
                      THEN coalesce(t_max, 0.0) 
                      ELSE coalesce(a_max, 0.0) 
                 END as base_w, 
                 dg_in
            WITH l, base_w, (1.0 + {LIB_FANOUT_GAMMA} * log(1 + dg_in) / log(2)) as multiplier
            SET l.weight = CASE WHEN base_w <= 0 THEN 0.01
                                WHEN base_w * multiplier > 1.0 THEN 1.0
                                ELSE base_w * multiplier END
        """, tx=tx)

        # 3. Broker Weight (hybrid: 0.70 * max + 0.30 * mean)
        self._run_query(f"""
            MATCH (b:Broker)
            OPTIONAL MATCH (b)-[:ROUTES]->(t:Topic)
            WITH b, max(t.weight) as max_w, avg(t.weight) as mean_w
            SET b.weight = coalesce({BROKER_HYBRID_MAX_COEFF} * max_w + {BROKER_HYBRID_MEAN_COEFF} * mean_w, 0.01)
        """, tx=tx)

        # 4. Node Weight (max hosted component weight)
        self._run_query("""
            MATCH (n:Node)
            OPTIONAL MATCH (c)-[:RUNS_ON]->(n) WHERE c:Application OR c:Broker
            WITH n, max(c.weight) as hosted_max
            SET n.weight = coalesce(hosted_max, 0.01)
        """, tx=tx)
        
        # 5. app_to_lib Edge Weights (inherits from App)
        self._run_query("""
            MATCH (app)-[d:DEPENDS_ON {dependency_type: 'app_to_lib'}]->(lib:Library)
            SET d.weight = coalesce(app.weight, 0.01)
        """, tx=tx)

        # 6. broker_to_broker Edge Weights (inherits from Node)
        # Step 6: Population of Rule 6 weights computed in Phase 5 from finalized Node weights.
        self._run_query("""
            MATCH (b1:Broker)-[d:DEPENDS_ON {dependency_type: 'broker_to_broker'}]->(b2:Broker)
            MATCH (b1)-[:RUNS_ON]->(n:Node)<-[:RUNS_ON]-(b2)
            WITH d, max(n.weight) as node_w
            SET d.weight = coalesce(node_w, 0.01)
        """, tx=tx)

    def _derive_dependencies(self, tx: Any = None) -> None:
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
            WITH subscriber, publisher, count(t) as path_count, max(t.weight) as max_weight
            MERGE (subscriber)-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(publisher)
            SET d.weight = coalesce(max_weight, 0.01),
                d.path_count = path_count
        """, tx=tx)
        
        # Rule 1 (transitive): app depends on publisher via library chain
        # App-A -[USES]-> Lib-X -[SUBSCRIBES_TO]-> Topic-T <-[PUBLISHES_TO]- App-B
        self._run_query("""
            MATCH (app:Application)-[:USES*1..3]->(lib)-[:SUBSCRIBES_TO]->(t:Topic)<-[:PUBLISHES_TO]-(publisher)
            WHERE app <> publisher
              AND (publisher:Application OR publisher:Library)
            WITH app, publisher, count(DISTINCT t) as path_count, max(t.weight) as max_weight
            MERGE (app)-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(publisher)
            ON CREATE SET d.weight = coalesce(max_weight, 0.01), d.path_count = path_count
            ON MATCH SET d.weight = CASE WHEN coalesce(max_weight, 0.01) > coalesce(d.weight, 0.0) 
                                         THEN max_weight ELSE d.weight END,
                         d.path_count = CASE WHEN path_count > coalesce(d.path_count, 0) 
                                             THEN path_count ELSE d.path_count END
        """, tx=tx)
        
        # Rule 1 (transitive, reverse): app publishes via library chain
        # App-A -[SUBSCRIBES_TO]-> Topic-T <-[PUBLISHES_TO]- Lib-Y <-[USES*]- App-B
        self._run_query("""
            MATCH (subscriber)-[:SUBSCRIBES_TO]->(t:Topic)<-[:PUBLISHES_TO]-(lib)<-[:USES*1..3]-(app:Application)
            WHERE subscriber <> app
              AND (subscriber:Application OR subscriber:Library)
            WITH subscriber, app, count(DISTINCT t) as path_count, max(t.weight) as max_weight
            MERGE (subscriber)-[d:DEPENDS_ON {dependency_type: 'app_to_app'}]->(app)
            ON CREATE SET d.weight = coalesce(max_weight, 0.01), d.path_count = path_count
            ON MATCH SET d.weight = CASE WHEN coalesce(max_weight, 0.01) > coalesce(d.weight, 0.0) 
                                         THEN max_weight ELSE d.weight END,
                         d.path_count = CASE WHEN path_count > coalesce(d.path_count, 0) 
                                             THEN path_count ELSE d.path_count END
        """, tx=tx)
        
        # Rule 2: app_to_broker — app depends on broker that routes its topics
        self._run_query("""
            MATCH (app)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)<-[:ROUTES]-(broker:Broker)
            WHERE app:Application OR app:Library
            WITH app, broker, max(t.weight) as max_w, count(t) as path_count
            MERGE (app)-[d:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(broker)
            SET d.weight = coalesce(max_w, 0.01), d.path_count = path_count
        """, tx=tx)
 
        # Rule 2 (transitive): app depends on broker via library chain
        self._run_query("""
            MATCH (app:Application)-[:USES*1..3]->(lib)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)<-[:ROUTES]-(broker:Broker)
            WITH app, broker, max(t.weight) as max_w, count(DISTINCT t) as path_count
            MERGE (app)-[d:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(broker)
            ON CREATE SET d.weight = coalesce(max_w, 0.01), d.path_count = path_count
            ON MATCH SET d.weight = CASE WHEN coalesce(max_w, 0.01) > coalesce(d.weight, 0) THEN coalesce(max_w, 0.01) ELSE d.weight END,
                         d.path_count = CASE WHEN path_count > coalesce(d.path_count, 0) THEN path_count ELSE d.path_count END
        """, tx=tx)
 
        # Rule 3: node_to_node — lifted from component dependencies
        self._run_query("""
            MATCH (a)-[d_ab:DEPENDS_ON]->(b),
                  (a)-[:RUNS_ON]->(n1:Node),
                  (b)-[:RUNS_ON]->(n2:Node)
            WHERE n1 <> n2
            WITH n1, n2, max(d_ab.weight) as lifted_max, count(*) as dep_count
            MERGE (n1)-[d:DEPENDS_ON {dependency_type: 'node_to_node'}]->(n2)
            SET d.weight = coalesce(lifted_max, 0.01), d.path_count = dep_count
        """, tx=tx)
 
        # Rule 4: node_to_broker — lifted from hosted app broker usage
        self._run_query("""
            MATCH (app)-[dep:DEPENDS_ON {dependency_type: 'app_to_broker'}]->(broker:Broker),
                  (app)-[:RUNS_ON]->(n:Node)
            WITH n, broker, max(dep.weight) as lifted_max, count(*) as dep_count
            MERGE (n)-[d:DEPENDS_ON {dependency_type: 'node_to_broker'}]->(broker)
            SET d.weight = coalesce(lifted_max, 0.01), d.path_count = dep_count
        """, tx=tx)
 
        # Rule 6: broker_to_broker — colocation dependency via shared node
        self._run_query("""
            MATCH (b1:Broker)-[:RUNS_ON]->(n:Node)<-[:RUNS_ON]-(b2:Broker)
            WHERE b1 <> b2
            WITH b1, b2, count(DISTINCT n) as path_count
            MERGE (b1)-[d:DEPENDS_ON {dependency_type: 'broker_to_broker'}]->(b2)
            SET d.path_count = path_count
        """, tx=tx)
 
        # Rule 5: app_to_lib — app depends on shared library
        self._run_query("""
            MATCH (app)-[:USES]->(lib:Library)
            WHERE app:Application OR app:Library
            WITH app, lib, count(*) as path_count
            MERGE (app)-[d:DEPENDS_ON {dependency_type: 'app_to_lib'}]->(lib)
            SET d.path_count = path_count
        """, tx=tx)

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
                    # Ensure name is present, fallback to ID
                    name = record["name"] or record["id"]
                    props["name"] = name
                    
                    # Clean up props to avoid duplication
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
                f"labels(t)[0] as ttype, d.dependency_type as dep_type, d.weight as weight, d.path_count as path_count"
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
                    path_count=record["path_count"] or 1,
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
        all_dependency_types = ["app_to_app", "app_to_lib", "app_to_broker", "node_to_node", "node_to_broker", "broker_to_broker"]
        
        stats = {}
        with self.driver.session(database=self.database) as session:
            # Capture total metrics
            result = session.run("MATCH (n) RETURN count(n) as c")
            stats["total_nodes"] = result.single()["c"]
            
            result = session.run("MATCH ()-[r]->() RETURN count(r) as c")
            stats["total_relationships"] = result.single()["c"]
            
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

    def _reconstruct_component_dict(self, comp: ComponentData) -> Dict[str, Any]:
        """
        Reconstruct a component dictionary with nested sub-objects (system_hierarchy, code_metrics)
        from flattened Neo4j properties.
        """
        props = comp.properties
        # Base fields
        res = {"id": comp.id, "name": props.get("name", comp.id), "weight": comp.weight}
        
        # 1. System Hierarchy reconstruction
        sh = {}
        for key in ["system_name", "domain_name", "component_name", "config_item_name"]:
            if val := props.get(key):
                if val != "": # Don't export empty strings
                    sh[key] = val
        if sh:
            res["system_hierarchy"] = sh

        # 2. Code Metrics reconstruction (Size, Complexity, Cohesion, Coupling, Quality)
        cm = {"size": {}, "complexity": {}, "cohesion": {}, "coupling": {}, "quality": {}}
        
        # Mapping definition for un-flattening
        metrics_mapping = {
            "size": {
                "cm_total_loc": "total_loc", "cm_total_classes": "total_classes", 
                "cm_total_methods": "total_methods", "cm_total_fields": "total_fields"
            },
            "complexity": {
                "cm_total_wmc": "total_wmc", "cm_avg_wmc": "avg_wmc", "cm_max_wmc": "max_wmc"
            },
            "cohesion": {
                "cm_avg_lcom": "avg_lcom", "cm_max_lcom": "max_lcom"
            },
            "coupling": {
                "cm_avg_cbo": "avg_cbo", "cm_max_cbo": "max_cbo", "cm_avg_rfc": "avg_rfc",
                "cm_max_rfc": "max_rfc", "cm_avg_fanin": "avg_fanin", "cm_max_fanin": "max_fanin",
                "cm_avg_fanout": "avg_fanout", "cm_max_fanout": "max_fanout"
            },
            "quality": {
                "sqale_debt_ratio": "sqale_debt_ratio", "bugs": "bugs", 
                "vulnerabilities": "vulnerabilities", "duplicated_lines_density": "duplicated_lines_density"
            }
        }
        
        for section, fields in metrics_mapping.items():
            for flat_key, nest_key in fields.items():
                if flat_key in props:
                    cm[section][nest_key] = props[flat_key]
        
        # Filter out empty sections
        cm = {k: v for k, v in cm.items() if v}
        if cm:
            res["code_metrics"] = cm

        # 3. Type-specific logic and property normalization
        if comp.component_type == "Topic":
            res["size"] = props.get("size", 256)
            # Canonical lowercase for QoS keys to match input expectations
            res["qos"] = {
                "reliability": str(props.get("qos_reliability", "best_effort")).lower(),
                "durability": str(props.get("qos_durability", "volatile")).lower(),
                "transport_priority": str(props.get("qos_transport_priority", "medium")).lower(),
            }
        elif comp.component_type == "Application":
            res.update({
                "role": props.get("role", "pubsub"),
                "app_type": props.get("app_type", "service"),
                "criticality": props.get("criticality", "LOW"),
            })
            if props.get("version"): res["version"] = props["version"]
        elif comp.component_type == "Library":
            if props.get("version"): res["version"] = props["version"]
        elif comp.component_type == "Node":
            for key in ["ip_address", "cpu_cores", "memory_gb", "os_type"]:
                if key in props: res[key] = props[key]
        elif comp.component_type == "Broker":
            for key in ["type", "max_connections", "host"]:
                if key in props: res[key] = props[key]
        
        return res

    def _get_metadata_dict(self) -> Dict[str, Any]:
        """
        Retrieve graph metadata from the :Metadata node and reconstruct the nested structure.
        """
        query = "MATCH (m:Metadata) RETURN properties(m) as props"
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            record = result.single()
            if not record:
                return {}
            
            props = record["props"]
            return {
                "scale": {
                    "apps": props.get("scale_apps", 0),
                    "topics": props.get("scale_topics", 0),
                    "brokers": props.get("scale_brokers", 0),
                    "nodes": props.get("scale_nodes", 0),
                    "libs": props.get("scale_libs", 0)
                },
                "seed": props.get("seed"),
                "generation_mode": props.get("generation_mode"),
                "domain": props.get("domain"),
                "scenario": props.get("scenario")
            }

    def export_json(self) -> Dict[str, Any]:
        """
        Export graph as JSON (compatible with data generation format).
        Consolidated via get_graph_data to ensure logic consistency.
        """
        # Fetch everything: all component types, all dependency types, and raw structural edges
        graph_data = self.get_graph_data(include_raw=True)
        
        data = {
            "metadata": self._get_metadata_dict(),
            "nodes": [], "brokers": [], "topics": [], 
            "applications": [], "libraries": [],
            "relationships": {
                "runs_on": [], "routes": [], "publishes_to": [],
                "subscribes_to": [], "connects_to": [], "uses": [],
                "depends_on": [] # include pre-computed dependencies
            }
        }
        
        # Category mapping for components
        type_to_category = {
            "Node": "nodes",
            "Broker": "brokers",
            "Topic": "topics",
            "Application": "applications",
            "Library": "libraries"
        }
        
        # Process components
        for comp in graph_data.components:
            category = type_to_category.get(comp.component_type)
            if not category:
                continue
                
            comp_dict = self._reconstruct_component_dict(comp)
            data[category].append(comp_dict)
            
        # Process edges
        for edge in graph_data.edges:
            rel_key = edge.relation_type.lower()
            if rel_key in data["relationships"]:
                edge_dict = {"from": edge.source_id, "to": edge.target_id, "weight": edge.weight}
                
                # Include metadata for DEPENDS_ON
                if edge.relation_type == "DEPENDS_ON":
                    edge_dict.update({
                        "dependency_type": edge.dependency_type,
                        "path_count": edge.path_count
                    })
                
                data["relationships"][rel_key].append(edge_dict)
                
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