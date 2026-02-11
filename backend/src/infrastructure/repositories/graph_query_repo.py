from typing import List, Optional, Dict, Any, Tuple
import logging
from src.core.graph_exporter import GraphExporter, ComponentData, EdgeData, STRUCTURAL_REL_TYPES, GraphData

logger = logging.getLogger(__name__)

class GraphQueryRepository:
    """
    Repository for executing Cypher queries against Neo4j to retrieve graph data.
    Decouples raw DB access from the API layer.
    """
    
    def __init__(self, exporter: GraphExporter):
        self.exporter = exporter

    def _execute_edge_query(self, query: str, node_ids: List[str], edge_type: str) -> List[EdgeData]:
        """Execute edge query and parse results."""
        with self.exporter.driver.session() as session:
            result = session.run(query, node_ids=node_ids)
            edges = []
            
            for record in result:
                props = dict(record["props"])
                props.pop("weight", None)
                props.pop("dependency_type", None)
                
                # Handle both structural and depends_on queries
                relation_type = record.get("relation_type", "DEPENDS_ON")
                dependency_type = record.get("dependency_type", relation_type)
                
                edges.append(EdgeData(
                    source_id=record["source_id"],
                    target_id=record["target_id"],
                    source_type=record["source_type"],
                    target_type=record["target_type"],
                    dependency_type=dependency_type,
                    relation_type=relation_type,
                    weight=float(record["weight"]),
                    properties=props
                ))
            
            logger.info(f"Fetched {len(edges)} {edge_type} edges")
            if edges:
                edge_counts = {}
                for e in edges:
                    edge_counts[e.relation_type] = edge_counts.get(e.relation_type, 0) + 1
                logger.info(f"Edge breakdown: {edge_counts}")
            
            return edges

    def fetch_limited_nodes(self, limit: int, node_types: Optional[List[str]] = None) -> Tuple[List[ComponentData], List[str]]:
        """Fetch top N nodes sorted by weight, optionally filtering by node types."""
        # Build WHERE clause for node types
        if node_types:
            type_conditions = " OR ".join([f"n:{t}" for t in node_types])
            where_clause = f"WHERE {type_conditions}"
        else:
            where_clause = "WHERE n:Application OR n:Broker OR n:Node OR n:Topic"
        
        query = f"""
        MATCH (n)
        {where_clause}
        RETURN n.id AS id, labels(n)[0] AS type,
               COALESCE(n.weight, 1.0) AS weight, properties(n) AS props
        ORDER BY COALESCE(n.weight, 0.0) DESC
        LIMIT $limit
        """
        
        with self.exporter.driver.session() as session:
            result = session.run(query, limit=limit)
            components = []
            node_ids = []
            
            for record in result:
                props = dict(record["props"])
                props.pop("id", None)
                props.pop("weight", None)
                
                node_id = record["id"]
                node_ids.append(node_id)
                components.append(ComponentData(
                    id=node_id,
                    component_type=record["type"],
                    weight=float(record["weight"]),
                    properties=props
                ))
            
            type_info = f" (types={node_types})" if node_types else " (all types)"
            logger.info(f"Fetched {len(components)} nodes (limit={limit}){type_info}")
            return components, node_ids

    def _fetch_structural_edges(self, node_ids: List[str], edge_limit: Optional[int]) -> List[EdgeData]:
        """Fetch structural relationships, prioritizing RUNS_ON."""
        edges = []
        
        # First, fetch RUNS_ON relationships (highest priority)
        runs_on_limit = edge_limit if edge_limit else None
        runs_on_query = f"""
        MATCH (s)-[r:RUNS_ON]->(t)
        WHERE s.id IN $node_ids AND t.id IN $node_ids
        RETURN s.id AS source_id, t.id AS target_id,
               labels(s)[0] AS source_type, labels(t)[0] AS target_type,
               type(r) AS relation_type, COALESCE(r.weight, 1.0) AS weight,
               properties(r) AS props
        ORDER BY COALESCE(r.weight, 0.0) DESC
        {f'LIMIT {runs_on_limit}' if runs_on_limit else ''}
        """
        
        edges.extend(self._execute_edge_query(runs_on_query, node_ids, "RUNS_ON"))
        
        # If we have a limit and reached it, return early
        if edge_limit and len(edges) >= edge_limit:
            return edges[:edge_limit]
        
        # Fetch remaining structural relationships (PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, CONNECTS_TO)
        remaining_limit = edge_limit - len(edges) if edge_limit else None
        other_rel_types = "|".join(r for r in STRUCTURAL_REL_TYPES if r != "RUNS_ON")
        
        if other_rel_types:
            other_query = f"""
            MATCH (s)-[r:{other_rel_types}]->(t)
            WHERE s.id IN $node_ids AND t.id IN $node_ids
            RETURN s.id AS source_id, t.id AS target_id,
                   labels(s)[0] AS source_type, labels(t)[0] AS target_type,
                   type(r) AS relation_type, COALESCE(r.weight, 1.0) AS weight,
                   properties(r) AS props
            ORDER BY COALESCE(r.weight, 0.0) DESC
            {f'LIMIT {remaining_limit}' if remaining_limit else ''}
            """
            
            edges.extend(self._execute_edge_query(other_query, node_ids, "other_structural"))
        
        return edges

    def _fetch_depends_on_edges(self, node_ids: List[str], edge_limit: Optional[int]) -> List[EdgeData]:
        """Fetch derived DEPENDS_ON relationships."""
        limit_clause = f"LIMIT {edge_limit}" if edge_limit else ""
        
        query = f"""
        MATCH (s)-[r:DEPENDS_ON]->(t)
        WHERE s.id IN $node_ids AND t.id IN $node_ids
        RETURN s.id AS source_id, t.id AS target_id,
               labels(s)[0] AS source_type, labels(t)[0] AS target_type,
               COALESCE(r.dependency_type, 'unknown') AS dependency_type,
               COALESCE(r.weight, 1.0) AS weight, properties(r) AS props
        ORDER BY COALESCE(r.weight, 0.0) DESC
        {limit_clause}
        """
        
        return self._execute_edge_query(query, node_ids, "depends_on")

    def fetch_limited_edges(self, node_ids: List[str], fetch_structural: bool, edge_limit: Optional[int]) -> List[EdgeData]:
        """Fetch edges between limited nodes."""
        if fetch_structural:
            return self._fetch_structural_edges(node_ids, edge_limit)
        else:
            return self._fetch_depends_on_edges(node_ids, edge_limit)

    def get_graph_data(self, node_limit: int = 1000, fetch_structural: bool = False, edge_limit: Optional[int] = None, node_types: Optional[List[str]] = None) -> GraphData:
        """
        Retrieve graph data from Neo4j.
        
        Args:
            node_limit: Maximum number of nodes to retrieve (sorted by weight DESC)
            fetch_structural: If True, fetch structural relationships; if False, fetch DEPENDS_ON
            edge_limit: Maximum edges to retrieve (None = no limit)
            node_types: Node types to include (None = all types)
        
        Returns:
            GraphData with components and their edges
        """
        components, node_ids = self.fetch_limited_nodes(node_limit, node_types)
        edges = self.fetch_limited_edges(node_ids, fetch_structural, edge_limit)
        
        return GraphData(
            components=components,
            edges=edges
        )

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
        
        with self.exporter.driver.session() as session:
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
        
        with self.exporter.driver.session() as session:
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
        """Fetch topology data with drill-dawn support."""
        with self.exporter.driver.session() as session:
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
                logger.info(f"Node {node_id} is of type {node_type}")
                
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

    def get_structural_stats(self) -> Dict[str, int]:
        """Get counts of structural relationships."""
        with self.exporter.driver.session() as session:
            rel_types = "|".join(["PUBLISHES_TO", "SUBSCRIBES_TO", "RUNS_ON", "ROUTES", "CONNECTS_TO", "USES"])
            query = f"""
            MATCH ()-[r:{rel_types}]->()
            RETURN type(r) AS rel_type, count(r) AS count
            """
            result = session.run(query)
            counts = {}
            for record in result:
                counts[record["rel_type"]] = record["count"]
            return counts


    def get_pub_sub_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Fetch data for message flow analysis."""
        with self.exporter.driver.session() as session:
            # Get all PUBLISHES_TO relationships
            publish_query = """
            MATCH (app:Application)-[r:PUBLISHES_TO]->(topic:Topic)
            RETURN app.id AS app_id, app.name AS app_name, 
                   topic.id AS topic_id, topic.name AS topic_name,
                   COALESCE(r.weight, 1.0) AS weight
            """
            
            # Get all SUBSCRIBES_TO relationships
            subscribe_query = """
            MATCH (app:Application)-[r:SUBSCRIBES_TO]->(topic:Topic)
            RETURN app.id AS app_id, app.name AS app_name,
                   topic.id AS topic_id, topic.name AS topic_name,
                   COALESCE(r.weight, 1.0) AS weight
            """
            
            # Get all topic-broker relationships
            topic_broker_query = """
            MATCH (topic:Topic)-[r:ROUTES]->(broker:Broker)
            RETURN topic.id AS topic_id, topic.name AS topic_name,
                   broker.id AS broker_id, broker.name AS broker_name,
                   COALESCE(r.weight, 1.0) AS weight
            """
            
            # Get all applications
            app_query = """
            MATCH (app:Application)
            RETURN app.id AS id, COALESCE(app.name, app.id) AS name
            """
            
            publishes = [dict(record) for record in session.run(publish_query)]
            subscribes = [dict(record) for record in session.run(subscribe_query)]
            topic_brokers = [dict(record) for record in session.run(topic_broker_query)]
            all_apps = [dict(record) for record in session.run(app_query)]
            
            return publishes, subscribes, topic_brokers, all_apps
