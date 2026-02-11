from typing import Dict, Any, List, Optional, Tuple
import logging
from src.core.graph_exporter import GraphExporter, STRUCTURAL_REL_TYPES
from src.infrastructure.repositories.graph_query_repo import GraphQueryRepository

logger = logging.getLogger(__name__)

class GraphService:
    """
    Service for retrieving raw graph data, topology, and managing the graph.
    Encapsulates interactions with GraphExporter and GraphQueryRepository.
    """

    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password

    def check_connection(self) -> bool:
        """Verify connection to Neo4j."""
        exporter = GraphExporter(self.uri, self.user, self.password)
        try:
            exporter.driver.verify_connectivity()
            return True
        finally:
            exporter.close()

    def clear_database(self) -> None:
        """Clear all data from the database."""
        exporter = GraphExporter(self.uri, self.user, self.password)
        try:
            with exporter.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
        finally:
            exporter.close()

    def get_components(self, component_type: Optional[str] = None, min_weight: Optional[float] = None, limit: int = 100) -> Dict[str, Any]:
        """Get components from the graph with optional filtering."""
        exporter = GraphExporter(self.uri, self.user, self.password)
        try:
            if component_type:
                graph_data = exporter.get_graph_data(component_types=[component_type])
            else:
                graph_data = exporter.get_graph_data()
            
            components = [c.to_dict() for c in graph_data.components]
            
            # Apply weight filter if specified
            if min_weight is not None:
                components = [c for c in components if c.get("weight", 0) >= min_weight]
            
            # Apply limit
            components = components[:limit]
            
            return {
                "count": len(components),
                "components": components
            }
        finally:
            exporter.close()

    def get_edges(self, dependency_type: Optional[str] = None, min_weight: Optional[float] = None, limit: int = 100) -> Dict[str, Any]:
        """Get edges from the graph with optional filtering."""
        exporter = GraphExporter(self.uri, self.user, self.password)
        try:
            if dependency_type:
                graph_data = exporter.get_graph_data(dependency_types=[dependency_type])
            else:
                graph_data = exporter.get_graph_data()
            
            edges = [e.to_dict() for e in graph_data.edges]
            
            # Apply weight filter if specified
            if min_weight is not None:
                edges = [e for e in edges if e.get("weight", 0) >= min_weight]
            
            # Apply limit
            edges = edges[:limit]
            
            return {
                "count": len(edges),
                "edges": edges
            }
        finally:
            exporter.close()

    def get_limited_graph_data(self, node_limit: int, fetch_structural: bool, edge_limit: Optional[int], node_types: Optional[List[str]]) -> Any:
        """Get limited graph subset optimized for visualization."""
        exporter = GraphExporter(self.uri, self.user, self.password)
        try:
            repo = GraphQueryRepository(exporter)
            return repo.get_limited_graph_data(node_limit, fetch_structural, edge_limit, node_types)
        finally:
            exporter.close()

    def search_nodes(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search for nodes by ID or label."""
        exporter = GraphExporter(self.uri, self.user, self.password)
        try:
            repo = GraphQueryRepository(exporter)
            return repo.search_nodes(query, limit)
        finally:
            exporter.close()

    def get_node_connections(self, node_id: str, fetch_structural: bool, depth: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Fetch connections for a specific node."""
        exporter = GraphExporter(self.uri, self.user, self.password)
        try:
            repo = GraphQueryRepository(exporter)
            return repo.get_node_connections(node_id, fetch_structural, depth)
        finally:
            exporter.close()

    def get_topology_data(self, node_id: Optional[str], node_limit: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Fetch topology data with drill-down support."""
        exporter = GraphExporter(self.uri, self.user, self.password)
        try:
            repo = GraphQueryRepository(exporter)
            return repo.get_topology_data(node_id, node_limit)
        finally:
            exporter.close()

    def export_neo4j_data(self) -> Dict[str, Any]:
        """Export complete Neo4j graph data in file format."""
        exporter = GraphExporter(self.uri, self.user, self.password)
        try:
            return exporter.export_graph_data()
        finally:
            exporter.close()
            
    def export_graph(self, include_structural: bool = True) -> Dict[str, Any]:
        """Export the complete graph (components and edges)."""
        exporter = GraphExporter(self.uri, self.user, self.password)
        try:
            # Get dependency graph
            graph_data = exporter.get_graph_data()
            components_dict = {c.id: c for c in graph_data.components}
            edges_list = [e.to_dict() for e in graph_data.edges]
            
            # Also get structural relationships if requested
            if include_structural:
                with exporter.driver.session() as session:
                    # Get all structural relationships
                    rel_types = "|".join(STRUCTURAL_REL_TYPES)
                    query = f"MATCH (s)-[r:{rel_types}]->(t) RETURN s.id AS source_id, t.id AS target_id, labels(s)[0] AS source_type, labels(t)[0] AS target_type, type(r) AS relation_type, COALESCE(r.weight, 1.0) AS weight, properties(r) AS props"
                    result = session.run(query)
                    for record in result:
                        props = dict(record["props"])
                        props.pop("weight", None)
                        
                        edge_dict = {
                            "source": record["source_id"],
                            "target": record["target_id"],
                            "source_type": record["source_type"],
                            "target_type": record["target_type"],
                            "relation_type": record["relation_type"],
                            "dependency_type": record["relation_type"],
                            "weight": float(record["weight"]),
                            **props
                        }
                        edges_list.append(edge_dict)
            
            components_list = [c.to_dict() for c in components_dict.values()]
            
            return {
                "components": components_list,
                "edges": edges_list,
                "stats": {
                    "component_count": len(components_list),
                    "edge_count": len(edges_list)
                }
            }
        finally:
            exporter.close()
