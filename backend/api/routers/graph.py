"""
Graph generation, import, export, and query endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
import logging

from api.models import (
    Neo4jCredentials,
    GenerateGraphRequest,
    GenerateGraphFileRequest,
    ImportGraphRequest
)
from api.dependencies import DEFAULT_NEO4J_URI, DEFAULT_NEO4J_USER, DEFAULT_NEO4J_PASSWORD
from src.application.services.generation_service import GenerationService
from src.application.services.import_service import ImportService
from src.application.services.graph_service import GraphService

router = APIRouter(prefix="/api/v1/graph", tags=["graph"])
logger = logging.getLogger(__name__)


@router.post("/generate", response_model=Dict[str, Any])
async def generate_graph(request: GenerateGraphRequest):
    """
    Generate a synthetic graph with specified scale and seed.
    
    Scales: tiny (5 apps), small (15 apps), medium (50 apps), 
            large (150 apps), xlarge (500 apps)
    """
    try:
        logger.info(f"Generating graph: scale={request.scale}, seed={request.seed}")
        service = GenerationService(scale=request.scale, seed=request.seed)
        graph_data = service.generate()
        
        return {
            "success": True,
            "message": f"Graph generated successfully with scale '{request.scale}'",
            "metadata": graph_data.get("metadata", {}),
            "stats": {
                "nodes": len(graph_data.get("nodes", [])),
                "brokers": len(graph_data.get("brokers", [])),
                "topics": len(graph_data.get("topics", [])),
                "applications": len(graph_data.get("applications", []))
            },
            "graph_data": graph_data
        }
    except Exception as e:
        logger.error(f"Graph generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Graph generation failed: {str(e)}")


@router.post("/generate-file")
async def generate_graph_file(request: GenerateGraphFileRequest):
    """
    Generate a synthetic graph and return it as JSON (for download).
    This endpoint does not require database credentials.
    
    Scales: tiny (5 apps), small (15 apps), medium (50 apps), 
            large (150 apps), xlarge (500 apps)
    """
    try:
        logger.info(f"Generating graph file: scale={request.scale}, seed={request.seed}")
        service = GenerationService(scale=request.scale, seed=request.seed)
        graph_data = service.generate()
        
        # Return the graph data directly as JSON
        return {
            "success": True,
            "message": f"Graph generated successfully with scale '{request.scale}'",
            "metadata": graph_data.get("metadata", {}),
            "stats": {
                "nodes": len(graph_data.get("nodes", [])),
                "brokers": len(graph_data.get("brokers", [])),
                "topics": len(graph_data.get("topics", [])),
                "applications": len(graph_data.get("applications", []))
            },
            "graph_data": graph_data
        }
    except Exception as e:
        logger.error(f"Graph file generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Graph generation failed: {str(e)}")


@router.post("/import", response_model=Dict[str, Any])
async def import_graph(request: ImportGraphRequest):
    """
    Import graph data into Neo4j database.
    
    This will:
    1. Import nodes, brokers, topics, applications
    2. Create structural relationships
    3. Derive DEPENDS_ON relationships
    4. Calculate component weights
    """
    try:
        logger.info(f"Importing graph data (clear={request.clear_database})")
        creds = request.credentials
        
        with ImportService(creds.uri, creds.user, creds.password, creds.database) as service:
            stats = service.import_graph(request.graph_data, clear=request.clear_database)
        
        return {
            "success": True,
            "message": "Graph imported successfully",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Graph import failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Graph import failed: {str(e)}")


@router.post("/generate-and-import", response_model=Dict[str, Any])
async def generate_and_import_graph(
    credentials: Neo4jCredentials,
    scale: str = Query(default="medium", description="Graph scale"),
    seed: int = Query(default=42, description="Random seed"),
    clear_database: bool = Query(default=False, description="Clear database before import")
):
    """
    Convenience endpoint to generate and immediately import a graph.
    """
    try:
        # Generate
        logger.info(f"Generating graph: scale={scale}, seed={seed}")
        gen_service = GenerationService(scale=scale, seed=seed)
        graph_data = gen_service.generate()
        
        # Import
        logger.info(f"Importing generated graph (clear={clear_database})")
        with ImportService(credentials.uri, credentials.user, credentials.password, credentials.database) as imp_service:
            stats = imp_service.import_graph(graph_data, clear=clear_database)
        
        return {
            "success": True,
            "message": f"Graph generated (scale={scale}) and imported successfully",
            "generation": {
                "scale": scale,
                "seed": seed,
                "metadata": graph_data.get("metadata", {})
            },
            "import_stats": stats
        }
    except Exception as e:
        logger.error(f"Generate and import failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Operation failed: {str(e)}")


@router.delete("/clear", response_model=Dict[str, Any])
@router.post("/clear", response_model=Dict[str, Any])
async def clear_database(credentials: Neo4jCredentials):
    """
    Clear all data from the Neo4j database.
    WARNING: This will delete all nodes and relationships!
    
    Supports both DELETE and POST methods for compatibility.
    """
    try:
        logger.warning("Clearing Neo4j database - all data will be deleted")
        
        service = GraphService(credentials.uri, credentials.user, credentials.password)
        service.clear_database()
        
        logger.info("Database cleared successfully")
        
        return {
            "success": True,
            "message": "Database cleared successfully"
        }
    except Exception as e:
        logger.error(f"Failed to clear database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")


@router.post("/export", response_model=Dict[str, Any])
async def export_graph(
    credentials: Neo4jCredentials,
    include_structural: bool = Query(default=True, description="Include structural relationships (RUNS_ON, PUBLISHES_TO, etc.)")
):
    """
    Export the complete graph from Neo4j.
    Includes both derived DEPENDS_ON edges and structural relationships.
    """
    try:
        logger.info(f"Exporting graph data (include_structural={include_structural})")
        service = GraphService(credentials.uri, credentials.user, credentials.password)
        result = service.export_graph(include_structural=include_structural)
        
        return {
            "success": True,
            "components": result["components"],
            "edges": result["edges"],
            "stats": result["stats"]
        }
    except Exception as e:
        logger.error(f"Graph export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Graph export failed: {str(e)}")


@router.post("/export-limited")
async def export_limited_graph(
    credentials: Neo4jCredentials,
    node_limit: int = Query(default=1000, description="Max nodes to retrieve (sorted by weight)"),
    edge_limit: Optional[int] = Query(default=None, description="Max edges to retrieve (None = no limit)"),
    fetch_structural: bool = Query(default=False, description="Fetch structural relationships vs DEPENDS_ON"),
    node_types: Optional[List[str]] = Query(default=None, description="Node types to include (None = all types)")
):
    """
    Export limited graph subset optimized for performance.
    Fetches top N nodes by weight and edges between them.
    """
    try:
        logger.info(f"Exporting limited graph: nodes={node_limit}, edges={edge_limit}, structural={fetch_structural}, types={node_types}")
        
        service = GraphService(credentials.uri, credentials.user, credentials.password)
        graph_data = service.get_limited_graph_data(node_limit, fetch_structural, edge_limit, node_types)
        
        return {
            "success": True,
            "components": [c.to_dict() for c in graph_data.components],
            "edges": [e.to_dict() for e in graph_data.edges],
            "stats": {
                "component_count": len(graph_data.components),
                "edge_count": len(graph_data.edges),
                "node_limit": node_limit,
                "edge_limit": edge_limit,
                "limited": True
            }
        }
    except Exception as e:
        logger.error(f"Limited graph export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Limited graph export failed: {str(e)}")


@router.post("/export-neo4j-data")
async def export_neo4j_data(credentials: Neo4jCredentials):
    """
    Export complete Neo4j graph data in the same format as input files.
    Returns data compatible with GraphImporter, suitable for re-importing.
    
    This endpoint uses the export_graph_data() method from GraphExporter which
    produces a JSON structure matching the format of input/dataset.json.
    """
    try:
        logger.info("Exporting Neo4j graph data to file format")
        
        service = GraphService(credentials.uri, credentials.user, credentials.password)
        graph_data = service.export_neo4j_data()
        
        return {
            "success": True,
            "message": "Graph data exported successfully",
            "graph_data": graph_data,
            "stats": {
                "nodes": len(graph_data.get("nodes", [])),
                "brokers": len(graph_data.get("brokers", [])),
                "topics": len(graph_data.get("topics", [])),
                "applications": len(graph_data.get("applications", [])),
                "libraries": len(graph_data.get("libraries", []))
            }
        }
    except Exception as e:
        logger.error(f"Neo4j data export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Neo4j data export failed: {str(e)}")


@router.get("/search-nodes")
async def search_nodes(
    query: str = Query(..., description="Search query for node ID or label"),
    limit: int = Query(default=20, description="Maximum number of results"),
    uri: str = Query(default=DEFAULT_NEO4J_URI, description="Neo4j URI"),
    user: str = Query(default=DEFAULT_NEO4J_USER, description="Neo4j username"),
    password: str = Query(default=DEFAULT_NEO4J_PASSWORD, description="Neo4j password"),
    database: str = Query(default="neo4j", description="Neo4j database")
):
    """
    Search for nodes across the entire database by ID or label.
    Returns matching nodes without their connections.
    """
    try:
        logger.info(f"Searching nodes with query: {query}, limit={limit}")
        
        service = GraphService(uri, user, password)
        nodes = service.search_nodes(query, limit)
                
        logger.info(f"Found {len(nodes)} nodes matching query: {query}")
        return {
            "success": True,
            "query": query,
            "count": len(nodes),
            "nodes": nodes
        }
    
    except Exception as e:
        logger.error(f"Node search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Node search failed: {str(e)}")


@router.post("/node-connections")
async def get_node_connections(
    credentials: Neo4jCredentials,
    node_id: str = Query(..., description="Node ID to fetch connections for"),
    fetch_structural: bool = Query(default=False, description="Fetch structural relationships vs DEPENDS_ON"),
    depth: int = Query(default=1, description="Depth level for fetching connections (1, 2, or 3)")
):
    """
    Fetch all connections (edges and connected nodes) for a specific node at specified depth.
    Returns nodes and edges that can be merged into existing graph.
    depth=1: immediate neighbors
    depth=2: neighbors and their neighbors  
    depth=3: three levels of connections
    """
    try:
        logger.info(f"Fetching connections for node: {node_id}, structural={fetch_structural}, depth={depth}")
        
        # Clamp depth to valid range
        depth = max(1, min(3, depth))
        
        service = GraphService(credentials.uri, credentials.user, credentials.password)
        components, edges = service.get_node_connections(node_id, fetch_structural, depth)
        
        logger.info(f"Fetched {len(components)} connected nodes and {len(edges)} edges for node {node_id} at depth {depth}")
        
        return {
            "success": True,
            "node_id": node_id,
            "depth": depth,
            "components": components,
            "edges": edges,
            "stats": {
                "connected_nodes": len(components),
                "edges": len(edges)
            }
        }
    except Exception as e:
        logger.error(f"Failed to fetch node connections: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch node connections: {str(e)}")


@router.post("/topology")
async def get_topology_data(
    credentials: Neo4jCredentials,
    node_id: Optional[str] = Query(None, description="Optional node ID to filter topology (drill-down)"),
    node_limit: int = Query(default=1000, description="Maximum number of nodes to retrieve")
):
    """
    Fetch topology data with hierarchical drill-down:
    - No node_id: Show all Node nodes and CONNECTS_TO relationships
    - Node type clicked: Show Applications and Brokers running on that Node (RUNS_ON)
    - Application type clicked: Show Topics (PUBLISHES_TO/SUBSCRIBES_TO) and Libraries (USES)
    """
    try:
        logger.info(f"Fetching topology data, node_id={node_id}, limit={node_limit}")
        
        service = GraphService(credentials.uri, credentials.user, credentials.password)
        components, edges = service.get_topology_data(node_id, node_limit)
            
        logger.info(f"Fetched {len(components)} components and {len(edges)} edges for topology view")
            
        return {
            "success": True,
            "node_id": node_id,
            "components": components,
            "edges": edges,
            "stats": {
                "nodes": len(components),
                "edges": len(edges)
            }
        }
    except Exception as e:
        logger.error(f"Failed to fetch topology data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch topology data: {str(e)}")
