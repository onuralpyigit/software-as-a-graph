from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, List, Optional
import logging

from api.models import (
    Neo4jCredentials,
    GenerateGraphRequest,
    GenerateGraphFileRequest,
    ImportGraphRequest,
    GraphGenerationResponse,
    GraphImportResponse,
    GraphGenerateImportResponse,
    GenericSuccessResponse,
    GraphExportResponse,
    LimitedGraphExportResponse,
    Neo4jExportResponse,
    SearchNodesResponse,
    NodeConnectionsResponse,
    TopologyResponse
)
from tools.generation import GenerationService
from src.core.ports.graph_repository import IGraphRepository
from api.dependencies import get_repository, get_generation_service
from api.presenters import graph_presenter

router = APIRouter(prefix="/api/v1/graph", tags=["graph"])
logger = logging.getLogger(__name__)


@router.post("/generate", response_model=GraphGenerationResponse)
async def generate_graph(
    request: GenerateGraphRequest,
    service: GenerationService = Depends(get_generation_service)
):
    """
    Generate a synthetic graph with specified scale and seed.
    """
    try:
        logger.info(f"Generating graph: scale={request.scale}, seed={request.seed}")
        graph_data = service.generate()
        return graph_presenter.format_generation_response(graph_data, request.scale)
    except Exception as e:
        logger.error(f"Graph generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Graph generation failed: {e}")


@router.post("/generate-file", response_model=GraphGenerationResponse)
async def generate_graph_file(
    request: GenerateGraphFileRequest,
    service: GenerationService = Depends(get_generation_service)
):
    """
    Generate a synthetic graph and return it as JSON (for download).
    """
    try:
        logger.info(f"Generating graph file: scale={request.scale}, seed={request.seed}")
        graph_data = service.generate()
        return graph_presenter.format_generation_response(graph_data, request.scale)
    except Exception as e:
        logger.error(f"Graph file generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Graph generation failed: {e}")


from api.dependencies import get_repository, get_client
from saag import Client

@router.post("/import", response_model=GraphImportResponse)
async def import_graph(
    request: ImportGraphRequest,
    client: Client = Depends(get_client)
):
    """Import graph data into Neo4j database."""
    try:
        logger.info(f"Importing graph data (clear={request.clear_database})")
        # client.import_topology returns a dict of stats
        stats_dict = client.import_topology(graph_data=request.graph_data, clear=request.clear_database)
        
        # Format expects the details dictionary directly wrapper
        return graph_presenter.format_import_response(stats_dict)
    except Exception as e:
        logger.error(f"Graph import failed: {e}")
        raise HTTPException(status_code=500, detail=f"Graph import failed: {e}")


@router.post("/generate-and-import", response_model=GraphGenerateImportResponse)
async def generate_and_import_graph(
    credentials: Neo4jCredentials,
    scale: str = Query(default="medium", description="Graph scale"),
    seed: int = Query(default=42, description="Random seed"),
    domain: Optional[str] = Query(default=None, description="Domain dataset"),
    scenario: Optional[str] = Query(default=None, description="QoS Scenario"),
    clear_database: bool = Query(default=False, description="Clear database before import"),
    repo: IGraphRepository = Depends(get_repository)
):
    """Convenience endpoint to generate and immediately import a graph."""
    try:
        logger.info(f"Generating and importing graph: scale={scale}, seed={seed}")
        gen_service = GenerationService(scale=scale, seed=seed, domain=domain, scenario=scenario)
        graph_data = gen_service.generate()
        
        repo.save_graph(graph_data, clear=clear_database)
        stats = repo.get_statistics()
        
        return graph_presenter.format_generate_import_response(graph_data, stats, scale, seed)
    except Exception as e:
        logger.error(f"Generate and import failed: {e}")
        raise HTTPException(status_code=500, detail=f"Operation failed: {e}")


@router.delete("/clear", response_model=GenericSuccessResponse)
@router.post("/clear", response_model=GenericSuccessResponse)
async def clear_database(
    credentials: Neo4jCredentials,
    repo: IGraphRepository = Depends(get_repository)
):
    """Clear all data from the Neo4j database."""
    try:
        logger.warning("Clearing Neo4j database")
        repo.save_graph({"nodes": [], "brokers": [], "topics": [], "applications": [], "libraries": [], "relationships": {}}, clear=True)
        return {"success": True, "message": "Database cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {e}")


@router.post("/export", response_model=GraphExportResponse)
async def export_graph(
    credentials: Neo4jCredentials,
    include_structural: bool = Query(default=True, description="Include structural relationships"),
    repo: IGraphRepository = Depends(get_repository)
):
    """Export the complete graph from Neo4j."""
    try:
        logger.info(f"Exporting graph data (include_structural={include_structural})")
        graph_data = repo.get_graph_data(include_raw=include_structural)
        stats = repo.get_statistics()
        return graph_presenter.format_export_response(graph_data, stats)
    except Exception as e:
        logger.error(f"Graph export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Graph export failed: {e}")


@router.post("/export-limited", response_model=LimitedGraphExportResponse)
async def export_limited_graph(
    credentials: Neo4jCredentials,
    node_limit: int = Query(default=1000, description="Max nodes"),
    edge_limit: Optional[int] = Query(default=None, description="Max edges"),
    fetch_structural: bool = Query(default=False, description="Fetch structural vs DEPENDS_ON"),
    node_types: Optional[List[str]] = Query(default=None, description="Node types"),
    repo: IGraphRepository = Depends(get_repository)
):
    """Export limited graph subset optimized for performance."""
    try:
        logger.info(f"Exporting limited graph: nodes={node_limit}")
        if hasattr(repo, 'get_limited_graph_data'):
            graph_subset = repo.get_limited_graph_data(node_limit, fetch_structural, edge_limit, node_types)
            components = graph_subset.components
            edges = graph_subset.edges
        else:
            graph_data = repo.get_graph_data(component_types=node_types, include_raw=fetch_structural)
            components = graph_data.components[:node_limit]
            comp_ids = {c.id for c in components}
            edges = [e for e in graph_data.edges if e.source_id in comp_ids and e.target_id in comp_ids]
            if edge_limit:
                 edges = edges[:edge_limit]
        
        return graph_presenter.format_limited_export_response(components, edges, node_limit, edge_limit)
    except Exception as e:
        logger.error(f"Limited graph export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Limited graph export failed: {e}")


@router.post("/export-neo4j-data", response_model=Neo4jExportResponse)
async def export_neo4j_data(
    credentials: Neo4jCredentials,
    repo: IGraphRepository = Depends(get_repository)
):
    """Export complete Neo4j graph data in input file format."""
    try:
        logger.info("Exporting Neo4j graph data to file format")
        graph_data = repo.export_json()
        return graph_presenter.format_neo4j_export_response(graph_data)
    except Exception as e:
        logger.error(f"Neo4j data export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Neo4j data export failed: {e}")


@router.get("/search-nodes", response_model=SearchNodesResponse)
async def search_nodes(
    query: str = Query(..., description="Search query"),
    limit: int = Query(default=20, description="Max results"),
    repo: IGraphRepository = Depends(get_repository)
):
    """Search for nodes across the entire database."""
    try:
        logger.info(f"Searching nodes with query: {query}")
        nodes = repo.search_nodes(query, limit)
        return {
            "success": True,
            "query": query,
            "count": len(nodes),
            "nodes": nodes
        }
    except Exception as e:
        logger.error(f"Node search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Node search failed: {e}")


@router.post("/node-connections", response_model=NodeConnectionsResponse)
async def get_node_connections(
    credentials: Neo4jCredentials,
    node_id: str = Query(..., description="Node ID"),
    fetch_structural: bool = Query(default=False, description="Fetch structural"),
    depth: int = Query(default=1, description="Depth level"),
    repo: IGraphRepository = Depends(get_repository)
):
    """Fetch all connections for a specific node at specified depth."""
    try:
        logger.info(f"Fetching connections for node: {node_id}, depth={depth}")
        depth = max(1, min(3, depth))
        components, edges = repo.get_node_connections(node_id, fetch_structural, depth)
        return graph_presenter.format_node_connections_response(node_id, depth, components, edges)
    except Exception as e:
        logger.error(f"Failed to fetch node connections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch node connections: {e}")


@router.post("/topology", response_model=TopologyResponse)
async def get_topology_data(
    credentials: Neo4jCredentials,
    node_id: Optional[str] = Query(None, description="Optional node ID"),
    node_limit: int = Query(default=1000, description="Max nodes"),
    repo: IGraphRepository = Depends(get_repository)
):
    """Fetch topology data with hierarchical drill-down."""
    try:
        logger.info(f"Fetching topology data, node_id={node_id}")
        components, edges = repo.get_topology_data(node_id, node_limit)
        return graph_presenter.format_topology_response(node_id, components, edges)
    except Exception as e:
        logger.error(f"Failed to fetch topology data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch topology data: {e}")
