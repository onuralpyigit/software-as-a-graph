"""
Distributed System Graph API

FastAPI application exposing graph generation, import, analysis, and query capabilities.
"""

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import os
from datetime import datetime

# Core imports
from src.core.graph_generator import GraphGenerator
from src.core.graph_importer import GraphImporter
from src.core.graph_exporter import GraphExporter, GraphData, ComponentData, EdgeData, STRUCTURAL_REL_TYPES

# Analysis imports
from src.analysis.analyzer import GraphAnalyzer
from src.analysis.classifier import BoxPlotClassifier, CriticalityLevel
from src.analysis.structural_analyzer import StructuralAnalyzer
from src.analysis.quality_analyzer import QualityAnalyzer
from src.analysis.problem_detector import ProblemDetector

# Simulation imports
from src.simulation import Simulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Distributed System Graph API",
    description="API for generating, analyzing, and querying distributed system graphs",
    version="1.0.0"
)

# Configure CORS to allow frontend access from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when allow_origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Pydantic Models
# ============================================================================

# Default Neo4j connection settings — read from environment for Docker,
# fallback to localhost for local development.
_DEFAULT_NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
_DEFAULT_NEO4J_USER = os.environ.get("NEO4J_USERNAME", "neo4j")
_DEFAULT_NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")


def _resolve_neo4j_uri(client_uri: str) -> str:
    """Rewrite localhost/127.0.0.1 URIs to the Docker-internal Neo4j hostname.

    When the browser sends bolt://localhost:7687, this is unreachable from
    inside the API container.  If NEO4J_URI is set (Docker), replace the
    host portion so the connection goes to the correct container.
    """
    env_uri = os.environ.get("NEO4J_URI")
    if not env_uri:
        return client_uri  # Not running in Docker — use as-is

    import re
    # Match bolt:// or neo4j:// with localhost or 127.0.0.1
    if re.match(r"^(bolt|neo4j)(\+s?s?)?://(localhost|127\.0\.0\.1)(:\d+)?$", client_uri):
        return env_uri
    return client_uri


class Neo4jCredentials(BaseModel):
    uri: str = Field(default=_DEFAULT_NEO4J_URI, description="Neo4j connection URI")
    user: str = Field(default=_DEFAULT_NEO4J_USER, description="Neo4j username")
    password: str = Field(default=_DEFAULT_NEO4J_PASSWORD, description="Neo4j password")
    database: str = Field(default="neo4j", description="Neo4j database name")
    node_type: Optional[str] = Field(default=None, description="Filter by specific node type")

    def __init__(self, **data):
        super().__init__(**data)
        # Rewrite localhost URIs when running inside Docker
        object.__setattr__(self, 'uri', _resolve_neo4j_uri(self.uri))


class GraphRequestWithCredentials(BaseModel):
    credentials: Neo4jCredentials = Field(..., description="Neo4j connection credentials")


class GenerateGraphRequest(GraphRequestWithCredentials):
    scale: str = Field(default="medium", description="Graph scale: tiny, small, medium, large, xlarge")
    seed: int = Field(default=42, description="Random seed for reproducibility")


class GenerateGraphFileRequest(BaseModel):
    """Request for generating a graph file without database credentials"""
    scale: str = Field(default="medium", description="Graph scale: tiny, small, medium, large, xlarge")
    seed: int = Field(default=42, description="Random seed for reproducibility")


class ImportGraphRequest(GraphRequestWithCredentials):
    graph_data: Dict[str, Any] = Field(..., description="Graph data structure to import")
    clear_database: bool = Field(default=False, description="Clear database before import")


class AnalysisResponse(BaseModel):
    timestamp: str
    summary: Dict[str, Any]
    components: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    problems: List[Dict[str, Any]]
    stats: Dict[str, Any]


class ComponentQueryParams(BaseModel):
    component_type: Optional[str] = Field(None, description="Filter by component type")
    min_weight: Optional[float] = Field(None, description="Minimum weight threshold")
    criticality_level: Optional[str] = Field(None, description="Filter by criticality level")


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    neo4j_connected: bool
    message: Optional[str] = None


class EventSimulationRequest(GraphRequestWithCredentials):
    source_app: str = Field(..., description="Source application ID for event simulation")
    num_messages: int = Field(default=100, description="Number of messages to simulate")
    duration: float = Field(default=10.0, description="Simulation duration in seconds")


class FailureSimulationRequest(GraphRequestWithCredentials):
    target_id: str = Field(..., description="Target component ID to simulate failure")
    layer: str = Field(default="system", description="Analysis layer: app, infra, mw-app, mw-infra, system")
    cascade_probability: float = Field(default=1.0, description="Cascade propagation probability (0.0-1.0)")


class ExhaustiveSimulationRequest(GraphRequestWithCredentials):
    layer: str = Field(default="system", description="Analysis layer: app, infra, mw-app, mw-infra, system")
    cascade_probability: float = Field(default=1.0, description="Cascade propagation probability (0.0-1.0)")


class ReportRequest(GraphRequestWithCredentials):
    layers: List[str] = Field(default=["app", "infra", "system"], description="Layers to include in report")


# ============================================================================
# Helper Functions for Limited Graph Export
# ============================================================================

def _fetch_limited_nodes(exporter: GraphExporter, limit: int, node_types: Optional[List[str]] = None) -> tuple[List[ComponentData], List[str]]:
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
    
    with exporter.driver.session() as session:
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


def _execute_edge_query(exporter: GraphExporter, query: str, node_ids: List[str], edge_type: str) -> List[EdgeData]:
    """Execute edge query and parse results."""
    with exporter.driver.session() as session:
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


def _fetch_structural_edges(exporter: GraphExporter, node_ids: List[str], edge_limit: Optional[int]) -> List[EdgeData]:
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
    
    edges.extend(_execute_edge_query(exporter, runs_on_query, node_ids, "RUNS_ON"))
    
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
        
        edges.extend(_execute_edge_query(exporter, other_query, node_ids, "other_structural"))
    
    return edges


def _fetch_depends_on_edges(exporter: GraphExporter, node_ids: List[str], edge_limit: Optional[int]) -> List[EdgeData]:
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
    
    return _execute_edge_query(exporter, query, node_ids, "depends_on")


def _fetch_limited_edges(exporter: GraphExporter, node_ids: List[str], fetch_structural: bool, edge_limit: Optional[int]) -> List[EdgeData]:
    """Fetch edges between limited nodes."""
    if fetch_structural:
        return _fetch_structural_edges(exporter, node_ids, edge_limit)
    else:
        return _fetch_depends_on_edges(exporter, node_ids, edge_limit)


def get_limited_graph_data(exporter: GraphExporter, node_limit: int = 1000, fetch_structural: bool = False, edge_limit: Optional[int] = None, node_types: Optional[List[str]] = None) -> GraphData:
    """
    Retrieve limited graph data from Neo4j using LIMIT clause.
    
    Args:
        exporter: GraphExporter instance
        node_limit: Maximum number of nodes to retrieve (sorted by weight DESC)
        fetch_structural: If True, fetch structural relationships; if False, fetch DEPENDS_ON
        edge_limit: Maximum edges to retrieve (None = no limit)
        node_types: Node types to include (None = all types)
    
    Returns:
        GraphData with limited components and their edges
    """
    components, node_ids = _fetch_limited_nodes(exporter, node_limit, node_types)
    edges = _fetch_limited_edges(exporter, node_ids, fetch_structural, edge_limit)
    
    return GraphData(
        components=components,
        edges=edges
    )


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Distributed System Graph API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "stats": "/api/v1/stats",
            "generate": "/api/v1/graph/generate",
            "import": "/api/v1/graph/import",
            "export": "/api/v1/graph/export",
            "export_limited": "/api/v1/graph/export-limited",
            "export_neo4j_data": "/api/v1/graph/export-neo4j-data",
            "generate_and_import": "/api/v1/graph/generate-and-import",
            "analyze_full": "/api/v1/analysis/full",
            "analyze_type": "/api/v1/analysis/type/{component_type}",
            "analyze_layer": "/api/v1/analysis/layer/{layer}",
            "classify": "/api/v1/classify",
            "components": "/api/v1/components",
            "edges": "/api/v1/edges",
            "critical_components": "/api/v1/components/critical",
            "critical_edges": "/api/v1/edges/critical",
            "simulate_event": "/api/v1/simulation/event",
            "simulate_failure": "/api/v1/simulation/failure",
            "simulate_exhaustive": "/api/v1/simulation/exhaustive",
            "simulation_report": "/api/v1/simulation/report"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Verifies API is running.
    Note: Neo4j connection is validated via the /api/v1/connect endpoint.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        neo4j_connected=False,
        message="API is running. Configure Neo4j connection in settings."
    )


@app.post("/api/v1/connect", response_model=Dict[str, Any])
async def test_connection(credentials: Neo4jCredentials):
    """
    Test Neo4j database connection with provided credentials.
    This endpoint is used by the frontend to validate credentials.
    """
    try:
        logger.info(f"Testing Neo4j connection to {credentials.uri}")
        exporter = GraphExporter(credentials.uri, credentials.user, credentials.password)
        
        # Test connection with lightweight verification (no data fetching)
        exporter.driver.verify_connectivity()
        exporter.close()
        
        return {
            "success": True,
            "message": "Successfully connected to Neo4j",
            "neo4j_connected": True
        }
    except Exception as e:
        logger.error(f"Neo4j connection test failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Connection failed: {str(e)}")


# ============================================================================
# Graph Generation & Import Endpoints
# ============================================================================

@app.post("/api/v1/graph/generate", response_model=Dict[str, Any])
async def generate_graph(request: GenerateGraphRequest):
    """
    Generate a synthetic graph with specified scale and seed.
    
    Scales: tiny (5 apps), small (15 apps), medium (50 apps), 
            large (150 apps), xlarge (500 apps)
    """
    try:
        logger.info(f"Generating graph: scale={request.scale}, seed={request.seed}")
        generator = GraphGenerator(scale=request.scale, seed=request.seed)
        graph_data = generator.generate()
        
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


@app.post("/api/v1/graph/generate-file")
async def generate_graph_file(request: GenerateGraphFileRequest):
    """
    Generate a synthetic graph and return it as JSON (for download).
    This endpoint does not require database credentials.
    
    Scales: tiny (5 apps), small (15 apps), medium (50 apps), 
            large (150 apps), xlarge (500 apps)
    """
    try:
        logger.info(f"Generating graph file: scale={request.scale}, seed={request.seed}")
        generator = GraphGenerator(scale=request.scale, seed=request.seed)
        graph_data = generator.generate()
        
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


@app.post("/api/v1/graph/import", response_model=Dict[str, Any])
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
        
        with GraphImporter(creds.uri, creds.user, creds.password, creds.database) as importer:
            stats = importer.import_graph(request.graph_data, clear=request.clear_database)
        
        return {
            "success": True,
            "message": "Graph imported successfully",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Graph import failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Graph import failed: {str(e)}")


@app.post("/api/v1/graph/generate-and-import", response_model=Dict[str, Any])
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
        generator = GraphGenerator(scale=scale, seed=seed)
        graph_data = generator.generate()
        
        # Import
        logger.info(f"Importing generated graph (clear={clear_database})")
        with GraphImporter(credentials.uri, credentials.user, credentials.password, credentials.database) as importer:
            stats = importer.import_graph(graph_data, clear=clear_database)
        
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


@app.delete("/api/v1/graph/clear", response_model=Dict[str, Any])
@app.post("/api/v1/graph/clear", response_model=Dict[str, Any])
async def clear_database(credentials: Neo4jCredentials):
    """
    Clear all data from the Neo4j database.
    WARNING: This will delete all nodes and relationships!
    
    Supports both DELETE and POST methods for compatibility.
    """
    try:
        logger.warning("Clearing Neo4j database - all data will be deleted")
        
        with GraphExporter(
            uri=credentials.uri,
            user=credentials.user,
            password=credentials.password
        ) as exporter:
            with exporter.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared successfully")
        
        return {
            "success": True,
            "message": "Database cleared successfully"
        }
    except Exception as e:
        logger.error(f"Failed to clear database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")


@app.post("/api/v1/graph/export", response_model=Dict[str, Any])
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
        exporter = GraphExporter(credentials.uri, credentials.user, credentials.password)
        
        # Get dependency graph
        graph_data = exporter.get_graph_data()
        components_dict = {c.id: c for c in graph_data.components}
        edges_list = [e.to_dict() for e in graph_data.edges]
        
        # Also get structural relationships if requested
        if include_structural:
            # Query structural relationships directly
            with exporter.driver.session() as session:
                # Get all structural relationships
                rel_types = "|".join(STRUCTURAL_REL_TYPES)
                query = f"""
                MATCH (s)-[r:{rel_types}]->(t)
                RETURN s.id AS source_id, t.id AS target_id,
                       labels(s)[0] AS source_type, labels(t)[0] AS target_type,
                       type(r) AS relation_type, COALESCE(r.weight, 1.0) AS weight,
                       properties(r) AS props
                """
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
        
        exporter.close()
        
        components_list = [c.to_dict() for c in components_dict.values()]
        
        return {
            "success": True,
            "components": components_list,
            "edges": edges_list,
            "stats": {
                "component_count": len(components_list),
                "edge_count": len(edges_list)
            }
        }
    except Exception as e:
        logger.error(f"Graph export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Graph export failed: {str(e)}")


@app.post("/api/v1/graph/export-limited")
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
        
        with GraphExporter(
            uri=credentials.uri,
            user=credentials.user,
            password=credentials.password
        ) as exporter:
            graph_data = get_limited_graph_data(exporter, node_limit, fetch_structural, edge_limit, node_types)
        
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


@app.post("/api/v1/graph/export-neo4j-data")
async def export_neo4j_data(credentials: Neo4jCredentials):
    """
    Export complete Neo4j graph data in the same format as input files.
    Returns data compatible with GraphImporter, suitable for re-importing.
    
    This endpoint uses the export_graph_data() method from GraphExporter which
    produces a JSON structure matching the format of input/dataset.json.
    """
    try:
        logger.info("Exporting Neo4j graph data to file format")
        
        with GraphExporter(
            uri=credentials.uri,
            user=credentials.user,
            password=credentials.password
        ) as exporter:
            # Use the export_graph_data method which returns data in input file format
            graph_data = exporter.export_graph_data()
        
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


@app.get("/api/v1/graph/search-nodes")
async def search_nodes(
    query: str = Query(..., description="Search query for node ID or label"),
    limit: int = Query(default=20, description="Maximum number of results"),
    uri: str = Query(default=_DEFAULT_NEO4J_URI, description="Neo4j URI"),
    user: str = Query(default=_DEFAULT_NEO4J_USER, description="Neo4j username"),
    password: str = Query(default=_DEFAULT_NEO4J_PASSWORD, description="Neo4j password"),
    database: str = Query(default="neo4j", description="Neo4j database")
):
    """
    Search for nodes across the entire database by ID or label.
    Returns matching nodes without their connections.
    """
    try:
        logger.info(f"Searching nodes with query: {query}, limit={limit}")
        
        with GraphExporter(
            uri=uri,
            user=user,
            password=password
        ) as exporter:
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
            
            with exporter.driver.session() as session:
                result = session.run(cypher_query, search_term=query, limit=limit)
                nodes = []
                for record in result:
                    nodes.append({
                        "id": record["id"],
                        "type": record["type"],
                        "label": record["label"],
                        "weight": float(record["weight"])
                    })
                
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


@app.post("/api/v1/graph/node-connections")
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
        
        with GraphExporter(
            uri=credentials.uri,
            user=credentials.user,
            password=credentials.password
        ) as exporter:
            # First, fetch the center node itself
            center_node_query = """
            MATCH (center {id: $node_id})
            RETURN center.id AS id, labels(center)[0] AS type,
                   COALESCE(center.weight, 1.0) AS weight, properties(center) AS props
            """
            
            # Determine relationship types based on view
            if fetch_structural:
                rel_types = "|".join(["PUBLISHES_TO", "SUBSCRIBES_TO", "RUNS_ON", "ROUTES", "CONNECTS_TO", "USES"])
                # Use variable-length path pattern for multi-hop traversal
                query = f"""
                MATCH (center {{id: $node_id}})
                MATCH path = (center)-[:{rel_types}*1..{depth}]-(connected)
                WITH DISTINCT connected
                RETURN connected.id AS id, labels(connected)[0] AS type,
                       COALESCE(connected.weight, 1.0) AS weight, properties(connected) AS props
                """
                
                # Fetch all edges in the subgraph at the specified depth (both directions)
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
                # Use variable-length path pattern for multi-hop traversal
                query = f"""
                MATCH (center {{id: $node_id}})
                MATCH path = (center)-[:DEPENDS_ON*1..{depth}]-(connected)
                WITH DISTINCT connected
                RETURN connected.id AS id, labels(connected)[0] AS type,
                       COALESCE(connected.weight, 1.0) AS weight, properties(connected) AS props
                """
                
                # Fetch all DEPENDS_ON edges in the subgraph at the specified depth (both directions)
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
            
            # Fetch center node and connected nodes
            with exporter.driver.session() as session:
                # First, add the center node itself
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
                
                # Then fetch all connected nodes at specified depth
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


@app.post("/api/v1/graph/topology")
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
        
        with GraphExporter(
            uri=credentials.uri,
            user=credentials.user,
            password=credentials.password
        ) as exporter:
            with exporter.driver.session() as session:
                components = []
                edges = []
                
                if node_id:
                    # First, determine the type of the selected node
                    type_query = """
                    MATCH (n {id: $node_id})
                    RETURN labels(n)[0] AS type
                    """
                    result = session.run(type_query, node_id=node_id)
                    record = result.single()
                    
                    if not record:
                        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
                    
                    node_type = record["type"]
                    logger.info(f"Node {node_id} is of type {node_type}")
                    
                    if node_type == "Node":
                        # Level 2: Show Applications and Brokers running on this Node
                        node_query = """
                        MATCH (center:Node {id: $node_id})
                        RETURN center.id AS id, labels(center)[0] AS type,
                               COALESCE(center.weight, 1.0) AS weight,
                               properties(center) AS props
                        UNION
                        MATCH (center:Node {id: $node_id})<-[:RUNS_ON]-(entity)
                        WHERE entity:Application OR entity:Broker
                        RETURN entity.id AS id, labels(entity)[0] AS type,
                               COALESCE(entity.weight, 1.0) AS weight,
                               properties(entity) AS props
                        """
                        
                        result = session.run(node_query, node_id=node_id)
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
                        
                        # Fetch RUNS_ON edges
                        edges_query = """
                        MATCH (entity)-[r:RUNS_ON]->(center:Node {id: $node_id})
                        WHERE entity:Application OR entity:Broker
                        RETURN entity.id AS source_id, center.id AS target_id,
                               labels(entity)[0] AS source_type, labels(center)[0] AS target_type,
                               COALESCE(r.weight, 1.0) AS weight,
                               properties(r) AS props
                        """
                        result = session.run(edges_query, node_id=node_id)
                        
                        for record in result:
                            props = dict(record["props"])
                            props.pop("weight", None)
                            edges.append({
                                "source": record["source_id"],
                                "target": record["target_id"],
                                "source_type": record["source_type"],
                                "target_type": record["target_type"],
                                "relation_type": "RUNS_ON",
                                "weight": float(record["weight"]),
                                **props
                            })
                    
                    elif node_type == "Application":
                        # Level 3: Show Topics and Libraries related to this Application
                        node_query = """
                        MATCH (center:Application {id: $node_id})
                        RETURN center.id AS id, labels(center)[0] AS type,
                               COALESCE(center.weight, 1.0) AS weight,
                               properties(center) AS props
                        UNION
                        MATCH (center:Application {id: $node_id})-[:PUBLISHES_TO|SUBSCRIBES_TO]->(topic:Topic)
                        RETURN topic.id AS id, labels(topic)[0] AS type,
                               COALESCE(topic.weight, 1.0) AS weight,
                               properties(topic) AS props
                        UNION
                        MATCH (center:Application {id: $node_id})-[:USES]->(lib:Library)
                        RETURN lib.id AS id, labels(lib)[0] AS type,
                               COALESCE(lib.weight, 1.0) AS weight,
                               properties(lib) AS props
                        """
                        
                        result = session.run(node_query, node_id=node_id)
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
                        
                        # Fetch PUBLISHES_TO, SUBSCRIBES_TO, and USES edges
                        edges_query = """
                        MATCH (center:Application {id: $node_id})-[r:PUBLISHES_TO|SUBSCRIBES_TO|USES]->(target)
                        WHERE target:Topic OR target:Library
                        RETURN center.id AS source_id, target.id AS target_id,
                               labels(center)[0] AS source_type, labels(target)[0] AS target_type,
                               type(r) AS relation_type,
                               COALESCE(r.weight, 1.0) AS weight,
                               properties(r) AS props
                        """
                        result = session.run(edges_query, node_id=node_id)
                        
                        for record in result:
                            props = dict(record["props"])
                            props.pop("weight", None)
                            edges.append({
                                "source": record["source_id"],
                                "target": record["target_id"],
                                "source_type": record["source_type"],
                                "target_type": record["target_type"],
                                "relation_type": record["relation_type"],
                                "weight": float(record["weight"]),
                                **props
                            })
                    
                    elif node_type == "Broker":
                        # Level 3 for Broker: Show Topics it routes
                        node_query = """
                        MATCH (center:Broker {id: $node_id})
                        RETURN center.id AS id, labels(center)[0] AS type,
                               COALESCE(center.weight, 1.0) AS weight,
                               properties(center) AS props
                        UNION
                        MATCH (center:Broker {id: $node_id})-[:ROUTES]->(topic:Topic)
                        RETURN topic.id AS id, labels(topic)[0] AS type,
                               COALESCE(topic.weight, 1.0) AS weight,
                               properties(topic) AS props
                        """
                        
                        result = session.run(node_query, node_id=node_id)
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
                        
                        # Fetch ROUTES edges
                        edges_query = """
                        MATCH (center:Broker {id: $node_id})-[r:ROUTES]->(topic:Topic)
                        RETURN center.id AS source_id, topic.id AS target_id,
                               labels(center)[0] AS source_type, labels(topic)[0] AS target_type,
                               COALESCE(r.weight, 1.0) AS weight,
                               properties(r) AS props
                        """
                        result = session.run(edges_query, node_id=node_id)
                        
                        for record in result:
                            props = dict(record["props"])
                            props.pop("weight", None)
                            edges.append({
                                "source": record["source_id"],
                                "target": record["target_id"],
                                "source_type": record["source_type"],
                                "target_type": record["target_type"],
                                "relation_type": "ROUTES",
                                "weight": float(record["weight"]),
                                **props
                            })
                    
                    elif node_type == "Library":
                        # Level 3 for Library: Show Libraries it uses and Applications using it
                        node_query = """
                        MATCH (center:Library {id: $node_id})
                        RETURN center.id AS id, labels(center)[0] AS type,
                               COALESCE(center.weight, 1.0) AS weight,
                               properties(center) AS props
                        UNION
                        MATCH (center:Library {id: $node_id})-[:USES]->(lib:Library)
                        RETURN lib.id AS id, labels(lib)[0] AS type,
                               COALESCE(lib.weight, 1.0) AS weight,
                               properties(lib) AS props
                        UNION
                        MATCH (app:Application)-[:USES]->(center:Library {id: $node_id})
                        RETURN app.id AS id, labels(app)[0] AS type,
                               COALESCE(app.weight, 1.0) AS weight,
                               properties(app) AS props
                        """
                        
                        result = session.run(node_query, node_id=node_id)
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
                        
                        # Fetch USES edges (both outgoing to other Libraries and incoming from Applications)
                        edges_query = """
                        MATCH (center:Library {id: $node_id})-[r:USES]->(lib:Library)
                        RETURN center.id AS source_id, lib.id AS target_id,
                               labels(center)[0] AS source_type, labels(lib)[0] AS target_type,
                               COALESCE(r.weight, 1.0) AS weight,
                               properties(r) AS props
                        UNION
                        MATCH (app:Application)-[r:USES]->(center:Library {id: $node_id})
                        RETURN app.id AS source_id, center.id AS target_id,
                               labels(app)[0] AS source_type, labels(center)[0] AS target_type,
                               COALESCE(r.weight, 1.0) AS weight,
                               properties(r) AS props
                        """
                        result = session.run(edges_query, node_id=node_id)
                        
                        for record in result:
                            props = dict(record["props"])
                            props.pop("weight", None)
                            edges.append({
                                "source": record["source_id"],
                                "target": record["target_id"],
                                "source_type": record["source_type"],
                                "target_type": record["target_type"],
                                "relation_type": "USES",
                                "weight": float(record["weight"]),
                                **props
                            })
                    
                    elif node_type == "Topic":
                        # Level 3 for Topic: Show Brokers that route it and Applications that publish/subscribe
                        node_query = """
                        MATCH (center:Topic {id: $node_id})
                        RETURN center.id AS id, labels(center)[0] AS type,
                               COALESCE(center.weight, 1.0) AS weight,
                               properties(center) AS props
                        UNION
                        MATCH (broker:Broker)-[:ROUTES]->(center:Topic {id: $node_id})
                        RETURN broker.id AS id, labels(broker)[0] AS type,
                               COALESCE(broker.weight, 1.0) AS weight,
                               properties(broker) AS props
                        UNION
                        MATCH (app:Application)-[:PUBLISHES_TO|SUBSCRIBES_TO]->(center:Topic {id: $node_id})
                        RETURN app.id AS id, labels(app)[0] AS type,
                               COALESCE(app.weight, 1.0) AS weight,
                               properties(app) AS props
                        """
                        
                        result = session.run(node_query, node_id=node_id)
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
                        
                        # Fetch ROUTES and PUBLISHES_TO/SUBSCRIBES_TO edges
                        edges_query = """
                        MATCH (broker:Broker)-[r:ROUTES]->(center:Topic {id: $node_id})
                        RETURN broker.id AS source_id, center.id AS target_id,
                               labels(broker)[0] AS source_type, labels(center)[0] AS target_type,
                               type(r) AS relation_type,
                               COALESCE(r.weight, 1.0) AS weight,
                               properties(r) AS props
                        UNION
                        MATCH (app:Application)-[r:PUBLISHES_TO|SUBSCRIBES_TO]->(center:Topic {id: $node_id})
                        RETURN app.id AS source_id, center.id AS target_id,
                               labels(app)[0] AS source_type, labels(center)[0] AS target_type,
                               type(r) AS relation_type,
                               COALESCE(r.weight, 1.0) AS weight,
                               properties(r) AS props
                        """
                        result = session.run(edges_query, node_id=node_id)
                        
                        for record in result:
                            props = dict(record["props"])
                            props.pop("weight", None)
                            edges.append({
                                "source": record["source_id"],
                                "target": record["target_id"],
                                "source_type": record["source_type"],
                                "target_type": record["target_type"],
                                "relation_type": record["relation_type"],
                                "weight": float(record["weight"]),
                                **props
                            })
                    else:
                        # For other node types, just return the node itself
                        node_query = """
                        MATCH (n {id: $node_id})
                        RETURN n.id AS id, labels(n)[0] AS type,
                               COALESCE(n.weight, 1.0) AS weight,
                               properties(n) AS props
                        """
                        result = session.run(node_query, node_id=node_id)
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
                
                else:
                    # Level 1: Full topology - fetch all Node nodes and CONNECTS_TO relationships
                    node_query = f"""
                    MATCH (n:Node)
                    RETURN n.id AS id, labels(n)[0] AS type,
                           COALESCE(n.weight, 1.0) AS weight,
                           properties(n) AS props
                    ORDER BY COALESCE(n.weight, 0.0) DESC
                    LIMIT $limit
                    """
                    
                    result = session.run(node_query, limit=node_limit)
                    node_ids = []
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
                        node_ids.append(record["id"])
                    
                    # Fetch CONNECTS_TO edges between these nodes
                    edges_query = """
                    MATCH (s:Node)-[r:CONNECTS_TO]->(t:Node)
                    WHERE s.id IN $node_ids AND t.id IN $node_ids
                    RETURN s.id AS source_id, t.id AS target_id,
                           labels(s)[0] AS source_type, labels(t)[0] AS target_type,
                           COALESCE(r.weight, 1.0) AS weight,
                           properties(r) AS props
                    ORDER BY COALESCE(r.weight, 0.0) DESC
                    """
                    result = session.run(edges_query, node_ids=node_ids)
                    
                    for record in result:
                        props = dict(record["props"])
                        props.pop("weight", None)
                        edges.append({
                            "source": record["source_id"],
                            "target": record["target_id"],
                            "source_type": record["source_type"],
                            "target_type": record["target_type"],
                            "relation_type": "CONNECTS_TO",
                            "weight": float(record["weight"]),
                            **props
                        })
        
        logger.info(f"Topology data: {len(components)} nodes, {len(edges)} edges")
        
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


# ============================================================================
# Analysis Endpoints
# ============================================================================

@app.post("/api/v1/analysis/full", response_model=Dict[str, Any])
async def analyze_full_system(credentials: Neo4jCredentials):
    """
    Run complete system analysis including:
    - Structural metrics (centrality, clustering, etc.)
    - Quality scores (reliability, maintainability, availability)
    - Problem detection
    """
    try:
        logger.info("Running full system analysis")
        
        with GraphAnalyzer(credentials.uri, credentials.user, credentials.password) as analyzer:
            result = analyzer.analyze_layer("system")
        
        # Create a map of component IDs to names from structural data
        component_names = {c.id: c.structural.name if c.structural and hasattr(c.structural, 'name') else c.id 
                          for c in result.quality.components}
        
        return {
            "success": True,
            "analysis": {
                "context": result.layer_name,
                "layer": result.layer,
                "description": result.description,
                "summary": {
                    "total_components": result.quality.classification_summary.total_components,
                    "total_edges": result.quality.classification_summary.total_edges,
                    "critical_count": result.quality.classification_summary.component_distribution.get("critical", 0),
                    "high_count": result.quality.classification_summary.component_distribution.get("high", 0),
                    "medium_count": result.quality.classification_summary.component_distribution.get("medium", 0),
                    "low_count": result.quality.classification_summary.component_distribution.get("low", 0),
                    "minimal_count": result.quality.classification_summary.component_distribution.get("minimal", 0),
                    "total_problems": result.problem_summary.total_problems,
                    "critical_problems": result.problem_summary.by_severity.get("CRITICAL", 0),
                    "components": dict(result.quality.classification_summary.component_distribution),
                    "edges": dict(result.quality.classification_summary.edge_distribution)
                },
                "stats": {
                    "nodes": result.structural.graph_summary.nodes,
                    "edges": result.structural.graph_summary.edges,
                    "density": result.structural.graph_summary.density,
                    "avg_degree": result.structural.graph_summary.avg_degree,
                    "avg_clustering": result.structural.graph_summary.avg_clustering,
                    "is_connected": result.structural.graph_summary.is_connected,
                    "num_components": result.structural.graph_summary.num_components,
                    "num_articulation_points": result.structural.graph_summary.num_articulation_points,
                    "num_bridges": result.structural.graph_summary.num_bridges,
                    "diameter": result.structural.graph_summary.diameter,
                    "avg_path_length": result.structural.graph_summary.avg_path_length
                },
                "components": [
                    {
                        "id": c.id,
                        "name": c.structural.name if c.structural and hasattr(c.structural, 'name') else c.id,
                        "type": c.type,
                        "criticality_level": c.levels.overall.value,
                        "criticality_levels": {
                            "reliability": c.levels.reliability.value,
                            "maintainability": c.levels.maintainability.value,
                            "availability": c.levels.availability.value,
                            "vulnerability": c.levels.vulnerability.value,
                            "overall": c.levels.overall.value
                        },
                        "scores": {
                            "reliability": c.scores.reliability,
                            "maintainability": c.scores.maintainability,
                            "availability": c.scores.availability,
                            "vulnerability": c.scores.vulnerability,
                            "overall": c.scores.overall
                        }
                    }
                    for c in result.quality.components
                ],
                "edges": [
                    {
                        "source": e.source,
                        "target": e.target,
                        "source_name": component_names.get(e.source, e.source),
                        "target_name": component_names.get(e.target, e.target),
                        "type": e.dependency_type,
                        "criticality_level": e.level.value,
                        "scores": {
                            "reliability": e.scores.reliability,
                            "maintainability": e.scores.maintainability,
                            "availability": e.scores.availability,
                            "vulnerability": e.scores.vulnerability,
                            "overall": e.scores.overall
                        }
                    }
                    for e in result.quality.edges
                ],
                "problems": [
                    {
                        "entity_id": p.entity_id,
                        "type": p.entity_type,
                        "category": p.category.value if hasattr(p.category, 'value') else str(p.category),
                        "severity": p.severity.value if hasattr(p.severity, 'value') else str(p.severity),
                        "name": p.name,
                        "description": p.description,
                        "recommendation": p.recommendation
                    }
                    for p in result.problems
                ]
            }
        }
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/v1/analysis/type/{component_type}", response_model=Dict[str, Any])
async def analyze_by_type(component_type: str, credentials: Neo4jCredentials):
    """
    Analyze components of a specific type.
    
    Valid types: Application, Broker, Node, Topic
    """
    valid_types = ["Application", "Broker", "Node", "Topic"]
    if component_type not in valid_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid component type. Must be one of: {', '.join(valid_types)}"
        )
    
    try:
        logger.info(f"Analyzing component type: {component_type}")
        
        with GraphAnalyzer(credentials.uri, credentials.user, credentials.password) as analyzer:
            # Analyze full system instead of specific layer
            result = analyzer.analyze_layer("system", context=f"Type: {component_type}")
            
            # Filter components by type
            filtered_components = [c for c in result.quality.components if c.type == component_type]
        
        # Create a map of component IDs to names from structural data
        component_names = {c.id: c.structural.name if c.structural and hasattr(c.structural, 'name') else c.id 
                          for c in result.quality.components}
        
        # Create a set of filtered component IDs for problem filtering
        filtered_component_ids = {c.id for c in filtered_components}
        
        # Filter edges related to these components
        filtered_edges = [
            e for e in result.quality.edges
            if e.source in filtered_component_ids or e.target in filtered_component_ids
        ]
        
        return {
            "success": True,
            "component_type": component_type,
            "analysis": {
                "context": f"Type: {component_type}",
                "layer": result.layer,
                "description": result.description,
                "summary": {
                    "total_components": len(filtered_components),
                    "critical_count": sum(1 for c in filtered_components if c.levels.overall.value == "critical"),
                    "high_count": sum(1 for c in filtered_components if c.levels.overall.value == "high"),
                    "components": {
                        "critical": sum(1 for c in filtered_components if c.levels.overall.value == "critical"),
                        "high": sum(1 for c in filtered_components if c.levels.overall.value == "high"),
                        "medium": sum(1 for c in filtered_components if c.levels.overall.value == "medium"),
                        "low": sum(1 for c in filtered_components if c.levels.overall.value == "low"),
                        "minimal": sum(1 for c in filtered_components if c.levels.overall.value == "minimal")
                    },
                    "edges": {
                        "critical": sum(1 for e in filtered_edges if e.level.value == "critical"),
                        "high": sum(1 for e in filtered_edges if e.level.value == "high"),
                        "medium": sum(1 for e in filtered_edges if e.level.value == "medium"),
                        "low": sum(1 for e in filtered_edges if e.level.value == "low"),
                        "minimal": sum(1 for e in filtered_edges if e.level.value == "minimal")
                    }
                },
                "stats": {
                    "nodes": len(filtered_components),
                    "edges": len(filtered_edges)
                },
                "components": [
                    {
                        "id": c.id,
                        "name": c.structural.name if c.structural and hasattr(c.structural, 'name') else c.id,
                        "type": c.type,
                        "criticality_level": c.levels.overall.value,
                        "criticality_levels": {
                            "reliability": c.levels.reliability.value,
                            "maintainability": c.levels.maintainability.value,
                            "availability": c.levels.availability.value,
                            "vulnerability": c.levels.vulnerability.value,
                            "overall": c.levels.overall.value
                        },
                        "scores": {
                            "reliability": c.scores.reliability,
                            "maintainability": c.scores.maintainability,
                            "availability": c.scores.availability,
                            "vulnerability": c.scores.vulnerability,
                            "overall": c.scores.overall
                        }
                    }
                    for c in filtered_components
                ],
                "edges": [
                    {
                        "source": e.source,
                        "target": e.target,
                        "source_name": component_names.get(e.source, e.source),
                        "target_name": component_names.get(e.target, e.target),
                        "type": e.dependency_type,
                        "criticality_level": e.level.value,
                        "scores": {
                            "reliability": e.scores.reliability,
                            "maintainability": e.scores.maintainability,
                            "availability": e.scores.availability,
                            "vulnerability": e.scores.vulnerability,
                            "overall": e.scores.overall
                        }
                    }
                    for e in result.quality.edges
                    if e.source in filtered_component_ids or e.target in filtered_component_ids
                ],
                "problems": [
                    {
                        "entity_id": p.entity_id,
                        "type": p.entity_type,
                        "category": p.category.value if hasattr(p.category, 'value') else str(p.category),
                        "severity": p.severity.value if hasattr(p.severity, 'value') else str(p.severity),
                        "name": p.name,
                        "description": p.description,
                        "recommendation": p.recommendation
                    }
                    for p in result.problems if p.entity_id in filtered_component_ids
                ]
            }
        }
    except Exception as e:
        logger.error(f"Type analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/v1/analysis/layer/{layer}", response_model=Dict[str, Any])
async def analyze_layer(layer: str, credentials: Neo4jCredentials):
    """
    Analyze a specific architectural layer.
    
    Valid layers: app, infra, application, infrastructure, system
    """
    valid_layers = ["app", "infra", "application", "infrastructure", "system", "mw-app", "mw-infra"]
    if layer not in valid_layers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid layer. Must be one of: {', '.join(valid_layers)}"
        )
    
    try:
        logger.info(f"Analyzing layer: {layer}")
        
        with GraphAnalyzer(credentials.uri, credentials.user, credentials.password) as analyzer:
            result = analyzer.analyze_layer(layer)
        
        # Create a map of component IDs to names from structural data
        component_names = {c.id: c.structural.name if c.structural and hasattr(c.structural, 'name') else c.id 
                          for c in result.quality.components}
        
        return {
            "success": True,
            "layer": result.layer,
            "analysis": {
                "context": result.layer_name,
                "description": result.description,
                "summary": {
                    "total_components": result.quality.classification_summary.total_components,
                    "critical_count": result.quality.classification_summary.component_distribution.get("critical", 0),
                    "high_count": result.quality.classification_summary.component_distribution.get("high", 0),
                    "total_problems": result.problem_summary.total_problems,
                    "critical_problems": result.problem_summary.by_severity.get("CRITICAL", 0),
                    "components": dict(result.quality.classification_summary.component_distribution),
                    "edges": dict(result.quality.classification_summary.edge_distribution)
                },
                "stats": {
                    "nodes": result.structural.graph_summary.nodes,
                    "edges": result.structural.graph_summary.edges,
                    "density": result.structural.graph_summary.density,
                    "avg_degree": result.structural.graph_summary.avg_degree
                },
                "components": [
                    {
                        "id": c.id,
                        "name": c.structural.name if c.structural and hasattr(c.structural, 'name') else c.id,
                        "type": c.type,
                        "criticality_level": c.levels.overall.value,
                        "criticality_levels": {
                            "reliability": c.levels.reliability.value,
                            "maintainability": c.levels.maintainability.value,
                            "availability": c.levels.availability.value,
                            "vulnerability": c.levels.vulnerability.value,
                            "overall": c.levels.overall.value
                        },
                        "scores": {
                            "reliability": c.scores.reliability,
                            "maintainability": c.scores.maintainability,
                            "availability": c.scores.availability,
                            "vulnerability": c.scores.vulnerability,
                            "overall": c.scores.overall
                        }
                    }
                    for c in result.quality.components
                ],
                "edges": [
                    {
                        "source": e.source,
                        "target": e.target,
                        "source_name": component_names.get(e.source, e.source),
                        "target_name": component_names.get(e.target, e.target),
                        "type": e.dependency_type,
                        "criticality_level": e.level.value,
                        "scores": {
                            "reliability": e.scores.reliability,
                            "maintainability": e.scores.maintainability,
                            "availability": e.scores.availability,
                            "vulnerability": e.scores.vulnerability,
                            "overall": e.scores.overall
                        }
                    }
                    for e in result.quality.edges
                ],
                "problems": [
                    {
                        "entity_id": p.entity_id,
                        "type": p.entity_type,
                        "category": p.category.value if hasattr(p.category, 'value') else str(p.category),
                        "severity": p.severity.value if hasattr(p.severity, 'value') else str(p.severity),
                        "name": p.name,
                        "description": p.description,
                        "recommendation": p.recommendation
                    }
                    for p in result.problems
                ]
            }
        }
    except Exception as e:
        logger.error(f"Layer analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ============================================================================
# Component & Edge Query Endpoints
# ============================================================================

@app.post("/api/v1/components", response_model=Dict[str, Any])
async def get_components(
    credentials: Neo4jCredentials,
    component_type: Optional[str] = Query(None, description="Filter by component type"),
    min_weight: Optional[float] = Query(None, description="Minimum weight threshold"),
    limit: int = Query(100, description="Maximum number of components to return")
):
    """
    Get components from the graph with optional filtering.
    """
    try:
        logger.info(f"Querying components: type={component_type}, min_weight={min_weight}")
        
        exporter = GraphExporter(credentials.uri, credentials.user, credentials.password)
        
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
        
        exporter.close()
        
        return {
            "success": True,
            "count": len(components),
            "components": components
        }
    except Exception as e:
        logger.error(f"Component query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/api/v1/components/critical", response_model=Dict[str, Any])
async def get_critical_components(
    credentials: Neo4jCredentials,
    limit: int = Query(20, description="Maximum number of components to return")
):
    """
    Get the most critical components based on analysis.
    """
    try:
        logger.info("Querying critical components")
        
        with GraphAnalyzer(credentials.uri, credentials.user, credentials.password) as analyzer:
            result = analyzer.analyze_layer("system")
        
        # Sort by overall score descending
        components = sorted(
            result.quality.components,
            key=lambda c: c.scores.overall,
            reverse=True
        )[:limit]
        
        return {
            "success": True,
            "count": len(components),
            "components": [
                {
                    "id": c.id,
                    "type": c.type,
                    "criticality_level": c.levels.overall.value,
                    "criticality_levels": {
                        "reliability": c.levels.reliability.value,
                        "maintainability": c.levels.maintainability.value,
                        "availability": c.levels.availability.value,
                        "vulnerability": c.levels.vulnerability.value,
                        "overall": c.levels.overall.value
                    },
                    "overall_score": c.scores.overall,
                    "scores": {
                        "reliability": c.scores.reliability,
                        "maintainability": c.scores.maintainability,
                        "availability": c.scores.availability,
                        "vulnerability": c.scores.vulnerability
                    }
                }
                for c in components
            ]
        }
    except Exception as e:
        logger.error(f"Critical components query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/api/v1/edges", response_model=Dict[str, Any])
async def get_edges(
    credentials: Neo4jCredentials,
    dependency_type: Optional[str] = Query(None, description="Filter by dependency type"),
    min_weight: Optional[float] = Query(None, description="Minimum weight threshold"),
    limit: int = Query(100, description="Maximum number of edges to return")
):
    """
    Get edges from the graph with optional filtering.
    
    Valid dependency types: app_to_app, node_to_node, app_to_broker, node_to_broker
    """
    try:
        logger.info(f"Querying edges: type={dependency_type}, min_weight={min_weight}")
        
        exporter = GraphExporter(credentials.uri, credentials.user, credentials.password)
        
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
        
        exporter.close()
        
        return {
            "success": True,
            "count": len(edges),
            "edges": edges
        }
    except Exception as e:
        logger.error(f"Edge query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/api/v1/edges/critical", response_model=Dict[str, Any])
async def get_critical_edges(
    credentials: Neo4jCredentials,
    limit: int = Query(20, description="Maximum number of edges to return")
):
    """
    Get the most critical edges based on analysis.
    """
    try:
        logger.info("Querying critical edges")
        
        with GraphAnalyzer(credentials.uri, credentials.user, credentials.password) as analyzer:
            result = analyzer.analyze_layer("system")
        
        # Sort by overall score descending
        edges = sorted(
            result.quality.edges,
            key=lambda e: e.scores.overall,
            reverse=True
        )[:limit]
        
        return {
            "success": True,
            "count": len(edges),
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "type": e.dependency_type,
                    "criticality_level": e.level.value,
                    "overall_score": e.scores.overall,
                    "scores": {
                        "reliability": e.scores.reliability,
                        "maintainability": e.scores.maintainability,
                        "availability": e.scores.availability,
                        "vulnerability": e.scores.vulnerability
                    }
                }
                for e in edges
            ]
        }
    except Exception as e:
        logger.error(f"Critical edges query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/api/v1/stats", response_model=Dict[str, Any])
async def get_graph_stats(credentials: Neo4jCredentials):
    """
    Get overall graph statistics including structural relationships.
    """
    try:
        logger.info("Getting graph statistics")
        
        exporter = GraphExporter(credentials.uri, credentials.user, credentials.password)
        graph_data = exporter.get_graph_data()
        
        # Count by type
        type_counts = {}
        for component in graph_data.components:
            comp_type = component.component_type
            type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
        
        # Count by dependency type (derived DEPENDS_ON edges)
        dep_counts = {}
        for edge in graph_data.edges:
            dep_type = edge.dependency_type
            dep_counts[dep_type] = dep_counts.get(dep_type, 0) + 1
        
        # Count structural relationships (RUNS_ON, PUBLISHES_TO, etc.)
        structural_counts = {}
        with exporter.driver.session() as session:
            rel_types = "|".join(STRUCTURAL_REL_TYPES)
            query = f"""
            MATCH ()-[r:{rel_types}]->()
            RETURN type(r) AS rel_type, count(r) AS count
            """
            result = session.run(query)
            for record in result:
                structural_counts[record["rel_type"]] = record["count"]
        
        exporter.close()
        
        return {
            "success": True,
            "stats": {
                "total_nodes": len(graph_data.components),
                "total_edges": len(graph_data.edges),
                "total_structural_edges": sum(structural_counts.values()),
                "node_counts": type_counts,
                "edge_counts": dep_counts,
                "structural_edge_counts": structural_counts
            }
        }
    except Exception as e:
        logger.error(f"Stats query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/api/v1/stats/degree-distribution", response_model=Dict[str, Any])
async def get_degree_distribution_stats(credentials: Neo4jCredentials):
    """
    Get fast degree distribution statistics.
    
    Computes:
    - In-degree, out-degree, and total degree statistics (mean, median, max, min, std)
    - Hub nodes (degree > mean + 2*std)
    - Isolated nodes count
    
    Optionally filter by node_type to analyze specific component types.
    Runs in O(V+E) time - very fast even for large graphs.
    """
    import time
    import statistics
    from typing import Dict, List
    
    try:
        start_time = time.time()
        filter_msg = f" (filtered by type: {credentials.node_type})" if credentials.node_type else ""
        logger.info(f"Computing degree distribution statistics{filter_msg}")
        
        exporter = GraphExporter(credentials.uri, credentials.user, credentials.password)
        graph_data = exporter.get_graph_data()
        
        # Calculate degree for ALL components first
        in_degree: Dict[str, int] = {}
        out_degree: Dict[str, int] = {}
        component_types: Dict[str, str] = {}
        component_names: Dict[str, str] = {}
        
        # Initialize ALL components with 0 degree
        for component in graph_data.components:
            comp_id = component.id
            in_degree[comp_id] = 0
            out_degree[comp_id] = 0
            component_types[comp_id] = component.component_type
            # Get name from properties or use id as fallback
            component_names[comp_id] = component.properties.get('name', comp_id)
        
        # Count degrees from ALL edges in the system
        for edge in graph_data.edges:
            source = edge.source_id
            target = edge.target_id
            out_degree[source] = out_degree.get(source, 0) + 1
            in_degree[target] = in_degree.get(target, 0) + 1
        
        # Calculate total degree for ALL components
        total_degree_all = {comp_id: in_degree[comp_id] + out_degree[comp_id] 
                           for comp_id in in_degree.keys()}
        
        # Calculate hub threshold based on ALL components (before filtering)
        all_degrees = list(total_degree_all.values())
        if all_degrees:
            all_mean = statistics.mean(all_degrees)
            all_std = statistics.stdev(all_degrees) if len(all_degrees) > 1 else 0
            hub_threshold = all_mean + 2 * all_std
        else:
            hub_threshold = 0
        
        # Now filter to only the requested node type for display
        if credentials.node_type:
            filtered_ids = {comp_id for comp_id, comp_type in component_types.items() 
                          if comp_type == credentials.node_type}
            in_degree = {k: v for k, v in in_degree.items() if k in filtered_ids}
            out_degree = {k: v for k, v in out_degree.items() if k in filtered_ids}
            component_types = {k: v for k, v in component_types.items() if k in filtered_ids}
            component_names = {k: v for k, v in component_names.items() if k in filtered_ids}
        
        # Calculate total degree for filtered components
        total_degree = {comp_id: in_degree[comp_id] + out_degree[comp_id] 
                       for comp_id in in_degree.keys()}
        
        # Helper function to compute stats
        def compute_stats(degree_dict: Dict[str, int]) -> Dict[str, float]:
            values = list(degree_dict.values())
            if not values:
                return {"mean": 0, "median": 0, "max": 0, "min": 0, "std": 0}
            
            return {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "max": max(values),
                "min": min(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0
            }
        
        in_stats = compute_stats(in_degree)
        out_stats = compute_stats(out_degree)
        total_stats = compute_stats(total_degree)
        
        # Identify hub nodes using the system-wide threshold (degree > mean + 2*std from ALL components)
        hub_nodes = []
        
        for comp_id, degree in total_degree.items():
            if degree > hub_threshold:
                hub_nodes.append({
                    "id": comp_id,
                    "name": component_names.get(comp_id, comp_id),
                    "degree": degree,
                    "type": component_types.get(comp_id, "Unknown")
                })
        
        # Sort hubs by degree descending
        hub_nodes.sort(key=lambda x: x["degree"], reverse=True)
        
        # Count isolated nodes - nodes with ZERO connections in the ENTIRE system
        # Use total_degree_all (before filtering) to identify truly isolated nodes
        isolated_count = 0
        for comp_id in total_degree.keys():
            # Check if this filtered node has zero connections in the entire system
            if total_degree_all.get(comp_id, 0) == 0:
                isolated_count += 1
        
        # Total node count in the filtered set
        total_nodes = len(total_degree)
        
        computation_time = (time.time() - start_time) * 1000  # Convert to ms
        
        exporter.close()
        
        return {
            "in_degree": in_stats,
            "out_degree": out_stats,
            "total_degree": total_stats,
            "hub_nodes": hub_nodes,
            "isolated_nodes": isolated_count,
            "total_nodes": total_nodes,
            "hub_threshold": round(hub_threshold, 2),
            "computation_time_ms": round(computation_time, 2)
        }
    except Exception as e:
        logger.error(f"Degree distribution computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@app.post("/api/v1/stats/connectivity-density", response_model=Dict[str, Any])
async def get_connectivity_density_stats(credentials: Neo4jCredentials):
    """
    Get connectivity density statistics - measures how interconnected the system is.
    
    Density = (Actual Edges) / (Maximum Possible Edges)
    For a directed graph: max_edges = N * (N - 1)
    
    Optionally filter by node_type to show specific component types,
    but density is always calculated on the entire graph first.
    Fast computation in O(1) time using existing stats.
    Provides insights into system coupling and complexity.
    """
    import time
    
    try:
        start_time = time.time()
        filter_msg = f" (filtered by type: {credentials.node_type})" if credentials.node_type else ""
        logger.info(f"Computing connectivity density statistics{filter_msg}")
        
        exporter = GraphExporter(credentials.uri, credentials.user, credentials.password)
        graph_data = exporter.get_graph_data()
        
        # ALWAYS calculate density on the ENTIRE graph first
        total_nodes = len(graph_data.components)
        total_edges = len(graph_data.edges)
        
        # For directed graph: max edges = n * (n - 1)
        max_possible_edges = total_nodes * (total_nodes - 1) if total_nodes > 1 else 0
        
        # Calculate density on entire graph
        density = total_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        # Calculate degree for ALL components using ALL edges
        component_degrees = {}
        component_info = {}
        component_types = {}
        
        # Initialize all components
        for component in graph_data.components:
            comp_id = component.id
            component_degrees[comp_id] = 0
            component_types[comp_id] = component.component_type
            component_info[comp_id] = {
                "name": component.properties.get('name', comp_id),
                "type": component.component_type
            }
        
        # Count total degree (in + out) for each component using ALL edges
        for edge in graph_data.edges:
            source = edge.source_id
            target = edge.target_id
            if source in component_degrees:
                component_degrees[source] += 1
            if target in component_degrees:
                component_degrees[target] += 1
        
        # Now filter components by type if specified (for display only)
        if credentials.node_type:
            filtered_component_ids = {comp_id for comp_id, comp_type in component_types.items() 
                                     if comp_type == credentials.node_type}
            # Filter degrees and info to only show filtered components
            component_degrees = {k: v for k, v in component_degrees.items() if k in filtered_component_ids}
            component_info = {k: v for k, v in component_info.items() if k in filtered_component_ids}
        
        # Get top 10 most connected components (from filtered set if filter applied)
        most_dense_components = []
        sorted_components = sorted(component_degrees.items(), key=lambda x: x[1], reverse=True)
        
        for comp_id, degree in sorted_components[:10]:
            if degree > 0:  # Only include components with connections
                info = component_info.get(comp_id, {})
                most_dense_components.append({
                    "id": comp_id,
                    "name": info.get("name", comp_id),
                    "type": info.get("type", "Unknown"),
                    "degree": degree,
                    "density_contribution": round((degree / (2 * total_edges)) * 100, 2) if total_edges > 0 else 0
                })
        
        # Interpretation (based on overall graph density)
        if density < 0.05:
            interpretation = "Sparse - Low coupling, good modularity"
            category = "sparse"
        elif density < 0.15:
            interpretation = "Moderate - Balanced connectivity"
            category = "moderate"
        elif density < 0.30:
            interpretation = "Dense - High coupling"
            category = "dense"
        else:
            interpretation = "Very Dense - Very high coupling"
            category = "very_dense"
        
        computation_time = (time.time() - start_time) * 1000  # Convert to ms
        
        exporter.close()
        
        return {
            "success": True,
            "stats": {
                "density": round(density, 6),
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "max_possible_edges": max_possible_edges,
                "interpretation": interpretation,
                "category": category,
                "most_dense_components": most_dense_components
            },
            "computation_time_ms": round(computation_time, 2)
        }
    except Exception as e:
        logger.error(f"Connectivity density computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@app.post("/api/v1/stats/clustering-coefficient", response_model=Dict[str, Any])
async def get_clustering_coefficient_stats(credentials: Neo4jCredentials):
    """
    Get clustering coefficient statistics - measures how nodes tend to cluster together.
    
    Clustering coefficient indicates:
    - How much nodes tend to form tightly connected groups
    - Local connectivity patterns and modularity
    - Probability that neighbors of a node are also connected
    
    Optionally filter by node_type to show specific component types,
    but clustering is always calculated on the entire graph first.
    Runs in O(V * d^2) where d is average degree - fast for sparse graphs.
    Provides valuable insights into system modularity and component grouping.
    """
    import time
    from collections import defaultdict
    
    try:
        start_time = time.time()
        filter_msg = f" (filtered by type: {credentials.node_type})" if credentials.node_type else ""
        logger.info(f"Computing clustering coefficient statistics{filter_msg}")
        
        exporter = GraphExporter(credentials.uri, credentials.user, credentials.password)
        graph_data = exporter.get_graph_data()
        
        # Build adjacency lists (treating graph as undirected for clustering)
        neighbors = defaultdict(set)
        component_info = {}
        component_types = {}
        
        # Initialize ALL components
        for component in graph_data.components:
            comp_id = component.id
            component_types[comp_id] = component.component_type
            component_info[comp_id] = {
                "name": component.properties.get('name', comp_id),
                "type": component.component_type
            }
        
        # Build neighbor sets (undirected edges) using ALL edges
        for edge in graph_data.edges:
            source = edge.source_id
            target = edge.target_id
            neighbors[source].add(target)
            neighbors[target].add(source)
        
        # Calculate clustering coefficient for ALL nodes
        local_coefficients = {}
        
        for node in component_info.keys():
            node_neighbors = neighbors.get(node, set())
            degree = len(node_neighbors)
            
            # Need at least 2 neighbors to form triangles
            if degree < 2:
                local_coefficients[node] = 0.0
                continue
            
            # Count triangles: connections between neighbors
            triangles = 0
            neighbor_list = list(node_neighbors)
            
            for i in range(len(neighbor_list)):
                for j in range(i + 1, len(neighbor_list)):
                    if neighbor_list[j] in neighbors[neighbor_list[i]]:
                        triangles += 1
            
            # Clustering coefficient = actual triangles / possible triangles
            # Possible triangles = degree * (degree - 1) / 2
            possible = degree * (degree - 1) / 2
            local_coefficients[node] = triangles / possible if possible > 0 else 0.0
        
        # Calculate global average clustering coefficient on ALL nodes
        if local_coefficients:
            avg_clustering = sum(local_coefficients.values()) / len(local_coefficients)
        else:
            avg_clustering = 0.0
        
        # Now filter components by type if specified (for display only)
        if credentials.node_type:
            filtered_component_ids = {comp_id for comp_id, comp_type in component_types.items() 
                                     if comp_type == credentials.node_type}
            # Filter coefficients and info to only show filtered components
            local_coefficients_filtered = {k: v for k, v in local_coefficients.items() if k in filtered_component_ids}
            component_info_filtered = {k: v for k, v in component_info.items() if k in filtered_component_ids}
        else:
            local_coefficients_filtered = local_coefficients
            component_info_filtered = component_info
        
        # Find highly clustered components (from filtered set if filter applied)
        highly_clustered = []
        sorted_nodes = sorted(
            [(node, coef) for node, coef in local_coefficients_filtered.items() if coef > 0],
            key=lambda x: (x[1], len(neighbors[x[0]])),  # Sort by coefficient, then by degree
            reverse=True
        )
        
        for node, coef in sorted_nodes[:10]:
            info = component_info.get(node, {})
            highly_clustered.append({
                "id": node,
                "name": info.get("name", node),
                "type": info.get("type", "Unknown"),
                "coefficient": round(coef, 4),
                "degree": len(neighbors[node]),
                "triangles": int(coef * len(neighbors[node]) * (len(neighbors[node]) - 1) / 2)
            })
        
        # Calculate distribution stats (on ALL nodes, not filtered)
        non_zero_coefficients = [c for c in local_coefficients.values() if c > 0]
        
        if non_zero_coefficients:
            import statistics
            max_coef = max(local_coefficients.values())
            min_coef = min([c for c in local_coefficients.values() if c > 0], default=0)
            median_coef = statistics.median(non_zero_coefficients)
            std_coef = statistics.stdev(non_zero_coefficients) if len(non_zero_coefficients) > 1 else 0
        else:
            max_coef = min_coef = median_coef = std_coef = 0.0
        
        # Count nodes by clustering level (on ALL nodes)
        zero_clustering = sum(1 for c in local_coefficients.values() if c == 0)
        low_clustering = sum(1 for c in local_coefficients.values() if 0 < c < 0.3)
        medium_clustering = sum(1 for c in local_coefficients.values() if 0.3 <= c < 0.7)
        high_clustering = sum(1 for c in local_coefficients.values() if c >= 0.7)
        
        # Interpretation (based on overall graph clustering)
        if avg_clustering < 0.1:
            interpretation = "Low clustering - components operate independently with few local groups"
            category = "low"
        elif avg_clustering < 0.3:
            interpretation = "Moderate clustering - some component grouping and local connectivity"
            category = "moderate"
        elif avg_clustering < 0.6:
            interpretation = "High clustering - strong component grouping with modular structure"
            category = "high"
        else:
            interpretation = "Very high clustering - tightly connected groups, highly modular"
            category = "very_high"
        
        computation_time = (time.time() - start_time) * 1000
        
        exporter.close()
        
        return {
            "success": True,
            "stats": {
                "avg_clustering_coefficient": round(avg_clustering, 6),
                "max_coefficient": round(max_coef, 6),
                "min_coefficient": round(min_coef, 6),
                "median_coefficient": round(median_coef, 6),
                "std_coefficient": round(std_coef, 6),
                "interpretation": interpretation,
                "category": category,
                "zero_clustering_count": zero_clustering,
                "low_clustering_count": low_clustering,
                "medium_clustering_count": medium_clustering,
                "high_clustering_count": high_clustering,
                "total_nodes": len(component_info),
                "highly_clustered_components": highly_clustered
            },
            "computation_time_ms": round(computation_time, 2)
        }
    except Exception as e:
        logger.error(f"Clustering coefficient computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@app.post("/api/v1/stats/dependency-depth", response_model=Dict[str, Any])
async def get_dependency_depth_stats(credentials: Neo4jCredentials):
    """
    Get dependency depth statistics - measures the depth of dependency chains.
    
    Dependency depth indicates:
    - How deep the transitive dependency chains are
    - Potential failure propagation paths
    - System architectural complexity
    - Testing and refactoring impact radius
    
    Runs in O(V + E) using BFS - very fast.
    Provides critical insights into system architecture and change impact.
    """
    import time
    from collections import deque, defaultdict
    
    try:
        start_time = time.time()
        logger.info("Computing dependency depth statistics")
        
        exporter = GraphExporter(credentials.uri, credentials.user, credentials.password)
        graph_data = exporter.get_graph_data()
        
        # Build directed adjacency lists and component info
        outgoing = defaultdict(list)  # node -> nodes it depends on
        incoming = defaultdict(list)  # node -> nodes that depend on it
        component_info = {}
        
        # Initialize components
        for component in graph_data.components:
            comp_id = component.id
            component_info[comp_id] = {
                "name": component.properties.get('name', comp_id),
                "type": component.component_type
            }
        
        # Build adjacency lists
        for edge in graph_data.edges:
            source = edge.source_id
            target = edge.target_id
            outgoing[source].append(target)
            incoming[target].append(source)
        
        # Calculate dependency depth for each node using BFS
        node_depths = {}
        
        for node in component_info.keys():
            # BFS to find maximum depth from this node
            max_depth = 0
            visited = {node}
            queue = deque([(node, 0)])
            
            while queue:
                current, depth = queue.popleft()
                max_depth = max(max_depth, depth)
                
                # Follow outgoing dependencies
                for neighbor in outgoing[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
            
            node_depths[node] = max_depth
        
        # Calculate stats
        if node_depths:
            import statistics
            depths = list(node_depths.values())
            avg_depth = statistics.mean(depths)
            max_depth = max(depths)
            min_depth = min(depths)
            median_depth = statistics.median(depths)
            std_depth = statistics.stdev(depths) if len(depths) > 1 else 0
        else:
            avg_depth = max_depth = min_depth = median_depth = std_depth = 0
        
        # Find deepest dependency chains (top 10)
        deepest_components = []
        sorted_nodes = sorted(
            node_depths.items(),
            key=lambda x: (x[1], len(outgoing[x[0]])),  # Sort by depth, then by out-degree
            reverse=True
        )
        
        for node, depth in sorted_nodes[:10]:
            info = component_info.get(node, {})
            deepest_components.append({
                "id": node,
                "name": info.get("name", node),
                "type": info.get("type", "Unknown"),
                "depth": depth,
                "dependencies": len(outgoing[node]),
                "dependents": len(incoming[node])
            })
        
        # Count nodes by depth level
        depth_distribution = defaultdict(int)
        for depth in depths:
            depth_distribution[depth] += 1
        
        # Categorize by depth level
        shallow = sum(1 for d in depths if d == 0)
        low_depth = sum(1 for d in depths if 0 < d <= 2)
        medium_depth = sum(1 for d in depths if 2 < d <= 5)
        high_depth = sum(1 for d in depths if d > 5)
        
        # Find root nodes (no incoming dependencies) and leaf nodes (no outgoing dependencies)
        root_nodes = []
        leaf_nodes = []
        isolated_nodes = []
        
        for node in component_info.keys():
            out_count = len(outgoing[node])
            in_count = len(incoming[node])
            info = component_info[node]
            
            # Isolated nodes (no connections at all)
            if out_count == 0 and in_count == 0:
                isolated_nodes.append({
                    "id": node,
                    "name": info.get("name", node),
                    "type": info.get("type", "Unknown"),
                    "dependencies": 0,
                    "dependents": 0
                })
            # Leaf nodes (no outgoing, has incoming - foundation components)
            elif out_count == 0 and in_count > 0:
                leaf_nodes.append({
                    "id": node,
                    "name": info.get("name", node),
                    "type": info.get("type", "Unknown"),
                    "dependents": in_count
                })
            # Root nodes (no incoming, has outgoing - top-level/entry point components)
            elif in_count == 0 and out_count > 0:
                root_nodes.append({
                    "id": node,
                    "name": info.get("name", node),
                    "type": info.get("type", "Unknown"),
                    "dependencies": out_count
                })
        
        # If no true root nodes found, show nodes with fewest incoming dependencies
        if len(root_nodes) == 0:
            candidates = []
            for node in component_info.keys():
                in_count = len(incoming[node])
                out_count = len(outgoing[node])
                if out_count > 0:  # Must have dependencies to be meaningful
                    info = component_info[node]
                    candidates.append({
                        "id": node,
                        "name": info.get("name", node),
                        "type": info.get("type", "Unknown"),
                        "dependencies": out_count,
                        "dependents": in_count
                    })
            # Sort by fewest dependents, then most dependencies
            root_nodes = sorted(candidates, key=lambda x: (x["dependents"], -x["dependencies"]))[:10]
        else:
            # Sort and limit to top 10 by most dependencies
            root_nodes = sorted(root_nodes, key=lambda x: x["dependencies"], reverse=True)[:10]
        
        # Sort and limit leaf nodes to top 10 by most dependents
        leaf_nodes = sorted(leaf_nodes, key=lambda x: x["dependents"], reverse=True)[:10]
        
        # Interpretation
        if max_depth == 0:
            interpretation = "No dependencies - completely isolated components"
            category = "isolated"
        elif max_depth <= 2:
            interpretation = "Shallow dependencies - simple, flat architecture with minimal cascading"
            category = "shallow"
        elif max_depth <= 5:
            interpretation = "Moderate depth - balanced architecture with some layering"
            category = "moderate"
        elif max_depth <= 10:
            interpretation = "Deep dependencies - complex architecture with significant layering"
            category = "deep"
        else:
            interpretation = "Very deep dependencies - highly complex with extensive cascading chains"
            category = "very_deep"
        
        computation_time = (time.time() - start_time) * 1000
        
        exporter.close()
        
        return {
            "success": True,
            "stats": {
                "avg_depth": round(avg_depth, 3),
                "max_depth": max_depth,
                "min_depth": min_depth,
                "median_depth": round(median_depth, 3),
                "std_depth": round(std_depth, 3),
                "interpretation": interpretation,
                "category": category,
                "shallow_count": shallow,
                "low_depth_count": low_depth,
                "medium_depth_count": medium_depth,
                "high_depth_count": high_depth,
                "total_nodes": len(component_info),
                "deepest_components": deepest_components,
                "root_nodes": root_nodes,
                "leaf_nodes": leaf_nodes,
                "depth_distribution": dict(depth_distribution)
            },
            "computation_time_ms": round(computation_time, 2)
        }
    except Exception as e:
        logger.error(f"Dependency depth computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@app.post("/api/v1/stats/component-isolation", response_model=Dict[str, Any])
async def get_component_isolation_stats(credentials: Neo4jCredentials):
    """
    Get component isolation statistics - identifies isolated, source, and sink components.
    
    Component isolation reveals:
    - Completely isolated components (no connections)
    - Source components (only outgoing dependencies - entry points)
    - Sink components (only incoming dependencies - foundations)
    - Bi-directional components (both in and out)
    
    Runs in O(V + E) - extremely fast.
    Provides critical architectural insights about component roles and dependencies.
    """
    import time
    from collections import defaultdict
    
    try:
        start_time = time.time()
        logger.info("Computing component isolation statistics")
        
        exporter = GraphExporter(credentials.uri, credentials.user, credentials.password)
        graph_data = exporter.get_graph_data()
        
        # Build adjacency info
        incoming = defaultdict(set)  # nodes that depend on this node
        outgoing = defaultdict(set)  # nodes this node depends on
        component_info = {}
        
        # Initialize components
        for component in graph_data.components:
            comp_id = component.id
            component_info[comp_id] = {
                "name": component.properties.get('name', comp_id),
                "type": component.component_type
            }
        
        # Build adjacency lists
        for edge in graph_data.edges:
            source = edge.source_id
            target = edge.target_id
            outgoing[source].add(target)
            incoming[target].add(source)
        
        # Categorize components
        isolated_components = []
        source_components = []
        sink_components = []
        bidirectional_components = []
        
        for comp_id in component_info.keys():
            in_count = len(incoming[comp_id])
            out_count = len(outgoing[comp_id])
            info = component_info[comp_id]
            
            comp_data = {
                "id": comp_id,
                "name": info.get("name", comp_id),
                "type": info.get("type", "Unknown"),
                "in_degree": in_count,
                "out_degree": out_count
            }
            
            if in_count == 0 and out_count == 0:
                # Completely isolated
                isolated_components.append(comp_data)
            elif in_count == 0 and out_count > 0:
                # Source: only depends on others (entry points)
                source_components.append(comp_data)
            elif in_count > 0 and out_count == 0:
                # Sink: only depended upon (foundations)
                sink_components.append(comp_data)
            else:
                # Bidirectional: both dependencies
                bidirectional_components.append(comp_data)
        
        # Sort by degree for top components
        source_components.sort(key=lambda x: x["out_degree"], reverse=True)
        sink_components.sort(key=lambda x: x["in_degree"], reverse=True)
        bidirectional_components.sort(key=lambda x: x["in_degree"] + x["out_degree"], reverse=True)
        
        # Calculate percentages
        total = len(component_info)
        isolated_pct = (len(isolated_components) / total * 100) if total > 0 else 0
        source_pct = (len(source_components) / total * 100) if total > 0 else 0
        sink_pct = (len(sink_components) / total * 100) if total > 0 else 0
        bidirectional_pct = (len(bidirectional_components) / total * 100) if total > 0 else 0
        
        # Interpretation
        if isolated_pct > 50:
            interpretation = "Highly fragmented - majority of components are isolated with no connections"
            category = "fragmented"
            health = "poor"
        elif source_pct > 50:
            interpretation = "Top-heavy architecture - many entry points with fewer shared foundations"
            category = "top_heavy"
            health = "moderate"
        elif sink_pct > 50:
            interpretation = "Foundation-heavy - many shared base components with fewer consumers"
            category = "foundation_heavy"
            health = "moderate"
        elif bidirectional_pct > 70:
            interpretation = "Well-connected architecture - most components have both dependencies and dependents"
            category = "connected"
            health = "good"
        elif isolated_pct > 20:
            interpretation = "Significant fragmentation - notable portion of components are isolated"
            category = "partially_fragmented"
            health = "fair"
        else:
            interpretation = "Balanced architecture - healthy mix of entry points, foundations, and connected components"
            category = "balanced"
            health = "good"
        
        computation_time = (time.time() - start_time) * 1000
        
        exporter.close()
        
        return {
            "success": True,
            "stats": {
                "total_components": total,
                "isolated_count": len(isolated_components),
                "source_count": len(source_components),
                "sink_count": len(sink_components),
                "bidirectional_count": len(bidirectional_components),
                "isolated_percentage": round(isolated_pct, 2),
                "source_percentage": round(source_pct, 2),
                "sink_percentage": round(sink_pct, 2),
                "bidirectional_percentage": round(bidirectional_pct, 2),
                "interpretation": interpretation,
                "category": category,
                "health": health,
                "isolated_components": isolated_components[:20],  # Top 20
                "top_sources": source_components[:10],
                "top_sinks": sink_components[:10],
                "top_bidirectional": bidirectional_components[:10]
            },
            "computation_time_ms": round(computation_time, 2)
        }
    except Exception as e:
        logger.error(f"Component isolation computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@app.post("/api/v1/stats/message-flow-patterns", response_model=Dict[str, Any])
async def get_message_flow_patterns(credentials: Neo4jCredentials):
    """
    Get message flow pattern statistics - analyzes communication patterns in pub-sub system.
    
    Message flow analysis reveals:
    - Hot topics (most publishers/subscribers)
    - Communication bottlenecks (overloaded brokers/topics)
    - Broker utilization patterns
    - Isolated applications (no message flow)
    - Publisher/Subscriber balance
    
    Runs in O(V + E) - extremely fast.
    Critical for understanding system communication patterns and identifying bottlenecks.
    """
    import time
    from collections import defaultdict
    
    try:
        start_time = time.time()
        logger.info("Computing message flow pattern statistics")
        
        exporter = GraphExporter(credentials.uri, credentials.user, credentials.password)
        
        # Query structural relationships for pub-sub analysis
        with exporter.driver.session() as session:
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
            
            # Execute queries
            publish_result = session.run(publish_query)
            publishes = [dict(record) for record in publish_result]
            
            subscribe_result = session.run(subscribe_query)
            subscribes = [dict(record) for record in subscribe_result]
            
            topic_broker_result = session.run(topic_broker_query)
            topic_brokers = [dict(record) for record in topic_broker_result]
            
            app_result = session.run(app_query)
            all_apps = [dict(record) for record in app_result]
        
        # Analyze topics
        topic_stats = defaultdict(lambda: {
            "publishers": set(),
            "subscribers": set(),
            "brokers": set(),
            "name": "",
            "pub_weight": 0,
            "sub_weight": 0
        })
        
        # Count publishers per topic
        for pub in publishes:
            topic_id = pub["topic_id"]
            topic_stats[topic_id]["publishers"].add(pub["app_id"])
            topic_stats[topic_id]["name"] = pub["topic_name"]
            topic_stats[topic_id]["pub_weight"] += pub["weight"]
        
        # Count subscribers per topic
        for sub in subscribes:
            topic_id = sub["topic_id"]
            topic_stats[topic_id]["subscribers"].add(sub["app_id"])
            topic_stats[topic_id]["name"] = sub["topic_name"]
            topic_stats[topic_id]["sub_weight"] += sub["weight"]
        
        # Map topics to brokers
        for tb in topic_brokers:
            topic_id = tb["topic_id"]
            topic_stats[topic_id]["brokers"].add(tb["broker_id"])
            if not topic_stats[topic_id]["name"]:
                topic_stats[topic_id]["name"] = tb["topic_name"]
        
        # Analyze brokers
        broker_stats = defaultdict(lambda: {
            "topics": set(),
            "total_publishers": 0,
            "total_subscribers": 0,
            "name": ""
        })
        
        for tb in topic_brokers:
            broker_id = tb["broker_id"]
            topic_id = tb["topic_id"]
            broker_stats[broker_id]["topics"].add(topic_id)
            broker_stats[broker_id]["name"] = tb["broker_name"]
            
            # Count publishers and subscribers for this broker's topics
            if topic_id in topic_stats:
                broker_stats[broker_id]["total_publishers"] += len(topic_stats[topic_id]["publishers"])
                broker_stats[broker_id]["total_subscribers"] += len(topic_stats[topic_id]["subscribers"])
        
        # Analyze applications
        app_pub_count = defaultdict(int)
        app_sub_count = defaultdict(int)
        app_names = {}
        
        for app in all_apps:
            app_names[app["id"]] = app["name"]
        
        for pub in publishes:
            app_pub_count[pub["app_id"]] += 1
        
        for sub in subscribes:
            app_sub_count[sub["app_id"]] += 1
        
        # Identify hot topics (top 10 by total activity)
        hot_topics = []
        for topic_id, stats in topic_stats.items():
            total_activity = len(stats["publishers"]) + len(stats["subscribers"])
            topic_name = stats["name"]
            if isinstance(topic_name, dict):
                topic_name = topic_name.get("name", topic_id)
            hot_topics.append({
                "id": str(topic_id),
                "name": str(topic_name),
                "publishers": len(stats["publishers"]),
                "subscribers": len(stats["subscribers"]),
                "total_activity": total_activity,
                "brokers": len(stats["brokers"]),
                "pub_weight": round(stats["pub_weight"], 2),
                "sub_weight": round(stats["sub_weight"], 2)
            })
        
        hot_topics.sort(key=lambda x: x["total_activity"], reverse=True)
        
        # Identify broker utilization (top 10)
        broker_utilization = []
        for broker_id, stats in broker_stats.items():
            total_load = stats["total_publishers"] + stats["total_subscribers"]
            broker_name = stats["name"]
            if isinstance(broker_name, dict):
                broker_name = broker_name.get("name", broker_id)
            broker_utilization.append({
                "id": str(broker_id),
                "name": str(broker_name),
                "topics": len(stats["topics"]),
                "publishers": stats["total_publishers"],
                "subscribers": stats["total_subscribers"],
                "total_load": total_load
            })
        
        broker_utilization.sort(key=lambda x: x["total_load"], reverse=True)
        
        # Identify isolated applications (no pub/sub)
        isolated_apps = []
        for app in all_apps:
            app_id = app["id"]
            if app_pub_count[app_id] == 0 and app_sub_count[app_id] == 0:
                # Ensure name is a string
                app_name = app.get("name", app_id)
                if isinstance(app_name, dict):
                    app_name = app_name.get("name", app_id)
                isolated_apps.append({
                    "id": str(app_id),
                    "name": str(app_name)
                })
        
        # Identify top publishers and subscribers
        top_publishers = []
        for app_id, count in app_pub_count.items():
            if count > 0:
                app_name = app_names.get(app_id, app_id)
                if isinstance(app_name, dict):
                    app_name = app_name.get("name", app_id)
                top_publishers.append({
                    "id": str(app_id),
                    "name": str(app_name),
                    "topics": count,
                    "subscriptions": app_sub_count[app_id]
                })
        top_publishers.sort(key=lambda x: x["topics"], reverse=True)
        
        top_subscribers = []
        for app_id, count in app_sub_count.items():
            if count > 0:
                app_name = app_names.get(app_id, app_id)
                if isinstance(app_name, dict):
                    app_name = app_name.get("name", app_id)
                top_subscribers.append({
                    "id": str(app_id),
                    "name": str(app_name),
                    "topics": count,
                    "publications": app_pub_count[app_id]
                })
        top_subscribers.sort(key=lambda x: x["topics"], reverse=True)
        
        # Calculate overall statistics
        total_topics = len(topic_stats)
        total_brokers = len(broker_stats)
        total_apps = len(all_apps)
        active_apps = len(set(app_pub_count.keys()) | set(app_sub_count.keys()))
        isolated_count = len(isolated_apps)
        
        avg_publishers_per_topic = sum(len(s["publishers"]) for s in topic_stats.values()) / total_topics if total_topics > 0 else 0
        avg_subscribers_per_topic = sum(len(s["subscribers"]) for s in topic_stats.values()) / total_topics if total_topics > 0 else 0
        avg_topics_per_broker = sum(len(s["topics"]) for s in broker_stats.values()) / total_brokers if total_brokers > 0 else 0
        
        # Interpretation
        if isolated_count / total_apps > 0.3 if total_apps > 0 else False:
            interpretation = "High isolation - many applications not participating in message flow"
            category = "isolated"
            health = "poor"
        elif avg_publishers_per_topic < 1.5 and avg_subscribers_per_topic < 1.5:
            interpretation = "Sparse communication - low message flow activity across topics"
            category = "sparse"
            health = "moderate"
        elif len(hot_topics) > 0 and hot_topics[0]["total_activity"] > total_apps * 0.3:
            interpretation = "Bottleneck detected - some topics have very high activity concentration"
            category = "bottleneck"
            health = "fair"
        else:
            interpretation = "Balanced communication - healthy message flow patterns across the system"
            category = "balanced"
            health = "good"
        
        computation_time = (time.time() - start_time) * 1000
        
        exporter.close()
        
        return {
            "success": True,
            "stats": {
                "total_topics": total_topics,
                "total_brokers": total_brokers,
                "total_applications": total_apps,
                "active_applications": active_apps,
                "isolated_applications": isolated_count,
                "avg_publishers_per_topic": round(avg_publishers_per_topic, 2),
                "avg_subscribers_per_topic": round(avg_subscribers_per_topic, 2),
                "avg_topics_per_broker": round(avg_topics_per_broker, 2),
                "interpretation": interpretation,
                "category": category,
                "health": health,
                "hot_topics": hot_topics[:10],
                "broker_utilization": broker_utilization[:10],
                "isolated_applications": isolated_apps[:20],
                "top_publishers": top_publishers[:10],
                "top_subscribers": top_subscribers[:10]
            },
            "computation_time_ms": round(computation_time, 2)
        }
    except Exception as e:
        logger.error(f"Message flow pattern computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@app.post("/api/v1/stats/component-redundancy", response_model=Dict[str, Any])
async def get_component_redundancy_stats(credentials: Neo4jCredentials):
    """
    Get component redundancy statistics - identifies critical single points of failure.
    
    Component redundancy analysis reveals:
    - Single Points of Failure (SPOFs) - components whose failure disconnects the graph
    - Bridge components - components that connect different parts of the system
    - Redundant paths - availability of alternative routes between components
    - Resilience score - overall system redundancy and fault tolerance
    
    Runs in O(V + E) - extremely fast.
    Critical for understanding system resilience and identifying architectural risks.
    """
    import time
    from collections import defaultdict, deque
    
    try:
        start_time = time.time()
        logger.info("Computing component redundancy statistics")
        
        exporter = GraphExporter(credentials.uri, credentials.user, credentials.password)
        graph_data = exporter.get_graph_data()
        
        # Build adjacency lists
        adjacency = defaultdict(set)
        reverse_adjacency = defaultdict(set)
        component_info = {}
        
        # Initialize components
        for component in graph_data.components:
            comp_id = component.id
            component_info[comp_id] = {
                "name": component.properties.get('name', comp_id),
                "type": component.component_type
            }
        
        # Build adjacency structures
        for edge in graph_data.edges:
            source = edge.source_id
            target = edge.target_id
            adjacency[source].add(target)
            reverse_adjacency[target].add(source)
        
        # Calculate articulation points (SPOFs) using DFS
        def find_articulation_points():
            visited = set()
            disc = {}
            low = {}
            parent = {}
            ap = set()
            time_counter = [0]
            
            def dfs(u):
                children = 0
                visited.add(u)
                disc[u] = low[u] = time_counter[0]
                time_counter[0] += 1
                
                for v in adjacency[u]:
                    if v not in visited:
                        children += 1
                        parent[v] = u
                        dfs(v)
                        low[u] = min(low[u], low[v])
                        
                        # Check if u is an articulation point
                        if parent.get(u) is None and children > 1:
                            ap.add(u)
                        if parent.get(u) is not None and low[v] >= disc[u]:
                            ap.add(u)
                    elif v != parent.get(u):
                        low[u] = min(low[u], disc[v])
            
            for node in component_info.keys():
                if node not in visited:
                    dfs(node)
            
            return ap
        
        articulation_points = find_articulation_points()
        
        # Identify bridge components (high betweenness without full calculation)
        # A bridge component is one that many paths go through
        def identify_bridge_components():
            bridge_scores = defaultdict(int)
            
            # Sample paths and count how often each component appears
            sample_nodes = list(component_info.keys())[:min(50, len(component_info))]
            
            for start in sample_nodes:
                # BFS to find paths
                visited = set([start])
                queue = deque([(start, [start])])
                paths_through = defaultdict(int)
                
                while queue:
                    node, path = queue.popleft()
                    
                    if len(path) > 10:  # Limit path length
                        continue
                    
                    for neighbor in adjacency[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            new_path = path + [neighbor]
                            queue.append((neighbor, new_path))
                            
                            # Count components in path
                            for comp in new_path[1:-1]:  # Exclude start and end
                                paths_through[comp] += 1
                
                # Add to bridge scores
                for comp, count in paths_through.items():
                    bridge_scores[comp] += count
            
            # Get top bridge components
            sorted_bridges = sorted(bridge_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_bridges[:20]
        
        bridge_components = identify_bridge_components()
        
        # Calculate redundancy metrics
        total_components = len(component_info)
        spof_count = len(articulation_points)
        spof_percentage = (spof_count / total_components * 100) if total_components > 0 else 0
        
        # Identify components with multiple paths (redundant)
        redundant_count = 0
        for comp_id in component_info.keys():
            # A component is redundant if it has multiple incoming or outgoing connections
            if len(adjacency[comp_id]) > 1 or len(reverse_adjacency[comp_id]) > 1:
                redundant_count += 1
        
        redundancy_percentage = (redundant_count / total_components * 100) if total_components > 0 else 0
        
        # Calculate resilience score (0-100)
        # Higher is better: more redundancy, fewer SPOFs
        resilience_score = max(0, min(100, redundancy_percentage - (spof_percentage * 2)))
        
        # Prepare SPOF details
        spof_details = []
        for spof_id in list(articulation_points)[:20]:
            info = component_info[spof_id]
            spof_details.append({
                "id": spof_id,
                "name": info.get("name", spof_id),
                "type": info.get("type", "Unknown"),
                "in_degree": len(reverse_adjacency[spof_id]),
                "out_degree": len(adjacency[spof_id]),
                "is_critical": True
            })
        
        # Sort by total degree
        spof_details.sort(key=lambda x: x["in_degree"] + x["out_degree"], reverse=True)
        
        # Prepare bridge component details
        bridge_details = []
        for comp_id, bridge_score in bridge_components:
            if comp_id in component_info:
                info = component_info[comp_id]
                bridge_details.append({
                    "id": comp_id,
                    "name": info.get("name", comp_id),
                    "type": info.get("type", "Unknown"),
                    "bridge_score": bridge_score,
                    "in_degree": len(reverse_adjacency[comp_id]),
                    "out_degree": len(adjacency[comp_id])
                })
        
        # Interpretation
        if resilience_score >= 70:
            interpretation = "Highly resilient architecture - good redundancy with few single points of failure"
            category = "resilient"
            health = "good"
        elif resilience_score >= 50:
            interpretation = "Moderately resilient - some redundancy exists but critical components need attention"
            category = "moderate"
            health = "fair"
        elif resilience_score >= 30:
            interpretation = "Limited resilience - significant single points of failure present"
            category = "limited"
            health = "moderate"
        else:
            interpretation = "Low resilience - architecture is fragile with many critical dependencies"
            category = "fragile"
            health = "poor"
        
        computation_time = (time.time() - start_time) * 1000
        
        exporter.close()
        
        return {
            "success": True,
            "stats": {
                "total_components": total_components,
                "spof_count": spof_count,
                "spof_percentage": round(spof_percentage, 2),
                "redundant_count": redundant_count,
                "redundancy_percentage": round(redundancy_percentage, 2),
                "resilience_score": round(resilience_score, 2),
                "interpretation": interpretation,
                "category": category,
                "health": health,
                "single_points_of_failure": spof_details,
                "bridge_components": bridge_details
            },
            "computation_time_ms": round(computation_time, 2)
        }
    except Exception as e:
        logger.error(f"Component redundancy computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@app.post("/api/v1/stats/node-weight-distribution", response_model=Dict[str, Any])
async def get_node_weight_distribution_stats(credentials: Neo4jCredentials):
    """
    Get node weight distribution statistics - analyzes how component importance is distributed.
    
    Node weight distribution reveals:
    - Distribution of component weights (importance scores)
    - High-value vs low-value components
    - Weight concentration patterns
    - Critical component identification by weight
    
    Runs in O(V) - extremely fast.
    Provides insights into component importance hierarchy and architectural focus areas.
    """
    import time
    import statistics
    
    try:
        start_time = time.time()
        logger.info("Computing node weight distribution statistics")
        
        exporter = GraphExporter(credentials.uri, credentials.user, credentials.password)
        graph_data = exporter.get_graph_data()
        
        # Collect weights by component type
        weights_by_type = {}
        all_weights = []
        component_details = []
        
        for component in graph_data.components:
            comp_id = component.id
            comp_type = component.component_type
            weight = component.weight
            name = component.properties.get('name', comp_id)
            
            all_weights.append(weight)
            
            if comp_type not in weights_by_type:
                weights_by_type[comp_type] = []
            weights_by_type[comp_type].append(weight)
            
            component_details.append({
                "id": comp_id,
                "name": name,
                "type": comp_type,
                "weight": round(weight, 4)
            })
        
        # Sort components by weight
        component_details.sort(key=lambda x: x["weight"], reverse=True)
        
        # Overall statistics
        if all_weights:
            total_weight = sum(all_weights)
            avg_weight = statistics.mean(all_weights)
            median_weight = statistics.median(all_weights)
            std_weight = statistics.stdev(all_weights) if len(all_weights) > 1 else 0
            min_weight = min(all_weights)
            max_weight = max(all_weights)
        else:
            total_weight = avg_weight = median_weight = std_weight = min_weight = max_weight = 0
        
        # Statistics by type
        type_stats = {}
        for comp_type, weights in weights_by_type.items():
            if weights:
                type_stats[comp_type] = {
                    "count": len(weights),
                    "total_weight": round(sum(weights), 4),
                    "avg_weight": round(statistics.mean(weights), 4),
                    "median_weight": round(statistics.median(weights), 4),
                    "min_weight": round(min(weights), 4),
                    "max_weight": round(max(weights), 4),
                    "std_weight": round(statistics.stdev(weights), 4) if len(weights) > 1 else 0
                }
        
        # Weight distribution categories
        very_high_threshold = avg_weight + 2 * std_weight if std_weight > 0 else avg_weight * 2
        high_threshold = avg_weight + std_weight if std_weight > 0 else avg_weight * 1.5
        low_threshold = avg_weight - std_weight if std_weight > 0 else avg_weight * 0.5
        very_low_threshold = avg_weight - 2 * std_weight if std_weight > 0 else avg_weight * 0.25
        
        very_high_count = sum(1 for w in all_weights if w >= very_high_threshold)
        high_count = sum(1 for w in all_weights if high_threshold <= w < very_high_threshold)
        medium_count = sum(1 for w in all_weights if low_threshold <= w < high_threshold)
        low_count = sum(1 for w in all_weights if very_low_threshold <= w < low_threshold)
        very_low_count = sum(1 for w in all_weights if w < very_low_threshold)
        
        # Top weighted components (top 15)
        top_components = component_details[:15]
        
        # Calculate weight concentration (what % of total weight is in top 20% of components)
        sorted_weights = sorted(all_weights, reverse=True)
        top_20_percent_count = max(1, int(len(sorted_weights) * 0.2))
        top_20_weight = sum(sorted_weights[:top_20_percent_count])
        weight_concentration = (top_20_weight / total_weight * 100) if total_weight > 0 else 0
        
        # Interpretation
        if weight_concentration > 80:
            interpretation = "Highly concentrated - small number of critical components hold most importance"
            category = "concentrated"
            health = "moderate"
        elif weight_concentration > 60:
            interpretation = "Moderately concentrated - some components significantly more important than others"
            category = "moderate"
            health = "fair"
        elif weight_concentration > 40:
            interpretation = "Balanced distribution - importance spread across many components"
            category = "balanced"
            health = "good"
        else:
            interpretation = "Evenly distributed - most components have similar importance levels"
            category = "even"
            health = "good"
        
        computation_time = (time.time() - start_time) * 1000
        
        exporter.close()
        
        return {
            "success": True,
            "stats": {
                "total_components": len(graph_data.components),
                "total_weight": round(total_weight, 4),
                "avg_weight": round(avg_weight, 4),
                "median_weight": round(median_weight, 4),
                "min_weight": round(min_weight, 4),
                "max_weight": round(max_weight, 4),
                "std_weight": round(std_weight, 4),
                "weight_concentration": round(weight_concentration, 2),
                "interpretation": interpretation,
                "category": category,
                "health": health,
                "very_high_count": very_high_count,
                "high_count": high_count,
                "medium_count": medium_count,
                "low_count": low_count,
                "very_low_count": very_low_count,
                "top_components": top_components,
                "type_stats": type_stats
            },
            "computation_time_ms": round(computation_time, 2)
        }
    except Exception as e:
        logger.error(f"Node weight distribution computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@app.post("/api/v1/stats/edge-weight-distribution", response_model=Dict[str, Any])
async def get_edge_weight_distribution_stats(credentials: Neo4jCredentials):
    """
    Get edge weight distribution statistics - analyzes how dependency importance is distributed.
    
    Edge weight distribution reveals:
    - Distribution of dependency weights (connection strength)
    - Critical vs weak dependencies
    - Weight concentration patterns
    - Dependency type importance patterns
    
    Runs in O(E) - extremely fast.
    Provides insights into dependency criticality and architectural coupling patterns.
    """
    import time
    import statistics
    
    try:
        start_time = time.time()
        logger.info("Computing edge weight distribution statistics")
        
        exporter = GraphExporter(credentials.uri, credentials.user, credentials.password)
        graph_data = exporter.get_graph_data()
        
        # Collect weights by dependency type
        weights_by_type = {}
        all_weights = []
        edge_details = []
        
        # Build component names map for better display
        component_names = {}
        for component in graph_data.components:
            component_names[component.id] = component.properties.get('name', component.id)
        
        for edge in graph_data.edges:
            source_id = edge.source_id
            target_id = edge.target_id
            dep_type = edge.dependency_type
            weight = edge.weight
            
            all_weights.append(weight)
            
            if dep_type not in weights_by_type:
                weights_by_type[dep_type] = []
            weights_by_type[dep_type].append(weight)
            
            edge_details.append({
                "source": source_id,
                "target": target_id,
                "source_name": component_names.get(source_id, source_id),
                "target_name": component_names.get(target_id, target_id),
                "type": dep_type,
                "weight": round(weight, 4)
            })
        
        # Sort edges by weight
        edge_details.sort(key=lambda x: x["weight"], reverse=True)
        
        # Overall statistics
        if all_weights:
            total_weight = sum(all_weights)
            avg_weight = statistics.mean(all_weights)
            median_weight = statistics.median(all_weights)
            std_weight = statistics.stdev(all_weights) if len(all_weights) > 1 else 0
            min_weight = min(all_weights)
            max_weight = max(all_weights)
        else:
            total_weight = avg_weight = median_weight = std_weight = min_weight = max_weight = 0
        
        # Statistics by dependency type
        type_stats = {}
        for dep_type, weights in weights_by_type.items():
            if weights:
                type_stats[dep_type] = {
                    "count": len(weights),
                    "total_weight": round(sum(weights), 4),
                    "avg_weight": round(statistics.mean(weights), 4),
                    "median_weight": round(statistics.median(weights), 4),
                    "min_weight": round(min(weights), 4),
                    "max_weight": round(max(weights), 4),
                    "std_weight": round(statistics.stdev(weights), 4) if len(weights) > 1 else 0
                }
        
        # Weight distribution categories
        very_high_threshold = avg_weight + 2 * std_weight if std_weight > 0 else avg_weight * 2
        high_threshold = avg_weight + std_weight if std_weight > 0 else avg_weight * 1.5
        low_threshold = avg_weight - std_weight if std_weight > 0 else avg_weight * 0.5
        very_low_threshold = avg_weight - 2 * std_weight if std_weight > 0 else avg_weight * 0.25
        
        very_high_count = sum(1 for w in all_weights if w >= very_high_threshold)
        high_count = sum(1 for w in all_weights if high_threshold <= w < very_high_threshold)
        medium_count = sum(1 for w in all_weights if low_threshold <= w < high_threshold)
        low_count = sum(1 for w in all_weights if very_low_threshold <= w < low_threshold)
        very_low_count = sum(1 for w in all_weights if w < very_low_threshold)
        
        # Top weighted edges (top 15)
        top_edges = edge_details[:15]
        
        # Calculate weight concentration (what % of total weight is in top 20% of edges)
        sorted_weights = sorted(all_weights, reverse=True)
        top_20_percent_count = max(1, int(len(sorted_weights) * 0.2))
        top_20_weight = sum(sorted_weights[:top_20_percent_count])
        weight_concentration = (top_20_weight / total_weight * 100) if total_weight > 0 else 0
        
        # Interpretation
        if weight_concentration > 80:
            interpretation = "Highly concentrated - few critical dependencies carry most weight"
            category = "concentrated"
            health = "moderate"
        elif weight_concentration > 60:
            interpretation = "Moderately concentrated - some dependencies significantly stronger than others"
            category = "moderate"
            health = "fair"
        elif weight_concentration > 40:
            interpretation = "Balanced distribution - dependency importance spread across many edges"
            category = "balanced"
            health = "good"
        else:
            interpretation = "Evenly distributed - most dependencies have similar importance levels"
            category = "even"
            health = "good"
        
        computation_time = (time.time() - start_time) * 1000
        
        exporter.close()
        
        return {
            "success": True,
            "stats": {
                "total_edges": len(graph_data.edges),
                "total_weight": round(total_weight, 4),
                "avg_weight": round(avg_weight, 4),
                "median_weight": round(median_weight, 4),
                "min_weight": round(min_weight, 4),
                "max_weight": round(max_weight, 4),
                "std_weight": round(std_weight, 4),
                "weight_concentration": round(weight_concentration, 2),
                "interpretation": interpretation,
                "category": category,
                "health": health,
                "very_high_count": very_high_count,
                "high_count": high_count,
                "medium_count": medium_count,
                "low_count": low_count,
                "very_low_count": very_low_count,
                "top_edges": top_edges,
                "type_stats": type_stats
            },
            "computation_time_ms": round(computation_time, 2)
        }
    except Exception as e:
        logger.error(f"Edge weight distribution computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


# ============================================================================
# Simulation Endpoints
# ============================================================================

@app.post("/api/v1/simulation/event", response_model=Dict[str, Any])
async def simulate_event(request: EventSimulationRequest):
    """
    Run event simulation from a source application.
    
    Simulates message flow through the pub-sub system, measuring:
    - Throughput (messages published, delivered, dropped)
    - Latency metrics (avg, min, max, p50, p99)
    - Path analysis (topics, brokers, subscribers)
    - Component impacts
    """
    try:
        logger.info(f"Running event simulation: source={request.source_app}, messages={request.num_messages}")
        
        creds = request.credentials
        with Simulator(uri=creds.uri, user=creds.user, password=creds.password) as sim:
            result = sim.run_event_simulation(
                source_app=request.source_app,
                num_messages=request.num_messages,
                duration=request.duration
            )
        
        return {
            "success": True,
            "simulation_type": "event",
            "result": result.to_dict()
        }
    except Exception as e:
        logger.error(f"Event simulation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Event simulation failed: {str(e)}")


@app.post("/api/v1/simulation/failure", response_model=Dict[str, Any])
async def simulate_failure(request: FailureSimulationRequest):
    """
    Run failure simulation for a target component.
    
    Simulates component failure and analyzes:
    - Composite impact score
    - Reachability loss (connectivity degradation)
    - Fragmentation (component isolation)
    - Throughput loss (capacity degradation)
    - Cascade propagation (dependent failures)
    - Per-layer impacts
    
    Valid layers: app, infra, mw-app, mw-infra, system
    """
    valid_layers = ["app", "infra", "mw-app", "mw-infra", "system"]
    if request.layer not in valid_layers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid layer. Must be one of: {', '.join(valid_layers)}"
        )
    
    try:
        logger.info(f"Running failure simulation: target={request.target_id}, layer={request.layer}")
        
        creds = request.credentials
        with Simulator(uri=creds.uri, user=creds.user, password=creds.password) as sim:
            result = sim.run_failure_simulation(
                target_id=request.target_id,
                layer=request.layer,
                cascade_probability=request.cascade_probability
            )
        
        return {
            "success": True,
            "simulation_type": "failure",
            "result": result.to_dict()
        }
    except Exception as e:
        logger.error(f"Failure simulation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failure simulation failed: {str(e)}")


@app.post("/api/v1/simulation/exhaustive", response_model=Dict[str, Any])
async def simulate_exhaustive(request: ExhaustiveSimulationRequest):
    """
    Run exhaustive failure analysis for all components in a layer.
    
    Analyzes failure impact for every component, sorted by impact score.
    Useful for identifying the most critical components in the system.
    
    Valid layers: app, infra, mw-app, mw-infra, system
    
    Warning: This can take significant time for large graphs.
    """
    valid_layers = ["app", "infra", "mw-app", "mw-infra", "system"]
    if request.layer not in valid_layers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid layer. Must be one of: {', '.join(valid_layers)}"
        )
    
    try:
        logger.info(f"Running exhaustive failure analysis: layer={request.layer}")
        
        creds = request.credentials
        with Simulator(uri=creds.uri, user=creds.user, password=creds.password) as sim:
            results = sim.run_failure_simulation_exhaustive(
                layer=request.layer,
                cascade_probability=request.cascade_probability
            )
        
        # Convert results to dict
        results_dict = [r.to_dict() for r in results]
        
        # Calculate summary statistics
        if results_dict:
            impacts = [r["impact"]["composite_impact"] for r in results_dict]
            summary = {
                "total_components": len(results_dict),
                "avg_impact": sum(impacts) / len(impacts),
                "max_impact": max(impacts),
                "min_impact": min(impacts),
                "critical_count": sum(1 for i in impacts if i > 0.5),
                "high_count": sum(1 for i in impacts if 0.3 < i <= 0.5),
                "medium_count": sum(1 for i in impacts if 0.1 < i <= 0.3),
                "low_count": sum(1 for i in impacts if i <= 0.1)
            }
        else:
            summary = {
                "total_components": 0,
                "avg_impact": 0,
                "max_impact": 0,
                "min_impact": 0,
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0
            }
        
        return {
            "success": True,
            "simulation_type": "exhaustive",
            "layer": request.layer,
            "summary": summary,
            "results": results_dict
        }
    except Exception as e:
        logger.error(f"Exhaustive simulation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Exhaustive simulation failed: {str(e)}")


@app.post("/api/v1/simulation/report", response_model=Dict[str, Any])
async def generate_simulation_report(request: ReportRequest):
    """
    Generate comprehensive simulation report with analysis across multiple layers.
    
    Includes:
    - Graph summary statistics
    - Per-layer event and failure simulation metrics
    - Criticality classification (critical, high, medium, low, minimal)
    - SPOF (Single Point of Failure) detection
    - Top critical components
    - System health recommendations
    
    Valid layers: app, infra, mw-app, mw-infra, system
    """
    valid_layers = ["app", "infra", "mw-app", "mw-infra", "system"]
    for layer in request.layers:
        if layer not in valid_layers:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid layer '{layer}'. Must be one of: {', '.join(valid_layers)}"
            )
    
    try:
        logger.info(f"Generating simulation report: layers={request.layers}")
        
        creds = request.credentials
        with Simulator(uri=creds.uri, user=creds.user, password=creds.password) as sim:
            report = sim.generate_report(layers=request.layers)
        
        # Transform top_critical to match frontend expectations (nested structure)
        report_dict = report.to_dict()
        if "top_critical" in report_dict:
            report_dict["top_critical"] = [
                {
                    "id": comp["id"],
                    "type": comp["type"],
                    "level": comp["level"],
                    "scores": {
                        "event_impact": 0.0,
                        "failure_impact": 0.0,
                        "combined_impact": comp.get("combined_impact", 0.0),
                    },
                    "metrics": {
                        "cascade_count": comp.get("cascade_count", 0),
                        "message_throughput": 0,
                        "reachability_loss_percent": 0.0,
                    },
                }
                for comp in report_dict["top_critical"]
            ]
        
        return {
            "success": True,
            "report": report_dict
        }
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


# ============================================================================
# Classification Endpoint
# ============================================================================

@app.post("/api/v1/classify", response_model=Dict[str, Any])
async def classify_components(
    credentials: Neo4jCredentials,
    metrics: Optional[List[str]] = Query(None, description="Metrics to use for classification"),
    use_fuzzy: bool = Query(False, description="Use fuzzy logic classifier"),
    use_weights: bool = Query(True, description="Consider edge weights"),
    dependency_types: Optional[List[str]] = Query(None, description="Dependency types to analyze")
):
    """
    Classify components using BoxPlot or Fuzzy logic based on centrality metrics.
    
    Available metrics: betweenness, pagerank, degree
    """
    try:
        logger.info(f"Running classification: metrics={metrics}, fuzzy={use_fuzzy}, weights={use_weights}")
        
        with GraphAnalyzer(credentials.uri, credentials.user, credentials.password) as analyzer:
            # Get graph data first
            client = analyzer._get_client()
            graph_data = client.get_graph_data()
            
            # Filter by dependency types if specified
            if dependency_types:
                filtered_edges = [e for e in graph_data.edges if e.dependency_type in dependency_types]
                # Create filtered GraphData
                from src.core.graph_exporter import GraphData
                graph_data = GraphData(components=graph_data.components, edges=filtered_edges)
            
            # Run structural analysis
            structural = analyzer.structural.analyze(graph_data)
            
            # Prepare metrics for classification
            metrics_to_use = metrics or ['betweenness', 'pagerank', 'degree']
            metric_data = {}
            
            for metric_name in metrics_to_use:
                values = {}
                for comp_id, comp_metrics in structural.components.items():
                    if metric_name == 'betweenness':
                        values[comp_id] = comp_metrics.betweenness
                    elif metric_name == 'pagerank':
                        values[comp_id] = comp_metrics.pagerank
                    elif metric_name == 'degree':
                        values[comp_id] = comp_metrics.degree
                
                if values:
                    metric_data[metric_name] = values
            
            # Run classification for each metric
            classifications = {}
            for metric_name, values in metric_data.items():
                if use_fuzzy:
                    # Use fuzzy classifier (if implemented)
                    # For now, fall back to boxplot
                    classifier = BoxPlotClassifier()
                else:
                    classifier = BoxPlotClassifier()
                
                result = classifier.classify_scores(values, item_type="component", metric_name=metric_name)
                
                classifications[metric_name] = {
                    "statistics": {
                        "min_val": result.stats.min_val,
                        "max_val": result.stats.max_val,
                        "median": result.stats.median,
                        "q1": result.stats.q1,
                        "q3": result.stats.q3,
                        "iqr": result.stats.iqr,
                        "upper_fence": result.stats.upper_fence
                    },
                    "distribution": result.summary(),
                    "components": [
                        {
                            "id": item.id,
                            "level": item.level.value,
                            "score": item.score
                        }
                        for item in result.items
                    ]
                }
            
            # Create merged ranking combining all metrics
            component_scores = {}
            for metric_name, classification in classifications.items():
                for comp in classification["components"]:
                    if comp["id"] not in component_scores:
                        component_scores[comp["id"]] = {
                            "scores": {},
                            "levels": []
                        }
                    component_scores[comp["id"]]["scores"][metric_name] = comp["score"]
                    component_scores[comp["id"]]["levels"].append(comp["level"])
            
            # Calculate merged scores and determine dominant level
            merged_ranking = []
            level_priority = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            
            for comp_id, data in component_scores.items():
                merged_score = sum(data["scores"].values()) / len(data["scores"])
                
                # Dominant level is the highest priority level across all metrics
                dominant = max(data["levels"], key=lambda l: level_priority.get(l, 0))
                
                merged_ranking.append({
                    "id": comp_id,
                    "merged_score": merged_score,
                    "dominant_level": dominant,
                    "scores_by_metric": data["scores"]
                })
            
            # Sort by merged score descending
            merged_ranking.sort(key=lambda x: x["merged_score"], reverse=True)
            
            return {
                "success": True,
                "classifications": classifications,
                "merged_ranking": merged_ranking,
                "metadata": {
                    "metrics_used": metrics_to_use,
                    "use_fuzzy": use_fuzzy,
                    "use_weights": use_weights,
                    "dependency_types": dependency_types
                }
            }
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


# ============================================================================
# Validation Endpoints
# ============================================================================

class ValidationRequest(GraphRequestWithCredentials):
    layers: List[str] = Field(default=["app", "infra", "system"], description="Layers to validate")
    include_comparisons: bool = Field(default=True, description="Include detailed component comparisons")


class QuickValidationRequest(GraphRequestWithCredentials):
    predicted_file: Optional[str] = Field(None, description="Path to predicted scores JSON file")
    actual_file: Optional[str] = Field(None, description="Path to actual scores JSON file")
    predicted_data: Optional[Dict[str, float]] = Field(None, description="Predicted scores dictionary")
    actual_data: Optional[Dict[str, float]] = Field(None, description="Actual scores dictionary")


@app.post("/api/v1/validation/run-pipeline", response_model=Dict[str, Any])
async def run_validation_pipeline(request: ValidationRequest):
    """
    Run the full validation pipeline.
    
    This endpoint orchestrates:
    1. Graph analysis to get predicted criticality scores
    2. Failure simulation to get actual impact scores
    3. Statistical validation comparing predictions vs reality
    
    Args:
        request: Validation configuration with credentials and layers
        
    Returns:
        Complete validation results with metrics for each layer
    """
    try:
        from src.validation import ValidationPipeline, ValidationTargets
        
        logger.info(f"Starting validation pipeline for layers: {request.layers}")
        
        # Initialize pipeline
        pipeline = ValidationPipeline(
            uri=request.credentials.uri,
            user=request.credentials.user,
            password=request.credentials.password,
            targets=ValidationTargets()
        )
        
        # Run validation
        result = pipeline.run(
            layers=request.layers,
            include_comparisons=request.include_comparisons
        )
        
        return {
            "success": True,
            "result": result.to_dict()
        }
        
    except ImportError as e:
        logger.error(f"Validation module import failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Validation module not available: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Validation pipeline failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {str(e)}"
        )


@app.post("/api/v1/validation/quick", response_model=Dict[str, Any])
async def quick_validation(request: QuickValidationRequest):
    """
    Quick validation from provided or file-based data.
    
    Compare predicted scores against actual scores using
    statistical validation metrics without running the full pipeline.
    
    Args:
        request: Predicted and actual scores (as files or data)
        
    Returns:
        Validation metrics and results
    """
    try:
        from src.validation import QuickValidator, ValidationTargets
        import json
        
        logger.info("Starting quick validation")
        
        # Load data
        predicted_scores = {}
        actual_scores = {}
        
        if request.predicted_data:
            predicted_scores = request.predicted_data
        elif request.predicted_file:
            with open(request.predicted_file, 'r') as f:
                data = json.load(f)
                predicted_scores = data if isinstance(data, dict) else {}
        
        if request.actual_data:
            actual_scores = request.actual_data
        elif request.actual_file:
            with open(request.actual_file, 'r') as f:
                data = json.load(f)
                actual_scores = data if isinstance(data, dict) else {}
        
        if not predicted_scores or not actual_scores:
            raise HTTPException(
                status_code=400,
                detail="Must provide either files or data for both predicted and actual scores"
            )
        
        # Run quick validation
        validator = QuickValidator(targets=ValidationTargets())
        result = validator.validate(
            predicted_scores=predicted_scores,
            actual_scores=actual_scores
        )
        
        return {
            "success": True,
            "result": result.to_dict()
        }
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Quick validation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {str(e)}"
        )


@app.get("/api/v1/validation/layers", response_model=Dict[str, Any])
async def get_validation_layers():
    """
    Get available validation layers and their definitions.
    
    Returns:
        Dictionary of layer definitions with descriptions
    """
    try:
        from src.validation import LAYER_DEFINITIONS
        
        return {
            "success": True,
            "layers": LAYER_DEFINITIONS
        }
        
    except ImportError as e:
        logger.error(f"Validation module import failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Validation module not available: {str(e)}"
        )


@app.get("/api/v1/validation/targets", response_model=Dict[str, Any])
async def get_validation_targets():
    """
    Get default validation targets (success criteria).
    
    Returns:
        Dictionary of validation metrics and their target thresholds
    """
    try:
        from src.validation import ValidationTargets
        
        targets = ValidationTargets()
        
        return {
            "success": True,
            "targets": targets.to_dict()
        }
        
    except ImportError as e:
        logger.error(f"Validation module import failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Validation module not available: {str(e)}"
        )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
