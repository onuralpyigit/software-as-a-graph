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
# Core imports removed - now using services

# Application Services
from src.application.services.generation_service import GenerationService
from src.application.services.import_service import ImportService
from src.application.services.statistics_service import StatisticsService
from src.application.services.analysis_service import AnalysisService
from src.application.services.graph_service import GraphService
from src.application.services.simulation_service import SimulationService
from src.application.services.classification_service import ClassificationService
from src.application.services.validation_service import ValidationService
from src.application.container import Container

# Analysis imports
from src.analysis.analyzer import GraphAnalyzer
from src.domain.services.classifier import BoxPlotClassifier


# Simulation imports
# Removed direct simulator imports

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
# Helper functions have been moved to src.infrastructure.repositories.graph_query_repo
# ============================================================================


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
        service = GraphService(credentials.uri, credentials.user, credentials.password)
        service.check_connection()
        
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
        
        container = Container(uri=credentials.uri, user=credentials.user, password=credentials.password)
        service = container.analysis_service()
        result = service.analyze_layer("system")
        
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
        logger.error(f"Full analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/v1/analysis/type/{component_type}", response_model=Dict[str, Any])
async def analyze_by_type(component_type: str, credentials: Neo4jCredentials):
    """
    Run analysis filtered by component type.
    Accepts: node, app, broker, Application, Node, Broker
    """
    # Normalize component type (handle variations from frontend)
    type_mapping = {
        "application": "Application",
        "app": "Application",
        "node": "Node",
        "broker": "Broker",
    }
    
    normalized_type = type_mapping.get(component_type.lower())
    if not normalized_type:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid component type: {component_type}. Valid types: node, app, broker, Application, Node, Broker"
        )
        
    try:
        logger.info(f"Analyzing component type: {component_type} (normalized to {normalized_type})")
        
        container = Container(uri=credentials.uri, user=credentials.user, password=credentials.password)
        service = container.analysis_service()
        result = service.analyze_layer("system")
        
        # Filter components by type
        filtered_components = [c for c in result.quality.components if c.type == normalized_type]
        
        # Create a map of component IDs to names from structural data
        component_names = {c.id: c.structural.name if c.structural and hasattr(c.structural, 'name') else c.id 
                          for c in filtered_components}
        
        # Filter edges to only include those between filtered components
        filtered_component_ids = {c.id for c in filtered_components}
        filtered_edges = [e for e in result.quality.edges 
                         if e.source in filtered_component_ids or e.target in filtered_component_ids]
        
        return {
            "success": True,
            "layer": result.layer,
            "component_type": normalized_type,
            "analysis": {
                "context": f"{normalized_type} Components Analysis",
                "description": f"Analysis filtered by component type: {normalized_type}",
                "summary": {
                    "total_components": len(filtered_components),
                    "critical_count": sum(1 for c in filtered_components if c.levels.overall.value == "critical"),
                    "high_count": sum(1 for c in filtered_components if c.levels.overall.value == "high"),
                    "total_problems": sum(1 for p in result.problems if p.entity_id in filtered_component_ids),
                    "critical_problems": sum(1 for p in result.problems 
                                           if p.entity_id in filtered_component_ids and 
                                           (p.severity == "CRITICAL" or (hasattr(p.severity, 'value') and p.severity.value == "CRITICAL"))),
                    "components": {
                        level: sum(1 for c in filtered_components if c.levels.overall.value == level)
                        for level in ["critical", "high", "medium", "low", "minimal"]
                    },
                    "edges": {
                        level: sum(1 for e in filtered_edges if e.level.value == level)
                        for level in ["critical", "high", "medium", "low", "minimal"]
                    }
                },
                "stats": {
                    "nodes": len(filtered_components),
                    "edges": len(filtered_edges),
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
                    for e in filtered_edges
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
        
        container = Container(uri=credentials.uri, user=credentials.user, password=credentials.password)
        service = container.analysis_service()
        result = service.analyze_layer(layer)
        
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
        
        service = GraphService(credentials.uri, credentials.user, credentials.password)
        result = service.get_components(component_type, min_weight, limit)
        
        return {
            "success": True,
            "count": result["count"],
            "components": result["components"]
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
        
        container = Container(uri=credentials.uri, user=credentials.user, password=credentials.password)
        service = container.analysis_service()
        result = service.analyze_layer("system")
        
        # Sort components by overall score and take top N
        components = sorted(
            result.quality.components,
            key=lambda c: c.scores.overall,
            reverse=True
        )[:limit]
        
        # Format components for response
        formatted_components = [
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
                "scores": {
                    "reliability": c.scores.reliability,
                    "maintainability": c.scores.maintainability,
                    "availability": c.scores.availability,
                    "vulnerability": c.scores.vulnerability,
                    "overall": c.scores.overall
                }
            }
            for c in components
        ]
        
        return {
            "success": True,
            "count": len(formatted_components),
            "components": formatted_components
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
        
        service = GraphService(credentials.uri, credentials.user, credentials.password)
        result = service.get_edges(dependency_type, min_weight, limit)
        
        return {
            "success": True,
            "count": result["count"],
            "edges": result["edges"]
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
        
        container = Container(uri=credentials.uri, user=credentials.user, password=credentials.password)
        service = container.analysis_service()
        result = service.analyze_layer("system")
        
        # Sort edges by overall score and take top N
        edges = sorted(
            result.quality.edges,
            key=lambda e: e.scores.overall,
            reverse=True
        )[:limit]
        
        # Format edges for response
        formatted_edges = [
            {
                "source": e.source,
                "target": e.target,
                "criticality_level": e.level.value,
                "scores": {
                    "reliability": e.scores.reliability,
                    "maintainability": e.scores.maintainability,
                    "availability": e.scores.availability,
                    "vulnerability": e.scores.vulnerability,
                    "overall": e.scores.overall
                }
            }
            for e in edges
        ]
        
        return {
            "success": True,
            "count": len(formatted_edges),
            "edges": formatted_edges
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
        
        service = StatisticsService(credentials.uri, credentials.user, credentials.password)
        stats = service.get_graph_stats()
        
        return {
            "success": True,
            "stats": stats
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
    try:
        filter_msg = f" (filtered by type: {credentials.node_type})" if credentials.node_type else ""
        logger.info(f"Computing degree distribution statistics{filter_msg}")
        
        service = StatisticsService(credentials.uri, credentials.user, credentials.password)
        stats = service.get_degree_distribution(node_type=credentials.node_type)
        
        return stats
    except Exception as e:
        logger.error(f"Degree distribution computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@app.post("/api/v1/stats/connectivity-density", response_model=Dict[str, Any])
async def get_connectivity_density_stats(credentials: Neo4jCredentials):
    """
    Get connectivity density statistics - measures how interconnected the system is.
    """
    try:
        filter_msg = f" (filtered by type: {credentials.node_type})" if credentials.node_type else ""
        logger.info(f"Computing connectivity density statistics{filter_msg}")
        
        service = StatisticsService(credentials.uri, credentials.user, credentials.password)
        stats = service.get_connectivity_density(node_type=credentials.node_type)
        
        return {
            "success": True,
            "stats": stats,
            "computation_time_ms": stats.get("computation_time_ms", 0)
        }
    except Exception as e:
        logger.error(f"Connectivity density computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@app.post("/api/v1/stats/clustering-coefficient", response_model=Dict[str, Any])
async def get_clustering_coefficient_stats(credentials: Neo4jCredentials):
    """
    Get clustering coefficient statistics - measures how nodes tend to cluster together.
    """
    try:
        filter_msg = f" (filtered by type: {credentials.node_type})" if credentials.node_type else ""
        logger.info(f"Computing clustering coefficient statistics{filter_msg}")
        
        service = StatisticsService(credentials.uri, credentials.user, credentials.password)
        stats = service.get_clustering_coefficient(node_type=credentials.node_type)
        
        return {
            "success": True,
            "stats": stats,
            "computation_time_ms": stats.get("computation_time_ms", 0)
        }
    except Exception as e:
        logger.error(f"Clustering coefficient computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@app.post("/api/v1/stats/dependency-depth", response_model=Dict[str, Any])
async def get_dependency_depth_stats(credentials: Neo4jCredentials):
    """
    Get dependency depth statistics - measures the depth of dependency chains.
    """
    try:
        logger.info("Computing dependency depth statistics")
        
        service = StatisticsService(credentials.uri, credentials.user, credentials.password)
        stats = service.get_dependency_depth()
        
        return {
            "success": True,
            "stats": stats,
            "computation_time_ms": stats.get("computation_time_ms", 0)
        }
    except Exception as e:
        logger.error(f"Dependency depth computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@app.post("/api/v1/stats/component-isolation", response_model=Dict[str, Any])
async def get_component_isolation_stats(credentials: Neo4jCredentials):
    """
    Get component isolation statistics - identifies isolated, source, and sink components.
    """
    try:
        logger.info("Computing component isolation statistics")
        
        service = StatisticsService(credentials.uri, credentials.user, credentials.password)
        stats = service.get_component_isolation()
        
        return {
            "success": True,
            "stats": stats,
            "computation_time_ms": stats.get("computation_time_ms", 0)
        }
    except Exception as e:
        logger.error(f"Component isolation computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@app.post("/api/v1/stats/message-flow-patterns", response_model=Dict[str, Any])
async def get_message_flow_patterns(credentials: Neo4jCredentials):
    """
    Get message flow pattern statistics - analyzes communication patterns in pub-sub system.
    """
    try:
        logger.info("Computing message flow pattern statistics")
        
        service = StatisticsService(credentials.uri, credentials.user, credentials.password)
        stats = service.get_message_flow_patterns()
        
        return {
            "success": True,
            "stats": stats,
            "computation_time_ms": stats.get("computation_time_ms", 0)
        }
    except Exception as e:
        logger.error(f"Message flow pattern computation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@app.post("/api/v1/stats/component-redundancy", response_model=Dict[str, Any])
async def get_component_redundancy_stats(credentials: Neo4jCredentials):
    """
    Get component redundancy statistics - identifies SPOFs and bridge components.
    """
    import time
    try:
        logger.info("Computing component redundancy statistics")
        
        service = StatisticsService(credentials.uri, credentials.user, credentials.password)
        result = service.get_component_redundancy()
        
        return {
            "success": result.get("success", True),
            "stats": result.get("stats", {}),
            "computation_time_ms": result.get("computation_time_ms", 0)
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
        
        service = StatisticsService(credentials.uri, credentials.user, credentials.password)
        stats = service.get_node_weight_distribution()
        
        return {
            "success": True,
            "stats": stats
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
        
        service = StatisticsService(credentials.uri, credentials.user, credentials.password)
        stats = service.get_edge_weight_distribution()
        
        return {
            "success": True,
            "stats": stats
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
        container = Container(uri=creds.uri, user=creds.user, password=creds.password)
        service = container.simulation_service()
        
        result = service.run_event_simulation(
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
        container = Container(uri=creds.uri, user=creds.user, password=creds.password)
        service = container.simulation_service()
        
        result = service.run_failure_simulation(
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
        container = Container(uri=creds.uri, user=creds.user, password=creds.password)
        service = container.simulation_service()
        
        results = service.run_failure_simulation_exhaustive(
            layer=request.layer,
            cascade_probability=request.cascade_probability
        )
        
        # Create summary from results
        summary = {
            "total_components": len(results),
            "avg_impact": sum(r.impact.composite_impact for r in results) / len(results) if results else 0,
            "max_impact": max((r.impact.composite_impact for r in results), default=0),
            "critical_count": sum(1 for r in results if r.impact.composite_impact > 0.7),
            "high_count": sum(1 for r in results if 0.4 < r.impact.composite_impact <= 0.7),
            "medium_count": sum(1 for r in results if 0.2 < r.impact.composite_impact <= 0.4),
            "spof_count": sum(1 for r in results if r.impact.fragmentation > 0.01),
        }
        
        return {
            "success": True,
            "simulation_type": "exhaustive",
            "layer": request.layer,
            "summary": summary,
            "results": [r.to_dict() for r in results]
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
    
    Valid layers: app, infra, mw, system (or legacy: application, infrastructure, complete)
    """
    # Map legacy layer names to canonical names
    layer_aliases = {
        "application": "app",
        "infrastructure": "infra",
        "app_broker": "mw",
        "complete": "system",
    }
    
    valid_layers = ["app", "infra", "mw", "system"]
    mapped_layers = []
    
    for layer in request.layers:
        # Map legacy name to canonical name
        canonical_layer = layer_aliases.get(layer, layer)
        
        if canonical_layer not in valid_layers:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid layer '{layer}'. Must be one of: {', '.join(valid_layers + list(layer_aliases.keys()))}"
            )
        mapped_layers.append(canonical_layer)
    
    try:
        logger.info(f"Generating simulation report: layers={mapped_layers}")
        
        creds = request.credentials
        container = Container(uri=creds.uri, user=creds.user, password=creds.password)
        service = container.simulation_service()
        
        report = service.generate_report(layers=mapped_layers)
        
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
        from src.domain.models.validation.metrics import ValidationTargets
        
        logger.info(f"Starting validation pipeline for layers: {request.layers}")
        
        # Create container with credentials
        container = Container(
            uri=request.credentials.uri,
            user=request.credentials.user,
            password=request.credentials.password
        )
        
        try:
            # Get validation service
            validation_service = container.validation_service(targets=ValidationTargets())
            
            # Run validation
            result = validation_service.validate_layers(layers=request.layers)
            
            # Transform the result to match frontend expectations
            result_dict = result.to_dict()
            
            # Restructure response for frontend compatibility
            transformed_result = {
                "timestamp": result_dict["timestamp"],
                "summary": {
                    "total_components": result_dict["total_components"],
                    "layers_validated": len(result_dict["layers"]),
                    "layers_passed": result_dict["layers_passed"],
                    "all_passed": result_dict["all_passed"],
                },
                "layers": result_dict["layers"],
                "cross_layer_insights": result_dict.get("warnings", []),
                "targets": result_dict["targets"],
            }
            
            return {
                "success": True,
                "result": transformed_result
            }
        finally:
            container.close()
        
    except ImportError as e:
        logger.error(f"Validation module import failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Validation module not available: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Validation pipeline failed: {str(e)}")
        logger.exception("Full traceback:")
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
        from src.domain.models.validation.metrics import ValidationTargets
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
        
        # Create container with credentials (for potential graph access)
        container = Container(
            uri=request.credentials.uri,
            user=request.credentials.user,
            password=request.credentials.password
        )
        
        try:
            # Get validation service
            validation_service = container.validation_service(targets=ValidationTargets())
            
            # Run quick validation
            result = validation_service.validate_from_data(
                predicted=predicted_scores,
                actual=actual_scores
            )
            
            return {
                "success": True,
                "result": result.to_dict()
            }
        finally:
            container.close()
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Quick validation failed: {str(e)}")
        logger.exception("Full traceback:")
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
