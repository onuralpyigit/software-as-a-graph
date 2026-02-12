"""
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


# Default Neo4j connection settings
import os

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
