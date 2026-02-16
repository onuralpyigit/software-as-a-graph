"""
FastAPI dependency injection for API routes.

Provides:
  - Default Neo4j connection settings from environment
  - ``get_repository`` dependency: creates a request-scoped repository
    from the ``Neo4jCredentials`` body and ensures it is closed after use
"""

import os
import logging
from typing import AsyncGenerator

from fastapi import Depends
from src.core import create_repository
from src.core.interfaces import IGraphRepository
from api.models import Neo4jCredentials

# ── Configuration ────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

DEFAULT_NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_NEO4J_USER = os.environ.get("NEO4J_USERNAME", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")


# ── Dependencies ─────────────────────────────────────────────────────────

async def get_repository(credentials: Neo4jCredentials) -> AsyncGenerator[IGraphRepository, None]:
    """
    Request-scoped repository dependency.

    Creates a repository from the credentials provided in the request body,
    yields it for the endpoint to use, and closes it automatically when the
    request completes — even if the endpoint raises an exception.

    Usage in an endpoint::

        @router.post("/example")
        async def example(repo: IGraphRepository = Depends(get_repository)):
            data = repo.get_graph_data()
            return {"nodes": data.components}
    """
    repo = create_repository(
        uri=credentials.uri,
        user=credentials.user,
        password=credentials.password,
    )
    try:
        yield repo
    finally:
        repo.close()
