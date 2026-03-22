"""
Infrastructure Adapters (Deprecated: use src.infrastructure)
"""

from src.infrastructure import Neo4jRepository, create_repository, config

__all__ = [
    "Neo4jRepository",
    "create_repository",
    "config",
]
