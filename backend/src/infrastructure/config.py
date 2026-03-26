import os
import re

# Neo4j Environment Variable Defaults
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = "neo4j"


def resolve_neo4j_uri(client_uri: str) -> str:
    """Rewrite localhost/127.0.0.1 URIs to the Docker-internal Neo4j hostname.

    When the browser sends bolt://localhost:7687, this is unreachable from
    inside the API container. If NEO4J_URI is set (Docker), replace the
    host portion so the connection goes to the correct container.
    """
    env_uri = os.environ.get("NEO4J_URI")
    if not env_uri:
        return client_uri  # Not running in Docker — use as-is

    # Match bolt:// or neo4j:// with localhost or 127.0.0.1
    if re.match(r"^(bolt|neo4j)(\+s?s?)?://(localhost|127\.0\.0\.1)(:\d+)?$", client_uri):
        return env_uri
    return client_uri


def get_default_uri() -> str:
    return NEO4J_URI


def get_default_username() -> str:
    return NEO4J_USERNAME


def get_default_password() -> str:
    return NEO4J_PASSWORD


def get_default_database() -> str:
    return NEO4J_DATABASE
