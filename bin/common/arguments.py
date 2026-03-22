import argparse
import sys
import os
from typing import Optional
from src.core import AnalysisLayer

def add_neo4j_arguments(parser: argparse.ArgumentParser):
    """Add standard Neo4j connection arguments to a parser."""
    group = parser.add_argument_group("Neo4j connection")
    group.add_argument(
        "--uri", 
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"), 
        help="Neo4j Bolt URI (env: NEO4J_URI)"
    )
    group.add_argument(
        "--user", "-u", 
        default=os.getenv("NEO4J_USER", "neo4j"), 
        help="Neo4j username (env: NEO4J_USER)"
    )
    group.add_argument(
        "--password", "-p", 
        default=os.getenv("NEO4J_PASSWORD", "password"), 
        help="Neo4j password (env: NEO4J_PASSWORD)"
    )
    return group

def add_common_arguments(parser: argparse.ArgumentParser):
    """Add common runtime and output arguments to a parser."""
    group = parser.add_argument_group("Common options")
    group.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")
    group.add_argument("--quiet", "-q", action="store_true", help="Suppress non-essential console output")
    return group

def add_layer_argument(parser: argparse.ArgumentParser, default: str = "system"):
    """Add standard layer selection argument to a parser."""
    from src.core import AnalysisLayer
    parser.add_argument(
        "--layer", "-l",
        choices=[la.value for la in AnalysisLayer],
        default=default,
        help=f"Analysis layer (default: {default})",
    )
