"""
bin/_shared.py - Common CLI utilities
"""
import os
import logging
import argparse

def add_neo4j_args(parser: argparse.ArgumentParser):
    """Add standard Neo4j connection arguments."""
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

def add_common_args(parser: argparse.ArgumentParser):
    """Add standard runtime and output arguments."""
    group = parser.add_argument_group("Common options")
    group.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")
    group.add_argument("--quiet", "-q", action="store_true", help="Suppress non-essential console output")
    
    # Layer argument (present in almost all scripts)
    group.add_argument(
        "--layer", "-l",
        choices=["app", "infra", "mw", "system"],
        default="system",
        help="Analysis layer (default: system)",
    )
    
    # Common Output argument
    group.add_argument("--output", "-o", metavar="FILE", help="Export results to file")
    return group

def setup_logging(args: argparse.Namespace) -> None:
    """Configure python logging based on runtime args."""
    log_level = (
        logging.DEBUG if getattr(args, "verbose", False)
        else logging.WARNING if getattr(args, "quiet", False)
        else logging.INFO
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
