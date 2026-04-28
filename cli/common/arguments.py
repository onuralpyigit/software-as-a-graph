import argparse
import logging
import os


def add_neo4j_arguments(parser: argparse.ArgumentParser):
    """Add standard Neo4j connection arguments to a parser."""
    group = parser.add_argument_group("Neo4j connection")
    group.add_argument(
        "--uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j Bolt URI (env: NEO4J_URI)",
    )
    group.add_argument(
        "--user", "-u",
        default=os.getenv("NEO4J_USER", "neo4j"),
        help="Neo4j username (env: NEO4J_USER)",
    )
    group.add_argument(
        "--password", "-p",
        default=os.getenv("NEO4J_PASSWORD", "password"),
        help="Neo4j password (env: NEO4J_PASSWORD)",
    )
    return group


def add_runtime_arguments(parser: argparse.ArgumentParser):
    """Add common runtime arguments (verbose/quiet) to a parser."""
    group = parser.add_argument_group("Common options")
    group.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging")
    group.add_argument("--quiet", "-q", action="store_true", help="Suppress non-essential console output")
    return group


def add_common_arguments(parser: argparse.ArgumentParser):
    """Add standard runtime, layer, and output arguments (verbose/quiet/layer/output)."""
    group = add_runtime_arguments(parser)
    group.add_argument(
        "--layer", "-l",
        default="system",
        help="Analysis layer(s). Can be comma-separated (e.g. 'app,infra,system'). Defaults to 'system'.",
    )
    group.add_argument("--output", "-o", metavar="FILE", help="Export results to file")
    return group


def setup_logging(args: argparse.Namespace) -> None:
    """Configure Python logging based on --verbose / --quiet flags."""
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
