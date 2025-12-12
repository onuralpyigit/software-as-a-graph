#!/usr/bin/env python3
"""
Neo4j Utilities

Common utilities for working with Neo4j graph database including:
- Connection testing and diagnostics
- Database information and statistics
- Query templates and execution
- Performance analysis
- Data export utilities

This module does not require APOC and works with standard Neo4j installations.

Usage:
    from src.utils.neo4j_utils import Neo4jUtilities

    # Basic usage
    utils = Neo4jUtilities(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )

    if utils.test_connection():
        info = utils.get_database_info()
        print(info)

    utils.close()

    # Context manager usage (recommended)
    with Neo4jUtilities(uri="bolt://localhost:7687", user="neo4j", password="pass") as utils:
        info = utils.get_database_info()
        utils.export_to_json("output.json")
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import logging

# Gracefully handle missing neo4j driver
try:
    from neo4j import GraphDatabase, basic_auth
    from neo4j.exceptions import ServiceUnavailable, ClientError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None
    basic_auth = None
    ServiceUnavailable = Exception
    ClientError = Exception


class Neo4jUtilities:
    """
    Utility functions for Neo4j operations.

    Provides connection management, database info, querying, and export
    capabilities without requiring APOC plugins.
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        Initialize connection to Neo4j.

        Args:
            uri: Neo4j bolt URI (e.g., bolt://localhost:7687)
            user: Username for authentication
            password: Password for authentication
            database: Database name (default: neo4j)

        Raises:
            ImportError: If neo4j driver is not installed
            ServiceUnavailable: If connection fails
        """
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "neo4j driver not installed. Install with: pip install neo4j"
            )

        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.logger = logging.getLogger(__name__)

        self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed"""
        self.close()
        return False

    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
            self.driver = None

    def test_connection(self) -> bool:
        """
        Test database connection.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False

    def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information without requiring APOC.

        Returns:
            Dictionary containing node counts, relationship counts, and totals
        """
        with self.driver.session(database=self.database) as session:
            # Get node counts by label using standard Cypher
            node_counts = {}
            labels_result = session.run("CALL db.labels() YIELD label RETURN label")
            labels = [record['label'] for record in labels_result]

            for label in labels:
                count_result = session.run(
                    f"MATCH (n:`{label}`) RETURN count(n) as count"
                )
                node_counts[label] = count_result.single()['count']

            # Get relationship counts by type
            rel_counts = {}
            rel_types_result = session.run(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
            )
            rel_types = [record['relationshipType'] for record in rel_types_result]

            for rel_type in rel_types:
                count_result = session.run(
                    f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) as count"
                )
                rel_counts[rel_type] = count_result.single()['count']

            return {
                'database': self.database,
                'nodes_by_label': node_counts,
                'relationships_by_type': rel_counts,
                'total_nodes': sum(node_counts.values()),
                'total_relationships': sum(rel_counts.values())
            }

    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get schema information (constraints and indexes).

        Returns:
            Dictionary containing constraints and indexes
        """
        with self.driver.session(database=self.database) as session:
            # Get constraints
            constraints = []
            try:
                constraints_result = session.run("SHOW CONSTRAINTS")
                for record in constraints_result:
                    constraints.append(dict(record))
            except ClientError:
                # Older Neo4j versions may not support SHOW CONSTRAINTS
                self.logger.warning("Could not retrieve constraints (older Neo4j version?)")

            # Get indexes
            indexes = []
            try:
                indexes_result = session.run("SHOW INDEXES")
                for record in indexes_result:
                    indexes.append(dict(record))
            except ClientError:
                self.logger.warning("Could not retrieve indexes (older Neo4j version?)")

            return {
                'constraints': constraints,
                'indexes': indexes
            }

    def export_to_json(self, output_file: str, include_derived: bool = True):
        """
        Export entire graph to JSON.

        Args:
            output_file: Path to output JSON file
            include_derived: Include derived DEPENDS_ON relationships
        """
        self.logger.info(f"Exporting graph to {output_file}...")

        with self.driver.session(database=self.database) as session:
            # Export infrastructure nodes
            nodes = []
            result = session.run("""
                MATCH (n:Node)
                RETURN n.id AS id, n.name AS name, n.node_type AS node_type,
                       n.location AS location, n.zone AS zone
            """)
            for record in result:
                nodes.append(dict(record))

            # Export applications
            applications = []
            result = session.run("""
                MATCH (a:Application)
                RETURN a.id AS id, a.name AS name, a.role AS role
            """)
            for record in result:
                applications.append(dict(record))

            # Export topics with QoS
            topics = []
            result = session.run("""
                MATCH (t:Topic)
                RETURN t.id AS id, t.name AS name,
                       t.size AS message_size_bytes,
                       t.qos_durability AS qos_durability,
                       t.qos_reliability AS qos_reliability,
                       t.qos_transport_priority AS qos_transport_priority
            """)
            for record in result:
                topic_data = dict(record)
                # Restructure QoS
                qos = {
                    'durability': topic_data.pop('qos_durability', None),
                    'reliability': topic_data.pop('qos_reliability', None),
                    'transport_priority': topic_data.pop('qos_transport_priority', None)
                }
                topic_data['qos'] = {k: v for k, v in qos.items() if v is not None}
                topics.append(topic_data)

            # Export brokers
            brokers = []
            result = session.run("""
                MATCH (b:Broker)
                RETURN b.id AS id, b.name AS name
            """)
            for record in result:
                brokers.append(dict(record))

            # Export relationships
            relationships = {
                'runs_on': [],
                'publishes_to': [],
                'subscribes_to': [],
                'routes': [],
                'connects_to': []
            }

            # RUNS_ON (both applications and brokers)
            result = session.run("""
                MATCH (source)-[r:RUNS_ON]->(n:Node)
                RETURN source.id AS from_id, n.id AS to_id, labels(source)[0] AS source_type
            """)
            for record in result:
                relationships['runs_on'].append({
                    'from': record['from_id'],
                    'to': record['to_id']
                })

            # PUBLISHES_TO
            result = session.run("""
                MATCH (a:Application)-[r:PUBLISHES_TO]->(t:Topic)
                RETURN a.id AS from_id, t.id AS to_id, t.size AS message_size
            """)
            for record in result:
                relationships['publishes_to'].append({
                    'from': record['from_id'],
                    'to': record['to_id'],
                    'message_size': record['message_size']
                })

            # SUBSCRIBES_TO
            result = session.run("""
                MATCH (a:Application)-[r:SUBSCRIBES_TO]->(t:Topic)
                RETURN a.id AS from_id, t.id AS to_id
            """)
            for record in result:
                relationships['subscribes_to'].append({
                    'from': record['from_id'],
                    'to': record['to_id']
                })

            # ROUTES
            result = session.run("""
                MATCH (b:Broker)-[r:ROUTES]->(t:Topic)
                RETURN b.id AS from_id, t.id AS to_id
            """)
            for record in result:
                relationships['routes'].append({
                    'from': record['from_id'],
                    'to': record['to_id']
                })

            # CONNECTS_TO (physical topology)
            result = session.run("""
                MATCH (n1:Node)-[r:CONNECTS_TO]->(n2:Node)
                RETURN n1.id AS from_id, n2.id AS to_id,
                       r.bandwidth_mbps AS bandwidth_mbps, r.latency_ms AS latency_ms
            """)
            for record in result:
                relationships['connects_to'].append({
                    'from': record['from_id'],
                    'to': record['to_id'],
                    'bandwidth_mbps': record['bandwidth_mbps'],
                    'latency_ms': record['latency_ms']
                })

            # DEPENDS_ON (derived relationships)
            if include_derived:
                depends_on = []
                result = session.run("""
                    MATCH (source)-[d:DEPENDS_ON]->(target)
                    RETURN source.id AS from_id, target.id AS to_id,
                           d.dependency_type AS dependency_type,
                           d.weight AS weight,
                           d.topics AS topics,
                           labels(source)[0] AS source_type,
                           labels(target)[0] AS target_type
                """)
                for record in result:
                    depends_on.append({
                        'from': record['from_id'],
                        'to': record['to_id'],
                        'dependency_type': record['dependency_type'],
                        'weight': record['weight'],
                        'topics': record['topics'],
                        'source_type': record['source_type'],
                        'target_type': record['target_type']
                    })
                relationships['depends_on'] = depends_on

            # Build output
            output = {
                'metadata': {
                    'exported_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'source': 'neo4j',
                    'database': self.database,
                    'include_derived': include_derived
                },
                'nodes': nodes,
                'applications': applications,
                'topics': topics,
                'brokers': brokers,
                'relationships': relationships
            }

            # Write to file
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2, default=str)

            self.logger.info(
                f"Exported {len(nodes)} nodes, {len(applications)} apps, "
                f"{len(topics)} topics, {len(brokers)} brokers"
            )

    def run_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Run a Cypher query and return results.

        Args:
            query: Cypher query string
            params: Optional query parameters

        Returns:
            List of result dictionaries
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]

    def run_query_print(self, query: str, params: Optional[Dict] = None):
        """
        Run a Cypher query and print formatted results.

        Args:
            query: Cypher query string
            params: Optional query parameters
        """
        records = self.run_query(query, params)

        if not records:
            print("No results")
            return

        # Print header
        keys = list(records[0].keys())
        header = " | ".join(f"{k:20s}" for k in keys)
        print(header)
        print("-" * len(header))

        # Print rows
        for record in records:
            row = " | ".join(f"{str(record.get(k, '')):20s}" for k in keys)
            print(row)

        print(f"\n{len(records)} rows")

    def clear_database(self, confirm: bool = False):
        """
        Clear all data from database.

        Args:
            confirm: Must be True to proceed with deletion

        Raises:
            ValueError: If confirm is not True
        """
        if not confirm:
            raise ValueError("Set confirm=True to clear the database")

        with self.driver.session(database=self.database) as session:
            # Delete all relationships first
            session.run("MATCH ()-[r]->() DELETE r")
            # Then delete all nodes
            session.run("MATCH (n) DETACH DELETE n")

        self.logger.info("Database cleared")

    def get_statistics_summary(self) -> str:
        """
        Get a formatted statistics summary.

        Returns:
            Multi-line string with database statistics
        """
        info = self.get_database_info()

        lines = [
            "=" * 70,
            "DATABASE STATISTICS",
            "=" * 70,
            "",
            "Nodes by Label:"
        ]

        for label, count in sorted(info['nodes_by_label'].items()):
            lines.append(f"  {label:20s}: {count:6d}")
        lines.append(f"  {'TOTAL':20s}: {info['total_nodes']:6d}")

        lines.append("")
        lines.append("Relationships by Type:")
        for rel_type, count in sorted(info['relationships_by_type'].items()):
            lines.append(f"  {rel_type:20s}: {count:6d}")
        lines.append(f"  {'TOTAL':20s}: {info['total_relationships']:6d}")

        return "\n".join(lines)


def main():
    """Main entry point for CLI usage"""
    parser = argparse.ArgumentParser(
        description='Neo4j Utilities for pub-sub graph database operations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test                           # Test connection
  %(prog)s --info                           # Show database info
  %(prog)s --export graph.json              # Export graph to JSON
  %(prog)s --query "MATCH (n) RETURN n LIMIT 5"  # Run custom query
        """
    )

    # Connection options
    conn_group = parser.add_argument_group('Connection')
    conn_group.add_argument(
        '--uri', default='bolt://localhost:7687',
        help='Neo4j URI (default: bolt://localhost:7687)'
    )
    conn_group.add_argument(
        '--user', default='neo4j',
        help='Username (default: neo4j)'
    )
    conn_group.add_argument(
        '--password', default='password',
        help='Password (default: password)'
    )
    conn_group.add_argument(
        '--database', default='neo4j',
        help='Database name (default: neo4j)'
    )

    # Commands
    cmd_group = parser.add_argument_group('Commands')
    cmd_group.add_argument(
        '--test', action='store_true',
        help='Test connection'
    )
    cmd_group.add_argument(
        '--info', action='store_true',
        help='Show database info'
    )
    cmd_group.add_argument(
        '--schema', action='store_true',
        help='Show schema info (constraints and indexes)'
    )
    cmd_group.add_argument(
        '--export', metavar='FILE',
        help='Export graph to JSON file'
    )
    cmd_group.add_argument(
        '--query', metavar='CYPHER',
        help='Run Cypher query'
    )
    cmd_group.add_argument(
        '--clear', action='store_true',
        help='Clear database (requires confirmation)'
    )

    # Options
    parser.add_argument(
        '--no-derived', action='store_true',
        help='Exclude derived DEPENDS_ON relationships from export'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    # Check if neo4j is available
    if not NEO4J_AVAILABLE:
        print("Error: neo4j driver not installed")
        print("Install with: pip install neo4j")
        return 1

    # Create utilities instance
    try:
        utils = Neo4jUtilities(args.uri, args.user, args.password, args.database)
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return 1

    try:
        if args.test:
            print("Testing connection...")
            if utils.test_connection():
                print("Connection successful")
            else:
                print("Connection failed")
                return 1

        if args.info:
            print(utils.get_statistics_summary())

        if args.schema:
            print("\nSchema Information:")
            print("=" * 70)
            schema = utils.get_schema_info()

            print("\nConstraints:")
            if schema['constraints']:
                for c in schema['constraints']:
                    print(f"  - {c}")
            else:
                print("  None")

            print("\nIndexes:")
            if schema['indexes']:
                for idx in schema['indexes']:
                    print(f"  - {idx}")
            else:
                print("  None")

        if args.export:
            utils.export_to_json(args.export, include_derived=not args.no_derived)
            print(f"Graph exported to {args.export}")

        if args.query:
            print(f"\nRunning query: {args.query}\n")
            utils.run_query_print(args.query)

        if args.clear:
            response = input("Clear all data? This cannot be undone! [y/N]: ")
            if response.lower() == 'y':
                utils.clear_database(confirm=True)
                print("Database cleared")
            else:
                print("Cancelled")

    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        utils.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
