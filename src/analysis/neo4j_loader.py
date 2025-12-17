#!/usr/bin/env python3
"""
Neo4j Graph Loader
==================

Loads pub-sub graph data from Neo4j database for analysis.
Retrieves base entities and relationships, NOT derived DEPENDS_ON
(those are computed during analysis).

Usage:
    from src.analysis.neo4j_loader import Neo4jGraphLoader
    
    # Load from Neo4j
    loader = Neo4jGraphLoader(uri="bolt://localhost:7687", user="neo4j", password="pass")
    data = loader.load()
    
    # Use with GraphAnalyzer
    from src.analysis import GraphAnalyzer
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(data)
    result = analyzer.analyze()

Author: Software-as-a-Graph Research Project
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Gracefully handle missing neo4j driver
try:
    from neo4j import GraphDatabase, basic_auth
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None
    basic_auth = None
    ServiceUnavailable = Exception
    AuthError = Exception


@dataclass
class Neo4jConfig:
    """Configuration for Neo4j connection"""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    max_retries: int = 3


class Neo4jGraphLoader:
    """
    Loads pub-sub graph data from Neo4j database.
    
    Retrieves:
    - Nodes (infrastructure)
    - Brokers
    - Applications
    - Topics
    - Base relationships (PUBLISHES_TO, SUBSCRIBES_TO, RUNS_ON, ROUTES)
    
    Does NOT load DEPENDS_ON relationships - those are derived during analysis.
    """
    
    def __init__(self, 
                 uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "password",
                 database: str = "neo4j"):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j bolt URI
            user: Database username
            password: Database password
            database: Database name
        
        Raises:
            ImportError: If neo4j driver is not installed
        """
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "neo4j driver not installed. Install with: pip install neo4j"
            )
        
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.logger = logging.getLogger('neo4j_loader')
        
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=basic_auth(self.user, self.password)
            )
            # Verify connection
            self.driver.verify_connectivity()
            self.logger.info(f"Connected to Neo4j at {self.uri}")
        except ServiceUnavailable as e:
            self.logger.error(f"Could not connect to Neo4j at {self.uri}: {e}")
            raise
        except AuthError as e:
            self.logger.error(f"Authentication failed for Neo4j: {e}")
            raise
    
    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
            self.driver = None
            self.logger.info("Disconnected from Neo4j")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False
    
    def test_connection(self) -> bool:
        """Test if the connection is working"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def load(self) -> Dict[str, Any]:
        """
        Load complete graph data from Neo4j.
        
        Returns:
            Dictionary with nodes, brokers, applications, topics, and relationships
            in the format expected by GraphAnalyzer.
        """
        self.logger.info("Loading graph data from Neo4j...")
        
        data = {
            'nodes': self._load_nodes(),
            'brokers': self._load_brokers(),
            'applications': self._load_applications(),
            'topics': self._load_topics(),
            'relationships': {
                'publishes_to': self._load_publishes_to(),
                'subscribes_to': self._load_subscribes_to(),
                'runs_on': self._load_runs_on(),
                'routes': self._load_routes(),
                'connects_to': self._load_connects_to()
            }
        }
        
        # Log summary
        self.logger.info(
            f"Loaded: {len(data['nodes'])} nodes, {len(data['brokers'])} brokers, "
            f"{len(data['applications'])} applications, {len(data['topics'])} topics"
        )
        self.logger.info(
            f"Relationships: {len(data['relationships']['publishes_to'])} publishes_to, "
            f"{len(data['relationships']['subscribes_to'])} subscribes_to, "
            f"{len(data['relationships']['runs_on'])} runs_on, "
            f"{len(data['relationships']['routes'])} routes, "
            f"{len(data['relationships']['connects_to'])} connects_to"
        )
        
        return data
    
    def _load_nodes(self) -> List[Dict[str, Any]]:
        """Load infrastructure nodes"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n:Node)
                RETURN n.id AS id,
                       n.name AS name
            """)
            
            nodes = []
            for record in result:
                node = {k: v for k, v in dict(record).items() if v is not None}
                nodes.append(node)
            
            return nodes
    
    def _load_brokers(self) -> List[Dict[str, Any]]:
        """Load brokers with their host node"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (b:Broker)
                OPTIONAL MATCH (b)-[:RUNS_ON]->(n:Node)
                RETURN b.id AS id,
                       b.name AS name
            """)
            
            brokers = []
            for record in result:
                broker = {k: v for k, v in dict(record).items() if v is not None}
                brokers.append(broker)
            
            return brokers
    
    def _load_applications(self) -> List[Dict[str, Any]]:
        """Load applications with their host node"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (a:Application)
                OPTIONAL MATCH (a)-[:RUNS_ON]->(n:Node)
                RETURN a.id AS id,
                       a.name AS name,
                       a.role AS role
            """)
            
            applications = []
            for record in result:
                app = {k: v for k, v in dict(record).items() if v is not None}
                applications.append(app)
            
            return applications
    
    def _load_topics(self) -> List[Dict[str, Any]]:
        """Load topics with their broker and QoS settings"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (t:Topic)
                OPTIONAL MATCH (b:Broker)-[:ROUTES]->(t)
                RETURN t.id AS id,
                       t.name AS name,
                       t.size AS size,
                       t.qos_durability AS qos_durability,
                       t.qos_reliability AS qos_reliability,
                       t.qos_transport_priority AS qos_transport_priority
            """)
            
            topics = []
            for record in result:
                topic_dict = dict(record)
                
                # Build topic with QoS nested
                topic = {
                    'id': topic_dict['id'],
                    'name': topic_dict.get('name'),
                }

                if topic_dict.get('size'):
                    topic['size'] = topic_dict['size']
                
                # Collect QoS settings
                qos = {}
                for key in ['durability', 'reliability', 'transport_priority']:
                    qos_key = f'qos_{key}'
                    if topic_dict.get(qos_key) is not None:
                        qos[key] = topic_dict[qos_key]
                
                if qos:
                    topic['qos'] = qos
                
                # Remove None values
                topic = {k: v for k, v in topic.items() if v is not None}
                topics.append(topic)
            
            return topics
    
    def _load_publishes_to(self) -> List[Dict[str, str]]:
        """Load PUBLISHES_TO relationships"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (a:Application)-[r:PUBLISHES_TO]->(t:Topic)
                RETURN a.id AS from, t.id AS to
            """)
            
            rels = []
            for record in result:
                rel = {'from': record['from'], 'to': record['to']}
                rels.append(rel)
            
            return rels
    
    def _load_subscribes_to(self) -> List[Dict[str, str]]:
        """Load SUBSCRIBES_TO relationships"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (a:Application)-[r:SUBSCRIBES_TO]->(t:Topic)
                RETURN a.id AS from, t.id AS to
            """)
            
            return [{'from': r['from'], 'to': r['to']} for r in result]
    
    def _load_runs_on(self) -> List[Dict[str, str]]:
        """Load RUNS_ON relationships (apps/brokers to nodes)"""
        with self.driver.session(database=self.database) as session:
            # Applications running on nodes
            app_result = session.run("""
                MATCH (a:Application)-[:RUNS_ON]->(n:Node)
                RETURN a.id AS from, n.id AS to
            """)
            
            # Brokers running on nodes
            broker_result = session.run("""
                MATCH (b:Broker)-[:RUNS_ON]->(n:Node)
                RETURN b.id AS from, n.id AS to
            """)
            
            rels = [{'from': r['from'], 'to': r['to']} for r in app_result]
            rels.extend([{'from': r['from'], 'to': r['to']} for r in broker_result])
            
            return rels
    
    def _load_routes(self) -> List[Dict[str, str]]:
        """Load ROUTES relationships (brokers to topics)"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (b:Broker)-[:ROUTES]->(t:Topic)
                RETURN b.id AS from, t.id AS to
            """)
            
            return [{'from': r['from'], 'to': r['to']} for r in result]
        
    def _load_connects_to(self) -> List[Dict[str, str]]:
        """Load CONNECTS_TO relationships (nodes to nodes)"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (n1:Node)-[:CONNECTS_TO]->(n2:Node)
                RETURN n1.id AS from, n2.id AS to
            """)
            
            return [{'from': r['from'], 'to': r['to']} for r in result]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with counts of nodes and relationships by type
        """
        with self.driver.session(database=self.database) as session:
            # Node counts by label
            node_result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS label, count(*) AS count
            """)
            nodes_by_label = {r['label']: r['count'] for r in node_result}
            
            # Relationship counts by type
            rel_result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS type, count(*) AS count
            """)
            rels_by_type = {r['type']: r['count'] for r in rel_result}
            
            return {
                'nodes_by_label': nodes_by_label,
                'relationships_by_type': rels_by_type,
                'total_nodes': sum(nodes_by_label.values()),
                'total_relationships': sum(rels_by_type.values())
            }


def load_from_neo4j(uri: str = "bolt://localhost:7687",
                    user: str = "neo4j",
                    password: str = "password",
                    database: str = "neo4j") -> Dict[str, Any]:
    """
    Convenience function to load graph data from Neo4j.
    
    Args:
        uri: Neo4j bolt URI
        user: Database username
        password: Database password
        database: Database name
    
    Returns:
        Dictionary with graph data in GraphAnalyzer format
    """
    with Neo4jGraphLoader(uri, user, password, database) as loader:
        return loader.load()


# ============================================================================
# CLI for standalone usage
# ============================================================================

def main():
    """CLI entry point for Neo4j graph loader"""
    import argparse
    import json
    import sys
    
    parser = argparse.ArgumentParser(
        description='Load pub-sub graph data from Neo4j',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test connection
    python neo4j_loader.py --test
    
    # Load and print statistics
    python neo4j_loader.py --stats
    
    # Export to JSON file
    python neo4j_loader.py --output graph_data.json
    
    # Custom connection
    python neo4j_loader.py --uri bolt://server:7687 --user admin --password secret
        """
    )
    
    # Connection options
    parser.add_argument('--uri', default='bolt://localhost:7687',
                        help='Neo4j URI (default: bolt://localhost:7687)')
    parser.add_argument('--user', default='neo4j',
                        help='Username (default: neo4j)')
    parser.add_argument('--password', default='password',
                        help='Password (default: password)')
    parser.add_argument('--database', default='neo4j',
                        help='Database name (default: neo4j)')
    
    # Actions
    parser.add_argument('--test', action='store_true',
                        help='Test connection only')
    parser.add_argument('--stats', action='store_true',
                        help='Show database statistics')
    parser.add_argument('--output', '-o',
                        help='Output JSON file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        loader = Neo4jGraphLoader(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database
        )
        
        try:
            if args.test:
                if loader.test_connection():
                    print("✓ Connection successful")
                    return 0
                else:
                    print("✗ Connection failed")
                    return 1
            
            if args.stats:
                stats = loader.get_statistics()
                print("\n" + "="*50)
                print("DATABASE STATISTICS")
                print("="*50)
                print("\nNodes by Label:")
                for label, count in stats['nodes_by_label'].items():
                    print(f"  {label}: {count}")
                print(f"  Total: {stats['total_nodes']}")
                print("\nRelationships by Type:")
                for rel_type, count in stats['relationships_by_type'].items():
                    print(f"  {rel_type}: {count}")
                print(f"  Total: {stats['total_relationships']}")
                return 0
            
            # Load data
            data = loader.load()
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"✓ Data exported to {args.output}")
            else:
                # Print to stdout
                print(json.dumps(data, indent=2))
            
            return 0
            
        finally:
            loader.close()
            
    except ImportError as e:
        print(f"Error: {e}")
        print("Install neo4j driver: pip install neo4j")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())