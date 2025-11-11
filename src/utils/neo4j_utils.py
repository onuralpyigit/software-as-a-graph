#!/usr/bin/env python3
"""
Neo4j Utilities

Common utilities for working with Neo4j graph database including:
- Connection testing and diagnostics
- Database backup and restore
- Query templates and builders
- Performance analysis
- Data export utilities
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import time

try:
    from neo4j import GraphDatabase, basic_auth
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Error: neo4j driver not installed")
    print("Install with: pip install neo4j")
    sys.exit(1)


class Neo4jUtilities:
    """Utility functions for Neo4j operations"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """Initialize connection"""
        self.uri = uri
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
    
    def close(self):
        """Close connection"""
        self.driver.close()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def get_database_info(self) -> Dict:
        """Get database information"""
        with self.driver.session(database=self.database) as session:
            # Node counts by label
            node_counts = {}
            result = session.run("""
                CALL db.labels() YIELD label
                CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {})
                YIELD value
                RETURN label, value.count as count
            """)
            for record in result:
                node_counts[record['label']] = record['count']
            
            # Relationship counts by type
            rel_counts = {}
            result = session.run("""
                CALL db.relationshipTypes() YIELD relationshipType
                CALL apoc.cypher.run('MATCH ()-[r:' + relationshipType + ']->() RETURN count(r) as count', {})
                YIELD value
                RETURN relationshipType, value.count as count
            """)
            for record in result:
                rel_counts[record['relationshipType']] = record['count']
            
            # Database stats
            result = session.run("CALL dbms.queryJmx('org.neo4j:instance=kernel#0,name=Store file sizes')")
            
            return {
                'nodes_by_label': node_counts,
                'relationships_by_type': rel_counts,
                'total_nodes': sum(node_counts.values()),
                'total_relationships': sum(rel_counts.values())
            }
    
    def export_to_json(self, output_file: str):
        """Export entire graph to JSON"""
        print(f"Exporting graph to {output_file}...")
        
        with self.driver.session(database=self.database) as session:
            # Export nodes
            nodes = []
            applications = []
            topics = []
            brokers = []
            
            # Get all nodes
            result = session.run("MATCH (n:Node) RETURN n")
            for record in result:
                node = dict(record['n'])
                nodes.append(node)
            
            # Get applications
            result = session.run("MATCH (a:Application) RETURN a")
            for record in result:
                app = dict(record['a'])
                applications.append(app)
            
            # Get topics
            result = session.run("MATCH (t:Topic) RETURN t")
            for record in result:
                topic_data = dict(record['t'])
                # Restructure QoS
                qos = {
                    'durability': topic_data.pop('qos_durability', None),
                    'reliability': topic_data.pop('qos_reliability', None),
                    'history_depth': topic_data.pop('qos_history_depth', None),
                    'deadline_ms': topic_data.pop('qos_deadline_ms', None),
                    'lifespan_ms': topic_data.pop('qos_lifespan_ms', None),
                    'transport_priority': topic_data.pop('qos_transport_priority', None)
                }
                topic_data['qos'] = {k: v for k, v in qos.items() if v is not None}
                topics.append(topic_data)
            
            # Get brokers
            result = session.run("MATCH (b:Broker) RETURN b")
            for record in result:
                broker = dict(record['b'])
                brokers.append(broker)
            
            # Export relationships
            relationships = {
                'runs_on': [],
                'publishes_to': [],
                'subscribes_to': [],
                'routes': []
            }
            
            # RUNS_ON
            result = session.run("""
                MATCH (a:Application)-[r:RUNS_ON]->(n:Node)
                RETURN a.id as from, n.id as to
            """)
            for record in result:
                relationships['runs_on'].append({
                    'from': record['from'],
                    'to': record['to']
                })
            
            # PUBLISHES_TO
            result = session.run("""
                MATCH (a:Application)-[r:PUBLISHES_TO]->(t:Topic)
                RETURN a.id as from, t.id as to, r.period_ms as period_ms, r.msg_size as msg_size
            """)
            for record in result:
                relationships['publishes_to'].append({
                    'from': record['from'],
                    'to': record['to'],
                    'period_ms': record['period_ms'],
                    'msg_size': record['msg_size']
                })
            
            # SUBSCRIBES_TO
            result = session.run("""
                MATCH (a:Application)-[r:SUBSCRIBES_TO]->(t:Topic)
                RETURN a.id as from, t.id as to
            """)
            for record in result:
                relationships['subscribes_to'].append({
                    'from': record['from'],
                    'to': record['to']
                })
            
            # ROUTES
            result = session.run("""
                MATCH (b:Broker)-[r:ROUTES]->(t:Topic)
                RETURN b.id as from, t.id as to
            """)
            for record in result:
                relationships['routes'].append({
                    'from': record['from'],
                    'to': record['to']
                })
            
            # Build output
            output = {
                'nodes': nodes,
                'applications': applications,
                'topics': topics,
                'brokers': brokers,
                'relationships': relationships
            }
            
            # Write to file
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"✓ Exported {len(nodes)} nodes, {len(applications)} apps, "
                  f"{len(topics)} topics, {len(brokers)} brokers")
    
    def run_query(self, query: str, params: Optional[Dict] = None):
        """Run a Cypher query and print results"""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, params or {})
            
            # Print results
            records = list(result)
            if not records:
                print("No results")
                return
            
            # Print header
            keys = records[0].keys()
            header = " | ".join(f"{k:20s}" for k in keys)
            print(header)
            print("-" * len(header))
            
            # Print rows
            for record in records:
                row = " | ".join(f"{str(record[k]):20s}" for k in keys)
                print(row)
            
            print(f"\n{len(records)} rows")
    
    def clear_database(self):
        """Clear all data from database"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✓ Database cleared")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        with self.driver.session(database=self.database) as session:
            # Query execution stats
            result = session.run("""
                CALL dbms.queryJmx('org.neo4j:instance=kernel#0,name=Transactions')
                YIELD attributes
                RETURN attributes
            """)
            
            # Get cache stats
            cache_result = session.run("""
                CALL dbms.queryJmx('org.neo4j:instance=kernel#0,name=Page cache')
                YIELD attributes
                RETURN attributes
            """)
            
            return {
                'transactions': result.single()['attributes'] if result.peek() else {},
                'cache': cache_result.single()['attributes'] if cache_result.peek() else {}
            }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Neo4j Utilities')
    
    # Connection
    parser.add_argument('--uri', default='bolt://localhost:7687',
                       help='Neo4j URI')
    parser.add_argument('--user', default='neo4j',
                       help='Username')
    parser.add_argument('--password', default='password',
                       help='Password')
    parser.add_argument('--database', default='neo4j',
                       help='Database name')
    
    # Commands
    parser.add_argument('--test', action='store_true',
                       help='Test connection')
    parser.add_argument('--info', action='store_true',
                       help='Show database info')
    parser.add_argument('--export', metavar='FILE',
                       help='Export graph to JSON file')
    parser.add_argument('--query', metavar='CYPHER',
                       help='Run Cypher query')
    parser.add_argument('--clear', action='store_true',
                       help='Clear database')
    parser.add_argument('--stats', action='store_true',
                       help='Show performance statistics')
    
    args = parser.parse_args()
    
    # Create utilities instance
    utils = Neo4jUtilities(args.uri, args.user, args.password, args.database)
    
    try:
        if args.test:
            print("Testing connection...")
            if utils.test_connection():
                print("✓ Connection successful")
            else:
                print("❌ Connection failed")
                return 1
        
        if args.info:
            print("\n" + "=" * 70)
            print("DATABASE INFORMATION")
            print("=" * 70)
            info = utils.get_database_info()
            
            print("\nNodes:")
            for label, count in info['nodes_by_label'].items():
                print(f"  {label:20s}: {count:6d}")
            print(f"  {'Total':20s}: {info['total_nodes']:6d}")
            
            print("\nRelationships:")
            for rel_type, count in info['relationships_by_type'].items():
                print(f"  {rel_type:20s}: {count:6d}")
            print(f"  {'Total':20s}: {info['total_relationships']:6d}")
        
        if args.export:
            utils.export_to_json(args.export)
        
        if args.query:
            print(f"\nRunning query: {args.query}\n")
            utils.run_query(args.query)
        
        if args.clear:
            response = input("Clear all data? [y/N]: ")
            if response.lower() == 'y':
                utils.clear_database()
        
        if args.stats:
            print("\nPerformance Statistics:")
            stats = utils.get_performance_stats()
            print(json.dumps(stats, indent=2))
    
    finally:
        utils.close()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
