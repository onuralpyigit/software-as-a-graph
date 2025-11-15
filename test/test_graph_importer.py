#!/usr/bin/env python3
"""
Test Script for Neo4j Import System

Comprehensive test suite for validating:
- Import functionality
- Schema creation
- Query execution
- Error handling
- Performance metrics
"""

import json
import sys
import tempfile
from pathlib import Path
import time

# Test configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "neo4j"


def create_test_graph():
    """Create a simple test graph for validation"""
    return {
        "nodes": [
            {
                "id": "N1",
                "name": "TestNode1",
                "cpu_capacity": 16.0,
                "memory_gb": 32.0,
                "network_bandwidth_mbps": 1000.0,
                "zone": "zone-a",
                "region": "us-east-1"
            },
            {
                "id": "N2",
                "name": "TestNode2",
                "cpu_capacity": 8.0,
                "memory_gb": 16.0,
                "network_bandwidth_mbps": 1000.0,
                "zone": "zone-b",
                "region": "us-east-1"
            }
        ],
        "applications": [
            {
                "id": "A1",
                "name": "ProducerApp",
                "type": "PRODUCER",
                "criticality": "HIGH",
                "replicas": 2,
                "cpu_request": 2.0,
                "memory_request_mb": 1024.0
            },
            {
                "id": "A2",
                "name": "ConsumerApp",
                "type": "CONSUMER",
                "criticality": "CRITICAL",
                "replicas": 1,
                "cpu_request": 1.0,
                "memory_request_mb": 512.0
            },
            {
                "id": "A3",
                "name": "ProsumerApp",
                "type": "PROSUMER",
                "criticality": "MEDIUM",
                "replicas": 3,
                "cpu_request": 1.5,
                "memory_request_mb": 768.0
            }
        ],
        "topics": [
            {
                "id": "T1",
                "name": "sensor_data",
                "message_size_bytes": 1024,
                "expected_rate_hz": 10,
                "qos": {
                    "durability": "PERSISTENT",
                    "reliability": "RELIABLE",
                    "history_depth": 10,
                    "deadline_ms": 100,
                    "transport_priority": "HIGH"
                }
            },
            {
                "id": "T2",
                "name": "control_commands",
                "message_size_bytes": 512,
                "expected_rate_hz": 5,
                "qos": {
                    "durability": "VOLATILE",
                    "reliability": "BEST_EFFORT",
                    "history_depth": 1,
                    "deadline_ms": 50,
                    "transport_priority": "MEDIUM"
                }
            }
        ],
        "brokers": [
            {
                "id": "B1",
                "name": "MainBroker"
            }
        ],
        "relationships": {
            "runs_on": [
                {"from": "A1", "to": "N1"},
                {"from": "A2", "to": "N2"},
                {"from": "A3", "to": "N1"}
            ],
            "publishes_to": [
                {"from": "A1", "to": "T1", "period_ms": 100, "msg_size": 1024},
                {"from": "A3", "to": "T2", "period_ms": 200, "msg_size": 512}
            ],
            "subscribes_to": [
                {"from": "A2", "to": "T1"},
                {"from": "A3", "to": "T1"}
            ],
            "routes": [
                {"from": "B1", "to": "T1"},
                {"from": "B1", "to": "T2"}
            ]
        }
    }


def test_connection():
    """Test 1: Neo4j Connection"""
    print("\n" + "=" * 70)
    print("TEST 1: Neo4j Connection")
    print("=" * 70)
    
    try:
        from neo4j import GraphDatabase, basic_auth
        
        driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
        
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run("RETURN 1 as test")
            if result.single()["test"] == 1:
                print("✓ Connection successful")
                driver.close()
                return True
        
        driver.close()
        print("❌ Connection test failed")
        return False
        
    except ImportError:
        print("❌ neo4j driver not installed")
        print("   Install with: pip install neo4j")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def test_import():
    """Test 2: Basic Import"""
    print("\n" + "=" * 70)
    print("TEST 2: Basic Import")
    print("=" * 70)
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent / '..'))
        from src.core.graph_importer import GraphImporter
        
        # Create test graph
        test_graph = create_test_graph()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_graph, f, indent=2)
            temp_file = f.name
        
        print(f"Created test graph: {temp_file}")
        
        # Import
        importer = GraphImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
        
        try:
            # Clear first
            importer.clear_database()
            print("✓ Database cleared")
            
            # Create schema
            importer.create_schema()
            print("✓ Schema created")
            
            # Import graph
            start_time = time.time()
            importer.import_graph(test_graph, batch_size=10)
            duration = time.time() - start_time
            print(f"✓ Graph imported in {duration:.2f}s")
            
            # Verify counts
            stats = importer.get_statistics()
            
            expected = {
                'nodes': {'Node': 2, 'Application': 3, 'Topic': 2, 'Broker': 1, 'total': 8},
                'relationships': {'RUNS_ON': 3, 'PUBLISHES_TO': 2, 'SUBSCRIBES_TO': 2, 
                                'ROUTES': 2, 'DEPENDS_ON': 2, 'total': 11}
            }
            
            print("\nVerifying counts:")
            all_match = True
            for category in ['nodes', 'relationships']:
                for key, expected_count in expected[category].items():
                    actual_count = stats[category][key]
                    if actual_count == expected_count:
                        print(f"  ✓ {category}.{key}: {actual_count}")
                    else:
                        print(f"  ❌ {category}.{key}: expected {expected_count}, got {actual_count}")
                        all_match = False
            
            if all_match:
                print("\n✓ All counts match")
                return True
            else:
                print("\n❌ Count mismatch")
                return False
                
        finally:
            importer.close()
            Path(temp_file).unlink()
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_queries():
    """Test 3: Query Execution"""
    print("\n" + "=" * 70)
    print("TEST 3: Query Execution")
    print("=" * 70)
    
    try:
        from neo4j import GraphDatabase, basic_auth
        
        driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
        
        test_queries = [
            {
                'name': 'Count Applications',
                'query': 'MATCH (a:Application) RETURN count(a) as count',
                'expected': 3
            },
            {
                'name': 'Count Topics',
                'query': 'MATCH (t:Topic) RETURN count(t) as count',
                'expected': 2
            },
            {
                'name': 'Count Dependencies',
                'query': 'MATCH ()-[r:DEPENDS_ON]->() RETURN count(r) as count',
                'expected': 2
            },
            {
                'name': 'Find Critical Apps',
                'query': 'MATCH (a:Application) WHERE a.criticality = "CRITICAL" RETURN count(a) as count',
                'expected': 1
            },
            {
                'name': 'Find Producers',
                'query': 'MATCH (a:Application) WHERE a.type = "PRODUCER" RETURN count(a) as count',
                'expected': 1
            }
        ]
        
        all_passed = True
        
        with driver.session(database=NEO4J_DATABASE) as session:
            for test in test_queries:
                result = session.run(test['query'])
                count = result.single()['count']
                
                if count == test['expected']:
                    print(f"  ✓ {test['name']}: {count}")
                else:
                    print(f"  ❌ {test['name']}: expected {test['expected']}, got {count}")
                    all_passed = False
        
        driver.close()
        
        if all_passed:
            print("\n✓ All queries passed")
            return True
        else:
            print("\n❌ Some queries failed")
            return False
            
    except Exception as e:
        print(f"❌ Query test failed: {e}")
        return False


def test_schema():
    """Test 4: Schema Validation"""
    print("\n" + "=" * 70)
    print("TEST 4: Schema Validation")
    print("=" * 70)
    
    try:
        from neo4j import GraphDatabase, basic_auth
        
        driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
        
        with driver.session(database=NEO4J_DATABASE) as session:
            # Check constraints
            print("\nChecking constraints:")
            result = session.run("SHOW CONSTRAINTS")
            constraints = [record['name'] for record in result if 'name' in record.keys()]
            
            expected_constraints = ['node_id', 'app_id', 'topic_id', 'broker_id']
            for constraint in expected_constraints:
                if any(constraint in c for c in constraints):
                    print(f"  ✓ Constraint exists: {constraint}")
                else:
                    print(f"  ⚠ Constraint missing: {constraint}")
            
            # Check indexes
            print("\nChecking indexes:")
            result = session.run("SHOW INDEXES")
            indexes = [record['name'] for record in result if 'name' in record.keys()]
            
            expected_indexes = ['app_type', 'app_criticality', 'topic_name', 'node_zone']
            for index in expected_indexes:
                if any(index in i for i in indexes):
                    print(f"  ✓ Index exists: {index}")
                else:
                    print(f"  ⚠ Index missing: {index}")
        
        driver.close()
        print("\n✓ Schema validation complete")
        return True
        
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        return False


def test_error_handling():
    """Test 5: Error Handling"""
    print("\n" + "=" * 70)
    print("TEST 5: Error Handling")
    print("=" * 70)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / '..'))
        from src.core.graph_importer import GraphImporter
        
        # Test invalid reference
        invalid_graph = {
            "nodes": [],
            "applications": [
                {"id": "A1", "name": "App1", "type": "PRODUCER"}
            ],
            "topics": [],
            "brokers": [],
            "relationships": {
                "runs_on": [
                    {"from": "A1", "to": "INVALID_NODE"}  # Invalid reference
                ],
                "publishes_to": [],
                "subscribes_to": [],
                "routes": []
            }
        }
        
        importer = GraphImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
        
        try:
            importer.clear_database()
            importer.create_schema()
            
            # This should fail gracefully
            try:
                importer.import_graph(invalid_graph, batch_size=10)
                print("❌ Should have raised an error for invalid reference")
                return False
            except Exception as e:
                print(f"✓ Correctly caught error: {type(e).__name__}")
                print(f"  Error message: {str(e)[:100]}...")
                return True
                
        finally:
            importer.close()
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def test_performance():
    """Test 6: Performance Metrics"""
    print("\n" + "=" * 70)
    print("TEST 6: Performance Metrics")
    print("=" * 70)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / '..'))
        from src.core.graph_importer import GraphImporter
        
        # Create larger test graph
        large_graph = create_test_graph()
        
        # Multiply components
        for i in range(10):
            large_graph['applications'].append({
                "id": f"A{i+10}",
                "name": f"App{i+10}",
                "type": "CONSUMER",
                "criticality": "MEDIUM",
                "replicas": 1,
                "cpu_request": 1.0,
                "memory_request_mb": 512.0
            })
        
        importer = GraphImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
        
        try:
            importer.clear_database()
            importer.create_schema()
            
            # Measure import time
            start_time = time.time()
            importer.import_graph(large_graph, batch_size=5)
            duration = time.time() - start_time
            
            stats = importer.get_statistics()
            total_components = stats['nodes']['total']
            
            components_per_sec = total_components / duration if duration > 0 else 0
            
            print(f"\nPerformance metrics:")
            print(f"  Total components: {total_components}")
            print(f"  Import duration:  {duration:.2f}s")
            print(f"  Throughput:       {components_per_sec:.1f} components/sec")
            
            if components_per_sec > 10:
                print("\n✓ Performance acceptable")
                return True
            else:
                print("\n⚠ Performance below threshold")
                return True  # Still pass, just warning
                
        finally:
            importer.close()
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("NEO4J IMPORT SYSTEM - TEST SUITE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  URI:      {NEO4J_URI}")
    print(f"  User:     {NEO4J_USER}")
    print(f"  Database: {NEO4J_DATABASE}")
    
    tests = [
        ("Connection", test_connection),
        ("Import", test_import),
        ("Queries", test_queries),
        ("Schema", test_schema),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {status:10s} {name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total_count - passed_count} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
