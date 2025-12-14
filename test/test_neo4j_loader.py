#!/usr/bin/env python3
"""
Test Suite for Neo4j Graph Loader
=================================

Tests for the Neo4j loader functionality.
Uses mocking to test without requiring a real Neo4j instance.

Usage:
    python test_neo4j_loader.py
    python test_neo4j_loader.py -v

Author: Software-as-a-Graph Research Project
"""

import sys
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Mock Data
# ============================================================================

def get_mock_neo4j_data() -> Dict[str, List[Dict]]:
    """Get mock data that simulates Neo4j query results"""
    return {
        'nodes': [
            {'id': 'N1', 'name': 'Node1', 'node_type': 'compute', 'location': 'zone-a'},
            {'id': 'N2', 'name': 'Node2', 'node_type': 'edge', 'location': 'zone-b'}
        ],
        'brokers': [
            {'id': 'B1', 'name': 'Broker1', 'broker_type': 'mqtt', 'port': 1883, 'node': 'N1'}
        ],
        'applications': [
            {'id': 'A1', 'name': 'Publisher', 'role': 'pub', 'node': 'N1'},
            {'id': 'A2', 'name': 'Subscriber', 'role': 'sub', 'node': 'N2'}
        ],
        'topics': [
            {'id': 'T1', 'name': 'Topic1', 'broker': 'B1', 
             'qos_durability': 'transient', 'qos_reliability': 'reliable'}
        ],
        'publishes_to': [
            {'from': 'A1', 'to': 'T1'}
        ],
        'subscribes_to': [
            {'from': 'A2', 'to': 'T1'}
        ],
        'runs_on': [
            {'from': 'A1', 'to': 'N1'},
            {'from': 'A2', 'to': 'N2'},
            {'from': 'B1', 'to': 'N1'}
        ],
        'routes': [
            {'from': 'B1', 'to': 'T1'}
        ]
    }


def create_mock_record(data: Dict) -> Mock:
    """Create a mock Neo4j record"""
    record = Mock()
    record.__getitem__ = lambda self, key: data.get(key)
    record.get = lambda key, default=None: data.get(key, default)
    record.keys = lambda: data.keys()
    return record


def create_mock_result(records: List[Dict]) -> Mock:
    """Create a mock Neo4j result set"""
    result = Mock()
    mock_records = [create_mock_record(r) for r in records]
    result.__iter__ = lambda self: iter(mock_records)
    result.single = lambda: mock_records[0] if mock_records else None
    return result


# ============================================================================
# Test Classes
# ============================================================================

class TestNeo4jGraphLoaderWithMock(unittest.TestCase):
    """Tests for Neo4jGraphLoader using mocks"""
    
    def setUp(self):
        """Set up mocks for each test"""
        self.mock_data = get_mock_neo4j_data()
        
        # Create mock session
        self.mock_session = MagicMock()
        
        # Configure run method to return appropriate data based on query
        def mock_run(query, params=None):
            if 'MATCH (n:Node)' in query:
                return create_mock_result(self.mock_data['nodes'])
            elif 'MATCH (b:Broker)' in query:
                return create_mock_result(self.mock_data['brokers'])
            elif 'MATCH (a:Application)' in query:
                return create_mock_result(self.mock_data['applications'])
            elif 'MATCH (t:Topic)' in query:
                return create_mock_result(self.mock_data['topics'])
            elif 'PUBLISHES_TO' in query:
                return create_mock_result(self.mock_data['publishes_to'])
            elif 'SUBSCRIBES_TO' in query:
                return create_mock_result(self.mock_data['subscribes_to'])
            elif 'labels(n)' in query:  # Statistics query
                return create_mock_result([
                    {'label': 'Node', 'count': 2},
                    {'label': 'Broker', 'count': 1},
                    {'label': 'Application', 'count': 2},
                    {'label': 'Topic', 'count': 1}
                ])
            elif 'type(r)' in query:  # Relationship statistics
                return create_mock_result([
                    {'type': 'PUBLISHES_TO', 'count': 1},
                    {'type': 'SUBSCRIBES_TO', 'count': 1},
                    {'type': 'RUNS_ON', 'count': 3},
                    {'type': 'ROUTES', 'count': 1}
                ])
            elif 'RETURN 1' in query:  # Connection test
                return create_mock_result([{'test': 1}])
            else:
                return create_mock_result([])
        
        self.mock_session.run = mock_run
        self.mock_session.__enter__ = Mock(return_value=self.mock_session)
        self.mock_session.__exit__ = Mock(return_value=False)
    
    @patch('src.analysis.neo4j_loader.NEO4J_AVAILABLE', True)
    @patch('src.analysis.neo4j_loader.basic_auth')
    @patch('src.analysis.neo4j_loader.GraphDatabase')
    def test_connection(self, mock_graph_db, mock_basic_auth):
        """Test Neo4j connection"""
        from src.analysis.neo4j_loader import Neo4jGraphLoader
        
        # Setup mock driver
        mock_driver = MagicMock()
        mock_driver.verify_connectivity = Mock()
        mock_driver.session.return_value = self.mock_session
        mock_graph_db.driver.return_value = mock_driver
        mock_basic_auth.return_value = ('neo4j', 'password')
        
        # Create loader
        loader = Neo4jGraphLoader(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )
        
        # Verify connection was established
        mock_graph_db.driver.assert_called_once()
        mock_driver.verify_connectivity.assert_called_once()
        
        loader.close()
    
    @patch('src.analysis.neo4j_loader.NEO4J_AVAILABLE', True)
    @patch('src.analysis.neo4j_loader.basic_auth')
    @patch('src.analysis.neo4j_loader.GraphDatabase')
    def test_load_returns_correct_structure(self, mock_graph_db, mock_basic_auth):
        """Test that load() returns data in correct format"""
        from src.analysis.neo4j_loader import Neo4jGraphLoader
        
        # Setup mock
        mock_driver = MagicMock()
        mock_driver.verify_connectivity = Mock()
        mock_driver.session.return_value = self.mock_session
        mock_graph_db.driver.return_value = mock_driver
        mock_basic_auth.return_value = ('neo4j', 'password')
        
        loader = Neo4jGraphLoader()
        data = loader.load()
        loader.close()
        
        # Verify structure
        self.assertIn('nodes', data)
        self.assertIn('brokers', data)
        self.assertIn('applications', data)
        self.assertIn('topics', data)
        self.assertIn('relationships', data)
        
        # Verify relationships structure
        rels = data['relationships']
        self.assertIn('publishes_to', rels)
        self.assertIn('subscribes_to', rels)
        self.assertIn('runs_on', rels)
        self.assertIn('routes', rels)
    
    @patch('src.analysis.neo4j_loader.NEO4J_AVAILABLE', True)
    @patch('src.analysis.neo4j_loader.basic_auth')
    @patch('src.analysis.neo4j_loader.GraphDatabase')
    def test_test_connection(self, mock_graph_db, mock_basic_auth):
        """Test the test_connection method"""
        from src.analysis.neo4j_loader import Neo4jGraphLoader
        
        mock_driver = MagicMock()
        mock_driver.verify_connectivity = Mock()
        mock_driver.session.return_value = self.mock_session
        mock_graph_db.driver.return_value = mock_driver
        mock_basic_auth.return_value = ('neo4j', 'password')
        
        loader = Neo4jGraphLoader()
        result = loader.test_connection()
        loader.close()
        
        self.assertTrue(result)
    
    @patch('src.analysis.neo4j_loader.NEO4J_AVAILABLE', True)
    @patch('src.analysis.neo4j_loader.basic_auth')
    @patch('src.analysis.neo4j_loader.GraphDatabase')
    def test_get_statistics(self, mock_graph_db, mock_basic_auth):
        """Test get_statistics method"""
        from src.analysis.neo4j_loader import Neo4jGraphLoader
        
        mock_driver = MagicMock()
        mock_driver.verify_connectivity = Mock()
        mock_driver.session.return_value = self.mock_session
        mock_graph_db.driver.return_value = mock_driver
        mock_basic_auth.return_value = ('neo4j', 'password')
        
        loader = Neo4jGraphLoader()
        stats = loader.get_statistics()
        loader.close()
        
        self.assertIn('nodes_by_label', stats)
        self.assertIn('relationships_by_type', stats)
        self.assertIn('total_nodes', stats)
        self.assertIn('total_relationships', stats)
    
    @patch('src.analysis.neo4j_loader.NEO4J_AVAILABLE', True)
    @patch('src.analysis.neo4j_loader.basic_auth')
    @patch('src.analysis.neo4j_loader.GraphDatabase')
    def test_context_manager(self, mock_graph_db, mock_basic_auth):
        """Test context manager protocol"""
        from src.analysis.neo4j_loader import Neo4jGraphLoader
        
        mock_driver = MagicMock()
        mock_driver.verify_connectivity = Mock()
        mock_driver.session.return_value = self.mock_session
        mock_graph_db.driver.return_value = mock_driver
        mock_basic_auth.return_value = ('neo4j', 'password')
        
        with Neo4jGraphLoader() as loader:
            self.assertIsNotNone(loader.driver)
        
        # Verify close was called
        mock_driver.close.assert_called_once()


class TestGraphAnalyzerNeo4jIntegration(unittest.TestCase):
    """Tests for GraphAnalyzer.load_from_neo4j integration"""
    
    @patch('src.analysis.neo4j_loader.NEO4J_AVAILABLE', True)
    @patch('src.analysis.neo4j_loader.basic_auth')
    @patch('src.analysis.neo4j_loader.GraphDatabase')
    def test_load_from_neo4j(self, mock_graph_db, mock_basic_auth):
        """Test GraphAnalyzer.load_from_neo4j method"""
        from src.analysis import GraphAnalyzer
        
        # Setup mock data
        mock_data = get_mock_neo4j_data()
        mock_session = MagicMock()
        
        def mock_run(query, params=None):
            if 'MATCH (n:Node)' in query:
                return create_mock_result(mock_data['nodes'])
            elif 'MATCH (b:Broker)' in query:
                return create_mock_result(mock_data['brokers'])
            elif 'MATCH (a:Application)' in query:
                return create_mock_result(mock_data['applications'])
            elif 'MATCH (t:Topic)' in query:
                return create_mock_result(mock_data['topics'])
            elif 'PUBLISHES_TO' in query:
                return create_mock_result(mock_data['publishes_to'])
            elif 'SUBSCRIBES_TO' in query:
                return create_mock_result(mock_data['subscribes_to'])
            else:
                return create_mock_result([])
        
        mock_session.run = mock_run
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        
        mock_driver = MagicMock()
        mock_driver.verify_connectivity = Mock()
        mock_driver.session.return_value = mock_session
        mock_graph_db.driver.return_value = mock_driver
        mock_basic_auth.return_value = ('neo4j', 'password')
        
        # Test load_from_neo4j
        analyzer = GraphAnalyzer()
        analyzer.load_from_neo4j(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )
        
        # Verify data was loaded
        self.assertIsNotNone(analyzer.raw_data)
        self.assertIn('nodes', analyzer.raw_data)
        self.assertIn('applications', analyzer.raw_data)
    
    @patch('src.analysis.neo4j_loader.NEO4J_AVAILABLE', True)
    @patch('src.analysis.neo4j_loader.basic_auth')
    @patch('src.analysis.neo4j_loader.GraphDatabase')
    def test_analyze_after_neo4j_load(self, mock_graph_db, mock_basic_auth):
        """Test full analysis after loading from Neo4j"""
        from src.analysis import GraphAnalyzer
        
        # Setup mock data
        mock_data = get_mock_neo4j_data()
        mock_session = MagicMock()
        
        def mock_run(query, params=None):
            if 'MATCH (n:Node)' in query:
                return create_mock_result(mock_data['nodes'])
            elif 'MATCH (b:Broker)' in query:
                return create_mock_result(mock_data['brokers'])
            elif 'MATCH (a:Application)' in query:
                return create_mock_result(mock_data['applications'])
            elif 'MATCH (t:Topic)' in query:
                return create_mock_result(mock_data['topics'])
            elif 'PUBLISHES_TO' in query:
                return create_mock_result(mock_data['publishes_to'])
            elif 'SUBSCRIBES_TO' in query:
                return create_mock_result(mock_data['subscribes_to'])
            else:
                return create_mock_result([])
        
        mock_session.run = mock_run
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        
        mock_driver = MagicMock()
        mock_driver.verify_connectivity = Mock()
        mock_driver.session.return_value = mock_session
        mock_graph_db.driver.return_value = mock_driver
        mock_basic_auth.return_value = ('neo4j', 'password')
        
        # Load and analyze
        analyzer = GraphAnalyzer()
        analyzer.load_from_neo4j()
        result = analyzer.analyze()
        
        # Verify analysis results
        self.assertIsNotNone(result)
        self.assertGreater(result.graph_summary['total_nodes'], 0)


class TestNeo4jDriverNotInstalled(unittest.TestCase):
    """Tests for handling missing neo4j driver"""
    
    @patch('src.analysis.neo4j_loader.NEO4J_AVAILABLE', False)
    def test_import_error_when_neo4j_not_available(self):
        """Test ImportError is raised when neo4j driver not installed"""
        # Reimport with NEO4J_AVAILABLE = False
        import importlib
        import src.analysis.neo4j_loader as loader_module
        
        with patch.object(loader_module, 'NEO4J_AVAILABLE', False):
            with self.assertRaises(ImportError):
                loader_module.Neo4jGraphLoader()


class TestLoadFromNeo4jConvenienceFunction(unittest.TestCase):
    """Tests for load_from_neo4j convenience function"""
    
    @patch('src.analysis.neo4j_loader.NEO4J_AVAILABLE', True)
    @patch('src.analysis.neo4j_loader.basic_auth')
    @patch('src.analysis.neo4j_loader.GraphDatabase')
    def test_load_from_neo4j_function(self, mock_graph_db, mock_basic_auth):
        """Test the load_from_neo4j convenience function"""
        from src.analysis.neo4j_loader import load_from_neo4j
        
        mock_data = get_mock_neo4j_data()
        mock_session = MagicMock()
        
        def mock_run(query, params=None):
            if 'MATCH (n:Node)' in query:
                return create_mock_result(mock_data['nodes'])
            elif 'MATCH (b:Broker)' in query:
                return create_mock_result(mock_data['brokers'])
            elif 'MATCH (a:Application)' in query:
                return create_mock_result(mock_data['applications'])
            elif 'MATCH (t:Topic)' in query:
                return create_mock_result(mock_data['topics'])
            elif 'PUBLISHES_TO' in query:
                return create_mock_result(mock_data['publishes_to'])
            elif 'SUBSCRIBES_TO' in query:
                return create_mock_result(mock_data['subscribes_to'])
            else:
                return create_mock_result([])
        
        mock_session.run = mock_run
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        
        mock_driver = MagicMock()
        mock_driver.verify_connectivity = Mock()
        mock_driver.session.return_value = mock_session
        mock_graph_db.driver.return_value = mock_driver
        mock_basic_auth.return_value = ('neo4j', 'password')
        
        data = load_from_neo4j()
        
        self.assertIn('nodes', data)
        self.assertIn('applications', data)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestNeo4jGraphLoaderWithMock))
    suite.addTests(loader.loadTestsFromTestCase(TestGraphAnalyzerNeo4jIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestNeo4jDriverNotInstalled))
    suite.addTests(loader.loadTestsFromTestCase(TestLoadFromNeo4jConvenienceFunction))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())