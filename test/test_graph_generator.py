#!/usr/bin/env python3
"""
Test Suite - Simplified Graph Model Version 3.0

Tests for:
- GraphConfig validation
- GraphGenerator
- Anti-patterns
- GraphBuilder
- GraphExporter
- QoSPolicy

Usage:
    python test_graph_generation.py           # All tests
    python test_graph_generation.py --quick   # Quick tests
    python test_graph_generation.py -v        # Verbose
"""

import sys
import json
import unittest
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.graph_generator import GraphGenerator, GraphConfig
from src.core.graph_builder import GraphBuilder, ValidationResult, GraphDiffResult
from src.core.graph_exporter import GraphExporter
from src.core.graph_model import GraphModel, QoSPolicy, Application, Broker, Topic, Node, Edge, EdgeType


class TestGraphConfig(unittest.TestCase):
    """Tests for GraphConfig"""
    
    def test_valid_config(self):
        """Test valid configuration"""
        config = GraphConfig(scale='small', scenario='generic', seed=42)
        self.assertEqual(config.scale, 'small')
        self.assertEqual(config.scenario, 'generic')
        self.assertEqual(config.seed, 42)
    
    def test_invalid_scale(self):
        """Test invalid scale raises error"""
        with self.assertRaises(ValueError):
            GraphConfig(scale='invalid', scenario='generic')
    
    def test_invalid_scenario(self):
        """Test invalid scenario raises error"""
        with self.assertRaises(ValueError):
            GraphConfig(scale='small', scenario='invalid')
    
    def test_invalid_antipattern(self):
        """Test invalid antipattern raises error"""
        with self.assertRaises(ValueError):
            GraphConfig(scale='small', scenario='generic', antipatterns=['invalid'])
    
    def test_all_scales(self):
        """Test all valid scales"""
        scales = ['tiny', 'small', 'medium', 'large', 'xlarge', 'extreme']
        for scale in scales:
            config = GraphConfig(scale=scale, scenario='generic')
            self.assertEqual(config.scale, scale)
    
    def test_all_scenarios(self):
        """Test all valid scenarios"""
        scenarios = ['generic', 'iot', 'financial', 'ecommerce', 'analytics',
                    'smart_city', 'healthcare', 'autonomous_vehicle', 'gaming']
        for scenario in scenarios:
            config = GraphConfig(scale='tiny', scenario=scenario)
            self.assertEqual(config.scenario, scenario)


class TestGraphGenerator(unittest.TestCase):
    """Tests for GraphGenerator"""
    
    def test_basic_generation(self):
        """Test basic graph generation"""
        config = GraphConfig(scale='tiny', scenario='generic', seed=42)
        graph = GraphGenerator(config).generate()
        
        # Check structure
        self.assertIn('metadata', graph)
        self.assertIn('applications', graph)
        self.assertIn('brokers', graph)
        self.assertIn('topics', graph)
        self.assertIn('nodes', graph)
        self.assertIn('relationships', graph)
        
        # Check edges structure
        edges = graph['relationships']
        self.assertIn('publishes_to', edges)
        self.assertIn('subscribes_to', edges)
        self.assertIn('routes', edges)
        self.assertIn('runs_on', edges)
        self.assertIn('connects_to', edges)
    
    def test_reproducibility(self):
        """Test same seed produces identical graphs"""
        config1 = GraphConfig(scale='small', scenario='generic', seed=12345)
        config2 = GraphConfig(scale='small', scenario='generic', seed=12345)
        
        graph1 = GraphGenerator(config1).generate()
        graph2 = GraphGenerator(config2).generate()
        
        self.assertEqual(len(graph1['applications']), len(graph2['applications']))
        self.assertEqual(len(graph1['topics']), len(graph2['topics']))
    
    def test_scale_sizes(self):
        """Test scales produce expected sizes"""
        expected = {
            'tiny': {'nodes': 3, 'apps': 5, 'topics': 3, 'brokers': 1},
            'small': {'nodes': 5, 'apps': 12, 'topics': 8, 'brokers': 2}
        }
        
        for scale, expected_counts in expected.items():
            config = GraphConfig(scale=scale, scenario='generic', seed=42)
            graph = GraphGenerator(config).generate()
            
            self.assertEqual(len(graph['nodes']), expected_counts['nodes'])
            self.assertEqual(len(graph['applications']), expected_counts['apps'])
            self.assertEqual(len(graph['topics']), expected_counts['topics'])
            self.assertEqual(len(graph['brokers']), expected_counts['brokers'])
    
    def test_application_roles(self):
        """Test applications have valid roles"""
        config = GraphConfig(scale='small', scenario='generic', seed=42)
        graph = GraphGenerator(config).generate()
        
        valid_roles = {'pub', 'sub', 'pubsub'}
        for app in graph['applications']:
            self.assertIn('role', app)
            self.assertIn(app['role'], valid_roles)
    
    def test_topic_qos(self):
        """Test topics have QoS properties"""
        config = GraphConfig(scale='small', scenario='generic', seed=42)
        graph = GraphGenerator(config).generate()
        
        for topic in graph['topics']:
            self.assertIn('qos', topic)
            qos = topic['qos']
            self.assertIn('durability', qos)
            self.assertIn('reliability', qos)
            self.assertIn('transport_priority', qos)
    
    def test_topic_size(self):
        """Test topics have size attribute"""
        config = GraphConfig(scale='small', scenario='generic', seed=42)
        graph = GraphGenerator(config).generate()
        
        for topic in graph['topics']:
            self.assertIn('size', topic)
            self.assertIsInstance(topic['size'], int)
            self.assertGreater(topic['size'], 0)
    
    def test_edges_valid(self):
        """Test all edges reference valid IDs"""
        config = GraphConfig(scale='small', scenario='generic', seed=42)
        graph = GraphGenerator(config).generate()
        
        app_ids = {a['id'] for a in graph['applications']}
        broker_ids = {b['id'] for b in graph['brokers']}
        topic_ids = {t['id'] for t in graph['topics']}
        node_ids = {n['id'] for n in graph['nodes']}
        
        for pub in graph['relationships']['publishes_to']:
            self.assertIn(pub['from'], app_ids)
            self.assertIn(pub['to'], topic_ids)
        
        for sub in graph['relationships']['subscribes_to']:
            self.assertIn(sub['from'], app_ids)
            self.assertIn(sub['to'], topic_ids)
        
        for route in graph['relationships']['routes']:
            self.assertIn(route['from'], broker_ids)
            self.assertIn(route['to'], topic_ids)
    
    def test_custom_counts(self):
        """Test custom component counts"""
        config = GraphConfig(
            scale='tiny',
            scenario='generic',
            num_nodes=10,
            num_applications=20,
            num_topics=15,
            num_brokers=5,
            seed=42
        )
        graph = GraphGenerator(config).generate()
        
        self.assertEqual(len(graph['nodes']), 10)
        self.assertEqual(len(graph['applications']), 20)
        self.assertEqual(len(graph['topics']), 15)
        self.assertEqual(len(graph['brokers']), 5)


class TestAntipatterns(unittest.TestCase):
    """Tests for anti-pattern injection"""
    
    def test_spof(self):
        """Test SPOF antipattern"""
        config = GraphConfig(scale='small', scenario='generic', antipatterns=['spof'], seed=42)
        graph = GraphGenerator(config).generate()
        
        ap = graph['metadata'].get('antipatterns_applied', {}).get('spof')
        self.assertIsNotNone(ap)
    
    def test_god_topic(self):
        """Test god topic antipattern"""
        config = GraphConfig(scale='small', scenario='generic', antipatterns=['god_topic'], seed=42)
        graph = GraphGenerator(config).generate()
        
        ap = graph['metadata'].get('antipatterns_applied', {}).get('god_topic')
        self.assertIsNotNone(ap)
    
    def test_broker_overload(self):
        """Test broker overload antipattern"""
        config = GraphConfig(scale='small', scenario='generic', antipatterns=['broker_overload'], seed=42)
        graph = GraphGenerator(config).generate()
        
        ap = graph['metadata'].get('antipatterns_applied', {}).get('broker_overload')
        # May not apply if only 1 broker
        if len(graph['brokers']) >= 2:
            self.assertIsNotNone(ap)
    
    def test_multiple_antipatterns(self):
        """Test multiple antipatterns"""
        config = GraphConfig(
            scale='small', 
            scenario='generic', 
            antipatterns=['spof', 'tight_coupling'], 
            seed=42
        )
        graph = GraphGenerator(config).generate()
        
        applied = graph['metadata'].get('antipatterns_applied', {})
        self.assertIn('spof', applied)
        self.assertIn('tight_coupling', applied)


class TestGraphBuilder(unittest.TestCase):
    """Tests for GraphBuilder"""
    
    def test_build_from_dict(self):
        """Test building from dictionary"""
        config = GraphConfig(scale='tiny', scenario='generic', seed=42)
        graph_dict = GraphGenerator(config).generate()
        
        builder = GraphBuilder()
        model = builder.build_from_dict(graph_dict)
        
        self.assertIsInstance(model, GraphModel)
        self.assertEqual(len(model.applications), len(graph_dict['applications']))
        self.assertEqual(len(model.topics), len(graph_dict['topics']))
    
    def test_build_from_json(self):
        """Test building from JSON file"""
        config = GraphConfig(scale='tiny', scenario='generic', seed=42)
        graph_dict = GraphGenerator(config).generate()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(graph_dict, f, default=str)
            path = f.name
        
        try:
            builder = GraphBuilder()
            model = builder.build_from_json(path)
            self.assertIsInstance(model, GraphModel)
        finally:
            Path(path).unlink()
    
    def test_model_to_dict(self):
        """Test model to dict roundtrip"""
        config = GraphConfig(scale='tiny', scenario='generic', seed=42)
        graph_dict = GraphGenerator(config).generate()
        
        builder = GraphBuilder()
        model = builder.build_from_dict(graph_dict)
        result = model.to_dict()
        
        self.assertEqual(len(result['applications']), len(graph_dict['applications']))


class TestGraphExporter(unittest.TestCase):
    """Tests for GraphExporter"""
    
    def setUp(self):
        config = GraphConfig(scale='tiny', scenario='generic', seed=42)
        graph_dict = GraphGenerator(config).generate()
        self.model = GraphBuilder().build_from_dict(graph_dict)
        self.exporter = GraphExporter()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_export_json(self):
        """Test JSON export"""
        path = Path(self.temp_dir) / "test.json"
        self.exporter.export_to_json(self.model, str(path))
        
        self.assertTrue(path.exists())
        with open(path) as f:
            data = json.load(f)
        self.assertIn('applications', data)
    
    def test_export_graphml(self):
        """Test GraphML export"""
        path = Path(self.temp_dir) / "test.graphml"
        self.exporter.export_to_graphml(self.model, str(path))
        
        self.assertTrue(path.exists())
        content = path.read_text()
        self.assertIn('graphml', content)
    
    def test_export_gexf(self):
        """Test GEXF export"""
        path = Path(self.temp_dir) / "test.gexf"
        self.exporter.export_to_gexf(self.model, str(path))
        
        self.assertTrue(path.exists())
        content = path.read_text()
        self.assertIn('gexf', content)
    
    def test_export_dot(self):
        """Test DOT export"""
        path = Path(self.temp_dir) / "test.dot"
        self.exporter.export_to_dot(self.model, str(path))
        
        self.assertTrue(path.exists())
        content = path.read_text()
        self.assertIn('digraph', content)
    
    def test_export_csv(self):
        """Test CSV export"""
        result = self.exporter.export_to_csv(self.model, self.temp_dir, prefix="test")
        
        self.assertIn('vertices', result)
        self.assertIn('edges', result)
        self.assertTrue(Path(result['vertices']).exists())
        self.assertTrue(Path(result['edges']).exists())


class TestQoSPolicy(unittest.TestCase):
    """Tests for QoSPolicy"""
    
    def test_default_qos(self):
        """Test default QoS values"""
        qos = QoSPolicy()
        self.assertEqual(qos.durability, 'VOLATILE')
        self.assertEqual(qos.reliability, 'BEST_EFFORT')
        self.assertEqual(qos.transport_priority, 'MEDIUM')
    
    def test_qos_from_dict(self):
        """Test creating QoS from dict"""
        data = {
            'durability': 'PERSISTENT',
            'reliability': 'RELIABLE',
            'transport_priority': 'URGENT'
        }
        qos = QoSPolicy.from_dict(data)
        
        self.assertEqual(qos.durability, 'PERSISTENT')
        self.assertEqual(qos.reliability, 'RELIABLE')
        self.assertEqual(qos.transport_priority, 'URGENT')
    
    def test_qos_to_dict(self):
        """Test converting QoS to dict"""
        qos = QoSPolicy(
            durability='PERSISTENT',
            reliability='RELIABLE',
            transport_priority='HIGH'
        )
        data = qos.to_dict()
        
        self.assertEqual(data['durability'], 'PERSISTENT')
        self.assertEqual(data['reliability'], 'RELIABLE')
        self.assertEqual(data['transport_priority'], 'HIGH')
    
    def test_criticality_score(self):
        """Test QoS criticality score"""
        high_qos = QoSPolicy(
            durability='PERSISTENT',
            reliability='RELIABLE',
            transport_priority='URGENT'
        )
        low_qos = QoSPolicy(
            durability='VOLATILE',
            reliability='BEST_EFFORT',
            transport_priority='LOW'
        )
        
        high_score = high_qos.get_criticality_score()
        low_score = low_qos.get_criticality_score()
        
        self.assertGreater(high_score, low_score)
        self.assertGreaterEqual(high_score, 0)
        self.assertLessEqual(high_score, 1)


class TestBuilderAdvanced(unittest.TestCase):
    """Tests for advanced GraphBuilder features"""
    
    def setUp(self):
        config = GraphConfig(scale='tiny', scenario='generic', seed=42)
        self.graph_dict = GraphGenerator(config).generate()
        self.builder = GraphBuilder()
        self.model = self.builder.build_from_dict(self.graph_dict)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_csv_roundtrip(self):
        """Test CSV export and import roundtrip"""
        exporter = GraphExporter()
        csv_files = exporter.export_to_csv(self.model, self.temp_dir, prefix='test')
        
        model2 = self.builder.build_from_csv(csv_files['vertices'], csv_files['edges'])
        
        self.assertEqual(len(self.model.applications), len(model2.applications))
        self.assertEqual(len(self.model.brokers), len(model2.brokers))
        self.assertEqual(len(self.model.topics), len(model2.topics))
        self.assertEqual(len(self.model.nodes), len(model2.nodes))
        self.assertEqual(len(self.model.edges), len(model2.edges))
    
    def test_graph_merge(self):
        """Test merging multiple graphs"""
        config2 = GraphConfig(scale='tiny', scenario='iot', seed=99)
        model2 = self.builder.build_from_dict(GraphGenerator(config2).generate())
        
        merged = self.builder.merge([self.model, model2], prefix_ids=True)
        
        expected_apps = len(self.model.applications) + len(model2.applications)
        self.assertEqual(len(merged.applications), expected_apps)
    
    def test_graph_filter(self):
        """Test filtering a graph"""
        # Filter to keep only applications
        filtered = self.builder.filter(
            self.model,
            vertex_filter=lambda v: hasattr(v, 'role')
        )
        
        self.assertEqual(len(filtered.applications), len(self.model.applications))
        self.assertEqual(len(filtered.brokers), 0)
        self.assertEqual(len(filtered.topics), 0)
    
    def test_subgraph_extraction(self):
        """Test extracting a subgraph"""
        # Get first 2 app IDs
        app_ids = set(list(self.model.applications.keys())[:2])
        
        subgraph = self.builder.subgraph(self.model, app_ids)
        
        self.assertEqual(len(subgraph.applications), 2)
    
    def test_validation_strict(self):
        """Test strict validation"""
        result = self.builder.validate(self.model, strict=True)
        self.assertIsNotNone(result)
        self.assertTrue(result.is_valid or len(result.errors) > 0)
    
    def test_validation_summary(self):
        """Test validation summary"""
        result = self.builder.validate(self.model)
        summary = result.summary()
        self.assertIn('Valid:', summary)
    
    def test_cypher_batch_generation(self):
        """Test batch Cypher generation"""
        batch = self.builder.generate_cypher_batch(self.model)
        
        self.assertIn('UNWIND', batch)
        self.assertIn('Application', batch)
        self.assertIn('Topic', batch)
    
    def test_build_summary(self):
        """Test build summary"""
        summary = self.builder.get_build_summary(self.model)
        
        self.assertIn('Applications:', summary)
        self.assertIn('Relationships by Type:', summary)


class TestGraphModelQueries(unittest.TestCase):
    """Tests for GraphModel query methods"""
    
    def setUp(self):
        config = GraphConfig(scale='small', scenario='iot', seed=42)
        graph_dict = GraphGenerator(config).generate()
        self.model = GraphBuilder().build_from_dict(graph_dict)
    
    def test_get_publishers_of(self):
        """Test getting publishers of a topic"""
        topic_id = list(self.model.topics.keys())[0]
        pubs = self.model.get_publishers_of(topic_id)
        
        self.assertIsInstance(pubs, list)
        for pub_id in pubs:
            self.assertIn(pub_id, self.model.applications)
    
    def test_get_subscribers_of(self):
        """Test getting subscribers of a topic"""
        topic_id = list(self.model.topics.keys())[0]
        subs = self.model.get_subscribers_of(topic_id)
        
        self.assertIsInstance(subs, list)
    
    def test_get_broker_of(self):
        """Test getting broker of a topic"""
        topic_id = list(self.model.topics.keys())[0]
        broker = self.model.get_broker_of(topic_id)
        
        if broker:
            self.assertIn(broker, self.model.brokers)
    
    def test_get_neighbors(self):
        """Test getting neighbors of a vertex"""
        app_id = list(self.model.applications.keys())[0]
        neighbors = self.model.get_neighbors(app_id)
        
        self.assertIsInstance(neighbors, set)
    
    def test_get_orphan_topics(self):
        """Test finding orphan topics"""
        orphans = self.model.get_orphan_topics()
        self.assertIsInstance(orphans, list)
    
    def test_get_isolated_apps(self):
        """Test finding isolated applications"""
        isolated = self.model.get_isolated_apps()
        self.assertIsInstance(isolated, list)
    
    def test_get_critical_topics(self):
        """Test finding critical topics"""
        critical = self.model.get_critical_topics(threshold=0.5)
        self.assertIsInstance(critical, list)
    
    def test_topic_fanout_fanin(self):
        """Test topic fanout and fanin"""
        topic_id = list(self.model.topics.keys())[0]
        
        fanout = self.model.get_topic_fanout(topic_id)
        fanin = self.model.get_topic_fanin(topic_id)
        
        self.assertIsInstance(fanout, int)
        self.assertIsInstance(fanin, int)
        self.assertGreaterEqual(fanout, 0)
        self.assertGreaterEqual(fanin, 0)


class TestPerformance(unittest.TestCase):
    """Performance tests"""
    
    def test_small_speed(self):
        """Test small graph generation speed"""
        config = GraphConfig(scale='small', scenario='generic', seed=42)
        
        start = time.time()
        GraphGenerator(config).generate()
        elapsed = time.time() - start
        
        self.assertLess(elapsed, 1.0)
    
    def test_medium_speed(self):
        """Test medium graph generation speed"""
        config = GraphConfig(scale='medium', scenario='generic', seed=42)
        
        start = time.time()
        GraphGenerator(config).generate()
        elapsed = time.time() - start
        
        self.assertLess(elapsed, 5.0)

class TestSchemaValidation(unittest.TestCase):
    """Tests for JSON schema validation"""
    
    def setUp(self):
        self.builder = GraphBuilder()
    
    def test_valid_schema(self):
        """Test validation of valid v3.0 data"""
        config = GraphConfig(scale='tiny', scenario='generic', seed=42)
        data = GraphGenerator(config).generate()
        
        result = self.builder.validate_schema(data)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.schema_errors), 0)
    
    def test_missing_required_keys(self):
        """Test detection of missing required keys"""
        data = {'applications': []}  # Missing other required keys
        
        result = self.builder.validate_schema(data)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.schema_errors), 0)
    
    def test_invalid_role(self):
        """Test detection of invalid application role"""
        data = {
            'applications': [{'id': 'A1', 'name': 'App1', 'role': 'invalid_role'}],
            'brokers': [],
            'topics': [],
            'nodes': [],
            'edges': {
                'publishes_to': [],
                'subscribes_to': [],
                'routes': [],
                'runs_on': [],
                'connects_to': []
            }
        }
        
        result = self.builder.validate_schema(data)
        self.assertFalse(result.is_valid)
        self.assertTrue(any('role' in e for e in result.schema_errors))
    
    def test_invalid_qos_values(self):
        """Test detection of invalid QoS values"""
        data = {
            'applications': [],
            'brokers': [],
            'topics': [{'id': 'T1', 'name': 'T1', 'qos': {'durability': 'INVALID'}}],
            'nodes': [],
            'edges': {
                'publishes_to': [],
                'subscribes_to': [],
                'routes': [],
                'runs_on': [],
                'connects_to': []
            }
        }
        
        result = self.builder.validate_schema(data)
        self.assertFalse(result.is_valid)
        self.assertTrue(any('durability' in e for e in result.schema_errors))
    
    def test_build_validated(self):
        """Test build_from_dict_validated method"""
        config = GraphConfig(scale='tiny', scenario='generic', seed=42)
        data = GraphGenerator(config).generate()
        
        model, result = self.builder.build_from_dict_validated(data)
        
        self.assertIsInstance(model, GraphModel)
        self.assertTrue(result.is_valid)


class TestGraphComparison(unittest.TestCase):
    """Tests for graph comparison"""
    
    def setUp(self):
        self.builder = GraphBuilder()
        # Create two similar graphs
        config = GraphConfig(scale='tiny', scenario='generic', seed=42)
        self.data1 = GraphGenerator(config).generate()
        self.model1 = self.builder.build_from_dict(self.data1)
    
    def test_identical_graphs(self):
        """Test comparison of identical graphs"""
        model2 = self.builder.clone(self.model1)
        
        diff = self.builder.compare(self.model1, model2)
        
        self.assertFalse(diff.has_changes)
    
    def test_added_vertex(self):
        """Test detection of added vertex"""
        model2 = self.builder.clone(self.model1)
        model2.add_application(Application(id='A_NEW', name='NewApp', role='pubsub'))
        
        diff = self.builder.compare(self.model1, model2)
        
        self.assertTrue(diff.has_changes)
        self.assertIn('A_NEW', diff.added_vertices['applications'])
    
    def test_removed_vertex(self):
        """Test detection of removed vertex"""
        model2 = self.builder.clone(self.model1)
        # Remove first application
        first_app_id = list(model2.applications.keys())[0]
        del model2.applications[first_app_id]
        
        diff = self.builder.compare(self.model1, model2)
        
        self.assertTrue(diff.has_changes)
        self.assertIn(first_app_id, diff.removed_vertices['applications'])
    
    def test_modified_vertex(self):
        """Test detection of modified vertex"""
        model2 = self.builder.clone(self.model1)
        # Modify first application
        first_app_id = list(model2.applications.keys())[0]
        model2.applications[first_app_id].role = 'pub'  # Change role
        
        diff = self.builder.compare(self.model1, model2)
        
        self.assertTrue(diff.has_changes)
        modified_ids = [vid for vid, _, _ in diff.modified_vertices['applications']]
        self.assertIn(first_app_id, modified_ids)
    
    def test_added_edge(self):
        """Test detection of added edge"""
        model2 = self.builder.clone(self.model1)
        
        # Add a new edge
        model2.add_node(Node(id='N_NEW', name='NewNode'))
        model2.add_edge(Edge(source='N_NEW', target=list(model2.nodes.keys())[0], edge_type='CONNECTS_TO'))
        
        diff = self.builder.compare(self.model1, model2)
        
        self.assertTrue(diff.has_changes)
        # New node added
        self.assertIn('N_NEW', diff.added_vertices['nodes'])
    
    def test_diff_summary(self):
        """Test diff summary generation"""
        model2 = self.builder.clone(self.model1)
        model2.add_broker(Broker(id='B_NEW', name='NewBroker'))
        
        diff = self.builder.compare(self.model1, model2)
        summary = diff.summary()
        
        self.assertIn('Brokers', summary)
        self.assertIn('+1', summary)


class TestCloneAndUtilities(unittest.TestCase):
    """Tests for utility methods"""
    
    def setUp(self):
        self.builder = GraphBuilder()
        config = GraphConfig(scale='tiny', scenario='generic', seed=42)
        self.data = GraphGenerator(config).generate()
        self.model = self.builder.build_from_dict(self.data)
    
    def test_clone(self):
        """Test graph cloning"""
        cloned = self.builder.clone(self.model)
        
        # Same structure
        self.assertEqual(len(cloned.applications), len(self.model.applications))
        self.assertEqual(len(cloned.brokers), len(self.model.brokers))
        self.assertEqual(len(cloned.topics), len(self.model.topics))
        self.assertEqual(len(cloned.nodes), len(self.model.nodes))
        self.assertEqual(len(cloned.edges), len(self.model.edges))
        
        # Independent (modifying clone doesn't affect original)
        cloned.add_application(Application(id='A_CLONE', name='CloneApp', role='pub'))
        self.assertNotIn('A_CLONE', self.model.applications)
    
    def test_get_schema(self):
        """Test schema retrieval"""
        schema = self.builder.get_schema()
        
        self.assertIn('type', schema)
        self.assertIn('properties', schema)
        self.assertIn('applications', schema['properties'])
    
    def test_validation_result_to_dict(self):
        """Test ValidationResult serialization"""
        result = self.builder.validate(self.model)
        result_dict = result.to_dict()
        
        self.assertIn('is_valid', result_dict)
        self.assertIn('errors', result_dict)
        self.assertIn('warnings', result_dict)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run graph generation tests')
    parser.add_argument('--quick', action='store_true', help='Quick tests only')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    verbosity = 2 if args.verbose else 1
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    if args.quick:
        suite.addTests(loader.loadTestsFromTestCase(TestGraphConfig))
        suite.addTests(loader.loadTestsFromTestCase(TestGraphGenerator))
    else:
        suite.addTests(loader.loadTestsFromTestCase(TestGraphConfig))
        suite.addTests(loader.loadTestsFromTestCase(TestGraphGenerator))
        suite.addTests(loader.loadTestsFromTestCase(TestAntipatterns))
        suite.addTests(loader.loadTestsFromTestCase(TestGraphBuilder))
        suite.addTests(loader.loadTestsFromTestCase(TestGraphExporter))
        suite.addTests(loader.loadTestsFromTestCase(TestQoSPolicy))
        suite.addTests(loader.loadTestsFromTestCase(TestBuilderAdvanced))
        suite.addTests(loader.loadTestsFromTestCase(TestGraphModelQueries))
        suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
        suite.addTests(loader.loadTestsFromTestCase(TestSchemaValidation))
        suite.addTests(loader.loadTestsFromTestCase(TestGraphComparison))
        suite.addTests(loader.loadTestsFromTestCase(TestCloneAndUtilities))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()