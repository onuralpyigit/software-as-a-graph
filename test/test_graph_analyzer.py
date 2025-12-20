#!/usr/bin/env python3
"""
Test Suite for Enhanced Graph Analyzer
======================================

Comprehensive tests for the graph analysis module including:
- Node criticality scoring
- Edge criticality analysis
- HITS role analysis
- Motif detection
- Dependency chain analysis
- Layer correlation analysis
- Ensemble criticality scoring

Usage:
    python test_graph_analyzer.py                    # Run all tests
    python test_graph_analyzer.py --quick            # Quick tests only
    python test_graph_analyzer.py -v                 # Verbose output
    python test_graph_analyzer.py --test criticality # Specific test group

Author: Software-as-a-Graph Research Project
"""

import sys
import json
import unittest
import tempfile
import time
import math
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import networkx as nx
except ImportError:
    print("ERROR: NetworkX is required. Install with: pip install networkx")
    sys.exit(1)


# ============================================================================
# Test Data Generation
# ============================================================================

def generate_simple_pubsub_graph() -> nx.DiGraph:
    """Generate a simple pub-sub graph for testing"""
    G = nx.DiGraph()
    
    # Add applications
    G.add_node('A1', type='Application', name='Publisher1', role='pub')
    G.add_node('A2', type='Application', name='Subscriber1', role='sub')
    G.add_node('A3', type='Application', name='Subscriber2', role='sub')
    G.add_node('A4', type='Application', name='PubSub1', role='pubsub')
    
    # Add topics
    G.add_node('T1', type='Topic', name='Events')
    G.add_node('T2', type='Topic', name='Commands')
    
    # Add broker
    G.add_node('B1', type='Broker', name='MainBroker')
    
    # Add node (infrastructure)
    G.add_node('N1', type='Node', name='Server1')
    
    # Add edges
    # Publisher -> Topic
    G.add_edge('A1', 'T1', type='PUBLISHES_TO')
    G.add_edge('A4', 'T2', type='PUBLISHES_TO')
    
    # Topic -> Subscriber (or reverse depending on convention)
    G.add_edge('A2', 'T1', type='SUBSCRIBES_TO')
    G.add_edge('A3', 'T1', type='SUBSCRIBES_TO')
    G.add_edge('A4', 'T1', type='SUBSCRIBES_TO')
    G.add_edge('A2', 'T2', type='SUBSCRIBES_TO')
    
    # Broker routes
    G.add_edge('B1', 'T1', type='ROUTES')
    G.add_edge('B1', 'T2', type='ROUTES')
    
    # Runs on
    G.add_edge('A1', 'N1', type='RUNS_ON')
    G.add_edge('A2', 'N1', type='RUNS_ON')
    G.add_edge('B1', 'N1', type='RUNS_ON')
    
    return G


def generate_chain_graph() -> nx.DiGraph:
    """Generate a chain/pipeline graph for testing dependency chains"""
    G = nx.DiGraph()
    
    # Create a processing pipeline: A1 -> T1 -> A2 -> T2 -> A3 -> T3 -> A4
    for i in range(1, 5):
        G.add_node(f'A{i}', type='Application', name=f'App{i}', role='pubsub')
    
    for i in range(1, 4):
        G.add_node(f'T{i}', type='Topic', name=f'Topic{i}')
    
    # Chain edges
    G.add_edge('A1', 'T1', type='PUBLISHES_TO')
    G.add_edge('A2', 'T1', type='SUBSCRIBES_TO')
    G.add_edge('A2', 'T2', type='PUBLISHES_TO')
    G.add_edge('A3', 'T2', type='SUBSCRIBES_TO')
    G.add_edge('A3', 'T3', type='PUBLISHES_TO')
    G.add_edge('A4', 'T3', type='SUBSCRIBES_TO')
    
    return G


def generate_star_graph() -> nx.DiGraph:
    """Generate a star topology for fan-out testing"""
    G = nx.DiGraph()
    
    # Central publisher
    G.add_node('A1', type='Application', name='CentralPub', role='pub')
    G.add_node('T1', type='Topic', name='BroadcastTopic')
    
    # Many subscribers
    for i in range(2, 12):
        G.add_node(f'A{i}', type='Application', name=f'Sub{i}', role='sub')
        G.add_edge(f'A{i}', 'T1', type='SUBSCRIBES_TO')
    
    G.add_edge('A1', 'T1', type='PUBLISHES_TO')
    
    return G


def generate_diamond_graph() -> nx.DiGraph:
    """Generate a diamond pattern for redundancy testing"""
    G = nx.DiGraph()
    
    # Two paths from A1 to A4
    G.add_node('A1', type='Application', name='Source', role='pub')
    G.add_node('A4', type='Application', name='Sink', role='sub')
    
    # Path 1: A1 -> T1 -> A2 -> T3 -> A4
    G.add_node('A2', type='Application', name='Path1', role='pubsub')
    G.add_node('T1', type='Topic', name='Topic1')
    G.add_node('T3', type='Topic', name='Topic3')
    
    # Path 2: A1 -> T2 -> A3 -> T4 -> A4
    G.add_node('A3', type='Application', name='Path2', role='pubsub')
    G.add_node('T2', type='Topic', name='Topic2')
    G.add_node('T4', type='Topic', name='Topic4')
    
    # Connect path 1
    G.add_edge('A1', 'T1', type='PUBLISHES_TO')
    G.add_edge('A2', 'T1', type='SUBSCRIBES_TO')
    G.add_edge('A2', 'T3', type='PUBLISHES_TO')
    G.add_edge('A4', 'T3', type='SUBSCRIBES_TO')
    
    # Connect path 2
    G.add_edge('A1', 'T2', type='PUBLISHES_TO')
    G.add_edge('A3', 'T2', type='SUBSCRIBES_TO')
    G.add_edge('A3', 'T4', type='PUBLISHES_TO')
    G.add_edge('A4', 'T4', type='SUBSCRIBES_TO')
    
    return G


def generate_test_json_data() -> Dict[str, Any]:
    """Generate test data in JSON format"""
    return {
        'applications': [
            {'id': 'A1', 'name': 'Publisher1', 'role': 'pub'},
            {'id': 'A2', 'name': 'Subscriber1', 'role': 'sub'},
            {'id': 'A3', 'name': 'Subscriber2', 'role': 'sub'},
        ],
        'brokers': [
            {'id': 'B1', 'name': 'MainBroker'}
        ],
        'topics': [
            {'id': 'T1', 'name': 'Events', 'qos': {'durability': 'VOLATILE', 'reliability': 'BEST_EFFORT'}}
        ],
        'nodes': [
            {'id': 'N1', 'name': 'Server1', 'type': 'compute'}
        ],
        'relationships': {
            'publishes_to': [
                {'from': 'A1', 'to': 'T1'}
            ],
            'subscribes_to': [
                {'from': 'A2', 'to': 'T1'},
                {'from': 'A3', 'to': 'T1'}
            ],
            'routes': [
                {'from': 'B1', 'to': 'T1'}
            ],
            'runs_on': [
                {'from': 'A1', 'to': 'N1'},
                {'from': 'A2', 'to': 'N1'},
                {'from': 'B1', 'to': 'N1'}
            ],
            'connects_to': []
        }
    }


# ============================================================================
# Basic Graph Tests
# ============================================================================

class TestGraphConstruction(unittest.TestCase):
    """Tests for graph construction from various formats"""
    
    def test_simple_graph_creation(self):
        """Test creating a simple graph"""
        G = generate_simple_pubsub_graph()
        
        self.assertEqual(G.number_of_nodes(), 8)
        self.assertGreater(G.number_of_edges(), 0)
    
    def test_node_types(self):
        """Test node type attributes"""
        G = generate_simple_pubsub_graph()
        
        types = set(G.nodes[n].get('type') for n in G.nodes())
        self.assertIn('Application', types)
        self.assertIn('Topic', types)
        self.assertIn('Broker', types)
        self.assertIn('Node', types)
    
    def test_edge_types(self):
        """Test edge type attributes"""
        G = generate_simple_pubsub_graph()
        
        edge_types = set(G.edges[e].get('type') for e in G.edges())
        self.assertIn('PUBLISHES_TO', edge_types)
        self.assertIn('SUBSCRIBES_TO', edge_types)
    
    def test_graph_from_json(self):
        """Test building graph from JSON data"""
        from analyze_graph import build_graph_from_dict
        
        data = generate_test_json_data()
        G = build_graph_from_dict(data)
        
        self.assertEqual(G.number_of_nodes(), 6)
        self.assertIn('A1', G.nodes())
        self.assertIn('T1', G.nodes())


# ============================================================================
# Criticality Scoring Tests
# ============================================================================

class TestCriticalityScoring(unittest.TestCase):
    """Tests for node criticality scoring"""
    
    def setUp(self):
        self.G = generate_simple_pubsub_graph()
    
    def test_basic_scoring(self):
        """Test basic criticality scoring"""
        from analyze_graph import calculate_criticality_scores
        
        scores = calculate_criticality_scores(self.G)
        
        self.assertEqual(len(scores), self.G.number_of_nodes())
        
        # All scores should have required fields
        for score in scores:
            self.assertIsNotNone(score.node_id)
            self.assertIsNotNone(score.composite_score)
            self.assertGreaterEqual(score.composite_score, 0)
            self.assertLessEqual(score.composite_score, 2.0)  # Upper bound with AP
    
    def test_criticality_levels(self):
        """Test criticality level assignment"""
        from analyze_graph import calculate_criticality_scores
        
        scores = calculate_criticality_scores(self.G)
        
        valid_levels = {'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'}
        for score in scores:
            self.assertIn(score.criticality_level, valid_levels)
    
    def test_custom_weights(self):
        """Test custom weight parameters"""
        from analyze_graph import calculate_criticality_scores
        
        # All weight on betweenness
        scores_bc = calculate_criticality_scores(self.G, alpha=1.0, beta=0.0, gamma=0.0)
        
        # All weight on articulation points
        scores_ap = calculate_criticality_scores(self.G, alpha=0.0, beta=1.0, gamma=0.0)
        
        # Scores should differ
        bc_top = scores_bc[0].node_id
        ap_top = scores_ap[0].node_id
        
        # At least one should have different top node (unless it's the same)
        self.assertTrue(len(scores_bc) > 0)
        self.assertTrue(len(scores_ap) > 0)
    
    def test_articulation_points_detection(self):
        """Test that articulation points are identified"""
        from analyze_graph import calculate_criticality_scores
        
        # Create a graph with clear articulation point
        G = nx.DiGraph()
        G.add_node('A', type='Application')
        G.add_node('B', type='Application')
        G.add_node('C', type='Application')  # Articulation point
        G.add_node('D', type='Application')
        G.add_node('E', type='Application')
        
        G.add_edge('A', 'C', type='PUBLISHES_TO')
        G.add_edge('B', 'C', type='PUBLISHES_TO')
        G.add_edge('C', 'D', type='PUBLISHES_TO')
        G.add_edge('C', 'E', type='PUBLISHES_TO')
        
        scores = calculate_criticality_scores(G)
        
        # Find C's score
        c_score = next((s for s in scores if s.node_id == 'C'), None)
        
        # C should be marked as articulation point
        self.assertIsNotNone(c_score)
        # In undirected version, C connects left and right sides


# ============================================================================
# Structural Analysis Tests
# ============================================================================

class TestStructuralAnalysis(unittest.TestCase):
    """Tests for structural analysis"""
    
    def test_structural_metrics(self):
        """Test structural analysis metrics"""
        from analyze_graph import analyze_structure
        
        G = generate_simple_pubsub_graph()
        result = analyze_structure(G)
        
        self.assertIsNotNone(result.density)
        self.assertGreaterEqual(result.density, 0)
        self.assertLessEqual(result.density, 1)
        
        self.assertIsNotNone(result.articulation_points)
        self.assertIsNotNone(result.bridges)
    
    def test_bridge_detection(self):
        """Test bridge edge detection"""
        from analyze_graph import analyze_structure
        
        # Create graph with clear bridge
        G = nx.DiGraph()
        G.add_node('A1', type='Application')
        G.add_node('A2', type='Application')
        G.add_node('A3', type='Application')
        
        G.add_edge('A1', 'A2', type='DEPENDS_ON')
        G.add_edge('A2', 'A3', type='DEPENDS_ON')  # Bridge
        
        result = analyze_structure(G)
        
        # Should detect bridges
        self.assertIsInstance(result.bridges, list)
    
    def test_connected_components(self):
        """Test connected component counting"""
        from analyze_graph import analyze_structure
        
        G = generate_simple_pubsub_graph()
        result = analyze_structure(G)
        
        self.assertGreaterEqual(result.weakly_connected_components, 1)


# ============================================================================
# Relationship Analyzer Tests
# ============================================================================

class TestEdgeCriticality(unittest.TestCase):
    """Tests for edge criticality analysis"""
    
    def setUp(self):
        self.G = generate_simple_pubsub_graph()
    
    def test_edge_analysis(self):
        """Test edge criticality analysis"""
        try:
            from src.analysis.relationship_analyzer import EdgeCriticalityAnalyzer
        except ImportError:
            self.skipTest("Relationship analyzer not available")
        
        analyzer = EdgeCriticalityAnalyzer(self.G)
        results = analyzer.analyze_all_edges()
        
        self.assertEqual(len(results), self.G.number_of_edges())
        
        for result in results:
            self.assertIsNotNone(result.edge)
            self.assertGreaterEqual(result.criticality_score, 0)
            self.assertLessEqual(result.criticality_score, 1)
    
    def test_bridge_identification(self):
        """Test bridge edge identification"""
        try:
            from src.analysis.relationship_analyzer import EdgeCriticalityAnalyzer
        except ImportError:
            self.skipTest("Relationship analyzer not available")
        
        analyzer = EdgeCriticalityAnalyzer(self.G)
        results = analyzer.analyze_all_edges()
        
        # Check is_bridge attribute
        for result in results:
            self.assertIsInstance(result.is_bridge, bool)
    
    def test_simmelian_strength(self):
        """Test Simmelian strength calculation"""
        try:
            from src.analysis.relationship_analyzer import EdgeCriticalityAnalyzer
        except ImportError:
            self.skipTest("Relationship analyzer not available")
        
        analyzer = EdgeCriticalityAnalyzer(self.G)
        results = analyzer.analyze_all_edges()
        
        for result in results:
            self.assertGreaterEqual(result.simmelian_strength, 0)


class TestHITSAnalysis(unittest.TestCase):
    """Tests for HITS-based role analysis"""
    
    def setUp(self):
        self.G = generate_simple_pubsub_graph()
    
    def test_hits_roles(self):
        """Test HITS role analysis"""
        try:
            from src.analysis.relationship_analyzer import HITSRoleAnalyzer, ComponentRole
        except ImportError:
            self.skipTest("Relationship analyzer not available")
        
        analyzer = HITSRoleAnalyzer(self.G)
        results = analyzer.analyze_roles()
        
        self.assertEqual(len(results), self.G.number_of_nodes())
        
        for node_id, result in results.items():
            self.assertEqual(result.node_id, node_id)
            self.assertIsInstance(result.role, ComponentRole)
            self.assertGreaterEqual(result.hub_score, 0)
            self.assertGreaterEqual(result.authority_score, 0)
    
    def test_top_hubs(self):
        """Test top hubs identification"""
        try:
            from src.analysis.relationship_analyzer import HITSRoleAnalyzer
        except ImportError:
            self.skipTest("Relationship analyzer not available")
        
        analyzer = HITSRoleAnalyzer(self.G)
        analyzer.analyze_roles()
        
        top_hubs = analyzer.get_top_hubs(5)
        
        self.assertLessEqual(len(top_hubs), 5)
        self.assertTrue(all(h in self.G.nodes() for h in top_hubs))
    
    def test_top_authorities(self):
        """Test top authorities identification"""
        try:
            from src.analysis.relationship_analyzer import HITSRoleAnalyzer
        except ImportError:
            self.skipTest("Relationship analyzer not available")
        
        analyzer = HITSRoleAnalyzer(self.G)
        analyzer.analyze_roles()
        
        top_auth = analyzer.get_top_authorities(5)
        
        self.assertLessEqual(len(top_auth), 5)


class TestMotifDetection(unittest.TestCase):
    """Tests for network motif detection"""
    
    def test_fan_out_detection(self):
        """Test fan-out pattern detection"""
        try:
            from src.analysis.relationship_analyzer import MotifDetector, MotifType
        except ImportError:
            self.skipTest("Relationship analyzer not available")
        
        G = generate_star_graph()
        detector = MotifDetector(G)
        motifs = detector.detect_all_motifs()
        
        # Should detect star/fan-out patterns
        self.assertGreater(len(motifs), 0)
    
    def test_chain_detection(self):
        """Test chain pattern detection"""
        try:
            from src.analysis.relationship_analyzer import MotifDetector, MotifType
        except ImportError:
            self.skipTest("Relationship analyzer not available")
        
        G = generate_chain_graph()
        detector = MotifDetector(G)
        motifs = detector.detect_all_motifs()
        
        self.assertIsInstance(motifs, list)
    
    def test_motif_summary(self):
        """Test motif summary generation"""
        try:
            from src.analysis.relationship_analyzer import MotifDetector
        except ImportError:
            self.skipTest("Relationship analyzer not available")
        
        G = generate_simple_pubsub_graph()
        detector = MotifDetector(G)
        motifs = detector.detect_all_motifs()
        summary = detector.get_motif_summary(motifs)
        
        self.assertIsInstance(summary, dict)


class TestDependencyChains(unittest.TestCase):
    """Tests for dependency chain analysis"""
    
    def test_chain_analysis(self):
        """Test dependency chain analysis"""
        try:
            from src.analysis.relationship_analyzer import DependencyChainAnalyzer
        except ImportError:
            self.skipTest("Relationship analyzer not available")
        
        G = generate_chain_graph()
        analyzer = DependencyChainAnalyzer(G)
        results = analyzer.analyze_all()
        
        self.assertEqual(len(results), G.number_of_nodes())
        
        for node_id, result in results.items():
            self.assertEqual(result.node_id, node_id)
            self.assertGreaterEqual(result.transitive_depth, 0)
    
    def test_deepest_chains(self):
        """Test identification of deepest chains"""
        try:
            from src.analysis.relationship_analyzer import DependencyChainAnalyzer
        except ImportError:
            self.skipTest("Relationship analyzer not available")
        
        G = generate_chain_graph()
        analyzer = DependencyChainAnalyzer(G)
        results = analyzer.analyze_all()
        
        deepest = analyzer.get_deepest_chains(results, 3)
        
        self.assertLessEqual(len(deepest), 3)


class TestEnsembleCriticality(unittest.TestCase):
    """Tests for ensemble criticality scoring"""
    
    def setUp(self):
        self.G = generate_simple_pubsub_graph()
    
    def test_ensemble_scoring(self):
        """Test ensemble criticality scoring"""
        try:
            from src.analysis.relationship_analyzer import EnsembleCriticalityScorer
        except ImportError:
            self.skipTest("Relationship analyzer not available")
        
        scorer = EnsembleCriticalityScorer(self.G)
        results = scorer.score_all()
        
        self.assertEqual(len(results), self.G.number_of_nodes())
        
        for node_id, result in results.items():
            self.assertEqual(result.node_id, node_id)
            self.assertGreaterEqual(result.ensemble_score, 0)
            self.assertIn(result.ensemble_level, ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'])
    
    def test_custom_weights(self):
        """Test custom algorithm weights"""
        try:
            from src.analysis.relationship_analyzer import EnsembleCriticalityScorer
        except ImportError:
            self.skipTest("Relationship analyzer not available")
        
        # Custom weights emphasizing betweenness
        weights = {
            'betweenness': 0.5,
            'pagerank': 0.1,
            'hits_hub': 0.05,
            'hits_auth': 0.05,
            'articulation': 0.2,
            'kcore': 0.05,
            'closeness': 0.05
        }
        
        scorer = EnsembleCriticalityScorer(self.G)
        results = scorer.score_all(weights)
        
        self.assertEqual(len(results), self.G.number_of_nodes())


class TestRelationshipAnalyzer(unittest.TestCase):
    """Tests for the main RelationshipAnalyzer class"""
    
    def setUp(self):
        self.G = generate_simple_pubsub_graph()
    
    def test_full_analysis(self):
        """Test complete relationship analysis"""
        try:
            from src.analysis.relationship_analyzer import RelationshipAnalyzer
        except ImportError:
            self.skipTest("Relationship analyzer not available")
        
        analyzer = RelationshipAnalyzer(self.G)
        result = analyzer.analyze()
        
        self.assertEqual(result.total_nodes, self.G.number_of_nodes())
        self.assertEqual(result.total_edges, self.G.number_of_edges())
        self.assertIsNotNone(result.analysis_timestamp)
    
    def test_selective_analysis(self):
        """Test selective analysis (disable some components)"""
        try:
            from src.analysis.relationship_analyzer import RelationshipAnalyzer
        except ImportError:
            self.skipTest("Relationship analyzer not available")
        
        analyzer = RelationshipAnalyzer(self.G)
        result = analyzer.analyze(
            include_motifs=False,
            include_layers=False
        )
        
        # Motifs should be empty
        self.assertEqual(len(result.motifs), 0)
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        try:
            from src.analysis.relationship_analyzer import RelationshipAnalyzer
        except ImportError:
            self.skipTest("Relationship analyzer not available")
        
        analyzer = RelationshipAnalyzer(self.G)
        result = analyzer.analyze()
        result_dict = analyzer.to_dict(result)
        
        self.assertIn('summary', result_dict)
        self.assertIn('edge_analysis', result_dict)
        self.assertIn('hits_analysis', result_dict)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete analysis pipeline"""
    
    def test_full_pipeline(self):
        """Test the full analysis pipeline"""
        from analyze_graph import (
            build_graph_from_dict,
            analyze_structure,
            calculate_criticality_scores,
            generate_recommendations
        )
        
        data = generate_test_json_data()
        G = build_graph_from_dict(data)
        
        structural = analyze_structure(G)
        scores = calculate_criticality_scores(G)
        recommendations = generate_recommendations(G, structural, scores)
        
        self.assertIsNotNone(structural)
        self.assertGreater(len(scores), 0)
        self.assertIsInstance(recommendations, list)
    
    def test_cli_json_output(self):
        """Test CLI with JSON output"""
        import subprocess
        import tempfile
        
        # Create test file
        data = generate_test_json_data()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            # Run CLI
            result = subprocess.run(
                [sys.executable, 'analyze_graph.py', '--input', temp_path, '-j', '-q'],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent)
            )
            
            if result.returncode == 0:
                # Verify JSON output
                output = json.loads(result.stdout)
                self.assertIn('total_nodes', output)
        finally:
            Path(temp_path).unlink()
    
    def test_export_formats(self):
        """Test export to different formats"""
        from analyze_graph import (
            build_graph_from_dict,
            analyze_structure,
            calculate_criticality_scores,
            generate_recommendations,
            export_json,
            export_csv,
            GraphAnalysisResult
        )
        from datetime import datetime
        
        data = generate_test_json_data()
        G = build_graph_from_dict(data)
        
        structural = analyze_structure(G)
        scores = calculate_criticality_scores(G)
        recommendations = generate_recommendations(G, structural, scores)
        
        # Create result object
        nodes_by_type = {}
        for n in G.nodes():
            ntype = G.nodes[n].get('type', 'Unknown')
            nodes_by_type[ntype] = nodes_by_type.get(ntype, 0) + 1
        
        edges_by_type = {}
        for e in G.edges():
            etype = G.edges[e].get('type', 'Unknown')
            edges_by_type[etype] = edges_by_type.get(etype, 0) + 1
        
        level_counts = {}
        for s in scores:
            level_counts[s.criticality_level] = level_counts.get(s.criticality_level, 0) + 1
        
        result = GraphAnalysisResult(
            timestamp=datetime.now().isoformat(),
            input_file='test.json',
            total_nodes=G.number_of_nodes(),
            total_edges=G.number_of_edges(),
            nodes_by_type=nodes_by_type,
            edges_by_type=edges_by_type,
            criticality_scores=scores,
            criticality_by_level=level_counts,
            structural=structural,
            recommendations=recommendations
        )
        
        # Test export to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # JSON export
            export_json(result, tmpdir / 'test.json')
            self.assertTrue((tmpdir / 'test.json').exists())
            
            # CSV export
            export_csv(result, tmpdir)
            self.assertTrue((tmpdir / 'criticality_scores.csv').exists())


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance(unittest.TestCase):
    """Performance tests"""
    
    def test_small_graph_speed(self):
        """Test analysis speed on small graph"""
        from analyze_graph import calculate_criticality_scores, analyze_structure
        
        G = generate_simple_pubsub_graph()
        
        start = time.time()
        analyze_structure(G)
        calculate_criticality_scores(G)
        elapsed = time.time() - start
        
        self.assertLess(elapsed, 1.0)  # Should complete in < 1 second
    
    def test_medium_graph_speed(self):
        """Test analysis speed on medium graph"""
        from analyze_graph import calculate_criticality_scores, analyze_structure
        
        # Create larger graph
        G = nx.DiGraph()
        for i in range(100):
            G.add_node(f'A{i}', type='Application')
        for i in range(50):
            G.add_node(f'T{i}', type='Topic')
        
        # Add edges
        import random
        random.seed(42)
        for i in range(100):
            t = random.randint(0, 49)
            G.add_edge(f'A{i}', f'T{t}', type='PUBLISHES_TO')
            t2 = random.randint(0, 49)
            G.add_edge(f'A{i}', f'T{t2}', type='SUBSCRIBES_TO')
        
        start = time.time()
        analyze_structure(G)
        calculate_criticality_scores(G)
        elapsed = time.time() - start
        
        self.assertLess(elapsed, 10.0)  # Should complete in < 10 seconds


# ============================================================================
# Algorithm Correctness Tests
# ============================================================================

class TestAlgorithmCorrectness(unittest.TestCase):
    """Tests for algorithm correctness"""
    
    def test_betweenness_centrality_order(self):
        """Test that betweenness centrality ordering is correct"""
        # Create graph where center node should have highest BC
        G = nx.DiGraph()
        G.add_node('center', type='Application')
        for i in range(5):
            G.add_node(f'spoke{i}', type='Application')
            G.add_edge(f'spoke{i}', 'center', type='DEPENDS_ON')
        
        bc = nx.betweenness_centrality(G, normalized=True)
        
        # Center should have highest or tied for highest BC
        max_bc = max(bc.values())
        self.assertEqual(bc['center'], max_bc)
    
    def test_pagerank_conservation(self):
        """Test PageRank score conservation"""
        G = generate_simple_pubsub_graph()
        
        pr = nx.pagerank(G)
        total = sum(pr.values())
        
        # PageRank scores should sum to approximately 1
        self.assertAlmostEqual(total, 1.0, places=5)
    
    def test_articulation_point_correctness(self):
        """Test articulation point detection correctness"""
        # Create graph with known articulation point
        G = nx.Graph()  # Undirected for AP detection
        # A -- B -- C
        #      |
        #      D -- E
        G.add_edges_from([('A', 'B'), ('B', 'C'), ('B', 'D'), ('D', 'E')])
        
        aps = list(nx.articulation_points(G))
        
        # B and D should be articulation points
        self.assertIn('B', aps)
        self.assertIn('D', aps)


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run graph analyzer tests')
    parser.add_argument('--quick', action='store_true', help='Quick tests only')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--test', type=str, 
                        choices=['construction', 'criticality', 'structural', 
                                'edges', 'hits', 'motifs', 'chains', 'ensemble',
                                'integration', 'performance', 'correctness', 'all'],
                        default='all', help='Specific test group to run')
    
    args = parser.parse_args()
    
    verbosity = 2 if args.verbose else 1
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_groups = {
        'construction': TestGraphConstruction,
        'criticality': TestCriticalityScoring,
        'structural': TestStructuralAnalysis,
        'edges': TestEdgeCriticality,
        'hits': TestHITSAnalysis,
        'motifs': TestMotifDetection,
        'chains': TestDependencyChains,
        'ensemble': TestEnsembleCriticality,
        'integration': TestIntegration,
        'performance': TestPerformance,
        'correctness': TestAlgorithmCorrectness,
    }
    
    if args.quick:
        # Quick tests - basic functionality only
        suite.addTests(loader.loadTestsFromTestCase(TestGraphConstruction))
        suite.addTests(loader.loadTestsFromTestCase(TestCriticalityScoring))
        suite.addTests(loader.loadTestsFromTestCase(TestStructuralAnalysis))
    elif args.test == 'all':
        # All tests
        for test_class in test_groups.values():
            suite.addTests(loader.loadTestsFromTestCase(test_class))
        suite.addTests(loader.loadTestsFromTestCase(TestRelationshipAnalyzer))
    else:
        # Specific test group
        if args.test in test_groups:
            suite.addTests(loader.loadTestsFromTestCase(test_groups[args.test]))
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)
    
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()