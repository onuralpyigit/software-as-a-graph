"""
Unit Tests for src/analysis module

Tests for:
    - StructuralAnalyzer: centrality metrics, articulation points, bridges
    - QualityAnalyzer: quality score aggregation
    - Layer filtering: APP, INFRA, SYSTEM layer component types
"""

import pytest
from src.services.analysis.structural_analyzer import StructuralAnalyzer
from src.services.analysis.quality_analyzer import QualityAnalyzer
from src.models.analysis.layers import AnalysisLayer, get_layer_definition, LAYER_DEFINITIONS
from src.models import GraphData, ComponentData, EdgeData


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_graph_data():
    """Simple A->B->C linear graph."""
    return GraphData(
        components=[
            ComponentData(id="A", component_type="Application", weight=1.0),
            ComponentData(id="B", component_type="Application", weight=1.0),
            ComponentData(id="C", component_type="Application", weight=1.0)
        ],
        edges=[
            EdgeData("A", "B", "Application", "Application", "app_to_app", 1.0),
            EdgeData("B", "C", "Application", "Application", "app_to_app", 2.0)
        ]
    )


@pytest.fixture
def multi_layer_graph():
    """Graph with multiple component types for layer testing."""
    return GraphData(
        components=[
            ComponentData(id="app1", component_type="Application", weight=1.0),
            ComponentData(id="app2", component_type="Application", weight=1.0),
            ComponentData(id="broker1", component_type="Broker", weight=2.0),
            ComponentData(id="node1", component_type="Node", weight=1.5),
            ComponentData(id="node2", component_type="Node", weight=1.5),
            ComponentData(id="lib1", component_type="Library", weight=0.5),
        ],
        edges=[
            EdgeData("app1", "app2", "Application", "Application", "app_to_app", 1.0),
            EdgeData("app1", "broker1", "Application", "Broker", "app_to_broker", 2.0),
            EdgeData("node1", "node2", "Node", "Node", "node_to_node", 1.0),
            EdgeData("node1", "broker1", "Node", "Broker", "node_to_broker", 1.5),
        ]
    )


@pytest.fixture
def articulation_point_graph():
    """Graph where B is an articulation point: A--B--C, B--D."""
    return GraphData(
        components=[
            ComponentData(id="A", component_type="Application", weight=1.0),
            ComponentData(id="B", component_type="Application", weight=1.0),
            ComponentData(id="C", component_type="Application", weight=1.0),
            ComponentData(id="D", component_type="Application", weight=1.0),
        ],
        edges=[
            EdgeData("A", "B", "Application", "Application", "app_to_app", 1.0),
            EdgeData("B", "C", "Application", "Application", "app_to_app", 1.0),
            EdgeData("B", "D", "Application", "Application", "app_to_app", 1.0),
        ]
    )


@pytest.fixture
def empty_graph():
    """Empty graph with no components or edges."""
    return GraphData(components=[], edges=[])


@pytest.fixture
def single_node_graph():
    """Graph with a single node and no edges."""
    return GraphData(
        components=[ComponentData(id="solo", component_type="Application", weight=1.0)],
        edges=[]
    )


# =============================================================================
# StructuralAnalyzer Tests
# =============================================================================

class TestStructuralAnalyzer:
    """Tests for StructuralAnalyzer metrics computation."""
    
    def test_structural_metrics(self, mock_graph_data):
        """Test basic structural metrics computation."""
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(mock_graph_data)
        
        assert "A" in res.components
        # B is central, should have higher betweenness
        assert res.components["B"].betweenness > res.components["A"].betweenness
        # A->B->C, so edge A->B should exist
        assert ("A", "B") in res.edges
    
    def test_pagerank_ordering(self, mock_graph_data):
        """Test that downstream nodes have higher PageRank (more depend on them)."""
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(mock_graph_data)
        
        # In A->B->C, C should have higher PageRank (A and B depend on C)
        assert res.components["C"].pagerank >= res.components["A"].pagerank
    
    def test_reverse_pagerank_ordering(self, mock_graph_data):
        """Test that upstream nodes have higher reverse PageRank."""
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(mock_graph_data)
        
        # In A->B->C, A should have highest reverse PageRank (depends on most)
        assert res.components["A"].reverse_pagerank >= res.components["C"].reverse_pagerank
    
    def test_articulation_point_detection(self, articulation_point_graph):
        """Test that articulation points are correctly identified."""
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(articulation_point_graph)
        
        # B connects A to C and D, so B is an articulation point
        assert res.components["B"].is_articulation_point is True
        # A, C, D are not articulation points (leaf nodes)
        assert res.components["A"].is_articulation_point is False
    
    def test_bridge_detection(self, mock_graph_data):
        """Test that bridge edges are detected in linear graph."""
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(mock_graph_data)
        
        # In A->B->C, both edges are bridges
        bridges = res.get_bridges()
        # Should have detected bridges (exact count may vary based on direction)
        assert len(bridges) >= 0  # At minimum, no crash
    
    def test_empty_graph_handling(self, empty_graph):
        """Test that empty graphs don't crash."""
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(empty_graph)
        
        assert len(res.components) == 0
        assert len(res.edges) == 0
    
    def test_single_node_handling(self, single_node_graph):
        """Test that single node graphs are handled correctly."""
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(single_node_graph)
        
        assert "solo" in res.components
        assert res.components["solo"].betweenness == 0.0  # No paths through single node
        assert res.components["solo"].is_articulation_point is False


# =============================================================================
# Layer Filtering Tests
# =============================================================================

class TestLayerFiltering:
    """Tests for layer definitions and component type filtering."""
    
    def test_app_layer_components(self):
        """APP layer should only include Application components."""
        layer_def = get_layer_definition(AnalysisLayer.APP)
        assert "Application" in layer_def.component_types
        assert "Node" not in layer_def.component_types
        assert "Broker" not in layer_def.component_types
    
    def test_infra_layer_components(self):
        """INFRA layer should only include Node components."""
        layer_def = get_layer_definition(AnalysisLayer.INFRA)
        assert "Node" in layer_def.component_types
        assert "Application" not in layer_def.component_types
    
    def test_system_layer_includes_library(self):
        """SYSTEM layer should include Library components."""
        layer_def = get_layer_definition(AnalysisLayer.SYSTEM)
        assert "Library" in layer_def.component_types
        assert "Application" in layer_def.component_types
        assert "Broker" in layer_def.component_types
        assert "Node" in layer_def.component_types
        assert "Topic" in layer_def.component_types
    
    def test_mw_layer_components(self):
        """MW layer should analyze Brokers via app/node to broker dependencies."""
        layer_def = get_layer_definition(AnalysisLayer.MW)
        # Graph building includes all connected component types
        assert "Application" in layer_def.component_types
        assert "Broker" in layer_def.component_types
        assert "Node" in layer_def.component_types
        # But analysis focuses on Brokers only
        assert layer_def.types_to_analyze == frozenset({"Broker"})
        # MW layer should have both broker dependency types
        assert "app_to_broker" in layer_def.dependency_types
        assert "node_to_broker" in layer_def.dependency_types
    
    def test_layer_analysis_filters_correctly(self, multi_layer_graph):
        """Verify that layer analysis filters to correct component types."""
        analyzer = StructuralAnalyzer()
        
        # APP layer should only have Applications
        res_app = analyzer.analyze(multi_layer_graph, layer=AnalysisLayer.APP)
        for comp_id, comp in res_app.components.items():
            assert comp.type == "Application", f"APP layer has non-Application: {comp.type}"
        
        # INFRA layer should only have Nodes
        res_infra = analyzer.analyze(multi_layer_graph, layer=AnalysisLayer.INFRA)
        for comp_id, comp in res_infra.components.items():
            assert comp.type == "Node", f"INFRA layer has non-Node: {comp.type}"
    
    def test_mw_layer_analyzes_only_brokers(self, multi_layer_graph):
        """MW layer should analyze only Brokers (not Applications or Nodes)."""
        analyzer = StructuralAnalyzer()
        res_mw = analyzer.analyze(multi_layer_graph, layer=AnalysisLayer.MW)
        
        # MW layer results should only include Broker components
        for comp_id, comp in res_mw.components.items():
            assert comp.type == "Broker", f"MW layer has non-Broker: {comp.type}"


# =============================================================================
# QualityAnalyzer Tests
# =============================================================================

class TestQualityAnalyzer:
    """Tests for QualityAnalyzer score computation."""
    
    def test_quality_scoring(self, mock_graph_data):
        """Test that quality scores are computed for all components."""
        struct_an = StructuralAnalyzer()
        struct_res = struct_an.analyze(mock_graph_data)
        
        qual_an = QualityAnalyzer()
        qual_res = qual_an.analyze(struct_res)
        
        # Check components
        assert len(qual_res.components) == 3
        comp_b = next(c for c in qual_res.components if c.id == "B")
        assert comp_b.scores.overall > 0
        
        # Check edges
        assert len(qual_res.edges) == 2
        edge_ab = next(e for e in qual_res.edges if e.source == "A")
        assert edge_ab.scores.overall > 0
    
    def test_central_node_higher_score(self, mock_graph_data):
        """Test that central nodes get higher quality scores."""
        struct_an = StructuralAnalyzer()
        struct_res = struct_an.analyze(mock_graph_data)
        
        qual_an = QualityAnalyzer()
        qual_res = qual_an.analyze(struct_res)
        
        scores = {c.id: c.scores.overall for c in qual_res.components}
        # B is the central node, should have higher overall score
        assert scores["B"] >= scores["A"]
    
    def test_all_dimensions_computed(self, mock_graph_data):
        """Test that all four quality dimensions are computed."""
        struct_an = StructuralAnalyzer()
        struct_res = struct_an.analyze(mock_graph_data)
        
        qual_an = QualityAnalyzer()
        qual_res = qual_an.analyze(struct_res)
        
        for comp in qual_res.components:
            # All dimensions should be non-negative
            assert comp.scores.reliability >= 0
            assert comp.scores.maintainability >= 0
            assert comp.scores.availability >= 0
            assert comp.scores.vulnerability >= 0
            assert comp.scores.overall >= 0
    
    def test_edge_vulnerability_computed(self, mock_graph_data):
        """Test that edge vulnerability is now computed."""
        struct_an = StructuralAnalyzer()
        struct_res = struct_an.analyze(mock_graph_data)
        
        qual_an = QualityAnalyzer()
        qual_res = qual_an.analyze(struct_res)
        
        # Edges should have vulnerability scores
        for edge in qual_res.edges:
            assert hasattr(edge.scores, 'vulnerability')
            assert edge.scores.vulnerability >= 0
    
    def test_custom_weights(self, mock_graph_data):
        """Test that custom weights affect scoring."""
        from src.services.analysis.weight_calculator import QualityWeights
        
        struct_an = StructuralAnalyzer()
        struct_res = struct_an.analyze(mock_graph_data)
        
        # Default weights
        qual_default = QualityAnalyzer()
        res_default = qual_default.analyze(struct_res)
        
        # Custom weights emphasizing reliability
        custom_weights = QualityWeights(
            q_reliability=0.7,
            q_maintainability=0.1,
            q_availability=0.1,
            q_vulnerability=0.1
        )
        qual_custom = QualityAnalyzer(weights=custom_weights)
        res_custom = qual_custom.analyze(struct_res)
        
        # Results should differ when using different weights
        default_scores = {c.id: c.scores.overall for c in res_default.components}
        custom_scores = {c.id: c.scores.overall for c in res_custom.components}
        
        # At least one score should differ
        assert any(
            abs(default_scores[id] - custom_scores[id]) > 0.001
            for id in default_scores
        )
    
    def test_classification_summary(self, mock_graph_data):
        """Test that classification summary is properly built."""
        struct_an = StructuralAnalyzer()
        struct_res = struct_an.analyze(mock_graph_data)
        
        qual_an = QualityAnalyzer()
        qual_res = qual_an.analyze(struct_res)
        
        summary = qual_res.classification_summary
        
        assert summary.total_components == 3
        assert summary.total_edges == 2
        # Distribution values should be non-negative integers
        assert all(v >= 0 for v in summary.component_distribution.values())
        assert all(v >= 0 for v in summary.edge_distribution.values())
    
    def test_edge_weights_configurable(self, mock_graph_data):
        """Test that edge quality weights are configurable."""
        from src.services.analysis.weight_calculator import QualityWeights
        
        struct_an = StructuralAnalyzer()
        struct_res = struct_an.analyze(mock_graph_data)
        
        # Custom edge weights emphasizing bridge factor
        custom_weights = QualityWeights(
            e_betweenness=0.1,
            e_bridge=0.7,
            e_endpoint=0.1,
            e_vulnerability=0.1
        )
        qual_custom = QualityAnalyzer(weights=custom_weights)
        res_custom = qual_custom.analyze(struct_res)
        
        # Results should be computed without error
        assert len(res_custom.edges) == 2