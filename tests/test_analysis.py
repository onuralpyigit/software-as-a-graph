import pytest
from src.analysis.structural_analyzer import StructuralAnalyzer
from src.analysis.quality_analyzer import QualityAnalyzer
from src.core.graph_exporter import GraphData, ComponentData, EdgeData

@pytest.fixture
def mock_graph_data():
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

def test_structural_metrics(mock_graph_data):
    analyzer = StructuralAnalyzer()
    res = analyzer.analyze(mock_graph_data)
    
    assert "A" in res.components
    # B is central, should have higher betweenness
    assert res.components["B"].betweenness > res.components["A"].betweenness
    # A->B->C, so edge A->B should exist
    assert ("A", "B") in res.edges

def test_quality_scoring(mock_graph_data):
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