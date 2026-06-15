import pytest
from types import SimpleNamespace
from saag.core.criticality import CriticalityRanking, CompatNamespace
from saag.models import ComponentFacade, PredictionResult, AnalysisResult

def test_compat_namespace():
    # Test attribute and dict-like key access
    ns = CompatNamespace(foo="bar", num=42)
    
    assert ns.foo == "bar"
    assert ns["foo"] == "bar"
    assert ns.num == 42
    assert ns["num"] == 42
    
    # Missing attributes/keys
    with pytest.raises(AttributeError):
        _ = ns.nonexistent
    with pytest.raises(AttributeError):
        _ = ns["nonexistent"]
    assert ns.get("nonexistent") is None
    
    # Str representation
    assert "foo" in str(ns)

def test_criticality_ranking_dto():
    ranking = CriticalityRanking(
        id="c1",
        type="Application",
        scores={"overall": 0.8},
        levels={"overall": "CRITICAL"},
        overall=0.8,
        level="CRITICAL",
        provenance="gnn",
        name="Test App",
        blast_radius=5,
        cascade_depth=3,
        is_articulation_point=True
    )
    
    assert ranking.id == "c1"
    assert ranking.overall == 0.8
    assert ranking.is_articulation_point is True

def test_component_facade_mapping():
    ranking = CriticalityRanking(
        id="c1",
        type="Application",
        scores={"overall": 0.8, "reliability": 0.7},
        levels={"overall": "CRITICAL", "reliability": "HIGH"},
        overall=0.8,
        level="CRITICAL",
        provenance="gnn",
        name="Test App",
        blast_radius=5,
        cascade_depth=3,
        is_articulation_point=True
    )
    
    facade = ComponentFacade(ranking)
    
    assert facade.id == "c1"
    assert facade.name == "Test App"
    assert facade.type == "Application"
    assert facade.rmav_score == 0.8
    assert facade.is_critical is True
    assert facade.blast_radius == 5
    assert facade.cascade_depth == 3
    assert facade.criticality_level == "CRITICAL"
    assert facade.criticality_levels == {"overall": "CRITICAL", "reliability": "HIGH"}
    assert facade.scores == {"overall": 0.8, "reliability": 0.7}
    
    # Structural mock
    struct = facade.structural
    assert struct.name == "Test App"
    assert struct.is_articulation_point is True
    assert struct.blast_radius == 5
    assert struct.cascade_depth == 3
    
    # to_dict
    d = facade.to_dict()
    assert d["id"] == "c1"
    assert d["rmav_score"] == 0.8
    assert d["is_critical"] is True

def test_prediction_result_from_dict():
    raw_dict = {
        "prediction_mode": "gnn_only",
        "node_scores": {
            "c1": {
                "component": "c1",
                "composite_score": 0.85,
                "reliability_score": 0.75,
                "maintainability_score": 0.65,
                "availability_score": 0.55,
                "security_score": 0.45,
                "criticality_level": "CRITICAL"
            }
        },
        "edge_scores": [
            {
                "source": "c1",
                "target": "c2",
                "edge_type": "DEPENDS_ON",
                "criticality_level": "high",
                "composite_score": 0.6
            }
        ],
        "_structural_cache": {
            "c1": {
                "name": "App One",
                "type": "Application",
                "blast_radius": 4,
                "cascade_depth": 2,
                "is_articulation_point": True
            }
        }
    }
    
    pred = PredictionResult(raw_dict)
    assert pred.raw.prediction_mode == "gnn_only"
    
    comps = pred.all_components
    assert len(comps) == 1
    c = comps[0]
    assert c.id == "c1"
    assert c.name == "App One"
    assert c.type == "Application"
    assert c.rmav_score == 0.85
    assert c.criticality_level == "CRITICAL"
    assert c.blast_radius == 4
    assert c.cascade_depth == 2
    assert c.structural.is_articulation_point is True
    
    assert len(pred.edges) == 1
    edge = pred.edges[0]
    assert edge.source == "c1"
    assert edge.target == "c2"
