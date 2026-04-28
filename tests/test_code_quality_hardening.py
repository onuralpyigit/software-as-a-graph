import pytest
import networkx as nx
from saag.analysis.structural_analyzer import StructuralAnalyzer
from saag.core.metrics import StructuralMetrics
from saag.core.utils import serialization

def test_serialization_flat_aliases():
    """Verify that flatten_component correctly adds flat aliases for code metrics."""
    comp = {
        "id": "app1",
        "name": "App 1",
        "code_metrics": {
            "size": {"total_loc": 1000},
            "complexity": {"avg_wmc": 15.5},
            "cohesion": {"avg_lcom": 25.0},
            "coupling": {"avg_fanin": 5, "avg_fanout": 3}
        }
    }
    
    flat = serialization.flatten_component(comp, "Application")
    
    assert flat["loc"] == 1000
    assert flat["cyclomatic_complexity"] == 15.5
    assert flat["lcom"] == 25.0
    assert flat["coupling_afferent"] == 5
    assert flat["coupling_efferent"] == 3
    # Check that original cm_* fields are also still there
    assert flat["cm_total_loc"] == 1000
    assert flat["cm_avg_lcom"] == 25.0

def test_single_node_normalization_hardening():
    """Verify that a single-node population gets 1.0 instead of 0.0 (Hardening Issue 5)."""
    # Create a single library with some raw metrics
    m = StructuralMetrics(id="lib1", name="Lib 1", type="Library")
    m.loc_norm = 5000.0   # raw value
    m.complexity_norm = 45.0
    m.lcom_norm = 35.0
    m.coupling_afferent = 2
    m.coupling_efferent = 4
    m.instability_code = 4 / (2 + 4)
    
    components = {"lib1": m}
    
    # Run the normalization part of the analyzer
    # _compute_code_quality_metrics is a static method but can be called directly
    StructuralAnalyzer._compute_code_quality_metrics(components)
    
    # Since it's a population of 1, span is 0. 
    # v7 logic should return 1.0 (most critical) instead of 0.0.
    assert m.loc_norm == 1.0
    assert m.complexity_norm == 1.0
    assert m.lcom_norm == 1.0
    
    # Verify CQP v7 formula: 0.10*LOC + 0.35*CC + 0.30*INS + 0.25*LCOM
    # CQP = 0.10*1.0 + 0.35*1.0 + 0.30*(4/6) + 0.25*1.0
    expected_cqp = 0.10*1.0 + 0.35*1.0 + 0.30*(4/6) + 0.25*1.0
    assert m.code_quality_penalty == pytest.approx(expected_cqp)

def test_cqp_formula_weights_v7():
    """Verify the new CQP v7 weights with multiple nodes."""
    m1 = StructuralMetrics(id="app1", name="App 1", type="Application")
    m1.loc_norm = 100.0
    m1.complexity_norm = 10.0
    m1.lcom_norm = 5.0
    m1.instability_code = 0.5
    
    m2 = StructuralMetrics(id="app2", name="App 2", type="Application")
    m2.loc_norm = 200.0
    m2.complexity_norm = 20.0
    m2.lcom_norm = 10.0
    m2.instability_code = 0.5
    
    components = {"app1": m1, "app2": m2}
    StructuralAnalyzer._compute_code_quality_metrics(components)
    
    # app2 should be 1.0 in all normalized fields (it's the max)
    assert m2.loc_norm == 1.0
    assert m2.complexity_norm == 1.0
    assert m2.lcom_norm == 1.0
    
    # app1 should be 0.0 in all normalized fields (it's the min)
    assert m1.loc_norm == 0.0
    assert m1.complexity_norm == 0.0
    assert m1.lcom_norm == 0.0
    
    # Check app2 score: 0.10*1.0 + 0.35*1.0 + 0.30*0.5 + 0.25*1.0 = 0.1+0.35+0.15+0.25 = 0.85
    assert m2.code_quality_penalty == pytest.approx(0.85)
    # Check app1 score: 0.10*0.0 + 0.35*0.0 + 0.30*0.5 + 0.25*0.0 = 0.15
    assert m1.code_quality_penalty == pytest.approx(0.15)
