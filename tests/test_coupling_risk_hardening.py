import pytest
from saag.prediction.analyzer import QualityAnalyzer
from saag.core.metrics import StructuralMetrics
from saag.core.models import COUPLING_PATH_DELTA

def test_coupling_risk_capping():
    """Verify that CouplingRisk is correctly modulated and capped at 1.0."""
    # Create a balanced node (max base CouplingRisk)
    m = StructuralMetrics(id="app1", name="App 1", type="Application")
    m.in_degree_raw = 5
    m.out_degree_raw = 5
    # path_complexity significant enough to push CR > 1.0
    # Formula: CR = (1 - |2*(5/10)-1|) * (1 + 0.10 * 2.0) = 1.0 * 1.2 = 1.2
    m.path_complexity = 2.0
    
    analyzer = QualityAnalyzer()
    # We need to simulate the _compute_rmav scoring or test the logic
    # Since _compute_rmav is complex, we can test the specific logic if extracted,
    # or just run a full scoring pass.
    
    # Let's check how to invoke scoring for a single node
    # QualityAnalyzer.predict(structural_results)
    
    from saag.analysis.models import LayerAnalysisResult, StructuralAnalysisResult, GraphSummary
    from saag.core.layers import AnalysisLayer
    
    # Mock analysis result
    sar = StructuralAnalysisResult(
        layer=AnalysisLayer.SYSTEM,
        components={"app1": m},
        edges={},
        graph_summary=GraphSummary(layer="system", nodes=1, edges=10)
    )
    
    # Predict
    # analyzer.analyze expects StructuralAnalysisResult as the first argument
    results = analyzer.analyze(sar)
    
    # Check CouplingRisk in the output
    # Results.components is a List[ComponentQuality]
    score = next(c for c in results.components if c.id == "app1")
    
    # The ComponentQuality has dimension scores and maybe raw derived terms?
    # Actually, CouplingRisk is not typically exported in the output JSON, 
    # it's an internal scalar. But it affects the Maintainability score.
    
    # Wait, in analyzer.py, M score is calculated using coupling_risk.
    # We can verify M score and compare it with the expected capped vs uncapped value.
    
    # Expected M if capped at 1.0:
    # M = 0.35*BT + 0.30*w_out_n + 0.15*cqp + 0.12*1.0 + 0.08*(1-cc)
    # Metrics are all 0 by default. CC=0.0 -> (1-CC)=1.0
    # M = 0.12 * 1.0 + 0.08 = 0.20
    assert score.scores.maintainability == pytest.approx(0.20)

def test_coupling_risk_modulation():
    """Verify that path_complexity actually increases CouplingRisk."""
    # Instability = 0.2 -> Base CR = 1 - |2*0.2 - 1| = 1 - 0.6 = 0.4
    # PC = 1.0 -> CR = 0.4 * (1 + 0.1*1.0) = 0.44
    
    m3 = StructuralMetrics(id="app3", name="App 3", type="Application")
    m3.in_degree_raw = 8
    m3.out_degree_raw = 2
    m3.path_complexity = 1.0 # 0.4 * 1.1 = 0.44
    
    from saag.analysis.models import StructuralAnalysisResult, GraphSummary
    from saag.core.layers import AnalysisLayer
    
    sar = StructuralAnalysisResult(
        layer=AnalysisLayer.SYSTEM,
        components={"app3": m3},
        edges={},
        graph_summary=GraphSummary(layer="system", nodes=1, edges=10)
    )
    
    # Predict
    analyzer = QualityAnalyzer()
    results = analyzer.analyze(sar)
    score = next(c for c in results.components if c.id == "app3")
    
    # M = 0.12 * 0.44 + 0.08 = 0.0528 + 0.08 = 0.1328
    assert score.scores.maintainability == pytest.approx(0.1328)
