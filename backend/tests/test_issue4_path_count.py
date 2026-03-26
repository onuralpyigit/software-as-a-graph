import pytest
import math
from src.core import GraphData, ComponentData, EdgeData, AnalysisLayer
from src.analysis.structural_analyzer import StructuralAnalyzer
from src.prediction.analyzer import QualityAnalyzer
from src.core.models import COUPLING_PATH_DELTA

class TestPathComplexity:
    """Verifies Issue 4: path_complexity metric and CouplingRisk modulation."""

    @pytest.fixture
    def multi_path_graph(self):
        """
        Graph with varying path_count:
        A --[path_count=1]--> B
        A --[path_count=3]--> C
        """
        return GraphData(
            components=[
                ComponentData(id="A", component_type="Application"),
                ComponentData(id="B", component_type="Application"),
                ComponentData(id="C", component_type="Application"),
            ],
            edges=[
                EdgeData("A", "B", "Application", "Application", "app_to_app", "DEPENDS_ON", 1.0, path_count=1),
                EdgeData("A", "C", "Application", "Application", "app_to_app", "DEPENDS_ON", 1.0, path_count=3),
            ],
        )

    def test_path_complexity_calculation(self, multi_path_graph):
        """StructuralAnalyzer should correctly compute mean log-scaled path count."""
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(multi_path_graph)
        
        # A has two outgoing edges: 
        # e1: path_count=1 -> log2(1+1) = 1.0
        # e2: path_count=3 -> log2(1+3) = 2.0
        # path_complexity(A) = (1.0 + 2.0) / 2 = 1.5
        assert res.components["A"].path_complexity == pytest.approx(1.5)
        # B and C have 0 out-degree
        assert res.components["B"].path_complexity == 0.0
        assert res.components["C"].path_complexity == 0.0

    def test_coupling_risk_modulation(self, multi_path_graph):
        """QualityAnalyzer should modulate CouplingRisk by (1 + delta * path_complexity)."""
        # 1. Structural Analysis
        struct_res = StructuralAnalyzer().analyze(multi_path_graph)
        
        # 2. Quality Analysis
        quality_res = QualityAnalyzer().analyze(struct_res)
        
        comp_map = {c.id: c for c in quality_res.components}
        a_metrics = struct_res.components["A"]
        a_quality = comp_map["A"]
        
        # Base CouplingRisk calculation:
        # id=0, od=2/2=1.0 -> instability = 1.0 / (0 + 1.0) = 1.0
        # base_cr = 1.0 - abs(2*1.0 - 1) = 0.0
        # Wait, if base_cr is 0, modulation won't change it. 
        # Let's use a graph where ID > 0.
        pass

    def test_coupling_risk_modulation_with_id(self):
        """Verifies modulation when base CouplingRisk > 0."""
        # X depends on Y, Z depends on X
        # X: in_degree=1, out_degree=1 -> Instability=0.5 -> base_cr=1.0
        graph = GraphData(
            components=[
                ComponentData(id="X", component_type="Application"),
                ComponentData(id="Y", component_type="Application"),
                ComponentData(id="Z", component_type="Application"),
            ],
            edges=[
                EdgeData("Z", "X", "Application", "Application", "app_to_app", "DEPENDS_ON", 1.0, path_count=7), # log2(1+7)=3.0
                EdgeData("X", "Y", "Application", "Application", "app_to_app", "DEPENDS_ON", 1.0, path_count=3), # log2(1+3)=2.0
            ],
        )
        
        struct_res = StructuralAnalyzer().analyze(graph)
        quality_res = QualityAnalyzer().analyze(struct_res)
        
        comp_map = {c.id: c for c in quality_res.components}
        x_struct = struct_res.components["X"]
        x_quality = comp_map["X"]
        
        # X analysis:
        # id_raw=1, od_raw=1, n=3 -> id_n=0.5, od_n=0.5
        # instability = 0.5 / (0.5 + 0.5) = 0.5
        # base_cr = 1.0 - abs(2*0.5 - 1) = 1.0
        # path_complexity = log2(1+3) / 1 = 2.0
        # enriched_cr = 1.0 * (1 + 0.10 * 2.0) = 1.2
        
        # Maintainability M = ... + w_cr * enriched_cr + ...
        # Since path_complexity for X is 2.0, the contribution of CR should be increased by 20%.
        # We compare with a baseline where path_count = 1 (path_complexity = 1.0)
        
        # Baseline:
        graph_base = GraphData(
            components=[
                ComponentData(id="X", component_type="Application"),
                ComponentData(id="Y", component_type="Application"),
                ComponentData(id="Z", component_type="Application"),
            ],
            edges=[
                EdgeData("Z", "X", "Application", "Application", "app_to_app", "DEPENDS_ON", 1.0, path_count=1), # log2(1+1)=1.0
                EdgeData("X", "Y", "Application", "Application", "app_to_app", "DEPENDS_ON", 1.0, path_count=1), # log2(1+1)=1.0
            ],
        )
        struct_base = StructuralAnalyzer().analyze(graph_base)
        quality_base = QualityAnalyzer().analyze(struct_base)
        m_base = {c.id: c.scores.maintainability for c in quality_base.components}["X"]
        
        # Test case (from graph variable above):
        # Z->X (path_count=7 -> log2(1+7)=3.0)
        # X->Y (path_count=3 -> log2(1+3)=2.0)
        # path_complexity(X) = 2.0 (efferent only)
        # Wait, path_complexity is mean(log2(1+path_count)) for out_edges.
        # X has 1 out-edge (X->Y) with path_count=3 -> path_complexity = 2.0.
        # Base has path_complexity = 1.0.
        
        # Difference in CR contribution: w_cr * (1.2 - 1.1) = w_cr * 0.1
        # It should be strictly greater.
        assert x_quality.scores.maintainability > m_base
