import pytest
from saag.core import GraphData, ComponentData, EdgeData
from saag.analysis.structural_analyzer import StructuralAnalyzer
from saag.prediction.analyzer import QualityAnalyzer
from saag.prediction.weight_calculator import QualityWeights

class TestAvailabilityDecoupling:
    """Verifies Issue 5: Decoupling structural SPOF from QoS weight."""

    @pytest.fixture
    def spof_graph(self):
        """
        Graph where X is a structural SPOF (articulation point).
        Y -> X -> Z
        """
        return GraphData(
            components=[
                ComponentData(id="X", component_type="Application"),
                ComponentData(id="Y", component_type="Application"),
                ComponentData(id="Z", component_type="Application"),
            ],
            edges=[
                EdgeData("Y", "X", "Application", "Application", "app_to_app", "DEPENDS_ON", 1.0),
                EdgeData("X", "Z", "Application", "Application", "app_to_app", "DEPENDS_ON", 1.0),
            ],
        )

    def test_structural_spof_not_masked_by_low_qos(self, spof_graph):
        """
        Structural SPOFs should retain baseline availability risk even if QoS weight is minimal.
        """
        # 1. High QoS Scenario
        # X has high weight (e.g. 1.0)
        # QSPOF = 1.0 * AP_c_dir
        for c in spof_graph.components:
            if c.id == "X":
                c.weight = 1.0
        
        struct_res_high = StructuralAnalyzer().analyze(spof_graph)
        quality_res_high = QualityAnalyzer().analyze(struct_res_high)
        a_high = {c.id: c.scores.availability for c in quality_res_high.components}["X"]
        
        # 2. Low QoS Scenario
        # X has low weight (e.g. 0.01)
        # In v2: QSPOF = 0.01 * AP_c_dir ≈ 0 -> A ≈ 0 (masking)
        # In v3: A = 0.35 * AP_c_dir + 0.25 * (0.01 * AP_c_dir) + ... + 0.05 * 0.01
        for c in spof_graph.components:
            if c.id == "X":
                c.weight = 0.01
        
        struct_res_low = StructuralAnalyzer().analyze(spof_graph)
        quality_res_low = QualityAnalyzer().analyze(struct_res_low)
        a_low = {c.id: c.scores.availability for c in quality_res_low.components}["X"]
        
        # In v2, a_low would be very close to 0 because QSPOF was 45% and AP_c_dir was only 15%.
        # Now AP_c_dir is 35% directly.
        
        # Verification: a_low should still be significant (at least 35% of its structural potential)
        ap_c_dir = struct_res_low.components["X"].ap_c_directed
        assert ap_c_dir >= 0.5 # It's a linear SPOF
        
        # Expected baseline contribution: 0.35 * ap_c_dir
        expected_baseline = 0.35 * ap_c_dir
        assert a_low >= expected_baseline
        
        # It should still be lower than a_high because QSPOF and w(v) terms still contribute
        assert a_low < a_high
        
        print(f"SPOF AP_c_dir: {ap_c_dir:.4f}")
        print(f"Availability (High QoS): {a_high:.4f}")
        print(f"Availability (Low QoS): {a_low:.4f}")

    def test_availability_weights_sum_to_one(self):
        """Verify that the new AHP weights sum to 1.0."""
        w = QualityWeights()
        total = (
            w.a_qspof + 
            w.a_bridge_ratio + 
            w.a_ap_c_directed + 
            w.a_cdi + 
            w.a_qos_weight
        )
        assert total == pytest.approx(1.0)
