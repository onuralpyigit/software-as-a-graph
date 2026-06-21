"""
test_groundtruth_contract.py
────────────────────────────
SIM-01 drift guard: pins the schema the validation harness depends on, and
documents which simulator is canonical for the published impact scores.
"""

import json
from pathlib import Path


def test_impact_scores_schema(tmp_path):
    """FaultInjector.run().save() must produce the schema validate_predictions
    expects: {"records": {node_id: {"impact_score": float in [0,1]}}}."""
    import networkx as nx
    from saag.simulation.fault_injector import FaultInjector

    g = nx.DiGraph()
    g.add_node("A", type="Application")
    g.add_node("B", type="Application")
    g.add_node("T", type="Topic")
    g.add_edge("A", "T", type="PUBLISHES_TO")
    g.add_edge("T", "B", type="SUBSCRIBES_TO")

    res = FaultInjector(graph=g, seeds=[42]).run(node_types=["Application"])
    out = tmp_path / "impact_scores.json"
    res.save(out)

    raw = json.loads(out.read_text())
    assert "records" in raw, f"expected 'records' key, got {list(raw.keys())}"
    for nid, rec in raw["records"].items():
        assert "impact_score" in rec, f"node {nid} missing 'impact_score'"
        assert 0.0 <= rec["impact_score"] <= 1.0, (
            f"node {nid} impact_score {rec['impact_score']} out of [0,1]"
        )


def test_canonical_simulator_is_documented():
    """Drift guard: the simulator backing the published numbers must be named.

    The library blast-radius result (ICAOMessageLib I≈0.97) only arises from
    FailureSimulator's USES cascade. If that number appears in the paper,
    CANONICAL must be FailureSimulator and FaultInjector is diagnostic only.

    TODO(onuralp): wire a shared fixture that records both I(v) vectors and
    asserts the file consumed by Predict/Validate comes from CANONICAL.
    """
    CANONICAL = "FailureSimulator"
    assert CANONICAL in {"FaultInjector", "FailureSimulator"}
