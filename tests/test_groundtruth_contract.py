"""
test_groundtruth_contract.py
────────────────────────────
SIM-01 drift guard: pins the schema the validation harness depends on, and
enforces which simulator is canonical for the published impact scores.

Two engines emit differently-scaled "impact", and for a long time the one this
file *declared* canonical was not the one the pipeline actually consumed. These
tests now check the artifact end-to-end: the labeler names itself, declares the
dimensions it genuinely measured and the nodes it never touched, and the tensor
the trainer masks on round-trips back to that artifact.
"""

import json
from pathlib import Path

import pytest


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


#: The engine that labels nodes for the Predict stage. FaultInjector, because it
#: is what every cached artifact and published result cell actually consumed, and
#: because it is already deterministic and multi-seed with per-node variance.
#:
#: The earlier claim that only FailureSimulator's USES cascade could produce the
#: ICAOMessageLib library blast radius is obsolete: FaultInjector derives
#: DEPENDS_ON(app_to_lib) from USES edges (fault_injector.py:222-224) and
#: cascades them at prob 1.0 (:409-411). Libraries simply are not in its default
#: node_types, which is a configuration choice, not an engine limitation.
CANONICAL_LABELER = "FaultInjector"

#: The engine that supplies the Validate stage's RMAV oracle. These two measure
#: DIFFERENT quantities (see saag/simulation/models.py:336-350) and must never be
#: mixed within a single stage.
CANONICAL_VALIDATION_ORACLE = "FailureSimulator"


def _build_labeled_artifact(tmp_path):
    """Run the canonical labeler on a graph with a deliberate coverage gap."""
    import networkx as nx
    from saag.simulation.fault_injector import FaultInjector

    g = nx.DiGraph()
    g.add_node("A", type="Application")
    g.add_node("B", type="Application")
    g.add_node("T", type="Topic")       # never injected -> must be unlabeled
    g.add_node("N", type="Node")        # never injected -> must be unlabeled
    g.add_edge("A", "T", type="PUBLISHES_TO")
    g.add_edge("T", "B", type="SUBSCRIBES_TO")

    res = FaultInjector(graph=g, seeds=[42]).run(node_types=["Application"])
    out = tmp_path / "impact_scores.json"
    res.save(out)
    return g, json.loads(out.read_text())


def test_artifact_declares_its_labeler(tmp_path):
    """The artifact must name the engine that produced it.

    Two simulators emit differently-scaled impact scores; a consumer that cannot
    tell them apart cannot know what its numbers mean.
    """
    _, raw = _build_labeled_artifact(tmp_path)
    assert raw["labeler"] == CANONICAL_LABELER
    assert raw["seeds_used"], "artifact must record the seeds it used"
    assert raw["labeled_node_types"] == ["Application"]


def test_artifact_declares_which_dimensions_it_measured(tmp_path):
    """Unmeasured label dimensions must be declared, not silently zeroed."""
    from saag.prediction.data_preparation import extract_simulation_dict

    _, raw = _build_labeled_artifact(tmp_path)
    declared = set(raw["labeled_dimensions"])

    assert "maintainability" not in declared, (
        "FaultInjector emits a scalar impact; it does not measure maintainability"
    )
    assert "security" not in declared

    # The parsed dict must carry exactly the declared dimensions — no fabricated
    # zeros standing in for dimensions the engine never computed.
    parsed = extract_simulation_dict(raw)
    assert parsed, "expected at least one parsed record"
    for node_id, dims in parsed.items():
        assert set(dims) == declared, (
            f"{node_id}: parsed dimensions {sorted(dims)} != declared {sorted(declared)}"
        )


def test_unlabeled_nodes_are_recorded_not_dropped(tmp_path):
    """Nodes outside the sweep must be listed, so the gap survives into training.

    Previously they vanished through a set intersection at evaluation time, so a
    42-58% coverage gap was invisible in every report.
    """
    g, raw = _build_labeled_artifact(tmp_path)

    labeled = set(raw["records"])
    unlabeled = set(raw["unlabeled_node_ids"])

    assert unlabeled == {"T", "N"}
    assert labeled & unlabeled == set(), "a node cannot be both labeled and unlabeled"
    assert labeled | unlabeled == set(g.nodes), "every node must be accounted for"


def test_label_mask_round_trips_the_artifact(tmp_path):
    """The tensor the trainer masks on must agree with the artifact exactly.

    This is the contract the old TODO asked for: the file consumed by Predict
    must be traceable to the labeler that wrote it.
    """
    pytest.importorskip("torch")
    from saag.prediction.data_preparation import (
        extract_simulation_dict,
        networkx_to_hetero_data,
    )

    g, raw = _build_labeled_artifact(tmp_path)
    for node, attrs in g.nodes(data=True):
        attrs["component_type"] = attrs["type"]

    sim = extract_simulation_dict(raw)
    result = networkx_to_hetero_data(g, structural_metrics={}, simulation_results=sim)

    masked_out = set()
    for node_type in result.present_node_types:
        store = result.hetero_data[node_type]
        if not hasattr(store, "label_mask"):
            continue
        for local_idx, name in enumerate(result.node_id_map[node_type]):
            if bool(store.label_mask[local_idx]):
                assert name in raw["records"], f"{name} masked in but absent from artifact"
                assert store.y[local_idx, 0].item() == pytest.approx(
                    raw["records"][name]["impact_score"]
                ), f"{name}: label does not round-trip the artifact"
            else:
                masked_out.add(name)

    assert masked_out == set(raw["unlabeled_node_ids"])

    # dimension_mask must match what the artifact declared it measured.
    from saag.prediction.data_preparation import LABEL_COLS

    measured = {dim for dim, col in LABEL_COLS.items() if result.dimension_mask[col]}
    assert measured == set(raw["labeled_dimensions"])


def test_validation_oracle_is_the_other_engine():
    """Validate reads FailureSimulator; the two engines are never mixed."""
    import inspect

    from saag.validation import service as validation_service

    src = inspect.getsource(validation_service)
    assert "run_failure_simulation_exhaustive" in src, (
        "validation must source its oracle from FailureSimulator's exhaustive sweep"
    )
    assert CANONICAL_VALIDATION_ORACLE != CANONICAL_LABELER, (
        "labeler and validation oracle must stay distinct engines"
    )
