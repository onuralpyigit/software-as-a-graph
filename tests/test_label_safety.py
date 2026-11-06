"""
Guards on the path from a label artifact to a training run.

Each of these used to pass silently: a runtime-label file read as all zeros, a
label file from another scenario aligned on shared ids like ``A0``, and a label
too noisy to agree with itself was trained on. The checks live in
``saag/prediction/label_artifacts.py``.
"""

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

import cli.train_graph as train_graph
from saag.core.graph_io import build_graph_from_json, load_graph, topology_fingerprint
from saag.prediction.data_preparation import extract_simulation_dict
from saag.prediction.label_artifacts import (
    MIN_LABEL_RETEST,
    LabelArtifactError,
    check_stability,
    check_topology,
    select_labels,
)
from saag.simulation.fault_injector import FaultInjector

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = sorted((ROOT / "data" / "scenarios").glob("*_system.json"))
NETWORK = sorted((ROOT / "data" / "network").glob("*_system.json"))


def _chain() -> nx.DiGraph:
    g = nx.DiGraph()
    for a in ("P", "S"):
        g.add_node(a, type="Application")
    g.add_node("T", type="Topic", frequency=10.0)
    g.add_edge("P", "T", type="PUBLISHES_TO")
    g.add_edge("S", "T", type="SUBSCRIBES_TO")
    return g


def _runtime_artifact(stability=0.9, n_draws=2):
    return {
        "schema_version": "1.0",
        "labeler": "MessageFlowSimulator/runtime",
        "labeled_dimensions": ["i_flow", "i_recovery"],
        "label_stability": {"i_flow": {"n_draws": n_draws, "worst_spearman": stability}},
        "records": {
            "A": {"i_flow": {"mean": 0.4, "std": 0.0, "per_seed": {"1": 0.4}}, "i_recovery": None},
            "B": {"i_flow": {"mean": 0.0, "std": 0.0, "per_seed": {"1": 0.0}}, "i_recovery": None},
        },
    }


# ── Topology fingerprint ─────────────────────────────────────────────────────

@pytest.mark.parametrize("path", SCENARIOS + NETWORK, ids=lambda p: p.stem)
def test_both_graph_loaders_agree_on_the_fingerprint(path):
    # Simulation loads through load_graph and training through
    # build_graph_from_json, so a disagreement would refuse every real run.
    assert topology_fingerprint(load_graph(path)) == topology_fingerprint(
        build_graph_from_json(json.loads(path.read_text())))


def test_fingerprint_ignores_insertion_order_and_derived_edges():
    a = _chain()
    b = nx.DiGraph()
    b.add_node("T", type="Topic", frequency=10.0)
    b.add_node("S", type="Application")
    b.add_node("P", type="Application")
    b.add_edge("S", "T", type="SUBSCRIBES_TO")
    b.add_edge("P", "T", type="PUBLISHES_TO")
    b.add_edge("S", "P", type="DEPENDS_ON")
    assert topology_fingerprint(a) == topology_fingerprint(b)


@pytest.mark.parametrize("change", [
    lambda g: g.remove_edge("S", "T"),
    lambda g: g.add_node("X", type="Application"),
    lambda g: g.nodes["T"].update(frequency=50.0),
    lambda g: g.nodes["T"].update(qos={"reliability": "RELIABLE"}),
], ids=["edge", "node", "rate", "qos"])
def test_fingerprint_sees_what_changes_a_label(change):
    g = _chain()
    before = topology_fingerprint(g)
    change(g)
    assert topology_fingerprint(g) != before


# ── Artifact provenance ──────────────────────────────────────────────────────

def test_fault_injector_artifact_records_how_it_was_made():
    g = _chain()
    raw = FaultInjector(g, seeds=[1], propagation_threshold=0.5, qos_factor_mode="none",
                        replica_aware=True).run(node_types=["Application"]).to_dict()

    assert raw["schema_version"] == "2.2"
    assert raw["topology_fingerprint"] == topology_fingerprint(g)
    cfg = raw["labeler_config"]
    assert cfg["propagation_threshold"] == 0.5
    assert cfg["qos_factor_mode"] == "none"
    assert cfg["failure_model"] == "step"
    assert cfg["replica_aware"] is True
    assert cfg["network_gating"] is False


def test_runtime_artifact_records_the_topology():
    pytest.importorskip("simpy")
    from saag.simulation.runtime.labels import runtime_labels

    g = load_graph(ROOT / "data" / "scenarios" / "tiny_system.json")
    raw = runtime_labels(g, seeds=[1], settings={"duration": 5.0}, candidates=["A9"])
    assert raw["topology_fingerprint"] == topology_fingerprint(g)


# ── Selecting a label ────────────────────────────────────────────────────────

def test_runtime_artifact_is_refused_instead_of_read_as_zeros():
    with pytest.raises(ValueError, match="i_flow"):
        extract_simulation_dict(_runtime_artifact())
    with pytest.raises(LabelArtifactError, match="--label"):
        select_labels(_runtime_artifact())


def test_record_without_a_score_is_an_error_not_a_zero():
    with pytest.raises(ValueError, match="impact_score"):
        extract_simulation_dict({"labeler": "FaultInjector", "records": {"A": {"cascade_depth": 1}}})


def test_fault_injector_default_label_is_unchanged():
    raw = {"labeler": "FaultInjector", "records": {"A": {"impact_score": 0.5}}}
    assert select_labels(raw) == {"A": {"composite": 0.5, "reliability": 0.5}}


def test_runtime_label_fills_only_the_composite_and_keeps_unmeasured_absent():
    assert select_labels(_runtime_artifact(), "i_flow") == {
        "A": {"composite": 0.4}, "B": {"composite": 0.0}}
    # i_recovery is None for every node: nothing was measured, so nothing is labelled.
    with pytest.raises(LabelArtifactError, match="i_recovery"):
        select_labels(_runtime_artifact(), "i_recovery")


@pytest.mark.parametrize("raw, label", [
    ({"labeler": "FaultInjector", "records": {"A": {"impact_score": 0.5}}}, "i_flow"),
    (_runtime_artifact(), "i_nonsense"),
    ({"labeler": "FailureSimulator", "records": {"A": {"impact_score": 0.5}}}, "impact_score"),
])
def test_wrong_label_or_labeler_is_refused(raw, label):
    with pytest.raises(LabelArtifactError):
        select_labels(raw, label)


# ── Topology check ───────────────────────────────────────────────────────────

def test_labels_from_another_scenario_are_refused():
    atm = load_graph(ROOT / "data" / "scenarios" / "atm_system.json")
    micro = load_graph(ROOT / "data" / "scenarios" / "microservices_system.json")
    raw = FaultInjector(atm, seeds=[1]).run(node_types=["Application"]).to_dict()
    check_topology(raw, atm)
    with pytest.raises(LabelArtifactError, match="atm_system"):
        check_topology(raw, micro)


def test_artifact_without_a_fingerprint_still_loads():
    # Caches built before schema 2.2 carry none; refusing them would break every
    # stored cache, so they pass with a warning.
    check_topology({"labeler": "FaultInjector", "records": {}}, _chain())


# ── Stability gate ───────────────────────────────────────────────────────────

def test_gate_refuses_a_label_that_does_not_agree_with_itself():
    with pytest.raises(LabelArtifactError, match="i_flow"):
        check_stability(_runtime_artifact(stability=-0.011), "i_flow")
    assert check_stability(_runtime_artifact(stability=0.9), "i_flow") == 0.9


def test_gate_threshold_and_override():
    low = MIN_LABEL_RETEST - 0.01
    with pytest.raises(LabelArtifactError):
        check_stability(_runtime_artifact(stability=low), "i_flow")
    assert check_stability(_runtime_artifact(stability=low), "i_flow", min_retest=0.5) == low
    assert check_stability(_runtime_artifact(stability=low), "i_flow", allow_unstable=True) == low


@pytest.mark.parametrize("value", [None, float("nan")], ids=["none", "nan"])
def test_unmeasured_runtime_stability_is_refused_not_coerced(value):
    with pytest.raises(LabelArtifactError, match="never measured"):
        check_stability(_runtime_artifact(stability=value, n_draws=1), "i_flow")


def test_unmeasured_fault_injector_stability_warns(caplog):
    # A single-seed fault-inject run, or a stored artifact older than the block.
    raw = {"labeler": "FaultInjector",
           "label_stability": {"n_seeds": 1, "test_retest_spearman": None},
           "records": {"A": {"impact_score": 0.5}}}
    with caplog.at_level(logging.WARNING):
        assert check_stability(raw, "impact_score") is None
    assert "stability" in caplog.text


def test_fault_injector_labels_clear_the_gate_on_the_corpus():
    raw = FaultInjector(load_graph(ROOT / "data" / "scenarios" / "microservices_system.json"),
                        seeds=[42, 123, 456, 789, 2024]).run(node_types=["Application"]).to_dict()
    assert check_stability(raw, "impact_score") >= MIN_LABEL_RETEST


# ── The training CLI ─────────────────────────────────────────────────────────

def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def _train(tmp_path, simulated, *extra, topology=None):
    gnn = MagicMock()
    gnn.train.return_value = MagicMock(gnn_metrics=None, summary=MagicMock(return_value={}),
                                       top_critical_nodes=MagicMock(return_value=[]))
    argv = [
        "train_graph.py", "--layer", "app",
        "--input", _write(tmp_path, "topology.json", topology or {
            "applications": [{"id": "A"}, {"id": "B"}]}),
        "--structural", _write(tmp_path, "s.json", {"A": {"pagerank": 0.1}, "B": {"pagerank": 0.2}}),
        "--rm", _write(tmp_path, "rm.json", {"A": {"overall": 0.7}, "B": {"overall": 0.1}}),
        "--simulated", _write(tmp_path, "labels.json", simulated),
        "--checkpoint", str(tmp_path / "ckpt"),
        *extra,
    ]
    with patch.object(sys, "argv", argv), patch("cli.train_graph.ConsoleDisplay"), \
            patch("saag.Client"), patch("saag.prediction.GNNService", return_value=gnn):
        train_graph.main()
    return gnn


def test_cli_refuses_runtime_labels_without_a_label_choice(tmp_path):
    with pytest.raises(SystemExit):
        _train(tmp_path, _runtime_artifact())


def test_cli_trains_on_the_chosen_runtime_label_and_records_it(tmp_path):
    raw = _runtime_artifact()
    raw["topology_fingerprint"] = topology_fingerprint(
        build_graph_from_json({"applications": [{"id": "A"}, {"id": "B"}]}))
    gnn = _train(tmp_path, raw, "--label", "i_flow")

    assert gnn.train.call_args.kwargs["simulation_results"] == {
        "A": {"composite": 0.4}, "B": {"composite": 0.0}}
    prov = json.loads((tmp_path / "ckpt" / "label_provenance.json").read_text())
    assert prov["labeler"] == "MessageFlowSimulator/runtime"
    assert prov["label"] == "i_flow"
    assert prov["retest"] == 0.9
    assert prov["topology_fingerprint"] == raw["topology_fingerprint"]


def test_cli_refuses_labels_from_a_different_topology(tmp_path):
    raw = _runtime_artifact()
    raw["topology_fingerprint"] = "0" * 16
    with pytest.raises(SystemExit):
        _train(tmp_path, raw, "--label", "i_flow")


def test_cli_refuses_an_unstable_label_unless_told_otherwise(tmp_path):
    raw = _runtime_artifact(stability=-0.011)
    with pytest.raises(SystemExit):
        _train(tmp_path, raw, "--label", "i_flow")
    gnn = _train(tmp_path, raw, "--label", "i_flow", "--allow-unstable-labels")
    assert gnn.train.called
