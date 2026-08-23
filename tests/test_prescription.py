"""
tests/test_prescription.py
"""
from types import SimpleNamespace

import pytest

from saag.core.models import ComponentData, EdgeData, GraphData
from saag.infrastructure.memory_repo import MemoryRepository
from saag.prescription.rules import _predicted_critical, _smells, compile_policy
from saag.prescription.service import PrescribeService
from saag.client import Client
from saag import Pipeline

@pytest.fixture
def repo_with_congested_topology():
    """
    Seed repository with a topology designed to trigger all three prescriptive rules:
    - Topic T1 has 2 publishers (AppA, AppC) and 2 subscribers (AppB, AppD), triggering Topic Splitting.
    - NodeMain hosts AppA, AppB, and BrokerMain, triggering Physical Anti-Affinity.
    - Topic T1 uses BEST_EFFORT reliability and VOLATILE durability, triggering QoS hardening.
    """
    repo = MemoryRepository()
    graph_data = {
        "applications": [
            {"id": "AppA", "name": "App A", "role": ["pub"]},
            {"id": "AppB", "name": "App B", "role": ["sub"]},
            {"id": "AppC", "name": "App C", "role": ["pub"]},
            {"id": "AppD", "name": "App D", "role": ["sub"]},
        ],
        "brokers": [
            {"id": "BrokerMain", "name": "Broker Main"}
        ],
        "nodes": [
            {"id": "NodeMain", "name": "Node Main"},
            {"id": "NodeOther", "name": "Node Other"},
        ],
        "topics": [
            {
                "id": "T1",
                "name": "Topic T1",
                "size": 1024,
                "qos": {
                    "reliability": "BEST_EFFORT",
                    "durability": "VOLATILE",
                    "transport_priority": "MEDIUM"
                }
            }
        ],
        "relationships": {
            "publishes_to": [
                {"source": "AppA", "target": "T1", "weight": 0.8},
                {"source": "AppC", "target": "T1", "weight": 0.8}
            ],
            "subscribes_to": [
                {"source": "AppB", "target": "T1", "weight": 0.8},
                {"source": "AppD", "target": "T1", "weight": 0.8}
            ],
            "routes": [
                {"source": "BrokerMain", "target": "T1", "weight": 1.0}
            ],
            "runs_on": [
                {"source": "AppA", "target": "NodeMain", "weight": 1.0},
                {"source": "AppB", "target": "NodeMain", "weight": 1.0},
                {"source": "AppC", "target": "NodeOther", "weight": 1.0},
                {"source": "AppD", "target": "NodeOther", "weight": 1.0},
                {"source": "BrokerMain", "target": "NodeMain", "weight": 1.0}
            ],
            "connects_to": [
                {"source": "NodeMain", "target": "NodeOther", "weight": 1.0}
            ]
        }
    }
    repo.save_graph(graph_data)
    return repo

def test_prescribe_rule_compilation(repo_with_congested_topology):
    client = Client(repo=repo_with_congested_topology)
    analysis = client.analyze(layer="system")
    # Criticality levels (SPOF/CRITICAL) now come from the Predict step, not Analyze.
    prediction = client.predict(analysis, mode="rm")

    # Run prescribe compiler
    service = PrescribeService(repo_with_congested_topology)
    policy = service.compile_policy(analysis_result=analysis.raw, prediction_result=prediction)

    # Verify logical subgraph refactoring (Topic splitting T1)
    assert len(policy.topic_splits) == 1
    assert policy.topic_splits[0].topic == "T1"
    assert set(policy.topic_splits[0].publishers) == {"AppA", "AppC"}
    assert set(policy.topic_splits[0].subscribers) == {"AppB", "AppD"}

    # Verify physical locality anti-affinity
    # NodeMain hosts AppA, AppB, BrokerMain, and NodeMain is an AP/critical, triggering anti-affinity
    assert len(policy.node_reallocations) > 0
    realloc_comps = [r.component for r in policy.node_reallocations]
    assert "AppB" in realloc_comps

    # Verify transport contract hardening
    assert len(policy.qos_upgrades) == 1
    assert policy.qos_upgrades[0].topic == "T1"
    assert policy.qos_upgrades[0].target_reliability == "RELIABLE"
    assert policy.qos_upgrades[0].target_durability == "TRANSIENT"

def test_missing_prediction_result_degrades_to_splits_only(repo_with_congested_topology, caplog):
    """Regression test for the defect this suite did not catch: every caller
    that omits `prediction_result` (as reproduce/run_prescribe_all.py and
    cli/prescribe_graph.py both did) silently loses Rules 2 and 3, because
    Analyze is structural-only and analysis_result.quality is always None.

    On this fixture NodeMain hosts a SPOF-triggering co-location and T1 is a
    hardening candidate, so the full policy (with prediction_result) must
    contain a reallocation and a QoS upgrade -- and the degraded policy
    (without it) must contain neither, plus the loud warning that names why.
    """
    client = Client(repo=repo_with_congested_topology)
    analysis = client.analyze(layer="system")
    service = PrescribeService(repo_with_congested_topology)

    degraded = service.compile_policy(analysis_result=analysis.raw, prediction_result=None)
    assert len(degraded.topic_splits) > 0
    assert degraded.node_reallocations == []
    assert degraded.qos_upgrades == []

    prediction = client.predict(analysis, mode="rm")
    full = service.compile_policy(analysis_result=analysis.raw, prediction_result=prediction)
    assert len(full.node_reallocations) > 0
    assert len(full.qos_upgrades) > 0

    import logging
    with caplog.at_level(logging.WARNING, logger="saag.prescription.service"):
        service.prescribe(analysis_result=analysis.raw, prediction_result=None, layer="system")
    assert any("prediction_result" in record.message for record in caplog.records)


def test_closed_loop_prescriptive_verification(repo_with_congested_topology):
    client = Client(repo=repo_with_congested_topology)
    analysis = client.analyze(layer="system")

    # Run prescribe usecase
    res = client.prescribe(analysis_result=analysis, layer="system")

    # Verify prescribe results
    assert res is not None
    assert hasattr(res, "original_sri")
    assert hasattr(res, "mutated_sri")
    assert hasattr(res, "sri_improvement")

    # Every candidate edit is verified on its own counterfactual graph, and only
    # the ones that beat the simulator's seed noise at every propagation
    # threshold are applied. On a topology this small that can legitimately be
    # none of them — an empty applied_changes is the filter working, not a
    # failure, which is why this does not assert len(applied_changes) > 0.
    assert res.candidate_policy is not None
    assert len(res.edit_verdicts) == len(res.candidate_policy.edits())
    assert res.n_accepted + res.n_rejected == len(res.edit_verdicts)

    # applied_changes must describe exactly the accepted subset.
    assert len(res.policy.edits()) == res.n_accepted
    assert len(res.applied_changes) == res.n_accepted
    if res.n_accepted == 0:
        assert res.applied_changes == []
        assert res.sri_improvement == 0.0

    # Every rejected edit must say why it was rejected.
    for verdict in res.edit_verdicts:
        if not verdict.accepted:
            assert verdict.reason, f"{verdict.kind}/{verdict.target} rejected without a reason"

    # accepted = the applied subset reduced overall system risk.
    assert isinstance(res.accepted, bool)
    assert res.accepted == (res.sri_improvement > 0)

def test_pipeline_integration(repo_with_congested_topology):
    # Run the full pipeline including the prescribe step
    pipeline = Pipeline(repo=repo_with_congested_topology)
    res = pipeline.analyze().simulate().validate().prescribe().run()

    assert res.prescription is not None
    assert res.prescription.sri_improvement is not None
    assert res.prescription.accepted == (res.prescription.sri_improvement > 0)


def test_prescribe_extreme_kappa_matches_the_documented_acceptance_rule(repo_with_congested_topology):
    """At an astronomically high kappa, most edits get rejected -- but not
    necessarily all of them.

    docs/prescription.md L5: the bar `mean_delta > kappa * sigma_seed`
    collapses to `mean_delta > 0` whenever `sigma_seed == 0` (the simulator is
    deterministic for that edit), no matter how large kappa is raised. A prior
    version of this test asserted `n_accepted == 0` unconditionally, which only
    held because this fixture's edits happened to all have non-zero seed
    spread -- coincidence, not the acceptance rule. This derives the expected
    verdict from each edit's own per-threshold statistics instead, so it keeps
    testing the rule itself even if that coincidence stops holding.
    """
    client = Client(repo=repo_with_congested_topology)
    analysis = client.analyze(layer="system")

    kappa = 1e9
    res = client.prescribe(analysis_result=analysis, layer="system", kappa=kappa)

    assert res.edit_verdicts, "fixture must compile at least one candidate edit"
    for verdict in res.edit_verdicts:
        expected_accept = all(
            stat.mean_delta > kappa * stat.sigma_seed
            for stat in verdict.per_threshold.values()
        )
        assert verdict.accepted == expected_accept, (
            f"{verdict.kind}/{verdict.target}: accepted={verdict.accepted} but "
            f"per-threshold stats say {expected_accept}"
        )
        if not verdict.accepted:
            assert verdict.reason, f"{verdict.kind}/{verdict.target} rejected without a reason"

    if res.n_accepted == 0:
        # Nothing cleared the bar: the early-return path reports the
        # unchanged baseline rather than running a no-op mutation.
        assert res.policy.is_empty()
        assert res.applied_changes == []
        assert res.mutated_sri == res.original_sri
        assert res.sri_improvement == 0.0
        assert res.accepted is False
    else:
        # A zero-variance edit cleared the bar despite kappa=1e9: the
        # accepted subset must still be exactly what applied_changes reports.
        assert len(res.policy.edits()) == res.n_accepted
        assert len(res.applied_changes) == res.n_accepted


def test_prescribe_result_to_dict_includes_accepted():
    """PrescribeResult.to_dict() must surface the accept/reject gate for JSON export."""
    from saag.prescription.models import PrescribeResult, PrescriptionPolicy

    result = PrescribeResult(
        original_sri=0.5,
        mutated_sri=0.3,
        sri_improvement=0.2,
        original_metrics={},
        mutated_metrics={},
        policy=PrescriptionPolicy(),
        accepted=True,
    )
    d = result.to_dict()
    assert d["accepted"] is True


# ── Rule compilation, in isolation ────────────────────────────────────────────

def _graph(components, edges):
    """Build a GraphData from (id, type, props) and (src, tgt, relation) tuples."""
    return GraphData(
        components=[
            ComponentData(id=cid, component_type=ctype, properties=props or {})
            for cid, ctype, props in components
        ],
        edges=[
            EdgeData(
                source_id=src, target_id=tgt,
                source_type="", target_type="",
                dependency_type="", relation_type=relation,
            )
            for src, tgt, relation in edges
        ],
    )


def test_predicted_critical_from_dict_node_scores():
    prediction = {"node_scores": {
        "A1": {"criticality_level": "HIGH"},
        "A2": {"criticality_level": "LOW"},
    }}

    assert _predicted_critical(prediction) == {"A1"}


def test_predicted_critical_from_raw_facade_node_scores():
    prediction = SimpleNamespace(raw=SimpleNamespace(node_scores={
        "A1": SimpleNamespace(criticality_level="CRITICAL"),
        "A2": SimpleNamespace(criticality_level="MINIMAL"),
    }))

    assert _predicted_critical(prediction) == {"A1"}


def test_predicted_critical_from_all_components():
    prediction = SimpleNamespace(all_components=[
        SimpleNamespace(id="A1", criticality_level="high"),
        SimpleNamespace(id="A2", criticality_level="medium"),
    ])

    assert _predicted_critical(prediction) == {"A1"}


def test_smells_tolerate_a_problem_whose_name_is_empty():
    """A problem object with an empty name must not be mistaken for a mapping."""
    prediction = SimpleNamespace(problems=[SimpleNamespace(name="", entity_id="A1")])

    assert _smells(None, prediction) == (set(), set())


def test_qos_rule_fires_for_a_critical_publisher_with_no_subscribers():
    """A critical channel is critical whether or not anyone is listening yet."""
    graph = _graph(
        components=[("T1", "Topic", None), ("AppA", "Application", None)],
        edges=[("AppA", "T1", "PUBLISHES_TO")],
    )
    prediction = {"node_scores": {"AppA": {"criticality_level": "CRITICAL"}}}

    policy = compile_policy(graph, analysis_result=None, prediction_result=prediction)

    assert [u.topic for u in policy.qos_upgrades] == ["T1"]


def test_topic_split_fires_for_a_god_publisher_with_no_subscribers():
    graph = _graph(
        components=[("T1", "Topic", None), ("AppA", "Application", None), ("AppB", "Application", None)],
        edges=[("AppA", "T1", "PUBLISHES_TO"), ("AppB", "T1", "PUBLISHES_TO")],
    )
    prediction = SimpleNamespace(
        problems=[{"name": "God Component", "entity_id": "AppA"}])

    policy = compile_policy(graph, analysis_result=None, prediction_result=prediction)

    assert [s.topic for s in policy.topic_splits] == ["T1"]
    assert policy.topic_splits[0].publishers == ["AppA", "AppB"]


def test_qos_rule_skips_an_already_hardened_topic():
    graph = _graph(
        components=[
            ("T1", "Topic", {"qos_reliability": "RELIABLE", "qos_durability": "TRANSIENT"}),
            ("AppA", "Application", None),
        ],
        edges=[("AppA", "T1", "PUBLISHES_TO")],
    )
    prediction = {"node_scores": {"T1": {"criticality_level": "CRITICAL"}}}

    policy = compile_policy(graph, analysis_result=None, prediction_result=prediction)

    assert policy.qos_upgrades == []


def test_node_reallocation_keeps_the_first_process_in_place():
    graph = _graph(
        components=[("N1", "Node", None)] + [(f"App{c}", "Application", None) for c in "ABC"],
        edges=[(f"App{c}", "N1", "RUNS_ON") for c in "ABC"],
    )
    prediction = {"node_scores": {"N1": {"criticality_level": "CRITICAL"}}}

    policy = compile_policy(graph, analysis_result=None, prediction_result=prediction)

    # Hosted ids are sorted, so "first" is lexicographic and reproducible.
    assert [r.component for r in policy.node_reallocations] == ["AppB", "AppC"]
    assert [r.to_node for r in policy.node_reallocations] == ["N1_AppB", "N1_AppC"]
