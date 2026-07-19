"""
tests/test_prescription.py
"""
import pytest
from saag.infrastructure.memory_repo import MemoryRepository
from saag.prescription.service import PrescribeService
from saag.usecases.prescribe_graph import PrescribeGraphUseCase
from saag.client import Client
from saag import Pipeline

@pytest.fixture
def repo_with_vulnerable_topology():
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

def test_prescribe_rule_compilation(repo_with_vulnerable_topology):
    client = Client(repo=repo_with_vulnerable_topology)
    analysis = client.analyze(layer="system")
    # Criticality levels (SPOF/CRITICAL) now come from the Predict step, not Analyze.
    prediction = client.predict(analysis, mode="rmav")

    # Run prescribe compiler
    service = PrescribeService(repo_with_vulnerable_topology)
    policy = service.compile_policy(analysis_result=analysis.raw, prediction_result=prediction)
    
    # Verify logical subgraph refactoring (Topic splitting T1)
    assert len(policy.topic_splits) == 1
    assert policy.topic_splits[0]["topic"] == "T1"
    assert set(policy.topic_splits[0]["publishers"]) == {"AppA", "AppC"}
    assert set(policy.topic_splits[0]["subscribers"]) == {"AppB", "AppD"}
    
    # Verify physical locality anti-affinity
    # NodeMain hosts AppA, AppB, BrokerMain, and NodeMain is an AP/critical, triggering anti-affinity
    assert len(policy.node_reallocations) > 0
    realloc_comps = [r["component"] for r in policy.node_reallocations]
    assert "AppB" in realloc_comps
    
    # Verify transport contract hardening
    assert len(policy.qos_upgrades) == 1
    assert policy.qos_upgrades[0]["topic"] == "T1"
    assert policy.qos_upgrades[0]["target_reliability"] == "RELIABLE"
    assert policy.qos_upgrades[0]["target_durability"] == "TRANSIENT"

def test_closed_loop_prescriptive_verification(repo_with_vulnerable_topology):
    client = Client(repo=repo_with_vulnerable_topology)
    analysis = client.analyze(layer="system")
    
    # Run prescribe usecase
    res = client.prescribe(analysis_result=analysis, layer="system")
    
    # Verify prescribe results
    assert res is not None
    assert hasattr(res, "original_sri")
    assert hasattr(res, "mutated_sri")
    assert hasattr(res, "sri_improvement")
    assert len(res.applied_changes) > 0

    # Verify the accept/reject gate: accepted must be a bool consistent with the
    # sign of sri_improvement (accepted = mutated policy reduced overall system risk).
    assert isinstance(res.accepted, bool)
    assert res.accepted == (res.sri_improvement > 0)

def test_pipeline_integration(repo_with_vulnerable_topology):
    # Run the full pipeline including the prescribe step
    pipeline = Pipeline(repo=repo_with_vulnerable_topology)
    res = pipeline.analyze().simulate().validate().prescribe().run()

    assert res.prescription is not None
    assert res.prescription.sri_improvement is not None
    assert res.prescription.accepted == (res.prescription.sri_improvement > 0)


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
