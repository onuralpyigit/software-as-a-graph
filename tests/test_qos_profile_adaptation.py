"""
tests/test_qos_profile_adaptation.py — QoS-profile weight adaptation sees real QoS
==================================================================================

``QualityAnalyzer._derive_qos_weights`` shifts the RM composite toward
Reliability for reliability-critical systems and toward Maintainability for
volatile ones, from the topic QoS profile ``StructuralAnalyzer`` collects.

The repositories store topic QoS flat (``qos_reliability`` ...) while the
collector read only a nested ``qos`` dict, so the profile was empty for every
repository-loaded system. An empty profile has a reliability signal of 0, which
is the volatile branch: every system with topics was scored at w_R = 0.65,
w_M = 0.35. The existing tests only built ``qos_profile`` dicts by hand, so they
never exercised the collector. These go through the full repository path.
"""

from __future__ import annotations

import pytest

from saag.analysis.analyzer import QualityAnalyzer
from saag.analysis.structural_analyzer import StructuralAnalyzer
from saag.analysis.weight_calculator import QualityWeights
from saag.core.models import GraphData
from saag.infrastructure.memory_repo import MemoryRepository


def _topology(qos: dict) -> dict:
    """Three apps on one topic, one broker, one host, one shared library."""
    apps = ["a1", "a2", "a3"]
    return {
        "metadata": {"scenario": "qos_profile_fixture", "generation_mode": "manual", "seed": 42},
        "nodes": [{"id": "n", "name": "host"}],
        "brokers": [{"id": "b", "name": "broker"}],
        "topics": [{"id": "t", "name": "t", "size": 1024, "frequency": 14.0, "qos": qos}],
        "applications": [
            {"id": "a1", "name": "a1", "role": "pub"},
            {"id": "a2", "name": "a2", "role": "sub"},
            {"id": "a3", "name": "a3", "role": "sub"},
        ],
        "libraries": [{"id": "l", "name": "lib"}],
        "relationships": {
            "runs_on": [{"from": c, "to": "n"} for c in apps + ["b"]],
            "routes": [{"from": "b", "to": "t"}],
            "publishes_to": [{"from": "a1", "to": "t"}],
            "subscribes_to": [{"from": "a2", "to": "t"}, {"from": "a3", "to": "t"}],
            "connects_to": [],
            "uses": [{"from": a, "to": "l"} for a in apps],
        },
    }


def _profile_from_repository(qos: dict) -> dict:
    repo = MemoryRepository()
    try:
        repo.save_graph(_topology(qos), clear=True)
        repo.derive_dependencies()
        return StructuralAnalyzer._collect_qos_profile(repo.get_graph_data(include_raw=True))
    finally:
        repo.close()


RELIABLE = {"reliability": "RELIABLE", "durability": "TRANSIENT_LOCAL", "transport_priority": "HIGH"}
VOLATILE = {"reliability": "BEST_EFFORT", "durability": "VOLATILE", "transport_priority": "LOW"}


def test_profile_reads_repository_stored_qos():
    profile = _profile_from_repository(RELIABLE)
    assert profile == {
        "durability": {"transient_local": 1},
        "reliability": {"reliable": 1},
        "priority": {"high": 1},
        "total_topics": 1,
    }


def test_profile_still_reads_nested_qos():
    topic = type("C", (), {"component_type": "Topic", "properties": {"qos": dict(RELIABLE)}})()
    profile = StructuralAnalyzer._collect_qos_profile(GraphData(components=[topic], edges=[]))
    assert profile["reliability"] == {"reliable": 1}
    assert profile["durability"] == {"transient_local": 1}
    assert profile["priority"] == {"high": 1}


@pytest.mark.parametrize("qos, direction", [(RELIABLE, 1), (VOLATILE, -1)])
def test_adaptation_moves_weights_in_the_declared_direction(qos, direction):
    base = QualityWeights()
    w = QualityAnalyzer._derive_qos_weights(_profile_from_repository(qos), base)
    assert (w.q_reliability - base.q_reliability) * direction > 0
    assert w.q_reliability + w.q_maintainability == pytest.approx(1.0)
