"""Guards for edge-removal simulation (closes failure-simulation.md L8).

Edge criticality labels used to be a projection of node labels through a
hand-picked bridge multiplier (``I*(u) x {1.0, 0.1}``), and
``SimulationService.classify_edges`` always returned ``[]``. These tests pin the
properties that make the new measurement trustworthy: the edge is really
removed, both endpoints stay up, and the reported number is a delta against a
control rather than the raw impact floor.
"""

from __future__ import annotations

import pytest

from saag.simulation.failure_simulator import FailureSimulator
from saag.simulation.graph import SimulationGraph


def _topology():
    """Two publishers, one shared topic, two subscribers, one broker."""
    return {
        "nodes": [{"id": "N0", "name": "host0"}],
        "brokers": [{"id": "B0", "name": "broker0"}],
        "topics": [
            {"id": "T0", "name": "/t0", "size": 64,
             "qos": {"reliability": "RELIABLE", "durability": "PERSISTENT",
                     "transport_priority": "HIGH"}},
            {"id": "T1", "name": "/t1", "size": 64,
             "qos": {"reliability": "RELIABLE", "durability": "PERSISTENT",
                     "transport_priority": "HIGH"}},
        ],
        "applications": [
            {"id": "P0", "name": "pub0", "role": "pub"},
            {"id": "P1", "name": "pub1", "role": "pub"},
            {"id": "S0", "name": "sub0", "role": "sub"},
            {"id": "S1", "name": "sub1", "role": "sub"},
        ],
        "libraries": [],
        "relationships": {
            "runs_on": [{"from": c, "to": "N0"} for c in ("P0", "P1", "S0", "S1", "B0")],
            "routes": [{"from": "B0", "to": "T0"}, {"from": "B0", "to": "T1"}],
            "publishes_to": [{"from": "P0", "to": "T0"}, {"from": "P1", "to": "T1"}],
            "subscribes_to": [{"from": "S0", "to": "T0"}, {"from": "S1", "to": "T1"}],
            "connects_to": [],
            "uses": [],
        },
    }


@pytest.fixture
def simulator():
    from saag.infrastructure.memory_repo import MemoryRepository

    repo = MemoryRepository()
    repo.save_graph(_topology(), clear=True)
    graph = SimulationGraph(repo.get_graph_data(include_raw=True))
    return FailureSimulator(graph)


def test_failed_edge_hides_only_that_relationship(simulator):
    """Severing P0->T0 must not affect P1->T1 or either endpoint's other links."""
    graph = simulator.graph
    assert "P0" in graph.get_publishers("T0")

    graph.fail_edge("P0", "T0")
    assert "P0" not in graph.get_publishers("T0")
    assert "P1" in graph.get_publishers("T1")     # unrelated edge intact
    assert graph.is_active("P0")                  # endpoint still up
    assert graph.is_active("T0")

    graph.recover_edge("P0", "T0")
    assert "P0" in graph.get_publishers("T0")


def test_reset_clears_severed_edges(simulator):
    simulator.graph.fail_edge("P0", "T0")
    simulator.graph.reset()
    assert "P0" in simulator.graph.get_publishers("T0")


def test_removal_is_measured_against_a_control(simulator):
    """The impact floor must be subtracted, or every edge inherits it.

    ``_calculate_impact`` is non-zero on a pristine graph because topics that
    already lack a publisher or subscriber count as lost throughput. An edge
    label that did not difference that out would report the floor as signal.
    """
    simulator.graph.reset()
    simulator._compute_baseline()
    simulator._baseline_computed = True

    floor = simulator._null_impact().composite_impact
    assert floor >= 0.0

    # RUNS_ON carries no traffic in this cascade model, so removing one must
    # measure as exactly zero rather than as the floor.
    result = simulator.simulate_edge_removal("P0", "N0", "RUNS_ON")
    assert result.combined_impact == 0.0

    # A publish edge does carry traffic, so it must measure strictly above zero.
    carrying = simulator.simulate_edge_removal("P0", "T0", "PUBLISHES_TO")
    assert carrying.combined_impact > 0.0


def test_simulation_is_restored_after_each_measurement(simulator):
    """A sweep must not leave the graph mutated between candidates."""
    simulator.graph.reset()
    simulator._compute_baseline()
    simulator._baseline_computed = True

    first = simulator.simulate_edge_removal("P0", "T0", "PUBLISHES_TO")
    second = simulator.simulate_edge_removal("P0", "T0", "PUBLISHES_TO")

    assert first.combined_impact == second.combined_impact
    assert not simulator.graph._failed_edges


def test_sweep_returns_ranked_candidates(simulator):
    results = simulator.simulate_edge_removal_sweep(top_q=10)

    assert results, "candidate set must not be empty on a connected topology"
    impacts = [e.combined_impact for e in results]
    assert impacts == sorted(impacts, reverse=True)
    assert all(e.evaluated for e in results)
    assert all(e.relationship for e in results)


def test_classify_edges_is_populated():
    """L8 said this always returned []; it must now return measurements."""
    from saag.infrastructure.memory_repo import MemoryRepository
    from saag.simulation.service import SimulationService

    repo = MemoryRepository()
    repo.save_graph(_topology(), clear=True)
    edges = SimulationService(repo).classify_edges(layer="system")

    assert isinstance(edges, list)
    assert edges, "edge criticality must be populated by the removal sweep"
