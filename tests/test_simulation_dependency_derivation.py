"""
SimulationGraph's DEPENDS_ON projection (Rules 1-6, docs/graph-model.md §4.4).

`saag/simulation/` never reads derived edges, so it re-derives the projection from
raw structural edges. These tests pin that derivation to the authoritative one in
`MemoryRepository` and cover each rule in isolation.

Regression context: the derivation used to implement one rule (direct `USES`) and
returned the subscribed *Topic* instead of that topic's publishers, so Applications
were sinks in the transposed graph and IM(v) was identically 0.0 for every component
the validation scores. See tests/test_maintainability_dimension.py.
"""
import json

import pytest

from saag.core.models import compute_effective_edge_weight
from saag.infrastructure.memory_repo import MemoryRepository
from saag.simulation.graph import SimulationGraph
from saag.usecases.model_graph import ModelGraphUseCase

#: Scenarios exercising every rule: brokers, colocation, and library chains.
PARITY_SCENARIOS = ("atm", "microservices", "hub_and_spoke", "av")


def _empty_graph():
    return {
        "metadata": {"seed": 1},
        "nodes": [], "brokers": [], "topics": [], "applications": [], "libraries": [],
        "relationships": {
            "runs_on": [], "routes": [], "publishes_to": [],
            "subscribes_to": [], "connects_to": [], "uses": [],
        },
    }


def _build(data) -> SimulationGraph:
    repo = MemoryRepository()
    repo.save_graph(data, clear=True)
    return SimulationGraph(repo.get_graph_data(include_raw=True))


def _arcs(graph: SimulationGraph):
    """{(dependent, dependency, dependency_type)} — weights dropped."""
    return {(s, t, d) for s, t, _, d in graph.get_dependency_edges()}


def _weight_of(graph: SimulationGraph, src, tgt, dep_type) -> float:
    return next(w for s, t, w, d in graph.get_dependency_edges()
                if (s, t, d) == (src, tgt, dep_type))


# ---------------------------------------------------------------------------
# Per-rule derivation
# ---------------------------------------------------------------------------

def test_rule1_subscriber_depends_on_publisher():
    """Rule 1 — the arc runs subscriber -> publisher, against the data flow."""
    data = _empty_graph()
    data["topics"] = [{"id": "T", "name": "T", "weight": 0.5}]
    data["applications"] = [{"id": "PUB", "name": "PUB"}, {"id": "SUB", "name": "SUB"}]
    data["relationships"]["publishes_to"] = [{"from": "PUB", "to": "T"}]
    data["relationships"]["subscribes_to"] = [{"from": "SUB", "to": "T"}]

    arcs = _arcs(_build(data))
    assert ("SUB", "PUB", "app_to_app") in arcs
    # The Topic is never a DEPENDS_ON endpoint — this was the original bug.
    assert not any("T" in (s, t) for s, t, _ in arcs)


def test_rule2_participant_depends_on_routing_broker():
    data = _empty_graph()
    data["topics"] = [{"id": "T", "name": "T", "weight": 0.5}]
    data["brokers"] = [{"id": "B", "name": "B"}]
    data["applications"] = [{"id": "PUB", "name": "PUB"}]
    data["relationships"]["publishes_to"] = [{"from": "PUB", "to": "T"}]
    data["relationships"]["routes"] = [{"from": "B", "to": "T"}]

    assert ("PUB", "B", "app_to_broker") in _arcs(_build(data))


def test_rules3_and_4_lift_onto_hosting_nodes():
    data = _empty_graph()
    data["topics"] = [{"id": "T", "name": "T", "weight": 0.5}]
    data["brokers"] = [{"id": "B", "name": "B"}]
    data["nodes"] = [{"id": "N1", "name": "N1"}, {"id": "N2", "name": "N2"}]
    data["applications"] = [{"id": "PUB", "name": "PUB"}, {"id": "SUB", "name": "SUB"}]
    data["relationships"]["publishes_to"] = [{"from": "PUB", "to": "T"}]
    data["relationships"]["subscribes_to"] = [{"from": "SUB", "to": "T"}]
    data["relationships"]["routes"] = [{"from": "B", "to": "T"}]
    data["relationships"]["runs_on"] = [
        {"from": "PUB", "to": "N1"}, {"from": "SUB", "to": "N2"}, {"from": "B", "to": "N1"},
    ]

    arcs = _arcs(_build(data))
    assert ("N2", "N1", "node_to_node") in arcs       # SUB@N2 depends on PUB@N1
    assert ("N2", "B", "node_to_broker") in arcs
    # Same node on both ends is not a node_to_node dependency.
    assert ("N1", "N1", "node_to_node") not in arcs


def test_rule5_app_to_lib_uses_harmonic_coupling():
    data = _empty_graph()
    data["applications"] = [{"id": "A", "name": "A"}]
    data["libraries"] = [{"id": "L", "name": "L"}]
    data["relationships"]["uses"] = [{"from": "A", "to": "L"}]

    g = _build(data)
    assert ("A", "L", "app_to_lib") in _arcs(g)
    w_a = g.components["A"].weight
    w_l = g.components["L"].weight
    assert _weight_of(g, "A", "L", "app_to_lib") == pytest.approx(
        2.0 * w_a * w_l / (w_a + w_l), abs=1e-9
    )


def test_rule6_colocated_brokers_are_bidirectional():
    data = _empty_graph()
    data["nodes"] = [{"id": "N", "name": "N"}]
    data["brokers"] = [{"id": "B1", "name": "B1"}, {"id": "B2", "name": "B2"}]
    data["relationships"]["runs_on"] = [{"from": "B1", "to": "N"}, {"from": "B2", "to": "N"}]

    arcs = _arcs(_build(data))
    assert ("B1", "B2", "broker_to_broker") in arcs
    assert ("B2", "B1", "broker_to_broker") in arcs


# ---------------------------------------------------------------------------
# Parity with the authoritative derivation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", PARITY_SCENARIOS)
def test_derivation_matches_repository_edge_set(scenario):
    """The simulation-side projection must agree arc-for-arc with MemoryRepository.

    Set equality on (dependent, dependency, type). Weights are deliberately allowed
    to diverge on pairs whose topics are reachable through more than one path kind:
    see test_collapse_does_not_double_count_shared_topics.
    """
    repo = MemoryRepository()
    with open(f"data/scenarios/{scenario}_system.json") as fh:
        ModelGraphUseCase(repo).execute(json.load(fh), clear=True)

    derived = _arcs(SimulationGraph(repo.get_graph_data(include_raw=True)))

    repo.derive_dependencies()
    reference = {(r["from"], r["to"], r["dependency_type"])
                 for r in repo.data["relationships"]["depends_on"]}

    assert derived == reference


def test_collapse_does_not_double_count_shared_topics():
    """A topic reachable directly *and* via a library enters the union once.

    `memory_repo._collapse_paths` flattens its per-kind dict into a multiset, so such
    a topic is counted twice there. `Neo4jRepository` MERGEs each kind separately and
    keeps the max, which is what this derivation matches. Measured divergence between
    the two adapters: 0 affected pairs on atm, 64 on microservices (dw <= 0.248), 139
    on av (dw <= 0.374).
    """
    data = _empty_graph()
    data["topics"] = [{"id": "T", "name": "T", "weight": 0.5}]
    data["applications"] = [{"id": "PUB", "name": "PUB"}, {"id": "SUB", "name": "SUB"}]
    data["libraries"] = [{"id": "L", "name": "L"}]
    # SUB reaches T both directly and through the library L that it uses.
    data["relationships"]["publishes_to"] = [{"from": "PUB", "to": "T"}]
    data["relationships"]["subscribes_to"] = [
        {"from": "SUB", "to": "T"}, {"from": "L", "to": "T"},
    ]
    data["relationships"]["uses"] = [{"from": "SUB", "to": "L"}]

    g = _build(data)
    topic_weight = g.components["T"].weight
    assert _weight_of(g, "SUB", "PUB", "app_to_app") == pytest.approx(
        compute_effective_edge_weight([topic_weight]), abs=1e-9
    ), "shared topic entered the probabilistic union more than once"


def test_derivation_ignores_failed_edges():
    """IM(v) asks about the intact architecture, so severed links must not change it."""
    repo = MemoryRepository()
    with open("data/scenarios/atm_system.json") as fh:
        ModelGraphUseCase(repo).execute(json.load(fh), clear=True)
    graph_data = repo.get_graph_data(include_raw=True)

    before = _arcs(SimulationGraph(graph_data))

    severed = SimulationGraph(graph_data)
    topic, subs = next(iter(severed._subscribers.items()))
    severed._failed_edges.add((subs[0][0], topic))
    severed._dependency_edges = None  # force a fresh derivation

    assert _arcs(severed) == before
