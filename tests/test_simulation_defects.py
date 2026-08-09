"""
Regression tests for defects found while simplifying the Simulate capability.

Each test here fails against the pre-fix implementation. They are grouped in one
file because they are unrelated one-line defects rather than a coherent feature.
"""

import pytest

from saag.core.models import GraphData, ComponentData, EdgeData
from saag.simulation.graph import SimulationGraph
from saag.simulation.event_simulator import EventSimulator
from saag.simulation.failure_simulator import FailureSimulator
from saag.simulation.processor import ComplexityProcessor
from saag.simulation.models import EventScenario, EventType, FailureScenario


def _pubsub_graph():
    """App1 -> Topic1 -> Broker1 -> Sub1, plus a second publisher on Topic1."""
    return GraphData(
        components=[
            ComponentData(id="App1", component_type="Application", properties={"layer": "app"}),
            ComponentData(id="App2", component_type="Application", properties={"layer": "app"}),
            ComponentData(id="Topic1", component_type="Topic", properties={"layer": "mw"}),
            ComponentData(id="Broker1", component_type="Broker", properties={"layer": "infra"}),
            ComponentData(id="Sub1", component_type="Application", properties={"layer": "app"}),
        ],
        edges=[
            EdgeData(source_id="App1", target_id="Topic1", source_type="Application",
                     target_type="Topic", relation_type="PUBLISHES_TO",
                     dependency_type="pubsub", weight=1.0),
            EdgeData(source_id="App2", target_id="Topic1", source_type="Application",
                     target_type="Topic", relation_type="PUBLISHES_TO",
                     dependency_type="pubsub", weight=1.0),
            EdgeData(source_id="Sub1", target_id="Topic1", source_type="Application",
                     target_type="Topic", relation_type="SUBSCRIBES_TO",
                     dependency_type="pubsub", weight=1.0),
            EdgeData(source_id="Topic1", target_id="Broker1", source_type="Topic",
                     target_type="Broker", relation_type="ROUTES",
                     dependency_type="routing", weight=1.0),
        ],
    )


# --- B1: EventType was missing the two members event_simulator dispatches on ---

def test_event_type_has_poisson_members():
    assert EventType.FAIL_COMPONENT.value == "fail_component"
    assert EventType.RECOVER_COMPONENT.value == "recover_component"


def test_poisson_failure_injection_runs():
    """Previously raised AttributeError inside _schedule_poisson_failures."""
    sim = EventSimulator(SimulationGraph(graph_data=_pubsub_graph()))
    result = sim.simulate(EventScenario(
        source_app="App1", num_messages=50, duration=10.0,
        seed=42, failure_rate=2.0, mean_recovery_time=1.0,
    ))
    assert any(e["event"] == "fail" for e in result.poisson_failure_log)
    assert any(e["event"] == "recover" for e in result.poisson_failure_log)


def test_failed_broker_does_not_deliver_directly():
    """
    A brokered topic whose brokers have all failed must drop, not fall through
    to the brokerless direct-delivery path.
    """
    sim = EventSimulator(SimulationGraph(graph_data=_pubsub_graph()))
    result = sim.simulate(EventScenario(
        source_app="App1", num_messages=200, duration=30.0,
        seed=21, failure_rate=10.0, failure_targets=["Broker1"],
        mean_recovery_time=0.0,
    ))
    assert result.metrics.messages_dropped > 0
    assert result.metrics.delivery_rate < 100.0


def test_brokerless_topic_still_delivers_directly():
    """The DDS-style path must survive the fix above."""
    data = _pubsub_graph()
    data.components = [c for c in data.components if c.id != "Broker1"]
    data.edges = [e for e in data.edges if e.target_id != "Broker1"]

    sim = EventSimulator(SimulationGraph(graph_data=data))
    result = sim.simulate(EventScenario(source_app="App1", num_messages=10, duration=5.0))
    assert result.metrics.messages_delivered == 10


# --- B3: ComplexityProcessor read a non-existent SimulationGraph attribute ---

def test_complexity_processor_folds_in_library_complexity():
    """Previously raised AttributeError on graph.out_edges whenever beta > 0."""
    data = GraphData(
        components=[
            ComponentData(id="App1", component_type="Application",
                          properties={"layer": "app", "complexity": 10.0}),
            ComponentData(id="App2", component_type="Application",
                          properties={"layer": "app", "complexity": 20.0}),
            ComponentData(id="Lib1", component_type="Library",
                          properties={"layer": "app", "complexity": 50.0}),
        ],
        edges=[
            EdgeData(source_id="App1", target_id="Lib1", source_type="Application",
                     target_type="Library", relation_type="USES",
                     dependency_type="library", weight=1.0),
        ],
    )
    graph = SimulationGraph(graph_data=data)
    scenario = EventScenario(source_app="App1", base_processing_latency=0.001,
                             complexity_scale_factor=1.0, library_complexity_weight=0.3)

    ComplexityProcessor(graph, scenario).process()

    app1 = graph.components["App1"].properties["processing_latency"]
    app2 = graph.components["App2"].properties["processing_latency"]
    # App1 is the least complex app (c_norm=0) but carries Lib1's penalty;
    # App2 is the most complex (c_norm=1) and uses nothing.
    assert app1 == pytest.approx(0.001 + 0.3 * 0.0)   # single library normalises to 0
    assert app2 == pytest.approx(0.002)


def test_get_used_libraries_is_inverse_of_get_uses_consumers():
    data = GraphData(
        components=[
            ComponentData(id="App1", component_type="Application", properties={"layer": "app"}),
            ComponentData(id="Lib1", component_type="Library", properties={"layer": "app"}),
        ],
        edges=[
            EdgeData(source_id="App1", target_id="Lib1", source_type="Application",
                     target_type="Library", relation_type="USES",
                     dependency_type="library", weight=1.0),
        ],
    )
    graph = SimulationGraph(graph_data=data)
    assert graph.get_used_libraries("App1") == ["Lib1"]
    assert graph.get_uses_consumers("Lib1") == ["App1"]
    assert graph.get_used_libraries("Lib1") == []


# --- B5: get_broker_routing returned the topic-keyed index ---

def test_get_broker_routing_is_keyed_by_broker():
    graph = SimulationGraph(graph_data=_pubsub_graph())
    routing = graph.get_broker_routing()
    assert routing == {"Broker1": ["Topic1"]}


# --- B4: SimulationMode.CLASSIFY forwarded `edges` into a method without **kwargs ---

def _classify_topology():
    return {
        "nodes": [{"id": "N0", "name": "host0"}],
        "brokers": [{"id": "B0", "name": "broker0"}],
        "topics": [{"id": "T0", "name": "/t0", "size": 64,
                    "qos": {"reliability": "RELIABLE", "durability": "PERSISTENT",
                            "transport_priority": "HIGH"}}],
        "applications": [{"id": "P0", "name": "pub0", "role": "pub"},
                         {"id": "S0", "name": "sub0", "role": "sub"}],
        "libraries": [],
        "relationships": {
            "runs_on": [{"from": c, "to": "N0"} for c in ("P0", "S0", "B0")],
            "routes": [{"from": "B0", "to": "T0"}],
            "publishes_to": [{"from": "P0", "to": "T0"}],
            "subscribes_to": [{"from": "S0", "to": "T0"}],
            "connects_to": [],
            "uses": [],
        },
    }


@pytest.mark.parametrize("edges", [True, False])
def test_classify_mode_does_not_raise(edges):
    """Previously raised TypeError: unexpected keyword argument 'edges'."""
    from saag.infrastructure.memory_repo import MemoryRepository
    from saag.simulation.service import SimulationService
    from saag.usecases.simulate_graph import SimulateGraphUseCase
    from saag.usecases.models import SimulationMode

    repo = MemoryRepository()
    repo.save_graph(_classify_topology(), clear=True)
    usecase = SimulateGraphUseCase(SimulationService(repo))

    result = usecase.execute(layer="system", mode=SimulationMode.CLASSIFY, edges=edges)
    assert isinstance(result, list)


# --- B6: (id, weight) tuples were inserted into a Set[str] ---

def test_affected_publishers_counts_distinct_apps():
    """
    Topic1 has two publishers. Killing the only broker breaks the topic, so both
    publishers are affected — but each must be counted exactly once.
    """
    graph = SimulationGraph(graph_data=_pubsub_graph())
    sim = FailureSimulator(graph)
    result = sim.simulate(FailureScenario(target_ids=["Broker1"]))
    assert result.impact.affected_publishers == 2
    assert result.impact.affected_subscribers == 1


# --- FaultInjector: I*(v) must reflect the final cascade wave, not the
# in-loop `topic_loss` left over from the second-to-last wave. Sub1 is a
# subscriber of Topic1 (published solely by the injected App1) *and* a
# publisher of Topic2 (subscribed to by SubSub1). App1's failure orphans
# Topic1 in wave 0, which deterministically fails Sub1 in that same wave
# (propagation_threshold=0.2, feed loss=1.0). Sub1's own failure as a
# *publisher* of Topic2 is only visible once topic_loss is recomputed
# against the post-wave-0 failed_nodes state — the recomputation that a
# cascade_depth_limit of 1 (stop after wave 0) prevents from ever running
# in-loop. SubSub1's reported loss is the regression signal: 0.0 pre-fix,
# 1.0 once the final state is recomputed after the loop.

def test_fault_injector_impact_reflects_final_wave():
    import networkx as nx
    from saag.simulation.fault_injector import FaultInjector

    g = nx.DiGraph()
    for n in ("App1", "Sub1", "SubSub1"):
        g.add_node(n, type="Application")
    for n in ("Topic1", "Topic2"):
        g.add_node(n, type="Topic")
    g.add_edge("App1", "Topic1", type="PUBLISHES_TO", rate_hz=10.0)
    g.add_edge("Sub1", "Topic1", type="SUBSCRIBES_TO")
    g.add_edge("Sub1", "Topic2", type="PUBLISHES_TO", rate_hz=10.0)
    g.add_edge("SubSub1", "Topic2", type="SUBSCRIBES_TO")

    injector = FaultInjector(
        g, seeds=[42], cascade_depth_limit=1,
        propagation_threshold=0.2, qos_factor_mode="none",
    )
    result = injector.run(node_ids=["App1"])
    rec = result.records["App1"]

    assert "Sub1" in rec.impacted_subscriber_ids
    assert rec.per_subscriber_feed_loss.get("SubSub1", 0.0) == pytest.approx(1.0)


def test_fault_injector_impact_no_unbound_error_on_empty_cascade():
    """A node with no dependents and no pub/sub edges produces a one-wave
    cascade with zero impact; I*(v) must resolve cleanly rather than
    referencing a while-loop-local `topic_loss` that a differently-shaped
    graph could leave unset."""
    import networkx as nx
    from saag.simulation.fault_injector import FaultInjector

    g = nx.DiGraph()
    g.add_node("App1", type="Application")
    injector = FaultInjector(g, seeds=[42], propagation_threshold=0.2)
    result = injector.run(node_ids=["App1"])
    assert result.records["App1"].impact_score == 0.0
