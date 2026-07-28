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
