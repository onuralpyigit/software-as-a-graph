import pytest
from src.core.graph_exporter import GraphData, ComponentData, EdgeData
from src.simulation.simulation_graph import SimulationGraph
from src.simulation.event_simulator import EventSimulator, EventScenario
from src.simulation.failure_simulator import FailureSimulator, FailureScenario

@pytest.fixture
def raw_graph_data():
    return GraphData(
        components=[
            ComponentData("App1", "Application"),
            ComponentData("App2", "Application"),
            ComponentData("Topic1", "Topic"),
            ComponentData("Node1", "Node"),
        ],
        edges=[
            # App1 publishes to Topic1
            EdgeData("App1", "Topic1", "Application", "Topic", "PUBLISHES_TO"),
            # App2 subscribes to Topic1 (App2 -> Topic1 in graph structure usually means Sub?)
            # Wait, usually (App)-[:SUBSCRIBES_TO]->(Topic)
            EdgeData("App2", "Topic1", "Application", "Topic", "SUBSCRIBES_TO"),
            # App1 runs on Node1
            EdgeData("App1", "Node1", "Application", "Node", "RUNS_ON"),
        ]
    )

def test_event_simulation(raw_graph_data):
    graph = SimulationGraph(raw_graph_data)
    sim = EventSimulator(graph)
    
    # App1 Publishes to Topic1. App2 Subscribes to Topic1.
    # Event from App1 should reach App2.
    res = sim.simulate(EventScenario("App1", "test"))
    
    assert "Topic1" in res.affected_topics
    assert "App2" in res.reached_subscribers
    assert res.hops == 2

def test_failure_simulation(raw_graph_data):
    graph = SimulationGraph(raw_graph_data)
    sim = FailureSimulator(graph)
    
    # Kill Node1. App1 runs on Node1. App1 should fail.
    res = sim.simulate(FailureScenario("Node1", "test"))
    
    assert "App1" in res.cascaded_failures
    assert res.impact_counts["Application"] == 1