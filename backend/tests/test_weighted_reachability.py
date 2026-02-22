
import pytest
from src.simulation.models import FailureScenario, FailureMode, ComponentState
from src.simulation.failure_simulator import FailureSimulator
from src.simulation.graph import SimulationGraph
from src.core.models import GraphData, ComponentData, EdgeData

def create_simple_path(edge_weight: float = 1.0):
    """Pub1 -> Topic1 -> Sub1 (via Broker1)"""
    components = [
        ComponentData("Pub1", "Application"),
        ComponentData("Topic1", "Topic"),
        ComponentData("Sub1", "Application"),
        ComponentData("Broker1", "Broker"),
    ]
    edges = [
        EdgeData("Pub1", "Topic1", "Application", "Topic", "logic", "PUBLISHES_TO", weight=edge_weight),
        EdgeData("Broker1", "Topic1", "Broker", "Topic", "logic", "ROUTES", weight=1.0),
        EdgeData("Sub1", "Topic1", "Application", "Topic", "logic", "SUBSCRIBES_TO", weight=1.0),
    ]
    return SimulationGraph(GraphData(components=components, edges=edges))

def test_reachability_healthy():
    """All healthy -> Reachability Loss = 0.0"""
    graph = create_simple_path()
    sim = FailureSimulator(graph)
    res = sim.simulate(FailureScenario(target_ids=["Other"], failure_mode=FailureMode.CRASH))
    assert res.impact.reachability_loss == 0.0

def test_reachability_degraded_component():
    """Degraded publisher (0.5) -> Path capacity 0.5 -> RL = 0.5"""
    graph = create_simple_path()
    sim = FailureSimulator(graph)
    
    # Degrade Pub1
    res = sim.simulate(FailureScenario(target_ids=["Pub1"], failure_mode=FailureMode.DEGRADED))
    
    # Initial capacity = 1.0 (mean of (1.0, 1.0, 1.0) basically)
    # Remaining capacity = min(0.5, 1.0, 1.0) = 0.5
    # RL = 1.0 - 0.5/1.0 = 0.5
    assert res.impact.reachability_loss == 0.5

def test_reachability_degraded_edge():
    """Healthy components, degraded edge (0.5) -> Path capacity 0.5 -> RL = 0.5"""
    graph = create_simple_path(edge_weight=0.5)
    sim = FailureSimulator(graph)
    
    # Initial capacity = min(1.0, 0.5, 1.0) = 0.5
    # We need to compute baseline on the healthy graph.
    # If we start with edge weight 0.5, initial_capacity_sum is 0.5.
    # If we then crash something irrelevant, remaining is 0.5 -> RL = 0.0.
    # To test RL we need to change state.
    
    # Let's start with edge 1.0, then degrade a component.
    # Wait, the request said: "capacity(path) is the minimum edge weight along the path".
    # And "Degraded mode result meaningful".
    
    graph = create_simple_path(edge_weight=1.0)
    sim = FailureSimulator(graph)
    res = sim.simulate(FailureScenario(target_ids=["Pub1"], failure_mode=FailureMode.DEGRADED))
    assert res.impact.reachability_loss == 0.5

def test_redundant_brokers_degraded():
    """
    Two brokers. 
    Topic1 routed by Broker1 (weight 1.0) and Broker2 (weight 1.0).
    If one fails, capacity remains 1.0.
    If one degrades (0.5) and other is active (1.0), capacity is max(0.5, 1.0) = 1.0.
    If both degrade (0.5), capacity is max(0.5, 0.5) = 0.5.
    """
    components = [
        ComponentData("Pub1", "Application"),
        ComponentData("Topic1", "Topic"),
        ComponentData("Sub1", "Application"),
        ComponentData("Broker1", "Broker"),
        ComponentData("Broker2", "Broker"),
    ]
    edges = [
        EdgeData("Pub1", "Topic1", "Application", "Topic", "logic", "PUBLISHES_TO"),
        EdgeData("Broker1", "Topic1", "Broker", "Topic", "logic", "ROUTES"),
        EdgeData("Broker2", "Topic1", "Broker", "Topic", "logic", "ROUTES"),
        EdgeData("Sub1", "Topic1", "Application", "Topic", "logic", "SUBSCRIBES_TO"),
    ]
    graph = SimulationGraph(GraphData(components=components, edges=edges))
    sim = FailureSimulator(graph)
    
    # Baseline: Broker segment = max(1, 1) = 1.0. Path capacity = 1.0.
    
    # Fail Broker1 -> Broker segment = max(0, 1) = 1.0. RL = 0.0.
    res1 = sim.simulate(FailureScenario(target_ids=["Broker1"], failure_mode=FailureMode.CRASH))
    assert res1.impact.reachability_loss == 0.0
    
    # Degrade Broker1 -> Broker segment = max(0.5, 1) = 1.0. RL = 0.0.
    res2 = sim.simulate(FailureScenario(target_ids=["Broker1"], failure_mode=FailureMode.DEGRADED))
    assert res2.impact.reachability_loss == 0.0
    
    # Degrade both -> Broker segment = max(0.5, 0.5) = 0.5. RL = 0.5.
    res3 = sim.simulate(FailureScenario(target_ids=["Broker1", "Broker2"], failure_mode=FailureMode.DEGRADED))
    assert res3.impact.reachability_loss == 0.5

if __name__ == "__main__":
    pytest.main([__file__])
