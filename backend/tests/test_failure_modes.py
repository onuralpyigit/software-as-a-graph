
import pytest
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.simulation.models import FailureScenario, FailureMode, CascadeRule, ImpactMetrics
from src.simulation.failure_simulator import FailureSimulator
from src.simulation.graph import SimulationGraph
from src.core.models import GraphData, ComponentData, EdgeData

def create_test_graph(n_publishers: int):
    """Create a graph with N publishers to 1 topic and 1 subscriber."""
    components = [
        ComponentData("Topic1", "Topic"),
        ComponentData("Sub1", "Application"),
        ComponentData("Broker1", "Broker"),
    ]
    edges = [
        EdgeData("Broker1", "Topic1", "Broker", "Topic", "logic", "ROUTES"),
        EdgeData("Sub1", "Topic1", "Application", "Topic", "logic", "SUBSCRIBES_TO"),
    ]
    
    for i in range(1, n_publishers + 1):
        pid = f"Pub{i}"
        components.append(ComponentData(pid, "Application"))
        edges.append(EdgeData(pid, "Topic1", "Application", "Topic", "logic", "PUBLISHES_TO"))
    
    return SimulationGraph(GraphData(components=components, edges=edges))

def test_degraded_single_publisher():
    """Single publisher degraded (0.5) > threshold (0.3) -> Topic lives."""
    graph = create_test_graph(n_publishers=1)
    sim = FailureSimulator(graph)
    
    # Degrade the only publisher
    scenario = FailureScenario(target_ids=["Pub1"], failure_mode=FailureMode.DEGRADED)
    res = sim.simulate(scenario)
    
    # SL = 0.5 >= 0.3. Topic1 should NOT be in failed_set.
    assert "Topic1" not in res.cascaded_failures
    assert "Sub1" not in res.cascaded_failures

def test_degraded_crushed_multi_publisher():
    """
    4 publishers. 3 fail, 1 degraded. 
    SL = (0.5 + 0 + 0 + 0) / 4 = 0.125 < 0.3 -> Topic fails.
    """
    graph = create_test_graph(n_publishers=4)
    sim = FailureSimulator(graph)
    
    # Fail 3, Degrade 1
    # Note: simulate currently takes one failure_mode for ALL target_ids.
    # To test mixed, we rely on the fact that propagation handles performance.
    # We'll fail 3 and degrade 1 by using multiple targets with a custom scenario 
    # if we had one, but currently simulate() applies scenario.failure_mode to all.
    # So let's simulate a case where 3 publishers are failed and we fail the 4th?
    # No, let's just use 4 publishers and Fail 3.
    # SL = (1.0 + 0 + 0 + 0) / 4 = 0.25 < 0.3 -> Topic fails.
    
    # Scenario: 3 publishers fail (Crash)
    scenario = FailureScenario(target_ids=["Pub1", "Pub2", "Pub3"], failure_mode=FailureMode.CRASH)
    res = sim.simulate(scenario)
    
    # SL = 0.25 < 0.3. Topic1 should fail.
    assert "Topic1" in res.cascaded_failures
    assert "Sub1" in res.cascaded_failures
    
    # Re-test: 2 publishers fail.
    # SL = (1.0 + 1.0 + 0 + 0) / 4 = 0.5 >= 0.3. Topic1 lives.
    scenario2 = FailureScenario(target_ids=["Pub1", "Pub2"], failure_mode=FailureMode.CRASH)
    res2 = sim.simulate(scenario2)
    assert "Topic1" not in res2.cascaded_failures

def test_degraded_starvation_boundary():
    """
    3 publishers. 2 degraded (0.5 each).
    SL = (0.5 + 0.5 + 1.0) / 3 = 0.66 > 0.3 -> Topic lives.
    Wait, let's try to get closer to 0.3.
    3 publishers. Fail 2. SL = 1/3 = 0.33 > 0.3 -> Lives.
    4 publishers. Fail 3. SL = 1/4 = 0.25 < 0.3 -> Fails.
    """
    # 4 pubs, fail 3
    graph = create_test_graph(n_publishers=4)
    sim = FailureSimulator(graph)
    res = sim.simulate(FailureScenario(target_ids=["Pub1", "Pub2", "Pub3"], failure_mode=FailureMode.CRASH))
    assert "Topic1" in res.cascaded_failures
    
    # 3 pubs, fail 2
    graph3 = create_test_graph(n_publishers=3)
    sim3 = FailureSimulator(graph3)
    res3 = sim3.simulate(FailureScenario(target_ids=["Pub1", "Pub2"], failure_mode=FailureMode.CRASH))
    assert "Topic1" not in res3.cascaded_failures

if __name__ == "__main__":
    test_degraded_single_publisher()
    test_degraded_crushed_multi_publisher()
    test_degraded_starvation_boundary()
