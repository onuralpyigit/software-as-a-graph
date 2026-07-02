
import pytest
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from saag.simulation.models import FailureScenario, FailureMode, CascadeRule, ImpactMetrics
from saag.simulation.failure_simulator import FailureSimulator
from saag.simulation.graph import SimulationGraph
from saag.core.models import GraphData, ComponentData, EdgeData

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
    """Single publisher degraded (0.5) > threshold (0.2) -> Topic lives."""
    graph = create_test_graph(n_publishers=1)
    sim = FailureSimulator(graph)

    # Degrade the only publisher
    scenario = FailureScenario(target_ids=["Pub1"], failure_mode=FailureMode.DEGRADED)
    res = sim.simulate(scenario)

    # avg_pub_impact = 0.5 < (1 - 0.2) = 0.8. Topic1 should NOT be in failed_set.
    assert "Topic1" not in res.cascaded_failures
    assert "Sub1" not in res.cascaded_failures

def test_degraded_crushed_multi_publisher():
    """
    5 publishers. 4 fail, 1 lives.
    avg_pub_impact = (1.0*4 + 0) / 5 = 0.8 >= (1 - 0.2) -> Topic fails.

    Canonical propagation_threshold is 0.2 (FailureSimulator default, matching
    the paper's committed default); the starvation boundary is therefore
    avg_pub_impact >= 0.8, not the pre-fix 0.3-threshold boundary of 0.7.
    """
    graph = create_test_graph(n_publishers=5)
    sim = FailureSimulator(graph)

    # Scenario: 4 of 5 publishers fail (Crash)
    scenario = FailureScenario(target_ids=["Pub1", "Pub2", "Pub3", "Pub4"], failure_mode=FailureMode.CRASH)
    res = sim.simulate(scenario)

    # avg_pub_impact = 0.8 >= 0.8. Topic1 should fail.
    assert "Topic1" in res.cascaded_failures
    assert "Sub1" in res.cascaded_failures

    # Re-test: 3 of 5 publishers fail.
    # avg_pub_impact = 0.6 < 0.8. Topic1 lives.
    scenario2 = FailureScenario(target_ids=["Pub1", "Pub2", "Pub3"], failure_mode=FailureMode.CRASH)
    res2 = sim.simulate(scenario2)
    assert "Topic1" not in res2.cascaded_failures

def test_degraded_starvation_boundary():
    """
    Boundary check around the canonical propagation_threshold = 0.2
    (starvation fires when avg_pub_impact >= 1 - 0.2 = 0.8).

    5 publishers. Fail 4. avg_pub_impact = 4/5 = 0.8 >= 0.8 -> Fails.
    5 publishers. Fail 3. avg_pub_impact = 3/5 = 0.6 < 0.8 -> Lives.
    """
    # 5 pubs, fail 4
    graph = create_test_graph(n_publishers=5)
    sim = FailureSimulator(graph)
    res = sim.simulate(FailureScenario(target_ids=["Pub1", "Pub2", "Pub3", "Pub4"], failure_mode=FailureMode.CRASH))
    assert "Topic1" in res.cascaded_failures

    # 5 pubs, fail 3
    graph3 = create_test_graph(n_publishers=5)
    sim3 = FailureSimulator(graph3)
    res3 = sim3.simulate(FailureScenario(target_ids=["Pub1", "Pub2", "Pub3"], failure_mode=FailureMode.CRASH))
    assert "Topic1" not in res3.cascaded_failures

if __name__ == "__main__":
    test_degraded_single_publisher()
    test_degraded_crushed_multi_publisher()
    test_degraded_starvation_boundary()
