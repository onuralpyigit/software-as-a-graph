
import pytest
from saag.simulation.models import FailureScenario, FailureMode, CascadeRule
from saag.simulation.failure_simulator import FailureSimulator
from saag.simulation.graph import SimulationGraph
from saag.core.models import GraphData, ComponentData, EdgeData

def test_partition_mode_skips_physical_cascade():
    """
    Test PARTITION mode skips Rule 1 (Physical Cascade).
    Graph:
        Node1 (Node)
        App1 (Application) --[RUNS_ON]--> Node1
        Topic1 (Topic)
        App1 --[PUBLISHES_TO]--> Topic1
        
    In CRASH mode: Node1 fails -> App1 fails (Physical) -> Topic1 fails (Logical).
    In PARTITION mode: Node1 fails (partitioned) -> App1 LIVES (no physical cascade).
    """
    components = [
        ComponentData("Node1", "Node"),
        ComponentData("App1", "Application"),
        ComponentData("Topic1", "Topic"),
        ComponentData("Broker1", "Broker"), 
    ]
    edges = [
        EdgeData("App1", "Node1", "Application", "Node", "infra", "RUNS_ON"),
        EdgeData("App1", "Topic1", "Application", "Topic", "logic", "PUBLISHES_TO"),
        EdgeData("Broker1", "Topic1", "Broker", "Topic", "logic", "ROUTES"),
    ]
    
    graph = SimulationGraph(GraphData(components=components, edges=edges))
    
    # CASE 1: CRASH mode
    sim_crash = FailureSimulator(graph)
    res_crash = sim_crash.simulate(FailureScenario(target_ids=["Node1"], failure_mode=FailureMode.CRASH))
    assert "App1" in res_crash.cascaded_failures
    
    # CASE 2: PARTITION mode
    graph.reset()
    sim_part = FailureSimulator(graph)
    scenario_part = FailureScenario(target_ids=["Node1"], failure_mode=FailureMode.PARTITION)
    res_part = sim_part.simulate(scenario_part)
    
    # App1 should NOT be failed in PARTITION mode because Rule 1 is skipped
    assert "App1" not in res_part.cascaded_failures
    assert len(res_part.cascaded_failures) == 0

if __name__ == "__main__":
    pytest.main([__file__])
