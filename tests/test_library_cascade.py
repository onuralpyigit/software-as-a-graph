
import pytest
from saag.simulation.models import FailureScenario, FailureMode, CascadeRule
from saag.simulation.failure_simulator import FailureSimulator
from saag.simulation.graph import SimulationGraph
from saag.core.models import GraphData, ComponentData, EdgeData

def test_library_cascade_rule():
    """
    Test Rule 4: Library failure -> Using applications fail.
    Graph:
        Lib1 (Library)
        App1 (Application) --[USES]--> Lib1
        App2 (Application) --[USES]--> Lib1
        Topic1 (Topic)
        App1 --[PUBLISHES_TO]--> Topic1
        App3 (Application) --[SUBSCRIBES_TO]--> Topic1
    
    If Lib1 fails:
        1. App1 and App2 fail (Rule 4)
        2. Topic1 fails (Rule 3 - App1 was only publisher)
        3. App3 fails (Rule 3 - Topic1 failed)
    """
    components = [
        ComponentData("Lib1", "Library"),
        ComponentData("App1", "Application"),
        ComponentData("App2", "Application"),
        ComponentData("App3", "Application"),
        ComponentData("Topic1", "Topic"),
        ComponentData("Broker1", "Broker"), 
    ]
    edges = [
        EdgeData("App1", "Lib1", "Application", "Library", "logic", "USES"),
        EdgeData("App2", "Lib1", "Application", "Library", "logic", "USES"),
        EdgeData("App1", "Topic1", "Application", "Topic", "logic", "PUBLISHES_TO"),
        EdgeData("App3", "Topic1", "Application", "Topic", "logic", "SUBSCRIBES_TO"),
        EdgeData("Broker1", "Topic1", "Broker", "Topic", "logic", "ROUTES"),
    ]
    
    graph = SimulationGraph(GraphData(components=components, edges=edges))
    sim = FailureSimulator(graph)
    
    # Fail Lib1
    scenario = FailureScenario(target_ids=["Lib1"], failure_mode=FailureMode.CRASH)
    res = sim.simulate(scenario)
    
    # Check cascaded failures
    assert "App1" in res.cascaded_failures
    assert "App2" in res.cascaded_failures
    assert "Topic1" in res.cascaded_failures
    assert "App3" in res.cascaded_failures
    
    # Check sequence
    lib_event = next(e for e in res.cascade_sequence if e.component_id == "Lib1")
    assert lib_event.depth == 0
    
    app1_event = next(e for e in res.cascade_sequence if e.component_id == "App1")
    assert app1_event.depth == 1
    assert app1_event.cause == "uses_library:Lib1"
    
    topic_event = next(e for e in res.cascade_sequence if e.component_id == "Topic1")
    assert topic_event.depth > 1
    assert "sl_starvation" in topic_event.cause

if __name__ == "__main__":
    pytest.main([__file__])
