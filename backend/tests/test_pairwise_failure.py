
import pytest
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.simulation.models import FailureScenario, ImpactMetrics
from src.simulation.failure_simulator import FailureSimulator
from src.simulation.graph import SimulationGraph
from src.core.models import GraphData, ComponentData, EdgeData

def test_pairwise_logic():
    """
    Verify that FailureSimulator correctly handles multiple initial targets
    and that pairwise impact can be greater than single impact.
    """
    # Create GraphData
    components = [
        ComponentData(id="Node1", component_type="Node"),
        ComponentData(id="Node2", component_type="Node"),
        ComponentData(id="App1", component_type="Application"),
        ComponentData(id="App2", component_type="Application"),
        ComponentData(id="Broker1", component_type="Broker"),
        ComponentData(id="T1", component_type="Topic"),
    ]
    edges = [
        EdgeData("App1", "Node1", "Application", "Node", "structural", "RUNS_ON"),
        EdgeData("App2", "Node2", "Application", "Node", "structural", "RUNS_ON"),
        EdgeData("App1", "T1", "Application", "Topic", "logic", "PUBLISHES_TO"),
        EdgeData("App2", "T1", "Application", "Topic", "logic", "SUBSCRIBES_TO"),
        EdgeData("Broker1", "T1", "Broker", "Topic", "logic", "ROUTES"),
    ]
    
    graph_data = GraphData(components=components, edges=edges)
    graph = SimulationGraph(graph_data)
    sim = FailureSimulator(graph)
    
    # 1. Test single failure (Node1)
    res1 = sim.simulate(FailureScenario(target_ids=["Node1"]))
    assert "App1" in res1.cascaded_failures
    assert "T1" in res1.cascaded_failures # Publisher starvation
    impact1 = res1.impact.composite_impact
    
    # 2. Test another single failure (Node2)
    res2 = sim.simulate(FailureScenario(target_ids=["Node2"]))
    assert "App2" in res2.cascaded_failures
    assert "T1" in res2.cascaded_failures # Subscriber starvation (now implemented)
    impact2 = res2.impact.composite_impact
    
    # 3. Test pairwise failure
    res_pair = sim.simulate(FailureScenario(target_ids=["Node1", "Node2"]))
    assert "App1" in res_pair.cascaded_failures
    assert "App2" in res_pair.cascaded_failures
    assert "T1" in res_pair.cascaded_failures
    impact_pair = res_pair.impact.composite_impact
    
    print(f"\nImpact Node1: {impact1:.4f}")
    print(f"Impact Node2: {impact2:.4f}")
    print(f"Impact Pair:  {impact_pair:.4f}")
    
    assert impact_pair >= impact1
    assert impact_pair >= impact2

def test_superadditivity_scenario():
    """
    Scenario where I(v1, v2) >> I(v1) + I(v2).
    Two redundant bridges. Individually they don't break anything.
    Together they isolate a whole segment.
    """
    components = [
        ComponentData("SegA", "Node"),
        ComponentData("SegB", "Node"),
        ComponentData("Bridge1", "Node"),
        ComponentData("Bridge2", "Node"),
        ComponentData("AppA", "Application"),
        ComponentData("AppB", "Application"),
        ComponentData("T", "Topic"),
        ComponentData("B1", "Broker"),
    ]
    
    # We use bidirectional CONNECTS_TO to ensure redundancy works correctly
    # in the simulation graph connect component logic.
    edges = [
        EdgeData("AppA", "SegA", "Application", "Node", "structural", "RUNS_ON"),
        EdgeData("AppB", "SegB", "Application", "Node", "structural", "RUNS_ON"),
        
        # Segment A to Bridges
        EdgeData("SegA", "Bridge1", "Node", "Node", "network", "CONNECTS_TO"),
        EdgeData("Bridge1", "SegA", "Node", "Node", "network", "CONNECTS_TO"),
        EdgeData("SegA", "Bridge2", "Node", "Node", "network", "CONNECTS_TO"),
        EdgeData("Bridge2", "SegA", "Node", "Node", "network", "CONNECTS_TO"),
        
        # Bridges to Segment B
        EdgeData("Bridge1", "SegB", "Node", "Node", "network", "CONNECTS_TO"),
        EdgeData("SegB", "Bridge1", "Node", "Node", "network", "CONNECTS_TO"),
        EdgeData("Bridge2", "SegB", "Node", "Node", "network", "CONNECTS_TO"),
        EdgeData("SegB", "Bridge2", "Node", "Node", "network", "CONNECTS_TO"),
        
        EdgeData("AppA", "T", "Application", "Topic", "logic", "PUBLISHES_TO"),
        EdgeData("AppB", "T", "Application", "Topic", "logic", "SUBSCRIBES_TO"),
        EdgeData("B1", "SegA", "Broker", "Node", "structural", "RUNS_ON"),
        EdgeData("B1", "T", "Broker", "Topic", "logic", "ROUTES"),
    ]
    
    graph_data = GraphData(components=components, edges=edges)
    graph = SimulationGraph(graph_data)
    sim = FailureSimulator(graph)
    
    # Impact of Bridge1
    res1 = sim.simulate(FailureScenario(target_ids=["Bridge1"]))
    # SegA and SegB still connected via Bridge2.
    assert res1.impact.reachability_loss == 0.0
    
    # Impact of Bridge2
    res2 = sim.simulate(FailureScenario(target_ids=["Bridge2"]))
    # SegA and SegB still connected via Bridge1.
    assert res2.impact.reachability_loss == 0.0
    
    # Joint Impact
    res_pair = sim.simulate(FailureScenario(target_ids=["Bridge1", "Bridge2"]))
    # Network partition between SegA and SegB. AppA -> T -> AppB path broken.
    assert res_pair.impact.reachability_loss > 0.0
    
    print(f"\nSuperadditivity Detection:")
    print(f"  I(Bridge1):    {res1.impact.composite_impact:.4f}")
    print(f"  I(Bridge2):    {res2.impact.composite_impact:.4f}")
    print(f"  I(B1+B2):      {res_pair.impact.composite_impact:.4f} (DETECTED)")
    
    assert res_pair.impact.composite_impact > (res1.impact.composite_impact + res2.impact.composite_impact)

if __name__ == "__main__":
    test_pairwise_logic()
    test_superadditivity_scenario()
