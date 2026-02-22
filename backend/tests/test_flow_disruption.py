
import pytest
from src.core.models import GraphData, ComponentData, EdgeData
from src.simulation.graph import SimulationGraph
from src.simulation.event_simulator import EventSimulator
from src.simulation.failure_simulator import FailureSimulator
from src.simulation.models import (
    EventScenario, 
    FailureScenario, 
    ComponentState,
    CascadeRule
)

@pytest.fixture
def sample_graph_data():
    return GraphData(
        components=[
            ComponentData(id="App1", component_type="Application", properties={"layer": "app"}),
            ComponentData(id="App2", component_type="Application", properties={"layer": "app"}),
            ComponentData(id="Topic1", component_type="Topic", properties={"layer": "mw"}),
            ComponentData(id="Broker1", component_type="Broker", properties={"layer": "infra"}),
            ComponentData(id="Node1", component_type="Node", properties={"layer": "infra"})
        ],
        edges=[
            EdgeData(source_id="App1", target_id="Topic1", source_type="Application", target_type="Topic", 
                     relation_type="PUBLISHES_TO", dependency_type="pubsub", weight=1.0),
            EdgeData(source_id="App2", target_id="Topic1", source_type="Application", target_type="Topic", 
                     relation_type="SUBSCRIBES_TO", dependency_type="pubsub", weight=1.0),
            EdgeData(source_id="Topic1", target_id="Broker1", source_type="Topic", target_type="Broker", 
                     relation_type="ROUTES", dependency_type="routing", weight=1.0),
            EdgeData(source_id="App1", target_id="Node1", source_type="Application", target_type="Node", 
                     relation_type="RUNS_ON", dependency_type="deployment"),
            EdgeData(source_id="Broker1", target_id="Node1", source_type="Broker", target_type="Node", 
                     relation_type="RUNS_ON", dependency_type="deployment")
        ]
    )

def test_flow_disruption_calculation(sample_graph_data):
    graph = SimulationGraph(graph_data=sample_graph_data)
    event_sim = EventSimulator(graph)
    fail_sim = FailureSimulator(graph)
    
    # 1. Run baseline event simulation
    # App1 publishes to Topic1, App2 subscribes to Topic1. Flow: (App1, Topic1, App2)
    scenario = EventScenario(source_app="App1", num_messages=10, duration=1.0)
    event_res = event_sim.simulate(scenario)
    
    assert len(event_res.successful_flows) == 1
    assert ("App1", "Topic1", "App2") in event_res.successful_flows
    
    # 2. Set baseline flows in failure simulator
    fail_sim.set_baseline_flows(event_res.successful_flows)
    
    # 3. Simulate failure of a critical component (Broker1)
    # This should break the flow (App1, Topic1, App2)
    failure_scenario = FailureScenario(target_ids=["Broker1"])
    failure_res = fail_sim.simulate(failure_scenario)
    
    # Flow Disruption should be 1.0 (100% of flows broken)
    assert failure_res.impact.flow_disruption == 1.0
    
    # Composite impact should include flow disruption
    # weights: reachability(0.35), fragmentation(0.25), throughput(0.25), flow_disruption(0.15)
    # fragmentation will be 0 (only 1 island), throughput loss will be 1.0, reachability loss will be 1.0
    # composite = 0.35*1 + 0.25*0 + 0.25*1 + 0.15*1 = 0.35 + 0.25 + 0.15 = 0.75
    assert failure_res.impact.composite_impact == pytest.approx(0.75)

def test_flow_disruption_survival(sample_graph_data):
    # Add a second redundant broker
    sample_graph_data.components.append(ComponentData(id="Broker2", component_type="Broker", properties={"layer": "infra"}))
    sample_graph_data.edges.append(EdgeData(source_id="Topic1", target_id="Broker2", source_type="Topic", target_type="Broker", 
                                          relation_type="ROUTES", dependency_type="routing", weight=1.0))
    sample_graph_data.edges.append(EdgeData(source_id="Broker2", target_id="Node1", source_type="Broker", target_type="Node", 
                                          relation_type="RUNS_ON", dependency_type="deployment"))
    
    graph = SimulationGraph(graph_data=sample_graph_data)
    event_sim = EventSimulator(graph)
    fail_sim = FailureSimulator(graph)
    
    # Baseline
    event_res = event_sim.simulate(EventScenario(source_app="App1", num_messages=10, duration=1.0))
    assert len(event_res.successful_flows) == 1
    fail_sim.set_baseline_flows(event_res.successful_flows)
    
    # Fail ONE broker - flow should survive because of Broker2
    failure_res = fail_sim.simulate(FailureScenario(target_ids=["Broker1"]))
    assert failure_res.impact.flow_disruption == 0.0
    
    # Fail BOTH brokers - flow should break
    # Using cascade: fail Node1 which hosts both brokers
    failure_res = fail_sim.simulate(FailureScenario(target_ids=["Node1"], cascade_rule=CascadeRule.PHYSICAL))
    assert failure_res.impact.flow_disruption == 1.0
