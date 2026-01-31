import pytest
from src.domain.models import GraphData, ComponentData, EdgeData
from src.domain.models.simulation.graph import SimulationGraph
from src.domain.services.event_simulator import EventSimulator, EventScenario
from src.domain.services.failure_simulator import FailureSimulator, FailureScenario, ImpactMetrics

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
            EdgeData("App1", "Topic1", "Application", "Topic", "PUBLISHES_TO", "PUBLISHES_TO"),
            # App2 subscribes to Topic1
            EdgeData("App2", "Topic1", "Application", "Topic", "SUBSCRIBES_TO", "SUBSCRIBES_TO"),
            # App1 runs on Node1
            EdgeData("App1", "Node1", "Application", "Node", "RUNS_ON", "RUNS_ON"),
        ]
    )


def test_event_simulation(raw_graph_data):
    graph = SimulationGraph(graph_data=raw_graph_data)
    sim = EventSimulator(graph)
    
    # App1 Publishes to Topic1. App2 Subscribes to Topic1.
    # Event from App1 should reach App2.
    res = sim.simulate(EventScenario("App1", "test"))
    
    assert "Topic1" in res.affected_topics
    assert "App2" in res.reached_subscribers
    # Check that messages were published (hops is on Message, not EventResult)
    assert res.metrics.messages_published > 0


def test_failure_simulation(raw_graph_data):
    graph = SimulationGraph(graph_data=raw_graph_data)
    sim = FailureSimulator(graph)
    
    # Kill Node1. App1 runs on Node1. App1 should fail.
    res = sim.simulate(FailureScenario("Node1", "test"))
    
    assert "App1" in res.cascaded_failures
    # Fixed: use impact.cascade_by_type instead of impact_counts
    assert res.impact.cascade_by_type.get("Application", 0) >= 1


def test_configurable_impact_weights(raw_graph_data):
    """Test that impact weights are configurable."""
    # ImpactMetrics imported at top level now
    
    # Create metrics with losses
    metrics = ImpactMetrics(
        reachability_loss=0.5,
        fragmentation=0.5,
        throughput_loss=0.5,
    )
    
    # Default weights: 0.4, 0.3, 0.3
    default_impact = metrics.composite_impact
    assert default_impact == pytest.approx(0.5, abs=0.01)  # 0.4*0.5 + 0.3*0.5 + 0.3*0.5 = 0.5
    
    # Custom weights: emphasize reachability
    custom_metrics = ImpactMetrics(
        reachability_loss=0.5,
        fragmentation=0.5,
        throughput_loss=0.5,
        impact_weights={"reachability": 0.8, "fragmentation": 0.1, "throughput": 0.1}
    )
    custom_impact = custom_metrics.composite_impact
    # 0.8*0.5 + 0.1*0.5 + 0.1*0.5 = 0.5 (same total but different weights)
    assert custom_impact == pytest.approx(0.5, abs=0.01)
    
    # Different losses to show weight effect
    metrics1 = ImpactMetrics(reachability_loss=1.0, fragmentation=0.0, throughput_loss=0.0)
    metrics2 = ImpactMetrics(
        reachability_loss=1.0, 
        fragmentation=0.0, 
        throughput_loss=0.0,
        impact_weights={"reachability": 1.0, "fragmentation": 0.0, "throughput": 0.0}
    )
    assert metrics1.composite_impact == pytest.approx(0.4, abs=0.01)  # Default: 0.4*1.0
    assert metrics2.composite_impact == pytest.approx(1.0, abs=0.01)  # Custom: 1.0*1.0


def test_layer_filtering(raw_graph_data):
    """Test that layer filtering works as expected."""
    graph = SimulationGraph(graph_data=raw_graph_data)
    
    # System layer should include everything
    system_comps = graph.get_components_by_layer("system")
    assert "App1" in system_comps
    assert "Node1" in system_comps
    assert "Topic1" in system_comps
    
    # App layer should simulate Apps and Topics but analyze only Apps
    app_comps = graph.get_components_by_layer("app")
    assert "App1" in app_comps
    assert "Topic1" in app_comps
    
    analyze_apps = graph.get_analyze_components_by_layer("app")
    assert "App1" in analyze_apps
    assert "Topic1" not in analyze_apps  # Topics are part of graph but not analyzed primarily
    
    # Infra layer should simulate Nodes and hosted components but analyze only Nodes
    infra_comps = graph.get_components_by_layer("infra")
    assert "Node1" in infra_comps
    assert "App1" in infra_comps  # App1 runs on Node1, so it's included for connectivity
    
    analyze_infra = graph.get_analyze_components_by_layer("infra")
    assert "Node1" in analyze_infra
    assert "App1" not in analyze_infra