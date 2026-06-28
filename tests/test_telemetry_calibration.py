import pytest
from saag.simulation.models import (
    FailureScenario,
    FailureMode,
    CascadeRule,
    RuntimeTelemetryProfile,
)
from saag.simulation.failure_simulator import FailureSimulator
from saag.simulation.graph import SimulationGraph
from saag.core.models import GraphData, ComponentData, EdgeData


def create_calibration_test_graph():
    """
    Create a graph with:
    - 3 Applications (Pub1, Sub1, Sub2)
    - 2 Topics (Topic1, Topic2)
    - 1 Broker (Broker1)
    
    Connections:
    Pub1 -> PUBLISHES_TO -> Topic1
    Pub1 -> PUBLISHES_TO -> Topic2
    Broker1 -> ROUTES -> Topic1
    Broker1 -> ROUTES -> Topic2
    Sub1 -> SUBSCRIBES_TO -> Topic1
    Sub2 -> SUBSCRIBES_TO -> Topic2
    """
    components = [
        ComponentData("Pub1", "Application"),
        ComponentData("Sub1", "Application"),
        ComponentData("Sub2", "Application"),
        ComponentData("Topic1", "Topic"),
        ComponentData("Topic2", "Topic"),
        ComponentData("Broker1", "Broker"),
    ]
    edges = [
        EdgeData("Pub1", "Topic1", "Application", "Topic", "logic", "PUBLISHES_TO"),
        EdgeData("Pub1", "Topic2", "Application", "Topic", "logic", "PUBLISHES_TO"),
        EdgeData("Broker1", "Topic1", "Broker", "Topic", "logic", "ROUTES"),
        EdgeData("Broker1", "Topic2", "Broker", "Topic", "logic", "ROUTES"),
        EdgeData("Sub1", "Topic1", "Application", "Topic", "logic", "SUBSCRIBES_TO"),
        EdgeData("Sub2", "Topic2", "Application", "Topic", "logic", "SUBSCRIBES_TO"),
    ]
    return SimulationGraph(GraphData(components=components, edges=edges))


def test_default_backward_compatible():
    """Ensure that default instantiation behaves without errors (backward compatible)."""
    graph = create_calibration_test_graph()
    sim = FailureSimulator(graph)
    
    # Run a simple cascade simulation
    scenario = FailureScenario(target_ids=["Pub1"], failure_mode=FailureMode.CRASH)
    res = sim.simulate(scenario)
    
    # Pub1 failure cascades to Topic1 & Topic2, which starves Sub1
    assert "Topic1" in res.cascaded_failures
    assert "Topic2" in res.cascaded_failures
    assert "Sub1" in res.cascaded_failures


def test_edge_failure_calibration():
    """Ensure edge-specific failure probabilities from telemetry profile are respected."""
    graph = create_calibration_test_graph()
    
    # Set the Pub1 -> Topic1 edge cascade probability to 0.0 (no failure spreads)
    telemetry = RuntimeTelemetryProfile(
        edge_failure_correlation={
            ("Pub1", "Topic1"): 0.0,
        }
    )
    sim = FailureSimulator(graph, telemetry_profile=telemetry)
    
    scenario = FailureScenario(target_ids=["Pub1"], failure_mode=FailureMode.CRASH, cascade_probability=1.0)
    res = sim.simulate(scenario)
    
    # Pub1 -> Topic1 did NOT cascade (prob = 0.0)
    assert "Topic1" not in res.cascaded_failures
    # Pub1 -> Topic2 DID cascade (uses default scenario cascade_probability = 1.0)
    assert "Topic2" in res.cascaded_failures
    # Sub1 did not cascade because Topic1 is active
    assert "Sub1" not in res.cascaded_failures


def test_starvation_boundary_calibration():
    """Ensure component starvation bounds can be customized via telemetry."""
    # Create graph with 2 publishers to 1 topic
    components = [
        ComponentData("Pub1", "Application"),
        ComponentData("Pub2", "Application"),
        ComponentData("Topic1", "Topic"),
        ComponentData("Sub1", "Application"),
    ]
    edges = [
        EdgeData("Pub1", "Topic1", "Application", "Topic", "logic", "PUBLISHES_TO"),
        EdgeData("Pub2", "Topic1", "Application", "Topic", "logic", "PUBLISHES_TO"),
        EdgeData("Sub1", "Topic1", "Application", "Topic", "logic", "SUBSCRIBES_TO"),
    ]
    graph = SimulationGraph(GraphData(components=components, edges=edges))
    
    # Normal starvation threshold is 0.3.
    # If 1 of 2 publishers fails (crash = 1.0 impact), average publisher impact is 0.5.
    # 0.5 >= (1 - 0.3) = 0.7 is False -> Topic lives under default settings.
    sim = FailureSimulator(graph)
    res = sim.simulate(FailureScenario(target_ids=["Pub1"], failure_mode=FailureMode.CRASH))
    assert "Topic1" not in res.cascaded_failures

    # Customize starvation bound on Pub1 to 0.6.
    # 0.5 >= (1 - 0.6) = 0.4 is True -> Topic fails.
    telemetry = RuntimeTelemetryProfile(
        custom_starvation_bounds={
            "Pub1": 0.6,
        }
    )
    sim_calibrated = FailureSimulator(graph, telemetry_profile=telemetry)
    res_calibrated = sim_calibrated.simulate(FailureScenario(target_ids=["Pub1"], failure_mode=FailureMode.CRASH))
    assert "Topic1" in res_calibrated.cascaded_failures


def test_throughput_loss_calibration():
    """Ensure throughput loss calculation is scaled by empirical msg_rate_per_sec."""
    graph = create_calibration_test_graph()
    
    # Topic1 and Topic2 have default weights of 1.0.
    # With no telemetry:
    # Fail Pub1, which fails Topic1 and Topic2 (both publishers and subscribers are disconnected/failed)
    # total_weight = 2.0. If we fail Broker1, Topic1 and Topic2 route is lost -> lost capacity = 2.0. Throughput loss = 1.0.
    # If we set msg_rate_per_sec to Topic1 = 9.0 and Topic2 = 1.0:
    telemetry = RuntimeTelemetryProfile(
        msg_rate_per_sec={
            "Topic1": 9.0,
            "Topic2": 1.0,
        }
    )
    
    # Scenario: fail only Topic2. Since Topic1 is routed and has publishers/subscribers, lost capacity is Topic2 weight (1.0).
    # Total volume = 10.0. throughput_loss should be 1.0 / 10.0 = 0.10.
    sim = FailureSimulator(graph, telemetry_profile=telemetry)
    
    # Let's run simulation targeting Topic2
    res = sim.simulate(FailureScenario(target_ids=["Topic2"], failure_mode=FailureMode.CRASH))
    
    # Throughput loss should be 0.10
    assert abs(res.impact.throughput_loss - 0.10) < 1e-6
    
    # Now target Topic1 instead. Lost capacity = 9.0. Total = 10.0. throughput_loss should be 0.90.
    res2 = sim.simulate(FailureScenario(target_ids=["Topic1"], failure_mode=FailureMode.CRASH))
    assert abs(res2.impact.throughput_loss - 0.90) < 1e-6
