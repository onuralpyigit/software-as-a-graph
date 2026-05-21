import pytest
import networkx as nx
from saag.simulation.message_flow_simulator import MessageFlowSimulator

def test_simulator_honors_topic_frequency_periodic():
    """
    Test that the simulator honors topic.frequency as the periodic publish rate
    and matches frequency * duration within ±5% over a 60-s simulated window.
    """
    g = nx.DiGraph()
    # Topic frequency is 50.0 Hz
    g.add_node("App1", type="Application")
    g.add_node("App2", type="Application")
    g.add_node("/telemetry", type="Topic", frequency=50.0)

    g.add_edge("App1", "/telemetry", type="PUBLISHES_TO")
    g.add_edge("App2", "/telemetry", type="SUBSCRIBES_TO")

    # Duration = 60.0 seconds
    sim = MessageFlowSimulator(
        graph=g,
        duration=60.0,
        seed=42,
        default_publish_rate_hz=10.0,
    )

    result = sim.run()
    stats = result.topic_stats["/telemetry"]

    expected_count = 50.0 * 60.0  # 3000
    actual_count = stats.total_published

    # Acceptance check: ±5% tolerance
    assert abs(actual_count - expected_count) <= 0.05 * expected_count, (
        f"Expected periodic message count around {expected_count}, got {actual_count}"
    )

def test_simulator_honors_topic_frequency_poisson():
    """
    Test that the simulator honors topic.frequency as the Poisson publish rate
    and matches frequency * duration within ±5% over a 60-s simulated window.
    """
    g = nx.DiGraph()
    # Topic frequency is 50.0 Hz, workload type is poisson
    g.add_node("App1", type="Application")
    g.add_node("App2", type="Application")
    g.add_node("/telemetry", type="Topic", frequency=50.0, workload_type="poisson")

    g.add_edge("App1", "/telemetry", type="PUBLISHES_TO")
    g.add_edge("App2", "/telemetry", type="SUBSCRIBES_TO")

    # Duration = 60.0 seconds
    sim = MessageFlowSimulator(
        graph=g,
        duration=60.0,
        seed=123,  # different seed
        default_publish_rate_hz=10.0,
    )

    result = sim.run()
    stats = result.topic_stats["/telemetry"]

    expected_count = 50.0 * 60.0  # 3000
    actual_count = stats.total_published

    # Acceptance check: ±5% tolerance
    assert abs(actual_count - expected_count) <= 0.05 * expected_count, (
        f"Expected Poisson message count around {expected_count}, got {actual_count}"
    )

def test_multiple_publishers_share_frequency():
    """
    Test that if multiple publishers exist for the same topic,
    the frequency is shared (divided) equally among them to keep the aggregate topic rate constant.
    """
    g = nx.DiGraph()
    g.add_node("App1", type="Application")
    g.add_node("App2", type="Application")
    g.add_node("App3", type="Application")
    g.add_node("/telemetry", type="Topic", frequency=60.0)

    # 2 publishers publishing to the same topic
    g.add_edge("App1", "/telemetry", type="PUBLISHES_TO")
    g.add_edge("App2", "/telemetry", type="PUBLISHES_TO")
    g.add_edge("App3", "/telemetry", type="SUBSCRIBES_TO")

    sim = MessageFlowSimulator(
        graph=g,
        duration=60.0,
        seed=42,
    )

    # App1 and App2 should each have a rate of 60.0 / 2 = 30.0 Hz
    rate1 = sim.generate_workload("/telemetry")
    assert rate1 == pytest.approx(30.0)

    result = sim.run()
    stats = result.topic_stats["/telemetry"]

    expected_count = 60.0 * 60.0  # 3600
    actual_count = stats.total_published

    assert abs(actual_count - expected_count) <= 0.05 * expected_count, (
        f"Expected aggregate message count around {expected_count}, got {actual_count}"
    )
