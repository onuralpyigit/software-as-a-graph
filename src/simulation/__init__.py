"""
Simulation Module - Version 5.0

Comprehensive simulation capabilities for pub-sub systems.

Features:
- Failure simulation with cascade propagation
- Event-driven discrete event simulation
- Layer-specific analysis and reporting
- Impact scoring based on message path disruption
- Neo4j integration for loading graphs directly from database

Key Design Decision:
    Simulations use ORIGINAL edge types:
    - PUBLISHES_TO, SUBSCRIBES_TO (message flow)
    - ROUTES (broker responsibility)
    - RUNS_ON, CONNECTS_TO (infrastructure)
    
    NOT derived DEPENDS_ON relationships.
    
    This enables accurate simulation of actual message paths
    through the pub-sub system.

Layers:
    - application: App-to-app communication via topics
    - infrastructure: Node-to-broker connections
    - app_broker: How applications connect through brokers
    - node_broker: Infrastructure deployment topology

Usage:
    from src.simulation import (
        SimulationGraph,
        FailureSimulator,
        EventSimulator,
        create_simulation_graph,
    )
    
    # Load from JSON
    graph = SimulationGraph.from_json("system.json")
    
    # Generate synthetic graph
    graph = create_simulation_graph(applications=10, brokers=2)
    
    # Load from Neo4j
    from src.simulation import load_graph_from_neo4j
    graph = load_graph_from_neo4j(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )
    
    # Failure simulation
    simulator = FailureSimulator(cascade=True)
    result = simulator.simulate_failure(graph, "broker_1")
    print(f"Impact: {result.impact_score:.4f}")
    
    # Full campaign
    campaign = simulator.simulate_all(graph)
    for comp_id, impact in campaign.ranked_by_impact()[:10]:
        print(f"{comp_id}: {impact:.4f}")
    
    # Event-driven simulation
    event_sim = EventSimulator(seed=42)
    result = event_sim.run(graph, duration=5000, message_rate=100)
    print(f"Delivered: {result.stats.delivery_rate:.2%}")

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

__version__ = "5.0.0"

# Simulation Graph
from .simulation_graph import (
    # Enums
    ComponentType,
    EdgeType,
    ComponentStatus,
    # Data Classes
    QoSPolicy,
    Component,
    Edge,
    # Main Class
    SimulationGraph,
    # Factory
    create_simulation_graph,
)

# Failure Simulator
from .failure_simulator import (
    # Enums
    FailureMode,
    # Data Classes
    FailureResult,
    LayerFailureResult,
    CampaignResult,
    # Main Class
    FailureSimulator,
    # Factory Functions
    simulate_single_failure,
    simulate_all_components,
)

# Event Simulator
from .event_simulator import (
    # Enums
    EventType,
    MessageStatus,
    # Data Classes
    Event,
    Message,
    ComponentLoad,
    SimulationStats,
    LayerStats,
    SimulationResult,
    # Main Class
    EventSimulator,
    # Factory Functions
    run_event_simulation,
    run_stress_test,
)

# Neo4j Client
from .neo4j_client import (
    # Config
    Neo4jConfig,
    # Main Class
    Neo4jSimulationClient,
    # Factory Functions
    load_graph_from_neo4j,
    check_neo4j_available,
)


__all__ = [
    # Version
    "__version__",
    
    # Simulation Graph - Enums
    "ComponentType",
    "EdgeType",
    "ComponentStatus",
    # Simulation Graph - Data Classes
    "QoSPolicy",
    "Component",
    "Edge",
    # Simulation Graph - Main
    "SimulationGraph",
    "create_simulation_graph",
    
    # Failure Simulator - Enums
    "FailureMode",
    # Failure Simulator - Data Classes
    "FailureResult",
    "LayerFailureResult",
    "CampaignResult",
    # Failure Simulator - Main
    "FailureSimulator",
    "simulate_single_failure",
    "simulate_all_components",
    
    # Event Simulator - Enums
    "EventType",
    "MessageStatus",
    # Event Simulator - Data Classes
    "Event",
    "Message",
    "ComponentLoad",
    "SimulationStats",
    "LayerStats",
    "SimulationResult",
    # Event Simulator - Main
    "EventSimulator",
    "run_event_simulation",
    "run_stress_test",
    
    # Neo4j Client
    "Neo4jConfig",
    "Neo4jSimulationClient",
    "load_graph_from_neo4j",
    "check_neo4j_available",
]
