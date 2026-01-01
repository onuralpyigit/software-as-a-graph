"""
Simulation Module - Version 5.0

Comprehensive simulation capabilities for pub-sub systems.

Features:
- Failure simulation with cascade propagation
- Event-driven discrete event simulation
- Component-type specific simulation
- Neo4j integration for live graph data
- Impact scoring and statistics

Simulators:
1. FailureSimulator: Tests system resilience by failing components
2. EventSimulator: Simulates message flow with QoS modeling

Key Design Decision:
Simulations use ORIGINAL edge types (PUBLISHES_TO, SUBSCRIBES_TO, etc.)
NOT derived DEPENDS_ON relationships. This allows accurate simulation
of message flow paths through the pub-sub system.

Usage:
    from src.simulation import (
        SimulationGraph,
        FailureSimulator,
        EventSimulator,
        load_graph_from_neo4j,
    )
    
    # Load graph from JSON
    graph = SimulationGraph.from_json("system.json")
    
    # Or from Neo4j
    graph = load_graph_from_neo4j(
        uri="bolt://localhost:7687",
        password="secret"
    )
    
    # Failure simulation
    simulator = FailureSimulator(cascade=True)
    result = simulator.simulate_failure(graph, "broker1")
    print(f"Impact: {result.impact_score:.4f}")
    
    # Full campaign
    campaign = simulator.simulate_all_failures(graph)
    for comp_id, impact in campaign.ranked_by_impact[:10]:
        print(f"{comp_id}: {impact:.4f}")
    
    # Event simulation
    event_sim = EventSimulator(graph, seed=42)
    result = event_sim.run(duration=5000, message_rate=100)
    print(f"Delivered: {result.statistics.delivery_rate:.2%}")

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

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
    PropagationRule,
    # Data Classes
    FailureResult,
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
    SimulationState,
    # Data Classes
    Event,
    Message,
    ComponentLoad,
    SimulationStatistics,
    SimulationResult,
    # Main Class
    EventSimulator,
    # Factory Functions
    run_event_simulation,
    run_stress_test,
)

# Neo4j Client
from .neo4j_client import (
    Neo4jConfig,
    Neo4jSimulationClient,
    load_graph_from_neo4j,
)

__all__ = [
    # === Simulation Graph ===
    # Enums
    "ComponentType",
    "EdgeType",
    "ComponentStatus",
    # Data Classes
    "QoSPolicy",
    "Component",
    "Edge",
    # Main Class
    "SimulationGraph",
    # Factory
    "create_simulation_graph",
    
    # === Failure Simulator ===
    # Enums
    "FailureMode",
    "PropagationRule",
    # Data Classes
    "FailureResult",
    "CampaignResult",
    # Main Class
    "FailureSimulator",
    # Factory Functions
    "simulate_single_failure",
    "simulate_all_components",
    
    # === Event Simulator ===
    # Enums
    "EventType",
    "MessageStatus",
    "SimulationState",
    # Data Classes
    "Event",
    "Message",
    "ComponentLoad",
    "SimulationStatistics",
    "SimulationResult",
    # Main Class
    "EventSimulator",
    # Factory Functions
    "run_event_simulation",
    "run_stress_test",
    
    # === Neo4j Client ===
    "Neo4jConfig",
    "Neo4jSimulationClient",
    "load_graph_from_neo4j",
]

__version__ = "5.0.0"
