"""
Simulation Module

Pub-sub system simulation for impact analysis and validation.

This module provides comprehensive simulation capabilities for distributed
pub-sub systems, working directly on RAW structural relationships
(PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON) without DEPENDS_ON derivation.

Simulators:
    - EventSimulator: Discrete event simulation for message flow
    - FailureSimulator: Failure injection and cascade propagation
    - Simulator: Main facade orchestrating all simulations

Features:
    - Event-driven simulation with throughput/latency/drop metrics
    - Failure simulation with cascade and impact analysis
    - Multi-layer analysis (app, infra, mw-app, mw-infra, system)
    - Criticality classification based on simulation results
    - Comprehensive reporting and export
    - Direct Neo4j graph data retrieval

Layers:
    - app: Application layer (Applications only)
    - infra: Infrastructure layer (Nodes only)
    - mw-app: Middleware-Application (Applications + Brokers)
    - mw-infra: Middleware-Infrastructure (Nodes + Brokers)
    - system: Complete system (all components)

Example:
    >>> from src.simulation import Simulator
    >>> 
    >>> with Simulator(uri="bolt://localhost:7687") as sim:
    ...     # Event simulation
    ...     event_result = sim.run_event_simulation("App1", num_messages=100)
    ...     print(f"Delivery rate: {event_result.metrics.delivery_rate}%")
    ...     
    ...     # Failure simulation
    ...     failure_result = sim.run_failure_simulation("Broker1")
    ...     print(f"Impact: {failure_result.impact.composite_impact}")
    ...     
    ...     # Full report
    ...     report = sim.generate_report(layers=["app", "infra", "system"])
"""

# Simulation Graph
from .simulation_graph import (
    SimulationGraph,
    ComponentState,
    RelationType,
    ComponentInfo,
    TopicInfo,
)

# Event Simulator
from .event_simulator import (
    EventSimulator,
    EventScenario,
    EventResult,
    RuntimeMetrics,
    EventType,
    Event,
    Message,
)

# Failure Simulator
from .failure_simulator import (
    FailureSimulator,
    FailureScenario,
    FailureResult,
    ImpactMetrics,
    FailureMode,
    CascadeRule,
    CascadeEvent,
)

# Main Simulator
from .simulator import (
    Simulator,
    SimulationReport,
    LayerMetrics,
    ComponentCriticality,
)

# Display Functions
from ..visualization.display import (
    display_event_result,
    display_failure_result,
    display_exhaustive_results,
    display_simulation_report as display_report,
    level_color,
)


__all__ = [
    # Simulation Graph
    "SimulationGraph",
    "ComponentState",
    "RelationType",
    "ComponentInfo",
    "TopicInfo",
    
    # Event Simulator
    "EventSimulator",
    "EventScenario",
    "EventResult",
    "RuntimeMetrics",
    "EventType",
    "Event",
    "Message",
    
    # Failure Simulator
    "FailureSimulator",
    "FailureScenario",
    "FailureResult",
    "ImpactMetrics",
    "FailureMode",
    "CascadeRule",
    "CascadeEvent",
    
    # Main Simulator
    "Simulator",
    "SimulationReport",
    "LayerMetrics",
    "ComponentCriticality",
    
    # Display Functions
    "display_event_result",
    "display_failure_result",
    "display_exhaustive_results",
    "display_report",
    "level_color",
]