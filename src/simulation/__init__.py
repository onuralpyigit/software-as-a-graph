"""
Simulation Module

Pub-sub system simulation for impact analysis and validation.

Components:
- SimulationGraph: Graph representation using raw structural relationships
- EventSimulator: Discrete event simulation for message flow analysis
- FailureSimulator: Failure injection and cascade propagation
- Simulator: Main facade orchestrating simulations and reporting

Features:
- Event-driven simulation with throughput/latency/drop metrics
- Failure simulation with cascade and impact analysis
- Multi-layer analysis (application, infrastructure, complete)
- Criticality classification based on simulation results
- Comprehensive reporting and export

Author: Software-as-a-Graph Research Project
"""

from .simulation_graph import (
    SimulationGraph,
    ComponentState,
    RelationType,
    ComponentInfo,
    TopicInfo,
)

from .event_simulator import (
    EventSimulator,
    AsyncEventSimulator,
    EventScenario,
    EventResult,
    RuntimeMetrics,
    EventType,
    Event,
    Message,
)

from .failure_simulator import (
    FailureSimulator,
    BatchFailureSimulator,
    FailureScenario,
    FailureResult,
    ImpactMetrics,
    FailureMode,
    CascadeRule,
)

from .simulator import (
    Simulator,
    SimulationReport,
    LayerMetrics,
    ComponentCriticality,
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
    "AsyncEventSimulator",
    "EventScenario",
    "EventResult",
    "RuntimeMetrics",
    "EventType",
    "Event",
    "Message",
    
    # Failure Simulator
    "FailureSimulator",
    "BatchFailureSimulator",
    "FailureScenario",
    "FailureResult",
    "ImpactMetrics",
    "FailureMode",
    "CascadeRule",
    
    # Main Simulator
    "Simulator",
    "SimulationReport",
    "LayerMetrics",
    "ComponentCriticality",
]