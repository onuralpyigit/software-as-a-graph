"""
Simulation Module for Pub-Sub Systems
======================================

Comprehensive simulation capabilities including:
- FailureSimulator: Component failure and cascade simulation
- EventDrivenSimulator: Message flow and performance simulation

Usage:
    from src.simulation import FailureSimulator, EventDrivenSimulator
    
    # Failure simulation
    simulator = FailureSimulator(cascade_threshold=0.7)
    result = simulator.simulate_single_failure(graph, 'broker1')
    
    # Event-driven simulation
    event_sim = EventDrivenSimulator()
    result = event_sim.simulate(graph, duration_ms=10000, message_rate=100)
"""

from .failure_simulator import (
    # Main class
    FailureSimulator,
    
    # Enums
    FailureType,
    FailureMode,
    RecoveryStrategy,
    AttackStrategy,
    
    # Data classes
    FailureEvent,
    ImpactMetrics,
    SimulationResult,
    BatchSimulationResult
)

from .event_simulator import (
    # Main class
    EventDrivenSimulator,
    
    # Enums
    EventType,
    MessageState,
    ComponentState,
    QoSLevel,
    
    # Data classes
    SimEvent,
    Message,
    ComponentStats,
    SimulationMetrics,
    EventSimulationResult
)

__all__ = [
    # Failure Simulator
    'FailureSimulator',
    'FailureType',
    'FailureMode',
    'RecoveryStrategy',
    'AttackStrategy',
    'FailureEvent',
    'ImpactMetrics',
    'SimulationResult',
    'BatchSimulationResult',
    
    # Event-Driven Simulator
    'EventDrivenSimulator',
    'EventType',
    'MessageState',
    'ComponentState',
    'QoSLevel',
    'SimEvent',
    'Message',
    'ComponentStats',
    'SimulationMetrics',
    'EventSimulationResult'
]

__version__ = '1.0.0'