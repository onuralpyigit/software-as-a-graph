"""
Simulation Module for Pub-Sub Systems
======================================

Comprehensive simulation capabilities including:
- FailureSimulator: Component failure and cascade simulation
- EventDrivenSimulator: Message flow and performance simulation

Data is loaded directly from Neo4j using Neo4jGraphLoader.

Usage:
    from src.simulation import Neo4jGraphLoader, FailureSimulator, EventDrivenSimulator
    
    # Load graph from Neo4j
    loader = Neo4jGraphLoader(uri="bolt://localhost:7687", user="neo4j", password="password")
    graph = loader.load_graph()
    loader.close()
    
    # Failure simulation
    simulator = FailureSimulator(cascade_threshold=0.7)
    result = simulator.simulate_single_failure(graph, 'broker1')
    
    # Event-driven simulation
    event_sim = EventDrivenSimulator()
    result = event_sim.simulate(graph, duration_ms=10000, message_rate=100)
"""

from .neo4j_loader import (
    # Main classes
    Neo4jGraphLoader,
    SimulationGraph,
    Component,
    Dependency,
    
    # Enums
    ComponentType,
    DependencyType
)

from .failure_simulator import (
    # Main class
    FailureSimulator,
    
    # Enums
    FailureType,
    FailureMode,
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
    QoSLevel,
    
    # Data classes
    SimEvent,
    Message,
    ComponentStats,
    SimulationMetrics,
    EventSimulationResult
)

__all__ = [
    # Graph Loading
    'Neo4jGraphLoader',
    'SimulationGraph',
    'Component',
    'Dependency',
    'ComponentType',
    'DependencyType',
    
    # Failure Simulator
    'FailureSimulator',
    'FailureType',
    'FailureMode',
    'AttackStrategy',
    'FailureEvent',
    'ImpactMetrics',
    'SimulationResult',
    'BatchSimulationResult',
    
    # Event-Driven Simulator
    'EventDrivenSimulator',
    'EventType',
    'MessageState',
    'QoSLevel',
    'SimEvent',
    'Message',
    'ComponentStats',
    'SimulationMetrics',
    'EventSimulationResult'
]

__version__ = '2.0.0'