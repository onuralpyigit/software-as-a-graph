"""
Simulation Facade

Main entry point for running simulations using Neo4j data.
"""

from typing import Dict, Any, List, Optional
import logging

from src.core.graph_exporter import GraphExporter
from .simulation_graph import SimulationGraph
from .failure_simulator import FailureSimulator, FailureScenario, FailureResult
from .event_simulator import EventSimulator, EventScenario, EventResult

class Simulator:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.exporter = GraphExporter(uri, user, password)
        self.logger = logging.getLogger(__name__)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    def close(self): self.exporter.close()

    def run_failure_simulation(
        self, 
        target_node_id: str,
        component_type: Optional[str] = None,
        layer: Optional[str] = None
    ) -> FailureResult:
        """
        Run failure simulation.
        Optionally restricted to a specific Component Type or Layer.
        """
        # 1. Fetch Data
        if layer:
            data = self.exporter.get_layer(layer)
        elif component_type:
            # For type simulation, we usually need the connections between them.
            # graph_exporter.get_subgraph_by_component_type returns (Type)->(Type)
            data = self.exporter.get_subgraph_by_component_type(component_type)
        else:
            data = self.exporter.get_graph_data()

        # 2. Build Sim Graph
        sim_graph = SimulationGraph(data)
        
        # 3. Run Sim
        sim = FailureSimulator(sim_graph, propagation_threshold=0.4)
        scenario = FailureScenario([target_node_id], f"Failure of {target_node_id}")
        
        return sim.simulate(scenario)

    def run_event_simulation(
        self, 
        source_node_id: str
    ) -> EventResult:
        """Run event flow simulation on the full graph."""
        data = self.exporter.get_graph_data()
        sim_graph = SimulationGraph(data)
        
        sim = EventSimulator(sim_graph)
        scenario = EventScenario(source_node_id, f"Event from {source_node_id}")
        
        return sim.simulate(scenario)