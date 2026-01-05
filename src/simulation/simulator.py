"""
Simulator Facade
"""
from typing import Optional
from src.core.graph_exporter import GraphExporter
from .simulation_graph import SimulationGraph
from .event_simulator import EventSimulator, EventScenario, EventResult
from .failure_simulator import FailureSimulator, FailureScenario, FailureResult

class Simulator:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.exporter = GraphExporter(uri, user, password)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    def close(self): self.exporter.close()

    def run_event_sim(self, source_id: str) -> EventResult:
        data = self.exporter.get_structural_graph()
        sim_graph = SimulationGraph(data)
        
        sim = EventSimulator(sim_graph)
        return sim.simulate(EventScenario(source_id, f"Event from {source_id}"))

    def run_failure_sim(self, target_id: str, 
                        threshold: float = 0.5, 
                        probability: float = 0.7, 
                        depth: int = 5) -> FailureResult:
        """
        Runs a failure simulation with configurable cascade parameters.
        Defaults are based on the PhD Progress Report Section 4.4.2.
        """
        data = self.exporter.get_structural_graph()
        sim_graph = SimulationGraph(data)
        
        sim = FailureSimulator(sim_graph)
        scenario = FailureScenario(
            target_node=target_id,
            description=f"Failure of {target_id}",
            cascade_threshold=threshold,
            cascade_probability=probability,
            max_depth=depth
        )
        return sim.simulate(scenario)