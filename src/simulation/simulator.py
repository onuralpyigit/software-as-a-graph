"""
Simulator Facade
"""
import logging
from typing import List
from src.core.graph_exporter import GraphExporter
from .simulation_graph import SimulationGraph
from .event_simulator import EventSimulator, EventScenario, EventResult
from .failure_simulator import FailureSimulator, FailureScenario, FailureResult

class Simulator:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.exporter = GraphExporter(uri, user, password)
        self.logger = logging.getLogger(__name__)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    def close(self): self.exporter.close()

    def run_event_sim(self, source_id: str) -> EventResult:
        """Run event simulation (Application Layer Logic)."""
        data = self.exporter.get_structural_graph()
        sim_graph = SimulationGraph(data)
        sim = EventSimulator(sim_graph)
        return sim.simulate(EventScenario(source_id, f"Event Source: {source_id}"))

    def run_failure_sim(self, target_id: str, 
                        layer: str = "complete",
                        threshold: float = 0.5, 
                        probability: float = 0.7, 
                        depth: int = 5) -> FailureResult:
        """Runs a single failure simulation."""
        data = self.exporter.get_structural_graph()
        sim_graph = SimulationGraph(data)
        
        sim = FailureSimulator(sim_graph)
        scenario = FailureScenario(
            target_node=target_id,
            description=f"Failure: {target_id} [{layer}]",
            layer=layer,
            cascade_threshold=threshold,
            cascade_probability=probability,
            max_depth=depth
        )
        return sim.simulate(scenario)

    def run_exhaustive_failure_sim(self, 
                                   layer: str = "complete",
                                   threshold: float = 0.5, 
                                   probability: float = 0.7, 
                                   depth: int = 5) -> List[FailureResult]:
        """
        Runs failure simulation for EVERY component in the graph (or specified layer).
        Returns a list of results suitable for dataset generation.
        """
        self.logger.info(f"Starting exhaustive simulation for layer: {layer}")
        
        # 1. Fetch Data
        data = self.exporter.get_structural_graph()
        sim_graph = SimulationGraph(data)
        
        # 2. Identify Targets
        # We use the graph wrapper to filter nodes by layer if needed
        # (Though currently sim_graph is built from full 'data', we can filter iterating nodes)
        full_graph = sim_graph.graph
        
        if layer == "infrastructure":
            targets = [n for n, d in full_graph.nodes(data=True) if d.get("type") == "Node"]
        elif layer == "application":
            targets = [n for n, d in full_graph.nodes(data=True) if d.get("type") in ["Application", "Broker", "Topic"]]
        else:
            targets = list(full_graph.nodes())
            
        self.logger.info(f"Found {len(targets)} targets to simulate.")
        
        # 3. Iterate and Simulate
        results = []
        sim = FailureSimulator(sim_graph)
        
        for i, target in enumerate(targets):
            if i % 10 == 0:
                self.logger.info(f"Simulating {i}/{len(targets)}: {target}")
                
            scenario = FailureScenario(
                target_node=target,
                description=f"Exhaustive: {target}",
                layer=layer,
                cascade_threshold=threshold,
                cascade_probability=probability,
                max_depth=depth
            )
            try:
                # Reset is handled inside sim.simulate
                res = sim.simulate(scenario)
                results.append(res)
            except Exception as e:
                self.logger.error(f"Failed to simulate target {target}: {e}")
                
        return results