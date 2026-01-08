"""
Simulator Facade

Orchestrates Event and Failure simulations.
Supports batch processing and report generation.
"""
import logging
from typing import List, Dict, Any
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
        """Run a single event propagation simulation."""
        data = self.exporter.get_structural_graph()
        sim_graph = SimulationGraph(data)
        sim = EventSimulator(sim_graph)
        return sim.simulate(EventScenario(source_id, f"Event Source: {source_id}"))

    def run_failure_sim(self, target_id: str, layer: str = "complete", **kwargs) -> FailureResult:
        """Run a single failure simulation."""
        data = self.exporter.get_structural_graph()
        sim_graph = SimulationGraph(data)
        sim = FailureSimulator(sim_graph)
        
        scenario = FailureScenario(
            target_node=target_id,
            description=f"Failure: {target_id}",
            layer=layer,
            cascade_threshold=kwargs.get("threshold", 0.5),
            cascade_probability=kwargs.get("probability", 0.7),
            max_depth=kwargs.get("depth", 5)
        )
        return sim.simulate(scenario)

    def generate_evaluation_report(self) -> Dict[str, Any]:
        """
        Runs exhaustive simulations for both Application and Infrastructure layers.
        Returns a summarized report.
        """
        data = self.exporter.get_structural_graph()
        sim_graph = SimulationGraph(data)
        sim = FailureSimulator(sim_graph)
        
        report = {
            "summary": {},
            "critical_components": []
        }
        
        # 1. Exhaustive Simulation
        results = []
        nodes = sim_graph.graph.nodes(data=True)
        
        for node_id, attrs in nodes:
            scenario = FailureScenario(target_node=node_id, description="Batch")
            res = sim.simulate(scenario)
            results.append(res)
            
        # 2. Aggregation & Statistics
        results.sort(key=lambda x: x.impact_score, reverse=True)
        
        # Top 5 Critical Components
        for r in results[:5]:
            report["critical_components"].append({
                "id": r.initial_failure,
                "type": r.failure_type,
                "impact_score": r.impact_score,
                "reachability_loss": r.app_reachability_loss,
                "fragmentation": r.infra_fragmentation
            })
            
        # Layer Stats
        app_scores = [r.app_reachability_loss for r in results]
        infra_scores = [r.infra_fragmentation for r in results]
        
        report["summary"] = {
            "total_scenarios": len(results),
            "avg_reachability_loss": sum(app_scores) / len(app_scores) if app_scores else 0,
            "avg_fragmentation": sum(infra_scores) / len(infra_scores) if infra_scores else 0,
            "max_impact": results[0].impact_score if results else 0
        }
        
        return report

    def run_exhaustive_failure_sim(self, layer: str = "complete", **kwargs) -> List[FailureResult]:
        """Runs batch simulation for dataset generation."""
        data = self.exporter.get_structural_graph()
        sim_graph = SimulationGraph(data)
        sim = FailureSimulator(sim_graph)
        
        targets = [n for n, d in sim_graph.graph.nodes(data=True)]
        if layer == "infrastructure":
            targets = [n for n, d in sim_graph.graph.nodes(data=True) if d.get("type") == "Node"]
        elif layer == "application":
            targets = [n for n, d in sim_graph.graph.nodes(data=True) if d.get("type") in ["Application", "Broker"]]

        results = []
        for target in targets:
            scenario = FailureScenario(
                target_node=target,
                description="Exhaustive",
                layer=layer,
                cascade_threshold=kwargs.get("threshold", 0.5),
                cascade_probability=kwargs.get("probability", 0.7),
                max_depth=kwargs.get("depth", 5)
            )
            results.append(sim.simulate(scenario))
        return results