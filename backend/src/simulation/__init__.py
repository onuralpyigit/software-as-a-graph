"""
Compatibility shim: src.simulation

Maps the old Simulator(uri, user, password) API to the new
SimulationService + Neo4jGraphRepository combo.
"""
from src.adapters.outbound.neo4j_repo import Neo4jGraphRepository
from src.application.services.simulation_service import SimulationService


class Simulator:
    """Backward-compatible facade wrapping SimulationService."""

    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self._repo = Neo4jGraphRepository(uri=uri, user=user, password=password)
        self._service = SimulationService(repository=self._repo)

    def __enter__(self):
        self._repo.__enter__()
        return self

    def __exit__(self, *args):
        self._repo.__exit__(*args)

    def close(self):
        self._repo.close()

    def run_event_simulation(self, source_app, num_messages=100, duration=10.0, **kwargs):
        """Run event simulation from a specific source application."""
        return self._service.run_event_simulation(
            source_app=source_app,
            num_messages=num_messages,
            duration=duration,
            **kwargs,
        )

    def run_failure_simulation(self, target_id, layer="system", cascade_probability=0.5, **kwargs):
        """Run failure simulation for a specific component."""
        return self._service.run_failure_simulation(
            target_id=target_id,
            layer=layer,
            cascade_probability=cascade_probability,
            **kwargs,
        )

    def run_failure_simulation_exhaustive(self, layer="system", cascade_probability=0.5):
        """Run failure simulation for every component in a layer."""
        return self._service.run_failure_simulation_exhaustive(
            layer=layer,
            cascade_probability=cascade_probability,
        )

    def generate_report(self, layers=None):
        """Generate comprehensive simulation report."""
        if layers is None:
            layers = ["app", "infra", "mw", "system"]

        from src.domain.models.simulation import SimulationReport
        layer_metrics = {}
        for layer in layers:
            metrics = self._service.analyze_layer(layer)
            layer_metrics[layer] = metrics

        component_criticality = self._service.classify_components("system")
        edge_criticality = self._service.classify_edges("system")

        return SimulationReport(
            layers=layer_metrics,
            component_criticality=component_criticality,
            edge_criticality=edge_criticality,
        )


__all__ = ["Simulator"]
