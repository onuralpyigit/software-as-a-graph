from typing import Any, Dict, List, Optional
from saag.visualization.service import VisualizationService
from .models import VisOptions


class VisualizeGraphUseCase:
    """Use case for generating the visualization dashboard."""

    def __init__(self, service: VisualizationService):
        self.service = service

    def execute(
        self,
        layers: List[str],
        output_file: str,
        options: VisOptions,
        precomputed: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        return self.service.generate_dashboard(
            output_file=output_file,
            layers=layers,
            include_network=options.include_network,
            include_matrix=options.include_matrix,
            include_validation=options.include_validation,
            include_per_dim_scatter=options.include_per_dim_scatter,
            antipatterns_file=options.antipatterns_file,
            multi_seed=options.multi_seed,
            cascade_file=options.cascade_file,
            triage_k=options.triage_k,
            precomputed=precomputed,
        )
