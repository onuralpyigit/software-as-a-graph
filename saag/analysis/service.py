from typing import Dict, List

from .models import LayerAnalysisResult
from .structural_analyzer import StructuralAnalyzer
from saag.core.layers import AnalysisLayer, get_layer_definition
from saag.core.ports.graph_repository import IGraphRepository


class AnalysisService:
    """
    Service for running structural graph analysis (Step 2).

    Orchestrates StructuralAnalyzer only — RM/GNN scoring, anti-pattern
    detection, and explanations are produced by the Predict stage (Step 3),
    see saag.prediction.service.PredictionService.
    """

    def __init__(self, repository: IGraphRepository):
        self.repository = repository

    def analyze_layers(self, layers: List[str]) -> Dict[str, LayerAnalysisResult]:
        """
        Run structural analysis (Step 2) on each layer in *layers*.

        Dependencies are derived and the graph loaded **once**, then reused for
        every layer projection. Each result has only ``structural`` populated;
        RM/GNN quality scoring, anti-pattern detection, and explanations are
        produced separately by the Predict stage (Step 3) — see
        saag.prediction.service.PredictionService.

        Returns a dict keyed by canonical layer name.
        """
        # Pre-analysis stage: derive DEPENDS_ON edges and finalise their weights.
        self.repository.derive_dependencies()
        # include_raw=True: StructuralAnalyzer._compute_pubsub_metrics needs the
        # raw PUBLISHES_TO/SUBSCRIBES_TO/ROUTES edges (not just DEPENDS_ON) to score
        # pubsub_degree/broker_exposure/publisher_spof. extract_layer_subgraph filters
        # G to dependency_type in defn.dependency_types, so raw edges never leak into
        # the analysis topology itself.
        graph_data = self.repository.get_graph_data(include_raw=True)

        analyzer = StructuralAnalyzer()
        results: Dict[str, LayerAnalysisResult] = {}

        for layer in layers:
            layer_enum = _resolve_layer(layer)
            layer_def = get_layer_definition(layer_enum)
            results[layer_enum.value] = LayerAnalysisResult(
                layer=layer_enum.value,
                layer_name=layer_def.name,
                description=layer_def.description,
                structural=analyzer.analyze(graph_data, layer=layer_enum),
            )

        return results

    def analyze_layer(self, layer: str) -> LayerAnalysisResult:
        """Run structural analysis (Step 2) on a single layer."""
        layer_enum = _resolve_layer(layer)
        return self.analyze_layers([layer])[layer_enum.value]


def _resolve_layer(layer: str) -> AnalysisLayer:
    """Resolve a layer name or alias, falling back to SYSTEM when unknown."""
    if isinstance(layer, AnalysisLayer):
        return layer
    try:
        return AnalysisLayer.from_string(layer)
    except ValueError:
        return AnalysisLayer.SYSTEM
