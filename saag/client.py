"""
saag/client.py
"""
import json
from typing import TYPE_CHECKING, Optional, List, Dict, Any

from .models import AnalysisResult, PredictionResult, ValidationResult, ValidationPipelineFacade, ImportResult

if TYPE_CHECKING:
    # Import-time only: pulling saag.prescription in eagerly would drag the
    # validation stack (and numpy) into every `import saag`.
    from saag.prescription.models import PrescribeResult

class Client:
    """
    Step-by-step programmatic client for SoftwareAsAGraph.
    """
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password", repo=None):
        if repo:
            self.repo = repo
        else:
            from saag.infrastructure import create_repository
            self.repo = create_repository(uri=neo4j_uri, user=user, password=password)
        
    def import_topology(self, filepath: Optional[str] = None, graph_data: Optional[Dict[str, Any]] = None, clear: bool = False, dry_run: bool = False):
        """Import a JSON topology into the graph database, either from a file or raw dict."""
        if graph_data is None:
            if not filepath:
                raise ValueError("Either filepath or graph_data must be provided.")
            with open(filepath, "r") as f:
                graph_data = json.load(f)
            
        from saag.usecases.model_graph import ModelGraphUseCase
        uc = ModelGraphUseCase(self.repo)
        
        stats = uc.execute(graph_data, clear=clear, dry_run=dry_run)
        return ImportResult(stats)


    def analyze(self, layer: str = "app", **kwargs) -> AnalysisResult:
        """Analyze the structural graph topology.

        Extra keyword arguments are accepted and ignored — Pipeline.analyze
        forwards RM options here, but they apply to the Predict stage
        (see ``predict()``), not to structural analysis.
        """
        from saag.usecases.analyze_graph import AnalyzeGraphUseCase
        from saag.analysis.service import AnalysisService

        service = AnalysisService(self.repo)
        uc = AnalyzeGraphUseCase(service)
        result = uc.execute(layer=layer)
        return AnalysisResult(result)

    def predict(
        self,
        analysis_result: AnalysisResult,
        mode: str = "gnn",
        gnn_checkpoint: Optional[str] = None,
        use_ahp: bool = False,
        equal_weights: bool = False,
        ahp_shrinkage: float = 0.7,
        normalization_method: str = "robust",
        winsorize: bool = True,
        winsorize_limit: float = 0.05,
        run_sensitivity: bool = False,
        active_patterns: Optional[List[str]] = None,
    ) -> PredictionResult:
        """Run the unified Prediction stage (Step 3) on a prior AnalysisResult.

        Consumes the StructuralAnalysisResult produced by analyze() — no
        repository access. Always computes rule-based RM scores; blends in
        GNN-derived criticality ranks when a trained checkpoint is available
        at ``gnn_checkpoint`` (falls back to RM otherwise). Also runs
        anti-pattern detection and generates a human-readable explanation.

        Parameters
        ----------
        analysis_result:
            Output of a preceding client.analyze() call for the same layer.
        mode:
            'gnn' for raw GNN scores (default) when a checkpoint is available.
        gnn_checkpoint:
            Path to a GNN checkpoint directory. Defaults to output/gnn_checkpoints.
        use_ahp, equal_weights, ahp_shrinkage:
            RM dimension weighting. Mutually exclusive modes: AHP-derived
            weights (shrunk toward uniform by ``ahp_shrinkage``), equal 0.25
            weights, or the default fixed weights.
        normalization_method, winsorize, winsorize_limit:
            How Tier-1 metrics are scaled to [0, 1] before the RM weighted sum.
        run_sensitivity:
            Run the Kendall τ weight-stability analysis after scoring.
        active_patterns:
            Anti-pattern IDs to run. ``None`` runs the whole catalogue; an empty
            list skips detection entirely. The distinction matters for callers
            that only want RM scores: DEEP_PIPELINE enumerates every simple
            source-to-sink path and does not terminate in practical time on
            topologies of a few hundred components.
        """
        from saag.prediction.service import PredictionService

        layer_result = analysis_result.raw
        structural = layer_result.structural
        layer = getattr(layer_result, "layer", "system")

        checkpoint = gnn_checkpoint or "output/gnn_checkpoints"
        service = PredictionService(
            use_ahp=use_ahp,
            equal_weights=equal_weights,
            ahp_shrinkage=ahp_shrinkage,
            normalization_method=normalization_method,
            winsorize=winsorize,
            winsorize_limit=winsorize_limit,
            gnn_checkpoint_dir=checkpoint,
            prefer_gnn=(mode == "gnn"),
        )

        result = service.predict_quality_with_gnn(
            structural, structural.graph, layer=layer, run_sensitivity=run_sensitivity,
            active_patterns=active_patterns,
        )
        return PredictionResult(result)

    def triage(
        self,
        prediction_result: Any,
        k: int = 10,
        node_types: Optional[List[str]] = None,
    ) -> Any:
        """Run the Triage bridge: scope Pathway A's (RM) root-cause diagnosis
        to the Top-K components Pathway B's ranking (GNN when a checkpoint
        was available at predict()-time, RM otherwise) flagged as critical.

        Consumes the result produced by predict() -- no repository access.
        Each shortlisted component is joined back to its RM
        CriticalityProfile pattern, elevated dimensions, priority action,
        and stakeholder roles by id, never read off the ranking itself (a
        GNN result carries no root cause of its own).
        """
        from saag.usecases.triage_graph import TriageGraphUseCase

        raw = getattr(prediction_result, "raw", prediction_result)
        layer = getattr(raw, "layer", "system")
        uc = TriageGraphUseCase()
        return uc.execute(raw, k=k, layer=layer, node_types=node_types)

    def detect_antipatterns(self, prediction_result: Any, active_patterns: Optional[List[str]] = None) -> List[Any]:
        """Return anti-patterns detected during the Predict stage.

        Problems are computed as part of predict(); this method simply surfaces
        them with an optional pattern filter rather than re-running detection.
        """
        problems = getattr(prediction_result, "problems", []) or []
        if active_patterns:
            problems = [p for p in problems if getattr(p, "pattern", None) in active_patterns]
        return problems

    def simulate(self, layer: str = "system", mode: str = "exhaustive", target_id: Optional[str] = None, **kwargs) -> Any:
        """Run graph simulations (failure analysis, event propagation)."""
        from saag.usecases.simulate_graph import SimulateGraphUseCase
        from saag.simulation.service import SimulationService
        from saag.usecases.models import SimulationMode
        
        service = SimulationService(self.repo)
        uc = SimulateGraphUseCase(service)
        
        try:
            mode_enum = SimulationMode(mode)
        except ValueError:
            mode_enum = SimulationMode.EXHAUSTIVE
            
        return uc.execute(layer=layer, mode=mode_enum, target_id=target_id, **kwargs)

    def validate(self, layers: Optional[List[str]] = None, **kwargs) -> ValidationResult:
        """Validate the criticality model against ground truth simulation results."""
        if layers is None:
            layers = ["app", "infra", "mw", "system"]
            
        from saag.usecases.validate_graph import ValidateGraphUseCase
        from saag.analysis.service import AnalysisService
        from saag.prediction.service import PredictionService
        from saag.simulation.service import SimulationService
        from saag.validation.service import ValidationService
        
        analysis_service = AnalysisService(self.repo)
        gnn_checkpoint = kwargs.get("gnn_checkpoint") or kwargs.get("gnn_checkpoint_dir")
        prediction_service = PredictionService(gnn_checkpoint_dir=gnn_checkpoint)
        simulation_service = SimulationService(self.repo)
        validation_service = ValidationService(
            analysis_service=analysis_service,
            prediction_service=prediction_service,
            simulation_service=simulation_service
        )
        
        uc = ValidateGraphUseCase(validation_service)
        pipeline_result = uc.execute(layers=layers)
        
        return ValidationPipelineFacade(pipeline_result)

    def prescribe(
        self,
        analysis_result: Any,
        prediction_result: Optional[Any] = None,
        layer: str = "system",
        gnn_checkpoint: Optional[str] = None,
        **kwargs: Any,
    ) -> "PrescribeResult":
        """Run the prescriptive Stage 6 optimization on critical items and smells.

        ``kwargs`` is forwarded to ``PrescribeService.prescribe`` — notably
        ``kappa``, ``seeds`` and ``thresholds``, which parameterise the per-edit
        acceptance filter.
        """
        from saag.prescription.service import PrescribeService
        from saag.usecases.prescribe_graph import PrescribeGraphUseCase

        service = PrescribeService(self.repo)
        uc = PrescribeGraphUseCase(service)
        
        # Unwrap SDK facades if necessary
        raw_analysis = getattr(analysis_result, "raw", analysis_result)
        raw_prediction = getattr(prediction_result, "raw", prediction_result)
        
        return uc.execute(
            analysis_result=raw_analysis,
            prediction_result=raw_prediction,
            layer=layer,
            gnn_checkpoint=gnn_checkpoint,
            **kwargs,
        )

    def visualize(self, output: str = "report.html", layers: Optional[List[str]] = None, **kwargs) -> str:
        """Render the logic to an HTML report."""
        if layers is None:
            layers = ["system"]
            
        from saag.usecases.visualize_graph import VisualizeGraphUseCase
        from saag.usecases.models import VisOptions
        from saag.analysis.service import AnalysisService
        from saag.prediction.service import PredictionService
        from saag.simulation.service import SimulationService
        from saag.validation.service import ValidationService
        from saag.visualization.service import VisualizationService
        
        analysis_service = AnalysisService(self.repo)
        prediction_service = PredictionService()
        if hasattr(self.repo, 'driver'):
             simulation_service = SimulationService(self.repo)
        else:
             simulation_service = None # or some fallback
             
        validation_service = ValidationService(analysis_service, prediction_service, simulation_service)
        
        viz_service = VisualizationService(
            analysis_service=analysis_service,
            prediction_service=prediction_service,
            simulation_service=simulation_service,
            validation_service=validation_service,
            repository=self.repo
        )
        
        uc = VisualizeGraphUseCase(viz_service)
        
        options = VisOptions()
        if "include_network" in kwargs: options.include_network = kwargs["include_network"]
        if "include_matrix" in kwargs: options.include_matrix = kwargs["include_matrix"]
        if "include_validation" in kwargs: options.include_validation = kwargs["include_validation"]
        if "antipatterns_file" in kwargs: options.antipatterns_file = kwargs["antipatterns_file"]
        if "multi_seed" in kwargs: options.multi_seed = kwargs["multi_seed"]
        if "cascade_file" in kwargs: options.cascade_file = kwargs["cascade_file"]
        
        return uc.execute(layers=layers, output_file=output, options=options)

    def export_topology(self) -> Dict[str, Any]:
        """Export the current graph back to the canonical nested input format."""
        return self.repo.export_json()

    def get_graph_data(
        self, 
        component_types: Optional[List[str]] = None, 
        dependency_types: Optional[List[str]] = None, 
        include_raw: bool = False
    ) -> Any:
        """Export graph data in flat analysis format (components/edges lists)."""
        return self.repo.get_graph_data(
            component_types=component_types, 
            dependency_types=dependency_types, 
            include_raw=include_raw
        )

