"""
saag/client.py
"""
import json
from typing import Optional, List, Dict, Any

from .models import AnalysisResult, PredictionResult, ValidationResult, ValidationPipelineFacade, ImportResult

class Client:
    """
    Step-by-step programmatic client for SoftwareAsAGraph.
    """
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password", repo=None):
        if repo:
            self.repo = repo
        else:
            from src.infrastructure import create_repository
            self.repo = create_repository(uri=neo4j_uri, user=user, password=password)
        
    def import_topology(self, filepath: Optional[str] = None, graph_data: Optional[Dict[str, Any]] = None, clear: bool = False, dry_run: bool = False):
        """Import a JSON topology into the graph database, either from a file or raw dict."""
        if graph_data is None:
            if not filepath:
                raise ValueError("Either filepath or graph_data must be provided.")
            with open(filepath, "r") as f:
                graph_data = json.load(f)
            
        from src.usecases.model_graph import ModelGraphUseCase
        uc = ModelGraphUseCase(self.repo)
        
        stats = uc.execute(graph_data, clear=clear, dry_run=dry_run)
        return ImportResult(stats)


    def analyze(self, layer: str = "app", **kwargs) -> AnalysisResult:
        """Analyze the structural graph topology."""
        from src.usecases.analyze_graph import AnalyzeGraphUseCase
        from src.analysis.service import AnalysisService
        
        service = AnalysisService(self.repo)
        uc = AnalyzeGraphUseCase(service)
        result = uc.execute(layer=layer)
        return AnalysisResult(result)

    def predict(self, layer: str = "app", detect_problems: bool = False, **kwargs) -> PredictionResult:
        """Predict criticality scores and optionally detect anti-patterns."""
        from src.usecases.analyze_graph import AnalyzeGraphUseCase
        from src.analysis.service import AnalysisService
        from src.usecases.predict_graph import PredictGraphUseCase
        from src.prediction.service import PredictionService
        
        # We need structural analysis first
        analysis_service = AnalysisService(self.repo)
        analyze_uc = AnalyzeGraphUseCase(analysis_service)
        structural_result = analyze_uc.execute(layer=layer)
        
        prediction_service = PredictionService()
        predict_uc = PredictGraphUseCase(prediction_service)
        quality, problems = predict_uc.execute(
            layer=layer, 
            structural_result=structural_result,
            detect_problems=detect_problems
        )
        
        return PredictionResult(quality, problems)

    def detect_antipatterns(self, prediction: PredictionResult) -> List[Any]:
        """Detect architectural anti-patterns from the GNN prediction results."""
        from src.prediction.service import PredictionService
        service = PredictionService()
        return service.detect_problems(prediction.raw)

    def simulate(self, layer: str = "system", mode: str = "exhaustive", target_id: Optional[str] = None, **kwargs) -> Any:
        """Run graph simulations (failure analysis, event propagation)."""
        from src.usecases.simulate_graph import SimulateGraphUseCase
        from src.simulation.service import SimulationService
        from src.usecases.models import SimulationMode
        
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
            
        from src.usecases.validate_graph import ValidateGraphUseCase
        from src.analysis.service import AnalysisService
        from src.prediction.service import PredictionService
        from src.simulation.service import SimulationService
        from src.validation.service import ValidationService
        
        analysis_service = AnalysisService(self.repo)
        prediction_service = PredictionService()
        simulation_service = SimulationService(self.repo)
        validation_service = ValidationService(
            analysis_service=analysis_service,
            prediction_service=prediction_service,
            simulation_service=simulation_service
        )
        
        uc = ValidateGraphUseCase(validation_service)
        pipeline_result = uc.execute(layers=layers)
        
        return ValidationPipelineFacade(pipeline_result)

    def visualize(self, output: str = "report.html", layers: Optional[List[str]] = None, **kwargs) -> str:
        """Render the logic to an HTML report."""
        if layers is None:
            layers = ["system"]
            
        from src.usecases.visualize_graph import VisualizeGraphUseCase
        from src.usecases.models import VisOptions
        from src.analysis.service import AnalysisService
        from src.prediction.service import PredictionService
        from src.simulation.service import SimulationService
        from src.validation.service import ValidationService
        from src.visualization.service import VisualizationService
        
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

