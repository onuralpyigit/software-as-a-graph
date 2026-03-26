"""
saag/client.py
"""
import json
from typing import Optional, List, Dict, Any

from .models import AnalysisResult, PredictionResult, ValidationResult, ValidationPipelineFacade

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
        
    def import_topology(self, filepath: Optional[str] = None, graph_data: Optional[Dict[str, Any]] = None, clear: bool = False) -> Dict[str, Any]:
        """Import a JSON topology into the graph database, either from a file or raw dict."""
        if not graph_data:
            if not filepath:
                raise ValueError("Either filepath or graph_data must be provided.")
            with open(filepath, "r") as f:
                graph_data = json.load(f)
            
        from src.usecases.model_graph import ModelGraphUseCase
        uc = ModelGraphUseCase(self.repo)
        
        result = uc.execute(graph_data, clear=clear)
        stats = {
            "nodes_imported": result.nodes_imported,
            "edges_imported": result.edges_imported,
            "duration_ms": result.duration_ms,
            "success": result.success,
        }
        if result.details:
            stats.update(result.details)
        return stats

    def analyze(self, layer: str = "app", **kwargs) -> AnalysisResult:
        """Analyze the structural graph topology."""
        from src.usecases.analyze_graph import AnalyzeGraphUseCase
        uc = AnalyzeGraphUseCase(self.repo)
        raw_analysis = uc.execute(layer=layer)
        return AnalysisResult(raw_analysis)

    def predict(self, analysis: AnalysisResult, equal_weights: bool = False, ahp_shrinkage: float = 0.7) -> PredictionResult:
        """Predict quality metrics using the statistical quality analyser.

        Args:
            analysis:      Result of a prior ``analyze()`` call.
            equal_weights: If True, use equal 0.25 weights for all Q(v) dimensions.
            ahp_shrinkage: Shrinkage factor for AHP weights (default: 0.7).
        """
        from src.prediction.service import PredictionService
        service = PredictionService(
            use_ahp=True, 
            equal_weights=equal_weights,
            ahp_shrinkage=ahp_shrinkage
        )
        from src.usecases.predict_graph import PredictGraphUseCase
        uc = PredictGraphUseCase(self.repo, prediction_service=service)

        layer_str = getattr(analysis.raw.layer, "value", str(analysis.raw.layer))
        quality, _ = uc.execute(
            layer=layer_str,
            structural_result=analysis.raw,
            detect_problems=False,
        )
        return PredictionResult(quality)

    def detect_antipatterns(self, prediction: PredictionResult) -> List[Any]:
        """Detect architectural anti-patterns from the GNN prediction results."""
        from src.prediction.service import PredictionService
        service = PredictionService()
        return service.detect_problems(prediction.raw)

    def simulate(self, layer: str = "system", mode: str = "exhaustive", **kwargs) -> Any:
        """Simulate resilience scenarios (e.g. cascading failures)."""
        from src.usecases.simulate_graph import SimulateGraphUseCase
        from src.usecases.models import SimulationMode
        
        uc = SimulateGraphUseCase(self.repo)
        
        sim_mode_enum = SimulationMode.EXHAUSTIVE
        for m in SimulationMode:
            if m.value.lower() == mode.lower():
                sim_mode_enum = m
                break
                
        return uc.execute(layer=layer, mode=sim_mode_enum, **kwargs)

    def validate(self, layers: Optional[List[str]] = None) -> ValidationPipelineFacade:
        """Validate pipeline accuracy across specified layers."""
        if layers is None:
            layers = ["system"]
            
        from src.usecases.validate_graph import ValidateGraphUseCase
        uc = ValidateGraphUseCase(self.repo)
        pipeline_result = uc.execute(layers=layers)
        
        return ValidationPipelineFacade(pipeline_result)

    def visualize(self, output: str = "report.html", layers: Optional[List[str]] = None, **kwargs) -> str:
        """Render the logic to an HTML report."""
        if layers is None:
            layers = ["system"]
            
        from src.usecases.visualize_graph import VisualizeGraphUseCase
        from src.usecases.models import VisOptions
        
        uc = VisualizeGraphUseCase(self.repo)
        
        options = VisOptions()
        if "include_network" in kwargs: options.include_network = kwargs["include_network"]
        if "include_matrix" in kwargs: options.include_matrix = kwargs["include_matrix"]
        if "include_validation" in kwargs: options.include_validation = kwargs["include_validation"]
        if "antipatterns_file" in kwargs: options.antipatterns_file = kwargs["antipatterns_file"]
        if "multi_seed" in kwargs: options.multi_seed = kwargs["multi_seed"]
        
        return uc.execute(layers=layers, output_file=output, options=options)
