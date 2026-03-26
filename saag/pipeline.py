"""
saag/pipeline.py
"""
import logging
from typing import Optional, List, Any

from .client import Client
from .models import PipelineExecutionResult

logger = logging.getLogger(__name__)

class Pipeline:
    """
    A fluent builder for executing the full SoftwareAsAGraph analytical pipeline sequentially.
    """
    def __init__(
        self, 
        file_path: Optional[str] = None, 
        neo4j_uri: str = "bolt://localhost:7687", 
        user: str = "neo4j", 
        password: str = "password",
        repo=None
    ):
        if repo is not None:
            self.client = Client(repo=repo)
        else:
            self.client = Client(neo4j_uri=neo4j_uri, user=user, password=password)
        self.file_path = file_path
        
        self._layer: str = "system"
        self._analyze_kwargs: dict = {}
        self._simulate_kwargs: dict = {}
        self._visualize_kwargs: dict = {}
        
        # State tracking flags
        self._do_analyze = False
        self._do_simulate = False
        self._do_validate = False
        self._do_visualize = False

    @staticmethod
    def from_json(
        filepath: str, 
        neo4j_uri: str = "bolt://localhost:7687", 
        user: str = "neo4j", 
        password: str = "password",
        clear: bool = False,
        repo=None
    ) -> "Pipeline":
        """Initialize a new Pipeline starting from a JSON topology export."""
        pipeline = Pipeline(
            file_path=filepath, 
            neo4j_uri=neo4j_uri, 
            user=user, 
            password=password,
            repo=repo
        )
        pipeline._clear = clear
        return pipeline

    def analyze(self, layer: str = "system", **kwargs) -> "Pipeline":
        """Stage: Analyze structural metrics."""
        self._layer = layer
        self._do_analyze = True
        self._analyze_kwargs = kwargs
        return self

    def predict(self, **kwargs) -> "Pipeline":
        """Stage: Predict."""
        self._predict_kwargs = kwargs
        return self
        
    def simulate(self, layer: str = "system", mode: str = "exhaustive", **kwargs) -> "Pipeline":
        """Stage: Simulate cascading failures."""
        self._layer = layer
        self._do_simulate = True
        self._simulate_kwargs = {"mode": mode, **kwargs}
        return self

    def validate(self, layers: Optional[List[str]] = None) -> "Pipeline":
        """Stage: Validate prediction vs simulation ground truth."""
        self._do_validate = True
        self._validate_layers = layers
        return self

    def visualize(self, output: str = "report.html", layers: Optional[List[str]] = None, **kwargs) -> "Pipeline":
        """Stage: Generate HTML dashboard report."""
        self._do_visualize = True
        self._visualize_kwargs = {"output": output, "layers": layers, **kwargs}
        return self

    def run(self) -> PipelineExecutionResult:
        """Execute all configured stages sequentially and compile results."""
        result = PipelineExecutionResult()
        
        # 1. Import
        if getattr(self, "file_path", None):
            logger.info(f"Importing topology from {self.file_path}")
            clear = getattr(self, "_clear", False)
            self.client.import_topology(self.file_path, clear=clear)
        
        # 2. Analyze
        if self._do_analyze:
            logger.info(f"Running structural analysis on layer: {self._layer}")
            result.analysis = self.client.analyze(layer=self._layer, **self._analyze_kwargs)
            
            # 3. Predict & Detect Problems
            logger.info("Predicting quality metrics and detecting antipatterns...")
            predict_kwargs = getattr(self, "_predict_kwargs", {})
            result.prediction = self.client.predict(result.analysis, **predict_kwargs)
            result.problems = self.client.detect_antipatterns(result.prediction)
            
        # 4. Simulate
        if self._do_simulate:
            logger.info("Running fault simulation...")
            result.simulation = self.client.simulate(layer=self._layer, **self._simulate_kwargs)
            
        # 5. Validate
        if self._do_validate:
            logger.info("Validating pipeline predictions against simulation...")
            validate_layers = getattr(self, "_validate_layers", [self._layer]) or [self._layer]
            result.validation = self.client.validate(layers=validate_layers)
            
        # 6. Visualize
        if self._do_visualize:
            out_file = self._visualize_kwargs.pop("output", "report.html")
            vis_layers = self._visualize_kwargs.pop("layers", [self._layer]) or [self._layer]
            logger.info(f"Generating visualization report to {out_file}...")
            self.client.visualize(output=out_file, layers=vis_layers, **self._visualize_kwargs)
            
        logger.info("Pipeline execution complete.")
        return result
