"""
saag/pipeline.py
"""
import logging
from typing import List, Optional

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
        self._predict_kwargs: dict = {}
        self._simulate_kwargs: dict = {}
        self._visualize_kwargs: dict = {}

        self._do_analyze = False
        self._do_predict = False
        self._do_simulate = False
        self._do_validate = False
        self._do_prescribe = False
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
        """Stage 2: Deterministic structural analysis — topology metrics only
        (PageRank, betweenness, closeness, articulation points, etc.).

        Produces a fully deterministic AnalysisResult from topology and metadata.
        No RMAV/Q scores or anti-patterns here — see predict() for the unified
        Prediction Step that produces those.
        """
        self._layer = layer
        self._do_analyze = True
        self._analyze_kwargs = kwargs
        return self

    def predict(self, mode: str = "gnn", gnn_checkpoint: Optional[str] = None) -> "Pipeline":
        """Stage 3: Unified Prediction Step — rule-based (RMAV) + ML (GNN) scoring.

        Always computes the AHP-weighted RMAV composite (deterministic, closed-form).
        When a trained GNN checkpoint is available, blends in a Heterogeneous Graph
        Transformer (HGT / HGTConv) inference pass that learns patterns the RMAV
        composite cannot encode (nonlinear interactions, multi-hop motifs); falls
        back to RMAV otherwise. Also runs anti-pattern detection and generates a
        human-readable explanation. This replaces the legacy "Quality Scoring" step
        that used to live inside Analyze.
        Requires analyze() to have been configured first.

        Parameters
        ----------
        mode:
            'gnn' for raw GNN scores (default) when a checkpoint is available.
        gnn_checkpoint:
            Path to a GNN checkpoint directory. Defaults to output/gnn_checkpoints.
        """
        self._do_predict = True
        self._predict_kwargs = {"mode": mode, "gnn_checkpoint": gnn_checkpoint}
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

    def prescribe(self) -> "Pipeline":
        """Stage 6: Prescriptive remediation generation."""
        self._do_prescribe = True
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
            self.client.import_topology(self.file_path, clear=getattr(self, "_clear", False))

        # 2. Analyze — deterministic: structural metrics only
        if self._do_analyze:
            logger.info(f"Analyzing layer '{self._layer}': structural metrics")
            result.analysis = self.client.analyze(layer=self._layer, **self._analyze_kwargs)

        # 3. Simulate — counterfactual cascade engine; generates ground-truth labels
        if self._do_simulate:
            logger.info("Running fault simulation (cascade ground truth)...")
            result.simulation = self.client.simulate(layer=self._layer, **self._simulate_kwargs)

        # 4. Predict — unified: RMAV (always) + GNN (when available) + anti-patterns
        if self._do_predict:
            if result.analysis is None:
                raise RuntimeError(
                    "predict() requires an AnalysisResult. "
                    "Either call analyze() in this pipeline or pass a stored result via client.predict()."
                )

            # Fail-fast check for GNN checkpoint / simulation labels
            checkpoint_dir = self._predict_kwargs.get("gnn_checkpoint") or "output/gnn_checkpoints"
            from pathlib import Path
            p = Path(checkpoint_dir)
            has_ckpt = p.exists() and (p / "service_config.json").exists() and (
                (p / "node_model.pt").exists() or (p / "best_model.pt").exists()
            )
            if not has_ckpt and not self._do_simulate:
                raise RuntimeError(
                    f"GNN Prediction requested but no trained GNN checkpoint was found at '{checkpoint_dir}' "
                    "and no fault simulation was run to generate training labels. "
                    "To run GNN prediction, you must either provide a valid checkpoint or run the simulate stage "
                    "first to generate labels for training."
                )

            logger.info("Running unified Prediction step (RMAV + GNN)...")
            result.prediction = self.client.predict(result.analysis, **self._predict_kwargs)

        # 5. Validate — compare Predict/Analyze output against Simulate ground truth
        if self._do_validate:
            logger.info("Validating against simulation ground truth...")
            validate_layers = getattr(self, "_validate_layers", [self._layer]) or [self._layer]
            gnn_checkpoint = self._predict_kwargs.get("gnn_checkpoint")
            result.validation = self.client.validate(layers=validate_layers, gnn_checkpoint=gnn_checkpoint)

        # 6. Prescribe — generate recommendations and verify them in closed loop
        if self._do_prescribe:
            if result.analysis is None:
                raise RuntimeError(
                    "prescribe() requires an AnalysisResult. "
                    "Make sure to call analyze() in the pipeline."
                )
            logger.info("Running prescriptive remediation (Stage 6)...")
            gnn_checkpoint = self._predict_kwargs.get("gnn_checkpoint")
            result.prescription = self.client.prescribe(
                analysis_result=result.analysis,
                prediction_result=result.prediction,
                layer=self._layer,
                gnn_checkpoint=gnn_checkpoint
            )

        # 7. Visualize
        if self._do_visualize:
            out_file = self._visualize_kwargs.pop("output", "report.html")
            vis_layers = self._visualize_kwargs.pop("layers", [self._layer]) or [self._layer]
            logger.info(f"Generating visualization report → {out_file}")
            self.client.visualize(output=out_file, layers=vis_layers, **self._visualize_kwargs)

        logger.info("Pipeline execution complete.")
        return result
