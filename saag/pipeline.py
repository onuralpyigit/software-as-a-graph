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
        #: simulate()'s own layer, kept separate from self._layer so that
        #: `.analyze("app").simulate("infra")` cannot silently redirect
        #: analyze's layer to "infra" — the two used to share one field.
        #: None means "use whatever analyze()/self._layer resolved to".
        self._simulate_layer: Optional[str] = None
        self._predict_kwargs: dict = {}
        self._triage_kwargs: dict = {}
        self._simulate_kwargs: dict = {}
        self._visualize_kwargs: dict = {}
        self._clear: bool = False
        self._validate_layers: Optional[List[str]] = None

        self._do_analyze = False
        self._do_predict = False
        self._do_triage = False
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

    def analyze(self, layer: str = "system") -> "Pipeline":
        """Stage 2: Deterministic structural analysis — topology metrics only
        (PageRank, betweenness, closeness, articulation points, etc.).

        Produces a fully deterministic AnalysisResult from topology and metadata.
        No RM/Q scores or anti-patterns here — see predict() for the unified
        Prediction Step that produces those.
        """
        self._layer = layer
        self._do_analyze = True
        return self

    def predict(
        self,
        mode: str = "gnn",
        gnn_checkpoint: Optional[str] = None,
        **kwargs,
    ) -> "Pipeline":
        """Stage 3: Unified Prediction Step — rule-based (RM) + ML (GNN) scoring.

        Always computes the AHP-weighted RM composite (deterministic, closed-form).
        When a trained GNN checkpoint is available, blends in a Heterogeneous Graph
        Transformer (HGT / HGTConv) inference pass that learns patterns the RM
        composite cannot encode (nonlinear interactions, multi-hop motifs); falls
        back to RM otherwise. Also runs anti-pattern detection and generates a
        human-readable explanation. This replaces the legacy "Quality Scoring" step
        that used to live inside Analyze.
        Requires analyze() to have been configured first.

        Parameters
        ----------
        mode:
            'gnn' for raw GNN scores (default) when a checkpoint is available.
        gnn_checkpoint:
            Path to a GNN checkpoint directory. Defaults to output/gnn_checkpoints.
        **kwargs:
            Forwarded to ``Client.predict`` — the RM weighting/normalisation
            options (``use_ahp``, ``equal_weights``, ``ahp_shrinkage``,
            ``normalization_method``, ``winsorize``, ``winsorize_limit``,
            ``run_sensitivity``, ``active_patterns``). These belong to the
            Predict stage, not analyze() — pass them here, not to analyze().
        """
        self._do_predict = True
        self._predict_kwargs = {"mode": mode, "gnn_checkpoint": gnn_checkpoint, **kwargs}
        return self

    def triage(self, k: int = 10, node_types: Optional[List[str]] = None) -> "Pipeline":
        """Triage bridge — scope Pathway A's (RM) root-cause diagnosis to the
        Top-K components Pathway B's ranking (GNN when a checkpoint was
        available at predict()-time, RM otherwise) flagged as critical.

        Requires predict() to have been configured first. Each shortlisted
        component is joined back to its RM CriticalityProfile pattern,
        elevated dimensions, priority action, and stakeholder roles by id —
        never read off the ranking itself, since a GNN result carries no
        root cause of its own.
        """
        self._do_triage = True
        self._triage_kwargs = {"k": k, "node_types": node_types}
        return self

    def simulate(self, layer: Optional[str] = None, mode: str = "exhaustive", **kwargs) -> "Pipeline":
        """Stage 4: Simulate cascading failures.

        layer defaults to whatever analyze() was configured with when not
        given explicitly here.
        """
        self._simulate_layer = layer
        self._do_simulate = True
        self._simulate_kwargs = {"mode": mode, **kwargs}
        return self

    def validate(self, layers: Optional[List[str]] = None) -> "Pipeline":
        """Stage 5: Validate prediction vs simulation ground truth."""
        self._do_validate = True
        self._validate_layers = layers
        return self

    def prescribe(self) -> "Pipeline":
        """Stage 6: Prescriptive remediation generation."""
        self._do_prescribe = True
        return self

    def visualize(self, output: str = "report.html", layers: Optional[List[str]] = None, **kwargs) -> "Pipeline":
        """Stage 7: Generate HTML dashboard report."""
        self._do_visualize = True
        self._visualize_kwargs = {"output": output, "layers": layers, **kwargs}
        return self

    def run(self) -> PipelineExecutionResult:
        """Execute all configured stages sequentially and compile results.

        Execution order (Import -> Analyze -> Simulate -> Predict -> Validate
        -> Prescribe -> Visualize) does not match pipeline stage *identity*
        numbers (Predict is Stage 3, Simulate is Stage 4) — Simulate runs
        before Predict so a first run can generate the ground-truth labels
        GNN training needs before Predict's checkpoint-dependent path runs.
        See ARCHITECTURE.md's first-run sequencing note. Comments below name
        both: execution position and stage identity.
        """
        result = PipelineExecutionResult()

        # Execution step 1 — Import
        if getattr(self, "file_path", None):
            logger.info(f"Importing topology from {self.file_path}")
            self.client.import_topology(self.file_path, clear=self._clear)

        # Execution step 2 (Stage 2: Analyze) — deterministic structural metrics only
        if self._do_analyze:
            logger.info(f"Analyzing layer '{self._layer}': structural metrics")
            result.analysis = self.client.analyze(layer=self._layer)

        # Execution step 3 (Stage 4: Simulate) — counterfactual cascade engine;
        # generates ground-truth labels. Uses its own layer, defaulting to
        # analyze()'s, so simulate("infra") cannot redirect analyze's layer.
        if self._do_simulate:
            logger.info("Running fault simulation (cascade ground truth)...")
            sim_layer = self._simulate_layer if self._simulate_layer is not None else self._layer
            result.simulation = self.client.simulate(layer=sim_layer, **self._simulate_kwargs)

        # Execution step 4 (Stage 3: Predict) — unified: RM (always) + GNN
        # (when available) + anti-patterns
        if self._do_predict:
            if result.analysis is None:
                raise RuntimeError(
                    "predict() requires an AnalysisResult. "
                    "Either call analyze() in this pipeline or pass a stored result via client.predict()."
                )

            # Fail-fast check for GNN checkpoint / simulation labels. Reuses
            # PredictionService's own checkpoint probe rather than a second
            # copy of the same three-file existence test (a third copy lived
            # in cli/run.py, at a *different* default path — collapsed here).
            checkpoint_dir = self._predict_kwargs.get("gnn_checkpoint") or "output/gnn_checkpoints"
            from saag.prediction.service import PredictionService
            has_ckpt = PredictionService._has_checkpoint(checkpoint_dir)
            if not has_ckpt and not self._do_simulate:
                raise RuntimeError(
                    f"GNN Prediction requested but no trained GNN checkpoint was found at '{checkpoint_dir}' "
                    "and no fault simulation was run to generate training labels. "
                    "To run GNN prediction, you must either provide a valid checkpoint or run the simulate stage "
                    "first to generate labels for training."
                )

            logger.info("Running unified Prediction step (RM + GNN)...")
            result.prediction = self.client.predict(result.analysis, **self._predict_kwargs)

        # Execution step 4b — Triage bridge: scope Pathway A's RM diagnosis
        # to Pathway B's Top-K shortlist. Runs immediately after Predict,
        # on whatever ranking Predict produced (GNN or RM fallback).
        if self._do_triage:
            if result.prediction is None:
                raise RuntimeError(
                    "triage() requires a PredictionResult. "
                    "Make sure to call predict() in the pipeline."
                )
            logger.info("Running Triage bridge...")
            result.triage = self.client.triage(result.prediction, **self._triage_kwargs)

        # Execution step 5 (Stage 5: Validate) — compare Predict/Analyze
        # output against Simulate ground truth
        if self._do_validate:
            logger.info("Validating against simulation ground truth...")
            validate_layers = self._validate_layers or [self._layer]
            gnn_checkpoint = self._predict_kwargs.get("gnn_checkpoint")
            result.validation = self.client.validate(layers=validate_layers, gnn_checkpoint=gnn_checkpoint)

        # Execution step 6 (Stage 6: Prescribe) — generate recommendations
        # and verify them in closed loop
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

        # Execution step 7 (Stage 7: Visualize)
        if self._do_visualize:
            out_file = self._visualize_kwargs.pop("output", "report.html")
            vis_layers = self._visualize_kwargs.pop("layers", [self._layer]) or [self._layer]
            logger.info(f"Generating visualization report → {out_file}")
            self.client.visualize(output=out_file, layers=vis_layers, **self._visualize_kwargs)

        logger.info("Pipeline execution complete.")
        return result
