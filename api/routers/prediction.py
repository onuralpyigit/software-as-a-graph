"""
Prediction endpoints: GNN training and inference.
"""

import json
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import asyncio
import logging

from api.dependencies import get_client
from api.presenters.prediction_presenter import (
    format_checkpoint_info,
    format_edge_scores,
    format_metrics,
    format_node_scores,
    format_summary,
)
from api.presenters.triage_presenter import (
    categorize_by_stakeholder,
    format_triage_result,
)
from saag import Client

router = APIRouter(prefix="/api/v1/graph/prediction", tags=["prediction"])
logger = logging.getLogger(__name__)


# ── Request / Response models ─────────────────────────────────────────────────

class TrainRequest(BaseModel):
    credentials: Dict[str, Any] = Field(..., description="Neo4j connection credentials")
    layer: str = Field(default="app", description="Graph layer to train on")
    checkpoint_name: str = Field(default="", description="Optional checkpoint folder name (default: auto datetime)")
    hidden: int = Field(default=64, description="Hidden dimension size")
    heads: int = Field(default=4, description="Number of attention heads")
    layers: int = Field(default=3, description="Number of GNN layers")
    dropout: float = Field(default=0.2, description="Dropout rate")
    epochs: int = Field(default=300, description="Maximum training epochs")
    lr: float = Field(default=3e-4, description="Learning rate")
    patience: int = Field(default=30, description="Early-stopping patience")
    train_ratio: float = Field(default=0.6, description="Training split fraction")
    val_ratio: float = Field(default=0.2, description="Validation split fraction")
    use_ahp: bool = Field(default=False, description="Use AHP weights for RM")
    predict_edges: bool = Field(default=True, description="Also predict edge criticality")
    variant: str = Field(
        default="hetero_qos",
        description=(
            "Model architecture variant: "
            "'hetero_qos' (QoS-aware HGT, default), "
            "'homo_unweighted' (flat GAT, no edge_attr), "
            "'homo_scalar' (flat GAT, scalar weight), "
            "'topology_rm' (RM baseline, no GNN). "
            "Paper-name mapping (reproduce/EXPERIMENTS.md §3 Model Variants): "
            "hetero_qos=HGL-QoS, homo_unweighted=GL, homo_scalar=GL-QoS."
        ),
    )



class GNNScoreModel(BaseModel):
    component: str
    node_name: str = ""
    composite_score: float
    reliability_score: float
    maintainability_score: float
    criticality_level: str
    source: str


class GNNEdgeScoreModel(BaseModel):
    source: str
    source_name: str = ""
    target: str
    target_name: str = ""
    edge_type: str
    composite_score: float
    criticality_level: str


class GNNMetricsModel(BaseModel):
    spearman_rho: Optional[float] = None
    f1_score: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    ndcg_10: Optional[float] = None


class TrainSummaryModel(BaseModel):
    total_components: int
    critical: int
    high: int
    medium: int
    low: int
    minimal: int
    critical_edges: int


class TrainResponse(BaseModel):
    success: bool
    layer: str
    checkpoint_dir: str
    summary: TrainSummaryModel
    gnn_metrics: Optional[GNNMetricsModel] = None
    top_critical: List[GNNScoreModel]
    top_critical_edges: List[GNNEdgeScoreModel]


class PredictRequest(BaseModel):
    credentials: Dict[str, Any] = Field(..., description="Neo4j connection credentials")
    layer: str = Field(default="app", description="Graph layer to analyse")
    checkpoint_dir: str = Field(default="", description="Path to saved checkpoints (empty = repo output/gnn_checkpoints)")


class PredictResponse(BaseModel):
    success: bool
    layer: str
    checkpoint_dir: str
    summary: TrainSummaryModel
    scores: List[GNNScoreModel]
    edge_scores: List[GNNEdgeScoreModel]


class StakeholderGroupModel(BaseModel):
    role_name: str
    focus: str
    count: int
    items: List[Dict[str, Any]]


class StakeholderSummaryModel(BaseModel):
    devops_sre: StakeholderGroupModel
    architect: StakeholderGroupModel
    developer: StakeholderGroupModel


class TriageRequest(BaseModel):
    credentials: Dict[str, Any] = Field(..., description="Neo4j connection credentials")
    layer: str = Field(default="system", description="Graph layer to triage")
    checkpoint_dir: str = Field(default="", description="Optional GNN checkpoint path (falls back to RM in cold start)")
    k: int = Field(default=10, ge=1, description="Number of top critical components to shortlist")
    node_types: Optional[List[str]] = Field(default=None, description="Optional node type filter (e.g. ['Application', 'Topic'])")


class TriageResponse(BaseModel):
    success: bool
    layer: str
    k: int
    ranking_source: str
    population: int
    stakeholders: StakeholderSummaryModel
    entries: List[Dict[str, Any]]


# ── Endpoints ─────────────────────────────────────────────────────────────────

# Anchor to repo root (api/routers/ → api/ → repo root)
# so the API always shares the same output/ directory as the CLI scripts.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GNN_CHECKPOINTS_DIR: Path = _REPO_ROOT / "output" / "gnn_checkpoints"


class CheckpointInfo(BaseModel):
    path: str
    name: str
    layer: str
    hidden_channels: int
    num_heads: int
    num_layers: int
    dropout: float
    predict_edges: bool
    has_node_model: bool
    has_edge_model: bool


class CheckpointListResponse(BaseModel):
    checkpoints: List[CheckpointInfo]


@router.get("/checkpoints", response_model=CheckpointListResponse)
async def list_checkpoints():
    """
    Return all valid GNN checkpoints found under ``output/gnn_checkpoints/``,
    sorted newest-first (directory names are YYYY-MM-DD_HH-MM-SS).
    """
    found: List[CheckpointInfo] = []
    ckpt_root = _GNN_CHECKPOINTS_DIR.resolve()
    if ckpt_root.exists():
        for sub in sorted(ckpt_root.iterdir(), reverse=True):
            if sub.is_dir():
                info = format_checkpoint_info(sub)
                if info:
                    found.append(CheckpointInfo(**info))
    return CheckpointListResponse(checkpoints=found)


@router.post("/train", response_model=TrainResponse)
async def train_gnn(
    request: TrainRequest,
    client: Client = Depends(get_client),
):
    """
    Train a Heterogeneous Graph Transformer (HGT) on the current
    graph topology.  Runs structural analysis and failure simulation as
    prerequisites, then trains the GNN and saves model checkpoints.
    """
    try:
        from saag.prediction import GNNService, extract_structural_metrics_dict, \
            extract_rm_scores_dict, extract_simulation_dict
        from saag.simulation import SimulationService
    except ImportError as e:
        raise HTTPException(status_code=501, detail=f"GNN module not available: {e}")

    def _run_training():
        import re
        raw_name = (request.checkpoint_name or "").strip()
        # Sanitise: keep only alphanumeric, dashes, underscores, dots
        safe_name = re.sub(r"[^\w.\-]", "_", raw_name) if raw_name else ""
        folder_name = safe_name if safe_name else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ckpt_dir = _GNN_CHECKPOINTS_DIR / folder_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        logger.info("GNN training: layer=%s epochs=%d checkpoint_dir=%s", request.layer, request.epochs, ckpt_dir)

        # Step 2+3: structural analysis + RM scores
        from saag.analysis.structural_analyzer import StructuralAnalyzer
        from saag.core.layers import AnalysisLayer
        graph_data = client.repo.get_graph_data()
        struct_analyzer = StructuralAnalyzer()
        layer_enum = AnalysisLayer.from_string(request.layer)
        struct_result = struct_analyzer.analyze(graph_data, layer=layer_enum)
        nx_graph = struct_result.graph
        if nx_graph is None:
            import networkx as nx
            nx_graph = nx.DiGraph()
        if nx_graph.number_of_nodes() == 0:
            raise ValueError(
                f"Layer '{request.layer}' has no nodes. "
                "Make sure the graph is imported and the correct layer is selected."
            )
        structural_dict = extract_structural_metrics_dict(struct_result)

        from saag.prediction.service import PredictionService
        pred_svc = PredictionService(use_ahp=request.use_ahp)
        quality_result = pred_svc.predict_quality(struct_result)
        rm_dict = extract_rm_scores_dict(quality_result)

        # Step 4: simulation ground truth
        sim_svc = SimulationService(client.repo)
        sim_results = sim_svc.run_failure_simulation_exhaustive(layer=request.layer)
        simulation_dict = extract_simulation_dict(sim_results)

        # Train GNN
        gnn_svc = GNNService(
            hidden_channels=request.hidden,
            num_heads=request.heads,
            num_layers=request.layers,
            dropout=request.dropout,
            predict_edges=request.predict_edges,
            checkpoint_dir=str(ckpt_dir),
        )
        gnn_result = gnn_svc.train(
            graph=nx_graph,
            structural_metrics=structural_dict,
            simulation_results=simulation_dict,
            rm_scores=rm_dict,
            train_ratio=request.train_ratio,
            val_ratio=request.val_ratio,
            num_epochs=request.epochs,
            lr=request.lr,
            patience=request.patience,
        )

        # Persist the layer into service_config.json so the predict page can read it
        cfg_path = ckpt_dir / "service_config.json"
        if cfg_path.exists():
            try:
                cfg_data = json.loads(cfg_path.read_text())
                cfg_data["layer"] = request.layer
                cfg_path.write_text(json.dumps(cfg_data, indent=2))
            except Exception:
                pass

        name_lookup = {node: attrs.get("name", node) for node, attrs in nx_graph.nodes(data=True)}
        return ckpt_dir, gnn_result, name_lookup

    try:
        ckpt_dir, gnn_result, name_lookup = await asyncio.to_thread(_run_training)

        return TrainResponse(
            success=True,
            layer=request.layer,
            checkpoint_dir=str(ckpt_dir),
            summary=format_summary(gnn_result),
            gnn_metrics=format_metrics(gnn_result.gnn_metrics),
            top_critical=format_node_scores(gnn_result.top_critical_nodes(n=10), name_lookup),
            top_critical_edges=format_edge_scores(gnn_result.top_critical_edges(n=10), name_lookup),
        )
    except Exception as e:
        logger.error("GNN training failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


@router.delete("/checkpoints/{name}")
async def delete_checkpoint(name: str):
    """
    Delete a checkpoint directory from ``output/gnn_checkpoints/``.
    Only names that are direct children of the checkpoints root are accepted.
    """
    import shutil
    import re

    # Validate name: no path separators or special characters (prevents traversal)
    if not re.fullmatch(r"[\w.\-]+", name):
        raise HTTPException(status_code=400, detail="Invalid checkpoint name")

    target = (_GNN_CHECKPOINTS_DIR / name).resolve()
    # Make sure the resolved path is still inside the checkpoints dir (no traversal)
    if not str(target).startswith(str(_GNN_CHECKPOINTS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid checkpoint path")

    if not target.exists():
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    try:
        shutil.rmtree(target)
        logger.info("Deleted checkpoint: %s", target)
        return {"deleted": name}
    except Exception as e:
        logger.error("Failed to delete checkpoint %s: %s", name, e)
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")


@router.post("/predict", response_model=PredictResponse)
async def predict_gnn(
    request: PredictRequest,
    client: Client = Depends(get_client),
):
    """
    Run GNN inference on the current graph topology using saved checkpoints.
    Requires a trained model in ``checkpoint_dir``.
    """
    try:
        from saag.prediction import GNNService, extract_structural_metrics_dict, \
            extract_rm_scores_dict
    except ImportError as e:
        raise HTTPException(status_code=501, detail=f"GNN module not available: {e}")

    def _run_inference():
        logger.info("GNN inference: layer=%s checkpoint=%s", request.layer, request.checkpoint_dir)

        # Resolve empty checkpoint_dir to default repo path
        ckpt_dir = request.checkpoint_dir.strip() or str(_GNN_CHECKPOINTS_DIR)

        # Step 2+3: structural analysis + RM scores (needed for features AND metadata)
        from saag.analysis.structural_analyzer import StructuralAnalyzer
        from saag.core.layers import AnalysisLayer
        graph_data = client.repo.get_graph_data()
        struct_analyzer = StructuralAnalyzer()
        layer_enum = AnalysisLayer.from_string(request.layer)
        struct_result = struct_analyzer.analyze(graph_data, layer=layer_enum)
        nx_graph = struct_result.graph
        if nx_graph is None:
            import networkx as nx
            nx_graph = nx.DiGraph()
        if nx_graph.number_of_nodes() == 0:
            raise ValueError(
                f"Layer '{request.layer}' has no nodes. "
                "Make sure the graph is imported and the correct layer is selected."
            )
        structural_dict = extract_structural_metrics_dict(struct_result)

        from saag.prediction.service import PredictionService
        pred_svc = PredictionService(use_ahp=False)
        quality_result = pred_svc.predict_quality(struct_result)
        rm_dict = extract_rm_scores_dict(quality_result)

        # Load trained model — pass graph so from_checkpoint can reconstruct PyG metadata
        gnn_svc = GNNService.from_checkpoint(ckpt_dir, graph=nx_graph)

        gnn_result = gnn_svc.predict(
            graph=nx_graph,
            structural_metrics=structural_dict,
            rm_scores=rm_dict,
            mode="gnn",
        )
        name_lookup = {node: attrs.get("name", node) for node, attrs in nx_graph.nodes(data=True)}
        return ckpt_dir, gnn_result, name_lookup

    try:
        ckpt_dir, gnn_result, name_lookup = await asyncio.to_thread(_run_inference)

        ranked = sorted(
            gnn_result.node_scores.values(),
            key=lambda s: s.composite_score,
            reverse=True,
        )

        return PredictResponse(
            success=True,
            layer=request.layer,
            checkpoint_dir=ckpt_dir,
            summary=format_summary(gnn_result),
            scores=format_node_scores(ranked, name_lookup),
            edge_scores=format_edge_scores(gnn_result.top_critical_edges(n=20), name_lookup),
        )
    except Exception as e:
        logger.error("GNN inference failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@router.post("/triage", response_model=TriageResponse)
async def triage_components_endpoint(
    request: TriageRequest,
    client: Client = Depends(get_client),
):
    """
    Run Triage bridge: shortlists Top-K critical components (via GNN or RM fallback)
    and formats root-cause attribution categorized by stakeholder role (DevOps/SRE,
    Architect, Developer).
    """
    from saag.usecases.triage import TriageUseCase
    from saag.prediction.service import PredictionService
    from saag.analysis.structural_analyzer import StructuralAnalyzer
    from saag.core.layers import AnalysisLayer

    def _run_triage():
        graph_data = client.repo.get_graph_data()
        struct_analyzer = StructuralAnalyzer()
        layer_enum = AnalysisLayer.from_string(request.layer)
        struct_result = struct_analyzer.analyze(graph_data, layer=layer_enum)
        nx_graph = struct_result.graph

        ckpt_dir = request.checkpoint_dir.strip() or str(_GNN_CHECKPOINTS_DIR)
        has_gnn = Path(ckpt_dir).exists() and (
            (Path(ckpt_dir) / "node_model.pt").exists() or (Path(ckpt_dir) / "best_model.pt").exists()
        )

        pred_svc = PredictionService(
            gnn_checkpoint_dir=ckpt_dir if has_gnn else None,
            prefer_gnn=has_gnn,
        )

        pred_result = pred_svc.predict_quality_with_gnn(
            structural_result=struct_result,
            graph=nx_graph,
            layer=request.layer,
        )

        triage_uc = TriageUseCase()
        triage_res = triage_uc.execute(
            prediction_result=pred_result,
            k=request.k,
            layer=request.layer,
            node_types=request.node_types,
        )

        name_lookup = {}
        if nx_graph is not None:
            name_lookup = {node: attrs.get("name", node) for node, attrs in nx_graph.nodes(data=True)}

        stakeholder_data = categorize_by_stakeholder(triage_res, name_lookup=name_lookup)
        return triage_res, stakeholder_data

    try:
        triage_res, stakeholder_data = await asyncio.to_thread(_run_triage)
        return TriageResponse(
            success=True,
            layer=triage_res.layer,
            k=triage_res.k,
            ranking_source=triage_res.ranking_source,
            population=triage_res.population,
            stakeholders=stakeholder_data["stakeholders"],
            entries=[e.to_dict() for e in triage_res.entries],
        )
    except Exception as e:
        logger.error("Triage execution failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Triage execution failed: {e}")

