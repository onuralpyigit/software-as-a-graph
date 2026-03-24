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
from saag import Client

router = APIRouter(prefix="/api/v1/prediction", tags=["prediction"])
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
    use_ahp: bool = Field(default=False, description="Use AHP weights for RMAV")
    predict_edges: bool = Field(default=True, description="Also predict edge criticality")


class GNNScoreModel(BaseModel):
    component: str
    node_name: str = ""
    composite_score: float
    reliability_score: float
    maintainability_score: float
    availability_score: float
    vulnerability_score: float
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
    ensemble_metrics: Optional[GNNMetricsModel] = None
    ensemble_alpha: Optional[List[float]] = None
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _metrics_from_eval(m) -> Optional[GNNMetricsModel]:
    if m is None:
        return None
    return GNNMetricsModel(
        spearman_rho=getattr(m, "spearman_rho", None),
        f1_score=getattr(m, "f1_score", None),
        rmse=getattr(m, "rmse", None),
        mae=getattr(m, "mae", None),
        ndcg_10=getattr(m, "ndcg_10", None),
    )


def _node_score_model(s, name_lookup: dict = {}) -> GNNScoreModel:
    d = s.to_dict()
    node_id = d["component"]
    return GNNScoreModel(
        component=node_id,
        node_name=name_lookup.get(node_id, node_id),
        composite_score=d["composite_score"],
        reliability_score=d["reliability_score"],
        maintainability_score=d["maintainability_score"],
        availability_score=d["availability_score"],
        vulnerability_score=d["vulnerability_score"],
        criticality_level=d["criticality_level"].upper(),
        source=d.get("source", "GNN"),
    )


def _edge_score_model(e, name_lookup: dict = {}) -> GNNEdgeScoreModel:
    d = e.to_dict()
    src = d["source"]
    tgt = d["target"]
    return GNNEdgeScoreModel(
        source=src,
        source_name=name_lookup.get(src, src),
        target=tgt,
        target_name=name_lookup.get(tgt, tgt),
        edge_type=d["edge_type"],
        composite_score=d["composite_score"],
        criticality_level=d["criticality_level"].upper(),
    )


def _build_summary(result) -> TrainSummaryModel:
    scores = result.ensemble_scores or result.node_scores
    levels = [s.criticality_level.upper() for s in scores.values()]
    return TrainSummaryModel(
        total_components=len(scores),
        critical=levels.count("CRITICAL"),
        high=levels.count("HIGH"),
        medium=levels.count("MEDIUM"),
        low=levels.count("LOW"),
        minimal=levels.count("MINIMAL"),
        critical_edges=sum(
            1 for e in result.edge_scores if e.criticality_level.upper() == "CRITICAL"
        ),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

# Anchor to repo root (backend/api/routers/ → ../../.. → repo root)
# so the API always shares the same output/ directory as the CLI scripts.
_REPO_ROOT = Path(__file__).resolve().parents[3]
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
    has_ensemble: bool


class CheckpointListResponse(BaseModel):
    checkpoints: List[CheckpointInfo]


def _read_checkpoint_info(directory: Path) -> Optional[CheckpointInfo]:
    """Return a CheckpointInfo if *directory* looks like a valid GNN checkpoint."""
    cfg_path = directory / "service_config.json"
    node_path = directory / "node_model.pt"
    best_path = directory / "best_model.pt"
    # Accept either node_model.pt or best_model.pt as the model file marker
    has_any_model = node_path.exists() or best_path.exists()
    if not cfg_path.exists() or not has_any_model:
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return None
    return CheckpointInfo(
        path=str(directory),
        name=directory.name,
        layer=cfg.get("layer", ""),
        hidden_channels=cfg.get("hidden_channels", 64),
        num_heads=cfg.get("num_heads", 4),
        num_layers=cfg.get("num_layers", 3),
        dropout=cfg.get("dropout", 0.2),
        predict_edges=cfg.get("predict_edges", True),
        has_node_model=node_path.exists(),
        has_edge_model=(directory / "edge_model.pt").exists(),
        has_ensemble=(directory / "ensemble.pt").exists() or best_path.exists(),
    )


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
                info = _read_checkpoint_info(sub)
                if info:
                    found.append(info)
    return CheckpointListResponse(checkpoints=found)


@router.post("/train", response_model=TrainResponse)
async def train_gnn(
    request: TrainRequest,
    client: Client = Depends(get_client),
):
    """
    Train a Heterogeneous Graph Attention Network (HeteroGAT) on the current
    graph topology.  Runs structural analysis and failure simulation as
    prerequisites, then trains the GNN and saves model checkpoints.
    """
    try:
        from src.prediction import GNNService, extract_structural_metrics_dict, \
            extract_rmav_scores_dict, extract_simulation_dict
        from src.simulation import SimulationService
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

        # Step 2+3: structural analysis + RMAV scores
        from src.analysis.structural_analyzer import StructuralAnalyzer
        from src.core.layers import AnalysisLayer
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

        from src.prediction.service import PredictionService
        pred_svc = PredictionService(use_ahp=request.use_ahp)
        quality_result = pred_svc.predict_quality(struct_result)
        rmav_dict = extract_rmav_scores_dict(quality_result)

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
            rmav_scores=rmav_dict,
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

        top_nodes = [_node_score_model(s, name_lookup) for s in gnn_result.top_critical_nodes(n=10)]
        top_edges = [_edge_score_model(e, name_lookup) for e in gnn_result.top_critical_edges(n=10)]

        return TrainResponse(
            success=True,
            layer=request.layer,
            checkpoint_dir=str(ckpt_dir),
            summary=_build_summary(gnn_result),
            gnn_metrics=_metrics_from_eval(gnn_result.gnn_metrics),
            ensemble_metrics=_metrics_from_eval(gnn_result.ensemble_metrics),
            ensemble_alpha=gnn_result.ensemble_alpha,
            top_critical=top_nodes,
            top_critical_edges=top_edges,
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
        from src.prediction import GNNService, extract_structural_metrics_dict, \
            extract_rmav_scores_dict
    except ImportError as e:
        raise HTTPException(status_code=501, detail=f"GNN module not available: {e}")

    def _run_inference():
        logger.info("GNN inference: layer=%s checkpoint=%s", request.layer, request.checkpoint_dir)

        # Resolve empty checkpoint_dir to default repo path
        ckpt_dir = request.checkpoint_dir.strip() or str(_GNN_CHECKPOINTS_DIR)

        # Step 2+3: structural analysis + RMAV scores (needed for features AND metadata)
        from src.analysis.structural_analyzer import StructuralAnalyzer
        from src.core.layers import AnalysisLayer
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

        from src.prediction.service import PredictionService
        pred_svc = PredictionService(use_ahp=False)
        quality_result = pred_svc.predict_quality(struct_result)
        rmav_dict = extract_rmav_scores_dict(quality_result)

        # Load trained model — pass graph so from_checkpoint can reconstruct PyG metadata
        gnn_svc = GNNService.from_checkpoint(ckpt_dir, graph=nx_graph)

        gnn_result = gnn_svc.predict(
            graph=nx_graph,
            structural_metrics=structural_dict,
            rmav_scores=rmav_dict,
        )
        name_lookup = {node: attrs.get("name", node) for node, attrs in nx_graph.nodes(data=True)}
        return ckpt_dir, gnn_result, name_lookup

    try:
        ckpt_dir, gnn_result, name_lookup = await asyncio.to_thread(_run_inference)

        scores = [_node_score_model(s, name_lookup) for s in sorted(
            (gnn_result.ensemble_scores or gnn_result.node_scores).values(),
            key=lambda s: s.composite_score,
            reverse=True,
        )]
        edge_scores = [_edge_score_model(e, name_lookup) for e in gnn_result.top_critical_edges(n=20)]

        return PredictResponse(
            success=True,
            layer=request.layer,
            checkpoint_dir=ckpt_dir,
            summary=_build_summary(gnn_result),
            scores=scores,
            edge_scores=edge_scores,
        )
    except Exception as e:
        logger.error("GNN inference failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
