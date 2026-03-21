"""
GNN Service
===========
High-level integration point between the GNN module and the existing
Software-as-a-Graph pipeline.

This service mirrors the interface of ``AnalysisService`` and
``SimulationService`` so it can be dropped into the existing six-step
pipeline without structural changes.  It adds a **Step 3.5** between
Prediction (Step 3) and Failure Simulation (Step 4):

    Step 3  →  RMAV Q*(v)
    Step 3.5 → GNN  Q_GNN(v)   ← NEW
    Ensemble → Q_ens(v) = α·Q_GNN + (1−α)·Q_RMAV
    Step 4  →  I*(v) (unchanged)
    Step 5  →  validate Q_ens vs I*  (apples-to-apples with Step 3)

Usage
-----
Typical inference workflow (no training data available):

    >>> service = GNNService.from_checkpoint("output/gnn_checkpoints/best_model.pt")
    >>> gnn_result = service.predict(
    ...     graph=nx_graph,
    ...     structural_metrics=structural_dict,
    ...     rmav_scores=rmav_dict,
    ... )
    >>> gnn_result.node_scores          # {node_name: GNNCriticalityScore}
    >>> gnn_result.edge_scores          # {(src, dst): GNNCriticalityScore}
    >>> gnn_result.ensemble_scores      # merged with RMAV via EnsembleGNN

Training workflow:

    >>> service = GNNService()
    >>> service.train(
    ...     graph=nx_graph,
    ...     structural_metrics=structural_dict,
    ...     simulation_results=sim_dict,
    ...     rmav_scores=rmav_dict,
    ... )
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .classifier import BoxPlotClassifier

from .data_preparation import (
    GraphConversionResult,
    NODE_TYPES,
    create_node_splits,
    extract_rmav_scores_dict,
    extract_simulation_dict,
    extract_structural_metrics_dict,
    networkx_to_hetero_data,
)
from .models import (
    EdgeCriticalityGNN,
    EnsembleGNN,
    NodeCriticalityGNN,
    build_edge_gnn,
    build_node_gnn,
)
from .trainer import EvalMetrics, GNNTrainer, evaluate

logger = logging.getLogger(__name__)


# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class GNNCriticalityScore:
    """Criticality prediction for a single component or relationship.

    Mirrors the structure of ``CompositeCriticalityScore`` from the
    existing codebase for seamless downstream compatibility.
    """
    component: str
    composite_score: float
    reliability_score: float
    maintainability_score: float
    availability_score: float
    vulnerability_score: float
    source: str = "GNN"           # "GNN", "RMAV", or "Ensemble"

    criticality_level: str = "MINIMAL"    # Calculated via adaptive thresholds

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "composite_score": round(self.composite_score, 4),
            "reliability_score": round(self.reliability_score, 4),
            "maintainability_score": round(self.maintainability_score, 4),
            "availability_score": round(self.availability_score, 4),
            "vulnerability_score": round(self.vulnerability_score, 4),
            "criticality_level": self.criticality_level,
            "source": self.source,
        }


@dataclass
class GNNEdgeCriticalityScore:
    """Criticality prediction for a pub-sub relationship (directed edge)."""
    source_node: str
    target_node: str
    edge_type: str
    composite_score: float
    reliability_score: float
    maintainability_score: float
    availability_score: float
    vulnerability_score: float

    @property
    def criticality_level(self) -> str:
        if self.composite_score >= 0.75:
            return "CRITICAL"
        elif self.composite_score >= 0.55:
            return "HIGH"
        elif self.composite_score >= 0.35:
            return "MEDIUM"
        return "LOW"

    def to_dict(self) -> dict:
        return {
            "source": self.source_node,
            "target": self.target_node,
            "edge_type": self.edge_type,
            "composite_score": round(self.composite_score, 4),
            "reliability_score": round(self.reliability_score, 4),
            "maintainability_score": round(self.maintainability_score, 4),
            "availability_score": round(self.availability_score, 4),
            "vulnerability_score": round(self.vulnerability_score, 4),
            "criticality_level": self.criticality_level,
        }


@dataclass
class GNNAnalysisResult:
    """Complete GNN analysis result for a single graph.

    Designed to complement (not replace) the existing ``LayerAnalysisResult``
    produced by ``AnalysisService.analyze_layer()``.
    """

    # Node-level predictions
    node_scores: Dict[str, GNNCriticalityScore] = field(default_factory=dict)
    # Edge-level predictions  
    edge_scores: List[GNNEdgeCriticalityScore] = field(default_factory=list)
    # Ensemble predictions (GNN + RMAV blended)
    ensemble_scores: Dict[str, GNNCriticalityScore] = field(default_factory=dict)
    # Validation metrics (when simulation ground truth is available)
    gnn_metrics: Optional[EvalMetrics] = None
    ensemble_metrics: Optional[EvalMetrics] = None
    # Learned ensemble alpha (per RMAV dimension)
    ensemble_alpha: Optional[List[float]] = None
    # Adaptive classification stats
    stats: Dict[str, Any] = field(default_factory=dict)

    def top_critical_nodes(self, n: int = 10, use_ensemble: bool = True) -> List[GNNCriticalityScore]:
        scores = self.ensemble_scores if use_ensemble and self.ensemble_scores else self.node_scores
        sorted_scores = sorted(scores.values(), key=lambda s: s.composite_score, reverse=True)
        return sorted_scores[:n]

    def top_critical_edges(self, n: int = 10) -> List[GNNEdgeCriticalityScore]:
        return sorted(self.edge_scores, key=lambda e: e.composite_score, reverse=True)[:n]

    def to_dict(self) -> dict:
        return {
            "node_scores": {k: v.to_dict() for k, v in self.node_scores.items()},
            "edge_scores": [e.to_dict() for e in self.edge_scores],
            "ensemble_scores": {k: v.to_dict() for k, v in self.ensemble_scores.items()},
            "gnn_metrics": self.gnn_metrics.to_dict() if self.gnn_metrics else None,
            "ensemble_metrics": self.ensemble_metrics.to_dict() if self.ensemble_metrics else None,
            "ensemble_alpha": self.ensemble_alpha,
        }

    def summary(self) -> dict:
        scores = self.ensemble_scores or self.node_scores
        levels = [s.criticality_level for s in scores.values()]
        return {
            "total_components": len(scores),
            "critical": levels.count("CRITICAL"),
            "high": levels.count("HIGH"),
            "medium": levels.count("MEDIUM"),
            "low": levels.count("LOW"),
            "minimal": levels.count("MINIMAL"),
            "critical_edges": sum(
                1 for e in self.edge_scores if e.criticality_level == "CRITICAL"
            ),
        }


# ── Service ────────────────────────────────────────────────────────────────────

class GNNService:
    """Integrates GNN criticality prediction into the existing pipeline.

    Parameters
    ----------
    hidden_channels:
        GNN embedding dimension (default 64).
    num_heads:
        GAT attention heads (default 4).
    num_layers:
        Message-passing depth (default 3).
    dropout:
        Dropout probability (default 0.2).
    predict_edges:
        Whether to also score relationships (edges) via
        :class:`EdgeCriticalityGNN` (default True).
    checkpoint_dir:
        Directory for model checkpoints.
    device:
        Computation device.  Auto-detected if None.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        predict_edges: bool = True,
        checkpoint_dir: str = "output/gnn_checkpoints",
        device: Optional[torch.device] = None,
    ):
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.predict_edges = predict_edges
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Models initialised lazily (require graph metadata)
        self._node_model: Optional[NodeCriticalityGNN] = None
        self._edge_model: Optional[EdgeCriticalityGNN] = None
        self._ensemble: Optional[EnsembleGNN] = None
        self._conversion_result: Optional[GraphConversionResult] = None

        logger.info(
            "GNNService initialised | device=%s | hidden=%d | heads=%d | layers=%d",
            self.device, hidden_channels, num_heads, num_layers,
        )

    # ── Model initialisation ──────────────────────────────────────────────────

    def _init_models(self, metadata: Tuple) -> None:
        """Build models from PyG metadata."""
        logger.info("Initialising GNN models from metadata.")
        self._node_model = build_node_gnn(
            metadata, self.hidden_channels, self.num_heads, self.num_layers, self.dropout
        )
        if self.predict_edges:
            self._edge_model = build_edge_gnn(
                metadata, self.hidden_channels, self.num_heads, self.num_layers, self.dropout
            )
        self._ensemble = EnsembleGNN()
        self._node_model.to(self.device)
        if self._edge_model:
            self._edge_model.to(self.device)
        self._ensemble.to(self.device)

    # ── Main public API ───────────────────────────────────────────────────────

    def train(
        self,
        graph,
        structural_metrics=None,
        simulation_results=None,
        rmav_scores=None,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        num_epochs: int = 300,
        lr: float = 3e-4,
        patience: int = 30,
        inductive_graphs: Optional[List] = None,
        seeds: Optional[List[int]] = None,
    ) -> GNNAnalysisResult:
        """Train GNN models on a labelled graph.

        Converts the NetworkX graph to HeteroData, initialises models,
        trains with early stopping, and returns predictions on the full graph.

        Parameters
        ----------
        graph:
            NetworkX DiGraph (structural graph from Step 1).
        structural_metrics:
            Dict or StructuralAnalysisResult from Step 2.
        simulation_results:
            Dict or list of FailureResult objects from Step 4.
        rmav_scores:
            Dict or QualityAnalysisResult from Step 3.
        train_ratio, val_ratio:
            Node split fractions for transductive training.
        num_epochs:
            Maximum training epochs.
        lr:
            Learning rate.
        patience:
            Early-stopping patience.
        inductive_graphs:
            Optional list of additional HeteroData graphs for
            inductive multi-graph training (e.g., all 8 domain scenarios).
        """
        # Normalise inputs
        if structural_metrics is not None and not isinstance(structural_metrics, dict):
            structural_metrics = extract_structural_metrics_dict(structural_metrics)
        if simulation_results is not None and isinstance(simulation_results, list):
            simulation_results = extract_simulation_dict(simulation_results)
        if rmav_scores is not None and not isinstance(rmav_scores, dict):
            rmav_scores = extract_rmav_scores_dict(rmav_scores)

        # Convert to HeteroData
        conv = networkx_to_hetero_data(
            graph, structural_metrics, simulation_results, rmav_scores
        )
        self._conversion_result = conv
        data = conv.hetero_data
        create_node_splits(data, train_ratio, val_ratio)

        # Prepare DataLoader for training
        if inductive_graphs:
            logger.info("Multi-graph inductive training with %d additional graphs.", len(inductive_graphs))
            all_graphs = [data] + inductive_graphs
            from torch_geometric.loader import DataLoader
            train_loader = DataLoader(all_graphs, batch_size=1, shuffle=True)
            training_input = train_loader
        else:
            training_input = data

        # Handle multi-seed training
        training_seeds = seeds if seeds else [42]
        all_metrics = []
        
        for seed in training_seeds:
            if len(training_seeds) > 1:
                logger.info("── Training seed %d ────────────────────────────────", seed)
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            # Initialise models for this seed
            self._init_models(data.metadata())

            # Train node model
            model_to_train = self._edge_model if self.predict_edges else self._node_model
            trainer = GNNTrainer(
                model=model_to_train,
                checkpoint_dir=str(self.checkpoint_dir),
                lr=lr,
                num_epochs=num_epochs,
                patience=patience,
            )
            trainer.train(training_input)

            # Evaluate on primary graph test set
            test_metrics = evaluate(model_to_train, data, "test_mask", self.device)
            all_metrics.append(test_metrics)

        # Average metrics across seeds if multiple
        if len(all_metrics) > 1:
            avg_rho = sum(m.spearman_rho for m in all_metrics) / len(all_metrics)
            avg_f1 = sum(m.f1_score for m in all_metrics) / len(all_metrics)
            avg_rmse = sum(m.rmse for m in all_metrics) / len(all_metrics)
            avg_mae = sum(m.mae for m in all_metrics) / len(all_metrics)
            avg_ndcg = sum(m.ndcg_10 for m in all_metrics) / len(all_metrics)
            logger.info("Average metrics over %d seeds: rho=%.4f, f1=%.4f, ndcg=%.4f", 
                        len(all_metrics), avg_rho, avg_f1, avg_ndcg)

        # Train ensemble (fine-tune α) - using the best model from the last seed
        if simulation_results and rmav_scores:
            self._train_ensemble(data)

        self._save_service_config()
        return self.predict_from_data(data, simulation_results)

    def predict(
        self,
        graph,
        structural_metrics=None,
        rmav_scores=None,
        simulation_results=None,
    ) -> GNNAnalysisResult:
        """Run inference on a graph without training.

        Requires models to be loaded via :meth:`from_checkpoint`.

        Parameters
        ----------
        graph:
            NetworkX DiGraph.
        structural_metrics:
            Structural analysis results (for node features).
        rmav_scores:
            Existing RMAV predictions (for ensemble blending).
        simulation_results:
            Optional: if provided, validation metrics are computed.
        """
        if self._node_model is None:
            raise RuntimeError(
                "Models not initialised. Call train() or from_checkpoint() first."
            )

        if structural_metrics is not None and not isinstance(structural_metrics, dict):
            structural_metrics = extract_structural_metrics_dict(structural_metrics)
        if simulation_results is not None and isinstance(simulation_results, list):
            simulation_results = extract_simulation_dict(simulation_results)
        if rmav_scores is not None and not isinstance(rmav_scores, dict):
            rmav_scores = extract_rmav_scores_dict(rmav_scores)

        conv = networkx_to_hetero_data(graph, structural_metrics, simulation_results, rmav_scores)
        self._conversion_result = conv
        return self.predict_from_data(conv.hetero_data, simulation_results)

    def predict_from_data(self, data, simulation_results=None) -> GNNAnalysisResult:
        """Run inference directly on a HeteroData object."""
        if self._node_model is None:
            raise RuntimeError("Models not initialised.")

        conv = self._conversion_result
        self._node_model.eval()
        if self._edge_model:
            self._edge_model.eval()

        data_dev = data.to(self.device)
        x_dict = {nt: data_dev[nt].x for nt in data_dev.node_types
                  if hasattr(data_dev[nt], "x")}
        edge_index_dict = {rel: data_dev[rel].edge_index for rel in data_dev.edge_types}
        edge_attr_dict = {rel: data_dev[rel].edge_attr for rel in data_dev.edge_types
                         if hasattr(data_dev[rel], "edge_attr")}

        with torch.no_grad():
            if self._edge_model:
                pred_dict, edge_pred_dict = self._edge_model(x_dict, edge_index_dict, edge_attr_dict)
            else:
                pred_dict = self._node_model(x_dict, edge_index_dict, edge_attr_dict)
                edge_pred_dict = {}

        result = GNNAnalysisResult()

        # ── Node scores ───────────────────────────────────────────────────────
        for nt, preds in pred_dict.items():
            if conv is None or nt not in conv.node_id_map:
                continue
            node_names = conv.node_id_map[nt]
            preds_cpu = preds.cpu().numpy()
            for i, name in enumerate(node_names):
                result.node_scores[name] = GNNCriticalityScore(
                    component=name,
                    composite_score=float(preds_cpu[i, 0]),
                    reliability_score=float(preds_cpu[i, 1]),
                    maintainability_score=float(preds_cpu[i, 2]),
                    availability_score=float(preds_cpu[i, 3]),
                    vulnerability_score=float(preds_cpu[i, 4]),
                    source="GNN",
                )

        # ── Edge scores ───────────────────────────────────────────────────────
        for rel, e_preds in edge_pred_dict.items():
            src_type, edge_type, dst_type = rel
            if conv is None:
                continue
            if rel not in conv.edge_name_map:
                continue
            edge_names = conv.edge_name_map[rel]
            e_preds_cpu = e_preds.cpu().numpy()
            for i, (src_name, dst_name) in enumerate(edge_names):
                result.edge_scores.append(
                    GNNEdgeCriticalityScore(
                        source_node=src_name,
                        target_node=dst_name,
                        edge_type=edge_type,
                        composite_score=float(e_preds_cpu[i, 0]),
                        reliability_score=float(e_preds_cpu[i, 1]),
                        maintainability_score=float(e_preds_cpu[i, 2]),
                        availability_score=float(e_preds_cpu[i, 3]),
                        vulnerability_score=float(e_preds_cpu[i, 4]),
                    )
                )

        # ── Ensemble (GNN + RMAV) ─────────────────────────────────────────────
        if self._ensemble is not None and hasattr(data_dev[list(data_dev.node_types)[0]], "y_rmav"):
            result.ensemble_scores = self._compute_ensemble_scores(data_dev, pred_dict, conv)
            result.ensemble_alpha = self._ensemble.alpha.detach().cpu().tolist()
        else:
            # Fall back to GNN-only
            result.ensemble_scores = {k: v for k, v in result.node_scores.items()}

        # ── Validation metrics ────────────────────────────────────────────────
        # ── Validation metrics ────────────────────────────────────────────────
        if simulation_results:
            create_node_splits(data_dev, seed=42)
            result.gnn_metrics = evaluate(
                self._node_model, data_dev, "test_mask", self.device
            )
            logger.info("GNN test metrics:\n%s", result.gnn_metrics)

        # ── Adaptive Classification ───────────────────────────────────────────
        classifier = BoxPlotClassifier()
        
        # 1. Classify Node Scores (GNN)
        gnn_node_data = [
            {"id": k, "score": v.composite_score} 
            for k, v in result.node_scores.items()
        ]
        gnn_classification = classifier.classify(gnn_node_data, metric_name="GNN Criticality")
        for k, item in zip(result.node_scores.keys(), gnn_classification.items):
             # Ensure we match by ID if sorted order changed (classify returns sorted items)
             # But it's easier to just lookup
             pass
        
        # Sort-safe update
        gnn_lookup = {item.id: item.level.value for item in gnn_classification.items}
        for k, v in result.node_scores.items():
            v.criticality_level = gnn_lookup.get(k, "MINIMAL")
            
        result.stats["gnn_composite"] = gnn_classification.stats.to_dict()

        # 2. Classify Ensemble Scores if they exist
        if result.ensemble_scores:
            ens_node_data = [
                {"id": k, "score": v.composite_score} 
                for k, v in result.ensemble_scores.items()
            ]
            ens_classification = classifier.classify(ens_node_data, metric_name="Ensemble Criticality")
            ens_lookup = {item.id: item.level.value for item in ens_classification.items}
            for k, v in result.ensemble_scores.items():
                v.criticality_level = ens_lookup.get(k, "MINIMAL")
            
            result.stats["ensemble_composite"] = ens_classification.stats.to_dict()

        return result

    # ── Ensemble helpers ──────────────────────────────────────────────────────

    def _train_ensemble(self, data, num_epochs: int = 100, lr: float = 1e-3) -> None:
        """Fine-tune ensemble alpha on labelled training nodes."""
        if self._ensemble is None:
            return
        logger.info("Fine-tuning ensemble weights…")
        opt = torch.optim.Adam(self._ensemble.parameters(), lr=lr)
        data = data.to(self.device)

        for epoch in range(num_epochs):
            self._ensemble.train()
            opt.zero_grad()

            with torch.no_grad():
                x_dict = {nt: data[nt].x for nt in data.node_types
                          if hasattr(data[nt], "x")}
                ei = {rel: data[rel].edge_index for rel in data.edge_types}
                ea = {rel: data[rel].edge_attr for rel in data.edge_types
                      if hasattr(data[rel], "edge_attr")}
                if self._edge_model:
                    pred_dict, _ = self._edge_model(x_dict, ei, ea)
                else:
                    pred_dict = self._node_model(x_dict, ei, ea)

            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            for nt in data.node_types:
                store = data[nt]
                if not (hasattr(store, "y") and hasattr(store, "y_rmav")
                        and hasattr(store, "train_mask") and nt in pred_dict):
                    continue
                mask = store.train_mask
                if mask.sum() == 0:
                    continue
                gnn_scores = pred_dict[nt][mask]
                rmav_scores = store.y_rmav[mask]
                blended = self._ensemble(gnn_scores, rmav_scores)
                target = store.y[mask]
                loss = torch.nn.functional.mse_loss(blended, target)
                total_loss = total_loss + loss

            total_loss.backward()
            opt.step()

        alpha_vals = self._ensemble.alpha.detach().cpu().tolist()
        logger.info(
            "Ensemble alpha (composite/R/M/A/V): %s",
            [f"{a:.3f}" for a in alpha_vals],
        )

    def _compute_ensemble_scores(self, data, pred_dict, conv) -> Dict[str, GNNCriticalityScore]:
        out: Dict[str, GNNCriticalityScore] = {}
        for nt, gnn_preds in pred_dict.items():
            store = data[nt]
            if conv is None or nt not in conv.node_id_map:
                continue
            node_names = conv.node_id_map[nt]

            if hasattr(store, "y_rmav"):
                rmav_t = store.y_rmav.to(self.device)
            else:
                rmav_t = None

            blended = self._ensemble(gnn_preds, rmav_t).detach().cpu().numpy()
            for i, name in enumerate(node_names):
                out[name] = GNNCriticalityScore(
                    component=name,
                    composite_score=float(blended[i, 0]),
                    reliability_score=float(blended[i, 1]),
                    maintainability_score=float(blended[i, 2]),
                    availability_score=float(blended[i, 3]),
                    vulnerability_score=float(blended[i, 4]),
                    source="Ensemble",
                )
        return out

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> Path:
        """Save all models and service configuration."""
        save_dir = Path(path) if path else self.checkpoint_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        if self._node_model:
            torch.save(self._node_model.state_dict(), save_dir / "node_model.pt")
        if self._edge_model:
            torch.save(self._edge_model.state_dict(), save_dir / "edge_model.pt")
        if self._ensemble:
            torch.save(self._ensemble.state_dict(), save_dir / "ensemble.pt")

        self._save_service_config(save_dir)
        logger.info("GNNService saved to '%s'.", save_dir)
        return save_dir

    def _save_service_config(self, save_dir: Optional[Path] = None) -> None:
        d = save_dir or self.checkpoint_dir
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "service_config.json", "w") as f:
            json.dump(
                {
                    "hidden_channels": self.hidden_channels,
                    "num_heads": self.num_heads,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "predict_edges": self.predict_edges,
                },
                f, indent=2,
            )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        metadata: Optional[Tuple] = None,
        graph=None,
        device: Optional[torch.device] = None,
    ) -> "GNNService":
        """Load a previously trained GNNService from disk.

        Parameters
        ----------
        checkpoint_dir:
            Path to the directory created by :meth:`save`.
        metadata:
            PyG metadata tuple ``(node_types, edge_types_as_triples)``.
            Required if ``graph`` is not provided.
        graph:
            NetworkX DiGraph used to reconstruct metadata automatically.
        """
        ckpt_dir = Path(checkpoint_dir)
        cfg_path = ckpt_dir / "service_config.json"

        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            print(f"DEBUG: Loaded GNN config from {cfg_path}: {cfg}")
        else:
            cfg = {}

        service = cls(
            hidden_channels=cfg.get("hidden_channels", 64),
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 3),
            dropout=cfg.get("dropout", 0.2),
            predict_edges=cfg.get("predict_edges", True),
            checkpoint_dir=str(ckpt_dir),
            device=device,
        )

        if metadata is None and graph is not None:
            conv = networkx_to_hetero_data(graph)
            metadata = conv.hetero_data.metadata()

        if metadata is not None:
            service._init_models(metadata)
            nm_path = ckpt_dir / "node_model.pt"
            em_path = ckpt_dir / "edge_model.pt"
            ens_path = ckpt_dir / "ensemble.pt"

            if nm_path.exists() and service._node_model:
                node_sd = torch.load(nm_path, map_location=service.device)
                service._node_model.load_state_dict(node_sd, strict=False)
                logger.info("Loaded node model from '%s' (strict=False).", nm_path)
            if em_path.exists() and service._edge_model:
                edge_sd = torch.load(em_path, map_location=service.device)
                service._edge_model.load_state_dict(edge_sd, strict=False)
                logger.info("Loaded edge model from '%s' (strict=False).", em_path)
            if ens_path.exists() and service._ensemble:
                service._ensemble.load_state_dict(
                    torch.load(ens_path, map_location=service.device)
                )
                logger.info("Loaded ensemble from '%s'.", ens_path)

        return service
