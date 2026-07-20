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
    Step 4  →  I*(v) (unchanged)
    Step 5  →  validate Q_gnn vs I*  (apples-to-apples with Step 3)

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from saag.core.metrics import ComponentQuality, EdgeQuality, QualityScores, QualityLevels, StructuralMetrics
from saag.core.criticality import CriticalityLevel

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
    security_score: float
    source: str = "GNN"           # "GNN", "RMAV", or "Ensemble"

    criticality_level: str = "MINIMAL"    # Calculated via adaptive thresholds

    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "composite_score": round(self.composite_score, 4),
            "reliability_score": round(self.reliability_score, 4),
            "maintainability_score": round(self.maintainability_score, 4),
            "availability_score": round(self.availability_score, 4),
            "security_score": round(self.security_score, 4),
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
    security_score: float

    criticality_level: str = "MINIMAL"    # Calculated via adaptive thresholds (per-scenario box-plot)

    def to_dict(self) -> dict:
        return {
            "source": self.source_node,
            "target": self.target_node,
            "edge_type": self.edge_type,
            "composite_score": round(self.composite_score, 4),
            "reliability_score": round(self.reliability_score, 4),
            "maintainability_score": round(self.maintainability_score, 4),
            "availability_score": round(self.availability_score, 4),
            "security_score": round(self.security_score, 4),
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
    # Effective prediction mode (Literal["ensemble", "gnn_only", "rmav_only"])
    prediction_mode: str = "gnn_only"
    # Internal: structural metadata for shimming
    _structural_cache: Dict[str, Any] = field(default_factory=dict, repr=False)
    layer: str = "system"

    @property
    def components(self) -> List[ComponentQuality]:
        """Backward-compatibility shim for anti-pattern detection."""
        from saag.core.criticality import CriticalityLevel
        comps = []
        scores_map = self.ensemble_scores or self.node_scores
        for node_id, score in scores_map.items():
            qs = QualityScores(
                reliability=score.reliability_score,
                maintainability=score.maintainability_score,
                availability=score.availability_score,
                security=score.security_score,
                overall=score.composite_score
            )
            def _to_level(val: float) -> CriticalityLevel:
                if val >= 0.75: return CriticalityLevel.CRITICAL
                if val >= 0.55: return CriticalityLevel.HIGH
                if val >= 0.35: return CriticalityLevel.MEDIUM
                return CriticalityLevel.LOW if val >= 0.15 else CriticalityLevel.MINIMAL

            ql = QualityLevels(
                reliability=_to_level(qs.reliability),
                maintainability=_to_level(qs.maintainability),
                availability=_to_level(qs.availability),
                security=_to_level(qs.security),
                # Reuse the adaptive, per-scenario box-plot level already assigned
                # to `score.criticality_level` rather than recomputing with fixed
                # absolute cutoffs, so this shim can't disagree with node_scores.
                overall=CriticalityLevel[score.criticality_level],
            )
            s_dict = self._structural_cache.get(node_id, {})
            sm = StructuralMetrics(id=node_id, name=s_dict.get("name", node_id), type=s_dict.get("type", "Application"))
            for k, v in s_dict.items():
                if hasattr(sm, k): setattr(sm, k, v)
            comps.append(ComponentQuality(id=node_id, type=sm.type, scores=qs, levels=ql, structural=sm))
        return comps

    @property
    def edges(self) -> List[EdgeQuality]:
        """Backward-compatibility shim for anti-pattern detection."""
        from saag.core.metrics import EdgeMetrics
        from saag.core.criticality import CriticalityLevel
        eqs = []
        for es in self.edge_scores:
            qs = QualityScores(reliability=es.reliability_score, maintainability=es.maintainability_score, 
                               availability=es.availability_score, security=es.security_score, 
                               overall=es.composite_score)
            # Reuse the adaptive, per-scenario box-plot level already assigned to
            # `es.criticality_level` rather than recomputing with fixed cutoffs.
            eqs.append(EdgeQuality(source=es.source_node, target=es.target_node, source_type="GNN_Node",
                                   target_type="GNN_Node", dependency_type=es.edge_type, scores=qs, level=CriticalityLevel[es.criticality_level]))
        return eqs

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
            "prediction_mode": self.prediction_mode,
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
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._node_model = None
        self._edge_model = None
        self._ensemble = None
        self._best_seed = 42
        self.layer = "unknown"
        self._conversion_result: Optional[GraphConversionResult] = None

        logger.info(
            "GNNService initialised | device=%s | hidden=%d | heads=%d | layers=%d",
            self.device, hidden_channels, num_heads, num_layers,
        )

    # ── Model initialisation ──────────────────────────────────────────────────

    def _init_models(self, metadata: Tuple) -> None:
        """Build models from PyG metadata."""
        logger.info("Initialising GNN models from metadata.")
        self.metadata = metadata
        self._node_model = build_node_gnn(
            metadata, self.hidden_channels, self.num_heads, self.num_layers, self.dropout
        )
        if self.predict_edges:
            self._edge_model = build_edge_gnn(
                metadata, self.hidden_channels, self.num_heads, self.num_layers, self.dropout
            )
        self._ensemble = None
        self._node_model.to(self.device)
        if self._edge_model:
            self._edge_model.to(self.device)

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
        inductive_graphs: Optional[List['HeteroData']] = None,
        seeds: Optional[List[int]] = None,
        mode: str = "gnn",
        layer: str = "app",
        qos_enabled: bool = True,
        weight_decay: float = 1e-4,
        warmup_T0: Optional[int] = None,
        multitask_weight: float = 0.5,
        rmav_consistency_weight: float = 0.1,
        ranking_weight: float = 0.3,
        pairwise_ranking_weight: float = 0.1,
    ) -> GNNAnalysisResult:
        """Process graphs and train the GNN model using a multi-seed approach.

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
            graph, structural_metrics, simulation_results, rmav_scores, qos_enabled=qos_enabled
        )
        self._conversion_result = conv
        data = conv.hetero_data
        
        # Transductive bias acknowledgment (Issue G5)
        logger.info("Training mode: Transductive (neighbourhood context includes test nodes).")
        self.layer = layer
        logger.info("Training GNNService for layer '%s'.", layer)
        if inductive_graphs:
            logger.info("Generality Validation: Inductive multi-graph training enabled (%d scenarios).", len(inductive_graphs))

        # Prepare DataLoader for training
        if inductive_graphs:
            logger.info("Multi-graph inductive training with %d additional graphs.", len(inductive_graphs))
            all_graphs = [data] + inductive_graphs
            from torch_geometric.loader import DataLoader
            train_loader = DataLoader(all_graphs, batch_size=1, shuffle=True)
            training_input = train_loader
        else:
            training_input = data

        # Handle multi-seed training (Issue G6)
        training_seeds = seeds if seeds else [42]
        all_metrics = []
        best_val_rho = -1.0
        best_state: Optional[Dict[str, Tensor]] = None
        best_seed = training_seeds[0]
        
        for seed in training_seeds:
            if len(training_seeds) > 1:
                logger.info("── Training seed %d ────────────────────────────────", seed)
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            # Fresh split per seed (Issue G6/G8)
            create_node_splits(data, train_ratio, val_ratio, seed=seed)
            if inductive_graphs:
                for ig in inductive_graphs:
                    create_node_splits(ig, train_ratio, val_ratio, seed=seed)

            # IQR-normalize labels to reduce outlier impact on loss scale.
            # Applied per-graph so every scenario's labels land on the same
            # (0,1) scale — otherwise the primary graph's normalized labels
            # and the inductive graphs' raw-scale labels would produce
            # inconsistent gradients for the same scale-sensitive loss terms.
            from .data_preparation import normalize_labels_robust
            normalize_labels_robust(data)
            if inductive_graphs:
                for ig in inductive_graphs:
                    normalize_labels_robust(ig)

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
                weight_decay=weight_decay,
                warmup_T0=warmup_T0,
                multitask_weight=multitask_weight,
                rmav_consistency_weight=rmav_consistency_weight,
                ranking_weight=ranking_weight,
                pairwise_ranking_weight=pairwise_ranking_weight,
            )
            _, best_val_metrics = trainer.train(
                training_input, primary_data=data if inductive_graphs else None
            )

            # Track global best across seeds
            if best_val_metrics and best_val_metrics.spearman_rho > best_val_rho:
                best_val_rho = best_val_metrics.spearman_rho
                best_state = {k: v.cpu().clone() for k, v in model_to_train.state_dict().items()}
                best_seed = seed
                self._best_seed = seed
                logger.info("  [Best Seed Updated] Seed %d achieved Val Rho: %.4f", seed, best_val_rho)

            # Evaluate on primary graph test set for this seed
            test_metrics = evaluate(model_to_train, data, "test_mask", self.device)
            all_metrics.append(test_metrics)

        # Restore best model and masks
        if best_state is not None:
            logger.info("Restoring best model from seed %d (Val Rho: %.4f)", best_seed, best_val_rho)
            model_to_restore = self._edge_model if self.predict_edges else self._node_model
            model_to_restore.load_state_dict(best_state)
            # Re-apply best seed splits to ensure splits are correct
            create_node_splits(data, train_ratio, val_ratio, seed=best_seed)

        # ── Final Inference ──────────────────────────────────────────────────
        # Best state is already loaded in self._node_model via GNNTrainer.train
        self._save_service_config()
        return self.predict_from_data(data, simulation_results, mode=mode)

    def predict(
        self,
        graph,
        structural_metrics=None,
        rmav_scores=None,
        eval_labels=None,
        mode: str = "gnn",
        qos_enabled: bool = True,
        **kwargs,
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
        eval_labels:
            Optional: if provided, validation metrics are computed.
        mode:
            Ablation mode: 'rmav', 'gnn', or 'ensemble'.
        """
        # Backward compatibility with simulation_results parameter
        simulation_results = kwargs.pop("simulation_results", None)
        if eval_labels is None:
            eval_labels = simulation_results

        if self._node_model is None:
            raise RuntimeError(
                "Models not initialised. Call train() or from_checkpoint() first."
            )

        if structural_metrics is not None and not isinstance(structural_metrics, dict):
            structural_metrics = extract_structural_metrics_dict(structural_metrics)
        if eval_labels is not None and isinstance(eval_labels, list):
            eval_labels = extract_simulation_dict(eval_labels)
        if rmav_scores is not None and not isinstance(rmav_scores, dict):
            rmav_scores = extract_rmav_scores_dict(rmav_scores)

        conv = networkx_to_hetero_data(graph, structural_metrics, eval_labels, rmav_scores, qos_enabled=qos_enabled)
        self._conversion_result = conv
        # ── Run prediction ────────────────────────────────────────────────────
        return self.predict_from_data(
            conv.hetero_data, 
            eval_labels, 
            mode=mode, 
            structural_metrics=structural_metrics,
            layer=getattr(graph, "layer", "system") if hasattr(graph, "layer") else "system"
        )

    def predict_from_data(
        self, 
        data, 
        eval_labels=None, 
        mode: str = "gnn",
        structural_metrics: Optional[Dict[str, Any]] = None,
        layer: str = "system",
        **kwargs,
    ) -> GNNAnalysisResult:
        """Run inference directly on a HeteroData object.
        
        Parameters
        ----------
        data: HeteroData
        eval_labels: Optional dict
        mode: 'rmav', 'gnn', or 'ensemble'
        """
        # Support backward compatibility with simulation_results parameter
        simulation_results = kwargs.pop("simulation_results", None)
        if eval_labels is None:
            eval_labels = simulation_results

        if self._node_model is None:
            raise RuntimeError("Models not initialised.")

        conv = self._conversion_result
        self._node_model.eval()
        if self._edge_model:
            self._edge_model.eval()

        data_dev = data.to(self.device)
        
        # Filter to only the node types and edge types supported by the model to prevent size mismatch
        model_node_types = set(self._node_model.node_types)
        model_edge_types = set(self._node_model.edge_types)

        x_dict = {nt: data_dev[nt].x for nt in data_dev.node_types
                  if nt in model_node_types and hasattr(data_dev[nt], "x")}
        edge_index_dict = {rel: data_dev[rel].edge_index for rel in data_dev.edge_types
                           if rel in model_edge_types}
        edge_attr_dict = {rel: data_dev[rel].edge_attr for rel in data_dev.edge_types
                          if rel in model_edge_types and hasattr(data_dev[rel], "edge_attr")}
        
        # ── STRICT INDEPENDENCE INVARIANT ASSERTIONS ──────────────────────────
        # Ensure that during inference (forward pass), no evaluation/simulation labels (y)
        # are ever passed to or consumed by the GNN models.
        for nt, x in x_dict.items():
            assert not hasattr(x, "y"), f"Violation of Independence Guarantee: Feature tensor for '{nt}' contains target label attribute 'y'."
            assert x.shape[1] != 5 or nt == "Broker" or nt == "Node", (
                f"Violation of Independence Guarantee: Input features for '{nt}' has 5 dimensions "
                "which matches the label dimension, indicating potential leak of target labels."
            )
        for k in x_dict:
            assert k != "y" and k != "y_edge", "Violation of Independence Guarantee: Target label key present in input dict."

        # ── Raw GNN Inference ────────────────────────────────────────────────
        with torch.no_grad():
            if self._edge_model:
                pred_dict, edge_pred_dict = self._edge_model(x_dict, edge_index_dict, edge_attr_dict)
            else:
                pred_dict = self._node_model(x_dict, edge_index_dict, edge_attr_dict)
                edge_pred_dict = {}

        result = GNNAnalysisResult()
        result.mode = mode

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
                     security_score=float(preds_cpu[i, 4]),
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
                        security_score=float(e_preds_cpu[i, 4]),
                    )
                )

        # ── Assemble Results based on Mode (Issue G11, G12) ───────────────────
        has_rmav = any(hasattr(data_dev[nt], "y_rmav") for nt in data_dev.node_types)
        
        if mode == "rmav":
            if not has_rmav:
                logger.warning("Mode 'rmav' requested but no RMAV scores available. Falling back to GNN.")
                result.prediction_mode = "gnn_only"
                self._populate_node_scores(result, pred_dict, conv)
            else:
                result.prediction_mode = "rmav_only"
                self._populate_scores_from_rmav(result, data_dev, conv)
        else: # gnn, or deprecated ensemble
            result.prediction_mode = "gnn_only"
            self._populate_node_scores(result, pred_dict, conv)
            if mode == "ensemble":
                logger.warning("Ensemble mode is deprecated and removed. Falling back to GNN-only.")

        # ── Edge scores (always GNN) ──────────────────────────────────────────
        self._populate_edge_scores(result, edge_pred_dict, conv)

        # ── Validation metrics (Issue G9, G10) ────────────────────────────────
        if eval_labels:
            create_node_splits(data_dev, seed=self._best_seed)
            # 1. GNN Validation
            result.gnn_metrics = evaluate(self._node_model, data_dev, "test_mask", self.device)

        # ── Adaptive Classification ───────────────────────────────────────────
        classifier = BoxPlotClassifier()
        
        # Classify Final Result Node Scores
        node_data = [
            {"id": k, "score": v.composite_score} 
            for k, v in result.node_scores.items()
        ]
        classification = classifier.classify(node_data, metric_name="Criticality")
        result.stats["classification"] = classification

        # Assign classification levels back to each score object
        lookup = {item.id: item.level.value for item in classification.items}
        for k, v in result.node_scores.items():
            v.criticality_level = lookup.get(k, "MINIMAL").upper()
        result.stats["composite"] = classification.stats.to_dict()

        # Classify edge scores against this scenario's own edge-score distribution
        # (mirrors the node classification above; previously edges used fixed
        # absolute cutoffs, inconsistent across scenarios of different scale).
        if result.edge_scores:
            edge_data = [
                {"id": i, "score": e.composite_score}
                for i, e in enumerate(result.edge_scores)
            ]
            edge_classification = classifier.classify(edge_data, metric_name="EdgeCriticality")
            edge_lookup = {item.id: item.level.value for item in edge_classification.items}
            for i, e in enumerate(result.edge_scores):
                e.criticality_level = edge_lookup.get(i, "MINIMAL").upper()
            result.stats["edge_composite"] = edge_classification.stats.to_dict()

        # Attach context for shimming
        result.layer = layer
        if structural_metrics:
            result._structural_cache = structural_metrics

        return result

    def detect_problems(self, result: GNNAnalysisResult) -> List[Any]:
        """Unified SDK entry point for anti-pattern detection on GNN results."""
        from .problem_detector import ProblemDetector
        detector = ProblemDetector()
        return detector.detect(result)

    # ── Score Population Helpers ──────────────────────────────────────────────

    def _populate_node_scores(self, result: GNNAnalysisResult, pred_dict: Dict[str, Tensor], conv: GraphConversionResult) -> None:
        """Fill result.node_scores from GNN predictions."""
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
                    security_score=float(preds_cpu[i, 4]),
                    source="GNN",
                )

    def _populate_scores_from_rmav(self, result: GNNAnalysisResult, data: 'HeteroData', conv: GraphConversionResult) -> None:
        """Fill result.node_scores from RMAV ground truth inside data_dev."""
        for nt in data.node_types:
            store = data[nt]
            if not hasattr(store, "y_rmav") or conv is None or nt not in conv.node_id_map:
                continue
            node_names = conv.node_id_map[nt]
            rmav_cpu = store.y_rmav.cpu().numpy()
            for i, name in enumerate(node_names):
                result.node_scores[name] = GNNCriticalityScore(
                    component=name,
                    composite_score=float(rmav_cpu[i, 0]),
                    reliability_score=float(rmav_cpu[i, 1]),
                    maintainability_score=float(rmav_cpu[i, 2]),
                    availability_score=float(rmav_cpu[i, 3]),
                    security_score=float(rmav_cpu[i, 4]),
                    source="RMAV",
                )

    def _populate_edge_scores(self, result: GNNAnalysisResult, edge_pred_dict: Dict[Tuple, Tensor], conv: GraphConversionResult) -> None:
        """Fill result.edge_scores from GNN predictions."""
        for rel, e_preds in edge_pred_dict.items():
            src_type, edge_type, dst_type = rel
            if conv is None or rel not in conv.edge_name_map:
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
                        security_score=float(e_preds_cpu[i, 4]),
                    )
                )

    def _format_scores_as_dict(self, scores: Dict[str, GNNCriticalityScore], conv: GraphConversionResult) -> Dict[str, Tensor]:
        """Convert result.node_scores back to a pred_dict-like structure for evaluation."""
        out = {}
        # We need to map back which score belongs to which node type
        # We can use conv.node_id_map which is {type: [names]}
        for nt, names in conv.node_id_map.items():
            type_scores = []
            for name in names:
                if name in scores:
                    s = scores[name]
                    type_scores.append([
                        s.composite_score, s.reliability_score, s.maintainability_score, 
                        s.availability_score, s.security_score
                    ])
                else:
                    type_scores.append([0.0] * 5)
            out[nt] = torch.tensor(type_scores, device=self.device)
        return out

    # ── Ensemble helpers ──────────────────────────────────────────────────────

    def _train_ensemble(self, data, num_epochs: int = 100, lr: float = 1e-3) -> None:
        """Deprecated: ensemble blending is removed."""
        pass
        
    def _compute_ensemble_scores(self, data, pred_dict, conv) -> Dict[str, GNNCriticalityScore]:
        """Deprecated: ensemble blending is removed."""
        return {}

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
        metadata_json = None
        if hasattr(self, "metadata") and self.metadata is not None:
            node_types, edge_types = self.metadata
            metadata_json = {
                "node_types": list(node_types),
                "edge_types": [list(et) for et in edge_types]
            }
        with open(d / "service_config.json", "w") as f:
            from .data_preparation import NODE_TYPE_TO_DIM
            json.dump(
                {
                    "hidden_channels": self.hidden_channels,
                    "num_heads": self.num_heads,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "predict_edges": self.predict_edges,
                    "node_feature_dims": NODE_TYPE_TO_DIM,
                    "best_seed": self._best_seed,
                    "layer": self.layer,
                    "feature_version": 3,
                    "default_mode": "gnn",
                    "metadata": metadata_json,
                },
                f, indent=2,
            )

    @staticmethod
    def _validate_feature_dims(cfg: dict) -> None:
        """Validate checkpoint feature dims against current code; raise on mismatch for v2+."""
        from .data_preparation import NODE_TYPE_TO_DIM
        feature_version = cfg.get("feature_version", 1)
        if feature_version < 3:
            logger.warning(
                "Checkpoint uses feature_version=%d (current is 3). "
                "Node dims changed: Broker 18→19, Topic 18→22 "
                "(+log1p_frequency_norm, +topic_qos_criticality_ord), Node 18→20. "
                "Re-training required. Loading in strict=False mode.",
                feature_version,
            )
        if "node_feature_dims" not in cfg:
            if "hidden_channels" in cfg:
                logger.warning("Checkpoint lacks 'node_feature_dims'. Proceeding with risk.")
            return
        saved = cfg["node_feature_dims"]
        mismatches = [
            (nt, saved[nt], dim)
            for nt, dim in NODE_TYPE_TO_DIM.items()
            if nt in saved and saved[nt] != dim
        ]
        if not mismatches:
            return
        if feature_version >= 2:
            details = ", ".join(f"{nt}: ckpt={s} code={c}" for nt, s, c in mismatches)
            raise ValueError(f"GNN Feature Dimension Mismatch ({details}). Re-training required.")
        logger.warning(
            "Feature dim mismatches (loading with strict=False): %s",
            ", ".join(f"{nt}:{s}→{c}" for nt, s, c in mismatches),
        )

    def _load_model_weights(self, ckpt_dir: Path) -> None:
        """Load node_model, edge_model, and ensemble state dicts from ckpt_dir."""
        paths = {
            "node": ckpt_dir / "node_model.pt",
            "edge": ckpt_dir / "edge_model.pt",
            "ensemble": ckpt_dir / "ensemble.pt",
        }
        models = {
            "node": self._node_model,
            "edge": self._edge_model,
            "ensemble": self._ensemble,
        }
        for key, path in paths.items():
            model = models[key]
            if path.exists() and model is not None:
                sd = torch.load(path, map_location=self.device)
                strict = key == "ensemble"
                model.load_state_dict(sd, strict=strict)
                logger.info("Loaded %s model from '%s'.", key, path)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        metadata: Optional[Tuple] = None,
        graph=None,
        device: Optional[torch.device] = None,
        layer: Optional[str] = None,
    ) -> "GNNService":
        """Load a previously trained GNNService from disk."""
        ckpt_dir = Path(checkpoint_dir)
        cfg_path = ckpt_dir / "service_config.json"
        cfg = json.load(open(cfg_path)) if cfg_path.exists() else {}

        ckpt_layer = cfg.get("layer", "unknown")
        if layer and ckpt_layer != "unknown" and layer != ckpt_layer:
            raise ValueError(
                f"GNN Layer Mismatch: Checkpoint trained for '{ckpt_layer}', "
                f"inference requested for '{layer}'."
            )

        cls._validate_feature_dims(cfg)

        service = cls(
            hidden_channels=cfg.get("hidden_channels", 64),
            num_heads=cfg.get("num_heads", 4),
            num_layers=cfg.get("num_layers", 3),
            dropout=cfg.get("dropout", 0.2),
            predict_edges=cfg.get("predict_edges", True),
            checkpoint_dir=str(ckpt_dir),
            device=device,
        )
        service._best_seed = cfg.get("best_seed", 42)
        service.layer = ckpt_layer

        if metadata is None:
            if "metadata" in cfg and cfg["metadata"] is not None:
                md = cfg["metadata"]
                node_types = md["node_types"]
                edge_types = [tuple(et) for et in md["edge_types"]]
                metadata = (node_types, edge_types)
            elif graph is not None:
                conv = networkx_to_hetero_data(graph)
                metadata = conv.hetero_data.metadata()

        if metadata is not None:
            service._init_models(metadata)
            service._load_model_weights(ckpt_dir)

        return service
