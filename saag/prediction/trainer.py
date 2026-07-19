"""
GNN Trainer
===========
Provides the training loop and evaluation metrics for component and
relationship criticality prediction models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch import Tensor

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from .models import CriticalityLoss

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    """Stores quantitative performance metrics for a GNN model.

    Compatible with Step 5 validation metrics (Spearman rho, F1, etc.).
    """
    spearman_rho: float
    f1_score: float
    rmse: float
    mae: float
    top_5_overlap: float
    top_10_overlap: float
    ndcg_10: float
    precision: float = 0.0
    recall: float = 0.0
    accuracy: float = 0.0
    # Per-node-type Spearman ρ — populated by evaluate() for Figure 4 (Block F)
    per_node_type: Dict[str, float] = None  # type: ignore[assignment]
    # Calibration metadata — MW26 disclosure fields
    calibration: str = "rank_matched"
    n_critical_in_truth: int = 0
    macro_f1: float = 0.0
    bce_loss: float = 0.0
    regression_slope: float = 0.0
    regression_intercept: float = 0.0
    regression_r2: float = 0.0

    def __post_init__(self):
        if self.per_node_type is None:
            object.__setattr__(self, "per_node_type", {})

    def to_dict(self) -> dict:
        d = {
            "spearman_rho": round(self.spearman_rho, 4),
            "f1_score":     self.f1_score if self.f1_score is None or not _isnan_f(self.f1_score) else None,
            "macro_f1":     round(self.macro_f1, 4),
            "bce_loss":     round(self.bce_loss, 4),
            "regression_slope": round(self.regression_slope, 4),
            "regression_intercept": round(self.regression_intercept, 4),
            "regression_r2": round(self.regression_r2, 4),
            "rmse":         round(self.rmse, 4),
            "mae":          round(self.mae, 4),
            "top_5_overlap":  round(self.top_5_overlap, 4),
            "top_10_overlap": round(self.top_10_overlap, 4),
            "ndcg_10":      round(self.ndcg_10, 4),
            "precision":    self.precision if self.precision is None or not _isnan_f(self.precision) else None,
            "recall":       self.recall    if self.recall    is None or not _isnan_f(self.recall)    else None,
            "accuracy":     round(self.accuracy, 4),
            "calibration":           self.calibration,
            "n_critical_in_truth":   self.n_critical_in_truth,
        }
        if self.per_node_type:
            d["per_node_type"] = {nt: round(r, 4) for nt, r in self.per_node_type.items()}
        return d

    @property
    def spearman(self) -> float:
        """Alias for spearman_rho (backward compatibility)."""
        return self.spearman_rho

    def __str__(self) -> str:
        lines = [
            f"  Spearman ρ: {self.spearman_rho:.4f}",
            f"  F1 Score:   {self.f1_score:.4f}",
            f"  Macro F1:   {self.macro_f1:.4f}",
            f"  BCE Loss:   {self.bce_loss:.4f}",
            f"  Slope:      {self.regression_slope:.4f}",
            f"  Intercept:  {self.regression_intercept:.4f}",
            f"  R2 Coeff:   {self.regression_r2:.4f}",
            f"  Precision:  {self.precision:.4f}",
            f"  Recall:     {self.recall:.4f}",
            f"  Accuracy:   {self.accuracy:.4f}",
            f"  RMSE:       {self.rmse:.4f}",
            f"  MAE:        {self.mae:.4f}",
            f"  NDCG@10:    {self.ndcg_10:.4f}",
        ]
        if self.per_node_type:
            lines.append("  Per-type ρ:")
            for nt, r in sorted(self.per_node_type.items()):
                lines.append(f"    {nt:<20} {r:.4f}")
        return "\n".join(lines)


def get_inductive_subgraph(data: 'HeteroData', mask_name: str) -> 'HeteroData':
    """Isolate graph nodes by partition to enforce the Inductive Split Protocol."""
    has_mask = False
    for nt in data.node_types:
        store = data[nt]
        if hasattr(store, mask_name) and store[mask_name].sum() > 0:
            has_mask = True
            break
    if not has_mask:
        return data

    idx_dict = {}
    for nt in data.node_types:
        store = data[nt]
        if hasattr(store, mask_name):
            idx_dict[nt] = torch.where(store[mask_name])[0]
        else:
            device = None
            if hasattr(store, "x") and isinstance(store.x, Tensor):
                device = store.x.device
            else:
                for k, val in store.items():
                    if isinstance(val, Tensor):
                        device = val.device
                        break
            idx_dict[nt] = torch.arange(store.num_nodes, device=device)
    return data.subgraph(idx_dict)


class GNNTrainer:
    """Manages the training process for HGT models with early stopping."""

    def __init__(
        self,
        model: nn.Module,
        checkpoint_dir: str = "output/gnn_checkpoints",
        lr: float = 3e-4,
        num_epochs: int = 300,
        patience: int = 30,
        weight_decay: float = 1e-4,
        warmup_T0: Optional[int] = None,
        multitask_weight: float = 0.5,
        rmav_consistency_weight: float = 0.1,
        ranking_weight: float = 0.3,
        pairwise_ranking_weight: float = 0.1,
    ):
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.lr = lr
        self.num_epochs = num_epochs
        self.patience = patience
        self.weight_decay = weight_decay
        # T_0 for CosineAnnealingWarmRestarts; defaults to num_epochs // 4
        self.warmup_T0 = warmup_T0 or max(50, num_epochs // 4)

        self.loss_fn = CriticalityLoss(
            multitask_weight=multitask_weight,
            rmav_consistency_weight=rmav_consistency_weight,
            ranking_weight=ranking_weight,
            pairwise_ranking_weight=pairwise_ranking_weight,
        )
        self.device = next(model.parameters()).device
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _compute_val_loss(self, data: "HeteroData") -> float:
        """Compute loss on validation-masked *labelled* nodes (no grad)."""
        self.model.eval()
        with torch.no_grad():
            val_data = get_inductive_subgraph(data, "val_mask")
            val_data = val_data.to(self.device)
            x_dict = {nt: val_data[nt].x for nt in val_data.node_types if hasattr(val_data[nt], "x")}
            ei_dict = {rel: val_data[rel].edge_index for rel in val_data.edge_types}
            ea_dict = {rel: val_data[rel].edge_attr for rel in val_data.edge_types if hasattr(val_data[rel], "edge_attr")}
            output = self.model(x_dict, ei_dict, ea_dict)
            node_preds = output[0] if isinstance(output, tuple) else output

            total = 0.0
            count = 0
            for nt, preds in node_preds.items():
                if nt not in {"Application", "Library"}:
                    continue
                store = val_data[nt]
                if not (hasattr(store, "y") and hasattr(store, "val_mask")):
                    continue
                mask = store.val_mask
                if mask.sum() == 0:
                    continue
                # Sub-mask to labelled nodes only (|y_composite| > 0)
                labelled = mask & (store.y[:, 0].abs() > 1e-6)
                if labelled.sum() == 0:
                    continue
                rmav_target = store.y_rmav if hasattr(store, "y_rmav") else None
                loss, _ = self.loss_fn(preds, store.y, labelled, rmav_target)
                total += loss.item()
                count += 1
        return total / max(count, 1)

    def _run_epoch(self, loader, optimizer) -> float:
        """Run one training epoch; return average loss."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            # Enforce Inductive Split Protocol by isolating training subgraph
            train_batch = get_inductive_subgraph(batch, "train_mask")
            
            x_dict = {nt: train_batch[nt].x for nt in train_batch.node_types if hasattr(train_batch[nt], "x")}
            ei_dict = {rel: train_batch[rel].edge_index for rel in train_batch.edge_types}
            ea_dict = {rel: train_batch[rel].edge_attr for rel in train_batch.edge_types if hasattr(train_batch[rel], "edge_attr")}
            output = self.model(x_dict, ei_dict, ea_dict)
            node_preds = output[0] if isinstance(output, tuple) else output
            batch_loss = self._node_loss(node_preds, train_batch)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += batch_loss.item()
            num_batches += 1
        return epoch_loss / max(num_batches, 1)

    def _node_loss(self, node_preds: Dict, batch) -> Tensor:
        """Accumulate loss over all *labelled* train nodes (|y_composite| > 0)."""
        total = torch.tensor(0.0, device=self.device, requires_grad=True)
        for nt, preds in node_preds.items():
            if nt not in {"Application", "Library"}:
                continue
            store = batch[nt]
            if not (hasattr(store, "y") and hasattr(store, "train_mask")):
                continue
            mask = store.train_mask
            if mask.sum() == 0:
                continue
            # Sub-mask: only train on nodes that have non-zero ground-truth labels
            labelled = mask & (store.y[:, 0].abs() > 1e-6)
            if labelled.sum() == 0:
                continue
            rmav_target = store.y_rmav if hasattr(store, "y_rmav") else None
            loss, _ = self.loss_fn(preds, store.y, labelled, rmav_target)
            total = total + loss
        return total

    def _update_best(
        self,
        val_metrics: EvalMetrics,
        val_loss: float,
        best_combined: float,
        best_val_loss: float,
    ) -> Tuple[float, bool]:
        """Compute combined metric and return (combined_score, is_improved)."""
        loss_improvement = max(0.0, 1.0 - val_loss / (best_val_loss + 1e-8))
        combined = 0.6 * val_metrics.spearman_rho + 0.4 * loss_improvement
        return combined, combined > best_combined

    def _log_split_sizes(self, batch) -> None:
        for nt in batch.node_types:
            if hasattr(batch[nt], "train_mask"):
                n_train = batch[nt].train_mask.sum().item()
                n_val = batch[nt].val_mask.sum().item()
                if n_train > 0 or n_val > 0:
                    logger.info("  [%s] Train: %d | Val: %d", nt, n_train, n_val)

    def train(
        self, data: "HeteroData", primary_data: Optional["HeteroData"] = None
    ) -> Tuple[Dict[str, List[float]], Optional[EvalMetrics]]:
        """Run the full training loop with combined-metric early stopping.

        Parameters
        ----------
        data:
            Either a single HeteroData graph, or a DataLoader over multiple
            graphs for inductive multi-scenario training.
        primary_data:
            When ``data`` is a multi-graph DataLoader, the specific graph
            whose val_mask should drive validation/early-stopping/checkpoint
            selection. Without this, validation would run against whatever
            graph happens to land first in the loader's shuffled iteration
            order, which is not guaranteed to be the scenario being trained
            toward. Ignored when ``data`` is a single HeteroData (in that
            case the single graph is always the validation target).
        """
        logger.info(
            "Starting training | epochs=%d | lr=%.2e | device=%s",
            self.num_epochs, self.lr, self.device,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.warmup_T0, T_mult=2, eta_min=self.lr * 0.01,
        )
        loader = DataLoader([data], batch_size=1) if isinstance(data, HeteroData) else data
        if primary_data is not None:
            first_batch = primary_data.to(self.device)
        else:
            first_batch = next(iter(loader))
            first_batch = first_batch.to(self.device)
        self._log_split_sizes(first_batch)

        best_combined = -1.0
        best_val_loss = float("inf")
        best_val_metrics: Optional[EvalMetrics] = None
        epochs_without_improvement = 0
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_rho": []}

        for epoch in range(1, self.num_epochs + 1):
            avg_loss = self._run_epoch(loader, optimizer)
            scheduler.step()

            val_metrics = evaluate(self.model, first_batch, "val_mask", self.device)
            val_loss = self._compute_val_loss(first_batch)
            self.model.train()

            if epoch == 1:
                best_val_loss = val_loss

            combined, improved = self._update_best(
                val_metrics, val_loss, best_combined, best_val_loss
            )
            if improved:
                best_combined = combined
                best_val_loss = min(best_val_loss, val_loss)
                best_val_metrics = val_metrics
                epochs_without_improvement = 0
                self._save_checkpoint("best_model.pt")
            else:
                epochs_without_improvement += 1

            history["train_loss"].append(avg_loss)
            history["val_loss"].append(val_loss)
            history["val_rho"].append(val_metrics.spearman_rho)

            if epoch % 20 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d | Loss: %.4f | Val ρ: %.4f | Combined: %.4f | %s",
                    epoch, avg_loss, val_metrics.spearman_rho, combined,
                    "↑" if improved else " ",
                )

            if epochs_without_improvement >= self.patience:
                logger.info("Early stopping at epoch %d (combined=%.4f).", epoch, best_combined)
                break

        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
            logger.info("Restored best model (combined=%.4f).", best_combined)

        return history, best_val_metrics

    def _save_checkpoint(self, name: str):
        path = self.checkpoint_dir / name
        torch.save(self.model.state_dict(), path)



def evaluate(
    model: nn.Module,
    data: 'HeteroData',
    mask_name: str,
    device: torch.device,
) -> EvalMetrics:
    """Compute validation/test metrics, including per-node-type Spearman ρ under Inductive Split Protocol.

    Parameters
    ----------
    model:
        Trained GNN model.
    data:
        HeteroData graph.
    mask_name:
        Name of the boolean mask in data (e.g., "test_mask").
    device:
        Torch device.
    """
    model.eval()
    
    # Enforce Inductive Split Protocol by isolating evaluation subgraph
    sub_data = get_inductive_subgraph(data, mask_name)
    sub_data = sub_data.to(device)

    with torch.no_grad():
        x_dict = {nt: sub_data[nt].x for nt in sub_data.node_types if hasattr(sub_data[nt], "x")}
        ei_dict = {rel: sub_data[rel].edge_index for rel in sub_data.edge_types}
        ea_dict = {rel: sub_data[rel].edge_attr for rel in sub_data.edge_types if hasattr(sub_data[rel], "edge_attr")}

        if hasattr(model, "predict_edges") and getattr(model, "predict_edges", False):
            node_preds, _ = model(x_dict, ei_dict, ea_dict)
        else:
            node_preds = model(x_dict, ei_dict, ea_dict)

    y_pred, y_true = _collect_samples(node_preds, sub_data, mask_name)
    metrics = evaluate_scores(y_pred, y_true)

    # Populate per-node-type Spearman ρ (Block F prerequisite)
    metrics.per_node_type = _compute_per_type_rho(node_preds, sub_data, mask_name)

    return metrics


def _isnan_f(x: float) -> bool:
    """True iff x is a float NaN.  Safe for None."""
    import math
    try:
        return math.isnan(x)
    except (TypeError, ValueError):
        return False


def evaluate_scores(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    calibration: str = "rank_matched",
) -> EvalMetrics:
    """Compute metrics from pre-collected arrays (N, 5) with robust scaling normalization.

    Parameters
    ----------
    y_pred, y_true : np.ndarray, shape (N, 5)
        Column 0 is the composite score.
    calibration : {"rank_matched", "fixed"}
        - ``rank_matched`` (default): binarize predictions by selecting the
          top-K nodes as critical, where K equals the number of ground-truth
          critical nodes.  Makes F1 comparable across variants whose raw
          outputs span different scales (sigmoid in [0,1], unbounded logits,
          raw betweenness).
        - ``fixed``: legacy behaviour, binarize both at the adaptive
          ``gt_threshold``.
    """
    if y_pred.shape[0] == 0:
        return EvalMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           calibration=calibration, n_critical_in_truth=0)

    # ── Label Normalization Hardening: Robust scaling transform on ground-truth ──
    if y_true.shape[0] > 0:
        for col in range(y_true.shape[1]):
            t_col = y_true[:, col]
            t_median = np.median(t_col)
            t_q75, t_q25 = np.percentile(t_col, [75, 25])
            t_iqr = t_q75 - t_q25
            t_scaled = (t_col - t_median) / (t_iqr + 1e-9)
            y_true[:, col] = 1.0 / (1.0 + np.exp(-t_scaled))

    # Composite scores (column 0)
    p_comp = y_pred[:, 0]
    t_comp = y_true[:, 0]
    n = len(p_comp)

    # ── Spearman ρ ────────────────────────────────────────────────────────────
    if n > 1 and not (np.all(p_comp == p_comp[0]) or np.all(t_comp == t_comp[0])):
        rho, _ = spearmanr(p_comp, t_comp)
        rho = float(rho) if not np.isnan(rho) else 0.0
    else:
        rho = 0.0

    # ── Adaptive ground-truth threshold ────────────────────────────────────────
    # When labels are derived from RMAV (all in [0, 1] but max < 0.5),
    # use the 90th-percentile as the critical threshold.
    gt_threshold = 0.5
    if np.max(t_comp) < 0.5 and np.max(t_comp) > 1e-6:
        gt_threshold = float(np.percentile(t_comp, 90))

    y_true_bin = (t_comp >= gt_threshold).astype(int)
    n_critical = int(y_true_bin.sum())

    # ── Classification metrics ─────────────────────────────────────────────────
    macro_f1_val = 0.0
    if n_critical == 0 or n_critical == n:
        # Degenerate label distribution: F1 is undefined.
        f1 = prec = rec = float("nan")
        acc = float(accuracy_score(y_true_bin, np.zeros(n, dtype=int)))
        calib_label = f"{calibration}_degenerate"
    else:
        # Both branches use exact top-K integer indexing so that |P| = K = |G|
        # exactly, enforcing Precision = Recall = F1 for every seed cell.
        # The "fixed"/legacy threshold approach (p_comp >= sorted_threshold) can
        # select |P| > K when predictions tie at the K-th boundary; we replace
        # it with argsort indexing for identical cardinality guarantees.
        near_constant = (np.std(p_comp) < 1e-9)
        if near_constant:
            # Constant or near-constant predictions — worst-case random assignment.
            y_pred_bin = np.zeros(n, dtype=int)
        else:
            # Stable argsort: ties broken by original node index (deterministic).
            order = np.argsort(-p_comp, kind="stable")
            y_pred_bin = np.zeros(n, dtype=int)
            y_pred_bin[order[:n_critical]] = 1

        # Cardinality guard: |P| must equal K = |G| exactly.
        n_pred_positive = int(y_pred_bin.sum())
        if n_pred_positive != n_critical:
            logger.warning(
                "evaluate_scores: cardinality mismatch |P|=%d != K=%d; "
                "forcing top-K re-selection.",
                n_pred_positive, n_critical,
            )
            order = np.argsort(-p_comp, kind="stable")
            y_pred_bin = np.zeros(n, dtype=int)
            y_pred_bin[order[:n_critical]] = 1

        f1   = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
        prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
        rec  = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
        acc  = float(accuracy_score(y_true_bin, y_pred_bin))
        calib_label = "rank_matched"  # both branches now enforce rank-matched cardinality
        macro_f1_val = float(f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0))

    # ── Continuous BCE (Soft Labels) ───────────────────────────────────────────
    p_clipped = np.clip(p_comp, 1e-7, 1 - 1e-7)
    bce = -np.mean(t_comp * np.log(p_clipped) + (1.0 - t_comp) * np.log(1.0 - p_clipped))
    bce_loss = float(bce) if not np.isnan(bce) else 0.0

    # ── Regression metrics ─────────────────────────────────────────────────────
    rmse = float(np.sqrt(np.mean((p_comp - t_comp) ** 2)))
    mae  = float(np.mean(np.abs(p_comp - t_comp)))

    # ── Regression Curve ───────────────────────────────────────────────────────
    if n > 1:
        mean_p = np.mean(p_comp)
        mean_t = np.mean(t_comp)
        num = np.sum((p_comp - mean_p) * (t_comp - mean_t))
        den = np.sum((p_comp - mean_p) ** 2)
        if den > 1e-12:
            slope = num / den
            intercept = mean_t - slope * mean_p
            ss_res = np.sum((t_comp - (slope * p_comp + intercept)) ** 2)
            ss_tot = np.sum((t_comp - mean_t) ** 2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            slope = 0.0
            intercept = mean_t
            r2 = 0.0
    else:
        slope = 0.0
        intercept = 0.0
        r2 = 0.0

    # ── Top-K overlaps and NDCG ────────────────────────────────────────────────
    top_5_pred = np.argsort(p_comp)[-5:]
    top_5_true = np.argsort(t_comp)[-5:]
    top_5_overlap = len(set(top_5_pred) & set(top_5_true)) / 5.0

    top_10_pred = np.argsort(p_comp)[-10:]
    top_10_true = np.argsort(t_comp)[-10:]
    top_10_overlap = len(set(top_10_pred) & set(top_10_true)) / 10.0 if n >= 10 else 0.0

    ndcg = _compute_ndcg(p_comp, t_comp, k=10)

    return EvalMetrics(
        spearman_rho=rho,
        f1_score=f1,
        rmse=rmse,
        mae=mae,
        top_5_overlap=float(top_5_overlap),
        top_10_overlap=float(top_10_overlap),
        ndcg_10=float(ndcg),
        precision=prec,
        recall=rec,
        accuracy=acc,
        calibration=calib_label,
        n_critical_in_truth=n_critical,
        macro_f1=macro_f1_val,
        bce_loss=bce_loss,
        regression_slope=float(slope),
        regression_intercept=float(intercept),
        regression_r2=float(r2),
    )


def _collect_samples(
    node_preds: Dict[str, Tensor],
    data: 'HeteroData',
    mask_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to gather masked predictions and targets.

    Only includes nodes where the true composite label (column 0) is non-zero,
    i.e. nodes that were actually in the failure simulation. This prevents
    unlabelled nodes (y=0) from diluting the Spearman ρ.
    """
    all_preds = []
    all_targets = []

    for nt, preds in node_preds.items():
        if nt not in {"Application", "Library"}:
            continue
        store = data[nt]
        if not (hasattr(store, "y") and hasattr(store, mask_name)):
            continue
        mask = getattr(store, mask_name)
        if mask.sum() == 0:
            continue

        y_masked = store.y[mask].detach().cpu().numpy()
        p_masked = preds[mask].detach().cpu().numpy()

        # Filter to labelled nodes (composite score > 0)
        labelled = np.abs(y_masked[:, 0]) > 1e-6
        if labelled.sum() >= 2:
            all_preds.append(p_masked[labelled])
            all_targets.append(y_masked[labelled])
        elif labelled.sum() > 0:
            # Include partial if we have at least 1 (will be concatenated with others)
            all_preds.append(p_masked[labelled])
            all_targets.append(y_masked[labelled])

    if not all_preds:
        return np.empty((0, 5)), np.empty((0, 5))

    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)



def _compute_per_type_rho(
    node_preds: Dict[str, Tensor],
    data: 'HeteroData',
    mask_name: str,
) -> Dict[str, float]:
    """Compute per-node-type Spearman ρ on composite score (column 0).

    Returns a dict mapping node type name to rho.  Types with fewer than 3
    samples are omitted (insufficient for Spearman correlation).
    """
    result: Dict[str, float] = {}
    for nt, preds in node_preds.items():
        store = data[nt]
        if not (hasattr(store, "y") and hasattr(store, mask_name)):
            continue
        mask = getattr(store, mask_name)
        if mask.sum() < 3:
            continue
        p_comp = preds[mask].detach().cpu().numpy()[:, 0]
        t_comp = store.y[mask].detach().cpu().numpy()[:, 0]
        if np.all(p_comp == p_comp[0]) or np.all(t_comp == t_comp[0]):
            result[nt] = 0.0
            continue
        rho, _ = spearmanr(p_comp, t_comp)
        result[nt] = float(rho) if not np.isnan(rho) else 0.0
    return result


def _compute_ndcg(y_score, y_true, k=10):
    k = min(k, len(y_true))
    if k == 0:
        return 0.0

    # DCG
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order[:k]]
    dcg = np.sum(y_true_sorted / np.log2(np.arange(2, k + 2)))

    # IDCG
    y_true_ideal = np.sort(y_true)[::-1][:k]
    idcg = np.sum(y_true_ideal / np.log2(np.arange(2, k + 2)))

    return dcg / idcg if idcg > 0 else 0.0
