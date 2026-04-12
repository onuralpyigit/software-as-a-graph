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
from sklearn.metrics import f1_score
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

    def to_dict(self) -> dict:
        return {
            "spearman_rho": round(self.spearman_rho, 4),
            "f1_score": round(self.f1_score, 4),
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            "top_5_overlap": round(self.top_5_overlap, 4),
            "top_10_overlap": round(self.top_10_overlap, 4),
            "ndcg_10": round(self.ndcg_10, 4),
        }

    @property
    def spearman(self) -> float:
        """Alias for spearman_rho (backward compatibility)."""
        return self.spearman_rho

    def __str__(self) -> str:
        return (
            f"  Spearman ρ: {self.spearman_rho:.4f}\n"
            f"  F1 Score:   {self.f1_score:.4f}\n"
            f"  RMSE:       {self.rmse:.4f}\n"
            f"  MAE:        {self.mae:.4f}\n"
            f"  NDCG@10:    {self.ndcg_10:.4f}"
        )


class GNNTrainer:
    """Manages the training process for HeteroGAT models with early stopping."""

    def __init__(
        self,
        model: nn.Module,
        checkpoint_dir: str = "output/gnn_checkpoints",
        lr: float = 3e-4,
        num_epochs: int = 300,
        patience: int = 30,
        weight_decay: float = 1e-4,
    ):
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.lr = lr
        self.num_epochs = num_epochs
        self.patience = patience
        self.weight_decay = weight_decay

        self.loss_fn = CriticalityLoss()
        self.device = next(model.parameters()).device
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self, data: 'HeteroData') -> Tuple[Dict[str, List[float]], Optional[EvalMetrics]]:
        """Run the full training loop with early stopping.
        
        Returns:
            Tuple of (history_dict, best_val_metrics).
        """
        logger.info(
            "Starting training | epochs=%d | lr=%.2e | device=%s",
            self.num_epochs, self.lr, self.device
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs
        )

        # Handle single HeteroData vs DataLoader
        if isinstance(data, HeteroData):
            loader = DataLoader([data], batch_size=1)
        else:
            loader = data

        # Initial logging of split sizes (first batch only)
        first_batch = next(iter(loader))
        first_batch = first_batch.to(self.device)
        for nt in first_batch.node_types:
            if hasattr(first_batch[nt], "train_mask"):
                n_train = first_batch[nt].train_mask.sum().item()
                n_val = first_batch[nt].val_mask.sum().item()
                if n_train > 0 or n_val > 0:
                    logger.info("  [%s] Train: %d | Val: %d", nt, n_train, n_val)

        best_val_rho = -1.0
        best_val_metrics: Optional[EvalMetrics] = None
        epochs_without_improvement = 0
        history = {"train_loss": [], "val_loss": [], "val_rho": []}

        for epoch in range(1, self.num_epochs + 1):
            # ── Train ────────────────────────────────────────────────────────
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                # Forward pass
                x_dict = {nt: batch[nt].x for nt in batch.node_types if hasattr(batch[nt], "x")}
                ei_dict = {rel: batch[rel].edge_index for rel in batch.edge_types}
                ea_dict = {rel: batch[rel].edge_attr for rel in batch.edge_types if hasattr(batch[rel], "edge_attr")}

                # Handle both NodeCriticalityGNN and EdgeCriticalityGNN
                output = self.model(x_dict, ei_dict, ea_dict)
                if isinstance(output, tuple):
                    node_preds, _ = output
                else:
                    node_preds = output

                # Compute loss over training nodes of all types
                batch_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                for nt, preds in node_preds.items():
                    store = batch[nt]
                    if not (hasattr(store, "y") and hasattr(store, "train_mask")):
                        continue
                    mask = store.train_mask
                    if mask.sum() == 0:
                        continue

                    # Pass both simulation labels (y) and RMAV scores (y_rmav)
                    rmav_target = store.y_rmav if hasattr(store, "y_rmav") else None
                    loss, _ = self.loss_fn(preds, store.y, mask, rmav_target)
                    batch_loss = batch_loss + loss

                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                num_batches += 1

            scheduler.step()
            avg_epoch_loss = epoch_loss / max(num_batches, 1)

            # ── Validation ───────────────────────────────────────────────────
            self.model.eval()
            # If DataLoader, evaluate on all validation sets or just the first one?
            # Standard multi-graph behavior: evaluate on the loader but validation usually needs consistent set.
            # For simplicity, we evaluate on the first batch (often the primary graph) or the full loader if small.
            val_metrics = evaluate(self.model, first_batch, "val_mask", self.device)
            val_rho = val_metrics.spearman_rho

            history["train_loss"].append(avg_epoch_loss)
            history["val_rho"].append(val_rho)

            if val_rho > best_val_rho:
                best_val_rho = val_rho
                best_val_metrics = val_metrics
                epochs_without_improvement = 0
                self._save_checkpoint("best_model.pt")
            else:
                epochs_without_improvement += 1

            if epoch % 20 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d | Loss: %.4f | Val ρ: %.4f | Heads: %s",
                    epoch, avg_epoch_loss, val_rho,
                    "↑" if epochs_without_improvement == 0 else " "
                )

            if epochs_without_improvement >= self.patience:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

        # Restore best model
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
            logger.info("Restored best model with validation rho: %.4f", best_val_rho)

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
    """Compute validation/test metrics.

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
    data = data.to(device)

    with torch.no_grad():
        x_dict = {nt: data[nt].x for nt in data.node_types if hasattr(data[nt], "x")}
        ei_dict = {rel: data[rel].edge_index for rel in data.edge_types}
        ea_dict = {rel: data[rel].edge_attr for rel in data.edge_types if hasattr(data[rel], "edge_attr")}

        if hasattr(model, "predict_edges") and getattr(model, "predict_edges", False):
            node_preds, _ = model(x_dict, ei_dict, ea_dict)
        else:
            node_preds = model(x_dict, ei_dict, ea_dict)

    all_preds = []
    all_targets = []

    for nt, preds in node_preds.items():
        store = data[nt]
        if not (hasattr(store, "y") and hasattr(store, mask_name)):
            continue
        mask = getattr(store, mask_name)
        if mask.sum() == 0:
            continue

        all_preds.append(preds[mask].cpu().numpy())
        all_targets.append(store.y[mask].cpu().numpy())

    if not all_preds:
        return EvalMetrics(0, 0, 0, 0, 0, 0, 0)

    y_pred = np.concatenate(all_preds, axis=0)  # (N, 5)
    y_true = np.concatenate(all_targets, axis=0) # (N, 5)

    # Composite scores (column 0)
    p_comp = y_pred[:, 0]
    t_comp = y_true[:, 0]

    # Spearman rho
    # Silencing ConstantInputWarning if predictions or targets are constant
    if len(p_comp) > 1 and (np.all(p_comp == p_comp[0]) or np.all(t_comp == t_comp[0])):
        rho = 0.0
    else:
        rho, _ = spearmanr(p_comp, t_comp)
        if np.isnan(rho):
            rho = 0.0

    # F1 (threshold at 0.5)
    f1 = f1_score(t_comp >= 0.5, p_comp >= 0.5, zero_division=0)

    # Error metrics
    rmse = np.sqrt(np.mean((p_comp - t_comp)**2))
    mae = np.mean(np.abs(p_comp - t_comp))

    # Overlap and NDCG
    top_5_pred = np.argsort(p_comp)[-5:]
    top_5_true = np.argsort(t_comp)[-5:]
    top_5_overlap = len(set(top_5_pred) & set(top_5_true)) / 5.0

    top_10_pred = np.argsort(p_comp)[-10:]
    top_10_true = np.argsort(t_comp)[-10:]
    top_10_overlap = len(set(top_10_pred) & set(top_10_true)) / 10.0 if len(t_comp) >= 10 else 0

    ndcg = _compute_ndcg(p_comp, t_comp, k=10)

    return EvalMetrics(
        spearman_rho=float(rho),
        f1_score=float(f1),
        rmse=float(rmse),
        mae=float(mae),
        top_5_overlap=float(top_5_overlap),
        top_10_overlap=float(top_10_overlap),
        ndcg_10=float(ndcg),
    )


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
