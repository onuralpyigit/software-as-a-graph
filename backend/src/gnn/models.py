"""
GNN Models
==========
Provides two cooperating neural network architectures for predicting
component and relationship criticality in distributed publish-subscribe
systems.

Architecture overview
---------------------

NodeCriticalityGNN
    Heterogeneous Graph Attention Network (HeteroGAT) that learns
    node-level criticality embeddings for each of the five component
    types (Application, Broker, Topic, Node, Library) and for each
    of the seven relationship types.  The model predicts all five
    RMAV ground-truth dimensions simultaneously via multi-task MLP
    heads:
        output ∈ ℝ^(N×5)  →  [I*(v), IR(v), IM(v), IA(v), IV(v)]

    Each attention layer uses *type-specific* projection matrices so
    that an Application→Topic message is handled differently from a
    Node→Broker message — crucial for capture of QoS semantics.

EdgeCriticalityGNN
    An edge-scoring head that sits on top of ``NodeCriticalityGNN``.
    Given the learned node embeddings h_u and h_v, it predicts the
    criticality of the dependency edge (u, v) via:
        score(u,v) = MLP([h_u ‖ h_v ‖ edge_feat])
    This enables *relationship-level* identification of which pub-sub
    data flows are most dangerous to lose.

EnsembleGNN
    A lightweight ensemble that combines the GNN's learned scores with
    the existing RMAV topology scores via a learnable convex combination:
        Q_ens(v) = α·Q_GNN(v) + (1−α)·Q_RMAV(v),  α ∈ [0,1]
    α is a learned scalar parameter, initialised to 0.5.  The ensemble
    can be fine-tuned end-to-end or used in inference-only mode.

Hyperparameters (defaults)
--------------------------
hidden_channels      64   — embedding dimension per layer
num_heads            4    — GAT attention heads
num_layers           3    — depth of message-passing
dropout              0.2  — applied after each layer
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ── Lazy PyG import helper ─────────────────────────────────────────────────────

def _require_pyg():
    try:
        import torch_geometric  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyTorch Geometric is required for GNN functionality.\n"
            "Install with:  pip install torch-geometric"
        ) from exc


# ── Node type / relation metadata (mirrors data_preparation.py) ───────────────

NODE_TYPES: List[str] = ["Application", "Broker", "Topic", "Node", "Library"]
EDGE_TYPES: List[str] = [
    "PUBLISHES_TO",
    "SUBSCRIBES_TO",
    "ROUTES",
    "RUNS_ON",
    "CONNECTS_TO",
    "USES",
    "DEPENDS_ON",
]

NODE_FEATURE_DIM = 18   # 13 metrics + 5 type one-hot
EDGE_FEATURE_DIM = 8    # 1 weight + 7 edge-type one-hot
NUM_LABEL_DIMS   = 5    # composite, reliability, maintainability, availability, vulnerability


# ── Building blocks ────────────────────────────────────────────────────────────

class ResidualMLP(nn.Module):
    """Two-layer MLP with residual connection and layer norm."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        # Residual projection when dims differ
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        return self.norm(h + self.proj(x))


class TypeAwareLinear(nn.Module):
    """Per-node-type linear projection.

    Maintains separate weight matrices for each node type so the model
    can learn type-specific feature transformations before message passing.
    """

    def __init__(self, node_types: List[str], in_dim: int, out_dim: int):
        super().__init__()
        self.projections = nn.ModuleDict(
            {nt: nn.Linear(in_dim, out_dim) for nt in node_types}
        )

    def forward(self, x_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {
            node_type: self.projections[node_type](x)
            for node_type, x in x_dict.items()
            if node_type in self.projections
        }


# ── Core GNN model ─────────────────────────────────────────────────────────────

class NodeCriticalityGNN(nn.Module):
    """Heterogeneous GAT for node-level criticality prediction.

    Parameters
    ----------
    metadata:
        Tuple of (node_types, edge_types_as_triples) from HeteroData.metadata().
    hidden_channels:
        Width of internal embeddings (default 64).
    num_heads:
        Number of GAT attention heads (default 4).  hidden_channels must be
        divisible by num_heads.
    num_layers:
        Number of HeteroConv message-passing layers (default 3).
    dropout:
        Dropout probability (default 0.2).
    out_dims:
        Number of output dimensions per node (default 5 = RMAV + composite).
    """

    def __init__(
        self,
        metadata: Tuple,
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        out_dims: int = NUM_LABEL_DIMS,
    ):
        _require_pyg()
        super().__init__()
        from torch_geometric.nn import GATConv, HeteroConv

        node_types, edge_types = metadata
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.out_dims = out_dims

        # ── Input projection: type-specific linear + norm ─────────────────────
        self.input_proj = nn.ModuleDict(
            {
                nt: nn.Sequential(
                    nn.Linear(NODE_FEATURE_DIM, hidden_channels),
                    nn.LayerNorm(hidden_channels),
                    nn.GELU(),
                )
                for nt in node_types
            }
        )

        # ── Stacked HeteroConv layers (each wrapping GATConv per relation) ────
        head_dim = hidden_channels // num_heads
        assert hidden_channels % num_heads == 0, (
            f"hidden_channels ({hidden_channels}) must be divisible by "
            f"num_heads ({num_heads})."
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer_idx in range(num_layers):
            conv_dict = {}
            for rel in edge_types:
                src_type, rel_type, dst_type = rel
                # Edge features fed as edge_attr; add_self_loops=False because
                # heterogeneous graphs may not have symmetric src/dst types.
                conv_dict[rel] = GATConv(
                    in_channels=hidden_channels,
                    out_channels=head_dim,
                    heads=num_heads,
                    concat=True,
                    edge_dim=EDGE_FEATURE_DIM,
                    dropout=dropout,
                    add_self_loops=False,
                )
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))
            # Per-type layer norm after each message-passing step
            self.norms.append(
                nn.ModuleDict(
                    {nt: nn.LayerNorm(hidden_channels) for nt in node_types}
                )
            )

        # ── Multi-task output heads ────────────────────────────────────────────
        # Each RMAV dimension gets its own head for specialised learning.
        # The composite head aggregates all four dimension predictions.
        self.rmav_heads = nn.ModuleDict(
            {
                dim: ResidualMLP(hidden_channels, hidden_channels // 2, 1, dropout)
                for dim in ["reliability", "maintainability", "availability", "vulnerability"]
            }
        )
        self.composite_head = ResidualMLP(
            hidden_channels + 4, hidden_channels // 2, 1, dropout
        )

    def encode(self, x_dict: Dict[str, Tensor], edge_index_dict, edge_attr_dict=None) -> Dict[str, Tensor]:
        """Run message passing and return per-node embeddings (h_dict)."""
        # Project inputs to hidden dimension
        h_dict: Dict[str, Tensor] = {}
        for nt, x in x_dict.items():
            if nt in self.input_proj:
                h_dict[nt] = self.input_proj[nt](x)

        # Message-passing layers
        for layer_idx, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            # Build edge_attr_dict compatible with HeteroConv
            h_new = conv(h_dict, edge_index_dict, edge_attr_dict=edge_attr_dict if edge_attr_dict else None)

            # Residual + normalise
            for nt, h in h_new.items():
                if nt in h_dict and h.shape == h_dict[nt].shape:
                    h = h + h_dict[nt]  # residual
                if nt in norm_dict:
                    h = norm_dict[nt](h)
                h = F.dropout(F.gelu(h), p=self.dropout, training=self.training)
                h_dict[nt] = h

        return h_dict

    def decode(self, h_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Apply multi-task prediction heads to node embeddings.

        Returns a dict mapping node_type → Tensor of shape (N, 5) where
        column order is [composite, reliability, maintainability, availability, vulnerability].
        """
        out: Dict[str, Tensor] = {}
        for nt, h in h_dict.items():
            # Per-dimension RMAV scores
            r = torch.sigmoid(self.rmav_heads["reliability"](h))
            m = torch.sigmoid(self.rmav_heads["maintainability"](h))
            a = torch.sigmoid(self.rmav_heads["availability"](h))
            v = torch.sigmoid(self.rmav_heads["vulnerability"](h))

            # Composite: feed h + all four dimension predictions
            composite_in = torch.cat([h, r, m, a, v], dim=-1)
            composite = torch.sigmoid(self.composite_head(composite_in))

            # Stack: [composite, R, M, A, V]
            out[nt] = torch.cat([composite, r, m, a, v], dim=-1)  # (N, 5)
        return out

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict,
        edge_attr_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """End-to-end forward pass.

        Returns dict mapping node_type → Tensor shape (N, 5):
            col 0: composite criticality
            col 1: reliability
            col 2: maintainability
            col 3: availability
            col 4: vulnerability
        """
        h_dict = self.encode(x_dict, edge_index_dict, edge_attr_dict)
        return self.decode(h_dict)

    def get_embeddings(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict,
        edge_attr_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Return raw node embeddings (before prediction heads).

        Used by :class:`EdgeCriticalityGNN` for edge scoring.
        """
        return self.encode(x_dict, edge_index_dict, edge_attr_dict)


# ── Edge criticality model ─────────────────────────────────────────────────────

class EdgeCriticalityGNN(nn.Module):
    """Predicts criticality of pub-sub relationships (edges).

    Uses embeddings from :class:`NodeCriticalityGNN` as input.
    For each edge (u, v), the criticality score is computed as:
        score(u,v) = MLP([h_u ‖ h_v ‖ edge_feat])

    Both node and edge criticality are predicted simultaneously by sharing
    the :class:`NodeCriticalityGNN` backbone.

    Parameters
    ----------
    node_gnn:
        A :class:`NodeCriticalityGNN` instance (may be pre-trained).
    hidden_channels:
        MLP hidden dimension (default 64).
    dropout:
        Dropout probability (default 0.2).
    out_dims:
        Number of edge output dimensions (default 5 = RMAV + composite).
    """

    def __init__(
        self,
        node_gnn: NodeCriticalityGNN,
        hidden_channels: int = 64,
        dropout: float = 0.2,
        out_dims: int = NUM_LABEL_DIMS,
    ):
        super().__init__()
        self.node_gnn = node_gnn
        self.out_dims = out_dims
        self.predict_edges = True

        # MLP: h_u (D) + h_v (D) + edge_feat (8) → out_dims
        embed_dim = node_gnn.hidden_channels
        in_dim = embed_dim * 2 + EDGE_FEATURE_DIM

        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_dims),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict,
        edge_attr_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[Tuple, Tensor]]:
        """Run forward pass.

        Returns
        -------
        node_preds:
            Dict mapping node_type → Tensor (N, 5) of node criticalities.
        edge_preds:
            Dict mapping (src_type, rel, dst_type) → Tensor (E, 5) of
            edge criticalities in column order [composite, R, M, A, V].
        """
        # Get node embeddings and predictions from backbone
        h_dict = self.node_gnn.get_embeddings(x_dict, edge_index_dict, edge_attr_dict)
        node_preds = self.node_gnn.decode(h_dict)

        edge_preds: Dict[Tuple, Tensor] = {}

        for rel, edge_index in edge_index_dict.items():
            src_type, _, dst_type = rel
            if src_type not in h_dict or dst_type not in h_dict:
                continue

            src_idx, dst_idx = edge_index[0], edge_index[1]
            h_src = h_dict[src_type][src_idx]
            h_dst = h_dict[dst_type][dst_idx]

            # Fetch or zero-pad edge attributes
            if edge_attr_dict and rel in edge_attr_dict:
                e_feat = edge_attr_dict[rel]
            else:
                e_feat = torch.zeros(
                    edge_index.size(1), EDGE_FEATURE_DIM,
                    device=h_src.device,
                )

            edge_input = torch.cat([h_src, h_dst, e_feat], dim=-1)
            edge_preds[rel] = self.edge_mlp(edge_input)

        return node_preds, edge_preds


# ── Ensemble model ─────────────────────────────────────────────────────────────

class EnsembleGNN(nn.Module):
    """Learnable convex combination of GNN + RMAV scores.

    Q_ens(v) = α · Q_GNN(v)  +  (1−α) · Q_RMAV(v)

    ``α`` is a learned scalar parameter per output dimension, initialised
    to 0.5 (equal weight).  When fine-tuned on labelled data, the model
    discovers the optimal per-dimension blending.

    In ``inference_only`` mode (no RMAV scores available), the ensemble
    passes GNN predictions through unchanged.
    """

    def __init__(self, num_dims: int = NUM_LABEL_DIMS):
        super().__init__()
        # α stored in logit space so sigmoid(α_logit) ∈ (0, 1)
        self.alpha_logit = nn.Parameter(
            torch.zeros(num_dims)  # sigmoid(0) = 0.5
        )

    @property
    def alpha(self) -> Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(
        self,
        gnn_scores: Tensor,
        rmav_scores: Optional[Tensor] = None,
    ) -> Tensor:
        """Blend GNN and RMAV scores.

        Parameters
        ----------
        gnn_scores:
            Tensor (N, D) of GNN predicted scores.
        rmav_scores:
            Tensor (N, D) of RMAV topology scores.  When None, GNN
            scores are returned unchanged.

        Returns
        -------
        Tensor (N, D) of blended scores.
        """
        if rmav_scores is None:
            return gnn_scores
        alpha = self.alpha.to(gnn_scores.device)
        return alpha * gnn_scores + (1 - alpha) * rmav_scores


# ── Loss functions ─────────────────────────────────────────────────────────────

class CriticalityLoss(nn.Module):
    """Multi-task loss for criticality prediction.

    Combines:
    * MSE loss on composite scores (primary task)
    * MSE loss on each RMAV dimension (auxiliary tasks)
    * Ranking loss via ListMLE on composite scores (encourages correct
      ordering of components by criticality)

    Parameters
    ----------
    rmav_weight:
        Weight of per-dimension RMAV auxiliary losses relative to composite.
    ranking_weight:
        Weight of the ranking component.
    """

    def __init__(
        self,
        rmav_weight: float = 0.5,
        ranking_weight: float = 0.3,
    ):
        super().__init__()
        self.rmav_weight = rmav_weight
        self.ranking_weight = ranking_weight
        self.mse = nn.MSELoss(reduction="mean")

    def forward(
        self,
        pred: Tensor,   # (N, 5)
        target: Tensor, # (N, 5)
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute total loss.

        Parameters
        ----------
        pred:
            Predicted scores (N, 5).
        target:
            Ground-truth simulation scores (N, 5).
        mask:
            Optional boolean mask selecting labelled nodes.

        Returns
        -------
        total_loss, component_losses
        """
        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        if pred.shape[0] == 0:
            dummy = torch.tensor(0.0, device=pred.device, requires_grad=True)
            return dummy, {"composite": 0.0, "rmav": 0.0, "ranking": 0.0}

        # Composite MSE  (col 0)
        loss_composite = self.mse(pred[:, 0], target[:, 0])

        # Per-dimension RMAV MSE  (cols 1-4)
        loss_rmav = self.mse(pred[:, 1:], target[:, 1:])

        # Ranking loss: negative log-likelihood of the correct ordering
        loss_ranking = self._listmle_loss(pred[:, 0], target[:, 0])

        total = (
            loss_composite
            + self.rmav_weight * loss_rmav
            + self.ranking_weight * loss_ranking
        )

        components = {
            "composite": loss_composite.item(),
            "rmav": loss_rmav.item(),
            "ranking": loss_ranking.item(),
        }
        return total, components

    @staticmethod
    def _listmle_loss(scores: Tensor, targets: Tensor) -> Tensor:
        """ListMLE ranking loss (differentiable approximation to Spearman).

        Maximises the log-likelihood of the permutation induced by ``scores``
        matching the permutation induced by ``targets``.
        """
        # Sort by descending target
        _, idx = torch.sort(targets, descending=True)
        sorted_scores = scores[idx]

        # Log-sum-exp trick for numerical stability
        n = sorted_scores.shape[0]
        cumulative_log_sum_exp = []
        running = torch.tensor(-float("inf"), device=scores.device)
        for i in range(n - 1, -1, -1):
            running = torch.logaddexp(running, sorted_scores[i])
            cumulative_log_sum_exp.insert(0, running)

        log_probs = torch.stack(
            [sorted_scores[i] - cumulative_log_sum_exp[i] for i in range(n)]
        )
        return -log_probs.mean()


# ── Model factory ──────────────────────────────────────────────────────────────

def build_node_gnn(
    metadata: Tuple,
    hidden_channels: int = 64,
    num_heads: int = 4,
    num_layers: int = 3,
    dropout: float = 0.2,
) -> NodeCriticalityGNN:
    """Convenience factory for :class:`NodeCriticalityGNN`."""
    _require_pyg()
    return NodeCriticalityGNN(
        metadata=metadata,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )


def build_edge_gnn(
    metadata: Tuple,
    hidden_channels: int = 64,
    num_heads: int = 4,
    num_layers: int = 3,
    dropout: float = 0.2,
) -> EdgeCriticalityGNN:
    """Convenience factory for :class:`EdgeCriticalityGNN`."""
    _require_pyg()
    node_gnn = build_node_gnn(metadata, hidden_channels, num_heads, num_layers, dropout)
    return EdgeCriticalityGNN(node_gnn, hidden_channels=hidden_channels, dropout=dropout)
