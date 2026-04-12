"""
Prediction Domain Models

Data structures for quality analysis results, classifications, and problem detection.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.core.metrics import ComponentQuality, EdgeQuality
from src.core.criticality import CriticalityLevel, BoxPlotStats

logger = logging.getLogger(__name__)


@dataclass
class QualityAnalysisResult:
    """Complete quality analysis result for a single layer."""
    timestamp: str
    layer: str
    context: str
    components: List[ComponentQuality]
    edges: List[EdgeQuality]
    classification_summary: Any  # Avoid circular or complex imports for now
    weights: Any = None
    stats: Dict[str, BoxPlotStats] = field(default_factory=dict)
    sensitivity: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "timestamp": self.timestamp,
            "layer": self.layer,
            "context": self.context,
            "components": [c.to_dict() for c in self.components],
            "edges": [e.to_dict() for e in self.edges],
            "classification_summary": self.classification_summary.to_dict() if hasattr(self.classification_summary, "to_dict") else self.classification_summary,
        }
        if self.sensitivity:
            result["sensitivity"] = self.sensitivity
        return result

    def get_critical_components(self) -> List[ComponentQuality]:
        return [c for c in self.components if c.levels.overall == CriticalityLevel.CRITICAL]

    def get_high_priority(self) -> List[ComponentQuality]:
        return [c for c in self.components if c.levels.overall >= CriticalityLevel.HIGH]

    def get_by_type(self, comp_type: str) -> List[ComponentQuality]:
        return [c for c in self.components if c.type == comp_type]

    def get_critical_edges(self) -> List[EdgeQuality]:
        return [e for e in self.edges if e.level == CriticalityLevel.CRITICAL]

    def get_requiring_attention(self) -> tuple[List[ComponentQuality], List[EdgeQuality]]:
        comps = [c for c in self.components if c.requires_attention]
        edges = [e for e in self.edges if e.level >= CriticalityLevel.HIGH]
        return comps, edges


@dataclass
class DetectedProblem:
    """A detected architectural problem or risk."""
    entity_id: str
    entity_type: str           # Component | Edge | System
    category: str              # ProblemCategory value
    severity: str              # CRITICAL | HIGH | MEDIUM | LOW
    name: str
    description: str
    recommendation: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @property
    def priority(self) -> int:
        return {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(self.severity, 0)


@dataclass
class ProblemSummary:
    """Aggregated problem counts."""
    total_problems: int
    by_severity: Dict[str, int]
    by_category: Dict[str, int]
    affected_components: int
    affected_edges: int

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @property
    def has_critical(self) -> bool:
        return self.by_severity.get("CRITICAL", 0) > 0

    @property
    def requires_attention(self) -> int:
        return self.by_severity.get("CRITICAL", 0) + self.by_severity.get("HIGH", 0)


# ── GNN Support Classes (Restored) ─────────────────────────────────────────────

def _require_pyg():
    try:
        import torch_geometric  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyTorch Geometric is required for GNN functionality.\n"
            "Install with:  pip install torch-geometric"
        ) from exc


# Node type / relation metadata
NODE_TYPES: List[str] = ["Application", "Broker", "Topic", "Node", "Library"]

# Type-specific input dimensions (Base=18, CQ=5)
NODE_TYPE_TO_DIM: Dict[str, int] = {
    "Application": 23,
    "Library": 23,
    "Broker": 18,
    "Topic": 18,
    "Node": 18,
}

EDGE_FEATURE_DIM = 8    # 1 weight + 7 edge-type one-hot
NUM_LABEL_DIMS   = 5    # composite, reliability, maintainability, availability, vulnerability


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


class NodeCriticalityGNN(nn.Module):
    """Heterogeneous GAT for node-level criticality prediction."""

    def __init__(
        self,
        metadata: Tuple,
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        out_dims: int = 5,
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

        # Input projection
        self.input_proj = nn.ModuleDict(
            {
                nt: nn.Sequential(
                    nn.Linear(NODE_TYPE_TO_DIM.get(nt, 18), hidden_channels),
                    nn.LayerNorm(hidden_channels),
                    nn.GELU(),
                )
                for nt in node_types
            }
        )

        # HeteroConv layers
        head_dim = hidden_channels // num_heads
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for rel in edge_types:
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
            self.norms.append(
                nn.ModuleDict(
                    {nt: nn.LayerNorm(hidden_channels) for nt in node_types}
                )
            )

        # Output heads
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
        h_dict: Dict[str, Tensor] = {}
        for nt, x in x_dict.items():
            if nt in self.input_proj:
                h_dict[nt] = self.input_proj[nt](x)

        for conv, norm_dict in zip(self.convs, self.norms):
            h_new = conv(h_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for nt, h in h_new.items():
                if nt in h_dict and h.shape == h_dict[nt].shape:
                    h = h + h_dict[nt]
                if nt in norm_dict:
                    h = norm_dict[nt](h)
                h = F.dropout(F.gelu(h), p=self.dropout, training=self.training)
                h_dict[nt] = h
        return h_dict

    def decode(self, h_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        for nt, h in h_dict.items():
            r = torch.sigmoid(self.rmav_heads["reliability"](h))
            m = torch.sigmoid(self.rmav_heads["maintainability"](h))
            a = torch.sigmoid(self.rmav_heads["availability"](h))
            v = torch.sigmoid(self.rmav_heads["vulnerability"](h))
            composite_in = torch.cat([h, r, m, a, v], dim=-1)
            composite = torch.sigmoid(self.composite_head(composite_in))
            out[nt] = torch.cat([composite, r, m, a, v], dim=-1)
        return out

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        h_dict = self.encode(x_dict, edge_index_dict, edge_attr_dict)
        return self.decode(h_dict)

    def get_embeddings(self, x_dict, edge_index_dict, edge_attr_dict=None):
        return self.encode(x_dict, edge_index_dict, edge_attr_dict)


class EdgeCriticalityGNN(nn.Module):
    """Predicts criticality of pub-sub relationships (edges)."""

    def __init__(
        self,
        node_gnn: NodeCriticalityGNN,
        hidden_channels: int = 64,
        dropout: float = 0.2,
        out_dims: int = 5,
    ):
        super().__init__()
        self.node_gnn = node_gnn
        self.out_dims = out_dims
        self.predict_edges = True
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

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        h_dict = self.node_gnn.get_embeddings(x_dict, edge_index_dict, edge_attr_dict)
        node_preds = self.node_gnn.decode(h_dict)
        edge_preds = {}
        for rel, edge_index in edge_index_dict.items():
            src_type, _, dst_type = rel
            if src_type not in h_dict or dst_type not in h_dict:
                continue
            src_idx, dst_idx = edge_index[0], edge_index[1]
            h_src = h_dict[src_type][src_idx]
            h_dst = h_dict[dst_type][dst_idx]
            if edge_attr_dict and rel in edge_attr_dict:
                e_feat = edge_attr_dict[rel]
            else:
                e_feat = torch.zeros(edge_index.size(1), EDGE_FEATURE_DIM, device=h_src.device)
            edge_input = torch.cat([h_src, h_dst, e_feat], dim=-1)
            edge_preds[rel] = self.edge_mlp(edge_input)
        return node_preds, edge_preds


class EnsembleGNN(nn.Module):
    """Learnable convex combination of GNN + RMAV scores."""

    def __init__(self, num_dims: int = 5):
        super().__init__()
        self.alpha_logit = nn.Parameter(torch.zeros(num_dims))

    @property
    def alpha(self) -> Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(self, gnn_scores: Tensor, rmav_scores: Optional[Tensor] = None) -> Tensor:
        if rmav_scores is None:
            return gnn_scores
        alpha = self.alpha.to(gnn_scores.device)
        return alpha * gnn_scores + (1 - alpha) * rmav_scores


# ── Loss functions ─────────────────────────────────────────────────────────────

class CriticalityLoss(nn.Module):
    """Multi-task loss for criticality prediction with consistency regularization.
    
    Now implements Issue G2: 
    - Directed loss for composite/rank/multitask on labeled nodes.
    - Consistency regularization (against RMAV) on unlabeled nodes only.
    """

    def __init__(
        self,
        multitask_weight: float = 0.5,
        rmav_consistency_weight: float = 0.1, # Reduced per G2 fix
        ranking_weight: float = 0.3,
    ):
        super().__init__()
        self.multitask_weight = multitask_weight
        self.rmav_consistency_weight = rmav_consistency_weight
        self.ranking_weight = ranking_weight
        self.mse = nn.MSELoss(reduction="mean")

    def forward(
        self,
        pred: Tensor,         # (N, 5)
        target: Tensor,       # (N, 5) - Simulation ground truth
        mask: Tensor,         # (N,)   - Training mask (labeled nodes)
        rmav_target: Optional[Tensor] = None, # (N, 5) - RMAV scores (for consistency)
    ) -> Tuple[Tensor, Dict[str, float]]:
        # 1. Supervised Loss (on labeled nodes only)
        labeled_pred = pred[mask]
        labeled_target = target[mask]
        
        if labeled_pred.shape[0] == 0:
            supervised_loss = torch.tensor(0.0, device=pred.device)
            loss_composite = supervised_loss
            loss_multitask = supervised_loss
            loss_ranking = supervised_loss
        else:
            # Composite MSE (col 0)
            loss_composite = self.mse(labeled_pred[:, 0], labeled_target[:, 0])
            
            # Multi-task sub-score MSE (cols 1-4)
            loss_multitask = self.mse(labeled_pred[:, 1:], labeled_target[:, 1:])
            
            # Ranking loss
            loss_ranking = self._listmle_loss(labeled_pred[:, 0], labeled_target[:, 0])
            
            supervised_loss = (
                loss_composite 
                + self.multitask_weight * loss_multitask 
                + self.ranking_weight * loss_ranking
            )

        # 2. Consistency Regularization (on unlabeled nodes only)
        loss_rmav_consistency = torch.tensor(0.0, device=pred.device)
        if rmav_target is not None:
            unlabeled_mask = ~mask
            unlabeled_pred = pred[unlabeled_mask]
            unlabeled_rmav = rmav_target[unlabeled_mask]
            
            if unlabeled_pred.shape[0] > 0:
                # Compare GNN sub-scores with RMAV counterparts
                loss_rmav_consistency = self.mse(unlabeled_pred[:, 1:], unlabeled_rmav[:, 1:])

        total = supervised_loss + self.rmav_consistency_weight * loss_rmav_consistency

        components = {
            "composite": loss_composite.item(),
            "multitask": loss_multitask.item(),
            "ranking": loss_ranking.item(),
            "consistency": loss_rmav_consistency.item(),
        }
        return total, components

    @staticmethod
    def _listmle_loss(scores: Tensor, targets: Tensor) -> Tensor:
        _, idx = torch.sort(targets, descending=True)
        sorted_scores = scores[idx]
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


def build_node_gnn(metadata, hidden_channels=64, num_heads=4, num_layers=3, dropout=0.2):
    return NodeCriticalityGNN(metadata, hidden_channels, num_heads, num_layers, dropout)


def build_edge_gnn(metadata, hidden_channels=64, num_heads=4, num_layers=3, dropout=0.2):
    node_gnn = build_node_gnn(metadata, hidden_channels, num_heads, num_layers, dropout)
    return EdgeCriticalityGNN(node_gnn, hidden_channels=hidden_channels, dropout=dropout)
