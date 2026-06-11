"""
Prediction Domain Models

Data structures for quality analysis results, classifications, and problem detection.
GNN architecture: HGT-based heterogeneous graph neural network for criticality prediction.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from saag.core.metrics import ComponentQuality, EdgeQuality
from saag.core.criticality import CriticalityLevel, BoxPlotStats

# Import canonical dimension constants to avoid drift between data prep and model
from .data_preparation import NODE_TYPE_TO_DIM, EDGE_FEATURE_DIM

logger = logging.getLogger(__name__)


@dataclass
class QualityAnalysisResult:
    """Complete quality analysis result for a single layer."""
    timestamp: str
    layer: str
    context: str
    components: List[ComponentQuality]
    edges: List[EdgeQuality]
    classification_summary: Any
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
    entity_type: str
    category: str
    severity: str
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


# ── GNN Support Classes ─────────────────────────────────────────────────────────

def _require_pyg():
    try:
        import torch_geometric  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PyTorch Geometric is required for GNN functionality.\n"
            "Install with:  pip install torch-geometric"
        ) from exc


NODE_TYPES: List[str] = ["Application", "Broker", "Topic", "Node", "Library"]
NUM_LABEL_DIMS = 5  # composite, reliability, maintainability, availability, security


class ResidualMLP(nn.Module):
    """Two-layer MLP with residual connection and layer norm."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        return self.norm(h + self.proj(x))


class EdgeFeatureEncoder(nn.Module):
    """Aggregates edge features into destination-node embeddings via scatter-mean.

    Called before each HGTConv layer to inject edge information, since HGTConv
    does not accept raw edge_attr tensors.
    """

    def __init__(self, edge_feat_dim: int, hidden_channels: int):
        super().__init__()
        self.proj = nn.Linear(hidden_channels * 2 + edge_feat_dim, hidden_channels)

    def forward(
        self,
        h_dict: Dict[str, Tensor],
        edge_index_dict: Dict,
        edge_attr_dict: Dict,
    ) -> Dict[str, Tensor]:
        try:
            from torch_scatter import scatter_mean
        except ImportError:
            # Fallback: manual scatter via index_add
            def scatter_mean(src, index, dim, dim_size):
                out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
                count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
                out.index_add_(0, index, src)
                count.index_add_(0, index, torch.ones(src.size(0), 1, device=src.device))
                return out / count.clamp(min=1)

        augmented = {k: v.clone() for k, v in h_dict.items()}
        for rel, edge_index in edge_index_dict.items():
            src_type, _, dst_type = rel
            if rel not in edge_attr_dict or dst_type not in h_dict or src_type not in h_dict:
                continue
            h_src = h_dict[src_type][edge_index[0]]
            h_dst = h_dict[dst_type][edge_index[1]]
            e_feat = edge_attr_dict[rel]
            fused = torch.cat([h_src, h_dst, e_feat], dim=-1)
            e = self.proj(fused)                        # (E, hidden)
            n_dst = h_dict[dst_type].size(0)
            aggr = scatter_mean(e, edge_index[1], dim=0, dim_size=n_dst)
            augmented[dst_type] = augmented[dst_type] + aggr
        return augmented



class TypedEdgeEncoder(nn.Module):
    """Per-relation-type edge encoder for EdgeCriticalityGNN.

    Uses relation-specific linear projections instead of a shared MLP,
    which better captures the semantics of different edge types.
    """

    def __init__(
        self,
        edge_feat_dim: int,
        hidden_channels: int,
        edge_types: List[Tuple],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.type_proj = nn.ModuleDict({
            self._rel_key(rel): nn.Linear(edge_feat_dim, hidden_channels)
            for rel in edge_types
        })
        self.fuse = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
        )
        self.out_head = ResidualMLP(hidden_channels, hidden_channels // 2, NUM_LABEL_DIMS, dropout)

    @staticmethod
    def _rel_key(rel: Tuple) -> str:
        return "__".join(str(r) for r in rel)

    def forward(self, h_src: Tensor, h_dst: Tensor, e_feat: Tensor, rel: Tuple) -> Tensor:
        key = self._rel_key(rel)
        if key in self.type_proj:
            e_proj = self.type_proj[key](e_feat)
        else:
            e_proj = torch.zeros(e_feat.size(0), h_src.size(-1), device=h_src.device)
        fused = self.fuse(torch.cat([h_src, h_dst, e_proj], dim=-1))
        return torch.sigmoid(self.out_head(fused))


class EdgeAwareHGTConv(nn.Module):
    """Heterogeneous Graph Transformer Convolution with native edge feature injection.

    Instead of pre-aggregating edge features into destination nodes, this layer
    projects edge features directly into the Key and Value spaces of each individual
    edge before message passing. This avoids information smoothing in dense networks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        heads: int = 4,
        edge_feat_dim: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_types, self.edge_types = metadata
        self.heads = heads
        self.head_dim = out_channels // heads
        
        # Projections for each node type
        self.k_proj = nn.ModuleDict({
            nt: nn.Linear(in_channels, out_channels) for nt in self.node_types
        })
        self.q_proj = nn.ModuleDict({
            nt: nn.Linear(in_channels, out_channels) for nt in self.node_types
        })
        self.v_proj = nn.ModuleDict({
            nt: nn.Linear(in_channels, out_channels) for nt in self.node_types
        })
        
        # Projections for each relation type (edges)
        self.k_edge_proj = nn.ModuleDict({
            self._rel_key(rel): nn.Linear(edge_feat_dim, out_channels) for rel in self.edge_types
        })
        self.v_edge_proj = nn.ModuleDict({
            self._rel_key(rel): nn.Linear(edge_feat_dim, out_channels) for rel in self.edge_types
        })
        
        # Relation-specific query-key attention projection and message projection
        self.relation_att = nn.ParameterDict({
            self._rel_key(rel): nn.Parameter(torch.ones(heads)) for rel in self.edge_types
        })
        self.relation_msg = nn.ModuleDict({
            self._rel_key(rel): nn.Linear(out_channels, out_channels) for rel in self.edge_types
        })
        
        # Output projections to combine heads
        self.out_proj = nn.ModuleDict({
            nt: nn.Linear(out_channels, out_channels) for nt in self.node_types
        })

    @staticmethod
    def _rel_key(rel: Tuple[str, str, str]) -> str:
        return "__".join(rel)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], Tensor]] = None,
    ) -> Dict[str, Tensor]:
        try:
            from torch_scatter import scatter_add
        except ImportError:
            # Fallback scatter_add
            def scatter_add(src, index, dim, dim_size):
                out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
                out.index_add_(0, index, src)
                return out

        from torch_geometric.utils import softmax
        import math

        # Project node features to Query, Key, Value spaces
        k_dict = {nt: self.k_proj[nt](x) for nt, x in x_dict.items() if nt in self.k_proj}
        q_dict = {nt: self.q_proj[nt](x) for nt, x in x_dict.items() if nt in self.q_proj}
        v_dict = {nt: self.v_proj[nt](x) for nt, x in x_dict.items() if nt in self.v_proj}
        
        # Initialize output dict with zeros
        out_dict = {nt: torch.zeros_like(x) for nt, x in x_dict.items()}
        
        # Loop over relations
        for rel, edge_index in edge_index_dict.items():
            src_type, _, dst_type = rel
            if src_type not in k_dict or dst_type not in q_dict:
                continue
            if edge_index.numel() == 0:
                continue
                
            rel_k = self._rel_key(rel)
            
            # 1. Fetch keys/queries/values for the edges
            k_src = k_dict[src_type][edge_index[0]]  # (E, out_channels)
            v_src = v_dict[src_type][edge_index[0]]  # (E, out_channels)
            q_dst = q_dict[dst_type][edge_index[1]]  # (E, out_channels)
            
            # 2. Inject edge features (if provided) directly into the Key and Value of the specific edge
            if edge_attr_dict and rel in edge_attr_dict and rel_k in self.k_edge_proj:
                e_feat = edge_attr_dict[rel]
                k_src = k_src + self.k_edge_proj[rel_k](e_feat)
                v_src = v_src + self.v_edge_proj[rel_k](e_feat)
                
            # 3. Reshape for multi-head attention
            # (E, out_channels) -> (E, heads, head_dim)
            k_src = k_src.view(-1, self.heads, self.head_dim)
            v_src = v_src.view(-1, self.heads, self.head_dim)
            q_dst = q_dst.view(-1, self.heads, self.head_dim)
            
            # 4. Compute attention coefficients: (q * k).sum(dim=-1) / sqrt(head_dim)
            # alpha shape: (E, heads)
            alpha = (q_dst * k_src).sum(dim=-1) / math.sqrt(self.head_dim)
            
            # Scale by relation-specific attention multiplier
            alpha = alpha * self.relation_att[rel_k].unsqueeze(0)
            
            # Softmax attention coefficients over incoming edges for each destination node
            # alpha shape: (E, heads)
            alpha = softmax(alpha, edge_index[1], num_nodes=x_dict[dst_type].size(0))
            
            # 5. Compute relation-specific message transformation
            # Project value back from head space to transform it
            v_src = v_src.view(-1, self.out_channels)
            msg = self.relation_msg[rel_k](v_src)  # (E, out_channels)
            msg = msg.view(-1, self.heads, self.head_dim)
            
            # Apply attention weights to message
            msg = msg * alpha.unsqueeze(-1)  # (E, heads, head_dim)
            msg = msg.reshape(-1, self.out_channels)  # (E, out_channels)
            
            # 6. Scatter-add message to destination nodes
            n_dst = x_dict[dst_type].size(0)
            dst_msg = scatter_add(msg, edge_index[1], dim=0, dim_size=n_dst)
            
            out_dict[dst_type] = out_dict[dst_type] + dst_msg
            
        # Post-process node aggregations (run through out_proj)
        for nt, h in out_dict.items():
            if nt in self.out_proj:
                out_dict[nt] = self.out_proj[nt](h)
                
        return out_dict


class NodeCriticalityGNN(nn.Module):
    """Heterogeneous Graph Transformer (HGT) for node-level criticality prediction.

    Architecture:
    - Per-type input projections → hidden_channels
    - N layers of EdgeAwareHGTConv with native edge features
    - Optional bidirectional pass (forward + reverse) for upstream/downstream awareness
    - Four RMAV output heads + one composite head (all sigmoid-activated)
    """

    def __init__(
        self,
        metadata: Tuple,
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        use_bidirectional: bool = True,
    ):
        _require_pyg()
        super().__init__()

        node_types, edge_types = metadata
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.out_dims = NUM_LABEL_DIMS
        self.use_bidirectional = use_bidirectional

        # Per-type input projections — dims sourced from data_preparation constants
        self.input_proj = nn.ModuleDict({
            nt: nn.Sequential(
                nn.Linear(NODE_TYPE_TO_DIM.get(nt, 18), hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.GELU(),
            )
            for nt in node_types
        })

        # EdgeAwareHGTConv layers with native edge feature injection and per-layer residual norms
        self.convs = nn.ModuleList([
            EdgeAwareHGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=num_heads,
                edge_feat_dim=EDGE_FEATURE_DIM,
            )
            for _ in range(num_layers)
        ])
        # Unused, kept for backward compatibility and parameter inspection scripts
        self.edge_encoders = nn.ModuleList([
            EdgeFeatureEncoder(EDGE_FEATURE_DIM, hidden_channels)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(hidden_channels) for nt in node_types})
            for _ in range(num_layers)
        ])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

        # Optional reverse-direction EdgeAwareHGTConv for bidirectional awareness
        if use_bidirectional:
            rev_edge_types = [
                (dst, "rev__" + etype, src)
                for (src, etype, dst) in edge_types
            ]
            rev_metadata = (node_types, rev_edge_types)
            self.rev_conv = EdgeAwareHGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=rev_metadata,
                heads=num_heads,
                edge_feat_dim=EDGE_FEATURE_DIM,
            )
        else:
            self.rev_conv = None

        # Output heads (unchanged from prior architecture)
        self.rmas_heads = nn.ModuleDict({
            dim: ResidualMLP(hidden_channels, hidden_channels // 2, 1, dropout)
            for dim in ["reliability", "maintainability", "availability", "security"]
        })
        self.composite_head = ResidualMLP(hidden_channels + 4, hidden_channels // 2, 1, dropout)

    def _apply_reverse_pass(
        self, h: Dict[str, Tensor], edge_index_dict: Dict, edge_attr_dict: Optional[Dict] = None
    ) -> Dict[str, Tensor]:
        """Single reverse-direction HGTConv pass for upstream signal propagation."""
        rev_ei = {
            (dst, "rev__" + etype, src): torch.stack([ei[1], ei[0]])
            for (src, etype, dst), ei in edge_index_dict.items()
        }
        if not rev_ei:
            return h
        rev_ea = None
        if edge_attr_dict:
            rev_ea = {
                (dst, "rev__" + etype, src): edge_attr_dict[(src, etype, dst)]
                for (src, etype, dst) in edge_index_dict.keys()
                if (src, etype, dst) in edge_attr_dict
            }
        h_rev = self.rev_conv(h, rev_ei, rev_ea)
        for nt, h_r in h_rev.items():
            if nt in h:
                h[nt] = h[nt] + 0.5 * h_r
        return h

    def encode(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict,
        edge_attr_dict: Optional[Dict] = None,
    ) -> Dict[str, Tensor]:
        h: Dict[str, Tensor] = {
            nt: self.input_proj[nt](x)
            for nt, x in x_dict.items()
            if nt in self.input_proj
        }

        for conv, norm_d, drop in zip(
            self.convs, self.norms, self.dropouts
        ):
            h_new = conv(h, edge_index_dict, edge_attr_dict)
            for nt, h_n in h_new.items():
                residual = h.get(nt, torch.zeros_like(h_n))
                h[nt] = drop(F.gelu(norm_d[nt](h_n + residual)))

        if self.use_bidirectional and self.rev_conv is not None:
            h = self._apply_reverse_pass(h, edge_index_dict, edge_attr_dict)

        return h

    def decode(self, h_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        for nt, h in h_dict.items():
            r = torch.sigmoid(self.rmas_heads["reliability"](h))
            m = torch.sigmoid(self.rmas_heads["maintainability"](h))
            a = torch.sigmoid(self.rmas_heads["availability"](h))
            s = torch.sigmoid(self.rmas_heads["security"](h))
            composite_in = torch.cat([h, r, m, a, s], dim=-1)
            composite = torch.sigmoid(self.composite_head(composite_in))
            out[nt] = torch.cat([composite, r, m, a, s], dim=-1)
        return out

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        h_dict = self.encode(x_dict, edge_index_dict, edge_attr_dict)
        return self.decode(h_dict)

    def get_embeddings(self, x_dict, edge_index_dict, edge_attr_dict=None):
        return self.encode(x_dict, edge_index_dict, edge_attr_dict)


class EdgeCriticalityGNN(nn.Module):
    """Predicts criticality of pub-sub relationships using TypedEdgeEncoder."""

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
        self.typed_edge_enc = TypedEdgeEncoder(
            edge_feat_dim=EDGE_FEATURE_DIM,
            hidden_channels=hidden_channels,
            edge_types=node_gnn.edge_types,
            dropout=dropout,
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
                e_feat = torch.zeros(
                    edge_index.size(1), EDGE_FEATURE_DIM, device=h_src.device
                )
            edge_preds[rel] = self.typed_edge_enc(h_src, h_dst, e_feat, rel)
        return node_preds, edge_preds


class EnsembleGNN(nn.Module):
    """Learnable convex combination of GNN + RMAV scores.

    Available as optional "ensemble" mode for research/comparison.
    Default prediction mode is "gnn" — GNN-only output.
    RMAV scores are used as regularization targets during training
    but are NOT blended into the default output.
    """

    def __init__(self, num_dims: int = NUM_LABEL_DIMS):
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
    """Multi-task criticality loss with consistency regularization and pairwise ranking.

    Loss components:
    - MSE on composite score (labeled nodes)
    - MSE on RMAV sub-scores (labeled nodes, multitask)
    - ListMLE ranking loss on composite (labeled nodes)
    - Pairwise margin ranking loss on composite (labeled nodes)
    - RMAV consistency regularization on unlabeled nodes
    """

    def __init__(
        self,
        multitask_weight: float = 0.5,
        rmav_consistency_weight: float = 0.1,
        ranking_weight: float = 0.3,
        pairwise_ranking_weight: float = 0.1,
    ):
        super().__init__()
        self.multitask_weight = multitask_weight
        self.rmav_consistency_weight = rmav_consistency_weight
        self.ranking_weight = ranking_weight
        self.pairwise_ranking_weight = pairwise_ranking_weight
        self.mse = nn.MSELoss(reduction="mean")

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        mask: Tensor,
        rmav_target: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        labeled_pred = pred[mask]
        labeled_target = target[mask]

        if labeled_pred.shape[0] == 0:
            zero = torch.tensor(0.0, device=pred.device)
            return zero, {"composite": 0.0, "multitask": 0.0, "ranking": 0.0,
                          "pairwise": 0.0, "consistency": 0.0}

        loss_composite = self.mse(labeled_pred[:, 0], labeled_target[:, 0])
        loss_multitask = self.mse(labeled_pred[:, 1:], labeled_target[:, 1:])
        loss_ranking = self._listmle_loss(labeled_pred[:, 0], labeled_target[:, 0])
        loss_pairwise = self._pairwise_margin_loss(labeled_pred[:, 0], labeled_target[:, 0])

        supervised_loss = (
            loss_composite
            + self.multitask_weight * loss_multitask
            + self.ranking_weight * loss_ranking
            + self.pairwise_ranking_weight * loss_pairwise
        )

        loss_consistency = torch.tensor(0.0, device=pred.device)
        if rmav_target is not None:
            unlabeled_mask = ~mask
            unlabeled_pred = pred[unlabeled_mask]
            unlabeled_rmav = rmav_target[unlabeled_mask]
            if unlabeled_pred.shape[0] > 0:
                loss_consistency = self.mse(unlabeled_pred[:, 1:], unlabeled_rmav[:, 1:])

        total = supervised_loss + self.rmav_consistency_weight * loss_consistency

        components = {
            "composite": loss_composite.item(),
            "multitask": loss_multitask.item(),
            "ranking": loss_ranking.item(),
            "pairwise": loss_pairwise.item(),
            "consistency": loss_consistency.item(),
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

    @staticmethod
    def _pairwise_margin_loss(
        scores: Tensor, targets: Tensor, margin: float = 0.05
    ) -> Tensor:
        """For all pairs (i,j) where target_i > target_j, penalize score_i < score_j + margin."""
        n = scores.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=scores.device)
        s_diff = scores.unsqueeze(0) - scores.unsqueeze(1)    # (n, n)
        t_diff = targets.unsqueeze(0) - targets.unsqueeze(1)  # (n, n)
        should_rank = (t_diff > margin).float()
        loss = torch.clamp(margin - s_diff, min=0.0) * should_rank
        return loss.sum() / (should_rank.sum() + 1e-8)


def build_node_gnn(
    metadata,
    hidden_channels: int = 64,
    num_heads: int = 4,
    num_layers: int = 3,
    dropout: float = 0.2,
    use_bidirectional: bool = True,
) -> NodeCriticalityGNN:
    return NodeCriticalityGNN(
        metadata, hidden_channels, num_heads, num_layers, dropout,
        use_bidirectional=use_bidirectional,
    )


def build_edge_gnn(
    metadata,
    hidden_channels: int = 64,
    num_heads: int = 4,
    num_layers: int = 3,
    dropout: float = 0.2,
    use_bidirectional: bool = True,
) -> EdgeCriticalityGNN:
    node_gnn = build_node_gnn(
        metadata, hidden_channels, num_heads, num_layers, dropout, use_bidirectional
    )
    return EdgeCriticalityGNN(node_gnn, hidden_channels=hidden_channels, dropout=dropout)
