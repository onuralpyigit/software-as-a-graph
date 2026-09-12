"""
saag/prediction/models/baselines.py — Homogeneous GAT baselines
================================================================

Block A: Two homogeneous GAT baselines for the Middleware 2026 comparison.

Models
------
HomogeneousGAT_Unweighted
    Flat single-relation GAT on a homogeneous projection of the graph.
    No edge_attr — topology structure only.
    Corresponds to the «topology-only» ablation in paper Table 3.

HomogeneousGAT_ScalarWeighted
    Same flat GAT but uses a scalar edge weight (the QoS aggregate w(e))
    as a single edge feature.  Corresponds to the «scalar-QoS» ablation.

Both models:
- Accept the same (x_dict, edge_index_dict, edge_attr_dict) interface as
  NodeCriticalityGNN for drop-in trainer compatibility.
- Project all node types to a common hidden embedding before GATConv.
- Output shape: {node_type: (N, NUM_LABEL_DIMS)} matching the RM multi-task
  convention (composite, reliability, maintainability).

Usage
-----
  from saag.prediction.models.baselines import HomogeneousGAT_Unweighted

  model = HomogeneousGAT_Unweighted(
      node_type_dims={"Application": 23, "Broker": 19, "Topic": 22, "Node": 20, "Library": 23},
      hidden_channels=64,
      num_heads=4,
      num_layers=3,
  )
  out = model(x_dict, edge_index_dict, edge_attr_dict)
  # out: {"Application": Tensor(N_app, 3), "Broker": Tensor(N_brk, 3), ...}
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from saag.prediction.models.core import NUM_LABEL_DIMS
from saag.prediction.data_preparation import NODE_TYPE_TO_DIM

logger = logging.getLogger(__name__)


# ── Shared backbone ───────────────────────────────────────────────────────────

class _ResidualMLP(nn.Module):
    """Two-layer MLP with residual connection and layer norm."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        # LayerNorm on a 1-element vector collapses to 0 (mean=x, var=0 → output=beta=0).
        # Use Identity when out_dim=1 so the scalar residual is preserved.
        self.norm = nn.LayerNorm(out_dim) if out_dim > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        h = F.gelu(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        return self.norm(h + self.proj(x))


class _HomoGATBase(nn.Module):
    """Shared base for homogeneous GAT baselines.

    Converts the heterogeneous input (x_dict) to a flat homogeneous tensor by
    projecting each node type to a common hidden_channels space, concatenating
    the type one-hot, then running GATConv layers on the union of all edges.

    The flat graph preserves all edges but loses their type semantics — this is
    exactly the ablation we want for the «homogeneous» baselines.
    """

    def __init__(
        self,
        node_type_dims: Dict[str, int],
        hidden_channels: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        edge_dim: Optional[int],  # None → no edge features
    ):
        super().__init__()
        self._require_pyg()

        self.node_types = list(node_type_dims.keys())
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_p = dropout
        #: Width of the edge channel handed to GATConv, and the number of
        #: leading ``edge_attr`` columns ``_build_homo_edge_attr`` slices out.
        #: ``None`` means the model sees no edge features at all.
        self.edge_dim = edge_dim

        # Per-type input projections → common hidden space
        self.input_proj = nn.ModuleDict({
            nt: nn.Sequential(
                nn.Linear(dim, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.GELU(),
            )
            for nt, dim in node_type_dims.items()
        })

        # GATConv layers (homogeneous)
        from torch_geometric.nn import GATConv
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // num_heads,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            )
            # After concat: heads × (hidden//heads) = hidden
            self.norms.append(nn.LayerNorm(hidden_channels))

        # RM output heads (same as NodeCriticalityGNN for fair comparison)
        self.rm_heads = nn.ModuleDict({
            dim: _ResidualMLP(hidden_channels, hidden_channels // 2, 1, dropout)
            for dim in ["reliability", "maintainability"]
        })
        self.composite_head = _ResidualMLP(hidden_channels + 2, hidden_channels // 2, 1, dropout)

    @staticmethod
    def _require_pyg():
        try:
            import torch_geometric  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "PyTorch Geometric is required for GNN baselines.\n"
                "Install with: pip install torch-geometric"
            ) from exc

    def _build_homo_graph(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tuple[int, int]]]:
        """Merge all node types into a flat tensor; remap edge indices globally.

        Returns
        -------
        x_flat : (N_total, hidden_channels) — projected node embeddings
        edge_index_flat : (2, E_total) — globally remapped edges
        type_offsets : {node_type: (start, end)} — global index ranges per type
        """
        # Project each type to hidden space
        parts = []
        offsets: Dict[str, Tuple[int, int]] = {}
        cursor = 0
        for nt in self.node_types:
            if nt not in x_dict or nt not in self.input_proj:
                continue
            h = self.input_proj[nt](x_dict[nt])  # (N_t, hidden)
            start = cursor
            cursor += h.size(0)
            offsets[nt] = (start, cursor)
            parts.append(h)

        if not parts:
            return torch.zeros(0, self.hidden_channels), torch.zeros(2, 0, dtype=torch.long), offsets

        x_flat = torch.cat(parts, dim=0)  # (N_total, hidden)

        # Remap edge indices
        edge_parts = []
        for (src_type, _, dst_type), ei in edge_index_dict.items():
            if src_type not in offsets or dst_type not in offsets:
                continue
            src_offset = offsets[src_type][0]
            dst_offset = offsets[dst_type][0]
            ei_global = ei.clone()
            ei_global[0] += src_offset
            ei_global[1] += dst_offset
            edge_parts.append(ei_global)

        if edge_parts:
            edge_index_flat = torch.cat(edge_parts, dim=1)
        else:
            edge_index_flat = torch.zeros(2, 0, dtype=torch.long, device=x_flat.device)

        return x_flat, edge_index_flat, offsets

    def _decode(self, h: Tensor) -> Tensor:
        """RM output heads → (N, NUM_LABEL_DIMS) tensor."""
        r = torch.sigmoid(self.rm_heads["reliability"](h))
        m = torch.sigmoid(self.rm_heads["maintainability"](h))
        composite_in = torch.cat([h, r, m], dim=-1)
        composite = torch.sigmoid(self.composite_head(composite_in))
        return torch.cat([composite, r, m], dim=-1)  # (N, NUM_LABEL_DIMS)

    def _scatter_to_types(
        self,
        h_flat: Tensor,
        offsets: Dict[str, Tuple[int, int]],
    ) -> Dict[str, Tensor]:
        """Split flat output back to per-type tensors."""
        out: Dict[str, Tensor] = {}
        for nt, (start, end) in offsets.items():
            out[nt] = h_flat[start:end]
        return out


# ── Concrete baseline classes ─────────────────────────────────────────────────

class HomogeneousGAT_Unweighted(_HomoGATBase):
    """Flat GAT on all edges, no edge_attr.

    Ablation: topology structure only (no QoS signal at all).

    Paper name: **GL** (Section 7.2, docs/research/jss/draft.md). Internal
    identifier ``homo_unweighted`` (CLI ``--variant``, checkpoint/output naming) predates the
    paper's terminology and is kept as-is to avoid a wide, low-value rename across
    api/routers/prediction.py, cli/train_graph.py, scripts/train_all_variants.sh, and
    tests/test_baselines.py.
    """

    def __init__(
        self,
        node_type_dims: Optional[Dict[str, int]] = None,
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        **kwargs,
    ):
        dims = node_type_dims or NODE_TYPE_TO_DIM
        super().__init__(dims, hidden_channels, num_heads, num_layers, dropout, edge_dim=None)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict,
        edge_attr_dict: Optional[Dict] = None,  # ignored
    ) -> Dict[str, Tensor]:
        device = next(self.parameters()).device
        x_flat, ei_flat, offsets = self._build_homo_graph(x_dict, edge_index_dict)
        x_flat = x_flat.to(device)
        ei_flat = ei_flat.to(device)

        h = x_flat
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, ei_flat)           # no edge_attr
            h = F.dropout(F.gelu(norm(h_new + h)), p=self.dropout_p, training=self.training)

        out_flat = self._decode(h)
        return self._scatter_to_types(out_flat, offsets)


class HomogeneousGAT_ScalarWeighted(_HomoGATBase):
    """Flat GAT reading the leading ``edge_dim`` columns of the edge encoding.

    At the default ``edge_dim=1`` this is the scalar QoS aggregate
    (w(e) = 0.3·R + 0.4·D + 0.3·P) alone, without per-dimension decomposition —
    the configuration behind every published GAT-QoS / GAT-N-QoS number, sitting
    between Unweighted and HeteroQoS.

    ``edge_dim=16`` widens it to the full encoding, the same channel HGT-QoS
    consumes. That is not an ablation but a control: it removes the edge-feature
    asymmetry from the Section 7.2 typed-vs-untyped comparison, which otherwise
    confounds relational typing with per-dimension QoS access. Until this
    parameter existed the width was hard-wired to 1 and any ``edge_dim`` passed
    by a caller was silently swallowed by ``**kwargs``.

    Display name: **GAT-QoS** (saag/evaluation/variant_registry.py). Internal
    identifier ``homo_scalar`` kept as-is; see HomogeneousGAT_Unweighted's docstring for why.
    """

    def __init__(
        self,
        node_type_dims: Optional[Dict[str, int]] = None,
        hidden_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        edge_dim: int = 1,
        **kwargs,
    ):
        dims = node_type_dims or NODE_TYPE_TO_DIM
        super().__init__(
            dims, hidden_channels, num_heads, num_layers, dropout, edge_dim=edge_dim
        )

    def _build_homo_edge_attr(
        self,
        edge_index_dict: Dict,
        edge_attr_dict: Optional[Dict],
    ) -> Tensor:
        """Collect the leading ``self.edge_dim`` edge-feature columns.

        At the default ``edge_dim=1`` this is dim 0 alone — the QoS aggregate
        weight w(e) — which is what every published GAT-QoS number was trained
        on. ``edge_dim=16`` takes the whole encoding, the same channel HGT-QoS
        reads, which is what makes the RQ2 edge-channel control a control.
        """
        width = self.edge_dim or 1
        scalar_parts = []
        for rel, ei in edge_index_dict.items():
            E = ei.size(1)
            if edge_attr_dict and rel in edge_attr_dict:
                scalar_parts.append(edge_attr_dict[rel][:, :width].float())
            else:
                # No edge_attr for this relation: fall back to unit weight in
                # dim 0 and zeros elsewhere, preserving the previous behaviour
                # of the width-1 case. data_preparation writes edge_attr for
                # every relation, so this is defensive only.
                fallback = torch.zeros(E, width)
                fallback[:, 0] = 1.0
                scalar_parts.append(fallback)

        if scalar_parts:
            return torch.cat(scalar_parts, dim=0)
        return torch.zeros(0, width)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict,
        edge_attr_dict: Optional[Dict] = None,
    ) -> Dict[str, Tensor]:
        device = next(self.parameters()).device
        x_flat, ei_flat, offsets = self._build_homo_graph(x_dict, edge_index_dict)
        x_flat = x_flat.to(device)
        ei_flat = ei_flat.to(device)

        # Build scalar edge weights (1-d)
        ea_flat = self._build_homo_edge_attr(edge_index_dict, edge_attr_dict).to(device)

        h = x_flat
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, ei_flat, edge_attr=ea_flat)
            h = F.dropout(F.gelu(norm(h_new + h)), p=self.dropout_p, training=self.training)

        out_flat = self._decode(h)
        return self._scatter_to_types(out_flat, offsets)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_baseline(
    variant: str,
    node_type_dims: Optional[Dict[str, int]] = None,
    hidden_channels: int = 64,
    num_heads: int = 4,
    num_layers: int = 3,
    dropout: float = 0.2,
    edge_dim: Optional[int] = None,
) -> nn.Module:
    """Instantiate a baseline model by variant name.

    Parameters
    ----------
    variant:
        One of ``"homo_unweighted"`` or ``"homo_scalar"``.
    node_type_dims:
        Per-type feature dimensions.  Defaults to ``NODE_TYPE_TO_DIM``.
    edge_dim:
        Width of the edge channel for ``"homo_scalar"``.  ``None`` keeps that
        class's own default of 1, so existing callers are unaffected.  Ignored
        by ``"homo_unweighted"``, which takes no edge features by construction.
        Resolve it from a variant id with
        ``saag.evaluation.variant_registry.edge_dim`` rather than hard-coding.
    """
    kwargs = dict(
        node_type_dims=node_type_dims or NODE_TYPE_TO_DIM,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )
    if variant == "homo_unweighted":
        return HomogeneousGAT_Unweighted(**kwargs)
    elif variant == "homo_scalar":
        if edge_dim is not None:
            kwargs["edge_dim"] = edge_dim
        return HomogeneousGAT_ScalarWeighted(**kwargs)
    else:
        raise ValueError(
            f"Unknown baseline variant '{variant}'. "
            "Use 'homo_unweighted' or 'homo_scalar'."
        )
