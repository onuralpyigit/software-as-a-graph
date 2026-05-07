#!/usr/bin/env python3
"""
tools/extract_attention.py — Block G: ATM Case Study — Attention Extraction
============================================================================

Extracts HGT attention weights from a trained NodeCriticalityGNN and writes
them to attention_weights.json for rendering as an annotated subgraph.

HGT uses multi-head attention internally. This tool monkey-patches HGTConv
forward() to capture `alpha` (attention coefficients) per layer × head ×
edge. Compatible with torch_geometric >= 2.0.

Output: output/atm_case_study/attention_weights.json
  {
    "layer_<l>": {
      "<src_type>__<rel>__<dst_type>": {
        "edges": [[src_idx, dst_idx], ...],
        "heads": [[head0_alpha, ...], ...],   # (E, H)
        "mean_alpha": [float, ...]            # (E,) mean across heads
      }
    }
  }

Usage
-----
  # Use ATM checkpoint if available, otherwise train fresh on atm_system.json
  python tools/extract_attention.py
  python tools/extract_attention.py --scenario av_system --checkpoint output/my_ckpt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_RESULTS_DIR = Path("output/atm_case_study")
_LOSO_CACHE  = Path("output/loso_cache")
_SCENARIOS   = Path("data/scenarios")


# ── Attention capture ─────────────────────────────────────────────────────────

class _AttentionCapture:
    """Context manager that patches HGTConv to expose attention weights."""

    def __init__(self):
        self.weights: Dict[int, Dict] = {}  # layer_idx → per-rel attention

    def attach(self, model):
        """Register forward hooks on every HGTConv layer."""
        self._hooks = []
        for layer_idx, conv in enumerate(model.convs):
            hook = conv.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(hook)
        return self

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _make_hook(self, layer_idx: int):
        capture = self

        def hook(module, inputs, outputs):
            # HGTConv returns (out_dict, alpha_dict) if return_attention_weights=True
            # but vanilla forward returns just out_dict. We access internal alpha if stored.
            # PyG >= 2.3: HGTConv stores _alpha on the module after forward.
            alpha_dict = getattr(module, "_alpha", None)
            if alpha_dict is None:
                return
            capture.weights[layer_idx] = {}
            for rel_key, alpha in alpha_dict.items():
                if alpha is None:
                    continue
                src_t, etype, dst_t = rel_key
                key = f"{src_t}__{etype}__{dst_t}"
                capture.weights[layer_idx][key] = alpha.detach().cpu().numpy()

        return hook


def _extract_via_return_attention_weights(
    model, x_dict, edge_index_dict, edge_attr_dict
) -> Dict[str, Dict]:
    """Alternative: call each HGTConv with return_attention_weights=True."""
    import torch
    h = {nt: model.input_proj[nt](x_dict[nt]) for nt in x_dict if nt in model.input_proj}
    all_weights: Dict[str, Dict] = {}

    for layer_idx, (conv, edge_enc) in enumerate(zip(model.convs, model.edge_encoders)):
        if edge_attr_dict:
            h = edge_enc(h, edge_index_dict, edge_attr_dict)
        try:
            h_new, alphas = conv(h, edge_index_dict, return_attention_weights=True)
        except TypeError:
            # PyG version doesn't support return_attention_weights for HGTConv
            h_new = conv(h, edge_index_dict)
            all_weights[f"layer_{layer_idx}"] = {}
            h = {nt: h_new.get(nt, h.get(nt)) for nt in h}
            continue

        layer_key = f"layer_{layer_idx}"
        all_weights[layer_key] = {}
        if isinstance(alphas, dict):
            for rel, (ei, alpha) in alphas.items():
                src_t, etype, dst_t = rel
                key = f"{src_t}__{etype}__{dst_t}"
                alpha_np = alpha.detach().cpu().numpy()  # (E, H)
                ei_np = ei.cpu().numpy().T.tolist()       # List of [src, dst]
                all_weights[layer_key][key] = {
                    "edges":      ei_np,
                    "heads":      alpha_np.tolist(),
                    "mean_alpha": alpha_np.mean(axis=-1).tolist(),
                }

        if isinstance(h_new, dict):
            for nt in h_new:
                if h_new[nt] is not None:
                    h[nt] = h_new[nt]

    return all_weights


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_scenario(scenario: str) -> Tuple[Any, Dict, Dict, Dict]:
    """Load graph + metrics from cache or raw scenario JSON."""
    from cli.loso_evaluate import _build_graph_from_json

    json_path = _SCENARIOS / f"{scenario}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Scenario JSON not found: {json_path}")
    topology = json.loads(json_path.read_text())
    g = _build_graph_from_json(topology)

    cache = _LOSO_CACHE / scenario
    struct, sim, rmav = {}, {}, {}
    if cache.exists():
        for fname, d in [("structural_metrics.json", struct),
                         ("failure_impact.json", sim),
                         ("quality_scores.json", rmav)]:
            p = cache / fname
            if p.exists():
                d.update(json.loads(p.read_text()))

    return g, struct, sim, rmav


# ── Main extraction ───────────────────────────────────────────────────────────

def run_extraction(
    scenario: str,
    checkpoint_dir: Optional[Path],
    output_dir: Path,
    seed: int = 42,
    hidden: int = 64,
    num_heads: int = 4,
    num_layers: int = 3,
    dropout: float = 0.2,
    num_epochs: int = 100,
) -> Path:
    import torch
    from saag.prediction.data_preparation import networkx_to_hetero_data, create_node_splits
    from saag.prediction.models.core import build_node_gnn

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"  Loading scenario: {scenario}")
    g, struct, sim, rmav = _load_scenario(scenario)

    conv = networkx_to_hetero_data(g, struct, sim, rmav)
    data = conv.hetero_data
    create_node_splits(data, seed=seed)

    x_dict  = {nt: data[nt].x for nt in data.node_types if hasattr(data[nt], "x")}
    ei_dict = {rel: data[rel].edge_index for rel in data.edge_types}
    ea_dict = {rel: data[rel].edge_attr for rel in data.edge_types
               if hasattr(data[rel], "edge_attr")}

    # Build or load model
    ckpt = None
    if checkpoint_dir and (checkpoint_dir / "best_model.pt").exists():
        print(f"  Loading checkpoint: {checkpoint_dir}")
        model = build_node_gnn(data.metadata(), hidden_channels=hidden,
                               num_heads=num_heads, num_layers=num_layers, dropout=dropout)
        model.load_state_dict(torch.load(checkpoint_dir / "best_model.pt", map_location="cpu"))
    else:
        print(f"  No checkpoint found — training {num_epochs} epochs on {scenario}")
        from saag.prediction.trainer import GNNTrainer
        model = build_node_gnn(data.metadata(), hidden_channels=hidden,
                               num_heads=num_heads, num_layers=num_layers, dropout=dropout)
        ckpt_dir = output_dir / "checkpoint"
        trainer = GNNTrainer(model=model, checkpoint_dir=str(ckpt_dir),
                             lr=3e-4, num_epochs=num_epochs, patience=20)
        trainer.train(data)

    model.eval()
    print("  Extracting attention weights ...")

    with torch.no_grad():
        attn_data = _extract_via_return_attention_weights(model, x_dict, ei_dict, ea_dict)

    # Attach node ID mapping for interpretability
    node_id_map = conv.node_id_map  # {type: [nid, ...]}

    output: Dict[str, Any] = {
        "scenario": scenario,
        "seed": seed,
        "model_config": {
            "hidden_channels": hidden, "num_heads": num_heads,
            "num_layers": num_layers, "dropout": dropout,
        },
        "node_id_map": {nt: list(ids) for nt, ids in node_id_map.items()},
        "attention_by_layer": attn_data,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "attention_weights.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  Saved: {out_path}")
    return out_path


# ── Console summary ───────────────────────────────────────────────────────────

def _print_attention_summary(attn_path: Path):
    data = json.loads(attn_path.read_text())
    attn = data.get("attention_by_layer", {})
    node_ids = data.get("node_id_map", {})

    print(f"\n  Attention Summary — {data.get('scenario', '?')}")
    for layer_key in sorted(attn):
        print(f"  {layer_key}:")
        rels = attn[layer_key]
        if not rels:
            print("    (no attention captured — HGTConv version may not expose alpha)")
            continue
        for rel_key, info in rels.items():
            edges = info.get("edges", [])
            means = info.get("mean_alpha", [])
            top_n = sorted(zip(means, edges), reverse=True)[:5]
            print(f"    {rel_key}  (E={len(edges)} edges)")
            for alpha, (src, dst) in top_n:
                src_t = rel_key.split("__")[0]
                dst_t = rel_key.split("__")[2]
                src_name = (node_ids.get(src_t, []) + ["?"])[src] if isinstance(src, int) else "?"
                dst_name = (node_ids.get(dst_t, []) + ["?"])[dst] if isinstance(dst, int) else "?"
                print(f"      α={alpha:.4f}  {src_name} → {dst_name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Block G: Extract HGT attention weights.")
    p.add_argument("--scenario", default="atm_system",
                   help="Scenario name (default: atm_system)")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to checkpoint dir with best_model.pt")
    p.add_argument("--output-dir", type=Path, default=_RESULTS_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=100,
                   help="Epochs to train if no checkpoint (default: 100)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n  Block G — ATM Case Study: Attention Extraction")
    out_path = run_extraction(
        scenario=args.scenario,
        checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir,
        seed=args.seed,
        hidden=args.hidden,
        num_heads=args.heads,
        num_layers=args.layers,
        num_epochs=args.epochs,
    )
    _print_attention_summary(out_path)
    print("\n  Done. Run: python tools/render_attention_subgraph.py")


if __name__ == "__main__":
    main()
