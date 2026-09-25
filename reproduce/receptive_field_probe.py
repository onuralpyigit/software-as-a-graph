#!/usr/bin/env python3
"""
reproduce/receptive_field_probe.py — what each learned engine can see at an Application
=======================================================================================

On the native multigraph every relation points away from Application
(Application -> Topic / Node / Library), and the untyped GATs aggregate along edge
direction only, so an Application receives no messages in them. HGT reaches
Applications through its reverse-direction pass. This script measures both facts.

1. **Gradient probe** (every LOSO scenario). Each Application's composite output
   is back-propagated to every node-feature row and edge attribute; the inputs
   with non-zero gradient are its receptive field. Receptive field is a property
   of architecture and graph, not of the weights, so random initialisation is
   sufficient.
2. **Edge-deletion check** (trained checkpoints). Deletes every edge and reports
   the largest change in any Application prediction. Zero means the trained
   model's Application scores do not depend on the graph.

Usage:
    PYTHONPATH=. python reproduce/receptive_field_probe.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import torch

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce._provenance import stamp

logger = logging.getLogger(__name__)

#: variant id -> checkpoint written by reproduce/realworld_zeroshot.py (seed 42).
CHECKPOINTS = {
    "gl_full_cap": "output/realworld_zeroshot_attribution/gl_full_cap/seed_42/best_model.pt",
    "gl_full_qos16_nfmask": "output/realworld_zeroshot_attribution/gl_full_qos16_nfmask/seed_42/best_model.pt",
    "gl_full_qos16_cap": "output/realworld_zeroshot_attribution/gl_full_qos16_cap/seed_42/best_model.pt",
    "gl_qos16_prior": "output/realworld_zeroshot_cpu_hybrid_gat/gl_qos16_prior/seed_42/best_model.pt",
}
CHECK_SCENARIOS = ("enterprise_system", "healthcare_system")


def _inputs(data):
    x = {nt: data[nt].x for nt in data.node_types}
    ei = {r: data[r].edge_index for r in data.edge_types}
    ea = {r: data[r].edge_attr for r in data.edge_types if "edge_attr" in data[r]}
    return x, ei, ea


def gradient_probe(model, data) -> Dict[str, float]:
    """Mean receptive field of an Application output, in nodes and as a share of the graph."""
    model.eval()
    x, ei, ea = _inputs(data)
    x = {nt: v.clone().requires_grad_(True) for nt, v in x.items()}
    ea = {r: v.clone().requires_grad_(True) for r, v in ea.items()}
    y = model(x, ei, ea)["Application"][:, 0]
    leaves = list(x.values()) + list(ea.values())
    n_total = sum(v.shape[0] for v in x.values())
    sizes, other_apps, edges = [], [], []
    for i in range(y.shape[0]):
        grads = torch.autograd.grad(y[i], leaves, retain_graph=True, allow_unused=True)
        seen = {nt: (g.abs().sum(1) > 0) if g is not None else torch.zeros(x[nt].shape[0], dtype=torch.bool)
                for nt, g in zip(x, grads[:len(x)])}
        sizes.append(sum(int(s.sum()) for s in seen.values()) or 1)
        other_apps.append(int(seen["Application"].sum()) - int(seen["Application"][i]))
        edges.append(sum(int((g.abs().sum(1) > 0).sum()) for g in grads[len(x):] if g is not None))
    n = len(sizes)
    return {
        "mean_rf_nodes": sum(sizes) / n,
        "mean_rf_share": sum(sizes) / n / n_total,
        "share_of_apps_seeing_other_apps": sum(a > 0 for a in other_apps) / n,
        "mean_edges_with_gradient": sum(edges) / n,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=Path, default=Path("output/loso_cache"))
    p.add_argument("--output", type=Path, default=Path("results/receptive_field_probe.json"))
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    from cli.loso_evaluate import _build_training_hetero, _graft_qos_edge_attr, load_scenario_bundle
    from saag.evaluation import variant_registry as registry
    from saag.prediction.models.baselines import build_baseline
    from saag.prediction.models.core import build_node_gnn

    torch.manual_seed(0)
    probe: Dict[str, Dict] = {}
    for sd in sorted(p for p in args.cache_dir.iterdir() if p.is_dir()):
        bundle = load_scenario_bundle(sd)
        if bundle is None:
            continue
        data = _build_training_hetero(bundle, True, False)
        models = {
            "hgl_qos": build_node_gnn(data.metadata(), 32, 4, 3, 0.0, use_bidirectional=True),
            "hgl_qos_uni": build_node_gnn(data.metadata(), 32, 4, 3, 0.0, use_bidirectional=False),
            "gl_full_qos16_cap": build_baseline("homo_scalar", hidden_channels=32, num_heads=4,
                                                num_layers=3, dropout=0.0, edge_dim=16),
        }
        probe[sd.name] = {
            "n_nodes": sum(data[nt].num_nodes for nt in data.node_types),
            "n_app": data["Application"].num_nodes,
            **{v: gradient_probe(m, data) for v, m in models.items()},
        }
        print(f"{sd.name:28s} HGT-QoS share {probe[sd.name]['hgl_qos']['mean_rf_share']:.3f}   "
              f"GAT-QoS nodes {probe[sd.name]['gl_full_qos16_cap']['mean_rf_nodes']:.1f}")

    deletion: Dict[str, Dict[str, float]] = {}
    for variant, ckpt in CHECKPOINTS.items():
        if not Path(ckpt).exists():
            deletion[variant] = {"skipped": f"{ckpt} not found"}
            continue
        edge_dim = registry.edge_dim(variant, "loso")
        prior = registry.VARIANTS[variant].family == "hybrid"
        model = build_baseline(
            "homo_unweighted" if edge_dim is None else "homo_scalar",
            hidden_channels=registry.hidden_for(variant, 64, "loso"), num_heads=4,
            num_layers=3, dropout=0.2, edge_dim=edge_dim, topo_prior=prior,
        )
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state))
        model.eval()
        deletion[variant] = {}
        for scen in CHECK_SCENARIOS:
            bundle = load_scenario_bundle(args.cache_dir / scen)
            # Inputs as the zero-shot harness builds them (rank-normalised features).
            node_qos = registry.node_qos_for(variant, "loso")
            data = _build_training_hetero(bundle, node_qos, True, prior)
            if edge_dim is not None and not node_qos:
                _graft_qos_edge_attr(data, bundle, True)
            x, ei, ea = _inputs(data)
            with torch.no_grad():
                full = model(x, ei, ea)["Application"][:, 0]
                bare = model(x, {r: e[:, :0] for r, e in ei.items()},
                             {r: e[:0] for r, e in ea.items()})["Application"][:, 0]
            deletion[variant][scen] = float((full - bare).abs().max())
        print(f"{variant:24s} max |Δ| with every edge deleted: {max(deletion[variant].values()):.1e}")

    args.output.write_text(json.dumps({
        "provenance": stamp(cache_dir=str(args.cache_dir), probe_hidden=32, seed=0),
        "gradient_probe": probe,
        "edge_deletion_max_abs_delta": deletion,
    }, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
