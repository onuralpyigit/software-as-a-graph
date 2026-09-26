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

``--substrate projection`` runs both on the DEPENDS_ON projection arms of
PREREGISTRATION.md Amendment 9 instead. There the gradient probe also reports how
often an Application's receptive field is exactly {v} plus its dependents within
three hops, which is what a 3-layer source-to-target GAT should see.

Usage:
    PYTHONPATH=. python reproduce/receptive_field_probe.py
    PYTHONPATH=. python reproduce/receptive_field_probe.py --substrate projection \
        --output results/receptive_field_probe_dependency_graph.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import networkx as nx
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
    "gl_full_qos_cap": "output/realworld_zeroshot_capacity/gl_full_qos_cap/seed_42/best_model.pt",
}
#: Amendment 9's arms, same convention (seed 42 of the zero-shot run).
CHECKPOINTS_PROJECTION = {
    v: f"output/realworld_zeroshot_dependency_graph/{v}/seed_42/best_model.pt"
    for v in ("gl_proj_cap", "gl_proj_qos16_cap", "gl_proj_qos16_indeg_prior", "hgl_proj_qos")
}
CHECK_SCENARIOS = ("enterprise_system", "healthcare_system")


def _inputs(data):
    x = {nt: data[nt].x for nt in data.node_types}
    ei = {r: data[r].edge_index for r in data.edge_types}
    ea = {r: data[r].edge_attr for r in data.edge_types if "edge_attr" in data[r]}
    return x, ei, ea


def _dependents_within(graph: nx.DiGraph, node, hops: int) -> set:
    """``node`` plus every node with a path of at most ``hops`` edges into it."""
    seen, frontier = {node}, {node}
    for _ in range(hops):
        frontier = {u for v in frontier for u in graph.predecessors(v)} - seen
        seen |= frontier
    return seen


def gradient_probe(model, data, expected=None) -> Dict[str, float]:
    """Mean receptive field of an Application output, in nodes and as a share of the graph.

    ``expected`` (optional): per Application, the set of ``(node_type, index)``
    it should see; the share of Applications whose receptive field equals it
    exactly is then reported too.
    """
    model.eval()
    x, ei, ea = _inputs(data)
    x = {nt: v.clone().requires_grad_(True) for nt, v in x.items()}
    ea = {r: v.clone().requires_grad_(True) for r, v in ea.items()}
    y = model(x, ei, ea)["Application"][:, 0]
    leaves = list(x.values()) + list(ea.values())
    n_total = sum(v.shape[0] for v in x.values())
    sizes, other_apps, edges, exact = [], [], [], []
    for i in range(y.shape[0]):
        grads = torch.autograd.grad(y[i], leaves, retain_graph=True, allow_unused=True)
        seen = {nt: (g.abs().sum(1) > 0) if g is not None else torch.zeros(x[nt].shape[0], dtype=torch.bool)
                for nt, g in zip(x, grads[:len(x)])}
        sizes.append(sum(int(s.sum()) for s in seen.values()) or 1)
        other_apps.append(int(seen["Application"].sum()) - int(seen["Application"][i]))
        edges.append(sum(int((g.abs().sum(1) > 0).sum()) for g in grads[len(x):] if g is not None))
        if expected is not None:
            got = {(nt, int(j)) for nt, s in seen.items() for j in s.nonzero().flatten()}
            exact.append(got == expected[i])
    n = len(sizes)
    out = {
        "mean_rf_nodes": sum(sizes) / n,
        "mean_rf_share": sum(sizes) / n / n_total,
        "share_of_apps_seeing_other_apps": sum(a > 0 for a in other_apps) / n,
        "mean_edges_with_gradient": sum(edges) / n,
    }
    if expected is not None:
        out["share_rf_equals_3hop_dependents"] = sum(exact) / n
    return out


def _expected_3hop(bundle) -> list:
    """Per Application (in tensor order), its 3-hop dependent set as (type, index)."""
    from saag.prediction.data_preparation import networkx_to_hetero_data

    conv = networkx_to_hetero_data(bundle.graph, bundle.structural)
    index = conv.node_name_to_idx
    return [
        {index[u] for u in _dependents_within(bundle.graph, app, 3)}
        for app in conv.node_id_map["Application"]
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=Path, default=Path("output/loso_cache"))
    p.add_argument("--output", type=Path, default=Path("results/receptive_field_probe.json"))
    p.add_argument("--substrate", choices=["native", "projection"], default="native",
                   help="projection: probe Amendment 9's DEPENDS_ON-projection arms")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    from cli.loso_evaluate import (
        _build_training_hetero, _dependency_bundle, _graft_qos_edge_attr, load_scenario_bundle,
    )
    from saag.evaluation import variant_registry as registry
    from saag.prediction.models.baselines import build_baseline
    from saag.prediction.models.core import build_node_gnn

    projection = args.substrate == "projection"
    load = (lambda d: _dependency_bundle(load_scenario_bundle(d))) if projection else load_scenario_bundle

    torch.manual_seed(0)
    probe: Dict[str, Dict] = {}
    for sd in sorted(p for p in args.cache_dir.iterdir() if p.is_dir()):
        bundle = load(sd)
        if bundle is None:
            continue
        data = _build_training_hetero(bundle, True, False)
        hgt, gat = ("hgl_proj_qos", "gl_proj_qos16_cap") if projection else ("hgl_qos", "gl_full_qos16_cap")
        models = {
            hgt: build_node_gnn(data.metadata(), 32, 4, 3, 0.0, use_bidirectional=True),
            gat: build_baseline("homo_scalar", hidden_channels=32, num_heads=4,
                                num_layers=3, dropout=0.0, edge_dim=16),
        }
        if not projection:
            models["hgl_qos_uni"] = build_node_gnn(data.metadata(), 32, 4, 3, 0.0,
                                                   use_bidirectional=False)
        expected = _expected_3hop(bundle) if projection else None
        probe[sd.name] = {
            "n_nodes": sum(data[nt].num_nodes for nt in data.node_types),
            "n_app": data["Application"].num_nodes,
            **{v: gradient_probe(m, data, expected if v == gat else None) for v, m in models.items()},
        }
        print(f"{sd.name:28s} HGT share {probe[sd.name][hgt]['mean_rf_share']:.3f}   "
              f"GAT nodes {probe[sd.name][gat]['mean_rf_nodes']:.1f}"
              + (f"   GAT = 3-hop dependents {probe[sd.name][gat]['share_rf_equals_3hop_dependents']:.3f}"
                 if projection else ""))

    deletion: Dict[str, Dict[str, float]] = {}
    for variant, ckpt in (CHECKPOINTS_PROJECTION if projection else CHECKPOINTS).items():
        if not Path(ckpt).exists():
            deletion[variant] = {"skipped": f"{ckpt} not found"}
            continue
        edge_dim = registry.edge_dim(variant, "loso")
        prior = registry.prior_for(variant) or False
        is_hgt = variant == "hgl_proj_qos"
        if not is_hgt:
            model = build_baseline(
                "homo_unweighted" if edge_dim is None else "homo_scalar",
                hidden_channels=registry.hidden_for(variant, 64, "loso"), num_heads=4,
                num_layers=3, dropout=0.2, edge_dim=edge_dim, topo_prior=bool(prior),
            )
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state.get("model_state_dict", state))
            model.eval()
        deletion[variant] = {}
        for scen in CHECK_SCENARIOS:
            bundle = load(args.cache_dir / scen)
            # Inputs as the zero-shot harness builds them (rank-normalised features).
            node_qos = registry.node_qos_for(variant, "loso")
            data = _build_training_hetero(bundle, node_qos, True, prior)
            if edge_dim is not None and not node_qos:
                _graft_qos_edge_attr(data, bundle, True)
            if is_hgt:
                model = build_node_gnn(data.metadata(), registry.hidden_for(variant, 64, "loso"),
                                       4, 3, 0.2, use_bidirectional=True)
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                model.eval()
            x, ei, ea = _inputs(data)
            with torch.no_grad():
                full = model(x, ei, ea)["Application"][:, 0]
                bare = model(x, {r: e[:, :0] for r, e in ei.items()},
                             {r: e[:0] for r, e in ea.items()})["Application"][:, 0]
            deletion[variant][scen] = float((full - bare).abs().max())
        print(f"{variant:24s} max |Δ| with every edge deleted: {max(deletion[variant].values()):.1e}")

    args.output.write_text(json.dumps({
        "provenance": stamp(cache_dir=str(args.cache_dir), probe_hidden=32, seed=0,
                            **({"substrate": "projection"} if projection else {})),
        "gradient_probe": probe,
        "edge_deletion_max_abs_delta": deletion,
    }, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
