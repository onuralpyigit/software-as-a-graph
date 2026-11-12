#!/usr/bin/env python3
"""
Example script that walks through Step 3, Predict (docs/prediction.md), on the
15-component mini-ATM topology and builds an interactive HTML page.

It shows the two inputs a ranking is made from (the dependency projection and
the model's feature tensor), the closed-form references term by term, the
trained HGT-QoS model across its five seeds, the dual-engine consensus and
divergence sets, the cold-start fallback, and finally how each ranking compares
with the simulated impact it is meant to forecast.

The trained checkpoints are read from the git-ignored ``output/`` folder.
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

# Add project root to sys.path to support direct execution
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

from saag import Client
from saag.core.graph_io import load_graph
from saag.infrastructure.memory_repo import MemoryRepository
from saag.prediction.data_preparation import (
    KEYS_BY_TYPE, extract_rm_scores_dict, extract_structural_metrics_dict, networkx_to_hetero_data)
from saag.prediction.service import PredictionService
from saag.prediction.structural_predictor import derive_flow_projection
from saag.simulation.fault_injector import RECOMMENDED_SEEDS, FaultInjector

from walkthrough_page import write_page

SCRIPT_DIR = Path(__file__).parent
TOPOLOGY = SCRIPT_DIR / "mini_atm.json"
TEMPLATE = SCRIPT_DIR / "predict_walkthrough.template.html"
CHECKPOINTS = ROOT / "output/realworld_zeroshot/hgl_qos"
SEEDS = list(RECOMMENDED_SEEDS)
TOP_K = 3


def r4(x):
    return None if x is None else round(float(x), 4)


def closed_form(topology: dict) -> dict:
    """The training-free references, with the terms each one is built from."""
    from reproduce.idyn_rate_expansion import closed_forms
    from reproduce.referee_round7 import raw_rankers

    rankers, forms = raw_rankers(topology), closed_forms(topology)
    rel, topics = topology["relationships"], {t["id"]: t for t in topology["topics"]}
    pubs, subs = {}, {}
    for r in rel["publishes_to"]:
        pubs.setdefault(r["to"], []).append(r["from"])
    for r in rel["subscribes_to"]:
        subs.setdefault(r["to"], []).append(r["from"])
    flow = derive_flow_projection(topology)
    apps = [a["id"] for a in topology["applications"]]
    terms = {a: [{"topic": t, "subs": len(subs.get(t, [])), "pubs": len(p), "rate": topics[t]["frequency"]}
                 for t, p in pubs.items() if a in p] for a in apps}
    return {
        "flow": [[u, v, r4(d.get("weight", d.get("qos_weight", 1.0)))] for u, v, d in flow.edges(data=True)],
        "dependents": {v: sorted(flow.predecessors(v)) for v in flow.nodes},
        "reach": {v: sorted(nx.ancestors(flow, v)) for v in flow.nodes},
        "terms": terms,
        "scores": {
            "InDeg": {a: r4(rankers["InDeg"].get(a, 0.0)) for a in flow.nodes},
            "Reach": {a: r4(rankers["Reach"].get(a, 0.0)) for a in flow.nodes},
            "Analytic-I*": {a: r4(rankers["Analytic-I*"].get(a, 0.0)) for a in apps},
            "Rate-weighted": {a: r4(forms["Rate-I_dyn"].get(a, 0.0)) for a in apps},
        },
    }


def model_inputs(client, analysis, rm_result) -> dict:
    """The feature rows the trained model reads for Applications, as it reads them (rank-normalised)."""
    conv = networkx_to_hetero_data(
        client._raw_structural_graph(), extract_structural_metrics_dict(analysis.raw.structural),
        None, extract_rm_scores_dict(rm_result.raw), rank_normalize_features=True)
    data, ids = conv.hetero_data, conv.node_id_map
    return {
        "node_counts": {t: len(v) for t, v in ids.items()},
        "feature_dims": {t: int(data[t].x.shape[1]) for t in data.node_types},
        "edge_types": {f"{s} {r} {d}": int(data[(s, r, d)].edge_index.shape[1]) for s, r, d in data.edge_types},
        "edge_dim": int(data[data.edge_types[0]].edge_attr.shape[1]),
        "app_keys": KEYS_BY_TYPE["Application"],
        "app_rows": {n: [r4(v) for v in data["Application"].x[i].tolist()] for i, n in enumerate(ids["Application"])},
    }


def trained(client, analysis) -> dict:
    """HGT-QoS through the SDK, one run per seed, with its three output heads."""
    runs = {}
    for seed in SEEDS:
        result = client.predict(analysis, mode="gnn", gnn_checkpoint=str(CHECKPOINTS / f"seed_{seed}"), diagnose=False)
        if result.raw.prediction_mode != "gnn_only":
            sys.exit(f"GNN inference did not run for seed {seed} (prediction_mode = {result.raw.prediction_mode}).")
        runs[seed] = {n: {"composite": r4(s.composite_score), "reliability": r4(s.reliability_score),
                          "maintainability": r4(s.maintainability_score)} for n, s in result.raw.node_scores.items()}
    ids = sorted(runs[SEEDS[0]])
    return {
        "seeds": SEEDS, "runs": runs,
        "mean": {n: r4(np.mean([runs[s][n]["composite"] for s in SEEDS])) for n in ids},
        "std": {n: r4(np.std([runs[s][n]["composite"] for s in SEEDS])) for n in ids},
    }


def dual_engine(client, analysis) -> dict:
    """Consensus and divergence between the trained model and Topo-QoS."""
    service = PredictionService(gnn_checkpoint_dir=str(CHECKPOINTS / f"seed_{SEEDS[0]}"), prefer_gnn=True)
    result = service.predict_dual(analysis.raw.structural, graph=client._raw_structural_graph(),
                                  layer="app", diagnose=False, k=TOP_K, divergence_threshold=2)
    dual = result.dual_result
    return {"k": TOP_K, "threshold": 2, "gnn_ranks": dual.gnn_ranks, "topo_ranks": dual.topo_ranks,
            "topo_scores": {k: r4(v) for k, v in dual.topo_scores.items()},
            "divergence": dual.rank_divergences, "consensus": dual.consensus_top_k,
            "escalations": dual.divergence_escalations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "predict_walkthrough.html",
                        help="Where to write the HTML walkthrough")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    if not CHECKPOINTS.exists():
        sys.exit(f"Trained checkpoints not found: {CHECKPOINTS.relative_to(ROOT)} (see reproduce/README.md).")

    topology = json.loads(TOPOLOGY.read_text())
    repo = MemoryRepository()
    repo.save_graph(topology, clear=True)
    repo.derive_dependencies()
    client = Client(repo=repo)
    analysis = client.analyze(layer="app")
    cold = client.predict(analysis, diagnose=False)          # no checkpoint passed: the RM fallback

    data = {
        "closed": closed_form(topology),
        "inputs": model_inputs(client, analysis, cold),
        "trained": trained(client, analysis),
        "dual": dual_engine(client, analysis),
        "cold": {"mode": cold.raw.prediction_mode, "scores": {c.id: r4(c.rm_score) for c in cold.all_components}},
    }
    labels = FaultInjector(load_graph(TOPOLOGY), seeds=SEEDS).run(node_types=["Application", "Library"]).records
    data["istar"] = {n: r4(rec.impact_score) for n, rec in labels.items()}

    apps = [a["id"] for a in topology["applications"]]
    rankings = {**data["closed"]["scores"], "HGT-QoS": data["trained"]["mean"], "RM Q(v)": data["cold"]["scores"]}
    data["agreement"] = {k: r4(spearmanr([v[a] for a in apps], [data["istar"][a] for a in apps])[0])
                         for k, v in rankings.items()}
    for name, rho in data["agreement"].items():
        print(f"  {name:<14} rho vs I* = {rho:+.3f}")

    write_page(TEMPLATE, args.output, data)
    print(f"\nWalkthrough written to {args.output}")


if __name__ == "__main__":
    main()
