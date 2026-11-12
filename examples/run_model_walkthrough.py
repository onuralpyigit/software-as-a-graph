#!/usr/bin/env python3
"""
Example script that walks through Step 1, Model (docs/graph-model.md), on the
15-component mini-ATM topology and builds an interactive HTML page.

It shows how a topic's weight follows from its QoS contract, size and rate,
how components inherit weight from the topics they touch, how the derivation
rules turn raw pub-sub structure into DEPENDS_ON edges and weight them, and
what each of the four layer projections keeps.

The repository returns finished weights and edges only, so the script retraces
every one of them with the public helper functions in saag/core/models.py and
checks the result against what the repository stored.
"""

import argparse
import collections
import json
import logging
import math
import sys
import warnings
from pathlib import Path

# Add project root to sys.path to support direct execution
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from saag import Client
from saag.core.models import (
    LIB_FANOUT_GAMMA, TOPIC_FREQ_WEIGHT_PSI, TOPIC_QOS_WEIGHT_BETA, TOPIC_SIZE_WEIGHT_ALPHA, QoSPolicy,
    compute_effective_edge_weight, compute_freq_norm, compute_harmonic_coupling, compute_lifted_edge_weight,
    compute_power_mean_weight, compute_size_norm, compute_topic_weight)
from saag.infrastructure.memory_repo import MemoryRepository

from walkthrough_page import write_page

SCRIPT_DIR = Path(__file__).parent
TOPOLOGY = SCRIPT_DIR / "mini_atm.json"
TEMPLATE = SCRIPT_DIR / "model_walkthrough.template.html"
LAYERS = ("app", "infra", "mw", "system")
TOLERANCE = 1e-3


def r4(x):
    return None if x is None else round(float(x), 4)


def pairs(topology: dict, key: str) -> list:
    return [(r["from"], r["to"]) for r in topology["relationships"][key]]


def topic_weights(topology: dict, stored: dict) -> dict:
    """w(t) term by term."""
    out = {}
    for t in topology["topics"]:
        qos = QoSPolicy(**{k: t["qos"][k] for k in ("durability", "reliability", "transport_priority")})
        scores = {
            "reliability": [t["qos"]["reliability"], QoSPolicy.RELIABILITY_SCORES[t["qos"]["reliability"]], QoSPolicy.W_RELIABILITY],
            "durability": [t["qos"]["durability"], QoSPolicy.DURABILITY_SCORES[t["qos"]["durability"]], QoSPolicy.W_DURABILITY],
            "priority": [t["qos"]["transport_priority"], QoSPolicy.PRIORITY_SCORES[t["qos"]["transport_priority"]], QoSPolicy.W_PRIORITY],
        }
        weight = compute_topic_weight(qos, t["size"], t["frequency"])
        assert abs(weight - stored[t["id"]]) < TOLERANCE, (t["id"], weight, stored[t["id"]])
        out[t["id"]] = {"qos": scores, "qos_score": r4(qos.calculate_weight()), "size": t["size"],
                        "size_norm": r4(compute_size_norm(t["size"])), "frequency": t["frequency"],
                        "freq_norm": r4(compute_freq_norm(t["frequency"])), "weight": r4(weight)}
    return out


def component_weights(topology: dict, stored: dict) -> dict:
    """w(v) for applications, brokers, the library and hosts, with the inputs each one aggregates."""
    touches = collections.defaultdict(list)
    for a, t in pairs(topology, "publishes_to") + pairs(topology, "subscribes_to"):
        touches[a].append(t)
    routes = collections.defaultdict(list)
    for b, t in pairs(topology, "routes"):
        routes[b].append(t)
    users = collections.defaultdict(list)
    for a, lib in pairs(topology, "uses"):
        users[lib].append(a)
    hosted = collections.defaultdict(list)
    for c, n in pairs(topology, "runs_on"):
        hosted[n].append(c)

    out = {}

    def record(cid, rule, inputs, weight, note=""):
        assert abs(weight - stored[cid]) < TOLERANCE, (cid, weight, stored[cid])
        out[cid] = {"rule": rule, "inputs": [[i, r4(stored[i])] for i in inputs], "weight": r4(weight), "note": note}

    for a in (x["id"] for x in topology["applications"]):
        record(a, "power_mean", touches[a], compute_power_mean_weight([stored[t] for t in touches[a]]))
    for b in (x["id"] for x in topology["brokers"]):
        record(b, "power_mean", routes[b], compute_power_mean_weight([stored[t] for t in routes[b]]))
    for lib in (x["id"] for x in topology["libraries"]):
        base, fan = max(stored[a] for a in users[lib]), len(users[lib])
        factor = 1.0 + LIB_FANOUT_GAMMA * math.log2(1 + fan)
        record(lib, "library", users[lib], min(1.0, base * factor),
               f"{base:.4f} x (1 + {LIB_FANOUT_GAMMA} x log2(1 + {fan})) = {base:.4f} x {factor:.4f}")
    for n in (x["id"] for x in topology["nodes"]):
        record(n, "max", hosted[n], max(stored[c] for c in hosted[n]))
    return out


def derived_edges(topology: dict, stored: dict, edges: list) -> list:
    """Each DEPENDS_ON edge with the structure it was derived from and its weight recomputed."""
    pubs, subs, routed, host = (collections.defaultdict(set) for _ in range(4))
    for a, t in pairs(topology, "publishes_to"):
        pubs[a].add(t)
    for a, t in pairs(topology, "subscribes_to"):
        subs[a].add(t)
    for b, t in pairs(topology, "routes"):
        routed[b].add(t)
    host = dict(pairs(topology, "runs_on"))
    weight_of = {(e.source_id, e.target_id): e.weight for e in edges}
    logical = {k: w for k, w in weight_of.items() if host.get(k[0]) and host.get(k[1])}   # app and broker edges

    out = []
    for e in sorted(edges, key=lambda e: (e.dependency_type, e.source_id, e.target_id)):
        u, v, kind = e.source_id, e.target_id, e.dependency_type
        if kind == "app_to_app":
            via = sorted(subs[u] & pubs[v])
            weight, how = compute_effective_edge_weight([stored[t] for t in via]), "union"
        elif kind == "app_to_broker":
            via = sorted((pubs[u] | subs[u]) & routed[v])
            weight, how = compute_effective_edge_weight([stored[t] for t in via]), "union"
        elif kind == "app_to_lib":
            via = [u, v]
            weight, how = compute_harmonic_coupling(stored[u], stored[v]), "harmonic"
        elif kind == "node_to_node":
            lifted = {k: w for k, w in logical.items() if host[k[0]] == u and host[k[1]] == v}
            via = [f"{a}>{b}" for a, b in sorted(lifted)]
            weight, how = compute_lifted_edge_weight(list(lifted.values())), "max"
        elif kind == "node_to_broker":
            lifted = {k: w for k, w in logical.items() if host[k[0]] == u and k[1] == v}
            via = [f"{a}>{b}" for a, b in sorted(lifted)]
            weight, how = compute_lifted_edge_weight(list(lifted.values())), "max"
        else:
            raise ValueError(f"no retrace for rule {kind}")
        assert abs(weight - e.weight) < TOLERANCE, (u, v, kind, weight, e.weight)
        terms = ([[t, r4(stored[t])] for t in via] if how != "max"
                 else [[k, r4(weight_of[tuple(k.split(">"))])] for k in via])
        out.append({"source": u, "target": v, "rule": kind, "how": how, "terms": terms, "weight": r4(e.weight)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "model_walkthrough.html",
                        help="Where to write the HTML walkthrough")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    topology = json.loads(TOPOLOGY.read_text())
    repo = MemoryRepository()
    repo.save_graph(topology, clear=True)
    repo.derive_dependencies()
    graph = repo.get_graph_data(include_raw=True)
    stored = {c.id: c.weight for c in graph.components}
    dependencies = [e for e in graph.edges if e.relation_type == "DEPENDS_ON"]
    raw = collections.Counter(e.relation_type for e in graph.edges if e.relation_type != "DEPENDS_ON")

    client = Client(repo=repo)
    layers = {}
    for layer in LAYERS:
        structural = client.analyze(layer=layer).raw.structural
        layers[layer] = {"components": dict(collections.Counter(m.type for m in structural.components.values())),
                         "ids": sorted(structural.components),
                         "edges": dict(collections.Counter(e.dependency_type for e in structural.edges.values()))}

    data = {
        "constants": {"beta": TOPIC_QOS_WEIGHT_BETA, "alpha": TOPIC_SIZE_WEIGHT_ALPHA, "psi": TOPIC_FREQ_WEIGHT_PSI,
                      "gamma": LIB_FANOUT_GAMMA},
        "types": {c.id: c.component_type for c in graph.components},
        "raw_counts": dict(raw),
        "topics": topic_weights(topology, stored),
        "components": component_weights(topology, stored),
        "edges": derived_edges(topology, stored, dependencies),
        "layers": layers,
    }
    by_rule = collections.Counter(e["rule"] for e in data["edges"])
    print(f"{sum(raw.values())} raw edges -> {len(data['edges'])} derived DEPENDS_ON edges: "
          + ", ".join(f"{n} {k}" for k, n in sorted(by_rule.items())))
    print("Every topic weight, component weight and edge weight was recomputed and matches the repository.")

    write_page(TEMPLATE, args.output, data)
    print(f"Walkthrough written to {args.output}")


if __name__ == "__main__":
    main()
