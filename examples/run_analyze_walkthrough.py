#!/usr/bin/env python3
"""
Example script that walks through Step 2, Analyze (docs/structural-analysis.md),
on the 15-component mini-ATM topology and builds an interactive HTML page.

Analyze turns the dependency graph into one structural metric vector per
component and scores nothing. The page lets a reader pick a metric and see it
on the system, shows what each field is used for downstream (from
saag/core/metric_registry.py), and recomputes a few metrics independently with
NetworkX to show which are textbook quantities and which are the method's own.
"""

import argparse
import dataclasses
import inspect
import json
import logging
import re
import sys
import warnings
from pathlib import Path

# Add project root to sys.path to support direct execution
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import networkx as nx

from saag import Client
from saag.core.metric_registry import METRIC_ROLES
from saag.core.metrics import StructuralMetrics
from saag.infrastructure.memory_repo import MemoryRepository

from walkthrough_page import write_page

SCRIPT_DIR = Path(__file__).parent
TOPOLOGY = SCRIPT_DIR / "mini_atm.json"
TEMPLATE = SCRIPT_DIR / "analyze_walkthrough.template.html"
LAYERS = ("system", "app")


def r4(x):
    return None if x is None else round(float(x), 4)


def field_catalogue() -> list:
    """Every numeric metric field with its group and description, read from the dataclass source."""
    numeric = {f.name for f in dataclasses.fields(StructuralMetrics) if f.type in ("float", "int", float, int)}
    out, group = [], ""
    for line in inspect.getsource(StructuralMetrics).splitlines():
        header = re.match(r"\s*# === (.+?) ===", line)
        if header:
            group = re.sub(r"\s*\(.*\)", "", header.group(1)).replace(" Metrics", "")
            continue
        field = re.match(r"\s*(\w+): \w+ = [^#]+#\s*(.+)", line)
        if field and field.group(1) in numeric:
            name = field.group(1)
            out.append({"name": name, "group": group, "text": field.group(2).strip(),
                        "roles": sorted(r.value for r in METRIC_ROLES.get(name, ()))})
    return out


def layer_view(client, layer: str, fields: list) -> dict:
    structural = client.analyze(layer=layer).raw.structural
    summary = dataclasses.asdict(structural.graph_summary)
    return {
        "structural": structural,
        "values": {cid: {f["name"]: r4(getattr(m, f["name"])) for f in fields} for cid, m in structural.components.items()},
        "types": {cid: m.type for cid, m in structural.components.items()},
        "flags": {cid: {"articulation": bool(m.is_articulation_point), "directed_ap": bool(m.is_directed_ap),
                        "isolated": bool(m.is_isolated)} for cid, m in structural.components.items()},
        "edges": [[u, v, r4(e.weight)] for (u, v), e in structural.edges.items()],
        "summary": {k: r4(v) for k, v in summary.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
    }


def independent_checks(structural) -> list:
    """Recompute a few metrics from the layer's dependency graph with NetworkX alone."""
    graph = nx.DiGraph()
    graph.add_nodes_from(structural.components)
    for (u, v), e in structural.edges.items():
        graph.add_edge(u, v, weight=e.weight)
    ours = structural.components

    def gap(reference: dict, field: str) -> float:
        return max(abs(reference[n] - getattr(ours[n], field)) for n in ours)

    weight_in = {n: sum(d["weight"] for _, _, d in graph.in_edges(n, data=True)) for n in graph}
    weight_out = {n: sum(d["weight"] for _, _, d in graph.out_edges(n, data=True)) for n in graph}
    # Betweenness needs a distance: a heavier dependency is a shorter path.
    distance = graph.copy()
    for _, _, d in distance.edges(data=True):
        d["weight"] = 1.0 / (d["weight"] + 1e-6)
    checks = [
        ("in_degree_raw", "count of incoming dependency edges", dict(graph.in_degree()), None),
        ("out_degree_raw", "count of outgoing dependency edges", dict(graph.out_degree()), None),
        ("dependency_weight_in", "sum of incoming edge weights", weight_in, None),
        ("dependency_weight_out", "sum of outgoing edge weights", weight_out, None),
        ("clustering_coefficient", "local clustering on the undirected graph", nx.clustering(graph.to_undirected()), None),
        ("pagerank", "PageRank weighted by dependency weight", nx.pagerank(graph, alpha=0.85, weight="weight"),
         nx.pagerank(graph, alpha=0.85, weight=None)),
        ("reverse_pagerank", "weighted PageRank on the reversed graph",
         nx.pagerank(graph.reverse(), alpha=0.85, weight="weight"), nx.pagerank(graph.reverse(), alpha=0.85, weight=None)),
        ("betweenness", "betweenness with distance = 1 / dependency weight",
         nx.betweenness_centrality(distance, weight="weight"), nx.betweenness_centrality(graph)),
    ]
    out = []
    for metric, reference, values, unweighted in checks:
        g = gap(values, metric)
        assert g < 1e-3, (metric, g)
        out.append({"metric": metric, "reference": reference, "gap": r4(g),
                    "unweighted_gap": None if unweighted is None else r4(gap(unweighted, metric))})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "analyze_walkthrough.html",
                        help="Where to write the HTML walkthrough")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    repo = MemoryRepository()
    repo.save_graph(json.loads(TOPOLOGY.read_text()), clear=True)
    repo.derive_dependencies()
    client = Client(repo=repo)

    fields = field_catalogue()
    layers = {layer: layer_view(client, layer, fields) for layer in LAYERS}
    checks = independent_checks(layers["system"]["structural"])
    for view in layers.values():
        del view["structural"]

    system = layers["system"]["values"]
    for f in fields:
        column = [row[f["name"]] for row in system.values()]
        f["varies"] = len(set(column)) > 1
    data = {"fields": fields, "layers": layers, "checks": checks,
            "role_counts": {role: sum(role in f["roles"] for f in fields) for role in ("scoring", "detection", "gnn_feature", "descriptive")}}

    print(f"{len(fields)} numeric metric fields per component; {sum(f['varies'] for f in fields)} vary on this system")
    print("Roles: " + ", ".join(f"{k} {v}" for k, v in data["role_counts"].items()))
    for c in checks:
        print(f"  {c['metric']:<24} reproduced as {c['reference']}")

    write_page(TEMPLATE, args.output, data)
    print(f"\nWalkthrough written to {args.output}")


if __name__ == "__main__":
    main()
