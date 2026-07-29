"""Graph loading and tabular output for the validation CLI."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import networkx as nx
import numpy as np

from .scoring import NodeScores


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (int, float, bool)): return obj
        import numpy as np
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        return super(NpEncoder, self).default(obj)


def load_graph(path: str) -> Tuple[nx.DiGraph, dict]:
    """
    Load a SaG system.json / dataset.json and build a typed DiGraph.

    Returns (G, raw_data).
    G nodes carry attribute 'ntype' ∈ {Application, Broker, Topic,
    InfraNode, Library} and 'label'.
    """
    raw = json.loads(Path(path).read_text())
    G = nx.DiGraph()

    # ── nodes ──────────────────────────────────────────────────────────────────
    def _add(collection, ntype):
        for item in raw.get(collection, []):
            nid = item.get("id") or item.get("name")
            model_type = "Node" if ntype == "InfraNode" else ntype
            G.add_node(nid, ntype=ntype, type=model_type, label=item.get("name", nid), raw=item)

    _add("applications", "Application")
    _add("brokers",      "Broker")
    _add("topics",       "Topic")
    _add("nodes",        "InfraNode")
    _add("libraries",    "Library")

    # ── edges ──────────────────────────────────────────────────────────────────
    rels = raw.get("relationships", {})

    def _edges(key, src_field, tgt_field, etype):
        # Support both root-level and relationships-level keys
        # Support both singular and plural (publishes vs publishes_to)
        items = raw.get(key, []) + rels.get(key, [])
        if not items and "_" not in key:
            items += raw.get(f"{key}_to", []) + rels.get(f"{key}_to", [])
        
        for e in items:
            s = e.get(src_field) or e.get("from") or e.get("app") or e.get("src")
            t = e.get(tgt_field) or e.get("to") or e.get("topic") or e.get("tgt")
            if s and t and G.has_node(s) and G.has_node(t):
                G.add_edge(s, t, etype=etype, type=etype)

    _edges("publishes",    "app", "topic", "PUBLISHES_TO")
    _edges("subscribes",   "app", "topic", "SUBSCRIBES_TO")
    _edges("routes",       "from", "to",   "ROUTES")
    _edges("runs_on",      "from", "to",   "RUNS_ON")
    _edges("uses",         "from", "to",   "USES")
    # Legacy support
    _edges("publish_edges",   "app", "topic", "PUBLISHES_TO")
    _edges("subscribe_edges", "app", "topic", "SUBSCRIBES_TO")
    _edges("broker_connections", "broker", "topic", "ROUTES")

    # ── derive DEPENDS_ON (app_to_app via shared topics) ──────────────────────
    pub_map: Dict[str, List[str]] = defaultdict(list)   # topic → publishers
    sub_map: Dict[str, List[str]] = defaultdict(list)   # topic → subscribers

    for u, v, d in G.edges(data=True):
        if d["etype"] == "PUBLISHES_TO":
            pub_map[v].append(u)
        elif d["etype"] == "SUBSCRIBES_TO":
            sub_map[v].append(u)

    for topic, pubs in pub_map.items():
        for sub in sub_map.get(topic, []):
            for pub in pubs:
                if pub != sub and not G.has_edge(sub, pub):
                    G.add_edge(sub, pub, etype="DEPENDS_ON", type="DEPENDS_ON", via=topic)

    return G, raw


def write_csv(node_scores: Dict[str, NodeScores], path: str):
    import csv
    rows = sorted(node_scores.values(), key=lambda ns: ns.Q, reverse=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "node_id", "node_type", "Q", "R", "M", "A", "S",
                    "I", "cascade_depth", "nodes_affected",
                    "is_articulation_point", "degree_centrality"])
        for rank, ns in enumerate(rows, 1):
            w.writerow([
                rank, ns.node_id, ns.node_type,
                round(ns.Q, 5), round(ns.R, 5), round(ns.M, 5),
                round(ns.A, 5), round(ns.S, 5),
                round(ns.I, 5), ns.cascade_depth, ns.nodes_affected,
                ns.is_articulation_point, round(ns.degree_centrality, 5),
            ])
