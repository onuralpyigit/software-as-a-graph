#!/usr/bin/env python3
"""
reproduce/qos_corpus_diagnostic.py
==================================

Measures how much *within-graph* information the declared QoS contract actually
carries in each scenario of the LOSO corpus.

Motivation: the predictive pathway's architectural claim rests on per-dimension
QoS encoding — 7 of the 16 edge-feature dimensions (indices 9-15: reliability,
durability, priority, has_deadline, deadline_ns_log, max_blocking_ms_log,
qos_heterogeneity_flag; see the module docstring of
``saag/prediction/data_preparation.py``). A dimension that takes the *same
value on every edge of a graph* cannot discriminate between components of that
graph, no matter how the model attends to it. If that holds on most of the
corpus, an ablation of the QoS channel is averaging a treatment that is inert
where it is measured, and the resulting near-zero effect says nothing about
what QoS encoding does when QoS varies.

This script reports, per scenario:

  * **Declared-source entropy** — Shannon entropy (nats) of the ``qos.reliability``,
    ``qos.durability`` and ``qos.transport_priority`` values over Topic nodes,
    plus ``criticality``. H = 0 means every topic declares the identical value.
  * **As-encoded dispersion** — the standard deviation and distinct-value count
    of each QoS edge-feature column, computed on the ``edge_attr`` tensors that
    actually reach the model, restricted to the QoS-bearing edge types
    (``_QOS_EDGE_TYPES``). std = 0 means the column is a constant offset for
    that whole graph.

The two views are reported together because they can disagree: ``criticality``
stays non-degenerate even in scenarios whose QoS triple is constant, which is
why a QoS-weighted *structural* baseline can still score well on a graph where
the *edge* QoS channel is uniform.

Read-only. Loads cached scenario bundles, computes summary statistics, writes
JSON + Markdown. Does not modify any model, cache, or training code.

Usage:
    PYTHONPATH=. python reproduce/qos_corpus_diagnostic.py
    PYTHONPATH=. python reproduce/qos_corpus_diagnostic.py \\
        --cache-dir output/loso_cache --output results/qos_corpus_diagnostic.md
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.loso_evaluate import discover_scenarios  # noqa: E402
from saag.prediction.data_preparation import _QOS_EDGE_TYPES  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

#: Edge-feature column indices carrying declared QoS, with display names.
#: Index 0 is the aggregate scalar w(e) the homogeneous baseline receives;
#: 9-15 are the per-dimension decomposition only the typed model receives.
QOS_EDGE_COLUMNS: List[tuple] = [
    (0, "w(e) aggregate"),
    (9, "reliability_score"),
    (10, "durability_score"),
    (11, "priority_score"),
    (12, "has_deadline"),
    (13, "deadline_ns_log"),
    (14, "max_blocking_ms_log"),
    (15, "qos_heterogeneity_flag"),
]

#: Declared Topic fields whose entropy is reported.
DECLARED_FIELDS = ["reliability", "durability", "transport_priority"]


def _entropy(counts: Counter) -> float:
    """Shannon entropy in nats. 0.0 when every observation shares one value."""
    n = sum(counts.values())
    if n == 0:
        return float("nan")
    return -sum((v / n) * math.log(v / n) for v in counts.values() if v)


def _declared_stats(graph) -> Dict[str, Any]:
    """Entropy and value counts of the declared QoS fields over Topic nodes."""
    topics = [
        attrs for _, attrs in graph.nodes(data=True)
        if attrs.get("type") == "Topic"
    ]
    out: Dict[str, Any] = {"n_topics": len(topics)}
    for field in DECLARED_FIELDS:
        # QoS may be nested under `qos` or flattened onto the node, and the
        # corpus is not case-consistent (some topologies emit `RELIABLE`,
        # others `reliable`), so fold case before counting or the same
        # contract would register as two distinct values.
        vals = Counter(
            str(t.get("qos", {}).get(field, t.get(field, ""))).upper()
            for t in topics
        )
        out[field] = {"entropy": _entropy(vals), "values": dict(vals)}
    crit = Counter(str(t.get("criticality", "")).upper() for t in topics)
    out["criticality"] = {"entropy": _entropy(crit), "values": dict(crit)}
    out["qos_triple_entropy"] = sum(
        out[f]["entropy"] for f in DECLARED_FIELDS
    )
    return out


def _encoded_stats(hetero_data) -> Dict[str, Any]:
    """Dispersion of each QoS edge-feature column, as the model receives it.

    Pooled over the QoS-bearing relations only: on every other edge type dims
    9-15 are zero by construction, so including them would report a spread
    that is an artefact of edge-type mixing rather than of declared QoS.
    """
    blocks = []
    for rel in hetero_data.edge_types:
        if rel[1] not in _QOS_EDGE_TYPES:
            continue
        store = hetero_data[rel]
        attr = getattr(store, "edge_attr", None)
        if attr is None or attr.numel() == 0:
            continue
        blocks.append(attr.detach().cpu().numpy())

    if not blocks:
        return {"n_qos_edges": 0, "columns": {}}

    mat = np.concatenate(blocks, axis=0)
    cols: Dict[str, Any] = {}
    for idx, name in QOS_EDGE_COLUMNS:
        if idx >= mat.shape[1]:
            continue
        col = mat[:, idx]
        cols[name] = {
            "std": float(col.std()),
            "distinct": int(len(np.unique(np.round(col, 6)))),
            "mean": float(col.mean()),
        }
    return {"n_qos_edges": int(mat.shape[0]), "columns": cols}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=Path("output/loso_cache"))
    p.add_argument("--output", type=Path,
                   default=Path("results/qos_corpus_diagnostic.md"))
    p.add_argument("--json-output", type=Path,
                   default=Path("results/qos_corpus_diagnostic.json"))
    p.add_argument("--skip", default="", help="Comma-separated scenario ids to skip")
    args = p.parse_args()

    skip = [s.strip() for s in args.skip.split(",") if s.strip()]
    bundles = discover_scenarios(args.cache_dir, skip)
    if not bundles:
        print(f"No scenario bundles found under {args.cache_dir}", file=sys.stderr)
        sys.exit(1)

    payload: Dict[str, Any] = {"cache_dir": str(args.cache_dir), "scenarios": {}}
    for b in bundles:
        payload["scenarios"][b.scenario_id] = {
            "n_nodes": b.n_nodes,
            "declared": _declared_stats(b.graph),
            "encoded": _encoded_stats(b.hetero_data),
        }

    degenerate = sorted(
        sid for sid, d in payload["scenarios"].items()
        if d["declared"]["qos_triple_entropy"] <= 1e-12
    )
    payload["qos_degenerate_scenarios"] = degenerate
    payload["n_degenerate"] = len(degenerate)
    payload["n_scenarios"] = len(payload["scenarios"])

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n")

    # ── Markdown ──────────────────────────────────────────────────────────────
    sids = sorted(payload["scenarios"])
    lines: List[str] = []
    lines.append("# QoS Corpus Diagnostic\n")
    lines.append(
        "How much *within-graph* information the declared QoS contract carries "
        "in each scenario. `H` is Shannon entropy (nats) over Topic nodes; "
        "`H = 0` means every topic in that graph declares the identical value, "
        "so the corresponding edge-feature dimensions are a constant offset for "
        "the whole graph and cannot discriminate between its components.\n"
    )
    lines.append(
        f"**{len(degenerate)} of {len(sids)} scenarios have a fully degenerate "
        f"QoS triple** (H(rel) = H(dur) = H(pri) = 0): "
        + ", ".join(f"`{s}`" for s in degenerate) + ".\n"
    )

    lines.append("\n## Declared QoS entropy (source)\n")
    lines.append("| scenario | topics | H(reliability) | H(durability) | H(priority) | **H(triple)** | H(criticality) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for sid in sids:
        d = payload["scenarios"][sid]["declared"]
        lines.append(
            f"| {sid} | {d['n_topics']} | "
            f"{d['reliability']['entropy']:.3f} | "
            f"{d['durability']['entropy']:.3f} | "
            f"{d['transport_priority']['entropy']:.3f} | "
            f"**{d['qos_triple_entropy']:.3f}** | "
            f"{d['criticality']['entropy']:.3f} |"
        )
    lines.append(
        "\nNote that `criticality` stays non-degenerate everywhere. It is a "
        "Topic *node* feature (`topic_qos_criticality_ord`), not one of the 7 "
        "QoS *edge* dimensions, which is why a QoS-weighted structural score "
        "can still rank well on a graph whose edge QoS channel is uniform.\n"
    )

    lines.append("\n## As-encoded dispersion (what reaches the model)\n")
    lines.append(
        "Standard deviation of each QoS edge-feature column over the QoS-bearing "
        "relations (`PUBLISHES_TO`, `SUBSCRIBES_TO`, `DEPENDS_ON`). "
        "`0` marks a column that is constant across every such edge in the graph.\n"
    )
    col_names = [name for _, name in QOS_EDGE_COLUMNS]
    lines.append("| scenario | QoS edges | " + " | ".join(col_names) + " |")
    lines.append("|---" * (len(col_names) + 2) + "|")
    for sid in sids:
        e = payload["scenarios"][sid]["encoded"]
        cells = []
        for name in col_names:
            c = e["columns"].get(name)
            cells.append("—" if c is None else f"{c['std']:.3f}")
        lines.append(f"| {sid} | {e['n_qos_edges']} | " + " | ".join(cells) + " |")

    lines.append("\n## Per-dimension corpus summary\n")
    lines.append(
        "How many scenarios each QoS edge dimension actually varies in. A "
        "dimension live in 0 scenarios is dead weight in the architecture: it "
        "occupies a feature slot and a projection row in every model ever "
        "trained on this corpus, and has never carried a bit of information.\n"
    )
    lines.append("| edge dimension | live in | constant value where dead |")
    lines.append("|---|---:|---|")
    for _, name in QOS_EDGE_COLUMNS:
        live, dead_vals = [], set()
        for sid in sids:
            c = payload["scenarios"][sid]["encoded"]["columns"].get(name)
            if c is None:
                continue
            if c["std"] > 1e-9:
                live.append(sid)
            else:
                dead_vals.add(round(c["mean"], 4))
        dead_str = ", ".join(str(v) for v in sorted(dead_vals)) if dead_vals else "—"
        lines.append(f"| `{name}` | {len(live)}/{len(sids)} | {dead_str} |")

    dead_everywhere = [
        name for _, name in QOS_EDGE_COLUMNS
        if all(
            payload["scenarios"][sid]["encoded"]["columns"].get(name, {}).get("std", 0.0)
            <= 1e-9
            for sid in sids
        )
    ]
    payload["dimensions_dead_in_every_scenario"] = dead_everywhere
    if dead_everywhere:
        lines.append(
            "\n**"
            + ", ".join(f"`{n}`" for n in dead_everywhere)
            + f" vary in no scenario at all.** The generator emits no deadline "
            "or blocking fields, so these dimensions are identically zero across "
            "the whole corpus. Any claim about a 16-dimensional edge vector "
            "carrying 7 QoS dimensions should be read against this table.\n"
        )

    args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")

    print(f"Wrote {args.output}")
    print(f"Wrote {args.json_output}")
    print(f"\nQoS-degenerate scenarios ({len(degenerate)}/{len(sids)}):")
    for s in degenerate:
        print(f"  {s}")
    if dead_everywhere:
        print(f"\nDimensions constant in ALL {len(sids)} scenarios:")
        for n in dead_everywhere:
            print(f"  {n}")


if __name__ == "__main__":
    main()
