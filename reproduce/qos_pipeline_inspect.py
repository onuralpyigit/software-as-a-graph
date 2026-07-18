#!/usr/bin/env python3
"""
reproduce/qos_pipeline_inspect.py — Stage-by-stage QoS attribute trace
===================================================================

Block 0, Task 0.5: Generates diagnostic source data for the QoS edge-encoding pipeline described
in the paper's §3.1 ("Each edge is encoded as a 16-dimensional feature vector..."). Auxiliary
inspection tooling, not itself a numbered table/figure in the current JSS paper.

Traces QoS attributes through every stage of the pipeline:
  Stage 1: Raw QoS fields per Topic (from topology JSON)
  Stage 2: QoSPolicy.calculate_weight() result per Topic
  Stage 3: edge_attr slice (dims 9-15) per relation triple in HeteroData
  Stage 4: Model parameter summary (if checkpoint provided)

Usage
-----
  python reproduce/qos_pipeline_inspect.py --scenario data/scenarios/atm_system.json
  python reproduce/qos_pipeline_inspect.py --scenario data/scenarios/atm_system.json \\
      --checkpoint output/gnn_checkpoints/atm_hetero_qos/ --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path for direct execution
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _banner(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_table(headers: List[str], rows: List[List[str]], col_widths: Optional[List[int]] = None) -> None:
    if not rows:
        print("  (no data)")
        return
    if col_widths is None:
        col_widths = [max(len(str(r[i])) for r in [headers] + rows) + 2 for i in range(len(headers))]
    fmt = "  " + "".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  " + "-" * (sum(col_widths)))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))


# ── Stage 1: Raw QoS from topology JSON ──────────────────────────────────────

def stage1_raw_qos(topology: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract raw QoS fields per Topic from the topology JSON."""
    topics = topology.get("topics", [])
    result = {}
    for t in topics:
        tid = t.get("id", t.get("name", "?"))
        qos = t.get("qos", t.get("qos_profile", {})) or {}
        result[tid] = qos
    return result


def _print_stage1(qos_by_topic: Dict[str, Dict[str, Any]]) -> None:
    _banner("Stage 1 — Raw QoS fields per Topic (topology JSON)")
    if not qos_by_topic:
        print("  ⚠  No topics with QoS fields found in topology JSON.")
        print("  Paper §5.1 must disclose that QoS heterogeneity is synthetic.")
        return
    rows = []
    for tid, qos in sorted(qos_by_topic.items()):
        rows.append([
            tid[:40],
            qos.get("reliability", "—"),
            qos.get("durability", "—"),
            qos.get("transport_priority", qos.get("priority", "—")),
            str(qos.get("deadline_ns", "—")),
            str(qos.get("max_blocking_ms", "—")),
        ])
    _print_table(
        ["Topic ID", "Reliability", "Durability", "Priority", "Deadline(ns)", "MaxBlocking(ms)"],
        rows, [42, 14, 18, 10, 14, 16]
    )


# ── Stage 2: QoSPolicy scoring ────────────────────────────────────────────────

def stage2_qos_weights(qos_by_topic: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """Compute QoSPolicy.calculate_weight() for each Topic."""
    from saag.core.models import QoSPolicy
    scores = {}
    for tid, qos in qos_by_topic.items():
        policy = QoSPolicy.from_dict(qos) if qos else QoSPolicy()
        scores[tid] = policy.calculate_weight()
    return scores


def _print_stage2(scores: Dict[str, float]) -> None:
    _banner("Stage 2 — QoSPolicy.calculate_weight() per Topic")
    if not scores:
        print("  (no topics)")
        return
    rows = sorted(scores.items(), key=lambda x: -x[1])
    print(f"  Gini coefficient of QoS weight distribution: {_gini([v for _, v in rows]):.4f}")
    print()
    _print_table(
        ["Topic ID", "QoS Weight", "Bar"],
        [[tid[:40], f"{w:.4f}", "#" * int(w * 30)] for tid, w in rows],
        [42, 12, 32]
    )


def _gini(values: List[float]) -> float:
    if not values:
        return 0.0
    n = len(values)
    if n == 1:
        return 0.0
    s = sorted(values)
    total = sum(s)
    if total == 0:
        return 0.0
    cumsum = 0.0
    lorenz = []
    for v in s:
        cumsum += v
        lorenz.append(cumsum / total)
    # Gini = 1 - 2 * area under Lorenz curve
    area = sum((lorenz[i] + lorenz[i - 1]) for i in range(1, n)) / (2 * (n - 1))
    return 1.0 - 2.0 * area


# ── Stage 3: edge_attr in HeteroData ──────────────────────────────────────────

def stage3_edge_attr(graph, structural_metrics=None, simulation_results=None):
    """Build HeteroData and dump edge_attr shapes + QoS slice stats."""
    from saag.prediction.data_preparation import (
        EDGE_FEATURE_DIM,
        networkx_to_hetero_data,
    )
    conv = networkx_to_hetero_data(graph, structural_metrics, simulation_results)
    data = conv.hetero_data
    return data, EDGE_FEATURE_DIM, conv


def _print_stage3(data, EDGE_FEATURE_DIM: int, conv) -> None:
    import torch
    _banner(f"Stage 3 — HeteroData edge_attr (expected dim = {EDGE_FEATURE_DIM})")
    print(f"  Node types  : {list(data.node_types)}")
    print(f"  Edge types  : {len(list(data.edge_types))} relation triples")
    print()

    rows = []
    qos_warnings = []
    for rel in data.edge_types:
        ea = data[rel].edge_attr
        shape_str = f"({ea.shape[0]} × {ea.shape[1]})"
        dim_ok = "✓" if ea.shape[1] == EDGE_FEATURE_DIM else f"✗ (got {ea.shape[1]})"
        qos_slice = ea[:, 9:] if ea.shape[1] >= 16 else None
        if rel[1] in {"PUBLISHES_TO", "SUBSCRIBES_TO"} and qos_slice is not None:
            qos_sum = qos_slice.abs().sum().item()
            qos_str = f"sum(|dims9-15|)={qos_sum:.4f}"
            if qos_sum < 1e-6:
                qos_warnings.append(f"  ⚠  {rel}: QoS dims 9-15 are ALL ZERO — QoS not flowing!")
        else:
            qos_str = "n/a (non-pub/sub)"
        rows.append([str(rel), shape_str, dim_ok, qos_str])

    _print_table(
        ["Relation triple", "Shape (E×D)", "Dim OK?", "QoS dims check"],
        rows, [55, 14, 10, 40]
    )

    if qos_warnings:
        print()
        for w in qos_warnings:
            print(w)
        print()
        print("  ACTION REQUIRED: QoS must flow to edge_attr for the paper claim to hold.")
    else:
        print()
        print("  ✓ All pub/sub edges carry non-zero QoS attributes (dims 9-15).")


def _print_stage3_sample(data, rel, n_samples: int = 3) -> None:
    """Print the first n edge feature vectors for a relation triple."""
    ea = data[rel].edge_attr
    labels = [
        "weight", "path_cnt_norm",
        "pub", "sub", "routes", "runs_on", "connects", "uses", "depends",
        "reliability", "durability", "priority",
        "has_deadline", "deadline_log", "blocking_log", "qos_hetero_flag"
    ]
    print(f"\n  Sample edge_attr for {rel} (first {min(n_samples, ea.shape[0])} edges):")
    for i in range(min(n_samples, ea.shape[0])):
        row = ea[i].tolist()
        parts = [f"{lbl}={v:.3f}" for lbl, v in zip(labels, row)]
        print(f"    [{i}] " + "  ".join(parts))


# ── Stage 4: Model parameter summary ─────────────────────────────────────────

def stage4_model_params(checkpoint_dir: str, data) -> None:
    """Load checkpoint and print edge_dim per relation triple."""
    import torch
    from saag.prediction.gnn_service import GNNService
    _banner("Stage 4 — Trained model parameter inspection")
    try:
        svc = GNNService.from_checkpoint(checkpoint_dir)
        model = svc._node_model
        if model is None:
            print("  Model not loaded.")
            return

        total = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total:,}")
        print()

        # Probe EdgeFeatureEncoder projection dimensions
        for i, enc in enumerate(getattr(model, "edge_encoders", [])):
            proj = getattr(enc, "proj", None)
            if proj is not None:
                print(f"  EdgeFeatureEncoder[{i}].proj: {proj.in_features} → {proj.out_features}")

        # Probe input projections
        print()
        for nt, proj in getattr(model, "input_proj", {}).items():
            fc = proj[0]
            print(f"  input_proj[{nt}]: {fc.in_features} → {fc.out_features}")

    except Exception as e:
        print(f"  Could not load checkpoint: {e}")


# ── JSON output ───────────────────────────────────────────────────────────────

def _to_json_report(
    qos_by_topic: Dict[str, Dict],
    qos_scores: Dict[str, float],
    data,
    EDGE_FEATURE_DIM: int,
) -> Dict[str, Any]:
    import torch
    report: Dict[str, Any] = {
        "edge_feature_dim": EDGE_FEATURE_DIM,
        "gini_coefficient": _gini(list(qos_scores.values())),
        "topics": {
            tid: {
                "raw_qos": qos_by_topic.get(tid, {}),
                "qos_weight": round(w, 4),
            }
            for tid, w in qos_scores.items()
        },
        "relations": [],
    }
    for rel in data.edge_types:
        ea = data[rel].edge_attr
        qos_slice = ea[:, 9:] if ea.shape[1] >= 16 else None
        report["relations"].append({
            "relation": list(rel),
            "shape": list(ea.shape),
            "dim_ok": ea.shape[1] == EDGE_FEATURE_DIM,
            "qos_dims_nonzero": float(qos_slice.abs().sum().item()) > 0 if qos_slice is not None else None,
        })
    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage-by-stage QoS pipeline inspector (Block 0, Task 0.5).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--scenario", required=True, type=Path,
                   help="Path to topology JSON (e.g., data/scenarios/atm_system.json)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional: path to trained checkpoint directory (for Stage 4)")
    p.add_argument("--sample-edges", type=int, default=3, metavar="N",
                   help="Number of sample edge vectors to print per pub/sub relation")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON report to stdout instead of rich output")
    p.add_argument("--output", type=Path, default=None,
                   help="Save JSON report to this file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.scenario.exists():
        print(f"Error: scenario file not found: {args.scenario}", file=sys.stderr)
        sys.exit(1)

    topology = json.loads(args.scenario.read_text())
    scenario_name = args.scenario.stem

    # ── Build graph ───────────────────────────────────────────────────────────
    from cli.loso_evaluate import _build_graph_from_json
    graph = _build_graph_from_json(topology)

    if not args.json:
        print(f"\nQoS Pipeline Inspector — scenario: {scenario_name}")
        print(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    qos_by_topic = stage1_raw_qos(topology)
    if not args.json:
        _print_stage1(qos_by_topic)

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    qos_scores = stage2_qos_weights(qos_by_topic)
    if not args.json:
        _print_stage2(qos_scores)

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    data, EDGE_FEATURE_DIM, conv = stage3_edge_attr(graph)
    if not args.json:
        _print_stage3(data, EDGE_FEATURE_DIM, conv)

        # Sample edge vectors for pub/sub relations
        for rel in data.edge_types:
            if rel[1] in {"PUBLISHES_TO", "SUBSCRIBES_TO"}:
                _print_stage3_sample(data, rel, args.sample_edges)

    # ── Stage 4 ───────────────────────────────────────────────────────────────
    if args.checkpoint and not args.json:
        stage4_model_params(args.checkpoint, data)

    # ── JSON output ───────────────────────────────────────────────────────────
    if args.json or args.output:
        report = _to_json_report(qos_by_topic, qos_scores, data, EDGE_FEATURE_DIM)
        report["scenario"] = scenario_name
        if args.json:
            print(json.dumps(report, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2))
            if not args.json:
                print(f"\n  JSON report saved to: {args.output}")

    if not args.json:
        print()
        print("  Done. Use --json --output results/qos_inspect.json to save for §3.2.")


if __name__ == "__main__":
    main()
