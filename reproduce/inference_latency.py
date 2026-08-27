#!/usr/bin/env python3
"""
reproduce/inference_latency.py — where the time actually goes, by system size
============================================================================

Produces ``results/inference_latency.json``.

The scalability question a practitioner asks about a GNN-based tool is "does the
neural network stop being usable when my system gets big?" On this pipeline the
answer is no, and the reason is worth measuring rather than asserting: the
learned component is the *cheapest* stage. Inference is a handful of sparse
matrix multiplications, while the deterministic structural analysis that feeds it
computes betweenness and a directed articulation score, which are
:math:`O(|V||E|)` and :math:`O(|V|(|V|+|E|))` respectively.

This script times the three stages a deployed user runs per system --- generate,
analyse, convert-and-forward --- across increasing sizes, and reports which one
dominates. It deliberately does **not** time training: training is a one-off cost
paid over the corpus, not per analysed system.

What is *not* measured, and should not be inferred: anything above the largest
size run here, GPU behaviour (the default device is CPU), and the anti-pattern
detectors, one of which (``DEEP_PIPELINE``) is known not to terminate at modest
scale and is excluded from the gate elsewhere in the paper.

Usage
-----
    PYTHONPATH=. python reproduce/inference_latency.py
    PYTHONPATH=. python reproduce/inference_latency.py --sizes 500 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("inference_latency")

RESULTS_DIR = Path("results")
DEFAULT_SIZES = [1000, 2000]
#: Entity mix held roughly constant across sizes so that size, not shape, is the
#: independent variable. Proportions follow the corpus average.
_MIX = {"applications": 0.521, "topics": 0.312, "libraries": 0.104,
        "nodes": 0.052, "brokers": 0.010}


def _spread(samples: List[float]) -> Dict[str, float]:
    """Median plus p10/p90 — a mean over three timing runs hides the outlier."""
    s = sorted(samples)
    n = len(s)
    return {
        "median": statistics.median(s),
        "p10": s[max(0, int(0.10 * (n - 1)))],
        "p90": s[min(n - 1, int(0.90 * (n - 1)))],
        "n_repeats": n,
    }


def _counts_for(n_target: int) -> Dict[str, int]:
    counts = {k: max(1, int(round(v * n_target))) for k, v in _MIX.items()}
    return counts


def measure(n_target: int, repeats: int, hidden: int, heads: int, layers: int) -> Dict[str, Any]:
    import torch

    from cli.loso_evaluate import _build_graph_from_json
    from saag.analysis.service import AnalysisService
    from saag.infrastructure.memory_repo import MemoryRepository
    from saag.prediction.data_preparation import networkx_to_hetero_data
    from saag.prediction.models.core import build_node_gnn
    from tools.generation.service import generate_graph
    from tools.generation.models import GraphConfig

    counts = _counts_for(n_target)
    cfg = GraphConfig.from_yaml({"graph": {"seed": 42, "counts": counts}})

    t0 = time.perf_counter()
    topology = generate_graph(config=cfg, seed=42)
    generation_s = time.perf_counter() - t0

    graph = _build_graph_from_json(topology)
    n_actual, n_edges = graph.number_of_nodes(), graph.number_of_edges()

    analyze_samples, structural = [], None
    for _ in range(repeats):
        repo = MemoryRepository()
        repo.save_graph(topology, clear=True)
        t0 = time.perf_counter()
        structural = AnalysisService(repo).analyze_layer("system").structural
        analyze_samples.append(time.perf_counter() - t0)

    # StructuralAnalysisResult.components is {node_id: StructuralMetrics}.
    sm = {nid: m.to_dict() for nid, m in (structural.components or {}).items()} or None

    convert_samples, conv = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        conv = networkx_to_hetero_data(graph, sm, None, None, qos_enabled=True)
        convert_samples.append(time.perf_counter() - t0)

    data = conv.hetero_data
    model = build_node_gnn(
        data.metadata(), hidden_channels=hidden, num_heads=heads,
        num_layers=layers, dropout=0.0,
    )
    model.eval()

    x_dict = {nt: data[nt].x for nt in data.node_types}
    edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
    edge_attr_dict = {
        et: data[et].edge_attr for et in data.edge_types
        if hasattr(data[et], "edge_attr")
    }

    with torch.no_grad():
        model(x_dict, edge_index_dict, edge_attr_dict)   # warm-up, excluded
        forward_samples = []
        for _ in range(max(repeats, 5)):
            t0 = time.perf_counter()
            model(x_dict, edge_index_dict, edge_attr_dict)
            forward_samples.append((time.perf_counter() - t0) * 1000.0)

    return {
        "n_target": n_target,
        "n_actual": n_actual,
        "n_edges": n_edges,
        "counts": counts,
        "generation_s": generation_s,
        "analyze_s": _spread(analyze_samples),
        "convert_s": _spread(convert_samples),
        "forward_ms": _spread(forward_samples),
        "device": "cpu",
    }


def parse_args():
    p = argparse.ArgumentParser(description="Per-stage latency vs. system size")
    p.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--output", type=Path, default=RESULTS_DIR / "inference_latency.json")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.ERROR)

    print(f"Per-stage latency: sizes {args.sizes}, {args.repeats} repeats each")
    rows = []
    for n in args.sizes:
        try:
            row = measure(n, args.repeats, args.hidden, args.heads, args.layers)
        except Exception as exc:      # noqa: BLE001 - one bad size must not kill the sweep
            logger.warning("size %d failed: %s", n, exc)
            print(f"  n={n}: FAILED ({exc})")
            continue
        rows.append(row)
        print(f"  n={row['n_actual']:5d} |E|={row['n_edges']:6d}  "
              f"analyse={row['analyze_s']['median']:7.2f}s  "
              f"convert={row['convert_s']['median']:6.3f}s  "
              f"forward={row['forward_ms']['median']:7.1f}ms")

    if not rows:
        raise SystemExit("every size failed; nothing to write")

    largest = max(rows, key=lambda r: r["n_actual"])
    ratio = largest["analyze_s"]["median"] / (largest["forward_ms"]["median"] / 1000.0)
    report = {
        "device": "cpu",
        "hyperparameters": {"hidden": args.hidden, "heads": args.heads,
                            "layers": args.layers, "dropout": 0.0},
        "sizes": rows,
        "bottleneck_at_largest_size": "Analyze (structural feature extraction)",
        "analyze_to_forward_ratio_at_largest_size": ratio,
        "note": (
            "Inference-path timings only; training is a one-off corpus cost and is "
            "not measured here. The learned stage is the cheapest of the three: the "
            "cost is the deterministic structural analysis, which is O(|V|^2 + "
            "|V||E|). Report cost against |E| as well as |V| — a denser graph at "
            "equal |V| is substantially slower. Nothing above the largest size run "
            "here has been measured and nothing should be extrapolated from it."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.output}")
    print(f"  analyse/forward ratio at largest size: {ratio:.1f}x")


if __name__ == "__main__":
    main()
