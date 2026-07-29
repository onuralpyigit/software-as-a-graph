#!/usr/bin/env python3
"""
validate_graph.py — SaG Statistical Validation CLI
====================================================
Statistically proves that topology-based Q(v) predictions agree with
simulation-derived cascade impact I(v) as proxy ground truth.

Pipeline
--------
1. Load graph (JSON / Neo4j) and compute Q(v) scores
2. Derive simulation ground truth I(v) via cascade failure simulation
3. Run full statistical battery:
   • Spearman ρ  (primary gate, topology-class dependent)
   • Kendall τ   (robustness cross-check)
   • Wilcoxon signed-rank vs. degree-centrality baseline
   • Bootstrap 95% CI on ρ
4. Compute specialist metrics:
   • ICR@K  — In-Cluster Recall at K
   • RCR    — Rank Consistency Rate across seeds
   • BCE    — Binary Classification Error on top-K
   • SPOF-F1 — Articulation-point detection F1
   • FTR    — False Top Rate (critical false positives)
   • PG     — Predictive Gain over degree-centrality baseline
5. Node-type stratified reporting (Application, Broker, Topic, Infra, Library)
6. Topology-class gate evaluation (sparse / medium / dense / hub-spoke)
7. Multi-seed stability sweep  (seeds: 42, 123, 456, 789, 2024)

The implementation lives in the `cli.validation` package; this module is the
argument parser and command dispatcher.

Usage
-----
# Single run — topology-only baseline
python cli/validate_graph.py single --input data/system.json

# Single run — QoS-enriched
python cli/validate_graph.py single --input data/system.json --qos

# Multi-seed stability sweep
python cli/validate_graph.py sweep --input data/system.json --qos

# Full report (sweep + topology-class gates + node-type strata)
python cli/validate_graph.py report --input data/system.json \\
    --output output/validation_report.json --qos

# Run against ATM dataset with custom top-k
python cli/validate_graph.py report --input datasets/atm_system.json \\
    --top-k 10 --qos --output output/atm_validation.json

# Methodological-guard harness on pre-computed JSON artifacts
python cli/validate_graph.py harness \\
    --predictions output/predictions.json \\
    --ground-truth cascade=output/simulation/impact_scores.json \\
    --ground-truth latency=output/latency_delta.json \\
    --out output/harness_report.json

For the full flag reference see docs/cli-pipeline-guide.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

# Add project root to sys.path to support direct execution (python cli/validate_graph.py)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.common.arguments import setup_logging

# Imported flat and called unqualified: tests patch these names on this module,
# which only works when the call resolves through this module's globals.
from cli.validation.graph_io import NpEncoder, load_graph, write_csv
from cli.validation.harness import (
    GroundTruthSource, build_report, load_ground_truth, load_predictions,
)
from cli.validation.reporting import (
    _check_min_apps, print_ablation_report, print_single_report, print_sweep_report,
    write_latex_table,
)
from cli.validation.runners import run_ablation, run_single, run_sweep
from cli.validation.statistics import GATE_THRESHOLDS, classify_topology

SUBCOMMANDS = ("single", "sweep", "report", "compare", "harness")


class _GraphContext:
    """Graph and derived run parameters shared by the graph-based subcommands."""

    def __init__(self, args):
        self.G, self.raw = load_graph(args.input)
        self.seeds = [int(s) for s in args.seeds.split(",")]
        self.use_color = not args.no_color
        _check_min_apps(self.G, self.use_color)

        n_total = self.G.number_of_nodes()
        self.top_k_frac = (
            0.20 if args.top_k is None else max(0.01, args.top_k / max(n_total, 1))
        )
        self.topo_class = classify_topology(self.G)
        self.gnn_model = getattr(args, "gnn_model", None)

    def sweep_kwargs(self, args) -> dict:
        return dict(
            seeds=self.seeds, qos=args.qos, top_k_frac=self.top_k_frac,
            depth_limit=args.cascade, B=args.bootstrap, alpha=args.alpha,
            gnn_model=self.gnn_model,
        )


def _write_json(path, payload) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, cls=NpEncoder))


def _cmd_single(args) -> int:
    ctx = _GraphContext(args)
    vr, node_scores = run_single(
        ctx.G, ctx.raw, seed=ctx.seeds[0], qos=args.qos,
        top_k_frac=ctx.top_k_frac, depth_limit=args.cascade,
        B=args.bootstrap, alpha=args.alpha, gnn_model=ctx.gnn_model,
    )
    print_single_report(vr, ctx.topo_class, ctx.use_color)

    if args.verbose:
        print("\nTop-20 nodes by Q(v):")
        ranked = sorted(node_scores.values(), key=lambda ns: ns.Q, reverse=True)[:20]
        for rank, ns in enumerate(ranked, 1):
            print(f"  {rank:3d}. {ns.node_id:30s} Q={ns.Q:.4f}  I={ns.I:.4f}  "
                  f"{'AP ' if ns.is_articulation_point else '   '}{ns.node_type}")

    if args.csv:
        csv_path = (args.output or "validation").replace(".json", "") + "_nodes.csv"
        write_csv(node_scores, csv_path)
        print(f"\nPer-node CSV written to: {csv_path}")

    if args.output:
        _write_json(args.output, {"validation": asdict(vr), "topology_class": ctx.topo_class})
        print(f"JSON report written to: {args.output}")

    return 0 if vr.overall_pass else 1


def _cmd_sweep(args) -> int:
    ctx = _GraphContext(args)
    sr = run_sweep(ctx.G, ctx.raw, **ctx.sweep_kwargs(args))
    print_sweep_report(sr, ctx.use_color)

    if args.output:
        _write_json(args.output, asdict(sr))
        print(f"JSON report written to: {args.output}")

    return 0 if sr.all_gates_pass_rate == 1.0 else 1


def _cmd_report(args) -> int:
    ctx = _GraphContext(args)
    sr = run_sweep(ctx.G, ctx.raw, **ctx.sweep_kwargs(args))
    print_sweep_report(sr, ctx.use_color)
    if sr.per_seed:
        print_single_report(sr.per_seed[0], ctx.topo_class, ctx.use_color)

    if args.output:
        _write_json(args.output, {
            "sweep": asdict(sr),
            "topology_class": ctx.topo_class,
            "gate_thresholds": GATE_THRESHOLDS.get(ctx.topo_class),
        })
        print(f"\nFull report written to: {args.output}")

    return 0 if sr.all_gates_pass_rate == 1.0 else 1


def _cmd_compare(args) -> int:
    ctx = _GraphContext(args)
    ar = run_ablation(
        ctx.G, ctx.raw, seeds=ctx.seeds, top_k_frac=ctx.top_k_frac,
        depth_limit=args.cascade, B=args.bootstrap, alpha=args.alpha,
    )
    print_ablation_report(ar, ctx.use_color)

    if args.output:
        _write_json(args.output, asdict(ar))
        print(f"Ablation JSON written to: {args.output}")

    if args.latex:
        latex_path = (args.output or "ablation").replace(".json", "") + "_table.tex"
        write_latex_table(ar, latex_path)
        print(f"LaTeX table written to:   {latex_path}")

    # Exit 0 only if Δρ > 0 and significant (validates the QoS claim)
    return 0 if (ar.delta_rho > 0 and ar.rho_lift_significant) else 1


def _cmd_harness(args) -> int:
    """Operates on pre-computed JSON artifacts — no graph is loaded."""
    preds = load_predictions(args.predictions)

    sources: List[GroundTruthSource] = []
    for spec in args.ground_truth:
        name, _, rest = spec.partition("=")
        path_str, _, tag = rest.partition(":")
        sources.append(load_ground_truth(
            Path(path_str), name,
            qos_coupled=(tag == "qos"),
            independence="declared independent" if tag != "qos" else "QoS-coupled",
        ))

    if not sources:
        print("error: harness subcommand requires at least one --ground-truth source",
              file=sys.stderr)
        return 2

    text, blob = build_report(preds, sources)
    print(text)
    if args.out:
        _write_json(args.out, blob)
        print(f"\nJSON report → {args.out}")
    return 0


COMMANDS = {
    "single": _cmd_single,
    "sweep": _cmd_sweep,
    "report": _cmd_report,
    "compare": _cmd_compare,
    "harness": _cmd_harness,
}


def _parse_args():
    p = argparse.ArgumentParser(
        prog="validate_graph.py",
        description="SaG topology prediction vs. simulation-based ground-truth validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=False)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input",   required=True, help="Path to system.json")
    common.add_argument("--qos",     action="store_true", help="Enable QoS-enriched scoring")
    common.add_argument("--gnn-model", default=None, help="Path to GNN checkpoint directory")
    common.add_argument("--top-k",   type=int, default=None,
                        help="K for classification metrics (default: 20%% of nodes)")
    common.add_argument("--seeds",   default="42,123,456,789,2024",
                        help="Comma-separated seed list")
    common.add_argument("--cascade", type=int, default=5, help="Cascade depth limit")
    common.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples")
    common.add_argument("--alpha",   type=float, default=0.05, help="Significance level")
    common.add_argument("--output",  default=None, help="Write JSON report to path")
    common.add_argument("--csv",     action="store_true", help="Write per-node CSV")
    common.add_argument("--latex",   action="store_true", help="Write LaTeX ablation table")
    common.add_argument("--verbose", action="store_true")
    common.add_argument("--no-color", action="store_true")

    # subcommands
    sub.add_parser("single",  parents=[common], help="One-seed run (first seed only)")
    sub.add_parser("sweep",   parents=[common], help="Multi-seed stability sweep")
    sub.add_parser("report",  parents=[common], help="Full sweep + strata + gates JSON report")
    sub.add_parser("compare", parents=[common],
                   help="Ablation: topology-only vs QoS-enriched side-by-side")

    harness_p = sub.add_parser(
        "harness",
        help="Methodological-guard harness: validate pre-computed Q(v) vs I(v) JSON files",
    )
    harness_p.add_argument("--predictions", required=True, type=Path,
                            help="Q(v) JSON ({node_id:{type,Q}} or list).")
    harness_p.add_argument("--ground-truth", action="append", default=[],
                            metavar="NAME=PATH",
                            help="Ground-truth source (e.g. cascade=output/impact_scores.json). "
                                 "Repeatable. Append ':qos' to mark QoS-coupled.")
    harness_p.add_argument("--out", type=Path, default=None,
                            help="Optional JSON report path.")
    harness_p.add_argument("--no-color", action="store_true")

    return p.parse_args()


def main():
    # Default to "single" when no subcommand is given, so that
    # `validate_graph.py --input x.json` keeps working. Only the first
    # positional token can be a subcommand, so checking it alone avoids
    # mistaking an option *value* (e.g. `--output single`) for a command.
    if not sys.argv[1:] or sys.argv[1].startswith("-"):
        sys.argv.insert(1, "single")

    args = _parse_args()
    setup_logging(args)

    handler = COMMANDS.get(args.command)
    if handler is None:
        print(f"error: unknown command {args.command!r}; expected one of "
              f"{', '.join(SUBCOMMANDS)}", file=sys.stderr)
        sys.exit(2)

    sys.exit(handler(args))


if __name__ == "__main__":
    main()
