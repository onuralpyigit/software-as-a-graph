#!/usr/bin/env python3
"""
simulate_graph.py
─────────────────
CLI entry point for the SaG simulation pipeline.

Two simulation modes, each accessible as a subcommand:

  fault-inject   Systematic BFS cascade fault injection.
                 Produces ``impact_scores.json`` – the I(v) ground-truth
                 vector used to validate Q(v) predictions via Spearman ρ.

  message-flow   Discrete-event pub-sub message flow simulation (SimPy).
                 Models publishers, broker queues, subscribers, QoS
                 enforcement, and optional runtime fault injection.

  combined       Run both modes in sequence.

EXAMPLES
────────

  # Ground-truth generation for the ATM dataset (all Application + Broker nodes)
  python cli/simulate_graph.py fault-inject \
      --input data/atm_system.json \\
      --output output/simulation/ \\
      --seeds 42,123,456,789,2024 \\
      --export-json

  # Single-node fault injection for ConflictDetector only
  python cli/simulate_graph.py fault-inject \
      --input data/atm_system.json \\
      --nodes ConflictDetector \\
      --output output/simulation/ \\
      --export-json

  # Message-flow simulation, 300 s, fault ConflictDetector at t=150 s
  python cli/simulate_graph.py message-flow \
      --input data/atm_system.json \\
      --duration 300 \\
      --fault-node ConflictDetector \\
      --fault-time 150 \\
      --seed 42 \\
      --output output/simulation/ \\
      --export-json

  # Run both in one pass
  python cli/simulate_graph.py combined \
      --input data/atm_system.json \\
      --output output/simulation/ \\
      --seeds 42,123,456 \\
      --duration 200 \\
      --fault-node ASTERIX_Broker \\
      --export-json

OUTPUT FILES
────────────
  fault-inject  →  output/impact_scores.json
                   output/impact_scores_summary.txt
  message-flow  →  output/message_flow_results.json
                   output/message_flow_summary.txt
  combined      →  all of the above
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path to support direct execution (python cli/simulate_graph.py)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from typing import Any, Dict, List, Optional

from cli.common.arguments import setup_logging

logger = logging.getLogger("simulate_graph")


# ─────────────────────────────────────────────────────────────────────────────
# Graph loader (mirrors analyze_graph.py's approach)
# ─────────────────────────────────────────────────────────────────────────────

def _load_graph(input_path: Path):
    """
    Load a SaG graph from a JSON file and return a NetworkX DiGraph.

    Tries to use the project's GraphBuilder / GraphExporter pipeline first.
    Falls back to a lightweight inline loader for environments where the
    full src/ package is not importable.
    """
    import networkx as nx

    # ── Try project pipeline ─────────────────────────────────────────────
    try:
        from saag.core.graph_builder import GraphBuilder
        from saag.core.graph_exporter import GraphExporter

        builder = GraphBuilder()
        model = builder.build_from_json(str(input_path))
        exporter = GraphExporter()
        g = exporter.export_to_networkx(model)
        g.graph["id"] = input_path.stem
        logger.info("Graph loaded via GraphBuilder: %d nodes, %d edges",
                    len(g.nodes), len(g.edges))
        return g
    except Exception as exc:
        logger.debug("GraphBuilder unavailable (%s); using fallback loader.", exc)

    # ── Fallback loader ───────────────────────────────────────────────────
    with open(input_path) as fh:
        data = json.load(fh)

    g = nx.DiGraph()
    g.graph["id"] = input_path.stem

    # Nodes
    for app in data.get("applications", []):
        g.add_node(app["id"], type="Application",
                   name=app.get("name", app["id"]),
                   **{k: v for k, v in app.items() if k not in ("id", "name")})
    for broker in data.get("brokers", []):
        g.add_node(broker["id"], type="Broker",
                   name=broker.get("name", broker["id"]),
                   **{k: v for k, v in broker.items() if k not in ("id", "name")})
    for topic in data.get("topics", []):
        g.add_node(topic["id"], type="Topic",
                   name=topic.get("name", topic["id"]),
                   **{k: v for k, v in topic.items() if k not in ("id", "name")})
    for node in data.get("nodes", []):
        g.add_node(node["id"], type="Node",
                   name=node.get("name", node["id"]),
                   **{k: v for k, v in node.items() if k not in ("id", "name")})

    # Edges
    # Support both flat and nested 'relationships' structure
    rels = data.get("relationships", {})
    
    # 1. PUBLISHES_TO
    pub_list = data.get("publishes", []) + data.get("publish_edges", []) + rels.get("publishes_to", [])
    for pub in pub_list:
        app_id = pub.get("application_id") or pub.get("source") or pub.get("from")
        topic_id = pub.get("topic_id") or pub.get("target") or pub.get("to")
        if app_id and topic_id:
            g.add_edge(app_id, topic_id, type="PUBLISHES_TO",
                       rate_hz=pub.get("rate_hz", 10.0),
                       qos_profile=pub.get("qos_profile", {}))

    # 2. SUBSCRIBES_TO
    sub_list = data.get("subscribes", []) + data.get("subscribe_edges", []) + rels.get("subscribes_to", [])
    for sub in sub_list:
        app_id = sub.get("application_id") or sub.get("source") or sub.get("from")
        topic_id = sub.get("topic_id") or sub.get("target") or sub.get("to")
        if app_id and topic_id:
            g.add_edge(app_id, topic_id, type="SUBSCRIBES_TO",
                       qos_profile=sub.get("qos_profile", {}))

    # 3. ROUTES
    # Handle both dict-based broker_routes and list-based relationships["routes"]
    for broker_id, topic_ids in (data.get("broker_routes") or {}).items():
        if isinstance(topic_ids, list):
            for tid in topic_ids:
                g.add_edge(broker_id, tid, type="ROUTES")
    
    for route in rels.get("routes", []):
        src = route.get("source") or route.get("broker_id") or route.get("from")
        tgt = route.get("target") or route.get("topic_id") or route.get("to")
        if src and tgt:
            g.add_edge(src, tgt, type="ROUTES")

    # 4. RUNS_ON (important for physical fault propagation)
    for run in rels.get("runs_on", []):
        src = run.get("source") or run.get("application_id") or run.get("from")
        tgt = run.get("target") or run.get("node_id") or run.get("to")
        if src and tgt:
            g.add_edge(src, tgt, type="RUNS_ON")

    # 5. CONNECTS_TO
    for conn in rels.get("connects_to", []):
        src = conn.get("source") or conn.get("from")
        tgt = conn.get("target") or conn.get("to")
        if src and tgt:
            g.add_edge(src, tgt, type="CONNECTS_TO")

    # 6. USES
    # Note: Important for library-mediated publishing/subscribing
    for use in rels.get("uses", []) or data.get("uses", []):
        src = use.get("source") or use.get("from")
        tgt = use.get("target") or use.get("to")
        if src and tgt:
            g.add_edge(src, tgt, type="USES")

    logger.info("Graph loaded via fallback: %d nodes, %d edges",
                len(g.nodes), len(g.edges))
    return g


# ─────────────────────────────────────────────────────────────────────────────
# fault-inject subcommand
# ─────────────────────────────────────────────────────────────────────────────

def _run_fault_inject(args: argparse.Namespace) -> None:
    from saag.simulation.fault_injector import FaultInjector

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    seeds = _parse_seeds(args.seeds)
    node_types = [t.strip() for t in args.node_types.split(",")]
    node_ids: Optional[List[str]] = (
        [n.strip() for n in args.nodes.split(",")]
        if args.nodes
        else None
    )

    logger.info("═" * 60)
    logger.info("FAULT INJECTION")
    logger.info("  Input      : %s", input_path)
    logger.info("  Output     : %s", output_dir)
    logger.info("  Node types : %s", node_types)
    logger.info("  Node IDs   : %s", node_ids or "all")
    logger.info("  Seeds      : %s", seeds)
    logger.info("  Cascade lim: %s", args.cascade_depth or "unlimited")
    logger.info("═" * 60)

    t0 = time.perf_counter()
    g = _load_graph(input_path)

    injector = FaultInjector(
        graph=g,
        seeds=seeds,
        cascade_depth_limit=args.cascade_depth,
    )
    result = injector.run(node_types=node_types, node_ids=node_ids)
    elapsed = time.perf_counter() - t0

    # ── Print summary ────────────────────────────────────────────────────
    _print_fault_inject_summary(result, elapsed)

    # ── Export ────────────────────────────────────────────────────────────
    if args.export_json:
        out_json = output_dir / "impact_scores.json"
        result.save(out_json)
        logger.info("Impact scores written → %s", out_json)

    _write_fault_inject_text_summary(result, elapsed, output_dir)


def _print_fault_inject_summary(result, elapsed: float) -> None:
    print()
    print("=" * 70)
    print("FAULT INJECTION SUMMARY")
    print("=" * 70)
    print(f"  Nodes injected          : {result.total_nodes_injected}")
    print(f"  Total subscribers       : {result.total_subscribers}")
    print(f"  Seeds used              : {result.seeds_used}")
    print(f"  Elapsed                 : {elapsed:.2f}s")
    print()
    print(f"  {'Rank':<5}  {'Node ID':<30}  {'Type':<14}  {'I(v)':>7}  "
          f"{'Depth':>5}  {'Orphaned':>8}  {'Impacted':>8}")
    print(f"  {'─'*5}  {'─'*30}  {'─'*14}  {'─'*7}  {'─'*5}  {'─'*8}  {'─'*8}")
    for row in result.top_k_by_impact[:20]:
        print(f"  {row['rank']:<5}  {row['node_id']:<30}  "
              f"{row['node_type']:<14}  {row['impact_score']:>7.4f}  "
              f"{row['cascade_depth']:>5}  {row['orphaned_topics']:>8}  "
              f"{row['impacted_subscribers']:>8}")
    print("=" * 70)
    print()


def _write_fault_inject_text_summary(result, elapsed: float, output_dir: Path) -> None:
    lines = ["FAULT INJECTION SUMMARY", "=" * 70, ""]
    lines.append(f"Nodes injected   : {result.total_nodes_injected}")
    lines.append(f"Total subscribers: {result.total_subscribers}")
    lines.append(f"Seeds used       : {result.seeds_used}")
    lines.append(f"Elapsed          : {elapsed:.2f}s")
    lines.append("")
    lines.append(f"{'Rank':<5}  {'Node ID':<30}  {'Type':<14}  "
                 f"{'I(v)':>7}  {'StdDev':>7}  {'Depth':>5}  "
                 f"{'Orphaned':>8}  {'Impacted':>8}")
    lines.append("-" * 90)
    for row in result.top_k_by_impact:
        lines.append(
            f"{row['rank']:<5}  {row['node_id']:<30}  {row['node_type']:<14}  "
            f"{row['impact_score']:>7.4f}  {row['impact_score_std']:>7.4f}  "
            f"{row['cascade_depth']:>5}  {row['orphaned_topics']:>8}  "
            f"{row['impacted_subscribers']:>8}"
        )
    summary_path = output_dir / "impact_scores_summary.txt"
    summary_path.write_text("\n".join(lines))
    logger.info("Text summary written → %s", summary_path)


# ─────────────────────────────────────────────────────────────────────────────
# message-flow subcommand
# ─────────────────────────────────────────────────────────────────────────────

def _run_message_flow(args: argparse.Namespace) -> None:
    try:
        from saag.simulation.message_flow_simulator import MessageFlowSimulator
    except ImportError as exc:
        logger.error("Cannot import MessageFlowSimulator: %s", exc)
        sys.exit(1)

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    fault_time = getattr(args, "fault_time", None)
    fault_node = getattr(args, "fault_node", None)

    logger.info("═" * 60)
    logger.info("MESSAGE FLOW SIMULATION")
    logger.info("  Input      : %s", input_path)
    logger.info("  Output     : %s", output_dir)
    logger.info("  Duration   : %.1f s", args.duration)
    logger.info("  Seed       : %d", args.seed)
    logger.info("  Fault node : %s", fault_node or "none")
    if fault_node:
        logger.info("  Fault time : %.1f s", fault_time or (args.duration / 2))
    logger.info("═" * 60)

    t0 = time.perf_counter()
    g = _load_graph(input_path)

    sim = MessageFlowSimulator(
        graph=g,
        duration=float(args.duration),
        fault_node=fault_node,
        fault_time=float(fault_time) if fault_time else None,
        seed=args.seed,
        default_queue_size=getattr(args, "default_queue_size", 100),
        default_publish_rate_hz=getattr(args, "default_rate", 10.0),
    )
    result = sim.run()
    elapsed = time.perf_counter() - t0

    _print_message_flow_summary(result, elapsed)

    if args.export_json:
        out_json = output_dir / "message_flow_results.json"
        result.save(out_json)
        logger.info("Message flow results written → %s", out_json)

    _write_message_flow_text_summary(result, elapsed, output_dir)


def _print_message_flow_summary(result, elapsed: float) -> None:
    print()
    print("=" * 70)
    print("MESSAGE FLOW SUMMARY")
    print("=" * 70)
    print(f"  Duration              : {result.simulation_duration:.1f} s")
    print(f"  Seed                  : {result.seed}")
    print(f"  Total published       : {result.total_messages_published:,}")
    print(f"  Total delivered       : {result.total_messages_delivered:,}")
    print(f"  System delivery rate  : {result.system_delivery_rate:.4f}")
    print(f"  System drop rate      : {result.system_drop_rate:.4f}")
    print(f"  Deadline violations   : {result.total_deadline_violations:,}")
    print(f"  Queue overflows       : {result.total_queue_overflows:,}")
    print(f"  Elapsed (wall)        : {elapsed:.2f}s")

    if result.fault_event:
        fe = result.fault_event
        print()
        print("  Fault event:")
        print(f"    Node       : {fe.faulted_node_id} ({fe.faulted_node_type})")
        print(f"    Time       : {fe.fault_time:.1f} s")
        print(f"    Orphaned   : {fe.cascade_orphaned_topics}")
        print(f"    Impacted   : {fe.cascade_impacted_subscribers}")
        print(f"    Rate before: {fe.delivery_rate_before:.4f}")
        print(f"    Rate after : {fe.delivery_rate_after:.4f}")

    print()
    print(f"  {'Topic':<30}  {'Delivery':>9}  {'P50 ms':>8}  {'P95 ms':>8}  "
          f"{'Deadline viol':>13}")
    print(f"  {'─'*30}  {'─'*9}  {'─'*8}  {'─'*8}  {'─'*13}")
    for tid, ts in sorted(result.topic_stats.items(),
                          key=lambda x: x[1].delivery_rate):
        p50 = f"{ts.latency_p50:.2f}" if ts.latency_p50 is not None else "—"
        p95 = f"{ts.latency_p95:.2f}" if ts.latency_p95 is not None else "—"
        print(f"  {ts.topic_name:<30}  {ts.delivery_rate:>9.4f}  {p50:>8}  "
              f"{p95:>8}  {ts.total_dropped_deadline:>13,}")
    print("=" * 70)
    print()


def _write_message_flow_text_summary(result, elapsed: float, output_dir: Path) -> None:
    lines = ["MESSAGE FLOW SIMULATION SUMMARY", "=" * 70, ""]
    lines.append(f"Duration           : {result.simulation_duration:.1f} s")
    lines.append(f"Seed               : {result.seed}")
    lines.append(f"Published          : {result.total_messages_published:,}")
    lines.append(f"Delivered          : {result.total_messages_delivered:,}")
    lines.append(f"System delivery    : {result.system_delivery_rate:.4f}")
    lines.append(f"Deadline violations: {result.total_deadline_violations:,}")
    lines.append(f"Queue overflows    : {result.total_queue_overflows:,}")
    lines.append(f"Elapsed (wall)     : {elapsed:.2f}s")
    if result.fault_event:
        fe = result.fault_event
        lines.extend([
            "",
            "FAULT EVENT",
            f"  Node    : {fe.faulted_node_id} ({fe.faulted_node_type})",
            f"  Time    : {fe.fault_time:.1f} s",
            f"  Orphaned: {', '.join(fe.cascade_orphaned_topics)}",
            f"  Impacted: {', '.join(fe.cascade_impacted_subscribers)}",
            f"  Rate before fault : {fe.delivery_rate_before:.4f}",
            f"  Rate after fault  : {fe.delivery_rate_after:.4f}",
        ])
    lines.extend(["", "TOPIC BREAKDOWN", "-" * 70])
    for tid, ts in sorted(result.topic_stats.items(),
                          key=lambda x: x[1].delivery_rate):
        lines.append(
            f"  {ts.topic_name:<30}  delivery={ts.delivery_rate:.4f}  "
            f"published={ts.total_published}  dropped_deadline={ts.total_dropped_deadline}"
        )

    summary_path = output_dir / "message_flow_summary.txt"
    summary_path.write_text("\n".join(lines))
    logger.info("Text summary written → %s", summary_path)


# ─────────────────────────────────────────────────────────────────────────────
# combined subcommand
# ─────────────────────────────────────────────────────────────────────────────

def _run_combined(args: argparse.Namespace) -> None:
    logger.info("Running COMBINED simulation (fault-inject + message-flow)")
    _run_fault_inject(args)
    _run_message_flow(args)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_seeds(seeds_str: str) -> List[int]:
    try:
        return [int(s.strip()) for s in seeds_str.split(",")]
    except ValueError:
        logger.error("Invalid seeds argument: %s  (expected comma-separated ints)", seeds_str)
        sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simulate_graph.py",
        description=(
            "SaG simulation pipeline: fault injection (ground-truth I(v)) "
            "and discrete-event message flow simulation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("EXAMPLES")[1] if "EXAMPLES" in __doc__ else "",
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # ── Shared arguments factory ──────────────────────────────────────────
    def _add_shared(p: argparse.ArgumentParser) -> None:
        p.add_argument("--input", required=True,
                       help="Path to graph JSON file.")
        p.add_argument("--output", default="output/simulation/",
                       help="Output directory.  Created if absent.  "
                            "Default: output/simulation/")
        p.add_argument("--export-json", action="store_true",
                       help="Write JSON result file(s) to --output.")
        p.add_argument("--verbose", "-v", action="store_true",
                       help="Enable DEBUG logging.")

    # ── fault-inject ──────────────────────────────────────────────────────
    fi = subparsers.add_parser(
        "fault-inject",
        help="BFS cascade fault injection → I(v) ground-truth impact scores.",
        description=(
            "Injects single-node faults and traces cascading pub-sub failures "
            "to produce per-node ground-truth impact scores I(v)."
        ),
    )
    _add_shared(fi)
    fi.add_argument(
        "--nodes",
        default=None,
        metavar="ID1,ID2,...",
        help="Comma-separated node IDs to inject.  If omitted, injects all "
             "nodes matching --node-types.",
    )
    fi.add_argument(
        "--node-types",
        default="Application,Broker",
        metavar="TYPE1,TYPE2",
        help="Comma-separated node types to inject.  Default: Application,Broker",
    )
    fi.add_argument(
        "--seeds",
        default="42",
        metavar="42,123,...",
        help="Comma-separated seeds for multi-seed stability testing.  "
             "Default: 42",
    )
    fi.add_argument(
        "--cascade-depth",
        type=int,
        default=0,
        metavar="N",
        help="Maximum cascade wave depth (0 = unlimited).  Default: 0",
    )

    # ── message-flow ─────────────────────────────────────────────────────
    mf = subparsers.add_parser(
        "message-flow",
        help="Discrete-event pub-sub message flow simulation (SimPy).",
        description=(
            "Models publishers, broker queues, subscribers and QoS in "
            "simulated time.  Optionally injects a node fault at --fault-time."
        ),
    )
    _add_shared(mf)
    mf.add_argument(
        "--duration",
        type=float,
        default=100.0,
        metavar="SECONDS",
        help="Simulation duration in simulated seconds.  Default: 100.0",
    )
    mf.add_argument(
        "--fault-node",
        default=None,
        metavar="NODE_ID",
        help="Node ID to fault during simulation.  If omitted, no fault is injected.",
    )
    mf.add_argument(
        "--fault-time",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Simulated time at which to inject the fault.  "
             "Default: duration / 2.",
    )
    mf.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for jitter in intervals and processing times.  Default: 42",
    )
    mf.add_argument(
        "--default-rate",
        type=float,
        default=10.0,
        metavar="HZ",
        help="Fallback publish rate (Hz) when not in graph metadata.  Default: 10.0",
    )
    mf.add_argument(
        "--default-queue-size",
        type=int,
        default=100,
        metavar="N",
        help="Fallback broker queue capacity.  Default: 100",
    )

    # ── combined ──────────────────────────────────────────────────────────
    co = subparsers.add_parser(
        "combined",
        help="Run fault-inject followed by message-flow.",
        description="Runs both simulation modes in sequence.",
    )
    _add_shared(co)
    # Fault-inject flags
    co.add_argument("--nodes", default=None, metavar="ID1,ID2,...")
    co.add_argument("--node-types", default="Application,Broker", metavar="TYPE1,TYPE2")
    co.add_argument("--seeds", default="42", metavar="42,123,...")
    co.add_argument("--cascade-depth", type=int, default=0, metavar="N")
    # Message-flow flags
    co.add_argument("--duration", type=float, default=100.0, metavar="SECONDS")
    co.add_argument("--fault-node", default=None, metavar="NODE_ID")
    co.add_argument("--fault-time", type=float, default=None, metavar="SECONDS")
    co.add_argument("--seed", type=int, default=42)
    co.add_argument("--default-rate", type=float, default=10.0, metavar="HZ")
    co.add_argument("--default-queue-size", type=int, default=100, metavar="N")

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    setup_logging(args)

    dispatch = {
        "fault-inject": _run_fault_inject,
        "message-flow": _run_message_flow,
        "combined": _run_combined,
    }

    handler = dispatch.get(args.subcommand)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()