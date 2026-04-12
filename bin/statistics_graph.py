#!/usr/bin/env python3
"""
bin/statistics_graph.py — Cross-Cutting Statistics CLI
======================================================
Computes and displays chart statistics for a pub-sub graph topology.

Two data source modes
  Live (default)  — connects to Neo4j, exports graph data, then computes stats
  File (--input)  — reads a pre-exported dataset JSON directly (no DB required)

Usage examples
  # All charts from a live Neo4j instance
  python bin/statistics_graph.py

  # Specific charts only
  python bin/statistics_graph.py --chart topic_bandwidth app_balance criticality_io

  # Standalone file mode — no Neo4j required
  python bin/statistics_graph.py --input output/dataset.json

  # Save results to JSON
  python bin/statistics_graph.py --output output/stats.json

  # Minimal one-line-per-chart summary
  python bin/statistics_graph.py --format minimal

  # Raw JSON dump (machine-readable)
  python bin/statistics_graph.py --format json --output output/stats.json
"""

import sys
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path bootstrap — same pattern used by all other bin/ scripts
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
backend_path = project_root / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

import argparse
from bin._shared import add_neo4j_args, add_common_args, setup_logging
from bin.common.console import ConsoleDisplay, Colors
from api.presenters.statistics_presenter import serialise_numpy
from api.statistics import extract_cross_cutting_data, compute_all_extras_statistics

# ---------------------------------------------------------------------------
# Chart registry — id, display name, summary key hints for rendering
# ---------------------------------------------------------------------------
CHART_REGISTRY: Dict[str, Dict[str, Any]] = {
    "topic_bandwidth": {
        "title": "Topic Bandwidth (Size × Subscribers)",
        "summary_keys": [
            ("total_topics",    "Total Topics",    None),
            ("size_mean",       "Avg Size",        ".2f"),
            ("size_max",        "Max Size",        ".0f"),
            ("sub_mean",        "Avg Subscribers", ".2f"),
            ("sub_max",         "Max Subscribers", "d"),
            ("zero_sub_count",  "Unsubscribed",    "d"),
            ("bw_mean",         "Avg Bandwidth",   ".2f"),
            ("bw_median",       "Median Bandwidth",".2f"),
            ("outlier_count",   "Outliers (IQR)",  "d"),
        ],
    },
    "app_balance": {
        "title": "App Pub/Sub Balance",
        "summary_keys": [
            ("total_apps",       "Total Apps",       "d"),
            ("high_io_count",    "High-I/O Apps",    "d"),
            ("consumer_count",   "Consumer Apps",    "d"),
            ("producer_count",   "Producer Apps",    "d"),
            ("low_io_count",     "Low-I/O Apps",     "d"),
            ("pub_mean",         "Avg Publishes",    ".2f"),
            ("sub_mean",         "Avg Subscribes",   ".2f"),
            ("outlier_count",    "Outliers (IQR)",   "d"),
        ],
    },
    "topic_fanout": {
        "title": "Topic Fanout Patterns",
        "summary_keys": [
            ("total_topics",    "Total Topics",     "d"),
            ("one_to_n_count",  "1:N (Broadcast)",  "d"),
            ("n_to_1_count",    "N:1 (Aggregator)", "d"),
            ("n_to_m_count",    "N:M (Mesh)",       "d"),
            ("orphan_count",    "Orphan Topics",    "d"),
            ("pub_mean",        "Avg Publishers",   ".2f"),
            ("sub_mean",        "Avg Subscribers",  ".2f"),
            ("pub_max",         "Max Publishers",   "d"),
            ("sub_max",         "Max Subscribers",  "d"),
        ],
    },
    "cross_node_heatmap": {
        "title": "Cross-Node Communication Heatmap",
        "summary_keys": [
            ("node_count",         "Infrastructure Nodes",  "d"),
            ("total_traffic",      "Total Traffic",         "d"),
            ("intra_node_traffic", "Intra-Node Traffic",    "d"),
            ("inter_node_traffic", "Inter-Node Traffic",    "d"),
            ("intra_pct",          "Intra-Node %",          ".1f"),
            ("outlier_count",      "Outlier Pairs (IQR)",   "d"),
        ],
    },
    "node_comm_load": {
        "title": "Node Communication Load",
        "summary_keys": [
            ("node_count",    "Infrastructure Nodes", "d"),
            ("load_mean",     "Avg Load",             ".2f"),
            ("load_max",      "Max Load",             ".0f"),
            ("load_cv",       "Coeff. of Variation",  ".3f"),
            ("outlier_count", "Outliers (IQR)",       "d"),
        ],
    },
    "domain_comm": {
        "title": "Domain Communication Matrix",
        "summary_keys": [
            ("domain_count",        "Domains",              "d"),
            ("cross_domain_pairs",  "Cross-Domain Pairs",   "d"),
            ("total_cross_traffic", "Total Cross Traffic",  "d"),
            ("outlier_count",       "Outlier Pairs (IQR)",  "d"),
        ],
    },
    "criticality_io": {
        "title": "Criticality × I/O Load",
        "summary_keys": [
            ("total_apps",      "Total Apps",          "d"),
            ("crit_count",      "Critical Apps",       "d"),
            ("crit_pct",        "Critical %",          ".1f"),
            ("crit_io_mean",    "Critical Avg I/O",    ".2f"),
            ("crit_io_max",     "Critical Max I/O",    "d"),
            ("norm_io_mean",    "Normal Avg I/O",      ".2f"),
            ("crit_norm_ratio", "Crit/Normal Ratio",   ".3f"),
            ("outlier_count",   "Outliers (IQR)",      "d"),
        ],
    },
    "lib_dependency": {
        "title": "Library Dependency Density",
        "summary_keys": [
            ("total_libs",       "Total Libraries",      "d"),
            ("total_apps",       "App Consumers",        "d"),
            ("in_degree_mean",   "Avg In-Degree (libs)", ".2f"),
            ("in_degree_max",    "Max In-Degree (libs)", "d"),
            ("out_degree_mean",  "Avg Out-Degree (apps)",".2f"),
            ("outlier_count",    "Outliers (IQR)",       "d"),
        ],
    },
    "node_critical_density": {
        "title": "Node Critical Density",
        "summary_keys": [
            ("node_count",        "Infrastructure Nodes",  "d"),
            ("total_critical",    "Total Critical Apps",   "d"),
            ("density_mean",      "Avg Critical Density",  ".3f"),
            ("density_max",       "Max Critical Density",  ".3f"),
            ("high_density_nodes","High-Density Nodes",    "d"),
        ],
    },
    "domain_diversity": {
        "title": "Domain Diversity",
        "summary_keys": [
            ("css_count",    "Domains (CSS)",      "d"),
            ("app_mean",     "Avg Apps/Domain",    ".2f"),
            ("app_max",      "Max Apps/Domain",    "d"),
            ("topic_mean",   "Avg Topics/Domain",  ".2f"),
            ("topic_max",    "Max Topics/Domain",  "d"),
            ("io_mean",      "Avg I/O/Domain",     ".2f"),
            ("io_max",       "Max I/O/Domain",     "d"),
        ],
    },
    "qos_risk": {
        "title": "Topic QoS Risk Scatter",
        "summary_keys": [
            ("total_topics",    "Total Topics",     "d"),
            ("risk_mean",       "Avg Risk Score",   ".2f"),
            ("risk_std",        "Risk Std Dev",    ".2f"),
            ("outlier_count",   "Outliers (IQR)",   "d"),
        ],
    },
}

ALL_CHART_IDS = list(CHART_REGISTRY.keys())


# Display helpers
# ---------------------------------------------------------------------------
C = Colors  # shorthand

COL_LABEL = 28
COL_VALUE = 14

OUTLIER_WARN_THRESHOLD = 3  # highlight outlier_count in yellow at this value
OUTLIER_CRIT_THRESHOLD = 6  # highlight in red above this


def _kv(label: str, value: Any, fmt: Optional[str] = None) -> str:
    """Format a single key-value row."""
    if value is None:
        formatted = ConsoleDisplay.colored("n/a", C.GRAY)
    elif fmt and isinstance(value, (int, float)):
        try:
            formatted = format(value, fmt)
        except Exception:
            formatted = str(value)
    else:
        formatted = str(value)

    # Colour-code outlier counts
    if "outlier" in label.lower() and isinstance(value, int):
        if value >= OUTLIER_CRIT_THRESHOLD:
            formatted = ConsoleDisplay.colored(formatted, C.RED, bold=True)
        elif value >= OUTLIER_WARN_THRESHOLD:
            formatted = ConsoleDisplay.colored(formatted, C.YELLOW)
        else:
            formatted = ConsoleDisplay.colored(formatted, C.GREEN)

    return f"    {label:<{COL_LABEL}} {formatted}"


def _print_chart(display: ConsoleDisplay, chart_id: str, chart_data: Dict[str, Any]) -> None:
    """Render one chart section to stdout."""
    meta = CHART_REGISTRY[chart_id]
    display.print_subheader(meta["title"])

    summary = chart_data.get("summary", {})

    if not summary:
        print(ConsoleDisplay.colored("    (no data)", C.GRAY))
        return

    for key, label, fmt in meta["summary_keys"]:
        if key in summary:
            print(_kv(label, summary[key], fmt))

    # Generic fallback: print remaining summary keys not listed in registry
    listed = {k for k, _, __ in meta["summary_keys"]}
    extras = {k: v for k, v in summary.items() if k not in listed and not isinstance(v, (dict, list))}
    if extras:
        print()
        for k, v in extras.items():
            label = k.replace("_", " ").title()
            print(_kv(label, v))


def _print_chart_minimal(chart_id: str, chart_data: Dict[str, Any]) -> None:
    """One-line summary for minimal format."""
    meta = CHART_REGISTRY.get(chart_id, {})
    title = meta.get("title", chart_id)
    summary = chart_data.get("summary", {})

    parts = []
    for key, label, fmt in meta.get("summary_keys", []):
        if key in summary:
            v = summary[key]
            if fmt and isinstance(v, (int, float)):
                try:
                    parts.append(f"{label}={format(v, fmt)}")
                except Exception:
                    parts.append(f"{label}={v}")
            else:
                parts.append(f"{label}={v}")
        if len(parts) >= 4:
            break

    summary_str = "  |  ".join(parts) if parts else "(no data)"
    chart_label = ConsoleDisplay.colored(f"{title:<40}", C.CYAN)
    print(f"  {chart_label}  {summary_str}")


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------
def _load_from_file(path: Path) -> Dict[str, Any]:
    """Load raw graph JSON from a local file."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with open(path) as f:
        return json.load(f)


def _load_from_neo4j(uri: str, user: str, password: str) -> Dict[str, Any]:
    """Export raw graph JSON from a live Neo4j instance."""
    from saag import Client
    client = Client(neo4j_uri=uri, user=user, password=password)
    return client.repo.export_json()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and display cross-cutting statistics for a pub-sub graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data source
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--input", "-i",
        metavar="FILE",
        type=Path,
        help="Path to a pre-exported dataset JSON file (standalone mode, no Neo4j required).",
    )

    # Chart selection
    parser.add_argument(
        "--chart",
        nargs="+",
        metavar="CHART",
        choices=ALL_CHART_IDS,
        default=None,
        help=(
            "One or more chart IDs to compute and display. "
            "Defaults to all charts. "
            f"Available: {', '.join(ALL_CHART_IDS)}"
        ),
    )

    # Output format
    parser.add_argument(
        "--format",
        choices=["table", "minimal", "json"],
        default="table",
        help=(
            "Output format. "
            "'table' (default): rich per-chart tables. "
            "'minimal': one summary line per chart. "
            "'json': raw JSON dump (use with --output for machine-readable results)."
        ),
    )

    add_neo4j_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args)

    display = ConsoleDisplay()
    selected_charts: List[str] = args.chart or ALL_CHART_IDS

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    if args.format != "json":
        display.print_header("Graph Statistics")

        if args.input:
            display.print_step(f"Source: file  →  {args.input}")
        else:
            display.print_step(f"Source: Neo4j  →  {args.uri}")
        display.print_step(f"Charts:  {', '.join(selected_charts)}")

    # ------------------------------------------------------------------
    # Acquire raw data
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        if args.input:
            if args.format != "json":
                display.print_step(f"Loading {args.input.name}…")
            raw_data = _load_from_file(args.input)
        else:
            if args.format != "json":
                display.print_step("Exporting graph data from Neo4j…")
            raw_data = _load_from_neo4j(args.uri, args.user, args.password)
    except FileNotFoundError as exc:
        display.print_error(str(exc))
        sys.exit(1)
    except Exception as exc:
        display.print_error(f"Failed to load graph data: {exc}")
        if getattr(args, "verbose", False):
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------
    # Compute statistics
    # ------------------------------------------------------------------
    try:
        if args.format != "json":
            display.print_step("Computing statistics…")

        def default_risk_weight_fn(_, value: str) -> float:
            # Default weights for QoS risk scoring
            mapping = {"High": 3.0, "Medium": 2.0, "Low": 1.0, "NOT_FOUND": 1.0}
            return mapping.get(value, 1.0)

        cc = extract_cross_cutting_data(raw_data)
        all_stats = compute_all_extras_statistics(cc, risk_weight_fn=default_risk_weight_fn)
        all_stats = serialise_numpy(all_stats)

    except Exception as exc:
        display.print_error(f"Statistics computation failed: {exc}")
        if getattr(args, "verbose", False):
            import traceback
            traceback.print_exc()
        sys.exit(1)

    elapsed_ms = (time.time() - t0) * 1000

    # Filter to requested charts
    filtered_stats = {k: v for k, v in all_stats.items() if k in selected_charts}

    # ------------------------------------------------------------------
    # Render output
    # ------------------------------------------------------------------
    if args.format == "json":
        output_payload = {
            "charts": filtered_stats,
            "computation_time_ms": round(elapsed_ms, 1),
        }
        json_str = json.dumps(output_payload, indent=2, default=str)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json_str)
            display.print_success(f"Statistics saved to {args.output}")
        else:
            print(json_str)
        return

    if args.format == "minimal":
        display.print_subheader("Summary")
        for chart_id in selected_charts:
            if chart_id in filtered_stats:
                _print_chart_minimal(chart_id, filtered_stats[chart_id])
            else:
                print(f"  {chart_id:<40}  (no data)")
    else:
        # table mode
        for chart_id in selected_charts:
            if chart_id in filtered_stats:
                _print_chart(display, chart_id, filtered_stats[chart_id])
            else:
                display.print_subheader(CHART_REGISTRY.get(chart_id, {}).get("title", chart_id))
                print(ConsoleDisplay.colored("    (no data for this chart)", C.GRAY))

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    if args.format != "json":
        print()
        display.print_success(
            f"Computed {len(filtered_stats)} chart(s) in {elapsed_ms:.0f} ms"
        )

    # ------------------------------------------------------------------
    # Optional JSON save
    # ------------------------------------------------------------------
    if args.output:
        try:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "charts": filtered_stats,
                "computation_time_ms": round(elapsed_ms, 1),
            }
            output_path.write_text(json.dumps(payload, indent=2, default=str))
            display.print_step(f"Results saved → {args.output}")
        except Exception as exc:
            display.print_error(f"Could not write output file: {exc}")
            if getattr(args, "verbose", False):
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
