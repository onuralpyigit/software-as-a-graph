#!/usr/bin/env python3
"""
cli/predict_graph.py — Unified Prediction & Anti-Pattern CLI
=============================================================
Runs the full Step 3 prediction pipeline in one command:

  Step 3a  RMAV quality scoring  (always)
  Step 3b  GNN inference         (opt-in: --gnn-model PATH)
  Step 3c  Anti-pattern scan     (on by default; skip with --no-antipatterns)

Exit codes (CI/CD gate):
  0 — clean (no anti-patterns, or --no-antipatterns)
  1 — MEDIUM anti-patterns detected
  2 — HIGH or CRITICAL anti-patterns detected  → blocks deployment

Usage examples
--------------
  # Minimal — RMAV + antipatterns on system layer
  python cli/predict_graph.py

  # Multi-layer
  python cli/predict_graph.py --layer app,system

  # AHP-weighted RMAV + GNN ensemble
  python cli/predict_graph.py --use-ahp --gnn-model output/gnn_checkpoints/best

  # Strict CI gate — only CRITICAL patterns block
  python cli/predict_graph.py --severity critical --output-antipatterns results/ap.json

  # Filter to specific patterns
  python cli/predict_graph.py --pattern SPOF,FAILURE_HUB,GOD_COMPONENT

  # Baseline equal weights, no GNN, skip antipatterns
  python cli/predict_graph.py --equal-weights --no-antipatterns

  # Print the full pattern catalog and exit
  python cli/predict_graph.py --catalog
"""

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import argparse
from saag import Client
from saag.models import PredictionResult
from cli.common.arguments import add_neo4j_arguments, add_common_arguments, setup_logging
from cli.common.console import ConsoleDisplay

logger = logging.getLogger("predict_graph")


# ═══════════════════════════════════════════════════════════════════════════════
# Argument parser
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict_graph.py",
        description=(
            "Unified prediction CLI: RMAV scoring, optional GNN inference, "
            "and architectural anti-pattern detection in one step."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage examples")[1] if "Usage examples" in __doc__ else "",
    )

    # ── RMAV / weighting ──────────────────────────────────────────────────────
    weight_grp = parser.add_argument_group("Weighting (RMAV)")
    weight_grp.add_argument(
        "--use-ahp", action="store_true",
        help="Use AHP-derived dimension weights (recommended for thesis results)",
    )
    weight_grp.add_argument(
        "--equal-weights", action="store_true",
        help="Override to equal 0.25 per dimension — baseline ablation condition",
    )
    weight_grp.add_argument(
        "--ahp-shrinkage", type=float, default=0.7, metavar="λ",
        help="Shrinkage factor λ ∈ [0,1] blending AHP weights toward equal weights "
             "(default: 0.7). Ignored when --equal-weights is set.",
    )

    # ── GNN inference ─────────────────────────────────────────────────────────
    gnn_grp = parser.add_argument_group("GNN inference (Step 3b, optional)")
    gnn_grp.add_argument(
        "--gnn-model", metavar="PATH", default=None,
        help="Path to a trained GNN checkpoint directory. "
             "When provided, runs HeteroGAT inference and reports ensemble scores.",
    )

    # ── Anti-pattern detection ────────────────────────────────────────────────
    ap_grp = parser.add_argument_group("Anti-pattern detection")
    ap_grp.add_argument(
        "--no-antipatterns", action="store_true",
        help="Skip anti-pattern detection entirely. "
             "Exit code is always 0 when this flag is set.",
    )
    ap_grp.add_argument(
        "--severity", metavar="LEVELS", default=None,
        help="Comma-separated severity filter for reporting and exit-code logic. "
             "Accepted values: critical, high, medium  (default: all three). "
             "Example: --severity critical,high",
    )
    ap_grp.add_argument(
        "--pattern", metavar="IDS", default=None,
        help="Comma-separated pattern IDs to run (default: full catalog). "
             "Example: --pattern SPOF,FAILURE_HUB,GOD_COMPONENT,CYCLIC_DEPENDENCY",
    )
    ap_grp.add_argument(
        "--catalog", action="store_true",
        help="Print the full anti-pattern catalog (ID, severity, category, description) "
             "and exit. No analysis is run.",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    out_grp = parser.add_argument_group("Output")
    out_grp.add_argument(
        "--output-antipatterns", metavar="FILE", default=None,
        help="Write the anti-pattern report to a separate JSON file. "
             "This file feeds --antipatterns in visualize_graph.py.",
    )
    out_grp.add_argument(
        "--no-exit-code", action="store_true",
        help="Always exit with code 0 (disables CI/CD blocking behaviour).",
    )

    add_neo4j_arguments(parser)
    add_common_arguments(parser)  # adds --layer, --output, --verbose, --quiet
    return parser


# ═══════════════════════════════════════════════════════════════════════════════
# Catalog printer
# ═══════════════════════════════════════════════════════════════════════════════

def print_catalog(display: ConsoleDisplay) -> None:
    """Print the full anti-pattern catalog and exit."""
    try:
        from saag.analysis.antipattern_detector import CATALOG
    except ImportError as exc:
        display.print_error(f"Cannot load anti-pattern catalog: {exc}")
        sys.exit(1)

    _SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    entries = sorted(CATALOG.values(), key=lambda s: (_SEV_ORDER.get(s.severity, 9), s.id))

    display.print_header("Anti-Pattern Catalog")

    current_sev = None
    for spec in entries:
        if spec.severity != current_sev:
            current_sev = spec.severity
            display.print_step(f"── {current_sev} ──")
        print(f"  {spec.id:<22}  [{spec.category:<16}]  {spec.description[:72]}")
        print(f"  {'':22}  Risk:    {spec.risk[:72]}")
        print(f"  {'':22}  Fix:     {spec.recommendation[:72]}")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# Blast-radius / cascade-depth helpers  (Issue #2)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_propagation_metrics(nx_graph) -> dict[str, dict]:
    """
    For each node, compute:
      blast_radius  — number of nodes reachable from this node in the DEPENDS_ON graph
      cascade_depth — length of the longest directed path from this node
    Both are O(V+E) amortised via a single DFS per source.
    Returns {node_id: {"blast_radius": int, "cascade_depth": int}}.
    """
    try:
        import networkx as nx
    except ImportError:
        return {}

    result: dict[str, dict] = {}
    for node in nx_graph.nodes():
        reachable = nx.descendants(nx_graph, node)
        blast_radius = len(reachable)
        # Longest path on the ego sub-DAG
        sub = nx_graph.subgraph(reachable | {node})
        try:
            cascade_depth = nx.dag_longest_path_length(sub)
        except (nx.NetworkXError, nx.NetworkXUnfeasible):
            # Graph has cycles in the ego subgraph — use simple BFS depth instead
            lengths = nx.single_source_shortest_path_length(nx_graph, node)
            cascade_depth = max(lengths.values()) if lengths else 0
        result[node] = {"blast_radius": blast_radius, "cascade_depth": cascade_depth}
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Anti-pattern detection helper (supports --pattern filter)
# ═══════════════════════════════════════════════════════════════════════════════

def run_antipattern_detection(
    prediction,
    layer: str,
    active_patterns: list[str] | None,
    severity_filter: set[str] | None,
) -> list:
    """
    Run the anti-pattern detector against a PredictionResult.
    Uses AntiPatternDetector directly so --pattern and --severity filtering
    can be applied at the engine level rather than as a post-filter.

    Falls back to client.detect_antipatterns() when no pattern filter is set,
    preserving backward compatibility.
    """
    try:
        from saag.analysis.antipattern_detector import AntiPatternDetector
    except ImportError as exc:
        logger.error("AntiPatternDetector not available: %s", exc)
        return []

    quality = prediction.raw
    layer_name = getattr(quality, "layer", layer)
    if hasattr(layer_name, "value"):
        layer_name = layer_name.value

    # Build the shim expected by AntiPatternDetector.detect()
    shim = SimpleNamespace(
        quality=quality,
        components=quality.components,
        edges=getattr(quality, "edges", []),
    )

    detector = AntiPatternDetector(active_patterns=active_patterns)
    problems = detector.detect(shim, layer_name)

    # Apply severity filter (post-detection, preserves early-exit from detector)
    if severity_filter:
        problems = [p for p in problems if p.severity.lower() in severity_filter]

    return problems


# ═══════════════════════════════════════════════════════════════════════════════
# GNN inference helper
# ═══════════════════════════════════════════════════════════════════════════════

def run_gnn_inference(client: Client, nx_graph, analysis, layer: str, gnn_model: str, display: ConsoleDisplay):
    """
    Run GNN inference using a trained checkpoint.
    Returns a GNNAnalysisResult or None on failure.
    """
    try:
        from saag.prediction import GNNService, extract_structural_metrics_dict, extract_rmav_scores_dict
    except ImportError as exc:
        display.print_error(f"GNN module not available (PyTorch Geometric required): {exc}")
        return None

    try:
        structural_dict = extract_structural_metrics_dict(analysis.raw)
        rmav_dict = extract_rmav_scores_dict(analysis.raw.quality)

        gnn_svc = GNNService.from_checkpoint(gnn_model, graph=nx_graph)
        return gnn_svc.predict(
            graph=nx_graph,
            structural_metrics=structural_dict,
            rmav_scores=rmav_dict,
        )
    except Exception as exc:
        display.print_error(f"GNN inference failed: {exc}")
        logger.debug("GNN inference traceback", exc_info=True)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# RMAV dimension display helper  (Issue #4)
# ═══════════════════════════════════════════════════════════════════════════════

_RMAV_LABELS = {
    "reliability":      ("R", "Cascade / reliability risk"),
    "maintainability":  ("M", "Coupling / change fragility"),
    "availability":     ("A", "SPOF / availability loss"),
    "vulnerability":    ("V", "Outbound blast radius"),
}

def display_rmav_breakdown(components: list, top_n: int = 10) -> None:
    """
    Print a ranked table of components with per-RMAV dimension scores and the
    dominant risk dimension — so maintainability concerns (high M) are
    distinguished from SPOF concerns (high A) at a glance.
    """
    if not components:
        return

    # Sort by composite score descending
    ranked = sorted(components, key=lambda c: c.scores.overall, reverse=True)[:top_n]

    print()
    print(f"  {'Rank':<4} {'Component':<32} {'Q':>5}  {'R':>5}  {'M':>5}  {'A':>5}  {'V':>5}  {'Dominant risk':<28}  {'SPOF'}")
    print(f"  {'─'*4} {'─'*32} {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*28}  {'─'*4}")

    for rank, comp in enumerate(ranked, 1):
        s = comp.scores
        dim_scores = {
            "reliability":     s.reliability,
            "maintainability": s.maintainability,
            "availability":    s.availability,
            "vulnerability":   s.vulnerability,
        }
        dominant_dim = max(dim_scores, key=dim_scores.get)
        _, dominant_label = _RMAV_LABELS[dominant_dim]
        is_spof = getattr(comp.structural, "is_articulation_point", False)
        spof_mark = "  ✗" if is_spof else ""

        print(
            f"  {rank:<4} {str(comp.id)[:31]:<32} "
            f"{s.overall:>5.3f}  {s.reliability:>5.3f}  {s.maintainability:>5.3f}  "
            f"{s.availability:>5.3f}  {s.vulnerability:>5.3f}  "
            f"{dominant_label:<28}  {spof_mark}"
        )
    print()


def display_propagation_metrics(components: list, prop_metrics: dict, top_n: int = 10) -> None:
    """
    Print blast-radius and cascade-depth for the top-ranked components
    by composite Q score.  (Issue #2)
    """
    if not components or not prop_metrics:
        return

    ranked = sorted(components, key=lambda c: c.scores.overall, reverse=True)[:top_n]

    print()
    print(f"  {'Component':<32} {'Q':>5}  {'Blast radius':>12}  {'Cascade depth':>13}")
    print(f"  {'─'*32} {'─'*5}  {'─'*12}  {'─'*13}")
    for comp in ranked:
        pm = prop_metrics.get(comp.id, {})
        print(
            f"  {str(comp.id)[:31]:<32} {comp.scores.overall:>5.3f}  "
            f"{pm.get('blast_radius', '?'):>12}  {pm.get('cascade_depth', '?'):>13}"
        )
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args)
    display = ConsoleDisplay()

    # ── --catalog: print catalog and exit ─────────────────────────────────────
    if args.catalog:
        print_catalog(display)
        sys.exit(0)

    # ── Validate conflicting weight flags ─────────────────────────────────────
    if args.equal_weights and args.use_ahp:
        parser.error("--equal-weights and --use-ahp are mutually exclusive.")

    # ── Parse layers (multi-layer support)  (Issue #3) ───────────────────────
    layers = [l.strip() for l in args.layer.split(",") if l.strip()]
    if not layers:
        layers = ["system"]

    # ── Parse --severity filter ───────────────────────────────────────────────
    severity_filter: set[str] | None = None
    if args.severity:
        severity_filter = {s.strip().upper() for s in args.severity.split(",")}
        valid_sevs = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
        unknown_sevs = severity_filter - valid_sevs
        if unknown_sevs:
            parser.error(f"Unknown severity values: {', '.join(sorted(unknown_sevs))}. "
                         f"Choose from: {', '.join(sorted(valid_sevs))}")
        # Normalise back to lowercase for comparison later
        severity_filter = {s.lower() for s in severity_filter}

    # ── Parse --pattern filter ────────────────────────────────────────────────
    active_patterns: list[str] | None = None
    if args.pattern:
        active_patterns = [p.strip().upper() for p in args.pattern.split(",") if p.strip()]
        try:
            from saag.analysis.antipattern_detector import CATALOG
            unknown_pats = set(active_patterns) - set(CATALOG.keys())
            if unknown_pats:
                parser.error(
                    f"Unknown pattern IDs: {', '.join(sorted(unknown_pats))}. "
                    f"Run --catalog to see all available patterns."
                )
        except ImportError:
            pass  # Will fail later with a cleaner error

    # ── Header ────────────────────────────────────────────────────────────────
    layer_label = ", ".join(l.upper() for l in layers)
    mode_parts = []
    if args.use_ahp:
        mode_parts.append(f"AHP λ={args.ahp_shrinkage}")
    elif args.equal_weights:
        mode_parts.append("equal weights")
    else:
        mode_parts.append("default weights")
    if args.gnn_model:
        mode_parts.append("GNN ensemble")
    if not args.no_antipatterns:
        if args.pattern:
            mode_parts.append(f"patterns: {args.pattern}")
        else:
            mode_parts.append("full anti-pattern scan")

    display.print_header(f"Prediction — {layer_label}  [{' · '.join(mode_parts)}]")

    # ── Connect ───────────────────────────────────────────────────────────────
    client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)

    # ── Per-layer results accumulator ─────────────────────────────────────────
    all_problems: list = []
    all_output: dict = {"layers": {}}

    for layer in layers:
        display.print_step(f"[{layer.upper()}] Structural analysis…")

        analysis = client.analyze(
            layer=layer,
            equal_weights=args.equal_weights,
            use_ahp=args.use_ahp,
            ahp_shrinkage=args.ahp_shrinkage,
        )

        # ── RMAV prediction ──────────────────────────────────────────────────
        display.print_step(f"[{layer.upper()}] RMAV quality scoring…")
        # Quality scores are already computed inside client.analyze(); wrap for uniform access.
        prediction = PredictionResult(analysis.raw.quality)

        components = prediction.raw.components if prediction.raw else []
        total_components = len(components)

        # ── Blast-radius / cascade-depth  (Issue #2) ─────────────────────────
        nx_graph = getattr(analysis.raw, "graph", None)
        prop_metrics: dict = {}
        if nx_graph is not None:
            display.print_step(f"[{layer.upper()}] Computing failure propagation metrics…")
            prop_metrics = compute_propagation_metrics(nx_graph)

        # ── RMAV breakdown display  (Issue #4) ───────────────────────────────
        display.print_step(f"[{layer.upper()}] Top components by RMAV score:")
        display_rmav_breakdown(components, top_n=10)

        # ── Propagation metrics display  (Issue #2) ──────────────────────────
        if prop_metrics:
            display.print_step(f"[{layer.upper()}] Failure propagation (blast radius / cascade depth):")
            display_propagation_metrics(components, prop_metrics, top_n=10)

        # ── GNN inference (Step 3b, optional) ────────────────────────────────
        gnn_result = None
        if args.gnn_model:
            display.print_step(f"[{layer.upper()}] GNN inference from checkpoint: {args.gnn_model}")
            gnn_result = run_gnn_inference(client, nx_graph, analysis, layer, args.gnn_model, display)
            if gnn_result:
                top_nodes = gnn_result.top_critical_nodes(n=10)
                print()
                print(f"  GNN / ensemble top-10 components:")
                print(f"  {'Rank':<4} {'Component':<32} {'Q_ens':>6}  {'R':>6}  {'M':>6}  {'A':>6}  {'V':>6}  {'Source'}")
                print(f"  {'─'*4} {'─'*32} {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*10}")
                for rank, ns in enumerate(top_nodes, 1):
                    print(
                        f"  {rank:<4} {str(ns.component)[:31]:<32} "
                        f"{ns.composite_score:>6.3f}  {ns.reliability_score:>6.3f}  "
                        f"{ns.maintainability_score:>6.3f}  {ns.availability_score:>6.3f}  "
                        f"{ns.vulnerability_score:>6.3f}  {ns.source}"
                    )
                    if gnn_result.ensemble_alpha:
                        alpha_str = "  ".join(f"{a:.2f}" for a in gnn_result.ensemble_alpha)
                        print(f"\n  Ensemble α per dimension (Q R M A V): {alpha_str}")
                print()

        # ── Anti-pattern detection  (Issue #1, #5, #6, #7) ───────────────────
        layer_problems: list = []
        if not args.no_antipatterns:
            display.print_step(f"[{layer.upper()}] Anti-pattern scan…")
            layer_problems = run_antipattern_detection(
                prediction=prediction,
                layer=layer,
                active_patterns=active_patterns,
                severity_filter=severity_filter,
            )
            all_problems.extend(layer_problems)

            display.display_antipatterns(layer_problems, [layer], total_components)
        else:
            display.print_step(f"[{layer.upper()}] Anti-pattern scan skipped (--no-antipatterns).")

        # ── Accumulate layer output ───────────────────────────────────────────
        layer_entry: dict = {
            "total_components": total_components,
            "rmav": {
                c.id: {
                    "overall":          c.scores.overall,
                    "reliability":      c.scores.reliability,
                    "maintainability":  c.scores.maintainability,
                    "availability":     c.scores.availability,
                    "vulnerability":    c.scores.vulnerability,
                    "is_spof":          getattr(c.structural, "is_articulation_point", False),
                    **prop_metrics.get(c.id, {}),
                }
                for c in components
            },
            "antipatterns": [p.to_dict() for p in layer_problems],
        }
        if gnn_result:
            layer_entry["gnn"] = gnn_result.to_dict()
        all_output["layers"][layer] = layer_entry

    # ── Persist combined output ────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(all_output, fh, indent=2, default=str)
        display.print_success(f"Full prediction report saved → {args.output}")

    # ── Persist antipattern-only output (for visualize_graph.py) ─────────────
    if args.output_antipatterns and not args.no_antipatterns:
        ap_path = Path(args.output_antipatterns)
        ap_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ap_path, "w") as fh:
            json.dump([p.to_dict() for p in all_problems], fh, indent=2)
        display.print_success(f"Anti-pattern report saved → {args.output_antipatterns}")

    # ── Summary line ──────────────────────────────────────────────────────────
    if not args.no_antipatterns:
        n_critical = sum(1 for p in all_problems if p.severity == "CRITICAL")
        n_high     = sum(1 for p in all_problems if p.severity == "HIGH")
        n_medium   = sum(1 for p in all_problems if p.severity == "MEDIUM")
        print()
        print(f"  Anti-pattern summary: {len(all_problems)} total  "
              f"({n_critical} CRITICAL  {n_high} HIGH  {n_medium} MEDIUM)")

    # ── CI/CD exit codes  (Issue #6) ─────────────────────────────────────────
    if args.no_antipatterns or args.no_exit_code:
        sys.exit(0)

    # Determine worst severity found (respecting --severity filter)
    severities_found = {p.severity for p in all_problems}
    if "CRITICAL" in severities_found or "HIGH" in severities_found:
        display.print_error(
            "DEPLOYMENT GATE: HIGH or CRITICAL anti-patterns detected. "
            "Resolve before releasing."
        )
        sys.exit(2)
    elif "MEDIUM" in severities_found:
        display.print_step(
            "WARNING: MEDIUM anti-patterns detected. "
            "Deployment allowed; architectural debt review recommended."
        )
        sys.exit(1)
    else:
        display.print_success("No anti-patterns detected. Prediction complete.")
        sys.exit(0)


if __name__ == "__main__":
    main()