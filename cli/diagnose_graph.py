#!/usr/bin/env python3
"""
cli/diagnose_graph.py — Diagnose & Anti-Pattern CLI (Step 4)
==============================================================
Runs the Diagnose stage in one command: deterministic ISO-RM root-cause
attribution, anti-pattern detection, natural-language explanation, and
(optionally) the Triage Bridge. Requires no GNN checkpoint — this is the
zero-GNN cold-start path; for Step 3 (GNN blast-radius ranking),
see `saag-predict`.

Exit codes (CI/CD gate):
  0 — clean (no anti-patterns, or --no-antipatterns)
  1 — MEDIUM anti-patterns detected
  2 — HIGH or CRITICAL anti-patterns detected  → blocks deployment

Usage examples
--------------
  # Minimal — RM scoring + antipatterns on system layer
  python cli/diagnose_graph.py

  # Multi-layer
  python cli/diagnose_graph.py --layer app,system

  # AHP-weighted RM + Triage bridge, grouped by stakeholder
  python cli/diagnose_graph.py --use-ahp --triage-k 10 --by-stakeholder

  # Strict CI gate — only CRITICAL patterns block
  python cli/diagnose_graph.py --severity critical --output-antipatterns results/ap.json

  # Filter to specific patterns
  python cli/diagnose_graph.py --pattern SPOF,FAILURE_HUB,GOD_COMPONENT

  # Baseline equal weights, skip antipatterns
  python cli/diagnose_graph.py --equal-weights --no-antipatterns

  # Print the full pattern catalog and exit
  python cli/diagnose_graph.py --catalog
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path to support direct execution (python cli/diagnose_graph.py)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from saag import Client
from cli.common.arguments import add_neo4j_arguments, add_common_arguments, setup_logging
from cli.common.console import ConsoleDisplay

logger = logging.getLogger("diagnose_graph")


# ═══════════════════════════════════════════════════════════════════════════════
# Argument parser
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diagnose_graph.py",
        description=(
            "Diagnose CLI: RM root-cause attribution, anti-pattern detection, "
            "explanation, and the Triage Bridge (Step 4) — no GNN "
            "checkpoint required."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage examples")[1] if "Usage examples" in __doc__ else "",
    )

    # ── RM / weighting ──────────────────────────────────────────────────────
    weight_grp = parser.add_argument_group("Weighting (RM)")
    weight_grp.add_argument(
        "--use-ahp", action="store_true",
        help="Use AHP-derived dimension weights (recommended for thesis results)",
    )
    weight_grp.add_argument(
        "--equal-weights", action="store_true",
        help="Override to q_reliability=q_maintainability=0.5 and r_alpha=0.5 "
             "(equal at every level of the RM composite) — baseline ablation condition",
    )
    weight_grp.add_argument(
        "--ahp-shrinkage", type=float, default=0.7, metavar="λ",
        help="Shrinkage factor λ ∈ [0,1] blending AHP weights toward equal weights "
             "(default: 0.7). Ignored when --equal-weights is set.",
    )
    weight_grp.add_argument(
        "--norm", type=str, choices=["robust", "minmax", "zscore", "rank"], default="robust",
        help="Normalization applied to Tier-1 metrics before the RM weighted sum "
             "(default: robust, i.e. rank-based)",
    )
    weight_grp.add_argument(
        "--winsorize", action="store_true",
        help="Cap raw metric values above the 95th percentile before normalization",
    )
    weight_grp.add_argument(
        "--sensitivity", action="store_true",
        help="Run Kendall τ weight sensitivity analysis after scoring",
    )

    # ── Triage bridge ────────────────────────────────────────────────────────
    triage_grp = parser.add_argument_group("Triage bridge (optional)")
    triage_grp.add_argument(
        "--triage-k", type=int, default=None, metavar="K",
        help="Shortlist the Top-K critical components by RM score and print "
             "each one's root-cause diagnosis: pattern, elevated dimensions, "
             "priority action, and stakeholder roles.",
    )
    triage_grp.add_argument(
        "--by-stakeholder", action="store_true",
        help="Group triage remediation actions by stakeholder role (DevOps/SRE, Architect, Developer).",
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
             "Example: --pattern SPOF,FAILURE_HUB,GOD_COMPONENT,CYCLE",
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
# RM dimension display helper
# ═══════════════════════════════════════════════════════════════════════════════

_RM_LABELS = {
    "fault_tolerance":  "Cascade / propagation risk",
    "availability":     "SPOF / availability loss",
    "maintainability":  "Coupling / change fragility",
}

def display_rm_breakdown(components: list, top_n: int = 10) -> None:
    """
    Print a ranked table of components with per-RM dimension scores and the
    dominant risk dimension — so maintainability concerns (high M) are
    distinguished from SPOF concerns (high A) at a glance.

    The dominant-dimension max() runs over Fault Tolerance, Availability and
    Maintainability — the three orthogonal signals — not Reliability itself,
    which is their weighted combination and would double-count.
    """
    if not components:
        return

    # Sort by composite score descending
    ranked = sorted(components, key=lambda c: c.scores.overall, reverse=True)[:top_n]

    print()
    print(f"  {'Rank':<4} {'Component':<32} {'Q':>5}  {'R':>5}  {'M':>5}  {'FT':>5}  {'A':>5}  {'Dominant risk':<28}  {'SPOF'}")
    print(f"  {'─'*4} {'─'*32} {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*28}  {'─'*4}")

    for rank, comp in enumerate(ranked, 1):
        s = comp.scores
        dim_scores = {
            "fault_tolerance": s.fault_tolerance,
            "availability":    s.availability,
            "maintainability": s.maintainability,
        }
        dominant_dim = max(dim_scores, key=dim_scores.get)
        dominant_label = _RM_LABELS[dominant_dim]
        is_spof = getattr(comp.structural, "is_articulation_point", False)
        spof_mark = "  ✗" if is_spof else ""

        print(
            f"  {rank:<4} {str(comp.id)[:31]:<32} "
            f"{s.overall:>5.3f}  {s.reliability:>5.3f}  {s.maintainability:>5.3f}  "
            f"{s.fault_tolerance:>5.3f}  {s.availability:>5.3f}  "
            f"{dominant_label:<28}  {spof_mark}"
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

    # ── Parse layers (multi-layer support) ────────────────────────────────────
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
    if not args.no_antipatterns:
        if args.pattern:
            mode_parts.append(f"patterns: {args.pattern}")
        else:
            mode_parts.append("full anti-pattern scan")

    display.print_header(f"Diagnosis — {layer_label}  [{' · '.join(mode_parts)}]")

    # ── Connect ───────────────────────────────────────────────────────────────
    client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)

    # ── Per-layer results accumulator ─────────────────────────────────────────
    all_problems: list = []
    all_failed_patterns: list[str] = []
    all_output: dict = {"layers": {}}

    for layer in layers:
        display.print_step(f"[{layer.upper()}] Structural analysis…")
        analysis = client.analyze(layer=layer)

        display.print_step(f"[{layer.upper()}] RM quality scoring & diagnosis…")
        diagnosis = client.diagnose(
            analysis,
            k=args.triage_k,
            detect_problems=not args.no_antipatterns,
            active_patterns=active_patterns,
            run_sensitivity=args.sensitivity,
            use_ahp=args.use_ahp,
            equal_weights=args.equal_weights,
            ahp_shrinkage=args.ahp_shrinkage,
            normalization_method=args.norm,
            winsorize=args.winsorize,
        )

        components = diagnosis.raw.components if diagnosis.raw else []
        total_components = len(components)

        # ── RM breakdown display ────────────────────────────────────────────
        display.print_step(f"[{layer.upper()}] Top components by RM score:")
        display_rm_breakdown(components, top_n=10)

        # ── Anti-pattern detection ────────────────────────────────────────────
        layer_problems: list = []
        if not args.no_antipatterns:
            display.print_step(f"[{layer.upper()}] Anti-pattern scan…")
            layer_problems = diagnosis.problems
            if severity_filter:
                layer_problems = [p for p in layer_problems if p.severity.lower() in severity_filter]
            layer_failed_patterns = getattr(diagnosis.raw, "failed_patterns", []) or []

            all_problems.extend(layer_problems)
            all_failed_patterns.extend(layer_failed_patterns)
            if layer_failed_patterns:
                display.print_error(
                    f"[{layer.upper()}] {len(layer_failed_patterns)} detector(s) "
                    f"crashed and were skipped: {', '.join(layer_failed_patterns)}"
                )

            display.display_antipatterns(layer_problems, [layer], total_components)
        else:
            display.print_step(f"[{layer.upper()}] Anti-pattern scan skipped (--no-antipatterns).")

        # ── Triage bridge (optional) ──────────────────────────────────────────
        triage_result = diagnosis.triage
        if triage_result:
            print()
            print(f"  Triage shortlist (ranking source: {triage_result.ranking_source}):")
            print(f"  {'Rank':<4} {'Component':<32} {'Pattern':<22} {'Level':<9} {'Roles'}")
            print(f"  {'─'*4} {'─'*32} {'─'*22} {'─'*9} {'─'*20}")
            for entry in triage_result.entries:
                print(
                    f"  {entry.rank:<4} {str(entry.component_id)[:31]:<32} "
                    f"{entry.pattern:<22} {entry.level:<9} {', '.join(entry.roles)}"
                )
            print()

            if args.by_stakeholder:
                from api.presenters.triage_presenter import categorize_by_stakeholder
                stakeholder_view = categorize_by_stakeholder(triage_result)
                print(f"  Remediation Actions by Stakeholder Role:")
                print(f"  {'─'*60}")
                for role_key, group in stakeholder_view["stakeholders"].items():
                    print(f"  ► {group['role_name']} ({group['count']} components):")
                    print(f"    Focus: {group['focus']}")
                    for item in group["items"]:
                        if item["priority_action"]:
                            print(f"    • {item['component_id']} ({item['criticality_level']}): {item['priority_action']}")
                    print()

        # ── Accumulate layer output ───────────────────────────────────────────
        layer_entry: dict = {
            "total_components": total_components,
            "rm": {
                c.id: {
                    "overall":          c.scores.overall,
                    "reliability":      c.scores.reliability,
                    "maintainability":  c.scores.maintainability,
                    "fault_tolerance":  c.scores.fault_tolerance,
                    "availability":     c.scores.availability,
                    "is_spof":          getattr(c.structural, "is_articulation_point", False),
                }
                for c in components
            },
            "antipatterns": [p.to_dict() for p in layer_problems],
        }
        if triage_result:
            layer_entry["triage"] = triage_result.to_dict()
        all_output["layers"][layer] = layer_entry

    # ── Persist combined output ────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(all_output, fh, indent=2, default=str)
        display.print_success(f"Full diagnosis report saved → {args.output}")

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

    # ── CI/CD exit codes ─────────────────────────────────────────────────────
    if args.no_antipatterns or args.no_exit_code:
        sys.exit(0)

    # A crashed detector can silently miss a CRITICAL/HIGH finding it would
    # otherwise have surfaced — "no findings for this pattern" must not look
    # the same as a clean scan when the truth is "the scan didn't run".
    if all_failed_patterns:
        display.print_error(
            f"DEPLOYMENT GATE: {len(all_failed_patterns)} anti-pattern detector(s) "
            f"crashed: {', '.join(sorted(set(all_failed_patterns)))}. "
            "Findings for these patterns are incomplete."
        )
        sys.exit(2)

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
        display.print_success("No anti-patterns detected. Diagnosis complete.")
        sys.exit(0)


if __name__ == "__main__":
    main()
