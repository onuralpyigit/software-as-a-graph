#!/usr/bin/env python3
"""
bin/detect_antipatterns.py — Pub-Sub Architectural Anti-Pattern & Bad Smell Detector
=====================================================================================
Catalogs and detects architectural anti-patterns and bad smells in distributed
publish-subscribe systems using graph topology analysis.

Each finding tells you:
  • WHAT kind of architectural risk is present (named pattern)
  • WHICH components are involved
  • WHY it is dangerous (root-cause explanation)
  • HOW to fix it (concrete refactoring recommendation)
  • WHICH RMAV quality dimension it primarily degrades

Anti-Pattern Catalog (12 patterns across 3 severity tiers):

  CRITICAL ─────────────────────────────────────────────────────────────────
    SPOF               Single Point of Failure — structural graph cut vertex
    SYSTEMIC_RISK      Correlated failure cluster — CRITICAL clique
    CYCLIC_DEPENDENCY  Circular pub-sub feedback loop (SCC > 1)

  HIGH ─────────────────────────────────────────────────────────────────────
    GOD_COMPONENT      Dependency magnet — absorbs too many responsibilities
    BOTTLENECK_EDGE    High-traffic bridge with no redundant path
    BROKER_OVERLOAD    Broker saturation — disproportionate routing share
    DEEP_PIPELINE      Excessive processing chain depth — latency amplifier

  MEDIUM ───────────────────────────────────────────────────────────────────
    TOPIC_FANOUT       Topic fan-out explosion — broadcast blast radius
    CHATTY_PAIR        Bidirectional tight coupling through topics
    QOS_MISMATCH       Publisher/subscriber QoS incompatibility
    ORPHANED_TOPIC     Topic with no publishers OR no subscribers
    UNSTABLE_INTERFACE High churn potential — extreme coupling imbalance

Usage:
    # Detect all patterns in the system layer
    python bin/detect_antipatterns.py --layer system

    # Detect patterns in the application layer, show only CRITICAL/HIGH
    python bin/detect_antipatterns.py --layer app --severity critical,high

    # Detect a specific subset of patterns
    python bin/detect_antipatterns.py --all --pattern spof,broker_overload,cyclic_dependency

    # Print the full anti-pattern catalog (no Neo4j needed)
    python bin/detect_antipatterns.py --catalog

    # Export findings to JSON for downstream tooling
    python bin/detect_antipatterns.py --layer system --output results/smells.json

    # Full scan across all layers, verbose, export JSON
    python bin/detect_antipatterns.py --all --output results/smells.json --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap — allow running from project root as `python bin/<script>`
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from src.core import create_repository, AnalysisLayer
from src.analysis import (
    AnalysisService,
    SmellDetector,
    CATALOG,
    PatternSpec,
    DetectedSmell,
    SmellReport
)
from src.cli.console import ConsoleDisplay

logger = logging.getLogger("detect_antipatterns")


# Logic moved to src.analysis.smells
# Logic moved to src.analysis.smells


# =============================================================================
# Console Display Helpers
# =============================================================================

class SmellConsoleDisplay:
    """Rich terminal rendering for smell detection results."""

    SEVERITY_COLORS = {
        "CRITICAL": "\033[91m",   # bright red
        "HIGH":     "\033[93m",   # yellow
        "MEDIUM":   "\033[94m",   # blue
    }
    RMAV_ICONS = {
        "Reliability":     "R",
        "Maintainability": "M",
        "Availability":    "A",
        "Vulnerability":   "V",
    }
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    GRAY   = "\033[90m"
    DIM    = "\033[2m"

    def _c(self, text: str, color: str) -> str:
        return f"{color}{text}{self.RESET}"

    def _severity_color(self, sev: str) -> str:
        return self.SEVERITY_COLORS.get(sev, "")

    def print_banner(self) -> None:
        line = "═" * 70
        print(f"\n{self._c(line, self.CYAN)}")
        print(self._c(
            "  Software-as-a-Graph  ·  Pub-Sub Anti-Pattern & Bad Smell Detector",
            self.CYAN + self.BOLD))
        print(f"{self._c(line, self.CYAN)}\n")

    def print_catalog(self) -> None:
        """Print the full anti-pattern catalog in a readable format."""
        self.print_banner()
        print(self._c("  ANTI-PATTERN CATALOG  (12 patterns across 3 severity tiers)\n",
                      self.BOLD))
        for tier, sev in [("CRITICAL", "CRITICAL"), ("HIGH", "HIGH"), ("MEDIUM", "MEDIUM")]:
            color = self._severity_color(sev)
            print(self._c(f"  ── {tier} ──────────────────────────────────────────────────────",
                          color))
            for pid, spec in CATALOG.items():
                if spec.severity != sev:
                    continue
                rmav_icon = self.RMAV_ICONS.get(spec.rmav_dimension, "?")
                print(f"\n  {self._c(f'[{pid}]', color + self.BOLD)}"
                      f"  {self._c(spec.name, self.BOLD)}"
                      f"  {self._c(f'[RMAV:{rmav_icon}]', self.GRAY)}")
                # Word-wrap description
                words = spec.description.split()
                line_buf, col = "  ", 2
                for w in words:
                    if col + len(w) + 1 > 72:
                        print(self._c(line_buf, self.GRAY))
                        line_buf, col = "  ", 2
                    line_buf += w + " "
                    col += len(w) + 1
                if line_buf.strip():
                    print(self._c(line_buf, self.GRAY))
                print(f"  {self._c('Recommendation:', self.BOLD)} "
                      f"{spec.recommendation[:120]}...")
            print()

    def print_report(self, report: SmellReport, severity_filter: Optional[List[str]] = None) -> None:
        """Print the full detection report to stdout."""
        self.print_banner()

        # ── Summary KPIs ──────────────────────────────────────────────────
        print(self._c("  SCAN SUMMARY", self.BOLD))
        print(f"  Layers analyzed:     {', '.join(report.layers_analyzed)}")
        print(f"  Components scanned:  {report.total_components}")
        print(f"  Total smells found:  {self._c(str(report.total_smells), self.BOLD)}")
        print()
        for sev in ("CRITICAL", "HIGH", "MEDIUM"):
            count = report.by_severity.get(sev, 0)
            color = self._severity_color(sev)
            bar = self._c("█" * count, color) if count else self._c("─", self.GRAY)
            print(f"  {self._c(f'{sev:<10}', color)}  {bar}  {self._c(str(count), color + self.BOLD)}")
        print()

        # ── Pattern breakdown ─────────────────────────────────────────────
        if report.by_pattern:
            print(self._c("  BY PATTERN", self.BOLD))
            for pid, count in sorted(report.by_pattern.items(),
                                     key=lambda x: -x[1]):
                spec = CATALOG.get(pid)
                sev_color = self._severity_color(spec.severity) if spec else ""
                label = f"{pid:<24}" if spec else f"{pid:<24}"
                print(f"  {self._c(label, sev_color)}  {count}")
            print()

        # ── Individual findings ───────────────────────────────────────────
        smells = report.smells
        if severity_filter:
            sf = [s.upper() for s in severity_filter]
            smells = [s for s in smells if s.severity in sf]

        if not smells:
            print(self._c("  ✓  No smells found matching the active filters.\n", self.GREEN))
            return

        print(self._c(f"  FINDINGS ({len(smells)})", self.BOLD))
        print()

        prev_sev = None
        for i, smell in enumerate(smells, 1):
            if smell.severity != prev_sev:
                color = self._severity_color(smell.severity)
                print(self._c(
                    f"  {'─' * 3} {smell.severity} {'─' * (62 - len(smell.severity))}",
                    color))
                print()
                prev_sev = smell.severity

            color  = self._severity_color(smell.severity)
            rmav_i = self.RMAV_ICONS.get(smell.rmav_dimension, "?")

            print(f"  {self._c(f'#{i:02d}', self.BOLD)}  "
                  f"{self._c(f'[{smell.pattern_id}]', color + self.BOLD)}"
                  f"  {self._c(smell.pattern_name, self.BOLD)}"
                  f"  {self._c(f'[RMAV:{rmav_i}]', self.GRAY)}"
                  f"  {self._c(f'layer={smell.layer}', self.DIM)}")

            comps_display = ", ".join(smell.component_ids[:5])
            if len(smell.component_ids) > 5:
                comps_display += f" … (+{len(smell.component_ids) - 5} more)"
            print(f"       {self._c('Components:', self.BOLD)} {comps_display}")

            # Evidence
            ev_parts = [f"{k}={v}" for k, v in list(smell.metric_evidence.items())[:4]]
            print(f"       {self._c('Evidence:  ', self.BOLD)} {self._c(', '.join(ev_parts), self.GRAY)}")

            # Risk (first 140 chars)
            risk_short = smell.risk[:140].rstrip() + ("…" if len(smell.risk) > 140 else "")
            print(f"       {self._c('Risk:      ', self.BOLD)} {risk_short}")

            # Recommendation (first numbered point only)
            rec_first = smell.recommendation.split("2.")[0].strip()
            print(f"       {self._c('Fix:       ', self.BOLD)} {rec_first}")
            print()

    def print_success(self, msg: str) -> None:
        print(f"  {self._c('✓', self.GREEN)} {msg}")

    def print_error(self, msg: str) -> None:
        red = "\033[91m"
        print(f"  {self._c('✗', red)} {msg}")


# =============================================================================
# CLI — Argument Parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="detect_antipatterns",
        description="Pub-Sub Anti-Pattern & Bad Smell Detector — graph topology analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --layer system                        Scan complete system layer
  %(prog)s --layer app --severity critical,high  Only CRITICAL and HIGH smells
  %(prog)s --all                                 Scan all four layers
  %(prog)s --all --pattern spof,broker_overload  Specific patterns only
  %(prog)s --catalog                             Print full catalog (no Neo4j needed)
  %(prog)s --layer system --output smells.json   Export findings to JSON
  %(prog)s --layer system --use-ahp              Use AHP-derived RMAV weights

pattern IDs (case-insensitive):
  spof, systemic_risk, cyclic_dependency,
  god_component, bottleneck_edge, broker_overload, deep_pipeline,
  topic_fanout, chatty_pair, qos_mismatch, orphaned_topic, unstable_interface
""",
    )

    # ── Mode ──────────────────────────────────────────────────────────────
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--layer", "-l",
        choices=["app", "infra", "mw", "system"],
        default="system",
        help="Analysis layer to scan (default: system)",
    )
    mode.add_argument(
        "--all", "-a",
        action="store_true",
        help="Scan all four layers (app, infra, mw, system)",
    )
    mode.add_argument(
        "--catalog",
        action="store_true",
        help="Print full anti-pattern catalog and exit (no Neo4j required)",
    )

    # ── Neo4j connection ──────────────────────────────────────────────────
    neo4j = parser.add_argument_group("Neo4j connection")
    neo4j.add_argument("--uri",      default="bolt://localhost:7687", help="Neo4j Bolt URI")
    neo4j.add_argument("--user",     default="neo4j",                help="Neo4j username")
    neo4j.add_argument("--password", default="password",             help="Neo4j password")

    # ── Detection options ─────────────────────────────────────────────────
    detection = parser.add_argument_group("Detection options")
    detection.add_argument(
        "--pattern", "-P",
        metavar="ID[,ID…]",
        help="Comma-separated list of pattern IDs to run (default: all)",
    )
    detection.add_argument(
        "--severity", "-S",
        metavar="LEVEL[,LEVEL…]",
        help="Filter output to these severity levels, e.g. critical,high",
    )
    detection.add_argument(
        "--use-ahp",
        action="store_true",
        help="Use AHP-derived RMAV weights instead of default equal weights",
    )

    # ── Output ────────────────────────────────────────────────────────────
    out = parser.add_argument_group("Output")
    out.add_argument("--output", "-o", metavar="FILE",  help="Export findings to JSON file")
    out.add_argument("--json",         action="store_true", help="Print JSON to stdout")
    out.add_argument("--quiet",  "-q", action="store_true", help="Suppress human-readable output")
    out.add_argument("--verbose","-v", action="store_true", help="Enable debug logging")

    return parser


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    parser  = build_parser()
    args    = parser.parse_args()
    display = SmellConsoleDisplay()

    # ── Logging ───────────────────────────────────────────────────────────
    log_level = (
        logging.DEBUG   if args.verbose else
        logging.WARNING if args.quiet   else
        logging.INFO
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Catalog mode (no Neo4j needed) ───────────────────────────────────
    if args.catalog:
        display.print_catalog()
        return 0

    # ── Parse active patterns ─────────────────────────────────────────────
    active_patterns: Optional[List[str]] = None
    if args.pattern:
        active_patterns = [p.strip().upper() for p in args.pattern.split(",") if p.strip()]
        unknown = set(active_patterns) - set(CATALOG.keys())
        if unknown:
            display.print_error(f"Unknown pattern IDs: {sorted(unknown)}")
            display.print_error(f"Valid IDs: {sorted(CATALOG.keys())}")
            return 1

    # ── Parse severity filter ─────────────────────────────────────────────
    severity_filter: Optional[List[str]] = None
    if args.severity:
        severity_filter = [s.strip().upper() for s in args.severity.split(",") if s.strip()]
        valid_sevs = {"CRITICAL", "HIGH", "MEDIUM"}
        bad_sevs = set(severity_filter) - valid_sevs
        if bad_sevs:
            display.print_error(f"Unknown severity levels: {sorted(bad_sevs)}. "
                                f"Use: critical, high, medium")
            return 1

    # ── Layers to scan ───────────────────────────────────────────────────
    layers_to_scan = (
        ["app", "infra", "mw", "system"] if args.all else [args.layer]
    )

    # ── Connect to Neo4j ─────────────────────────────────────────────────
    try:
        repo = create_repository(uri=args.uri, user=args.user, password=args.password)
    except Exception as exc:
        display.print_error(f"Cannot connect to Neo4j at {args.uri}: {exc}")
        display.print_error("Ensure Neo4j is running and data has been imported.")
        return 1

    try:
        analyzer = AnalysisService(repo, use_ahp=args.use_ahp)
        detector = SmellDetector(active_patterns=active_patterns)

        all_smells: List[DetectedSmell] = []
        total_components = 0

        for layer in layers_to_scan:
            logger.info("Analyzing layer: %s", layer)
            try:
                layer_result = analyzer.analyze_layer(layer)
            except Exception as exc:
                logger.warning("Analysis failed for layer %s: %s", layer, exc)
                if args.verbose:
                    logger.exception("Layer analysis error")
                continue

            total_components += len(layer_result.quality.components)
            layer_smells = detector.detect(layer_result, layer)
            all_smells.extend(layer_smells)
            logger.info("Layer %s: %d components, %d smells detected",
                        layer, len(layer_result.quality.components), len(layer_smells))

        # ── Build report ─────────────────────────────────────────────────
        by_severity: Dict[str, int] = {}
        by_pattern:  Dict[str, int] = {}
        by_layer:    Dict[str, int] = {}
        for s in all_smells:
            by_severity[s.severity]   = by_severity.get(s.severity, 0) + 1
            by_pattern[s.pattern_id]  = by_pattern.get(s.pattern_id, 0) + 1
            by_layer[s.layer]         = by_layer.get(s.layer, 0) + 1

        report = SmellReport(
            generated_at=datetime.utcnow().isoformat() + "Z",
            layers_analyzed=layers_to_scan,
            total_components=total_components,
            total_smells=len(all_smells),
            by_severity=by_severity,
            by_pattern=by_pattern,
            by_layer=by_layer,
            smells=all_smells,
        )

        # ── Human-readable output ─────────────────────────────────────────
        if not args.quiet:
            display.print_report(report, severity_filter=severity_filter)

        # ── JSON stdout ───────────────────────────────────────────────────
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, default=str))

        # ── File export ───────────────────────────────────────────────────
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as fh:
                json.dump(report.to_dict(), fh, indent=2, default=str)
            if not args.quiet:
                display.print_success(f"Report saved → {args.output}")

        # Exit code: 0 if clean, 2 if any CRITICAL found
        if by_severity.get("CRITICAL", 0) > 0:
            return 2
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        display.print_error(f"Unexpected error: {exc}")
        if args.verbose:
            logger.exception("Fatal error")
        return 1
    finally:
        repo.close()


if __name__ == "__main__":
    sys.exit(main())
