#!/usr/bin/env python3
"""
bin/detect_antipatterns.py — Pub-Sub Architectural Anti-Pattern & Bad Smell Detector
=====================================================================================
Catalogs and detects architectural anti-patterns and bad smells in distributed
publish-subscribe systems using graph topology analysis.
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

from src.adapters import create_repository
from src.prediction import PredictionService, QualityAnalysisResult, DetectedProblem
from src.usecases import AnalyzeGraphUseCase, PredictGraphUseCase

logger = logging.getLogger("detect_antipatterns")


# =============================================================================
# Console Display Helpers
# =============================================================================

class SmellConsoleDisplay:
    """Rich terminal rendering for smell detection results."""

    SEVERITY_COLORS = {
        "CRITICAL": "\033[91m",   # bright red
        "HIGH":     "\033[93m",   # yellow
        "MEDIUM":   "\033[94m",   # blue
        "LOW":      "\033[90m",   # gray
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
        return self.SEVERITY_COLORS.get(sev.upper(), "")

    def print_banner(self) -> None:
        line = "═" * 70
        print(f"\n{self._c(line, self.CYAN)}")
        print(self._c(
            "  Software-as-a-Graph  ·  Pub-Sub Anti-Pattern & Bad Smell Detector",
            self.CYAN + self.BOLD))
        print(f"{self._c(line, self.CYAN)}\n")

    def print_report(self, problems: List[DetectedProblem], layers: List[str], total_components: int) -> None:
        """Print the full detection report to stdout."""
        self.print_banner()

        # ── Summary KPIs ──────────────────────────────────────────────────
        print(self._c("  SCAN SUMMARY", self.BOLD))
        print(f"  Layers analyzed:     {', '.join(layers)}")
        print(f"  Components scanned:  {total_components}")
        print(f"  Total smells found:  {self._c(str(len(problems)), self.BOLD)}")
        print()
        
        by_sev = {}
        for p in problems:
            by_sev[p.severity] = by_sev.get(p.severity, 0) + 1
            
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            count = by_sev.get(sev, 0)
            color = self._severity_color(sev)
            bar = self._c("█" * count, color) if count else self._c("─", self.GRAY)
            print(f"  {self._c(f'{sev:<10}', color)}  {bar}  {self._c(str(count), color + self.BOLD)}")
        print()

        if not problems:
            print(self._c("  ✓  No smells found matching the active filters.\n", self.GREEN))
            return

        print(self._c(f"  FINDINGS ({len(problems)})", self.BOLD))
        print()

        prev_sev = None
        for i, problem in enumerate(problems, 1):
            if problem.severity != prev_sev:
                color = self._severity_color(problem.severity)
                print(self._c(
                    f"  {'─' * 3} {problem.severity} {'─' * (62 - len(problem.severity))}",
                    color))
                print()
                prev_sev = problem.severity

            color  = self._severity_color(problem.severity)

            print(f"  {self._c(f'#{i:02d}', self.BOLD)}  "
                  f"{self._c(f'[{problem.name}]', color + self.BOLD)}"
                  f"  {self._c(problem.entity_id, self.BOLD)} "
                  f"({problem.entity_type})")

            # Description
            print(f"       {self._c('Description:', self.BOLD)} {problem.description}")

            # Evidence
            ev_parts = [f"{k}={v}" for k, v in list(problem.evidence.items())[:4]]
            print(f"       {self._c('Evidence:   ', self.BOLD)} {self._c(', '.join(ev_parts), self.GRAY)}")

            # Recommendation
            print(f"       {self._c('Fix:        ', self.BOLD)} {problem.recommendation}")
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
    )

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

    neo4j = parser.add_argument_group("Neo4j connection")
    neo4j.add_argument("--uri",      default="bolt://localhost:7687", help="Neo4j Bolt URI")
    neo4j.add_argument("--user",     default="neo4j",                help="Neo4j username")
    neo4j.add_argument("--password", default="password",             help="Neo4j password")

    detection = parser.add_argument_group("Detection options")
    detection.add_argument(
        "--severity", "-S",
        metavar="LEVEL[,LEVEL…]",
        help="Filter output to these severity levels, e.g. critical,high",
    )

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

    # ── Parse severity filter ─────────────────────────────────────────────
    severity_filter: Optional[List[str]] = None
    if args.severity:
        severity_filter = [s.strip().upper() for s in args.severity.split(",") if s.strip()]

    # ── Layers to scan ───────────────────────────────────────────────────
    layers_to_scan = (
        ["app", "infra", "mw", "system"] if args.all else [args.layer]
    )

    # ── Connect to Neo4j ─────────────────────────────────────────────────
    try:
        repo = create_repository(uri=args.uri, user=args.user, password=args.password)
    except Exception as exc:
        display.print_error(f"Cannot connect to Neo4j at {args.uri}: {exc}")
        return 1

    try:
        analyze_uc = AnalyzeGraphUseCase(repo)
        predict_uc = PredictGraphUseCase(repo)

        all_problems: List[DetectedProblem] = []
        total_components = 0

        for layer in layers_to_scan:
            logger.info("Analyzing layer: %s", layer)
            try:
                structural_result = analyze_uc.execute(layer)
                quality_result, problems = predict_uc.execute(
                    layer=layer,
                    structural_result=structural_result,
                    detect_problems=True
                )
            except Exception as exc:
                logger.warning("Analysis failed for layer %s: %s", layer, exc)
                continue

            total_components += len(quality_result.components)
            if problems:
                if severity_filter:
                    problems = [p for p in problems if p.severity.upper() in severity_filter]
                all_problems.extend(problems)

        # ── Output ───────────────────────────────────────────────────────
        if not args.quiet:
            display.print_report(all_problems, layers_to_scan, total_components)

        if args.json:
            print(json.dumps([p.to_dict() for p in all_problems], indent=2))

        if args.output:
            with open(args.output, "w") as f:
                json.dump([p.to_dict() for p in all_problems], f, indent=2)
            if not args.quiet:
                display.print_success(f"Report saved → {args.output}")

        # Exit code 2 if CRITICAL found
        if any(p.severity == "CRITICAL" for p in all_problems):
            return 2
        return 0

    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        display.print_error(f"Unexpected error: {exc}")
        return 1
    finally:
        repo.close()


if __name__ == "__main__":
    sys.exit(main())
