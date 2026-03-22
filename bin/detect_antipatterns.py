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

from src.infrastructure import create_repository
from src.prediction import PredictionService, QualityAnalysisResult, DetectedProblem
from src.usecases import AnalyzeGraphUseCase, PredictGraphUseCase
from common.console import ConsoleDisplay
from common.arguments import add_neo4j_arguments, add_common_arguments

logger = logging.getLogger("detect_antipatterns")


# =============================================================================
# Console Display Helpers
# =============================================================================



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

    add_neo4j_arguments(parser)

    detection = parser.add_argument_group("Detection options")
    detection.add_argument(
        "--severity", "-S",
        metavar="LEVEL[,LEVEL…]",
        help="Filter output to these severity levels, e.g. critical,high",
    )

    out = parser.add_argument_group("Output")
    out.add_argument("--output", "-o", metavar="FILE",  help="Export findings to JSON file")
    out.add_argument("--json",         action="store_true", help="Print JSON to stdout")
    add_common_arguments(parser)

    return parser


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    parser  = build_parser()
    args    = parser.parse_args()
    display = ConsoleDisplay()

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
            display.display_antipatterns(all_problems, layers_to_scan, total_components)

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
