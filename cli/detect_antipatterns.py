#!/usr/bin/env python3
"""
cli/detect_antipatterns.py — Pub-Sub Architectural Anti-Pattern Detector
========================================================================
Detects bad smells from GNN predictions and structural metrics.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path to support direct execution (python cli/detect_antipatterns.py)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from saag import Client
from cli.common.arguments import add_neo4j_arguments, add_common_arguments, setup_logging
from cli.common.console import ConsoleDisplay

def main():
    parser = argparse.ArgumentParser(
        description="Pub-Sub Anti-Pattern & Bad Smell Detector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--use-ahp", action="store_true", help="Use AHP-derived weights instead of default fixed weights")
    parser.add_argument("--ahp-shrinkage", type=float, default=0.7, help="Shrinkage factor λ for AHP weights [0, 1] (default: 0.7)")
    parser.add_argument("--severity", type=str, help="Filter by severity (comma-separated, e.g. 'critical,high')")
    parser.add_argument("--pattern", type=str, help="Filter by pattern ID (comma-separated, e.g. 'SPOF,CYCLE')")
    parser.add_argument("--catalog", action="store_true", help="Print the anti-pattern catalog and exit")
    parser.add_argument("--no-exit-code", action="store_true", help="Always exit with code 0 (disables CI/CD blocking)")
    
    add_neo4j_arguments(parser)
    add_common_arguments(parser)
    args = parser.parse_args()
    setup_logging(args)

    display = ConsoleDisplay()

    if args.catalog:
        from saag.analysis.antipattern_detector import CATALOG
        display.print_header("Anti-Pattern Catalog")
        for pid, spec in sorted(CATALOG.items(), key=lambda kv: (kv[1].severity, kv[0])):
            color = display.severity_color(spec.severity)
            print(f"\n  {display.colored(f'[{pid}]', color, bold=True)} {display.colored(spec.name, display.Colors.WHITE, bold=True)}")
            print(f"  {'Category:':<12} {spec.category}")
            print(f"  {'Severity:':<12} {display.colored(spec.severity, color)}")
            print(f"  {'Description:':<12} {spec.description}")
            print(f"  {'Risk:':<12} {spec.risk}")
            print(f"  {'Fix:':<12} {display.colored(spec.recommendation, display.Colors.GREEN)}")
        sys.exit(0)

    display.print_header("Architectural Anti-Pattern Detection")
    
    client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
    
    layers = [args.layer]
    if args.layer.lower() == "all":
        layers = ["app", "infra", "mw", "system"]
    elif "," in args.layer:
        layers = [l.strip() for l in args.layer.split(",") if l.strip()]

    active_patterns = None
    if args.pattern:
        active_patterns = [p.strip().upper() for p in args.pattern.split(",") if p.strip()]

    all_problems = []
    
    for layer in layers:
        display.print_step(f"Analyzing layer '{layer}' for bad smells...")
        analysis = client.analyze(layer=layer)
        
        display.print_step(f"[{layer.upper()}] Scanning for anti-patterns...")
        prediction = client.predict(
            analysis,
            use_ahp=args.use_ahp,
            ahp_shrinkage=args.ahp_shrinkage,
            active_patterns=active_patterns,
        )
        
        problems = client.detect_antipatterns(prediction, active_patterns=active_patterns)
        
        # Apply severity filter
        if args.severity:
            allowed_sevs = {s.strip().upper() for s in args.severity.split(",")}
            problems = [p for p in problems if p.severity.upper() in allowed_sevs]
        
        all_problems.extend(problems)
        total_components = len(getattr(analysis.raw, "all_components", []))
        display.display_antipatterns(problems, [layer], total_components)
    
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([p.to_dict() for p in all_problems], f, indent=2)
        display.print_success(f"Detailed anti-pattern report saved to {args.output}")
    else:
        display.print_success("Anti-pattern detection complete.")

    if args.no_exit_code:
        return 0

    # CI Exit Codes (0=clean, 1=medium, 2=high/critical)
    severities = {p.severity.upper() for p in all_problems}
    if "CRITICAL" in severities or "HIGH" in severities:
        return 2
    elif "MEDIUM" in severities:
        return 1
    else:
        return 0

if __name__ == "__main__":
    sys.exit(main())

