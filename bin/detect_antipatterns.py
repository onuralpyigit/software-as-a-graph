#!/usr/bin/env python3
"""
bin/detect_antipatterns.py — Pub-Sub Architectural Anti-Pattern Detector
========================================================================
Detects bad smells from GNN predictions and structural metrics.
"""

import sys
from pathlib import Path

# Provide resolving so `saag` and `bin._shared` can be accessed natively
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from saag import Client
from bin._shared import add_neo4j_args, add_common_args, setup_logging
from bin.common.console import ConsoleDisplay

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
    
    add_neo4j_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args)

    display = ConsoleDisplay()

    if args.catalog:
        from src.analysis.antipattern_detector import CATALOG
        display.print_header("Anti-Pattern Catalog")
        for pid, spec in CATALOG.items():
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
    
    display.print_step(f"Analyzing layer '{args.layer}' for bad smells...")
    analysis = client.analyze(
        layer=args.layer, 
        use_ahp=args.use_ahp, 
        ahp_shrinkage=args.ahp_shrinkage
    )
    
    display.print_step("Generating criticality predictions...")
    prediction = client.predict(analysis)
    
    display.print_step("Scanning for structural and probabilistic anti-patterns...")
    active_patterns = None
    if args.pattern:
        active_patterns = [p.strip().upper() for p in args.pattern.split(",")]
        
    problems = client.detect_antipatterns(prediction, active_patterns=active_patterns)
    
    # Apply severity filter
    if args.severity:
        allowed_sevs = {s.strip().upper() for s in args.severity.split(",")}
        problems = [p for p in problems if p.severity.upper() in allowed_sevs]
    
    # Report results
    total_components = len(analysis.raw.components)
    display.display_antipatterns(problems, [args.layer], total_components)
    
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([p.to_dict() for p in problems], f, indent=2)
        display.print_success(f"Detailed anti-pattern report saved to {args.output}")
    else:
        display.print_success("Anti-pattern detection complete.")

    # CI Exit Codes
    if any(p.severity == "CRITICAL" for p in problems):
        sys.exit(2)
    elif any(p.severity == "HIGH" for p in problems):
        sys.exit(2)
    elif problems:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
