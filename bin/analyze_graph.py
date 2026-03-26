#!/usr/bin/env python3
"""
Graph Analysis CLI

Multi-layer graph analysis for distributed pub-sub systems.
Applies graph topology analysis to predict critical components using
DEPENDS_ON relationships derived from the system model.
"""
import sys
from pathlib import Path

# Provide resolving so `saag` and `bin._shared` can be accessed natively if script run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from saag import Client
from bin._shared import add_neo4j_args, add_common_args, setup_logging
from bin.common.console import ConsoleDisplay

def main():
    parser = argparse.ArgumentParser(description="Multi-layer graph analysis for distributed pub-sub systems.")
    
    # Specific args
    parser.add_argument("--use-ahp", action="store_true", help="Use AHP-derived weights instead of default fixed weights")
    parser.add_argument("--equal-weights", action="store_true", help="Use equal 0.25 weights for all Q(v) dimensions (baseline)")
    parser.add_argument("--ahp-shrinkage", type=float, default=0.7, help="Shrinkage factor λ for AHP weights [0, 1] (default: 0.7)")
    
    add_neo4j_args(parser)
    add_common_args(parser)
    
    args = parser.parse_args()
    setup_logging(args)
    console = ConsoleDisplay()

    console.print_header("Structural Graph Analysis")
    console.print_step(f"Connecting to Neo4j at {args.uri}...")
    
    try:
        client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
        console.print_step(f"Analyzing layer: {args.layer}...")
        
        # Analyze returns AnalysisResult which wraps StructuralAnalysisResult
        result = client.analyze(layer=args.layer)
        
        console.print_success("Analysis Complete!")
        
        # Display summary stats
        summary = result.raw.graph_summary.to_dict()
        console.display_structural_summary(summary)
        
        # Display top components by betweenness (bottlenecks)
        comps = [c.to_dict() for c in result.raw.components.values()]
        console.display_top_components(comps, metric="betweenness", n=5)

        if args.output:
            result.save(args.output)
            console.print_success(f"Full analysis results saved to {args.output}")

    except Exception as e:
        console.print_error(f"Analysis failed: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()