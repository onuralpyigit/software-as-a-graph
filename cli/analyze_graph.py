#!/usr/bin/env python3
"""
Graph Analysis CLI

Multi-layer graph analysis for distributed pub-sub systems.
Applies graph topology analysis to predict critical components using
DEPENDS_ON relationships derived from the system model.
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path to support direct execution (python cli/analyze_graph.py)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saag import Client
from cli.common.arguments import add_neo4j_arguments, add_common_arguments, setup_logging
from cli.common.console import ConsoleDisplay

def main():
    parser = argparse.ArgumentParser(
        description="Multi-layer structural graph analysis for distributed pub-sub systems.",
        epilog="Computes structural metrics M(v) and the graph summary S(G) only. "
               "RM weighting, normalization, and sensitivity options belong to the "
               "Predict stage — see `saag-predict --help`.",
    )

    add_neo4j_arguments(parser)
    add_common_arguments(parser)
    
    args = parser.parse_args()
    setup_logging(args)
    console = ConsoleDisplay()

    console.print_header("Structural Graph Analysis")
    console.print_step(f"Connecting to Neo4j at {args.uri}...")
    
    try:
        client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
        layers = [args.layer]
        if args.layer.lower() == "all":
            layers = ["app", "infra", "mw", "system"]
        elif "," in args.layer:
            layers = [l.strip() for l in args.layer.split(",") if l.strip()]

        for layer in layers:
            console.print_step(f"Analyzing layer: {layer}...")
            
            # Analyze returns AnalysisResult which wraps StructuralAnalysisResult
            result = client.analyze(layer=layer)
            
            console.display_layer_result(result.raw)

            if args.output:
                out_path = args.output
                if len(layers) > 1:
                    base, ext = out_path.rsplit('.', 1) if '.' in out_path else (out_path, 'json')
                    out_path = f"{base}_{layer}.{ext}"
                result.save(out_path)
                console.print_success(f"Full analysis results for {layer} saved to {out_path}")
        
        console.print_success("Analysis Complete!")

    except Exception as e:
        console.print_error(f"Analysis failed: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()