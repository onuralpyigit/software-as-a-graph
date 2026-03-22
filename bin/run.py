#!/usr/bin/env python3
"""
bin/run.py — Software-as-a-Graph Pipeline Orchestrator
======================================================
Executes the analytical pipeline using the saag SDK.
"""

import sys
from pathlib import Path

# Provide resolving so `saag` and `bin._shared` can be accessed natively
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from saag import Pipeline
from bin._shared import add_neo4j_args, add_common_args, setup_logging

def main():
    parser = argparse.ArgumentParser(
        description="Run the Software-as-a-Graph pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Stage flags
    group = parser.add_argument_group("Pipeline Stages")
    group.add_argument("--all", action="store_true", help="Run all stages")
    group.add_argument("--import-file", metavar="FILE", help="Run import stage with given JSON topology file")
    group.add_argument("--analyze", action="store_true", help="Run analysis stage")
    group.add_argument("--predict", action="store_true", help="Run prediction stage")
    group.add_argument("--simulate", action="store_true", help="Run simulation stage")
    group.add_argument("--validate", action="store_true", help="Run validation stage")
    group.add_argument("--visualize", action="store_true", help="Run visualization stage")
    
    # Stage-specific options
    opts = parser.add_argument_group("Stage Options")
    opts.add_argument("--clear", action="store_true", help="Clear DB before import")
    opts.add_argument("--use-ahp", action="store_true", help="Use AHP-derived weights instead of default fixed weights")
    opts.add_argument("--sim-mode", default="exhaustive", help="Simulation mode (e.g., exhaustive, monte_carlo)")
    opts.add_argument("--no-network", action="store_true", help="Skip interactive network in visualization")
    opts.add_argument("--no-matrix", action="store_true", help="Skip dependency matrix in visualization")
    opts.add_argument("--no-validation", action="store_true", help="Skip validation metrics in visualization")

    add_neo4j_args(parser)
    add_common_args(parser)
    args = parser.parse_args()
    setup_logging(args)
    
    # Ensure at least one stage is selected
    if not any([args.all, args.import_file, args.analyze, args.predict, args.simulate, args.validate, args.visualize]):
        parser.error("No stages selected. Use --all or specific stage flags.")
        return 1
        
    # Build pipeline
    if args.import_file or args.all:
        import_path = args.import_file if args.import_file else "output/graph.json"
        
        # Check if the file exists before passing to from_json
        if not Path(import_path).exists():
            print(f"Error: Import file '{import_path}' not found.", file=sys.stderr)
            return 1
            
        pipeline = Pipeline.from_json(import_path, clear=args.clear, neo4j_uri=args.uri, user=args.user, password=args.password)
    else:
        pipeline = Pipeline(neo4j_uri=args.uri, user=args.user, password=args.password)
        
    if args.analyze or args.all:
        pipeline.analyze(layer=args.layer, use_ahp=args.use_ahp)
        
    if args.predict or args.all:
        pipeline.predict()
        
    if args.simulate or args.all:
        pipeline.simulate(layer=args.layer, mode=args.sim_mode)
        
    if args.validate or args.all:
        layers = ["app", "infra", "mw", "system"] if args.all else [l.strip() for l in args.layer.split(",")]
        pipeline.validate(layers=layers)
        
    if args.visualize or args.all:
        layers = ["app", "infra", "mw", "system"] if args.all else [l.strip() for l in args.layer.split(",")]
        out_path = args.output if args.output else "dashboard.html"
        pipeline.visualize(
            output=out_path,
            layers=layers,
            include_network=not args.no_network,
            include_matrix=not args.no_matrix,
            include_validation=not args.no_validation
        )
        
    # Execute pipeline
    result = pipeline.run()
    
    # Save generic result if output provided and not visualizing
    if args.output and not (args.visualize or args.all):
        result.save(args.output)
        if not getattr(args, "quiet", False):
            print(f"Pipeline executed successfully. Result saved to {args.output}")
    else:
        if not getattr(args, "quiet", False):
            print("Pipeline executed successfully.")

    return 0
    
if __name__ == "__main__":
    sys.exit(main())