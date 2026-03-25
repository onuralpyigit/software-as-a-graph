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
    group.add_argument("--all", action="store_true", help="Run all stages (generate -> import -> analyze -> simulate -> validate -> visualize)")
    group.add_argument("--generate", action="store_true", help="Run graph generation stage")
    group.add_argument("--input", "-i", metavar="FILE", help="System topology JSON file (input for import, output for generate)")
    group.add_argument("--analyze", action="store_true", help="Run analysis stage (includes GNN prediction & problem detection)")
    group.add_argument("--predict", action="store_true", help="Explicitly run prediction stage")
    group.add_argument("--simulate", action="store_true", help="Run failure simulation stage")
    group.add_argument("--validate", action="store_true", help="Run validation stage (compare prediction vs simulation)")
    group.add_argument("--visualize", action="store_true", help="Run visualization stage (generates HTML dashboard)")
    
    # Stage-specific options
    opts = parser.add_argument_group("Stage Options")
    opts.add_argument("--config", type=Path, help="Path to graph configuration YAML file (for generation)")
    opts.add_argument("--scale", choices=["tiny", "small", "medium", "large", "xlarge"], help="Preset graph scale (for generation)")
    opts.add_argument("--output-dir", metavar="DIR", help="Directory for all intermediate and final outputs")
    opts.add_argument("--clear", "--clean", action="store_true", dest="clear", help="Clear Neo4j DB before import")
    opts.add_argument("--use-ahp", action="store_true", help="Use AHP-derived weights instead of default fixed weights")
    opts.add_argument("--gnn-model", metavar="PATH", help="Path to GNN model checkpoint directory")
    opts.add_argument("--sim-mode", default="exhaustive", help="Simulation mode (e.g., exhaustive, monte_carlo)")
    opts.add_argument("--no-network", action="store_true", help="Skip interactive network in visualization")
    opts.add_argument("--no-matrix", action="store_true", help="Skip dependency matrix in visualization")
    opts.add_argument("--no-validation", action="store_true", help="Skip validation metrics in visualization")

    add_neo4j_args(parser)
    add_common_args(parser)
    
    # Compatibility shim: old scripts might pass --import-file or --import
    parser.add_argument("--import-file", dest="input", help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    setup_logging(args)
    
    # Ensure at least one stage is selected
    stages = [args.all, args.generate, args.input, args.analyze, args.predict, args.simulate, args.validate, args.visualize]
    if not any(stages):
        parser.error("No stages selected. Use --all or specific stage flags.")
        return 1
        
    # 0. Generation Stage (Pre-Pipeline)
    # If using --all, we assume generation is desired if --config or --scale provided
    do_generate = args.generate or (args.all and (args.config or args.scale))
    if do_generate:
        from bin.common.dispatcher import dispatch_generate
        if not args.input and not args.all:
            parser.error("--generate requires --input to specify where to save the graph.")
        
        # Default output if not provided
        gen_output = args.input or (Path(args.output_dir) / "graph.json" if args.output_dir else "output/graph.json")
        args.output = str(gen_output) # dispatch_generate expects output in args.output
        
        print(f"[*] Stage 1/6: Generating graph...")
        dispatch_generate(args)
        # Update input path for subsequent stages
        args.input = args.output

    # 1. Initialize Pipeline
    # If no input provided but analyze/simulate/etc was requested, we operate on existing DB
    if args.input or args.all:
        import_path = args.input if args.input else "output/graph.json"
        
        # Check if the file exists before passing to from_json
        if not Path(import_path).exists():
            if not do_generate:
                print(f"Error: Input file '{import_path}' not found.", file=sys.stderr)
                return 1
            
        pipeline = Pipeline.from_json(import_path, clear=args.clear, neo4j_uri=args.uri, user=args.user, password=args.password)
    else:
        pipeline = Pipeline(neo4j_uri=args.uri, user=args.user, password=args.password)
        
    # 2. Configure Pipeline Stages
    if args.analyze or args.all:
        pipeline.analyze(layer=args.layer, use_ahp=args.use_ahp)
        
    if args.predict or args.all:
        pipeline.predict(gnn_model=args.gnn_model)
        
    if args.simulate or args.all:
        pipeline.simulate(layer=args.layer, mode=args.sim_mode)
        
    if args.validate or args.all:
        # Split comma-separated layers
        layers = [l.strip() for l in args.layer.split(",")] if args.layer else ["system"]
        if args.all: layers = ["app", "infra", "mw", "system"]
        pipeline.validate(layers=layers)
        
    if args.visualize or args.all:
        layers = [l.strip() for l in args.layer.split(",")] if args.layer else ["system"]
        if args.all: layers = ["app", "infra", "mw", "system"]
        
        out_path = args.output if args.output else "dashboard.html"
        if args.output_dir:
            out_path = str(Path(args.output_dir) / (args.output or "dashboard.html"))
            
        pipeline.visualize(
            output=out_path,
            layers=layers,
            include_network=not args.no_network,
            include_matrix=not args.no_matrix,
            include_validation=not args.no_validation
        )
        
    # 3. Execute pipeline
    print(f"[*] Running analytical stages...")
    result = pipeline.run()
    
    # 4. Post-Execution reporting
    if result.prediction and not getattr(args, "quiet", False):
        from src.explanation import CLIFormatter
        CLIFormatter.print_critical_report(result.prediction.raw, problems=result.problems, limit_top=5)
    
    # Save generic result if output path provided and not just visualizing
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