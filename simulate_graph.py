#!/usr/bin/env python3
"""
Graph Simulation CLI
"""

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.simulation import Simulator

def main():
    parser = argparse.ArgumentParser(description="Graph Simulation CLI")
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    
    # Simulation Type
    type_group = parser.add_mutually_exclusive_group(required=True)
    type_group.add_argument("--event", help="Run event simulation from Source Node ID")
    type_group.add_argument("--failure", help="Run failure simulation on Target Node ID")
    type_group.add_argument("--exhaustive", action="store_true", 
                            help="Run failure simulation on ALL components to build a dataset")
    
    # Layer Scope
    parser.add_argument("--layer", choices=["application", "infrastructure", "complete"],
                        default="complete", help="Layer to analyze (default: complete)")

    # Cascade Parameters
    parser.add_argument("--threshold", type=float, default=0.5, help="Cascade threshold (default: 0.5)")
    parser.add_argument("--probability", type=float, default=0.7, help="Cascade probability (default: 0.7)")
    parser.add_argument("--depth", type=int, default=5, help="Max cascade depth (default: 5)")
    
    # Output
    parser.add_argument("--output", default="simulation_dataset.json", 
                        help="Output file for exhaustive simulation dataset (default: simulation_dataset.json)")
    
    args = parser.parse_args()

    print(f"Connecting to {args.uri}...")
    
    try:
        with Simulator(args.uri, args.user, args.password) as sim:
            if args.event:
                print(f"\n--- Event Simulation: {args.event} ---")
                res = sim.run_event_sim(args.event)
                print(json.dumps(res.to_dict(), indent=2))
                
            elif args.failure:
                print(f"\n--- Failure Simulation: {args.failure} ---")
                print(f"Layer: {args.layer.upper()}")
                
                res = sim.run_failure_sim(
                    args.failure, 
                    layer=args.layer,
                    threshold=args.threshold,
                    probability=args.probability,
                    depth=args.depth
                )
                print(json.dumps(res.to_dict(), indent=2))
            
            elif args.exhaustive:
                print(f"\n--- Exhaustive Failure Simulation ---")
                print(f"Layer: {args.layer.upper()}")
                print(f"This may take a while depending on graph size...")
                
                results = sim.run_exhaustive_failure_sim(
                    layer=args.layer,
                    threshold=args.threshold,
                    probability=args.probability,
                    depth=args.depth
                )
                
                # Convert to dataset format (list of flat dicts)
                dataset = [r.to_flat_dict() for r in results]
                
                # Save
                with open(args.output, 'w') as f:
                    json.dump(dataset, f, indent=2)
                    
                print(f"\nCompleted! Simulated {len(dataset)} scenarios.")
                print(f"Dataset saved to: {args.output}")
                
    except Exception as e:
        print(f"Simulation Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()