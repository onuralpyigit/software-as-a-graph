#!/usr/bin/env python3
"""
Graph Simulation CLI

Executes Event and Failure simulations on the Raw Graph Model.
Supports single-target tests, exhaustive datasets, and summary reports.
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
    
    # Action Group
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--event", help="Run event simulation from Source Node ID")
    action.add_argument("--failure", help="Run failure simulation on Target Node ID")
    action.add_argument("--report", action="store_true", help="Generate full system evaluation report")
    action.add_argument("--exhaustive", action="store_true", help="Run batch simulation for dataset export")
    
    # Options
    parser.add_argument("--layer", choices=["application", "infrastructure", "complete"], default="complete")
    parser.add_argument("--output", "-o", help="Output file path for JSON results")
    parser.add_argument("--threshold", type=float, default=0.5)
    
    args = parser.parse_args()
    
    print(f"Connecting to {args.uri}...")
    
    try:
        with Simulator(args.uri, args.user, args.password) as sim:
            
            # 1. Event Simulation
            if args.event:
                print(f"\n>> Simulating Event Propagation from: {args.event}")
                res = sim.run_event_sim(args.event)
                print(json.dumps(res.to_dict(), indent=2))
                if args.output:
                    with open(args.output, 'w') as f: json.dump(res.to_dict(), f, indent=2)

            # 2. Single Failure Simulation
            elif args.failure:
                print(f"\n>> Simulating Failure: {args.failure}")
                res = sim.run_failure_sim(args.failure, layer=args.layer, threshold=args.threshold)
                print(json.dumps(res.to_dict(), indent=2))

            # 3. Full Report (Evaluation)
            elif args.report:
                print("\n>> Generating System Evaluation Report...")
                report = sim.generate_evaluation_report()
                print(json.dumps(report, indent=2))
                if args.output:
                    with open(args.output, 'w') as f: json.dump(report, f, indent=2)
                    print(f"Report saved to {args.output}")

            # 4. Exhaustive Dataset
            elif args.exhaustive:
                print(f"\n>> Running Exhaustive Simulation (Layer: {args.layer})...")
                results = sim.run_exhaustive_failure_sim(layer=args.layer, threshold=args.threshold)
                dataset = [r.to_flat_dict() for r in results]
                
                outfile = args.output if args.output else "simulation_dataset.json"
                with open(outfile, 'w') as f: json.dump(dataset, f, indent=2)
                print(f"Completed {len(dataset)} runs. Saved to {outfile}")

    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())