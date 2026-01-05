#!/usr/bin/env python3
"""
Graph Simulation CLI
Updated to support PhD Progress Report Methodology (Fall 2025).
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
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--event", help="Run event simulation from Source Node ID")
    group.add_argument("--failure", help="Run failure simulation on Target Node ID")
    
    # Cascade Parameters (Report Section 4.4.2)
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="Cascade threshold (default: 0.5)")
    parser.add_argument("--probability", type=float, default=0.7, 
                        help="Cascade probability (default: 0.7)")
    parser.add_argument("--depth", type=int, default=5, 
                        help="Max cascade depth (default: 5)")
    
    args = parser.parse_args()

    try:
        with Simulator(args.uri, args.user, args.password) as sim:
            if args.event:
                print(f"--- Simulating Event from {args.event} ---")
                res = sim.run_event_sim(args.event)
                print(json.dumps(res.to_dict(), indent=2))
                
            elif args.failure:
                print(f"--- Simulating Failure of {args.failure} ---")
                print(f"Params: Threshold={args.threshold}, Prob={args.probability}, Depth={args.depth}")
                
                res = sim.run_failure_sim(
                    args.failure, 
                    threshold=args.threshold,
                    probability=args.probability,
                    depth=args.depth
                )
                print(json.dumps(res.to_dict(), indent=2))
                
    except Exception as e:
        print(f"Simulation Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()