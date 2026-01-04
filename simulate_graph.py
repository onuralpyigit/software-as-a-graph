#!/usr/bin/env python3
"""
Simulation CLI

Run failure or event simulations on the Graph Model.
"""

import argparse
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.simulation.simulator import Simulator

# Colors
RED = "\033[91m"; GREEN = "\033[92m"; CYAN = "\033[96m"; RESET = "\033[0m"

def print_failure_result(res):
    print(f"\n{RED}=== Failure Simulation Result ==={RESET}")
    print(f"Scenario: {res.scenario}")
    print(f"Initial: {res.initial_failures}")
    print(f"Cascaded: {res.cascaded_failures}")
    print(f"Total Impact: {res.total_impact} components")
    print(f"Steps: {res.propagation_steps}")
    print("\nImpact by Type:")
    for c_type, nodes in res.affected_components.items():
        print(f"  {c_type}: {len(nodes)} nodes")

def print_event_result(res):
    print(f"\n{CYAN}=== Event Simulation Result ==={RESET}")
    print(f"Scenario: {res.scenario}")
    print(f"Source: {res.source}")
    print(f"Reached: {len(res.reached_nodes)} nodes")
    print(f"Unreachable: {len(res.unreachable_nodes)} nodes")
    print(f"Max Hops: {res.max_hops}")
    if res.bottlenecks:
        print(f"Potential Bottlenecks: {', '.join(res.bottlenecks)}")

def main():
    parser = argparse.ArgumentParser(description="Graph Simulation CLI")
    
    # Connection args
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--failure", help="Node ID to fail")
    mode_group.add_argument("--event", help="Source Node ID for event")
    
    # Filters
    parser.add_argument("--type", help="Restrict to Component Type (e.g. Application)")
    parser.add_argument("--layer", help="Restrict to Layer (e.g. application)")
    
    args = parser.parse_args()
    
    try:
        with Simulator(uri=args.uri, user=args.user, password=args.password) as sim:
            print(f"{GREEN}Connected to Neo4j.{RESET}")
            
            if args.failure:
                print(f"Simulating failure for {args.failure}...")
                res = sim.run_failure_simulation(
                    args.failure, 
                    component_type=args.type,
                    layer=args.layer
                )
                print_failure_result(res)
                
            elif args.event:
                print(f"Simulating event from {args.event}...")
                res = sim.run_event_simulation(args.event)
                print_event_result(res)
                
    except Exception as e:
        print(f"{RED}Error:{RESET} {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())