#!/usr/bin/env python3
"""
Prescription Optimization CLI

Generates prescriptive refactoring recommendations (logical topic splitting, 
physical host anti-affinity container reallocations, and transport QoS contract hardening)
and evaluates resilience improvements in a closed-loop simulation on a mutated graph.
"""
import argparse
import sys
import json
from pathlib import Path

# Add project root to sys.path to support direct execution (python cli/prescribe_graph.py)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saag import Client
from cli.common.arguments import add_neo4j_arguments, add_common_arguments, setup_logging
from cli.common.console import ConsoleDisplay

def main():
    parser = argparse.ArgumentParser(
        description="Prescriptive architecture optimization and closed-loop validation."
    )
    
    # Specific args
    parser.add_argument("--gnn-checkpoint", type=str, help="Path to trained GNN model checkpoint (optional)")
    parser.add_argument("--input", "-i", type=str, help="Path to system topology JSON file to run in-memory (optional)")
    
    add_neo4j_arguments(parser)
    add_common_arguments(parser)
    
    args = parser.parse_args()
    setup_logging(args)
    console = ConsoleDisplay()

    console.print_header("Prescriptive Architecture Optimization (Stage 6)")
    
    try:
        if args.input:
            from saag.infrastructure.memory_repo import MemoryRepository
            console.print_step(f"Loading graph JSON in-memory: {args.input}...")
            repo = MemoryRepository()
            with open(args.input) as f:
                data = json.load(f)
            repo.save_graph(data, clear=True)
            repo.derive_dependencies()
            client = Client(repo=repo)
        else:
            console.print_step(f"Connecting to Neo4j at {args.uri}...")
            client = Client(neo4j_uri=args.uri, user=args.user, password=args.password)
        
        console.print_step(f"Running baseline analysis on layer: {args.layer}...")
        analysis = client.analyze(layer=args.layer)
        
        console.print_step("Generating refactoring recommendations and validating in closed-loop...")
        res = client.prescribe(
            analysis_result=analysis,
            layer=args.layer,
            gnn_checkpoint=args.gnn_checkpoint
        )
        
        console.print_success("Prescriptive Remediation Generated!")
        
        # Display the compiled policy
        console.print_header("Compiled Optimization Policy Delta(G)")
        
        policy = res.policy
        splits = policy.topic_splits
        reallocs = policy.node_reallocations
        upgrades = policy.qos_upgrades
        
        print(f"Logical Topic Splits: {len(splits)}")
        for split in splits:
            print(f"  - Topic '{split['topic']}' split per publisher: {', '.join(split['publishers'])}")
            
        print(f"Physical Host Reallocations: {len(reallocs)}")
        for realloc in reallocs:
            print(f"  - Component '{realloc['component']}' moved from '{realloc['from_node']}' -> '{realloc['to_node']}'")
            
        print(f"Transport QoS Upgrades: {len(upgrades)}")
        for upgrade in upgrades:
            print(f"  - Topic '{upgrade['topic']}': {upgrade['original_reliability']}/{upgrade['original_durability']} -> {upgrade['target_reliability']}/{upgrade['target_durability']}")
            
        console.print_header("Closed-Loop Resilience Summary")
        print(f"Baseline System Risk Index (SRI): {res.original_sri:.4f}")
        print(f"Mutated System Risk Index (SRI) : {res.mutated_sri:.4f}")

        improvement = res.sri_improvement
        if improvement > 0:
            console.print_success(f"Resilience Improved by: {improvement:.4f}")
        elif improvement < 0:
            console.print_error(f"Resilience Degraded by: {abs(improvement):.4f}")
        else:
            print("No change in System Risk Index.")

        if res.accepted:
            console.print_success("Policy Status: ACCEPTED")
        else:
            console.print_error("Policy Status: REJECTED - mutated risk did not improve")
            
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w") as f:
                json.dump(res.to_dict(), f, indent=2, default=str)
            console.print_success(f"Prescription optimization result saved to {args.output}")
            
    except Exception as e:
        console.print_error(f"Prescription generation failed: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
