#!/usr/bin/env python3
"""
Graph Simulation CLI (Refactored)

Comprehensive simulation for distributed pub-sub systems.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.infrastructure import Container


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive simulation for distributed pub-sub systems.")
    
    action_group = parser.add_argument_group("Action (one required)")
    action_mutex = action_group.add_mutually_exclusive_group(required=True)
    action_mutex.add_argument("--event", "-e", metavar="APP_ID", help="Event simulation from source")
    action_mutex.add_argument("--failure", "-f", metavar="COMP_ID", help="Failure simulation for target")
    action_mutex.add_argument("--exhaustive", action="store_true", help="Exhaustive failure analysis")
    action_mutex.add_argument("--report", "-r", action="store_true", help="Generate full simulation report")
    
    sim_group = parser.add_argument_group("Simulation Parameters")
    sim_group.add_argument("--layer", "-l", choices=["app", "infra", "mw-app", "mw-infra", "system"], default="system", help="Analysis layer")
    sim_group.add_argument("--layers", help="Comma-separated layers for report")
    sim_group.add_argument("--messages", "-m", type=int, default=100, help="Number of messages")
    sim_group.add_argument("--duration", "-d", type=float, default=10.0, help="Simulation duration in seconds")
    sim_group.add_argument("--cascade-prob", type=float, default=1.0, help="Cascade probability")
    
    neo4j_group = parser.add_argument_group("Neo4j Connection")
    neo4j_group.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j connection URI")
    neo4j_group.add_argument("--user", "-u", default="neo4j", help="Neo4j username")
    neo4j_group.add_argument("--password", "-p", default="password", help="Neo4j password")
    
    parser.add_argument("--output", "-o", metavar="FILE", help="Export to JSON file")
    parser.add_argument("--json", action="store_true", help="Output as JSON to stdout")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    log_level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    
    container = Container(uri=args.uri, user=args.user, password=args.password)
    display = container.display_service()
    
    try:
        sim = container.simulation_service()
        
        if args.event:
            result = sim.run_event_simulation(source_app=args.event, num_messages=args.messages, duration=args.duration)
            if args.json: print(json.dumps(result.to_dict(), indent=2))
            elif not args.quiet: display.display_event_result(result)
            if args.output:
                with open(args.output, 'w') as f: json.dump(result.to_dict(), f, indent=2)
                if not args.quiet: print(f"\n{display.colored(f'Results saved to: {args.output}', display.Colors.GREEN)}")
        
        elif args.failure:
            result = sim.run_failure_simulation(target_id=args.failure, layer=args.layer, cascade_probability=args.cascade_prob)
            if args.json: print(json.dumps(result.to_dict(), indent=2))
            elif not args.quiet: display.display_failure_result(result)
            if args.output:
                with open(args.output, 'w') as f: json.dump(result.to_dict(), f, indent=2)
                if not args.quiet: print(f"\n{display.colored(f'Results saved to: {args.output}', display.Colors.GREEN)}")
        
        elif args.exhaustive:
            results = sim.run_failure_simulation_exhaustive(layer=args.layer, cascade_probability=args.cascade_prob)
            if args.json: print(json.dumps([r.to_dict() for r in results], indent=2))
            elif not args.quiet: display.display_exhaustive_results(results)
            if args.output:
                with open(args.output, 'w') as f: json.dump([r.to_dict() for r in results], f, indent=2)
                if not args.quiet: print(f"\n{display.colored(f'Results saved to: {args.output}', display.Colors.GREEN)}")
        
        elif args.report:
            layers = args.layers.split(",") if args.layers else ["app", "infra", "system"]
            report = sim.generate_report(layers=layers)
            if args.json: print(json.dumps(report.to_dict(), indent=2))
            elif not args.quiet: display.display_simulation_report(report)
            if args.output:
                sim.export_report(report, args.output)
                if not args.quiet: print(f"\n{display.colored(f'Report saved to: {args.output}', display.Colors.GREEN)}")
        
        return 0
    except Exception as e:
        print(display.colored(f"Error: {e}", display.Colors.RED), file=sys.stderr)
        if args.verbose: logging.exception("Simulation failed")
        return 1
    finally:
        container.close()


if __name__ == "__main__":
    sys.exit(main())