#!/usr/bin/env python3
"""
Graph Simulation CLI

Comprehensive simulation for distributed pub-sub systems.
Supports event-driven and failure simulation on the raw graph model.

Simulation Types:
    - Event: Message flow simulation (throughput, latency, drops)
    - Failure: Component failure with cascade propagation
    - Report: Full analysis with criticality classification

Layers:
    app      : Application layer
    infra    : Infrastructure layer
    mw-app   : Middleware-Application layer
    mw-infra : Middleware-Infrastructure layer
    system   : Complete system

Usage:
    python simulate_graph.py --event App1
    python simulate_graph.py --failure Broker1
    python simulate_graph.py --report --layers app,infra,system
    python simulate_graph.py --exhaustive --layer system

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.simulation import (
    Simulator,
    EventResult,
    FailureResult,
    SimulationReport,
)
from src.visualization.display import (
    display_event_result,
    display_failure_result,
    display_exhaustive_results,
    display_simulation_report as display_report,
)
from src.visualization.display import Colors, colored


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive simulation for distributed pub-sub systems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Simulation Types:
  --event APP_ID    Event simulation from source application
  --failure COMP_ID Failure simulation for target component
  --exhaustive      Failure analysis for all components
  --report          Full simulation report with analysis

Layers:
  app      Application layer (Applications only)
  infra    Infrastructure layer (Nodes only)
  mw-app   Middleware-Application (Applications + Brokers)
  mw-infra Middleware-Infrastructure (Nodes + Brokers)
  system   Complete system (all components)

Examples:
  %(prog)s --event App1 --messages 100
  %(prog)s --failure Broker1
  %(prog)s --exhaustive --layer system
  %(prog)s --report --layers app,infra,system
  %(prog)s --report --output results/report.json
        """
    )
    
    # Action (required)
    action_group = parser.add_argument_group("Action (one required)")
    action_mutex = action_group.add_mutually_exclusive_group(required=True)
    action_mutex.add_argument(
        "--event", "-e",
        metavar="APP_ID",
        help="Run event simulation from source application"
    )
    action_mutex.add_argument(
        "--failure", "-f",
        metavar="COMP_ID",
        help="Run failure simulation for target component"
    )
    action_mutex.add_argument(
        "--exhaustive",
        action="store_true",
        help="Run exhaustive failure analysis for all components"
    )
    action_mutex.add_argument(
        "--report", "-r",
        action="store_true",
        help="Generate full simulation report"
    )
    
    # Simulation parameters
    sim_group = parser.add_argument_group("Simulation Parameters")
    sim_group.add_argument(
        "--layer", "-l",
        choices=["app", "infra", "mw-app", "mw-infra", "system"],
        default="system",
        help="Analysis layer (default: system)"
    )
    sim_group.add_argument(
        "--layers",
        help="Comma-separated layers for report (default: app,infra,system)"
    )
    sim_group.add_argument(
        "--messages", "-m",
        type=int,
        default=100,
        help="Number of messages for event simulation (default: 100)"
    )
    sim_group.add_argument(
        "--duration", "-d",
        type=float,
        default=10.0,
        help="Event simulation duration in seconds (default: 10.0)"
    )
    sim_group.add_argument(
        "--cascade-prob",
        type=float,
        default=1.0,
        help="Cascade propagation probability (default: 1.0)"
    )
    
    # Neo4j connection
    neo4j_group = parser.add_argument_group("Neo4j Connection")
    neo4j_group.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j connection URI (default: bolt://localhost:7687)"
    )
    neo4j_group.add_argument(
        "--user", "-u",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    neo4j_group.add_argument(
        "--password", "-p",
        default="password",
        help="Neo4j password (default: password)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Export results to JSON file"
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout"
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with debug information"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    
    try:
        # Create container and repository
        from src.infrastructure import Container
        container = Container(uri=args.uri, user=args.user, password=args.password)
        repository = container.graph_repository()
        
        with Simulator(
            uri=args.uri,
            user=args.user,
            password=args.password,
            repository=repository
        ) as sim:
            
            # === Event Simulation ===
            if args.event:
                result = sim.run_event_simulation(
                    source_app=args.event,
                    num_messages=args.messages,
                    duration=args.duration,
                )
                
                if args.json:
                    print(json.dumps(result.to_dict(), indent=2))
                elif not args.quiet:
                    display_event_result(result)
                
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(result.to_dict(), f, indent=2)
                    if not args.quiet:
                        print(f"\n{colored(f'Results saved to: {args.output}', Colors.GREEN)}")
            
            # === Failure Simulation ===
            elif args.failure:
                result = sim.run_failure_simulation(
                    target_id=args.failure,
                    layer=args.layer,
                    cascade_probability=args.cascade_prob,
                )
                
                if args.json:
                    print(json.dumps(result.to_dict(), indent=2))
                elif not args.quiet:
                    display_failure_result(result)
                
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(result.to_dict(), f, indent=2)
                    if not args.quiet:
                        print(f"\n{colored(f'Results saved to: {args.output}', Colors.GREEN)}")
            
            # === Exhaustive Analysis ===
            elif args.exhaustive:
                results = sim.run_failure_simulation_exhaustive(
                    layer=args.layer,
                    cascade_probability=args.cascade_prob,
                )
                
                if args.json:
                    print(json.dumps([r.to_dict() for r in results], indent=2))
                elif not args.quiet:
                    display_exhaustive_results(results)
                
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump([r.to_dict() for r in results], f, indent=2)
                    if not args.quiet:
                        print(f"\n{colored(f'Results saved to: {args.output}', Colors.GREEN)}")
            
            # === Report ===
            elif args.report:
                layers = args.layers.split(",") if args.layers else ["app", "infra", "system"]
                
                report = sim.generate_report(layers=layers)
                
                if args.json:
                    print(json.dumps(report.to_dict(), indent=2))
                elif not args.quiet:
                    display_report(report)
                
                if args.output:
                    sim.export_report(report, args.output)
                    if not args.quiet:
                        print(f"\n{colored(f'Report saved to: {args.output}', Colors.GREEN)}")
        

        
        return 0
    
    except KeyboardInterrupt:
        print(f"\n{colored('Simulation interrupted.', Colors.YELLOW)}")
        return 130
    
    except Exception as e:
        logging.exception("Simulation failed")
        print(f"{colored(f'Error: {e}', Colors.RED)}", file=sys.stderr)
        return 1

    finally:
        container.close()


if __name__ == "__main__":
    sys.exit(main())