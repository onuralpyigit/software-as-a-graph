#!/usr/bin/env python3
"""
Unified CLI Entry Point

Single entry point for all graph analysis, simulation, and validation commands.
Dispatches to the appropriate module based on the command.

Usage:
    python -m src.adapters.inbound.cli <command> [options]
    
Commands:
    analyze   - Run graph analysis
    simulate  - Run simulations (event/failure)
    validate  - Validate analysis vs simulation results
    visualize - Generate dashboards and visualizations
    import    - Import graph data into Neo4j
    export    - Export graph data to JSON
    generate  - Generate synthetic graph data
    benchmark - Run benchmarks
"""

import sys
import argparse
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="graph-cli",
        description="Unified CLI for graph analysis, simulation, and validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "command",
        choices=["analyze", "simulate", "validate", "visualize", "import", "export", "generate", "benchmark"],
        help="Command to execute",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to the command",
    )
    
    args = parser.parse_args(argv)
    
    # Map commands to bin scripts
    command_map = {
        "analyze": "bin.analyze_graph",
        "simulate": "bin.simulate_graph",
        "validate": "bin.validate_graph",
        "visualize": "bin.visualize_graph",
        "import": "bin.import_graph",
        "export": "bin.export_graph",
        "generate": "bin.generate_graph",
        "benchmark": "bin.benchmark",
    }
    
    module_name = command_map.get(args.command)
    if not module_name:
        parser.print_help()
        return 1
    
    # Import and run the module's main function
    try:
        import importlib
        module = importlib.import_module(module_name)
        
        # Set up sys.argv for the subcommand
        sys.argv = [module_name] + args.args
        
        if hasattr(module, "main"):
            return module.main() or 0
        else:
            print(f"Error: Module {module_name} does not have a main() function")
            return 1
            
    except ImportError as e:
        print(f"Error: Could not import {module_name}: {e}")
        return 1
    except Exception as e:
        print(f"Error running {args.command}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
