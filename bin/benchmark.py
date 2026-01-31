#!/usr/bin/env python3
"""
Benchmark Suite for Graph-Based Criticality Prediction

Evaluates the Software-as-a-Graph methodology across varying system scales
and configurations to validate predictive accuracy and measure performance.

Usage:
    python benchmark.py --scales small,medium,large --runs 3
    python benchmark.py --config measure_config.yaml
    python benchmark.py --full-suite --output results/benchmark
"""

import sys
import argparse
import time
import logging
import yaml
from pathlib import Path
from typing import List

# Ensure src is in path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.benchmark import (
    BenchmarkRunner, 
    BenchmarkScenario, 
    ReportGenerator,
    BenchmarkSummary
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Graph Methodology Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration mode
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--scales", 
        help="Comma-separated list of scales (tiny,small,medium,large,xlarge)"
    )
    group.add_argument(
        "--config", 
        type=Path, 
        help="Path to YAML configuration file defining scenarios"
    )
    group.add_argument(
        "--full-suite",
        action="store_true",
        help="Run comprehensive full benchmark suite"
    )
    
    # Common options
    parser.add_argument(
        "--layers", 
        default="app,infra,system",
        help="Comma-separated list of layers to analyze"
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per configuration")
    parser.add_argument("--output", default="results/benchmark", help="Output directory")
    
    # Neo4j connection
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j User")
    parser.add_argument("--password", default="password", help="Neo4j Password")
    
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser.parse_args()

def load_scenarios_from_config(config_path: Path) -> List[BenchmarkScenario]:
    """Load benchmark scenarios from a YAML file."""
    if not config_path.exists():
        print(f"Error: Config file {config_path} not found.")
        sys.exit(1)
        
    with open(config_path) as f:
        data = yaml.safe_load(f)
        
    scenarios = []
    for item in data.get("scenarios", []):
        # Allow defining graph config inline or via file reference in the graph config tool
        # But here we need to map to BenchmarkScenario
        # Case 1: Scale preset
        scale = item.get("scale")
        
        # Case 2: Custom graph config file
        graph_config_file = item.get("graph_config")
        if graph_config_file:
            graph_config_path = Path(graph_config_file)
            if not graph_config_path.is_absolute():
                 # Resolve relative to benchmark config file
                 graph_config_path = config_path.parent / graph_config_path
        else:
            graph_config_path = None
            
        scenarios.append(BenchmarkScenario(
            name=item.get("name", "Unnamed"),
            scale=scale,
            config_path=graph_config_path,
            layers=item.get("layers", ["app", "infra", "system"]),
            runs=item.get("runs", 1)
        ))
    return scenarios

def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    output_dir = Path(args.output)
    
    # Initialize Runner
    runner = BenchmarkRunner(
        output_dir=output_dir,
        uri=args.uri,
        user=args.user,
        password=args.password,
        verbose=args.verbose
    )
    
    # Define Scenarios
    scenarios = []
    
    if args.config:
        print(f"Loading suite configuration from {args.config}...")
        scenarios = load_scenarios_from_config(args.config)
        
    elif args.full_suite:
        # Define full suite
        scales = ["tiny", "small", "medium"] # Large/XLarge can be slow
        layers = ["app", "infra", "system"]
        for scale in scales:
            scenarios.append(BenchmarkScenario(
                name=f"Full-{scale}",
                scale=scale,
                layers=layers,
                runs=args.runs
            ))
            
    else:
        # CLI fallback
        scales = args.scales.split(",") if args.scales else ["medium"]
        layers = args.layers.split(",")
        for scale in scales:
            scenarios.append(BenchmarkScenario(
                name=f"CLI-{scale}",
                scale=scale.strip(),
                layers=[l.strip() for l in layers],
                runs=args.runs
            ))

    print(f"Starting Benchmark Suite ({len(scenarios)} scenarios)...")
    start_time = time.time()
    
    try:
        total_records = 0
        for i, scenario in enumerate(scenarios):
             print(f"\nScenario {i+1}/{len(scenarios)}: {scenario.name} "
                   f"(Scale={scenario.scale or 'Custom'}, Runs={scenario.runs})")
             
             records = runner.run_scenario(scenario)
             total_records += len(records)
             
             passed = sum(1 for r in records if r.passed)
             print(f"  Completed {len(records)} runs. Passed: {passed}/{len(records)}")
             
        duration = time.time() - start_time
        summary = runner.aggregate_results(duration)
        
        # Reports
        reporter = ReportGenerator(output_dir)
        json_path = reporter.save_json(summary)
        md_path = reporter.generate_markdown(summary)
        
        print(f"\nBenchmark Complete in {duration:.1f}s")
        print(f"Results saved to:")
        print(f"  - {json_path}")
        print(f"  - {md_path}")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
