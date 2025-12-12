#!/usr/bin/env python3
"""
Graph Generation CLI - Simplified Version 3.0

Generates pub-sub system graphs with the simplified model:

Vertices:
- Application: {id, name, role (pub|sub|pubsub)}
- Broker: {id, name}
- Topic: {id, name, size, qos {durability, reliability, transport_priority}}
- Node: {id, name}

Edges:
- PUBLISHES_TO (App → Topic)
- SUBSCRIBES_TO (App → Topic)
- ROUTES (Broker → Topic)
- RUNS_ON (App/Broker → Node)
- CONNECTS_TO (Node → Node)

Usage:
    python generate_graph.py --scale small --output system.json
    python generate_graph.py --scale medium --scenario financial --output financial.json
    python generate_graph.py --scale large --antipatterns spof god_topic --output problematic.json
    python generate_graph.py --scale xlarge --preview
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.graph_generator import GraphGenerator, GraphConfig


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def supports_color() -> bool:
    """Check if terminal supports color output"""
    if os.getenv('NO_COLOR'):
        return False
    if os.getenv('FORCE_COLOR'):
        return True
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


if not supports_color():
    for attr in dir(Colors):
        if not attr.startswith('_'):
            setattr(Colors, attr, '')


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.END}\n")


def print_section(text: str):
    """Print section header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.BLUE}{'-'*len(text)}{Colors.END}")


def ok(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.END} {text}")


def err(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.END} {text}")


def warn(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")


def info(text: str):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ{Colors.END} {text}")


def validate_graph(graph: Dict) -> Tuple[bool, List[str], List[str]]:
    """Validate graph structure"""
    errors = []
    warnings = []
    
    # Check required keys
    required = ['metadata', 'applications', 'brokers', 'topics', 'nodes', 'relationships']
    for key in required:
        if key not in graph:
            errors.append(f"Missing required key: {key}")
    
    if errors:
        return False, errors, warnings
    
    # Collect all IDs
    app_ids = {a['id'] for a in graph.get('applications', [])}
    broker_ids = {b['id'] for b in graph.get('brokers', [])}
    topic_ids = {t['id'] for t in graph.get('topics', [])}
    node_ids = {n['id'] for n in graph.get('nodes', [])}
    all_ids = app_ids | broker_ids | topic_ids | node_ids
    
    # Validate edges
    edges = graph.get('relationships', {})
    
    for pub in edges.get('publishes_to', []):
        if pub.get('from') not in app_ids:
            errors.append(f"PUBLISHES_TO: unknown app {pub.get('from')}")
        if pub.get('to') not in topic_ids:
            errors.append(f"PUBLISHES_TO: unknown topic {pub.get('to')}")
    
    for sub in edges.get('subscribes_to', []):
        if sub.get('from') not in app_ids:
            errors.append(f"SUBSCRIBES_TO: unknown app {sub.get('from')}")
        if sub.get('to') not in topic_ids:
            errors.append(f"SUBSCRIBES_TO: unknown topic {sub.get('to')}")
    
    for route in edges.get('routes', []):
        if route.get('from') not in broker_ids:
            errors.append(f"ROUTES: unknown broker {route.get('from')}")
        if route.get('to') not in topic_ids:
            errors.append(f"ROUTES: unknown topic {route.get('to')}")
    
    for runs in edges.get('runs_on', []):
        if runs.get('from') not in (app_ids | broker_ids):
            errors.append(f"RUNS_ON: unknown component {runs.get('from')}")
        if runs.get('to') not in node_ids:
            errors.append(f"RUNS_ON: unknown node {runs.get('to')}")
    
    for conn in edges.get('connects_to', []):
        if conn.get('from') not in node_ids:
            errors.append(f"CONNECTS_TO: unknown node {conn.get('from')}")
        if conn.get('to') not in node_ids:
            errors.append(f"CONNECTS_TO: unknown node {conn.get('to')}")
    
    # Check connectivity
    connected_apps = set()
    for pub in edges.get('publishes_to', []):
        connected_apps.add(pub['from'])
    for sub in edges.get('subscribes_to', []):
        connected_apps.add(sub['from'])
    
    disconnected = app_ids - connected_apps
    if disconnected:
        warnings.append(f"{len(disconnected)} applications have no pub/sub connections")
    
    return len(errors) == 0, errors, warnings


def export_graph(graph: Dict, output_path: str, formats: List[str]) -> List[str]:
    """Export graph to multiple formats"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    exported = []
    
    # JSON export
    if 'json' in formats:
        json_path = output.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(graph, f, indent=2, default=str)
        exported.append(str(json_path))
        ok(f"JSON: {json_path}")
    
    # Other formats via GraphBuilder/Exporter
    other = [f for f in formats if f != 'json']
    if other:
        try:
            from src.core.graph_builder import GraphBuilder
            from src.core.graph_exporter import GraphExporter
            
            builder = GraphBuilder()
            model = builder.build_from_dict(graph)
            exporter = GraphExporter()
            
            for fmt in other:
                try:
                    path = str(output.with_suffix(f'.{fmt}'))
                    if fmt == 'graphml':
                        exporter.export_to_graphml(model, path)
                    elif fmt == 'gexf':
                        exporter.export_to_gexf(model, path)
                    elif fmt == 'dot':
                        exporter.export_to_dot(model, path)
                    else:
                        warn(f"Unknown format: {fmt}")
                        continue
                    exported.append(path)
                    ok(f"{fmt.upper()}: {path}")
                except Exception as e:
                    warn(f"Failed to export {fmt}: {e}")
        except ImportError as e:
            warn(f"Export modules not available: {e}")
    
    return exported


def print_preview(config: GraphConfig):
    """Print preview of what will be generated"""
    print_header("GENERATION PREVIEW")
    
    scale_defaults = GraphGenerator.SCALES.get(config.scale, {})
    
    print(f"{Colors.BOLD}Scale:{Colors.END} {config.scale}")
    print(f"{Colors.BOLD}Scenario:{Colors.END} {config.scenario}")
    print(f"{Colors.BOLD}Seed:{Colors.END} {config.seed}")
    
    print_section("Vertices to Generate")
    print(f"  Applications: {config.num_applications or scale_defaults.get('apps', '?')}")
    print(f"  Brokers:      {config.num_brokers or scale_defaults.get('brokers', '?')}")
    print(f"  Topics:       {config.num_topics or scale_defaults.get('topics', '?')}")
    print(f"  Nodes:        {config.num_nodes or scale_defaults.get('nodes', '?')}")
    
    num_apps = config.num_applications or scale_defaults.get('apps', 50)
    num_topics = config.num_topics or scale_defaults.get('topics', 25)
    num_brokers = config.num_brokers or scale_defaults.get('brokers', 3)
    num_nodes = config.num_nodes or scale_defaults.get('nodes', 10)
    
    est_pub = int(num_apps * 0.6 * 3)
    est_sub = int(num_apps * 0.7 * 5)
    est_routes = num_topics
    est_runs = num_apps + num_brokers
    est_connects = num_nodes + num_nodes // 3
    
    print_section("Estimated Edges")
    print(f"  PUBLISHES_TO:  ~{est_pub}")
    print(f"  SUBSCRIBES_TO: ~{est_sub}")
    print(f"  ROUTES:        ~{est_routes}")
    print(f"  RUNS_ON:       ~{est_runs}")
    print(f"  CONNECTS_TO:   ~{est_connects}")
    print(f"  Total:         ~{est_pub + est_sub + est_routes + est_runs + est_connects}")
    
    if config.antipatterns:
        print_section("Anti-patterns to Inject")
        for ap in config.antipatterns:
            print(f"  • {ap}")
    
    print(f"\n{Colors.DIM}Use without --preview to generate{Colors.END}")


def print_statistics(graph: Dict, gen_time: float):
    """Print graph statistics"""
    print_header("GRAPH STATISTICS")
    
    # Metadata
    meta = graph.get('metadata', {})
    print_section("Configuration")
    print(f"  Scale:    {meta.get('scale', 'N/A')}")
    print(f"  Scenario: {meta.get('scenario', 'N/A')}")
    print(f"  Seed:     {meta.get('seed', 'N/A')}")
    print(f"  Time:     {gen_time:.3f}s")
    
    # Vertices
    print_section("Vertices")
    print(f"  Applications: {len(graph['applications']):>6}")
    print(f"  Brokers:      {len(graph['brokers']):>6}")
    print(f"  Topics:       {len(graph['topics']):>6}")
    print(f"  Nodes:        {len(graph['nodes']):>6}")
    
    # Edges
    edges = graph['relationships']
    print_section("Edges")
    print(f"  PUBLISHES_TO:  {len(edges.get('publishes_to', [])):>6}")
    print(f"  SUBSCRIBES_TO: {len(edges.get('subscribes_to', [])):>6}")
    print(f"  ROUTES:        {len(edges.get('routes', [])):>6}")
    print(f"  RUNS_ON:       {len(edges.get('runs_on', [])):>6}")
    print(f"  CONNECTS_TO:   {len(edges.get('connects_to', [])):>6}")
    total = sum(len(v) for v in edges.values())
    print(f"  {Colors.BOLD}Total:{Colors.END}          {total:>6}")
    
    # Role distribution
    print_section("Application Roles")
    role_counts = {}
    for app in graph['applications']:
        role = app.get('role', 'unknown')
        role_counts[role] = role_counts.get(role, 0) + 1
    
    total_apps = len(graph['applications'])
    for role, count in sorted(role_counts.items()):
        pct = count / total_apps * 100 if total_apps else 0
        bar = '█' * int(pct / 5)
        print(f"  {role:<8} {count:>4} ({pct:>5.1f}%) {bar}")
    
    # QoS distribution
    print_section("QoS Distribution")
    qos_counts = {'durability': {}, 'reliability': {}, 'transport_priority': {}}
    for topic in graph['topics']:
        qos = topic.get('qos', {})
        for key in qos_counts:
            val = qos.get(key, 'UNKNOWN')
            qos_counts[key][val] = qos_counts[key].get(val, 0) + 1
    
    for qos_type, counts in qos_counts.items():
        print(f"  {qos_type}:")
        for val, count in sorted(counts.items()):
            pct = count / len(graph['topics']) * 100 if graph['topics'] else 0
            print(f"    {val:<18} {count:>4} ({pct:>5.1f}%)")
    
    # Anti-patterns
    ap_applied = meta.get('antipatterns_applied')
    if ap_applied:
        print_section("Anti-patterns Applied")
        for ap_name, ap_info in ap_applied.items():
            print(f"  {ap_name}: {ap_info}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate pub-sub system graphs (simplified model)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --scale small --output system.json
  %(prog)s --scale medium --scenario financial --output financial.json
  %(prog)s --scale large --antipatterns spof god_topic --output bad.json
  %(prog)s --scale xlarge --preview
        """
    )
    
    # Scale and scenario
    parser.add_argument(
        '--scale', '-s',
        choices=['tiny', 'small', 'medium', 'large', 'xlarge', 'extreme'],
        help='Scale preset'
    )
    parser.add_argument(
        '--scenario', '-S',
        choices=['generic', 'iot', 'financial', 'ecommerce', 'analytics', 
                'smart_city', 'healthcare', 'autonomous_vehicle', 'gaming'],
        default='generic',
        help='Domain scenario (default: generic)'
    )
    
    # Output
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument(
        '--formats', '-f',
        nargs='+',
        choices=['json', 'graphml', 'gexf', 'dot'],
        default=['json'],
        help='Export formats (default: json)'
    )
    
    # Override counts
    parser.add_argument('--num-nodes', type=int, help='Number of nodes')
    parser.add_argument('--num-apps', type=int, help='Number of applications')
    parser.add_argument('--num-topics', type=int, help='Number of topics')
    parser.add_argument('--num-brokers', type=int, help='Number of brokers')
    
    # Anti-patterns
    parser.add_argument(
        '--antipatterns', '-a',
        nargs='+',
        choices=['spof', 'broker_overload', 'god_topic', 'single_broker',
                'tight_coupling', 'chatty', 'bottleneck', 'circular_dependency'],
        default=[],
        help='Anti-patterns to inject'
    )
    
    # Options
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--preview', action='store_true', help='Preview without generating')
    parser.add_argument('--validate', action='store_true', help='Validate after generation')
    parser.add_argument('--stats', action='store_true', help='Show detailed statistics')
    parser.add_argument('--validate-only', action='store_true', help='Validate existing file')
    parser.add_argument('--input', '-i', help='Input file for validation')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-q', '--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    # Validate-only mode
    if args.validate_only:
        if not args.input:
            err("--input required for --validate-only")
            return 1
        
        print_header("GRAPH VALIDATION")
        info(f"Validating: {args.input}")
        
        try:
            with open(args.input) as f:
                graph = json.load(f)
            
            is_valid, errors, warnings = validate_graph(graph)
            
            if errors:
                print_section("Errors")
                for e in errors[:10]:
                    print(f"  {Colors.RED}•{Colors.END} {e}")
                if len(errors) > 10:
                    print(f"  ... and {len(errors) - 10} more")
            
            if warnings:
                print_section("Warnings")
                for w in warnings[:10]:
                    print(f"  {Colors.YELLOW}•{Colors.END} {w}")
            
            if is_valid:
                ok("Validation passed")
                return 0
            else:
                err("Validation failed")
                return 1
        
        except FileNotFoundError:
            err(f"File not found: {args.input}")
            return 1
        except json.JSONDecodeError as e:
            err(f"Invalid JSON: {e}")
            return 1
    
    # Check scale required
    if not args.scale:
        parser.print_help()
        print()
        err("--scale is required")
        return 1
    
    # Create config
    try:
        config = GraphConfig(
            scale=args.scale,
            scenario=args.scenario,
            num_nodes=args.num_nodes,
            num_applications=args.num_apps,
            num_topics=args.num_topics,
            num_brokers=args.num_brokers,
            antipatterns=args.antipatterns or [],
            seed=args.seed
        )
    except ValueError as e:
        err(f"Invalid configuration: {e}")
        return 1
    
    # Preview mode
    if args.preview:
        print_preview(config)
        return 0
    
    # Check output
    if not args.output:
        err("--output is required")
        return 1
    
    # Generate
    if not args.quiet:
        print_header("GRAPH GENERATION")
        info(f"Scale: {args.scale}, Scenario: {args.scenario}")
        if args.antipatterns:
            info(f"Anti-patterns: {', '.join(args.antipatterns)}")
    
    start = time.time()
    
    try:
        generator = GraphGenerator(config)
        graph = generator.generate()
        gen_time = time.time() - start
        
        if not args.quiet:
            ok(f"Generated in {gen_time:.3f}s")
    
    except Exception as e:
        err(f"Generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    # Validate
    if args.validate:
        is_valid, errors, warnings = validate_graph(graph)
        if not is_valid:
            err("Validation failed")
            for e in errors[:5]:
                print(f"  {e}")
            return 1
        ok("Validation passed")
    
    # Export
    if not args.quiet:
        info("Exporting...")
    
    try:
        exported = export_graph(graph, args.output, args.formats)
        if not args.quiet and exported:
            ok(f"Exported {len(exported)} file(s)")
    except Exception as e:
        err(f"Export failed: {e}")
        return 1
    
    # Statistics
    if args.stats or args.verbose:
        print_statistics(graph, gen_time)
    
    if not args.quiet:
        print()
        ok("Complete!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())