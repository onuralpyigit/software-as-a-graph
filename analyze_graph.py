#!/usr/bin/env python3
"""
Software-as-a-Graph Analysis CLI

Analyzes multi-layer graph models to identify critical components,
dependencies, and architectural risks in distributed pub-sub systems.

Usage:
    python analyze_graph.py --layer application
    python analyze_graph.py --all
    python analyze_graph.py --layer complete
    python analyze_graph.py --output results/analysis.json

Author: Software-as-a-Graph Research Project
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis import (
    GraphAnalyzer,
    LayerAnalysisResult,
    MultiLayerAnalysisResult,
    CriticalityLevel,
    AnalysisLayer,
)

# ============================================================================
# ANSI Color Codes
# ============================================================================
COLORS = {
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "MAGENTA": "\033[95m",
    "CYAN": "\033[96m",
    "WHITE": "\033[97m",
    "GRAY": "\033[90m",
    "BOLD": "\033[1m",
    "RESET": "\033[0m",
}

# Disable colors if not a TTY
if not sys.stdout.isatty():
    COLORS = {k: "" for k in COLORS}

RED = COLORS["RED"]
GREEN = COLORS["GREEN"]
YELLOW = COLORS["YELLOW"]
BLUE = COLORS["BLUE"]
MAGENTA = COLORS["MAGENTA"]
CYAN = COLORS["CYAN"]
WHITE = COLORS["WHITE"]
GRAY = COLORS["GRAY"]
BOLD = COLORS["BOLD"]
RESET = COLORS["RESET"]


# ============================================================================
# Display Functions
# ============================================================================

def print_header(title: str, char: str = "=") -> None:
    """Print a section header."""
    width = 70
    print(f"\n{BOLD}{char * width}{RESET}")
    print(f"{BOLD} {title}{RESET}")
    print(f"{BOLD}{char * width}{RESET}")


def print_subheader(title: str) -> None:
    """Print a subsection header."""
    print(f"\n{BOLD}>> {title}{RESET}")


def level_color(level: CriticalityLevel) -> str:
    """Get color for criticality level."""
    return {
        CriticalityLevel.CRITICAL: RED,
        CriticalityLevel.HIGH: YELLOW,
        CriticalityLevel.MEDIUM: BLUE,
        CriticalityLevel.LOW: WHITE,
        CriticalityLevel.MINIMAL: GRAY,
    }.get(level, RESET)


def format_level(level: CriticalityLevel) -> str:
    """Format criticality level with color."""
    color = level_color(level)
    return f"{color}{level.value.upper():8}{RESET}"


def display_graph_summary(result: LayerAnalysisResult) -> None:
    """Display graph topology summary."""
    summary = result.structural.graph_summary
    
    print(f"\n  {CYAN}Layer:{RESET}        {result.layer_name}")
    print(f"  {CYAN}Description:{RESET}  {result.description}")
    print(f"  {CYAN}Nodes:{RESET}        {summary.nodes}")
    print(f"  {CYAN}Edges:{RESET}        {summary.edges}")
    print(f"  {CYAN}Density:{RESET}      {summary.density:.4f}")
    print(f"  {CYAN}Connected:{RESET}    {GREEN if summary.is_connected else RED}{'Yes' if summary.is_connected else 'No'}{RESET}")
    print(f"  {CYAN}Components:{RESET}   {summary.num_components}")
    
    if summary.num_articulation_points > 0:
        print(f"  {CYAN}SPOFs:{RESET}        {RED}{summary.num_articulation_points}{RESET}")
    if summary.num_bridges > 0:
        print(f"  {CYAN}Bridges:{RESET}      {YELLOW}{summary.num_bridges}{RESET}")
    
    # Node type breakdown
    if summary.node_types:
        types_str = ", ".join(f"{k}: {v}" for k, v in summary.node_types.items())
        print(f"  {CYAN}Node Types:{RESET}   {types_str}")
    
    # Edge type breakdown
    if summary.edge_types:
        types_str = ", ".join(f"{k}: {v}" for k, v in summary.edge_types.items())
        print(f"  {CYAN}Edge Types:{RESET}   {types_str}")


def display_classification_summary(result: LayerAnalysisResult) -> None:
    """Display classification distribution."""
    summary = result.quality.classification_summary
    
    print_subheader("Classification Distribution")
    
    # Component distribution
    comp_dist = summary.component_distribution
    total = sum(comp_dist.values())
    
    print(f"\n  Components ({total} total):")
    for level in [CriticalityLevel.CRITICAL, CriticalityLevel.HIGH, 
                  CriticalityLevel.MEDIUM, CriticalityLevel.LOW, CriticalityLevel.MINIMAL]:
        count = comp_dist.get(level.value, 0)
        pct = count / total * 100 if total > 0 else 0
        bar_len = int(pct / 5)  # Scale to max 20 chars
        bar = "█" * bar_len
        color = level_color(level)
        print(f"    {color}{level.value.upper():8}{RESET} {count:3}  ({pct:5.1f}%)  {color}{bar}{RESET}")
    
    # Edge distribution
    edge_dist = summary.edge_distribution
    total_edges = sum(edge_dist.values())
    
    if total_edges > 0:
        print(f"\n  Dependencies ({total_edges} total):")
        for level in [CriticalityLevel.CRITICAL, CriticalityLevel.HIGH]:
            count = edge_dist.get(level.value, 0)
            if count > 0:
                color = level_color(level)
                print(f"    {color}{level.value.upper():8}{RESET} {count:3}")


def display_critical_components(result: LayerAnalysisResult, limit: int = 15) -> None:
    """Display top critical components."""
    critical = [c for c in result.quality.components if c.levels.overall >= CriticalityLevel.HIGH]
    critical.sort(key=lambda x: x.scores.overall, reverse=True)
    
    print_subheader(f"Critical Components (Box-Plot Outliers)")
    
    if not critical:
        print(f"  {GREEN}✓ No critical components detected.{RESET}")
        return
    
    # Header
    print(f"\n  {'-' * 78}")
    print(f"  {'ID':<20} {'Type':<12} {'Overall':<8} {'R':<8} {'M':<8} {'A':<8} {'Flags':<10}")
    print(f"  {'-' * 78}")
    
    for c in critical[:limit]:
        # Build flags
        flags = []
        if c.structural.is_articulation_point:
            flags.append(f"{RED}SPOF{RESET}")
        if c.structural.bridge_ratio > 0.3:
            flags.append(f"{YELLOW}BR{RESET}")
        flags_str = " ".join(flags) if flags else "-"
        
        # Color by overall level
        color = level_color(c.levels.overall)
        
        # Level indicators
        r_mark = level_color(c.levels.reliability) + c.levels.reliability.symbol + RESET
        m_mark = level_color(c.levels.maintainability) + c.levels.maintainability.symbol + RESET
        a_mark = level_color(c.levels.availability) + c.levels.availability.symbol + RESET
        
        print(
            f"  {c.id:<20} {c.type:<12} "
            f"{color}{c.scores.overall:.3f}{RESET}    "
            f"{r_mark:<8} {m_mark:<8} {a_mark:<8} {flags_str}"
        )
    
    if len(critical) > limit:
        print(f"\n  {GRAY}... and {len(critical) - limit} more{RESET}")


def display_critical_edges(result: LayerAnalysisResult, limit: int = 10) -> None:
    """Display critical edges/dependencies."""
    critical_edges = [e for e in result.quality.edges if e.level >= CriticalityLevel.HIGH]
    critical_edges.sort(key=lambda x: x.scores.overall, reverse=True)
    
    if not critical_edges:
        return
    
    print_subheader("Critical Dependencies")
    
    print(f"\n  {'-' * 65}")
    print(f"  {'Source':<15} {'→':<3} {'Target':<15} {'Type':<12} {'Score':<8} {'Bridge'}")
    print(f"  {'-' * 65}")
    
    for e in critical_edges[:limit]:
        color = level_color(e.level)
        bridge_mark = f"{RED}Yes{RESET}" if e.is_bridge else f"{GRAY}No{RESET}"
        
        print(
            f"  {e.source:<15} → {e.target:<15} "
            f"{e.type:<12} {color}{e.scores.overall:.3f}{RESET}    {bridge_mark}"
        )


def display_problems(result: LayerAnalysisResult, limit: int = 20) -> None:
    """Display detected problems."""
    problems = result.problems
    
    print_subheader("Detected Problems & Risks")
    
    if not problems:
        print(f"  {GREEN}✓ No architectural problems detected.{RESET}")
        return
    
    # Summary
    summary = result.problem_summary
    print(f"\n  Total: {summary.total_problems} problems")
    print(f"  By Severity: ", end="")
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        count = summary.by_severity.get(sev, 0)
        if count > 0:
            color = RED if sev == "CRITICAL" else (YELLOW if sev == "HIGH" else BLUE)
            print(f"{color}{sev}: {count}{RESET}  ", end="")
    print()
    
    # Details
    print(f"\n  {'-' * 75}")
    
    for i, p in enumerate(problems[:limit]):
        if i > 0:
            print(f"  {'-' * 75}")
        
        sev_color = RED if p.severity == "CRITICAL" else (YELLOW if p.severity == "HIGH" else BLUE)
        
        print(f"  [{sev_color}{p.severity:8}{RESET}] {BOLD}{p.name}{RESET}")
        print(f"             Entity: {CYAN}{p.entity_id}{RESET} ({p.entity_type})")
        print(f"             Category: {p.category}")
        
        # Wrap description
        desc_lines = _wrap_text(p.description, 55)
        print(f"             Issue: {desc_lines[0]}")
        for line in desc_lines[1:]:
            print(f"                    {line}")
        
        # Wrap recommendation
        rec_lines = _wrap_text(p.recommendation, 55)
        print(f"             Fix: {GREEN}{rec_lines[0]}{RESET}")
        for line in rec_lines[1:]:
            print(f"                  {GREEN}{line}{RESET}")
    
    if len(problems) > limit:
        print(f"\n  {GRAY}... and {len(problems) - limit} more{RESET}")


def display_layer_result(result: LayerAnalysisResult) -> None:
    """Display complete analysis result for a single layer."""
    print_header(f"Analysis: {result.layer_name}")
    
    display_graph_summary(result)
    display_classification_summary(result)
    display_critical_components(result)
    display_critical_edges(result)
    display_problems(result)


def display_multi_layer_result(results: MultiLayerAnalysisResult) -> None:
    """Display analysis results for multiple layers."""
    print_header("Multi-Layer Analysis Results", "═")
    print(f"\n  Timestamp: {results.timestamp}")
    print(f"  Layers analyzed: {', '.join(results.layers.keys())}")
    
    for layer_name, layer_result in results.layers.items():
        display_layer_result(layer_result)
    
    # Cross-layer insights
    if results.cross_layer_insights:
        print_header("Cross-Layer Insights", "-")
        for insight in results.cross_layer_insights:
            print(f"  • {insight}")


def _wrap_text(text: str, width: int) -> List[str]:
    """Wrap text to specified width."""
    words = text.split()
    lines = []
    current_line = []
    current_len = 0
    
    for word in words:
        if current_len + len(word) + 1 <= width:
            current_line.append(word)
            current_len += len(word) + 1
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_len = len(word)
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return lines or [""]


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze multi-layer graph models for distributed pub-sub systems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --layer application
  %(prog)s --all
  %(prog)s --output results/analysis.json
  %(prog)s --layer complete
        """
    )
    # Layer selection
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument(
        "--layer", "-l",
        choices=["application", "infrastructure", "complete", "app_broker", "node_broker"],
        default="complete",
        help="Analysis layer (default: complete)"
    )
    layer_group.add_argument(
        "--all", "-a",
        action="store_true",
        help="Analyze all primary layers (application, infrastructure, complete)"
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Export results to JSON file"
    )
    
    # Neo4j authentication
    parser.add_argument(
        "--uri", "-n",
        default="bolt://localhost:7687",
        help="Neo4j connection URI (default: bolt://localhost:7687)"
    )

    parser.add_argument(
        "--user", "-u",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    parser.add_argument(
        "--password", "-p",
        default="password",
        help="Neo4j password (default: password)"
    )
    
    # Analysis options
    parser.add_argument(
        "--k-factor", "-k",
        type=float,
        default=1.5,
        help="Box-plot IQR multiplier for outlier detection (default: 1.5)"
    )
    parser.add_argument(
        "--damping", "-d",
        type=float,
        default=0.85,
        help="PageRank damping factor (default: 0.85)"
    )
    
    # Display options
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output (useful with --output)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with debug information"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    try:
        # Create analyzer
        analyzer_kwargs = {
            "damping_factor": args.damping,
            "k_factor": args.k_factor,
        }
        analyzer_kwargs["uri"] = args.uri
        analyzer_kwargs["user"] = args.user
        analyzer_kwargs["password"] = args.password
        
        with GraphAnalyzer(**analyzer_kwargs) as analyzer:
            # Run analysis
            if args.all:
                results = analyzer.analyze_all_layers()
            else:
                layer_result = analyzer.analyze_layer(args.layer)
                results = MultiLayerAnalysisResult(
                    timestamp=datetime.now().isoformat(),
                    layers={layer_result.layer: layer_result},
                    cross_layer_insights=[],
                )
            
            # Export if requested
            if args.output:
                analyzer.export_results(results, args.output)
                if not args.quiet:
                    print(f"\n{GREEN}Results exported to: {args.output}{RESET}")
            
            # Output
            if args.json:
                print(json.dumps(results.to_dict(), indent=2))
            elif not args.quiet:
                if args.all or len(results.layers) > 1:
                    display_multi_layer_result(results)
                else:
                    layer_result = list(results.layers.values())[0]
                    display_layer_result(layer_result)
                
                # Final summary
                all_problems = results.get_all_problems()
                critical_count = sum(1 for p in all_problems if p.severity == "CRITICAL")
                high_count = sum(1 for p in all_problems if p.severity == "HIGH")
                
                print_header("Summary", "-")
                if critical_count > 0:
                    print(f"\n  {RED}⚠ {critical_count} CRITICAL issues require immediate attention{RESET}")
                if high_count > 0:
                    print(f"  {YELLOW}⚠ {high_count} HIGH priority issues should be reviewed{RESET}")
                if critical_count == 0 and high_count == 0:
                    print(f"\n  {GREEN}✓ No high-priority issues detected{RESET}")
                
                print()
        
        return 0
    
    except FileNotFoundError as e:
        print(f"{RED}Error: {e}{RESET}", file=sys.stderr)
        return 1
    except Exception as e:
        logging.exception("Analysis failed")
        print(f"{RED}Error: {e}{RESET}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())