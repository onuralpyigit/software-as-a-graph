#!/usr/bin/env python3
"""
Analyze Graph CLI - Version 5.0

Command-line interface for graph-based analysis of distributed pub-sub systems.

Features:
- Component-type-specific analysis (Application, Topic, Node, Broker)
- Centrality algorithms via Neo4j GDS (PageRank, Betweenness, Degree)
- Box-plot statistical classification
- Problem detection (Reliability, Maintainability, Availability)
- Anti-pattern detection
- Critical edge analysis

Usage:
    # Full analysis (all features)
    python analyze_graph.py --full

    # Analyze specific component type
    python analyze_graph.py --component-type Application
    python analyze_graph.py --component-type Broker

    # Analyze all component types
    python analyze_graph.py --all-types

    # Detect problems
    python analyze_graph.py --problems

    # Detect anti-patterns
    python analyze_graph.py --antipatterns

    # Analyze critical edges
    python analyze_graph.py --edges

    # Export results to JSON
    python analyze_graph.py --full --output results.json

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List

try:
    from src.analysis import (
        GDSAnalyzer,
        CriticalityLevel,
        ProblemSeverity,
        PatternSeverity,
    )
    HAS_ANALYSIS = True
except ImportError:
    HAS_ANALYSIS = False


# =============================================================================
# Output Formatting
# =============================================================================

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
BLUE = "\033[94m"
GRAY = "\033[90m"
CYAN = "\033[96m"

SEVERITY_COLORS = {
    "critical": RED,
    "high": YELLOW,
    "medium": BLUE,
    "low": GREEN,
    "minimal": GRAY,
}


def severity_color(level: str) -> str:
    """Get ANSI color for a severity/criticality level"""
    return SEVERITY_COLORS.get(level.lower(), RESET)


def print_header(title: str) -> None:
    """Print a section header"""
    print(f"\n{BOLD}{CYAN}{'=' * 70}{RESET}")
    print(f"{BOLD}{CYAN}{title.center(70)}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 70}{RESET}\n")


def print_subheader(title: str) -> None:
    """Print a subsection header"""
    print(f"\n{BOLD}{title}{RESET}")
    print(f"{'-' * 50}")


def print_metric_box(stats: Dict[str, Any]) -> None:
    """Print box-plot statistics"""
    print(f"  Min: {stats.get('min', 0):.4f}  |  Q1: {stats.get('q1', 0):.4f}  |  "
          f"Median: {stats.get('median', 0):.4f}  |  Q3: {stats.get('q3', 0):.4f}  |  "
          f"Max: {stats.get('max', 0):.4f}")
    print(f"  IQR: {stats.get('iqr', 0):.4f}  |  Upper Fence: {stats.get('upper_fence', 0):.4f}  |  "
          f"Mean: {stats.get('mean', 0):.4f}  |  Std Dev: {stats.get('std_dev', 0):.4f}")


def print_classification_summary(summary: Dict[str, int]) -> None:
    """Print classification level summary"""
    levels = ["critical", "high", "medium", "low", "minimal"]
    parts = []
    for level in levels:
        count = summary.get(level, 0)
        if count > 0:
            color = severity_color(level)
            parts.append(f"{color}{level.upper()}: {count}{RESET}")
    print(f"  Distribution: {' | '.join(parts)}")


# =============================================================================
# Analysis Output
# =============================================================================

def print_component_type_results(results: Dict[str, Any], verbose: bool = False) -> None:
    """Print component type analysis results"""
    for comp_type, result in results.items():
        print_subheader(f"{comp_type} Analysis ({result['component_count']} components)")
        
        # Summary
        composite = result.get("composite_classification", {})
        if composite:
            print_classification_summary(composite.get("summary", {}))
        
        # Top critical components
        metrics = result.get("metrics", [])
        critical = [m for m in metrics if m.get("criticality_level") == "critical"]
        high = [m for m in metrics if m.get("criticality_level") == "high"]
        
        top_critical = sorted(
            critical + high,
            key=lambda x: x.get("composite_score", 0),
            reverse=True
        )[:5]
        
        if top_critical:
            print(f"\n  Top Critical {comp_type}s:")
            for i, comp in enumerate(top_critical, 1):
                level = comp.get("criticality_level", "unknown")
                color = severity_color(level)
                print(f"    {i}. {color}{comp['component_id']}{RESET}")
                print(f"       Composite: {comp.get('composite_score', 0):.4f}  |  "
                      f"Betweenness: {comp.get('betweenness', 0):.4f}  |  "
                      f"PageRank: {comp.get('pagerank', 0):.4f}")
                if comp.get("is_articulation_point"):
                    print(f"       {RED}⚠ Articulation Point (SPOF){RESET}")
        
        if verbose:
            # Box-plot stats
            stats = composite.get("statistics", {})
            if stats:
                print(f"\n  Box-Plot Statistics:")
                print_metric_box(stats)


def print_problem_results(result: Dict[str, Any], verbose: bool = False) -> None:
    """Print problem detection results"""
    problems = result.get("problems", [])
    summary = result.get("summary", {})
    
    print(f"  Total Problems: {summary.get('total', len(problems))}")
    print(f"  By Severity: {RED}CRITICAL: {summary.get('critical', 0)}{RESET} | "
          f"{YELLOW}HIGH: {summary.get('high', 0)}{RESET} | "
          f"{BLUE}MEDIUM: {summary.get('medium', 0)}{RESET} | "
          f"{GREEN}LOW: {summary.get('low', 0)}{RESET}")
    
    if not problems:
        print(f"\n  {GREEN}✓ No significant problems detected{RESET}")
        return
    
    # Group by quality attribute
    by_qa = {}
    for problem in problems:
        for qa in problem.get("quality_attributes", ["unknown"]):
            if qa not in by_qa:
                by_qa[qa] = []
            by_qa[qa].append(problem)
    
    for qa, qa_problems in by_qa.items():
        print(f"\n  {qa.upper()} Issues ({len(qa_problems)}):")
        
        # Sort by severity
        sorted_problems = sorted(
            qa_problems,
            key=lambda x: {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(
                x.get("severity", "low"), 0
            ),
            reverse=True
        )
        
        for problem in sorted_problems[:5 if not verbose else len(sorted_problems)]:
            severity = problem.get("severity", "unknown")
            color = severity_color(severity)
            print(f"    {color}[{severity.upper()}]{RESET} {problem.get('title', 'Unknown')}")
            
            if verbose:
                print(f"      Components: {', '.join(problem.get('affected_components', [])[:5])}")
                print(f"      Impact: {problem.get('impact', 'N/A')[:80]}")
                
                # Show symptoms
                symptoms = problem.get("symptoms", [])
                if symptoms:
                    print(f"      Symptoms:")
                    for symptom in symptoms[:3]:
                        print(f"        - {symptom.get('name')}: {symptom.get('value', 0):.2f} "
                              f"(threshold: {symptom.get('threshold', 0):.2f})")


def print_antipattern_results(result: Dict[str, Any], verbose: bool = False) -> None:
    """Print anti-pattern detection results"""
    patterns = result.get("patterns", [])
    summary = result.get("summary", {})
    
    print(f"  Total Anti-Patterns: {summary.get('total', len(patterns))}")
    
    if not patterns:
        print(f"\n  {GREEN}✓ No anti-patterns detected{RESET}")
        return
    
    # Sort by severity
    sorted_patterns = sorted(
        patterns,
        key=lambda x: {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(
            x.get("severity", "low"), 0
        ),
        reverse=True
    )
    
    for pattern in sorted_patterns[:10 if not verbose else len(sorted_patterns)]:
        severity = pattern.get("severity", "unknown")
        color = severity_color(severity)
        pattern_type = pattern.get("type", "unknown").replace("_", " ").title()
        
        print(f"\n    {color}[{severity.upper()}]{RESET} {pattern_type}")
        print(f"      Components: {', '.join(pattern.get('affected_components', [])[:5])}")
        
        if verbose:
            print(f"      Description: {pattern.get('description', 'N/A')[:100]}")
            print(f"      Impact: {pattern.get('impact', 'N/A')[:80]}")
            print(f"      Recommendation: {pattern.get('recommendation', 'N/A')[:80]}")


def print_edge_results(result: Dict[str, Any], verbose: bool = False) -> None:
    """Print edge analysis results"""
    edges = result.get("edges", [])
    summary = result.get("summary", {})
    
    print(f"  Total Edges: {summary.get('total', len(edges))}")
    print(f"  Critical: {summary.get('critical', 0)} | Bridges: {summary.get('bridges', 0)}")
    
    # Show critical edges
    critical_edges = [e for e in edges if e.get("level") == "critical"]
    
    if critical_edges:
        print(f"\n  Critical Edges:")
        for edge in critical_edges[:10]:
            bridge_mark = f"{RED}[BRIDGE]{RESET}" if edge.get("is_bridge") else ""
            print(f"    {edge.get('source_id')} → {edge.get('target_id')} "
                  f"(score: {edge.get('criticality_score', 0):.4f}) {bridge_mark}")


def print_full_results(result: Dict[str, Any], verbose: bool = False) -> None:
    """Print complete analysis results"""
    timestamp = result.get("timestamp", datetime.now().isoformat())
    summary = result.get("summary", {})
    
    print_header("GRAPH ANALYSIS RESULTS")
    print(f"  Timestamp: {timestamp}")
    print(f"  Total Nodes: {summary.get('graph', {}).get('total_nodes', 0)}")
    print(f"  Total DEPENDS_ON: {summary.get('graph', {}).get('total_depends_on', 0)}")
    
    # Component type analysis
    by_component_type = result.get("by_component_type", {})
    if by_component_type:
        print_header("COMPONENT TYPE ANALYSIS")
        print_component_type_results(by_component_type, verbose)
    
    # Problems
    problems = result.get("problems")
    if problems:
        print_header("PROBLEM DETECTION")
        print_problem_results(problems, verbose)
    
    # Anti-patterns
    antipatterns = result.get("antipatterns")
    if antipatterns:
        print_header("ANTI-PATTERN DETECTION")
        print_antipattern_results(antipatterns, verbose)
    
    # Edges
    edges = result.get("edges")
    if edges:
        print_header("CRITICAL EDGE ANALYSIS")
        print_edge_results(edges, verbose)
    
    # Summary
    print_header("SUMMARY")
    
    if "critical_by_type" in summary:
        print("  Critical Components by Type:")
        for comp_type, count in summary["critical_by_type"].items():
            if count > 0:
                print(f"    {RED}• {comp_type}: {count}{RESET}")
    
    if "problems" in summary:
        prob = summary["problems"]
        if prob.get("critical", 0) > 0:
            print(f"\n  {RED}⚠ {prob['critical']} CRITICAL problems require immediate attention{RESET}")
    
    if "antipatterns" in summary:
        ap = summary["antipatterns"]
        if ap.get("critical", 0) > 0:
            print(f"  {RED}⚠ {ap['critical']} CRITICAL anti-patterns detected{RESET}")


# =============================================================================
# Main Entry Point
# =============================================================================

def run_analysis(args) -> Optional[Dict[str, Any]]:
    """Run the requested analysis"""
    if not HAS_ANALYSIS:
        print(f"{RED}Error: Analysis module not available.{RESET}")
        print("Make sure you're running from the project root and all dependencies are installed.")
        return None
    
    try:
        with GDSAnalyzer(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
            k_factor=args.k_factor,
        ) as analyzer:
            
            # Full analysis
            if args.full:
                print(f"{CYAN}Running full analysis...{RESET}")
                result = analyzer.analyze_all(
                    include_component_types=True,
                    include_layers=True,
                    include_problems=True,
                    include_antipatterns=True,
                    include_edges=True,
                    weighted=not args.unweighted,
                )
                return result.to_dict()
            
            # Component type analysis
            if args.component_type:
                print(f"{CYAN}Analyzing {args.component_type} components...{RESET}")
                result = analyzer.analyze_component_type(
                    args.component_type,
                    weighted=not args.unweighted,
                )
                return {args.component_type: result.to_dict()}
            
            if args.all_types:
                print(f"{CYAN}Analyzing all component types...{RESET}")
                results = analyzer.analyze_all_component_types(
                    weighted=not args.unweighted
                )
                return {ct: r.to_dict() for ct, r in results.items()}
            
            # Problem detection
            if args.problems:
                print(f"{CYAN}Detecting problems...{RESET}")
                result = analyzer.detect_problems()
                return {"problems": result.to_dict()}
            
            # Anti-pattern detection
            if args.antipatterns:
                print(f"{CYAN}Detecting anti-patterns...{RESET}")
                result = analyzer.detect_antipatterns()
                return {"antipatterns": result.to_dict()}
            
            # Edge analysis
            if args.edges:
                print(f"{CYAN}Analyzing edges...{RESET}")
                result = analyzer.analyze_edges()
                return {"edges": result.to_dict()}
            
            # Default: full analysis
            print(f"{CYAN}Running full analysis (default)...{RESET}")
            result = analyzer.analyze_all(
                include_component_types=True,
                include_layers=True,
                include_problems=True,
                include_antipatterns=True,
                include_edges=True,
                weighted=not args.unweighted,
            )
            return result.to_dict()
            
    except Exception as e:
        print(f"{RED}Error during analysis: {e}{RESET}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Graph-based analysis of distributed pub-sub systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis with all features
  python analyze_graph.py --full

  # Analyze Application components only
  python analyze_graph.py --component-type Application

  # Analyze all component types separately
  python analyze_graph.py --all-types

  # Detect problems only
  python analyze_graph.py --problems

  # Detect anti-patterns
  python analyze_graph.py --antipatterns

  # Export results to JSON
  python analyze_graph.py --full --output results.json

  # Verbose output with details
  python analyze_graph.py --full --verbose
        """
    )
    
    # Neo4j connection
    conn_group = parser.add_argument_group("Neo4j Connection")
    conn_group.add_argument(
        "--uri", "-u",
        default="bolt://localhost:7687",
        help="Neo4j bolt URI (default: bolt://localhost:7687)"
    )
    conn_group.add_argument(
        "--user", "-U",
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    conn_group.add_argument(
        "--password", "-p",
        default="password",
        help="Neo4j password (default: password)"
    )
    conn_group.add_argument(
        "--database", "-d",
        default="neo4j",
        help="Neo4j database name (default: neo4j)"
    )
    
    # Analysis options
    analysis_group = parser.add_argument_group("Analysis Options")
    analysis_group.add_argument(
        "--full", "-f",
        action="store_true",
        help="Run full analysis (component types, problems, anti-patterns, edges)"
    )
    analysis_group.add_argument(
        "--component-type", "-c",
        choices=["Application", "Broker", "Node", "Topic"],
        help="Analyze specific component type"
    )
    analysis_group.add_argument(
        "--all-types", "-a",
        action="store_true",
        help="Analyze all component types separately"
    )
    analysis_group.add_argument(
        "--problems",
        action="store_true",
        help="Run problem detection (R/M/A issues)"
    )
    analysis_group.add_argument(
        "--antipatterns",
        action="store_true",
        help="Run anti-pattern detection"
    )
    analysis_group.add_argument(
        "--edges",
        action="store_true",
        help="Run critical edge analysis"
    )
    
    # Classification options
    class_group = parser.add_argument_group("Classification Options")
    class_group.add_argument(
        "--k-factor", "-k",
        type=float,
        default=1.5,
        help="Box-plot k-factor for outlier detection (default: 1.5)"
    )
    class_group.add_argument(
        "--unweighted",
        action="store_true",
        help="Use unweighted algorithms (ignore edge weights)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        help="Output file path (JSON format)"
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with detailed information"
    )
    output_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output (only errors)"
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Run analysis
    result = run_analysis(args)
    
    if result is None:
        sys.exit(1)
    
    # Output results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n{GREEN}✓ Results saved to: {args.output}{RESET}")
    
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    elif not args.quiet:
        # Pretty print results
        if args.full or not any([args.component_type, args.all_types, 
                                  args.problems, args.antipatterns, args.edges]):
            print_full_results(result, args.verbose)
        elif args.component_type or args.all_types:
            print_header(f"COMPONENT TYPE ANALYSIS")
            print_component_type_results(result, args.verbose)
        elif args.problems:
            print_header("PROBLEM DETECTION")
            print_problem_results(result.get("problems", {}), args.verbose)
        elif args.antipatterns:
            print_header("ANTI-PATTERN DETECTION")
            print_antipattern_results(result.get("antipatterns", {}), args.verbose)
        elif args.edges:
            print_header("CRITICAL EDGE ANALYSIS")
            print_edge_results(result.get("edges", {}), args.verbose)
    
    print(f"\n{GREEN}✓ Analysis complete{RESET}")


if __name__ == "__main__":
    main()