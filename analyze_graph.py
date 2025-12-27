#!/usr/bin/env python3
"""
Graph Analyzer CLI - Version 4.0

Analyzes pub-sub system graphs using Neo4j GDS for:
- Centrality analysis (PageRank, Betweenness, Degree)
- Quality attributes (Reliability, Maintainability, Availability)
- Box-plot statistical classification
- Anti-pattern detection

Usage:
    # Full analysis with all quality attributes
    python analyze_graph.py --uri bolt://localhost:7687
    
    # Specific quality attributes
    python analyze_graph.py --reliability --maintainability
    
    # Classification with custom k-factor
    python analyze_graph.py --classify --k-factor 1.5
    
    # Anti-pattern detection
    python analyze_graph.py --antipatterns
    
    # Export results
    python analyze_graph.py --output results/ --format json

Requirements:
    - Neo4j with GDS plugin
    - DEPENDS_ON relationships in database
    - pip install neo4j

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis import (
    GDSClient,
    GDSClassifier,
    ReliabilityAnalyzer,
    MaintainabilityAnalyzer,
    AvailabilityAnalyzer,
    AntiPatternDetector,
    CriticalityLevel,
    Severity,
    QualityAttribute,
)


# =============================================================================
# Terminal Colors
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    @classmethod
    def disable(cls):
        for attr in ['HEADER', 'BLUE', 'CYAN', 'GREEN', 'YELLOW', 'RED', 'END', 'BOLD', 'DIM']:
            setattr(cls, attr, '')


def use_colors() -> bool:
    import os
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and not os.getenv('NO_COLOR')


# =============================================================================
# Output Helpers
# =============================================================================

def print_header(text: str) -> None:
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}")


def print_section(title: str) -> None:
    print(f"\n{Colors.CYAN}{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.DIM}{'-'*50}{Colors.END}")


def print_subsection(title: str) -> None:
    print(f"\n  {Colors.BLUE}{title}{Colors.END}")


def print_kv(key: str, value, indent: int = 2) -> None:
    print(f"{' '*indent}{Colors.DIM}{key}:{Colors.END} {value}")


def print_success(text: str) -> None:
    print(f"{Colors.GREEN}✓{Colors.END} {text}")


def print_error(text: str) -> None:
    print(f"{Colors.RED}✗{Colors.END} {text}", file=sys.stderr)


def print_warning(text: str) -> None:
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")


def severity_color(severity) -> str:
    """Get color for severity"""
    if hasattr(severity, 'value'):
        severity = severity.value
    colors = {
        "critical": Colors.RED,
        "high": Colors.YELLOW,
        "medium": Colors.BLUE,
        "low": Colors.GREEN,
        "info": Colors.DIM,
    }
    return colors.get(severity, Colors.END)


def score_color(score: float) -> str:
    """Get color for score"""
    if score >= 80:
        return Colors.GREEN
    elif score >= 60:
        return Colors.YELLOW
    elif score >= 40:
        return Colors.YELLOW
    return Colors.RED


def level_color(level) -> str:
    """Get color for criticality level"""
    if hasattr(level, 'value'):
        level = level.value
    colors = {
        "critical": Colors.RED,
        "high": Colors.YELLOW,
        "medium": Colors.BLUE,
        "low": Colors.GREEN,
        "minimal": Colors.DIM,
    }
    return colors.get(level, Colors.END)


# =============================================================================
# Result Display
# =============================================================================

def print_graph_stats(stats: Dict) -> None:
    """Print graph statistics"""
    print_section("Graph Statistics")
    
    nodes = stats.get("nodes", {})
    print_subsection("Nodes")
    for label, count in sorted(nodes.items()):
        print_kv(label, count, indent=4)
    
    deps = stats.get("dependencies", {})
    if deps:
        print_subsection("DEPENDS_ON Relationships")
        for dep_type, info in sorted(deps.items()):
            if isinstance(info, dict):
                print_kv(dep_type, f"{info['count']} (avg weight: {info.get('avg_weight', 0):.2f})", indent=4)
            else:
                print_kv(dep_type, info, indent=4)


def print_classification_result(result, verbose: bool = False) -> None:
    """Print classification result"""
    print_subsection(f"Classification: {result.metric_name}")
    
    # Summary
    print(f"    Total items: {len(result.items)}")
    for level in CriticalityLevel:
        count = result.summary.get(level, 0)
        if count > 0:
            color = level_color(level)
            print(f"    {color}{level.value.upper()}: {count}{Colors.END}")
    
    # Statistics
    stats = result.stats
    print(f"    Thresholds: Q1={stats.q1:.4f}, Median={stats.median:.4f}, Q3={stats.q3:.4f}")
    print(f"    Upper fence: {stats.upper_fence:.4f} (k={stats.k_factor})")
    
    # Top items
    if verbose and result.items:
        print(f"\n    Top Critical/High Items:")
        shown = 0
        for item in result.items:
            if item.level in (CriticalityLevel.CRITICAL, CriticalityLevel.HIGH) and shown < 10:
                color = level_color(item.level)
                print(f"      {color}{item.level.value:8}{Colors.END} {item.id:20} ({item.item_type}) = {item.score:.4f}")
                shown += 1


def print_analysis_result(result, verbose: bool = False) -> None:
    """Print quality attribute analysis result"""
    attr = result.quality_attribute.value.upper()
    score = result.score
    color = score_color(score)
    
    print_subsection(f"{attr}: {color}{score:.1f}/100{Colors.END}")
    
    # Summary
    summary = result.summary()
    by_sev = summary.get("by_severity", {})
    if by_sev:
        parts = []
        for sev in ["critical", "high", "medium", "low"]:
            if by_sev.get(sev, 0) > 0:
                parts.append(f"{sev}: {by_sev[sev]}")
        if parts:
            print(f"    Findings: {', '.join(parts)}")
    
    # Critical components
    if result.critical_components:
        print(f"    Critical components: {len(result.critical_components)}")
    
    # Top findings
    if verbose:
        critical_findings = [f for f in result.findings if f.severity == Severity.CRITICAL]
        high_findings = [f for f in result.findings if f.severity == Severity.HIGH]
        
        if critical_findings:
            print(f"\n    {Colors.RED}Critical Issues:{Colors.END}")
            for f in critical_findings[:5]:
                print(f"      • {f.component_id}: {f.description}")
        
        if high_findings:
            print(f"\n    {Colors.YELLOW}High Issues:{Colors.END}")
            for f in high_findings[:5]:
                print(f"      • {f.component_id}: {f.description}")
    
    # Recommendations
    if result.recommendations:
        print(f"\n    Recommendations:")
        for rec in result.recommendations[:3]:
            print(f"      → {rec}")


def print_antipattern_result(result, verbose: bool = False) -> None:
    """Print anti-pattern detection result"""
    print_section("Anti-Pattern Detection")
    
    summary = result.summary
    print(f"  Total patterns: {summary['total']}")
    
    by_sev = summary.get("by_severity", {})
    if by_sev:
        parts = []
        for sev in ["critical", "high", "medium", "low"]:
            if by_sev.get(sev, 0) > 0:
                color = severity_color(sev)
                parts.append(f"{color}{sev}: {by_sev[sev]}{Colors.END}")
        if parts:
            print(f"  By severity: {', '.join(parts)}")
    
    by_type = summary.get("by_type", {})
    if by_type:
        print(f"\n  By type:")
        for ptype, count in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"    {ptype}: {count}")
    
    # Show critical/high patterns
    if verbose:
        critical = result.by_severity(result.patterns[0].severity.__class__("critical")) if result.patterns else []
        high = result.by_severity(result.patterns[0].severity.__class__("high")) if result.patterns else []
        
        if critical:
            print(f"\n  {Colors.RED}Critical Patterns:{Colors.END}")
            for p in critical[:5]:
                print(f"    • [{p.pattern_type.value}] {p.description[:60]}...")
        
        if high:
            print(f"\n  {Colors.YELLOW}High Patterns:{Colors.END}")
            for p in high[:5]:
                print(f"    • [{p.pattern_type.value}] {p.description[:60]}...")
    
    # Recommendations
    if result.recommendations:
        print(f"\n  Recommendations:")
        for rec in result.recommendations:
            print(f"    → {rec}")


def print_overall_summary(results: Dict) -> None:
    """Print overall analysis summary"""
    print_section("Overall Summary")
    
    scores = []
    if "reliability" in results and results["reliability"]:
        scores.append(("Reliability", results["reliability"].score))
    if "maintainability" in results and results["maintainability"]:
        scores.append(("Maintainability", results["maintainability"].score))
    if "availability" in results and results["availability"]:
        scores.append(("Availability", results["availability"].score))
    
    if scores:
        overall = sum(s for _, s in scores) / len(scores)
        color = score_color(overall)
        print(f"\n  {Colors.BOLD}Overall Score: {color}{overall:.1f}/100{Colors.END}")
        
        for name, score in scores:
            color = score_color(score)
            print(f"    {name:20} {color}{score:5.1f}{Colors.END}")
    
    # Count critical issues
    total_critical = 0
    total_high = 0
    for key in ["reliability", "maintainability", "availability"]:
        if key in results and results[key]:
            total_critical += len([f for f in results[key].findings if f.severity == Severity.CRITICAL])
            total_high += len([f for f in results[key].findings if f.severity == Severity.HIGH])
    
    if total_critical > 0 or total_high > 0:
        print(f"\n  {Colors.RED}⚠ Action Required:{Colors.END}")
        if total_critical > 0:
            print(f"    {Colors.RED}{total_critical} CRITICAL issues{Colors.END}")
        if total_high > 0:
            print(f"    {Colors.YELLOW}{total_high} HIGH issues{Colors.END}")


# =============================================================================
# Export
# =============================================================================

def export_results(results: Dict, output_dir: Path, formats: List[str]) -> None:
    """Export results to files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Prepare export data
    export_data = {
        "timestamp": timestamp,
        "graph_stats": results.get("graph_stats", {}),
    }
    
    # Add analysis results
    for key in ["reliability", "maintainability", "availability"]:
        if key in results and results[key]:
            export_data[key] = results[key].to_dict()
    
    # Add classification
    if "classification" in results:
        export_data["classification"] = {
            metric: cr.to_dict() for metric, cr in results["classification"].items()
        }
    
    # Add anti-patterns
    if "antipatterns" in results and results["antipatterns"]:
        export_data["antipatterns"] = results["antipatterns"].to_dict()
    
    if "json" in formats:
        json_file = output_dir / f"analysis_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(export_data, f, indent=2, default=str)
        print_success(f"JSON exported: {json_file}")
    
    if "summary" in formats:
        summary_file = output_dir / f"summary_{timestamp}.txt"
        with open(summary_file, "w") as f:
            f.write(f"Graph Analysis Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            for key in ["reliability", "maintainability", "availability"]:
                if key in results and results[key]:
                    r = results[key]
                    f.write(f"{key.capitalize()}: {r.score:.1f}/100\n")
                    f.write(f"  Findings: {len(r.findings)}\n")
                    f.write(f"  Critical components: {len(r.critical_components)}\n\n")
            
            if "antipatterns" in results and results["antipatterns"]:
                ap = results["antipatterns"]
                f.write(f"Anti-patterns: {ap.summary['total']}\n")
        
        print_success(f"Summary exported: {summary_file}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze pub-sub system graphs using Neo4j GDS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_graph.py --uri bolt://localhost:7687
    python analyze_graph.py --reliability --maintainability --verbose
    python analyze_graph.py --classify --k-factor 1.5
    python analyze_graph.py --antipatterns --output results/
        """,
    )
    
    # Neo4j connection
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    parser.add_argument("--database", default="neo4j", help="Database name")
    
    # Analysis selection
    parser.add_argument("--all", action="store_true", help="Run all analyses (default)")
    parser.add_argument("--reliability", "-r", action="store_true", help="Analyze reliability")
    parser.add_argument("--maintainability", "-m", action="store_true", help="Analyze maintainability")
    parser.add_argument("--availability", "-a", action="store_true", help="Analyze availability")
    parser.add_argument("--classify", "-c", action="store_true", help="Run box-plot classification")
    parser.add_argument("--antipatterns", "-p", action="store_true", help="Detect anti-patterns")
    
    # Classification options
    parser.add_argument("--k-factor", type=float, default=1.5, help="Box-plot k-factor (default: 1.5)")
    parser.add_argument("--weighted", action="store_true", default=True, help="Use weighted algorithms")
    parser.add_argument("--no-weighted", dest="weighted", action="store_false", help="Use unweighted algorithms")
    
    # Dependency types
    parser.add_argument(
        "--dep-types",
        nargs="+",
        default=["app_to_app", "node_to_node"],
        choices=["app_to_app", "node_to_node", "app_to_broker", "node_to_broker"],
        help="Dependency types to analyze",
    )
    
    # Output options
    parser.add_argument("--output", "-o", type=Path, help="Output directory for results")
    parser.add_argument("--format", nargs="+", default=["json", "summary"], help="Output formats")
    parser.add_argument("--json", action="store_true", help="Output as JSON to stdout")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    parser.add_argument("--no-color", action="store_true", help="Disable colors")
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Handle colors
    if args.no_color or not use_colors():
        Colors.disable()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    
    # Determine what to run
    run_all = args.all or not any([args.reliability, args.maintainability, args.availability, args.classify, args.antipatterns])
    run_reliability = args.reliability or run_all
    run_maintainability = args.maintainability or run_all
    run_availability = args.availability or run_all
    run_classify = args.classify or run_all
    run_antipatterns = args.antipatterns or run_all
    
    results = {}
    
    try:
        if not args.quiet:
            print_header("Graph Analysis (GDS Edition)")
        
        # Connect to Neo4j
        with GDSClient(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
        ) as gds:
            
            if not args.quiet:
                print_section("Connection")
                print_kv("URI", args.uri)
                print_kv("Database", args.database)
            
            # Get graph stats
            stats = gds.get_graph_stats()
            results["graph_stats"] = stats
            
            if not args.quiet and not args.json:
                print_graph_stats(stats)
            
            # Create projection
            projection_name = "analysis_projection"
            if not args.quiet:
                print_section("Creating Graph Projection")
                print_kv("Dependency types", ", ".join(args.dep_types))
                print_kv("Weighted", args.weighted)
            
            projection = gds.create_projection(
                projection_name,
                dependency_types=args.dep_types,
                include_weights=args.weighted,
            )
            
            if not args.quiet:
                print_kv("Nodes", projection.node_count)
                print_kv("Relationships", projection.relationship_count)
            
            # Run classification
            if run_classify:
                if not args.quiet:
                    print_section("Box-Plot Classification")
                    print_kv("K-factor", args.k_factor)
                
                classifier = GDSClassifier(gds, k_factor=args.k_factor)
                
                results["classification"] = {}
                
                # Betweenness
                bc_result = classifier.classify_by_betweenness(projection_name, weighted=args.weighted)
                results["classification"]["betweenness"] = bc_result
                if not args.quiet and not args.json:
                    print_classification_result(bc_result, verbose=args.verbose)
                
                # PageRank
                pr_result = classifier.classify_by_pagerank(projection_name, weighted=args.weighted)
                results["classification"]["pagerank"] = pr_result
                if not args.quiet and not args.json:
                    print_classification_result(pr_result, verbose=args.verbose)
                
                # Composite
                comp_result = classifier.classify_by_composite(projection_name, weighted=args.weighted)
                results["classification"]["composite"] = comp_result
                if not args.quiet and not args.json:
                    print_classification_result(comp_result, verbose=args.verbose)
            
            # Run quality attribute analyses
            if run_reliability:
                if not args.quiet:
                    print_section("Reliability Analysis")
                
                analyzer = ReliabilityAnalyzer(gds, {"use_weights": args.weighted})
                result = analyzer.analyze(projection_name)
                results["reliability"] = result
                
                if not args.quiet and not args.json:
                    print_analysis_result(result, verbose=args.verbose)
            
            if run_maintainability:
                if not args.quiet:
                    print_section("Maintainability Analysis")
                
                analyzer = MaintainabilityAnalyzer(gds, {"use_weights": args.weighted})
                result = analyzer.analyze(projection_name)
                results["maintainability"] = result
                
                if not args.quiet and not args.json:
                    print_analysis_result(result, verbose=args.verbose)
            
            if run_availability:
                if not args.quiet:
                    print_section("Availability Analysis")
                
                analyzer = AvailabilityAnalyzer(gds, {"use_weights": args.weighted})
                result = analyzer.analyze(projection_name)
                results["availability"] = result
                
                if not args.quiet and not args.json:
                    print_analysis_result(result, verbose=args.verbose)
            
            # Run anti-pattern detection
            if run_antipatterns:
                detector = AntiPatternDetector(gds)
                ap_result = detector.detect_all()
                results["antipatterns"] = ap_result
                
                if not args.quiet and not args.json:
                    print_antipattern_result(ap_result, verbose=args.verbose)
            
            # Cleanup projection
            gds.drop_projection(projection_name)
            
            # Overall summary
            if not args.quiet and not args.json:
                print_overall_summary(results)
            
            # JSON output
            if args.json:
                output = {
                    "timestamp": datetime.now().isoformat(),
                    "graph_stats": stats,
                }
                for key in ["reliability", "maintainability", "availability"]:
                    if key in results and results[key]:
                        output[key] = results[key].to_dict()
                if "classification" in results:
                    output["classification"] = {m: r.to_dict() for m, r in results["classification"].items()}
                if "antipatterns" in results:
                    output["antipatterns"] = results["antipatterns"].to_dict()
                
                print(json.dumps(output, indent=2, default=str))
            
            # Export to files
            if args.output:
                export_results(results, args.output, args.format)
        
        return 0
        
    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        print("Install neo4j driver: pip install neo4j")
        return 1
    
    except KeyboardInterrupt:
        print_warning("\nAnalysis interrupted")
        return 130
    
    except Exception as e:
        print_error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())