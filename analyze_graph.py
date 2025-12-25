#!/usr/bin/env python3
"""
Graph Analyzer CLI (GDS Edition)
================================

Simplified graph analysis using Neo4j Graph Data Science (GDS).
Analyzes DEPENDS_ON relationships for:
- Reliability: SPOFs, cascade risks, redundancy
- Maintainability: Coupling, cycles, modularity  
- Availability: Connectivity, fault tolerance, recovery

Usage:
    # Basic analysis (all quality attributes)
    python analyze_graph.py --uri bolt://localhost:7687
    
    # Specific quality attributes
    python analyze_graph.py --reliability --maintainability
    
    # Export results
    python analyze_graph.py --output results/ --format json
    
    # Verbose output
    python analyze_graph.py --verbose

Requirements:
    - Neo4j with GDS plugin installed
    - DEPENDS_ON relationships in the database
    - pip install neo4j

Author: Software-as-a-Graph Research Project
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# Add src to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from src.analysis.gds_client import GDSClient
from src.analysis.gds_analyzers import (
    ReliabilityAnalyzer,
    MaintainabilityAnalyzer,
    AvailabilityAnalyzer,
    AnalysisResult,
    QualityAttribute,
    Severity
)


# ============================================================================
# Terminal Colors
# ============================================================================

class Colors:
    """ANSI color codes"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    @classmethod
    def disable(cls):
        for attr in ['HEADER', 'BLUE', 'CYAN', 'GREEN', 'YELLOW', 'RED', 'ENDC', 'BOLD', 'DIM']:
            setattr(cls, attr, '')


def print_header(text: str):
    """Print formatted header"""
    width = 70
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*width}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^{width}}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*width}{Colors.ENDC}")


def print_section(text: str):
    """Print section header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.DIM}{'-'*50}{Colors.ENDC}")


def print_success(text: str):
    print(f"{Colors.GREEN}✓{Colors.ENDC} {text}")


def print_error(text: str):
    print(f"{Colors.RED}✗{Colors.ENDC} {text}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠{Colors.ENDC} {text}")


def severity_color(severity: Severity) -> str:
    """Get color for severity"""
    colors = {
        Severity.CRITICAL: Colors.RED,
        Severity.HIGH: Colors.YELLOW,
        Severity.MEDIUM: Colors.CYAN,
        Severity.LOW: Colors.DIM
    }
    return colors.get(severity, '')


def score_color(score: float) -> str:
    """Get color for score"""
    if score >= 80:
        return Colors.GREEN
    elif score >= 60:
        return Colors.YELLOW
    else:
        return Colors.RED


# ============================================================================
# Combined Analysis Result
# ============================================================================

@dataclass
class ComprehensiveResult:
    """Combined results from all analyses"""
    reliability: Optional[AnalysisResult] = None
    maintainability: Optional[AnalysisResult] = None
    availability: Optional[AnalysisResult] = None
    graph_stats: Dict[str, Any] = field(default_factory=dict)
    overall_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'overall_score': round(self.overall_score, 2),
            'timestamp': self.timestamp,
            'graph_stats': self.graph_stats
        }
        if self.reliability:
            result['reliability'] = self.reliability.to_dict()
        if self.maintainability:
            result['maintainability'] = self.maintainability.to_dict()
        if self.availability:
            result['availability'] = self.availability.to_dict()
        return result


# ============================================================================
# Analysis Orchestrator
# ============================================================================

class GDSAnalysisOrchestrator:
    """Orchestrates GDS-based quality analysis"""
    
    def __init__(self, gds_client: GDSClient, config: Optional[Dict[str, Any]] = None):
        self.gds = gds_client
        self.config = config or {}
        self.logger = logging.getLogger('GDSOrchestrator')
    
    def analyze(self,
                analyze_reliability: bool = True,
                analyze_maintainability: bool = True,
                analyze_availability: bool = True,
                dependency_types: Optional[List[str]] = None,
                use_weights: bool = True,
                weight_property: str = 'weight') -> ComprehensiveResult:
        """
        Run comprehensive analysis using GDS.
        
        Args:
            analyze_reliability: Include reliability analysis
            analyze_maintainability: Include maintainability analysis
            analyze_availability: Include availability analysis
            dependency_types: DEPENDS_ON types to include (default: all)
            use_weights: Use weighted algorithms (default: True)
            weight_property: Name of the weight property (default: 'weight')
            
        Returns:
            ComprehensiveResult with all findings
        """
        self.logger.info(f"Starting GDS-based analysis (weighted={use_weights})...")
        
        result = ComprehensiveResult()
        
        # Get graph statistics
        self.logger.info("Gathering graph statistics...")
        result.graph_stats = self.gds.get_graph_statistics()
        
        # Log weight information
        weight_stats = result.graph_stats.get('weight_statistics', {})
        if weight_stats:
            self.logger.info(f"Weight stats: avg={weight_stats.get('avg', 'N/A')}, "
                           f"range=[{weight_stats.get('min', 'N/A')}, {weight_stats.get('max', 'N/A')}]")
        
        # Create projection
        projection_name = 'analysis_projection'
        self.logger.info("Creating GDS projection...")
        projection = self.gds.create_depends_on_projection(
            projection_name=projection_name,
            dependency_types=dependency_types,
            include_weights=use_weights,
            weight_property=weight_property
        )
        
        self.logger.info(f"Projection created: {projection.node_count} nodes, "
                        f"{projection.relationship_count} relationships")
        
        # Analyzer config with weighted setting
        analyzer_config = {'use_weights': use_weights}
        
        try:
            # Run analyses
            scores = []
            
            if analyze_reliability:
                self.logger.info("Running reliability analysis...")
                analyzer = ReliabilityAnalyzer(self.gds, config=analyzer_config)
                result.reliability = analyzer.analyze(projection_name)
                scores.append(result.reliability.score * 1.2)  # Weight higher
            
            if analyze_maintainability:
                self.logger.info("Running maintainability analysis...")
                analyzer = MaintainabilityAnalyzer(self.gds, config=analyzer_config)
                result.maintainability = analyzer.analyze(projection_name)
                scores.append(result.maintainability.score)
            
            if analyze_availability:
                self.logger.info("Running availability analysis...")
                analyzer = AvailabilityAnalyzer(self.gds, config=analyzer_config)
                result.availability = analyzer.analyze(projection_name)
                scores.append(result.availability.score * 1.1)  # Weight higher
            
            # Calculate overall score
            if scores:
                result.overall_score = sum(scores) / len(scores)
                result.overall_score = max(0, min(100, result.overall_score))
            
        finally:
            # Cleanup projection
            self.gds.cleanup_projections()
        
        self.logger.info(f"Analysis complete. Overall score: {result.overall_score:.1f}/100")
        return result


# ============================================================================
# Output Functions
# ============================================================================

def print_graph_summary(stats: Dict[str, Any]):
    """Print graph statistics"""
    print_section("GRAPH SUMMARY")
    
    print(f"  Total Nodes:  {stats.get('total_nodes', 0)}")
    print(f"  Total Edges:  {stats.get('total_edges', 0)}")
    print(f"  Density:      {stats.get('density', 0):.4f}")
    
    print("\n  Nodes by Type:")
    for node_type, count in stats.get('node_counts', {}).items():
        print(f"    {node_type}: {count}")
    
    print("\n  DEPENDS_ON by Type:")
    for dep_type, count in stats.get('edge_counts', {}).items():
        print(f"    {dep_type}: {count}")
    
    # Weight statistics
    weight_stats = stats.get('weight_statistics', {})
    if weight_stats:
        print(f"\n  {Colors.CYAN}Weight Statistics:{Colors.ENDC}")
        print(f"    Edges with weight: {weight_stats.get('count', 0)}")
        print(f"    Average weight:    {weight_stats.get('avg', 0):.4f}")
        print(f"    Weight range:      [{weight_stats.get('min', 0):.2f}, {weight_stats.get('max', 0):.2f}]")
        print(f"    Median:            {weight_stats.get('median', 0):.4f}")
        print(f"    90th percentile:   {weight_stats.get('p90', 0):.4f}")
    
    # Weight by dependency type
    weight_by_type = stats.get('weight_by_dependency_type', {})
    if weight_by_type:
        print("\n  Weight by Dependency Type:")
        for dep_type, info in weight_by_type.items():
            print(f"    {dep_type}: avg={info.get('avg_weight', 0):.2f}, max={info.get('max_weight', 0):.2f}")


def print_analysis_result(result: AnalysisResult, verbose: bool = False):
    """Print single analysis result"""
    attr = result.quality_attribute.value.upper()
    color = score_color(result.score)
    
    print_section(f"{attr} ANALYSIS")
    print(f"  Score: {color}{Colors.BOLD}{result.score:.1f}/100{Colors.ENDC}")
    
    # Findings summary
    by_severity = {}
    for finding in result.findings:
        by_severity[finding.severity] = by_severity.get(finding.severity, 0) + 1
    
    print("\n  Findings:")
    for severity in Severity:
        count = by_severity.get(severity, 0)
        if count > 0:
            scolor = severity_color(severity)
            print(f"    {scolor}{severity.value.upper()}{Colors.ENDC}: {count}")
    
    # Top critical components
    if result.critical_components:
        print("\n  Critical Components:")
        for comp in result.critical_components[:5]:
            print(f"    • {comp.component_id} ({comp.component_type}): "
                  f"{comp.criticality_score:.3f}")
    
    # Show findings detail if verbose
    if verbose and result.findings:
        print("\n  Detailed Findings:")
        for finding in result.findings[:10]:
            scolor = severity_color(finding.severity)
            print(f"\n    {scolor}[{finding.severity.value.upper()}]{Colors.ENDC} "
                  f"{finding.component_id}")
            print(f"      Category: {finding.category}")
            print(f"      {finding.description}")
            print(f"      Impact: {finding.impact}")
            print(f"      Fix: {finding.recommendation}")
    
    # Recommendations
    if result.recommendations:
        print("\n  Recommendations:")
        for rec in result.recommendations:
            print(f"    {rec}")


def print_overall_summary(result: ComprehensiveResult):
    """Print overall summary"""
    print_section("OVERALL HEALTH")
    
    color = score_color(result.overall_score)
    print(f"\n  {Colors.BOLD}Overall Score: {color}{result.overall_score:.1f}/100{Colors.ENDC}")
    
    # Individual scores
    print("\n  Quality Attribute Scores:")
    if result.reliability:
        rc = score_color(result.reliability.score)
        print(f"    Reliability:     {rc}{result.reliability.score:.1f}{Colors.ENDC}")
    if result.maintainability:
        mc = score_color(result.maintainability.score)
        print(f"    Maintainability: {mc}{result.maintainability.score:.1f}{Colors.ENDC}")
    if result.availability:
        ac = score_color(result.availability.score)
        print(f"    Availability:    {ac}{result.availability.score:.1f}{Colors.ENDC}")
    
    # Count all critical findings
    total_critical = 0
    total_high = 0
    if result.reliability:
        total_critical += len([f for f in result.reliability.findings if f.severity == Severity.CRITICAL])
        total_high += len([f for f in result.reliability.findings if f.severity == Severity.HIGH])
    if result.maintainability:
        total_critical += len([f for f in result.maintainability.findings if f.severity == Severity.CRITICAL])
        total_high += len([f for f in result.maintainability.findings if f.severity == Severity.HIGH])
    if result.availability:
        total_critical += len([f for f in result.availability.findings if f.severity == Severity.CRITICAL])
        total_high += len([f for f in result.availability.findings if f.severity == Severity.HIGH])
    
    if total_critical > 0 or total_high > 0:
        print(f"\n  {Colors.RED}⚠ Action Required:{Colors.ENDC}")
        if total_critical > 0:
            print(f"    {Colors.RED}{total_critical} CRITICAL issues{Colors.ENDC}")
        if total_high > 0:
            print(f"    {Colors.YELLOW}{total_high} HIGH issues{Colors.ENDC}")


def export_results(result: ComprehensiveResult, output_dir: str, formats: List[str]):
    """Export results to files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if 'json' in formats:
        json_file = output_path / f"analysis_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print_success(f"JSON exported: {json_file}")
    
    if 'summary' in formats:
        summary_file = output_path / f"summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Graph Analysis Summary\n")
            f.write(f"{'='*50}\n")
            f.write(f"Timestamp: {result.timestamp}\n")
            f.write(f"Overall Score: {result.overall_score:.1f}/100\n\n")
            
            if result.reliability:
                f.write(f"Reliability: {result.reliability.score:.1f}/100\n")
                f.write(f"  - {len(result.reliability.findings)} findings\n")
                f.write(f"  - {len(result.reliability.critical_components)} critical components\n\n")
            
            if result.maintainability:
                f.write(f"Maintainability: {result.maintainability.score:.1f}/100\n")
                f.write(f"  - {len(result.maintainability.findings)} findings\n")
                f.write(f"  - {len(result.maintainability.critical_components)} critical components\n\n")
            
            if result.availability:
                f.write(f"Availability: {result.availability.score:.1f}/100\n")
                f.write(f"  - {len(result.availability.findings)} findings\n")
                f.write(f"  - {len(result.availability.critical_components)} critical components\n")
        
        print_success(f"Summary exported: {summary_file}")


# ============================================================================
# CLI
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description='Analyze DEPENDS_ON graph using Neo4j GDS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis
    python analyze_graph.py
    
    # Full analysis with Neo4j
    python analyze_graph.py --uri bolt://localhost:7687 --full
    
    # Export results
    python analyze_graph.py --output results/ --format json summary
    
    # Specific quality attributes
    python analyze_graph.py --reliability --availability
    
    # Filter by dependency type
    python analyze_graph.py --dep-types app_to_app node_to_node
        """
    )
    
    # Neo4j connection
    conn_group = parser.add_argument_group('Neo4j Connection')
    conn_group.add_argument('--uri', default='bolt://localhost:7687',
                           help='Neo4j URI (default: bolt://localhost:7687)')
    conn_group.add_argument('--user', '-u', default='neo4j',
                           help='Neo4j username (default: neo4j)')
    conn_group.add_argument('--password', '-p', default='password',
                           help='Neo4j password (default: password)')
    conn_group.add_argument('--database', '-d', default='neo4j',
                           help='Neo4j database (default: neo4j)')
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument('--full', '-f', action='store_true',
                               help='Run all analyses')
    analysis_group.add_argument('--reliability', '-R', action='store_true',
                               help='Analyze reliability')
    analysis_group.add_argument('--maintainability', '-M', action='store_true',
                               help='Analyze maintainability')
    analysis_group.add_argument('--availability', '-A', action='store_true',
                               help='Analyze availability')
    analysis_group.add_argument('--dep-types', nargs='+',
                               choices=['app_to_app', 'node_to_node', 
                                       'app_to_broker', 'node_to_broker'],
                               help='DEPENDS_ON types to analyze')
    analysis_group.add_argument('--weighted', '-W', action='store_true', default=True,
                               help='Use weighted analysis (default: True)')
    analysis_group.add_argument('--no-weighted', action='store_true',
                               help='Disable weighted analysis (use unweighted)')
    analysis_group.add_argument('--weight-property', default='weight',
                               help='Name of the weight property on DEPENDS_ON (default: weight)')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output', '-o',
                             help='Output directory for results')
    output_group.add_argument('--format', nargs='+',
                             choices=['json', 'summary'],
                             default=['json'],
                             help='Output formats')
    
    # Display options
    display_group = parser.add_argument_group('Display Options')
    display_group.add_argument('--verbose', '-v', action='store_true',
                              help='Verbose output with detailed findings')
    display_group.add_argument('--quiet', '-q', action='store_true',
                              help='Minimal output')
    display_group.add_argument('--no-color', action='store_true',
                              help='Disable colored output')
    display_group.add_argument('--json-stdout', action='store_true',
                              help='Output JSON to stdout')
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger('analyze_graph')
    
    # Handle colors
    if not sys.stdout.isatty() or args.no_color:
        Colors.disable()
    
    # Determine analyses to run
    if args.full or (not args.reliability and not args.maintainability and not args.availability):
        # Default: run all
        analyze_reliability = True
        analyze_maintainability = True
        analyze_availability = True
    else:
        analyze_reliability = args.reliability
        analyze_maintainability = args.maintainability
        analyze_availability = args.availability
    
    # Determine weighted mode
    use_weights = not args.no_weighted
    
    # Print header
    if not args.quiet:
        print_header("GRAPH ANALYZER (GDS)")
        print(f"\n  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Neo4j: {args.uri}")
        print(f"  Weighted: {'Yes' if use_weights else 'No'}")
        if use_weights:
            print(f"  Weight Property: {args.weight_property}")
        analyses = []
        if analyze_reliability:
            analyses.append("Reliability")
        if analyze_maintainability:
            analyses.append("Maintainability")
        if analyze_availability:
            analyses.append("Availability")
        print(f"  Analyzing: {', '.join(analyses)}")
    
    try:
        # Connect to Neo4j
        if not args.quiet:
            print_section("CONNECTING TO NEO4J")
        
        gds = GDSClient(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database
        )
        
        print_success(f"Connected to {args.uri}")
        
        # Run analysis
        orchestrator = GDSAnalysisOrchestrator(gds)
        result = orchestrator.analyze(
            analyze_reliability=analyze_reliability,
            analyze_maintainability=analyze_maintainability,
            analyze_availability=analyze_availability,
            dependency_types=args.dep_types,
            use_weights=use_weights,
            weight_property=args.weight_property
        )
        
        # Output results
        if args.json_stdout:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            # Print graph summary
            print_graph_summary(result.graph_stats)
            
            # Print analysis results
            if result.reliability:
                print_analysis_result(result.reliability, verbose=args.verbose)
            
            if result.maintainability:
                print_analysis_result(result.maintainability, verbose=args.verbose)
            
            if result.availability:
                print_analysis_result(result.availability, verbose=args.verbose)
            
            # Print overall summary
            print_overall_summary(result)
        
        # Export if requested
        if args.output:
            if not args.quiet:
                print_section("EXPORTING RESULTS")
            export_results(result, args.output, args.format)
        
        # Close connection
        gds.close()
        
        if not args.quiet:
            print_success("\nAnalysis complete!")
        
        # Exit code based on score
        if result.overall_score < 50:
            sys.exit(2)  # Critical issues
        elif result.overall_score < 70:
            sys.exit(1)  # Warnings
        sys.exit(0)
        
    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        print_warning("Install with: pip install neo4j")
        sys.exit(1)
        
    except Exception as e:
        print_error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()