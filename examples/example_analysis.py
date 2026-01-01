#!/usr/bin/env python3
"""
Example: Graph Analysis Usage - Version 5.0

Demonstrates the analysis module capabilities:
1. Component-type-specific analysis
2. Box-plot classification
3. Problem detection
4. Anti-pattern detection
5. Edge criticality analysis

Requirements:
- Neo4j with GDS plugin running
- Graph data imported into Neo4j

Usage:
    # With Neo4j connection
    python examples/example_analysis.py --uri bolt://localhost:7687 --password secret
    
    # Demo mode (no Neo4j required)
    python examples/example_analysis.py --demo

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

import argparse
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Demo Mode (No Neo4j Required)
# =============================================================================

def run_demo_mode():
    """Run demonstration without Neo4j connection"""
    print("\n" + "=" * 70)
    print("ANALYSIS MODULE DEMO (No Neo4j Required)")
    print("=" * 70)
    
    from src.analysis import (
        BoxPlotClassifier,
        CriticalityLevel,
        ClassificationResult,
    )
    
    # ==========================================================================
    # Demo 1: Box-Plot Classification
    # ==========================================================================
    print("\n" + "-" * 50)
    print("DEMO 1: Box-Plot Classification")
    print("-" * 50)
    
    # Simulated centrality scores for applications
    app_scores = [
        {"id": "app_order_service", "type": "Application", "score": 0.95},
        {"id": "app_payment_gateway", "type": "Application", "score": 0.88},
        {"id": "app_inventory_service", "type": "Application", "score": 0.72},
        {"id": "app_user_service", "type": "Application", "score": 0.65},
        {"id": "app_notification_service", "type": "Application", "score": 0.45},
        {"id": "app_analytics_service", "type": "Application", "score": 0.38},
        {"id": "app_logging_service", "type": "Application", "score": 0.25},
        {"id": "app_cache_service", "type": "Application", "score": 0.18},
        {"id": "app_health_check", "type": "Application", "score": 0.12},
        {"id": "app_config_service", "type": "Application", "score": 0.08},
    ]
    
    print(f"\nClassifying {len(app_scores)} applications by composite score...")
    
    classifier = BoxPlotClassifier(k_factor=1.5)
    result = classifier.classify(app_scores, metric_name="composite")
    
    print(f"\nBox-Plot Statistics:")
    stats = result.stats
    print(f"  Min:     {stats.min_val:.4f}")
    print(f"  Q1:      {stats.q1:.4f}")
    print(f"  Median:  {stats.median:.4f}")
    print(f"  Q3:      {stats.q3:.4f}")
    print(f"  Max:     {stats.max_val:.4f}")
    print(f"  IQR:     {stats.iqr:.4f}")
    print(f"  Upper Fence: {stats.upper_fence:.4f}")
    
    print(f"\nClassification Distribution:")
    for level in CriticalityLevel:
        count = result.summary.get(level, 0)
        if count > 0:
            print(f"  {level.value.upper():10s}: {count}")
    
    print(f"\nTop Critical Components:")
    for item in result.get_high_and_above():
        outlier_mark = " [OUTLIER]" if item.is_outlier else ""
        print(f"  {item.rank}. {item.id}: {item.score:.4f} ({item.level.value}){outlier_mark}")
    
    # ==========================================================================
    # Demo 2: Multi-Type Classification
    # ==========================================================================
    print("\n" + "-" * 50)
    print("DEMO 2: Component-Type Specific Classification")
    print("-" * 50)
    
    # Mixed component data
    mixed_scores = [
        # Applications
        {"id": "app_order", "type": "Application", "score": 0.92},
        {"id": "app_payment", "type": "Application", "score": 0.85},
        {"id": "app_user", "type": "Application", "score": 0.55},
        {"id": "app_logging", "type": "Application", "score": 0.15},
        # Brokers
        {"id": "broker_main", "type": "Broker", "score": 0.95},
        {"id": "broker_backup", "type": "Broker", "score": 0.35},
        # Topics
        {"id": "topic_orders", "type": "Topic", "score": 0.88},
        {"id": "topic_payments", "type": "Topic", "score": 0.75},
        {"id": "topic_events", "type": "Topic", "score": 0.45},
        {"id": "topic_logs", "type": "Topic", "score": 0.12},
        # Nodes
        {"id": "node_app_server_1", "type": "Node", "score": 0.78},
        {"id": "node_app_server_2", "type": "Node", "score": 0.65},
        {"id": "node_db_server", "type": "Node", "score": 0.55},
    ]
    
    print(f"\nClassifying {len(mixed_scores)} components by type...")
    
    by_type = classifier.classify_by_type(mixed_scores, metric_name="composite")
    
    for comp_type, type_result in by_type.items():
        print(f"\n{comp_type} ({len(type_result.items)} components):")
        print(f"  Statistics: Q1={type_result.stats.q1:.3f}, Median={type_result.stats.median:.3f}, Q3={type_result.stats.q3:.3f}")
        
        critical = type_result.get_high_and_above()
        if critical:
            print(f"  Critical/High Components:")
            for item in critical:
                print(f"    - {item.id}: {item.score:.3f} ({item.level.value})")
    
    # ==========================================================================
    # Demo 3: Problem Detection (Simulated)
    # ==========================================================================
    print("\n" + "-" * 50)
    print("DEMO 3: Problem Detection (Simulated)")
    print("-" * 50)
    
    from src.analysis import (
        ProblemType,
        ProblemSeverity,
        QualityAttribute,
        Problem,
        Symptom,
    )
    
    # Simulate detected problems
    simulated_problems = [
        Problem(
            problem_type=ProblemType.SINGLE_POINT_OF_FAILURE,
            severity=ProblemSeverity.CRITICAL,
            title="Single Point of Failure: broker_main",
            description="Broker handles all message traffic with no redundancy",
            affected_components=["broker_main"],
            symptoms=[
                Symptom(
                    name="High Betweenness",
                    description="Component sits on many shortest paths",
                    metric="betweenness",
                    value=0.95,
                    threshold=0.75
                ),
                Symptom(
                    name="Articulation Point",
                    description="Removal disconnects the graph",
                    metric="is_articulation_point",
                    value=1.0,
                    threshold=1.0
                ),
            ],
            impact="System-wide message delivery failure if broker fails",
            recommendation="Deploy backup broker with active-passive failover",
            quality_attributes=[QualityAttribute.RELIABILITY],
        ),
        Problem(
            problem_type=ProblemType.HIGH_COUPLING,
            severity=ProblemSeverity.HIGH,
            title="High Coupling: app_order_service",
            description="Application has too many direct dependencies",
            affected_components=["app_order_service"],
            symptoms=[
                Symptom(
                    name="High Degree",
                    description="Component has many connections",
                    metric="degree",
                    value=25,
                    threshold=15
                ),
            ],
            impact="Changes propagate widely, difficult to modify",
            recommendation="Introduce message bus or event-driven decoupling",
            quality_attributes=[QualityAttribute.MAINTAINABILITY],
        ),
        Problem(
            problem_type=ProblemType.BOTTLENECK,
            severity=ProblemSeverity.MEDIUM,
            title="Bottleneck: topic_orders",
            description="Topic handles disproportionate message volume",
            affected_components=["topic_orders"],
            symptoms=[
                Symptom(
                    name="High PageRank",
                    description="Component is highly connected",
                    metric="pagerank",
                    value=0.88,
                    threshold=0.70
                ),
            ],
            impact="Performance degradation under high load",
            recommendation="Consider topic partitioning or load balancing",
            quality_attributes=[QualityAttribute.AVAILABILITY],
        ),
    ]
    
    print(f"\nDetected {len(simulated_problems)} problems:\n")
    
    for problem in simulated_problems:
        severity_colors = {
            ProblemSeverity.CRITICAL: "\033[91m",
            ProblemSeverity.HIGH: "\033[93m",
            ProblemSeverity.MEDIUM: "\033[94m",
            ProblemSeverity.LOW: "\033[92m",
        }
        RESET = "\033[0m"
        color = severity_colors.get(problem.severity, RESET)
        
        print(f"{color}[{problem.severity.value.upper()}]{RESET} {problem.title}")
        print(f"  Quality: {', '.join(qa.value for qa in problem.quality_attributes)}")
        print(f"  Impact: {problem.impact[:60]}...")
        print(f"  Symptoms:")
        for symptom in problem.symptoms:
            print(f"    - {symptom.name}: {symptom.value:.2f} (threshold: {symptom.threshold:.2f})")
        print(f"  Recommendation: {problem.recommendation[:60]}...")
        print()
    
    # ==========================================================================
    # Demo 4: Anti-Pattern Detection (Simulated)
    # ==========================================================================
    print("-" * 50)
    print("DEMO 4: Anti-Pattern Detection (Simulated)")
    print("-" * 50)
    
    from src.analysis import (
        AntiPatternType,
        PatternSeverity,
        AntiPattern,
    )
    
    simulated_antipatterns = [
        AntiPattern(
            pattern_type=AntiPatternType.GOD_TOPIC,
            severity=PatternSeverity.HIGH,
            affected_components=["topic_events"],
            description="Topic 'topic_events' has 35% of all connections in the system",
            impact="Single point of failure for event-driven communication",
            recommendation="Split into domain-specific topics (user_events, order_events, etc.)",
            quality_attributes=["reliability", "maintainability"],
            metrics={"connection_percentage": 0.35, "total_connections": 42},
        ),
        AntiPattern(
            pattern_type=AntiPatternType.HUB_AND_SPOKE,
            severity=PatternSeverity.MEDIUM,
            affected_components=["broker_main"],
            description="Centralized topology with broker_main handling 85% of traffic",
            impact="No fault tolerance, performance bottleneck",
            recommendation="Implement distributed broker cluster",
            quality_attributes=["availability", "reliability"],
            metrics={"traffic_percentage": 0.85},
        ),
        AntiPattern(
            pattern_type=AntiPatternType.CHATTY_APPLICATION,
            severity=PatternSeverity.LOW,
            affected_components=["app_analytics_service"],
            description="Application publishes to 12 topics (above normal range)",
            impact="Network overhead, tight coupling",
            recommendation="Consolidate related events into fewer topics",
            quality_attributes=["maintainability"],
            metrics={"topic_count": 12, "threshold": 8},
        ),
    ]
    
    print(f"\nDetected {len(simulated_antipatterns)} anti-patterns:\n")
    
    for pattern in simulated_antipatterns:
        severity_colors = {
            PatternSeverity.CRITICAL: "\033[91m",
            PatternSeverity.HIGH: "\033[93m",
            PatternSeverity.MEDIUM: "\033[94m",
            PatternSeverity.LOW: "\033[92m",
        }
        RESET = "\033[0m"
        color = severity_colors.get(pattern.severity, RESET)
        
        pattern_name = pattern.pattern_type.value.replace("_", " ").title()
        print(f"{color}[{pattern.severity.value.upper()}]{RESET} {pattern_name}")
        print(f"  Affected: {', '.join(pattern.affected_components)}")
        print(f"  Description: {pattern.description[:60]}...")
        print(f"  Recommendation: {pattern.recommendation[:60]}...")
        print()
    
    # ==========================================================================
    # Demo 5: JSON Export
    # ==========================================================================
    print("-" * 50)
    print("DEMO 5: JSON Export")
    print("-" * 50)
    
    export_data = {
        "classification": result.to_dict(),
        "by_component_type": {ct: r.to_dict() for ct, r in by_type.items()},
        "problems": [p.to_dict() for p in simulated_problems],
        "antipatterns": [p.to_dict() for p in simulated_antipatterns],
    }
    
    json_output = json.dumps(export_data, indent=2, default=str)
    print(f"\nExported data structure ({len(json_output)} bytes):")
    print(f"  - classification.items: {len(export_data['classification']['items'])} items")
    print(f"  - by_component_type: {list(export_data['by_component_type'].keys())}")
    print(f"  - problems: {len(export_data['problems'])} problems")
    print(f"  - antipatterns: {len(export_data['antipatterns'])} patterns")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nTo run with actual Neo4j data:")
    print("  python examples/example_analysis.py --uri bolt://localhost:7687 --password your_password")
    print()


# =============================================================================
# Live Mode (With Neo4j)
# =============================================================================

def run_live_mode(args):
    """Run with actual Neo4j connection"""
    print("\n" + "=" * 70)
    print("ANALYSIS MODULE - LIVE MODE")
    print("=" * 70)
    
    try:
        from src.analysis import GDSAnalyzer
    except ImportError as e:
        print(f"\nError: Could not import analysis module: {e}")
        print("Make sure you're running from the project root.")
        return
    
    print(f"\nConnecting to Neo4j at {args.uri}...")
    
    try:
        with GDSAnalyzer(
            uri=args.uri,
            user=args.user,
            password=args.password,
            database=args.database,
            k_factor=args.k_factor,
        ) as analyzer:
            
            print("Connected successfully!\n")
            
            # Get graph statistics
            print("-" * 50)
            print("GRAPH STATISTICS")
            print("-" * 50)
            stats = analyzer.gds.get_graph_stats()
            print(f"  Total Nodes: {stats.get('total_nodes', 0)}")
            print(f"  DEPENDS_ON Edges: {stats.get('depends_on_total', 0)}")
            print(f"  By Type:")
            for comp_type, count in stats.get('nodes', {}).items():
                if count > 0:
                    print(f"    {comp_type}: {count}")
            
            # Component type analysis
            print("\n" + "-" * 50)
            print("COMPONENT TYPE ANALYSIS")
            print("-" * 50)
            
            type_results = analyzer.analyze_all_component_types()
            
            for comp_type, type_result in type_results.items():
                if type_result.component_count > 0:
                    print(f"\n{comp_type} ({type_result.component_count} components):")
                    
                    # Summary
                    summary = type_result.composite_classification.summary
                    print(f"  Distribution: CRITICAL={summary.get(CriticalityLevel.CRITICAL, 0)}, "
                          f"HIGH={summary.get(CriticalityLevel.HIGH, 0)}, "
                          f"MEDIUM={summary.get(CriticalityLevel.MEDIUM, 0)}")
                    
                    # Top critical
                    critical = type_result.get_critical_components()[:3]
                    if critical:
                        print(f"  Top Critical:")
                        for comp in critical:
                            ap_mark = " [AP]" if comp.is_articulation_point else ""
                            print(f"    - {comp.component_id}: {comp.composite_score:.4f}{ap_mark}")
            
            # Problem detection
            print("\n" + "-" * 50)
            print("PROBLEM DETECTION")
            print("-" * 50)
            
            problems = analyzer.detect_problems()
            print(f"  Total: {problems.total_count}")
            print(f"  Critical: {problems.critical_count}")
            
            if problems.problems:
                print(f"\n  Top Problems:")
                for problem in problems.problems[:5]:
                    print(f"    [{problem.severity.value.upper()}] {problem.title}")
            
            # Anti-pattern detection
            print("\n" + "-" * 50)
            print("ANTI-PATTERN DETECTION")
            print("-" * 50)
            
            antipatterns = analyzer.detect_antipatterns()
            print(f"  Total: {antipatterns.total_count}")
            
            if antipatterns.patterns:
                print(f"\n  Detected Patterns:")
                for pattern in antipatterns.patterns[:5]:
                    pattern_name = pattern.pattern_type.value.replace("_", " ").title()
                    print(f"    [{pattern.severity.value.upper()}] {pattern_name}")
            
            # Edge analysis
            print("\n" + "-" * 50)
            print("EDGE ANALYSIS")
            print("-" * 50)
            
            edges = analyzer.analyze_edges()
            print(f"  Total Edges: {edges.edge_count}")
            print(f"  Critical Edges: {len(edges.get_critical_edges())}")
            print(f"  Bridge Edges: {len(edges.get_bridges())}")
            
            print("\n" + "=" * 70)
            print("ANALYSIS COMPLETE")
            print("=" * 70)
            
            # Export option
            if args.output:
                result = analyzer.analyze_all()
                with open(args.output, "w") as f:
                    json.dump(result.to_dict(), f, indent=2, default=str)
                print(f"\nResults exported to: {args.output}")
            
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        print("\nMake sure:")
        print("  1. Neo4j is running with GDS plugin")
        print("  2. Graph data has been imported")
        print("  3. Connection credentials are correct")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Example: Graph Analysis Usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo mode (no Neo4j required)
  python examples/example_analysis.py --demo
  
  # Live mode with Neo4j
  python examples/example_analysis.py --uri bolt://localhost:7687 --password secret
  
  # Export results to JSON
  python examples/example_analysis.py --uri bolt://localhost:7687 --password secret --output results.json
        """
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode without Neo4j connection"
    )
    parser.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j bolt URI"
    )
    parser.add_argument(
        "--user",
        default="neo4j",
        help="Neo4j username"
    )
    parser.add_argument(
        "--password",
        default="password",
        help="Neo4j password"
    )
    parser.add_argument(
        "--database",
        default="neo4j",
        help="Neo4j database name"
    )
    parser.add_argument(
        "--k-factor",
        type=float,
        default=1.5,
        help="Box-plot k-factor for classification"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for JSON results"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo_mode()
    else:
        run_live_mode(args)


if __name__ == "__main__":
    main()
