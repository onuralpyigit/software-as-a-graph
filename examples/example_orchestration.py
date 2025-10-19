"""
Example: Orchestration System Usage

Demonstrates:
1. Complete analysis pipeline
2. Report generation (multiple formats)
3. End-to-end workflows
4. Integration of all modules
"""

import sys
from pathlib import Path
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent / '..'))

from src.orchestration.report_generator import (
    ReportGenerator, ReportConfig, ReportType, ReportFormat
)
from src.core.graph_builder import GraphBuilder
from src.core.graph_exporter import GraphExporter
from src.analysis.criticality_scorer import CompositeCriticalityScorer
from src.analysis.centrality_analyzer import CentralityAnalyzer
from src.simulation.failure_simulator import FailureSimulator
from src.simulation.impact_calculator import ImpactCalculator
from src.visualization.metrics_dashboard import MetricsDashboard


def create_example_system():
    """Create example system"""
    
    G = nx.DiGraph()
    
    components = {
        'WebApp': {'type': 'Application', 'criticality_score': 0.7},
        'APIGateway': {'type': 'Application', 'criticality_score': 0.9},
        'AuthService': {'type': 'Application', 'criticality_score': 0.95},
        'UserService': {'type': 'Application', 'criticality_score': 0.8},
        'OrderService': {'type': 'Application', 'criticality_score': 0.9},
        'MainBroker': {'type': 'Broker', 'criticality_score': 0.85},
        'UserDB': {'type': 'Node', 'criticality_score': 0.9},
        'OrderDB': {'type': 'Node', 'criticality_score': 0.95},
    }
    
    for name, attrs in components.items():
        G.add_node(name, **attrs)
    
    edges = [
        ('WebApp', 'APIGateway'),
        ('APIGateway', 'AuthService'),
        ('APIGateway', 'UserService'),
        ('APIGateway', 'OrderService'),
        ('UserService', 'UserDB'),
        ('OrderService', 'OrderDB'),
        ('OrderService', 'MainBroker'),
    ]
    
    G.add_edges_from(edges, weight=1.0)
    
    return G


def run_complete_analysis(graph: nx.DiGraph) -> dict:
    """Run complete analysis pipeline"""
    
    print("\n[Running complete analysis pipeline...]")
    
    results = {}
    
    # Topology
    print("  - Analyzing topology...")
    results['topology'] = {
        'node_count': len(graph),
        'edge_count': len(graph.edges()),
        'density': nx.density(graph),
        'avg_degree': sum(dict(graph.degree()).values()) / len(graph) if len(graph) > 0 else 0
    }
    
    # Criticality
    print("  - Analyzing criticality...")
    scorer = CompositeCriticalityScorer()
    crit_scores = scorer.calculate_all_scores(graph)
    results['criticality'] = {
        node: score.composite_score 
        for node, score in crit_scores.items()
    }
    
    # Centrality
    print("  - Analyzing centrality...")
    centrality = CentralityAnalyzer()
    results['centrality'] = {
        'hubs': centrality.identify_hubs(graph, threshold=0.7),
        'bottlenecks': centrality.identify_bottlenecks(graph, threshold=0.7)
    }
    
    # Health
    print("  - Calculating health metrics...")
    dashboard = MetricsDashboard()
    results['health'] = dashboard.get_system_health(graph)
    
    # Simulation (critical component)
    print("  - Running failure simulation...")
    critical_component = max(
        results['criticality'].items(),
        key=lambda x: x[1]
    )[0]
    
    simulator = FailureSimulator()
    sim_result = simulator.simulate_single_failure(graph, critical_component)
    
    results['simulation'] = {
        'component': critical_component,
        'impact_score': sim_result.impact_score,
        'affected_components': sim_result.affected_components,
        'resilience_score': sim_result.resilience_score
    }
    
    # Recommendations
    print("  - Generating recommendations...")
    results['recommendations'] = generate_recommendations(results)
    
    print("  ✓ Analysis complete!")
    
    return results


def generate_recommendations(results: dict) -> list:
    """Generate recommendations from analysis results"""
    
    recommendations = []
    
    # Check criticality
    if 'criticality' in results:
        critical_count = sum(1 for s in results['criticality'].values() if s > 0.8)
        if critical_count > 3:
            recommendations.append(
                f"High number of critical components ({critical_count}). "
                "Consider implementing redundancy for critical services."
            )
    
    # Check health
    if 'health' in results:
        health = results['health']['overall_health']
        if health < 75:
            recommendations.append(
                f"System health ({health:.1f}%) is below optimal. "
                "Review component connectivity and redundancy."
            )
    
    # Check simulation
    if 'simulation' in results:
        impact = results['simulation']['impact_score']
        if impact > 0.5:
            recommendations.append(
                f"Failure of {results['simulation']['component']} would have significant impact. "
                "Implement failover mechanisms."
            )
    
    if not recommendations:
        recommendations.append("System is in good health. Continue monitoring.")
    
    return recommendations


def example_1_executive_report():
    """Example 1: Generate executive report"""
    
    print("\n" + "=" * 70)
    print("Example 1: Executive Summary Report")
    print("=" * 70)
    
    # Create system and analyze
    print("\n[Step 1] Creating and analyzing system...")
    graph = create_example_system()
    results = run_complete_analysis(graph)
    
    # Generate report
    print("\n[Step 2] Generating executive report...")
    generator = ReportGenerator()
    
    config = ReportConfig(
        report_type=ReportType.EXECUTIVE,
        format=ReportFormat.HTML,
        title="Q4 System Health Review",
        author="System Administrator"
    )
    
    html = generator.generate_report(
        results,
        config,
        output_path='output/report_executive.html'
    )
    
    print(f"  ✓ Saved to: report_executive.html")
    
    # Also generate markdown version
    print("\n[Step 3] Generating markdown version...")
    config.format = ReportFormat.MARKDOWN
    generator.generate_report(
        results,
        config,
        output_path='output/report_executive.md'
    )
    print(f"  ✓ Saved to: report_executive.md")


def example_2_technical_report():
    """Example 2: Generate technical report"""
    
    print("\n" + "=" * 70)
    print("Example 2: Technical Analysis Report")
    print("=" * 70)
    
    graph = create_example_system()
    results = run_complete_analysis(graph)
    
    print("\n[Generating detailed technical report...]")
    
    generator = ReportGenerator()
    config = ReportConfig(
        report_type=ReportType.TECHNICAL,
        format=ReportFormat.HTML,
        include_raw_data=True,
        max_detail_level=5
    )
    
    generator.generate_report(
        results,
        config,
        output_path='output/report_technical.html'
    )
    
    print(f"  ✓ Saved to: report_technical.html")


def example_3_comprehensive_report():
    """Example 3: Generate comprehensive report"""
    
    print("\n" + "=" * 70)
    print("Example 3: Comprehensive Report")
    print("=" * 70)
    
    graph = create_example_system()
    results = run_complete_analysis(graph)
    
    print("\n[Generating comprehensive report with all details...]")
    
    generator = ReportGenerator()
    config = ReportConfig(
        report_type=ReportType.COMPREHENSIVE,
        format=ReportFormat.HTML,
        include_charts=True,
        include_recommendations=True,
        include_raw_data=True
    )
    
    generator.generate_report(
        results,
        config,
        output_path='output/report_comprehensive.html'
    )
    
    print(f"  ✓ Saved to: report_comprehensive.html")


def example_4_comparison_report():
    """Example 4: Generate comparison report"""
    
    print("\n" + "=" * 70)
    print("Example 4: Comparison Report")
    print("=" * 70)
    
    # Create baseline
    print("\n[Step 1] Creating baseline system...")
    baseline_graph = create_example_system()
    baseline_results = run_complete_analysis(baseline_graph)
    
    # Create modified system
    print("\n[Step 2] Creating modified system...")
    current_graph = create_example_system()
    # Simulate some changes
    current_graph.add_node('NewService', type='Application', criticality_score=0.6)
    current_graph.add_edge('APIGateway', 'NewService')
    current_results = run_complete_analysis(current_graph)
    
    # Generate comparison
    print("\n[Step 3] Generating comparison report...")
    generator = ReportGenerator()
    
    comparison_data = {
        'baseline': baseline_results,
        'current': current_results
    }
    
    config = ReportConfig(
        report_type=ReportType.COMPARISON,
        format=ReportFormat.HTML
    )
    
    generator.generate_report(
        comparison_data,
        config,
        output_path='output/report_comparison.html'
    )
    
    print(f"  ✓ Saved to: report_comparison.html")


def example_5_health_report():
    """Example 5: Generate health report"""
    
    print("\n" + "=" * 70)
    print("Example 5: System Health Report")
    print("=" * 70)
    
    graph = create_example_system()
    results = run_complete_analysis(graph)
    
    print("\n[Generating health status report...]")
    
    generator = ReportGenerator()
    config = ReportConfig(
        report_type=ReportType.HEALTH,
        format=ReportFormat.HTML
    )
    
    generator.generate_report(
        results,
        config,
        output_path='output/report_health.html'
    )
    
    print(f"  ✓ Saved to: report_health.html")


def example_6_json_export():
    """Example 6: Export as JSON"""
    
    print("\n" + "=" * 70)
    print("Example 6: JSON Export")
    print("=" * 70)
    
    graph = create_example_system()
    results = run_complete_analysis(graph)
    
    print("\n[Exporting results as JSON...]")
    
    generator = ReportGenerator()
    json_report = generator.generate_json_report(
        results,
        include_metadata=True
    )
    
    Path('output/report_results.json').write_text(json_report)
    print(f"  ✓ Saved to: report_results.json")


def example_7_executive_summary():
    """Example 7: Generate executive summary"""
    
    print("\n" + "=" * 70)
    print("Example 7: Executive Summary")
    print("=" * 70)
    
    graph = create_example_system()
    results = run_complete_analysis(graph)
    
    print("\n[Generating executive summary...]")
    
    generator = ReportGenerator()
    summary = generator.generate_executive_summary(results)
    
    print(f"\n{summary}")


def example_8_complete_workflow():
    """Example 8: Complete end-to-end workflow"""
    
    print("\n" + "=" * 70)
    print("Example 8: Complete End-to-End Workflow")
    print("=" * 70)
    
    # Step 1: Build system
    print("\n[Step 1] Building system model...")
    graph = create_example_system()
    print(f"  ✓ System with {len(graph)} components")
    
    # Step 2: Run analysis
    print("\n[Step 2] Running comprehensive analysis...")
    results = run_complete_analysis(graph)
    
    # Step 3: Generate all reports
    print("\n[Step 3] Generating reports...")
    generator = ReportGenerator()
    
    reports_generated = []
    
    # Executive report
    generator.generate_report(
        results,
        ReportConfig(ReportType.EXECUTIVE, ReportFormat.HTML),
        'output/final_executive.html'
    )
    reports_generated.append('final_executive.html')
    
    # Technical report
    generator.generate_report(
        results,
        ReportConfig(ReportType.TECHNICAL, ReportFormat.HTML),
        'output/final_technical.html'
    )
    reports_generated.append('final_technical.html')
    
    # Health report
    generator.generate_report(
        results,
        ReportConfig(ReportType.HEALTH, ReportFormat.HTML),
        'output/final_health.html'
    )
    reports_generated.append('final_health.html')
    
    # JSON export
    json_data = generator.generate_json_report(results)
    Path('output/final_results.json').write_text(json_data)
    reports_generated.append('final_results.json')
    
    print(f"  ✓ Generated {len(reports_generated)} reports")
    
    # Step 4: Summary
    print("\n[Step 4] Executive Summary:")
    print(f"\n{generator.generate_executive_summary(results)}")
    
    print("\n[Step 5] Generated Files:")
    for report in reports_generated:
        print(f"  - output/{report}")


def main():
    """Run all examples"""
    
    print("\n" + "=" * 70)
    print("ORCHESTRATION SYSTEM - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    
    # Create output directory
    Path('output').mkdir(exist_ok=True)
    
    try:
        example_1_executive_report()
        example_2_technical_report()
        example_3_comprehensive_report()
        example_4_comparison_report()
        example_5_health_report()
        example_6_json_export()
        example_7_executive_summary()
        example_8_complete_workflow()
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print("\n📚 Summary of Capabilities:")
        print("  ✓ Complete analysis orchestration")
        print("  ✓ Executive summaries")
        print("  ✓ Technical reports")
        print("  ✓ Comprehensive reports")
        print("  ✓ Comparison reports")
        print("  ✓ Health reports")
        print("  ✓ JSON/Markdown/HTML export")
        print("  ✓ Custom report templates")
        print("  ✓ End-to-end workflows")
        
        print("\n📖 Usage in Your Code:")
        print("""
from refactored.orchestration.report_generator import ReportGenerator, ReportConfig, ReportType, ReportFormat

# Run analysis
results = run_analysis(graph)

# Generate report
generator = ReportGenerator()
config = ReportConfig(
    report_type=ReportType.EXECUTIVE,
    format=ReportFormat.HTML
)
generator.generate_report(results, config, 'report.html')
        """)
        
        print("\n📁 Generated Files:")
        print("  Check the output/ directory for all generated reports")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
