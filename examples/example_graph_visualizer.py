#!/usr/bin/env python3
"""
Graph Visualizer Examples
=========================

Demonstrates how to use the GraphVisualizer to create visualizations
of distributed pub-sub systems, including topology, analysis results,
simulation impact, and validation comparisons.

Examples:
1. Basic topology visualization
2. Criticality heatmap
3. Full workflow with all results
4. Custom styling options
5. HTML report generation
6. Exporting for external tools

Usage:
    python example_graph_visualizer.py

Author: Software-as-a-Graph Research Project
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import GraphAnalyzer
from src.simulation import GraphSimulator
from src.validation import GraphValidator
from src.visualization import (
    GraphVisualizer,
    VisualizationConfig,
    ColorScheme,
    visualize_system,
    MATPLOTLIB_AVAILABLE,
)


# ============================================================================
# Sample Data
# ============================================================================

SAMPLE_PUBSUB_DATA = {
    "nodes": [
        {"id": "N1", "name": "ComputeNode1", "type": "compute"},
        {"id": "N2", "name": "ComputeNode2", "type": "compute"},
        {"id": "N3", "name": "EdgeNode", "type": "edge"}
    ],
    "brokers": [
        {"id": "B1", "name": "MainBroker", "node": "N1"},
        {"id": "B2", "name": "BackupBroker", "node": "N2"}
    ],
    "applications": [
        {"id": "A1", "name": "SensorReader", "role": "pub", "node": "N3"},
        {"id": "A2", "name": "DataProcessor", "role": "both", "node": "N1"},
        {"id": "A3", "name": "Analytics", "role": "sub", "node": "N1"},
        {"id": "A4", "name": "Dashboard", "role": "sub", "node": "N2"},
        {"id": "A5", "name": "Alerting", "role": "sub", "node": "N2"},
        {"id": "A6", "name": "Logger", "role": "sub", "node": "N1"}
    ],
    "topics": [
        {"id": "T1", "name": "sensor/data", "broker": "B1"},
        {"id": "T2", "name": "processed/data", "broker": "B1"},
        {"id": "T3", "name": "alerts", "broker": "B2"}
    ],
    "relationships": {
        "publishes_to": [
            {"from": "A1", "to": "T1"},
            {"from": "A2", "to": "T2"},
            {"from": "A2", "to": "T3"}
        ],
        "subscribes_to": [
            {"from": "A2", "to": "T1"},
            {"from": "A3", "to": "T2"},
            {"from": "A4", "to": "T2"},
            {"from": "A5", "to": "T3"},
            {"from": "A6", "to": "T1"},
            {"from": "A6", "to": "T2"}
        ],
        "runs_on": [
            {"from": "A1", "to": "N3"},
            {"from": "A2", "to": "N1"},
            {"from": "A3", "to": "N1"},
            {"from": "A4", "to": "N2"},
            {"from": "A5", "to": "N2"},
            {"from": "A6", "to": "N1"},
            {"from": "B1", "to": "N1"},
            {"from": "B2", "to": "N2"}
        ],
        "routes": [
            {"from": "B1", "to": "T1"},
            {"from": "B1", "to": "T2"},
            {"from": "B2", "to": "T3"}
        ]
    }
}


def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def print_subheader(title: str):
    """Print a formatted subheader"""
    print(f"\n── {title} ──")


# ============================================================================
# Example Functions
# ============================================================================

def example_1_basic_topology():
    """Example 1: Basic topology visualization"""
    print_header("Example 1: Basic Topology Visualization")
    
    # Setup
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    analysis_result = analyzer.analyze()
    
    # Create visualizer
    visualizer = GraphVisualizer(analyzer)
    visualizer.set_analysis_result(analysis_result)
    
    # Create output directory
    output_dir = Path(tempfile.mkdtemp())
    
    if MATPLOTLIB_AVAILABLE:
        print_subheader("Generating Topology Visualization")
        
        # Default topology
        path = visualizer.plot_topology(
            str(output_dir / 'topology_default.png'),
            layout='spring',
            color_by='type'
        )
        print(f"Generated: {path}")
        
        # Colored by criticality
        path = visualizer.plot_topology(
            str(output_dir / 'topology_criticality.png'),
            layout='spring',
            color_by='criticality',
            highlight_critical=True
        )
        print(f"Generated: {path}")
        
        print(f"\nOutput directory: {output_dir}")
    else:
        print("Matplotlib not available - skipping image generation")
    
    return output_dir


def example_2_criticality_heatmap():
    """Example 2: Criticality heatmap visualization"""
    print_header("Example 2: Criticality Heatmap")
    
    # Setup
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    analysis_result = analyzer.analyze()
    
    # Create visualizer
    visualizer = GraphVisualizer(analyzer)
    visualizer.set_analysis_result(analysis_result)
    
    # Create output directory
    output_dir = Path(tempfile.mkdtemp())
    
    if MATPLOTLIB_AVAILABLE:
        print_subheader("Generating Criticality Heatmap")
        
        path = visualizer.plot_criticality_heatmap(
            str(output_dir / 'criticality.png'),
            top_n=10
        )
        print(f"Generated: {path}")
        
        # Show top critical components
        print_subheader("Top Critical Components")
        top_critical = sorted(
            analysis_result.criticality_scores,
            key=lambda x: x.composite_score,
            reverse=True
        )[:5]
        
        for score in top_critical:
            print(f"  {score.node_id}: {score.composite_score:.4f} ({score.level.value})")
    else:
        print("Matplotlib not available - skipping image generation")
    
    return output_dir


def example_3_full_workflow():
    """Example 3: Full workflow with analysis, simulation, and validation"""
    print_header("Example 3: Full Workflow")
    
    # Step 1: Analysis
    print_subheader("Step 1: Run Analysis")
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    analysis_result = analyzer.analyze()
    print(f"Analyzed {analyzer.G.number_of_nodes()} components")
    
    # Step 2: Simulation
    print_subheader("Step 2: Run Simulation")
    simulator = GraphSimulator(seed=42)
    simulation_result = simulator.simulate_all_single_failures(analyzer.G)
    print(f"Ran {simulation_result.total_simulations} simulations")
    
    # Step 3: Validation
    print_subheader("Step 3: Run Validation")
    validator = GraphValidator(analyzer, seed=42)
    validation_result = validator.validate()
    print(f"Validation status: {validation_result.status.value}")
    
    # Step 4: Visualize
    print_subheader("Step 4: Generate Visualizations")
    
    visualizer = GraphVisualizer(analyzer)
    visualizer.set_analysis_result(analysis_result)
    visualizer.set_simulation_result(simulation_result)
    visualizer.set_validation_result(validation_result)
    
    output_dir = Path(tempfile.mkdtemp())
    
    if MATPLOTLIB_AVAILABLE:
        # Topology
        path = visualizer.plot_topology(str(output_dir / 'topology.png'))
        print(f"Generated: topology.png")
        
        # Criticality
        path = visualizer.plot_criticality_heatmap(str(output_dir / 'criticality.png'))
        print(f"Generated: criticality.png")
        
        # Impact comparison
        path = visualizer.plot_impact_comparison(str(output_dir / 'comparison.png'))
        print(f"Generated: comparison.png")
        
        # Validation scatter
        path = visualizer.plot_validation_scatter(str(output_dir / 'scatter.png'))
        print(f"Generated: scatter.png")
        
        # Impact distribution
        path = visualizer.plot_impact_distribution(str(output_dir / 'distribution.png'))
        print(f"Generated: distribution.png")
    
    # HTML report (always available)
    path = visualizer.generate_html_report(str(output_dir / 'report.html'))
    print(f"Generated: report.html")
    
    print(f"\nOutput directory: {output_dir}")
    
    return output_dir


def example_4_custom_styling():
    """Example 4: Custom styling options"""
    print_header("Example 4: Custom Styling")
    
    # Setup
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    analysis_result = analyzer.analyze()
    
    # Custom configuration
    config = VisualizationConfig(
        figsize=(16, 12),
        dpi=200,
        node_size=1200,
        font_size=12,
        edge_width=2.0,
        show_labels=True,
        show_legend=True
    )
    
    # Create visualizer with custom config
    visualizer = GraphVisualizer(analyzer, config)
    visualizer.set_analysis_result(analysis_result)
    
    output_dir = Path(tempfile.mkdtemp())
    
    if MATPLOTLIB_AVAILABLE:
        print_subheader("Different Layout Algorithms")
        
        layouts = ['spring', 'circular', 'shell', 'kamada_kawai']
        for layout in layouts:
            path = visualizer.plot_topology(
                str(output_dir / f'topology_{layout}.png'),
                layout=layout
            )
            print(f"Generated: topology_{layout}.png")
    else:
        print("Matplotlib not available - skipping image generation")
    
    return output_dir


def example_5_html_report():
    """Example 5: HTML report generation"""
    print_header("Example 5: HTML Report Generation")
    
    # Setup
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    analysis_result = analyzer.analyze()
    
    simulator = GraphSimulator(seed=42)
    simulation_result = simulator.simulate_all_single_failures(analyzer.G)
    
    validator = GraphValidator(analyzer, seed=42)
    validation_result = validator.validate()
    
    # Create visualizer
    visualizer = GraphVisualizer(analyzer)
    visualizer.set_analysis_result(analysis_result)
    visualizer.set_simulation_result(simulation_result)
    visualizer.set_validation_result(validation_result)
    
    output_dir = Path(tempfile.mkdtemp())
    
    print_subheader("Generating HTML Report")
    
    # With embedded images (if matplotlib available)
    path = visualizer.generate_html_report(
        str(output_dir / 'report_full.html'),
        title="IoT System Analysis Report",
        include_images=True
    )
    print(f"Generated: report_full.html (with images)")
    
    # Without embedded images (smaller file)
    path = visualizer.generate_html_report(
        str(output_dir / 'report_minimal.html'),
        title="IoT System Analysis Report",
        include_images=False
    )
    print(f"Generated: report_minimal.html (no images)")
    
    print(f"\nOutput directory: {output_dir}")
    print("\nOpen report_full.html in a browser to view the interactive report!")
    
    return output_dir


def example_6_export_for_tools():
    """Example 6: Exporting graph data for external tools"""
    print_header("Example 6: Export for External Tools")
    
    # Setup
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    analysis_result = analyzer.analyze()
    
    visualizer = GraphVisualizer(analyzer)
    visualizer.set_analysis_result(analysis_result)
    
    output_dir = Path(tempfile.mkdtemp())
    
    print_subheader("Exporting Graph Data")
    
    # JSON (for D3.js, web visualization)
    path = visualizer.export_graph_data(
        str(output_dir / 'graph.json'),
        format='json'
    )
    print(f"Generated: graph.json (for D3.js)")
    
    print(f"\nOutput directory: {output_dir}")
    
    # Show JSON structure
    print_subheader("JSON Export Structure")
    import json
    with open(output_dir / 'graph.json') as f:
        data = json.load(f)
    
    print(f"  Nodes: {len(data['nodes'])}")
    print(f"  Links: {len(data['links'])}")
    print(f"\n  Sample node: {json.dumps(data['nodes'][0], indent=4)}")
    
    return output_dir


def example_7_convenience_function():
    """Example 7: Using the visualize_system convenience function"""
    print_header("Example 7: Convenience Function")
    
    # Setup
    analyzer = GraphAnalyzer()
    analyzer.load_from_dict(SAMPLE_PUBSUB_DATA)
    analysis_result = analyzer.analyze()
    
    simulator = GraphSimulator(seed=42)
    simulation_result = simulator.simulate_all_single_failures(analyzer.G)
    
    validator = GraphValidator(analyzer, seed=42)
    validation_result = validator.validate()
    
    output_dir = Path(tempfile.mkdtemp())
    
    print_subheader("Using visualize_system()")
    
    # One function call generates all visualizations
    files = visualize_system(
        analyzer,
        str(output_dir),
        simulation_result=simulation_result,
        validation_result=validation_result
    )
    
    print(f"\nGenerated {len(files)} files:")
    for f in files:
        print(f"  - {Path(f).name}")
    
    print(f"\nOutput directory: {output_dir}")
    
    return output_dir


def example_color_scheme():
    """Bonus: Demonstrate color scheme"""
    print_header("Bonus: Color Scheme Reference")
    
    print_subheader("Component Type Colors")
    for type_name, color in ColorScheme.TYPE_COLORS.items():
        print(f"  {type_name}: {color}")
    
    print_subheader("Criticality Level Colors")
    for level, color in ColorScheme.CRITICALITY_COLORS.items():
        print(f"  {level}: {color}")
    
    print_subheader("Dependency Type Colors")
    for dep_type, color in ColorScheme.DEPENDENCY_COLORS.items():
        print(f"  {dep_type}: {color}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print(" GRAPH VISUALIZER EXAMPLES")
    print("=" * 60)
    
    print(f"\nMatplotlib available: {MATPLOTLIB_AVAILABLE}")
    if not MATPLOTLIB_AVAILABLE:
        print("Install matplotlib for full visualization support: pip install matplotlib")
    
    # Run examples
    example_1_basic_topology()
    example_2_criticality_heatmap()
    example_3_full_workflow()
    example_4_custom_styling()
    example_5_html_report()
    example_6_export_for_tools()
    example_7_convenience_function()
    example_color_scheme()
    
    print("\n" + "=" * 60)
    print(" ALL EXAMPLES COMPLETED")
    print("=" * 60)
    print()


if __name__ == '__main__':
    main()