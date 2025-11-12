#!/usr/bin/env python3
"""
Quick Graph Visualization

Generates a simple visualization of the analyzed graph using matplotlib.
Shows the multi-layer structure with colored nodes by layer.
"""

import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import json

def visualize_graph():
    """Create a visualization of the demo graph"""
    
    # Load the GraphML file
    graphml_path = "/tmp/demo_results/system_graph.graphml"
    
    if not Path(graphml_path).exists():
        print("❌ GraphML file not found. Please run standalone_demo.py first.")
        return
    
    print("Loading graph from GraphML...")
    G = nx.read_graphml(graphml_path)
    
    # Define colors for different layers
    layer_colors = {
        'application': '#FF6B6B',    # Red
        'topic': '#4ECDC4',          # Teal
        'broker': '#FFE66D',         # Yellow
        'infrastructure': '#95E1D3'  # Mint
    }
    
    # Get node colors based on layer
    node_colors = []
    for node in G.nodes():
        layer = G.nodes[node].get('layer', 'unknown')
        node_colors.append(layer_colors.get(layer, '#CCCCCC'))
    
    # Load criticality scores
    json_path = "/tmp/demo_results/analysis_results.json"
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Get node sizes based on betweenness centrality
    betweenness = results['structural_analysis']['betweenness']
    node_sizes = []
    for node in G.nodes():
        bc = betweenness.get(node, 0)
        # Scale sizes between 300 and 2000
        size = 300 + (bc * 10000)
        node_sizes.append(size)
    
    # Create figure
    plt.figure(figsize=(20, 14))
    
    # Use spring layout for positioning
    print("Computing graph layout...")
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw the graph
    print("Drawing graph...")
    
    # Draw edges first
    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True,
                          arrowsize=20, width=1.5,
                          edge_color='#888888')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.9,
                          linewidths=2,
                          edgecolors='#333333')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, 
                           font_size=8,
                           font_weight='bold',
                           font_family='sans-serif')
    
    # Add title and legend
    plt.title("Smart City IoT System - Multi-Layer Graph\n"
             "Node size = Betweenness Centrality | Color = Layer",
             fontsize=16, fontweight='bold', pad=20)
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=layer_colors['application'], 
                  markersize=15, label='Applications'),
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=layer_colors['topic'], 
                  markersize=15, label='Topics'),
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=layer_colors['broker'], 
                  markersize=15, label='Brokers'),
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor=layer_colors['infrastructure'], 
                  markersize=15, label='Infrastructure')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', 
              fontsize=12, framealpha=0.9)
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    output_path = "/tmp/demo_results/graph_visualization.png"
    print(f"Saving visualization to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    
    print(f"✓ Visualization saved: {output_path}")
    print(f"  Image size: {plt.gcf().get_size_inches()} inches")
    print(f"  DPI: 300")
    
    # Also create a criticality-focused view
    print("\nCreating criticality-focused visualization...")
    plt.figure(figsize=(20, 14))
    
    # Load criticality scores
    csv_path = "/tmp/demo_results/criticality_ranking.csv"
    criticality_scores = {}
    with open(csv_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    criticality_scores[parts[0]] = float(parts[1])
    
    # Color nodes by criticality score (red = high, blue = low)
    node_colors_crit = []
    for node in G.nodes():
        score = criticality_scores.get(node, 0)
        # Red for high criticality, blue for low
        if score > 0.4:
            node_colors_crit.append('#FF0000')  # Red
        elif score > 0.3:
            node_colors_crit.append('#FF6B6B')  # Light red
        elif score > 0.2:
            node_colors_crit.append('#FFB84D')  # Orange
        elif score > 0.1:
            node_colors_crit.append('#FFDE59')  # Yellow
        else:
            node_colors_crit.append('#90CAF9')  # Blue
    
    # Draw
    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True,
                          arrowsize=20, width=1.5,
                          edge_color='#888888')
    
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors_crit,
                          node_size=node_sizes,
                          alpha=0.9,
                          linewidths=2,
                          edgecolors='#333333')
    
    nx.draw_networkx_labels(G, pos, 
                           font_size=8,
                           font_weight='bold',
                           font_family='sans-serif')
    
    plt.title("Smart City IoT System - Criticality View\n"
             "Color intensity = Criticality Score | Size = Betweenness",
             fontsize=16, fontweight='bold', pad=20)
    
    # Criticality legend
    crit_legend = [
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='#FF0000', 
                  markersize=15, label='Very High (>0.4)'),
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='#FF6B6B', 
                  markersize=15, label='High (0.3-0.4)'),
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='#FFB84D', 
                  markersize=15, label='Medium (0.2-0.3)'),
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='#FFDE59', 
                  markersize=15, label='Low (0.1-0.2)'),
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='#90CAF9', 
                  markersize=15, label='Very Low (<0.1)')
    ]
    
    plt.legend(handles=crit_legend, loc='upper right', 
              fontsize=12, framealpha=0.9)
    
    plt.axis('off')
    plt.tight_layout()
    
    output_path_crit = "/tmp/demo_results/criticality_visualization.png"
    print(f"Saving criticality visualization to {output_path_crit}...")
    plt.savefig(output_path_crit, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    print(f"✓ Criticality visualization saved: {output_path_crit}")
    print("\n✓ Both visualizations completed successfully!")
    print("\nOpen these files to see the graph structure:")
    print(f"  1. {output_path}")
    print(f"  2. {output_path_crit}")


if __name__ == "__main__":
    print("="*80)
    print("Graph Visualization Generator".center(80))
    print("="*80)
    print()
    
    try:
        visualize_graph()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have matplotlib installed:")
        print("  pip install matplotlib")
        import traceback
        traceback.print_exc()
