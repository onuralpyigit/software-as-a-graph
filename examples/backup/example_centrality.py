"""
Example: Centrality Analyzer Usage

Demonstrates how to:
1. Compute various centrality metrics
2. Identify key nodes (hubs, bottlenecks, core, influential)
3. Compare centrality scores for specific nodes
4. Generate comprehensive centrality reports
5. Compute composite centrality scores
"""

import sys
from pathlib import Path
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent / '..'))

from src.analysis.centrality_analyzer import (
    CentralityAnalyzer, CentralityType
)


def create_example_network():
    """Create an example network for demonstration"""
    
    G = nx.DiGraph()
    
    # Create a network with clear structural properties:
    # - Hub nodes (high degree)
    # - Bottleneck nodes (high betweenness)
    # - Core nodes (high closeness)
    
    # Add nodes
    nodes = [
        # Frontend applications
        "WebUI", "MobileApp", "AdminPortal",
        
        # API Gateway (bottleneck)
        "APIGateway",
        
        # Core services (central)
        "AuthService", "UserService", "OrderService",
        
        # Data services
        "PaymentService", "InventoryService", "NotificationService",
        
        # Message brokers (hubs)
        "MainBroker", "EventBroker",
        
        # Data stores
        "UserDB", "OrderDB", "InventoryDB",
        
        # External services
        "EmailService", "SMSService", "PaymentGateway"
    ]
    
    G.add_nodes_from(nodes)
    
    # Add edges to create structure
    edges = [
        # Frontend -> API Gateway (bottleneck)
        ("WebUI", "APIGateway", 10),
        ("MobileApp", "APIGateway", 8),
        ("AdminPortal", "APIGateway", 5),
        
        # API Gateway -> Core Services
        ("APIGateway", "AuthService", 15),
        ("APIGateway", "UserService", 12),
        ("APIGateway", "OrderService", 10),
        
        # Core services interconnections
        ("AuthService", "UserService", 8),
        ("UserService", "OrderService", 6),
        ("OrderService", "PaymentService", 10),
        ("OrderService", "InventoryService", 8),
        
        # Services -> Message Brokers (hubs)
        ("UserService", "MainBroker", 5),
        ("OrderService", "MainBroker", 12),
        ("PaymentService", "MainBroker", 8),
        ("InventoryService", "MainBroker", 6),
        
        ("OrderService", "EventBroker", 10),
        ("NotificationService", "EventBroker", 5),
        
        # Message Brokers -> Subscribers
        ("MainBroker", "NotificationService", 8),
        ("MainBroker", "InventoryService", 6),
        ("MainBroker", "OrderService", 4),
        
        ("EventBroker", "NotificationService", 7),
        ("EventBroker", "InventoryService", 5),
        
        # Services -> Databases
        ("UserService", "UserDB", 15),
        ("OrderService", "OrderDB", 20),
        ("InventoryService", "InventoryDB", 12),
        
        # Notification -> External
        ("NotificationService", "EmailService", 8),
        ("NotificationService", "SMSService", 6),
        
        # Payment -> External
        ("PaymentService", "PaymentGateway", 10),
    ]
    
    G.add_weighted_edges_from(edges)
    
    return G


def example_1_basic_centrality():
    """Example 1: Compute basic centrality metrics"""
    
    print("\n" + "=" * 70)
    print("Example 1: Basic Centrality Metrics")
    print("=" * 70)
    
    # Create network
    print("\n[Step 1] Creating example network...")
    G = create_example_network()
    print(f"  ✓ Created network with {len(G)} nodes and {len(G.edges())} edges")
    
    # Initialize analyzer
    analyzer = CentralityAnalyzer()
    
    # Compute degree centrality
    print("\n[Step 2] Computing degree centrality...")
    degree_result = analyzer.compute_degree_centrality(G)
    
    print(f"\n  Top 5 nodes by degree centrality:")
    for i, node in enumerate(degree_result.top_nodes[:5], 1):
        score_obj = degree_result.scores[node]
        print(f"    {i}. {node:20s} - Score: {score_obj.score:6.3f}, "
              f"Normalized: {score_obj.normalized_score:6.3f}, "
              f"Percentile: {score_obj.percentile:5.1f}%")
    
    # Compute betweenness centrality
    print("\n[Step 3] Computing betweenness centrality...")
    betweenness_result = analyzer.compute_betweenness_centrality(G)
    
    print(f"\n  Top 5 nodes by betweenness centrality:")
    for i, node in enumerate(betweenness_result.top_nodes[:5], 1):
        score_obj = betweenness_result.scores[node]
        print(f"    {i}. {node:20s} - Score: {score_obj.score:6.3f}, "
              f"Normalized: {score_obj.normalized_score:6.3f}, "
              f"Percentile: {score_obj.percentile:5.1f}%")
    
    # Compute closeness centrality
    print("\n[Step 4] Computing closeness centrality...")
    closeness_result = analyzer.compute_closeness_centrality(G)
    
    print(f"\n  Top 5 nodes by closeness centrality:")
    for i, node in enumerate(closeness_result.top_nodes[:5], 1):
        score_obj = closeness_result.scores[node]
        print(f"    {i}. {node:20s} - Score: {score_obj.score:6.3f}, "
              f"Normalized: {score_obj.normalized_score:6.3f}, "
              f"Percentile: {score_obj.percentile:5.1f}%")


def example_2_pagerank():
    """Example 2: PageRank and eigenvector centrality"""
    
    print("\n" + "=" * 70)
    print("Example 2: PageRank and Eigenvector Centrality")
    print("=" * 70)
    
    G = create_example_network()
    analyzer = CentralityAnalyzer()
    
    # Compute PageRank
    print("\n[Step 1] Computing PageRank...")
    pagerank_result = analyzer.compute_pagerank(G, alpha=0.85)
    
    print(f"\n  Top 5 nodes by PageRank:")
    for i, node in enumerate(pagerank_result.top_nodes[:5], 1):
        score_obj = pagerank_result.scores[node]
        print(f"    {i}. {node:20s} - Score: {score_obj.score:6.4f}, "
              f"Percentile: {score_obj.percentile:5.1f}%")
    
    print(f"\n  Statistics:")
    stats = pagerank_result.statistics
    print(f"    Mean:   {stats['mean']:8.6f}")
    print(f"    Median: {stats['median']:8.6f}")
    print(f"    Std:    {stats['std']:8.6f}")
    print(f"    Min:    {stats['min']:8.6f}")
    print(f"    Max:    {stats['max']:8.6f}")
    
    # Try eigenvector centrality
    print("\n[Step 2] Computing eigenvector centrality...")
    try:
        eigenvector_result = analyzer.compute_eigenvector_centrality(G)
        
        print(f"\n  Top 5 nodes by eigenvector centrality:")
        for i, node in enumerate(eigenvector_result.top_nodes[:5], 1):
            score_obj = eigenvector_result.scores[node]
            print(f"    {i}. {node:20s} - Score: {score_obj.score:6.4f}, "
                  f"Percentile: {score_obj.percentile:5.1f}%")
    except Exception as e:
        print(f"  ⚠️  Eigenvector centrality computation failed: {e}")


def example_3_identify_key_nodes():
    """Example 3: Identify hubs, bottlenecks, and core nodes"""
    
    print("\n" + "=" * 70)
    print("Example 3: Identify Key Node Types")
    print("=" * 70)
    
    G = create_example_network()
    analyzer = CentralityAnalyzer()
    
    # Identify hubs (high degree)
    print("\n[Step 1] Identifying hub nodes (high degree)...")
    hubs = analyzer.identify_hubs(G, threshold=0.7)
    print(f"  ✓ Found {len(hubs)} hub nodes:")
    for hub in hubs:
        degree = G.degree(hub)
        print(f"    - {hub:20s} (degree: {degree})")
    
    # Identify bottlenecks (high betweenness)
    print("\n[Step 2] Identifying bottleneck nodes (high betweenness)...")
    bottlenecks = analyzer.identify_bottlenecks(G, threshold=0.7)
    print(f"  ✓ Found {len(bottlenecks)} bottleneck nodes:")
    for bottleneck in bottlenecks[:5]:
        print(f"    - {bottleneck}")
    
    # Identify core nodes (high closeness)
    print("\n[Step 3] Identifying core nodes (high closeness)...")
    core_nodes = analyzer.identify_core_nodes(G, threshold=0.7)
    print(f"  ✓ Found {len(core_nodes)} core nodes:")
    for core in core_nodes[:5]:
        print(f"    - {core}")
    
    # Identify influential nodes (high PageRank)
    print("\n[Step 4] Identifying influential nodes (high PageRank)...")
    influential = analyzer.identify_influential_nodes(G, threshold=0.7)
    print(f"  ✓ Found {len(influential)} influential nodes:")
    for node in influential[:5]:
        print(f"    - {node}")


def example_4_compare_node_centralities():
    """Example 4: Compare different centrality scores for specific nodes"""
    
    print("\n" + "=" * 70)
    print("Example 4: Compare Centrality Scores for Specific Nodes")
    print("=" * 70)
    
    G = create_example_network()
    analyzer = CentralityAnalyzer()
    
    # Select interesting nodes to compare
    nodes_to_compare = ["APIGateway", "MainBroker", "OrderService", "WebUI"]
    
    print("\n[Comparing centrality scores for key nodes]")
    
    for node in nodes_to_compare:
        print(f"\n  Node: {node}")
        print("  " + "-" * 60)
        
        comparison = analyzer.compare_centralities(
            G, 
            node,
            centrality_types=[
                CentralityType.DEGREE,
                CentralityType.BETWEENNESS,
                CentralityType.CLOSENESS,
                CentralityType.PAGERANK
            ]
        )
        
        for ctype, score in comparison.items():
            print(f"    {ctype.value:20s}: {score:6.4f}")


def example_5_composite_centrality():
    """Example 5: Compute composite centrality scores"""
    
    print("\n" + "=" * 70)
    print("Example 5: Composite Centrality Scores")
    print("=" * 70)
    
    G = create_example_network()
    analyzer = CentralityAnalyzer()
    
    # Compute composite centrality with default weights
    print("\n[Step 1] Computing composite centrality (default weights)...")
    print("  Weights: Degree=0.2, Betweenness=0.3, Closeness=0.2, PageRank=0.3")
    
    composite = analyzer.compute_composite_centrality(G)
    
    # Sort and display top nodes
    sorted_composite = sorted(composite.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n  Top 10 nodes by composite centrality:")
    for i, (node, score) in enumerate(sorted_composite[:10], 1):
        print(f"    {i:2d}. {node:20s} - Score: {score:6.4f}")
    
    # Try custom weights emphasizing betweenness (for finding bottlenecks)
    print("\n[Step 2] Computing composite centrality (bottleneck-focused)...")
    print("  Weights: Degree=0.1, Betweenness=0.6, Closeness=0.1, PageRank=0.2")
    
    custom_composite = analyzer.compute_composite_centrality(
        G,
        weights={
            CentralityType.DEGREE: 0.1,
            CentralityType.BETWEENNESS: 0.6,
            CentralityType.CLOSENESS: 0.1,
            CentralityType.PAGERANK: 0.2
        }
    )
    
    sorted_custom = sorted(custom_composite.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n  Top 10 nodes (bottleneck-focused):")
    for i, (node, score) in enumerate(sorted_custom[:10], 1):
        print(f"    {i:2d}. {node:20s} - Score: {score:6.4f}")


def example_6_all_centralities():
    """Example 6: Compute all centrality metrics at once"""
    
    print("\n" + "=" * 70)
    print("Example 6: Compute All Centrality Metrics")
    print("=" * 70)
    
    G = create_example_network()
    analyzer = CentralityAnalyzer()
    
    # Compute all centralities (fast metrics only)
    print("\n[Step 1] Computing all centrality metrics (fast only)...")
    results = analyzer.analyze_all_centralities(G, include_expensive=False)
    
    print(f"  ✓ Computed {len(results)} centrality metrics")
    
    print("\n  Summary of metrics computed:")
    for ctype, result in results.items():
        top_node = result.top_nodes[0] if result.top_nodes else "N/A"
        print(f"    {ctype.value:30s}: Top node = {top_node}")


def example_7_comprehensive_report():
    """Example 7: Generate comprehensive centrality report"""
    
    print("\n" + "=" * 70)
    print("Example 7: Comprehensive Centrality Report")
    print("=" * 70)
    
    G = create_example_network()
    analyzer = CentralityAnalyzer()
    
    # Generate report
    print("\n[Generating comprehensive centrality report]...")
    report = analyzer.generate_centrality_report(G, include_all=False)
    
    # Display summary
    print("\n  Report Summary:")
    print(f"    Total Nodes:       {report['summary']['total_nodes']}")
    print(f"    Total Edges:       {report['summary']['total_edges']}")
    print(f"    Metrics Computed:  {report['summary']['metrics_computed']}")
    
    # Display key nodes
    print("\n  Key Node Categories:")
    for category, nodes in report['key_nodes'].items():
        print(f"\n    {category.capitalize()}:")
        for node in nodes[:5]:
            print(f"      - {node}")
    
    # Display composite centrality
    if 'composite_centrality' in report:
        print("\n  Top 5 by Composite Centrality:")
        for node in report['composite_centrality']['top_nodes'][:5]:
            score = report['composite_centrality']['scores'][node]
            print(f"    - {node:20s}: {score:6.4f}")


def example_8_top_nodes_comparison():
    """Example 8: Compare top nodes across different metrics"""
    
    print("\n" + "=" * 70)
    print("Example 8: Compare Top Nodes Across Metrics")
    print("=" * 70)
    
    G = create_example_network()
    analyzer = CentralityAnalyzer()
    
    metrics = [
        CentralityType.DEGREE,
        CentralityType.BETWEENNESS,
        CentralityType.CLOSENESS,
        CentralityType.PAGERANK
    ]
    
    print("\n[Comparing top 5 nodes across different metrics]\n")
    
    # Create a comparison table
    comparison = {}
    for metric in metrics:
        top_nodes = analyzer.get_top_nodes(G, metric, n=5)
        comparison[metric] = top_nodes
    
    # Display comparison
    print(f"{'Rank':<6}", end="")
    for metric in metrics:
        print(f"{metric.value:<20}", end="")
    print()
    print("-" * 86)
    
    for rank in range(5):
        print(f"{rank+1:<6}", end="")
        for metric in metrics:
            node, score = comparison[metric][rank]
            print(f"{node:<20}", end="")
        print()
    
    print("\n  Observations:")
    print("    - Degree: Identifies nodes with most connections")
    print("    - Betweenness: Identifies nodes on many shortest paths (bottlenecks)")
    print("    - Closeness: Identifies nodes close to all others (core)")
    print("    - PageRank: Identifies nodes with important connections")


def main():
    """Run all examples"""
    
    print("\n" + "=" * 70)
    print("CENTRALITY ANALYZER - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    
    try:
        example_1_basic_centrality()
        example_2_pagerank()
        example_3_identify_key_nodes()
        example_4_compare_node_centralities()
        example_5_composite_centrality()
        example_6_all_centralities()
        example_7_comprehensive_report()
        example_8_top_nodes_comparison()
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print("\n📚 Summary of Capabilities:")
        print("  ✓ Degree, in-degree, out-degree centrality")
        print("  ✓ Betweenness centrality (bottleneck detection)")
        print("  ✓ Closeness centrality (core node detection)")
        print("  ✓ Eigenvector centrality & PageRank")
        print("  ✓ Katz, harmonic, load centrality")
        print("  ✓ Current flow metrics")
        print("  ✓ Composite centrality scores")
        print("  ✓ Key node identification (hubs, bottlenecks, core, influential)")
        print("  ✓ Comprehensive reporting")
        
        print("\n📖 Usage in Your Code:")
        print("""
from src.analysis.centrality_analyzer import CentralityAnalyzer, CentralityType

analyzer = CentralityAnalyzer()

# Compute specific centrality
degree_result = analyzer.compute_degree_centrality(graph)
betweenness_result = analyzer.compute_betweenness_centrality(graph)

# Identify key nodes
hubs = analyzer.identify_hubs(graph, threshold=0.8)
bottlenecks = analyzer.identify_bottlenecks(graph, threshold=0.8)

# Generate comprehensive report
report = analyzer.generate_centrality_report(graph)
        """)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
