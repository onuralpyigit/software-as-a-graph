#!/usr/bin/env python3
"""
Dependency Analyzer Demo
========================

Demonstrates the multi-layer dependency graph analyzer on pub-sub systems.
This approach applies graph algorithms DIRECTLY rather than computing
a single criticality score.

Benefits:
- More interpretable: Shows WHY something is critical
- Multiple perspectives: Different algorithms reveal different aspects
- Anti-pattern detection: Structural issues identified automatically
- Actionable: Specific recommendations for each finding
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.dependency_analyzer import (
    MultiLayerDependencyAnalyzer,
    ReportGenerator,
    DependencyGraphBuilder,
    GraphAlgorithmAnalyzer,
    AntiPatternDetector,
    CriticalComponentIdentifier
)


# ============================================================================
# Color Helpers
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.HEADER}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {text}{Colors.END}")
    print(f"{Colors.HEADER}{'='*70}{Colors.END}")


def print_section(text: str):
    print(f"\n{Colors.CYAN}▶ {text}{Colors.END}")


# ============================================================================
# Create Comprehensive Sample System
# ============================================================================

def create_comprehensive_system() -> dict:
    """
    Create a comprehensive pub-sub system with various patterns and anti-patterns
    for demonstration purposes.
    """
    return {
        "metadata": {
            "name": "E-Commerce Platform",
            "description": "Microservices-based e-commerce system with pub-sub messaging"
        },
        "nodes": [
            {"id": "N1", "type": "kubernetes_node", "zone": "us-east-1a"},
            {"id": "N2", "type": "kubernetes_node", "zone": "us-east-1b"},
            {"id": "N3", "type": "kubernetes_node", "zone": "us-east-1c"},
            {"id": "N4", "type": "database_server", "zone": "us-east-1a"},
            {"id": "N5", "type": "cache_server", "zone": "us-east-1b"},
        ],
        "brokers": [
            {"id": "B1", "type": "kafka", "node": "N1", "cluster": "main"},
            {"id": "B2", "type": "kafka", "node": "N2", "cluster": "main"},
            {"id": "B3", "type": "rabbitmq", "node": "N3", "cluster": "async"},
        ],
        "topics": [
            # Core business topics
            {"id": "T_orders", "name": "orders.created", "broker": "B1"},
            {"id": "T_payments", "name": "payments.processed", "broker": "B1"},
            {"id": "T_inventory", "name": "inventory.updated", "broker": "B1"},
            {"id": "T_shipping", "name": "shipping.status", "broker": "B2"},
            {"id": "T_notifications", "name": "notifications.send", "broker": "B2"},
            
            # God topic (anti-pattern) - too many connections
            {"id": "T_events", "name": "system.events", "broker": "B1"},
            
            # Analytics topics
            {"id": "T_analytics", "name": "analytics.events", "broker": "B3"},
            {"id": "T_logs", "name": "system.logs", "broker": "B3"},
            
            # Commands
            {"id": "T_commands", "name": "admin.commands", "broker": "B1"},
        ],
        "applications": [
            # Core services
            {"id": "A_orderSvc", "type": "service", "name": "OrderService", "node": "N1"},
            {"id": "A_paymentSvc", "type": "service", "name": "PaymentService", "node": "N1"},
            {"id": "A_inventorySvc", "type": "service", "name": "InventoryService", "node": "N2"},
            {"id": "A_shippingSvc", "type": "service", "name": "ShippingService", "node": "N2"},
            {"id": "A_notifSvc", "type": "service", "name": "NotificationService", "node": "N3"},
            
            # Frontend/Gateway
            {"id": "A_gateway", "type": "gateway", "name": "APIGateway", "node": "N1"},
            {"id": "A_webApp", "type": "frontend", "name": "WebApplication", "node": "N3"},
            {"id": "A_mobileApp", "type": "frontend", "name": "MobileApp", "node": "N3"},
            
            # Analytics & Monitoring
            {"id": "A_analytics", "type": "analytics", "name": "AnalyticsService", "node": "N4"},
            {"id": "A_monitor", "type": "monitoring", "name": "MonitoringService", "node": "N4"},
            
            # Admin
            {"id": "A_admin", "type": "admin", "name": "AdminDashboard", "node": "N5"},
            
            # Workers
            {"id": "A_emailWorker", "type": "worker", "name": "EmailWorker", "node": "N3"},
            {"id": "A_smsWorker", "type": "worker", "name": "SMSWorker", "node": "N3"},
        ],
        "publishes": [
            # Order flow
            {"application": "A_gateway", "topic": "T_orders"},
            {"application": "A_orderSvc", "topic": "T_orders"},
            {"application": "A_orderSvc", "topic": "T_events"},
            {"application": "A_orderSvc", "topic": "T_analytics"},
            
            # Payment flow
            {"application": "A_paymentSvc", "topic": "T_payments"},
            {"application": "A_paymentSvc", "topic": "T_events"},
            {"application": "A_paymentSvc", "topic": "T_analytics"},
            
            # Inventory
            {"application": "A_inventorySvc", "topic": "T_inventory"},
            {"application": "A_inventorySvc", "topic": "T_events"},
            
            # Shipping
            {"application": "A_shippingSvc", "topic": "T_shipping"},
            {"application": "A_shippingSvc", "topic": "T_events"},
            {"application": "A_shippingSvc", "topic": "T_notifications"},
            
            # Notifications
            {"application": "A_notifSvc", "topic": "T_notifications"},
            {"application": "A_notifSvc", "topic": "T_events"},
            
            # Analytics publishes processed data
            {"application": "A_analytics", "topic": "T_analytics"},
            
            # Admin commands
            {"application": "A_admin", "topic": "T_commands"},
            
            # Logging
            {"application": "A_orderSvc", "topic": "T_logs"},
            {"application": "A_paymentSvc", "topic": "T_logs"},
            {"application": "A_inventorySvc", "topic": "T_logs"},
            {"application": "A_shippingSvc", "topic": "T_logs"},
            {"application": "A_notifSvc", "topic": "T_logs"},
        ],
        "subscribes": [
            # Order service listens to payments and inventory
            {"application": "A_orderSvc", "topic": "T_payments"},
            {"application": "A_orderSvc", "topic": "T_inventory"},
            {"application": "A_orderSvc", "topic": "T_commands"},
            
            # Payment service listens to orders
            {"application": "A_paymentSvc", "topic": "T_orders"},
            {"application": "A_paymentSvc", "topic": "T_commands"},
            
            # Inventory listens to orders
            {"application": "A_inventorySvc", "topic": "T_orders"},
            {"application": "A_inventorySvc", "topic": "T_commands"},
            
            # Shipping listens to orders and payments
            {"application": "A_shippingSvc", "topic": "T_orders"},
            {"application": "A_shippingSvc", "topic": "T_payments"},
            {"application": "A_shippingSvc", "topic": "T_commands"},
            
            # Notification service listens to everything
            {"application": "A_notifSvc", "topic": "T_orders"},
            {"application": "A_notifSvc", "topic": "T_payments"},
            {"application": "A_notifSvc", "topic": "T_shipping"},
            
            # Frontend apps listen to events
            {"application": "A_webApp", "topic": "T_events"},
            {"application": "A_webApp", "topic": "T_notifications"},
            {"application": "A_mobileApp", "topic": "T_events"},
            {"application": "A_mobileApp", "topic": "T_notifications"},
            
            # Analytics listens to everything
            {"application": "A_analytics", "topic": "T_orders"},
            {"application": "A_analytics", "topic": "T_payments"},
            {"application": "A_analytics", "topic": "T_inventory"},
            {"application": "A_analytics", "topic": "T_shipping"},
            {"application": "A_analytics", "topic": "T_events"},
            
            # Monitor listens to logs and events
            {"application": "A_monitor", "topic": "T_logs"},
            {"application": "A_monitor", "topic": "T_events"},
            
            # Workers
            {"application": "A_emailWorker", "topic": "T_notifications"},
            {"application": "A_smsWorker", "topic": "T_notifications"},
            
            # Admin dashboard
            {"application": "A_admin", "topic": "T_events"},
            {"application": "A_admin", "topic": "T_analytics"},
        ],
        "routes": [
            {"from": "B1", "to": "B2"},
            {"from": "B2", "to": "B1"},
            {"from": "B1", "to": "B3"},
            {"from": "B2", "to": "B3"},
        ]
    }


# ============================================================================
# Main Demo
# ============================================================================

def main():
    print_header("MULTI-LAYER DEPENDENCY ANALYZER DEMO")
    
    print("""
This demo shows how to analyze pub-sub systems using graph algorithms
DIRECTLY, without relying on a single criticality score.

Approach:
  1. Build multi-layer DEPENDS_ON graph
  2. Apply graph algorithms (Betweenness, PageRank, HITS, etc.)
  3. Identify critical nodes from MULTIPLE perspectives
  4. Detect anti-patterns from graph structure
  5. Generate actionable recommendations
""")
    
    # Create sample system
    print_section("Creating Sample E-Commerce Pub-Sub System")
    system_data = create_comprehensive_system()
    print(f"  Applications: {len(system_data['applications'])}")
    print(f"  Topics: {len(system_data['topics'])}")
    print(f"  Brokers: {len(system_data['brokers'])}")
    print(f"  Nodes: {len(system_data['nodes'])}")
    
    # Run analysis
    print_section("Running Multi-Layer Dependency Analysis")
    analyzer = MultiLayerDependencyAnalyzer()
    result = analyzer.analyze(system_data)
    
    # Print detailed results
    print_section("Graph Metrics")
    gm = result.graph_metrics
    print(f"  Total nodes in dependency graph: {gm['nodes']}")
    print(f"  Total edges (dependencies): {gm['edges']}")
    print(f"  Graph density: {gm['density']:.4f}")
    print(f"  Is connected: {gm['is_connected']}")
    
    print_section("Layer Breakdown")
    for layer, metrics in result.layer_metrics.items():
        print(f"  {layer.upper()}:")
        print(f"    Nodes: {metrics['node_count']}")
        print(f"    Internal edges: {metrics['internal_edges']}")
        print(f"    Cross-layer (in/out): {metrics['incoming_cross_layer']}/{metrics['outgoing_cross_layer']}")
    
    # Critical Nodes
    print_section("Critical Nodes Identified")
    print(f"  Total: {len(result.critical_nodes)}")
    print(f"\n  {Colors.YELLOW}Top Critical Components:{Colors.END}")
    
    for i, node in enumerate(result.critical_nodes[:10], 1):
        reasons = [r.value for r in node.reasons]
        print(f"\n  {i}. {Colors.GREEN}{node.node_id}{Colors.END} ({node.layer}/{node.node_type})")
        print(f"     Reasons: {', '.join(reasons)}")
        print(f"     Impact: {node.impact_description}")
        print(f"     Recommendation: {node.recommendation}")
        if node.metrics:
            metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                   for k, v in list(node.metrics.items())[:3])
            print(f"     Metrics: {metrics_str}")
    
    # Critical Edges
    print_section("Critical Edges (Dependencies) Identified")
    print(f"  Total: {len(result.critical_edges)}")
    
    for edge in result.critical_edges[:5]:
        reasons = [r.value for r in edge.reasons]
        print(f"\n  • {edge.source} → {edge.target}")
        print(f"    Type: {edge.dependency_type}")
        print(f"    Reasons: {', '.join(reasons)}")
        print(f"    Impact: {edge.impact_description}")
    
    # Anti-Patterns
    print_section("Anti-Patterns Detected")
    print(f"  Total: {len(result.anti_patterns)}")
    
    severity_counts = result.summary['anti_patterns_by_severity']
    print(f"  By severity: {Colors.RED}Critical={severity_counts.get('critical', 0)}{Colors.END}, "
          f"{Colors.YELLOW}High={severity_counts.get('high', 0)}{Colors.END}, "
          f"Medium={severity_counts.get('medium', 0)}, Low={severity_counts.get('low', 0)}")
    
    type_counts = result.summary['anti_patterns_by_type']
    print(f"  By type: {dict(type_counts)}")
    
    for pattern in result.anti_patterns:
        severity_color = {
            'critical': Colors.RED,
            'high': Colors.YELLOW,
            'medium': Colors.CYAN,
            'low': Colors.GREEN
        }.get(pattern.severity, '')
        
        print(f"\n  {severity_color}[{pattern.severity.upper()}]{Colors.END} {pattern.pattern_type.value}")
        print(f"    Components: {', '.join(pattern.affected_components[:5])}"
              f"{'...' if len(pattern.affected_components) > 5 else ''}")
        print(f"    Description: {pattern.description}")
        print(f"    Impact: {pattern.impact}")
        print(f"    Recommendation: {pattern.recommendation}")
    
    # Algorithm Results Summary
    print_section("Algorithm Results Summary")
    
    # Betweenness
    bc = result.algorithm_results.get('betweenness', {})
    if bc.get('high_nodes'):
        print(f"\n  Betweenness Centrality (routing importance):")
        for node, score in list(bc['high_nodes'].items())[:5]:
            print(f"    • {node}: {score:.4f}")
    
    # HITS
    hits = result.algorithm_results.get('hits', {})
    if hits.get('high_hubs'):
        print(f"\n  HITS Hubs (key data sources):")
        for node, score in list(hits['high_hubs'].items())[:5]:
            print(f"    • {node}: {score:.4f}")
    
    if hits.get('high_authorities'):
        print(f"\n  HITS Authorities (key data sinks):")
        for node, score in list(hits['high_authorities'].items())[:5]:
            print(f"    • {node}: {score:.4f}")
    
    # Communities
    communities = result.algorithm_results.get('communities', {})
    print(f"\n  Communities detected: {communities.get('num_communities', 0)}")
    print(f"  Modularity: {communities.get('modularity', 0):.4f}")
    print(f"  Cross-community edges: {len(communities.get('cross_community_edges', []))}")
    
    # K-Core
    kcore = result.algorithm_results.get('kcore', {})
    print(f"\n  K-Core Analysis:")
    print(f"    Maximum k: {kcore.get('max_k', 0)}")
    print(f"    Innermost core size: {len(kcore.get('innermost_core', []))}")
    
    # Paths
    paths = result.algorithm_results.get('paths', {})
    print(f"\n  Path Analysis:")
    print(f"    Maximum dependency depth: {paths.get('max_depth', 0)}")
    avg_path = paths.get('average_path_length', 0)
    if avg_path != float('inf'):
        print(f"    Average path length: {avg_path:.2f}")
    
    # Export results
    print_section("Exporting Results")
    output_dir = Path("demo_output/dependency_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON
    analyzer.export_results(result, str(output_dir / "results.json"))
    print(f"  ✓ JSON: {output_dir / 'results.json'}")
    
    # HTML
    html_report = ReportGenerator.generate_html_report(result)
    with open(output_dir / "report.html", "w") as f:
        f.write(html_report)
    print(f"  ✓ HTML: {output_dir / 'report.html'}")
    
    # Text
    text_report = ReportGenerator.generate_text_report(result)
    with open(output_dir / "report.txt", "w") as f:
        f.write(text_report)
    print(f"  ✓ Text: {output_dir / 'report.txt'}")
    
    # Save system definition
    with open(output_dir / "system.json", "w") as f:
        json.dump(system_data, f, indent=2)
    print(f"  ✓ System: {output_dir / 'system.json'}")
    
    # Summary
    print_header("ANALYSIS COMPLETE")
    print(f"""
Key Findings:
  • {len(result.critical_nodes)} critical nodes identified through graph algorithms
  • {len(result.critical_edges)} critical dependencies identified  
  • {len(result.anti_patterns)} anti-patterns detected

This approach provides:
  ✓ INTERPRETABLE results - each finding explains WHY
  ✓ MULTIPLE perspectives - different algorithms reveal different aspects
  ✓ ACTIONABLE insights - specific recommendations for each issue
  ✓ NO arbitrary scores - direct structural analysis

Files exported to: {output_dir}/
""")


if __name__ == "__main__":
    main()
