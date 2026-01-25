"""
Simulation Display Module

Terminal display formatting for simulation results.
Reuses Colors and formatting utilities from analysis.display.

Provides:
    - Display functions for event simulation results
    - Display functions for failure simulation results
    - Display functions for exhaustive analysis and reports
"""

from __future__ import annotations
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .event_simulator import EventResult
    from .failure_simulator import FailureResult
    from .simulator import SimulationReport

# Reuse shared display utilities from analysis module
from ..analysis.display import (
    Colors,
    colored,
    print_header,
    print_subheader,
)


# =============================================================================
# Simulation-Specific Utilities
# =============================================================================

def level_color(level: str) -> str:
    """Get color for criticality level (simulation-specific)."""
    return {
        "critical": Colors.RED,
        "high": Colors.YELLOW,
        "medium": Colors.BLUE,
        "low": Colors.WHITE,
        "minimal": Colors.GRAY,
    }.get(level.lower(), Colors.RESET)


# =============================================================================
# Event Simulation Display
# =============================================================================

def display_event_result(result: "EventResult") -> None:
    """Display event simulation results."""
    # Helpers for name resolution
    names = result.component_names
    def get_name(comp_id: str) -> str:
        return f"{comp_id} ({names.get(comp_id, comp_id)})" if comp_id in names else comp_id

    print_header(f"Event Simulation: {get_name(result.source_app)}")
    
    print(f"\n  {colored('Scenario:', Colors.CYAN)}     {result.scenario}")
    print(f"  {colored('Duration:', Colors.CYAN)}     {result.duration * 1000:.2f} ms")
    
    # Throughput metrics
    print_subheader("Throughput Metrics")
    m = result.metrics
    print(f"\n  Messages Published:  {m.messages_published}")
    print(f"  Messages Delivered:  {colored(str(m.messages_delivered), Colors.GREEN)}")
    
    drop_color = Colors.RED if m.messages_dropped > 0 else Colors.GRAY
    print(f"  Messages Dropped:    {colored(str(m.messages_dropped), drop_color)}")
    print(f"  Delivery Rate:       {m.delivery_rate:.1f}%")
    print(f"  Drop Rate:           {m.drop_rate:.1f}%")
    
    # Latency metrics
    print_subheader("Latency Metrics")
    print(f"\n  Average Latency:     {m.avg_latency * 1000:.3f} ms")
    
    if m.min_latency != float('inf'):
        print(f"  Min Latency:         {m.min_latency * 1000:.3f} ms")
    else:
        print(f"  Min Latency:         N/A")
    
    print(f"  Max Latency:         {m.max_latency * 1000:.3f} ms")
    print(f"  P50 Latency:         {m.p50_latency * 1000:.3f} ms")
    print(f"  P99 Latency:         {m.p99_latency * 1000:.3f} ms")
    print(f"  Throughput:          {m.throughput:.1f} msg/sec")
    
    # Path analysis
    if result.affected_topics:
        print_subheader("Path Analysis")
        print(f"\n  Affected Topics:     {', '.join(get_name(t) for t in result.affected_topics)}")
        print(f"  Brokers Used:        {', '.join(get_name(b) for b in result.brokers_used) or 'None'}")
        print(f"  Reached Subscribers: {len(result.reached_subscribers)}")
        if result.reached_subscribers:
            subs = result.reached_subscribers[:5]
            print(f"                       {', '.join(get_name(s) for s in subs)}")
            if len(result.reached_subscribers) > 5:
                print(f"                       ... and {len(result.reached_subscribers) - 5} more")
    
    # Drop reasons
    if result.drop_reasons:
        print_subheader("Drop Analysis")
        for reason, count in result.drop_reasons.items():
            print(f"  {reason}: {count}")
    
    # Top impacted components
    if result.component_impacts:
        print_subheader("Component Impacts (Top 5)")
        sorted_impacts = sorted(
            result.component_impacts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        print(f"\n  {'Component':<35} {'Impact Score':<15}")
        print(f"  {'-' * 50}")
        for comp_id, impact in sorted_impacts:
            print(f"  {get_name(comp_id):<35} {impact:.4f}")


# =============================================================================
# Failure Simulation Display
# =============================================================================

def display_failure_result(result: "FailureResult") -> None:
    """Display failure simulation results."""
    # Helpers for name resolution
    names = result.component_names
    def get_name(comp_id: str) -> str:
        return f"{comp_id} ({names.get(comp_id, comp_id)})" if comp_id in names else comp_id

    print_header(f"Failure Simulation: {get_name(result.target_id)}")
    
    print(f"\n  {colored('Target:', Colors.CYAN)}       {get_name(result.target_id)} ({result.target_type})")
    print(f"  {colored('Scenario:', Colors.CYAN)}     {result.scenario}")
    
    # Impact metrics
    print_subheader("Impact Metrics")
    impact = result.impact
    
    # Composite impact with color coding
    imp_color = Colors.RED if impact.composite_impact > 0.5 else (
        Colors.YELLOW if impact.composite_impact > 0.2 else Colors.GREEN
    )
    print(f"\n  {colored('Composite Impact:', Colors.WHITE, bold=True)} {colored(f'{impact.composite_impact:.4f}', imp_color)}")
    
    print(f"\n  Reachability Loss:   {impact.reachability_loss * 100:.1f}%")
    print(f"    Initial Paths:     {impact.initial_paths}")
    print(f"    Remaining Paths:   {impact.remaining_paths}")
    
    print(f"\n  Fragmentation:       {impact.fragmentation * 100:.1f}%")
    print(f"    Initial Components: {impact.initial_components}")
    print(f"    Failed Components:  {impact.failed_components}")
    
    print(f"\n  Throughput Loss:     {impact.throughput_loss * 100:.1f}%")
    print(f"    Affected Topics:   {impact.affected_topics}")
    
    # Affected entities
    print_subheader("Affected Entities")
    print(f"\n  Publishers Affected:  {impact.affected_publishers}")
    print(f"  Subscribers Affected: {impact.affected_subscribers}")
    
    # Cascade analysis
    print_subheader("Cascade Analysis")
    print(f"\n  Cascade Count:       {impact.cascade_count}")
    print(f"  Cascade Depth:       {impact.cascade_depth}")
    
    if impact.cascade_by_type:
        print(f"  Cascade by Type:")
        for comp_type, count in impact.cascade_by_type.items():
            print(f"    {comp_type}: {count}")
    
    # Cascaded failures
    if result.cascaded_failures:
        print_subheader("Cascaded Failures")
        for i, comp_id in enumerate(result.cascaded_failures[:10]):
            print(f"  {i+1}. {get_name(comp_id)}")
        if len(result.cascaded_failures) > 10:
            print(f"  ... and {len(result.cascaded_failures) - 10} more")
    
    # Layer impacts
    if result.layer_impacts:
        print_subheader("Per-Layer Impact")
        print(f"\n  {'Layer':<15} {'Impact':<10}")
        print(f"  {'-' * 25}")
        for layer, layer_impact in result.layer_impacts.items():
            color = Colors.RED if layer_impact > 0.3 else (
                Colors.YELLOW if layer_impact > 0.1 else Colors.GREEN
            )
            print(f"  {layer:<15} {colored(f'{layer_impact:.4f}', color)}")


# =============================================================================
# Exhaustive Analysis Display
# =============================================================================

def display_exhaustive_results(results: List["FailureResult"], limit: int = 15) -> None:
    """Display exhaustive failure simulation results."""
    print_header("Exhaustive Failure Analysis")
    
    print(f"\n  Total Components Analyzed: {len(results)}")
    
    if not results:
        print(f"  {colored('No results.', Colors.GRAY)}")
        return
    
    # Summary statistics
    impacts = [r.impact.composite_impact for r in results]
    avg_impact = sum(impacts) / len(impacts)
    max_impact = max(impacts)
    
    print(f"  Average Impact:           {avg_impact:.4f}")
    print(f"  Maximum Impact:           {max_impact:.4f}")
    
    # Count by impact level
    critical = sum(1 for r in results if r.impact.composite_impact > 0.5)
    high = sum(1 for r in results if 0.3 < r.impact.composite_impact <= 0.5)
    medium = sum(1 for r in results if 0.1 < r.impact.composite_impact <= 0.3)
    low = sum(1 for r in results if r.impact.composite_impact <= 0.1)
    
    print(f"\n  Impact Distribution:")
    print(f"    {colored('CRITICAL (>0.5):', Colors.RED)} {critical}")
    print(f"    {colored('HIGH (0.3-0.5):', Colors.YELLOW)} {high}")
    print(f"    {colored('MEDIUM (0.1-0.3):', Colors.BLUE)} {medium}")
    print(f"    {colored('LOW (≤0.1):', Colors.GREEN)} {low}")
    
    # Top components
    print_subheader(f"Top {limit} Components by Impact")
    
    print(f"\n  {'Component':<30} {'Type':<12} {'Impact':<10} {'Cascade':<10} {'Reach Loss':<10}")
    print(f"  {'-' * 75}")
    
    for r in results[:limit]:
        # Resolve name provided by result's own map (assuming all results share context or first one covers it)
        # Note: display_exhaustive_results takes List[FailureResult], each has its own component_names map.
        # But we can assume the target_id is present in result.component_names.
        r_name = f"{r.target_id} ({r.component_names.get(r.target_id, r.target_id)})"
        
        imp = r.impact.composite_impact
        color = Colors.RED if imp > 0.5 else (
            Colors.YELLOW if imp > 0.2 else Colors.GREEN
        )
        
        print(
            f"  {r_name:<30} {r.target_type:<12} "
            f"{colored(f'{imp:.4f}', color):<10} "
            f"{r.impact.cascade_count:<10} "
            f"{r.impact.reachability_loss*100:.1f}%"
        )
    
    if len(results) > limit:
        print(f"\n  {colored(f'... and {len(results) - limit} more', Colors.GRAY)}")


# =============================================================================
# Report Display
# =============================================================================

def display_report(report: "SimulationReport") -> None:
    """Display comprehensive simulation report."""
    print_header("Simulation Report", "═")
    
    print(f"\n  {colored('Timestamp:', Colors.CYAN)}  {report.timestamp}")
    
    # Graph summary
    print_subheader("Graph Summary")
    gs = report.graph_summary
    print(f"\n  Total Nodes:      {gs.get('total_nodes', 0)}")
    print(f"  Total Edges:      {gs.get('total_edges', 0)}")
    print(f"  Topics:           {gs.get('topics', 0)}")
    print(f"  Pub-Sub Paths:    {gs.get('pub_sub_paths', 0)}")
    
    if gs.get('component_types'):
        types_str = ", ".join(f"{k}: {v}" for k, v in gs['component_types'].items())
        print(f"  Component Types:  {types_str}")
    
    # Layer metrics
    for layer_name, metrics in report.layer_metrics.items():
        print_subheader(f"Layer: {layer_name.upper()}")
        
        print(f"\n  {colored('Event Simulation:', Colors.CYAN)}")
        print(f"    Throughput:        {metrics.event_throughput} messages")
        print(f"    Delivery Rate:     {metrics.event_delivery_rate:.1f}%")
        print(f"    Drop Rate:         {metrics.event_drop_rate:.1f}%")
        print(f"    Avg Latency:       {metrics.event_avg_latency_ms:.3f} ms")
        
        print(f"\n  {colored('Failure Simulation:', Colors.CYAN)}")
        print(f"    Avg Reach Loss:    {metrics.avg_reachability_loss * 100:.1f}%")
        print(f"    Avg Fragmentation: {metrics.avg_fragmentation * 100:.1f}%")
        print(f"    Avg Throughput Loss: {metrics.avg_throughput_loss * 100:.1f}%")
        print(f"    Max Impact:        {metrics.max_impact:.4f}")
        
        print(f"\n  {colored('Criticality:', Colors.CYAN)}")
        print(f"    Total Components:  {metrics.total_components}")
        print(f"    Critical:          {colored(str(metrics.critical_count), Colors.RED)}")
        print(f"    High:              {colored(str(metrics.high_count), Colors.YELLOW)}")
        print(f"    Medium:            {metrics.medium_count}")
        print(f"    SPOFs:             {metrics.spof_count}")
    
    # Top critical components
    if report.top_critical:
        print_subheader("Top Critical Components")
        
        names = report.component_names
        print(f"\n  {'ID':<30} {'Type':<12} {'Level':<10} {'Impact':<10} {'Cascade':<10}")
        print(f"  {'-' * 75}")
        
        for c in report.top_critical:
            cid = c['id']
            c_name = f"{cid} ({names.get(cid, cid)})" if cid in names else cid
            color = level_color(c['level'])
            print(
                f"  {c_name:<30} {c['type']:<12} "
                f"{colored(c['level'], color):<10} "
                f"{c['combined_impact']:.4f}     "
                f"{c['cascade_count']:>3}"
            )
    
    # Recommendations
    if report.recommendations:
        print_subheader("Recommendations")
        for i, rec in enumerate(report.recommendations, 1):
            # Color code by severity keyword
            if "CRITICAL" in rec:
                color = Colors.RED
            elif "HIGH" in rec or "SPOF" in rec:
                color = Colors.YELLOW
            elif "HEALTHY" in rec:
                color = Colors.GREEN
            else:
                color = Colors.WHITE
            
            print(f"\n  {i}. {colored(rec, color)}")
    
    # Final summary
    print_subheader("Summary")
    
    total_critical = sum(m.critical_count for m in report.layer_metrics.values())
    total_high = sum(m.high_count for m in report.layer_metrics.values())
    
    if total_critical > 0:
        print(f"\n  {colored(f'⚠ {total_critical} CRITICAL components require immediate attention', Colors.RED, bold=True)}")
    if total_high > 0:
        print(f"  {colored(f'⚠ {total_high} HIGH priority components should be reviewed', Colors.YELLOW)}")
    if total_critical == 0 and total_high == 0:
        print(f"\n  {colored('✓ No critical issues detected. System looks healthy!', Colors.GREEN, bold=True)}")
    
    print()
