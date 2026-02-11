"""
Simulation Display Methods

Extension methods for ConsoleDisplay to render simulation results.
These methods are designed to be added to the existing ConsoleDisplay class
in src/cli/display.py.

Usage:
    Copy these methods into the ConsoleDisplay class body, or import
    and monkey-patch if preferred.
"""

from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.services.event_simulator import EventResult
    from src.domain.services.failure_simulator import FailureResult
    from src.domain.models.simulation.metrics import SimulationReport, LayerMetrics


# =============================================================================
# Add these methods to the ConsoleDisplay class
# =============================================================================


def display_event_result(self, result: "EventResult") -> None:
    """Display event simulation results for a single publisher."""
    names = result.component_names
    get_name = lambda cid: f"{cid} ({names.get(cid, cid)})" if cid in names else cid

    self.print_header(f"Event Simulation: {get_name(result.source_app)}")
    print(f"\n  Scenario:     {result.scenario}")
    print(f"  Duration:     {result.duration * 1000:.2f} ms")

    # Throughput
    self.print_subheader("Throughput Metrics")
    m = result.metrics
    print(f"\n  Messages Published:  {m.messages_published}")
    print(f"  Messages Delivered:  {self.colored(str(m.messages_delivered), self.Colors.GREEN)}")
    drop_color = self.Colors.RED if m.messages_dropped > 0 else self.Colors.GRAY
    print(f"  Messages Dropped:    {self.colored(str(m.messages_dropped), drop_color)}")
    print(f"  Delivery Rate:       {m.delivery_rate:.1f}%")
    print(f"  Drop Rate:           {m.drop_rate:.1f}%")

    # Latency
    self.print_subheader("Latency Metrics")
    print(f"\n  Average Latency:     {m.avg_latency * 1000:.3f} ms")
    print(f"  Min Latency:         {m.min_latency * 1000:.3f} ms")
    print(f"  Max Latency:         {m.max_latency * 1000:.3f} ms")
    print(f"  P99 Latency:         {m.p99_latency * 1000:.3f} ms")
    print(f"  Throughput:          {m.throughput:.1f} msg/sec")

    # Topics and subscribers
    if result.affected_topics:
        self.print_subheader("Message Flow")
        print(f"\n  Affected Topics:     {', '.join(result.affected_topics)}")
        print(f"  Reached Subscribers: {', '.join(result.reached_subscribers)}")
        if result.brokers_used:
            print(f"  Brokers Used:        {', '.join(result.brokers_used)}")

    # Drop reasons
    if result.drop_reasons:
        self.print_subheader("Drop Reasons")
        for reason, count in sorted(result.drop_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason:<30} {count}")


def display_failure_result(self, result: "FailureResult") -> None:
    """Display failure simulation results for a single component."""
    names = result.component_names
    get_name = lambda cid: f"{cid} ({names.get(cid, cid)})" if cid in names else cid

    self.print_header(f"Failure Simulation: {get_name(result.target_id)}")
    print(f"\n  Target:       {get_name(result.target_id)} ({result.target_type})")

    if result.related_components:
        related_str = ", ".join(result.related_components)
        if len(related_str) > 80:
            related_str = related_str[:77] + "..."
        print(f"  Related:      {self.colored(related_str, self.Colors.GRAY)}")

    print(f"  Scenario:     {result.scenario}")

    # Impact metrics
    self.print_subheader("Impact Metrics")
    impact = result.impact
    imp_color = (
        self.Colors.RED if impact.composite_impact > 0.5
        else self.Colors.YELLOW if impact.composite_impact > 0.2
        else self.Colors.GREEN
    )
    print(f"\n  {self.colored('Composite Impact:', self.Colors.WHITE, bold=True)} "
          f"{self.colored(f'{impact.composite_impact:.4f}', imp_color)}")
    print(f"\n  Reachability Loss:   {impact.reachability_loss * 100:.1f}%")
    print(f"  Fragmentation:       {impact.fragmentation * 100:.1f}%")
    print(f"  Throughput Loss:     {impact.throughput_loss * 100:.1f}%")

    # Cascade
    self.print_subheader("Cascade Analysis")
    print(f"\n  Cascade Count:       {impact.cascade_count}")
    print(f"  Cascade Depth:       {impact.cascade_depth}")

    if impact.cascade_by_type:
        print(f"  Cascade by Type:")
        for ctype, count in sorted(impact.cascade_by_type.items()):
            print(f"    {ctype:<20} {count}")

    # Per-layer impacts
    if result.layer_impacts:
        self.print_subheader("Per-Layer Impact")
        for layer, val in sorted(result.layer_impacts.items()):
            bar_len = int(val * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            color = (
                self.Colors.RED if val > 0.5
                else self.Colors.YELLOW if val > 0.2
                else self.Colors.GREEN
            )
            print(f"  {layer:<8} {self.colored(bar, color)} {val * 100:.1f}%")

    # Cascaded failures
    if result.cascaded_failures:
        self.print_subheader(f"Cascaded Failures ({len(result.cascaded_failures)})")
        for i, cid in enumerate(result.cascaded_failures[:15], 1):
            print(f"  {i:>3}. {get_name(cid)}")
        if len(result.cascaded_failures) > 15:
            print(f"  ... and {len(result.cascaded_failures) - 15} more")


def display_exhaustive_results(self, results: List["FailureResult"], limit: int = 15) -> None:
    """Display exhaustive failure simulation results."""
    self.print_header("Exhaustive Failure Analysis")

    if not results:
        print("\n  No components analyzed.")
        return

    impacts = [r.impact.composite_impact for r in results]
    print(f"\n  Components Analyzed: {len(results)}")
    print(f"  Average Impact:      {sum(impacts) / len(impacts):.4f}")
    print(f"  Maximum Impact:      {max(impacts):.4f}")
    print(f"  SPOFs Detected:      {sum(1 for r in results if r.impact.cascade_count > 0)}")

    self.print_subheader(f"Top {min(limit, len(results))} Components by Impact")
    print(f"\n  {'#':<4} {'Component':<28} {'Type':<12} {'Impact':<10} {'Cascade':<10} {'Reach Loss':<12}")
    print(f"  {'-' * 76}")

    for i, r in enumerate(results[:limit], 1):
        imp = r.impact.composite_impact
        color = (
            self.Colors.RED if imp > 0.5
            else self.Colors.YELLOW if imp > 0.2
            else self.Colors.GREEN
        )
        name = r.component_names.get(r.target_id, r.target_id)
        label = f"{r.target_id}" if name == r.target_id else f"{name}"
        print(
            f"  {i:<4} {label:<28} {r.target_type:<12} "
            f"{self.colored(f'{imp:.4f}', color):<10} "
            f"{r.impact.cascade_count:<10} "
            f"{r.impact.reachability_loss * 100:.1f}%"
        )

        if r.related_components:
            related_str = ", ".join(r.related_components[:3])
            if len(r.related_components) > 3:
                related_str += f" (+{len(r.related_components) - 3})"
            print(f"       {self.colored('↳ ' + related_str, self.Colors.GRAY)}")


def display_simulation_report(self, report: "SimulationReport") -> None:
    """Display comprehensive simulation report."""
    self.print_header("Simulation Report", "═")
    print(f"\n  Timestamp:    {report.timestamp}")

    # Graph summary
    gs = report.graph_summary
    print(f"  Components:   {gs.get('total_nodes', 0)}")
    print(f"  Edges:        {gs.get('total_edges', 0)}")
    print(f"  Topics:       {gs.get('topics', 0)}")
    print(f"  Pub-Sub Paths:{gs.get('pub_sub_paths', 0)}")

    # Per-layer metrics
    for layer_name, metrics in report.layer_metrics.items():
        self.print_subheader(f"Layer: {layer_name.upper()}")
        _display_layer_metrics(self, metrics)

    # Top critical components
    if report.top_critical:
        self.print_subheader("Top Critical Components")
        print(f"\n  {'#':<4} {'Component':<25} {'Type':<14} {'Level':<10} {'Combined':<10} {'Cascade':<8}")
        print(f"  {'-' * 71}")
        for i, c in enumerate(report.top_critical, 1):
            color = self.level_color(c["level"])
            print(
                f"  {i:<4} {c['id']:<25} {c['type']:<14} "
                f"{self.colored(c['level'], color):<10} "
                f"{c['combined_impact']:<10.4f} {c.get('cascade_count', 0):<8}"
            )

    # Edge criticality (if present)
    if report.edge_criticality:
        self.print_subheader(f"Critical Edges ({len([e for e in report.edge_criticality if e.level in ('critical', 'high')])} high-priority)")
        for e in report.edge_criticality[:10]:
            if e.level in ("critical", "high"):
                color = self.level_color(e.level)
                print(f"  {e.source} → {e.target} [{e.relationship}] "
                      f"impact={e.combined_impact:.4f} {self.colored(e.level, color)}")

    # Library usage
    if report.library_usage:
        self.print_subheader("Library Usage")
        for app_id in sorted(report.library_usage.keys()):
            libs = report.library_usage[app_id]
            app_name = report.component_names.get(app_id, app_id)
            print(f"  {self.colored(app_name, self.Colors.WHITE, bold=True)} ({app_id})")
            for lib in sorted(libs):
                print(f"    - {lib}")

    # Recommendations
    if report.recommendations:
        self.print_subheader("Recommendations")
        for i, rec in enumerate(report.recommendations, 1):
            # Color based on severity keyword
            if rec.startswith("CRITICAL"):
                color = self.Colors.RED
            elif rec.startswith("SPOF") or rec.startswith("HIGH"):
                color = self.Colors.YELLOW
            elif rec.startswith("HEALTHY"):
                color = self.Colors.GREEN
            else:
                color = self.Colors.WHITE
            print(f"  {i}. {self.colored(rec, color)}")


def _display_layer_metrics(display, metrics: "LayerMetrics") -> None:
    """Display metrics for a single layer."""
    print(f"\n  Components:          {metrics.total_components}")

    # Event metrics
    print(f"\n  {display.colored('Event Simulation:', display.Colors.CYAN)}")
    print(f"    Throughput:        {metrics.event_throughput} messages")
    print(f"    Delivered:         {metrics.event_delivered}")
    print(f"    Dropped:           {metrics.event_dropped}")
    print(f"    Delivery Rate:     {metrics.event_delivery_rate:.1f}%")
    if metrics.event_avg_latency_ms > 0:
        print(f"    Avg Latency:       {metrics.event_avg_latency_ms:.3f} ms")
    if metrics.event_p99_latency_ms > 0:
        print(f"    P99 Latency:       {metrics.event_p99_latency_ms:.3f} ms")

    # Drop reasons
    if metrics.event_drop_reasons:
        print(f"    Drop Reasons:")
        for reason, count in sorted(metrics.event_drop_reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {reason:<28} {count}")

    # Failure metrics
    print(f"\n  {display.colored('Failure Simulation:', display.Colors.CYAN)}")
    print(f"    Avg Reachability Loss: {metrics.avg_reachability_loss * 100:.1f}%")
    print(f"    Avg Fragmentation:     {metrics.avg_fragmentation * 100:.1f}%")
    print(f"    Avg Throughput Loss:    {metrics.avg_throughput_loss * 100:.1f}%")
    print(f"    Max Impact:            {metrics.max_impact:.4f}")
    if metrics.max_impact_component:
        print(f"    Max Impact Component:  {metrics.max_impact_component}")

    # Criticality
    print(f"\n  {display.colored('Criticality:', display.Colors.CYAN)}")
    print(f"    Critical:          {display.colored(str(metrics.critical_count), display.Colors.RED)}")
    print(f"    High:              {display.colored(str(metrics.high_count), display.Colors.YELLOW)}")
    print(f"    Medium:            {metrics.medium_count}")
    print(f"    SPOFs:             {metrics.spof_count}")
