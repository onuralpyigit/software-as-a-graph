"""
Display Module

Consolidates display functions from analysis, simulation, and validation modules
into a single visualization module for consistent terminal output.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..analysis.analyzer import LayerAnalysisResult, MultiLayerAnalysisResult
    from ..simulation.event_simulator import EventResult
    from ..simulation.failure_simulator import FailureResult
    from ..simulation.simulator import SimulationReport
    from ..simulation.simulator import SimulationReport
    from ..domain.models.validation.results import ValidationResult, LayerValidationResult
    # Adapt PipelineResult import if we kept it or simulate it using dict
    # We returned a dict from ValidationService.validate_layers, so we might need to handle dicts in display functions
    # or wrap them back in objects. 
    # For now let's import from new locations.
    from ..domain.models.validation.metrics import ValidationTargets
    from ..analysis.classifier import CriticalityLevel

# We need to import Level enum at runtime if used for type hinting or logic
try:
    from ..analysis.classifier import CriticalityLevel
except ImportError:
    # Handle circular import or if module is not yet fully initialized
    pass


# =============================================================================
# Terminal Colors
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def colored(text: str, color: str, bold: bool = False) -> str:
    """Apply color to text."""
    style = Colors.BOLD if bold else ""
    return f"{style}{color}{text}{Colors.RESET}"


def level_color(level) -> str:
    """Get color for criticality level."""
    # Handle both Enum and string levels
    if hasattr(level, 'name'): # Enum
        return {
            "CRITICAL": Colors.RED,
            "HIGH": Colors.YELLOW,
            "MEDIUM": Colors.BLUE,
            "LOW": Colors.CYAN,
            "MINIMAL": Colors.GRAY,
        }.get(level.name, Colors.RESET)
    elif isinstance(level, str):
        return {
            "critical": Colors.RED,
            "high": Colors.YELLOW,
            "medium": Colors.BLUE,
            "low": Colors.CYAN,
            "minimal": Colors.GRAY,
        }.get(level.lower(), Colors.RESET)
    return Colors.RESET


def severity_color(severity: str) -> str:
    """Get color for severity string."""
    return {
        "CRITICAL": Colors.RED,
        "HIGH": Colors.YELLOW,
        "MEDIUM": Colors.BLUE,
        "LOW": Colors.GRAY,
    }.get(severity, Colors.RESET)


# =============================================================================
# Text Utilities
# =============================================================================

def wrap_text(text: str, width: int) -> List[str]:
    """Wrap text to specified width."""
    words = text.split()
    lines = []
    current_line = []
    current_len = 0
    
    for word in words:
        if current_len + len(word) + 1 <= width:
            current_line.append(word)
            current_len += len(word) + 1
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_len = len(word)
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return lines or [""]


# =============================================================================
# Common Display Functions
# =============================================================================

def print_header(title: str, char: str = "=", width: int = 78) -> None:
    """Print a formatted header."""
    print(f"\n{colored(char * width, Colors.CYAN)}")
    print(f"{colored(f' {title} '.center(width), Colors.CYAN, bold=True)}")
    print(f"{colored(char * width, Colors.CYAN)}")


def print_subheader(title: str, char: str = "-", width: int = 78) -> None:
    """Print a formatted subheader."""
    print(f"\n{colored(f' {title} ', Colors.WHITE, bold=True)}")
    print(f"{colored(char * width, Colors.GRAY)}")


# =============================================================================
# Analysis Display
# =============================================================================

def display_graph_summary(result: "LayerAnalysisResult") -> None:
    """Display graph structure summary."""
    print_subheader("Graph Summary")
    
    gs = result.structural.graph_summary
    
    # Basic stats
    print(f"  {'Nodes:':<20} {gs.nodes}")
    print(f"  {'Edges:':<20} {gs.edges}")
    print(f"  {'Density:':<20} {gs.density:.4f}")
    print(f"  {'Avg Degree:':<20} {gs.avg_degree:.2f}")
    print(f"  {'Avg Clustering:':<20} {gs.avg_clustering:.4f}")
    
    # Connectivity
    status = colored("Yes", Colors.GREEN) if gs.is_connected else colored("No", Colors.RED)
    print(f"  {'Connected:':<20} {status}")
    print(f"  {'Components:':<20} {gs.num_components}")
    
    # Resilience indicators
    ap_color = Colors.RED if gs.num_articulation_points > 0 else Colors.GREEN
    br_color = Colors.RED if gs.num_bridges > 0 else Colors.GREEN
    print(f"  {'Articulation Pts:':<20} {colored(str(gs.num_articulation_points), ap_color)}")
    print(f"  {'Bridges:':<20} {colored(str(gs.num_bridges), br_color)}")
    
    # Health indicator
    health = gs.connectivity_health
    health_color = {
        "ROBUST": Colors.GREEN,
        "MODERATE": Colors.YELLOW,
        "FRAGILE": Colors.RED,
        "DISCONNECTED": Colors.RED,
    }.get(health, Colors.WHITE)
    print(f"  {'Health:':<20} {colored(health, health_color, bold=True)}")
    
    # Node types
    if gs.node_types:
        print(f"\n  Node Types: ", end="")
        print(", ".join(f"{t}: {c}" for t, c in gs.node_types.items()))
    
    # Edge types
    if gs.edge_types:
        print(f"  Edge Types: ", end="")
        print(", ".join(f"{t}: {c}" for t, c in gs.edge_types.items()))


def display_classification_summary(result: "LayerAnalysisResult") -> None:
    """Display classification distribution."""
    print_subheader("Classification Summary")
    
    summary = result.quality.classification_summary
    
    # Component distribution
    print(f"\n  Components ({summary.total_components} total):")
    # Using CriticalityLevel enum values if available, otherwise iterating keys
    for level_name, count in sorted(summary.component_distribution.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            bar_len = min(count * 2, 40)
            bar = "█" * bar_len
            color = level_color(level_name)
            print(f"    {level_name:10} {colored(bar, color)} {count}")
    
    # Edge distribution
    if summary.total_edges > 0:
        print(f"\n  Edges ({summary.total_edges} total):")
        for level_name, count in sorted(summary.edge_distribution.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                bar_len = min(count * 2, 40)
                bar = "█" * bar_len
                color = level_color(level_name)
                print(f"    {level_name:10} {colored(bar, color)} {count}")


def display_critical_components(result: "LayerAnalysisResult", limit: int = 15) -> None:
    """Display top critical components."""
    print_subheader("Top Components by Criticality", width=96)
    
    components = result.quality.components[:limit]
    
    if not components:
        print(f"  {colored('No components found.', Colors.GRAY)}")
        return
    
    # Header
    header = f"  {'ID':<20} {'Name':<20} {'Type':<10} {'R':<6} {'M':<6} {'A':<6} {'V':<6} {'Q':<6} {'Level':<10}"
    print(colored(header, Colors.WHITE, bold=True))
    print(f"  {'-' * 96}")
    
    for c in components:
        level_str = c.levels.overall.name if hasattr(c.levels.overall, 'name') else str(c.levels.overall)
        color = level_color(c.levels.overall)
        
        # Indicator symbols
        ap_flag = colored("●", Colors.RED) if c.structural.is_articulation_point else " "
        
        # Component details
        print(
            f"  {c.id:<20} {c.structural.name[:20]:<20} {c.type:<10} "
            f"{c.scores.reliability:.3f}  {c.scores.maintainability:.3f}  "
            f"{c.scores.availability:.3f}  {c.scores.vulnerability:.3f}  "
            f"{colored(f'{c.scores.overall:.3f}', color)}  "
            f"{colored(level_str, color):<10} {ap_flag}"
        )
    
    if len(result.quality.components) > limit:
        print(f"\n  {colored(f'... and {len(result.quality.components) - limit} more', Colors.GRAY)}")
    
    # Legend
    print(f"\n  Legend: R=Reliability, M=Maintainability, A=Availability, V=Vulnerability, "
          f"Q=Overall")
    print(f"          {colored('●', Colors.RED)} = Articulation Point (SPOF)")


def display_critical_edges(result: "LayerAnalysisResult", limit: int = 10) -> None:
    """Display top critical edges."""
    print_subheader("Critical Edges", width=96)
    
    # Need to handle enum iteration safely
    critical_edges = []
    for e in result.quality.edges:
        # Assuming higher level means more critical, we'll implement simple filtering
        level_val = 0
        if hasattr(e.level, 'value'):
            if isinstance(e.level.value, int):
                level_val = e.level.value 
            # If string based enum, we can't easily compare, so we might skip this optimization for now
            # or rely on upstream sorting
    
    # Assuming result.quality.edges is already sorted or we just display first few.
    # The original code filtered by level >= HIGH. Let's try to replicate if we can get Level enum
    # For now, just display top ones
    critical_edges = result.quality.edges[:limit]
    
    if not critical_edges:
        print(f"  {colored('No critical edges detected.', Colors.GREEN)}")
        return
    
    # Map IDs to names
    node_names = {c.id: c.structural.name for c in result.quality.components}

    for e in critical_edges:
        color = level_color(e.level)
        bridge_flag = colored("🌉", Colors.RED) if e.structural and e.structural.is_bridge else ""
        
        src_name = node_names.get(e.source, e.source)
        tgt_name = node_names.get(e.target, e.target)
        
        src_display = f"{e.source} ({src_name})" if src_name != e.source else e.source
        tgt_display = f"{e.target} ({tgt_name})" if tgt_name != e.target else e.target
        
        level_str = e.level.name if hasattr(e.level, 'name') else str(e.level)

        print(
            f"  {src_display} → {tgt_display} "
            f"[{e.dependency_type}] "
            f"Score: {colored(f'{e.scores.overall:.3f}', color)} "
            f"{colored(level_str, color)} {bridge_flag}"
        )


def display_problems(result: "LayerAnalysisResult", limit: int = 10) -> None:
    """Display detected problems."""
    print_subheader("Detected Problems")
    
    # Create name lookup map
    node_names = {c.id: c.structural.name for c in result.quality.components}
    
    problems = result.problems
    
    if not problems:
        print(f"  {colored('✓ No architectural problems detected.', Colors.GREEN)}")
        return
    
    # Summary
    summary = result.problem_summary
    print(f"\n  Total: {summary.total_problems} problems")
    print(f"  By Severity: ", end="")
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        count = summary.by_severity.get(sev, 0)
        if count > 0:
            color = severity_color(sev)
            print(f"{colored(f'{sev}: {count}', color)}  ", end="")
    print()
    
    # Details
    print(f"\n  {'-' * 75}")
    
    for i, p in enumerate(problems[:limit]):
        if i > 0:
            print(f"  {'-' * 75}")
        
        sev_color = severity_color(p.severity)
        
        print(f"  [{colored(p.severity, sev_color, bold=True):>8}] {colored(p.name, Colors.WHITE, bold=True)}")
        
        # Resolve entity display name
        entity_display = p.entity_id
        if p.entity_type == "Component":
            name = node_names.get(p.entity_id)
            if name and name != p.entity_id:
                entity_display = f"{p.entity_id} ({name})"
        elif p.entity_type == "Edge" and "->" in p.entity_id:
            try:
                src, tgt = p.entity_id.split("->", 1)
                src_name = node_names.get(src, src)
                tgt_name = node_names.get(tgt, tgt)
                src_disp = f"{src} ({src_name})" if src_name != src else src
                tgt_disp = f"{tgt} ({tgt_name})" if tgt_name != tgt else tgt
                entity_display = f"{src_disp} -> {tgt_disp}"
            except ValueError:
                pass

        print(f"           Entity: {colored(entity_display, Colors.CYAN)} ({p.entity_type})")
        print(f"           Category: {p.category}")
        
        # Wrap description
        desc_lines = wrap_text(p.description, 60)
        print(f"           Issue: {desc_lines[0]}")
        for line in desc_lines[1:]:
            print(f"                  {line}")
        
        # Wrap recommendation
        rec_lines = wrap_text(p.recommendation, 60)
        print(f"           Fix: {colored(rec_lines[0], Colors.GREEN)}")
        for line in rec_lines[1:]:
            print(f"                {colored(line, Colors.GREEN)}")
    
    if len(problems) > limit:
        print(f"\n  {colored(f'... and {len(problems) - limit} more', Colors.GRAY)}")


def display_layer_result(result: "LayerAnalysisResult") -> None:
    """Display complete analysis result for a single layer."""
    print_header(f"{result.layer_name} Analysis")
    print(f"  {colored(result.description, Colors.GRAY)}")
    
    display_graph_summary(result)
    display_classification_summary(result)
    display_critical_components(result)
    display_critical_edges(result)
    display_problems(result)


def display_multi_layer_analysis_result(results: "MultiLayerAnalysisResult") -> None:
    """Display analysis results for multiple layers."""
    print_header("Multi-Layer Analysis Results", "═")
    print(f"\n  Timestamp: {results.timestamp}")
    print(f"  Layers: {', '.join(results.layers.keys())}")
    
    for layer_name, layer_result in results.layers.items():
        display_layer_result(layer_result)
    
    # Cross-layer insights
    if results.cross_layer_insights:
        print_subheader("Cross-Layer Insights")
        for insight in results.cross_layer_insights:
            print(f"  • {insight}")
    
    # Overall summary
    print_subheader("Overall Summary")
    summary = results.summary
    print(f"  Layers analyzed:     {summary['layers_analyzed']}")
    print(f"  Total components:    {summary['total_components']}")
    print(f"  Total problems:      {summary['total_problems']}")
    if summary['critical_problems'] > 0:
        print(f"  Critical problems:   {colored(str(summary['critical_problems']), Colors.RED, bold=True)}")


def display_final_analysis_summary(results: "MultiLayerAnalysisResult") -> None:
    """Display final summary with actionable items."""
    print_header("Action Items", "-")
    
    all_problems = results.get_all_problems()
    critical_count = sum(1 for p in all_problems if p.severity == "CRITICAL")
    high_count = sum(1 for p in all_problems if p.severity == "HIGH")
    
    # Collect all component names across layers
    all_names = {}
    for layer in results.layers.values():
        for c in layer.quality.components:
            all_names[c.id] = c.structural.name

    if critical_count > 0:
        print(f"\n  {colored(f'⚠ {critical_count} CRITICAL issues require immediate attention', Colors.RED, bold=True)}")
        for p in all_problems:
            if p.severity == "CRITICAL":
                # Resolve entity display name
                entity_display = p.entity_id
                if p.entity_type == "Component":
                    name = all_names.get(p.entity_id)
                    if name and name != p.entity_id:
                        entity_display = f"{p.entity_id} ({name})"
                elif p.entity_type == "Edge" and "->" in p.entity_id:
                    try:
                        src, tgt = p.entity_id.split("->", 1)
                        src_name = all_names.get(src, src)
                        tgt_name = all_names.get(tgt, tgt)
                        src_disp = f"{src} ({src_name})" if src_name != src else src
                        tgt_disp = f"{tgt} ({tgt_name})" if tgt_name != tgt else tgt
                        entity_display = f"{src_disp} -> {tgt_disp}"
                    except ValueError:
                        pass
                
                print(f"    • {entity_display}: {p.name}")
    
    if high_count > 0:
        print(f"\n  {colored(f'⚠ {high_count} HIGH priority issues should be reviewed', Colors.YELLOW)}")
    
    if critical_count == 0 and high_count == 0:
        print(f"\n  {colored('✓ No high-priority issues detected. System looks healthy!', Colors.GREEN, bold=True)}")
    
    print()


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


def display_simulation_report(report: "SimulationReport") -> None:
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


# =============================================================================
# Validation Result Display
# =============================================================================

def status_icon(passed: bool) -> str:
    """Get status icon for pass/fail."""
    if passed:
        return colored("✓", Colors.GREEN, bold=True)
    return colored("✗", Colors.RED, bold=True)


def status_text(passed: bool) -> str:
    """Get status text for pass/fail."""
    if passed:
        return colored("PASSED", Colors.GREEN, bold=True)
    return colored("FAILED", Colors.RED, bold=True)


def metric_color(value: float, target: float, higher_better: bool = True) -> str:
    """Get color for metric based on target."""
    if higher_better:
        passed = value >= target
    else:
        passed = value <= target
    
    if passed:
        return Colors.GREEN
    elif higher_better and value >= target * 0.9:
        return Colors.YELLOW
    elif not higher_better and value <= target * 1.1:
        return Colors.YELLOW
    return Colors.RED


def display_pipeline_validation_result(result: "PipelineResult") -> None:
    """Display the full validation pipeline result."""
    print_header("VALIDATION PIPELINE RESULTS", "═")
    
    print(f"\n  {colored('Timestamp:', Colors.CYAN)}      {result.timestamp}")
    print(f"  {colored('Components:', Colors.CYAN)}     {result.total_components}")
    
    # Overall status
    print(f"\n  {colored('Overall Status:', Colors.WHITE, bold=True)} {status_text(result.all_passed)}")
    print(f"  Layers Validated: {len(result.layers)}")
    print(f"  Layers Passed:    {result.layers_passed}")
    
    # Layer summary table
    print_subheader("Layer Summary")
    
    print(f"\n  {'Layer':<12} {'N':<6} {'Spearman':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'Top-5':<10} {'Status':<10}")
    print(f"  {'-' * 78}")
    
    targets = result.targets
    for layer_name, layer_result in result.layers.items():
        spearman_color = metric_color(layer_result.spearman, targets.spearman)
        f1_color = metric_color(layer_result.f1_score, targets.f1_score)
        prec_color = metric_color(layer_result.precision, targets.precision)
        recall_color = metric_color(layer_result.recall, targets.recall)
        top5_color = metric_color(layer_result.top_5_overlap, targets.top_5_overlap)
        
        print(
            f"  {layer_name:<12} {layer_result.matched_components:<6} "
            f"{colored(f'{layer_result.spearman:>8.4f}', spearman_color)}   "
            f"{colored(f'{layer_result.f1_score:>8.4f}', f1_color)}   "
            f"{colored(f'{layer_result.precision:>8.4f}', prec_color)}   "
            f"{colored(f'{layer_result.recall:>8.4f}', recall_color)}   "
            f"{colored(f'{layer_result.top_5_overlap:>8.4f}', top5_color)}   "
            f"{status_text(layer_result.passed)}"
        )
    
    # Validation targets
    print_subheader("Validation Targets")
    print(f"\n  Spearman ρ ≥ {targets.spearman:.2f}  |  F1 ≥ {targets.f1_score:.2f}  |  "
          f"Precision ≥ {targets.precision:.2f}  |  Recall ≥ {targets.recall:.2f}  |  "
          f"Top-5 ≥ {targets.top_5_overlap:.2f}")
    
    # Cross-layer insights
    if result.cross_layer_insights:
        print_subheader("Cross-Layer Insights")
        for insight in result.cross_layer_insights:
            print(f"  • {insight}")
    
    # Detailed layer results
    for layer_name, layer_result in result.layers.items():
        display_layer_validation_result(layer_result, targets)
    
    # Final summary
    print_header("FINAL SUMMARY", "-")
    
    if result.all_passed:
        print(f"\n  {colored('✓ All validation targets met!', Colors.GREEN, bold=True)}")
        print(f"\n  The graph analysis approach accurately predicts component criticality.")
        print(f"  Predicted quality scores correlate strongly with simulated failure impact.")
        print(f"\n  {colored('Conclusion:', Colors.CYAN)} The methodology is validated for production use.")
    else:
        print(f"\n  {colored('✗ Some validation targets not met.', Colors.RED, bold=True)}")
        print(f"\n  Review the metrics above to identify areas for improvement.")
        
        # Specific suggestions
        for layer_name, layer_result in result.layers.items():
            if not layer_result.passed:
                print(f"\n  {colored(f'Suggestions for {layer_name}:', Colors.YELLOW)}")
                
                if layer_result.spearman < targets.spearman:
                    print(f"    • Low Spearman correlation ({layer_result.spearman:.3f} < {targets.spearman})")
                    print(f"      Consider adjusting quality formula weights.")
                
                if layer_result.f1_score < targets.f1_score:
                    print(f"    • Low F1 score ({layer_result.f1_score:.3f} < {targets.f1_score})")
                    print(f"      Review classification thresholds.")
                
                if layer_result.precision < targets.precision:
                    print(f"    • Low precision ({layer_result.precision:.3f} < {targets.precision})")
                    print(f"      Model over-predicts critical components (too many false positives).")
                
                if layer_result.recall < targets.recall:
                    print(f"    • Low recall ({layer_result.recall:.3f} < {targets.recall})")
                    print(f"      Model misses critical components (too many false negatives).")
    
    print()


def display_layer_validation_result(result: "LayerValidationResult", targets: "ValidationTargets") -> None:
    """Display detailed results for a single layer."""
    print_subheader(f"Layer: {result.layer_name} ({result.layer})")
    
    print(f"\n  {colored('Data Alignment:', Colors.CYAN)}")
    print(f"    Predicted Components:  {result.predicted_components}")
    print(f"    Simulated Components:  {result.simulated_components}")
    print(f"    Matched Components:    {result.matched_components}")
    
    if result.warnings:
        print(f"\n  {colored('Warnings:', Colors.YELLOW)}")
        for warning in result.warnings:
            print(f"    ⚠ {warning}")
    
    if not result.validation_result:
        return
    
    overall = result.validation_result.overall
    
    # Correlation metrics
    print(f"\n  {colored('Correlation Metrics:', Colors.CYAN)}")
    print(f"    {'Metric':<15} {'Value':<12} {'Target':<12} {'Status':<10}")
    print(f"    {'-' * 49}")
    
    spearman_passed = overall.correlation.spearman >= targets.spearman
    print(f"    {'Spearman ρ':<15} {overall.correlation.spearman:>8.4f}     ≥{targets.spearman:<8.2f}   {status_icon(spearman_passed)}")
    
    pearson_passed = overall.correlation.pearson >= targets.pearson
    print(f"    {'Pearson r':<15} {overall.correlation.pearson:>8.4f}     ≥{targets.pearson:<8.2f}   {status_icon(pearson_passed)}")
    
    kendall_passed = overall.correlation.kendall >= targets.kendall
    print(f"    {'Kendall τ':<15} {overall.correlation.kendall:>8.4f}     ≥{targets.kendall:<8.2f}   {status_icon(kendall_passed)}")
    
    # Classification metrics
    print(f"\n  {colored('Classification Metrics:', Colors.CYAN)}")
    print(f"    {'Metric':<15} {'Value':<12} {'Target':<12} {'Status':<10}")
    print(f"    {'-' * 49}")
    
    f1_passed = overall.classification.f1_score >= targets.f1_score
    print(f"    {'F1 Score':<15} {overall.classification.f1_score:>8.4f}     ≥{targets.f1_score:<8.2f}   {status_icon(f1_passed)}")
    
    prec_passed = overall.classification.precision >= targets.precision
    print(f"    {'Precision':<15} {overall.classification.precision:>8.4f}     ≥{targets.precision:<8.2f}   {status_icon(prec_passed)}")
    
    recall_passed = overall.classification.recall >= targets.recall
    print(f"    {'Recall':<15} {overall.classification.recall:>8.4f}     ≥{targets.recall:<8.2f}   {status_icon(recall_passed)}")
    
    print(f"    {'Accuracy':<15} {overall.classification.accuracy:>8.4f}")
    
    # Confusion matrix
    cm = overall.classification.confusion_matrix
    print(f"\n    {colored('Confusion Matrix:', Colors.GRAY)}")
    print(f"                            Actual Critical    Actual Non-Critical")
    print(f"      Pred Critical           {colored(str(cm['tp']), Colors.GREEN):>3}                {colored(str(cm['fp']), Colors.RED):>3}")
    print(f"      Pred Non-Critical       {colored(str(cm['fn']), Colors.RED):>3}                {colored(str(cm['tn']), Colors.GREEN):>3}")
    
    # Ranking metrics
    print(f"\n  {colored('Ranking Metrics:', Colors.CYAN)}")
    print(f"    {'Metric':<15} {'Value':<12} {'Target':<12} {'Status':<10}")
    print(f"    {'-' * 49}")
    
    top5_passed = overall.ranking.top_5_overlap >= targets.top_5_overlap
    print(f"    {'Top-5 Overlap':<15} {overall.ranking.top_5_overlap:>8.4f}     ≥{targets.top_5_overlap:<8.2f}   {status_icon(top5_passed)}")
    
    top10_passed = overall.ranking.top_10_overlap >= targets.top_10_overlap
    print(f"    {'Top-10 Overlap':<15} {overall.ranking.top_10_overlap:>8.4f}     ≥{targets.top_10_overlap:<8.2f}   {status_icon(top10_passed)}")
    
    # Top-5 agreement
    if overall.ranking.top_5_predicted:
        # Helpers for name resolution
        names = result.component_names
        def get_name(comp_id: str) -> str:
            return f"{comp_id} ({names.get(comp_id, comp_id)})" if comp_id in names else comp_id

        print(f"\n    {colored('Top-5 Agreement:', Colors.GRAY)}")
        print(f"      Predicted:  {', '.join(get_name(c) for c in overall.ranking.top_5_predicted[:5])}")
        print(f"      Actual:     {', '.join(get_name(c) for c in overall.ranking.top_5_actual[:5])}")
        print(f"      Common:     {', '.join(get_name(c) for c in overall.ranking.top_5_common) or 'None'}")
    
    # Error metrics
    print(f"\n  {colored('Error Metrics:', Colors.CYAN)}")
    rmse_passed = overall.error.rmse <= targets.rmse_max
    print(f"    RMSE: {overall.error.rmse:.4f} {'≤' if rmse_passed else '>'} {targets.rmse_max} {status_icon(rmse_passed)}")
    print(f"    MAE:  {overall.error.mae:.4f}")
    print(f"    Max Error: {overall.error.max_error:.4f}")
    
    # Top mismatches (if available)
    if result.comparisons:
        # Helpers for name resolution
        names = result.component_names
        
        print(f"\n  {colored('Top Mismatches (by error):', Colors.CYAN)}")
        print(f"    {'Component':<35} {'Type':<12} {'Predicted':<10} {'Actual':<10} {'Error':<10} {'Class':<6}")
        print(f"    {'-' * 85}")
        
        for comp in result.comparisons[:5]:
            cls_color = Colors.GREEN if comp.classification in ("TP", "TN") else Colors.RED
            c_name = f"{comp.id} ({names.get(comp.id, comp.id)})" if comp.id in names else comp.id
            print(
                f"    {c_name:<35} {comp.type:<12} "
                f"{comp.predicted:>8.4f}   {comp.actual:>8.4f}   "
                f"{comp.error:>8.4f}   {colored(comp.classification, cls_color)}"
            )
    
    # Per-type breakdown
    if result.validation_result.by_type:
        print(f"\n  {colored('Per-Type Breakdown:', Colors.CYAN)}")
        print(f"    {'Type':<15} {'N':<6} {'Spearman':<10} {'F1':<10} {'Passed':<10}")
        print(f"    {'-' * 51}")
        
        for comp_type, type_result in result.validation_result.by_type.items():
            print(
                f"    {comp_type:<15} {type_result.sample_size:<6} "
                f"{type_result.correlation.spearman:>8.4f}   "
                f"{type_result.classification.f1_score:>8.4f}   "
                f"{status_icon(type_result.passed)}"
            )
