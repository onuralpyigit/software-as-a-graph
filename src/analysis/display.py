"""
Display Module

Terminal display formatting and colorized output for analysis results.
Extracted from CLI to enable reuse in other contexts.

Provides:
    - Colors: ANSI terminal color codes
    - Display functions for analysis results (graphs, components, problems)
    - Text formatting utilities
"""

from __future__ import annotations
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .analyzer import LayerAnalysisResult, MultiLayerAnalysisResult

from .classifier import CriticalityLevel


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


def level_color(level: CriticalityLevel) -> str:
    """Get color for criticality level."""
    return {
        CriticalityLevel.CRITICAL: Colors.RED,
        CriticalityLevel.HIGH: Colors.YELLOW,
        CriticalityLevel.MEDIUM: Colors.BLUE,
        CriticalityLevel.LOW: Colors.WHITE,
        CriticalityLevel.MINIMAL: Colors.GRAY,
    }.get(level, Colors.RESET)


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
# Display Functions
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
    for level in CriticalityLevel:
        count = summary.component_distribution.get(level.value, 0)
        if count > 0:
            bar_len = min(count * 2, 40)
            bar = "█" * bar_len
            color = level_color(level)
            print(f"    {level.value:10} {colored(bar, color)} {count}")
    
    # Edge distribution
    if summary.total_edges > 0:
        print(f"\n  Edges ({summary.total_edges} total):")
        for level in CriticalityLevel:
            count = summary.edge_distribution.get(level.value, 0)
            if count > 0:
                bar_len = min(count * 2, 40)
                bar = "█" * bar_len
                color = level_color(level)
                print(f"    {level.value:10} {colored(bar, color)} {count}")


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
        level_str = c.levels.overall.value.upper()
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
    
    critical_edges = [e for e in result.quality.edges if e.level >= CriticalityLevel.HIGH][:limit]
    
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

        print(
            f"  {src_display} → {tgt_display} "
            f"[{e.dependency_type}] "
            f"Score: {colored(f'{e.scores.overall:.3f}', color)} "
            f"{colored(e.level.value.upper(), color)} {bridge_flag}"
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


def display_multi_layer_result(results: "MultiLayerAnalysisResult") -> None:
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


def display_final_summary(results: "MultiLayerAnalysisResult") -> None:
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
