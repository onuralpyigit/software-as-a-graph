"""
Display Application Service
"""
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.group import Group
    from rich.text import Text
    from rich.bar import Bar
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.theme import Theme
    from rich.style import Style
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

if TYPE_CHECKING:
    from src.analysis.models import LayerAnalysisResult, MultiLayerAnalysisResult
    from src.simulation.models import SimulationReport, EventResult, FailureResult
    from src.validation.models import PipelineResult, LayerValidationResult, ValidationTargets


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


class ConsoleDisplay:
    """
    Service for formatting and displaying analysis, simulation, and validation results in the terminal.
    """
    Colors = Colors

    def __init__(self):
        if RICH_AVAILABLE:
            custom_theme = Theme({
                "info": "cyan",
                "warning": "yellow",
                "error": "red",
                "success": "green",
                "critical": "bold red",
                "high": "bold yellow",
                "medium": "bold blue",
                "low": "cyan",
                "minimal": "grey50",
                "header": "bold magenta",
                "step": "bold blue",
            })
            self.console = Console(theme=custom_theme)
        else:
            self.console = None

    @staticmethod
    def colored(text: str, color: str, bold: bool = False) -> str:
        """Apply color to text."""
        style = Colors.BOLD if bold else ""
        return f"{style}{color}{text}{Colors.RESET}"

    @staticmethod
    def level_color(level) -> str:
        """Get color for criticality level."""
        if hasattr(level, 'name'):
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

    @staticmethod
    def severity_color(severity: str) -> str:
        """Get color for severity string."""
        return {
            "CRITICAL": Colors.RED,
            "HIGH": Colors.YELLOW,
            "MEDIUM": Colors.BLUE,
            "LOW": Colors.GRAY,
        }.get(severity.upper(), Colors.RESET)

    def print_header(self, title: str, char: str = "=", width: int = 78) -> None:
        """Print a formatted header."""
        if RICH_AVAILABLE:
            self.console.print()
            self.console.print(Panel(
                Text(title.upper(), justify="center", style="header"),
                border_style="cyan",
                width=width,
                box=box.DOUBLE
            ))
        else:
            print(f"\n{self.colored(char * width, Colors.CYAN)}")
            print(f"{self.colored(f' {title} '.center(width), Colors.CYAN, bold=True)}")
            print(f"{self.colored(char * width, Colors.CYAN)}")

    def print_subheader(self, title: str, char: str = "-", width: int = 78) -> None:
        """Print a formatted subheader."""
        if RICH_AVAILABLE:
            self.console.print()
            self.console.print(Text(f" {title} ", style="bold white on grey23"))
            self.console.print(Text(char * width, style="grey50"))
        else:
            print(f"\n{self.colored(f' {title} ', Colors.WHITE, bold=True)}")
            print(f"{self.colored(char * width, Colors.GRAY)}")

    def print_step(self, msg: str) -> None:
        """Print a step indicator."""
        if RICH_AVAILABLE:
            self.console.print(f"  [step]→[/step] {msg}")
        else:
            print(f"  {self.colored('→', Colors.BLUE)} {msg}")

    def print_success(self, msg: str) -> None:
        """Print a success indicator."""
        if RICH_AVAILABLE:
            self.console.print(f"  [success]✓[/success] {msg}")
        else:
            print(f"  {self.colored('✓', Colors.GREEN)} {msg}")

    def print_error(self, msg: str) -> None:
        """Print an error indicator."""
        if RICH_AVAILABLE:
            self.console.print(f"  [error]✗[/error] {msg}")
        else:
            print(f"  {self.colored('✗', Colors.RED)} {msg}")

    def print_warning(self, msg: str) -> None:
        """Print a warning indicator."""
        if RICH_AVAILABLE:
            self.console.print(f"  [warning]⚠[/warning] {msg}")
        else:
            print(f"  {self.colored('⚠', Colors.YELLOW)} {msg}")

    def wrap_text(self, text: str, width: int) -> List[str]:
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

    # --- Analysis Display ---

    def display_graph_summary(self, result: "LayerAnalysisResult") -> None:
        """Display graph structure summary."""
        if not RICH_AVAILABLE:
            self.print_subheader("Graph Summary")
            gs = result.structural.graph_summary
            print(f"  {'Nodes:':<20} {gs.nodes}")
            print(f"  {'Edges:':<20} {gs.edges}")
            print(f"  {'Density:':<20} {gs.density:.4f}")
            print(f"  {'Avg Degree:':<20} {gs.avg_degree:.2f}")
            print(f"  {'Avg Clustering:':<20} {gs.avg_clustering:.4f}")
            status = self.colored("Yes", Colors.GREEN) if gs.is_connected else self.colored("No", Colors.RED)
            print(f"  {'Connected:':<20} {status}")
            print(f"  {'Components:':<20} {gs.num_components}")
            ap_color = Colors.RED if gs.num_articulation_points > 0 else Colors.GREEN
            br_color = Colors.RED if gs.num_bridges > 0 else Colors.GREEN
            print(f"  {'Articulation Pts:':<20} {self.colored(str(gs.num_articulation_points), ap_color)}")
            print(f"  {'Bridges:':<20} {self.colored(str(gs.num_bridges), br_color)}")
            health = gs.connectivity_health
            health_color = {"ROBUST": Colors.GREEN, "MODERATE": Colors.YELLOW, "FRAGILE": Colors.RED, "DISCONNECTED": Colors.RED}.get(health, Colors.WHITE)
            print(f"  {'Health:':<20} {self.colored(health, health_color, bold=True)}")
            if gs.node_types:
                print(f"\n  Node Types: {', '.join(f'{t}: {c}' for t, c in gs.node_types.items())}")
            if gs.edge_types:
                print(f"  Edge Types: {', '.join(f'{t}: {c}' for t, c in gs.edge_types.items())}")
            return

        self.print_subheader("Graph Topology Overview")
        gs = result.structural.graph_summary
        
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold white")
        
        table.add_row("Nodes / Edges", f"{gs.nodes} / {gs.edges}")
        table.add_row("Density", f"{gs.density:.4f}")
        table.add_row("Avg Degree", f"{gs.avg_degree:.2f}")
        table.add_row("Avg Clustering", f"{gs.avg_clustering:.4f}")
        
        status_text = "[green]Yes[/green]" if gs.is_connected else "[red]No[/red]"
        table.add_row("Connected", status_text)
        table.add_row("Components", str(gs.num_components))
        
        ap_style = "red" if gs.num_articulation_points > 0 else "green"
        br_style = "red" if gs.num_bridges > 0 else "green"
        table.add_row("Articulation Pts", f"[{ap_style}]{gs.num_articulation_points}[/{ap_style}]")
        table.add_row("Bridges", f"[{br_style}]{gs.num_bridges}[/{br_style}]")
        
        health = gs.connectivity_health
        health_style = {"ROBUST": "green", "MODERATE": "yellow", "FRAGILE": "red", "DISCONNECTED": "bold red"}.get(health, "white")
        table.add_row("Connectivity Health", f"[{health_style}]{health}[/{health_style}]")
        
        self.console.print(table)
        
        if gs.node_types or gs.edge_types:
            parts = []
            if gs.node_types:
                parts.append(f"[bold]Nodes:[/] " + ", ".join(f"{t}: {c}" for t, c in gs.node_types.items()))
            if gs.edge_types:
                parts.append(f"[bold]Edges:[/] " + ", ".join(f"{t}: {c}" for t, c in gs.edge_types.items()))
            self.console.print(Columns([Panel(p, padding=(0,1), border_style="grey23") for p in parts]))

    def display_classification_summary(self, result: "LayerAnalysisResult") -> None:
        """Display classification distribution."""
        if not RICH_AVAILABLE:
            self.print_subheader("Classification Summary")
            summary = result.quality.classification_summary
            print(f"\n  Components ({summary.total_components} total):")
            for level_name, count in sorted(summary.component_distribution.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    bar = "█" * min(count * 2, 40)
                    print(f"    {level_name:10} {self.colored(bar, self.level_color(level_name))} {count}")
            if summary.total_edges > 0:
                print(f"\n  Edges ({summary.total_edges} total):")
                for level_name, count in sorted(summary.edge_distribution.items(), key=lambda x: x[1], reverse=True):
                    if count > 0:
                        bar = "█" * min(count * 2, 40)
                        print(f"    {level_name:10} {self.colored(bar, self.level_color(level_name))} {count}")
            return

        self.print_subheader("Criticality Distribution")
        summary = result.quality.classification_summary
        
        def render_distribution(dist, title, total):
            if total == 0: return None
            table = Table(box=box.MINIMAL, show_header=True, padding=(0, 2), header_style="bold grey70")
            table.add_column("Level", style="bold")
            table.add_column("Count", justify="right")
            table.add_column("Visual", width=30)
            
            for lvl in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL"]:
                count = dist.get(lvl.lower(), 0)
                if count == 0 and lvl != "CRITICAL": continue
                
                pct = count / total if total > 0 else 0
                bar_len = int(pct * 28)
                bar = "█" * bar_len
                color = self.level_color(lvl)
                table.add_row(
                    Text(lvl, style=color),
                    str(count),
                    Text(bar, style=color)
                )
            return table

        comp_table = render_distribution(summary.component_distribution, "Components", summary.total_components)
        edge_table = render_distribution(summary.edge_distribution, "Edges", summary.total_edges)
        
        items = []
        if comp_table: items.append(Panel(comp_table, title="[bold]Components[/]", border_style="grey23"))
        if edge_table: items.append(Panel(edge_table, title="[bold]Edges[/]", border_style="grey23"))
        
        if items:
            self.console.print(Columns(items))

    def display_layer_critical_components(self, result: "LayerAnalysisResult", limit: int = 15) -> None:
        """Display top critical components."""
        if not RICH_AVAILABLE:
            self.print_subheader("Top Components by Criticality", width=96)
            components = result.quality.components[:limit]
            if not components:
                print(f"  {self.colored('No components found.', Colors.GRAY)}")
                return
            header = f"  {'ID':<20} {'Name':<20} {'Type':<10} {'R':<6} {'M':<6} {'A':<6} {'V':<6} {'Q':<6} {'Level':<10}"
            print(self.colored(header, Colors.WHITE, bold=True))
            print(f"  {'-' * 96}")
            for c in components:
                level_str = c.levels.overall.name if hasattr(c.levels.overall, 'name') else str(c.levels.overall)
                color = self.level_color(c.levels.overall)
                ap_flag = self.colored("●", Colors.RED) if c.structural.is_articulation_point else " "
                print(f"  {c.id:<20} {c.structural.name[:20]:<20} {c.type:<10} {c.scores.reliability:.3f}  {c.scores.maintainability:.3f}  {c.scores.availability:.3f}  {c.scores.vulnerability:.3f}  {self.colored(f'{c.scores.overall:.3f}', color)}  {self.colored(level_str, color):<10} {ap_flag}")
                
                # Show direct affected/related components
                related = []
                if c.type == "Application" and c.id in result.library_usage:
                    related = [f"Uses Lib: {l}" for l in result.library_usage[c.id]]
                elif c.type == "Node" and c.id in result.node_allocations:
                    related = [f"Hosts App: {a}" for a in result.node_allocations[c.id]]
                elif c.type == "Broker" and c.id in result.broker_routing:
                    related = [f"Routes Topic: {t}" for t in result.broker_routing[c.id]]
                
                if related:
                    # Wrap or truncate related list if too long
                    related_str = ", ".join(related)
                    if len(related_str) > 90:
                        related_str = related_str[:87] + "..."
                    print(f"    {self.colored('↳ ', Colors.GRAY)} {self.colored(related_str, Colors.GRAY)}")
                    
            if len(result.quality.components) > limit:
                print(f"\n  {self.colored(f'... and {len(result.quality.components) - limit} more', Colors.GRAY)}")
            return

        self.print_subheader(f"Top {limit} Critical Components")
        components = result.quality.components[:limit]
        if not components:
            self.console.print("  [grey50]No components found.[/]")
            return

        table = Table(
            box=box.MINIMAL_HEAVY_HEAD, 
            header_style="bold white", 
            padding=(0, 1),
            caption="[grey50]R: Reliability | M: Maintainability | A: Availability | V: Vulnerability | Q: Overall | S: SPOF[/]",
            caption_justify="left"
        )
        table.add_column("Rank", justify="center", style="grey70")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="bold")
        table.add_column("Type", style="grey70")
        table.add_column("R", justify="right")
        table.add_column("M", justify="right")
        table.add_column("A", justify="right")
        table.add_column("V", justify="right")
        table.add_column("Score (Q)", justify="right")
        table.add_column("Level", justify="left")
        table.add_column("S", justify="center") # SPOF flag

        for i, c in enumerate(components, 1):
            level_str = c.levels.overall.name if hasattr(c.levels.overall, 'name') else str(c.levels.overall)
            color = self.level_color(c.levels.overall)
            spof = "[red]●[/]" if c.structural.is_articulation_point else ""
            
            table.add_row(
                str(i),
                c.id,
                c.structural.name[:25],
                c.type,
                f"{c.scores.reliability:.3f}",
                f"{c.scores.maintainability:.3f}",
                f"{c.scores.availability:.3f}",
                f"{c.scores.vulnerability:.3f}",
                Text(f"{c.scores.overall:.4f}", style=color),
                Text(level_str.capitalize(), style=color),
                spof
            )
            
            # Sub-row for related entities if present
            related = []
            if c.type == "Application" and c.id in result.library_usage:
                related = [f"Uses Lib: {l}" for l in result.library_usage[c.id]]
            elif c.type == "Node" and c.id in result.node_allocations:
                related = [f"Hosts App: {a}" for a in result.node_allocations[c.id]]
            elif c.type == "Broker" and c.id in result.broker_routing:
                related = [f"Routes Topic: {t}" for t in result.broker_routing[c.id]]
            
            if related:
                rel_str = ", ".join(related)
                if len(rel_str) > 100: rel_str = rel_str[:97] + "..."
                # We can't easily add a second row with different span in rich.Table without complex hacks,
                # but we can use a Group or just print it after. 
                # Let's keep it simple and just add it as a new row with mostly empty cells or handle it via padding.
                # Actually, adding it as a row with col_span support is not there, so we'll just use a small marker.
        
        self.console.print(table)
        
        if len(result.quality.components) > limit:
            self.console.print(f"\n  [grey50]... and {len(result.quality.components) - limit} more components[/]")

    def display_problems(self, result: "LayerAnalysisResult", limit: int = 10) -> None:
        """Display detected problems."""
        if not RICH_AVAILABLE:
            self.print_subheader("Detected Problems")
            node_names = {c.id: c.structural.name for c in result.quality.components}
            problems = result.problems
            if not problems:
                print(f"  {self.colored('✓ No architectural problems detected.', Colors.GREEN)}")
                return
            summary = result.problem_summary
            print(f"\n  Total: {summary.total_problems} problems")
            print(f"  By Severity: ", end="")
            for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                count = summary.by_severity.get(sev, 0)
                if count > 0: print(f"{self.colored(f'{sev}: {count}', self.severity_color(sev))}  ", end="")
            print(f"\n\n  {'-' * 75}")
            for i, p in enumerate(problems[:limit]):
                if i > 0: print(f"  {'-' * 75}")
                sev_color = self.severity_color(p.severity)
                print(f"  [{self.colored(p.severity, sev_color, bold=True):>8}] {self.colored(p.name, Colors.WHITE, bold=True)}")
                entity_display = p.entity_id
                related_info = ""
                
                if p.entity_type == "Component":
                    name = node_names.get(p.entity_id)
                    if name: entity_display = f"{p.entity_id} ({name})"
                    
                    # Check for related components
                    related = []
                    if p.entity_id in result.library_usage:
                        related = [l for l in result.library_usage[p.entity_id]]
                        if related: related_info = f"Uses Libs: {', '.join(related)}"
                    elif p.entity_id in result.node_allocations:
                        related = [a for a in result.node_allocations[p.entity_id]]
                        if related: related_info = f"Hosts Apps: {', '.join(related)}"
                    elif p.entity_id in result.broker_routing:
                        related = [t for t in result.broker_routing[p.entity_id]]
                        if related: related_info = f"Routes Topics: {', '.join(related)}"

                print(f"           Entity: {self.colored(entity_display, Colors.CYAN)} ({p.entity_type})")
                if related_info:
                    # Wrap related info if too long
                    if len(related_info) > 70: related_info = related_info[:67] + "..."
                    print(f"           Related: {self.colored(related_info, Colors.GRAY)}")
                
                print(f"           Category: {p.category}")
                desc_lines = self.wrap_text(p.description, 60)
                print(f"           Issue: {desc_lines[0]}")
                for line in desc_lines[1:]: print(f"                  {line}")
                rec_lines = self.wrap_text(p.recommendation, 60)
                print(f"           Fix: {self.colored(rec_lines[0], Colors.GREEN)}")
                for line in rec_lines[1:]: print(f"                {self.colored(line, self.Colors.GREEN)}")
            return

        self.print_subheader("Architectural Findings")
        problems = result.problems
        if not problems:
            self.console.print("  [success]✓ No architectural problems detected.[/]")
            return

        summary = result.problem_summary
        stats = []
        for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            count = summary.by_severity.get(sev, 0)
            if count > 0:
                stats.append(f"[{self.severity_color(sev)}]{sev}: {count}[/]")
        
        self.console.print(f"  Total: [bold]{summary.total_problems}[/] findings  |  " + "  ".join(stats))
        self.console.print()

        node_names = {c.id: c.structural.name for c in result.quality.components}
        
        for p in problems[:limit]:
            sev_style = self.severity_color(p.severity)
            
            # Build content group
            content = []
            entity_name = node_names.get(p.entity_id, "")
            entity_line = f"[cyan]{p.entity_id}[/] " + (f"({entity_name})" if entity_name else "")
            content.append(f"[bold]Entity:[/] {entity_line} [[grey70]{p.entity_type}[/]]")
            
            # Check for related components
            related_info = ""
            if p.entity_type == "Component":
                related = []
                if p.entity_id in result.library_usage:
                    related = [l for l in result.library_usage[p.entity_id]]
                    if related: related_info = f"Uses Libs: {', '.join(related)}"
                elif p.entity_id in result.node_allocations:
                    related = [a for a in result.node_allocations[p.entity_id]]
                    if related: related_info = f"Hosts Apps: {', '.join(related)}"
                elif p.entity_id in result.broker_routing:
                    related = [t for t in result.broker_routing[p.entity_id]]
                    if related: related_info = f"Routes Topics: {', '.join(related)}"
            
            if related_info:
                if len(related_info) > 80: related_info = related_info[:77] + "..."
                content.append(f"[bold]Related:[/] [grey70]{related_info}[/]")

            content.append(f"[bold]Category:[/] {p.category}")
            content.append(f"\n[bold red]Issue:[/] {p.description}")
            content.append(f"[bold green]Recommendation:[/] {p.recommendation}")
            
            panel = Panel(
                Group(*content),
                title=f"[{sev_style}]{p.severity}[/] | [bold white]{p.name}[/]",
                border_style=sev_style,
                expand=False,
                padding=(1, 2)
            )
            self.console.print(panel)
            self.console.print()

        if len(problems) > limit:
            self.console.print(f"  [grey50]... and {len(problems) - limit} more findings[/]")

    def display_antipatterns(self, problems: List["DetectedProblem"], layers: List[str], total_components: int) -> None:
        """Display anti-pattern detection findings."""
        self.print_header("Anti-Pattern & Bad Smell Detection")
        
        # Summary KPIs
        self.print_subheader("Scan Summary")
        print(f"  Layers analyzed:     {', '.join(layers)}")
        print(f"  Components scanned:  {total_components}")
        print(f"  Total smells found:  {self.colored(str(len(problems)), self.Colors.WHITE, bold=True)}")
        print()
        
        by_sev = {}
        for p in problems:
            by_sev[p.severity] = by_sev.get(p.severity, 0) + 1
            
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            count = by_sev.get(sev, 0)
            color = self.severity_color(sev)
            bar = self.colored("█" * min(count, 40), color) if count else self.colored("─", self.Colors.GRAY)
            print(f"  {self.colored(f'{sev:<10}', color)}  {bar}  {self.colored(str(count), color, bold=True)}")
        print()

        if not problems:
            print(f"  {self.colored('✓  No smells found matching the active filters.', self.Colors.GREEN)}")
            return

        self.print_subheader(f"Findings ({len(problems)})")

        prev_sev = None
        for i, problem in enumerate(problems, 1):
            if problem.severity != prev_sev:
                color = self.severity_color(problem.severity)
                print(f"\n  {self.colored(f'--- {problem.severity} ---', color, bold=True)}")
                prev_sev = problem.severity

            color = self.severity_color(problem.severity)
            print(f"\n  {self.colored(f'#{i:02d}', self.Colors.WHITE, bold=True)}  "
                  f"{self.colored(f'[{problem.name}]', color, bold=True)}"
                  f"  {self.colored(problem.entity_id, self.Colors.WHITE, bold=True)} "
                  f"({problem.entity_type})")

            print(f"       {self.colored('Description:', self.Colors.WHITE, bold=True)} {problem.description}")
            ev_parts = [f"{k}={v}" for k, v in list(problem.evidence.items())[:4]]
            print(f"       {self.colored('Evidence:   ', self.Colors.WHITE, bold=True)} {self.colored(', '.join(ev_parts), self.Colors.GRAY)}")
            print(f"       {self.colored('Fix:        ', self.Colors.WHITE, bold=True)} {self.colored(problem.recommendation, self.Colors.GREEN)}")

    def display_sensitivity(self, result: "LayerAnalysisResult") -> None:
        """Display weight sensitivity analysis results."""
        sensitivity = result.quality.sensitivity
        if not sensitivity:
            return

        if not RICH_AVAILABLE:
            self.print_subheader("Reliability & Sensitivity Analysis")
            stability = sensitivity.get("top5_stability", 0.0)
            mean_tau = sensitivity.get("mean_kendall_tau", 0.0)
            status = "STABLE" if stability > 0.8 else "MODERATE" if stability > 0.5 else "UNSTABLE"
            color = Colors.GREEN if status == "STABLE" else Colors.YELLOW if status == "MODERATE" else Colors.RED
            print(f"  {'Ranking Stability:':<25} {self.colored(status, color, bold=True)} ({stability*100:.1f}% top-5 consistency)")
            print(f"  {'Mean Kendall Tau:':<25} {mean_tau:.4f} (Ranking similarity under noise)")
            if stability < 0.5: print(f"\n  {self.colored('⚠ Warning:', Colors.YELLOW)} Ranking is sensitive to weight changes. Results should be interpreted with caution.")
            else: print(f"\n  {self.colored('✓ Info:', Colors.GREEN)} Ranking is robust to minor weight fluctuations.")
            return

        self.print_subheader("Reliability & Sensitivity Analysis")
        stability = sensitivity.get("top5_stability", 0.0)
        mean_tau = sensitivity.get("mean_kendall_tau", 0.0)
        
        status = "STABLE" if stability > 0.8 else "MODERATE" if stability > 0.5 else "UNSTABLE"
        color = "green" if status == "STABLE" else "yellow" if status == "MODERATE" else "red"
        
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        table.add_row("Ranking Stability", f"[{color} bold]{status}[/] ({stability*100:.1f}% Top-5 consistency)")
        table.add_row("Mean Kendall Tau", f"[white]{mean_tau:.4f}[/] (Ranking similarity under noise)")
        self.console.print(table)
        
        if stability < 0.5:
            self.console.print(Panel("Ranking is sensitive to weight changes. Results should be interpreted with caution.", title="[warning]⚠ Warning[/]", border_style="yellow"))
        else:
            self.console.print(Panel("Ranking is robust to minor weight fluctuations.", title="[success]✓ Info[/]", border_style="green"))

    def display_library_usage(self, result: "LayerAnalysisResult") -> None:
        """Display library usage by applications."""
        if not result.library_usage:
            return

        self.print_subheader("Library Usage")
        # Get component names for nicer display
        comp_map = {c.id: c.structural.name for c in result.quality.components}
        
        sorted_apps = sorted(result.library_usage.keys())
        for app_id in sorted_apps:
            libs = result.library_usage[app_id]
            app_name = comp_map.get(app_id, app_id)
            print(f"  {self.colored(app_name, Colors.WHITE, bold=True)} ({app_id})")
            for lib in sorted(libs):
                print(f"    - {lib}")
            print()
    def display_gnn_prediction(self, prediction: Dict[str, Any], limit: int = 10) -> None:
        """Display GNN or Ensemble prediction results."""
        if not prediction:
            return

        if not RICH_AVAILABLE:
            is_ensemble = bool(prediction.get("ensemble_scores"))
            title = "GNN/Ensemble Criticality Prediction" if is_ensemble else "GNN Criticality Prediction"
            self.print_subheader(title)
            node_scores = prediction.get("ensemble_scores") or prediction.get("node_scores")
            if not node_scores:
                print(f"  {self.colored('No prediction scores available.', Colors.GRAY)}")
                return
            sorted_nodes = sorted(node_scores.values(), key=lambda s: s.get("composite_score", 0), reverse=True)
            header = f"  {'#':<4} {'Component':<30} {'Score':>7} {'Level':<10} {'R':>6} {'M':>6} {'A':>6} {'V':>6}"
            print(self.colored(header, Colors.WHITE, bold=True)); print("  " + "-" * 78)
            for i, s in enumerate(sorted_nodes[:limit], 1):
                level = s.get("criticality_level", "MINIMAL")
                color = self.level_color(level)
                score_str = f"{s.get('composite_score', 0):>7.4f}"
                print(f"  {i:<4} {s.get('component', '')[:29]:<30} {self.colored(score_str, color)} {self.colored(level[:10], color):<10} {s.get('reliability_score', 0):>6.3f} {s.get('maintainability_score', 0):>6.3f} {s.get('availability_score', 0):>6.3f} {s.get('vulnerability_score', 0):>6.3f}")
            if len(sorted_nodes) > limit: print(f"\n  {self.colored(f'... and {len(sorted_nodes) - limit} more components', Colors.GRAY)}")
            return

        is_ensemble = bool(prediction.get("ensemble_scores"))
        title = "GNN/Ensemble Criticality Prediction" if is_ensemble else "GNN Criticality Prediction"
        self.print_subheader(title)

        node_scores = prediction.get("ensemble_scores") or prediction.get("node_scores")
        if not node_scores:
            self.console.print("  [grey50]No prediction scores available.[/]")
            return

        sorted_nodes = sorted(node_scores.values(), key=lambda s: s.get("composite_score", 0), reverse=True)

        table = Table(box=box.MINIMAL_HEAVY_HEAD, header_style="bold cyan")
        table.add_column("#", justify="center")
        table.add_column("Component", width=30)
        table.add_column("Score", justify="right")
        table.add_column("Level", justify="left")
        table.add_column("R", justify="right")
        table.add_column("M", justify="right")
        table.add_column("A", justify="right")
        table.add_column("V", justify="right")

        for i, s in enumerate(sorted_nodes[:limit], 1):
            lvl = s.get("criticality_level", "MINIMAL").upper()
            color = self.level_color(lvl)
            table.add_row(
                str(i),
                s.get('component', '')[:29],
                Text(f"{s.get('composite_score', 0):.4f}", style=color),
                Text(lvl.capitalize(), style=color),
                f"{s.get('reliability_score', 0):.3f}",
                f"{s.get('maintainability_score', 0):.3f}",
                f"{s.get('availability_score', 0):.3f}",
                f"{s.get('vulnerability_score', 0):.3f}"
            )
        self.console.print(table)

        if len(sorted_nodes) > limit:
            self.console.print(f"\n  [grey50]... and {len(sorted_nodes) - limit} more components[/]")
            
        edge_scores = prediction.get("edge_scores", [])
        if edge_scores:
            self.print_subheader("Critical Relationships (GNN)")
            sorted_edges = sorted(edge_scores, key=lambda e: e.get("composite_score", 0), reverse=True)
            e_table = Table(box=box.SIMPLE, header_style="bold grey70")
            e_table.add_column("#")
            e_table.add_column("Source")
            e_table.add_column("Target")
            e_table.add_column("Type")
            e_table.add_column("Score", justify="right")
            
            for i, e in enumerate(sorted_edges[:limit], 1):
                lvl = e.get("criticality_level", "MINIMAL").upper()
                color = self.level_color(lvl)
                e_table.add_row(
                    str(i),
                    e.get('source', '')[:20],
                    e.get('target', '')[:20],
                    f"[cyan]{e.get('edge_type', '')[:15]}[/]",
                    Text(f"{e.get('composite_score', 0):.4f}", style=color)
                )
            self.console.print(e_table)

    def display_layer_result(self, result: "LayerAnalysisResult") -> None:
        """Display complete analysis result for a single layer."""
        if not RICH_AVAILABLE:
            self.print_header(f"{result.layer_name} Analysis")
            print(f"  {self.colored(result.description, Colors.GRAY)}")
            has_topics = any(c.type == "Topic" for c in result.structural.components.values())
            if not has_topics and result.layer in ("app", "mw"):
                self.print_warning(f"fan_out_criticality (FOC) is restricted in {result.layer} layer. Use 'system' layer for topic fan-out analysis.")
            self.display_graph_summary(result)
            self.display_classification_summary(result)
            self.display_layer_critical_components(result)
            self.display_sensitivity(result)
            if result.prediction:
                self.display_gnn_prediction(result.prediction)
            self.display_problems(result)
            return

        self.print_header(f"{result.layer_name} Layer Analysis")
        
        # Description and metadata panel
        info_content = [
            f"[bold cyan]Description:[/] {result.description}",
            f"[bold cyan]Timestamp:[/]   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        # Check for FOC visibility warning
        has_topics = any(c.type == "Topic" for c in result.structural.components.values())
        if not has_topics and result.layer in ("app", "mw"):
            info_content.append(f"\n[warning]⚠ Fan-out criticality (FOC) is restricted in {result.layer} layer.[/]")
            info_content.append("[grey50]Use 'system' layer for comprehensive topic fan-out analysis.[/]")

        self.console.print(Panel(Group(*info_content), border_style="grey23"))

        self.display_graph_summary(result)
        self.display_classification_summary(result)
        self.display_layer_critical_components(result)
        self.display_sensitivity(result)
        if result.prediction:
            self.display_gnn_prediction(result.prediction)
        self.display_problems(result)

    def display_multi_layer_analysis_result(self, results: "MultiLayerAnalysisResult") -> None:
        """Display analysis results for multiple layers."""
        self.print_header("Multi-Layer Analysis Results", "═")
        print(f"\n  Timestamp: {results.timestamp}\n  Layers: {', '.join(results.layers.keys())}")
        for layer_result in results.layers.values():
            self.display_layer_result(layer_result)
        if results.cross_layer_insights:
            self.print_subheader("Cross-Layer Insights")
            for insight in results.cross_layer_insights: print(f"  • {insight}")
        self.print_subheader("Overall Summary")
        summary = results.summary
        print(f"  Layers analyzed:     {summary['layers_analyzed']}\n  Total components:    {summary['total_components']}\n  Total problems:      {summary['total_problems']}")
        if summary.get('critical_problems', 0) > 0:
            print(f"  Critical problems:   {self.colored(str(summary['critical_problems']), Colors.RED, bold=True)}")

    # --- Simulation Display ---

    def display_event_result(self, result: "EventResult") -> None:
        """Display event simulation results."""
        names = result.component_names
        get_name = lambda cid: f"{cid} ({names.get(cid, cid)})" if cid in names else cid
        self.print_header(f"Event Simulation: {get_name(result.source_app)}")
        print(f"\n  {self.colored('Scenario:', Colors.CYAN)}     {result.scenario}\n  {self.colored('Duration:', Colors.CYAN)}     {result.duration * 1000:.2f} ms")
        self.print_subheader("Throughput Metrics")
        m = result.metrics
        print(f"\n  Messages Published:  {m.messages_published}\n  Messages Delivered:  {self.colored(str(m.messages_delivered), Colors.GREEN)}")
        drop_color = Colors.RED if m.messages_dropped > 0 else Colors.GRAY
        print(f"  Messages Dropped:    {self.colored(str(m.messages_dropped), drop_color)}\n  Delivery Rate:       {m.delivery_rate:.1f}%\n  Drop Rate:           {m.drop_rate:.1f}%")
        self.print_subheader("Latency Metrics")
        print(f"\n  Average Latency:     {m.avg_latency * 1000:.3f} ms\n  P99 Latency:         {m.p99_latency * 1000:.3f} ms\n  Throughput:          {m.throughput:.1f} msg/sec")

    def display_failure_result(self, result: "FailureResult") -> None:
        """Display failure simulation results."""
        names = result.component_names
        get_name = lambda cid: f"{cid} ({names.get(cid, cid)})" if cid in names else cid
        self.print_header(f"Failure Simulation: {get_name(result.target_id)}")
        print(f"\n  {self.colored('Target:', Colors.CYAN)}       {get_name(result.target_id)} ({result.target_type})")
        
        if result.related_components:
            related_str = ", ".join(result.related_components)
            if len(related_str) > 80: related_str = related_str[:77] + "..."
            print(f"  {self.colored('Related:', Colors.CYAN)}      {self.colored(related_str, Colors.GRAY)}")
            
        print(f"  {self.colored('Scenario:', Colors.CYAN)}     {result.scenario}")
        self.print_subheader("Impact Metrics")
        impact = result.impact
        imp_color = Colors.RED if impact.composite_impact > 0.5 else (Colors.YELLOW if impact.composite_impact > 0.2 else Colors.GREEN)
        print(f"\n  {self.colored('Composite Impact:', Colors.WHITE, bold=True)} {self.colored(f'{impact.composite_impact:.4f}', imp_color)}")
        print(f"\n  Reachability Loss:   {impact.reachability_loss * 100:.1f}%\n  Fragmentation:       {impact.fragmentation * 100:.1f}%\n  Throughput Loss:     {impact.throughput_loss * 100:.1f}%")
        self.print_subheader("Cascade Analysis")
        print(f"\n  Cascade Count:       {impact.cascade_count}\n  Cascade Depth:       {impact.cascade_depth}")
        if result.cascaded_failures:
            self.print_subheader("Cascaded Failures")
            for i, cid in enumerate(result.cascaded_failures[:10]): print(f"  {i+1}. {get_name(cid)}")

    def display_exhaustive_results(self, results: List["FailureResult"], limit: int = 15) -> None:
        """Display exhaustive failure simulation results."""
        self.print_header("Exhaustive Failure Analysis")
        impacts = [r.impact.composite_impact for r in results]
        print(f"\n  Total Components Analyzed: {len(results)}\n  Average Impact:           {sum(impacts)/len(impacts):.4f}\n  Maximum Impact:           {max(impacts):.4f}")
        self.print_subheader(f"Top {limit} Components by Impact")
        print(f"\n  {'Component':<30} {'Type':<12} {'Impact':<10} {'Cascade':<10}")
        print(f"  {'-' * 65}")
        for r in results[:limit]:
            r_name = f"{r.target_id} ({r.component_names.get(r.target_id, r.target_id)})"
            imp = r.impact.composite_impact
            color = Colors.RED if imp > 0.5 else (Colors.YELLOW if imp > 0.2 else Colors.GREEN)
            print(f"  {r_name:<30} {r.target_type:<12} {self.colored(f'{imp:.4f}', color):<10} {r.impact.cascade_count:<10}")
            
            if r.related_components:
                related_str = ", ".join(r.related_components)
                if len(related_str) > 70: related_str = related_str[:67] + "..."
                print(f"    {self.colored('↳ ', Colors.GRAY)} {self.colored(related_str, Colors.GRAY)}")

    def display_simulation_report(self, report: "SimulationReport") -> None:
        """Display comprehensive simulation report."""
        self.print_header("Simulation Report", "═")
        print(f"\n  {self.colored('Timestamp:', Colors.CYAN)}  {report.timestamp}")
        for layer_name, metrics in report.layer_metrics.items():
            self.print_subheader(f"Layer: {layer_name.upper()}")
            print(f"\n  {self.colored('Event Simulation:', Colors.CYAN)}\n    Throughput:        {metrics.event_throughput} messages\n    Delivery Rate:     {metrics.event_delivery_rate:.1f}%")
            print(f"\n  {self.colored('Failure Simulation:', Colors.CYAN)}\n    Avg Reach Loss:    {metrics.avg_reachability_loss * 100:.1f}%\n    Max Impact:        {metrics.max_impact:.4f}")
            print(f"\n  {self.colored('Criticality:', Colors.CYAN)}\n    Critical:          {self.colored(str(metrics.critical_count), Colors.RED)}\n    SPOFs:             {metrics.spof_count}")

        if report.library_usage:
            self.print_subheader("Library Usage")
            sorted_apps = sorted(report.library_usage.keys())
            for app_id in sorted_apps:
                libs = report.library_usage[app_id]
                app_name = report.component_names.get(app_id, app_id)
                print(f"  {self.colored(app_name, Colors.WHITE, bold=True)} ({app_id})")
                for lib in sorted(libs):
                    print(f"    - {lib}")
            print()
            
        if report.node_allocations:
            self.print_subheader("Node Allocations")
            sorted_nodes = sorted(report.node_allocations.keys())
            for node_id in sorted_nodes:
                apps = report.node_allocations[node_id]
                node_name = report.component_names.get(node_id, node_id)
                print(f"  {self.colored(node_name, Colors.WHITE, bold=True)} ({node_id})")
                for app in sorted(apps):
                    print(f"    - {app}")
            print()
            
        if report.broker_routing:
            self.print_subheader("Broker Routing")
            sorted_brokers = sorted(report.broker_routing.keys())
            for broker_id in sorted_brokers:
                topics = report.broker_routing[broker_id]
                broker_name = report.component_names.get(broker_id, broker_id)
                print(f"  {self.colored(broker_name, Colors.WHITE, bold=True)} ({broker_id})")
                for topic in sorted(topics):
                    print(f"    - {topic}")
            print()

    # --- Validation Display ---

    def status_text(self, passed: bool) -> str:
        return self.colored("PASSED", Colors.GREEN, bold=True) if passed else self.colored("FAILED", Colors.RED, bold=True)

    def metric_color(self, value: float, target: float, higher_better: bool = True) -> str:
        passed = value >= target if higher_better else value <= target
        if passed: return Colors.GREEN
        return Colors.YELLOW if (higher_better and value >= target * 0.9) or (not higher_better and value <= target * 1.1) else Colors.RED

    def display_pipeline_validation_result(self, result: "PipelineResult") -> None:
        """Display full validation results."""
        self.print_header("VALIDATION PIPELINE RESULTS", "═")
        print(f"\n  Timestamp:      {result.timestamp}\n  Components:     {result.total_components}")
        print(f"\n  Overall Status: {self.status_text(result.all_passed)}\n  Layers Validated: {len(result.layers)}\n  Layers Passed:    {result.layers_passed}")
        self.print_subheader("Layer Summary")
        print(f"\n  {'Layer':<12} {'N':<6} {'Spearman':<10} {'F1':<10} {'Status':<10}")
        print(f"  {'-' * 48}")
        targets = result.targets
        for layer_name, layer_res in result.layers.items():
            s_color = self.metric_color(layer_res.spearman, targets.spearman)
            f_color = self.metric_color(layer_res.f1_score, targets.f1_score)
            print(f"  {layer_name:<12} {layer_res.matched_components:<6} {self.colored(f'{layer_res.spearman:>8.4f}', s_color)}   {self.colored(f'{layer_res.f1_score:>8.4f}', f_color)}   {self.status_text(layer_res.passed)}")
        for layer_res in result.layers.values():
            self.display_layer_validation_result(layer_res, targets)

    def display_layer_validation_result(self, result: "LayerValidationResult", targets: "ValidationTargets") -> None:
        """Display detailed layer validation results."""
        self.print_subheader(f"Layer: {result.layer_name} ({result.layer})")
        print(f"\n  Matched Components:    {result.matched_components}")
        if result.warnings:
            print(f"\n  {self.colored('Warnings:', Colors.YELLOW)}")
            for w in result.warnings: print(f"    ⚠ {w}")
        if not result.validation_result: return
        ov = result.validation_result.overall
        print(f"\n  {self.colored('Correlation Metrics:', Colors.CYAN)}")
        print(f"    {'Spearman ρ':<15} {ov.correlation.spearman:>8.4f}     ≥{targets.spearman:<8.2f}")
        print(f"\n  {self.colored('Classification Metrics:', Colors.CYAN)}")
        print(f"    {'F1 Score':<15} {ov.classification.f1_score:>8.4f}     ≥{targets.f1_score:<8.2f}")
        cm = ov.classification.confusion_matrix
        print(f"\n    Confusion Matrix:\n                            Actual Critical    Actual Non-Critical\n      Pred Critical           {self.colored(str(cm['tp']), Colors.GREEN):>3}                {self.colored(str(cm['fp']), Colors.RED):>3}\n      Pred Non-Critical       {self.colored(str(cm['fn']), Colors.RED):>3}                {self.colored(str(cm['tn']), Colors.GREEN):>3}")
        print(f"\n  {self.colored('Error Metrics:', Colors.CYAN)}\n    RMSE: {ov.error.rmse:.4f} {'≤' if ov.error.rmse <= targets.rmse_max else '>'} {targets.rmse_max}")
    def display_gate_verdicts(self, gates: Dict[str, bool]) -> None:
        """Display G1-G8 gate verdicts."""
        if not gates:
            return
            
        print(f"\n  {self.colored('Gate Verdicts:', Colors.WHITE, bold=True)}")
        
        tier1_keys = ["G1_spearman", "G2_pvalue", "G3_f1", "G4_top5"]
        tier1 = {k: v for k, v in gates.items() if k in tier1_keys}
        tier2 = {k: v for k, v in gates.items() if k not in tier1_keys}
        
        print("    Tier 1 (Critical):", end=" ")
        for g, passed in tier1.items():
            status = "✓" if passed else "✗"
            color = Colors.GREEN if passed else Colors.RED
            label = g.split('_')[0]
            print(self.colored(f"{label}:{status}", color), end="  ")
        
        print("\n    Tier 2 (Reported):", end=" ")
        for g, passed in tier2.items():
            status = "✓" if passed else "✗"
            # Tier 2 failure is a warning, not a critical error
            color = Colors.CYAN if passed else Colors.YELLOW
            label = g.split('_')[0]
            print(self.colored(f"{label}:{status}", color), end="  ")
        print()

    def display_import_summary(self, stats: Any) -> None:
        """Display summary of graph import results."""
        self.print_subheader("Graph Import Summary")
        
        # Support both dict and ImportStats object
        if hasattr(stats, "to_dict"):
            data = stats.to_dict()
        elif isinstance(stats, dict):
            data = stats
        else:
            data = {}

        print(f"  {'Status:':<20} {self.colored('SUCCESS', Colors.GREEN) if data.get('success') else self.colored('FAILED', Colors.RED)}")
        print(f"  {'Total Nodes:':<20} {data.get('nodes_imported', 0)}")
        print(f"  {'Total Edges:':<20} {data.get('edges_imported', 0)}")
        print(f"  {'Duration:':<20} {data.get('duration_ms', 0):.2f} ms")
        
        # breakdown by label/type
        print(f"\n  {'Entity Breakdown:':<20}")
        entities = ["application", "broker", "node", "topic", "library"]
        for ent in entities:
            count = data.get(f"{ent}_count", 0)
            if count > 0:
                print(f"    - {ent.capitalize():<18} {count}")
                
        print(f"\n  {'Relationship Breakdown:':<20}")
        rels = ["runs_on", "routes", "publishes_to", "subscribes_to", "connects_to", "uses"]
        for rel in rels:
            count = data.get(f"{rel}_count", 0)
            if count > 0:
                formatted_rel = rel.replace("_", " ").title()
                print(f"    - {formatted_rel:<18} {count}")
                
        print(f"\n  {'Derived Dependencies:':<20}")
        deps = ["app_to_app", "app_to_lib", "app_to_broker", "node_to_node", "node_to_broker", "broker_to_broker"]
        for dep in deps:
            count = data.get(f"{dep}_count", 0)
            if count > 0:
                formatted_dep = dep.replace("_", " ").title()
                print(f"    - {formatted_dep:<18} {count}")


    def display_top_critical_components(self, components: List[Any], n: int = 10) -> None:
        """Display top-N critical components with their RMAV breakdown and cascade metrics."""
        if not components:
            return
            
        self.print_subheader(f"Top {min(len(components), n)} Critical Components (Ensemble)")
        
        header = f"  {'Rank':<5} {'Component':<25} {'Score':>7} {'Lvl':<9} {'R':>5} {'M':>5} {'A':>5} {'V':>5} {'Dom':<4} {'B/D':<6} {'S'}"
        print(self.colored(header, Colors.WHITE, bold=True))
        print("  " + "-" * 88)
        
        for i, s in enumerate(components[:n], 1):
            # Polymorphic attribute access (handles internal model and saag SDK facade)
            level = getattr(s, "criticality_level", "MINIMAL")
            color = self.level_color(level)
            
            comp_id = getattr(s, "id", "unknown")
            comp_name = getattr(s, "component", comp_id)
            if len(comp_name) > 24:
                comp_name = comp_name[:21] + "..."
            
            # Scores (facade uses .scores dict, internal uses direct attributes)
            if hasattr(s, "scores") and isinstance(s.scores, dict):
                comp_s = s.scores.get("overall", 0.0)
                r_s = s.scores.get("reliability", 0.0)
                m_s = s.scores.get("maintainability", 0.0)
                a_s = s.scores.get("availability", 0.0)
                v_s = s.scores.get("vulnerability", 0.0)
            else:
                comp_s = getattr(s, "composite_score", 0.0)
                r_s = getattr(s, "reliability_score", 0.0)
                m_s = getattr(s, "maintainability_score", 0.0)
                a_s = getattr(s, "availability_score", 0.0)
                v_s = getattr(s, "vulnerability_score", 0.0)
            
            # Dominant dimension logic
            dims = {"R": r_s, "M": m_s, "A": a_s, "V": v_s}
            dominant = max(dims, key=dims.get)
            
            # Structural info (blast radius, depth, SPOF)
            blast = 0
            depth = 0
            is_spof = False
            
            if hasattr(s, "structural") and s.structural:
                # saag facade
                blast = getattr(s.structural, "blast_radius", 0)
                depth = getattr(s.structural, "cascade_depth", 0)
                is_spof = getattr(s.structural, "is_articulation_point", False)
            elif hasattr(s, "blast_radius"):
                # GNN NodeScore direct item
                blast = s.blast_radius
                depth = s.cascade_depth
                is_spof = getattr(s, "is_articulation_point", False)
                
            spof_flag = self.colored("●", Colors.RED) if is_spof else " "
            bd_str = f"{blast}/{depth}"

            print(
                f"  {i:<5} {comp_name:<25} "
                f"{self.colored(f'{comp_s:>7.4f}', color)} "
                f"{self.colored(f'{level[:8]:<9}', color)} "
                f"{r_s:>5.2f} {m_s:>5.2f} "
                f"{a_s:>5.2f} {v_s:>5.2f} "
                f"{self.colored(dominant, Colors.CYAN):<4} "
                f"{bd_str:<6} "
                f"{spof_flag}"
            )

    def display_training_summary(self, summary: Dict[str, Any]) -> None:
        """Display summary of GNN training results (ensemble levels)."""
        self.print_subheader(f"Training Results ({summary.get('total_components', 0)} components)")
        
        for lvl in ["critical", "high", "medium", "low", "minimal"]:
            count = summary.get(lvl, 0)
            bar_len = min(count * 2, 40)
            bar = "█" * bar_len
            color = self.level_color(lvl)
            print(f"  {lvl.upper():<9} {count:>3}  {self.colored(bar, color)}")

    def display_training_metrics(self, metrics: Dict[str, Any]) -> None:
        """Display validation metrics in a clean list."""
        if not metrics:
            return
            
        print(f"\n  {self.colored('Validation Metrics (Test Set):', Colors.CYAN)}")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    - {k:<25} {v:.4f}")
            else:
                print(f"    - {k:<25} {v}")

    def display_prediction_summary(self, result: Any) -> None:
        """Display summary of GNN prediction results."""
        # result can be saag.models.PredictionResult or src.prediction.models.QualityAnalysisResult
        components = []
        if hasattr(result, "all_components"):
            components = result.all_components
        elif hasattr(result, "components"):
            components = result.components
            
        if not components:
            return

        self.print_subheader(f"Prediction Results ({len(components)} components)")
        
        # Count levels
        counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "MINIMAL": 0}
        for c in components:
            level = ""
            if hasattr(c, "criticality_level"):
                level = c.criticality_level.upper()
            elif hasattr(c, "levels") and hasattr(c.levels, "overall"):
                level = c.levels.overall.value.upper()
            
            if level in counts:
                counts[level] += 1
            else:
                counts["MINIMAL"] += 1
                
        for lvl in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL"]:
            count = counts[lvl]
            bar = "█" * min(count * 2, 40)
            color = self.level_color(lvl)
            print(f"  {lvl:<9} {count:>3}  {self.colored(bar, color)}")

        # Display Top 10
        sorted_comps = sorted(components, key=lambda c: getattr(c, "composite_score", getattr(c, "rmav_score", 0.0)), reverse=True)
        self.display_top_critical_components(sorted_comps, n=10)

    def display_simulation_summary(self, report: Any) -> None:
        """Display summary of failure simulation results."""
        if not report:
            return
            
        data = report
        if hasattr(report, "to_dict"):
            data = report.to_dict()
            
        # Case 1: SimulationReport dict/object
        if isinstance(data, dict) and "component_criticality" in data:
            comps = data.get("component_criticality", [])
            if not comps: return
            
            self.print_subheader(f"Simulation Results ({len(comps)} components)")
            
            counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "minimal": 0}
            for c in comps:
                lvl = c.get("level", "minimal").lower()
                if lvl in counts: counts[lvl] += 1
                    
            for lvl in ["critical", "high", "medium", "low", "minimal"]:
                count = counts[lvl]
                bar = "█" * min(count * 2, 40)
                print(f"  {lvl.upper():<9} {count:>3}  {self.colored(bar, self.level_color(lvl))}")
                
            self.print_subheader("Top 10 Most Impactful Components (Simulation)")
            header = f"  {'Rank':<5} {'Component':<30} {'Impact':>7} {'Reach.':>7} {'Casc.':>7}"
            print(self.colored(header, Colors.WHITE, bold=True))
            print("  " + "-" * 59)
            for i, c in enumerate(comps[:10], 1):
                level = c.get("level", "minimal")
                impact_val = c.get("combined_impact", 0.0)
                impact_str = f"{impact_val:>7.4f}"
                print(
                    f"  {i:<5} {c.get('id', 'unknown')[:29]:<30} "
                    f"{self.colored(impact_str, self.level_color(level))} "
                    f"{c.get('reachability_loss', 0.0):>7.4f} {c.get('cascade_count', 0):>7}"
                )
            
            recs = data.get("recommendations", [])
            if recs:
                print(f"\n  {self.colored('Recommendations:', Colors.YELLOW)}")
                for rec in recs: print(f"    - {rec}")
            return

        # Case 2: List of FailureResult objects
        if isinstance(report, list):
            self.print_subheader(f"Simulation Results ({len(report)} scenarios)")
            
            # Simple list view for exhaustive run
            header = f"  {'Rank':<5} {'Target':<30} {'Impact':>7} {'Reach.':>7} {'Casc.':>7}"
            print(self.colored(header, Colors.WHITE, bold=True))
            print("  " + "-" * 59)
            
            # Sort by impact
            sorted_results = sorted(report, key=lambda r: getattr(getattr(r, "impact", r), "composite_impact", 0.0), reverse=True)
            for i, r in enumerate(sorted_results[:10], 1):
                impact_obj = getattr(r, "impact", r)
                imp = getattr(impact_obj, "composite_impact", 0.0)
                reach = getattr(impact_obj, "reachability_loss", 0.0)
                cascade = getattr(impact_obj, "cascade_count", 0)
                
                target = getattr(r, "target_id", "Multiple")
                level = "minimal"
                if imp > 0.8: level = "critical"
                elif imp > 0.6: level = "high"
                elif imp > 0.4: level = "medium"
                elif imp > 0.2: level = "low"
                
                impact_str = f"{imp:>7.4f}"
                print(
                    f"  {i:<5} {target[:29]:<30} "
                    f"{self.colored(impact_str, self.level_color(level))} "
                    f"{reach:>7.4f} {cascade:>7}"
                )

    def display_validation_summary(self, result: Any) -> None:
        """Display summary of validation pipeline results."""
        # result is typically saag.models.ValidationPipelineFacade
        if not result:
            return
            
        layers_dict = {}
        if hasattr(result, "layers"):
            layers_dict = result.layers
        elif isinstance(result, dict):
            layers_dict = result.get("layers", result)
            
        if not layers_dict:
            return

        self.print_subheader(f"Validation Results ({len(layers_dict)} layers)")
        
        header = f"  {'Layer':<15} {'Spearman ρ':>12} {'F1 Score':>12} {'Status':<10}"
        print(self.colored(header, Colors.WHITE, bold=True))
        print("  " + "-" * 55)
        
        for layer_name, res in layers_dict.items():
            rho = getattr(res, "spearman_rho", 0.0)
            f1 = getattr(res, "f1_score", 0.0)
            
            # Simple heuristic for status
            status = "PASSED" if rho > 0.6 and f1 > 0.6 else "WEAK"
            status_color = Colors.GREEN if status == "PASSED" else Colors.YELLOW
            
            print(
                f"  {layer_name:<15} "
                f"{rho:>12.4f} {f1:>12.4f} "
                f"{self.colored(f'{status:<10}', status_color)}"
            )

    def display_visualization_summary(self, filepath: str) -> None:
        """Display success message for dashboard generation."""
        abs_path = os.path.abspath(filepath)
        size_kb = os.path.getsize(abs_path) / 1024
        
        self.print_subheader("Visualization Generated")
        print(f"  {self.colored('Dashboard:', Colors.CYAN)} {abs_path}")
        print(f"  {self.colored('File Size:', Colors.CYAN)} {size_kb:.0f} KB")
        print(f"  {self.colored('Status:', Colors.GREEN)} Ready for review")
    def display_structural_summary(self, summary: Dict[str, Any]) -> None:
        """Display summary of structural graph analysis results."""
        # Handle both flat dict and nested dict (from LayerAnalysisResult)
        gs = summary.get("graph_summary", summary)
        layer_name = summary.get("layer", gs.get("layer", "unknown"))
        
        self.print_subheader(f"Structural analysis: Layer '{layer_name}'")
        
        # Get counts safely (handle cases where they might be lists or ints)
        def get_count(target, key):
            val = target.get(key, 0)
            return len(val) if isinstance(val, list) else val

        nodes_count = get_count(gs, "nodes")
        edges_count = get_count(gs, "edges")
        
        col1 = 25
        print(f"  {'Total Nodes:':<{col1}} {nodes_count}")
        print(f"  {'Total Edges:':<{col1}} {edges_count}")
        print(f"  {'Graph Density:':<{col1}} {gs.get('density', 0.0):.4f}")
        print(f"  {'Avg Degree:':<{col1}} {gs.get('avg_degree', 0.0):.2f}")
        print(f"  {'Avg Clustering:':<{col1}} {gs.get('avg_clustering', 0.0):.4f}")
        
        print(f"\n  {'Connectivity Health:':<{col1}} {gs.get('connectivity_health', 'unknown')}")
        print(f"  {'Num. Components:':<{col1}} {gs.get('num_components', 0)}")
        print(f"  {'Is Connected:':<{col1}} {gs.get('is_connected', False)}")
        
        print(f"\n  {'Resilience Indicators:':<{col1}}")
        ap_count = gs.get('num_articulation_points', 0)
        br_count = gs.get('num_bridges', 0)
        
        spof_ratio = ap_count / nodes_count if nodes_count > 0 else 0.0
        bridge_ratio = br_count / edges_count if edges_count > 0 else 0.0
        
        print(f"    - {'Articulation Points:':<22} {ap_count} (SPOF Ratio: {spof_ratio:.1%})")
        print(f"    - {'Bridges:':<22} {br_count} (Bridge Ratio: {bridge_ratio:.1%})")
        
        print(f"\n  {'Mixing Pattern:':<{col1}} {gs.get('assortativity_pattern', 'unknown')} (r={gs.get('assortativity', 0.0):.3f})")

    def display_top_components(self, components: List[Dict[str, Any]], metric: str = "betweenness", n: int = 5) -> None:
        """Display top-N components sorted by a specific structural metric."""
        if not components:
            return
            
        self.print_subheader(f"Top {n} components by {metric.title()}")
        
        # Sort and take top N
        top_n = sorted(components, key=lambda c: c.get(metric, 0), reverse=True)[:n]
        
        header = f"  {'ID':<{30}} {'Type':<{15}} {metric.title():<{15}}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        
        for comp in top_n:
            comp_id = comp.get("id", "unknown")
            if len(comp_id) > 28:
                comp_id = comp_id[:25] + "..."
            print(f"  {comp_id:<30} {comp.get('type', 'Unknown'):<15} {comp.get(metric, 0.0):<15.4f}")

    def display_graph_data_summary(self, graph_data: Dict[str, Any], title: str = "Graph Data Summary") -> None:
        """Display summary of a graph data dictionary (nodes/relationships format)."""
        self.print_subheader(title)
        
        # In this project's generator, components are separated by key
        main_components = {
            "Nodes (Compute)": len(graph_data.get("nodes", [])),
            "Brokers": len(graph_data.get("brokers", [])),
            "Topics": len(graph_data.get("topics", [])),
            "Applications": len(graph_data.get("applications", [])),
            "Libraries": len(graph_data.get("libraries", [])),
        }
        
        total_entities = sum(main_components.values())
        print(f"  {'Total Entities:':<20} {total_entities}")
        for label, count in main_components.items():
            if count > 0:
                print(f"    - {label:<18} {count}")
            
        relationships = graph_data.get("relationships", {})
        # Separate structural edges (imported) from derived DEPENDS_ON (computed at import time)
        structural = {k: v for k, v in relationships.items() if k != "depends_on"}
        derived = relationships.get("depends_on", [])

        total_structural = sum(len(rels) for rels in structural.values())
        print(f"\n  {'Structural Edges:':<20} {total_structural}")
        for rel_type, rel_list in sorted(structural.items()):
            if rel_list:
                formatted_rel = rel_type.replace("_", " ").title()
                print(f"    - {formatted_rel:<18} {len(rel_list)}")

        if derived:
            print(f"\n  {'Derived (DEPENDS_ON):':<20} {len(derived)} (informational snapshot)")

    def display_dimensional_results(self, dimensional_validation: Dict[str, Any]) -> None:
        """Display dimension-specific metrics (RMAV)."""
        if not dimensional_validation:
            return
            
        self.print_subheader("Dimensional Analysis Detail")
        
        # Reliability
        if r := dimensional_validation.get("reliability"):
            print(f"  {self.colored('Reliability [IR(v)]:', Colors.CYAN):<30} ρ={r['spearman']:.4f}, CCR@5={r['ccr_5']:.3f}, CME={r['cme']:.4f}")
            
        # Maintainability
        if m := dimensional_validation.get("maintainability"):
            print(f"  {self.colored('Maintainability [IM(v)]:', Colors.CYAN):<30} ρ={m['spearman']:.4f}, κ_CTA={m['weighted_kappa_cta']:.3f}, BP={m['bottleneck_precision']:.3f}")
            
        # Availability
        if a := dimensional_validation.get("availability"):
            print(f"  {self.colored('Availability [IA(v)]:', Colors.CYAN):<30} ρ={a['spearman']:.4f}, SPOF_F1={a['spof_f1']:.3f}, RRI={a['rri']:.3f}")
            
        # Vulnerability
        if v := dimensional_validation.get("vulnerability"):
            print(f"  {self.colored('Vulnerability [IV(v)]:', Colors.CYAN):<30} ρ={v['spearman']:.4f}, AHCR@5={v['ahcr_5']:.3f}, FTR={v['ftr']:.3f}")

        # Composite
        if c := dimensional_validation.get("composite"):
            pg = c.get("predictive_gain", 0)
            pg_colored = self.colored(f"{pg:+.3f}", Colors.GREEN if pg > 0 else Colors.RESET)
            print(f"  {self.colored('Composite [I*(v)]:', Colors.WHITE, bold=True):<30} ρ={c['spearman']:.4f}, Predictive Gain={pg_colored}")
