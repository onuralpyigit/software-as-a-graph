"""
Display Application Service
"""
import sys
from typing import List, Dict, Any, Optional, TYPE_CHECKING

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
        print(f"\n{self.colored(char * width, Colors.CYAN)}")
        print(f"{self.colored(f' {title} '.center(width), Colors.CYAN, bold=True)}")
        print(f"{self.colored(char * width, Colors.CYAN)}")

    def print_subheader(self, title: str, char: str = "-", width: int = 78) -> None:
        """Print a formatted subheader."""
        print(f"\n{self.colored(f' {title} ', Colors.WHITE, bold=True)}")
        print(f"{self.colored(char * width, Colors.GRAY)}")

    def print_step(self, msg: str) -> None:
        """Print a step indicator."""
        print(f"  {self.colored('→', Colors.BLUE)} {msg}")

    def print_success(self, msg: str) -> None:
        """Print a success indicator."""
        print(f"  {self.colored('✓', Colors.GREEN)} {msg}")

    def print_error(self, msg: str) -> None:
        """Print an error indicator."""
        print(f"  {self.colored('✗', Colors.RED)} {msg}")

    def print_warning(self, msg: str) -> None:
        """Print a warning indicator."""
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

    def display_classification_summary(self, result: "LayerAnalysisResult") -> None:
        """Display classification distribution."""
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

    def display_critical_components(self, result: "LayerAnalysisResult", limit: int = 15) -> None:
        """Display top critical components."""
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

    def display_problems(self, result: "LayerAnalysisResult", limit: int = 10) -> None:
        """Display detected problems."""
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

        self.print_subheader("Reliability & Sensitivity Analysis")
        
        # Stability metrics
        stability = sensitivity.get("top5_stability", 0.0)
        mean_tau = sensitivity.get("mean_kendall_tau", 0.0)
        
        status = "STABLE" if stability > 0.8 else "MODERATE" if stability > 0.5 else "UNSTABLE"
        color = Colors.GREEN if status == "STABLE" else Colors.YELLOW if status == "MODERATE" else Colors.RED
        
        print(f"  {'Ranking Stability:':<25} {self.colored(status, color, bold=True)} ({stability*100:.1f}% top-5 consistency)")
        print(f"  {'Mean Kendall Tau:':<25} {mean_tau:.4f} (Ranking similarity under noise)")
        
        if stability < 0.5:
            print(f"\n  {self.colored('⚠ Warning:', Colors.YELLOW)} Ranking is sensitive to weight changes. Results should be interpreted with caution.")
        else:
            print(f"\n  {self.colored('✓ Info:', Colors.GREEN)} Ranking is robust to minor weight fluctuations.")

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

        is_ensemble = bool(prediction.get("ensemble_scores"))
        title = "GNN/Ensemble Criticality Prediction" if is_ensemble else "GNN Criticality Prediction"
        self.print_subheader(title)

        node_scores = prediction.get("ensemble_scores") or prediction.get("node_scores")
        if not node_scores:
            print(f"  {self.colored('No prediction scores available.', Colors.GRAY)}")
            return

        # Sort by composite score
        sorted_nodes = sorted(
            node_scores.values(),
            key=lambda s: s.get("composite_score", 0),
            reverse=True
        )

        header = f"  {'#':<4} {'Component':<30} {'Score':>7} {'Level':<10} {'R':>6} {'M':>6} {'A':>6} {'V':>6}"
        print(self.colored(header, Colors.WHITE, bold=True))
        print("  " + "-" * 78)

        for i, s in enumerate(sorted_nodes[:limit], 1):
            level = s.get("criticality_level", "MINIMAL")
            color = self.level_color(level)
            score_val = s.get("composite_score", 0)
            score_str = f"{score_val:>7.4f}"
            print(
                f"  {i:<4} {s.get('component', '')[:29]:<30} "
                f"{self.colored(score_str, color)} "
                f"{self.colored(level[:10], color):<10} "
                f"{s.get('reliability_score', 0):>6.3f} {s.get('maintainability_score', 0):>6.3f} "
                f"{s.get('availability_score', 0):>6.3f} {s.get('vulnerability_score', 0):>6.3f}"
            )

        if len(sorted_nodes) > limit:
            print(f"\n  {self.colored(f'... and {len(sorted_nodes) - limit} more components', Colors.GRAY)}")
            
        # Display edge scores if available
        edge_scores = prediction.get("edge_scores", [])
        if edge_scores:
            self.print_subheader("Top Critical Relationships (GNN)")
            sorted_edges = sorted(edge_scores, key=lambda e: e.get("composite_score", 0), reverse=True)
            print(f"  {'#':<4} {'Source':<20} {'→':<3} {'Target':<20} {'Type':<15} {'Score':>7}")
            print("  " + "-" * 74)
            for i, e in enumerate(sorted_edges[:limit], 1):
                level = e.get("criticality_level", "MINIMAL")
                color = self.level_color(level)
                score_val = e.get("composite_score", 0)
                score_str = f"{score_val:>7.4f}"
                print(
                    f"  {i:<4} {e.get('source', '')[:19]:<20} → {e.get('target', '')[:19]:<20} "
                    f"{self.colored(e.get('edge_type', '')[:14], Colors.CYAN):<15} "
                    f"{self.colored(score_str, color)}"
                )

    def display_layer_result(self, result: "LayerAnalysisResult") -> None:
        """Display complete analysis result for a single layer."""
        self.print_header(f"{result.layer_name} Analysis")
        print(f"  {self.colored(result.description, Colors.GRAY)}")
        self.display_graph_summary(result)
        self.display_classification_summary(result)
        self.display_critical_components(result)
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

    def display_import_summary(self, stats: Dict[str, Any]) -> None:
        """Display summary of graph import results."""
        self.print_subheader("Graph Import Summary")
        
        print(f"  {'Status:':<20} {self.colored('SUCCESS', Colors.GREEN) if stats.get('success') else self.colored('FAILED', Colors.RED)}")
        print(f"  {'Total Nodes:':<20} {stats.get('nodes_imported', 0)}")
        print(f"  {'Total Edges:':<20} {stats.get('edges_imported', 0)}")
        print(f"  {'Duration:':<20} {stats.get('duration_ms', 0):.2f} ms")
        
        # breakdown by label/type
        print(f"\n  {'Entity Breakdown:':<20}")
        entities = ["application", "broker", "node", "topic", "library"]
        for ent in entities:
            count = stats.get(f"{ent}_count", 0)
            if count > 0:
                print(f"    - {ent.capitalize():<18} {count}")
                
        print(f"\n  {'Relationship Breakdown:':<20}")
        rels = ["runs_on", "routes", "publishes_to", "subscribes_to", "connects_to", "uses"]
        for rel in rels:
            count = stats.get(f"{rel}_count", 0)
            if count > 0:
                formatted_rel = rel.replace("_", " ").title()
                print(f"    - {formatted_rel:<18} {count}")
                
        print(f"\n  {'Derived Dependencies:':<20}")
        deps = ["app_to_app", "node_to_node", "app_to_broker", "node_to_broker", "app_to_lib"]
        for dep in deps:
            count = stats.get(f"{dep}_count", 0)
            if count > 0:
                formatted_dep = dep.replace("_", " ").title()
                print(f"    - {formatted_dep:<18} {count}")

    def display_generation_summary(self, graph_data: Dict[str, Any]) -> None:
        """Display summary of generated graph data."""
        self.print_subheader("Graph Generation Summary")
        
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
        total_edges = sum(len(rels) for rels in relationships.values())
        
        print(f"\n  {'Total Relationships:':<20} {total_edges}")
        for rel_type, rel_list in sorted(relationships.items()):
            if rel_list:
                # Format relationship type: pub_sub_rel -> Pub Sub Rel
                formatted_rel = rel_type.replace("_", " ").title()
                print(f"    - {formatted_rel:<18} {len(rel_list)}")

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
