"""
Display Application Service
"""
import sys
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...domain.models.analysis.results import LayerAnalysisResult, MultiLayerAnalysisResult
    from ...domain.models.simulation.results import EventResult, FailureResult, SimulationReport
    from ...domain.models.validation.results import PipelineResult, LayerValidationResult
    from ...domain.models.validation.metrics import ValidationTargets


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


class DisplayService:
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
            if p.entity_type == "Component":
                name = node_names.get(p.entity_id)
                if name: entity_display = f"{p.entity_id} ({name})"
            print(f"           Entity: {self.colored(entity_display, Colors.CYAN)} ({p.entity_type})\n           Category: {p.category}")
            desc_lines = self.wrap_text(p.description, 60)
            print(f"           Issue: {desc_lines[0]}")
            for line in desc_lines[1:]: print(f"                  {line}")
            rec_lines = self.wrap_text(p.recommendation, 60)
            print(f"           Fix: {self.colored(rec_lines[0], Colors.GREEN)}")
            for line in rec_lines[1:]: print(f"                {self.colored(line, Colors.GREEN)}")

    def display_layer_result(self, result: "LayerAnalysisResult") -> None:
        """Display complete analysis result for a single layer."""
        self.print_header(f"{result.layer_name} Analysis")
        print(f"  {self.colored(result.description, Colors.GRAY)}")
        self.display_graph_summary(result)
        self.display_classification_summary(result)
        self.display_critical_components(result)
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
        print(f"\n  {self.colored('Target:', Colors.CYAN)}       {get_name(result.target_id)} ({result.target_type})\n  {self.colored('Scenario:', Colors.CYAN)}     {result.scenario}")
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

    def display_simulation_report(self, report: "SimulationReport") -> None:
        """Display comprehensive simulation report."""
        self.print_header("Simulation Report", "═")
        print(f"\n  {self.colored('Timestamp:', Colors.CYAN)}  {report.timestamp}")
        for layer_name, metrics in report.layer_metrics.items():
            self.print_subheader(f"Layer: {layer_name.upper()}")
            print(f"\n  {self.colored('Event Simulation:', Colors.CYAN)}\n    Throughput:        {metrics.event_throughput} messages\n    Delivery Rate:     {metrics.event_delivery_rate:.1f}%")
            print(f"\n  {self.colored('Failure Simulation:', Colors.CYAN)}\n    Avg Reach Loss:    {metrics.avg_reachability_loss * 100:.1f}%\n    Max Impact:        {metrics.max_impact:.4f}")
            print(f"\n  {self.colored('Criticality:', Colors.CYAN)}\n    Critical:          {self.colored(str(metrics.critical_count), Colors.RED)}\n    SPOFs:             {metrics.spof_count}")

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
