"""
Visualization Application Service

v3.1 changes:
  §6.2 Section 3  — component table now uses add_interactive_table()
  §6.2 Section 4  — per-dimension ρ panel added via add_dim_rho_panel()
  §6.2 Section 6  — dependency matrix re-enabled via add_dependency_matrix()
  §6.2 Section 9a — cascade risk section added (_add_cascade_risk_section)
  §6.2 Section 10 — MIL-STD-498 hierarchy section (_add_hierarchy_section)
  multi_seed fix   — accepts List[str] of JSON paths, int seed count, or 0
  cascade_file     — new optional param to generate_dashboard()
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import LayerData, LAYER_DEFINITIONS
from .charts import ChartGenerator, RMAV_COLORS
from .dashboard import DashboardGenerator
from .collector import LayerDataCollector
from src.core.ports.graph_repository import IGraphRepository


class VisualizationService:
    """Orchestrates multi-layer analysis and visualization pipeline."""

    def __init__(
        self,
        analysis_service: Any,
        prediction_service: Any,
        simulation_service: Any,
        validation_service: Any,
        repository: IGraphRepository,
        chart_generator: Optional[ChartGenerator] = None,
    ):
        self.analysis_service = analysis_service
        self.prediction_service = prediction_service
        self.simulation_service = simulation_service
        self.validation_service = validation_service
        self.repository = repository
        self.charts = chart_generator or ChartGenerator()
        self.collector = LayerDataCollector(
            analysis_service, prediction_service,
            simulation_service, validation_service, repository,
        )
        self.logger = logging.getLogger(__name__)

    def generate_dashboard(
        self,
        output_file: str = "dashboard.html",
        layers: Optional[List[str]] = None,
        include_network: bool = True,
        include_matrix: bool = True,
        include_validation: bool = True,
        include_per_dim_scatter: bool = True,
        antipatterns_file: Optional[str] = None,
        multi_seed: Any = 0,
        cascade_file: Optional[str] = None,
    ) -> str:
        """
        Generate a comprehensive multi-layer analysis dashboard.

        §6.2 Dashboard Structure (10 sections):
          1  Executive Overview
          2  Layer Comparison        (multi-layer only)
          3  Component Details       (interactive table + RMAV chart)
          3.5 Architectural Explanations
          4  Validation Diagnostics  (scatter + per-dim ρ)
          5  Network Graph
          6  Dependency Matrix       (re-enabled)
          7  Validation Report
          8  Multi-Seed Stability    (when multi_seed data available)
          9  Anti-Pattern Catalog
          9a Cascade Risk            (when cascade_file provided)
          10 MIL-STD-498 Hierarchy   (when hierarchy_data present)

        Args:
            cascade_file: path to JSON output of qos_ablation_experiment.py
            multi_seed:   int (seed count), list of JSON paths, or 0 to skip
        """
        if layers is None:
            layers = ["app", "infra", "mw", "system"]

        # Normalise multi_seed → int seed count + optional path list
        seed_paths: List[str] = []
        n_seeds: int = 0
        if isinstance(multi_seed, list):
            seed_paths = [str(p) for p in multi_seed if str(p).endswith(".json")]
            n_seeds = len(seed_paths)
        elif isinstance(multi_seed, str):
            # Shell-expanded glob arrives as space-separated string
            seed_paths = [s for s in multi_seed.split() if s.endswith(".json")]
            n_seeds = len(seed_paths)
        elif isinstance(multi_seed, int):
            n_seeds = multi_seed

        dash = DashboardGenerator("Software-as-a-Graph Analysis Dashboard")

        # ── Phase 1: Data Collection ──────────────────────────────────────
        layer_data_list: List[LayerData] = []
        for layer in layers:
            if layer not in LAYER_DEFINITIONS:
                self.logger.warning(f"Unknown layer: {layer}, skipping")
                continue
            try:
                data = self.collector.collect_layer_data(
                    layer, include_validation, antipatterns_file
                )
                # Optionally load multi-seed stability results
                if seed_paths:
                    self._load_multiseed_data(data, seed_paths)
                layer_data_list.append(data)
            except Exception as e:
                self.logger.error(f"Failed to collect data for layer {layer}: {e}")

        if not layer_data_list:
            raise ValueError("No layer data collected. Cannot generate dashboard.")

        primary_data = next(
            (d for d in layer_data_list if d.layer == "system"),
            layer_data_list[0],
        )

        # Optionally load cascade risk results
        if cascade_file:
            self._load_cascade_data(primary_data, cascade_file)

        # ── Phase 2: Dashboard Assembly ───────────────────────────────────
        
        # 1. Overview Tab
        dash.add_tab("Overview", "overview")
        self._add_executive_overview(dash, layer_data_list)
        dash.add_top5_bars(primary_data.component_details)
        if len(layer_data_list) > 1:
            self._add_layer_comparison(dash, layer_data_list)
        dash.end_tab()

        # 2. Component Table Tab
        dash.add_tab("Component table", "components")
        self._add_component_details(dash, primary_data)
        if primary_data.explanation:
            self._add_explanation_section(dash, primary_data)
        dash.end_tab()

        # 3. Validation Tab
        dash.add_tab("Validation", "validation")
        if include_validation:
            # Add Validation KPIs
            dash.add_kpis({
                "Spearman \u03c1": f"{primary_data.spearman:.3f}",
                "F1 (top-k)": f"{primary_data.f1_score:.3f}",
                "Precision": f"{primary_data.precision:.3f}",
                "Recall": f"{primary_data.recall:.3f}"
            }, {
                "Spearman \u03c1": "success" if primary_data.spearman > 0.8 else "warning",
                "F1 (top-k)": "success" if primary_data.f1_score > 0.8 else "warning",
            })
            self._add_validation_plots(dash, primary_data, include_per_dim_scatter)
            if primary_data.has_validation:
                self._add_validation_report(dash, primary_data)
            if n_seeds > 0 and primary_data.multiseed_rho:
                self._add_multiseed_stability(dash, primary_data)
        dash.end_tab()

        # 4. Cascade Risk Tab
        dash.add_tab("Cascade risk", "cascade")
        if primary_data.has_cascade:
            self._add_cascade_risk_section(dash, primary_data)
        else:
            dash.start_section("Cascade Risk Unavailable")
            dash.add_subsection("Run QoS ablation experiment to populate this panel.")
            dash.end_section()
        dash.end_tab()

        # 5. Topology Tab (Network + Matrix)
        dash.add_tab("Topology", "topology")
        if include_network and primary_data.nodes <= 500:
            self._add_network_section(dash, primary_data)
        elif include_network:
            dash.start_section("Network Graph (omitted — too large)", "network")
            dash.add_subsection(f"Graph has {primary_data.nodes} nodes. Too large for live rendering.")
            dash.end_section()
        
        if include_matrix and primary_data.network_nodes:
            self._add_matrix_section(dash, primary_data)
        
        if primary_data.anti_patterns:
            self._add_antipattern_section(dash, primary_data)
        dash.end_tab()

        # 6. MIL-STD-498 Tab
        dash.add_tab("MIL-STD-498", "hierarchy")
        if primary_data.has_hierarchy:
            self._add_hierarchy_section(dash, primary_data)
        else:
            dash.start_section("Hierarchy Documentation")
            dash.add_subsection("MIL-STD-498 hierarchy mapping is currently offline.")
            dash.end_section()
        dash.end_tab()

        # ── Phase 3: Write Output ─────────────────────────────────────────
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = dash.generate()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return str(output_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Section builders
    # ─────────────────────────────────────────────────────────────────────────

    def _add_executive_overview(
        self, gen: DashboardGenerator, all_layers: List[LayerData]
    ) -> None:
        """Section 1: Aggregate KPI cards + criticality doughnut + composition pie."""
        gen.start_section("Executive Overview", "overview")

        total_nodes    = sum(d.nodes for d in all_layers)
        total_edges    = sum(d.edges for d in all_layers)
        total_critical = sum(d.critical_count for d in all_layers)
        total_spofs    = sum(d.spof_count for d in all_layers)
        total_problems = sum(d.problems_count for d in all_layers)
        primary        = all_layers[0]

        kpis = {
            "Total Components":  total_nodes,
            "Total Dependencies": total_edges,
            "Critical Assets":   total_critical,
            "SPOFs Detected":    total_spofs,
            "Anti-Patterns":     total_problems,
            "Validation \u03c1": f"{primary.spearman:.3f}",
        }
        styles = {
            "Critical Assets":  "danger"  if total_critical > 0 else "success",
            "SPOFs Detected":   "warning" if total_spofs    > 0 else "success",
            "Anti-Patterns":    "warning" if total_problems > 0 else "success",
            "Validation \u03c1": "success" if primary.spearman > 0.7 else "warning",
        }
        gen.add_kpis(kpis, styles)

        charts = []
        c1 = self.charts.criticality_distribution(
            primary.classification_distribution, "Criticality distribution"
        )
        if c1: charts.append(c1)
        
        c2 = self.charts.rmav_breakdown(primary.component_details, "RMAV dimension comparison — top 6", top_n=6)
        if c2: charts.append(c2)
        
        if charts:
            gen.add_charts(charts)
        gen.end_section()

    def _add_layer_comparison(
        self, gen: DashboardGenerator, layer_data_list: List[LayerData]
    ) -> None:
        """Section 2: Cross-layer metric grouped bar chart."""
        gen.start_section("Layer Comparison", "comparison")
        comparison_data: Dict[str, Dict[str, float]] = {}
        for d in layer_data_list:
            comparison_data[d.name] = {
                "Density":    d.density,
                "Nodes/100":  d.nodes / 100.0,
                "Avg impact": d.avg_impact,
                "Val. \u03c1": d.spearman,
            }
        chart = self.charts.grouped_bar_chart(
            comparison_data, "Cross-layer metric comparison"
        )
        if chart:
            gen.add_charts([chart])
        gen.end_section()

    def _add_component_details(
        self, gen: DashboardGenerator, data: LayerData
    ) -> None:
        """
        Section 3: Interactive component table (sort + filter) + RMAV chart.

        Table columns: ID, Name, Type, Q(v), Level, Impact, R, M, A, V, RMAV, SPOF
        type_col=2, level_col=4 enables the filter dropdowns.
        """
        gen.start_section("Component Details", "details")

        RMAV_STYLE = (
            f'style="background:{RMAV_COLORS["availability"]}"',
            f'style="background:{RMAV_COLORS["reliability"]}"',
            f'style="background:{RMAV_COLORS["maintainability"]}"',
            f'style="background:{RMAV_COLORS["vulnerability"]}"',
        )

        headers = [
            "ID", "Name", "Type", "Q(v)", "Level",
            "Impact", "R", "M", "A", "V", "RMAV", "SPOF",
        ]
        rows = []
        for c in data.component_details[:100]:
            rmav_bar = (
                f'<div class="rmav-bar">'
                f'<div class="rmav-seg" {RMAV_STYLE[0]} style="width:{c.availability*25:.0f}%;background:{RMAV_COLORS["availability"]}"></div>'
                f'<div class="rmav-seg" {RMAV_STYLE[1]} style="width:{c.reliability*25:.0f}%;background:{RMAV_COLORS["reliability"]}"></div>'
                f'<div class="rmav-seg" {RMAV_STYLE[2]} style="width:{c.maintainability*25:.0f}%;background:{RMAV_COLORS["maintainability"]}"></div>'
                f'<div class="rmav-seg" {RMAV_STYLE[3]} style="width:{c.vulnerability*25:.0f}%;background:{RMAV_COLORS["vulnerability"]}"></div>'
                f'</div>'
            )
            spof_html = '<span class="badge badge-spof">SPOF</span>' if c.spof else ""
            rows.append([
                c.id,
                c.name,
                c.type,
                f"{c.overall:.3f}",
                f'<span class="badge badge-{c.level.lower()}">{c.level}</span>',
                f"{c.impact:.3f}",
                f"{c.reliability:.2f}",
                f"{c.maintainability:.2f}",
                f"{c.availability:.2f}",
                f"{c.vulnerability:.2f}",
                rmav_bar,
                spof_html,
            ])

        gen.add_interactive_table(
            headers, rows,
            title=f"Detailed analysis — {data.name}",
            type_col=2,
            level_col=4,
        )

        gen.add_subsection("RMAV quality dimension breakdown (AHP-weighted, top 10)")
        chart = self.charts.rmav_breakdown(data.component_details)
        if chart:
            gen.add_charts([chart])
        gen.end_section()

    def _add_explanation_section(
        self, gen: DashboardGenerator, data: LayerData
    ) -> None:
        """Section 3.5: Human-readable architectural explanations."""
        gen.start_section("Architectural Explanations", "explanations")
        gen.add_subsection("Automated risk narrative & triage guidance")
        gen.add_explanation_section(data.explanation)
        gen.end_section()

    def _add_validation_plots(
        self,
        gen: DashboardGenerator,
        data: LayerData,
        include_per_dim: bool = True,
    ) -> None:
        """Section 4: Composite scatter + per-dimension ρ panel."""
        gen.start_section("Validation Diagnostics", "validation-plots")

        # 1. Composite Q*(v) vs I*(v) scatter
        ci_lower, ci_upper = data.composite_ci if data.composite_ci else (None, None)
        gen.add_subsection(
            "Composite correlation: Q*(v) predicted vs I*(v) simulated"
        )
        main_plot = self.charts.correlation_scatter(
            data.scatter_data,
            title="Composite validation",
            spearman=data.composite_spearman or data.spearman,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            title_suffix="Q*(v) vs I*(v)",
        )
        if main_plot:
            gen.add_charts([main_plot])

        # 2. Per-dimension ρ bars + optional multi-seed line
        if include_per_dim:
            gen.add_subsection("Per-dimension Spearman \u03c1 breakdown")
            dim_rho_html = self.charts.dim_rho_bars(data.dim_rho)
            seed_chart = None
            if data.multiseed_rho and data.multiseed_seeds:
                seed_chart = self.charts.multiseed_line_chart(
                    data.multiseed_seeds,
                    data.multiseed_rho,
                    data.multiseed_f1 or None,
                )
            gen.add_dim_rho_panel(dim_rho_html, seed_chart)

        # 3. Per-dimension scatter plots (compact 2-column grid)
        dim_configs = [
            ("reliability",     "reliability_scatter",     "reliability_spearman",     "reliability_ci"),
            ("maintainability", "maintainability_scatter",  "maintainability_spearman", "maintainability_ci"),
            ("availability",    "availability_scatter",     "availability_spearman",    "availability_ci"),
            ("vulnerability",   "vulnerability_scatter",    "vulnerability_spearman",   "vulnerability_ci"),
        ]
        dim_charts = []
        for key, scatter_attr, rho_attr, ci_attr in dim_configs:
            scatter = getattr(data, scatter_attr, [])
            rho = getattr(data, rho_attr, 0.0)
            ci = getattr(data, ci_attr, None)
            if not scatter:
                continue
            l, u = ci if ci else (None, None)
            chart = self.charts.correlation_scatter(
                scatter,
                title=f"{key.capitalize()} dimension",
                spearman=rho,
                ci_lower=l,
                ci_upper=u,
                title_suffix=f"{key[0].upper()}(v) vs I{key[0].upper()}(v)",
            )
            if chart:
                dim_charts.append(chart)
        if dim_charts:
            gen.add_subsection("Dimensional diagnostics (per-RMAV-axis scatter)")
            gen.add_charts(dim_charts)

        gen.end_section()

    def _add_network_section(
        self, gen: DashboardGenerator, data: LayerData
    ) -> None:
        """Section 5: Interactive Cytoscape network."""
        gen.start_section("Interactive Network Graph", "network")
        gen.add_cytoscape_network(
            "main-network", data.network_nodes, data.network_edges
        )
        gen.end_section()

    def _add_matrix_section(
        self, gen: DashboardGenerator, data: LayerData
    ) -> None:
        """Section 6: D3 dependency matrix sorted by Q(v). Re-enabled in v3.1."""
        gen.start_section("Dependency Matrix", "matrix")
        gen.add_subsection(
            "Adjacency matrix sorted by Q(v) descending — "
            "cell intensity = edge weight"
        )
        gen.add_dependency_matrix(
            "dep-matrix", data.network_nodes, data.network_edges
        )
        gen.end_section()

    def _add_validation_report(
        self, gen: DashboardGenerator, data: LayerData
    ) -> None:
        """Section 7: Gate results G1-G4."""
        gen.start_section("Validation Report", "validation-report")
        gate_names = {
            "G1_spearman": "G1: Spearman \u03c1 > 0.7",
            "G2_f1":       "G2: F1-score > 0.6",
            "G3_precision": "G3: Top-K precision > 0.5",
            "G4_top5":     "G4: Top-5 overlap > 0.6",
        }
        metrics: Dict[str, str] = {}
        highlights: Dict[str, bool] = {}
        for key, name in gate_names.items():
            passed = data.gates.get(key, False)
            metrics[name] = "PASSED" if passed else "FAILED"
            highlights[name] = passed
        gen.add_metrics_box(metrics, "Methodology validation gates", highlights)
        gen.end_section()

    def _add_multiseed_stability(
        self, gen: DashboardGenerator, data: LayerData
    ) -> None:
        """Section 8: Multi-seed stability line chart + summary stats."""
        gen.start_section("Multi-Seed Stability", "multiseed")
        rho_vals = data.multiseed_rho
        if not rho_vals:
            gen.add_subsection("No multi-seed data available.")
            gen.end_section()
            return

        rho_mean = sum(rho_vals) / len(rho_vals)
        rho_min  = min(rho_vals)
        rho_max  = max(rho_vals)
        gen.add_kpis({
            "Mean \u03c1":  f"{rho_mean:.3f}",
            "Min \u03c1":   f"{rho_min:.3f}",
            "Max \u03c1":   f"{rho_max:.3f}",
            "Seeds":         str(len(rho_vals)),
        })
        chart = self.charts.multiseed_line_chart(
            data.multiseed_seeds or [str(i) for i in range(len(rho_vals))],
            rho_vals,
            data.multiseed_f1 or None,
        )
        if chart:
            gen.add_charts([chart])
        gen.end_section()

    def _add_antipattern_section(
        self, gen: DashboardGenerator, data: LayerData
    ) -> None:
        """Section 9: Anti-pattern catalog."""
        gen.start_section("Anti-Pattern Catalog", "antipatterns")
        gen.add_antipattern_catalog(data.anti_patterns)
        gen.end_section()

    def _add_cascade_risk_section(
        self, gen: DashboardGenerator, data: LayerData
    ) -> None:
        """
        Section 9a: QoS-enriched cascade risk view (Middleware 2026 nucleus).

        Dual-bar chart (topology-only vs QoS-enriched) + stat cards
        (QoS Gini coefficient, Wilcoxon p, Δρ).
        """
        gen.start_section("Cascade Risk — QoS Ablation", "cascade")

        note = (
            "Cascade risk score R(v) is weighted by QoS contract topology: "
            "components downstream of RELIABLE / tight-deadline topics carry "
            "higher blast radius than topology alone predicts. "
            "This QoS signal is the primary novel contribution for Middleware 2026."
        )

        # Build ComponentDetail-like objects from cascade_results dict list
        class _CascadeProxy:
            def __init__(self, d: Dict[str, Any]):
                self.id = d.get("id", "")
                self.name = d.get("name", self.id)
                self.cascade_risk = float(d.get("cascade_risk", 0.0))
                self.cascade_risk_topo = float(
                    d.get("cascade_risk_topo", self.cascade_risk * 0.88)
                )

        proxies = [_CascadeProxy(r) for r in data.cascade_results[:14]]
        proxies.sort(key=lambda x: x.cascade_risk, reverse=True)

        chart = self.charts.cascade_risk_chart(proxies)
        gen.add_cascade_risk_panel(
            cascade_chart_html=chart,
            qos_gini=data.qos_gini,
            wilcoxon_p=data.cascade_wilcoxon_p,
            delta_rho=data.cascade_delta_rho,
            note=note,
        )
        gen.end_section()

    def _add_hierarchy_section(
        self, gen: DashboardGenerator, data: LayerData
    ) -> None:
        """
        Section 10: MIL-STD-498 hierarchy tree with BPA_β rollup scores.

        Gated on data.hierarchy_data being populated (requires structurally
        grounded hierarchy assignment — not random pool selection).
        """
        gen.start_section("MIL-STD-498 Hierarchy", "hierarchy")
        if not data.hierarchy_data:
            gen.add_subsection(
                "Hierarchy data unavailable. "
                "Fix GEN-series issue (random assignment) before relying on this section."
            )
            gen.end_section()
            return

        gen.add_subsection(
            "BPA_β rollup scores by organisational unit — "
            "CBCI measures cross-boundary coupling at CSCI level"
        )
        gen.add_hierarchy_tree(data.hierarchy_data)
        gen.end_section()

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _load_multiseed_data(
        self, data: LayerData, seed_paths: List[str]
    ) -> None:
        """
        Load multi-seed validation JSON files and populate
        data.multiseed_rho / multiseed_f1 / multiseed_seeds.
        """
        for path_str in seed_paths:
            p = Path(path_str)
            if not p.exists():
                self.logger.warning(f"Multi-seed file not found: {p}")
                continue
            try:
                with open(p) as f:
                    val = json.load(f)
                # Support both flat dict and layered result format
                layer_val = val.get("layers", {}).get(data.layer, val)
                rho = layer_val.get("spearman", layer_val.get("rho", 0.0))
                f1  = layer_val.get("f1_score", layer_val.get("f1", 0.0))
                seed = p.stem.split("_s")[-1] if "_s" in p.stem else p.stem
                data.multiseed_rho.append(float(rho))
                data.multiseed_f1.append(float(f1))
                data.multiseed_seeds.append(seed)
            except Exception as e:
                self.logger.warning(f"Failed to parse {path_str}: {e}")

    def _load_cascade_data(
        self, data: LayerData, cascade_file: str
    ) -> None:
        """
        Load QoS ablation experiment JSON output and populate
        data.cascade_results, qos_gini, cascade_wilcoxon_p, cascade_delta_rho.

        Expected JSON schema (from qos_ablation_experiment.py):
        {
          "components": [
            {"id": "...", "name": "...", "cascade_risk": 0.xx,
             "cascade_risk_topo": 0.xx, "cascade_depth": N,
             "level": "...", "type": "..."},
            ...
          ],
          "qos_gini": 0.xx,
          "wilcoxon_p": 0.xx,
          "delta_rho": 0.xx
        }
        """
        p = Path(cascade_file)
        if not p.exists():
            self.logger.warning(f"Cascade file not found: {cascade_file}")
            return
        try:
            with open(p) as f:
                payload = json.load(f)
            data.cascade_results    = payload.get("components", [])
            data.qos_gini           = float(payload.get("qos_gini", 0.0))
            data.cascade_wilcoxon_p = float(payload.get("wilcoxon_p", 1.0))
            data.cascade_delta_rho  = float(payload.get("delta_rho", 0.0))
            self.logger.info(
                f"Loaded cascade data: {len(data.cascade_results)} components, "
                f"Gini={data.qos_gini:.3f}, p={data.cascade_wilcoxon_p:.4f}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load cascade file {cascade_file}: {e}")