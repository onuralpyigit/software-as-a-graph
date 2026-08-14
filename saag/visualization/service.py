"""
Visualization Application Service

Orchestrates Step 7. Work is split into three stages so that each can be
used on its own:

    collect()            services + files → List[LayerData]
    build_html()         List[LayerData]  → dashboard HTML  (pure)
    generate_dashboard() collect → build_html → write to disk

`build_html()` touches neither the repository nor the filesystem, which is
what lets `cli/visualize_graph.py --demo` render the production dashboard
from a fixture without a database.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import LayerData
from .charts import ChartGenerator
from .palette import RM_COLORS
from .dashboard import DashboardGenerator
from .collector import LayerDataCollector
from saag.core.layers import AnalysisLayer
from saag.core.ports.graph_repository import IGraphRepository
from saag.validation.models import ValidationTargets

DEFAULT_LAYERS = ["app", "infra", "mw", "system"]
DASHBOARD_TITLE = "Software-as-a-Graph Analysis Dashboard"

#: (gate key in LayerData.gates, display label, ValidationTargets attribute,
#: comparison symbol) for every gate ValidationService._evaluate_gates
#: computes (saag/validation/service.py:607-617). Labels are built from this
#: table rather than hardcoded so they cannot drift from the thresholds the
#: gates are actually evaluated against (they previously did: "F1-score >
#: 0.6" / "Top-K precision > 0.5" against real targets of 0.75 / 0.80).
#: G7 (CDCC) and G9 (FTR) were retired with the Vulnerability/Security
#: dimension — both specialist metrics were security-only. The gap in the
#: numbering is intentional; do not renumber or reuse it.
_GATE_SPECS = (
    ("G1_spearman", "Spearman ρ", "spearman", "≥"),
    ("G2_f1", "F1-score", "f1_score", "≥"),
    ("G3_precision", "Top-K precision", "precision", "≥"),
    ("G4_top5", "Top-5 overlap", "top_5_overlap", "≥"),
    ("G5_predictive_gain", "Predictive gain", "predictive_gain", ">"),
    ("G6_kappa_cta", "Weighted κ (CTA)", "weighted_kappa_cta", "≥"),
    ("G8_bottleneck_precision", "Bottleneck precision", "bottleneck_precision_target", "≥"),
)


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
        Collect every requested layer, render the dashboard, write it to disk
        and return the path written.

        Args:
            cascade_file: path to a QoS cascade-risk JSON (see _load_cascade_data
                          for the expected schema)
            multi_seed:   list of validation JSON paths, a space-separated
                          string of them, or 0 to skip the stability panel
        """
        layer_data_list = self.collect(
            layers=layers,
            include_validation=include_validation,
            antipatterns_file=antipatterns_file,
            multi_seed=multi_seed,
            cascade_file=cascade_file,
        )
        html = self.build_html(
            layer_data_list,
            include_network=include_network,
            include_matrix=include_matrix,
            include_validation=include_validation,
            include_per_dim_scatter=include_per_dim_scatter,
        )

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        return str(output_path)

    def collect(
        self,
        layers: Optional[List[str]] = None,
        include_validation: bool = True,
        antipatterns_file: Optional[str] = None,
        multi_seed: Any = 0,
        cascade_file: Optional[str] = None,
    ) -> List[LayerData]:
        """
        Gather one LayerData per requested layer from the analysis, simulation
        and validation services, enriched with any supplied result files.

        Unknown or failing layers are logged and skipped; raises ValueError
        only when nothing at all could be collected.
        """
        layers = layers or DEFAULT_LAYERS
        seed_paths = _normalise_seed_paths(multi_seed)

        layer_data_list: List[LayerData] = []
        for layer in layers:
            try:
                AnalysisLayer.from_string(layer)
            except ValueError:
                self.logger.warning(f"Unknown layer: {layer}, skipping")
                continue
            try:
                data = self.collector.collect_layer_data(
                    layer, include_validation, antipatterns_file
                )
                if seed_paths:
                    self._load_multiseed_data(data, seed_paths)
                layer_data_list.append(data)
            except Exception as e:
                self.logger.error(f"Failed to collect data for layer {layer}: {e}")

        if not layer_data_list:
            raise ValueError("No layer data collected. Cannot generate dashboard.")

        if cascade_file:
            self._load_cascade_data(_primary_layer(layer_data_list), cascade_file)

        return layer_data_list

    def build_html(
        self,
        layer_data_list: List[LayerData],
        include_network: bool = True,
        include_matrix: bool = True,
        include_validation: bool = True,
        include_per_dim_scatter: bool = True,
        title: str = DASHBOARD_TITLE,
    ) -> str:
        """
        Render collected layers into dashboard HTML. Pure: no repository
        access, no file I/O.

        Tabs, in order:
          Overview      KPIs, criticality doughnut, RM chart, top-5, layer comparison
          Components    interactive table + architectural explanations
          Validation    scatter, per-dimension ρ, gates, multi-seed stability
          Cascade risk  QoS ablation panel (when cascade data was loaded)
          Topology      Cytoscape network, dependency matrix, anti-patterns
          MIL-STD-498   hierarchy tree (when hierarchy data is present)
        """
        if not layer_data_list:
            raise ValueError("No layer data supplied. Cannot generate dashboard.")

        primary_data = _primary_layer(layer_data_list)
        dash = DashboardGenerator(title)

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
            if primary_data.multiseed_rho:
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

        return dash.generate()

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
        
        c2 = self.charts.rm_breakdown(primary.component_details, "RM dimension comparison — top 6", top_n=6)
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
        Section 3: Interactive component table (sort + filter) + RM chart.

        Table columns: ID, Name, Type, Q(v), Level, Impact, R, M, FT, A, RM, SPOF
        FT and A are Reliability's sub-characteristics, reported alongside R.
        type_col=2, level_col=4 enables the filter dropdowns.
        """
        gen.start_section("Component Details", "details")

        headers = [
            "ID", "Name", "Type", "Q(v)", "Level",
            "Impact", "R", "M", "FT", "A", "RM", "SPOF",
        ]
        rows = []
        for c in data.component_details[:100]:
            # Each dimension contributes up to 33 % of the bar width, so a
            # component scoring 1.0 across all three fills it completely.
            segments = "".join(
                f'<div class="rm-seg" style="width:{getattr(c, dim) * 33.3:.0f}%;'
                f'background:{RM_COLORS[dim]}"></div>'
                for dim in ("fault_tolerance", "availability", "maintainability")
            )
            rm_bar = f'<div class="rm-bar">{segments}</div>'
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
                f"{c.fault_tolerance:.2f}",
                f"{c.availability:.2f}",
                rm_bar,
                spof_html,
            ])

        gen.add_interactive_table(
            headers, rows,
            title=f"Detailed analysis — {data.name}",
            type_col=2,
            level_col=4,
        )

        gen.add_subsection("RM quality dimension breakdown (top 10)")
        chart = self.charts.rm_breakdown(data.component_details)
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
            gen.add_subsection("Dimensional diagnostics (per-RM-axis scatter)")
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
        """Section 7: Gate results (G1-G8, with G7 and G9 retired — see _GATE_SPECS)."""
        gen.start_section("Validation Report", "validation-report")
        targets = getattr(self.validation_service, "targets", None) or ValidationTargets()
        metrics: Dict[str, str] = {}
        highlights: Dict[str, bool] = {}
        for key, label, attr, op in _GATE_SPECS:
            if key not in data.gates:
                continue
            threshold = getattr(targets, attr)
            gate_number = key.split("_", 1)[0]
            name = f"{gate_number}: {label} {op} {threshold:.2f}"
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
        Cascade risk tab: QoS-enriched cascade risk (Middleware 2026 nucleus).

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

        ranked = sorted(
            data.cascade_results,
            key=lambda r: float(r.get("cascade_risk", 0.0)),
            reverse=True,
        )[:14]
        chart = self.charts.cascade_risk_chart(ranked)
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
        MIL-STD-498 tab: hierarchy tree with BPA_β rollup scores.

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

        Expected JSON schema:
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

# ─────────────────────────────────────────────────────────────────────────────
# Module helpers
# ─────────────────────────────────────────────────────────────────────────────


def _primary_layer(layer_data_list: List[LayerData]) -> LayerData:
    """The layer the single-layer sections render: 'system' when present."""
    return next(
        (d for d in layer_data_list if d.layer == "system"),
        layer_data_list[0],
    )


def _normalise_seed_paths(multi_seed: Any) -> List[str]:
    """
    Accept the several shapes --multi-seed arrives in and return JSON paths.

    A list of paths (argparse nargs='*'), a space-separated string (an
    unexpanded shell glob), or 0/None to skip the stability panel.
    """
    if isinstance(multi_seed, str):
        candidates = multi_seed.split()
    elif isinstance(multi_seed, (list, tuple)):
        candidates = [str(p) for p in multi_seed]
    else:
        return []
    return [c for c in candidates if c.endswith(".json")]
