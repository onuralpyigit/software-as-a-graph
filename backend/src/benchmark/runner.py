"""
Benchmark Runner

Executes the generate → import → analyze → simulate → validate pipeline
via application services (in-process, no subprocess overhead) and collects
timing + validation metrics into BenchmarkRecord objects.
"""
from __future__ import annotations

import logging
import random
import statistics
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core import create_repository
from src.generation import GenerationService, load_config
from src.analysis import AnalysisService
from src.simulation import SimulationService
from src.validation import ValidationService, ValidationTargets
from src.validation.metric_calculator import spearman_correlation

from .models import (
    AggregateResult,
    BenchmarkRecord,
    BenchmarkScenario,
    BenchmarkSummary,
)

# Default validation targets (matches thesis requirements)
DEFAULT_TARGETS = ValidationTargets(
    spearman=0.70,
    f1_score=0.80,
    precision=0.80,
    recall=0.80,
    top_5_overlap=0.60,
    top_10_overlap=0.50,
)

# Metric names in (record_attr, target_attr) pairs for DRY target counting
_TARGET_CHECKS: List[Tuple[str, str]] = [
    ("spearman",      "spearman"),
    ("f1_score",      "f1_score"),
    ("precision",     "precision"),
    ("recall",        "recall"),
    ("top5_overlap",  "top_5_overlap"),
    ("top10_overlap", "top_10_overlap"),
]


class BenchmarkRunner:
    """
    Executes benchmark scenarios using in-process application services.

    Each scenario produces one BenchmarkRecord per (run, layer) combination.
    After all scenarios finish, call ``aggregate_results()`` to build a
    BenchmarkSummary with per-group statistics.
    """

    def __init__(
        self,
        output_dir: Path,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        verbose: bool = False,
        targets: Optional[ValidationTargets] = None,
    ):
        self.output_dir = output_dir
        self.verbose = verbose
        self.targets = targets or DEFAULT_TARGETS
        self.logger = logging.getLogger("Benchmark")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.repo = create_repository(uri=uri, user=user, password=password)
        self.records: List[BenchmarkRecord] = []

        # Services will be initialized once
        self.analysis_service = AnalysisService(self.repo)
        self.simulation_service = SimulationService(self.repo)
        self.validation_service = ValidationService(
            analysis_service=self.analysis_service,
            simulation_service=self.simulation_service,
            targets=self.targets
        )

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "BenchmarkRunner":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release Neo4j resources."""
        self.repo.close()

    # ------------------------------------------------------------------
    # Pipeline stages (all return (result | None, elapsed_ms))
    # ------------------------------------------------------------------

    def _generate_data(
        self, scenario: BenchmarkScenario, seed: int
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Generate synthetic graph data in memory."""
        t0 = time.time()
        try:
            if scenario.config_path:
                config = load_config(scenario.config_path)
                config.seed = seed
                service = GenerationService(config=config)
            else:
                service = GenerationService(scale=scenario.scale, seed=seed)

            return service.generate(), (time.time() - t0) * 1000

        except Exception as e:
            self.logger.error("Generation failed: %s", e)
            if self.verbose:
                traceback.print_exc()
            return None, 0.0

    def _import_data(self, graph_data: Dict[str, Any]) -> Tuple[bool, float]:
        """Import graph data into Neo4j (clears DB first)."""
        t0 = time.time()
        try:
            self.repo.save_graph(graph_data, clear=True)
            return True, (time.time() - t0) * 1000

        except Exception as e:
            self.logger.error("Import failed: %s", e)
            if self.verbose:
                traceback.print_exc()
            return False, 0.0

    def _run_analysis(self, layer: str) -> Tuple[Optional[Dict], float]:
        """Run structural + quality analysis, return dict."""
        t0 = time.time()
        try:
            result = self.analysis_service.analyze_layer(layer)
            return result.to_dict(), (time.time() - t0) * 1000

        except Exception as e:
            self.logger.error("Analysis failed for layer %s: %s", layer, e)
            if self.verbose:
                traceback.print_exc()
            return None, 0.0

    def _run_simulation(self, layer: str) -> Tuple[Optional[List[Dict]], float]:
        """Run exhaustive failure simulation, return list of dicts."""
        t0 = time.time()
        try:
            results = self.simulation_service.run_failure_simulation_exhaustive(layer=layer)
            return [r.to_dict() for r in results], (time.time() - t0) * 1000

        except Exception as e:
            self.logger.error("Simulation failed for layer %s: %s", layer, e)
            if self.verbose:
                traceback.print_exc()
            return None, 0.0

    def _run_validation(
        self,
        layer: str,
        analysis_data: Dict,
        simulation_data: List[Dict],
    ) -> Tuple[Optional[Any], float]:
        """Validate predicted quality scores against simulated impact."""
        t0 = time.time()
        try:

            # Extract predicted scores from analysis
            pred_scores: Dict[str, float] = {}
            comp_types: Dict[str, str] = {}
            for c in analysis_data.get("quality_analysis", {}).get("components", []):
                cid = c.get("id")
                pred_scores[cid] = c.get("scores", {}).get("overall", 0.0)
                comp_types[cid] = c.get("type", "unknown")

            # Extract actual impact from simulation
            actual_scores: Dict[str, float] = {}
            for r in simulation_data:
                actual_scores[r["target_id"]] = r.get("impact", {}).get(
                    "composite_impact", 0.0
                )

            res = self.validation_service.validator.validate(
                predicted_scores=pred_scores,
                actual_scores=actual_scores,
                component_types=comp_types,
                layer=layer,
                context="Benchmark",
            )

            # --- Baseline Correlations ---
            m_ids = list(pred_scores.keys())
            m_actuals = [actual_scores.get(cid, 0.0) for cid in m_ids]
            
            # 1. Betweenness Centrality
            bc_scores: Dict[str, float] = {}
            for c in analysis_data.get("structural_analysis", {}).get("components", []):
                bc_scores[c["id"]] = c["metrics"].get("betweenness", 0.0)
            m_bc = [bc_scores.get(cid, 0.0) for cid in m_ids]
            spearman_bc, _ = spearman_correlation(m_bc, m_actuals)

            # 2. Degree Centrality
            deg_scores: Dict[str, float] = {}
            for c in analysis_data.get("structural_analysis", {}).get("components", []):
                deg_scores[c["id"]] = c["metrics"].get("degree", 0.0)
            m_deg = [deg_scores.get(cid, 0.0) for cid in m_ids]
            spearman_deg, _ = spearman_correlation(m_deg, m_actuals)

            # 3. Random Ranking
            rng = random.Random(42) # Deterministic random per validation pass
            m_rand = [rng.random() for _ in m_ids]
            spearman_rand, _ = spearman_correlation(m_rand, m_actuals)

            # Return (ValidationResult, extra_metrics_dict, elapsed_ms)
            extra = {
                "spearman_bc": spearman_bc,
                "spearman_degree": spearman_deg,
                "spearman_random": spearman_rand,
            }

            return (res, extra), (time.time() - t0) * 1000

        except Exception as e:
            self.logger.error("Validation failed for layer %s: %s", layer, e)
            if self.verbose:
                traceback.print_exc()
            return None, 0.0

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------

    def _fill_record_from_validation(
        self, record: BenchmarkRecord, val_result: Any
    ) -> None:
        """Extract metrics from a ValidationResult into *record*."""
        record.spearman = val_result.overall.correlation.spearman
        record.f1_score = val_result.overall.classification.f1_score
        record.precision = val_result.overall.classification.precision
        record.recall = val_result.overall.classification.recall
        record.top5_overlap = val_result.overall.ranking.top_5_overlap
        record.top10_overlap = val_result.overall.ranking.top_10_overlap
        record.rmse = val_result.overall.error.rmse
        record.passed = val_result.passed

        # Count how many targets are met
        record.targets_met = sum(
            getattr(record, rec_attr) >= getattr(self.targets, tgt_attr)
            for rec_attr, tgt_attr in _TARGET_CHECKS
        )

    # ------------------------------------------------------------------
    # Scenario execution
    # ------------------------------------------------------------------

    def run_scenario(
        self, scenario: BenchmarkScenario, seed_start: int = 42
    ) -> List[BenchmarkRecord]:
        """
        Run a full scenario: generate + import once per seed,
        then analyze/simulate/validate for each requested layer.
        """
        scenario_records: List[BenchmarkRecord] = []

        for i in range(scenario.runs):
            seed = seed_start + i

            # 1. Generate
            graph_data, gen_time = self._generate_data(scenario, seed)
            if not graph_data:
                self.logger.warning("Run %d: generation failed — skipping", i)
                continue

            # 2. Import
            ok, imp_time = self._import_data(graph_data)
            if not ok:
                self.logger.warning("Run %d: import failed — skipping", i)
                continue

            # 3–5. Per-layer pipeline
            for layer in scenario.layers:
                record = BenchmarkRecord(
                    run_id=f"{scenario.label}_{layer}_{seed}",
                    timestamp=datetime.now().isoformat(),
                    scale=scenario.label,
                    layer=layer,
                    seed=seed,
                    time_generation=gen_time,
                    time_import=imp_time,
                )

                # 3. Analysis
                an_data, an_time = self._run_analysis(layer)
                record.time_analysis = an_time

                if not an_data:
                    record.error = "Analysis failed"
                    self._commit_record(record, scenario_records)
                    continue

                stats = an_data.get("graph_summary", {})
                record.nodes = stats.get("nodes", 0)
                record.edges = stats.get("edges", 0)
                record.density = stats.get("density", 0.0)

                # 4. Simulation
                sim_data, sim_time = self._run_simulation(layer)
                record.time_simulation = sim_time

                if not sim_data:
                    record.error = "Simulation failed"
                    self._commit_record(record, scenario_records)
                    continue

                # 5. Validation
                (val_result, baseline), val_time = self._run_validation(layer, an_data, sim_data)
                record.time_validation = val_time

                if val_result:
                    self._fill_record_from_validation(record, val_result)
                    # Set baselines
                    record.spearman_bc = baseline.get("spearman_bc", 0.0)
                    record.spearman_degree = baseline.get("spearman_degree", 0.0)
                    record.spearman_random = baseline.get("spearman_random", 0.0)
                else:
                    record.error = "Validation failed"

                record.time_total = (
                    record.time_generation
                    + record.time_import
                    + record.time_analysis
                    + record.time_simulation
                    + record.time_validation
                )

                self._commit_record(record, scenario_records)

        return scenario_records

    def _commit_record(
        self, record: BenchmarkRecord, batch: List[BenchmarkRecord]
    ) -> None:
        """Append *record* to both the global and per-scenario lists."""
        self.records.append(record)
        batch.append(record)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_results(self, duration: float) -> BenchmarkSummary:
        """Build a BenchmarkSummary from all collected records."""
        summary = BenchmarkSummary(
            timestamp=datetime.now().isoformat(),
            duration=duration,
            total_runs=len(self.records),
            passed_runs=sum(1 for r in self.records if r.passed),
            records=self.records,
        )

        if not self.records:
            return summary

        summary.overall_pass_rate = (summary.passed_runs / summary.total_runs) * 100
        summary.overall_spearman = statistics.mean(r.spearman for r in self.records)
        summary.overall_f1 = statistics.mean(r.f1_score for r in self.records)

        # Identify best / worst by Spearman
        best = max(self.records, key=lambda r: r.spearman)
        worst = min(self.records, key=lambda r: r.spearman)
        summary.best_config = best.run_id
        summary.worst_config = worst.run_id

        # Group by (scale, layer)
        groups: Dict[Tuple[str, str], List[BenchmarkRecord]] = defaultdict(list)
        for r in self.records:
            groups[(r.scale, r.layer)].append(r)
            if r.scale not in summary.scales:
                summary.scales.append(r.scale)
            if r.layer not in summary.layers:
                summary.layers.append(r.layer)

        for (scale, layer), recs in groups.items():
            agg = self._aggregate_group(scale, layer, recs)
            summary.aggregates.append(agg)

        return summary

    @staticmethod
    def _aggregate_group(
        scale: str, layer: str, recs: List[BenchmarkRecord]
    ) -> AggregateResult:
        """Compute mean/std for a group of records."""
        n = len(recs)
        agg = AggregateResult(scale=scale, layer=layer, num_runs=n)

        def _mean(attr: str) -> float:
            return statistics.mean(getattr(r, attr) for r in recs)

        def _stdev(attr: str) -> float:
            vals = [getattr(r, attr) for r in recs]
            return statistics.stdev(vals) if len(vals) >= 2 else 0.0

        # Graph stats
        agg.avg_nodes = _mean("nodes")
        agg.avg_edges = _mean("edges")
        agg.avg_density = _mean("density")

        # Timing
        agg.avg_time_analysis = _mean("time_analysis")
        agg.avg_time_simulation = _mean("time_simulation")
        agg.avg_time_total = _mean("time_total")
        if agg.avg_time_analysis > 0:
            agg.speedup_ratio = agg.avg_time_simulation / agg.avg_time_analysis

        # Validation
        agg.avg_spearman = _mean("spearman")
        agg.std_spearman = _stdev("spearman")
        agg.avg_f1 = _mean("f1_score")
        agg.std_f1 = _stdev("f1_score")
        agg.avg_precision = _mean("precision")
        agg.avg_recall = _mean("recall")
        agg.avg_top5 = _mean("top5_overlap")
        agg.avg_top10 = _mean("top10_overlap")
        agg.avg_rmse = _mean("rmse")

        # Baseline averages
        agg.avg_spearman_bc = _mean("spearman_bc")
        agg.avg_spearman_degree = _mean("spearman_degree")
        agg.avg_spearman_random = _mean("spearman_random")

        # Pass rate
        agg.num_passed = sum(1 for r in recs if r.passed)
        agg.pass_rate = (agg.num_passed / n) * 100

        return agg