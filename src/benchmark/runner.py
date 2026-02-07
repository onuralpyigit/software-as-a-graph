import time
import json
import logging
import statistics
import traceback
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional, Any, Dict

# Domain & Application imports
from src.application.container import Container
from src.domain.models.validation.metrics import ValidationTargets
from src.application.services.generation_service import GenerationService, load_config
from src.domain.models.statistics import GraphConfig

from .models import BenchmarkRecord, BenchmarkSummary, AggregateResult, BenchmarkScenario
from .reporting import ReportGenerator

# Layer definitions mapping (kept for reference and potential validation)
LAYER_DEFINITIONS = {
    "app": {"name": "Application", "full": "Application Layer"},
    "infra": {"name": "Infrastructure", "full": "Infrastructure Layer"},
    "mw-app": {"name": "MW-Application", "full": "Middleware-Application Layer"},
    "mw-infra": {"name": "MW-Infrastructure", "full": "Middleware-Infrastructure Layer"},
    "system": {"name": "System", "full": "Complete System"},
}

VALIDATION_TARGETS = ValidationTargets(
    spearman=0.70,
    f1_score=0.80,
    precision=0.80,
    recall=0.80,
    top_5_overlap=0.40,
    top_10_overlap=0.60,
)

class BenchmarkRunner:
    """
    Executes comprehensive benchmark suite for the graph methodology.
    Uses internal application services for maximum performance and accuracy.
    """
    
    def __init__(
        self,
        output_dir: Path,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        verbose: bool = False,
    ):
        self.output_dir = output_dir
        self.uri = uri
        self.user = user
        self.password = password
        self.verbose = verbose
        
        # Initialize Dependency Container
        self.container = Container(
            uri=self.uri,
            user=self.user,
            password=self.password
        )
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("Benchmark")
        self.records: List[BenchmarkRecord] = []
        
    def _generate_data(self, scenario: BenchmarkScenario, seed: int) -> Tuple[Optional[Dict[str, Any]], float]:
        """Generate synthetic graph data based on scenario."""
        start = time.time()
        try:
            if scenario.config_path:
                # Load config from file
                config = load_config(scenario.config_path)
                # Override seed if possible or rely on config
                # Note: GenerationService uses config.seed if present. 
                # To enforce variation across runs, we might need to mutate the config.
                config.seed = seed
                service = GenerationService(config=config)
            else:
                # Use scale preset
                service = GenerationService(scale=scenario.scale, seed=seed)
                
            graph_data = service.generate()
            return graph_data, (time.time() - start) * 1000
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            if self.verbose: traceback.print_exc()
            return None, 0.0

    def _import_data(self, graph_data: Dict[str, Any]) -> Tuple[bool, float]:
        """Import graph data into Neo4j."""
        start = time.time()
        try:
            repo = self.container.graph_repository()
            repo.save_graph(graph_data, clear=True)
            return True, (time.time() - start) * 1000
        except Exception as e:
            self.logger.error(f"Import failed: {e}")
            if self.verbose: traceback.print_exc()
            return False, 0.0

    def _run_analysis(self, layer: str) -> Tuple[Optional[Any], float]:
        """Run structural and quality analysis."""
        start = time.time()
        try:
            service = self.container.analysis_service()
            result = service.analyze_layer(layer)
            # Return raw dict for consistency with previous structure or use object
            # Converting to dict to match existing parsing logic in run_scenario
            return result.to_dict(), (time.time() - start) * 1000
        except Exception as e:
            self.logger.error(f"Analysis failed for layer {layer}: {e}")
            if self.verbose: traceback.print_exc()
            return None, 0.0

    def _run_simulation(self, layer: str) -> Tuple[Optional[Any], float]:
        """Run failure simulation."""
        start = time.time()
        try:
            service = self.container.simulation_service()
            # return exhaustive failure results
            results = service.run_failure_simulation_exhaustive(layer=layer)
            # serialization for record storage
            return [r.to_dict() for r in results], (time.time() - start) * 1000
        except Exception as e:
            self.logger.error(f"Simulation failed for layer {layer}: {e}")
            if self.verbose: traceback.print_exc()
            return None, 0.0

    def _run_validation(self, layer: str, analysis_data: Dict, simulation_data: List[Dict]) -> Tuple[Optional[Any], float]:
        """Run validation against targets."""
        start = time.time()
        try:
            # We construct validation service with specific targets
            service = self.container.validation_service(targets=VALIDATION_TARGETS)
            
            # Extract predicted scores from analysis data
            # Extract predicted scores from analysis data
            # Analysis data structure: { "quality_analysis": { "components": [ ... ] } }
            pred_scores = {}
            comp_types = {}
            
            qa_data = analysis_data.get("quality_analysis", {})
            for c in qa_data.get("components", []):
                 cid = c.get("id")
                 score = c.get("scores", {}).get("overall", 0.0)
                 ctype = c.get("type", "unknown")
                 pred_scores[cid] = score
                 comp_types[cid] = ctype
            
            # Extract actual scores from simulation data
            # Simulation data is list of FailureResult dicts
            actual_scores = {}
            for r in simulation_data:
                target = r.get("target_id")
                impact = r.get("impact", {}).get("composite_impact", 0.0)
                actual_scores[target] = impact
                
            # Run fast validation
            res = service.validator.validate(
                predicted_scores=pred_scores,
                actual_scores=actual_scores,
                component_types=comp_types,
                layer=layer,
                context="Benchmark"
            )
            
            # Wrap in structure expected by _parse_validation_results
            # Or simplified since we are inside python now:
            return res, (time.time() - start) * 1000
            
        except Exception as e:
            self.logger.error(f"Validation failed for layer {layer}: {e}")
            if self.verbose: traceback.print_exc()
            return None, 0.0

    def run_scenario(self, scenario: BenchmarkScenario, seed_start: int = 42) -> List[BenchmarkRecord]:
        """Run a single benchmark scenario."""
        scenario_records = []
        scale_label = scenario.scale if scenario.scale else "custom"
        
        for i in range(scenario.runs):
            seed = seed_start + i
            
            # 1. Generate (In-Memory)
            graph_data, gen_time = self._generate_data(scenario, seed)
            if not graph_data:
                print(f"Generation failed for run {i}")
                continue
                
            # 2. Import (Direct DB)
            success, imp_time = self._import_data(graph_data)
            if not success: 
                print(f"Import failed for run {i}")
                continue
            
            # Per-layer processing
            for layer in scenario.layers:
                run_id = f"{scale_label}_{layer}_{seed}"
                
                record = BenchmarkRecord(
                     run_id=run_id,
                     timestamp=datetime.now().isoformat(),
                     scale=scale_label,
                     layer=layer,
                     seed=seed
                )
                record.time_generation = gen_time
                record.time_import = imp_time
                
                # 3. Analysis
                an_data, an_time = self._run_analysis(layer)
                record.time_analysis = an_time
                
                if an_data:
                    # Extract graph stats
                    # LayerAnalysisResult.to_dict() structure:
                    # { "graph_summary": { "nodes": X, "edges": Y, ... }, "quality_analysis": ... }
                    stats = an_data.get("graph_summary", {})
                    record.nodes = stats.get("nodes", 0)
                    record.edges = stats.get("edges", 0)
                    record.density = stats.get("density", 0.0)
                else:
                    record.error = "Analysis failed"
                    self.records.append(record)
                    scenario_records.append(record)
                    continue

                # 4. Simulation
                sim_data, sim_time = self._run_simulation(layer)
                record.time_simulation = sim_time
                if not sim_data:
                    record.error = "Simulation failed"
                    self.records.append(record)
                    scenario_records.append(record)
                    continue
                
                # 5. Validation
                # Pass data directly to avoid re-fetching/re-calculating
                val_result, val_time = self._run_validation(layer, an_data, sim_data)
                record.time_validation = val_time
                
                if val_result:
                    self._parse_validation_results(record, val_result)
                else:
                     record.error = "Validation failed"
                
                record.time_total = record.time_generation + record.time_import + record.time_analysis + record.time_simulation + record.time_validation
                
                self.records.append(record)
                scenario_records.append(record)
        
        return scenario_records

    def _parse_validation_results(self, record: BenchmarkRecord, val_result: Any):
        """Extract metrics from ValidationResult object into BenchmarkRecord."""
        # val_result is a ValidationResult object
        
        record.spearman = val_result.overall.correlation.spearman
        record.f1_score = val_result.overall.classification.f1_score
        record.precision = val_result.overall.classification.precision
        record.recall = val_result.overall.classification.recall
        record.top5_overlap = val_result.overall.ranking.top_5_overlap
        record.top10_overlap = val_result.overall.ranking.top_10_overlap
        record.rmse = val_result.overall.error.rmse
        
        record.passed = val_result.passed
        
        # Targets met
        targets_met = 0
        if record.spearman >= VALIDATION_TARGETS.spearman: targets_met += 1
        if record.f1_score >= VALIDATION_TARGETS.f1_score: targets_met += 1
        if record.precision >= VALIDATION_TARGETS.precision: targets_met += 1
        if record.recall >= VALIDATION_TARGETS.recall: targets_met += 1
        if record.top5_overlap >= VALIDATION_TARGETS.top_5_overlap: targets_met += 1
        if record.top10_overlap >= VALIDATION_TARGETS.top_10_overlap: targets_met += 1
        record.targets_met = targets_met

    def aggregate_results(self, duration: float) -> BenchmarkSummary:
        """Aggregate all collected records into a summary."""
        summary = BenchmarkSummary(
            timestamp=datetime.now().isoformat(),
            duration=duration,
            total_runs=len(self.records),
            passed_runs=sum(1 for r in self.records if r.passed),
            records=self.records
        )
        
        if summary.total_runs > 0:
            summary.overall_pass_rate = (summary.passed_runs / summary.total_runs) * 100
            summary.overall_spearman = statistics.mean([r.spearman for r in self.records])
            summary.overall_f1 = statistics.mean([r.f1_score for r in self.records])
            
        # Group by scale/layer
        groups = defaultdict(list)
        for r in self.records:
            groups[(r.scale, r.layer)].append(r)
            if r.scale not in summary.scales: summary.scales.append(r.scale)
            if r.layer not in summary.layers: summary.layers.append(r.layer)
            
        for (scale, layer), recs in groups.items():
            if not recs: continue
            
            agg = AggregateResult(scale=scale, layer=layer, num_runs=len(recs))
            
            agg.avg_nodes = statistics.mean([r.nodes for r in recs])
            agg.avg_edges = statistics.mean([r.edges for r in recs])
            agg.avg_density = statistics.mean([r.density for r in recs])
            
            agg.avg_time_analysis = statistics.mean([r.time_analysis for r in recs])
            agg.avg_time_simulation = statistics.mean([r.time_simulation for r in recs])
            agg.avg_time_total = statistics.mean([r.time_total for r in recs])
            if agg.avg_time_analysis > 0:
                agg.speedup_ratio = agg.avg_time_simulation / agg.avg_time_analysis
                
            agg.avg_spearman = statistics.mean([r.spearman for r in recs])
            agg.avg_f1 = statistics.mean([r.f1_score for r in recs])
            agg.avg_precision = statistics.mean([r.precision for r in recs])
            agg.avg_recall = statistics.mean([r.recall for r in recs])
            agg.avg_top5 = statistics.mean([r.top5_overlap for r in recs])
            agg.avg_top10 = statistics.mean([r.top10_overlap for r in recs])
            agg.avg_rmse = statistics.mean([r.rmse for r in recs])
            
            agg.num_passed = sum(1 for r in recs if r.passed)
            agg.pass_rate = (agg.num_passed / agg.num_runs) * 100
            
            summary.aggregates.append(agg)
            
        return summary

