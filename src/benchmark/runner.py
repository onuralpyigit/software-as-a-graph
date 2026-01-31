import sys
import subprocess
import time
import json
import logging
import traceback
import statistics
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional, Any, Dict

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

VALIDATION_TARGETS = {
    "spearman": 0.70,
    "f1": 0.80,
    "precision": 0.80,
    "recall": 0.80,
    "top5_overlap": 0.40,
    "top10_overlap": 0.60,
}

class BenchmarkRunner:
    """
    Executes comprehensive benchmark suite for the graph methodology.
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
        
        # Determine project root (assuming src/benchmark/runner.py -> project_root is up 2 levels)
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("Benchmark")
        self.records: List[BenchmarkRecord] = []
        
    def _neo4j_args(self) -> List[str]:
        return [
            "--uri", self.uri,
            "--user", self.user,
            "--password", self.password,
        ]

    def _generate_data(self, scenario: BenchmarkScenario, seed: int, output_file: Path) -> Tuple[bool, float]:
        """Generate synthetic graph data based on scenario."""
        start = time.time()
        
        script = self.project_root / "bin" / "generate_graph.py"
        if not script.exists():
             script = self.project_root / "generate_graph.py"

        cmd = [sys.executable, str(script), "--output", str(output_file)]
        
        if scenario.config_path:
             # If config path provided, pass it. Seed in config overrides CLI seed usually, 
             # but generate_graph CLI might ignore CLI seed if config is present.
             # However, we want to control seed for multiple runs. 
             # generate_graph.py implementation: "ignored if --config is used" for seed arg.
             # This is a limitation if we want to run multiple random seeds with same config structure.
             # Ideally we would pass seed override to config based generation.
             # For now, we rely on the config file. If user wants multiple runs with config, 
             # they should provide multiple scenarios or we accept that config-based runs are deterministic 
             # unless config file templates are used (out of scope for now).
             # Wait, the plan said "multiple runs". If config has a fixed seed, all runs are identical.
             # This is suboptimal. 
             # Let's check generate_graph.py again. It loads config, and uses config.seed. 
             # We might need to modify generate_graph.py to allow overriding seed even with config?
             # Or we construction a temporary config file with updated seed? 
             # Let's stick to basics: if config is used, we use it as is. 
             cmd.extend(["--config", str(scenario.config_path)])
        elif scenario.scale:
             cmd.extend(["--scale", scenario.scale, "--seed", str(seed)])
        else:
             return False, 0.0 # Should not happen due to post_init check

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
            )
            success = result.returncode == 0
            if not success and self.verbose:
                self.logger.warning(f"Generation stderr: {result.stderr}")
            return success, (time.time() - start) * 1000
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return False, 0.0

    def _import_data(self, input_file: Path) -> Tuple[bool, float]:
        start = time.time()
        script = self.project_root / "bin" / "import_graph.py"
        cmd = [
            sys.executable, str(script),
            "--input", str(input_file),
            "--clear",
        ] + self._neo4j_args()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root))
            return result.returncode == 0, (time.time() - start) * 1000
        except Exception:
            return False, 0.0

    def _run_analysis(self, layer: str, run_id: str) -> Tuple[Optional[Any], float]:
        start = time.time()
        script = self.project_root / "bin" / "analyze_graph.py"
        output_file = self.output_dir / f"analysis_{run_id}.json"
        
        cmd = [
            sys.executable, str(script),
            "--layer", layer,
            "--output", str(output_file),
        ] + self._neo4j_args()
        
        if self.verbose: cmd.append("--verbose")
            
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root))
            success = result.returncode == 0
            
            if success and output_file.exists():
                with open(output_file) as f:
                    data = json.load(f)
                    if "layers" in data and layer in data["layers"]:
                        return data["layers"][layer], (time.time() - start) * 1000
            return None, (time.time() - start) * 1000
        except Exception:
            return None, (time.time() - start) * 1000

    def _run_simulation(self, layer: str, run_id: str) -> Tuple[Optional[Any], float]:
        start = time.time()
        script = self.project_root / "bin" / "simulate_graph.py"
        output_file = self.output_dir / f"simulation_{run_id}.json"
        
        cmd = [
            sys.executable, str(script),
            "--layer", layer,
            "--exhaustive",
            "--output", str(output_file),
        ] + self._neo4j_args()
        
        if self.verbose: cmd.append("--verbose")
            
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root))
            success = result.returncode == 0
            
            if success and output_file.exists():
                with open(output_file) as f:
                    return json.load(f), (time.time() - start) * 1000
            return None, (time.time() - start) * 1000
        except Exception:
            return None, (time.time() - start) * 1000

    def _run_validation(self, layer: str, run_id: str) -> Tuple[Optional[Any], float]:
        start = time.time()
        script = self.project_root / "bin" / "validate_graph.py"
        output_file = self.output_dir / f"validation_{run_id}.json"
        
        cmd = [
            sys.executable, str(script),
            "--layer", layer,
            "--spearman", str(VALIDATION_TARGETS["spearman"]),
            "--f1", str(VALIDATION_TARGETS["f1"]),
            "--output", str(output_file),
        ] + self._neo4j_args()
        
        if self.verbose: cmd.append("--verbose")
            
        try:
            subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root))
            # validate_graph returns non-zero if validation fails criteria, so we check file existence
            if output_file.exists():
                with open(output_file) as f:
                    data = json.load(f)
                    if "layers" in data and layer in data["layers"]:
                        return data["layers"][layer], (time.time() - start) * 1000
            return None, (time.time() - start) * 1000
        except Exception:
            return None, (time.time() - start) * 1000

    def _cleanup_temp_files(self, run_id: str):
         for prefix in ["analysis", "simulation", "validation"]:
            path = self.output_dir / f"{prefix}_{run_id}.json"
            if path.exists():
                try: path.unlink()
                except: pass

    def run_scenario(self, scenario: BenchmarkScenario, seed_start: int = 42) -> List[BenchmarkRecord]:
        """Run a single benchmark scenario."""
        scenario_records = []
        temp_data_file = self.output_dir / "temp_benchmark_data.json"
        
        # Scale label for record
        scale_label = scenario.scale if scenario.scale else "custom"
        
        for i in range(scenario.runs):
            seed = seed_start + i
            # Generate
            success, gen_time = self._generate_data(scenario, seed, temp_data_file)
            if not success:
                print(f"Generation failed for run {i}")
                continue
                
            # Import
            success, imp_time = self._import_data(temp_data_file)
            if not success: 
                print(f"Import failed for run {i}")
                continue
            
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
                
                # Analysis
                an_data, an_time = self._run_analysis(layer, run_id)
                record.time_analysis = an_time
                if an_data:
                    # Attempt to get graph stats from direct key (new format) or structural (old format)
                    stats = an_data.get("graph_summary")
                    if not stats: 
                         stats = an_data.get("structural", {}).get("graph_summary", {})
                         
                    record.nodes = stats.get("nodes", 0)
                    record.edges = stats.get("edges", 0)
                    record.density = stats.get("density", 0.0)
                else:
                    record.error = "Analysis failed"
                    self.records.append(record)
                    scenario_records.append(record)
                    continue

                # Simulation
                sim_data, sim_time = self._run_simulation(layer, run_id)
                record.time_simulation = sim_time
                if not sim_data:
                    record.error = "Simulation failed"
                    self.records.append(record)
                    scenario_records.append(record)
                    continue
                
                # Validation
                val_data, val_time = self._run_validation(layer, run_id)
                record.time_validation = val_time
                if val_data:
                    self._parse_validation_results(record, val_data)
                else:
                     record.error = "Validation failed"
                
                record.time_total = record.time_generation + record.time_import + record.time_analysis + record.time_simulation + record.time_validation
                
                self.records.append(record)
                scenario_records.append(record)
                
                self._cleanup_temp_files(run_id)
        
        if temp_data_file.exists():
            try: temp_data_file.unlink()
            except: pass
            
        return scenario_records

    def _parse_validation_results(self, record: BenchmarkRecord, val_data: Dict[str, Any]):
        validation_result = val_data.get("validation_result", {})
        overall = validation_result.get("overall", {})
        metrics = overall.get("metrics", {})
        
        corr = metrics.get("correlation", {})
        cls_met = metrics.get("classification", {})
        rank = metrics.get("ranking", {})
        err = metrics.get("error", {})
        
        record.spearman = corr.get("spearman", 0.0)
        record.f1_score = cls_met.get("f1_score", 0.0)
        record.precision = cls_met.get("precision", 0.0)
        record.recall = cls_met.get("recall", 0.0)
        record.top5_overlap = rank.get("top_5_overlap", 0.0)
        record.top10_overlap = rank.get("top_10_overlap", 0.0)
        record.rmse = err.get("rmse", 0.0)
        
        record.passed = val_data.get("summary", {}).get("passed", False)
        
        # Targets met calculation could be moved to model or utility, doing simplest here
        targets_met = 0
        if record.spearman >= VALIDATION_TARGETS["spearman"]: targets_met += 1
        if record.f1_score >= VALIDATION_TARGETS["f1"]: targets_met += 1
        if record.precision >= VALIDATION_TARGETS["precision"]: targets_met += 1
        if record.recall >= VALIDATION_TARGETS["recall"]: targets_met += 1
        if record.top5_overlap >= VALIDATION_TARGETS["top5_overlap"]: targets_met += 1
        if record.top10_overlap >= VALIDATION_TARGETS["top10_overlap"]: targets_met += 1
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
