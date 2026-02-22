"""
Benchmark Data Models

Dataclasses for benchmark records, aggregation, scenarios, and summaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any


# =============================================================================
# BenchmarkRecord — one (scale, layer, seed) execution
# =============================================================================

@dataclass
class BenchmarkRecord:
    """Result of a single benchmark run for one scale × layer × seed."""

    # --- Identification ---
    run_id: str
    timestamp: str
    scale: str
    layer: str
    seed: int

    # --- Graph statistics ---
    nodes: int = 0
    edges: int = 0
    density: float = 0.0

    # --- Timing (milliseconds) ---
    time_generation: float = 0.0
    time_import: float = 0.0
    time_analysis: float = 0.0
    time_simulation: float = 0.0
    time_validation: float = 0.0
    time_total: float = 0.0

    # --- Core validation metrics ---
    spearman: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    rmse: float = 0.0
    top5_overlap: float = 0.0
    top10_overlap: float = 0.0
    auc_pr: float = 0.0

    # --- Baseline Spearman correlations ---
    spearman_bc: float = 0.0      # Raw Betweenness Centrality
    spearman_degree: float = 0.0  # Raw Degree Centrality
    spearman_random: float = 0.0  # Uniform Random ranking

    # --- Status ---
    passed: bool = False
    targets_met: int = 0
    targets_total: int = 6
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# AggregateResult — mean metrics for a (scale, layer) group
# =============================================================================

@dataclass
class AggregateResult:
    """Aggregated metrics for one scale × layer combination across runs."""

    scale: str
    layer: str
    num_runs: int

    # Graph stats (averages)
    avg_nodes: float = 0.0
    avg_edges: float = 0.0
    avg_density: float = 0.0

    # Timing (averages, ms)
    avg_time_analysis: float = 0.0
    avg_time_simulation: float = 0.0
    avg_time_total: float = 0.0
    speedup_ratio: float = 0.0  # simulation / analysis

    # Validation (averages)
    avg_spearman: float = 0.0
    std_spearman: float = 0.0
    avg_f1: float = 0.0
    std_f1: float = 0.0
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_top5: float = 0.0
    avg_top10: float = 0.0
    avg_rmse: float = 0.0
    avg_auc_pr: float = 0.0

    # Baseline averages
    avg_spearman_bc: float = 0.0
    avg_spearman_degree: float = 0.0
    avg_spearman_random: float = 0.0

    # Pass rate
    num_passed: int = 0
    pass_rate: float = 0.0


# =============================================================================
# BenchmarkSummary — top-level report object
# =============================================================================

@dataclass
class BenchmarkSummary:
    """Complete benchmark summary with records and aggregates."""

    timestamp: str
    duration: float  # seconds
    total_runs: int
    passed_runs: int

    scales: List[str] = field(default_factory=list)
    layers: List[str] = field(default_factory=list)

    records: List[BenchmarkRecord] = field(default_factory=list)
    aggregates: List[AggregateResult] = field(default_factory=list)

    # Overall metrics
    overall_spearman: float = 0.0
    overall_f1: float = 0.0
    overall_pass_rate: float = 0.0

    # Best / worst
    best_config: Optional[str] = None
    worst_config: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "duration": self.duration,
            "total_runs": self.total_runs,
            "passed_runs": self.passed_runs,
            "overall_pass_rate": self.overall_pass_rate,
            "overall_spearman": self.overall_spearman,
            "overall_f1": self.overall_f1,
            "scales": self.scales,
            "layers": self.layers,
            "best_config": self.best_config,
            "worst_config": self.worst_config,
            "aggregates": [asdict(a) for a in self.aggregates],
            "records": [r.to_dict() for r in self.records],
        }


# =============================================================================
# BenchmarkScenario — input configuration
# =============================================================================

@dataclass
class BenchmarkScenario:
    """Configuration for a benchmark scenario."""

    name: str
    scale: Optional[str] = None
    config_path: Optional[Path] = None
    layers: List[str] = field(default_factory=lambda: ["app", "infra", "system"])
    runs: int = 3

    def __post_init__(self):
        if not self.scale and not self.config_path:
            raise ValueError("Either 'scale' or 'config_path' must be provided")

    @property
    def label(self) -> str:
        return self.scale if self.scale else "custom"