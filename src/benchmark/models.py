from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

@dataclass
class BenchmarkRecord:
    """Single benchmark iteration result."""
    # Identification
    run_id: str
    timestamp: str
    scale: str
    layer: str
    seed: int
    
    # Graph statistics
    nodes: int = 0
    edges: int = 0
    density: float = 0.0
    components_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Performance timings (milliseconds)
    time_generation: float = 0.0
    time_import: float = 0.0
    time_analysis: float = 0.0
    time_simulation: float = 0.0
    time_validation: float = 0.0
    time_total: float = 0.0
    
    # Validation metrics
    spearman: float = 0.0
    spearman_p: float = 1.0
    pearson: float = 0.0
    kendall: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    accuracy: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    top5_overlap: float = 0.0
    top10_overlap: float = 0.0
    
    # Classification counts
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    # Status
    passed: bool = False
    targets_met: int = 0
    targets_total: int = 6
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AggregateResult:
    """Aggregated results for a scale/layer combination."""
    scale: str
    layer: str
    num_runs: int
    
    # Graph stats (averages)
    avg_nodes: float = 0.0
    avg_edges: float = 0.0
    avg_density: float = 0.0
    
    # Timing stats (averages in ms)
    avg_time_analysis: float = 0.0
    avg_time_simulation: float = 0.0
    avg_time_total: float = 0.0
    speedup_ratio: float = 0.0  # simulation_time / analysis_time
    
    # Validation stats
    avg_spearman: float = 0.0
    std_spearman: float = 0.0
    avg_f1: float = 0.0
    std_f1: float = 0.0
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_top5: float = 0.0
    avg_top10: float = 0.0
    avg_rmse: float = 0.0
    
    # Success rate
    pass_rate: float = 0.0
    num_passed: int = 0


@dataclass
class BenchmarkSummary:
    """Complete benchmark summary."""
    timestamp: str
    duration: float
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
    
    # Best/worst performers
    best_config: Optional[str] = None
    worst_config: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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

@dataclass
class BenchmarkScenario:
    """Configuration for a benchmark scenario."""
    name: str
    scale: Optional[str] = None  # Preset scale name
    config_path: Optional[Path] = None  # Path to graph config
    layers: List[str] = field(default_factory=lambda: ["app", "infra", "system"])
    runs: int = 3
    
    def __post_init__(self):
        if not self.scale and not self.config_path:
            raise ValueError("Either scale or config_path must be provided")

