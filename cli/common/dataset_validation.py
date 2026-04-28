#!/usr/bin/env python3
"""
cli/common/dataset_validation.py
================================
Topology-class validation harness for the Software-as-a-Graph methodology.

Validates the six-step pipeline under four structural conditions:

  CLASS 1  Fan-out dominated   → scenarios 01, 02   (AV, IoT)
  CLASS 2  Dense pubsub        → scenarios 03, 04   (Finance, Healthcare)
  CLASS 3  Anti-pattern / SPOF → scenario  05        (Hub-and-Spoke)
  CLASS 4  Sparse distributed  → scenario  06        (Microservices)

Two operating modes:

  Pipeline mode (default):
      Runs generate → run.py --all for each scenario; reads validation JSON from
      the result directory.  Requires a running Neo4j instance.

  Results mode (--from-results DIR):
      Reads pre-existing validation JSON files from DIR.  No Neo4j needed.

Usage:
    # Full pipeline run (Neo4j required)
    python cli/generate_graph.py validate \\
        --neo4j-uri bolt://localhost:7687 \\
        --neo4j-user neo4j --neo4j-password password

    # Read from existing pipeline outputs
    python cli/generate_graph.py validate --from-results output/topology_validation

    # Single topology class, verbose
    python cli/generate_graph.py validate --class fan_out --verbose

    # Dry-run — print plan without executing
    python cli/generate_graph.py validate --dry-run

Output:
    output/topology_validation/<class_id>/<scenario_name>_results/
    output/topology_validation/topology_report.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent.parent
# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------
CYAN  = "\033[96m"; GREEN = "\033[92m"; RED   = "\033[91m"
AMBER = "\033[93m"; BOLD  = "\033[1m";  RESET = "\033[0m"
DIM   = "\033[2m"

def _hdr(msg: str)  -> None: print(f"\n{CYAN}{BOLD}{'='*68}\n  {msg}\n{'='*68}{RESET}")
def _sub(msg: str)  -> None: print(f"\n  {CYAN}{BOLD}{msg}{RESET}")
def _ok(msg: str)   -> None: print(f"  {GREEN}✓{RESET}  {msg}")
def _warn(msg: str) -> None: print(f"  {AMBER}⚠ {RESET}  {msg}")
def _err(msg: str)  -> None: print(f"  {RED}✗{RESET}  {msg}")
def _info(msg: str) -> None: print(f"  {DIM}·{RESET}  {msg}")

def _gate(passed: bool) -> str:
    return f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"


# ===========================================================================
# Topology class definitions
# ===========================================================================

@dataclass
class TopoClass:
    """Definition of one topological validation class."""
    id: str
    label: str
    scenarios: List[str]           # scenario name prefixes (e.g. "scenario_01")
    primary_dimension: str         # RMAV dimension that should dominate
    discriminating_signal: str     # structural property that drives correctness
    # Gate thresholds for this class (may be tighter/looser than global defaults)
    spearman_min: float = 0.80
    f1_min: float       = 0.85
    precision_min: float = 0.85
    spof_f1_min: float  = 0.90
    ftr_max: float      = 0.20
    pg_min: float       = 0.03     # Predictive Gain
    # Per-class narrative
    expected_driver: str = ""
    expected_weak_gate: str = ""

TOPOLOGY_CLASSES: Dict[str, TopoClass] = {
    "fan_out": TopoClass(
        id="fan_out",
        label="Fan-out dominated",
        scenarios=["scenario_01", "scenario_02"],
        primary_dimension="reliability",
        discriminating_signal="subscriber fan-out on topics → broker/topic betweenness centrality",
        spearman_min=0.82,
        f1_min=0.85,
        precision_min=0.83,
        spof_f1_min=0.88,
        expected_driver="R(v): PageRank + DG_in → topics/brokers with extreme subscriber counts "
                        "should rank at the top of Q(v) and match exhaustive simulation I(v).",
        expected_weak_gate="Infrastructure layer ρ may be lower (≈0.54) due to homogeneous "
                           "node load distribution; app-layer ρ should exceed threshold.",
    ),
    "dense_pubsub": TopoClass(
        id="dense_pubsub",
        label="Dense pubsub",
        scenarios=["scenario_03", "scenario_04"],
        primary_dimension="availability",
        discriminating_signal="articulation-point density + QoS weight on PERSISTENT/RELIABLE paths",
        spearman_min=0.83,
        f1_min=0.87,
        precision_min=0.86,
        spof_f1_min=0.92,
        pg_min=0.03,
        expected_driver="A(v): AP_c_directed × (1 + 0.30×QSPOF) → pubsub apps sitting on "
                        "critical articulation points with PERSISTENT QoS should dominate.",
        expected_weak_gate="Maintainability M(v) is secondary here; COCR@5 may be moderate "
                           "because code-quality signal is present but not the primary driver.",
    ),
    "anti_pattern": TopoClass(
        id="anti_pattern",
        label="Anti-pattern / SPOF",
        scenarios=["scenario_05"],
        primary_dimension="vulnerability",
        discriminating_signal="extreme broker overload ratio (2 brokers / 70 apps) → V(v) top-tier",
        spearman_min=0.87,   # tighter: deliberate signal is strong
        f1_min=0.90,
        precision_min=0.90,
        spof_f1_min=0.95,   # brokers must be classified CRITICAL
        pg_min=0.05,
        expected_driver="V(v): DG_out + DepDensity → both overloaded brokers must appear in "
                        "CRITICAL tier.  Broker failure impact ≥ 0.5 × total_apps in simulation.",
        expected_weak_gate="ρ may be artificially high (very clear signal); the real test is "
                           "SPOF_F1 and whether Precision ≥ 0.90 (not falsely flagging unrelated apps).",
    ),
    "sparse": TopoClass(
        id="sparse",
        label="Sparse / well-distributed",
        scenarios=["scenario_06"],
        primary_dimension="vulnerability",
        discriminating_signal="low fan-in and fan-out per topic → classifier must not over-flag",
        spearman_min=0.78,   # slightly relaxed: weaker structural signal
        f1_min=0.82,
        precision_min=0.88,  # precision is the primary gate for this class
        spof_f1_min=0.85,
        ftr_max=0.15,        # tighter FTR: the key failure mode is false positives
        pg_min=0.03,
        expected_driver="Precision: box-plot IQR-based thresholds must not promote ordinary "
                        "components to CRITICAL.  FTR (False Top Rate) ≤ 0.15 is decisive.",
        expected_weak_gate="Recall may be lower than other classes because there are fewer "
                           "true critical components to find in a sparse, well-balanced graph.",
    ),
}

# Scenario → topology class reverse mapping
SCENARIO_TO_CLASS: Dict[str, str] = {}
for _cls in TOPOLOGY_CLASSES.values():
    for _s in _cls.scenarios:
        SCENARIO_TO_CLASS[_s] = _cls.id


# ===========================================================================
# Result structures
# ===========================================================================

@dataclass
class ScenarioValidation:
    scenario_name: str
    topo_class_id: str
    layer: str
    result_file: str
    # Core metrics
    spearman: float = 0.0
    kendall:  float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall:    float = 0.0
    top5_overlap: float = 0.0
    ndcg_10:  float = 0.0
    predictive_gain: float = 0.0
    # Dimensional spearman
    reliability_spearman:     float = 0.0
    availability_spearman:    float = 0.0
    maintainability_spearman: float = 0.0
    vulnerability_spearman:   float = 0.0
    # Specialist metrics
    spof_f1:  float = 0.0
    ccr_5:    float = 0.0
    cocr_5:   float = 0.0
    ahcr_5:   float = 0.0
    ftr:      float = 0.0
    # Gates
    gates: Dict[str, bool] = field(default_factory=dict)
    # Meta
    passed: bool = False
    elapsed_s: float = 0.0
    error: str = ""
    sample_size: int = 0


@dataclass
class TopoClassResult:
    topo_class_id: str
    label: str
    scenario_results: List[ScenarioValidation] = field(default_factory=list)
    # Aggregated across scenarios / layers
    mean_spearman:   float = 0.0
    mean_f1:         float = 0.0
    mean_precision:  float = 0.0
    mean_spof_f1:    float = 0.0
    mean_ftr:        float = 0.0
    mean_pg:         float = 0.0
    # Class-level pass/fail
    spearman_ok:   bool = False
    f1_ok:         bool = False
    precision_ok:  bool = False
    spof_f1_ok:    bool = False
    ftr_ok:        bool = False
    class_passed:  bool = False
    # Primary dimension dominance confirmed
    primary_dim_dominance: bool = False


@dataclass
class TopologyReport:
    generated_at: str
    class_results: Dict[str, TopoClassResult] = field(default_factory=dict)
    overall_passed: bool = False
    classes_passed: int = 0
    classes_total: int = 0
    summary_notes: List[str] = field(default_factory=list)


# ===========================================================================
# Result file discovery and parsing
# ===========================================================================

_VALIDATION_FILE_CANDIDATES = [
    "validation_result.json",
    "validation_results.json",
    "validate_result.json",
    "pipeline_validation.json",
]

def _find_validation_json(result_dir: Path) -> Optional[Path]:
    """Look for a validation result JSON in result_dir using common filenames."""
    for name in _VALIDATION_FILE_CANDIDATES:
        p = result_dir / name
        if p.exists():
            return p
    # Fallback: any JSON that looks like a validation result
    for p in result_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text())
            if "spearman" in str(data) or "f1_score" in str(data):
                return p
        except Exception:
            continue
    return None


def _safe_float(d: Any, *keys: str, default: float = 0.0) -> float:
    """Navigate nested dict with dotted keys and return float."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k)
        else:
            return default
    if d is None:
        return default
    try:
        return float(d)
    except (TypeError, ValueError):
        return default


def _safe_bool(d: Any, *keys: str, default: bool = False) -> bool:
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k)
        else:
            return default
    if d is None:
        return default
    return bool(d)


def parse_validation_json(
    path: Path,
    scenario_name: str,
    topo_class_id: str,
    layer: str = "app",
) -> ScenarioValidation:
    """Parse a validation result JSON into a ScenarioValidation record."""
    sv = ScenarioValidation(
        scenario_name=scenario_name,
        topo_class_id=topo_class_id,
        layer=layer,
        result_file=str(path),
    )
    try:
        raw = json.loads(path.read_text())

        # Handle PipelineResult wrapper (layers dict) vs. LayerValidationResult directly
        if "layers" in raw and isinstance(raw["layers"], dict):
            # PipelineResult → pick the requested layer, fall back to "app"
            layer_data = raw["layers"].get(layer) or raw["layers"].get("app") or next(
                iter(raw["layers"].values()), {}
            )
        else:
            layer_data = raw

        # Core correlation / classification / ranking
        sv.spearman   = _safe_float(layer_data, "spearman") or \
                        _safe_float(layer_data, "overall", "correlation", "spearman")
        sv.kendall    = _safe_float(layer_data, "overall", "correlation", "kendall")
        sv.f1_score   = _safe_float(layer_data, "f1_score") or \
                        _safe_float(layer_data, "overall", "classification", "f1_score")
        sv.precision  = _safe_float(layer_data, "overall", "classification", "precision") or \
                        _safe_float(layer_data, "precision")
        sv.recall     = _safe_float(layer_data, "overall", "classification", "recall") or \
                        _safe_float(layer_data, "recall")
        sv.top5_overlap = _safe_float(layer_data, "overall", "ranking", "top_5_overlap") or \
                          _safe_float(layer_data, "top_5_overlap")
        sv.ndcg_10    = _safe_float(layer_data, "overall", "ranking", "ndcg_at_10") or \
                        _safe_float(layer_data, "ndcg_10")
        sv.sample_size = int(_safe_float(layer_data, "predicted_count") or
                             _safe_float(layer_data, "predicted_components"))

        # Predictive gain
        sv.predictive_gain = _safe_float(layer_data, "composite", "predictive_gain") or \
                             _safe_float(layer_data, "predictive_gain")

        # Dimensional spearman
        dim = layer_data.get("dimensional_validation") or layer_data.get("dimensional") or {}
        sv.reliability_spearman     = _safe_float(dim, "reliability",     "spearman")
        sv.availability_spearman    = _safe_float(dim, "availability",    "spearman")
        sv.maintainability_spearman = _safe_float(dim, "maintainability", "spearman")
        sv.vulnerability_spearman   = _safe_float(dim, "vulnerability",   "spearman")

        # Specialist metrics
        sv.spof_f1  = _safe_float(dim, "availability",    "spof_f1")
        sv.ccr_5    = _safe_float(dim, "reliability",     "ccr_5")
        sv.cocr_5   = _safe_float(dim, "maintainability", "cocr_5")
        sv.ahcr_5   = _safe_float(dim, "vulnerability",   "ahcr_5")
        sv.ftr      = _safe_float(dim, "vulnerability",   "ftr")

        # Gates
        sv.gates  = layer_data.get("gates") or {}
        sv.passed = bool(layer_data.get("passed", False))

    except Exception as exc:
        sv.error = str(exc)

    return sv


# ===========================================================================
# Pipeline runner (subprocess mode)
# ===========================================================================

def _python_exe() -> str:
    venv = _ROOT / "software_system_env" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def run_pipeline_for_scenario(
    scenario_yaml: Path,
    output_dir: Path,
    layer: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_pass: str,
    dry_run: bool,
) -> Tuple[bool, Path, float]:
    """
    Generate dataset and run the full pipeline for one scenario YAML.
    Returns (success, result_dir, elapsed_seconds).
    """
    stem = scenario_yaml.stem
    dataset_path  = output_dir.parent / f"{stem}.json"
    result_dir    = output_dir / f"{stem}_results"
    result_dir.mkdir(parents=True, exist_ok=True)

    python = _python_exe()
    t0 = time.perf_counter()

    # -- Step A: generate dataset if missing
    if not dataset_path.exists():
        gen_cmd = [
            python, str(_ROOT / "bin" / "generate_graph.py"),
            "--config", str(scenario_yaml),
            "--output", str(dataset_path),
        ]
        _info(f"Generating {stem} ...")
        if not dry_run:
            r = subprocess.run(gen_cmd, capture_output=True, text=True)
            if r.returncode != 0:
                _err(f"Generation failed:\n{r.stderr[-800:]}")
                return False, result_dir, 0.0
    else:
        _info(f"Dataset exists: {dataset_path.name}")

    # -- Step B: full pipeline
    run_cmd = [
        python, str(_ROOT / "bin" / "run.py"),
        "--all",
        "--input",      str(dataset_path),
        "--output-dir", str(result_dir),
        "--layer",      layer,
        "--uri",        neo4j_uri,
        "--user",       neo4j_user,
        "--password",   neo4j_pass,
        "--clean",
    ]
    _info(f"Running pipeline → {result_dir.relative_to(_ROOT)}")

    if dry_run:
        _info(f"  [dry-run] CMD: {' '.join(run_cmd)}")
        return True, result_dir, 0.0

    log_file = result_dir / "pipeline.log"
    with open(log_file, "w") as log:
        r = subprocess.run(run_cmd, stdout=log, stderr=subprocess.STDOUT, text=True)

    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        _err(f"Pipeline failed (see {log_file})")
        return False, result_dir, elapsed

    return True, result_dir, elapsed


# ===========================================================================
# Aggregation
# ===========================================================================

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def aggregate_topo_class(
    cls_def: TopoClass,
    scenario_results: List[ScenarioValidation],
) -> TopoClassResult:
    valid = [sv for sv in scenario_results if not sv.error]

    result = TopoClassResult(
        topo_class_id=cls_def.id,
        label=cls_def.label,
        scenario_results=scenario_results,
    )

    if not valid:
        return result

    result.mean_spearman  = _mean([sv.spearman   for sv in valid])
    result.mean_f1        = _mean([sv.f1_score   for sv in valid])
    result.mean_precision = _mean([sv.precision  for sv in valid])
    result.mean_spof_f1   = _mean([sv.spof_f1    for sv in valid])
    result.mean_ftr       = _mean([sv.ftr        for sv in valid])
    result.mean_pg        = _mean([sv.predictive_gain for sv in valid])

    result.spearman_ok  = result.mean_spearman  >= cls_def.spearman_min
    result.f1_ok        = result.mean_f1        >= cls_def.f1_min
    result.precision_ok = result.mean_precision >= cls_def.precision_min
    result.spof_f1_ok   = result.mean_spof_f1   >= cls_def.spof_f1_min
    result.ftr_ok       = result.mean_ftr       <= cls_def.ftr_max

    # Primary dimension dominance: the declared primary dim should have the
    # highest dimensional spearman among the four dimensions
    dim_map = {
        "reliability":     [sv.reliability_spearman     for sv in valid],
        "availability":    [sv.availability_spearman     for sv in valid],
        "maintainability": [sv.maintainability_spearman  for sv in valid],
        "vulnerability":   [sv.vulnerability_spearman    for sv in valid],
    }
    dim_means = {k: _mean(v) for k, v in dim_map.items()}
    if dim_means:
        best_dim = max(dim_means, key=lambda k: dim_means[k])
        result.primary_dim_dominance = (best_dim == cls_def.primary_dimension)

    result.class_passed = (
        result.spearman_ok and result.f1_ok and result.precision_ok
    )
    return result


# ===========================================================================
# Report writer
# ===========================================================================

def build_report(class_results: Dict[str, TopoClassResult]) -> TopologyReport:
    passed = sum(1 for r in class_results.values() if r.class_passed)
    notes: List[str] = []

    for r in class_results.values():
        cls = TOPOLOGY_CLASSES[r.topo_class_id]
        if not r.class_passed:
            notes.append(
                f"{r.label}: class FAILED — "
                f"ρ={r.mean_spearman:.3f} (≥{cls.spearman_min}), "
                f"F1={r.mean_f1:.3f} (≥{cls.f1_min}), "
                f"Prec={r.mean_precision:.3f} (≥{cls.precision_min})"
            )
        if not r.primary_dim_dominance:
            notes.append(
                f"{r.label}: primary dimension '{cls.primary_dimension}' "
                "did not dominate — check RMAV weight calibration."
            )
        if r.topo_class_id == "sparse" and not r.ftr_ok:
            notes.append(
                f"Sparse/distributed: FTR={r.mean_ftr:.3f} > {cls.ftr_max} — "
                "box-plot classifier is over-flagging; check IQR thresholds."
            )

    return TopologyReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        class_results=class_results,
        overall_passed=(passed == len(class_results)),
        classes_passed=passed,
        classes_total=len(class_results),
        summary_notes=notes,
    )


def write_report(report: TopologyReport, path: Path) -> None:
    def _serial(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _serial(v) for k, v in asdict(obj).items()}
        if isinstance(obj, dict):
            return {k: _serial(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serial(i) for i in obj]
        return obj
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_serial(report), f, indent=2)
    _ok(f"Report → {path.relative_to(_ROOT)}")


# ===========================================================================
# Console summary
# ===========================================================================

def print_summary(report: TopologyReport) -> None:
    _hdr("Topology-Class Validation Summary")

    # Per-class table
    col = 26
    print(f"  {'CLASS':<{col}}  {'ρ':>6}  {'F1':>6}  {'Prec':>6}  "
          f"{'SPOF-F1':>7}  {'FTR':>5}  {'PG':>5}  {'Dim✓':>5}  {'STATUS':>6}")
    print(f"  {'-'*col}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*6}")

    for cls_id, r in report.class_results.items():
        cls = TOPOLOGY_CLASSES[cls_id]
        rho   = f"{r.mean_spearman:.3f}"
        f1    = f"{r.mean_f1:.3f}"
        prec  = f"{r.mean_precision:.3f}"
        spof  = f"{r.mean_spof_f1:.3f}"
        ftr_s = f"{r.mean_ftr:.3f}"
        pg_s  = f"{r.mean_pg:.3f}"
        dom   = f"{GREEN}yes{RESET}" if r.primary_dim_dominance else f"{AMBER}no {RESET}"
        st    = f"{GREEN}PASS{RESET}" if r.class_passed else f"{RED}FAIL{RESET}"

        # Colour metric cells that fail their threshold
        rho_c  = rho  if r.spearman_ok  else f"{AMBER}{rho}{RESET}"
        f1_c   = f1   if r.f1_ok        else f"{AMBER}{f1}{RESET}"
        prec_c = prec if r.precision_ok else f"{AMBER}{prec}{RESET}"
        spof_c = spof if r.spof_f1_ok   else f"{AMBER}{spof}{RESET}"
        ftr_c  = ftr_s if r.ftr_ok      else f"{AMBER}{ftr_s}{RESET}"

        print(f"  {r.label:<{col}}  {rho_c:>6}  {f1_c:>6}  {prec_c:>6}  "
              f"{spof_c:>7}  {ftr_c:>5}  {pg_s:>5}  {dom:>5}  {st:>6}")

    print()
    print(f"  Classes passed  : {report.classes_passed} / {report.classes_total}")
    overall_s = f"{GREEN}OVERALL PASS{RESET}" if report.overall_passed else f"{RED}OVERALL FAIL{RESET}"
    print(f"  Verdict         : {overall_s}")

    if report.summary_notes:
        print()
        _sub("Diagnosis notes")
        for note in report.summary_notes:
            _warn(note)

    # Per-scenario breakdown
    print()
    _sub("Per-scenario detail")
    for cls_id, r in report.class_results.items():
        print(f"\n  {CYAN}{r.label}{RESET}")
        for sv in r.scenario_results:
            err_suffix = f"  {RED}[{sv.error[:60]}]{RESET}" if sv.error else ""
            print(f"    {sv.scenario_name:<44s}  "
                  f"ρ={sv.spearman:.3f}  F1={sv.f1_score:.3f}  "
                  f"Prec={sv.precision:.3f}  n={sv.sample_size}"
                  f"{err_suffix}")
            if sv.gates:
                gate_str = "  ".join(
                    f"{k}={'✓' if v else '✗'}"
                    for k, v in sorted(sv.gates.items())
                )
                print(f"    {'':44s}  gates: {gate_str}")

    # Class-level threshold reference
    print()
    _sub("Threshold reference by class")
    print(f"  {'CLASS':<26}  {'ρ≥':>5}  {'F1≥':>5}  {'Prec≥':>5}  {'SPOF-F1≥':>9}  {'FTR≤':>5}  {'Primary dim'}")
    print(f"  {'-'*26}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*9}  {'-'*5}  {'-'*16}")
    for cls_id, cls in TOPOLOGY_CLASSES.items():
        r = report.class_results.get(cls_id)
        if r is None:
            continue
        print(f"  {cls.label:<26}  {cls.spearman_min:>5.2f}  {cls.f1_min:>5.2f}  "
              f"{cls.precision_min:>5.2f}  {cls.spof_f1_min:>9.2f}  "
              f"{cls.ftr_max:>5.2f}  {cls.primary_dimension}")


# ===========================================================================
# CLI
# ===========================================================================

def add_validation_arguments(p: argparse.ArgumentParser) -> None:
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--from-results", type=Path, metavar="DIR",
                      help="Read pre-existing validation JSON files from DIR")
    mode.add_argument("--dry-run", action="store_true",
                      help="Print the execution plan without running anything")

    p.add_argument("--input-dir",    type=Path, default=_ROOT / "data")
    p.add_argument("--output-dir",   type=Path, default=_ROOT / "output" / "topology_validation")
    p.add_argument("--layer",        type=str,  default="app",
                   choices=["app", "mw", "infra", "system"],
                   help="Pipeline layer to validate")
    p.add_argument("--class",        dest="topo_class", type=str, default=None,
                   choices=list(TOPOLOGY_CLASSES.keys()),
                   help="Validate a single topology class only")
    p.add_argument("--report",       type=Path,
                   default=_ROOT / "output" / "topology_validation" / "topology_report.json",
                   help="Output path for the JSON report")
    # Neo4j (pipeline mode only)
    p.add_argument("--neo4j-uri",      default="bolt://localhost:7687")
    p.add_argument("--neo4j-user",     default="neo4j")
    p.add_argument("--neo4j-password", default="password")
    p.add_argument("-v", "--verbose", action="store_true")


# ===========================================================================
# Main Routine for Validation
# ===========================================================================

def run_dataset_validation(args: argparse.Namespace) -> int:
    # Filter to requested class(es)
    active_classes = (
        {args.topo_class: TOPOLOGY_CLASSES[args.topo_class]}
        if args.topo_class else TOPOLOGY_CLASSES
    )

    _hdr("SaG Topology-Class Validation Harness")
    mode_label = (
        "read-results" if args.from_results else
        "dry-run"      if args.dry_run      else
        "pipeline"
    )
    print(f"  Mode       : {mode_label}")
    print(f"  Layer      : {args.layer}")
    print(f"  Classes    : {', '.join(active_classes)}")
    print(f"  Output dir : {args.output_dir}")

    class_results: Dict[str, TopoClassResult] = {}

    for cls_id, cls_def in active_classes.items():
        _hdr(f"Class: {cls_def.label}")
        _info(f"Primary dimension   : {cls_def.primary_dimension}")
        _info(f"Discriminating signal: {cls_def.discriminating_signal}")
        _info(f"Thresholds          : ρ≥{cls_def.spearman_min}  F1≥{cls_def.f1_min}  "
              f"Prec≥{cls_def.precision_min}  SPOF-F1≥{cls_def.spof_f1_min}  FTR≤{cls_def.ftr_max}")

        scenario_results: List[ScenarioValidation] = []

        for prefix in cls_def.scenarios:
            # Find the matching YAML
            yaml_files = sorted(args.input_dir.glob(f"{prefix}*.yaml"))
            if not yaml_files:
                _warn(f"No YAML found for prefix '{prefix}' in {args.input_dir}")
                continue
            yaml_path = yaml_files[0]
            scenario_name = yaml_path.stem

            if args.from_results:
                # --- Results mode: find JSON in from_results dir
                result_dir = args.from_results / cls_id / f"{scenario_name}_results"
                result_dir2 = args.from_results / f"{scenario_name}_results"
                # try both layouts
                result_json = _find_validation_json(result_dir) or _find_validation_json(result_dir2)
                if result_json is None:
                    _warn(f"No validation JSON found for {scenario_name} in {args.from_results}")
                    sv = ScenarioValidation(
                        scenario_name=scenario_name, topo_class_id=cls_id,
                        layer=args.layer, result_file="", error="result file not found",
                    )
                else:
                    _ok(f"Reading {result_json.relative_to(_ROOT)}")
                    sv = parse_validation_json(result_json, scenario_name, cls_id, args.layer)
                    if sv.error:
                        _err(f"Parse error: {sv.error}")
            else:
                # --- Pipeline mode: run generate → run.py --all
                _sub(f"Scenario: {scenario_name}")
                result_dir = args.output_dir / cls_id
                success, actual_result_dir, elapsed = run_pipeline_for_scenario(
                    scenario_yaml=yaml_path,
                    output_dir=result_dir,
                    layer=args.layer,
                    neo4j_uri=args.neo4j_uri,
                    neo4j_user=args.neo4j_user,
                    neo4j_pass=args.neo4j_password,
                    dry_run=args.dry_run,
                )
                if args.dry_run:
                    sv = ScenarioValidation(
                        scenario_name=scenario_name, topo_class_id=cls_id,
                        layer=args.layer, result_file="", elapsed_s=elapsed,
                    )
                elif not success:
                    sv = ScenarioValidation(
                        scenario_name=scenario_name, topo_class_id=cls_id,
                        layer=args.layer, result_file="", elapsed_s=elapsed,
                        error="pipeline execution failed",
                    )
                else:
                    result_json = _find_validation_json(actual_result_dir)
                    if result_json:
                        sv = parse_validation_json(result_json, scenario_name, cls_id, args.layer)
                        sv.elapsed_s = elapsed
                        if sv.error:
                            _err(f"Parse error: {sv.error}")
                        else:
                            _ok(f"{scenario_name}: ρ={sv.spearman:.3f}  F1={sv.f1_score:.3f}  "
                                f"Prec={sv.precision:.3f}  ({elapsed:.1f}s)")
                    else:
                        sv = ScenarioValidation(
                            scenario_name=scenario_name, topo_class_id=cls_id,
                            layer=args.layer, result_file="", elapsed_s=elapsed,
                            error="validation JSON not found after pipeline",
                        )

            scenario_results.append(sv)

        # Aggregate this class
        cr = aggregate_topo_class(cls_def, scenario_results)
        class_results[cls_id] = cr

        verdict = f"{GREEN}PASS{RESET}" if cr.class_passed else f"{RED}FAIL{RESET}"
        print(f"\n  Class result: {verdict}  "
              f"mean ρ={cr.mean_spearman:.3f}  "
              f"mean F1={cr.mean_f1:.3f}  "
              f"mean Prec={cr.mean_precision:.3f}")

    # Build and write report
    report = build_report(class_results)
    if not args.dry_run:
        write_report(report, args.report)

    print_summary(report)
    return 0 if report.overall_passed else 1
