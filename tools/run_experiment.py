#!/usr/bin/env python3
"""
tools/qos_gnn_ablation_experiment.py
====================================

Headline experiment for the Middleware 2026 paper:

    "QoS-Aware Heterogeneous Graph Attention for Pre-Deployment
     Cascade Prediction in Publish-Subscribe Middleware"

For each scenario in the suite, this script trains a HeteroGAT criticality
model TWICE per seed:

    Arm A — topology-only: QoS edge weights and QoS-derived node features
            are masked to a uniform constant.
    Arm B — QoS-aware:     full edge_attr[:, 0] (w(e)) and node features
            indices 10..12 (w, w_in, w_out) are passed through unchanged.

Both arms share architecture, hyperparameters, seeds, and train/val/test
splits. The within-architecture contrast isolates the QoS contribution.

Per-scenario outputs:
    - paired Spearman ρ, F1@K, NDCG@10, RMSE per arm per seed
    - Δρ = ρ(QoS-aware) - ρ(topology-only)
    - paired t-test and Wilcoxon-signed-rank p-values (alternative='greater')
    - 95 % bootstrap CI on Δρ (B = 2000)

Aggregate outputs:
    - regression of Δρ on QoS Gini coefficient (slope, R², p-value)
    - Table 5 (headline) and Table 4 (per-scenario summary) as LaTeX
    - Master JSON suitable for downstream figure scripts


PRE-REQUISITES
--------------

This script consumes pre-computed artifacts produced by the standard
SaG pipeline. For each scenario `<S>` you need:

    data/generated/<S>.json                 — generated topology JSON
    results/<S>/structural_metrics.json     — Step 2 output
    results/<S>/impact_scores.json          — Step 4 output (multi-seed)
    results/<S>/rmav_scores.json            — Step 3a output (optional, for ensemble)

If any are missing, the scenario is skipped with a warning. To produce
them, run:

    bash cli/run_scenarios.sh
    # or, per scenario:
    python cli/generate_graph.py --config data/scenario_XX_*.yaml \
        --output data/generated/scenario_XX.json
    python cli/import_graph.py --input data/generated/scenario_XX.json --clear
    python cli/analyze_graph.py --layer system --predict
    python cli/simulate_graph.py fault-inject \
        --input data/generated/scenario_XX.json \
        --export-json --output results/scenario_XX/impact_scores.json \
        --seeds 42 123 456 789 2024


USAGE
-----

    # Full headline matrix (8 + ATM scenarios, 5 seeds, 300 epochs)
    python tools/qos_gnn_ablation_experiment.py \
        --scenarios scenario_01,scenario_02,scenario_03,scenario_04,\
scenario_05,scenario_06,scenario_07,scenario_08,scenario_10 \
        --seeds 42,123,456,789,2024 \
        --epochs 300 \
        --output-dir output/qos_ablation/

    # Smoke test (3 scenarios, 2 seeds, 50 epochs)
    python tools/qos_gnn_ablation_experiment.py \
        --scenarios scenario_01,scenario_05,scenario_10 \
        --seeds 42,123 \
        --epochs 50 \
        --output-dir output/qos_ablation_smoke/

    # Resume an interrupted run (skip already-completed combos)
    python tools/qos_gnn_ablation_experiment.py ... --skip-existing


METHODOLOGICAL NOTES
--------------------

1. QoS masking is applied at THREE levels in the topology-only arm:
   (a) every NetworkX edge's `weight` attribute is set to 1.0
   (b) structural_metrics fields qos_aggregate_weight, qos_weighted_in_degree,
       qos_weighted_out_degree are set to 0.0
   (c) (if available) the native qos_enabled=False flag of
       networkx_to_hetero_data is used, which additionally zeros
       edge_attr[:, 0] inside the HeteroData conversion.

   This three-level masking is a STRICT contrast: a QoS-blind GNN cannot
   recover QoS information through any input channel. Document this
   in §6.2 of the paper.

2. Statistical tests use the PAIRED design (per-seed pairs). Reviewers
   will check this. Both `ttest_rel` and `wilcoxon` are reported; agreement
   between the two strengthens the claim.

3. The aggregate Δρ-vs-Gini regression is the §6.5 mechanism analysis.
   A monotonically positive slope with p < 0.05 is the headline supporting
   evidence for the paper's central hypothesis.

License: same as parent repository.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is importable when invoked as `python tools/...`
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Heavy deps — fail fast with a useful message if missing.
try:
    import networkx as nx
    import numpy as np
    from scipy import stats
except ImportError as e:  # pragma: no cover
    print(f"[fatal] missing scientific dependency: {e}", file=sys.stderr)
    print("        run: pip install networkx numpy scipy", file=sys.stderr)
    sys.exit(2)

# SaG framework imports — these are required.
try:
    from saag.prediction.gnn_service import GNNService
    from saag.prediction import data_preparation as data_prep
except ImportError as e:  # pragma: no cover
    print(f"[fatal] cannot import saag modules: {e}", file=sys.stderr)
    print("        run from repo root and ensure PYTHONPATH=. is set", file=sys.stderr)
    sys.exit(2)


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("qos-ablation")
# Tame noisy children when running long experiments.
logging.getLogger("saag.prediction.trainer").setLevel(logging.WARNING)
logging.getLogger("saag.prediction.gnn_service").setLevel(logging.WARNING)


# ── Default scenario list ────────────────────────────────────────────────────

DEFAULT_SCENARIOS: List[str] = [
    "scenario_01",  # AV / ROS 2
    "scenario_02",  # IoT smart city
    "scenario_03",  # Financial trading
    "scenario_04",  # Healthcare
    "scenario_05",  # Hub-and-spoke (anti-pattern)
    "scenario_06",  # Microservices
    "scenario_07",  # Enterprise xlarge
    "scenario_08",  # Tiny regression
    "scenario_10",  # ATM (real-world inductive holdout)
]
DEFAULT_SEEDS: List[int] = [42, 123, 456, 789, 2024]


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class ScenarioPaths:
    """Filesystem layout for a single scenario."""
    name: str
    graph_json: Path
    structural_json: Path
    simulation_json: Path
    rmav_json: Optional[Path] = None  # may be absent

    def is_complete(self) -> bool:
        return (
            self.graph_json.exists()
            and self.structural_json.exists()
            and self.simulation_json.exists()
        )

    def missing_files(self) -> List[str]:
        out = []
        if not self.graph_json.exists():
            out.append(str(self.graph_json))
        if not self.structural_json.exists():
            out.append(str(self.structural_json))
        if not self.simulation_json.exists():
            out.append(str(self.simulation_json))
        return out


@dataclass
class ArmResult:
    """Test-set metrics for one (scenario, qos_enabled, seed) cell."""
    scenario: str
    qos_enabled: bool
    seed: int
    spearman_rho: float
    f1_score: float
    rmse: float
    mae: float
    ndcg_10: float
    top_5_overlap: float
    top_10_overlap: float
    train_seconds: float


@dataclass
class AblationReport:
    """Per-scenario paired comparison between arms."""
    scenario: str
    n_nodes: int
    n_edges: int
    n_topics: int
    qos_gini: float
    seeds: List[int]
    base_rhos: List[float]   # qos_enabled=False
    enr_rhos: List[float]    # qos_enabled=True
    base_rho_mean: float
    base_rho_std: float
    enr_rho_mean: float
    enr_rho_std: float
    delta_rho: float
    delta_rho_ci_low: float
    delta_rho_ci_high: float
    paired_t_pvalue: float
    wilcoxon_pvalue: float
    base_f1_mean: float
    enr_f1_mean: float
    base_ndcg_mean: float
    enr_ndcg_mean: float
    base_arm_results: List[ArmResult]
    enr_arm_results: List[ArmResult]


# ── QoS Gini coefficient ─────────────────────────────────────────────────────

def compute_qos_gini(graph: nx.DiGraph) -> float:
    """
    Gini coefficient over per-edge QoS-derived weights, scaled by Topic frequency.
    A value of 0 means every edge carries the same QoS weight (homogeneous workload);
    higher values indicate increasing heterogeneity. This serves as the continuous
    moderator variable for the §6.5 mechanism analysis.

    For each edge, we scale the categorical QoS weight by log1p(frequency) of the
    associated Topic to capture continuous traffic rate variance.
    """
    weights = []
    for u, v, attrs in graph.edges(data=True):
        w = attrs.get("weight")
        if w is None:
            continue
        try:
            wf = float(w)
        except (TypeError, ValueError):
            continue
        if wf <= 0.0:
            continue

        # Scale by topic frequency if either endpoint is a Topic node
        hz = 1.0
        u_attrs = graph.nodes[u]
        v_attrs = graph.nodes[v]
        if u_attrs.get("type") == "Topic":
            hz = float(u_attrs.get("frequency", u_attrs.get("topic_frequency", 1.0)) or 1.0)
        elif v_attrs.get("type") == "Topic":
            hz = float(v_attrs.get("frequency", v_attrs.get("topic_frequency", 1.0)) or 1.0)

        scale = math.log1p(max(0.0, hz))
        scale = max(scale, 0.1)  # preserve positive weight
        weights.append(wf * scale)

    if len(weights) < 2:
        return 0.0

    arr = np.sort(np.asarray(weights, dtype=float))
    n = arr.size
    total = arr.sum()
    if total <= 0.0:
        return 0.0
    gini = (2.0 * np.sum((np.arange(1, n + 1)) * arr) - (n + 1) * total) / (n * total)
    return float(max(0.0, min(1.0, gini)))



# ── QoS masking ──────────────────────────────────────────────────────────────

# The HeteroData node-feature indices listed in docs/prediction.md as
# QoS-derived. These are zeroed in the topology-only arm.
QOS_NODE_FEATURE_INDICES: Tuple[int, ...] = (10, 11, 12)
# Structural metric keys corresponding to those indices.
QOS_STRUCTURAL_KEYS: Tuple[str, ...] = (
    "qos_aggregate_weight",
    "qos_weighted_in_degree",
    "qos_weighted_out_degree",
)


def mask_qos_in_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """Return a deep-copied graph with every edge `weight` set to 1.0."""
    g = graph.copy()
    for u, v, k, attrs in (g.edges(keys=True, data=True) if g.is_multigraph()
                           else ((u, v, None, d) for u, v, d in g.edges(data=True))):
        attrs["weight"] = 1.0
    return g


def mask_qos_in_structural_metrics(
    sm: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Return a deep-copied structural-metrics dict with QoS fields zeroed."""
    out: Dict[str, Dict[str, Any]] = {}
    for node_id, metrics in sm.items():
        new = dict(metrics) if isinstance(metrics, dict) else {}
        for key in QOS_STRUCTURAL_KEYS:
            if key in new:
                new[key] = 0.0
        out[node_id] = new
    return out


# Detect whether the native qos_enabled flag exists on networkx_to_hetero_data
# (i.e. whether Change 1 from the implementation plan has been merged).
_NATIVE_QOS_FLAG_AVAILABLE: bool = (
    "qos_enabled" in inspect.signature(data_prep.networkx_to_hetero_data).parameters
)
if _NATIVE_QOS_FLAG_AVAILABLE:
    logger.info("Native qos_enabled flag detected on networkx_to_hetero_data — "
                "the script will use it in addition to graph-level masking.")
else:
    logger.info("Native qos_enabled flag NOT detected — falling back to "
                "graph-level masking only. Both approaches produce a valid "
                "topology-only arm; see methodology note (1) in the script header.")


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "to_dict") and callable(o.to_dict):
        return o.to_dict()
    raise TypeError(f"object of type {type(o).__name__} is not JSON-serializable")


def graph_from_topology_json(path: Path) -> nx.DiGraph:
    """
    Build a typed NetworkX DiGraph from the SaG topology JSON.

    Mirrors the structure expected by `networkx_to_hetero_data`: nodes carry
    a `type` attribute; edges carry `type`, `weight`, and (optionally)
    `path_count` attributes.
    """
    payload = load_json(path)
    g = nx.DiGraph()

    # ── Nodes ────────────────────────────────────────────────────────────
    for n in payload.get("nodes", []):
        g.add_node(
            n["id"],
            type="Node",
            name=n.get("name", n["id"]),
            cpu_cores=n.get("cpu_cores", 0),
            memory_gb=n.get("memory_gb", 0),
        )
    for b in payload.get("brokers", []):
        g.add_node(
            b["id"],
            type="Broker",
            name=b.get("name", b["id"]),
            max_connections=b.get("max_connections", 0),
        )
    for t in payload.get("topics", []):
        qos = t.get("qos", {})
        g.add_node(
            t["id"],
            type="Topic",
            name=t.get("name", t["id"]),
            size=t.get("size", 0),
            qos_durability=qos.get("durability"),
            qos_reliability=qos.get("reliability"),
            qos_transport_priority=qos.get("transport_priority"),
        )
    for a in payload.get("applications", []):
        g.add_node(
            a["id"],
            type="Application",
            name=a.get("name", a["id"]),
            criticality=bool(a.get("criticality", False)),
            code_metrics=a.get("code_metrics", {}),
        )
    for lib in payload.get("libraries", []):
        g.add_node(
            lib["id"],
            type="Library",
            name=lib.get("name", lib["id"]),
            code_metrics=lib.get("code_metrics", {}),
        )

    # ── Edges ────────────────────────────────────────────────────────────
    rels = payload.get("relationships", {})
    edge_type_map = {
        "publishes_to": "PUBLISHES_TO",
        "subscribes_to": "SUBSCRIBES_TO",
        "routes": "ROUTES",
        "runs_on": "RUNS_ON",
        "connects_to": "CONNECTS_TO",
        "uses": "USES",
        "depends_on": "DEPENDS_ON",
    }
    for k, etype in edge_type_map.items():
        for r in rels.get(k, []):
            src = r.get("from") or r.get("source") or r.get("src")
            dst = r.get("to") or r.get("target") or r.get("dst")
            if src is None or dst is None:
                continue
            attrs = {"type": etype}
            if "weight" in r:
                attrs["weight"] = float(r["weight"])
            if "path_count" in r:
                attrs["path_count"] = int(r["path_count"])
            g.add_edge(src, dst, **attrs)

    # Ensure every edge has a numeric `weight` so the QoS Gini and the
    # data-preparation conversion both see a value.
    for _, _, attrs in g.edges(data=True):
        if "weight" not in attrs or attrs["weight"] is None:
            attrs["weight"] = 1.0

    return g


def discover_scenario_paths(
    name: str,
    data_dir: Path,
    results_dir: Path,
) -> ScenarioPaths:
    """
    Resolve the four canonical artifact paths for a scenario by name.

    The scenario name should match the prefix of the YAML file, e.g.
    "scenario_01" matches "scenario_01_autonomous_vehicle.yaml".
    """
    graph_json = data_dir / "generated" / f"{name}.json"
    res_dir = results_dir / name
    return ScenarioPaths(
        name=name,
        graph_json=graph_json,
        structural_json=res_dir / "structural_metrics.json",
        simulation_json=res_dir / "impact_scores.json",
        rmav_json=res_dir / "rmav_scores.json",
    )


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci_paired_diff(
    a: List[float],
    b: List[float],
    *,
    n_iter: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Bootstrap (alpha/2, 1-alpha/2) CI on the mean of (b - a), preserving
    the paired structure. Returns (low, high). Falls back to (NaN, NaN)
    when the input is degenerate.
    """
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    diffs = b_arr - a_arr
    n = diffs.size
    samples = rng.choice(diffs, size=(n_iter, n), replace=True)
    means = samples.mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# ── Single-arm runner ────────────────────────────────────────────────────────

def run_single_arm(
    *,
    scenario: str,
    qos_enabled: bool,
    seed: int,
    graph: nx.DiGraph,
    structural_metrics: Dict[str, Dict[str, Any]],
    simulation_results: Dict[str, Dict[str, float]],
    rmav_scores: Optional[Dict[str, Dict[str, float]]],
    layer: str,
    epochs: int,
    patience: int,
    hidden: int,
    heads: int,
    num_layers: int,
    dropout: float,
    lr: float,
    train_ratio: float,
    val_ratio: float,
    checkpoint_root: Path,
) -> ArmResult:
    """Train one model and return its test-set metrics."""

    arm_label = "qosT" if qos_enabled else "qosF"
    ckpt_dir = checkpoint_root / scenario / f"{arm_label}_seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Apply masking for the topology-only arm ──────────────────────────
    if not qos_enabled:
        train_graph = mask_qos_in_graph(graph)
        train_sm = mask_qos_in_structural_metrics(structural_metrics)
    else:
        train_graph = graph
        train_sm = structural_metrics

    # Determinism for the per-seed train/val/test split is ensured by passing
    # `seeds=[seed]` into GNNService.train; that path internally calls
    # create_node_splits(seed=seed) so masks are paired across arms.
    random.seed(seed)
    np.random.seed(seed)

    svc = GNNService(
        hidden_channels=hidden,
        num_heads=heads,
        num_layers=num_layers,
        dropout=dropout,
        predict_edges=False,        # node criticality only for the headline
        checkpoint_dir=str(ckpt_dir),
    )

    train_kwargs: Dict[str, Any] = dict(
        graph=train_graph,
        structural_metrics=train_sm,
        simulation_results=simulation_results,
        rmav_scores=rmav_scores,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        num_epochs=epochs,
        lr=lr,
        patience=patience,
        seeds=[seed],
        mode="gnn",
        layer=layer,
    )
    # If Change 1 has merged, also forward the native flag for cleaner
    # masking inside HeteroData.
    if _NATIVE_QOS_FLAG_AVAILABLE:
        train_kwargs["qos_enabled"] = qos_enabled  # type: ignore[arg-type]

    t0 = time.time()
    result = svc.train(**train_kwargs)
    elapsed = time.time() - t0

    m = result.gnn_metrics
    if m is None:
        # Fall back to NaNs rather than crashing the entire matrix
        # if one cell fails to produce metrics.
        logger.warning("[%s | %s | seed=%d] no gnn_metrics returned; "
                       "recording NaNs.", scenario, arm_label, seed)
        return ArmResult(
            scenario=scenario, qos_enabled=qos_enabled, seed=seed,
            spearman_rho=float("nan"), f1_score=float("nan"),
            rmse=float("nan"), mae=float("nan"), ndcg_10=float("nan"),
            top_5_overlap=float("nan"), top_10_overlap=float("nan"),
            train_seconds=elapsed,
        )

    return ArmResult(
        scenario=scenario,
        qos_enabled=qos_enabled,
        seed=seed,
        spearman_rho=float(getattr(m, "spearman_rho", float("nan"))),
        f1_score=float(getattr(m, "f1_score", float("nan"))),
        rmse=float(getattr(m, "rmse", float("nan"))),
        mae=float(getattr(m, "mae", float("nan"))),
        ndcg_10=float(getattr(m, "ndcg_10", float("nan"))),
        top_5_overlap=float(getattr(m, "top_5_overlap", float("nan"))),
        top_10_overlap=float(getattr(m, "top_10_overlap", float("nan"))),
        train_seconds=elapsed,
    )


# ── Per-scenario orchestrator ────────────────────────────────────────────────

def run_scenario_ablation(
    *,
    paths: ScenarioPaths,
    seeds: List[int],
    layer: str,
    epochs: int,
    patience: int,
    hidden: int,
    heads: int,
    num_layers: int,
    dropout: float,
    lr: float,
    train_ratio: float,
    val_ratio: float,
    checkpoint_root: Path,
    cache_dir: Path,
    skip_existing: bool,
) -> Optional[AblationReport]:
    """Run both arms × all seeds for one scenario; return AblationReport."""

    if not paths.is_complete():
        logger.warning("[%s] missing prerequisites: %s — skipping.",
                       paths.name, paths.missing_files())
        return None

    cache_path = cache_dir / f"{paths.name}.report.json"
    if skip_existing and cache_path.exists():
        logger.info("[%s] cached report present — loading.", paths.name)
        try:
            cached = load_json(cache_path)
            return AblationReport(
                **{k: v for k, v in cached.items()
                   if k not in {"base_arm_results", "enr_arm_results"}},
                base_arm_results=[ArmResult(**r) for r in cached["base_arm_results"]],
                enr_arm_results=[ArmResult(**r) for r in cached["enr_arm_results"]],
            )
        except Exception as e:
            logger.warning("[%s] cached report unreadable (%s); rerunning.",
                           paths.name, e)

    # ── Load inputs ──────────────────────────────────────────────────────
    logger.info("[%s] loading topology + simulation artifacts…", paths.name)
    graph = graph_from_topology_json(paths.graph_json)
    structural_metrics = load_json(paths.structural_json)
    simulation_results = load_json(paths.simulation_json)
    rmav_scores = (
        load_json(paths.rmav_json)
        if paths.rmav_json is not None and paths.rmav_json.exists()
        else None
    )

    qos_gini = compute_qos_gini(graph)
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    n_topics = sum(1 for _, d in graph.nodes(data=True) if d.get("type") == "Topic")
    logger.info("[%s] |V|=%d |E|=%d topics=%d QoS-Gini=%.4f",
                paths.name, n_nodes, n_edges, n_topics, qos_gini)

    # ── Per-seed paired runs ─────────────────────────────────────────────
    base_arm_results: List[ArmResult] = []
    enr_arm_results: List[ArmResult] = []
    for seed in seeds:
        for qos_enabled in (False, True):
            label = "QoS-ON" if qos_enabled else "QoS-OFF"
            logger.info("[%s] %s seed=%d — training …", paths.name, label, seed)
            try:
                arm = run_single_arm(
                    scenario=paths.name,
                    qos_enabled=qos_enabled,
                    seed=seed,
                    graph=graph,
                    structural_metrics=structural_metrics,
                    simulation_results=simulation_results,
                    rmav_scores=rmav_scores,
                    layer=layer,
                    epochs=epochs,
                    patience=patience,
                    hidden=hidden,
                    heads=heads,
                    num_layers=num_layers,
                    dropout=dropout,
                    lr=lr,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    checkpoint_root=checkpoint_root,
                )
            except Exception as e:
                logger.error("[%s] %s seed=%d FAILED: %s", paths.name, label, seed, e)
                arm = ArmResult(
                    scenario=paths.name, qos_enabled=qos_enabled, seed=seed,
                    spearman_rho=float("nan"), f1_score=float("nan"),
                    rmse=float("nan"), mae=float("nan"), ndcg_10=float("nan"),
                    top_5_overlap=float("nan"), top_10_overlap=float("nan"),
                    train_seconds=0.0,
                )
            (enr_arm_results if qos_enabled else base_arm_results).append(arm)
            logger.info("[%s] %s seed=%d ρ=%.4f F1=%.4f NDCG=%.4f (%.1fs)",
                        paths.name, label, seed,
                        arm.spearman_rho, arm.f1_score, arm.ndcg_10,
                        arm.train_seconds)

    # ── Pair-level statistics ────────────────────────────────────────────
    base_rhos = [a.spearman_rho for a in base_arm_results]
    enr_rhos = [a.spearman_rho for a in enr_arm_results]
    base_clean = [x for x in base_rhos if not math.isnan(x)]
    enr_clean = [x for x in enr_rhos if not math.isnan(x)]

    if len(base_clean) >= 2 and len(enr_clean) >= 2 and len(base_clean) == len(enr_clean):
        # Paired tests: alternative='greater' tests H1: enr > base
        try:
            ttest = stats.ttest_rel(enr_clean, base_clean, alternative="greater")
            paired_t_pvalue = float(ttest.pvalue)
        except Exception:
            paired_t_pvalue = float("nan")
        try:
            wil = stats.wilcoxon(enr_clean, base_clean, alternative="greater",
                                 zero_method="zsplit")
            wilcoxon_pvalue = float(wil.pvalue)
        except Exception:
            wilcoxon_pvalue = float("nan")
        ci_low, ci_high = bootstrap_ci_paired_diff(base_clean, enr_clean, seed=42)
    else:
        paired_t_pvalue = float("nan")
        wilcoxon_pvalue = float("nan")
        ci_low = ci_high = float("nan")

    base_mean = float(np.mean(base_clean)) if base_clean else float("nan")
    base_std = float(np.std(base_clean, ddof=1)) if len(base_clean) > 1 else 0.0
    enr_mean = float(np.mean(enr_clean)) if enr_clean else float("nan")
    enr_std = float(np.std(enr_clean, ddof=1)) if len(enr_clean) > 1 else 0.0
    delta_rho = enr_mean - base_mean if not (math.isnan(base_mean) or math.isnan(enr_mean)) else float("nan")

    base_f1_mean = float(np.nanmean([a.f1_score for a in base_arm_results]))
    enr_f1_mean = float(np.nanmean([a.f1_score for a in enr_arm_results]))
    base_ndcg_mean = float(np.nanmean([a.ndcg_10 for a in base_arm_results]))
    enr_ndcg_mean = float(np.nanmean([a.ndcg_10 for a in enr_arm_results]))

    report = AblationReport(
        scenario=paths.name,
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_topics=n_topics,
        qos_gini=qos_gini,
        seeds=list(seeds),
        base_rhos=base_rhos,
        enr_rhos=enr_rhos,
        base_rho_mean=base_mean,
        base_rho_std=base_std,
        enr_rho_mean=enr_mean,
        enr_rho_std=enr_std,
        delta_rho=delta_rho,
        delta_rho_ci_low=ci_low,
        delta_rho_ci_high=ci_high,
        paired_t_pvalue=paired_t_pvalue,
        wilcoxon_pvalue=wilcoxon_pvalue,
        base_f1_mean=base_f1_mean,
        enr_f1_mean=enr_f1_mean,
        base_ndcg_mean=base_ndcg_mean,
        enr_ndcg_mean=enr_ndcg_mean,
        base_arm_results=base_arm_results,
        enr_arm_results=enr_arm_results,
    )

    # Cache to disk for resumability.
    save_json(asdict(report), cache_path)
    logger.info("[%s] Δρ=%+.4f  CI=[%+.4f,%+.4f]  t-p=%.4g  W-p=%.4g",
                paths.name, delta_rho, ci_low, ci_high,
                paired_t_pvalue, wilcoxon_pvalue)
    return report


# ── Aggregate analysis ───────────────────────────────────────────────────────

def aggregate_analysis(reports: List[AblationReport]) -> Dict[str, Any]:
    """§6.5 mechanism analysis: regress Δρ on QoS Gini across scenarios."""
    if not reports:
        return {}
    gini = np.array([r.qos_gini for r in reports], dtype=float)
    delta = np.array([r.delta_rho for r in reports], dtype=float)
    mask = np.isfinite(gini) & np.isfinite(delta)
    out: Dict[str, Any] = {
        "n_scenarios": int(mask.sum()),
        "mean_delta_rho": float(np.nanmean(delta)),
        "median_delta_rho": float(np.nanmedian(delta)),
        "delta_rho_positive_count": int(np.sum(delta[mask] > 0)),
    }
    if mask.sum() >= 3:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            gini[mask], delta[mask]
        )
        out.update({
            "gini_delta_slope": float(slope),
            "gini_delta_intercept": float(intercept),
            "gini_delta_r_squared": float(r_value ** 2),
            "gini_delta_p_value": float(p_value),
            "gini_delta_std_err": float(std_err),
        })
    return out


# ── LaTeX table writers ──────────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    if not math.isfinite(p):
        return "---"
    if p < 0.001:
        return r"\textless 0.001"
    return f"{p:.3f}"


def _fmt_signed(x: float, digits: int = 3) -> str:
    if not math.isfinite(x):
        return "---"
    return f"{x:+.{digits}f}"


def _fmt_unsigned(x: float, digits: int = 3) -> str:
    if not math.isfinite(x):
        return "---"
    return f"{x:.{digits}f}"


def write_table_dataset_summary(reports: List[AblationReport], out_path: Path) -> None:
    """Table 4: per-scenario dataset summary (n, edges, topics, Gini)."""
    lines = [
        r"% Auto-generated by tools/qos_gnn_ablation_experiment.py",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Datasets used in the QoS ablation experiment.}",
        r"  \label{tab:datasets}",
        r"  \small",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    Scenario & $|V|$ & $|E|$ & Topics & QoS-Gini \\",
        r"    \midrule",
    ]
    for r in reports:
        esc = r.scenario.replace('_', r'\\_')
        lines.append(
            f"    {esc} & "
            f"{r.n_nodes} & {r.n_edges} & {r.n_topics} & "
            f"{r.qos_gini:.3f} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines))
    logger.info("Wrote LaTeX table: %s", out_path)


def write_table_headline(
    reports: List[AblationReport],
    aggregate: Dict[str, Any],
    out_path: Path,
) -> None:
    """Table 5: headline Δρ per scenario with paired test p-values."""
    lines = [
        r"% Auto-generated by tools/qos_gnn_ablation_experiment.py",
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Spearman~$\rho$ on the held-out test set, "
        r"averaged over five seeds. $\Delta\rho = \rho_{\text{QoS}} - "
        r"\rho_{\text{topo}}$. \emph{p}-values are one-sided paired tests "
        r"under $H_1{:}\;\rho_{\text{QoS}} > \rho_{\text{topo}}$.}",
        r"  \label{tab:headline}",
        r"  \small",
        r"  \begin{tabular}{lcccccccc}",
        r"    \toprule",
        r"    Scenario & QoS-Gini & "
        r"$\rho_{\text{topo}}$ & $\rho_{\text{QoS}}$ & "
        r"$\Delta\rho$ & 95\% CI & "
        r"$p_{t}$ & $p_{W}$ & NDCG@10 \\",
        r"    \midrule",
    ]
    for r in reports:
        esc = r.scenario.replace('_', r'\\_')
        ci = f"[{_fmt_signed(r.delta_rho_ci_low)}, {_fmt_signed(r.delta_rho_ci_high)}]"
        lines.append(
            f"    {esc} & "
            f"{r.qos_gini:.3f} & "
            f"{_fmt_unsigned(r.base_rho_mean)} & "
            f"{_fmt_unsigned(r.enr_rho_mean)} & "
            f"{_fmt_signed(r.delta_rho)} & "
            f"{ci} & "
            f"{_fmt_p(r.paired_t_pvalue)} & "
            f"{_fmt_p(r.wilcoxon_pvalue)} & "
            f"{_fmt_unsigned(r.enr_ndcg_mean)} \\\\"
        )
    lines += [r"    \midrule"]

    if aggregate.get("n_scenarios", 0) >= 3:
        # Build the aggregate row as plain string concatenation; .format() with
        # nested LaTeX braces tends to confuse static checkers.
        n_sc = aggregate["n_scenarios"]
        mean_dr = aggregate.get("mean_delta_rho", float("nan"))
        slope = aggregate.get("gini_delta_slope", float("nan"))
        r2 = aggregate.get("gini_delta_r_squared", float("nan"))
        pval = _fmt_p(aggregate.get("gini_delta_p_value", float("nan")))
        agg_line = (
            r"    \multicolumn{9}{l}{\textit{Aggregate (across "
            + f"{n_sc} scenarios):"
            + r"}} $\overline{\Delta\rho}="
            + f"{mean_dr:+.3f}"
            + r"$, slope$_{\text{Gini}\to\Delta\rho}="
            + f"{slope:+.3f}"
            + r"$, $R^2="
            + f"{r2:.3f}"
            + r"$, $p="
            + f"{pval}"
            + r"$} \\"
        )
        lines.append(agg_line)
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table*}",
        "",
    ]
    out_path.write_text("\n".join(lines))
    logger.info("Wrote LaTeX table: %s", out_path)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QoS ablation matrix for the Middleware 2026 paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--scenarios",
        type=str,
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated scenario names (default: 8 synthetic + ATM).",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated integer seeds.",
    )
    p.add_argument(
        "--layer",
        type=str,
        default="system",
        choices=["app", "infra", "mw", "system"],
        help="Layer over which to train (default: system).",
    )
    p.add_argument("--epochs", type=int, default=300, help="Max epochs per training run.")
    p.add_argument("--patience", type=int, default=30, help="Early-stopping patience.")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--val-ratio", type=float, default=0.2)

    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root containing data/generated/<scenario>.json (default: data/).",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Root containing results/<scenario>/{structural,impact,rmav}.json.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/qos_ablation"),
        help="Where to write checkpoints, JSON, and LaTeX tables.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip scenarios with a cached report under <output-dir>/cache/.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_dir / "checkpoints"
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Configuration:")
    logger.info("  scenarios   = %s", scenarios)
    logger.info("  seeds       = %s", seeds)
    logger.info("  layer       = %s", args.layer)
    logger.info("  epochs      = %d  patience = %d", args.epochs, args.patience)
    logger.info("  arch        = hidden=%d heads=%d layers=%d dropout=%.2f",
                args.hidden, args.heads, args.num_layers, args.dropout)
    logger.info("  output_dir  = %s", output_dir)

    reports: List[AblationReport] = []
    for name in scenarios:
        paths = discover_scenario_paths(name, args.data_dir, args.results_dir)
        report = run_scenario_ablation(
            paths=paths,
            seeds=seeds,
            layer=args.layer,
            epochs=args.epochs,
            patience=args.patience,
            hidden=args.hidden,
            heads=args.heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            checkpoint_root=checkpoint_root,
            cache_dir=cache_dir,
            skip_existing=args.skip_existing,
        )
        if report is not None:
            reports.append(report)

    if not reports:
        logger.error("No scenario reports were produced. Verify the "
                     "--data-dir and --results-dir paths.")
        return 1

    aggregate = aggregate_analysis(reports)

    master = {
        "config": {
            "scenarios": scenarios,
            "seeds": seeds,
            "layer": args.layer,
            "epochs": args.epochs,
            "hidden": args.hidden,
            "heads": args.heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "native_qos_flag_used": _NATIVE_QOS_FLAG_AVAILABLE,
        },
        "aggregate": aggregate,
        "reports": [asdict(r) for r in reports],
    }
    master_path = output_dir / "qos_gnn_ablation.json"
    save_json(master, master_path)
    logger.info("Wrote master JSON: %s", master_path)

    write_table_dataset_summary(reports, table_dir / "table_datasets.tex")
    write_table_headline(reports, aggregate, table_dir / "table_headline_delta_rho.tex")

    # ── Console summary ──────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(" QoS GNN ablation — final summary")
    print("=" * 78)
    print(f" {'Scenario':<22} {'Gini':>6} {'rho_topo':>9} {'rho_QoS':>9} "
          f"{'Δρ':>9} {'p_t':>9} {'p_W':>9}")
    print("-" * 78)
    for r in reports:
        print(f" {r.scenario:<22} {r.qos_gini:>6.3f} "
              f"{r.base_rho_mean:>9.4f} {r.enr_rho_mean:>9.4f} "
              f"{r.delta_rho:>+9.4f} {r.paired_t_pvalue:>9.4g} "
              f"{r.wilcoxon_pvalue:>9.4g}")
    print("-" * 78)
    if aggregate.get("n_scenarios", 0) >= 3:
        print(f" Aggregate: mean Δρ = {aggregate['mean_delta_rho']:+.4f}; "
              f"Δρ ~ Gini  slope = {aggregate.get('gini_delta_slope', float('nan')):+.4f}, "
              f"R² = {aggregate.get('gini_delta_r_squared', float('nan')):.3f}, "
              f"p = {aggregate.get('gini_delta_p_value', float('nan')):.4g}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    sys.exit(main())
