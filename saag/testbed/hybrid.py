"""
Two-tiered hybrid runtime simulation protocol.

Combines fast discrete-event screening in `MessageFlowSimulator` (Tier 1) with
targeted verification and gray-failure probing on the emulated/container DDS
testbed (Tier 2):

  Step 0  Steady-state baseline & calibration (Tier 2 -> Tier 1):
          Build the testbed once, run a fault-free baseline, fit
          `RuntimeTelemetryProfile` via `TelemetryCalibrator`, and record
          steady-state fidelity (`compare_steady_state`).
  Step 1  Full-system screening (Tier 1):
          Run calibrated `runtime_labels` under the runtime/DDS engine
          (`stations`, `dds`, `paired`, `per_process`, `replica_aware`), with
          multi-seed stability, optional parameter-uncertainty draws, and
          order-k references (`ref_flow_policy`, `ref_delivery_first_order`).
  Step 2  Shortlist & triage selection:
          Combine Top-K risk targets (from predictions or Tier 1), highest
          prediction-vs-Tier-1 rank disagreements, and low-impact negative
          controls (preferring sinks).
  Step 3  Targeted verification & gray-failure probing (Tier 2):
          Re-run shortlisted targets on the built testbed across crash and
          opt-in gray-failure probes, comparing simulated vs. observed impact,
          stall-set Jaccard, and DDS statuses, with explicit circularity,
          scaling, and fault-parity caveats.

This report is a Tier-2 advisory triage artifact: it never decides a blocking
CI pass/fail gate and never substitutes for `FailureSimulator` release gates.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from saag.core.graph_io import load_graph
from saag.simulation.runtime.faults import FaultSpec, RuntimeFaultMode
from saag.simulation.runtime.labels import runtime_labels
from saag.validation.fidelity import (
    compare_dds_status,
    compare_fault_response,
    compare_steady_state,
    diagnose_fault_residuals,
    observed_impact,
    stalled_in_trace,
)

from . import orchestrate, study
from .chaos import plan_actions
from .collect import load_run
from .deployment import build_deployment


#: Valid Tier 1 labels for shortlist ranking.
VALID_TIER1_LABELS: Tuple[str, ...] = ("i_flow", "i_delivery", "i_delivery_consumer")


@dataclass(frozen=True)
class HybridProtocolConfig:
    """Configuration for the four-step two-tiered hybrid runtime protocol."""

    # Timing & capacity
    duration_s: float = 60.0
    warmup_s: float = 5.0
    tier1_duration_s: Optional[float] = None
    target_utilization: float = 0.65
    max_host_load: Optional[float] = None

    # Tier 1 screening (MessageFlowSimulator runtime/DDS engine)
    tier1_seeds: Tuple[int, ...] = (42, 123, 456)
    tier1_label: str = "i_flow"
    behavior_model: str = "reactive"          # "reactive" | "open_loop"
    qos_semantics: str = "dds"                # "dds" | "engine"
    parameter_draws: int = 0
    workers: int = 1
    skip_calibration: bool = False

    # Step 2 shortlist & triage selection
    top_k: Optional[int] = None
    top_k_frac: float = 0.15
    n_disagreements: int = 3
    n_controls: int = 3
    prediction_layer: str = "system"

    # Step 3 Tier 2 verification & gray-failure probing
    mode: str = "local"                       # "local" | "docker"
    service_work: str = "sleep"               # "sleep" | "cpu"
    carry_payload: bool = False
    network_gating: bool = False
    repetitions: int = 3
    base_seed: int = 42
    fault_probes: Tuple[Dict[str, Any], ...] = ({"mode": "crash"},)

    def __post_init__(self) -> None:
        if self.tier1_label not in VALID_TIER1_LABELS:
            raise ValueError(
                f"tier1_label must be one of {VALID_TIER1_LABELS}, got {self.tier1_label!r}"
            )
        if self.behavior_model not in ("reactive", "open_loop"):
            raise ValueError(
                f"behavior_model must be 'reactive' or 'open_loop', got {self.behavior_model!r}"
            )
        if self.qos_semantics not in ("dds", "engine"):
            raise ValueError(
                f"qos_semantics must be 'dds' or 'engine', got {self.qos_semantics!r}"
            )
        if self.mode not in ("local", "docker"):
            raise ValueError(f"mode must be 'local' or 'docker', got {self.mode!r}")
        if self.repetitions < 1:
            raise ValueError(f"repetitions must be >= 1, got {self.repetitions}")
        if self.duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {self.duration_s}")


# ── Parsing & loading helpers ────────────────────────────────────────────────

def parse_fault_probe(spec: str) -> Dict[str, Any]:
    """Parse a CLI ``--fault-probe`` string into a probe dict.

    Syntax: ``<mode>`` or ``<mode>:<k1>=<v1>,<k2>=<v2>``.
    ``duration`` / ``duration_s`` sets the fault duration; all other pairs go
    into ``params`` (coercing numeric strings to int/float).
    """
    valid_modes = {m.value for m in RuntimeFaultMode}
    raw = spec.strip()
    if not raw:
        raise ValueError("fault probe specification cannot be empty")
    if ":" in raw:
        mode, rest = raw.split(":", 1)
    else:
        mode, rest = raw, ""
    mode = mode.strip().lower()
    if mode not in valid_modes:
        raise ValueError(
            f"unknown fault probe mode {mode!r}; expected one of {sorted(valid_modes)}"
        )

    duration_s: Optional[float] = None
    params: Dict[str, Any] = {}
    if rest.strip():
        for item in rest.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"fault probe parameter must be KEY=VALUE, got {item!r}")
            k, v = item.split("=", 1)
            k, v = k.strip(), v.strip()
            val: Any
            try:
                fval = float(v)
                val = int(fval) if fval.is_integer() and "." not in v else fval
            except ValueError:
                val = v
            if k in ("duration", "duration_s"):
                duration_s = float(val)
            else:
                params[k] = val

    out: Dict[str, Any] = {"mode": mode}
    if duration_s is not None:
        out["duration_s"] = duration_s
    if params:
        out["params"] = params
    return out


def _probe_id(probe: Mapping[str, Any], index: int) -> str:
    mode = str(probe.get("mode", "crash"))
    parts = [mode]
    if probe.get("duration_s") is not None:
        parts.append(f"d{probe['duration_s']}")
    for k, v in sorted((probe.get("params") or {}).items()):
        parts.append(f"{k}{v}")
    base = "_".join(parts)
    return base if index == 0 or len(parts) > 1 else f"{base}_{index}"


def load_prediction_scores(
    source: Union[Path, str, Mapping[str, Any]],
    layer: str = "system",
    score_key: Optional[str] = None,
) -> Dict[str, float]:
    """Extract a ``{component_id: float}`` score map from common SaG artifacts.

    Supports:
    - Flat ``{component_id: number}`` dicts
    - ``saag-predict`` JSON (``layers[layer].gnn.node_scores`` or ``layers[layer].rm``)
    - ``impact_scores.json`` (``records[id].impact_score``)
    - ``runtime_labels.json`` (``records[id][score_key | 'i_flow'].mean``)
    """
    if isinstance(source, (str, Path)):
        raw = json.loads(Path(source).read_text())
    else:
        raw = dict(source)

    if not isinstance(raw, Mapping):
        raise ValueError("predictions payload must be a JSON object")

    # 1. saag-predict multi-layer output
    if isinstance(raw.get("layers"), Mapping):
        layers = raw["layers"]
        entry = layers.get(layer) or next(iter(layers.values()), {})
        gnn = entry.get("gnn")
        if isinstance(gnn, Mapping) and isinstance(gnn.get("node_scores"), Mapping):
            key = score_key or "composite_score"
            return {
                cid: float(rec[key])
                for cid, rec in gnn["node_scores"].items()
                if isinstance(rec, Mapping) and isinstance(rec.get(key), (int, float))
            }
        rm = entry.get("rm")
        if isinstance(rm, Mapping):
            key = score_key or "overall"
            return {
                cid: float(rec[key])
                for cid, rec in rm.items()
                if isinstance(rec, Mapping) and isinstance(rec.get(key), (int, float))
            }
        return {}

    # 2. impact_scores.json / runtime_labels.json
    if isinstance(raw.get("records"), Mapping):
        out: Dict[str, float] = {}
        for cid, rec in raw["records"].items():
            if not isinstance(rec, Mapping):
                continue
            if score_key and score_key in rec:
                val = rec[score_key]
                if isinstance(val, Mapping) and isinstance(val.get("mean"), (int, float)):
                    out[cid] = float(val["mean"])
                elif isinstance(val, (int, float)):
                    out[cid] = float(val)
            elif isinstance(rec.get("impact_score"), (int, float)):
                out[cid] = float(rec["impact_score"])
            else:
                for fallback in ("i_flow", "i_delivery", "i_delivery_consumer"):
                    sub = rec.get(fallback)
                    if isinstance(sub, Mapping) and isinstance(sub.get("mean"), (int, float)):
                        out[cid] = float(sub["mean"])
                        break
        return out

    # 3. Flat {component_id: score}
    return {
        str(k): float(v)
        for k, v in raw.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


# ── Step 2: Shortlist & triage selection ─────────────────────────────────────

def _spearman_pair(a: Mapping[str, float], b: Mapping[str, float]) -> Optional[float]:
    from scipy.stats import spearmanr

    common = sorted(set(a) & set(b))
    if len(common) < 3:
        return None
    xs = [a[k] for k in common]
    ys = [b[k] for k in common]
    rho = spearmanr(xs, ys).correlation
    return None if math.isnan(rho) else round(float(rho), 6)


def _rank_positions(scores: Mapping[str, float], keys: Sequence[str]) -> Dict[str, float]:
    """Normalised descending rank in [0, 1] (0 = highest score, 1 = lowest)."""
    ordered = sorted(keys, key=lambda k: (-scores[k], k))
    denom = max(len(ordered) - 1, 1)
    return {k: i / denom for i, k in enumerate(ordered)}


def select_shortlist(
    tier1_records: Mapping[str, Mapping[str, Any]],
    predictions: Optional[Mapping[str, float]] = None,
    *,
    label: str = "i_flow",
    top_k: Optional[int] = None,
    top_k_frac: float = 0.15,
    n_disagreements: int = 3,
    n_controls: int = 3,
    sinks: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Select targets for Tier 2 verification from Tier 1 and optional predictions.

    Composition:
    1. **Top-K** risk targets ranked by ``predictions`` (if supplied and
       overlapping) or by Tier 1 ``label`` mean.
    2. **Disagreements** (up to ``n_disagreements``) with the largest rank gap
       between ``predictions`` and Tier 1 ``label``.
    3. **Negative controls** (up to ``n_controls``) with the lowest Tier 1
       impact, preferring declared sinks so false-positive impact is detectable
       without inflating primary rank correlations.
    """
    tier1_scores: Dict[str, float] = {}
    ref_flow: Dict[str, float] = {}
    ref_deliv: Dict[str, float] = {}
    for cid, rec in tier1_records.items():
        val = rec.get(label)
        if isinstance(val, Mapping) and val.get("mean") is not None:
            tier1_scores[cid] = float(val["mean"])
        if rec.get("ref_flow_policy") is not None:
            ref_flow[cid] = float(rec["ref_flow_policy"])
        if rec.get("ref_delivery_first_order") is not None:
            ref_deliv[cid] = float(rec["ref_delivery_first_order"])

    candidates = sorted(tier1_scores)
    if not candidates:
        return {
            "label": label,
            "k": 0,
            "targets": [],
            "primary_targets": [],
            "control_targets": [],
            "tier1_diagnostics": {},
        }

    pred_map = {k: float(v) for k, v in (predictions or {}).items() if k in tier1_scores}
    has_preds = bool(pred_map)
    k = top_k if top_k is not None else max(1, int(round(len(candidates) * top_k_frac)))
    k = max(1, min(k, len(candidates)))

    if has_preds:
        ranked_primary = sorted(
            pred_map, key=lambda c: (-pred_map[c], -tier1_scores[c], c)
        )
        top_reason = "top_k_predicted"
    else:
        ranked_primary = sorted(candidates, key=lambda c: (-tier1_scores[c], c))
        top_reason = "top_k_tier1"

    selected_order: List[str] = []
    reasons_by_id: Dict[str, List[str]] = {}
    divergence_by_id: Dict[str, float] = {}

    def _add(cid: str, reason: str) -> None:
        if cid not in reasons_by_id:
            selected_order.append(cid)
            reasons_by_id[cid] = []
        if reason not in reasons_by_id[cid]:
            reasons_by_id[cid].append(reason)

    for cid in ranked_primary[:k]:
        _add(cid, top_reason)

    # Disagreement targets: largest normalised rank gap between predictions and Tier 1
    if has_preds and len(pred_map) >= 3 and n_disagreements > 0:
        common = sorted(pred_map)
        r_pred = _rank_positions(pred_map, common)
        r_t1 = _rank_positions(tier1_scores, common)
        gaps = {c: abs(r_pred[c] - r_t1[c]) for c in common}
        divergence_by_id.update({c: round(g, 4) for c, g in gaps.items()})
        remaining = [c for c in common if c not in reasons_by_id]
        for cid in sorted(remaining, key=lambda c: (-gaps[c], c))[:n_disagreements]:
            _add(cid, "disagreement")

    primary_targets = list(selected_order)

    # Negative controls: lowest-impact remaining candidates, preferring sinks
    sink_set = set(sinks or ())
    remaining_all = [c for c in candidates if c not in reasons_by_id]
    control_pool = sorted(
        remaining_all,
        key=lambda c: (0 if c in sink_set else 1, abs(tier1_scores[c]), c),
    )
    control_targets: List[str] = []
    for cid in control_pool[: max(0, n_controls)]:
        _add(cid, "negative_control")
        control_targets.append(cid)

    targets = [
        {
            "node_id": cid,
            "node_type": tier1_records[cid].get("node_type"),
            "reasons": reasons_by_id[cid],
            "tier1_score": round(tier1_scores[cid], 6),
            "predicted_score": round(pred_map[cid], 6) if cid in pred_map else None,
            "rank_divergence": divergence_by_id.get(cid),
            "is_control": cid in control_targets,
        }
        for cid in selected_order
    ]

    diagnostics: Dict[str, Any] = {
        "n_tier1_candidates": len(candidates),
        "spearman_vs_ref_flow_policy": _spearman_pair(tier1_scores, ref_flow),
        "spearman_vs_ref_delivery_first_order": _spearman_pair(tier1_scores, ref_deliv),
    }
    if has_preds:
        diagnostics["n_predictions_matched"] = len(pred_map)
        diagnostics["spearman_predictions_vs_tier1"] = _spearman_pair(pred_map, tier1_scores)
        common = sorted(pred_map)
        k_j = max(1, min(k, len(common)))
        top_pred = set(sorted(common, key=lambda c: (-pred_map[c], c))[:k_j])
        top_t1 = set(sorted(common, key=lambda c: (-tier1_scores[c], c))[:k_j])
        diagnostics["topk_jaccard_predictions_vs_tier1"] = round(
            len(top_pred & top_t1) / len(top_pred | top_t1), 6
        )

    return {
        "label": label,
        "k": k,
        "targets": targets,
        "primary_targets": primary_targets,
        "control_targets": control_targets,
        "tier1_diagnostics": diagnostics,
    }


# ── Pre-flight parity & caveat assessment ────────────────────────────────────

def assess_probe_parity(
    fault_probes: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    mode: str,
    *,
    kappa: float = 1.0,
    behavior_model: str = "reactive",
    shortlist_targets: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Inspect fault probes against testbed capabilities and collect caveats."""
    caveats: List[Dict[str, str]] = []
    supported: List[Dict[str, Any]] = []
    unsupported: List[Dict[str, Any]] = []

    if mode == "local":
        caveats.append({
            "code": "local_mode_no_dds_transport",
            "severity": "info",
            "message": (
                "Tier 2 ran in local in-process mode (LocalTransport): useful for logic and "
                "smoke checks, but does not exercise Cyclone DDS discovery, liveliness, or containers."
            ),
        })

    if behavior_model == "reactive":
        caveats.append({
            "code": "circular_reactive_policies",
            "severity": "warning",
            "message": (
                "Reactive trigger policies (`policy_for`) are shared between MessageFlowSimulator "
                "and the testbed stubs; reactive cascade agreement validates timing/transport "
                "fidelity rather than independent application behaviour. Use --open-loop for "
                "confirmatory non-circular transport checks."
            ),
        })

    if kappa < 1.0 - 1e-9:
        caveats.append({
            "code": "rate_scaled_latency_caveat",
            "severity": "warning",
            "message": (
                f"Scenario rates were scaled by kappa={kappa:.4f} to fit single-host capacity; "
                "utilisation is preserved, but transport-to-service latency ratios are distorted."
            ),
        })

    rep_target = (
        shortlist_targets[0]
        if shortlist_targets
        else next(iter(plan.get("apps", ())), next(iter(plan.get("hosts", ())), "A0"))
    )
    links = plan.get("links") or []
    rep_link = f"{links[0][0]}|{links[0][1]}" if links else "H0|H1"
    stationless_targets = [
        t for t in (shortlist_targets or ())
        if t in plan.get("apps", {}) and not plan["apps"][t].get("subscribes")
    ]

    for idx, probe in enumerate(fault_probes):
        pid = _probe_id(probe, idx)
        p_mode = str(probe.get("mode", "crash"))
        sample_target = rep_link if p_mode == "link_flap" else rep_target
        spec = FaultSpec(
            sample_target,
            p_mode,
            1.0,
            probe.get("duration_s"),
            dict(probe.get("params") or {}),
        )
        cp = plan_actions([spec], dict(plan), duration_s=10.0, seed=0, local=(mode == "local"))
        if cp.unsupported or (p_mode == "link_flap" and not links):
            reason = (
                cp.unsupported[0].get("reason", "unsupported")
                if cp.unsupported
                else "scenario declares no CONNECTS_TO links for link_flap"
            )
            unsupported.append({"probe_id": pid, "probe": dict(probe), "reason": reason})
        else:
            supported.append({"probe_id": pid, "probe": dict(probe)})
            if p_mode == "link_flap" and mode == "docker":
                caveats.append({
                    "code": "link_flap_rediscovery_lag",
                    "severity": "info",
                    "message": (
                        "Cyclone DDS incurs a multi-second participant-rediscovery lag after "
                        "partition healing (~6 s on longer cuts), whereas MessageFlowSimulator "
                        "reconnects immediately."
                    ),
                })
            if p_mode == "slowdown" and stationless_targets:
                caveats.append({
                    "code": "slowdown_stationless_divergence",
                    "severity": "warning",
                    "message": (
                        f"Targets {stationless_targets} have no subscribed inputs (station-less): "
                        "MessageFlowSimulator stretches emission period under slowdown, whereas "
                        "testbed agents do not yet throttle station-less apps identically."
                    ),
                })

    if unsupported:
        caveats.append({
            "code": "unsupported_probes",
            "severity": "warning",
            "message": (
                f"{len(unsupported)} probe(s) cannot be emulated in {mode!r} mode and were skipped: "
                + ", ".join(f"{u['probe_id']} ({u['reason']})" for u in unsupported)
            ),
        })

    return {"supported": supported, "unsupported": unsupported, "caveats": caveats}


# ── Main 4-step orchestrator ─────────────────────────────────────────────────

def _failed_components_for_target(plan: Mapping[str, Any], target: str, probe_mode: str) -> List[str]:
    if probe_mode != "crash":
        return [target] if target not in plan["hosts"] else study._failed_with(plan, target)
    return study._failed_with(plan, target)


def run_hybrid_protocol(
    scenario: Union[Path, str],
    out_dir: Union[Path, str],
    config: Optional[HybridProtocolConfig] = None,
    predictions: Optional[Union[Path, str, Mapping[str, float]]] = None,
) -> Dict[str, Any]:
    """Execute the 4-step two-tiered hybrid runtime simulation protocol."""
    cfg = config or HybridProtocolConfig()
    scenario_path = Path(scenario)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tb_dir = out_path / "testbed"

    # ── Step 0: Optional capacity scaling + build + fault-free baseline ──────
    base_plan = build_deployment(
        scenario_path,
        target_utilization=cfg.target_utilization,
        qos_semantics=cfg.qos_semantics,
        open_loop=(cfg.behavior_model == "open_loop"),
        service_work=cfg.service_work,
        carry_payload=cfg.carry_payload,
        network_gating=cfg.network_gating,
    )
    peak_load = max(study.host_loads(base_plan).values(), default=0.0)
    kappa = (
        study.kappa_for(base_plan, cfg.max_host_load)
        if cfg.max_host_load is not None
        else 1.0
    )
    if kappa < 1.0 - 1e-9:
        active_scenario = study.write_scaled(scenario_path, kappa, out_path / "scaled_scenario.json")
    else:
        active_scenario = scenario_path

    eff_duration = cfg.duration_s / kappa
    eff_warmup = min(cfg.warmup_s / kappa, eff_duration * 0.25)
    tier1_duration = (cfg.tier1_duration_s if cfg.tier1_duration_s is not None else cfg.duration_s) / kappa

    orchestrate.build(
        active_scenario,
        tb_dir,
        duration_s=eff_duration,
        seed=cfg.base_seed,
        target_utilization=cfg.target_utilization,
        qos_semantics=cfg.qos_semantics,
        open_loop=(cfg.behavior_model == "open_loop"),
        service_work=cfg.service_work,
        carry_payload=cfg.carry_payload,
        network_gating=cfg.network_gating,
    )
    plan = json.loads((tb_dir / "deployment.json").read_text())

    # Fault-free baseline run on Tier 2
    cpu_p95: Optional[Dict[str, Optional[float]]] = None
    if cfg.mode == "docker":
        with study.CpuSampler(period_s=1.0) as sampler:
            base_meta = orchestrate.run(tb_dir, eff_duration, faults=[], mode=cfg.mode, seed=cfg.base_seed)
        t0 = float(base_meta["start_unix"])
        cpu_p95 = sampler.p95_by_container(t0 + eff_warmup, t0 + eff_duration)
    else:
        base_meta = orchestrate.run(tb_dir, eff_duration, faults=[], mode=cfg.mode, seed=cfg.base_seed)

    base_observed = load_run(tb_dir)
    base_attainment = study.attainment(base_observed, plan, eff_warmup)

    profile_dict: Optional[Dict[str, Any]] = None
    profile_obj: Optional[Any] = None
    if not cfg.skip_calibration:
        profile_dict = orchestrate.calibrate(tb_dir, warmup_s=eff_warmup)
        profile_obj = orchestrate.profile_from_json(profile_dict)

    base_simulated = orchestrate.simulate_like(tb_dir, calibrated=(profile_obj is not None))
    steady_state_report = compare_steady_state(
        base_simulated.window(eff_warmup, eff_duration),
        base_observed.window(eff_warmup, eff_duration),
    )

    step0_summary: Dict[str, Any] = {
        "kappa": round(kappa, 6),
        "peak_host_load_msg_s": round(peak_load, 2),
        "effective_duration_s": round(eff_duration, 3),
        "effective_warmup_s": round(eff_warmup, 3),
        "calibrated": profile_obj is not None,
        "calibration_counts": (
            {
                "topic_rates": len(profile_dict["msg_rate_per_sec"]),
                "service_time_dists": len(profile_dict["service_time_dist"]),
                "trigger_policies": len(profile_dict["trigger_policy"]),
            }
            if profile_dict is not None
            else None
        ),
        "attainment": base_attainment,
        "steady_state": steady_state_report,
        "docker_cpu_p95": cpu_p95,
    }

    # ── Step 1: Full-system Tier 1 screening (MessageFlowSimulator) ──────────
    graph = load_graph(active_scenario)
    tier1_settings: Dict[str, Any] = {
        "duration": tier1_duration,
        "qos_mode": "full",
        "qos_semantics": cfg.qos_semantics,
        "dds_status": True,
        "target_utilization": cfg.target_utilization,
        "behavior_model": cfg.behavior_model,
        "deployment_model": "stations",
        "rng_streams": "per_process",
        "estimator": "paired",
        "replica_aware": True,
        "network_gating": cfg.network_gating,
    }
    tier1_out = runtime_labels(
        graph,
        seeds=cfg.tier1_seeds,
        settings=tier1_settings,
        workers=cfg.workers,
        parameter_draws=cfg.parameter_draws,
        telemetry_profile=profile_obj,
    )
    (out_path / "tier1_runtime_labels.json").write_text(
        json.dumps(tier1_out, indent=2, sort_keys=True)
    )

    # ── Step 2: Shortlist & triage selection ─────────────────────────────────
    pred_scores = (
        load_prediction_scores(predictions, layer=cfg.prediction_layer)
        if predictions is not None
        else None
    )
    sinks = sorted(
        a for a, spec in plan["apps"].items()
        if spec.get("subscribes") and not spec.get("publishes")
    )
    shortlist = select_shortlist(
        tier1_out["records"],
        predictions=pred_scores,
        label=cfg.tier1_label,
        top_k=cfg.top_k,
        top_k_frac=cfg.top_k_frac,
        n_disagreements=cfg.n_disagreements,
        n_controls=cfg.n_controls,
        sinks=sinks,
    )
    (out_path / "shortlist.json").write_text(json.dumps(shortlist, indent=2, sort_keys=True))

    # ── Step 3: Targeted Tier 2 verification & gray-failure probing ──────────
    shortlist_ids = [t["node_id"] for t in shortlist["targets"]]
    parity = assess_probe_parity(
        cfg.fault_probes,
        plan,
        cfg.mode,
        kappa=kappa,
        behavior_model=cfg.behavior_model,
        shortlist_targets=shortlist_ids,
    )
    subs_map = study.subscribers_of(plan)
    fault_time = eff_duration / 2.0
    probes_report: Dict[str, Any] = {}

    for item in parity["supported"]:
        pid = item["probe_id"]
        probe = item["probe"]
        p_mode = str(probe.get("mode", "crash"))
        p_dur = float(probe["duration_s"]) / kappa if probe.get("duration_s") is not None else None
        p_params = dict(probe.get("params") or {})

        if p_mode == "link_flap":
            link_targets = [f"{a}|{b}" for a, b in (plan.get("links") or [])]
            probe_targets = link_targets[: max(1, shortlist["k"])]
            control_set: set = set()
        else:
            probe_targets = shortlist_ids
            control_set = set(shortlist["control_targets"])

        per_target: Dict[str, Any] = {}
        sim_primary: Dict[str, float] = {}
        obs_primary: Dict[str, float] = {}
        controls_summary: Dict[str, Any] = {}

        for target in probe_targets:
            rep_obs_impacts: List[float] = []
            rep_sim_impacts: List[float] = []
            rep_stall_jaccards: List[float] = []
            rep_unsupported: List[Any] = []
            last_dds_cmp: Optional[Dict[str, Any]] = None
            obs_stalled_union: set = set()
            sim_stalled_union: set = set()

            failed_set = (
                set() if p_mode == "link_flap"
                else set(_failed_components_for_target(plan, target, p_mode))
            )

            for rep in range(cfg.repetitions):
                rep_seed = study.run_seed(cfg.base_seed, pid, target, str(rep))
                spec = FaultSpec(target, p_mode, fault_time, p_dur, p_params)
                meta = orchestrate.run(
                    tb_dir, eff_duration, faults=[spec], mode=cfg.mode, seed=rep_seed
                )
                if meta.get("unsupported"):
                    rep_unsupported.extend(meta["unsupported"])
                obs_tr = load_run(tb_dir)
                sim_tr = orchestrate.simulate_like(
                    tb_dir, calibrated=(profile_obj is not None)
                )

                obs_imp = observed_impact(
                    obs_tr, subs_map, target, fault_time,
                    warmup_s=eff_warmup, failed=failed_set,
                )
                sim_imp = observed_impact(
                    sim_tr, subs_map, target, fault_time,
                    warmup_s=eff_warmup, failed=failed_set,
                )
                rep_obs_impacts.append(obs_imp)
                rep_sim_impacts.append(sim_imp)

                obs_st = stalled_in_trace(obs_tr, fault_time) - failed_set - {target}
                sim_st = stalled_in_trace(sim_tr, fault_time) - failed_set - {target}
                obs_stalled_union |= obs_st
                sim_stalled_union |= sim_st
                st_cmp = compare_fault_response({}, {}, sim_st, obs_st)
                rep_stall_jaccards.append(float(st_cmp["stall_jaccard"]))

                last_dds_cmp = compare_dds_status(
                    sim_tr, obs_tr, split_at=fault_time,
                    dead_instances=orchestrate._dead_instances(plan, meta["faults"]),
                )

            obs_mean = statistics.fmean(rep_obs_impacts)
            sim_mean = statistics.fmean(rep_sim_impacts)
            entry = {
                "is_control": target in control_set,
                "repetitions": cfg.repetitions,
                "observed_impact_mean": round(obs_mean, 6),
                "observed_impact_std": round(
                    statistics.stdev(rep_obs_impacts) if len(rep_obs_impacts) > 1 else 0.0, 6
                ),
                "simulated_impact_mean": round(sim_mean, 6),
                "simulated_impact_std": round(
                    statistics.stdev(rep_sim_impacts) if len(rep_sim_impacts) > 1 else 0.0, 6
                ),
                "abs_error": round(abs(sim_mean - obs_mean), 6),
                "stall_jaccard_mean": round(statistics.fmean(rep_stall_jaccards), 6),
                "observed_stalled": sorted(obs_stalled_union),
                "simulated_stalled": sorted(sim_stalled_union),
                "dds_status": last_dds_cmp,
                "unsupported": rep_unsupported,
            }
            per_target[target] = entry
            if target in control_set:
                controls_summary[target] = {
                    "observed_impact_mean": entry["observed_impact_mean"],
                    "simulated_impact_mean": entry["simulated_impact_mean"],
                }
            else:
                sim_primary[target] = sim_mean
                obs_primary[target] = obs_mean

        primary_cmp = compare_fault_response(sim_primary, obs_primary)
        all_sim_stalled = {a for e in per_target.values() for a in e["simulated_stalled"]}
        all_obs_stalled = {a for e in per_target.values() for a in e["observed_stalled"]}
        residual_diag = diagnose_fault_residuals(
            sim_primary, obs_primary,
            simulated_stalled=all_sim_stalled,
            observed_stalled=all_obs_stalled,
            current_profile=profile_obj,
        )
        pred_vs_obs = (
            compare_fault_response(
                {t: pred_scores[t] for t in obs_primary if t in pred_scores},
                obs_primary,
            )
            if pred_scores
            else None
        )
        probes_report[pid] = {
            "probe": probe,
            "n_primary_targets": len(obs_primary),
            "n_control_targets": len(controls_summary),
            "tier1_vs_tier2": primary_cmp,
            "residual_diagnosis": residual_diag,
            "predictions_vs_tier2": pred_vs_obs,
            "controls": controls_summary,
            "per_target": per_target,
        }

    report: Dict[str, Any] = {
        "schema_version": "1.0",
        "report_kind": "tier2_advisory_hybrid_verification",
        "scenario": str(scenario_path),
        "config": asdict(cfg),
        "step0_calibration": step0_summary,
        "step1_tier1_screening": {
            "label": cfg.tier1_label,
            "n_candidates": len(tier1_out["records"]),
            "seeds": tier1_out["seeds"],
            "calibrated": bool(tier1_out.get("calibrated", False)),
            "label_stability": tier1_out["label_stability"],
            "parameter_stability": tier1_out.get("parameter_stability"),
            "artifact": str(out_path / "tier1_runtime_labels.json"),
        },
        "step2_shortlist": shortlist,
        "step3_tier2_verification": {
            "probes": probes_report,
            "unsupported_probes": parity["unsupported"],
        },
        "caveats": parity["caveats"],
    }
    (out_path / "hybrid_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str)
    )
    return report
