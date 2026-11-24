#!/usr/bin/env python3
"""
Perturbed-counterpart calibration (RQ2/H5) and R7 batch drift/EMA/residual-feedback
evaluation on the executable counterpart (local in-process or Docker + Cyclone DDS).

Addresses two structural gaps of Study 1:
  1. H5 was undecidable when the counterpart used the exact declared parameters
     (uncalibrated S2 already had 0.000 delivery error). Here the counterpart runs
     perturbed rates and service times while the twin starts from declared priors.
  2. R7 batch drift detection (detect_drift), incremental EMA recalibration
     (TelemetryCalibrator.update_profile), and behavioural residual feedback
     (diagnose_fault_residuals + apply_residual_feedback) were unit-tested only.

Usage:
    python3 reproduce/calibration_drift_eval.py --mode local --scenarios tiny_system
    python3 reproduce/calibration_drift_eval.py --mode docker --scenarios tiny_system,atm,realworld_autoware_ros2
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from saag.core.graph_io import load_graph  # noqa: E402
from saag.simulation.calibration import TelemetryCalibrator, TelemetryTrace  # noqa: E402
from saag.simulation.message_flow_simulator import MessageFlowSimulator  # noqa: E402
from saag.simulation.runtime.faults import FaultSpec  # noqa: E402
from saag.testbed import orchestrate, study  # noqa: E402
from saag.testbed.collect import load_run  # noqa: E402
from saag.testbed.compose import write_testbed  # noqa: E402
from saag.testbed.deployment import build_deployment  # noqa: E402
from saag.validation.fidelity import (  # noqa: E402
    apply_residual_feedback,
    compare_steady_state,
    detect_drift,
    diagnose_fault_residuals,
    observed_impact,
    stalled_in_trace,
)

logging.disable(logging.WARNING)

BASE_SEED = 20261123


def resolve_scenario_path(name: str) -> Path:
    """Resolve short scenario name (e.g. 'atm', 'tiny_system', 'realworld_autoware_ros2')."""
    cand = [
        ROOT / "data" / "scenarios" / f"{name}.json",
        ROOT / "data" / "scenarios" / f"{name}_system.json",
    ]
    for p in cand:
        if p.exists():
            return p
    matches = sorted((ROOT / "data" / "scenarios").glob(f"*{name}*.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Scenario not found: {name}")


def perturb_plan(plan: Mapping[str, Any], rate_frac: float = 0.4,
                 service_mult: float = 2.0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return a perturbed copy of *plan* and metadata describing the injected shifts.

    Deterministically shifts ~rate_frac of topics (+50% / -35% alternating) and scales
    service_mean_s by *service_mult* for all subscribing apps on the busiest host.
    """
    out = copy.deepcopy(dict(plan))
    topics = sorted(out["topics"].keys())
    n_shift = max(1, int(round(len(topics) * rate_frac)))
    shifted_topics: Dict[str, float] = {}
    for idx, t in enumerate(topics[:n_shift]):
        mult = 1.50 if idx % 2 == 0 else 0.65
        out["topics"][t]["rate_hz"] = float(out["topics"][t]["rate_hz"]) * mult
        shifted_topics[t] = mult

    # Update per_publisher_rate consistently for shifted topics
    for a, spec in out["apps"].items():
        for t, mult in shifted_topics.items():
            if t in spec.get("per_publisher_rate", {}):
                spec["per_publisher_rate"][t] = float(spec["per_publisher_rate"][t]) * mult

    # Scale service_mean_s on the busiest host (with at least one subscribing app)
    loads = study.host_loads(plan)
    ranked_hosts = sorted(loads.keys(), key=lambda h: (-loads[h], h))
    shifted_apps: List[str] = []
    target_host = None
    for h in ranked_hosts:
        subs = [a for a in out["hosts"][h]["apps"] if out["apps"][a].get("subscribes")]
        if subs:
            target_host = h
            for a in sorted(subs):
                cur = float(out["apps"][a].get("service_mean_s", 0.0))
                if cur > 0:
                    out["apps"][a]["service_mean_s"] = cur * service_mult
                    shifted_apps.append(a)
            break

    meta = {
        "shifted_topics": shifted_topics,
        "target_host": target_host,
        "shifted_apps": shifted_apps,
        "service_mult": service_mult,
    }
    return out, meta


def pick_behavioral_target(plan: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    """Find (publisher_P, subscriber_M) where M has >=2 inputs and publishes at least 1 topic."""
    for app_id in sorted(plan["apps"]):
        spec = plan["apps"][app_id]
        if len(spec.get("subscribes", [])) >= 2 and spec.get("publishes"):
            for topic in spec["subscribes"]:
                pubs = [p for p in plan["topics"][topic]["publishers"] if p != app_id]
                if len(pubs) == 1:
                    return pubs[0], app_id
    return None


def _run_sim(graph: Any, plan: Mapping[str, Any], duration_s: float, seed: int,
             profile: Any = None, behavior_model: str = "open_loop",
             faults: Optional[List[FaultSpec]] = None) -> TelemetryTrace:
    kw: Dict[str, Any] = {
        "behavior_model": behavior_model,
        "deployment_model": "stations",
        "qos_semantics": "dds",
        "rng_streams": "per_process",
        "replica_aware": True,
        "target_utilization": plan["target_utilization"],
    }
    if profile is not None:
        kw["telemetry_profile"] = profile
    if faults:
        kw["faults"] = faults
    res = MessageFlowSimulator(graph, duration=duration_s, seed=seed, record_events=True, **kw).run()
    return TelemetryTrace.from_records(res.events, start_s=0.0, end_s=duration_s)


def _trim(trace: TelemetryTrace, warmup_s: float) -> TelemetryTrace:
    events = [e for e in trace.events if e["t"] >= warmup_s]
    return TelemetryTrace(events=events, start_s=warmup_s, end_s=trace.end_s)


def _metric_medians(comp: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    return {
        m: comp["metrics"][m]["median_rel_error"]
        for m in ("topic_rate", "delivery_rate", "latency_p50", "latency_p95")
    }


def evaluate_scenario(name: str, mode: str, duration_s: float, warmup_s: float,
                      reps: int, out_dir: Path) -> Dict[str, Any]:
    t0 = time.time()
    src = resolve_scenario_path(name)
    graph = load_graph(src)
    plan_nom = build_deployment(src, open_loop=True, qos_semantics="dds")
    plan_nom["source_scenario"] = str(src)

    work_dir = out_dir / name
    tb_nom = write_testbed(plan_nom, work_dir / "tb_nominal", duration_s, seed=BASE_SEED)

    plan_pert, pert_meta = perturb_plan(plan_nom)
    tb_pert = write_testbed(plan_pert, work_dir / "tb_perturbed", duration_s, seed=BASE_SEED + 1)

    # ── Part A: Unperturbed baseline (detect_drift false-positive check) ──
    sim_nom_ref = _trim(_run_sim(graph, plan_nom, duration_s, seed=42), warmup_s)
    fp_flags = []
    nom_delivery_errs = []
    for r in range(reps):
        orchestrate.run(tb_nom, duration_s, [], mode=mode, seed=BASE_SEED + 100 + r)
        obs = _trim(load_run(tb_nom), warmup_s)
        d = detect_drift(sim_nom_ref, obs)
        fp_flags.append(bool(d["drifted"]))
        comp = compare_steady_state(sim_nom_ref, obs)
        nom_delivery_errs.append(comp["metrics"]["delivery_rate"]["median_rel_error"])

    # ── Part B: Perturbed counterpart (non-trivial H5 + R7 drift + EMA) ──
    # Calibration trace on a disjoint seed
    orchestrate.run(tb_pert, duration_s, [], mode=mode, seed=BASE_SEED + 200)
    obs_cal = _trim(load_run(tb_pert), warmup_s)
    cal = TelemetryCalibrator(warmup_s=0.0, infer_policies=False)
    prof_batch = cal.calibrate(obs_cal, source="perturbed_cal")

    # Incremental EMA across 3 disjoint rolling windows starting from nominal baseline profile
    orchestrate.run(tb_nom, duration_s, [], mode=mode, seed=BASE_SEED + 150)
    prof_ema = cal.calibrate(_trim(load_run(tb_nom), warmup_s), source="nominal_prior")
    ema_trajectory: List[Dict[str, Any]] = []

    # Held-out evaluation traces on perturbed counterpart
    obs_eval_traces: List[TelemetryTrace] = []
    tp_flags: List[bool] = []
    topic_recall: List[float] = []
    s2_errs: List[Dict[str, Optional[float]]] = []
    s3_errs: List[Dict[str, Optional[float]]] = []

    sim_s3_batch = _trim(_run_sim(graph, plan_nom, duration_s, seed=42, profile=prof_batch), warmup_s)

    for r in range(reps):
        orchestrate.run(tb_pert, duration_s, [], mode=mode, seed=BASE_SEED + 300 + r)
        obs_e = _trim(load_run(tb_pert), warmup_s)
        obs_eval_traces.append(obs_e)

        # Drift detection against uncalibrated S2
        d = detect_drift(sim_nom_ref, obs_e)
        tp_flags.append(bool(d["drifted"]))
        shifted_set = set(pert_meta["shifted_topics"].keys())
        detected_set = set(d.get("drifted_topics", []))
        topic_recall.append(len(shifted_set & detected_set) / len(shifted_set) if shifted_set else 1.0)

        s2_errs.append(_metric_medians(compare_steady_state(sim_nom_ref, obs_e)))
        s3_errs.append(_metric_medians(compare_steady_state(sim_s3_batch, obs_e)))

        # EMA update step using this window, evaluated on the next (or first held-out) trace
        prof_ema = cal.update_profile(prof_ema, obs_e, alpha=0.5, source=f"window_{r + 1}")
        sim_ema = _trim(_run_sim(graph, plan_nom, duration_s, seed=42, profile=prof_ema), warmup_s)
        ema_comp = compare_steady_state(sim_ema, obs_cal)  # evaluate on disjoint calibration trace
        ema_trajectory.append({"window": r + 1, **_metric_medians(ema_comp)})

    def _med(rows: List[Dict[str, Optional[float]]], key: str) -> Optional[float]:
        vals = [row[key] for row in rows if row[key] is not None]
        return statistics.median(vals) if vals else None

    # ── Part C: Behavioural residual feedback (periodic prior vs all_of reality) ──
    beh_result: Dict[str, Any] = {"skipped": True}
    pair = pick_behavioral_target(plan_nom)
    if pair is not None:
        pub_p, app_m = pair
        fault_s = duration_s / 2.0
        plan_beh = copy.deepcopy(plan_nom)
        plan_beh["apps"][app_m]["policy"] = "all_of"
        tb_beh = write_testbed(plan_beh, work_dir / "tb_behavioral", duration_s, seed=BASE_SEED + 400)
        faults = [FaultSpec(pub_p, "crash", fault_s)]
        orchestrate.run(tb_beh, duration_s, faults, mode=mode, seed=BASE_SEED + 401)
        obs_beh = load_run(tb_beh)

        sim_before = _run_sim(graph, plan_nom, duration_s, seed=42, behavior_model="open_loop", faults=faults)
        subs = study.subscribers_of(plan_nom)
        stalled_obs = stalled_in_trace(obs_beh, fault_s) - {pub_p}
        stalled_before = stalled_in_trace(sim_before, fault_s) - {pub_p}
        imp_obs = {pub_p: observed_impact(obs_beh, subs, pub_p, fault_s, warmup_s=warmup_s)}
        imp_before = {pub_p: observed_impact(sim_before, subs, pub_p, fault_s, warmup_s=warmup_s)}

        def _jaccard(a, b):
            return len(a & b) / len(a | b) if (a | b) else 1.0

        j_before = _jaccard(stalled_before, stalled_obs)
        diag = diagnose_fault_residuals(
            imp_before, imp_obs,
            simulated_stalled=stalled_before,
            observed_stalled=stalled_obs,
            observed_trace=obs_beh,
            current_profile=prof_batch,
        )
        prof_after = apply_residual_feedback(prof_batch, diag)
        sim_after = _run_sim(graph, plan_nom, duration_s, seed=42, profile=prof_after,
                             behavior_model="reactive", faults=faults)
        stalled_after = stalled_in_trace(sim_after, fault_s) - {pub_p}
        j_after = _jaccard(stalled_after, stalled_obs)

        beh_result = {
            "skipped": False,
            "publisher_fault": pub_p,
            "reconfigured_app": app_m,
            "observed_stalled": sorted(stalled_obs),
            "sim_stalled_before": sorted(stalled_before),
            "sim_stalled_after": sorted(stalled_after),
            "stall_jaccard_before": j_before,
            "stall_jaccard_after": j_after,
            "inferred_policy_updates": diag.get("policy_updates", {}),
        }

    return {
        "scenario": name,
        "mode": mode,
        "duration_s": duration_s,
        "reps": reps,
        "wall_s": round(time.time() - t0, 2),
        "perturbation": pert_meta,
        "baseline_drift_fp_rate": sum(fp_flags) / len(fp_flags),
        "baseline_delivery_median_err": statistics.median([x for x in nom_delivery_errs if x is not None]),
        "perturbed_drift_tp_rate": sum(tp_flags) / len(tp_flags),
        "perturbed_topic_recall_median": statistics.median(topic_recall),
        "uncalibrated_S2": {k: _med(s2_errs, k) for k in ("topic_rate", "delivery_rate", "latency_p50", "latency_p95")},
        "calibrated_S3": {k: _med(s3_errs, k) for k in ("topic_rate", "delivery_rate", "latency_p50", "latency_p95")},
        "ema_trajectory": ema_trajectory,
        "behavioral_feedback": beh_result,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", default="tiny_system,atm,realworld_autoware_ros2")
    ap.add_argument("--mode", choices=("local", "docker"), default="local")
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--warmup", type=float, default=5.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "calibration_drift_eval")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in [s.strip() for s in args.scenarios.split(",") if s.strip()]:
        res = evaluate_scenario(name, args.mode, args.duration, args.warmup, args.reps, args.output)
        results[name] = res
        print(
            f"[{name}] mode={args.mode} FP={res['baseline_drift_fp_rate']:.2f} "
            f"TP={res['perturbed_drift_tp_rate']:.2f} topic_recall={res['perturbed_topic_recall_median']:.2f} | "
            f"delivery S2={res['uncalibrated_S2']['delivery_rate']:.3f} -> S3={res['calibrated_S3']['delivery_rate']:.3f} | "
            f"p50 S2={res['uncalibrated_S2']['latency_p50']:.3f} -> S3={res['calibrated_S3']['latency_p50']:.3f} | "
            f"wall={res['wall_s']}s",
            flush=True,
        )
    out_file = args.output / "results.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
