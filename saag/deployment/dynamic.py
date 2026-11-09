"""
Descriptive runtime report on shortlisted placements.

`verify.py` grades a placement on `FailureSimulator`, which has no time in it. This
report replays the same outage events on the runtime engine, so queueing, failover
leases, recovery and overload cascades become visible. It describes a placement and
decides nothing: `reliability_claim` and the recommendation are settled before it
runs, and neither reads it (CLAUDE.md, "Deployment reliability claims are held-out").
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence

from saag.core.graph_io import build_graph_from_json

from .instance import Placement
from .verify import apply_placement

#: The runtime engine with overload on: a failover can now fail from load it inherits.
DYNAMIC_SETTINGS: Dict[str, Any] = {
    "behavior_model": "reactive",
    "deployment_model": "stations",
    "rng_streams": "per_process",
    "estimator": "paired",
    "replica_aware": True,
    "overload_model": "saturation",
}


def _replay_event(graph, targets: Sequence[str], seed: int, duration: float,
                  network_gating: bool) -> Dict[str, Any]:
    from saag.simulation.message_flow_simulator import MessageFlowSimulator
    from saag.simulation.runtime.faults import FaultSpec

    # Bounded, so that recovery can be measured: an outage for a quarter of the run.
    faults = [FaultSpec(t, "transient_crash", duration / 2.0, duration / 4.0, {}) for t in targets]
    fe = MessageFlowSimulator(graph, duration=duration, seed=seed, faults=faults,
                              network_gating=network_gating, **DYNAMIC_SETTINGS).run().fault_event
    return {"delivery_loss": fe.i_dyn_paired,
            "time_to_recover_s": fe.time_to_recover_s,
            "overload_failures": fe.overload_failures or []}


def _worst_recovery(values: List[Optional[float]]) -> Optional[float]:
    """The slowest recovery; ``None`` also means delivery never recovered, which is worse."""
    return None if not values or any(v is None for v in values) else max(values)


def dynamic_report(
    topology: Mapping[str, Any],
    placements: Mapping[str, Placement],
    events: Mapping[str, Sequence[str]],
    p: Mapping[str, float],
    seeds: Sequence[int],
    duration: float = 30.0,
    network_gating: bool = False,
) -> Dict[str, Any]:
    """Replay every outage event on each placement, and summarise per placement."""
    rows: Dict[str, Any] = {}
    for name, placement in placements.items():
        graph = build_graph_from_json(apply_placement(dict(topology), placement))
        per_event: Dict[str, Any] = {}
        for event in sorted(events):
            runs = [_replay_event(graph, events[event], s, duration, network_gating) for s in seeds]
            per_event[event] = {
                "delivery_loss": statistics.fmean(r["delivery_loss"] for r in runs),
                "worst_time_to_recover_s": _worst_recovery([r["time_to_recover_s"] for r in runs]),
                "overload_failures": sorted({c for r in runs for c in r["overload_failures"]}),
            }
        rows[name] = {
            "delivery_loss": sum(p[e] * per_event[e]["delivery_loss"] for e in per_event),
            "worst_delivery_loss": max(v["delivery_loss"] for v in per_event.values()),
            "worst_time_to_recover_s": _worst_recovery(
                [v["worst_time_to_recover_s"] for v in per_event.values()]),
            "overload_failures": sorted({c for v in per_event.values() for c in v["overload_failures"]}),
            "per_event": per_event,
        }
    return {"descriptive_only": True, "engine": "MessageFlowSimulator/runtime",
            "settings": {**DYNAMIC_SETTINGS, "duration": duration, "network_gating": network_gating},
            "seeds": list(seeds), "placements": rows}


# ── Deployment warp ──────────────────────────────────────────────────────────

#: Runtime label the warp is read on: criticality-weighted end-to-end loss at the
#: sinks. It is weighted by declared sink criticality, not by node count, so it
#: does not reduce to reachability; and its crash retest is higher than i_delivery's.
WARP_LABEL = "i_flow"

#: The two regimes compared. ``logical`` keeps brokers and hosts out of the data
#: path, so the placement plays no part. ``deployed`` puts it in: broker stations,
#: host CPU shared by co-located applications, replica-aware host outages and
#: overload. CONNECTS_TO gating is not a third regime: it only acts when a host or
#: link fails, so an Application crash leaves it exactly inert (measured: 0.0 on all
#: 26 Applications of data/network/atm_two_tier_system.json).
WARP_REGIMES: Dict[str, Dict[str, Any]] = {
    "logical": {"deployment_model": "none"},
    "deployed": {"deployment_model": "stations", "overload_model": "saturation", "replica_aware": True},
}


def _paired_delta(after: Mapping[str, Any], before: Mapping[str, Any], seeds: Sequence[int],
                  kappa: float) -> Dict[str, Any]:
    """Seed-paired difference of one component's label between the two regimes.

    ``beyond_noise`` is None with fewer than two paired seeds: noise was not
    measured, which is not the same as no change.
    """
    diffs = [after[str(s)] - before[str(s)] for s in seeds
             if after.get(str(s)) is not None and before.get(str(s)) is not None]
    if not diffs:
        return {"mean": None, "std": None, "beyond_noise": None}
    mean = statistics.fmean(diffs)
    std = statistics.stdev(diffs) if len(diffs) > 1 else None
    beyond = None if std is None else (abs(mean) > kappa * std and abs(mean) > 1e-9)
    return {"mean": mean, "std": std, "beyond_noise": beyond}


def warp_report(
    topology: Mapping[str, Any],
    placement: Placement,
    seeds: Sequence[int],
    duration: float = 30.0,
    kappa: float = 1.0,
    capacities_rho: Optional[float] = None,
    workers: int = 1,
) -> Dict[str, Any]:
    """How each Application's runtime criticality moves once the deployment is in the data path.

    Every Application is crashed at mid-run under both `WARP_REGIMES`, on the same
    seeds and per-process random streams, so the difference is paired. It counts as
    beyond noise when its mean exceeds ``kappa`` times the standard deviation of the
    paired differences across seeds. Negative values are real: a crash can relieve
    contention on the brokers and hosts it shared.

    ``capacities_rho`` sizes undeclared broker and host capacities for that
    utilisation (`declare_capacities`); without it a topology that declares none
    shows little contention. Descriptive: nothing in the recommendation reads it.
    """
    from scipy.stats import spearmanr

    from saag.simulation.runtime.labels import RUNTIME_LABEL_SETTINGS, runtime_labels
    from saag.simulation.runtime.workload import declare_capacities

    graph = build_graph_from_json(apply_placement(dict(topology), placement))
    if capacities_rho is not None:
        graph = declare_capacities(graph, rho=capacities_rho)
    apps = sorted(n for n, d in graph.nodes(data=True) if d.get("type") == "Application")
    base = {**RUNTIME_LABEL_SETTINGS, "duration": duration}
    labels = {
        name: {c: rec[WARP_LABEL] or {} for c, rec in runtime_labels(
            graph, candidates=apps, seeds=seeds, settings={**base, **over}, workers=workers)["records"].items()}
        for name, over in WARP_REGIMES.items()
    }

    def ranks(regime: str) -> Dict[str, int]:
        scored = sorted((c for c in apps if labels[regime][c].get("mean") is not None),
                        key=lambda c: (-labels[regime][c]["mean"], c))
        return {c: i + 1 for i, c in enumerate(scored)}

    rank = {name: ranks(name) for name in WARP_REGIMES}
    components = {
        c: {
            "logical": labels["logical"][c].get("mean"),
            "deployed": labels["deployed"][c].get("mean"),
            "delta": _paired_delta(labels["deployed"][c].get("per_seed", {}),
                                   labels["logical"][c].get("per_seed", {}), seeds, kappa),
            "rank": {name: rank[name].get(c) for name in WARP_REGIMES},
        }
        for c in apps
    }
    # On the label values, not the ranks above, so ties stay ties.
    common = [c for c in apps if c in rank["logical"] and c in rank["deployed"]]
    rho = None
    if len(common) >= 3:
        value = spearmanr([components[c]["logical"] for c in common],
                          [components[c]["deployed"] for c in common])[0]
        rho = None if value != value else float(value)   # NaN when a ranking is constant
    return {
        "descriptive_only": True,
        "engine": "MessageFlowSimulator/runtime",
        "label": WARP_LABEL,
        "fault": "crash at mid-run, one Application at a time",
        "regimes": {name: {**base, **over} for name, over in WARP_REGIMES.items()},
        "seeds": list(seeds), "kappa": kappa, "capacities_rho": capacities_rho,
        "spearman": rho,
        "beyond_noise": sum(1 for c in apps if components[c]["delta"]["beyond_noise"]),
        "components": components,
    }
