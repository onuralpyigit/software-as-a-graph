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
