"""
Arrival-process models beyond the declared periodic rates.

No corpus topic declares ``workload_type``, so every publisher is strictly
periodic and no scenario ever sees a burst. These are opt-in arms; none changes
a topic's *mean* rate, so a subscriber calibrated to ``target_utilization`` is
still at that utilization on average -- bursts only change how the load arrives.

``mmpp``   two-state Markov-modulated Poisson process per publisher. In the
           burst state the rate is ``burst_ratio`` times the base; the calm
           state's rate is lowered so the long-run mean equals the declared
           rate. ``burst_fraction`` is the long-run share of time in the burst
           state and ``mean_burst_s`` the mean burst length.

`apply_zipf_rates` is a graph transform for the sensitivity arm on the
flat-rate scenarios (healthcare, iot_smart_city, financial_trading), where every
topic declares (nearly) the same frequency.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Tuple

WORKLOAD_MODELS: Tuple[str, ...] = ("declared", "mmpp")


@dataclass
class MMPP:
    burst_ratio: float = 4.0
    burst_fraction: float = 0.1
    mean_burst_s: float = 1.0
    #: publisher key -> (in_burst, state_ends_at)
    _state: Dict[str, Tuple[bool, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.burst_fraction < 1.0:
            raise ValueError("burst_fraction must lie in (0, 1)")
        if self.burst_ratio < 1.0:
            raise ValueError("burst_ratio must be >= 1")
        if self.burst_ratio * self.burst_fraction >= 1.0:
            raise ValueError(
                "burst_ratio * burst_fraction must be < 1, or the calm state would "
                "need a negative rate to keep the declared mean")
        if self.mean_burst_s <= 0:
            raise ValueError("mean_burst_s must be positive")

    @property
    def calm_ratio(self) -> float:
        f, b = self.burst_fraction, self.burst_ratio
        return (1.0 - f * b) / (1.0 - f)

    @property
    def mean_calm_s(self) -> float:
        return self.mean_burst_s * (1.0 - self.burst_fraction) / self.burst_fraction

    def interval(self, key: str, base_rate: float, now: float, rng: random.Random) -> float:
        """Next inter-arrival time for one publisher, from its own RNG stream."""
        in_burst, ends = self._state.get(key, (None, 0.0))
        if in_burst is None:
            in_burst = rng.random() < self.burst_fraction
            ends = now + rng.expovariate(1.0 / (self.mean_burst_s if in_burst else self.mean_calm_s))
        while now >= ends:
            in_burst = not in_burst
            ends += rng.expovariate(1.0 / (self.mean_burst_s if in_burst else self.mean_calm_s))
        self._state[key] = (in_burst, ends)
        rate = base_rate * (self.burst_ratio if in_burst else self.calm_ratio)
        return rng.expovariate(rate) if rate > 0 else 1.0


def apply_zipf_rates(graph, exponent: float = 1.0):
    """Copy of *graph* whose topic frequencies follow Zipf(exponent), same total.

    Topics are ranked by declared frequency (descending, ties by id) and
    reassigned rates proportional to ``1 / rank**exponent``, scaled so the
    system's total publication rate is unchanged.
    """
    g = graph.copy()
    topics = [(n, float(d.get("frequency") or 10.0))
              for n, d in g.nodes(data=True) if d.get("type") == "Topic"]
    if not topics:
        return g
    total = sum(f for _, f in topics)
    ranked = sorted(topics, key=lambda x: (-x[1], x[0]))
    weights = [1.0 / (i + 1) ** exponent for i in range(len(ranked))]
    scale = total / sum(weights)
    for (tid, _), w in zip(ranked, weights):
        g.nodes[tid]["frequency"] = w * scale
    return g


def declare_capacities(graph, rho: float = 0.65, target_utilization: float = 0.65):
    """Copy of *graph* with broker and host capacities a scenario did not declare.

    The corpus declares none, so overload has nothing to act on: a backup broker
    gets the default service time and cannot saturate. This sizes the system for its
    healthy load, before any fault and without reference to one:

    * a broker for ``max(its starting load, the fleet's mean load) / rho``, because a
      broker that starts idle is still a member of the fleet and not a spare of
      unlimited size;
    * a host for ``ceil(stationed residents x target_utilization / rho)`` cores.

    An explicit value in the scenario is kept. This is an assumption about the
    system, not a measurement of it.
    """
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must be in (0, 1], got {rho!r}")
    g = graph.copy()
    published, routers, residents = set(), defaultdict(list), defaultdict(list)
    subscribers = set()
    for src, tgt, data in g.edges(data=True):
        kind = data.get("type")
        if kind == "PUBLISHES_TO":
            published.add(tgt)
        elif kind == "SUBSCRIBES_TO":
            subscribers.add(src)
        elif kind == "ROUTES":
            routers[tgt].append(src)
        elif kind == "RUNS_ON":
            residents[tgt].append(src)

    brokers = sorted(n for n, d in g.nodes(data=True) if d.get("type") == "Broker")
    load = dict.fromkeys(brokers, 0.0)
    for topic, routing in routers.items():
        if topic in published:
            load[min(routing)] += float(g.nodes[topic].get("frequency") or 10.0)
    mean = sum(load.values()) / len(load) if load else 0.0
    for b in brokers:
        if max(load[b], mean) > 0:
            g.nodes[b].setdefault("capacity_msgs_s", max(load[b], mean) / rho)
    for host, comps in residents.items():
        stationed = sum(1 for c in comps if c in subscribers)
        g.nodes[host].setdefault("cpu_cores", max(1, math.ceil(stationed * target_utilization / rho)))
    return g
