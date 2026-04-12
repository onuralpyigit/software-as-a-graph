"""
fault_injector.py
─────────────────
Pub-sub-aware BFS cascade fault injector for the SaG pipeline.

PURPOSE
───────
Produces a proxy *ground-truth* impact score I(v) for every node v by
simulating its failure and tracing the resulting cascade through the
publish-subscribe dependency graph.  I(v) is used as the target variable
when computing Spearman ρ between the topology-derived Q(v) predictor and
observed impact.

ALGORITHM
─────────
For each candidate node v (Application or Broker by default):

  Wave 0 – Direct orphaning
    • Remove v from the graph.
    • Application: a topic is orphaned only if v was its SOLE live publisher.
    • Broker: a topic is orphaned only if v was its SOLE routing broker.
      Multi-path routing (another live broker also routes the topic) prevents
      orphaning.  [FIX: BUG-FI-1]

  Cascade propagation (waves 1, 2, …)
    • For each orphaned topic, find all subscribers.
    • A subscriber propagates the cascade when its feed-loss fraction exceeds
      *propagation_threshold* (default 1.0 = completely starved) AND it was
      itself a publisher on other topics.  [DESIGN-FI-1]
    • A set-based pending queue prevents duplicate processing when a node
      loses feeds from multiple topics in the same wave.  [FIX: BUG-FI-2]
    • Repeat until fixpoint or cascade_depth_limit is reached.

  I(v) computation
    • feed_loss_fraction(a) = |lost_feeds(a)| / |all_subscribed_feeds(a)|
    • I(v) = mean_{a ∈ all_subscribers} feed_loss_fraction(a)

MULTI-SEED STABILITY
────────────────────
When multiple seeds are supplied the cascade uses a seeded shuffle to break
ties in wave propagation order.  I(v) is the mean across seeds;
impact_score_std is the standard deviation.  The cascade trace recorded is
from the seed whose impact score is closest to the mean (median-representative
seed).  [FIX: DESIGN-FI-2]
"""

from __future__ import annotations

import logging
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import networkx as nx

from .simulation_results import (
    CascadeWave,
    FaultInjectionRecord,
    FaultInjectionResult,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Graph-index helper
# ─────────────────────────────────────────────────────────────────────────────

class _PubSubIndex:
    """
    Lightweight read-only index over the pub-sub layer of the SaG graph.

    Builds O(1) lookup structures from the raw NetworkX graph so the
    cascade loops spend no time on edge-type filtering.
    """

    def __init__(self, g: nx.DiGraph) -> None:
        # topic_id → set of publisher app_ids
        self.topic_publishers: Dict[str, Set[str]] = defaultdict(set)
        # topic_id → set of subscriber app_ids
        self.topic_subscribers: Dict[str, Set[str]] = defaultdict(set)
        # app_id → set of published topic_ids
        self.app_publishes: Dict[str, Set[str]] = defaultdict(set)
        # app_id → set of subscribed topic_ids
        self.app_subscribes: Dict[str, Set[str]] = defaultdict(set)
        # broker_id → set of topic_ids it routes
        self.broker_routes: Dict[str, Set[str]] = defaultdict(set)
        # topic_id → set of broker_ids that route it (inverse of broker_routes)
        self.topic_routers: Dict[str, Set[str]] = defaultdict(set)
        # node metadata
        self.node_type: Dict[str, str] = {}
        self.node_name: Dict[str, str] = {}

        for node, data in g.nodes(data=True):
            self.node_type[node] = data.get("type", "Unknown")
            self.node_name[node] = data.get("name", node)

        for src, tgt, data in g.edges(data=True):
            etype = data.get("type", "")
            if etype == "PUBLISHES_TO":
                self.topic_publishers[tgt].add(src)
                self.app_publishes[src].add(tgt)
            elif etype == "SUBSCRIBES_TO":
                self.topic_subscribers[tgt].add(src)
                self.app_subscribes[src].add(tgt)
            elif etype == "ROUTES":
                self.broker_routes[src].add(tgt)
                self.topic_routers[tgt].add(src)   # inverse index (BUG-FI-1)

        self.all_subscribers: Set[str] = {
            a for a, subs in self.app_subscribes.items() if subs
        }
        self.all_topics: Set[str] = (
            set(self.topic_publishers.keys()) | set(self.topic_subscribers.keys())
        )

    def publishers_of(self, topic: str) -> Set[str]:
        return self.topic_publishers.get(topic, set())

    def subscribers_of(self, topic: str) -> Set[str]:
        return self.topic_subscribers.get(topic, set())

    def topics_routed_by(self, broker: str) -> Set[str]:
        return self.broker_routes.get(broker, set())

    def live_routers_of(self, topic: str, failed: Set[str]) -> Set[str]:
        """Return brokers that route *topic* and are not in *failed*."""
        return self.topic_routers.get(topic, set()) - failed


# ─────────────────────────────────────────────────────────────────────────────
# Fault Injector
# ─────────────────────────────────────────────────────────────────────────────

class FaultInjector:
    """
    Runs systematic single-node fault injection across a SaG graph and
    produces I(v) ground-truth impact scores for every injected node.

    Parameters
    ----------
    graph : nx.DiGraph
        The full SaG graph exported by GraphExporter.  Must contain
        PUBLISHES_TO and SUBSCRIBES_TO edges (and optionally ROUTES for
        brokers).
    seeds : list[int], optional
        Seeds for multi-seed stability testing.  Default [42].
        Recommended: [42, 123, 456, 789, 2024].
    cascade_depth_limit : int, optional
        Maximum cascade waves.  0 = unlimited.  Default 0.
    propagation_threshold : float, optional
        Fraction of feeds that must be lost before a subscriber itself stops
        publishing (cascades further).  1.0 = completely starved (conservative
        default).  Lower values model nodes that fail on partial feed loss.
        For the ATM ConflictDetector, which needs both T_radar AND T_tracks,
        0.5 would be appropriate.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        seeds: Optional[List[int]] = None,
        cascade_depth_limit: int = 0,
        propagation_threshold: float = 1.0,
    ) -> None:
        self.graph = graph
        self.seeds = seeds or [42]
        self.cascade_depth_limit = cascade_depth_limit
        self.propagation_threshold = max(0.0, min(1.0, propagation_threshold))
        self._index = _PubSubIndex(graph)

    # ── Public API ──────────────────────────────────────────────────────────

    def run(
        self,
        node_types: Optional[List[str]] = None,
        node_ids: Optional[List[str]] = None,
    ) -> FaultInjectionResult:
        """
        Run fault injection for all eligible nodes.

        Parameters
        ----------
        node_types : list[str], optional
            Node types to inject.  Default: ["Application", "Broker"].
        node_ids : list[str], optional
            Explicit node IDs to inject (overrides node_types filter).
        """
        if node_types is None:
            node_types = ["Application", "Broker"]

        if node_ids:
            candidates = [n for n in node_ids if n in self.graph.nodes]
        else:
            candidates = [
                n
                for n, d in self.graph.nodes(data=True)
                if d.get("type", "") in node_types
            ]

        n_apps = sum(
            1 for _, d in self.graph.nodes(data=True) if d.get("type") == "Application"
        )
        n_brokers = sum(
            1 for _, d in self.graph.nodes(data=True) if d.get("type") == "Broker"
        )

        result = FaultInjectionResult(
            graph_id=self.graph.graph.get("id", ""),
            total_application_nodes=n_apps,
            total_broker_nodes=n_brokers,
            total_subscribers=len(self._index.all_subscribers),
            seeds_used=self.seeds,
        )

        logger.info(
            "Fault injection: %d candidates | %d subscriber(s) | seeds=%s | "
            "propagation_threshold=%.2f",
            len(candidates),
            len(self._index.all_subscribers),
            self.seeds,
            self.propagation_threshold,
        )

        for node_id in candidates:
            rec = self._inject_node(node_id)
            result.add_record(rec)
            logger.debug(
                "  %-30s  I(v)=%.4f  depth=%d  orphaned=%d  impacted=%d  std=%.4f",
                node_id,
                rec.impact_score,
                rec.cascade_depth,
                rec.total_orphaned_topics,
                rec.total_impacted_subscribers,
                rec.impact_score_std,
            )

        result.finalise()
        logger.info("Fault injection complete.  %d records.", result.total_nodes_injected)
        return result

    # ── Core injection logic ─────────────────────────────────────────────────

    def _inject_node(self, node_id: str) -> FaultInjectionRecord:
        node_type = self._index.node_type.get(node_id, "Unknown")
        node_name = self._index.node_name.get(node_id, node_id)

        seed_scores: Dict[int, float] = {}
        seed_records: List["_SingleSeedResult"] = []

        for seed in self.seeds:
            sr = self._cascade(node_id, node_type, seed)
            seed_scores[seed] = sr.impact_score
            seed_records.append(sr)

        mean_score = sum(seed_scores.values()) / len(seed_scores)
        std_score = (
            statistics.stdev(seed_scores.values()) if len(seed_scores) > 1 else 0.0
        )

        # DESIGN-FI-2 FIX: select the seed closest to the mean as primary trace
        primary = min(seed_records, key=lambda sr: abs(sr.impact_score - mean_score))

        return FaultInjectionRecord(
            node_id=node_id,
            node_type=node_type,
            node_name=node_name,
            impact_score=round(mean_score, 6),
            total_orphaned_topics=len(primary.all_orphaned_topics),
            total_impacted_subscribers=len(primary.impacted_subscribers),
            total_subscribers=len(self._index.all_subscribers),
            cascade_depth=primary.cascade_depth,
            directly_orphaned_topics=sorted(primary.directly_orphaned_topics),
            all_orphaned_topics=sorted(primary.all_orphaned_topics),
            impacted_subscriber_ids=sorted(primary.impacted_subscribers),
            per_subscriber_feed_loss={
                sub: round(v, 4)
                for sub, v in primary.per_subscriber_feed_loss.items()
            },
            cascade_waves=primary.waves,
            seed_impact_scores=seed_scores,
            impact_score_std=round(std_score, 6),
        )

    def _cascade(self, node_id: str, node_type: str, seed: int) -> "_SingleSeedResult":
        """Run one cascade simulation with a specific seed."""
        rng = random.Random(seed)
        idx = self._index

        failed_nodes: Set[str] = {node_id}
        orphaned_topics: Set[str] = set()
        impacted_subscribers: Set[str] = set()
        subscriber_lost_feeds: Dict[str, Set[str]] = defaultdict(set)
        waves: List[CascadeWave] = []
        wave_idx = 0

        # ── Wave 0: direct orphaning ──────────────────────────────────────

        directly_orphaned: Set[str] = set()

        if node_type == "Broker":
            # BUG-FI-1 FIX: check for redundant routing paths.
            # A topic is only orphaned if ALL of its routing brokers are failed.
            for t in idx.topics_routed_by(node_id):
                if not idx.live_routers_of(t, failed_nodes):
                    directly_orphaned.add(t)
        else:
            for t in idx.app_publishes.get(node_id, set()):
                if not (idx.publishers_of(t) - failed_nodes):
                    directly_orphaned.add(t)

        orphaned_topics.update(directly_orphaned)

        newly_impacted_w0: Set[str] = set()
        for t in directly_orphaned:
            for sub in idx.subscribers_of(t):
                if sub not in failed_nodes:
                    subscriber_lost_feeds[sub].add(t)
                    if sub not in impacted_subscribers:
                        newly_impacted_w0.add(sub)
                        impacted_subscribers.add(sub)

        waves.append(
            CascadeWave(
                wave_index=0,
                newly_orphaned_topics=sorted(directly_orphaned),
                newly_impacted_subscribers=sorted(newly_impacted_w0),
                newly_failed_publishers=[node_id],
            )
        )

        # ── Cascade waves 1, 2, … ─────────────────────────────────────────

        # BUG-FI-2 FIX: set-based pending queue eliminates duplicates.
        propagation_pending: Set[str] = {
            sub for sub in newly_impacted_w0
            if self._should_propagate(sub, subscriber_lost_feeds, idx)
        }

        while propagation_pending:
            wave_idx += 1
            if self.cascade_depth_limit and wave_idx > self.cascade_depth_limit:
                break

            wave_candidates = list(propagation_pending)
            rng.shuffle(wave_candidates)
            propagation_pending = set()

            wave_new_orphaned: Set[str] = set()
            wave_new_impacted: Set[str] = set()
            wave_new_failed: List[str] = []

            for silenced_pub in wave_candidates:
                if silenced_pub in failed_nodes:
                    continue
                failed_nodes.add(silenced_pub)
                wave_new_failed.append(silenced_pub)

                for t in idx.app_publishes.get(silenced_pub, set()):
                    if t in orphaned_topics:
                        continue
                    if idx.publishers_of(t) - failed_nodes:
                        continue  # still has live publishers
                    wave_new_orphaned.add(t)
                    orphaned_topics.add(t)
                    for sub2 in idx.subscribers_of(t):
                        if sub2 in failed_nodes:
                            continue
                        subscriber_lost_feeds[sub2].add(t)
                        if sub2 not in impacted_subscribers:
                            wave_new_impacted.add(sub2)
                            impacted_subscribers.add(sub2)
                        if self._should_propagate(sub2, subscriber_lost_feeds, idx):
                            propagation_pending.add(sub2)

            if not wave_new_orphaned and not wave_new_impacted:
                break

            waves.append(
                CascadeWave(
                    wave_index=wave_idx,
                    newly_orphaned_topics=sorted(wave_new_orphaned),
                    newly_impacted_subscribers=sorted(wave_new_impacted),
                    newly_failed_publishers=wave_new_failed,
                )
            )

        # ── I(v) computation ─────────────────────────────────────────────

        per_sub_loss: Dict[str, float] = {}
        for sub in idx.all_subscribers:
            all_feeds = idx.app_subscribes.get(sub, set())
            if not all_feeds:
                per_sub_loss[sub] = 0.0
                continue
            lost = subscriber_lost_feeds.get(sub, set())
            per_sub_loss[sub] = len(lost) / len(all_feeds)

        total_subs = len(idx.all_subscribers)
        impact_score = sum(per_sub_loss.values()) / total_subs if total_subs else 0.0

        return _SingleSeedResult(
            impact_score=impact_score,
            directly_orphaned_topics=directly_orphaned,
            all_orphaned_topics=orphaned_topics,
            impacted_subscribers=impacted_subscribers,
            per_subscriber_feed_loss=per_sub_loss,
            cascade_depth=wave_idx,
            waves=waves,
        )

    def _should_propagate(
        self,
        sub: str,
        subscriber_lost_feeds: Dict[str, Set[str]],
        idx: _PubSubIndex,
    ) -> bool:
        """
        Return True when *sub* has lost enough feeds to trigger cascade
        propagation, as determined by self.propagation_threshold.
        """
        if not idx.app_publishes.get(sub):
            return False  # not a publisher; cannot spread cascade
        all_feeds = idx.app_subscribes.get(sub, set())
        if not all_feeds:
            return False
        lost = subscriber_lost_feeds.get(sub, set())
        return (len(lost) / len(all_feeds)) >= self.propagation_threshold


# ─────────────────────────────────────────────────────────────────────────────
# Internal result container for a single seed run
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _SingleSeedResult:
    impact_score: float
    directly_orphaned_topics: Set[str]
    all_orphaned_topics: Set[str]
    impacted_subscribers: Set[str]
    per_subscriber_feed_loss: Dict[str, float]
    cascade_depth: int
    waves: List[CascadeWave]
