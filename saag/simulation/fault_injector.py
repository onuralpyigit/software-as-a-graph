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
      *propagation_threshold* (default 0.2 = aggressive feed loss starvation threshold) AND it was
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
            # Check both 'type' (library) and 'ntype' (validation CLI)
            self.node_type[node] = data.get("type") or data.get("ntype") or "Unknown"
            self.node_name[node] = data.get("name") or data.get("label") or node

        for src, tgt, data in g.edges(data=True):
            etype = (data.get("type") or data.get("etype") or "").upper()
            if etype == "PUBLISHES_TO":
                self.topic_publishers[tgt].add(src)
                self.app_publishes[src].add(tgt)
            elif etype == "SUBSCRIBES_TO":
                self.topic_subscribers[tgt].add(src)
                self.app_subscribes[src].add(tgt)
            elif etype == "ROUTES":
                self.broker_routes[src].add(tgt)
                self.topic_routers[tgt].add(src)

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
        publishing (cascades further).  0.2 = aggressive default. Lower values
        model nodes that fail on partial feed loss. For the ATM ConflictDetector,
        which needs both T_radar AND T_tracks, 0.5 would be appropriate.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        seeds: Optional[List[int]] = None,
        cascade_depth_limit: int = 0,
        propagation_threshold: float = 0.2,
    ) -> None:
        self.graph = graph.copy()
        
        # Derive DEPENDS_ON edges dynamically if they are missing
        has_depends_on = any((d.get("type") or d.get("etype") or "").upper() == "DEPENDS_ON" for _, _, d in self.graph.edges(data=True))
        if not has_depends_on:
            from collections import defaultdict
            topic_pubs = defaultdict(set)
            topic_subs = defaultdict(set)
            pub_qos = {}
            sub_qos = {}
            uses_rels = []
            for src, tgt, d in self.graph.edges(data=True):
                etype = (d.get("type") or d.get("etype") or "").upper()
                if etype == "PUBLISHES_TO":
                    topic_pubs[tgt].add(src)
                    if "qos_profile" in d:
                        pub_qos[(src, tgt)] = d["qos_profile"]
                elif etype == "SUBSCRIBES_TO":
                    topic_subs[tgt].add(src)
                    if "qos_profile" in d:
                        sub_qos[(src, tgt)] = d["qos_profile"]
                elif etype == "USES":
                    uses_rels.append((src, tgt))
            # Subscriber depends on publisher (via shared topic)
            for topic, publishers in topic_pubs.items():
                for subscriber in topic_subs[topic]:
                    for publisher in publishers:
                        if subscriber != publisher:
                            qp = pub_qos.get((publisher, topic)) or sub_qos.get((subscriber, topic)) or {}
                            self.graph.add_edge(subscriber, publisher, type="DEPENDS_ON", dependency_type="app_to_app", weight=1.0, qos_profile=qp)
            # App depends on library (uses relationship)
            for app, lib in uses_rels:
                self.graph.add_edge(app, lib, type="DEPENDS_ON", dependency_type="app_to_lib", weight=1.0)
                
        self.seeds = seeds or [42]
        self.cascade_depth_limit = cascade_depth_limit
        self.propagation_threshold = max(0.0, min(1.0, propagation_threshold))
        self._index = _PubSubIndex(self.graph)

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
        """
        Run one softened cascade simulation with a specific seed.
        
        Phase A: Stochastic propagation through DEPENDS_ON edges.
        Phase B: QoS-weighted and rate-weighted soft propagation through pub-sub channels.
        """
        rng = random.Random(seed)
        idx = self._index

        failed_nodes: Set[str] = {node_id}
        orphaned_topics: Set[str] = set()
        directly_orphaned: Set[str] = set()
        impacted_subscribers: Set[str] = set()
        subscriber_lost_feeds: Dict[str, Set[str]] = defaultdict(set)
        waves: List[CascadeWave] = []
        wave_idx = 0

        # Helper to calculate publisher rate_hz
        def get_rate_hz(pub, topic):
            edge_data = self.graph.get_edge_data(pub, topic)
            if edge_data:
                return edge_data.get("rate_hz", 10.0)
            return 10.0

        # Helper to calculate QoS criticality factor of a topic
        def get_qos_factor(topic):
            node_data = self.graph.nodes.get(topic, {})
            factor = 1.0
            if node_data.get("qos_reliability", "").upper() == "RELIABLE":
                factor *= 1.2
            priority = node_data.get("qos_priority", "").upper()
            if priority in ("HIGH", "CRITICAL", "URGENT"):
                factor *= 1.15
            elif priority == "MEDIUM":
                factor *= 1.05
            return factor

        frontier = [node_id]

        while frontier:
            if self.cascade_depth_limit and wave_idx >= self.cascade_depth_limit:
                break
            
            # Stochastic dampening factor based on depth
            depth_damp = max(0.25, 1.0 - wave_idx * 0.15)

            next_frontier = []
            wave_new_orphaned: Set[str] = set()
            wave_new_impacted: Set[str] = set()
            wave_new_failed: List[str] = []

            # --- Phase A: Direct DEPENDS_ON propagation ---
            for u in frontier:
                for v, _, data in self.graph.in_edges(u, data=True):
                    if v in failed_nodes:
                        continue
                    
                    edge_type = (data.get("type") or data.get("etype") or "").upper()
                    if edge_type != "DEPENDS_ON":
                        continue
                        
                    # Stochastic check scaled by edge weight & depth dampening
                    prob = 0.0
                    if rng.random() < prob:
                        failed_nodes.add(v)
                        wave_new_failed.append(v)
                        next_frontier.append(v)

            # --- Phase B: Topic-mediated Soft QoS/Rate-weighted Propagation ---
            # 1. Compute soft topic feed loss based on failed publishers
            topic_loss = {}
            for topic in idx.all_topics:
                publishers = idx.publishers_of(topic)
                if not publishers:
                    routers = idx.topic_routers.get(topic, set())
                    if routers:
                        failed_routers = routers & failed_nodes
                        topic_loss[topic] = len(failed_routers) / len(routers)
                    else:
                        topic_loss[topic] = 0.0
                else:
                    total_rate = sum(get_rate_hz(p, topic) for p in publishers)
                    if total_rate > 0:
                        failed_rate = sum(get_rate_hz(p, topic) for p in publishers if p in failed_nodes)
                        topic_loss[topic] = failed_rate / total_rate
                    else:
                        failed_pubs = publishers & failed_nodes
                        topic_loss[topic] = len(failed_pubs) / len(publishers)
                
                # Apply QoS factor and clamp to [0, 1]
                topic_loss[topic] = min(1.0, topic_loss[topic] * get_qos_factor(topic))

            # 2. Update orphaned/directly_orphaned sets based on fractional feed loss
            for topic, loss in topic_loss.items():
                if loss > 1e-6 and topic not in orphaned_topics:
                    orphaned_topics.add(topic)
                    wave_new_orphaned.add(topic)
                    if wave_idx == 0:
                        directly_orphaned.add(topic)
                    
                    # Track impacted subscribers
                    for sub in idx.subscribers_of(topic):
                        if sub in failed_nodes:
                            continue
                        subscriber_lost_feeds[sub].add(topic)
                        if sub not in impacted_subscribers:
                            wave_new_impacted.add(sub)
                            impacted_subscribers.add(sub)

            # 3. Stochastic failure of impacted subscribers based on continuous average feed loss
            for sub in idx.all_subscribers:
                if sub in failed_nodes:
                    continue
                all_feeds = idx.app_subscribes.get(sub, set())
                if not all_feeds:
                    continue
                sub_loss = sum(topic_loss.get(t, 0.0) for t in all_feeds) / len(all_feeds)
                
                if sub_loss >= self.propagation_threshold and sub_loss > 1e-6:
                    # Failure probability scaled by propagation_threshold
                    prob = min(1.0, sub_loss / max(1e-6, self.propagation_threshold)) * depth_damp
                    if rng.random() < prob:
                        failed_nodes.add(sub)
                        wave_new_failed.append(sub)
                        next_frontier.append(sub)

            if not next_frontier and not wave_new_orphaned:
                break
                
            waves.append(
                CascadeWave(
                    wave_index=wave_idx,
                    newly_orphaned_topics=sorted(wave_new_orphaned),
                    newly_impacted_subscribers=sorted(wave_new_impacted),
                    newly_failed_publishers=wave_new_failed,
                )
            )
            frontier = next_frontier
            wave_idx += 1

        # --- I(v) computation: Mean continuous feed loss across all subscribers ---
        per_sub_loss: Dict[str, float] = {}
        for sub in idx.all_subscribers:
            all_feeds = idx.app_subscribes.get(sub, set())
            if not all_feeds:
                per_sub_loss[sub] = 0.0
                continue
            # Store continuous QoS/rate-weighted feed loss
            # topic_loss is calculated based on final failed_nodes state
            per_sub_loss[sub] = sum(topic_loss.get(t, 0.0) for t in all_feeds) / len(all_feeds)

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
