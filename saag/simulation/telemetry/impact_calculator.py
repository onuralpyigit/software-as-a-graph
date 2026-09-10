"""
impact_calculator.py
────────────────────
Derives failure impact scores and ISO/IEC 25010 Quality Metrics from
operational telemetry deltas produced by RuntimeTelemetrySimulator.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from saag.simulation.models import ImpactMetrics
from saag.simulation.simulation_results import (
    CascadeWave,
    FaultInjectionRecord,
    FaultInjectionResult,
)
from .models import ComponentTelemetry, SystemTelemetry


class TelemetryImpactCalculator:
    """
    Translates raw system telemetry into normalized ground-truth impact scores
    and validation oracle metrics.

    Parameters
    ----------
    propagation_threshold : float
        Fraction of feed loss at which a subscriber is counted as starved (default 0.20).
    weights : dict, optional
        Weights for composite impact: delivery, starvation, qos, latency.
    """

    DEFAULT_WEIGHTS = {
        "delivery": 0.40,
        "starvation": 0.30,
        "qos": 0.15,
        "latency": 0.15,
    }

    def __init__(
        self,
        propagation_threshold: float = 0.20,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.propagation_threshold = propagation_threshold
        self.weights = weights or self.DEFAULT_WEIGHTS

    def compute_node_impact(
        self,
        pre_telemetry: SystemTelemetry,
        post_telemetry: SystemTelemetry,
        target_id: str,
        target_type: str = "Application",
        target_name: str = "",
    ) -> Dict[str, Any]:
        """
        Compute impact metrics for target_id by comparing pre-fault (or baseline)
        telemetry against post-fault (or degraded) telemetry.

        Crucially excludes target_id's own consumption so leaf consumers do not
        falsely generate high impact scores.
        """
        pre_comps = pre_telemetry.component_metrics
        post_comps = post_telemetry.component_metrics

        # Surviving subscribers (exclude target if it's a subscriber)
        all_subs = {
            cid for cid, c in pre_comps.items()
            if c.messages_received > 0 and cid != target_id
        }
        if not all_subs:
            # Fall back to any component with messages_received in post
            all_subs = {
                cid for cid, c in post_comps.items()
                if c.messages_received > 0 and cid != target_id
            }

        # 1. Delivery loss on surviving subscribers
        surv_rec_pre = sum(pre_comps[s].messages_received for s in all_subs if s in pre_comps)
        surv_rec_post = sum(post_comps[s].messages_received for s in all_subs if s in post_comps)

        dur_pre = max(1e-6, pre_telemetry.simulation_duration)
        dur_post = max(1e-6, post_telemetry.simulation_duration)

        rate_pre = surv_rec_pre / dur_pre
        rate_post = surv_rec_post / dur_post

        if rate_pre > 1e-9:
            delivery_loss = max(0.0, min(1.0, 1.0 - (rate_post / rate_pre)))
        else:
            delivery_loss = 0.0

        # 2. Per-subscriber feed loss and starvation reach
        per_sub_loss: Dict[str, float] = {}
        starved_subs: Set[str] = set()

        for sub_id in all_subs:
            sub_pre = pre_comps.get(sub_id)
            sub_post = post_comps.get(sub_id)

            sub_rate_pre = (sub_pre.messages_received / dur_pre) if sub_pre else 0.0
            sub_rate_post = (sub_post.messages_received / dur_post) if sub_post else 0.0

            if sub_rate_pre > 1e-9:
                s_loss = max(0.0, min(1.0, 1.0 - (sub_rate_post / sub_rate_pre)))
            else:
                s_loss = 0.0

            per_sub_loss[sub_id] = round(s_loss, 4)
            if s_loss >= self.propagation_threshold:
                starved_subs.add(sub_id)

        starvation_reach = len(starved_subs) / len(all_subs) if all_subs else 0.0

        # 3. QoS violation impact (deadlines missed, queue drops)
        vio_pre = len(pre_telemetry.qos_violations)
        vio_post = len(post_telemetry.qos_violations)
        delta_vio = max(0, vio_post - vio_pre)

        tot_expected = max(1, post_telemetry.total_messages_generated)
        qos_loss = min(1.0, delta_vio / tot_expected)

        # 4. Latency degradation (p95 shift)
        lat_pre = pre_telemetry.system_latency_p95_ms
        lat_post = post_telemetry.system_latency_p95_ms
        if lat_pre > 1e-6 and lat_post > lat_pre:
            latency_loss = min(1.0, (lat_post - lat_pre) / lat_pre)
        else:
            latency_loss = 0.0

        # 5. Composite impact score
        w = self.weights
        composite = (
            w.get("delivery", 0.40) * delivery_loss
            + w.get("starvation", 0.30) * starvation_reach
            + w.get("qos", 0.15) * qos_loss
            + w.get("latency", 0.15) * latency_loss
        )
        composite = round(max(0.0, min(1.0, composite)), 6)

        # Affected topics (topics where delivery dropped or drops occurred)
        affected_topics = set()
        for tid, t_post in post_telemetry.topic_metrics.items():
            t_pre = pre_telemetry.topic_metrics.get(tid)
            pre_del = t_pre.delivered_count if t_pre else 0
            if (t_post.delivered_count < pre_del or t_post.dropped_no_route > 0 or t_post.dropped_queue_full > 0):
                affected_topics.add(tid)

        return {
            "target_id": target_id,
            "target_type": target_type,
            "target_name": target_name or target_id,
            "impact_score": composite,
            "delivery_loss": round(delivery_loss, 4),
            "starvation_reach": round(starvation_reach, 4),
            "qos_loss": round(qos_loss, 4),
            "latency_loss": round(latency_loss, 4),
            "impacted_subscribers": sorted(starved_subs),
            "per_subscriber_feed_loss": per_sub_loss,
            "affected_topics": sorted(affected_topics),
            "total_subscribers": len(all_subs),
        }

    def to_fault_injection_record(
        self,
        impact_data: Dict[str, Any],
        seed_scores: Optional[Dict[int, float]] = None,
        cascade_depth: int = 1,
    ) -> FaultInjectionRecord:
        """Construct a FaultInjectionRecord compatible with impact_scores.json."""
        seed_scores = seed_scores or {42: impact_data["impact_score"]}
        scores_list = list(seed_scores.values())
        mean_score = sum(scores_list) / len(scores_list)
        std_score = (
            math.sqrt(sum((x - mean_score) ** 2 for x in scores_list) / (len(scores_list) - 1))
            if len(scores_list) > 1 else 0.0
        )

        impacted_subs = impact_data.get("impacted_subscribers", [])
        affected_topics = impact_data.get("affected_topics", [])

        waves = [
            CascadeWave(
                wave_index=0,
                newly_orphaned_topics=affected_topics,
                newly_impacted_subscribers=impacted_subs,
                newly_failed_publishers=[impact_data["target_id"]],
            )
        ]

        return FaultInjectionRecord(
            node_id=impact_data["target_id"],
            node_type=impact_data.get("target_type", "Application"),
            node_name=impact_data.get("target_name", impact_data["target_id"]),
            impact_score=round(mean_score, 6),
            total_orphaned_topics=len(affected_topics),
            total_impacted_subscribers=len(impacted_subs),
            total_subscribers=impact_data.get("total_subscribers", len(impacted_subs)),
            cascade_depth=cascade_depth,
            directly_orphaned_topics=affected_topics,
            all_orphaned_topics=affected_topics,
            impacted_subscriber_ids=impacted_subs,
            per_subscriber_feed_loss=impact_data.get("per_subscriber_feed_loss", {}),
            cascade_waves=waves,
            seed_impact_scores=seed_scores,
            impact_score_std=round(std_score, 6),
        )

    def to_fault_injection_result(
        self,
        graph_id: str,
        records: Dict[str, FaultInjectionRecord],
        seeds: List[int],
        unlabeled_node_ids: Optional[List[str]] = None,
        labeled_node_types: Optional[List[str]] = None,
    ) -> FaultInjectionResult:
        """Assemble a FaultInjectionResult matching GNN training requirements."""
        res = FaultInjectionResult(
            graph_id=graph_id,
            seeds_used=seeds,
            labeler="RuntimeTelemetrySimulator",
            labeled_node_types=labeled_node_types or ["Application", "Broker", "Node", "Library"],
            unlabeled_node_ids=unlabeled_node_ids or [],
        )
        for rec in records.values():
            res.add_record(rec)
        res.finalise()
        return res

    def to_impact_metrics(
        self,
        impact_data: Dict[str, Any],
        pre_telemetry: SystemTelemetry,
        post_telemetry: SystemTelemetry,
    ) -> ImpactMetrics:
        """Map telemetry deltas to ISO/IEC 25010 ImpactMetrics for validation gates."""
        delivery_loss = impact_data.get("delivery_loss", 0.0)
        starvation_reach = impact_data.get("starvation_reach", 0.0)
        composite = impact_data.get("impact_score", 0.0)

        # Fragmentation: fraction of nodes that are unreachable or failed
        n_comps = len(pre_telemetry.component_metrics) + len(pre_telemetry.node_metrics)
        failed_comps = sum(1 for c in post_telemetry.component_metrics.values() if c.is_failed)
        failed_comps += sum(1 for n in post_telemetry.node_metrics.values() if n.is_failed)
        fragmentation = failed_comps / max(1, n_comps)

        # Throughput loss
        thru_pre = pre_telemetry.total_messages_delivered / max(1e-6, pre_telemetry.simulation_duration)
        thru_post = post_telemetry.total_messages_delivered / max(1e-6, post_telemetry.simulation_duration)
        throughput_loss = max(0.0, min(1.0, 1.0 - (thru_post / (thru_pre + 1e-9))))

        # Flow disruption: fraction of topics that suffered total drop
        broken_topics = sum(
            1 for t in post_telemetry.topic_metrics.values()
            if t.published_count > 0 and t.delivered_count == 0
        )
        flow_disruption = broken_topics / max(1, len(post_telemetry.topic_metrics))

        metrics = ImpactMetrics(
            reachability_loss=delivery_loss,
            fragmentation=round(fragmentation, 4),
            throughput_loss=round(throughput_loss, 4),
            flow_disruption=round(flow_disruption, 4),
            affected_topics=len(impact_data.get("affected_topics", [])),
            affected_subscribers=len(impact_data.get("impacted_subscribers", [])),
            cascade_count=failed_comps,
            cascade_reach=starvation_reach,
            _manual_composite_impact=composite,
        )
        return metrics
