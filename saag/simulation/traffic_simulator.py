"""
Traffic Simulator

Analytical computation of expected pub-sub network and broker load for a set
of selected topics, given a message frequency (Hz) and duration (seconds).

No discrete-event simulation is required — metrics are derived directly from
the graph topology (publisher count, subscriber count, broker routing).

Metrics:
    Per topic:
        msgs_published_per_sec   = publisher_count × frequency_hz
        msgs_delivered_per_sec   = msgs_published_per_sec × subscriber_count
        bandwidth_in_bps         = msgs_published_per_sec × message_size_bytes
        bandwidth_out_bps        = msgs_delivered_per_sec × message_size_bytes
        bandwidth_total_bps      = bandwidth_in_bps + bandwidth_out_bps

    Per broker (accumulates across all its routed topics):
        msgs_inbound_per_sec     = Σ msgs_published_per_sec
        msgs_outbound_per_sec    = Σ msgs_delivered_per_sec
        bandwidth_bps            = Σ bandwidth_total_bps

    Aggregate:
        total_msgs_published     = Σ msgs_published_per_sec × duration_sec
        total_msgs_delivered     = Σ msgs_delivered_per_sec × duration_sec
        total_network_bps        = Σ bandwidth_total_bps
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from saag.core.ports.graph_repository import IGraphRepository


class TrafficSimulator:
    """
    Analytical pub-sub traffic load estimator.

    Reads raw structural topology data through an ``IGraphRepository`` port
    (supporting both ``MemoryRepository`` and ``Neo4jRepository``) and computes
    expected network and broker utilisation for a given set of topics,
    frequency, and duration.
    """

    def __init__(self, repository: IGraphRepository):
        self.repository = repository
        self.logger = logging.getLogger(__name__)

    @classmethod
    def from_neo4j(
        cls, uri: str, user: str, password: str, database: str = "neo4j"
    ) -> "TrafficSimulator":
        """Convenience constructor backed by a Neo4jRepository."""
        from saag.infrastructure.neo4j_repo import Neo4jRepository

        return cls(Neo4jRepository(uri=uri, user=user, password=password, database=database))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_all_apps(self) -> List[Dict[str, Any]]:
        """
        Return all Application nodes with the topic IDs they publish/subscribe to.
        """
        graph = self.repository.get_graph_data(include_raw=True)
        pub_by_app: Dict[str, Set[str]] = defaultdict(set)
        sub_by_app: Dict[str, Set[str]] = defaultdict(set)
        topic_ids = {c.id for c in graph.components if c.component_type == "Topic"}

        for edge in graph.edges:
            rel = (edge.relation_type or "").upper()
            if rel == "PUBLISHES_TO" and edge.target_id in topic_ids:
                pub_by_app[edge.source_id].add(edge.target_id)
            elif rel == "SUBSCRIBES_TO" and edge.target_id in topic_ids:
                sub_by_app[edge.source_id].add(edge.target_id)

        apps: List[Dict[str, Any]] = []
        for comp in graph.components:
            if comp.component_type != "Application":
                continue
            props = comp.properties or {}
            apps.append(
                {
                    "id": comp.id,
                    "name": props.get("name") or comp.id,
                    "weight": float(comp.weight if comp.weight is not None else 0.5),
                    "role": props.get("role"),
                    "hotstandby": props.get("hotstandby"),
                    "criticality": props.get("criticality"),
                    "pub_topic_ids": sorted(pub_by_app.get(comp.id, ())),
                    "sub_topic_ids": sorted(sub_by_app.get(comp.id, ())),
                }
            )
        apps.sort(key=lambda a: a["name"])
        return apps

    def get_all_topics(self) -> List[Dict[str, Any]]:
        """
        Return all Topic nodes with publisher/subscriber/broker metadata.
        """
        topics_map = self._load_topic_metadata(default_frequency=10.0)
        return [topics_map[tid] for tid in sorted(topics_map)]

    def simulate(
        self,
        topic_ids: List[str],
        frequency_hz: float,
        duration_sec: float,
        message_size_bytes: int = 1024,
        per_topic_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Compute expected traffic metrics for the selected topics.

        Args:
            topic_ids:           IDs of the topics to include in the simulation.
            frequency_hz:        Default message publication rate per publisher (Hz).
            duration_sec:        Default simulation window length (seconds).
            message_size_bytes:  Default assumed average message payload size (bytes).
            per_topic_params:    Optional per-topic overrides. Keys are topic IDs;
                                 values are dicts with any of ``frequency_hz``,
                                 ``duration_sec``, ``message_size_bytes``.
                                 Falls back to global defaults for absent keys.

        Returns:
            Dict with keys ``summary``, ``per_topic``, and ``broker_usage``.
        """
        if not topic_ids:
            return self._empty_result(frequency_hz, duration_sec, message_size_bytes)

        topic_data = self._fetch_topic_data(topic_ids)
        overrides = per_topic_params or {}

        per_topic: List[Dict[str, Any]] = []
        broker_load: Dict[str, Dict[str, Any]] = {}
        total_msgs_published = 0.0
        total_msgs_delivered = 0.0
        total_network_bps = 0.0

        for tid in topic_ids:
            if tid not in topic_data:
                self.logger.warning("Topic %s not found in graph — skipped.", tid)
                continue

            td = topic_data[tid]
            pub_count = max(1, td["publisher_count"])
            sub_count = max(0, td["subscriber_count"])

            # Resolve effective params (per-topic override → graph value → global default)
            p = overrides.get(tid, {})
            eff_hz = float(p.get("frequency_hz", td["frequency"] if td["frequency"] is not None else frequency_hz))
            eff_dur = float(p.get("duration_sec", duration_sec))
            # Prefer per-topic override, then graph value, then global default
            graph_size = td.get("size", 0)
            if "message_size_bytes" in p:
                eff_size = int(p["message_size_bytes"])
            elif graph_size > 0:
                eff_size = graph_size
            else:
                eff_size = message_size_bytes

            msgs_in_per_sec = pub_count * eff_hz
            msgs_out_per_sec = msgs_in_per_sec * sub_count
            msgs_total_per_sec = msgs_in_per_sec + msgs_out_per_sec

            msgs_published = msgs_in_per_sec * eff_dur
            msgs_delivered = msgs_out_per_sec * eff_dur

            bandwidth_in_bps = msgs_in_per_sec * eff_size
            bandwidth_out_bps = msgs_out_per_sec * eff_size
            bandwidth_total_bps = bandwidth_in_bps + bandwidth_out_bps

            per_topic.append(
                {
                    "topic_id": tid,
                    "topic_name": td["name"],
                    "weight": td["weight"],
                    "publisher_count": pub_count,
                    "subscriber_count": sub_count,
                    "broker_ids": td["broker_ids"],
                    "broker_names": td["broker_names"],
                    "frequency_hz": eff_hz,
                    "graph_frequency_hz": td["frequency"],
                    "duration_sec": eff_dur,
                    "message_size_bytes": eff_size,
                    "graph_size_bytes": td.get("size", 0) if td.get("size") is not None else 0,
                    "msgs_published_per_sec": round(msgs_in_per_sec, 4),
                    "msgs_delivered_per_sec": round(msgs_out_per_sec, 4),
                    "msgs_total_per_sec": round(msgs_total_per_sec, 4),
                    "msgs_published_total": round(msgs_published, 2),
                    "msgs_delivered_total": round(msgs_delivered, 2),
                    "bandwidth_in_bps": round(bandwidth_in_bps, 2),
                    "bandwidth_out_bps": round(bandwidth_out_bps, 2),
                    "bandwidth_total_bps": round(bandwidth_total_bps, 2),
                }
            )

            total_msgs_published += msgs_published
            total_msgs_delivered += msgs_delivered
            total_network_bps += bandwidth_total_bps

            # Accumulate per-broker load
            for i, broker_id in enumerate(td["broker_ids"]):
                b_name = td["broker_names"][i] if i < len(td["broker_names"]) else broker_id
                if broker_id not in broker_load:
                    broker_load[broker_id] = {
                        "broker_id": broker_id,
                        "broker_name": b_name,
                        "topics_routed": [],
                        "msgs_inbound_per_sec": 0.0,
                        "msgs_outbound_per_sec": 0.0,
                        "msgs_total_per_sec": 0.0,
                        "bandwidth_bps": 0.0,
                    }
                broker_load[broker_id]["topics_routed"].append(tid)
                broker_load[broker_id]["msgs_inbound_per_sec"] += msgs_in_per_sec
                broker_load[broker_id]["msgs_outbound_per_sec"] += msgs_out_per_sec
                broker_load[broker_id]["msgs_total_per_sec"] += msgs_total_per_sec
                broker_load[broker_id]["bandwidth_bps"] += bandwidth_total_bps

        # Round broker load values
        broker_usage = []
        for b in broker_load.values():
            broker_usage.append(
                {
                    **b,
                    "msgs_inbound_per_sec": round(b["msgs_inbound_per_sec"], 4),
                    "msgs_outbound_per_sec": round(b["msgs_outbound_per_sec"], 4),
                    "msgs_total_per_sec": round(b["msgs_total_per_sec"], 4),
                    "bandwidth_bps": round(b["bandwidth_bps"], 2),
                    "bandwidth_mbps": round(b["bandwidth_bps"] / 1_000_000, 6),
                }
            )

        # Sort broker usage by total msgs descending
        broker_usage.sort(key=lambda x: x["msgs_total_per_sec"], reverse=True)

        summary = {
            "selected_topics": len(topic_ids),
            "topics_found": len(topic_data),
            "frequency_hz": frequency_hz,
            "duration_sec": duration_sec,
            "message_size_bytes": message_size_bytes,
            "total_msgs_published": round(total_msgs_published, 2),
            "total_msgs_delivered": round(total_msgs_delivered, 2),
            "total_network_bps": round(total_network_bps, 2),
            "total_network_mbps": round(total_network_bps / 1_000_000, 6),
            "total_network_kbps": round(total_network_bps / 1_000, 4),
            "peak_topic_bps": round(
                max((t["bandwidth_total_bps"] for t in per_topic), default=0.0), 2
            ),
            "brokers_involved": len(broker_load),
        }

        return {
            "summary": summary,
            "per_topic": per_topic,
            "broker_usage": broker_usage,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_topic_metadata(
        self,
        topic_ids: Optional[List[str]] = None,
        default_frequency: Optional[float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        graph = self.repository.get_graph_data(include_raw=True)
        allowed = set(topic_ids) if topic_ids is not None else None
        comp_by_id = {c.id: c for c in graph.components}
        messaging_types = {"Application", "Library"}

        pubs_by_topic: Dict[str, Set[str]] = defaultdict(set)
        subs_by_topic: Dict[str, Set[str]] = defaultdict(set)
        brokers_by_topic: Dict[str, Set[str]] = defaultdict(set)

        for edge in graph.edges:
            rel = (edge.relation_type or "").upper()
            src = comp_by_id.get(edge.source_id)
            dst = comp_by_id.get(edge.target_id)
            if rel == "PUBLISHES_TO":
                if src is not None and src.component_type in messaging_types:
                    pubs_by_topic[edge.target_id].add(edge.source_id)
            elif rel == "SUBSCRIBES_TO":
                if src is not None and src.component_type in messaging_types:
                    subs_by_topic[edge.target_id].add(edge.source_id)
            elif rel == "ROUTES":
                if src is not None and src.component_type == "Broker":
                    brokers_by_topic[edge.target_id].add(edge.source_id)
                elif dst is not None and dst.component_type == "Broker":
                    brokers_by_topic[edge.source_id].add(edge.target_id)

        data: Dict[str, Dict[str, Any]] = {}
        for comp in graph.components:
            if comp.component_type != "Topic":
                continue
            if allowed is not None and comp.id not in allowed:
                continue
            props = comp.properties or {}
            nested_qos = props.get("qos") if isinstance(props.get("qos"), dict) else {}

            b_ids = sorted(brokers_by_topic.get(comp.id, ()))
            b_names = [
                (comp_by_id[bid].properties or {}).get("name") or bid
                if bid in comp_by_id else bid
                for bid in b_ids
            ]

            raw_freq = props.get("topic_frequency", props.get("frequency"))
            freq = float(raw_freq) if raw_freq is not None else default_frequency
            raw_deadline = props.get("qos_deadline_ms", props.get("deadline_ms"))
            raw_history = props.get("qos_history_depth", props.get("history_depth"))

            data[comp.id] = {
                "id": comp.id,
                "name": props.get("name") or comp.id,
                "weight": float(comp.weight if comp.weight is not None else 0.01),
                "publisher_count": len(pubs_by_topic.get(comp.id, ())),
                "subscriber_count": len(subs_by_topic.get(comp.id, ())),
                "broker_ids": b_ids,
                "broker_names": b_names,
                "qos_reliability": props.get("qos_reliability") or nested_qos.get("reliability"),
                "qos_durability": props.get("qos_durability") or nested_qos.get("durability"),
                "qos_transport_priority": (
                    props.get("qos_transport_priority") or nested_qos.get("transport_priority")
                ),
                "size": int(props.get("size") or 0),
                "frequency": freq,
                "criticality": props.get("topic_criticality", props.get("criticality")),
                "deadline_ms": float(raw_deadline) if raw_deadline is not None else None,
                "history_depth": int(raw_history) if raw_history is not None else 10,
            }
        return data

    def _fetch_topic_data(self, topic_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        return self._load_topic_metadata(topic_ids=topic_ids, default_frequency=None)

    @staticmethod
    def _empty_result(
        frequency_hz: float, duration_sec: float, message_size_bytes: int
    ) -> Dict[str, Any]:
        return {
            "summary": {
                "selected_topics": 0,
                "topics_found": 0,
                "frequency_hz": frequency_hz,
                "duration_sec": duration_sec,
                "message_size_bytes": message_size_bytes,
                "total_msgs_published": 0.0,
                "total_msgs_delivered": 0.0,
                "total_network_bps": 0.0,
                "total_network_mbps": 0.0,
                "total_network_kbps": 0.0,
                "peak_topic_bps": 0.0,
                "brokers_involved": 0,
            },
            "per_topic": [],
            "broker_usage": [],
        }
