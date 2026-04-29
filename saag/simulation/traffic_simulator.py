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
from typing import Any, Dict, List, Optional

from saag.adapters import create_repository


class TrafficSimulator:
    """
    Analytical pub-sub traffic load estimator.

    Fetches topology data from Neo4j and computes expected network and broker
    utilisation for a given set of topics, frequency, and duration.
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_all_apps(self) -> List[Dict[str, Any]]:
        """
        Return all Application nodes with the topic IDs they publish/subscribe to.
        """
        repo = create_repository(self.uri, self.user, self.password)
        try:
            with repo.driver.session(database=repo.database) as session:
                result = session.run(
                    """
                    MATCH (a:Application)
                    OPTIONAL MATCH (a)-[:PUBLISHES_TO]->(pt:Topic)
                    WITH a, collect(DISTINCT pt.id) AS pub_ids
                    OPTIONAL MATCH (a)-[:SUBSCRIBES_TO]->(st:Topic)
                    WITH a, pub_ids, collect(DISTINCT st.id) AS sub_ids
                    RETURN
                        a.id                       AS id,
                        COALESCE(a.name, a.id)     AS name,
                        COALESCE(a.weight, 0.5)    AS weight,
                        pub_ids,
                        sub_ids
                    ORDER BY name
                    """
                )
                apps: List[Dict[str, Any]] = []
                for rec in result:
                    apps.append(
                        {
                            "id": rec["id"],
                            "name": rec["name"],
                            "weight": float(rec["weight"]),
                            "pub_topic_ids": list(rec["pub_ids"]),
                            "sub_topic_ids": list(rec["sub_ids"]),
                        }
                    )
                return apps
        finally:
            repo.close()

    def get_all_topics(self) -> List[Dict[str, Any]]:
        """
        Return all Topic nodes with publisher/subscriber/broker metadata.
        """
        repo = create_repository(self.uri, self.user, self.password)
        try:
            with repo.driver.session(database=repo.database) as session:
                result = session.run(
                    """
                    MATCH (t:Topic)
                    OPTIONAL MATCH (pub)-[:PUBLISHES_TO]->(t)
                      WHERE pub:Application OR pub:Library
                    WITH t, count(DISTINCT pub) AS pub_count
                    OPTIONAL MATCH (sub)-[:SUBSCRIBES_TO]->(t)
                      WHERE sub:Application OR sub:Library
                    WITH t, pub_count, count(DISTINCT sub) AS sub_count
                    OPTIONAL MATCH (b:Broker)-[:ROUTES]->(t)
                    WITH t, pub_count, sub_count,
                         collect(DISTINCT b.id)   AS broker_ids,
                         collect(DISTINCT COALESCE(b.name, b.id)) AS broker_names
                    RETURN
                        t.id                   AS id,
                        COALESCE(t.name, t.id) AS name,
                        COALESCE(t.weight, 0.01) AS weight,
                        pub_count,
                        sub_count,
                        broker_ids,
                        broker_names,
                        t.qos_reliability          AS qos_reliability,
                        t.qos_durability           AS qos_durability,
                        t.qos_transport_priority   AS qos_transport_priority,
                        COALESCE(t.size, 0)        AS size
                    ORDER BY t.id
                    """
                )
                topics: List[Dict[str, Any]] = []
                for rec in result:
                    topics.append(
                        {
                            "id": rec["id"],
                            "name": rec["name"],
                            "weight": float(rec["weight"]),
                            "publisher_count": int(rec["pub_count"]),
                            "subscriber_count": int(rec["sub_count"]),
                            "broker_ids": list(rec["broker_ids"]),
                            "broker_names": list(rec["broker_names"]),
                            "qos_reliability": rec["qos_reliability"],
                            "qos_durability": rec["qos_durability"],
                            "qos_transport_priority": rec["qos_transport_priority"],
                            "size": int(rec["size"]),
                        }
                    )
                return topics
        finally:
            repo.close()

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

            # Resolve effective params (per-topic override → global default)
            p = overrides.get(tid, {})
            eff_hz = float(p.get("frequency_hz", frequency_hz))
            eff_dur = float(p.get("duration_sec", duration_sec))
            # Prefer the size stored on the Topic node; fall back to global default
            # for topics that have no size property (size == 0).
            eff_size = int(td["size"]) if td.get("size", 0) > 0 else message_size_bytes

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
                    "duration_sec": eff_dur,
                    "message_size_bytes": eff_size,
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

    def _fetch_topic_data(self, topic_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        repo = create_repository(self.uri, self.user, self.password)
        try:
            with repo.driver.session(database=repo.database) as session:
                result = session.run(
                    """
                    UNWIND $topic_ids AS topic_id
                    MATCH (t:Topic {id: topic_id})
                    OPTIONAL MATCH (pub)-[:PUBLISHES_TO]->(t)
                      WHERE pub:Application OR pub:Library
                    WITH t, count(DISTINCT pub) AS pub_count
                    OPTIONAL MATCH (sub)-[:SUBSCRIBES_TO]->(t)
                      WHERE sub:Application OR sub:Library
                    WITH t, pub_count, count(DISTINCT sub) AS sub_count
                    OPTIONAL MATCH (b:Broker)-[:ROUTES]->(t)
                    WITH t, pub_count, sub_count,
                         collect(DISTINCT b.id)   AS broker_ids,
                         collect(DISTINCT COALESCE(b.name, b.id)) AS broker_names
                    RETURN
                        t.id                     AS id,
                        COALESCE(t.name, t.id)   AS name,
                        COALESCE(t.weight, 0.01) AS weight,
                        pub_count,
                        sub_count,
                        broker_ids,
                        broker_names
                    """,
                    topic_ids=topic_ids,
                )
                data: Dict[str, Dict[str, Any]] = {}
                for rec in result:
                    data[rec["id"]] = {
                        "id": rec["id"],
                        "name": rec["name"],
                        "weight": float(rec["weight"]),
                        "publisher_count": int(rec["pub_count"]),
                        "subscriber_count": int(rec["sub_count"]),
                        "broker_ids": list(rec["broker_ids"]),
                        "broker_names": list(rec["broker_names"]),
                    }
                return data
        finally:
            repo.close()

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
