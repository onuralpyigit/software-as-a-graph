"""
In-Memory Graph Repository Adapter

Implements IGraphRepository using in-memory storage for testing.
"""

from typing import Dict, Any, List, Optional
import copy

from saag.core.ports.graph_repository import IGraphRepository
from saag.core.models import GraphData, ComponentData, EdgeData
from saag.core.utils import serialization


class MemoryRepository:
    """
    In-memory adapter implementing IGraphRepository.
    
    Useful for testing without Neo4j dependency.
    """
    
    def __init__(self) -> None:
        """Initialize in-memory repository."""
        self.data: Dict[str, Any] = {
            "metadata": {},
            "nodes": [],
            "brokers": [],
            "topics": [],
            "applications": [],
            "libraries": [],
            "relationships": {
                "runs_on": [],
                "routes": [],
                "publishes_to": [],
                "subscribes_to": [],
                "connects_to": [],
                "uses": [],
                "depends_on": [],
            },
        }
        self.derived_stats: Dict[str, int] = {}

    def close(self) -> None:
        """No-op for in-memory repository."""
        pass

    def derive_dependencies(self) -> None:
        """
        Pre-analysis stage: derive DEPENDS_ON relationships and finalise
        DEPENDS_ON edge weights.
        """
        self._derive_dependencies()
        self._finalize_dependency_weights()

    def _calculate_intrinsic_weights(self) -> None:
        """
        Step 3: Compute intrinsic weights for Topic nodes.
        w(topic) = beta * QoS_score + (1-beta) * Size_Norm
        """
        from saag.core.models import TOPIC_QOS_WEIGHT_BETA, MIN_TOPIC_WEIGHT, QoSPolicy
        import math

        rel_scores = QoSPolicy.RELIABILITY_SCORES
        dur_scores = QoSPolicy.DURABILITY_SCORES
        pri_scores = QoSPolicy.PRIORITY_SCORES
        
        beta = TOPIC_QOS_WEIGHT_BETA
        one_minus_beta = 1.0 - beta

        # 1. Update Topic weights
        for topic in self.data["topics"]:
            rel = topic.get("qos_reliability", "BEST_EFFORT")
            dur = topic.get("qos_durability", "VOLATILE")
            pri = topic.get("qos_transport_priority", "MEDIUM")
            size = topic.get("size", 256)

            rel_val = rel_scores.get(rel, 0.0)
            dur_val = dur_scores.get(dur, 0.0)
            
            if pri in ("HIGHEST", "CRITICAL"):
                pri_val = 1.0
            else:
                pri_val = pri_scores.get(pri, 0.0)

            qos_score = 0.30 * rel_val + 0.40 * dur_val + 0.30 * pri_val
            
            if size <= 0:
                size_norm = 0.0
            else:
                size_norm = math.log2(1.0 + size / 1024.0) / 50.0
                if size_norm > 1.0:
                    size_norm = 1.0
            
            weighted_sum = beta * qos_score + one_minus_beta * size_norm
            topic["weight"] = max(MIN_TOPIC_WEIGHT, weighted_sum)

        # 2. Update edge weights (Inherit from Topic)
        topic_weights = {t["id"]: t["weight"] for t in self.data["topics"]}

        for rel_type in ["publishes_to", "subscribes_to"]:
            for rel in self.data["relationships"].get(rel_type, []):
                topic_id = rel.get("to")
                if topic_id in topic_weights:
                    rel["weight"] = topic_weights[topic_id]

        for rel in self.data["relationships"].get("routes", []):
            topic_id = rel.get("to")
            if topic_id in topic_weights:
                rel["weight"] = topic_weights[topic_id]

    def _calculate_aggregate_weights(self) -> None:
        """
        Step 5: Compute aggregate weights for secondary infrastructure components.
        """
        from saag.core.models import (
            APP_HYBRID_MAX_COEFF, APP_HYBRID_MEAN_COEFF,
            BROKER_HYBRID_MAX_COEFF, BROKER_HYBRID_MEAN_COEFF,
            LIB_FANOUT_GAMMA
        )
        import math

        topic_weights = {t["id"]: t.get("weight", 1.0) for t in self.data["topics"]}

        def get_connected_topic_weights(node_id: str, rel_types: List[str]) -> List[float]:
            weights = []
            for r_type in rel_types:
                for rel in self.data["relationships"].get(r_type, []):
                    if rel.get("from") == node_id:
                        tgt = rel.get("to")
                        if tgt in topic_weights:
                            weights.append(topic_weights[tgt])
            return weights

        # 1. Application Weight (hybrid: 0.80 * max + 0.20 * mean)
        for app in self.data["applications"]:
            t_weights = get_connected_topic_weights(app["id"], ["publishes_to", "subscribes_to"])
            if t_weights:
                max_w = max(t_weights)
                mean_w = sum(t_weights) / len(t_weights)
                app["weight"] = APP_HYBRID_MAX_COEFF * max_w + APP_HYBRID_MEAN_COEFF * mean_w
            else:
                app["weight"] = 0.01

        # 2. Library Weight (propagated + fan-out multiplier)
        lib_uses = {}
        for rel in self.data["relationships"].get("uses", []):
            src = rel.get("from")
            tgt = rel.get("to")
            lib_uses.setdefault(tgt, set()).add(src)

        app_weights = {a["id"]: a.get("weight", 0.01) for a in self.data["applications"]}
        
        for lib in self.data["libraries"]:
            t_weights = get_connected_topic_weights(lib["id"], ["publishes_to", "subscribes_to"])
            t_max = max(t_weights) if t_weights else 0.0

            using_apps = lib_uses.get(lib["id"], set())
            using_app_ids = [aid for aid in using_apps if aid in app_weights]
            a_max = max(app_weights[aid] for aid in using_app_ids) if using_app_ids else 0.0
            dg_in = len(using_app_ids)

            base_w = max(t_max, a_max)
            multiplier = 1.0 + LIB_FANOUT_GAMMA * math.log2(1.0 + dg_in)
            
            if base_w <= 0:
                lib["weight"] = 0.01
            else:
                w = base_w * multiplier
                if w > 1.0:
                    w = 1.0
                lib["weight"] = max(0.01, w)

        # 1.5. Application Weight — library-mediated topics (second pass)
        lib_weights = {l["id"]: l.get("weight", 0.01) for l in self.data["libraries"]}
        app_uses_libs = {}
        for rel in self.data["relationships"].get("uses", []):
            src = rel.get("from")
            tgt = rel.get("to")
            if src in app_weights:
                app_uses_libs.setdefault(src, []).append(tgt)

        for app in self.data["applications"]:
            if app.get("weight", 1.0) <= 0.01:
                used_libs = app_uses_libs.get(app["id"], [])
                used_lib_weights = [lib_weights[lid] for lid in used_libs if lid in lib_weights]
                if used_lib_weights:
                    app["weight"] = max(used_lib_weights)

        # 3. Broker Weight (hybrid: 0.70 * max + 0.30 * mean)
        for broker in self.data["brokers"]:
            t_weights = get_connected_topic_weights(broker["id"], ["routes"])
            if t_weights:
                max_w = max(t_weights)
                mean_w = sum(t_weights) / len(t_weights)
                broker["weight"] = BROKER_HYBRID_MAX_COEFF * max_w + BROKER_HYBRID_MEAN_COEFF * mean_w
            else:
                broker["weight"] = 0.01

        # 4. Node Weight (max hosted component weight)
        final_app_weights = {a["id"]: a.get("weight", 0.01) for a in self.data["applications"]}
        broker_weights = {b["id"]: b.get("weight", 0.01) for b in self.data["brokers"]}

        node_components = {}
        for rel in self.data["relationships"].get("runs_on", []):
            src = rel.get("from")
            tgt = rel.get("to")
            weight = final_app_weights.get(src, broker_weights.get(src, 0.01))
            node_components.setdefault(tgt, []).append(weight)

        for node in self.data["nodes"]:
            hosted_weights = node_components.get(node["id"], [])
            if hosted_weights:
                node["weight"] = max(hosted_weights)
            else:
                node["weight"] = 0.01

    def _find_transitive_uses(self, start_id: str) -> List[str]:
        """Find libraries reachable from start_id via USES in 1..3 hops."""
        uses_map = {}
        for rel in self.data["relationships"].get("uses", []):
            src = rel.get("from")
            tgt = rel.get("to")
            uses_map.setdefault(src, []).append(tgt)

        visited = set()
        queue = [(start_id, 0)]
        while queue:
            node_id, depth = queue.pop(0)
            if depth > 0:
                visited.add(node_id)
            if depth < 3:
                for neighbor in uses_map.get(node_id, []):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
        return list(visited)

    def _derive_dependencies(self) -> None:
        """
        Derive DEPENDS_ON relationships from structural edges (Rule 1-6).
        """
        # Build component type and weight lookup
        comp_lookup = {}
        for key in ["nodes", "brokers", "topics", "applications", "libraries"]:
            ctype = key[:-1].capitalize()
            if key == "libraries":
                ctype = "Library"
            for item in self.data.get(key, []):
                comp_lookup[item["id"]] = {
                    "type": ctype,
                    "weight": item.get("weight", 0.01)
                }

        topic_weights = {t["id"]: t.get("weight", 0.01) for t in self.data["topics"]}

        # Helper to trace USES transitive paths for applications
        transitive_uses = {}
        for app in self.data["applications"]:
            transitive_uses[app["id"]] = self._find_transitive_uses(app["id"])

        subscribes_to = self.data["relationships"].get("subscribes_to", [])
        publishes_to = self.data["relationships"].get("publishes_to", [])
        routes = self.data["relationships"].get("routes", [])
        runs_on = self.data["relationships"].get("runs_on", [])

        # Group publishers and subscribers by Topic
        topic_publishers = {}
        for r in publishes_to:
            src = r["from"]
            tgt = r["to"]
            topic_publishers.setdefault(tgt, []).append(src)

        topic_subscribers = {}
        for r in subscribes_to:
            src = r["from"]
            tgt = r["to"]
            topic_subscribers.setdefault(tgt, []).append(src)

        topic_brokers = {}
        for r in routes:
            src = r["from"]
            tgt = r["to"]
            topic_brokers.setdefault(tgt, []).append(src)

        # Dictionary of (src_id, tgt_id, dep_type) -> {"direct_topics": set, "trans_sub_topics": set, "trans_pub_topics": set, "trans_broker_topics": set}
        app_app_candidates = {}
        app_broker_candidates = {}

        # -------------------------------------------------------------
        # Rule 1: app_to_app (direct & transitive)
        # -------------------------------------------------------------
        for t_id, publishers in topic_publishers.items():
            subscribers = topic_subscribers.get(t_id, [])
            
            # Direct Rule 1
            for sub in subscribers:
                if sub not in comp_lookup or comp_lookup[sub]["type"] not in ("Application", "Library"):
                    continue
                for pub in publishers:
                    if pub not in comp_lookup or comp_lookup[pub]["type"] not in ("Application", "Library"):
                        continue
                    if sub == pub:
                        continue
                    key = (sub, pub)
                    app_app_candidates.setdefault(key, {"direct": set(), "trans_sub": set(), "trans_pub": set()})
                    app_app_candidates[key]["direct"].add(t_id)

            # Transitive Rule 1 (sub via library chain)
            for lib in subscribers:
                for app in self.data["applications"]:
                    if lib in transitive_uses.get(app["id"], []):
                        for pub in publishers:
                            if pub not in comp_lookup or comp_lookup[pub]["type"] not in ("Application", "Library"):
                                continue
                            if app["id"] == pub:
                                continue
                            key = (app["id"], pub)
                            app_app_candidates.setdefault(key, {"direct": set(), "trans_sub": set(), "trans_pub": set()})
                            app_app_candidates[key]["trans_sub"].add(t_id)

            # Transitive Rule 1 (pub via library chain)
            for lib in publishers:
                for app in self.data["applications"]:
                    if lib in transitive_uses.get(app["id"], []):
                        for sub in subscribers:
                            if sub not in comp_lookup or comp_lookup[sub]["type"] not in ("Application", "Library"):
                                continue
                            if sub == app["id"]:
                                continue
                            key = (sub, app["id"])
                            app_app_candidates.setdefault(key, {"direct": set(), "trans_sub": set(), "trans_pub": set()})
                            app_app_candidates[key]["trans_pub"].add(t_id)

        # -------------------------------------------------------------
        # Rule 2: app_to_broker (direct & transitive)
        # -------------------------------------------------------------
        for t_id, brokers in topic_brokers.items():
            publishers = topic_publishers.get(t_id, [])
            subscribers = topic_subscribers.get(t_id, [])
            all_direct_apps = set(publishers + subscribers)

            for broker in brokers:
                if broker not in comp_lookup or comp_lookup[broker]["type"] != "Broker":
                    continue
                
                # Direct app_to_broker
                for app in all_direct_apps:
                    if app not in comp_lookup or comp_lookup[app]["type"] not in ("Application", "Library"):
                        continue
                    key = (app, broker)
                    app_broker_candidates.setdefault(key, {"direct": set(), "trans": set()})
                    app_broker_candidates[key]["direct"].add(t_id)

                # Transitive app_to_broker (via library)
                for lib in all_direct_apps:
                    for app in self.data["applications"]:
                        if lib in transitive_uses.get(app["id"], []):
                            key = (app["id"], broker)
                            app_broker_candidates.setdefault(key, {"direct": set(), "trans": set()})
                            app_broker_candidates[key]["trans"].add(t_id)

        # Create relationships dictionary to collect all DEPENDS_ON edges
        depends_on_edges = {}

        # Merge App-to-App
        for (src, tgt), paths in app_app_candidates.items():
            all_topics = paths["direct"] | paths["trans_sub"] | paths["trans_pub"]
            if not all_topics:
                continue
            path_count = max(len(paths["direct"]), len(paths["trans_sub"]), len(paths["trans_pub"]))
            max_weight = max(topic_weights[t] for t in all_topics)
            depends_on_edges[(src, tgt, "app_to_app")] = {
                "weight": max_weight,
                "path_count": path_count
            }

        # Merge App-to-Broker
        for (src, tgt), paths in app_broker_candidates.items():
            all_topics = paths["direct"] | paths["trans"]
            if not all_topics:
                continue
            path_count = max(len(paths["direct"]), len(paths["trans"]))
            max_weight = max(topic_weights[t] for t in all_topics)
            depends_on_edges[(src, tgt, "app_to_broker")] = {
                "weight": max_weight,
                "path_count": path_count
            }

        # Map hosted components to their Nodes
        comp_node = {}
        for r in runs_on:
            src = r["from"]
            tgt = r["to"]  # Node
            comp_node[src] = tgt

        # -------------------------------------------------------------
        # Rule 3: node_to_node
        # -------------------------------------------------------------
        node_node_groups = {}
        for (src, tgt, dep_type), edge_info in depends_on_edges.items():
            if dep_type in ("app_to_app", "app_to_broker"):
                n1 = comp_node.get(src)
                n2 = comp_node.get(tgt)
                if n1 and n2 and n1 != n2:
                    key = (n1, n2)
                    node_node_groups.setdefault(key, []).append(edge_info["weight"])

        for (n1, n2), weights in node_node_groups.items():
            depends_on_edges[(n1, n2, "node_to_node")] = {
                "weight": max(weights) if weights else 0.01,
                "path_count": len(weights)
            }

        # -------------------------------------------------------------
        # Rule 4: node_to_broker
        # -------------------------------------------------------------
        node_broker_groups = {}
        for (src, tgt, dep_type), edge_info in depends_on_edges.items():
            if dep_type == "app_to_broker":
                n = comp_node.get(src)
                if n:
                    key = (n, tgt)
                    node_broker_groups.setdefault(key, []).append(edge_info["weight"])

        for (n, broker), weights in node_broker_groups.items():
            depends_on_edges[(n, broker, "node_to_broker")] = {
                "weight": max(weights) if weights else 0.01,
                "path_count": len(weights)
            }

        # -------------------------------------------------------------
        # Rule 5: app_to_lib (direct USES)
        # -------------------------------------------------------------
        for rel in self.data["relationships"].get("uses", []):
            src = rel["from"]
            tgt = rel["to"]
            if src in comp_lookup and tgt in comp_lookup:
                if comp_lookup[src]["type"] in ("Application", "Library") and comp_lookup[tgt]["type"] == "Library":
                    depends_on_edges[(src, tgt, "app_to_lib")] = {
                        "weight": 0.01,  # Placeholder, finalized later
                        "path_count": 1
                    }

        # -------------------------------------------------------------
        # Rule 6: broker_to_broker (shared Nodes)
        # -------------------------------------------------------------
        # Find brokers sharing any Node
        node_brokers = {}
        for rel in runs_on:
            src = rel["from"]
            tgt = rel["to"] # Node
            if src in comp_lookup and comp_lookup[src]["type"] == "Broker":
                node_brokers.setdefault(tgt, []).append(src)

        broker_broker_groups = {}
        for node_id, brokers in node_brokers.items():
            for b1 in brokers:
                for b2 in brokers:
                    if b1 != b2:
                        key = (b1, b2)
                        broker_broker_groups.setdefault(key, set()).add(node_id)

        for (b1, b2), shared_nodes in broker_broker_groups.items():
            depends_on_edges[(b1, b2, "broker_to_broker")] = {
                "weight": 0.01,  # Placeholder, finalized later
                "path_count": len(shared_nodes)
            }

        # Store derived depends_on edges
        self.data["relationships"]["depends_on"] = []
        for (src, tgt, dep_type), edge_info in depends_on_edges.items():
            self.data["relationships"]["depends_on"].append({
                "from": src,
                "to": tgt,
                "dependency_type": dep_type,
                "weight": edge_info["weight"],
                "path_count": edge_info["path_count"]
            })

    def _finalize_dependency_weights(self) -> None:
        """
        Finalise DEPENDS_ON edge weights based on component weights.
        """
        # Lookup component weights
        comp_weights = {}
        for key in ["nodes", "brokers", "topics", "applications", "libraries"]:
            for item in self.data.get(key, []):
                comp_weights[item["id"]] = item.get("weight", 0.01)

        # Helper to find shared nodes between Brokers
        runs_on = self.data["relationships"].get("runs_on", [])
        broker_nodes = {}
        for rel in runs_on:
            src = rel["from"]
            tgt = rel["to"] # Node
            if src in comp_weights:  # Is a broker (or app)
                broker_nodes.setdefault(src, set()).add(tgt)

        for d in self.data["relationships"].get("depends_on", []):
            dep_type = d["dependency_type"]
            src = d["from"]
            tgt = d["to"]

            if dep_type == "app_to_lib":
                # Inherits weight from Application/Library src
                d["weight"] = comp_weights.get(src, 0.01)
            elif dep_type == "broker_to_broker":
                # Inherits max weight from shared Nodes
                nodes1 = broker_nodes.get(src, set())
                nodes2 = broker_nodes.get(tgt, set())
                shared = nodes1 & nodes2
                shared_weights = [comp_weights.get(n, 0.01) for n in shared]
                d["weight"] = max(shared_weights) if shared_weights else 0.01

        # Safety assertion: no DEPENDS_ON edge should carry the 0.01 placeholder
        # after finalization.  Rules 1–4 set real weights from topic/component
        # data; Rules 5–6 are finalized just above.  A remaining 0.01 means a
        # code path reached here before _calculate_aggregate_weights() ran, or
        # a new rule was added without a finalization step.
        _PLACEHOLDER = 0.01
        _PLACEHOLDER_RULES = {"app_to_lib", "broker_to_broker"}
        stale_edges = [
            (d["from"], d["to"], d["dependency_type"])
            for d in self.data["relationships"].get("depends_on", [])
            if d.get("dependency_type") in _PLACEHOLDER_RULES
            and abs(d.get("weight", 0.01) - _PLACEHOLDER) < 1e-9
            and d.get("weight", 0.01) <= _PLACEHOLDER  # allow float rounding above floor
        ]
        # Note: broker_to_broker edges sharing *only* nodes with weight 0.01 are
        # legitimately 0.01 (the node floor), so we only raise when an app_to_lib
        # edge is still at 0.01 — that means the consuming app itself has the
        # placeholder weight, indicating _calculate_aggregate_weights() did not
        # set a real weight for that app.
        stale_app_to_lib = [
            e for e in stale_edges if e[2] == "app_to_lib"
        ]
        if stale_app_to_lib:
            raise ValueError(
                f"_finalize_dependency_weights: {len(stale_app_to_lib)} app_to_lib "
                f"edge(s) still carry the 0.01 placeholder weight after finalization. "
                f"This means _calculate_aggregate_weights() did not set a real weight "
                f"for the consuming application. Affected edges: {stale_app_to_lib[:5]}"
            )

    def save_graph(self, data: Dict[str, Any], clear: bool = False) -> None:
        """
        Save graph data to in-memory storage, performing normalization/flattening
        to simulate Neo4j persistence behavior.
        """
        if clear:
            self.data = {
                "metadata": {},
                "nodes": [], "brokers": [], "topics": [], "applications": [], "libraries": [],
                "relationships": {
                    "runs_on": [], "routes": [], "publishes_to": [],
                    "subscribes_to": [], "connects_to": [], "uses": [],
                    "depends_on": [],
                },
            }

        # 1. Normalize and store Metadata
        if "metadata" in data:
            self.data["metadata"] = serialization.flatten_metadata(data["metadata"])

        # 2. Normalize and store Components
        mapping = {
            "nodes": "Node", "brokers": "Broker", "topics": "Topic", 
            "applications": "Application", "libraries": "Library"
        }
        for key, comp_type in mapping.items():
            for item in data.get(key, []):
                flattened = serialization.flatten_component(item, comp_type)
                existing = next((x for x in self.data[key] if x["id"] == flattened["id"]), None)
                if existing:
                    existing.update(flattened)
                else:
                    self.data[key].append(flattened)
        
        # 3. Store Relationships (simple copy for memory)
        for key in ["runs_on", "routes", "publishes_to", "subscribes_to", "connects_to", "uses"]:
            items = data.get("relationships", {}).get(key, [])
            for item in items:
                rel = {
                    "from": item.get("from", item.get("source")),
                    "to": item.get("to", item.get("target")),
                    "weight": item.get("weight", 1.0)
                }
                self.data["relationships"][key].append(rel)

        # 4. Compute weights
        self._calculate_intrinsic_weights()
        self._calculate_aggregate_weights()

    def get_graph_data(
        self,
        component_types: Optional[List[str]] = None,
        dependency_types: Optional[List[str]] = None,
        include_raw: bool = False,
    ) -> GraphData:
        """Retrieve graph data with optional type filtering."""
        components = []
        for key in ["nodes", "brokers", "topics", "applications", "libraries"]:
            ctype = key[:-1].capitalize()
            if key == "libraries": ctype = "Library"
            
            if component_types and ctype not in component_types:
                continue
                
            for item in self.data.get(key, []):
                components.append(ComponentData(
                    id=item["id"],
                    component_type=ctype,
                    weight=item.get("weight", 1.0),
                    properties={k: v for k, v in item.items() if k not in ["id", "weight"]}
                ))

        id_to_type = {c.id: c.component_type for c in components}
        edges = []

        # 1. Fetch DEPENDS_ON edges
        for item in self.data["relationships"].get("depends_on", []):
            dep_type = item.get("dependency_type")
            if dependency_types and dep_type not in dependency_types:
                continue
            
            src_id = item["from"]
            tgt_id = item["to"]
            
            edges.append(EdgeData(
                source_id=src_id,
                target_id=tgt_id,
                source_type=id_to_type.get(src_id, "Unknown"),
                target_type=id_to_type.get(tgt_id, "Unknown"),
                dependency_type=dep_type,
                relation_type="DEPENDS_ON",
                weight=item.get("weight", 1.0),
                path_count=item.get("path_count", 1),
                properties={k: v for k, v in item.items() if k not in ["from", "to", "weight", "dependency_type", "path_count"]}
            ))

        # 2. Optionally include raw structural edges
        if include_raw:
            for rel_type, items in self.data.get("relationships", {}).items():
                if rel_type == "depends_on":
                    continue
                for item in items:
                    src_id = item.get("from", item.get("source"))
                    tgt_id = item.get("to", item.get("target"))
                    edges.append(EdgeData(
                        source_id=src_id,
                        target_id=tgt_id,
                        source_type=id_to_type.get(src_id, "Unknown"),
                        target_type=id_to_type.get(tgt_id, "Unknown"),
                        dependency_type=rel_type.lower(),
                        relation_type=rel_type.upper(),
                        weight=item.get("weight", 1.0),
                        path_count=1,
                        properties={k: v for k, v in item.items() if k not in ["from", "to", "source", "target", "weight"]}
                    ))
        
        return GraphData(components=components, edges=edges)

    def get_layer_data(self, layer: str) -> GraphData:
        """Retrieve graph data for a specific layer project."""
        from saag.core.layers import get_layer_definition, AnalysisLayer, _resolve_layer
        canonical = _resolve_layer(layer)
        defn = get_layer_definition(AnalysisLayer.from_string(canonical))
        return self.get_graph_data(
            component_types=defn.component_types,
            dependency_types=defn.dependency_types
        )

    def get_statistics(self) -> Dict[str, int]:
        """Retrieve graph statistics."""
        stats = {}
        total_nodes = 0
        for key in ["nodes", "brokers", "topics", "applications", "libraries"]:
            count = len(self.data.get(key, []))
            stats[f"{key[:-1]}_count"] = count
            total_nodes += count
        
        total_rels = 0
        for key in self.data.get("relationships", {}):
            total_rels += len(self.data["relationships"][key])
            
        stats["total_nodes"] = total_nodes
        stats["total_relationships"] = total_rels
        return stats

    def export_json(self) -> Dict[str, Any]:
        """
        Export graph as JSON (compatible with data generation format).
        Consolidated via get_graph_data to ensure logic consistency and fidelity parity
        with the Neo4j persistence layer.
        """
        graph_data = self.get_graph_data(include_raw=True)
        return serialization.reconstruct_export_payload(graph_data, self.data["metadata"])
