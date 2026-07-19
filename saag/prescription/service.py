"""
saag/prescription/service.py
"""
import copy
import logging
from typing import Dict, Any, List, Optional

from saag.core.ports.graph_repository import IGraphRepository
from saag.simulation.service import SimulationService
from saag.validation.service import ValidationService
from saag.analysis.service import AnalysisService
from saag.prediction.service import PredictionService

from .models import PrescriptionPolicy, PrescribeResult

logger = logging.getLogger(__name__)

class PrescribeService:
    """
    Service for generating prescriptive architectural modifications (Stage 6)
    and validating them in a closed-loop simulation.
    """

    def __init__(self, repository: IGraphRepository, simulation_service: Optional[SimulationService] = None):
        self.repository = repository
        self.simulation_service = simulation_service or SimulationService(repository)

    def prescribe(
        self, 
        analysis_result: Any, 
        prediction_result: Optional[Any] = None, 
        layer: str = "system",
        gnn_checkpoint: Optional[str] = None
    ) -> PrescribeResult:
        """
        Generate optimization recommendations for components flagged as CRITICAL/HIGH,
        apply them to a mutated version of the graph, and verify resilience improvements.
        """
        logger.info(f"Starting Prescriptive Optimization for layer '{layer}'...")
        
        # 1. Compile transformation policy Delta(G)
        policy = self.compile_policy(analysis_result, prediction_result, layer)
        logger.info(f"Generated policy Delta(G): {len(policy.topic_splits)} splits, {len(policy.node_reallocations)} reallocations, {len(policy.qos_upgrades)} upgrades.")

        # 2. Get baseline simulation metrics
        # If original analysis has simulation metrics we can extract them,
        # but to be sure we have complete validation metrics, we calculate them.
        logger.info("Evaluating baseline system health and resilience...")
        baseline_sri, baseline_metrics, baseline_impact = self._evaluate_baseline(analysis_result, layer, gnn_checkpoint)

        # 3. Create mutated graph JSON
        original_json = self.repository.export_json()
        mutated_json = self._apply_policy_mutations(original_json, policy)

        # 4. Closed-loop validation on the mutated graph
        logger.info("Evaluating mutated system health and resilience (closed-loop simulation)...")
        mutated_sri, mutated_metrics, applied_changes, mutated_impact = self._evaluate_mutation(mutated_json, layer, policy, gnn_checkpoint)

        # 5. Per-component cascade-impact reduction (§6.7): restricted to remediated components
        # whose id is stable across the mutation (node reallocations, QoS upgrades). Topic splits
        # replace the original topic id with per-publisher sub-topics and have no stable
        # before/after counterpart, so they are excluded from this figure rather than approximated.
        remediated_ids = {r["component"] for r in policy.node_reallocations} | {u["topic"] for u in policy.qos_upgrades}
        impact_deltas: Dict[str, Dict[str, float]] = {}
        for cid in remediated_ids:
            i_before = baseline_impact.get(cid)
            i_after = mutated_impact.get(cid)
            if i_before is None or i_after is None:
                continue
            impact_deltas[cid] = {"before": i_before, "after": i_after}
            if i_before > 0:
                impact_deltas[cid]["reduction_frac"] = (i_before - i_after) / i_before

        reductions = [d["reduction_frac"] for d in impact_deltas.values() if "reduction_frac" in d]
        mean_reduction = sum(reductions) / len(reductions) if reductions else None

        # 6. Compile final results
        sri_improvement = round(baseline_sri - mutated_sri, 4)
        accepted = sri_improvement > 0
        logger.info(
            f"Closed-loop validation complete. SRI changed from {baseline_sri} to {mutated_sri} "
            f"({'ACCEPTED' if accepted else 'REJECTED'}, Improvement: {sri_improvement})."
        )

        return PrescribeResult(
            original_sri=baseline_sri,
            mutated_sri=mutated_sri,
            sri_improvement=sri_improvement,
            original_metrics=baseline_metrics,
            mutated_metrics=mutated_metrics,
            policy=policy,
            applied_changes=applied_changes,
            remediated_component_impact_deltas=impact_deltas,
            mean_cascade_impact_reduction=mean_reduction,
            accepted=accepted,
        )

    def compile_policy(
        self, 
        analysis_result: Any, 
        prediction_result: Optional[Any] = None, 
        layer: str = "system"
    ) -> PrescriptionPolicy:
        """
        Evaluate structural anomalies and predicted risks to build an optimization policy.
        """
        policy = PrescriptionPolicy()
        
        # Identify critical/high risk components
        critical_components = set()
        
        # Extract from GNN predictions if available
        if prediction_result:
            all_comps = getattr(prediction_result, "all_components", [])
            if not all_comps and isinstance(prediction_result, dict):
                for comp_id, comp_info in prediction_result.get("node_scores", {}).items():
                    level = comp_info.get("criticality_level", "MINIMAL").upper()
                    if level in ("CRITICAL", "HIGH"):
                        critical_components.add(comp_id)
            elif not all_comps and hasattr(prediction_result, "raw"):
                # Handle GNN prediction facade
                raw_pred = prediction_result.raw
                node_scores = getattr(raw_pred, "node_scores", {}) or getattr(raw_pred, "ensemble_scores", {})
                if isinstance(node_scores, dict):
                    for comp_id, comp_info in node_scores.items():
                        level = getattr(comp_info, "criticality_level", "MINIMAL").upper()
                        if level in ("CRITICAL", "HIGH"):
                            critical_components.add(comp_id)
            else:
                for comp in all_comps:
                    if comp.criticality_level.upper() in ("CRITICAL", "HIGH"):
                        critical_components.add(comp.id)
                        
        # Extract from structural analysis results
        # Handle both raw LayerAnalysisResult and SDK facade AnalysisResult.
        # analysis_result.quality is now always None (Analyze/Step 2 is
        # structural-only) — RMAV components come from the Predict stage
        # (prediction_result), so fall back to its raw QualityAnalysisResult.
        quality_comps = []
        analysis_quality = getattr(analysis_result, "quality", None)
        if analysis_quality is None:
            raw_prediction = getattr(prediction_result, "raw", prediction_result)
            if hasattr(raw_prediction, "components"):
                analysis_quality = raw_prediction
        if analysis_quality is not None and hasattr(analysis_quality, "components"):
            quality_comps = analysis_quality.components
        elif hasattr(analysis_result, "all_components"):
            for comp in analysis_result.all_components:
                # Check overall
                if comp.criticality_level.upper() in ("CRITICAL", "HIGH"):
                    critical_components.add(comp.id)
                # Check individual dimensions
                levels = getattr(comp, "criticality_levels", getattr(comp, "levels", {}))
                for dim, lvl in levels.items():
                    if str(lvl).upper() in ("CRITICAL", "HIGH"):
                        critical_components.add(comp.id)
                    
        for cq in quality_comps:
            # Check all dimensions: reliability, maintainability, availability, security, overall
            for dim in ("reliability", "maintainability", "availability", "security", "overall"):
                level_enum = getattr(cq.levels, dim, None)
                if level_enum:
                    level_str = getattr(level_enum, "name", str(level_enum)).upper()
                    if level_str in ("CRITICAL", "HIGH"):
                        critical_components.add(cq.id)
                
        # Identify problems (smells) — now produced by the Predict stage, not Analyze
        problems = getattr(prediction_result, "problems", None) or getattr(analysis_result, "problems", []) or []
        spof_components = set()
        god_components = set()
        
        for prob in problems:
            pid = getattr(prob, "name", "") or prob.get("name", "")
            entity_id = getattr(prob, "entity_id", "") or prob.get("entity_id", "")
            
            if "SPOF" in pid or "Single Point of Failure" in pid:
                spof_components.add(entity_id)
            if "God Component" in pid or "Bottleneck" in pid or "Hub" in pid:
                god_components.add(entity_id)

        # Retrieve raw graph data for connection lookups
        graph_data = self.repository.get_graph_data(include_raw=True)
        
        topic_publishers = {}
        topic_subscribers = {}
        node_hosted = {}
        
        for edge in graph_data.edges:
            src = edge.source_id
            tgt = edge.target_id
            rel = edge.relation_type
            
            if rel == "PUBLISHES_TO":
                topic_publishers.setdefault(tgt, []).append(src)
            elif rel == "SUBSCRIBES_TO":
                topic_subscribers.setdefault(tgt, []).append(src)
            elif rel == "RUNS_ON":
                node_hosted.setdefault(tgt, []).append(src)

        # --- Rule 1: Logical Subgraph Refactoring (Topic Splitting) ---
        for comp in graph_data.components:
            if comp.component_type == "Topic":
                tid = comp.id
                pubs = topic_publishers.get(tid, [])
                subs = topic_subscribers.get(tid, [])
                
                is_congested = len(pubs) > 1 and len(subs) > 1
                is_crit_multi = tid in critical_components and len(pubs) > 1
                connected_to_god = any(p in god_components or s in god_components for p in pubs for s in subs) and len(pubs) > 1
                
                if is_congested or is_crit_multi or connected_to_god:
                    policy.topic_splits.append({
                        "topic": tid,
                        "publishers": sorted(list(set(pubs))),
                        "subscribers": sorted(list(set(subs)))
                    })
                    
        # --- Rule 2: Physical Locality Anti-Affinity Rules ---
        for comp in graph_data.components:
            if comp.component_type == "Node":
                nid = comp.id
                hosted = node_hosted.get(nid, [])
                
                # If the host or any hosted component is a SPOF/critical, and it co-locates multiple processes
                is_spof_or_crit = (
                    nid in spof_components or
                    nid in critical_components or
                    any(cid in spof_components or cid in critical_components for cid in hosted)
                )
                if is_spof_or_crit and len(hosted) > 1:
                    # Move all components except the first one to new separate nodes
                    for idx, cid in enumerate(sorted(hosted)):
                        if idx == 0:
                            continue
                        policy.node_reallocations.append({
                            "component": cid,
                            "from_node": nid,
                            "to_node": f"{nid}_{cid}"
                        })
                        
        # --- Rule 3: Middleware Transport Contract Hardening ---
        for comp in graph_data.components:
            if comp.component_type == "Topic":
                tid = comp.id
                pubs = topic_publishers.get(tid, [])
                subs = topic_subscribers.get(tid, [])
                
                # Hardening critical channel QoS properties
                is_crit_channel = tid in critical_components or any(p in critical_components or s in critical_components for p in pubs for s in subs)
                
                props = comp.properties
                reliability = props.get("qos_reliability", "BEST_EFFORT")
                durability = props.get("qos_durability", "VOLATILE")
                
                if is_crit_channel and (reliability != "RELIABLE" or durability == "VOLATILE"):
                    policy.qos_upgrades.append({
                        "topic": tid,
                        "original_reliability": reliability,
                        "original_durability": durability,
                        "target_reliability": "RELIABLE",
                        "target_durability": "TRANSIENT"
                    })
                    
        return policy

    def _evaluate_baseline(self, analysis_result: Any, layer: str, gnn_checkpoint: Optional[str]) -> tuple[float, Dict[str, Any], Dict[str, float]]:
        """Calculate baseline SRI, verification metrics, and per-component I(v)."""
        try:
            # Try to get validation summary if already run in the pipeline
            # Wait, validate service is always on client
            from saag.client import Client
            client = Client(repo=self.repository)
            val_res = client.validate(layers=[layer], gnn_checkpoint=gnn_checkpoint)

            layer_val = val_res.layers.get(layer)
            if layer_val:
                # If we have validation facade, run evaluation directly to get metrics cleanly
                return self._run_evaluation(self.repository, layer, gnn_checkpoint)
        except Exception as e:
            logger.warning(f"Could not load pre-run validation, running manual evaluation: {e}")

        # Fallback: Run validation service manually
        return self._run_evaluation(self.repository, layer, gnn_checkpoint)

    def _evaluate_mutation(
        self,
        mutated_json: Dict[str, Any],
        layer: str,
        policy: PrescriptionPolicy,
        gnn_checkpoint: Optional[str]
    ) -> tuple[float, Dict[str, Any], List[str], Dict[str, float]]:
        """Run verification on the mutated graph."""
        from saag.infrastructure.memory_repo import MemoryRepository

        # Instantiate temporary in-memory repo containing mutated graph G'
        temp_repo = MemoryRepository()
        temp_repo.save_graph(mutated_json, clear=True)
        temp_repo.derive_dependencies()

        # Run closed-loop validation pipeline on the temporary repository
        sri, metrics, impact_by_component = self._run_evaluation(temp_repo, layer, gnn_checkpoint)

        # Track applied changes for the report
        applied_changes = []
        for split in policy.topic_splits:
            applied_changes.append(f"Split topic '{split['topic']}' into sub-topics per publisher: {', '.join(split['publishers'])}")
        for realloc in policy.node_reallocations:
            applied_changes.append(f"Moved process '{realloc['component']}' from SPOF node '{realloc['from_node']}' to isolated node '{realloc['to_node']}'")
        for upgrade in policy.qos_upgrades:
            applied_changes.append(f"Hardened QoS on topic '{upgrade['topic']}': Reliability -> {upgrade['target_reliability']}, Durability -> {upgrade['target_durability']}")

        return sri, metrics, applied_changes, impact_by_component

    def _run_evaluation(self, repo: IGraphRepository, layer: str, gnn_checkpoint: Optional[str]) -> tuple[float, Dict[str, Any], Dict[str, float]]:
        """Run full analysis, simulation, and validation to retrieve SRI, key metrics, and per-component I(v)."""
        from saag.client import Client
        client = Client(repo=repo)

        # Stage 2: Analyze
        analysis = client.analyze(layer=layer)

        # Stage 4: Simulate — canonical settings (propagation_threshold=0.2, §7.5); FailureResult
        # exposes both aggregate impact terms and the full I(v) = composite_impact per component.
        sim_results = client.simulate(layer=layer, propagation_threshold=0.2)
        impact_by_component = {r.target_id: r.impact.composite_impact for r in sim_results}

        # Stage 5: Validate (which internally triggers simulation and prediction comparison)
        validation_facade = client.validate(layers=[layer], gnn_checkpoint=gnn_checkpoint)
        layer_val = validation_facade.layers[layer]

        sri = layer_val.raw.system_health.get("SRI", 1.0)

        # Calculate averages from simulation results
        avg_reachability = sum(r.impact.reachability_loss for r in sim_results) / len(sim_results) if sim_results else 0.0
        avg_fragmentation = sum(r.impact.fragmentation for r in sim_results) / len(sim_results) if sim_results else 0.0
        avg_throughput = sum(r.impact.throughput_loss for r in sim_results) / len(sim_results) if sim_results else 0.0

        metrics = {
            "sri": sri,
            "avg_reachability_loss": avg_reachability,
            "avg_fragmentation": avg_fragmentation,
            "avg_throughput_loss": avg_throughput,
        }
        return sri, metrics, impact_by_component

    def _apply_policy_mutations(self, original_json: Dict[str, Any], policy: PrescriptionPolicy) -> Dict[str, Any]:
        """Apply Delta(G) modifications to the JSON topology export."""
        mutated = copy.deepcopy(original_json)
        
        # 1. Transport Contract Hardening (QoS upgrades)
        upgrades = {u["topic"]: u for u in policy.qos_upgrades}
        for topic in mutated.get("topics", []):
            tid = topic["id"]
            if tid in upgrades:
                qos = topic.setdefault("qos", {})
                qos["reliability"] = "RELIABLE"
                qos["durability"] = "TRANSIENT"
                
                # Update flat properties
                topic["qos_reliability"] = "RELIABLE"
                topic["qos_durability"] = "TRANSIENT"
                
        # 2. Node Reallocation (Anti-affinity isolation)
        reallocations = {r["component"]: r for r in policy.node_reallocations}
        new_nodes_to_add = {}
        
        runs_on = mutated.get("relationships", {}).get("runs_on", [])
        new_runs_on = []
        for rel in runs_on:
            comp_id = rel.get("from")
            node_id = rel.get("to")
            
            if comp_id in reallocations:
                realloc = reallocations[comp_id]
                target_node = realloc["to_node"]
                new_runs_on.append({
                    "from": comp_id,
                    "to": target_node,
                    "weight": rel.get("weight", 1.0)
                })
                new_nodes_to_add[target_node] = realloc["from_node"]
            else:
                new_runs_on.append(rel)
        mutated["relationships"]["runs_on"] = new_runs_on
        
        # Create new nodes
        for new_node_id, orig_node_id in new_nodes_to_add.items():
            orig_node = next((n for n in mutated.get("nodes", []) if n["id"] == orig_node_id), None)
            if orig_node:
                new_node = copy.deepcopy(orig_node)
                new_node["id"] = new_node_id
                new_node["name"] = f"Node {new_node_id}"
                mutated["nodes"].append(new_node)
                
        # Duplicate connects_to relationships
        connects_to = mutated.get("relationships", {}).get("connects_to", [])
        new_connects = []
        for conn in connects_to:
            src = conn.get("from")
            tgt = conn.get("to")
            new_connects.append(conn)
            
            for new_node_id, orig_node_id in new_nodes_to_add.items():
                if src == orig_node_id:
                    new_connects.append({
                        "from": new_node_id,
                        "to": tgt,
                        "weight": conn.get("weight", 1.0)
                    })
                if tgt == orig_node_id:
                    new_connects.append({
                        "from": src,
                        "to": new_node_id,
                        "weight": conn.get("weight", 1.0)
                    })
        mutated["relationships"]["connects_to"] = new_connects

        # 3. Logical Subgraph Refactoring (Topic Splitting)
        splits = {s["topic"]: s for s in policy.topic_splits}
        topics_to_remove = set()
        new_topics = []
        
        for topic in mutated.get("topics", []):
            tid = topic["id"]
            if tid in splits:
                topics_to_remove.add(tid)
                publishers = splits[tid]["publishers"]
                for pub in publishers:
                    sub_tid = f"{tid}_{pub}"
                    sub_topic = copy.deepcopy(topic)
                    sub_topic["id"] = sub_tid
                    sub_topic["name"] = f"{topic.get('name', tid)} for {pub}"
                    new_topics.append(sub_topic)
            else:
                new_topics.append(topic)
        mutated["topics"] = new_topics
        
        # publishes_to: P -> T becomes P -> T_P
        publishes_to = mutated.get("relationships", {}).get("publishes_to", [])
        new_publishes = []
        for rel in publishes_to:
            pub = rel.get("from")
            topic_id = rel.get("to")
            if topic_id in splits:
                new_publishes.append({
                    "from": pub,
                    "to": f"{topic_id}_{pub}",
                    "weight": rel.get("weight", 1.0)
                })
            else:
                new_publishes.append(rel)
        mutated["relationships"]["publishes_to"] = new_publishes
        
        # subscribes_to: S -> T becomes S -> T_P for all P
        subscribes_to = mutated.get("relationships", {}).get("subscribes_to", [])
        new_subscribes = []
        for rel in subscribes_to:
            sub = rel.get("from")
            topic_id = rel.get("to")
            if topic_id in splits:
                publishers = splits[topic_id]["publishers"]
                for pub in publishers:
                    new_subscribes.append({
                        "from": sub,
                        "to": f"{topic_id}_{pub}",
                        "weight": rel.get("weight", 1.0)
                    })
            else:
                new_subscribes.append(rel)
        mutated["relationships"]["subscribes_to"] = new_subscribes
        
        # routes: Broker -> T becomes Broker -> T_P for all P
        routes = mutated.get("relationships", {}).get("routes", [])
        new_routes = []
        for rel in routes:
            broker = rel.get("from")
            topic_id = rel.get("to")
            if topic_id in splits:
                publishers = splits[topic_id]["publishers"]
                for pub in publishers:
                    new_routes.append({
                        "from": broker,
                        "to": f"{topic_id}_{pub}",
                        "weight": rel.get("weight", 1.0)
                    })
            else:
                new_routes.append(rel)
        mutated["relationships"]["routes"] = new_routes
        
        return mutated
