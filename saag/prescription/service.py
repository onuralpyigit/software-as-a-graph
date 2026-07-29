"""
saag/prescription/service.py
"""
import logging
from typing import Any, Dict, Optional, Sequence, Tuple

from saag.core.ports.graph_repository import IGraphRepository

from .evaluator import GraphEvaluator, repo_from_json
from .models import PrescribeResult, PrescriptionPolicy
from .mutator import apply_policy
from .rules import compile_policy
from .verifier import EditVerifier

logger = logging.getLogger(__name__)


class PrescribeService:
    """
    Service for generating prescriptive architectural modifications (Stage 6)
    and validating them in a closed-loop simulation.
    """

    def __init__(self, repository: IGraphRepository):
        self.repository = repository

    def compile_policy(
        self,
        analysis_result: Any,
        prediction_result: Optional[Any] = None,
    ) -> PrescriptionPolicy:
        """Compile the candidate policy Delta(G) without verifying or applying it.

        Layer-independent: the rules read the raw graph, since a reallocation or
        a topic split has to be expressed over the physical topology whatever
        layer the risk was scored on.
        """
        return compile_policy(
            self.repository.get_graph_data(include_raw=True),
            analysis_result,
            prediction_result,
        )

    def prescribe(
        self,
        analysis_result: Any,
        prediction_result: Optional[Any] = None,
        layer: str = "system",
        gnn_checkpoint: Optional[str] = None,
        kappa: float = 1.0,
        seeds: Optional[Sequence[int]] = None,
        thresholds: Optional[Sequence[float]] = None,
    ) -> PrescribeResult:
        """
        Generate optimization recommendations for components flagged as CRITICAL/HIGH,
        verify each one independently, apply only those that pass, and re-check
        resilience end to end.

        Parameters
        ----------
        kappa:
            Acceptance multiple. An edit is kept only when its mean impact
            reduction exceeds ``kappa * sigma_seed`` at every threshold — it must
            beat the simulator's own seed noise, not merely register a positive
            delta.
        seeds:
            Simulation seeds used to estimate that noise. Defaults to
            ``verifier.DEFAULT_SEEDS``.
        thresholds:
            Propagation thresholds the edit must clear at *every* value, so an
            accepted edit is not an artifact of the 0.2 default. Defaults to
            ``verifier.DEFAULT_THRESHOLDS``.
        """
        logger.info(f"Starting Prescriptive Optimization for layer '{layer}'...")

        # 1. Compile the transformation policy Delta(G) — this is the *candidate* set.
        candidate_policy = self.compile_policy(analysis_result, prediction_result)
        logger.info(
            f"Generated candidate Delta(G): {len(candidate_policy.topic_splits)} splits, "
            f"{len(candidate_policy.node_reallocations)} reallocations, "
            f"{len(candidate_policy.qos_upgrades)} upgrades."
        )

        # 2. Baseline system health and resilience.
        logger.info("Evaluating baseline system health and resilience...")
        evaluator = GraphEvaluator(layer, gnn_checkpoint)
        baseline = evaluator.evaluate(self.repository)
        original_json = self.repository.export_json()

        # 3. Per-edit acceptance filter — each candidate is simulated alone.
        verifier = EditVerifier(evaluator, kappa=kappa, seeds=seeds, thresholds=thresholds)
        verdicts = verifier.verify(self.repository, original_json, candidate_policy)
        policy = PrescriptionPolicy.from_edits([
            edit for edit, verdict in zip(candidate_policy.edits(), verdicts)
            if verdict.accepted
        ])
        logger.info(
            "Per-edit filter: %d/%d edits accepted (kappa=%.2f).",
            len(policy.edits()), len(verdicts), kappa,
        )

        if policy.is_empty():
            # Nothing cleared the bar. Report that plainly rather than running a
            # no-op mutation and presenting an unchanged SRI as a result.
            logger.info("No candidate edit passed the acceptance filter; leaving the graph unchanged.")
            return PrescribeResult(
                original_sri=baseline.sri,
                mutated_sri=baseline.sri,
                sri_improvement=0.0,
                original_metrics=baseline.metrics,
                mutated_metrics=baseline.metrics,
                policy=policy,
                applied_changes=[],
                accepted=False,
                edit_verdicts=verdicts,
                candidate_policy=candidate_policy,
            )

        # 4. Closed-loop validation of the accepted subset.
        logger.info("Evaluating mutated system health and resilience (closed-loop simulation)...")
        mutated = evaluator.evaluate(repo_from_json(apply_policy(original_json, policy)))

        impact_deltas, mean_reduction = self._impact_deltas(
            policy, baseline.impact, mutated.impact)

        sri_improvement = round(baseline.sri - mutated.sri, 4)
        accepted = sri_improvement > 0
        logger.info(
            f"Closed-loop validation complete. SRI changed from {baseline.sri} to {mutated.sri} "
            f"({'ACCEPTED' if accepted else 'REJECTED'}, Improvement: {sri_improvement})."
        )

        return PrescribeResult(
            original_sri=baseline.sri,
            mutated_sri=mutated.sri,
            sri_improvement=sri_improvement,
            original_metrics=baseline.metrics,
            mutated_metrics=mutated.metrics,
            policy=policy,
            applied_changes=[edit.describe() for edit in policy.edits()],
            remediated_component_impact_deltas=impact_deltas,
            mean_cascade_impact_reduction=mean_reduction,
            accepted=accepted,
            edit_verdicts=verdicts,
            candidate_policy=candidate_policy,
        )

    @staticmethod
    def _impact_deltas(
        policy: PrescriptionPolicy,
        before: Dict[str, float],
        after: Dict[str, float],
    ) -> Tuple[Dict[str, Dict[str, float]], Optional[float]]:
        """Per-component cascade-impact reduction over the remediated components.

        Restricted to components whose id survives the mutation (reallocations,
        QoS upgrades). Topic splits replace the original topic id with
        per-publisher sub-topics and so have no before/after counterpart; they
        are excluded rather than approximated.
        """
        remediated = (
            {r.component for r in policy.node_reallocations}
            | {u.topic for u in policy.qos_upgrades}
        )

        deltas: Dict[str, Dict[str, float]] = {}
        for cid in remediated:
            i_before = before.get(cid)
            i_after = after.get(cid)
            if i_before is None or i_after is None:
                continue
            deltas[cid] = {"before": i_before, "after": i_after}
            if i_before > 0:
                deltas[cid]["reduction_frac"] = (i_before - i_after) / i_before

        reductions = [d["reduction_frac"] for d in deltas.values() if "reduction_frac" in d]
        mean_reduction = sum(reductions) / len(reductions) if reductions else None
        return deltas, mean_reduction
