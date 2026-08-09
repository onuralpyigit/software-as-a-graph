"""
saag/prescription/verifier.py

The per-edit acceptance filter. Every candidate edit is simulated on its own
counterfactual graph and kept only if it beats the simulator's own noise at every
propagation threshold. Applying a whole policy unconditionally is what previously
let regressing edits ride along with improving ones.
"""
import logging
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .evaluator import GraphEvaluator, repo_from_json
from .models import EditVerdict, PrescriptionPolicy, ThresholdStat
from .mutator import apply_policy

logger = logging.getLogger(__name__)

#: Propagation thresholds an edit must clear at every value, so that acceptance
#: cannot be an artifact of the canonical 0.2 default.
DEFAULT_THRESHOLDS = (0.1, 0.2, 0.5)

#: Seeds used to estimate the simulator's own noise on a delta. Three rather
#: than five: verification costs one exhaustive sweep per (edit x threshold x
#: seed), and because baseline and mutated runs are paired on (threshold, seed)
#: the shared noise largely cancels, leaving a spread a third seed already pins
#: well enough for a bar that edits clear by orders of magnitude or not at all.
DEFAULT_SEEDS = (42, 123, 456)


def mean_reduction(
    baseline_impact: Dict[str, float],
    mutated_impact: Dict[str, float],
) -> float:
    """Mean reduction in cascade impact over components common to both graphs.

    Positive means the edit lowered impact. Restricted to ids present before
    *and* after, because a topic split renames its target and so has no stable
    counterpart to difference against.
    """
    common = set(baseline_impact) & set(mutated_impact)
    if not common:
        return 0.0
    return statistics.fmean(
        baseline_impact[cid] - mutated_impact[cid] for cid in common
    )


def judge(
    kind: str,
    target: str,
    per_threshold: Dict[str, ThresholdStat],
    kappa: float,
    n_thresholds: int,
    failure_reason: str = "",
) -> EditVerdict:
    """Decide one edit's fate from its completed sweep.

    An edit is accepted iff, for **every** threshold,
    ``mean_delta > kappa * sigma_seed``. Requiring it everywhere stops an edit
    being accepted because it happened to help at one threshold; requiring it to
    beat sigma stops simulator noise being read as improvement. With a single
    seed sigma is 0 and the rule degenerates to a strictly positive mean.
    """
    if failure_reason or not per_threshold or len(per_threshold) != n_thresholds:
        return EditVerdict(
            kind=kind, target=target, kappa=kappa, accepted=False,
            reason=failure_reason or "incomplete_sweep",
            per_threshold=per_threshold,
        )

    verdict = EditVerdict(
        kind=kind, target=target, kappa=kappa, per_threshold=per_threshold)
    binding = min(per_threshold, key=lambda t: per_threshold[t].margin(kappa))
    stat = per_threshold[binding]
    verdict.accepted = stat.margin(kappa) > 0
    if not verdict.accepted:
        verdict.reason = (
            f"delta {stat.mean_delta:.6f} <= kappa*sigma "
            f"{kappa * stat.sigma_seed:.6f} at threshold {binding}"
        )
    return verdict


import concurrent.futures
import os


def _verify_single_edit_worker(
    args_tuple: Tuple[
        Any, Dict[str, Any], float, List[float], List[int],
        Dict[Tuple[float, int], Dict[str, float]], str, Optional[str],
    ]
) -> EditVerdict:
    """Top-level worker function for parallel candidate edit evaluation across CPU cores."""
    edit, original_json, kappa, thresholds, seeds, baselines, layer, gnn_checkpoint = args_tuple
    single = PrescriptionPolicy.from_edits([edit])
    failure_reason = ""
    per_threshold: Dict[str, ThresholdStat] = {}
    # Must match the layer/checkpoint the baselines were measured with
    # (EditVerifier.evaluator) — a bare GraphEvaluator() defaults to
    # layer="system", silently evaluating every candidate on the wrong
    # layer whenever prescribe runs with --layer != system.
    evaluator = GraphEvaluator(layer=layer, gnn_checkpoint=gnn_checkpoint)
    try:
        repo = repo_from_json(apply_policy(original_json, single))
        for threshold in thresholds:
            deltas = [
                mean_reduction(
                    baselines[(threshold, seed)],
                    evaluator.impact(repo, threshold=threshold, seed=seed),
                )
                for seed in seeds
            ]
            per_threshold[str(threshold)] = ThresholdStat(
                mean_delta=statistics.fmean(deltas),
                sigma_seed=statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            )
    except Exception as exc:
        logger.warning("Edit %s/%s failed to simulate in worker: %s", getattr(edit, 'KIND', 'UNKNOWN'), getattr(edit, 'target', 'UNKNOWN'), exc)
        failure_reason = f"simulation_error: {exc}"

    return judge(
        edit.KIND, edit.target, per_threshold,
        kappa, len(thresholds), failure_reason,
    )


class EditVerifier:
    """Runs the counterfactual sweep behind the acceptance filter."""

    def __init__(
        self,
        evaluator: GraphEvaluator,
        kappa: float = 1.0,
        seeds: Optional[Sequence[int]] = None,
        thresholds: Optional[Sequence[float]] = None,
    ):
        self.evaluator = evaluator
        self.kappa = kappa
        self.seeds = list(seeds or DEFAULT_SEEDS)
        self.thresholds = list(thresholds or DEFAULT_THRESHOLDS)

    def verify(
        self,
        base_repo: Any,
        original_json: Dict[str, Any],
        candidate_policy: PrescriptionPolicy,
        n_jobs: Optional[int] = -1,
    ) -> List[EditVerdict]:
        """Return one verdict per candidate edit, in ``candidate_policy.edits()`` order."""
        edits = candidate_policy.edits()
        if not edits:
            return []

        baselines = self._baselines(base_repo)
        
        # Determine CPU parallelism
        num_workers = n_jobs if (n_jobs is not None and n_jobs > 0) else (os.cpu_count() or 1)
        if n_jobs == 1 or len(edits) <= 1 or num_workers <= 1:
            # Serial execution path
            verdicts: List[EditVerdict] = []
            for edit in edits:
                single = PrescriptionPolicy.from_edits([edit])
                failure_reason = ""
                per_threshold: Dict[str, ThresholdStat] = {}
                try:
                    repo = repo_from_json(apply_policy(original_json, single))
                    per_threshold = self._sweep(repo, baselines)
                except Exception as exc:  # noqa: BLE001 - one bad edit must not abort the sweep
                    logger.warning("Edit %s/%s failed to simulate: %s", edit.KIND, edit.target, exc)
                    failure_reason = f"simulation_error: {exc}"

                verdicts.append(judge(
                    edit.KIND, edit.target, per_threshold,
                    self.kappa, len(self.thresholds), failure_reason,
                ))
            return verdicts

        # Parallel execution path using ProcessPoolExecutor
        tasks = [
            (
                edit, original_json, self.kappa, self.thresholds, self.seeds, baselines,
                self.evaluator.layer, self.evaluator.gnn_checkpoint,
            )
            for edit in edits
        ]
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                verdicts = list(executor.map(_verify_single_edit_worker, tasks))
            return verdicts
        except Exception as exc:
            logger.warning("ProcessPoolExecutor failed (%s); falling back to serial verification.", exc)
            # Fallback to serial execution if process pool allocation fails
            verdicts = []
            for edit in edits:
                single = PrescriptionPolicy.from_edits([edit])
                failure_reason = ""
                per_threshold = {}
                try:
                    repo = repo_from_json(apply_policy(original_json, single))
                    per_threshold = self._sweep(repo, baselines)
                except Exception as exc_inner:
                    failure_reason = f"simulation_error: {exc_inner}"

                verdicts.append(judge(
                    edit.KIND, edit.target, per_threshold,
                    self.kappa, len(self.thresholds), failure_reason,
                ))
            return verdicts

    def _baselines(self, base_repo: Any) -> Dict[Tuple[float, int], Dict[str, float]]:
        """Unmutated impact at every point of the sweep grid.

        Measured once and shared by every edit. Each mutated run is differenced
        against the baseline at its *own* (threshold, seed): differencing
        against a single fixed point would fold the threshold's own effect into
        every delta, swamping the effect of the edit being judged.
        """
        return {
            (threshold, seed): self.evaluator.impact(base_repo, threshold=threshold, seed=seed)
            for threshold in self.thresholds
            for seed in self.seeds
        }

    def _sweep(
        self,
        repo: Any,
        baselines: Dict[Tuple[float, int], Dict[str, float]],
    ) -> Dict[str, ThresholdStat]:
        """Measure one mutated graph across the grid, one statistic per threshold."""
        per_threshold: Dict[str, ThresholdStat] = {}
        for threshold in self.thresholds:
            deltas = [
                mean_reduction(
                    baselines[(threshold, seed)],
                    self.evaluator.impact(repo, threshold=threshold, seed=seed),
                )
                for seed in self.seeds
            ]
            per_threshold[str(threshold)] = ThresholdStat(
                mean_delta=statistics.fmean(deltas),
                # Spread across seeds at this threshold only — see ThresholdStat.
                sigma_seed=statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            )
        return per_threshold
