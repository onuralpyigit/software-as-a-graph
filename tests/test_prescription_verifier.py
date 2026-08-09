"""
tests/test_prescription_verifier.py

Unit tests for the per-edit acceptance filter: the pure arithmetic in ``judge``
and ``mean_reduction``, and the sweep grid ``EditVerifier`` drives. The grid
tests run against a fake evaluator, so no simulation happens here.
"""
import itertools

import pytest

from saag.prescription.models import PrescriptionPolicy, QosUpgrade, ThresholdStat
from saag.prescription.verifier import (
    DEFAULT_SEEDS,
    DEFAULT_THRESHOLDS,
    EditVerifier,
    judge,
    mean_reduction,
)


# ── judge: the acceptance rule ────────────────────────────────────────────────

def test_judge_accepts_when_every_threshold_beats_kappa_sigma():
    verdict = judge("qos_upgrade", "T1", {
        "0.1": ThresholdStat(mean_delta=0.05, sigma_seed=0.01),
        "0.2": ThresholdStat(mean_delta=0.04, sigma_seed=0.01),
    }, kappa=1.0, n_thresholds=2)

    assert verdict.accepted
    assert verdict.reason == ""
    assert verdict.worst_delta == pytest.approx(0.04)
    assert verdict.worst_margin == pytest.approx(0.03)


def test_judge_rejects_when_one_threshold_fails_and_names_it():
    verdict = judge("topic_split", "T1", {
        "0.1": ThresholdStat(mean_delta=0.05, sigma_seed=0.01),
        "0.5": ThresholdStat(mean_delta=0.002, sigma_seed=0.01),
    }, kappa=1.0, n_thresholds=2)

    assert not verdict.accepted
    # The binding threshold has to be in the reason, or the report says nothing
    # actionable about *why* the edit was declined.
    assert "0.5" in verdict.reason


def test_judge_sigma_is_per_threshold_not_pooled():
    """Sigma must be the spread across seeds at one threshold.

    Here each threshold's own noise is tiny (0.001) and every mean comfortably
    clears it, so the edit is accepted. Pooling the two thresholds' samples
    would produce a sigma on the order of the 0.04 gap *between* them, whose
    bar the 0.05 mean would not clear — the old behaviour, and the reason edits
    were rejected for being threshold-sensitive rather than noisy.
    """
    verdict = judge("qos_upgrade", "T1", {
        "0.1": ThresholdStat(mean_delta=0.09, sigma_seed=0.001),
        "0.5": ThresholdStat(mean_delta=0.05, sigma_seed=0.001),
    }, kappa=1.0, n_thresholds=2)

    assert verdict.accepted


def test_judge_zero_sigma_requires_strictly_positive_delta():
    verdict = judge("qos_upgrade", "T1", {
        "0.2": ThresholdStat(mean_delta=0.0, sigma_seed=0.0),
    }, kappa=1.0, n_thresholds=1)

    assert not verdict.accepted


def test_judge_incomplete_sweep_rejected_with_reason():
    verdict = judge("topic_split", "T1", {
        "0.1": ThresholdStat(mean_delta=0.5, sigma_seed=0.0),
    }, kappa=1.0, n_thresholds=3)

    assert not verdict.accepted
    assert verdict.reason == "incomplete_sweep"


def test_judge_propagates_simulation_failure_reason():
    verdict = judge("topic_split", "T1", {}, kappa=1.0, n_thresholds=3,
                    failure_reason="simulation_error: boom")

    assert not verdict.accepted
    assert verdict.reason == "simulation_error: boom"


def test_edit_verdict_to_dict_schema():
    verdict = judge("qos_upgrade", "T1", {
        "0.2": ThresholdStat(mean_delta=0.05, sigma_seed=0.01),
    }, kappa=2.0, n_thresholds=1)

    payload = verdict.to_dict()

    assert payload["schema"] == 2
    assert set(payload) == {
        "schema", "kind", "target", "kappa", "accepted", "reason",
        "worst_delta", "per_threshold",
    }
    assert payload["per_threshold"]["0.2"] == {"mean_delta": 0.05, "sigma_seed": 0.01}


# ── mean_reduction ────────────────────────────────────────────────────────────

def test_mean_reduction_restricted_to_common_ids():
    # T1 vanished from the mutated graph (a split renamed it), so it cannot be
    # differenced; only A1 and A2 count.
    assert mean_reduction(
        {"A1": 0.5, "A2": 0.3, "T1": 0.9},
        {"A1": 0.3, "A2": 0.1, "T1_AppA": 0.4},
    ) == pytest.approx(0.2)


def test_mean_reduction_empty_intersection_is_zero():
    assert mean_reduction({"A1": 0.5}, {"B1": 0.5}) == 0.0


# ── EditVerifier: the sweep grid ──────────────────────────────────────────────

class FakeEvaluator:
    """Records every (threshold, seed) asked for and returns a scripted impact.

    Impact is a function of the point *and* of whether the graph was mutated, so
    a delta computed against a mismatched baseline point is detectable.
    """

    def __init__(self):
        self.calls = []

    def impact(self, repo, *, threshold, seed):
        self.calls.append((threshold, seed))
        # Threshold contributes a large offset, seed a small one. A correctly
        # paired difference cancels the threshold term entirely.
        mutated = bool(getattr(repo, "_is_mutated", False))
        base = 100.0 * threshold + 0.001 * seed
        return {"A1": base - (1.0 if mutated else 0.0)}


class FakeRepo:
    def __init__(self, is_mutated=False):
        self._is_mutated = is_mutated


@pytest.fixture(autouse=True)
def stub_graph_rewrite(monkeypatch):
    """Skip the JSON round-trip: these tests are about the grid, not the mutator."""
    monkeypatch.setattr(
        "saag.prescription.verifier.apply_policy", lambda original_json, policy: original_json)
    monkeypatch.setattr(
        "saag.prescription.verifier.repo_from_json", lambda graph_json: FakeRepo(is_mutated=True))


@pytest.fixture
def one_edit_policy():
    return PrescriptionPolicy(qos_upgrades=[
        QosUpgrade(topic="T1", original_reliability="BEST_EFFORT", original_durability="VOLATILE")
    ])


def test_baseline_impact_measured_at_every_threshold_and_seed(one_edit_policy):
    evaluator = FakeEvaluator()
    verifier = EditVerifier(evaluator, thresholds=[0.1, 0.2], seeds=[1, 2])

    verifier.verify(FakeRepo(), {}, one_edit_policy)

    grid = set(itertools.product([0.1, 0.2], [1, 2]))
    # The baseline is swept over the whole grid, not measured once at a
    # single canonical point.
    assert set(evaluator.calls[:4]) == grid


def test_edit_delta_differences_matching_threshold_and_seed(one_edit_policy):
    evaluator = FakeEvaluator()
    verifier = EditVerifier(evaluator, thresholds=[0.1, 0.5], seeds=[1, 2, 3])

    verdict = verifier.verify(FakeRepo(), {}, one_edit_policy)[0]

    # The fake makes the mutated graph exactly 1.0 better at every point. If the
    # baseline were taken at a fixed threshold, the 100*threshold term would
    # leak into the delta and these would differ from 1.0 and from each other.
    assert verdict.per_threshold["0.1"].mean_delta == pytest.approx(1.0)
    assert verdict.per_threshold["0.5"].mean_delta == pytest.approx(1.0)
    # Paired differencing also cancels the seed term, so there is no noise left.
    assert verdict.per_threshold["0.1"].sigma_seed == pytest.approx(0.0)
    assert verdict.accepted


def test_verifier_never_requests_seed_none(one_edit_policy):
    evaluator = FakeEvaluator()

    EditVerifier(evaluator).verify(FakeRepo(), {}, one_edit_policy)

    assert evaluator.calls, "expected the verifier to run a sweep"
    assert all(seed is not None for _threshold, seed in evaluator.calls)
    assert {seed for _threshold, seed in evaluator.calls} == set(DEFAULT_SEEDS)
    assert {t for t, _seed in evaluator.calls} == set(DEFAULT_THRESHOLDS)


def test_verifier_honours_custom_kappa_seeds_thresholds(one_edit_policy):
    evaluator = FakeEvaluator()
    verifier = EditVerifier(evaluator, kappa=2.5, seeds=[1, 2], thresholds=[0.3])

    verdict = verifier.verify(FakeRepo(), {}, one_edit_policy)[0]

    assert set(verdict.per_threshold) == {"0.3"}
    assert verdict.kappa == 2.5
    assert {seed for _threshold, seed in evaluator.calls} == {1, 2}


def test_verifier_returns_no_verdicts_for_an_empty_policy():
    evaluator = FakeEvaluator()

    assert EditVerifier(evaluator).verify(FakeRepo(), {}, PrescriptionPolicy()) == []
    # An empty candidate set must not pay for a baseline sweep.
    assert evaluator.calls == []


def test_verifier_survives_an_edit_that_fails_to_simulate(monkeypatch, one_edit_policy):
    def explode(graph_json):
        raise RuntimeError("boom")

    monkeypatch.setattr("saag.prescription.verifier.repo_from_json", explode)

    verdict = EditVerifier(FakeEvaluator()).verify(FakeRepo(), {}, one_edit_policy)[0]

    assert not verdict.accepted
    assert "simulation_error" in verdict.reason


# ── parallel worker: must use the verifier's own layer/checkpoint ────────────
#
# `_verify_single_edit_worker` runs inside a separate process (the default
# path: EditVerifier.verify's n_jobs=-1), so it cannot capture `self.evaluator`
# by closure — it has to be told the layer and checkpoint explicitly. A bare
# `GraphEvaluator()` silently defaults to layer="system", which evaluates
# every candidate edit on the wrong layer whenever prescribe runs with
# --layer != system while the baselines were measured on that layer.

def test_worker_constructs_evaluator_with_configured_layer_and_checkpoint(
    monkeypatch, one_edit_policy
):
    from saag.prescription import verifier as verifier_mod

    seen = {}

    class RecordingEvaluator:
        def __init__(self, layer="system", gnn_checkpoint=None):
            seen["layer"] = layer
            seen["gnn_checkpoint"] = gnn_checkpoint

        def impact(self, repo, *, threshold, seed):
            return {"A1": 1.0}

    monkeypatch.setattr(verifier_mod, "GraphEvaluator", RecordingEvaluator)

    edit = one_edit_policy.edits()[0]
    args_tuple = (
        edit, {}, 1.0, [0.2], [42], {(0.2, 42): {"A1": 1.0}},
        "app", "output/gnn_checkpoints/app_ckpt",
    )
    verifier_mod._verify_single_edit_worker(args_tuple)

    assert seen == {"layer": "app", "gnn_checkpoint": "output/gnn_checkpoints/app_ckpt"}


def test_parallel_and_serial_paths_agree_on_evaluator_layer(monkeypatch):
    """End-to-end through EditVerifier.verify, forcing the real
    ProcessPoolExecutor path (>1 edit): the mutated-graph impact the parallel
    workers measure must come from the *same* layer as the baseline
    (self.evaluator.layer="app"), matching what the serial path measures.

    LayeredEvaluator.impact returns a layer-dependent value. The serial path
    always uses ``self.evaluator`` for both baseline and mutated impact, so it
    trivially agrees with itself — the real risk is a worker reconstructing
    ``GraphEvaluator()`` with its bare default (layer="system"), which would
    read the *wrong* layer's value and manufacture a nonzero delta out of
    nothing. Both paths should therefore report zero delta (baseline ==
    mutated) once the worker is threaded the configured layer correctly.
    """
    from saag.prescription import verifier as verifier_mod

    two_edit_policy = PrescriptionPolicy(qos_upgrades=[
        QosUpgrade(topic="T1", original_reliability="BEST_EFFORT", original_durability="VOLATILE"),
        QosUpgrade(topic="T2", original_reliability="BEST_EFFORT", original_durability="VOLATILE"),
    ])

    class LayeredEvaluator:
        def __init__(self, layer="system", gnn_checkpoint=None):
            self.layer = layer
            self.gnn_checkpoint = gnn_checkpoint

        def impact(self, repo, *, threshold, seed):
            return {"A1": 1.0 if self.layer == "app" else 5.0}

    monkeypatch.setattr(verifier_mod, "GraphEvaluator", LayeredEvaluator)
    verifier = EditVerifier(LayeredEvaluator(layer="app"), thresholds=[0.2], seeds=[42])

    serial = verifier.verify(FakeRepo(), {}, two_edit_policy, n_jobs=1)
    parallel = verifier.verify(FakeRepo(), {}, two_edit_policy, n_jobs=2)

    for verdicts, label in ((serial, "serial"), (parallel, "parallel")):
        for verdict in verdicts:
            stat = verdict.per_threshold["0.2"]
            assert stat.mean_delta == pytest.approx(0.0), (
                f"{label} path measured a nonzero delta from a self-vs-self "
                f"comparison — the mutated impact was read from a different "
                f"layer than the baseline"
            )
