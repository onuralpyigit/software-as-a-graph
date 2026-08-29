"""
tests/test_weight_calculation_revision.py — QoS-aware weight repairs
=====================================================================

Pins the corrections made to Intrinsic Topic Weighting (docs/graph-model.md
§4.3) and Aggregate Weight Propagation (§4.5):

1. QoS enum lookups are case-insensitive. ``integration_hub_migration_system``
   (all lowercase QoS values) previously scored ``QoS(t) = 0.0`` for every
   topic, collapsing every component weight in that scenario to the floor.
2. ``SizeNorm`` and ``FreqNorm`` attain a meaningful fraction of their range
   on realistic payloads, so beta/alpha/psi deliver the budget they declare.
3. The missing-frequency fallback no longer re-derives a rate from the
   topic's own reliability x priority score, which fed the QoS term back
   into the nominally independent frequency term.
4. Rules 3-4 (node_to_node, node_to_broker) use worst-case lift, not a
   probabilistic union, matching the corrected doc and both repositories.
5. ``compute_power_mean_weight``'s exponent resolves ``COMPONENT_POWER_MEAN_P``
   at call time rather than freezing it as an ordinary default argument value
   at function-definition time -- the latter made the constant silently
   un-patchable by any caller that omits ``p``, discovered while building
   ``reproduce/weight_global_sensitivity.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from saag.core.models import (
    MIN_TOPIC_WEIGHT,
    TOPIC_QOS_WEIGHT_BETA,
    TOPIC_SIZE_WEIGHT_ALPHA,
    TOPIC_FREQ_WEIGHT_PSI,
    TOPIC_SIZE_ENVELOPE_BYTES,
    TOPIC_DEFAULT_FREQUENCY_HZ,
    QoSPolicy,
    compute_size_norm,
    compute_freq_norm,
    compute_topic_weight,
    compute_lifted_edge_weight,
    compute_power_mean_weight,
    topic_weight_from_node_attrs,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
INTEGRATION_HUB = REPO_ROOT / "data" / "scenarios" / "integration_hub_migration_system.json"


# ── A1: case-insensitive QoS resolution ────────────────────────────────────

class TestCaseInsensitiveQoS:
    def test_lowercase_matches_uppercase(self):
        upper = QoSPolicy(durability="PERSISTENT", reliability="RELIABLE",
                           transport_priority="CRITICAL")
        lower = QoSPolicy(durability="persistent", reliability="reliable",
                           transport_priority="critical")
        assert lower.calculate_weight() == pytest.approx(upper.calculate_weight())
        assert lower.calculate_weight() == pytest.approx(1.0)

    def test_lowercase_no_longer_scores_zero(self):
        """The defect: a fully lowercase max-QoS policy used to score 0.0."""
        policy = QoSPolicy(durability="persistent", reliability="reliable",
                            transport_priority="critical")
        assert policy.calculate_weight() > 0.99

    def test_mixed_case_and_whitespace(self):
        policy = QoSPolicy(durability=" Transient_Local ", reliability="Reliable",
                            transport_priority="High")
        expected = QoSPolicy(durability="TRANSIENT_LOCAL", reliability="RELIABLE",
                              transport_priority="HIGH")
        assert policy.calculate_weight() == pytest.approx(expected.calculate_weight())

    def test_from_dict_canonicalises(self):
        policy = QoSPolicy.from_dict(
            {"durability": "volatile", "reliability": "best_effort", "transport_priority": "low"}
        )
        assert policy.durability == "VOLATILE"
        assert policy.reliability == "BEST_EFFORT"
        assert policy.transport_priority == "LOW"

    def test_from_node_attrs_canonicalises_nested_shape(self):
        policy = QoSPolicy.from_node_attrs(
            {"qos": {"durability": "reliable", "reliability": "reliable", "transport_priority": "critical"}}
        )
        # "reliable" is not a valid DURABILITY_SCORES key even canonicalised;
        # this checks canonicalisation happens, not that the value is valid.
        assert policy.durability == "RELIABLE"

    def test_unknown_value_still_scores_zero(self):
        """Canonicalisation must not turn an invalid value into a match."""
        policy = QoSPolicy(durability="not_a_real_value", reliability="BEST_EFFORT",
                            transport_priority="LOW")
        assert policy.calculate_weight() == pytest.approx(0.0)

    @pytest.mark.skipif(not INTEGRATION_HUB.exists(), reason="scenario corpus not present")
    def test_integration_hub_scenario_no_longer_floors(self):
        """Every topic in this scenario is authored in lowercase QoS. Before
        the fix every QoS(t) scored 0.0 and every component weight collapsed
        to MIN_TOPIC_WEIGHT."""
        data = json.loads(INTEGRATION_HUB.read_text())
        weights = [topic_weight_from_node_attrs(t) for t in data.get("topics", [])]
        assert weights, "scenario must contain topics"
        assert max(weights) > MIN_TOPIC_WEIGHT + 0.01
        # At least one topic is authored as ('reliable','persistent','critical')
        # or similar high-QoS combination and must now score near the top.
        assert any(w > 0.5 for w in weights)


# ── A2/A3: SizeNorm and FreqNorm envelopes ──────────────────────────────────

class TestSizeAndFreqNorm:
    def test_size_norm_bounds(self):
        assert compute_size_norm(0) == pytest.approx(0.0)
        assert compute_size_norm(TOPIC_SIZE_ENVELOPE_BYTES) == pytest.approx(1.0, abs=1e-4)

    def test_size_norm_monotone(self):
        sizes = [0, 1024, 32 * 1024, 256 * 1024, TOPIC_SIZE_ENVELOPE_BYTES, 10 * TOPIC_SIZE_ENVELOPE_BYTES]
        norms = [compute_size_norm(s) for s in sizes]
        assert norms == sorted(norms)

    def test_size_norm_realizes_meaningful_range_on_corpus_payloads(self):
        """The defect: the old /50.0-on-KiB divisor left SizeNorm in
        [0.0009, 0.101] for the 32B-32KiB range observed in the corpus,
        making alpha's declared budget undeliverable."""
        small, large = compute_size_norm(32), compute_size_norm(32 * 1024)
        assert large - small > 0.3

    def test_size_norm_capped_at_one(self):
        assert compute_size_norm(TOPIC_SIZE_ENVELOPE_BYTES * 1000) == pytest.approx(1.0)

    def test_freq_norm_bounds(self):
        assert compute_freq_norm(0) == pytest.approx(compute_freq_norm(None))
        assert compute_freq_norm(999.0) == pytest.approx(1.0, abs=1e-9)

    def test_freq_norm_unchanged_formula(self):
        """A3 is a naming change only: log10(1+f)/3.0 bit-for-bit."""
        for f in (1.0, 10.0, 100.0, 200.0):
            assert compute_freq_norm(f) == pytest.approx(min(math.log10(1.0 + f) / 3.0, 1.0))

    def test_beta_alpha_psi_sum_to_one(self):
        assert TOPIC_QOS_WEIGHT_BETA + TOPIC_SIZE_WEIGHT_ALPHA + TOPIC_FREQ_WEIGHT_PSI == pytest.approx(1.0)

    def test_topic_weight_bounded(self):
        policy = QoSPolicy(durability="PERSISTENT", reliability="RELIABLE", transport_priority="CRITICAL")
        w = compute_topic_weight(policy, size=10 ** 9, frequency=10 ** 6)
        assert MIN_TOPIC_WEIGHT <= w <= 1.0


# ── A4: frequency fallback no longer derives from QoS ───────────────────────

class TestFrequencyFallbackIndependence:
    def test_missing_frequency_uses_declared_default(self):
        assert compute_freq_norm(None) == pytest.approx(
            min(math.log10(1.0 + TOPIC_DEFAULT_FREQUENCY_HZ) / 3.0, 1.0)
        )

    def test_missing_frequency_independent_of_qos(self):
        """The removed defect: freq_norm used to be derived from
        reliability x priority when frequency was absent, so two policies
        with identical size but different QoS produced different freq_norm
        even though frequency was equally unknown for both."""
        low_qos = QoSPolicy(durability="VOLATILE", reliability="BEST_EFFORT", transport_priority="LOW")
        high_qos = QoSPolicy(durability="PERSISTENT", reliability="RELIABLE", transport_priority="CRITICAL")
        w_low = compute_topic_weight(low_qos, size=1024, frequency=None)
        w_high = compute_topic_weight(high_qos, size=1024, frequency=None)
        # The two must differ (QoS term differs) by exactly the QoS
        # contribution — the frequency term's contribution must be identical.
        freq_term_low = w_low - TOPIC_QOS_WEIGHT_BETA * low_qos.calculate_weight() - TOPIC_SIZE_WEIGHT_ALPHA * compute_size_norm(1024)
        freq_term_high = w_high - TOPIC_QOS_WEIGHT_BETA * high_qos.calculate_weight() - TOPIC_SIZE_WEIGHT_ALPHA * compute_size_norm(1024)
        assert freq_term_low == pytest.approx(freq_term_high, abs=1e-9)


# ── B2: worst-case lift for Rules 3-4 ────────────────────────────────────────

class TestLiftedEdgeWeight:
    def test_matches_max(self):
        assert compute_lifted_edge_weight([0.2, 0.9, 0.5]) == pytest.approx(0.9)

    def test_does_not_saturate_like_union(self):
        """The defect: compute_effective_edge_weight (noisy-OR) saturates
        when lifting several already-high dependencies; max does not exceed
        the strongest input."""
        weights = [0.9, 0.85, 0.8, 0.75]
        assert compute_lifted_edge_weight(weights) == pytest.approx(0.9)

    def test_floors_empty_input(self):
        assert compute_lifted_edge_weight([]) == pytest.approx(MIN_TOPIC_WEIGHT)

    def test_bounded_to_unit_interval(self):
        assert compute_lifted_edge_weight([1.5, 2.0]) == pytest.approx(1.0)


# ── COMPONENT_POWER_MEAN_P is genuinely live-patchable ──────────────────────

class TestPowerMeanPIsLive:
    def test_module_constant_change_takes_effect_without_explicit_p(self):
        """The defect: `p: float = COMPONENT_POWER_MEAN_P` as an ordinary
        default argument freezes at function-definition time, so patching
        the module constant silently had no effect on any caller (every
        caller in this codebase) that omits `p`."""
        from saag.core import models

        weights = [0.2, 0.9, 0.5]
        default_result = compute_power_mean_weight(weights)

        saved = models.COMPONENT_POWER_MEAN_P
        try:
            models.COMPONENT_POWER_MEAN_P = 1.0
            patched_result = compute_power_mean_weight(weights)
        finally:
            models.COMPONENT_POWER_MEAN_P = saved

        assert patched_result == pytest.approx(sum(weights) / len(weights))
        assert patched_result != pytest.approx(default_result)

    def test_explicit_p_still_overrides(self):
        weights = [0.2, 0.9, 0.5]
        assert compute_power_mean_weight(weights, p=1.0) == pytest.approx(
            sum(weights) / len(weights)
        )
